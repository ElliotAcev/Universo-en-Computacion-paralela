/*
 * Simulacion N-cuerpos: Galaxia 3D
 * Computacion Paralela - UTEM
 *
 * Compilar:
 *   nvcc main.cu -o galaxy -lGL -lGLU -lglfw -lGLEW -O3 -arch=sm_75
 *
 * Dependencias:
 *   sudo apt install libglfw3-dev libglew-dev
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// OpenGL / GLFW / GLEW
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// ─── Parametros de simulacion ──────────────────────────────────────────────
#define N_DEFAULT       350000      // numero de cuerpos por defecto
#define TILE_SIZE       256         // threads por bloque (multiplo de 32)
#define G               0.00015f    // constante gravitacional escalada
#define EPSILON2        0.08f       // suavizador (evita singularidades)
#define DT              0.004f      // paso de tiempo
#define MASS_CENTRAL    50000.0f    // masa del agujero negro central (dominante: ancla el centro)
#define NUM_ARMS        2           // brazos espirales

// ─── Ventana ───────────────────────────────────────────────────────────────
#define WIN_W           1280
#define WIN_H           720

// ─── Macros de error ───────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                       \
    cudaError_t e = (call);                                         \
    if (e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(e));         \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

#define GL_CHECK() do {                                             \
    GLenum err = glGetError();                                      \
    if (err != GL_NO_ERROR)                                         \
        fprintf(stderr, "GL error %s:%d: 0x%x\n",                  \
                __FILE__, __LINE__, err);                           \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════
//  ESTRUCTURAS
// ═══════════════════════════════════════════════════════════════════════════

// float4: x,y,z = posicion/velocidad,  w = masa (en pos) / 0 (en vel)
typedef float4 Body;

// Estado de camara (controlada por mouse)
typedef struct {
    float posX, posY, posZ; // posicion libre de la camara
    float yaw, pitch;       // orientacion (mouse-look)
    float speed;            // velocidad de vuelo (ajustable con scroll)
    double lastMouseX, lastMouseY;
    int dragging;
} Camera;

// Estado global de la aplicacion
typedef struct {
    int    N;               // numero de cuerpos actual
    int    paused;
    int    showHelp;
    double simTime;         // tiempo acumulado de simulacion
    long   steps;           // pasos ejecutados
    float  fps;
    double lastFPSTime;
    int    fpsFrames;
} AppState;

// ═══════════════════════════════════════════════════════════════════════════
//  KERNELS CUDA
// ═══════════════════════════════════════════════════════════════════════════

/*
 * Kernel de fuerzas gravitacionales con Shared Memory Tiling.
 *
 * Cada bloque carga TILE_SIZE cuerpos a shared memory en cada iteracion
 * del loop externo. Todos los threads del bloque calculan la fuerza de
 * interaccion contra esos TILE_SIZE cuerpos. Esto reduce accesos a VRAM
 * de O(N^2) a O(N^2 / TILE_SIZE).
 *
 * pos[i] = (x, y, z, masa)
 * acc[i] = aceleracion acumulada para el cuerpo i
 */
__global__ void kernelFuerzas(const Body* __restrict__ pos,
                               Body*       __restrict__ acc,
                               int N)
{
    extern __shared__ Body tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float3 pi  = {0,0,0};
    float3 ai  = {0,0,0};

    if (i < N) {
        pi.x = pos[i].x;
        pi.y = pos[i].y;
        pi.z = pos[i].z;
    }

    // Loop sobre tiles de TILE_SIZE cuerpos
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Carga cooperativa: cada thread carga UN cuerpo al tile
        int jGlobal = t * TILE_SIZE + threadIdx.x;
        tile[threadIdx.x] = (jGlobal < N) ? pos[jGlobal] : make_float4(0,0,0,0);
        __syncthreads();

        // Calcula fuerza contra los TILE_SIZE cuerpos del tile
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float rx = tile[j].x - pi.x;
            float ry = tile[j].y - pi.y;
            float rz = tile[j].z - pi.z;
            float dist2 = rx*rx + ry*ry + rz*rz + EPSILON2;
            // rsqrtf: instruccion nativa GPU, ~4 ciclos vs ~20 de sqrtf+div
            float inv3  = rsqrtf(dist2 * dist2 * dist2);
            float fmag  = G * tile[j].w * inv3;
            ai.x += fmag * rx;
            ai.y += fmag * ry;
            ai.z += fmag * rz;
        }
        __syncthreads();
    }

    if (i < N)
        acc[i] = make_float4(ai.x, ai.y, ai.z, 0.0f);
}

/*
 * Kernel de integracion Leapfrog.
 *
 * Leapfrog es un integrador simplectico: conserva la energia mecanica
 * mucho mejor que Euler explicito para orbitas a largo plazo.
 *
 * v(t+dt/2) = v(t-dt/2) + a(t)*dt
 * x(t+dt)   = x(t) + v(t+dt/2)*dt
 *
 * pos[i].w = masa (no se modifica)
 * vel[i].w = 0 (no usado)
 */
__global__ void kernelLeapfrog(Body* __restrict__ pos,
                                Body* __restrict__ vel,
                                const Body* __restrict__ acc,
                                int N, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // El agujero negro (masa >500) NO se integra aqui: su posicion la fija
    // kernelFijarBHalCoM, que lo coloca en el centro de masa de las estrellas
    // (asi siempre queda en el centro visual del bulbo, sin derivar).
    if (pos[i].w > 500.0f) return;

    // Actualiza velocidad (medio paso adelante)
    float vx = vel[i].x + acc[i].x * dt;
    float vy = vel[i].y + acc[i].y * dt;
    float vz = vel[i].z + acc[i].z * dt;

    // Actualiza posicion
    float px = pos[i].x + vx * dt;
    float py = pos[i].y + vy * dt;
    float pz = pos[i].z + vz * dt;

    vel[i] = make_float4(vx, vy, vz, 0.0f);
    pos[i] = make_float4(px, py, pz, pos[i].w);  // conserva masa
}

/*
 * Recentrado del centro de masa (anti-deriva).
 *
 * Fijar el agujero negro en el origen inyecta un pequeño impulso al sistema
 * (rompe la conservacion del momento), y con el tiempo el disco de estrellas
 * se corre respecto del agujero negro. Estos dos kernels recalculan el centro
 * de masa de las estrellas cada frame y lo devuelven al origen, manteniendo el
 * bulbo siempre centrado sobre el agujero negro supermasivo.
 *
 * com[0..2] = sum(m*pos),  com[3] = sum(m)   (solo estrellas, no el agujero)
 */
__global__ void kernelAcumCoM(const Body* __restrict__ pos, int N, float* com)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float m = pos[i].w;
    if (m > 500.0f) return;                 // excluye el agujero negro

    // Solo cuenta las estrellas CERCANAS al agujero (el bulbo), no todo el
    // disco: asi el agujero queda en el centro del bulbo brillante y no en el
    // centro de masa global (que un disco asimetrico desplazaria).
    float dx = pos[i].x - pos[0].x;
    float dy = pos[i].y - pos[0].y;
    float dz = pos[i].z - pos[0].z;
    if (dx*dx + dy*dy + dz*dz > 16.0f) return;   // radio ~4 (nucleo del bulbo)

    atomicAdd(&com[0], m * pos[i].x);
    atomicAdd(&com[1], m * pos[i].y);
    atomicAdd(&com[2], m * pos[i].z);
    atomicAdd(&com[3], m);
}

// Coloca el agujero negro (cuerpo 0) en el centro de masa de las estrellas.
__global__ void kernelFijarBHalCoM(Body* __restrict__ pos, const float* com)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float invM = (com[3] > 0.0f) ? 1.0f / com[3] : 0.0f;
        pos[0].x = com[0] * invM;
        pos[0].y = com[1] * invM;
        pos[0].z = com[2] * invM;
    }
}

/*
 * Kernel auxiliar: copia posiciones al VBO de OpenGL para render.
 * devVBO apunta directamente al buffer de la GPU compartido con OpenGL.
 * Sin cudaMemcpy, sin round-trip CPU.
 */
__global__ void kernelCopiaVBO(const Body* __restrict__ pos,
                                float*      __restrict__ devVBO,
                                int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // VBO layout: x,y,z,speed (speed = longitud de velocidad, para colorear)
    devVBO[i*4+0] = pos[i].x;
    devVBO[i*4+1] = pos[i].y;
    devVBO[i*4+2] = pos[i].z;
    devVBO[i*4+3] = pos[i].w;  // masa → usada para tamaño del punto
}

// ═══════════════════════════════════════════════════════════════════════════
//  INICIALIZACION DE LA GALAXIA
// ═══════════════════════════════════════════════════════════════════════════

static float randf() { return (float)rand() / (float)RAND_MAX; }

// Muestra gaussiana (Box-Muller): da bandas difusas con caida suave en vez
// de bordes duros, como las nubes de una galaxia real en lugar de una linea.
static float randGauss() {
    float u1 = fmaxf(randf(), 1e-6f), u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*
 * Genera una galaxia espiral con:
 *   - NUM_ARMS brazos logaritmicos
 *   - Distribucion radial tipo Plummer (sqrt para densidad decreciente)
 *   - Velocidades tangenciales keplerianamente correctas
 *   - Agujero negro central de masa MASS_CENTRAL
 *   - Disco delgado: dispersion vertical proporcional a exp(-r)
 */
void initGalaxy(Body* hPos, Body* hVel, int N)
{
    srand((unsigned)time(NULL));
    const float PI2 = 2.0f * 3.14159265f;

    // Agujero negro supermasivo central: UN solo cuerpo en el origen. No se
    // fija; participa en la dinamica como los demas, y al ser el mas masivo se
    // queda en el fondo del pozo gravitatorio -> centro del bulbo, co-moviendose
    // con las estrellas (como en una galaxia real).
    hPos[0] = make_float4(0.0f, 0.0f, 0.0f, MASS_CENTRAL);
    hVel[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // ── Regiones de formacion estelar (clumps): nudos densos repartidos a lo
    // largo de los brazos. Dan la estructura GRUMOSA e irregular de una galaxia
    // real (nudos brillantes, densidad desigual) en vez de brazos uniformes.
    const int   NUM_CLUMPS = 40;
    float clumpR[NUM_CLUMPS], clumpA[NUM_CLUMPS], clumpW[NUM_CLUMPS];
    for (int c = 0; c < NUM_CLUMPS; c++) {
        int   ca = rand() % NUM_ARMS;
        float cr = sqrtf(randf()) * 21.0f + 1.5f;
        clumpR[c] = cr;
        clumpA[c] = ((float)ca / NUM_ARMS) * PI2 + cr * 0.28f
                    + sinf(cr * 0.9f + ca * 2.1f) * 0.08f
                    + (randf() - 0.5f) * 0.25f;
        clumpW[c] = 0.8f + randf() * 2.2f;   // radio del nudo
    }

    // Estrellas del disco (y nubes de gas/polvo intercaladas en los brazos)
    //
    // Poblacion mixta, como en una galaxia real:
    //   ~10% gas/polvo  -> concentrado en bandas difusas sobre los brazos
    //   ~58% disco viejo -> repartido en TODO angulo (no traza brazos), da
    //                       el resplandor de fondo denso y continuo del disco
    //   ~32% estrellas jovenes de los brazos -> siguen la espiral con
    //                       dispersion gaussiana ancha (nube, no linea)
    for (int i = 1; i < N; i++) {
        int   arm    = rand() % NUM_ARMS;   // brazo aleatorio (no ligado al indice)
        float roll   = randf();
        int   isBulge = roll < 0.12f;                       // 12% bulbo central denso
        int   isHaze  = (!isBulge) && (roll < 0.12f + 0.22f);   // 22% neblina difusa (relleno)
        int   isGas   = (!isBulge) && (!isHaze) && (roll < 0.12f + 0.22f + 0.18f);
        int   isField = (!isBulge) && (!isHaze) && (!isGas) && (roll < 0.12f + 0.22f + 0.18f + 0.18f);
        // el resto (~30%) son estrellas de brazo

        // Radio base (disco extendido). El bulbo usa su propia concentracion.
        float r = sqrtf(randf()) * 22.0f + 0.3f;

        float angle;
        if (isBulge) {
            // Bulbo esferoidal: estrellas concentradas hacia el centro pero
            // dejando un hueco vacio en el nucleo (r>=1.2) para que la sombra
            // del agujero negro sea visible sobre el fondo negro.
            r     = powf(randf(), 2.2f) * 6.0f + 0.15f;
            angle = randf() * PI2;
        } else if (isField || isHaze) {
            // Disco / neblina: cualquier angulo, sin sesgo de brazo. La neblina
            // se concentra un poco mas hacia el centro (relleno luminoso continuo).
            if (isHaze) r = powf(randf(), 0.85f) * 22.0f + 0.3f;
            angle = randf() * PI2;
        } else if (isGas && randf() < 0.45f) {
            // Polvo INTER-BRAZO: gas repartido en todo angulo -> el disco entre
            // los brazos tambien tiene polvo tenue, no solo negro.
            angle = randf() * PI2;
        } else {
            // Estrella joven / gas de brazo. La MITAD se agrupa en un nudo de
            // formacion estelar (clump) -> estructura grumosa e irregular; la
            // otra mitad sigue el brazo de forma difusa.
            if (randf() < 0.5f) {
                int   c = rand() % NUM_CLUMPS;
                r     = clumpR[c] + randGauss() * clumpW[c] * 0.7f;
                angle = clumpA[c] + randGauss() * (clumpW[c] * 0.7f) / fmaxf(r, 1.0f);
                r     = fmaxf(r, 0.3f);
            } else {
                // Espiral abierta que barre ~1 vuelta -> 2 brazos claros y amplios
                float baseAngle = ((float)arm / NUM_ARMS) * PI2 + r * 0.28f;
                float wobble = sinf(r * 0.9f + arm * 2.1f) * 0.08f;
                baseAngle += wobble;
                // Banda ANCHA -> brazos gruesos y rellenos, no una cinta fina
                float armWidth = (isGas ? 0.75f : 0.60f) / fmaxf(r * 0.28f, 0.4f);
                angle = baseAngle + randGauss() * armWidth;
                r += randGauss() * (isGas ? 1.4f : 1.1f);
                r  = fmaxf(r, 0.3f);
            }
        }

        float x = cosf(angle) * r;
        float z = sinf(angle) * r;
        // Grosor vertical (da VOLUMEN, no una lamina plana): el bulbo es
        // esferoidal grueso; el disco tiene un grosor moderado que decae con r.
        float thickness = isBulge ? (r * 0.7f)
                                  : (isGas ? 1.6f : 1.0f) * (0.5f + expf(-r * 0.15f));
        float y = randGauss() * 0.5f * thickness;

        // Masa: estrellas siguen una IMF (muchas chicas, pocas gigantes);
        // el gas y la neblina son casi ingravidos (no perturban la dinamica).
        // La neblina usa una masa marcadora minima (<0.001) para el shader.
        float mass;
        if (isHaze)      mass = 0.0004f + randf() * 0.0003f;
        else if (isGas)  mass = 0.002f + randf() * 0.004f;
        else             mass = 0.02f + powf(randf(), 4.0f) * 0.9f;

        // Velocidad circular kepleriana: v_c = sqrt(G*M_enc / r)
        // Masa encerrada aproximada segun el RADIO real (no el indice del
        // arreglo): al generar r = sqrt(rand())*22, la fraccion de masa del
        // disco encerrada dentro de r es ~ (r/22)^2. Usar el indice en vez
        // del radio real desincroniza la velocidad orbital de cada estrella
        // y hace que la galaxia se colapse lentamente hacia el centro.
        float fracEnc = fminf(1.0f, (r / 22.0f) * (r / 22.0f));
        float Menc    = MASS_CENTRAL + (float)(N - 1) * 0.15f * fracEnc;
        float vc      = sqrtf(G * Menc / r) * 0.92f;

        // Perturbaciones termicas pequenas
        float dvx = (randf() - 0.5f) * 0.012f;
        float dvy = (randf() - 0.5f) * 0.005f;
        float dvz = (randf() - 0.5f) * 0.012f;

        hPos[i] = make_float4(x, y, z, mass);
        hVel[i] = make_float4(-sinf(angle)*vc + dvx,
                               dvy,
                               cosf(angle)*vc + dvz,
                               0.0f);
    }

    // Anular el momento lineal neto de las ESTRELLAS (sin tocar el agujero
    // negro, que debe quedar quieto en el origen). Si se incluyera al agujero
    // en la correccion, este —que arranca en reposo— recibiria una velocidad
    // y, por su masa enorme, se iria desplazando fuera del bulbo. Dejandolo
    // quieto con las estrellas de momento neto cero, el nucleo (agujero + bulbo)
    // permanece centrado en el origen.
    double vx = 0.0, vy = 0.0, vz = 0.0, sm = 0.0;
    for (int i = 1; i < N; i++) {
        double m = hPos[i].w;
        vx += m * hVel[i].x; vy += m * hVel[i].y; vz += m * hVel[i].z; sm += m;
    }
    if (sm > 0.0) {
        float vcx = (float)(vx / sm), vcy = (float)(vy / sm), vcz = (float)(vz / sm);
        for (int i = 1; i < N; i++) {
            hVel[i].x -= vcx; hVel[i].y -= vcy; hVel[i].z -= vcz;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SHADERS OPENGL (GLSL inline)
// ═══════════════════════════════════════════════════════════════════════════

static const char* VS_SOURCE = R"glsl(
#version 330 core

layout(location = 0) in vec3 inPos;    // x, y, z
layout(location = 1) in float inMass;  // masa -> tamaño del punto

uniform mat4 uMVP;
uniform float uMaxMass;

out float vMass;
out float vDist;   // distancia al centro (para colorear)
out float vTemp;   // "temperatura" pseudo-aleatoria por estrella (0=fria/roja, 1=caliente/azul)

// hash determinista por vertice: da variedad de color tipo poblacion estelar real
float hash(int i) {
    uint x = uint(i) * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return float(x) / 4294967295.0;
}

void main() {
    gl_Position  = uMVP * vec4(inPos, 1.0);
    bool  isHaze = inMass < 0.001;   // neblina de relleno (resplandor continuo)
    bool  isGas  = inMass < 0.008 && !isHaze;   // gas/polvo coloreado
    float jitter = mix(0.7, 1.3, hash(gl_VertexID * 7 + 3));  // variedad extra entre estrellas similares
    float persp  = 55.0 / max(gl_Position.w, 0.02);

    if (isHaze) {
        // Neblina: grande y muy tenue -> rellena los huecos entre estrellas
        // con luz continua (simula millones de estrellas no resueltas)
        float hazeSz = mix(9.0, 22.0, hash(gl_VertexID * 5 + 2));
        gl_PointSize  = clamp(hazeSz * persp, 4.0, 260.0);
    } else if (isGas) {
        // Nubes de polvo/gas: medianas y muy translucidas -> al solaparse
        // muchas, forman bandas continuas de nebulosa (no perlas separadas)
        float cloudSz = mix(3.5, 9.0, hash(gl_VertexID * 13 + 1));
        gl_PointSize  = clamp(cloudSz * persp, 2.0, 130.0);
    } else if (inMass > 500.0) {
        // Agujero negro: se dibuja en una pasada separada (alpha), no aqui
        gl_PointSize = 1.0;
    } else {
        // Estrella normal
        float massSz = mix(0.22, 1.8, pow(clamp(inMass / uMaxMass, 0.0, 1.0), 0.5));
        // Bulbo: las estrellas cerca del centro se agrandan/funden en un
        // resplandor continuo, como el nucleo denso de una galaxia real.
        float dist2d  = length(inPos.xz);
        float bulge   = 1.0 + 1.6 * exp(-dist2d * 0.55);
        gl_PointSize = clamp(massSz * jitter * persp * bulge, 1.3, 120.0);
    }

    vMass        = inMass;
    vDist        = length(inPos.xz);  // radio en el plano del disco
    vTemp        = hash(gl_VertexID);
}
)glsl";

static const char* FS_SOURCE = R"glsl(
#version 330 core

in float vMass;
in float vDist;
in float vTemp;
out vec4 fragColor;

uniform float uMaxDist;

// Paleta tipo clasificacion espectral estelar: M(roja) -> K -> G(amarilla) -> F -> A(blanca) -> B/O(azul)
vec3 starColor(float temp) {
    vec3 cM = vec3(1.0,  0.45, 0.30);
    vec3 cK = vec3(1.0,  0.70, 0.45);
    vec3 cG = vec3(1.0,  0.95, 0.75);
    vec3 cA = vec3(0.85, 0.90, 1.0);
    vec3 cB = vec3(0.55, 0.70, 1.0);

    vec3 c = mix(cM, cK, smoothstep(0.0, 0.25, temp));
    c      = mix(c,  cG, smoothstep(0.25, 0.5, temp));
    c      = mix(c,  cA, smoothstep(0.5, 0.75, temp));
    c      = mix(c,  cB, smoothstep(0.75, 1.0, temp));
    return c;
}

void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv) * 2.0;   // 0 = centro del sprite, 1 = borde

    // El agujero negro se dibuja en su propia pasada (alpha), no aqui.
    if (vMass > 500.0) discard;

    // ── Neblina de relleno: resplandor blanco-calido continuo y muy tenue ───
    if (vMass < 0.001) {
        float glow = pow(1.0 - smoothstep(0.0, 1.0, d), 2.0);
        if (glow < 0.005) discard;
        // Mas calido/dorado hacia el centro, mas azulado en el disco exterior
        float t = clamp(vDist / uMaxDist, 0.0, 1.0);
        vec3 warm = vec3(1.0, 0.9, 0.72);
        vec3 cool = vec3(0.72, 0.80, 1.0);
        vec3 c = mix(warm, cool, t);
        float a = glow * 0.07;   // muy tenue: se acumula por solape en luz continua
        fragColor = vec4(c * a, a);
        return;
    }

    // ── Gas / polvo cosmico: nebulosas densas y coloridas ───────────────────
    if (vMass < 0.008) {
        float cloud = 1.0 - smoothstep(0.0, 1.0, d);
        cloud = pow(cloud, 1.3);
        if (cloud < 0.01) discard;
        // Paleta de polvo tipo Andromeda: marron/tostado calido dominante,
        // con toques ocasionales de rojo (regiones HII) y azul (cumulos jovenes)
        vec3 cTan    = vec3(0.60, 0.45, 0.32);
        vec3 cBrown  = vec3(0.45, 0.30, 0.20);
        vec3 cRed    = vec3(0.95, 0.35, 0.30);   // regiones HII brillantes
        vec3 cBlue   = vec3(0.45, 0.55, 0.80);   // cumulos jovenes azules
        vec3 c = mix(cTan, cBrown, smoothstep(0.0, 0.5, vTemp));
        // Nudos de emision rojos brillantes (formacion estelar) mas intensos
        float hii = smoothstep(0.78, 0.90, vTemp);
        c = mix(c, cRed, hii);
        c = mix(c, cBlue, smoothstep(0.93, 1.0, vTemp));
        float a = cloud * mix(0.20, 0.55, hii);   // los nudos HII son mas densos/visibles
        fragColor = vec4(c * a, a);
        return;
    }

    // ── Estrellas normales: nucleo brillante + halo amplio ──────────────────
    // Halo mas ancho y suave que antes: al superponerse muchas estrellas
    // cercanas, se funden en un resplandor continuo (disco denso) en vez
    // de verse como puntos aislados con espacio negro entre ellos.
    float core = 1.0 - smoothstep(0.0, 0.3, d);
    float glow = pow(1.0 - smoothstep(0.0, 1.0, d), 2.0);
    float a    = clamp(core * 1.3 + glow * 0.6, 0.0, 1.0);
    if (a < 0.01) discard;

    // Sesgo de temperatura por poblacion: estrellas cercanas al centro tienden a ser
    // mas viejas/rojizas-amarillas; en los brazos hay mas azules jovenes.
    float t = clamp(vDist / uMaxDist, 0.0, 1.0);
    float temp = clamp(vTemp * 0.8 + t * 0.35, 0.0, 1.0);

    vec3 c = starColor(temp);
    c *= mix(0.7, 1.3, clamp(vMass * 3.0, 0.0, 1.0));  // estrellas masivas brillan mas

    // Resplandor calido del bulbo galactico: tinte dorado suave, sin sobre-
    // exponer (la densidad de estrellas ya lo hace brillante por acumulacion)
    float bulgeT = clamp(vDist / 5.0, 0.0, 1.0);
    c = mix(vec3(1.0, 0.85, 0.6), c, bulgeT);   // dorado calido en el centro
    a = min(a * mix(1.3, 1.0, bulgeT), 1.0);

    fragColor = vec4(c * a, a * 0.9);
}
)glsl";

// ── Agujero negro: pasada dedicada con ALPHA blending (puede oscurecer) ──
// A diferencia de las estrellas (aditivo), esta pasada usa alpha normal, asi
// el horizonte de eventos se dibuja negro OPACO y realmente tapa las estrellas
// de atras -> se ve como un agujero negro real, no un hueco en las particulas.
static const char* BH_VS_SOURCE = R"glsl(
#version 330 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in float inMass;
uniform mat4 uMVP;
void main() {
    gl_Position  = uMVP * vec4(inPos, 1.0);
    float persp  = 55.0 / max(gl_Position.w, 0.02);
    // Tamaño acotado: crece con la cercania pero sin desbordarse
    gl_PointSize = clamp(24.0 * persp, 14.0, 220.0);
}
)glsl";

static const char* BH_FS_SOURCE = R"glsl(
#version 330 core
out vec4 fragColor;
void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv) * 2.0;   // 0 centro, 1 borde

    float horizon = 1.0 - smoothstep(0.34, 0.40, d);       // disco negro opaco
    float ring    = smoothstep(0.40, 0.46, d) * (1.0 - smoothstep(0.52, 0.72, d));
    float glow    = (1.0 - smoothstep(0.40, 1.0, d));
    if (d > 1.0) discard;

    // Color del anillo de acrecion: interior blanco-ardiente -> naranja exterior
    vec3 ringCol = mix(vec3(1.0, 0.95, 0.8), vec3(1.0, 0.5, 0.15),
                       smoothstep(0.46, 0.72, d));
    vec3 col = ringCol * (ring * 2.2 + glow * 0.5);

    // Composicion: horizonte negro opaco domina el centro
    float a = max(horizon, clamp(ring * 1.5 + glow * 0.4, 0.0, 1.0));
    col = mix(col, vec3(0.0), horizon);   // centro negro solido
    fragColor = vec4(col, a);
}
)glsl";

// ── Fondo: estrellas lejanas + galaxias distantes (sin fisica) ──
// inBrightness codifica el tipo de punto:
//   [0, 1)  -> estrella de fondo normal (brillo = valor)
//   [1, 2)  -> parte de una galaxia lejana (fraccion = tono/color)
// (Las nebulosas ya NO son puntos: se dibujan como billboards con textura.)
#define NUM_BG_STARS       14000
#define NUM_BG_GALAXIES    14
#define STARS_PER_GALAXY   500
#define NUM_BG_BLUE        60
#define NUM_BG_TOTAL       (NUM_BG_STARS + NUM_BG_GALAXIES * STARS_PER_GALAXY \
                            + NUM_BG_BLUE)
#define NUM_NEBULAE        16      // nubes de nebulosa de fondo (billboards)
#define NUM_DUST           260     // nubes de polvo estelar en los brazos (billboards)

static const char* BG_VS_SOURCE = R"glsl(
#version 330 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in float inBrightness;
uniform mat4 uMVP;
out float vBrightness;
void main() {
    gl_Position  = uMVP * vec4(inPos, 1.0);
    float persp  = 40.0 / max(gl_Position.w, 0.02);
    if (inBrightness >= 1.0)
        gl_PointSize = clamp(3.0 * persp, 2.0, 12.0);    // motas de una galaxia lejana
    else
        gl_PointSize = mix(1.3, 3.2, inBrightness);      // estrella de fondo
    vBrightness  = inBrightness;
}
)glsl";

static const char* BG_FS_SOURCE = R"glsl(
#version 330 core
in float vBrightness;
out vec4 fragColor;
void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv) * 2.0;

    if (vBrightness >= 1.0) {
        // Galaxia lejana: nucleo brillante + tono calido/frio
        float hue = vBrightness - 1.0;
        float core = 1.0 - smoothstep(0.0, 0.4, d);
        float halo = 1.0 - smoothstep(0.2, 1.0, d);
        float a    = clamp(core * 1.2 + halo * 0.6, 0.0, 1.0);
        if (a < 0.01) discard;
        vec3 warm = vec3(1.0, 0.85, 0.6);
        vec3 cool = vec3(0.65, 0.78, 1.0);
        vec3 c = mix(warm, cool, hue) * 1.5;   // mas brillante
        fragColor = vec4(c * a, a);
        return;
    }

    float core = 1.0 - smoothstep(0.4, 1.0, d);
    float a    = core * mix(0.25, 0.85, vBrightness);
    if (a < 0.01) discard;
    // Las estrellas de fondo mas brillantes reciben un pequeño extra
    // para cruzar el umbral del bloom y tener su propio glow sutil.
    float boost = smoothstep(0.85, 1.0, vBrightness) * 1.4;
    vec3 col = vec3(0.85, 0.9, 1.0) * (a + boost * core);
    fragColor = vec4(col, a);
}
)glsl";

// ── Nebulosas: billboards (paneles orientados a la camara) con textura de
// nube PROCEDURAL (ruido fBm) en el fragment shader. No son puntos, asi que
// no parpadean y se ven como nubes de gas continuas y coloreadas.
//
// Cada nebulosa es un quad. Atributos por vertice:
//   loc 0: centro de la nube (vec3, world)
//   loc 1: esquina del quad en [-1,1] (vec2)
//   loc 2: params = (tamaño, hue, seed) (vec3)
// El quad se orienta hacia la camara usando uCamRight / uCamUp (uniforms).
static const char* NEB_VS_SOURCE = R"glsl(
#version 330 core
layout(location = 0) in vec3  inCenter;
layout(location = 1) in vec2  inCorner;
layout(location = 2) in vec3  inParams;   // x=size, y=hue, z=seed
uniform mat4 uMVP;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
out vec2  vUV;
out float vHue;
out float vSeed;
void main() {
    vec3 worldPos = inCenter
                  + uCamRight * (inCorner.x * inParams.x)
                  + uCamUp    * (inCorner.y * inParams.x);
    gl_Position = uMVP * vec4(worldPos, 1.0);
    vUV   = inCorner;      // [-1,1]
    vHue  = inParams.y;
    vSeed = inParams.z;
}
)glsl";

static const char* NEB_FS_SOURCE = R"glsl(
#version 330 core
in vec2  vUV;
in float vHue;
in float vSeed;
out vec4 fragColor;

// Ruido de valor + fBm (varias octavas) -> textura de nube turbulenta
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1,0));
    float c = hash21(i + vec2(0,1));
    float d = hash21(i + vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) { v += amp * vnoise(p); p *= 2.03; amp *= 0.5; }
    return v;
}

void main() {
    float r = length(vUV);
    if (r > 1.0) discard;
    // Caida radial suave (borde difuso de la nube)
    float falloff = pow(1.0 - smoothstep(0.0, 1.0, r), 1.5);

    // Textura de nube turbulenta; el seed desplaza el patron por nebulosa
    vec2 p = vUV * 2.2 + vec2(vSeed * 13.0, vSeed * 7.0);
    float n = fbm(p);
    n = pow(clamp(n * 1.4, 0.0, 1.0), 1.6);   // contraste -> filamentos/huecos

    float density = falloff * n;
    if (density < 0.01) discard;

    // Color por hue (mezcla nebulosa): magenta -> teal -> azul -> dorado
    vec3 cMagenta = vec3(0.75, 0.20, 0.65);
    vec3 cTeal    = vec3(0.10, 0.55, 0.65);
    vec3 cBlue    = vec3(0.25, 0.32, 0.85);
    vec3 cGold    = vec3(0.85, 0.50, 0.28);
    vec3 c = mix(cMagenta, cTeal, smoothstep(0.0, 0.35, vHue));
    c      = mix(c, cBlue, smoothstep(0.35, 0.7, vHue));
    c      = mix(c, cGold, smoothstep(0.7, 1.0, vHue));
    // Un poco mas brillante donde el gas es mas denso
    c *= 0.6 + 0.6 * n;

    float a = density * 0.28;   // translucido pero visible
    fragColor = vec4(c * a, a);
}
)glsl";

// Fragment shader de POLVO ESTELAR de la galaxia: misma textura procedural de
// nube, pero con paleta calida (polvo marron/tostado + nudos rojos HII + azul)
// para las bandas de polvo de los brazos. Reusa NEB_VS_SOURCE.
static const char* DUST_FS_SOURCE = R"glsl(
#version 330 core
in vec2  vUV;
in float vHue;
in float vSeed;
out vec4 fragColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1,0));
    float c = hash21(i + vec2(0,1));
    float d = hash21(i + vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) { v += amp * vnoise(p); p *= 2.03; amp *= 0.5; }
    return v;
}

void main() {
    float r = length(vUV);
    if (r > 1.0) discard;
    float falloff = pow(1.0 - smoothstep(0.0, 1.0, r), 1.4);
    vec2 p = vUV * 2.6 + vec2(vSeed * 13.0, vSeed * 7.0);
    float n = fbm(p);
    n = pow(clamp(n * 1.5, 0.0, 1.0), 1.7);
    float density = falloff * n;
    if (density < 0.01) discard;

    // Paleta de polvo: tostado/marron dominante, con nudos rojos (HII) y azul
    vec3 cTan   = vec3(0.60, 0.45, 0.32);
    vec3 cBrown = vec3(0.45, 0.30, 0.20);
    vec3 cRed   = vec3(0.95, 0.35, 0.28);
    vec3 cBlue  = vec3(0.45, 0.55, 0.85);
    vec3 c = mix(cTan, cBrown, smoothstep(0.0, 0.5, vHue));
    c      = mix(c, cRed,  smoothstep(0.80, 0.90, vHue));
    c      = mix(c, cBlue, smoothstep(0.93, 1.0, vHue));
    c *= 0.6 + 0.6 * n;

    float a = density * 0.30;
    fragColor = vec4(c * a, a);
}
)glsl";

// ═══════════════════════════════════════════════════════════════════════════
//  POST-PROCESADO: BLOOM (halo de luz alrededor de zonas brillantes)
// ═══════════════════════════════════════════════════════════════════════════
//
// La escena se renderiza a una textura HDR (valores > 1.0 permitidos, ya que
// el blending aditivo satura mucho). Se extraen los pixeles brillantes, se
// difuminan (blur gaussiano separable en 2 pasadas, varias iteraciones) y se
// suman de vuelta sobre la imagen original -> efecto "glow" cinematografico.

static const char* QUAD_VS_SOURCE = R"glsl(
#version 330 core
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
out vec2 vUV;
void main() {
    vUV = inUV;
    gl_Position = vec4(inPos, 0.0, 1.0);
}
)glsl";

static const char* BRIGHT_FS_SOURCE = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
void main() {
    vec3 c = texture(uScene, vUV).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    // Umbral bajo: asi TODO el disco (no solo el nucleo) aporta al bloom y,
    // al difuminarse, las miles de estrellas se funden en una nube continua
    // en vez de verse como puntos sueltos -- igual que en una foto real,
    // donde las estrellas no resueltas se perciben como luz difusa.
    fragColor = vec4(lum > 0.25 ? c : vec3(0.0), 1.0);
}
)glsl";

static const char* BLUR_FS_SOURCE = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uImage;
uniform vec2 uTexelSize;
uniform int uHorizontal;
void main() {
    float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec2 dir = (uHorizontal == 1) ? vec2(uTexelSize.x, 0.0) : vec2(0.0, uTexelSize.y);
    vec3 result = texture(uImage, vUV).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        result += texture(uImage, vUV + dir * float(i)).rgb * w[i];
        result += texture(uImage, vUV - dir * float(i)).rgb * w[i];
    }
    fragColor = vec4(result, 1.0);
}
)glsl";

static const char* COMPOSITE_FS_SOURCE = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
void main() {
    vec3 scene = texture(uScene, vUV).rgb;
    vec3 bloom = texture(uBloom, vUV).rgb;
    float EXPOSURE = 3.2;   // sube el brillo general antes del tonemap (imagen se veia muy apagada)
    vec3 hdr   = (scene + bloom * 1.3) * EXPOSURE;
    // Tonemap simple (Reinhard): hdr=0 -> 0 exacto, no debe "levantar" el negro del fondo
    vec3 mapped = hdr / (hdr + vec3(1.0));
    fragColor = vec4(mapped, 1.0);
}
)glsl";

// ═══════════════════════════════════════════════════════════════════════════
//  OPENGL: compilar shaders, matrices
// ═══════════════════════════════════════════════════════════════════════════

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, sizeof(buf), NULL, buf);
        fprintf(stderr, "Shader error:\n%s\n", buf);
    }
    return s;
}

static GLuint buildProgram(const char* vs, const char* fs)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, compileShader(GL_VERTEX_SHADER,   vs));
    glAttachShader(prog, compileShader(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
        fprintf(stderr, "Program link error:\n%s\n", buf);
    }
    return prog;
}

// Matrices column-major para OpenGL
typedef float Mat4[16];

static void mat4Identity(Mat4 m)
{
    memset(m, 0, sizeof(Mat4));
    m[0]=m[5]=m[10]=m[15]=1.0f;
}

static void mat4Perspective(Mat4 m, float fovY, float aspect, float n, float f)
{
    memset(m, 0, sizeof(Mat4));
    float t = tanf(fovY / 2.0f);
    m[0]  =  1.0f / (aspect * t);
    m[5]  =  1.0f / t;
    m[10] = -(f + n) / (f - n);
    m[11] = -1.0f;
    m[14] = -(2.0f * f * n) / (f - n);
}

static void mat4Mul(Mat4 out, const Mat4 a, const Mat4 b)
{
    Mat4 tmp;
    // Matrices column-major (convencion OpenGL): index = col*4 + row.
    // C = A*B  =>  C[col*4+row] = sum_k A[k*4+row] * B[col*4+k]
    for (int col=0; col<4; col++) for (int row=0; row<4; row++) {
        tmp[col*4+row] = 0;
        for (int k=0; k<4; k++) tmp[col*4+row] += a[k*4+row]*b[col*4+k];
    }
    memcpy(out, tmp, sizeof(Mat4));
}

static void mat4RotX(Mat4 m, float a)
{
    mat4Identity(m);
    m[5]=cosf(a); m[6]=-sinf(a);
    m[9]=sinf(a); m[10]=cosf(a);
}
static void mat4RotY(Mat4 m, float a)
{
    mat4Identity(m);
    m[0]=cosf(a); m[2]=sinf(a);
    m[8]=-sinf(a); m[10]=cosf(a);
}
static void mat4Trans(Mat4 m, float x, float y, float z)
{
    mat4Identity(m);
    m[12]=x; m[13]=y; m[14]=z;
}

// Vector "forward" de la camara (hacia donde mira) a partir de yaw/pitch
static void camForward(float yaw, float pitch, float* fx, float* fy, float* fz)
{
    *fx = sinf(yaw) * cosf(pitch);
    *fy = sinf(pitch);
    *fz = -cosf(yaw) * cosf(pitch);
}

// Matriz de vista tipo camara libre (FPS), construida directamente (lookAt manual)
static void buildViewMatrix(Mat4 m, float px, float py, float pz, float yaw, float pitch)
{
    float fx, fy, fz;
    camForward(yaw, pitch, &fx, &fy, &fz);

    // right = normalize(cross(forward, worldUp=(0,1,0)))
    float rx = -fz, ry = 0.0f, rz = fx;
    float rlen = sqrtf(rx*rx + rz*rz);
    if (rlen > 1e-6f) { rx /= rlen; rz /= rlen; }

    // up = cross(right, forward)
    float ux = ry*fz - rz*fy;
    float uy = rz*fx - rx*fz;
    float uz = rx*fy - ry*fx;

    m[0]=rx; m[1]=ux; m[2]=-fx; m[3]=0.0f;
    m[4]=ry; m[5]=uy; m[6]=-fy; m[7]=0.0f;
    m[8]=rz; m[9]=uz; m[10]=-fz; m[11]=0.0f;
    m[12] = -(rx*px + ry*py + rz*pz);
    m[13] = -(ux*px + uy*py + uz*pz);
    m[14] =  (fx*px + fy*py + fz*pz);
    m[15] = 1.0f;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CALLBACKS GLFW
// ═══════════════════════════════════════════════════════════════════════════

static Camera   g_cam   = {0.0f, 10.0f, 38.0f, 0.0f, -0.28f, 10.0f, 0, 0, 0};
static AppState g_app   = {N_DEFAULT, 0, 1, 0.0, 0, 0.0f, 0.0, 0};

static void cbMouseButton(GLFWwindow* w, int btn, int action, int mods)
{
    (void)mods;
    if (btn == GLFW_MOUSE_BUTTON_LEFT)
        g_cam.dragging = (action == GLFW_PRESS);
    if (action == GLFW_PRESS)
        glfwGetCursorPos(w, &g_cam.lastMouseX, &g_cam.lastMouseY);
}

static void cbMouseMove(GLFWwindow* w, double x, double y)
{
    if (!g_cam.dragging) return;
    float dx = (float)(x - g_cam.lastMouseX) * 0.005f;
    float dy = (float)(y - g_cam.lastMouseY) * 0.005f;
    g_cam.yaw   += dx;
    g_cam.pitch  = fmaxf(-1.5f, fminf(1.5f, g_cam.pitch + dy));
    g_cam.lastMouseX = x;
    g_cam.lastMouseY = y;
}

static void cbScroll(GLFWwindow* w, double dx, double dy)
{
    (void)w; (void)dx;
    // La rueda ajusta la velocidad de vuelo (no la distancia: la camara es libre)
    g_cam.speed = fmaxf(0.2f, fminf(80.0f, g_cam.speed * (dy > 0 ? 1.15f : 0.87f)));
}

static void cbKey(GLFWwindow* w, int key, int sc, int action, int mods)
{
    (void)sc; (void)mods;
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_P)
        g_app.paused = !g_app.paused;
    if (key == GLFW_KEY_H)
        g_app.showHelp = !g_app.showHelp;
}

static void cbResize(GLFWwindow* w, int width, int height)
{
    (void)w;
    glViewport(0, 0, width, height);
}

// ═══════════════════════════════════════════════════════════════════════════
//  BENCHMARK: comparacion CPU vs GPU
// ═══════════════════════════════════════════════════════════════════════════

void benchmarkCPUvsGPU(int N)
{
    printf("\n=== BENCHMARK CPU vs GPU (N=%d) ===\n", N);

    size_t bytes = N * sizeof(Body);
    Body* hPos = (Body*)malloc(bytes);
    Body* hVel = (Body*)malloc(bytes);
    Body* hAcc = (Body*)malloc(bytes);
    initGalaxy(hPos, hVel, N);

    // --- CPU (un paso de fuerzas) ---
    double t0 = glfwGetTime();
    for (int i = 0; i < N; i++) {
        float ax=0,ay=0,az=0;
        for (int j = 0; j < N; j++) {
            if (i==j) continue;
            float rx = hPos[j].x-hPos[i].x;
            float ry = hPos[j].y-hPos[i].y;
            float rz = hPos[j].z-hPos[i].z;
            float d2 = rx*rx+ry*ry+rz*rz+EPSILON2;
            float inv3 = 1.0f/(d2*sqrtf(d2));
            float f = G*hPos[j].w*inv3;
            ax+=f*rx; ay+=f*ry; az+=f*rz;
        }
        hAcc[i]=make_float4(ax,ay,az,0);
    }
    double cpuMs = (glfwGetTime()-t0)*1000.0;

    // --- GPU ---
    Body *dPos, *dAcc;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, bytes, cudaMemcpyHostToDevice));

    int grid = (N + TILE_SIZE - 1) / TILE_SIZE;
    size_t shMem = TILE_SIZE * sizeof(Body);

    // warmup
    kernelFuerzas<<<grid,TILE_SIZE,shMem>>>(dPos,dAcc,N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);
    cudaEventRecord(evStart);
    kernelFuerzas<<<grid,TILE_SIZE,shMem>>>(dPos,dAcc,N);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    float gpuMs=0;
    cudaEventElapsedTime(&gpuMs,evStart,evStop);

    double flops = 20.0 * (double)N * (double)N;
    printf("CPU: %.1f ms\n", cpuMs);
    printf("GPU: %.2f ms\n", gpuMs);
    printf("Speedup: %.1fx\n", cpuMs/gpuMs);
    printf("GPU GFLOPS: %.2f\n", flops / (gpuMs * 1e6));

    cudaFree(dPos); cudaFree(dAcc);
    free(hPos); free(hVel); free(hAcc);
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);
    printf("================================\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    int N = N_DEFAULT;
    if (argc > 1) N = atoi(argv[1]);
    if (N < 1024)  N = 1024;
    if (N > 500000) N = 500000;
    g_app.N = N;

    printf("=== Simulacion N-cuerpos: Galaxia 3D ===\n");
    printf("N = %d cuerpos\n", N);
    printf("TILE_SIZE = %d\n", TILE_SIZE);
    printf("Controles: WASD=moverse, arrastrar=mirar, SPACE/CTRL=subir/bajar,\n"
           "           SHIFT=velocidad, scroll=ajustar velocidad, P=pausa, H=ayuda, ESC=salir\n\n");

    // ── GLFW / OpenGL ──────────────────────────────────────────────────────
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    char title[128];
    snprintf(title, sizeof(title), "Galaxy N-body — N=%d", N);
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, title, NULL, NULL);
    if (!window) { fprintf(stderr, "Ventana GLFW failed\n"); glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // desactiva VSync para medir FPS real

    glfwSetMouseButtonCallback(window, cbMouseButton);
    glfwSetCursorPosCallback(window, cbMouseMove);
    glfwSetScrollCallback(window, cbScroll);
    glfwSetKeyCallback(window, cbKey);
    glfwSetFramebufferSizeCallback(window, cbResize);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "GLEW init failed\n"); return 1; }

    // ── Buffers GPU (CUDA) ─────────────────────────────────────────────────
    size_t bytes = N * sizeof(Body);
    Body *hPos = (Body*)malloc(bytes);
    Body *hVel = (Body*)malloc(bytes);
    initGalaxy(hPos, hVel, N);

    Body *dPos, *dVel, *dAcc;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dAcc, 0, bytes));

    // Acumulador del centro de masa de las estrellas (4 floats en GPU)
    float* dCoM;
    CUDA_CHECK(cudaMalloc(&dCoM, 4 * sizeof(float)));

    // ── VBO compartido CUDA-OpenGL ─────────────────────────────────────────
    // El VBO vive en la VRAM de la GPU.
    // CUDA escribe las posiciones directamente al VBO cada frame.
    // OpenGL lee el mismo VBO para render → cero cudaMemcpy.
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Atributo 0: posicion xyz
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Atributo 1: masa (para tamaño del punto)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Registrar VBO en CUDA
    cudaGraphicsResource* cudaVBORes;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBORes, vbo,
                                            cudaGraphicsMapFlagsWriteDiscard));

    // ── Fondo: estrellas lejanas + galaxias distantes (VBO estatico, sin CUDA) ──
    float* bgData = (float*)malloc(NUM_BG_TOTAL * 4 * sizeof(float));
    int bgIdx = 0;
    for (int i = 0; i < NUM_BG_STARS; i++, bgIdx++) {
        float theta = randf() * 2.0f * 3.14159265f;
        float phi   = acosf(2.0f * randf() - 1.0f);
        float rad   = 70.0f + randf() * 120.0f;
        bgData[bgIdx*4+0] = rad * sinf(phi) * cosf(theta);
        bgData[bgIdx*4+1] = rad * cosf(phi);
        bgData[bgIdx*4+2] = rad * sinf(phi) * sinf(theta);
        bgData[bgIdx*4+3] = randf();  // brillo
    }
    // Galaxias lejanas: grumos elipticos de tamaño y orientacion variados
    for (int g = 0; g < NUM_BG_GALAXIES; g++) {
        float gTheta = randf() * 2.0f * 3.14159265f;
        float gPhi   = acosf(2.0f * randf() - 1.0f);
        float gRad   = 130.0f + randf() * 100.0f;
        float cx = gRad * sinf(gPhi) * cosf(gTheta);
        float cy = gRad * cosf(gPhi);
        float cz = gRad * sinf(gPhi) * sinf(gTheta);
        float hue   = randf();
        float gSize = 2.0f + randf() * 4.5f;          // variedad de tamaño
        float flat  = 0.1f + randf() * 0.5f;          // que tan de canto se ve
        float tiltA = randf() * 2.0f * 3.14159265f;
        for (int i = 0; i < STARS_PER_GALAXY; i++, bgIdx++) {
            float rr    = sqrtf(randf()) * gSize;
            float aa    = randf() * 2.0f * 3.14159265f;
            float lx = cosf(aa) * rr;
            float lz = sinf(aa) * rr * flat;          // aplastado -> disco inclinado
            float ly = (randf() - 0.5f) * 0.6f;
            float rx = lx*cosf(tiltA) - ly*sinf(tiltA);
            float ry = lx*sinf(tiltA) + ly*cosf(tiltA);
            bgData[bgIdx*4+0] = cx + rx;
            bgData[bgIdx*4+1] = cy + ry;
            bgData[bgIdx*4+2] = cz + lz;
            bgData[bgIdx*4+3] = 1.0f + hue;  // >=1 marca "galaxia lejana"
        }
    }

    // (Las nebulosas se dibujan aparte como billboards con textura procedural.)

    // Estrellas azules brillantes dispersas (cumulos jovenes en el fondo)
    for (int s = 0; s < NUM_BG_BLUE; s++, bgIdx++) {
        float t = randf() * 2.0f * 3.14159265f, p = acosf(2.0f * randf() - 1.0f);
        float rad = 55.0f + randf() * 90.0f;
        bgData[bgIdx*4+0] = rad * sinf(p) * cosf(t);
        bgData[bgIdx*4+1] = rad * cosf(p);
        bgData[bgIdx*4+2] = rad * sinf(p) * sinf(t);
        bgData[bgIdx*4+3] = 0.97f;  // estrella de fondo muy brillante
    }
    GLuint bgVAO, bgVBO;
    glGenVertexArrays(1, &bgVAO);
    glGenBuffers(1, &bgVBO);
    glBindVertexArray(bgVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bgVBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_BG_TOTAL * 4 * sizeof(float), bgData, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    free(bgData);

    GLuint bgProg    = buildProgram(BG_VS_SOURCE, BG_FS_SOURCE);
    GLint  bgLocMVP  = glGetUniformLocation(bgProg, "uMVP");

    // ── Nebulosas: billboards con textura procedural ────────────────────────
    // Cada nebulosa = 1 quad (6 vertices). Por vertice: centro(3) + esquina(2)
    // + params(3: size, hue, seed) = 8 floats.
    const int NEB_VERTS = NUM_NEBULAE * 6;
    float* nebData = (float*)malloc(NEB_VERTS * 8 * sizeof(float));
    const float corner[6][2] = {
        {-1,-1}, {1,-1}, {1,1}, {-1,-1}, {1,1}, {-1,1}
    };
    int nv = 0;
    for (int n = 0; n < NUM_NEBULAE; n++) {
        float th  = randf() * 2.0f * 3.14159265f;
        float ph  = acosf(2.0f * randf() - 1.0f);
        float rad = 60.0f + randf() * 90.0f;
        float cx = rad * sinf(ph) * cosf(th);
        float cy = rad * cosf(ph);
        float cz = rad * sinf(ph) * sinf(th);
        float size = 18.0f + randf() * 26.0f;
        float hue  = randf();
        float seed = randf() * 10.0f;
        for (int v = 0; v < 6; v++, nv++) {
            nebData[nv*8+0] = cx;
            nebData[nv*8+1] = cy;
            nebData[nv*8+2] = cz;
            nebData[nv*8+3] = corner[v][0];
            nebData[nv*8+4] = corner[v][1];
            nebData[nv*8+5] = size;
            nebData[nv*8+6] = hue;
            nebData[nv*8+7] = seed;
        }
    }
    GLuint nebVAO, nebVBO;
    glGenVertexArrays(1, &nebVAO);
    glGenBuffers(1, &nebVBO);
    glBindVertexArray(nebVAO);
    glBindBuffer(GL_ARRAY_BUFFER, nebVBO);
    glBufferData(GL_ARRAY_BUFFER, NEB_VERTS * 8 * sizeof(float), nebData, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float)));
    glEnableVertexAttribArray(2);
    free(nebData);

    GLuint nebProg      = buildProgram(NEB_VS_SOURCE, NEB_FS_SOURCE);
    GLint  nebLocMVP    = glGetUniformLocation(nebProg, "uMVP");
    GLint  nebLocRight  = glGetUniformLocation(nebProg, "uCamRight");
    GLint  nebLocUp     = glGetUniformLocation(nebProg, "uCamUp");

    // ── Polvo estelar de los brazos: billboards con textura (escala galaxia) ──
    const int DUST_VERTS = NUM_DUST * 6;
    float* dustData = (float*)malloc(DUST_VERTS * 8 * sizeof(float));
    int dv = 0;
    const float PI2d = 2.0f * 3.14159265f;
    for (int c = 0; c < NUM_DUST; c++) {
        int   arm = rand() % NUM_ARMS;
        float rr  = sqrtf(randf()) * 20.0f + 2.0f;            // radio dentro del disco
        float baseAngle = ((float)arm / NUM_ARMS) * PI2d + rr * 0.28f;
        baseAngle += sinf(rr * 0.9f + arm * 2.1f) * 0.08f;    // ondulacion del brazo
        float ang = baseAngle + randGauss() * 0.35f;
        rr += randGauss() * 1.2f;
        float cx = cosf(ang) * rr;
        float cz = sinf(ang) * rr;
        float cy = randGauss() * 0.4f;                        // disco fino
        float size = 2.2f + randf() * 4.5f;
        // hue: mayormente polvo (bajo), a veces rojo HII (~0.85) o azul (~0.97)
        float hr = randf();
        float hue = (hr < 0.80f) ? randf() * 0.5f
                  : (hr < 0.93f) ? 0.85f
                                 : 0.97f;
        float seed = randf() * 10.0f;
        for (int v = 0; v < 6; v++, dv++) {
            dustData[dv*8+0] = cx;
            dustData[dv*8+1] = cy;
            dustData[dv*8+2] = cz;
            dustData[dv*8+3] = corner[v][0];
            dustData[dv*8+4] = corner[v][1];
            dustData[dv*8+5] = size;
            dustData[dv*8+6] = hue;
            dustData[dv*8+7] = seed;
        }
    }
    GLuint dustVAO, dustVBO;
    glGenVertexArrays(1, &dustVAO);
    glGenBuffers(1, &dustVBO);
    glBindVertexArray(dustVAO);
    glBindBuffer(GL_ARRAY_BUFFER, dustVBO);
    glBufferData(GL_ARRAY_BUFFER, DUST_VERTS * 8 * sizeof(float), dustData, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float)));
    glEnableVertexAttribArray(2);
    free(dustData);

    GLuint dustProg     = buildProgram(NEB_VS_SOURCE, DUST_FS_SOURCE);
    GLint  dustLocMVP   = glGetUniformLocation(dustProg, "uMVP");
    GLint  dustLocRight = glGetUniformLocation(dustProg, "uCamRight");
    GLint  dustLocUp    = glGetUniformLocation(dustProg, "uCamUp");

    // ── Shader program ─────────────────────────────────────────────────────
    GLuint prog = buildProgram(VS_SOURCE, FS_SOURCE);
    GLint locMVP     = glGetUniformLocation(prog, "uMVP");
    GLint locMaxMass = glGetUniformLocation(prog, "uMaxMass");
    GLint locMaxDist = glGetUniformLocation(prog, "uMaxDist");

    // Programa dedicado del agujero negro (alpha blending)
    GLuint bhProg     = buildProgram(BH_VS_SOURCE, BH_FS_SOURCE);
    GLint  bhLocMVP   = glGetUniformLocation(bhProg, "uMVP");

    // ── Estado OpenGL ──────────────────────────────────────────────────────
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);   // blending aditivo: zonas densas brillan
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glDisable(GL_DEPTH_TEST);            // transparencia aditiva no necesita depth

    // ── Post-procesado: bloom (render-to-texture HDR + blur + composite) ───
    int bloomW = WIN_W, bloomH = WIN_H;

    GLuint hdrFBO, hdrTex;
    glGenFramebuffers(1, &hdrFBO);
    glGenTextures(1, &hdrTex);
    glBindTexture(GL_TEXTURE_2D, hdrTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bloomW, bloomH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdrTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        fprintf(stderr, "HDR framebuffer incompleto\n");

    GLuint pingFBO[2], pingTex[2];
    glGenFramebuffers(2, pingFBO);
    glGenTextures(2, pingTex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, pingTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bloomW, bloomH, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, pingFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingTex[i], 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Quad de pantalla completa para los pases de post-procesado
    float quadVerts[] = {
        // pos.xy      // uv
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
    };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint brightProg = buildProgram(QUAD_VS_SOURCE, BRIGHT_FS_SOURCE);
    GLuint blurProg    = buildProgram(QUAD_VS_SOURCE, BLUR_FS_SOURCE);
    GLuint compProg    = buildProgram(QUAD_VS_SOURCE, COMPOSITE_FS_SOURCE);
    GLint  locBrightScene = glGetUniformLocation(brightProg, "uScene");
    GLint  locBlurImage   = glGetUniformLocation(blurProg, "uImage");
    GLint  locBlurTexel   = glGetUniformLocation(blurProg, "uTexelSize");
    GLint  locBlurHoriz   = glGetUniformLocation(blurProg, "uHorizontal");
    GLint  locCompScene   = glGetUniformLocation(compProg, "uScene");
    GLint  locCompBloom   = glGetUniformLocation(compProg, "uBloom");

    // ── Configuracion de kernels ───────────────────────────────────────────
    int gridSim  = (N + TILE_SIZE - 1) / TILE_SIZE;
    int gridCopy = (N + 255) / 256;
    size_t shMem = TILE_SIZE * sizeof(Body);

    // Benchmark inicial
    if (N <= 10000) benchmarkCPUvsGPU(N);

    printf("Iniciando loop de simulacion...\n");
    g_app.lastFPSTime = glfwGetTime();
    double lastFrameTime = glfwGetTime();

    // ─────────────────────────────────────────────────────────────────────
    //  LOOP PRINCIPAL
    // ─────────────────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now0 = glfwGetTime();
        float dt = (float)(now0 - lastFrameTime);
        lastFrameTime = now0;

        // ── Movimiento libre de camara (WASD + subir/bajar) ───────────────
        {
            float fx, fy, fz;
            camForward(g_cam.yaw, g_cam.pitch, &fx, &fy, &fz);
            float rx = -fz, rz = fx;
            float rlen = sqrtf(rx*rx + rz*rz);
            if (rlen > 1e-6f) { rx /= rlen; rz /= rlen; }

            float boost = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 3.0f : 1.0f;
            float v = g_cam.speed * boost * dt;

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { g_cam.posX += fx*v; g_cam.posY += fy*v; g_cam.posZ += fz*v; }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { g_cam.posX -= fx*v; g_cam.posY -= fy*v; g_cam.posZ -= fz*v; }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { g_cam.posX += rx*v; g_cam.posZ += rz*v; }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { g_cam.posX -= rx*v; g_cam.posZ -= rz*v; }
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)        g_cam.posY += v;
            if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) g_cam.posY -= v;
        }

        // ── Paso de fisica en GPU ─────────────────────────────────────────
        if (!g_app.paused) {
            // 1. Calcular fuerzas (O(N^2) con tiling)
            kernelFuerzas<<<gridSim, TILE_SIZE, shMem>>>(dPos, dAcc, N);

            // 2. Integrar (Leapfrog) — el agujero negro no se integra aqui
            kernelLeapfrog<<<gridSim, TILE_SIZE>>>(dPos, dVel, dAcc, N, DT);

            // 3. Colocar el agujero negro en el centro de masa de las estrellas
            //    -> siempre queda en el centro visual del bulbo, sin derivar.
            CUDA_CHECK(cudaMemset(dCoM, 0, 4 * sizeof(float)));
            kernelAcumCoM<<<gridSim, TILE_SIZE>>>(dPos, N, dCoM);
            kernelFijarBHalCoM<<<1, 1>>>(dPos, dCoM);

            g_app.steps++;
            g_app.simTime += DT;
        }

        // ── Copiar posiciones al VBO (CUDA→OpenGL sin pasar por CPU) ──────
        float* devVBOPtr;
        size_t vboSize;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBORes, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
                       (void**)&devVBOPtr, &vboSize, cudaVBORes));

        kernelCopiaVBO<<<gridCopy, 256>>>(dPos, devVBOPtr, N);

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBORes, 0));

        // ── Render de la escena a textura HDR (para el bloom) ──────────────
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
        glViewport(0, 0, bloomW, bloomH);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Matriz MVP: Perspectiva * Vista(camara libre FPS)
        Mat4 proj, view, mvp;
        mat4Perspective(proj, 0.85f, (float)fbW/fbH, 0.03f, 200.0f);
        buildViewMatrix(view, g_cam.posX, g_cam.posY, g_cam.posZ, g_cam.yaw, g_cam.pitch);
        mat4Mul(mvp, proj, view);

        // Fondo de estrellas lejanas (se dibuja primero, de fondo)
        glUseProgram(bgProg);
        glUniformMatrix4fv(bgLocMVP, 1, GL_FALSE, mvp);
        glBindVertexArray(bgVAO);
        glDrawArrays(GL_POINTS, 0, NUM_BG_TOTAL);

        // Nebulosas (billboards con textura): orientadas hacia la camara
        {
            float fx, fy, fz;
            camForward(g_cam.yaw, g_cam.pitch, &fx, &fy, &fz);
            float rx = -fz, ry = 0.0f, rz = fx;   // right = normalize(cross(fwd, up))
            float rl = sqrtf(rx*rx + rz*rz);
            if (rl > 1e-6f) { rx /= rl; rz /= rl; }
            float ux = ry*fz - rz*fy;             // up = cross(right, fwd)
            float uy = rz*fx - rx*fz;
            float uz = rx*fy - ry*fx;
            glUseProgram(nebProg);
            glUniformMatrix4fv(nebLocMVP, 1, GL_FALSE, mvp);
            glUniform3f(nebLocRight, rx, ry, rz);
            glUniform3f(nebLocUp, ux, uy, uz);
            glBindVertexArray(nebVAO);
            glDrawArrays(GL_TRIANGLES, 0, NUM_NEBULAE * 6);
        }

        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp);
        glUniform1f(locMaxMass, 0.92f);
        glUniform1f(locMaxDist, 24.0f);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, N);

        // Polvo estelar de los brazos (billboards con textura procedural)
        {
            float fx, fy, fz;
            camForward(g_cam.yaw, g_cam.pitch, &fx, &fy, &fz);
            float rx = -fz, ry = 0.0f, rz = fx;
            float rl = sqrtf(rx*rx + rz*rz);
            if (rl > 1e-6f) { rx /= rl; rz /= rl; }
            float ux = ry*fz - rz*fy;
            float uy = rz*fx - rx*fz;
            float uz = rx*fy - ry*fx;
            glUseProgram(dustProg);
            glUniformMatrix4fv(dustLocMVP, 1, GL_FALSE, mvp);
            glUniform3f(dustLocRight, rx, ry, rz);
            glUniform3f(dustLocUp, ux, uy, uz);
            glBindVertexArray(dustVAO);
            glDrawArrays(GL_TRIANGLES, 0, NUM_DUST * 6);
        }

        // (El agujero negro se dibuja al final, encima del bloom, para que su
        //  horizonte negro no sea "comido" por el resplandor del bulbo.)

        // ── Post-procesado: extraer brillo + blur gaussiano (bloom) ────────
        glDisable(GL_BLEND);
        glBindVertexArray(quadVAO);

        glUseProgram(brightProg);
        glBindFramebuffer(GL_FRAMEBUFFER, pingFBO[0]);
        glViewport(0, 0, bloomW, bloomH);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hdrTex);
        glUniform1i(locBrightScene, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(blurProg);
        glUniform1i(locBlurImage, 0);
        glUniform2f(locBlurTexel, 1.0f/bloomW, 1.0f/bloomH);
        int horizontal = 1, pingSrc = 0;
        const int BLUR_ITERS = 7;
        for (int i = 0; i < BLUR_ITERS; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, pingFBO[1 - pingSrc]);
            glUniform1i(locBlurHoriz, horizontal);
            glBindTexture(GL_TEXTURE_2D, pingTex[pingSrc]);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            pingSrc = 1 - pingSrc;
            horizontal = 1 - horizontal;
        }

        // ── Composicion final: escena + bloom -> pantalla, con tonemap ─────
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbW, fbH);
        glUseProgram(compProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hdrTex);
        glUniform1i(locCompScene, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, pingTex[pingSrc]);
        glUniform1i(locCompBloom, 1);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // ── Agujero negro ENCIMA del bloom: horizonte negro opaco y limpio ──
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(bhProg);
        glUniformMatrix4fv(bhLocMVP, 1, GL_FALSE, mvp);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, 1);   // cuerpo 0 = agujero negro
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        // ── FPS en titulo ─────────────────────────────────────────────────
        g_app.fpsFrames++;
        double now = glfwGetTime();
        if (now - g_app.lastFPSTime >= 0.5) {
            g_app.fps = (float)(g_app.fpsFrames / (now - g_app.lastFPSTime));
            g_app.fpsFrames = 0;
            g_app.lastFPSTime = now;
            snprintf(title, sizeof(title),
                     "Galaxy N-body | N=%d | %.0f FPS | paso=%ld | t=%.2f",
                     N, g_app.fps, g_app.steps, g_app.simTime);
            glfwSetWindowTitle(window, title);
        }

        glfwSwapBuffers(window);
    }

    // ── Limpieza ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVBORes));
    cudaFree(dPos); cudaFree(dVel); cudaFree(dAcc); cudaFree(dCoM);
    free(hPos); free(hVel);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &bgVBO);
    glDeleteVertexArrays(1, &bgVAO);
    glDeleteProgram(bgProg);
    glDeleteProgram(bhProg);
    glDeleteBuffers(1, &nebVBO);
    glDeleteVertexArrays(1, &nebVAO);
    glDeleteProgram(nebProg);
    glDeleteBuffers(1, &dustVBO);
    glDeleteVertexArrays(1, &dustVAO);
    glDeleteProgram(dustProg);
    glDeleteFramebuffers(1, &hdrFBO);
    glDeleteTextures(1, &hdrTex);
    glDeleteFramebuffers(2, pingFBO);
    glDeleteTextures(2, pingTex);
    glDeleteBuffers(1, &quadVBO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteProgram(brightProg);
    glDeleteProgram(blurProg);
    glDeleteProgram(compProg);
    glfwDestroyWindow(window);
    glfwTerminate();

    printf("Simulacion terminada. Pasos: %ld, tiempo simulado: %.4f\n",
           g_app.steps, g_app.simTime);
    return 0;
}
