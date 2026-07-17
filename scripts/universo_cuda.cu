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
// Radio visual del agujero negro, en unidades del mundo (la galaxia mide ~22).
// Semi-tamano del QUAD del agujero negro, en unidades del mundo. Ojo: no es el
// tamano del agujero: el fenomeno (sombra + disco + arcos lenteados) ocupa solo
// el ~45% central. El resto es margen para que los arcos no los corte el borde
// del quad (se veia el filo recto del cuadrado).
//
// Escala: la galaxia tiene radio ~22. Con 3.2 de quad, el fenomeno visible
// (sombra + disco) mide ~2.6 de diametro: un nucleo galactico creible. Un
// agujero negro supermasivo REAL es millones de veces mas pequeno que su
// galaxia (seria invisible), asi que esto ya es una licencia generosa.
#define RADIO_BH        3.2f
#define MASS_CENTRAL    50000.0f    // masa del agujero negro central (dominante: ancla el centro)
#define NUM_ARMS        2           // brazos espirales

// ─── Ritmo de la narrativa (segundos) ──────────────────────────────────────
// Cuanto dura el ACTO 1 antes de que la camara vaya sola al halo. Da tiempo a
// que la gravedad teja la telarana cosmica. Se puede saltar con G.
#define SEG_UNIVERSO    50.0f
// Espera tras aterrizar en el halo antes de que la galaxia se forme sola.
#define SEG_ESPERA_HALO  2.5f
// Cuanto tarda el halo en reorganizarse en galaxia (transicion suave).
#define SEG_MORPH       12.0f

// ─── Modo UNIVERSO (cosmologico) ───────────────────────────────────────────
// La caja donde vive el universo. Debe coincidir con --escala de exportar_ci.py
#define UNIVERSE_BOX        60.0f
#define INV_UNIVERSE_BOX    (1.0f / UNIVERSE_BOX)
// Gravedad del modo universo: mucho mas debil que la galaxia (no hay agujero
// negro dominante, solo materia difusa que se va agrupando sola).
#define G_UNIVERSE          0.0030f
#define EPSILON2_UNIVERSE   0.10f
#define DT_UNIVERSE         0.010f

// Parametros que cambian en tiempo de ejecucion segun el modo (universo/galaxia).
// Son __device__ para que los lea el kernel; el host los fija con
// cudaMemcpyToSymbol antes de cambiar de acto.
__device__ bool  g_periodic = false;      // caja periodica (solo modo universo)
__device__ float g_G        = G;          // constante gravitacional activa
__device__ float g_EPS2     = EPSILON2;   // suavizador activo

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

/*
 * Los tres actos de la narrativa:
 *   ACTO_UNIVERSO: la gravedad teje la telarana cosmica (caja periodica)
 *   ACTO_ZOOM:     la camara vuela hasta el halo mas denso
 *   ACTO_GALAXIA:  ese halo se convierte en una galaxia espiral
 * Se avanza con la tecla G (o automaticamente si --auto).
 */
typedef enum {
    ACTO_UNIVERSO = 0,
    ACTO_ZOOM     = 1,
    ACTO_MORPH    = 2,   // la materia del halo se reorganiza en galaxia
    ACTO_GALAXIA  = 3
} Acto;

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

    // ─── narrativa ───
    Acto   acto;
    float  actoT;           // segundos transcurridos dentro del acto actual
    int    autoAvance;      // 1 = la narrativa avanza sola con el tiempo
    float  zoomT;           // progreso del zoom [0..1]
    float  morphT;          // progreso de la transicion halo->galaxia [0..1]
    int    nGal;            // particulas que forman la galaxia (las del halo)
    int    bhIdx;           // indice REAL del agujero negro tras el morph
    float  haloX, haloY, haloZ;   // centro del halo elegido
    int    haloPop;         // particulas del halo
    // camara guardada al empezar el zoom (para interpolar desde ahi)
    float  zoomDesdeX, zoomDesdeY, zoomDesdeZ, zoomDesdeYaw, zoomDesdePitch;
    float  fade;            // fundido de la transicion [0..1]
    int    solicitarActo;   // la tecla G pide avanzar de acto
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
            // CAJA PERIODICA (convencion de imagen minima): cada par interactua
            // con la copia mas cercana a traves de los bordes que se envuelven.
            // Sin esto todo colapsaria a un unico punto en vez de formar la
            // telarana cosmica. rintf() redondea al entero mas cercano.
            if (g_periodic) {
                rx -= UNIVERSE_BOX * rintf(rx * INV_UNIVERSE_BOX);
                ry -= UNIVERSE_BOX * rintf(ry * INV_UNIVERSE_BOX);
                rz -= UNIVERSE_BOX * rintf(rz * INV_UNIVERSE_BOX);
            }
            float dist2 = rx*rx + ry*ry + rz*rz + g_EPS2;
            // rsqrtf: instruccion nativa GPU, ~4 ciclos vs ~20 de sqrtf+div
            float inv3  = rsqrtf(dist2 * dist2 * dist2);
            float fmag  = g_G * tile[j].w * inv3;
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

    // Masa ~0 = telarana cosmica congelada (decorado) o estrella ya absorbida.
    // Ponerles masa 0 evita que EJERZAN gravedad, pero seguian RECIBIENDOLA:
    // eran particulas de prueba cayendo hacia el agujero negro hasta que se
    // las tragaba. El fondo debe quedarse quieto.
    if (pos[i].w < 1e-6f) return;

    // Actualiza velocidad (medio paso adelante)
    float vx = vel[i].x + acc[i].x * dt;
    float vy = vel[i].y + acc[i].y * dt;
    float vz = vel[i].z + acc[i].z * dt;

    // Actualiza posicion
    float px = pos[i].x + vx * dt;
    float py = pos[i].y + vy * dt;
    float pz = pos[i].z + vz * dt;

    // En modo universo la caja se envuelve: lo que sale por un borde entra por
    // el opuesto (fronteras periodicas). fmodf puede dar negativo -> se corrige.
    if (g_periodic) {
        px = fmodf(px, UNIVERSE_BOX); if (px < 0.0f) px += UNIVERSE_BOX;
        py = fmodf(py, UNIVERSE_BOX); if (py < 0.0f) py += UNIVERSE_BOX;
        pz = fmodf(pz, UNIVERSE_BOX); if (pz < 0.0f) pz += UNIVERSE_BOX;
    }

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
/*
 * ─── BUSCADOR DE HALOS (modo universo) ─────────────────────────────────────
 *
 * Localiza el nodo mas denso de la telarana cosmica: el sitio donde nacera la
 * galaxia. Es un FoF ("friends of friends") casero por rejilla, la version
 * simple de lo que hacen Rockstar o AHF.
 */
#define HALO_GRID   16      // resolucion de la rejilla del buscador

__global__ void kernelContarCeldas(const Body* __restrict__ pos, int N, int* conteo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cx = (int)(pos[i].x * INV_UNIVERSE_BOX * HALO_GRID);
    int cy = (int)(pos[i].y * INV_UNIVERSE_BOX * HALO_GRID);
    int cz = (int)(pos[i].z * INV_UNIVERSE_BOX * HALO_GRID);
    cx = min(max(cx, 0), HALO_GRID - 1);
    cy = min(max(cy, 0), HALO_GRID - 1);
    cz = min(max(cz, 0), HALO_GRID - 1);

    atomicAdd(&conteo[(cx * HALO_GRID + cy) * HALO_GRID + cz], 1);
}

// Centro de masa de las particulas que caen dentro de la celda elegida.
__global__ void kernelCoMCelda(const Body* __restrict__ pos, int N,
                               int cx, int cy, int cz, float* com)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int px = (int)(pos[i].x * INV_UNIVERSE_BOX * HALO_GRID);
    int py = (int)(pos[i].y * INV_UNIVERSE_BOX * HALO_GRID);
    int pz = (int)(pos[i].z * INV_UNIVERSE_BOX * HALO_GRID);
    if (px != cx || py != cy || pz != cz) return;

    atomicAdd(&com[0], pos[i].x);
    atomicAdd(&com[1], pos[i].y);
    atomicAdd(&com[2], pos[i].z);
    atomicAdd(&com[3], 1.0f);
}

/*
 * Centro de masa de la GALAXIA ENTERA (no de lo que rodea al agujero negro).
 *
 * Antes solo contaba las estrellas a menos de 4 del agujero. Eso creaba un
 * bucle de realimentacion: si el agujero se desviaba un poco, su zona de
 * busqueda se iba con el, media el centro de masa de estrellas cada vez mas
 * descentradas, y se desviaba mas... hasta escaparse. Con 350k estrellas el
 * bulbo era tan denso que lo anclaba; con 27k (la galaxia que nace de un halo)
 * el ruido bastaba para disparar el bucle.
 *
 * Ahora se promedia toda la galaxia: un centro que NO depende de donde este el
 * agujero, asi que no hay realimentacion posible.
 *
 * com[0..2] = suma(m*pos),  com[3] = suma(m)
 */
__global__ void kernelAcumCoM(const Body* __restrict__ pos, int N, float* com,
                              int bh, const int* __restrict__ slot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || i == bh) return;

    // Solo cuenta la galaxia: la telarana cosmica (slot < 0) es decorado.
    if (slot != NULL && slot[i] < 0) return;

    float m = pos[i].w;
    if (m > 500.0f) return;      // otro cuerpo masivo (no deberia haber)
    if (m <= 0.0f) return;       // absorbida por el agujero: ya no cuenta

    atomicAdd(&com[0], m * pos[i].x);
    atomicAdd(&com[1], m * pos[i].y);
    atomicAdd(&com[2], m * pos[i].z);
    atomicAdd(&com[3], m);
}

// Coloca el agujero negro (cuerpo 0) en el centro de masa de las estrellas.
__global__ void kernelFijarBHalCoM(Body* __restrict__ pos, const float* com, int bh)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Si no hay estrellas cerca, NO tocar el agujero: antes se hacia
        // pos[bh] = com*0 = (0,0,0) y el agujero se teletransportaba a la
        // esquina del mundo.
        if (com[3] <= 0.0f) return;

        float invM = 1.0f / com[3];
        float cx = com[0] * invM;
        float cy = com[1] * invM;
        float cz = com[2] * invM;

        // SEGUIMIENTO SUAVE, no salto instantaneo.
        // El centro de masa medido tiene ruido estadistico ~1/sqrt(n_estrellas):
        // con 350k estrellas apenas se nota, pero con 27k (la galaxia que nace
        // de un halo) el agujero temblaba persiguiendo ese ruido. Filtramos.
        const float k = 0.06f;   // 0 = inmovil, 1 = salta al instante
        pos[bh].x += (cx - pos[bh].x) * k;
        pos[bh].y += (cy - pos[bh].y) * k;
        pos[bh].z += (cz - pos[bh].z) * k;
    }
}

/*
 * ─── TRANSICION SUAVE HALO -> GALAXIA ──────────────────────────────────────
 *
 * En vez de sustituir el universo por la galaxia de golpe (un cambiazo feo en
 * un solo frame), interpolamos cada particula desde donde esta hasta su sitio
 * en la galaxia. Visualmente: la materia del halo se reorganiza y se asienta
 * en un disco espiral, que es lo que ocurre de verdad cuando el gas de un halo
 * colapsa y forma una galaxia.
 *
 * s va de 0 (halo) a 1 (galaxia formada), con suavizado en los extremos.
 * Cada particula lleva su propio retardo segun su distancia al centro: las de
 * dentro se ordenan antes que las de fuera, como en un colapso real.
 */
// Radio alrededor del halo cuya materia formara la galaxia. Coincide con el
// radio del disco galactico, asi la galaxia ocupa justo la esfera de la que
// salio. La materia de FUERA no se toca: la telarana cosmica sigue ahi.
#define RADIO_HALO   28.0f

/*
 * Marca que particulas caen dentro del halo y les asigna un hueco en la
 * galaxia. slot[i] = -1 -> esa particula se queda quieta (sigue siendo
 * telarana cosmica). slot[i] >= 0 -> ocupara ese sitio de la galaxia.
 */
__global__ void kernelMarcarHalo(const Body* __restrict__ pos, int N,
                                 float cx, float cy, float cz,
                                 int* __restrict__ slot, int* __restrict__ contador,
                                 int* __restrict__ bhIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Distancia DIRECTA, sin imagen minima. Si usaramos la imagen periodica,
    // la materia del borde opuesto entraria en la galaxia y tendria que cruzar
    // todo el universo (o teletransportarse) para llegar: un salto visual
    // enorme. Por eso buscarHalo() prefiere halos centrados, donde la esfera
    // cabe entera sin tocar las fronteras.
    float dx = pos[i].x - cx, dy = pos[i].y - cy, dz = pos[i].z - cz;
    if (dx*dx + dy*dy + dz*dz <= RADIO_HALO * RADIO_HALO) {
        int k = atomicAdd(contador, 1);
        slot[i] = k;
        // initGalaxy pone el agujero negro en el hueco 0. Ese hueco se lo lleva
        // una particula cualquiera (la que gane el atomicAdd), NO la de indice 0.
        // Guardamos QUE indice acabo siendo el agujero, porque el recentrado y
        // el render lo necesitan.
        if (k == 0) *bhIdx = i;
    } else {
        slot[i] = -1;   // fuera del halo: sigue siendo telarana cosmica
    }
}

/*
 * Al terminar el morph, cada particula de la galaxia necesita SU velocidad
 * orbital (la que calculo initGalaxy), no la que traia del universo.
 *
 * Sin esto la galaxia no rota: las estrellas conservan las velocidades
 * aleatorias del Big Bang, el disco se deshace, el bulbo deriva... y como el
 * agujero negro se coloca en el centro de masa del bulbo, se va con el.
 *
 * La materia de fuera del halo (slot < 0) conserva su velocidad: sigue siendo
 * telarana cosmica y debe seguir su curso.
 */
/*
 * El agujero negro DESTRUYE lo que se le acerca demasiado (disrupcion de marea).
 *
 * Sin esto las estrellas atraviesan el centro y salen por el otro lado: el
 * suavizado del kernel de fuerzas (EPSILON2) evita la singularidad, pero
 * tambien deja pasar de largo.
 *
 * El radio no es el del horizonte sino el del DISCO DE ACRECION: una estrella
 * que se acerca tanto la despedazan las fuerzas de marea mucho antes de cruzar
 * el horizonte. Ademas, asi coincide con lo que se VE: ninguna estrella cruza
 * el disco brillante.
 *
 * Una estrella absorbida se esconde en el centro con masa 0: deja de tirar de
 * las demas (ya es parte del agujero) y queda invisible dentro de la sombra.
 */
__global__ void kernelAbsorber(Body* __restrict__ pos, Body* __restrict__ vel,
                               int N, int bh, float rHorizonte)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || i == bh) return;
    if (pos[i].w > 500.0f) return;      // el propio agujero negro
    if (pos[i].w < 1e-6f) return;       // telarana congelada o ya absorbida

    float dx = pos[i].x - pos[bh].x;
    float dy = pos[i].y - pos[bh].y;
    float dz = pos[i].z - pos[bh].z;
    if (dx*dx + dy*dy + dz*dz < rHorizonte * rHorizonte) {
        pos[i] = make_float4(pos[bh].x, pos[bh].y, pos[bh].z, 0.0f);
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void kernelAsignarVelGalaxia(Body* __restrict__ pos,
                                        Body* __restrict__ vel,
                                        const Body* __restrict__ velGal,
                                        const int* __restrict__ slot,
                                        int N, int nGal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int k = slot[i];

    if (k >= 0 && k < nGal) {
        vel[i] = velGal[k];              // la galaxia recibe sus velocidades orbitales
        return;
    }

    // ── La telarana pasa a ser DECORADO DE FONDO ──
    // Sus particulas pesan 1.0 (unidades del universo) mientras las estrellas de
    // la galaxia pesan ~0.001-0.006 (unidades de initGalaxy): son ~300x mas
    // pesadas, y quedan 37000 alrededor -> su masa total (~37000) compite con
    // la del agujero negro (50000). Tiraban de la galaxia, la deformaban y
    // arrastraban el bulbo... y el agujero negro, que se coloca en el centro de
    // masa del bulbo, se iba con el.
    // Solucion: masa despreciable (no ejercen gravedad) y quietas. Siguen
    // viendose (kernelCopiaVBO les da su propio tamano de render).
    pos[i].w = 1e-9f;
    vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void kernelMorph(Body* __restrict__ pos,
                            const Body* __restrict__ origen,
                            const Body* __restrict__ destino,
                            const int*  __restrict__ slot,
                            int N, int nGal, float s, float cx, float cy, float cz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int k = slot[i];
    if (k < 0 || k >= nGal) return;    // materia fuera del halo: NO se toca,
                                       // sigue formando la telarana cosmica

    // ── El agujero negro NO viaja ──
    // El hueco 0 (que initGalaxy reserva para el agujero) se lo lleva una
    // particula cualquiera, y si le tocaba una del borde del halo tenia que
    // cruzar toda la galaxia... con retardo por distancia. Resultado: llegaba
    // TARDE y desde un lado, aterrizando en una galaxia ya formada.
    // Ahora nace directamente en el centro del halo. El salto no se ve porque
    // arranca con un 8% de su tamano (uCrecer) y va creciendo ahi mismo.
    if (k == 0) {
        pos[i] = destino[0];
        return;
    }

    // Sin imagen minima: la particula viaja desde donde esta, en linea recta.
    // (kernelMarcarHalo ya se aseguro de que solo entren las que estan cerca
    //  de verdad, sin cruzar fronteras.)
    float ox = origen[i].x, oy = origen[i].y, oz = origen[i].z;
    float dx = ox - cx, dy = oy - cy, dz = oz - cz;

    // Retardo por distancia al centro del halo: el colapso ocurre de dentro
    // hacia fuera, como en un colapso gravitacional real.
    float d  = sqrtf(dx*dx + dy*dy + dz*dz) / RADIO_HALO;
    float retardo = fminf(0.55f, d * 0.55f);

    float t = (s - retardo) / fmaxf(0.05f, 1.0f - retardo);
    t = fminf(1.0f, fmaxf(0.0f, t));
    float e = t * t * (3.0f - 2.0f * t);          // suavizado (smoothstep)

    pos[i].x = ox + (destino[k].x - ox) * e;
    pos[i].y = oy + (destino[k].y - oy) * e;
    pos[i].z = oz + (destino[k].z - oz) * e;
    // la masa tambien transita (el universo es masa 1.0; la galaxia varia)
    pos[i].w = origen[i].w + (destino[k].w - origen[i].w) * e;
}

/*
 * Kernel auxiliar: copia posiciones al VBO de OpenGL para render.
 * devVBO apunta directamente al buffer de la GPU compartido con OpenGL.
 * Sin cudaMemcpy, sin round-trip CPU.
 */
__global__ void kernelCopiaVBO(const Body* __restrict__ pos,
                                float*      __restrict__ devVBO,
                                int N,
                                const int* __restrict__ slot,
                                float atenuacionWeb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // VBO layout: x,y,z,masa (la masa manda el tamaño/brillo del punto)
    devVBO[i*4+0] = pos[i].x;
    devVBO[i*4+1] = pos[i].y;
    devVBO[i*4+2] = pos[i].z;

    // La telarana pasa a segundo plano al formarse la galaxia: si no, sus
    // estrellas se pierden entre los miles de puntos del cubo (se ven iguales).
    //
    // Su tamano de render NO puede salir de la masa: al formarse la galaxia le
    // ponemos masa ~0 para que no ejerza gravedad (competia con el agujero
    // negro). Asi que aqui le damos un tamano propio, independiente de la
    // fisica: separamos "cuanto pesa" de "como se ve".
    float m = pos[i].w;
    bool esTelarana = (slot == NULL) || (slot[i] < 0);

    if (esTelarana) {
        // NUBES COSMICAS: buena parte de la telarana se dibuja como "neblina"
        // (el shader la pinta grande y muy tenue). Al solaparse muchas forman
        // luz continua: los filamentos dejan de ser puntos sueltos y pasan a
        // ser gas difuso, como en las visualizaciones cosmologicas reales.
        //
        // El marcador es la masa < 0.001, que activa esa rama del shader.
        // Es SOLO render: la fisica no lo ve (la telarana esta congelada).
        //
        // 45% neblina + 20% gas + 35% punto: las tres capas juntas dan
        // profundidad (nucleos brillantes sobre un halo difuso).
        int cara = i & 19;
        if      (cara < 9)  m = 0.00055f;              // 45% -> nube difusa grande
        else if (cara < 13) m = 0.005f;                // 20% -> nube media
        else                m = 0.85f * atenuacionWeb; // 35% -> punto nitido
    }
    else if (m <= 0.0f) m = -1.0f;    // absorbida por el agujero: no dibujar

    devVBO[i*4+3] = m;
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
/*
 * Carga las condiciones iniciales del universo (el "Big Bang") desde el binario
 * que genera scripts/exportar_ci.py.
 *
 * Esas CI son cosmologia real: campo gaussiano con espectro de potencia LCDM
 * (transferencia BBKS) + aproximacion de Zel'dovich, el mismo metodo que usan
 * los generadores profesionales (MUSIC / N-GenIC). Se generan en Python con
 * NumPy+FFT y se leen aqui, en vez de reimplementar la FFT en C++.
 *
 * Formato: int32 N | float32 box | float4 pos[N] | float4 vel[N]
 *
 * Devuelve el numero de particulas leidas, o 0 si fallo.
 */
/*
 * Busca en la GPU el halo mas denso y devuelve su centro (host).
 * Devuelve cuantas particulas tiene el halo encontrado.
 */
int buscarHalo(const Body* dPos, int N, float* cx, float* cy, float* cz)
{
    const int NC = HALO_GRID * HALO_GRID * HALO_GRID;
    int* dConteo; float* dCoM;
    CUDA_CHECK(cudaMalloc(&dConteo, NC * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCoM, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(dConteo, 0, NC * sizeof(int)));

    int grid = (N + TILE_SIZE - 1) / TILE_SIZE;
    kernelContarCeldas<<<grid, TILE_SIZE>>>(dPos, N, dConteo);
    CUDA_CHECK(cudaDeviceSynchronize());

    // El host elige la celda mas poblada (NC es pequeno: 4096 enteros).
    //
    // PERO no vale cualquiera: si el halo cae pegado a un borde, su esfera de
    // radio RADIO_HALO se sale de la caja y la galaxia nacería mutilada (falta
    // la materia del otro lado, que no podemos traer sin teletransportarla a
    // traves del universo). Por eso puntuamos cada celda con la poblacion
    // PENALIZADA por lo cerca que esta del borde: preferimos un halo algo menos
    // denso pero bien centrado.
    int* hConteo = (int*)malloc(NC * sizeof(int));
    CUDA_CHECK(cudaMemcpy(hConteo, dConteo, NC * sizeof(int), cudaMemcpyDeviceToHost));

    int mejor = 0;
    float mejorPuntos = -1.0f;
    for (int k = 0; k < NC; k++) {
        int kz = k % HALO_GRID;
        int ky = (k / HALO_GRID) % HALO_GRID;
        int kx = k / (HALO_GRID * HALO_GRID);
        // centro de la celda en coordenadas del mundo
        float px = (kx + 0.5f) / HALO_GRID * UNIVERSE_BOX;
        float py = (ky + 0.5f) / HALO_GRID * UNIVERSE_BOX;
        float pz = (kz + 0.5f) / HALO_GRID * UNIVERSE_BOX;
        // margen hasta el borde mas cercano en los 3 ejes
        float m = fminf(fminf(fminf(px, UNIVERSE_BOX - px), fminf(py, UNIVERSE_BOX - py)),
                        fminf(pz, UNIVERSE_BOX - pz));
        // 1.0 si la esfera cabe entera; baja hasta 0.15 pegado al borde
        float encaje = fminf(1.0f, m / RADIO_HALO);
        float puntos = hConteo[k] * (0.15f + 0.85f * encaje);
        if (puntos > mejorPuntos) { mejorPuntos = puntos; mejor = k; }
    }
    int poblacion = hConteo[mejor];
    free(hConteo);

    int gz = mejor % HALO_GRID;
    int gy = (mejor / HALO_GRID) % HALO_GRID;
    int gx = mejor / (HALO_GRID * HALO_GRID);

    // Centro de masa exacto de esa celda
    CUDA_CHECK(cudaMemset(dCoM, 0, 4 * sizeof(float)));
    kernelCoMCelda<<<grid, TILE_SIZE>>>(dPos, N, gx, gy, gz, dCoM);
    CUDA_CHECK(cudaDeviceSynchronize());

    float hCoM[4];
    CUDA_CHECK(cudaMemcpy(hCoM, dCoM, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    if (hCoM[3] > 0.0f) {
        *cx = hCoM[0] / hCoM[3];
        *cy = hCoM[1] / hCoM[3];
        *cz = hCoM[2] / hCoM[3];
    } else {
        // por si acaso: centro de la caja
        *cx = *cy = *cz = UNIVERSE_BOX * 0.5f;
    }

    cudaFree(dConteo); cudaFree(dCoM);
    return poblacion;
}

int loadUniverse(const char* ruta, Body** hPos, Body** hVel)
{
    FILE* f = fopen(ruta, "rb");
    if (!f) {
        fprintf(stderr, "\nERROR: no se pudo abrir '%s'\n", ruta);
        fprintf(stderr, "Genera las condiciones iniciales con:\n");
        fprintf(stderr, "    python scripts/exportar_ci.py --n 110000 --salida universo_ci.bin\n\n");
        return 0;
    }

    int n = 0;
    float box = 0.0f;
    if (fread(&n, sizeof(int), 1, f) != 1 || fread(&box, sizeof(float), 1, f) != 1) {
        fprintf(stderr, "ERROR: cabecera invalida en '%s'\n", ruta);
        fclose(f);
        return 0;
    }
    if (n <= 0 || n > 2000000) {
        fprintf(stderr, "ERROR: N invalido (%d) en '%s'\n", n, ruta);
        fclose(f);
        return 0;
    }

    *hPos = (Body*)malloc(n * sizeof(Body));
    *hVel = (Body*)malloc(n * sizeof(Body));
    if (!*hPos || !*hVel) { fprintf(stderr, "ERROR: sin memoria\n"); fclose(f); return 0; }

    size_t okP = fread(*hPos, sizeof(Body), n, f);
    size_t okV = fread(*hVel, sizeof(Body), n, f);
    fclose(f);

    if (okP != (size_t)n || okV != (size_t)n) {
        fprintf(stderr, "ERROR: archivo truncado (%zu/%zu de %d)\n", okP, okV, n);
        free(*hPos); free(*hVel);
        return 0;
    }

    printf("Universo cargado: %d particulas, caja = %.1f\n", n, box);
    if (fabsf(box - UNIVERSE_BOX) > 0.01f) {
        fprintf(stderr, "AVISO: la caja del archivo (%.1f) no coincide con UNIVERSE_BOX (%.1f).\n",
                box, UNIVERSE_BOX);
        fprintf(stderr, "       Regenera con --escala %.0f o ajusta UNIVERSE_BOX.\n", UNIVERSE_BOX);
    }
    return n;
}

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

        // Solo guardamos posicion y masa. La velocidad NO se puede calcular
        // todavia: necesita saber cuanta masa hay dentro del radio de cada
        // estrella, y eso solo se sabe cuando estan todas generadas.
        hPos[i] = make_float4(x, y, z, mass);
        hVel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // ─── Velocidades orbitales a partir de la masa encerrada REAL ──────────
    //
    // Antes esto se estimaba con  Menc = MASS_CENTRAL + (N-1)*0.15*(r/22)^2,
    // dando por hecho que cada estrella pesa 0.15. La masa media real es 0.121
    // (medido): la formula inflaba la masa del disco un 24%. Con N grande el
    // error se disimulaba con un factor 0.92 puesto a ojo, pero con N pequeno
    // (p.ej. la galaxia que nace de un halo) el disco pesa poco, el 0.92 ya no
    // compensa nada y deja las estrellas ~8% lentas: la galaxia se deshace.
    //
    // Ahora se SUMAN las masas de verdad: ordenamos por radio y acumulamos.
    // Sin constantes magicas y correcto para cualquier N.
    {
        int  n = N - 1;                       // sin contar el agujero negro
        int* orden = (int*)malloc(n * sizeof(int));
        float* radio = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            const Body& p = hPos[i + 1];
            radio[i] = sqrtf(p.x*p.x + p.z*p.z);   // radio cilindrico (disco en XZ)
            orden[i] = i;
        }
        // ordenar indices por radio (insertion sort seria O(n^2): usamos qsort)
        static float* radioOrden;              // el comparador necesita verlo
        radioOrden = radio;
        qsort(orden, n, sizeof(int), [](const void* a, const void* b) {
            float ra = radioOrden[*(const int*)a];
            float rb = radioOrden[*(const int*)b];
            return (ra < rb) ? -1 : (ra > rb) ? 1 : 0;
        });

        // Masa acumulada: recorriendo de dentro hacia fuera
        float acumulada = MASS_CENTRAL;        // el agujero negro esta en el centro
        for (int k = 0; k < n; k++) {
            int i = orden[k];
            float r = fmaxf(radio[i], 0.05f);
            // La masa de esta estrella no se atrae a si misma: se suma DESPUES
            float Menc = acumulada;
            acumulada += hPos[i + 1].w;

            float vc = sqrtf(G * Menc / r);    // velocidad circular exacta
            float angle = atan2f(hPos[i + 1].z, hPos[i + 1].x);

            // Perturbaciones termicas pequenas
            float dvx = (randf() - 0.5f) * 0.012f;
            float dvy = (randf() - 0.5f) * 0.005f;
            float dvz = (randf() - 0.5f) * 0.012f;

            hVel[i + 1] = make_float4(-sinf(angle)*vc + dvx,
                                       dvy,
                                       cosf(angle)*vc + dvz,
                                       0.0f);
        }
        printf("Galaxia: %d estrellas, masa estelar total %.0f (agujero negro %.0f)\n",
               n, acumulada - MASS_CENTRAL, (float)MASS_CENTRAL);
        free(orden); free(radio);
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

    // masa negativa = particula ABSORBIDA por el agujero negro. La marca la
    // pone kernelCopiaVBO; aqui simplemente no se dibuja.
    if (inMass < 0.0) { gl_PointSize = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

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
// ═══════════════════════════════════════════════════════════════════════════
//  AGUJERO NEGRO estilo GARGANTUA — con trazado de rayos relativista
// ═══════════════════════════════════════════════════════════════════════════
//
// La version anterior era una composicion 2D (un circulo negro con bandas
// pintadas encima) y por eso parecia una calcomania. Gargantua se ve
// tridimensional porque la luz SE CURVA de verdad alrededor de la esfera: ves
// el disco rodearla por detras, subir por encima y volver por debajo.
//
// Aqui cada pixel lanza un rayo desde la camara y lo INTEGRA paso a paso bajo
// la gravedad del agujero:
//   - si el rayo cae dentro del horizonte  -> negro (la sombra)
//   - si cruza el plano del disco          -> recoge su luz
//   - si escapa                            -> se va al fondo
// Como los rayos se curvan, el disco de detras aparece por arriba y por abajo:
// eso es la lente gravitacional, y sale sola de la integracion.
static const char* BH_VS_SOURCE = R"glsl(
#version 330 core
layout(location = 0) in vec2 inCorner;   // esquina del quad, en [-1,1]
uniform mat4  uMVP;
uniform vec3  uCentro;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uRadio;
out vec3 vWorld;    // posicion de este fragmento en el mundo
void main() {
    vWorld = uCentro + uCamRight * (inCorner.x * uRadio)
                     + uCamUp    * (inCorner.y * uRadio);
    gl_Position = uMVP * vec4(vWorld, 1.0);
}
)glsl";

static const char* BH_FS_SOURCE = R"glsl(
#version 330 core
in vec3 vWorld;
out vec4 fragColor;
uniform vec3  uCamPos;
uniform vec3  uCentro;
uniform float uRadio;
uniform float uTime;

// Gas a millones de grados: blanco en el interior, salmon hacia fuera.
vec3 discoColor(float t) {
    vec3 blanco = vec3(1.00, 0.99, 0.96);
    vec3 crema  = vec3(1.00, 0.88, 0.70);
    vec3 salmon = vec3(1.00, 0.62, 0.38);
    vec3 c = mix(blanco, crema, smoothstep(0.0, 0.45, t));
    return mix(c, salmon, smoothstep(0.45, 1.0, t));
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }
float ruido(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
               mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
}

void main() {
    // Rayo desde la camara hacia este pixel, en coordenadas del agujero negro
    vec3 p = uCamPos - uCentro;
    vec3 v = normalize(vWorld - uCamPos);

    // El fenomeno ocupa el 45% central del quad: deja margen para que los arcos
    // lenteados (que caen mas afuera que el disco) no los corte el cuadrado.
    float Rs   = uRadio * 0.085;         // radio del horizonte de sucesos
    float GM   = Rs * 0.5;               // GM/c^2 = Rs/2
    float rIn  = Rs * 2.4;               // borde interior del disco
    float rOut = uRadio * 0.40;          // borde exterior del disco
    float lejos = uRadio * 1.15;         // fuera de aqui ya no pasa nada

    vec3  col = vec3(0.0);
    float alpha = 0.0;
    bool  cayo = false;

    // Saltamos el vacio: llevamos el rayo justo al borde de la zona de accion.
    // Sin esto gastaba todos los pasos viajando y NO LLEGABA al agujero cuando
    // la camara estaba lejos: el agujero negro se volvia invisible de lejos.
    float haciaBH = -dot(p, v);                    // distancia al punto mas cercano
    float b2 = dot(p, p) - haciaBH * haciaBH;      // parametro de impacto^2
    if (b2 > lejos * lejos) discard;               // el rayo pasa de largo
    float entrada = haciaBH - sqrt(max(lejos * lejos - b2, 0.0));
    if (entrada > 0.0) p += v * entrada;

    // Jitter: si todos los rayos avanzan en fase, los pasos discretos dibujan
    // anillos concentricos. Desfasar cada pixel lo disuelve en ruido fino.
    float jitter = ruido(gl_FragCoord.xy * 0.7);

    for (int i = 0; i < 300; i++) {
        float r = length(p);

        if (r < Rs) { cayo = true; break; }        // cayo dentro del horizonte
        if (r > lejos * 1.6 && dot(v, p) > 0.0) break;   // ya escapo

        // Paso adaptativo: fino cerca (mucha curvatura), amplio lejos. Con el
        // jitter no deja bandeo, y asi 300 pasos alcanzan desde cualquier
        // distancia.
        float dt = clamp((r - Rs) * 0.16, uRadio * 0.004, uRadio * 0.030);
        if (i == 0) dt *= (0.4 + 0.6 * jitter);

        // Curvatura del rayo. Para fotones la desviacion es 1.5x la newtoniana
        // (relatividad general): por eso la sombra se ve mayor que el horizonte.
        vec3 a  = -normalize(p) * (1.5 * GM / (r * r));
        vec3 vN = normalize(v + a * dt);
        vec3 pN = p + vN * dt;

        // ¿Cruzo el plano del disco? (el disco vive en el plano XZ, y=0)
        if (p.y * pN.y < 0.0) {
            float f   = p.y / (p.y - pN.y);        // punto exacto del cruce
            vec3  hit = mix(p, pN, f);
            float rd  = length(hit.xz);
            if (rd > rIn && rd < rOut) {
                float t = (rd - rIn) / (rOut - rIn);

                // Bordes suaves: sin esto el disco corta en seco y se ve el filo
                float suave = smoothstep(0.0, 0.10, t) * (1.0 - smoothstep(0.72, 1.0, t));

                // Turbulencia girando (kepleriana: dentro va mas rapido)
                float ang = atan(hit.z, hit.x);
                float vel = uTime * 1.0 / max(pow(rd / rIn, 1.5), 0.35);
                float n   = ruido(vec2(ang * 2.2 + vel, rd / uRadio * 20.0));
                float brillo = (0.55 + 0.75 * n) * pow(1.0 - t, 0.7);

                // Doppler beaming con la velocidad tangencial real del gas.
                // Suave a proposito: con factor 1.25 el lado que se aleja caia
                // a 0.15 y desaparecia (medio disco "borrado"). En Interstellar
                // tambien lo atenuaron: la fisica real deja un lado casi negro
                // y visualmente queda mal.
                vec3  tang = normalize(vec3(-hit.z, 0.0, hit.x));
                float dop  = clamp(1.0 + 0.32 * dot(tang, -vN), 0.74, 1.5);

                // Corrimiento gravitacional: se apaga cerca del horizonte
                float grav = clamp(1.0 - Rs / rd, 0.25, 1.0);

                float emi = brillo * dop * grav * suave * 1.6;
                // (1-alpha): lo que ya recogimos por delante tapa lo de detras
                col   += discoColor(t) * emi * (1.0 - alpha);
                alpha += emi * 0.8 * (1.0 - alpha);
            }
        }

        p = pN;  v = vN;
    }

    // Si el rayo cayo al agujero, lo de DETRAS no se ve... pero la luz que
    // recogio ANTES de caer (el disco que pasa por DELANTE de la esfera) si:
    // esta entre la camara y el horizonte. Por eso NO se borra col.
    if (cayo) alpha = 1.0;

    if (alpha < 0.004) discard;
    fragColor = vec4(col, alpha);
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
#define NUM_DUST           220     // nubes de polvo en los brazos (menos que en la
                                   // galaxia original: esta nace del halo y tiene ~27k
                                   // estrellas en vez de 200k, el polvo la tapaba)

static const char* BG_VS_SOURCE = R"glsl(
#version 330 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in float inBrightness;
uniform mat4 uMVP;
// uCamPos: el fondo es un SKYBOX -> viaja con la camara, asi queda siempre
// "infinitamente lejos" y nunca te metes dentro de el. Sin esto, las estrellas
// de fondo (esfera de radio 70-190 alrededor del origen) quedaban DENTRO de la
// caja del universo [0,60]^3 y aparecian de golpe al volar.
uniform vec3 uCamPos;
out float vBrightness;
void main() {
    gl_Position  = uMVP * vec4(inPos + uCamPos, 1.0);
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
// uFade: 0 = universo (sin fondo: estamos mirando el cosmos entero)
//        1 = galaxia (fondo completo de estrellas y galaxias lejanas)
uniform float uFade;
void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv) * 2.0;
    if (uFade <= 0.001) discard;

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
        a *= uFade;
        fragColor = vec4(c * a, a);
        return;
    }

    float core = 1.0 - smoothstep(0.4, 1.0, d);
    float a    = core * mix(0.25, 0.85, vBrightness) * uFade;
    if (a < 0.01) discard;
    // Las estrellas de fondo mas brillantes reciben un pequeño extra
    // para cruzar el umbral del bloom y tener su propio glow sutil.
    float boost = smoothstep(0.85, 1.0, vBrightness) * 1.4 * uFade;
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
// uOrigen: a que punto se anclan estos billboards.
//   - POLVO de los brazos  -> el halo (viaja con la galaxia)
//   - NEBULOSAS de fondo   -> la camara (son decorado lejano, radio 60-150:
//     si se anclan al halo acaban ENCIMA de la camara como manchas gigantes)
uniform vec3 uOrigen;
out vec2  vUV;
out float vHue;
out float vSeed;
void main() {
    vec3 worldPos = inCenter + uOrigen
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
uniform float uFade;   // 0 = universo (sin decorado), 1 = galaxia

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

    // uFade: las nebulosas son decorado de la galaxia. Entran progresivamente
    // durante el morph en vez de aparecer de golpe al 100%.
    float a = density * 0.22 * uFade;   // translucido pero visible
    if (a < 0.002) discard;
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
uniform float uFade;   // 0 = universo (sin decorado), 1 = galaxia

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

    // Si el polvo se ve como bolas opacas NO es la opacidad: es que el blending
    // esta apagado (el bloom lo desactiva cada frame). Ver el glEnable(GL_BLEND)
    // al principio del render.
    float a = density * 0.26 * uFade;
    if (a < 0.002) discard;
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

// Camara: arranca fuera de la caja del universo, mirando hacia su centro.
// (La galaxia original arrancaba en {0,10,38}, pero el universo vive en
//  [0,UNIVERSE_BOX]^3, asi que hay que colocarla en otro sitio.)
static Camera   g_cam   = {UNIVERSE_BOX*0.5f, UNIVERSE_BOX*0.5f, -UNIVERSE_BOX*0.75f,
                           0.0f, 0.0f, 18.0f, 0, 0, 0};
// N, paused, showHelp, simTime, steps, fps, lastFPSTime, fpsFrames,
// acto, actoT, autoAvance, zoomT, haloX/Y/Z, haloPop, zoomDesde*, fade, solicitarActo
static AppState g_app   = {N_DEFAULT, 0, 1, 0.0, 0, 0.0f, 0.0, 0,
                           ACTO_UNIVERSO, 0.0f, 1, 0.0f, 0.0f, 0, 0,
                           0,0,0, 0, 0,0,0,0,0, 0.0f, 0};

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
    // G: avanza de acto (universo -> zoom al halo -> galaxia)
    if (key == GLFW_KEY_G)
        g_app.solicitarActo = 1;
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

// Ruta del binario con las condiciones iniciales (se puede pasar por argumento)
static const char* g_ciPath = "universo_ci.bin";

int main(int argc, char** argv)
{
    // Sin buffer: asi los mensajes de la narrativa salen al instante en la
    // consola (por defecto se quedarian retenidos al no ser una terminal).
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) g_ciPath = argv[1];
    int N = 0;   // lo decide el archivo de condiciones iniciales

    printf("=== El Nacimiento del Universo — N-body cosmologico en tiempo real ===\n");
    printf("Condiciones iniciales: %s\n", g_ciPath);
    printf("Caja periodica: %.1f x %.1f x %.1f\n", UNIVERSE_BOX, UNIVERSE_BOX, UNIVERSE_BOX);
    printf("TILE_SIZE = %d\n", TILE_SIZE);
    printf("Controles: WASD=moverse, arrastrar=mirar, SPACE/CTRL=subir/bajar,\n"
           "           SHIFT=velocidad, scroll=ajustar velocidad, P=pausa, ESC=salir\n\n");

    // ── GLFW / OpenGL ──────────────────────────────────────────────────────
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    char title[128];
    snprintf(title, sizeof(title), "El Nacimiento del Universo — N-body cosmologico");
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
    // MODO UNIVERSO: las condiciones iniciales vienen del binario que genera
    // exportar_ci.py (Big Bang LCDM + Zel'dovich). N lo decide el archivo.
    Body *hPos = NULL, *hVel = NULL;
    int nCargados = loadUniverse(g_ciPath, &hPos, &hVel);
    if (!nCargados) { glfwTerminate(); return 1; }
    N = nCargados;
    g_app.N = N;
    size_t bytes = (size_t)N * sizeof(Body);

    // Activamos la fisica del universo: caja periodica y gravedad difusa
    // (sin agujero negro dominante). Estos simbolos los lee el kernel.
    {
        bool  per  = true;
        float gU   = G_UNIVERSE;
        float epsU = EPSILON2_UNIVERSE;
        CUDA_CHECK(cudaMemcpyToSymbol(g_periodic, &per,  sizeof(bool)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_G,        &gU,   sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_EPS2,     &epsU, sizeof(float)));
    }

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

    // Buffers de la transicion halo -> galaxia (origen y destino del morph)
    Body *dPosOrigen, *dPosDestino, *dVelDestino;
    CUDA_CHECK(cudaMalloc(&dPosOrigen, bytes));
    CUDA_CHECK(cudaMalloc(&dPosDestino, bytes));
    CUDA_CHECK(cudaMalloc(&dVelDestino, bytes));   // velocidades orbitales de la galaxia
    // slot[i]: hueco de la galaxia que ocupara la particula i (-1 = se queda
    // como telarana cosmica). dContador cuenta cuantas caen en el halo.
    int *dSlot, *dContador, *dBhIdx;
    CUDA_CHECK(cudaMalloc(&dSlot, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dContador, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dBhIdx, sizeof(int)));

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
    GLint  bgLocFade = glGetUniformLocation(bgProg, "uFade");
    GLint  bgLocCam  = glGetUniformLocation(bgProg, "uCamPos");

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
    GLint  nebLocOrigen  = glGetUniformLocation(nebProg,  "uOrigen");
    GLint  dustLocOrigen = glGetUniformLocation(dustProg, "uOrigen");
    GLint  nebLocFade    = glGetUniformLocation(nebProg,  "uFade");
    GLint  dustLocFade   = glGetUniformLocation(dustProg, "uFade");
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
    GLint  bhLocMVP    = glGetUniformLocation(bhProg, "uMVP");
    GLint  bhLocTime   = glGetUniformLocation(bhProg, "uTime");
    GLint  bhLocCentro = glGetUniformLocation(bhProg, "uCentro");
    GLint  bhLocRight  = glGetUniformLocation(bhProg, "uCamRight");
    GLint  bhLocUp     = glGetUniformLocation(bhProg, "uCamUp");
    GLint  bhLocRadio  = glGetUniformLocation(bhProg, "uRadio");
    GLint  bhLocCam    = glGetUniformLocation(bhProg, "uCamPos");

    // Quad del agujero negro: 6 vertices con las esquinas en [-1,1].
    // El vertex shader lo orienta hacia la camara y lo escala con uRadio.
    GLuint bhVAO, bhVBO;
    {
        float esquinas[12] = { -1,-1,  1,-1,  1, 1,   -1,-1,  1, 1,  -1, 1 };
        glGenVertexArrays(1, &bhVAO);
        glGenBuffers(1, &bhVBO);
        glBindVertexArray(bhVAO);
        glBindBuffer(GL_ARRAY_BUFFER, bhVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(esquinas), esquinas, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }

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

        // ═════════════════════════════════════════════════════════════════
        //  NARRATIVA: universo -> zoom al halo -> galaxia
        //  Avanza SOLA con el tiempo; G la adelanta.
        // ═════════════════════════════════════════════════════════════════
        g_app.actoT += dt;

        if (g_app.autoAvance && !g_app.solicitarActo) {
            // ACTO 1: tras SEG_UNIVERSO la camara va sola al halo
            if (g_app.acto == ACTO_UNIVERSO && g_app.actoT >= SEG_UNIVERSO)
                g_app.solicitarActo = 1;
            // ACTO 2: cuando el vuelo termina, la galaxia se forma sola
            else if (g_app.acto == ACTO_ZOOM && g_app.zoomT >= 1.0f &&
                     g_app.actoT >= SEG_ESPERA_HALO)
                g_app.solicitarActo = 1;
        }

        if (g_app.solicitarActo) {
            g_app.solicitarActo = 0;
            g_app.actoT = 0.0f;

            if (g_app.acto == ACTO_UNIVERSO) {
                // ── ACTO 1 -> 2: buscar el halo mas denso y empezar el zoom ──
                g_app.haloPop = buscarHalo(dPos, N, &g_app.haloX, &g_app.haloY, &g_app.haloZ);
                printf("\n[ACTO 2] Halo mas denso: (%.1f, %.1f, %.1f) con %d particulas\n",
                       g_app.haloX, g_app.haloY, g_app.haloZ, g_app.haloPop);
                printf("         Volando hasta el...\n");
                // guardamos desde donde arranca la camara para interpolar
                g_app.zoomDesdeX = g_cam.posX; g_app.zoomDesdeY = g_cam.posY;
                g_app.zoomDesdeZ = g_cam.posZ;
                g_app.zoomDesdeYaw = g_cam.yaw; g_app.zoomDesdePitch = g_cam.pitch;
                g_app.zoomT = 0.0f;
                g_app.acto = ACTO_ZOOM;
            }
            else if (g_app.acto == ACTO_ZOOM) {
                // ── ACTO 2 -> 3: el halo EMPIEZA a reorganizarse en galaxia ──
                // No sustituimos de golpe: preparamos el destino y dejamos que
                // kernelMorph lleve cada particula suavemente hasta su sitio.
                printf("\n[ACTO 3] El halo colapsa y forma la galaxia...\n");

                // 1. Que particulas estan dentro del halo -> esas seran la galaxia.
                //    Las de fuera se quedan quietas: la telarana cosmica sigue ahi.
                CUDA_CHECK(cudaMemset(dContador, 0, sizeof(int)));
                kernelMarcarHalo<<<gridSim, TILE_SIZE>>>(dPos, N, g_app.haloX,
                                                         g_app.haloY, g_app.haloZ,
                                                         dSlot, dContador, dBhIdx);
                CUDA_CHECK(cudaMemcpy(&g_app.nGal, dContador, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&g_app.bhIdx, dBhIdx, sizeof(int), cudaMemcpyDeviceToHost));
                printf("         Materia dentro del halo (r=%.0f): %d particulas\n",
                       RADIO_HALO, g_app.nGal);
                printf("         El resto (%d) sigue siendo telarana cosmica\n", N - g_app.nGal);

                // 2. Generamos una galaxia con EXACTAMENTE esas particulas
                if (g_app.nGal > 1024) {
                    initGalaxy(hPos, hVel, g_app.nGal);
                    for (int i = 0; i < g_app.nGal; i++) {   // centrar en el halo
                        hPos[i].x += g_app.haloX;
                        hPos[i].y += g_app.haloY;
                        hPos[i].z += g_app.haloZ;
                    }
                    CUDA_CHECK(cudaMemcpy(dPosOrigen,  dPos, bytes, cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(dPosDestino, hPos,
                                          (size_t)g_app.nGal * sizeof(Body),
                                          cudaMemcpyHostToDevice));
                    // Las velocidades ORBITALES que calculo initGalaxy. Sin
                    // ellas la galaxia no rota y se deshace (bug historico).
                    CUDA_CHECK(cudaMemcpy(dVelDestino, hVel,
                                          (size_t)g_app.nGal * sizeof(Body),
                                          cudaMemcpyHostToDevice));
                    g_app.morphT = 0.0f;
                    g_app.acto = ACTO_MORPH;
                } else {
                    printf("         AVISO: halo demasiado pobre, no se forma galaxia.\n");
                }
            }
        }

        // ── Animacion del zoom (ACTO 2) ──────────────────────────────────
        // Solo mientras dura el vuelo (zoomT <= 1). Al aterrizar se marca con
        // zoomT = 2 y dejamos de tocar la camara para no pelear con el WASD.
        if (g_app.acto == ACTO_ZOOM && g_app.zoomT < 1.5f) {
            g_app.zoomT += dt * 0.14f;                 // ~7 segundos de vuelo (pausado)
            float t = fminf(1.0f, g_app.zoomT);
            float s = t * t * (3.0f - 2.0f * t);        // arranque y frenada suaves

            // destino: a una distancia prudente del halo, mirandolo de frente
            // A ~2.2 radios del centro y algo elevado: se ve la galaxia entera
            // y de tres cuartos (antes la camara acababa DENTRO del disco).
            float dstX = g_app.haloX - RADIO_HALO * 1.6f;
            float dstY = g_app.haloY + RADIO_HALO * 0.85f;
            float dstZ = g_app.haloZ - RADIO_HALO * 1.6f;
            g_cam.posX = g_app.zoomDesdeX + (dstX - g_app.zoomDesdeX) * s;
            g_cam.posY = g_app.zoomDesdeY + (dstY - g_app.zoomDesdeY) * s;
            g_cam.posZ = g_app.zoomDesdeZ + (dstZ - g_app.zoomDesdeZ) * s;

            // orientamos la camara hacia el halo
            float dx = g_app.haloX - g_cam.posX;
            float dy = g_app.haloY - g_cam.posY;
            float dz = g_app.haloZ - g_cam.posZ;
            float yawObj   = atan2f(dx, dz);
            float pitchObj = atan2f(dy, sqrtf(dx*dx + dz*dz));
            g_cam.yaw   = g_app.zoomDesdeYaw   + (yawObj   - g_app.zoomDesdeYaw)   * s;
            g_cam.pitch = g_app.zoomDesdePitch + (pitchObj - g_app.zoomDesdePitch) * s;

            if (t >= 1.0f && g_app.zoomT < 1.5f) {
                // Aterrizamos: a partir de aqui contamos la espera antes de que
                // la galaxia se forme sola (o el usuario pulse G).
                printf("         Llegamos al halo. La galaxia se formara sola...\n");
                g_app.zoomT = 2.0f;       // marca "ya llegamos" (>1 y no repite)
                g_app.actoT = 0.0f;       // reinicia el contador de espera
            }
        }

        // ── Movimiento libre de camara (WASD + subir/bajar) ───────────────
        // Bloqueado solo mientras la camara vuela sola hacia el halo.
        if (!(g_app.acto == ACTO_ZOOM && g_app.zoomT < 1.5f)) {
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

        // ── TRANSICION halo -> galaxia (ACTO_MORPH) ───────────────────────
        // Durante la transicion no hay fisica: cada particula viaja suavemente
        // desde el halo hasta su sitio en la galaxia (kernelMorph).
        if (g_app.acto == ACTO_MORPH && !g_app.paused) {
            g_app.morphT += dt / SEG_MORPH;
            float s = fminf(1.0f, g_app.morphT);

            kernelMorph<<<gridSim, TILE_SIZE>>>(dPos, dPosOrigen, dPosDestino, dSlot,
                                                N, g_app.nGal, s,
                                                g_app.haloX, g_app.haloY, g_app.haloZ);

            if (g_app.morphT >= 1.0f) {
                // Cada estrella recibe su velocidad orbital -> la galaxia rota
                kernelAsignarVelGalaxia<<<gridSim, TILE_SIZE>>>(dPos, dVel, dVelDestino,
                                                                dSlot, N, g_app.nGal);
                // Ya esta formada: activamos la fisica de la galaxia
                bool  per  = false;          // la galaxia no vive en caja periodica
                float gG   = G;
                float epsG = EPSILON2;
                CUDA_CHECK(cudaMemcpyToSymbol(g_periodic, &per,  sizeof(bool)));
                CUDA_CHECK(cudaMemcpyToSymbol(g_G,        &gG,   sizeof(float)));
                CUDA_CHECK(cudaMemcpyToSymbol(g_EPS2,     &epsG, sizeof(float)));
                g_app.acto = ACTO_GALAXIA;
                g_app.actoT = 0.0f;
                printf("         Galaxia formada. Vuela libre con WASD.\n");
            }
        }

        // ── Paso de fisica en GPU ─────────────────────────────────────────
        if (!g_app.paused && g_app.acto != ACTO_MORPH) {
            // 1. Calcular fuerzas (O(N^2) con tiling)
            kernelFuerzas<<<gridSim, TILE_SIZE, shMem>>>(dPos, dAcc, N);

            // 2. Integrar (Leapfrog). En modo universo la caja se envuelve.
            // El universo evoluciona con pasos mas largos que la galaxia
            float dtSim = (g_app.acto == ACTO_GALAXIA) ? DT : DT_UNIVERSE;
            kernelLeapfrog<<<gridSim, TILE_SIZE>>>(dPos, dVel, dAcc, N, dtSim);

            // En el UNIVERSO no hay agujero negro: la estructura emerge sola de
            // las fluctuaciones. Pero en la GALAXIA hay que recolocar el
            // agujero supermasivo en el centro de masa del bulbo cada frame, o
            // deriva visiblemente respecto de las estrellas.
            if (g_app.acto == ACTO_GALAXIA) {
                CUDA_CHECK(cudaMemset(dCoM, 0, 4 * sizeof(float)));
                kernelAcumCoM<<<gridSim, TILE_SIZE>>>(dPos, N, dCoM, g_app.bhIdx, dSlot);
                kernelFijarBHalCoM<<<1, 1>>>(dPos, dCoM, g_app.bhIdx);
                // Se traga lo que se acerca demasiado. El radio (0.42) es el
                // del DISCO DE ACRECION, no el del horizonte (0.085): antes
                // absorbia solo dentro del horizonte y las estrellas que
                // pasaban entre medias cruzaban el disco brillante a la vista.
                //
                // Y es fisicamente correcto: una estrella que se acerca tanto a
                // un agujero negro supermasivo la despedazan las fuerzas de
                // MAREA mucho antes de cruzar el horizonte, y sus restos
                // alimentan el disco. No necesita llegar al horizonte para
                // desaparecer.
                kernelAbsorber<<<gridSim, TILE_SIZE>>>(dPos, dVel, N, g_app.bhIdx,
                                                       RADIO_BH * 0.42f);
            }

            g_app.steps++;
            g_app.simTime += dtSim;
        }

        // ── Fundido del decorado de galaxia (nebulosas, polvo, fondo) ─────
        // Todo eso es de la GALAXIA: en el universo no debe verse, y debe
        // ENTRAR progresivamente durante el morph, no aparecer de golpe.
        float fadeDeco;
        {
            float m = (g_app.acto == ACTO_GALAXIA) ? 1.0f :
                      (g_app.acto == ACTO_MORPH)   ? fminf(1.0f, g_app.morphT) : 0.0f;
            fadeDeco = m * m * (3.0f - 2.0f * m);   // suavizado
        }

        // ── Copiar posiciones al VBO (CUDA→OpenGL sin pasar por CPU) ──────
        float* devVBOPtr;
        size_t vboSize;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBORes, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
                       (void**)&devVBOPtr, &vboSize, cudaVBORes));

        // En el ACTO 1 no hay slots todavia -> se pasa NULL y no se atenua nada.
        // Al formarse la galaxia la telarana baja a un 18% de su brillo.
        {
            const int* slotRender = (g_app.acto == ACTO_UNIVERSO || g_app.acto == ACTO_ZOOM)
                                    ? NULL : dSlot;
            float aten = 1.0f - 0.82f * fadeDeco;
            kernelCopiaVBO<<<gridCopy, 256>>>(dPos, devVBOPtr, N, slotRender, aten);
        }

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

        // ── Estado de dibujo de la escena (SE FIJA CADA FRAME) ────────────
        // No se puede dar por hecho: el bloom hace glDisable(GL_BLEND) mas
        // abajo, y la pasada del agujero negro (que antes lo reactivaba) ahora
        // es condicional. Sin esto, a partir del segundo frame los billboards
        // se dibujaban OPACOS: las nubes de polvo salian como bolas negras
        // tapando las estrellas.
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);   // aditivo: las zonas densas brillan
        glDisable(GL_DEPTH_TEST);            // la transparencia aditiva no usa depth

        // Fondo de estrellas lejanas (se dibuja primero, de fondo)
        glUseProgram(bgProg);
        glUniformMatrix4fv(bgLocMVP, 1, GL_FALSE, mvp);
        // El fondo de estrellas/galaxias lejanas es decorado de la GALAXIA:
        // en el ACTO 1 estamos mirando el cosmos entero, no tiene sentido.
        // Entra progresivamente durante el morph.
        {
            glUniform1f(bgLocFade, fadeDeco);
            // el skybox se mueve con la camara -> horizonte inalcanzable
            glUniform3f(bgLocCam, g_cam.posX, g_cam.posY, g_cam.posZ);
        }
        glBindVertexArray(bgVAO);
        glDrawArrays(GL_POINTS, 0, NUM_BG_TOTAL);

        // Nebulosas (billboards con textura): orientadas hacia la camara.
        // Son decorado de la GALAXIA (colocadas para una escala de ~22 unidades),
        // asi que no se dibujan en el universo: taparian la telarana cosmica.
        // Aparecen ya durante el morph, conforme la galaxia se forma.
        if (g_app.acto == ACTO_GALAXIA || g_app.acto == ACTO_MORPH) {
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
            // Skybox: las nebulosas viajan con la camara -> quedan siempre
            // lejanas, como el fondo de estrellas. Ancladas al halo se metian
            // dentro de la escena (radio 60-150, tamano 18-44).
            glUniform3f(nebLocOrigen, g_cam.posX, g_cam.posY, g_cam.posZ);
            glUniform1f(nebLocFade, fadeDeco);
            glUniform3f(nebLocRight, rx, ry, rz);
            glUniform3f(nebLocUp, ux, uy, uz);
            glBindVertexArray(nebVAO);
            glDrawArrays(GL_TRIANGLES, 0, NUM_NEBULAE * 6);
        }

        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp);
        // Estos uniforms estan calibrados para la GALAXIA (masas variadas, radio
        // ~24). En el UNIVERSO todas las particulas tienen masa 1.0 y la caja
        // mide UNIVERSE_BOX, asi que hay que reescalarlos o saldrian todas al
        // tamano maximo y con el degradado de color equivocado.
        // Durante el morph estos valores se interpolan tambien, si no habria un
        // salto visual (los puntos cambiarian de tamano/color de golpe).
        {
            float e = fadeDeco;
            glUniform1f(locMaxMass, 3.2f  + (0.92f - 3.2f) * e);
            glUniform1f(locMaxDist, UNIVERSE_BOX + (24.0f - UNIVERSE_BOX) * e);
        }

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, N);

        // Polvo estelar de los brazos (billboards con textura procedural).
        // Igual que las nebulosas: es decorado de la galaxia, no del universo.
        if (g_app.acto == ACTO_GALAXIA || g_app.acto == ACTO_MORPH) {
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
            glUniform3f(dustLocOrigen, g_app.haloX, g_app.haloY, g_app.haloZ);
            glUniform1f(dustLocFade, fadeDeco);
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
        // Solo en el acto GALAXIA: en el universo no hay agujero negro central
        // (el cuerpo 0 es una particula mas de materia oscura).
        // ── Agujero negro: billboard ENCIMA del bloom ──
        // Nace como semilla en el centro del halo en cuanto empieza el morph y
        // crece ahi mientras la materia cae. Su tamano es un RADIO DEL MUNDO,
        // asi la perspectiva funciona sola (antes, con point-sprite y clamp,
        // cerca se veia pequeno y lejos grande).
        if (g_app.acto == ACTO_GALAXIA || g_app.acto == ACTO_MORPH) {
            float crecer = (g_app.acto == ACTO_GALAXIA) ? 1.0f
                         : fminf(1.0f, g_app.morphT / 0.75f);
            // 8% al nacer -> tamano final; ^0.6 = crece rapido y luego se asienta
            float radio = RADIO_BH * (0.08f + 0.92f * powf(crecer, 0.6f));

            // Su posicion la mueve la GPU (kernelFijarBHalCoM): la leemos.
            Body bhPos;
            CUDA_CHECK(cudaMemcpy(&bhPos, dPos + g_app.bhIdx, sizeof(Body),
                                  cudaMemcpyDeviceToHost));

            float fx, fy, fz;
            camForward(g_cam.yaw, g_cam.pitch, &fx, &fy, &fz);
            float rx = -fz, ry = 0.0f, rz = fx;
            float rl = sqrtf(rx*rx + rz*rz);
            if (rl > 1e-6f) { rx /= rl; rz /= rl; }
            float ux = ry*fz - rz*fy, uy = rz*fx - rx*fz, uz = rx*fy - ry*fx;

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(bhProg);
            glUniformMatrix4fv(bhLocMVP, 1, GL_FALSE, mvp);
            glUniform1f(bhLocTime, (float)glfwGetTime());
            glUniform3f(bhLocCentro, bhPos.x, bhPos.y, bhPos.z);
            glUniform3f(bhLocRight, rx, ry, rz);
            glUniform3f(bhLocUp, ux, uy, uz);
            glUniform1f(bhLocRadio, radio);
            glUniform3f(bhLocCam, g_cam.posX, g_cam.posY, g_cam.posZ);
            glBindVertexArray(bhVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        }

        // ── FPS en titulo ─────────────────────────────────────────────────
        g_app.fpsFrames++;
        double now = glfwGetTime();
        if (now - g_app.lastFPSTime >= 0.5) {
            g_app.fps = (float)(g_app.fpsFrames / (now - g_app.lastFPSTime));
            g_app.fpsFrames = 0;
            g_app.lastFPSTime = now;
            if (g_app.acto == ACTO_UNIVERSO) {
                float faltan = SEG_UNIVERSO - g_app.actoT;
                snprintf(title, sizeof(title),
                         "ACTO 1: La telarana cosmica se teje... (zoom en %.0fs) | N=%d | %.0f FPS",
                         faltan > 0 ? faltan : 0.0f, N, g_app.fps);
            } else if (g_app.acto == ACTO_ZOOM) {
                snprintf(title, sizeof(title),
                         "ACTO 2: Volando al halo (%d particulas) | N=%d | %.0f FPS",
                         g_app.haloPop, N, g_app.fps);
            } else if (g_app.acto == ACTO_MORPH) {
                snprintf(title, sizeof(title),
                         "ACTO 3: El halo colapsa en galaxia... %.0f%% | N=%d | %.0f FPS",
                         fminf(1.0f, g_app.morphT) * 100.0f, N, g_app.fps);
            } else {
                snprintf(title, sizeof(title),
                         "ACTO 3: La galaxia ha nacido | N=%d | %.0f FPS | paso=%ld",
                         N, g_app.fps, g_app.steps);
            }
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
