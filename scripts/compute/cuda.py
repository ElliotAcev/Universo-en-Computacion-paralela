"""Backend CUDA (GPU NVIDIA) con un kernel N-body escrito a mano.

Se usa en PC1 (RTX 4060) y PC2 (RTX 3050).

POR QUE UN KERNEL PROPIO Y NO PYTORCH:
La primera version usaba tensores de PyTorch, que para calcular las fuerzas
construye arrays (N,N,3) en memoria -> O(N^2) de RAM. Medido en el cluster real,
eso hacia que la RTX 4060 tardase 0.43 s/universo mientras la AMD (con el kernel
OpenCL escrito a mano) tardaba 0.12 s. No era la GPU: era el algoritmo.

Este kernel es el mismo de main_cuda_referencia.cu: un hilo por particula y
SHARED-MEMORY TILING (cada bloque carga TILE_SIZE cuerpos a memoria compartida
y todos los hilos del bloque los reusan, reduciendo accesos a VRAM). Memoria
O(N) en vez de O(N^2).

CuPy compila el kernel en tiempo de ejecucion, igual que PyOpenCL con el suyo.
"""

import numpy as np

TILE = 256

KERNEL = r'''
extern "C" __global__ void aceleraciones(
    const float* __restrict__ pos,    // N*3 (x,y,z por particula)
    const float* __restrict__ masa,   // N
    float* __restrict__ acc,          // N*3 (salida)
    const int n, const float box, const float eps2, const float g)
{
    extern __shared__ float4 tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    if (i < n) { xi = pos[i*3+0]; yi = pos[i*3+1]; zi = pos[i*3+2]; }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int numTiles = (n + blockDim.x - 1) / blockDim.x;
    for (int t = 0; t < numTiles; t++) {
        // Carga cooperativa: cada hilo trae UN cuerpo a memoria compartida
        int j = t * blockDim.x + threadIdx.x;
        if (j < n) tile[threadIdx.x] = make_float4(pos[j*3+0], pos[j*3+1], pos[j*3+2], masa[j]);
        else       tile[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();

        // Todo el bloque reusa esos cuerpos desde shared memory
        for (int k = 0; k < blockDim.x; k++) {
            int jg = t * blockDim.x + k;
            if (jg >= n || jg == i) continue;

            float dx = tile[k].x - xi;
            float dy = tile[k].y - yi;
            float dz = tile[k].z - zi;

            // caja periodica: imagen minima
            dx -= box * rintf(dx / box);
            dy -= box * rintf(dy / box);
            dz -= box * rintf(dz / box);

            float d2   = dx*dx + dy*dy + dz*dz + eps2;
            float inv  = rsqrtf(d2);          // instruccion nativa de la GPU
            float inv3 = inv * inv * inv;
            float s    = g * tile[k].w * inv3;

            ax += s * dx;  ay += s * dy;  az += s * dz;
        }
        __syncthreads();
    }

    if (i < n) { acc[i*3+0] = ax; acc[i*3+1] = ay; acc[i*3+2] = az; }
}
'''


class CUDABackend:
    nombre = "CUDA (kernel propio)"

    def __init__(self, config=None):
        import cupy as cp   # si no esta -> excepcion -> auto prueba OpenCL
        self.cp = cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no hay GPU CUDA utilizable en este PC")

        config = config or {}
        self.G = np.float32(config.get("G", 1.0))
        self.eps2 = np.float32(config.get("EPS2", 2.5e-3))
        self.box = np.float32(config.get("BOX", 1.0))

        props = cp.cuda.runtime.getDeviceProperties(0)
        self.dispositivo = props["name"].decode()
        self.kernel = cp.RawKernel(KERNEL, "aceleraciones")

        # prueba real: si la GPU no puede operar, fallamos AQUI y no a mitad del lote
        _ = int(cp.zeros(8, dtype=cp.float32).sum())

    def aceleraciones(self, pos, masa):
        cp = self.cp
        n = np.int32(len(masa))
        d_pos = cp.asarray(np.ascontiguousarray(pos, dtype=np.float32).ravel())
        d_masa = cp.asarray(np.ascontiguousarray(masa, dtype=np.float32))
        d_acc = cp.empty_like(d_pos)

        bloques = (int(n) + TILE - 1) // TILE
        shmem = TILE * 16          # TILE * sizeof(float4)
        self.kernel((bloques,), (TILE,),
                    (d_pos, d_masa, d_acc, n, self.box, self.eps2, self.G),
                    shared_mem=shmem)

        return cp.asnumpy(d_acc).reshape(-1, 3).astype(np.float64)
