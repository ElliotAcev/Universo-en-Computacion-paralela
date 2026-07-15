"""Backend OpenCL (GPU AMD, y tambien NVIDIA/Intel).

Este es el que permite que PC3 (AMD Radeon) use SU GPU, ya que CUDA solo
funciona en NVIDIA. Aqui esta el kernel real del N-body escrito en OpenCL C:
un hilo por particula, que recorre todas las demas sumando la gravedad.

Nota: NVIDIA tambien expone OpenCL, asi que este backend se puede PROBAR en
PC1/PC2 antes de que PC3 lo use.
"""

import numpy as np

# Un hilo por particula. Cada hilo suma la atraccion de todas las demas,
# aplicando imagen minima para respetar la caja periodica.
KERNEL = """
__kernel void aceleraciones(
    __global const float *pos,    // N*3 (x,y,z por particula)
    __global const float *masa,   // N
    __global float *acc,          // N*3 (salida)
    const int n,
    const float box,
    const float eps2,
    const float g)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float xi = pos[i*3+0], yi = pos[i*3+1], zi = pos[i*3+2];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int j = 0; j < n; j++) {
        if (j == i) continue;                 // nadie se atrae a si mismo

        float dx = pos[j*3+0] - xi;
        float dy = pos[j*3+1] - yi;
        float dz = pos[j*3+2] - zi;

        // imagen minima: la copia mas cercana a traves de los bordes
        dx -= box * round(dx / box);
        dy -= box * round(dy / box);
        dz -= box * round(dz / box);

        float d2 = dx*dx + dy*dy + dz*dz + eps2;
        float inv = rsqrt(d2);
        float inv3 = inv * inv * inv;
        float s = g * masa[j] * inv3;

        ax += s * dx;  ay += s * dy;  az += s * dz;
    }
    acc[i*3+0] = ax;  acc[i*3+1] = ay;  acc[i*3+2] = az;
}
"""


class OpenCLBackend:
    nombre = "OpenCL"

    def __init__(self, config=None):
        import pyopencl as cl   # si falta -> excepcion -> auto pasa a CPU
        self.cl = cl

        # buscamos un dispositivo GPU de verdad (no basta con importar la libreria)
        dispositivo = None
        for plataforma in cl.get_platforms():
            gpus = plataforma.get_devices(device_type=cl.device_type.GPU)
            if gpus:
                dispositivo = gpus[0]
                break
        if dispositivo is None:
            raise RuntimeError("no hay ninguna GPU con OpenCL en este PC")

        config = config or {}
        self.G = np.float32(config.get("G", 1.0))
        self.eps2 = np.float32(config.get("EPS2", 2.5e-3))
        self.box = np.float32(config.get("BOX", 1.0))

        self.dispositivo = dispositivo.name
        self.ctx = cl.Context([dispositivo])
        self.cola = cl.CommandQueue(self.ctx)
        self.programa = cl.Program(self.ctx, KERNEL).build()   # compila el kernel
        # guardamos el kernel una sola vez (recrearlo en cada llamada es caro)
        self.kernel = cl.Kernel(self.programa, "aceleraciones")

    def aceleraciones(self, pos, masa):
        cl = self.cl
        n = np.int32(len(masa))
        pos32 = np.ascontiguousarray(pos, dtype=np.float32).ravel()
        masa32 = np.ascontiguousarray(masa, dtype=np.float32)
        acc32 = np.empty_like(pos32)

        mf = cl.mem_flags
        b_pos = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos32)
        b_masa = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masa32)
        b_acc = cl.Buffer(self.ctx, mf.WRITE_ONLY, acc32.nbytes)

        self.kernel(self.cola, (int(n),), None,
                    b_pos, b_masa, b_acc, n, self.box, self.eps2, self.G)
        cl.enqueue_copy(self.cola, acc32, b_acc)
        self.cola.finish()

        return acc32.reshape(-1, 3).astype(np.float64)
