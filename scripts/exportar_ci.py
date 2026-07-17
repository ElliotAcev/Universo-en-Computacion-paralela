"""
exportar_ci.py — Exporta condiciones iniciales del universo para la app CUDA

Genera el "Big Bang" con el mismo metodo cosmologico validado (espectro de
potencia LCDM + aproximacion de Zel'dovich) y lo vuelca a un archivo binario
que lee universo_cuda.cu.

Asi la app en tiempo real arranca con fisica real, sin reimplementar la FFT
en C++.

Formato del binario (little-endian):
    int32   N            numero de particulas
    float32 box          tamano de la caja
    float32 pos[N*4]     x,y,z,masa  por particula
    float32 vel[N*4]     vx,vy,vz,0  por particula

Uso:
    python scripts/exportar_ci.py --n 100000 --salida universo_ci.bin
"""

import argparse
import struct
import numpy as np

from simular_universo import condiciones_iniciales, BOX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="numero de particulas (se ajusta a un cubo perfecto)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--salida", type=str, default="universo_ci.bin")
    ap.add_argument("--escala", type=float, default=60.0,
                    help="escala el universo a las unidades de la app CUDA")
    args = ap.parse_args()

    print(f"Generando condiciones iniciales (Big Bang LCDM + Zel'dovich)...")
    pos, vel, masa, n = condiciones_iniciales(args.n, args.seed)
    lado = round(n ** (1 / 3))
    print(f"  Particulas: {n}  ({lado}^3)")

    # llevamos la caja unitaria a las unidades del motor CUDA
    k = args.escala / BOX
    pos = pos * k
    vel = vel * k
    box = BOX * k

    # masa por particula: la app usa pos.w como masa
    masa_particula = 1.0
    masa = np.full(n, masa_particula, dtype=np.float32)

    # empaquetamos como float4 (x,y,z,masa) y (vx,vy,vz,0)
    p4 = np.zeros((n, 4), dtype=np.float32)
    p4[:, :3] = pos
    p4[:, 3] = masa
    v4 = np.zeros((n, 4), dtype=np.float32)
    v4[:, :3] = vel

    with open(args.salida, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(struct.pack("<f", box))
        f.write(p4.tobytes())
        f.write(v4.tobytes())

    mb = (8 + n * 32) / 1e6
    print(f"  Caja: {box:.1f} unidades")
    print(f"  Velocidad media: {np.linalg.norm(vel, axis=1).mean():.4f}")
    print(f"  Guardado: {args.salida}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
