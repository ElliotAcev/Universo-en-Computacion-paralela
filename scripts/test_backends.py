"""
test_backends.py — Comprueba que CPU / CUDA / OpenCL calculan LO MISMO

Critico para el cluster: si PC3 (OpenCL) calculara una fisica distinta a PC1/PC2
(CUDA), los universos del dataset serian incompatibles entre si.

Compara cada backend contra el de CPU (referencia) y mide su velocidad.

Uso:
    python scripts/test_backends.py --n 1000
"""

import argparse
import time
import numpy as np

from compute import create_backend

CONFIG = {"G": 1.0, "EPS2": 2.5e-3, "BOX": 1.0}


def cronometrar(backend, pos, masa, repes=3):
    backend.aceleraciones(pos, masa)          # calentamiento
    t0 = time.perf_counter()
    for _ in range(repes):
        acc = backend.aceleraciones(pos, masa)
    return acc, (time.perf_counter() - t0) / repes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    pos = rng.uniform(0, CONFIG["BOX"], size=(args.n, 3))
    masa = np.full(args.n, 1.0 / args.n)

    print(f"Comparando backends con N={args.n}\n")

    # referencia: CPU
    cpu = create_backend("cpu", CONFIG)
    acc_ref, t_cpu = cronometrar(cpu, pos, masa)
    print(f"  {cpu.nombre:16s} [{cpu.dispositivo}]")
    print(f"      tiempo: {t_cpu*1000:8.2f} ms   (referencia)\n")

    ok_global = True
    for nombre in ("cuda", "opencl"):
        try:
            b = create_backend(nombre, CONFIG)
        except Exception as e:
            print(f"  {nombre.upper():16s} no disponible: {type(e).__name__}: {e}\n")
            continue

        acc, t = cronometrar(b, pos, masa)
        # comparamos contra la referencia de CPU
        err_abs = np.abs(acc - acc_ref).max()
        escala = np.abs(acc_ref).max()
        err_rel = err_abs / escala if escala > 0 else 0.0
        # float32 en GPU vs float64 en CPU -> toleramos ~1e-4 relativo
        ok = err_rel < 1e-3
        ok_global &= ok

        print(f"  {b.nombre:16s} [{b.dispositivo}]")
        print(f"      tiempo: {t*1000:8.2f} ms   ({t_cpu/t:5.1f}x vs CPU)")
        print(f"      error relativo vs CPU: {err_rel:.2e}  -> {'OK' if ok else 'DIFERENTE!'}\n")

    print("=" * 56)
    if ok_global:
        print("  Todos los backends disponibles calculan la MISMA fisica.")
    else:
        print("  AVISO: algun backend difiere demasiado. Revisar el kernel.")
    print("=" * 56)


if __name__ == "__main__":
    main()
