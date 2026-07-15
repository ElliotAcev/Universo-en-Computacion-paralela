"""
generar_universos.py — Generación DISTRIBUIDA de universos (envoltorio MPI)

El maestro reparte SEMILLAS entre los procesos (no manda universos enteros, solo
un número). Cada proceso genera y simula sus universos con esa semilla y guarda
los resultados. Así se crea el dataset del surrogate en paralelo entre los 3 PCs.

Reparto tolerante a las GPUs desiguales: se puede dar más carga a los ranks
rápidos con --pesos (ver el modelo de balanceo).

Uso local (simula 3 procesos en tu PC):
    mpiexec -n 3 python scripts/generar_universos.py --total 12 --n 1000 --steps 200

Uso en el clúster:
    mpiexec -hosts 3 <IP1> <IP2> <IP3> -n 3 python scripts/generar_universos.py --total 60 --n 1500 --steps 300

Salida:
    dataset/universo_seedNN.npz  (uno por semilla)
"""

import argparse
import os
import time
import numpy as np
from mpi4py import MPI

from simular_universo import simular, contraste_densidad


def repartir_semillas(total, size, pesos=None):
    """Devuelve, para cada rank, la lista de semillas que le tocan.
    Si 'pesos' es None, reparto equitativo (round-robin). Si hay pesos
    (velocidad relativa de cada GPU), se da más carga a los ranks rápidos."""
    semillas = list(range(total))
    if pesos is None:
        # round-robin: rank r se queda con semillas r, r+size, r+2*size, ...
        return [semillas[r::size] for r in range(size)]
    # reparto proporcional a los pesos
    pesos = np.array(pesos, float)
    frac = pesos / pesos.sum()
    cortes = np.floor(np.cumsum(frac) * total).astype(int)
    cortes[-1] = total
    reparto, ini = [], 0
    for c in cortes:
        reparto.append(semillas[ini:c])
        ini = c
    return reparto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=12, help="nº total de universos a generar")
    ap.add_argument("--n", type=int, default=1000, help="tamaño N de cada universo")
    ap.add_argument("--steps", type=int, default=200, help="pasos de simulación")
    ap.add_argument("--salida", type=str, default="dataset", help="carpeta de salida")
    ap.add_argument("--pesos", type=str, default="", help="pesos por rank separados por coma, ej: 1.0,0.6,0.4")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = MPI.Get_processor_name()

    pesos = [float(x) for x in args.pesos.split(",")] if args.pesos else None
    if pesos and len(pesos) != size:
        if rank == 0:
            print(f"AVISO: --pesos tiene {len(pesos)} valores pero hay {size} procesos. Ignorado.")
        pesos = None

    # el rank 0 decide el reparto y lo comparte
    reparto = repartir_semillas(args.total, size, pesos) if rank == 0 else None
    reparto = comm.bcast(reparto, root=0)
    mis_semillas = reparto[rank]

    if rank == 0:
        print(f"Generando {args.total} universos (N={args.n}, steps={args.steps}) entre {size} procesos")
        for r, s in enumerate(reparto):
            print(f"  rank {r}: {len(s)} universos -> semillas {s}")
        os.makedirs(args.salida, exist_ok=True)
    comm.Barrier()

    print(f"[rank {rank}@{host}] me tocan {len(mis_semillas)} universos", flush=True)

    # cada proceso simula sus universos
    t0 = time.perf_counter()
    for seed in mis_semillas:
        res = simular(args.n, args.steps, seed)
        ruta = os.path.join(args.salida, f"universo_seed{seed:02d}.npz")
        np.savez_compressed(ruta, **res)
        c = contraste_densidad(res["pos_final"])
        print(f"[rank {rank}] seed={seed:02d} listo  ({res['tiempo_s']:.1f}s, contraste={c:.2f})", flush=True)
    mi_tiempo = time.perf_counter() - t0

    # el maestro recoge los tiempos de cada rank (dato para el balanceo)
    tiempos = comm.gather(mi_tiempo, root=0)
    comm.Barrier()
    if rank == 0:
        print("\n=== Resumen de la generacion distribuida ===")
        for r, t in enumerate(tiempos):
            print(f"  rank {r}: {len(reparto[r])} universos en {t:.1f}s")
        t_paralelo = max(tiempos)       # el lote acaba cuando acaba el mas lento
        t_serie = sum(tiempos)          # el mismo trabajo en un solo proceso
        print(f"  Tiempo total (manda el mas lento): {t_paralelo:.1f}s")
        print(f"  Si fuera 1 solo proceso (~suma):   {t_serie:.1f}s")
        if t_paralelo > 0:
            print(f"  Speedup medido: {t_serie/t_paralelo:.2f}x")
        # guardamos las MEDICIONES REALES para el resumen de resultados
        np.savez(os.path.join(args.salida, "cluster_stats.npz"),
                 procesos=size,
                 universos=args.total,
                 tiempos_por_rank=np.array(tiempos),
                 t_paralelo=t_paralelo,
                 t_serie=t_serie,
                 speedup=t_serie / t_paralelo if t_paralelo > 0 else 0.0)


if __name__ == "__main__":
    main()
