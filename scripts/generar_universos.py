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
import traceback
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
    ap.add_argument("--local", action="store_true",
                    help="cada PC guarda en su propio disco (por defecto: los datos se "
                         "envian al maestro por MPI)")
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

    # cada proceso simula sus universos, capturando errores para poder decir
    # DESPUES en que PC fallo y por que (si no, el fallo se pierde)
    t0 = time.perf_counter()
    errores = []
    mis_resultados = []
    for seed in mis_semillas:
        try:
            res = simular(args.n, args.steps, seed)
            c = contraste_densidad(res["pos_final"])
            if args.local:
                # modo antiguo: cada PC guarda en SU disco
                ruta = os.path.join(args.salida, f"universo_seed{seed:02d}.npz")
                np.savez_compressed(ruta, **res)
            else:
                # modo por defecto: guardamos para enviarselo al maestro por MPI.
                # Usamos float32 para que viaje la mitad de datos por la red.
                mis_resultados.append({
                    "seed": seed,
                    "pos_inicial": res["pos_inicial"].astype(np.float32),
                    "pos_final": res["pos_final"].astype(np.float32),
                    "tiempo_s": res["tiempo_s"],
                    "n": res["n"],
                    "steps": res["steps"],
                })
            print(f"[rank {rank}] seed={seed:02d} listo  ({res['tiempo_s']:.1f}s, contraste={c:.2f})", flush=True)
        except Exception as e:
            errores.append({
                "seed": seed,
                "tipo": type(e).__name__,
                "mensaje": str(e),
                "traza": traceback.format_exc(),
            })
            print(f"[rank {rank}] ERROR en seed={seed:02d}: {type(e).__name__}: {e}", flush=True)
    mi_tiempo = time.perf_counter() - t0

    # el maestro recoge tiempos y errores de cada rank
    tiempos = comm.gather(mi_tiempo, root=0)
    todos_errores = comm.gather({"rank": rank, "host": host, "errores": errores}, root=0)

    # RECOGIDA DE DATOS: cada trabajador manda sus universos al maestro por MPI,
    # asi los .npz terminan todos en el PC del maestro (que entrena el surrogate)
    if not args.local:
        t_env = time.perf_counter()
        recogidos = comm.gather(mis_resultados, root=0)
        if rank == 0:
            guardados = 0
            for lote in recogidos:
                for r in lote:
                    ruta = os.path.join(args.salida, f"universo_seed{r['seed']:02d}.npz")
                    np.savez_compressed(ruta, **{k: v for k, v in r.items() if k != "seed"})
                    guardados += 1
            mb = sum(len(l) for l in recogidos) * args.n * 3 * 4 * 2 / 1e6
            print(f"\n  Datos recogidos por MPI: {guardados} universos "
                  f"(~{mb:.1f} MB) en {time.perf_counter()-t_env:.1f}s -> {args.salida}/")
    comm.Barrier()

    # informe de errores: quien fallo y por que
    if rank == 0:
        con_fallas = [x for x in todos_errores if x["errores"]]
        if con_fallas:
            print("\n" + "!" * 60)
            print("  SE PRODUJERON ERRORES — detalle por PC")
            print("!" * 60)
            for x in con_fallas:
                print(f"\n  rank {x['rank']} (host '{x['host']}') fallo en {len(x['errores'])} universo(s):")
                for e in x["errores"]:
                    print(f"    - seed {e['seed']:02d}: {e['tipo']}: {e['mensaje']}")
            print("\n  (traza completa del primer error:)")
            print(con_fallas[0]["errores"][0]["traza"])
            print("!" * 60)
        else:
            print("\n  Sin errores: los", size, "procesos completaron su trabajo.")
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
