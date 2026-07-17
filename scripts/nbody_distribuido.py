"""
nbody_distribuido.py — MODO B: UN universo gigante repartido entre los nodos

A diferencia de generar_universos.py (Modo A, donde cada PC simula universos
completos e independientes), aqui los nodos colaboran en UNA SOLA simulacion:

    - El universo se reparte: cada nodo POSEE un trozo de las particulas.
    - Cada paso, cada nodo calcula la aceleracion de SU trozo... contra TODAS.
    - Luego todos comparten sus posiciones nuevas (Allgatherv) y se sincronizan.

Esto es descomposicion de dominio con sincronizacion por paso: el paralelismo
distribuido "de verdad", el que hace posible simular un universo que NO cabe en
una sola GPU.

OJO — POR QUE HACE FALTA COMPARTIR TODO Y NO SOLO LAS FRONTERAS:
la gravedad es de LARGO ALCANCE: cada particula siente a todas las demas, no
solo a sus vecinas. Las "particulas fantasma" de solo la frontera valen para
fuerzas de corto alcance (gas/SPH) o si usas Barnes-Hut con multipolos.

Este script mide POR SEPARADO el tiempo de calculo y el de comunicacion, que es
justo lo que revela el muro de la latencia.

Uso local (3 procesos en un PC, red ~0 ms):
    mpiexec -n 3 python scripts/nbody_distribuido.py --n 20000 --steps 50

Uso en el cluster real (3 PCs por internet):
    mpiexec -env MPICH_NETMASK 100.64.0.0/255.192.0.0 -pwd ... -hosts 3 IP1 IP2 IP3 \
        -wdir C:\\universo C:\\universo\\venv\\Scripts\\python.exe \
        scripts\\nbody_distribuido.py --n 20000 --steps 50
"""

import argparse
import time
import numpy as np
from mpi4py import MPI

from simular_universo import condiciones_iniciales, contraste_densidad, BOX, G, EPS2, DT
from compute import create_backend


def repartir(n, size):
    """Reparte n particulas entre size nodos. Devuelve (inicio, cuantas) por nodo."""
    base, resto = divmod(n, size)
    tramos, ini = [], 0
    for r in range(size):
        c = base + (1 if r < resto else 0)
        tramos.append((ini, c))
        ini += c
    return tramos


def main():
    ap = argparse.ArgumentParser(description="Modo B: un universo repartido entre nodos")
    ap.add_argument("--n", type=int, default=20000, help="particulas del universo (se ajusta a un cubo)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", type=str, default="auto")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    host = MPI.Get_processor_name()

    # ── Condiciones iniciales: las genera el rank 0 y las reparte ──
    if rank == 0:
        pos, vel, masa, n = condiciones_iniciales(args.n, args.seed)
        print(f"MODO B — un universo de {n} particulas repartido entre {size} nodos")
        print(f"Pasos: {args.steps}\n")
    else:
        pos = vel = masa = None
        n = None
    n = comm.bcast(n, root=0)

    if rank != 0:
        pos = np.empty((n, 3), dtype=np.float64)
        vel = np.empty((n, 3), dtype=np.float64)
        masa = np.empty(n, dtype=np.float64)
    comm.Bcast(pos, root=0)
    comm.Bcast(vel, root=0)
    comm.Bcast(masa, root=0)

    backend = create_backend(args.backend, {"G": G, "EPS2": EPS2, "BOX": BOX})
    tramos = repartir(n, size)
    inicio, cuantas = tramos[rank]
    print(f"[rank {rank}@{host}] {backend.nombre} [{backend.dispositivo}] "
          f"| particulas {inicio}..{inicio+cuantas} ({cuantas})", flush=True)

    # Para Allgatherv: cuantos floats manda cada nodo y donde van
    counts = np.array([c * 3 for _, c in tramos], dtype=np.int32)
    displs = np.array([i * 3 for i, _ in tramos], dtype=np.int32)

    comm.Barrier()
    t_calculo = 0.0
    t_comunicacion = 0.0
    t0 = time.perf_counter()

    for paso in range(args.steps):
        # ── 1. CALCULO: cada nodo acelera SU trozo (contra todas) ──
        tc = time.perf_counter()
        acc = backend.aceleraciones_rango(pos, masa, inicio, cuantas)
        vel[inicio:inicio+cuantas] += acc * DT
        pos[inicio:inicio+cuantas] += vel[inicio:inicio+cuantas] * DT
        pos[inicio:inicio+cuantas] %= BOX          # caja periodica
        t_calculo += time.perf_counter() - tc

        # ── 2. COMUNICACION: todos comparten sus posiciones nuevas ──
        # Aqui es donde muerde la latencia: ocurre en CADA paso.
        tm = time.perf_counter()
        envio = np.ascontiguousarray(pos[inicio:inicio+cuantas]).reshape(-1)
        comm.Allgatherv(envio, [pos.reshape(-1), counts, displs, MPI.DOUBLE])
        t_comunicacion += time.perf_counter() - tm

    comm.Barrier()
    total = time.perf_counter() - t0

    # ── Informe ──
    calc = comm.gather(t_calculo, root=0)
    comu = comm.gather(t_comunicacion, root=0)
    tot = comm.gather(total, root=0)

    if rank == 0:
        print(f"\n=== Resultado (Modo B, {size} nodos, N={n}, {args.steps} pasos) ===")
        for r in range(size):
            print(f"  rank {r}: calculo {calc[r]:6.2f}s | comunicacion {comu[r]:6.2f}s "
                  f"| total {tot[r]:6.2f}s")
        tmax = max(tot)
        cmax = max(calc)
        mmax = max(comu)
        print(f"\n  Tiempo total          : {tmax:.2f}s")
        print(f"  Calculo (max)         : {cmax:.2f}s  ({cmax/tmax*100:.0f}%)")
        print(f"  Comunicacion (max)    : {mmax:.2f}s  ({mmax/tmax*100:.0f}%)")
        print(f"  Por paso              : {tmax/args.steps*1000:.1f} ms "
              f"(calculo {cmax/args.steps*1000:.1f} ms, red {mmax/args.steps*1000:.1f} ms)")
        mb = n * 3 * 8 / 1e6
        print(f"  Datos por paso        : {mb:.1f} MB compartidos entre todos")
        print(f"  Contraste final       : {contraste_densidad(pos):.2f}")
        if mmax > cmax:
            print(f"\n  >> MURO DE LA LATENCIA: la red se lleva el {mmax/tmax*100:.0f}% del tiempo.")
            print(f"     Repartir NO compensa aqui: la comunicacion cuesta mas que el calculo.")
        else:
            print(f"\n  >> El calculo domina ({cmax/tmax*100:.0f}%): repartir SI compensa.")


if __name__ == "__main__":
    main()
