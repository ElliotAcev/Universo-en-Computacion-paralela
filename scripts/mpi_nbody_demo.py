"""
mpi_nbody_demo.py

Demo MPI para el trabajo final: simulación N-body directa.
Ejemplo:
    mpiexec -n 4 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20

Qué paraleliza:
    Cada rank calcula la aceleración de un subconjunto de cuerpos.
    El estado global se sincroniza en cada paso mediante MPI.
"""

import argparse
import time
import numpy as np
from mpi4py import MPI

G = 1.0
EPS2 = 1e-3
DT = 0.002


def init_bodies(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    r = rng.gamma(shape=2.0, scale=0.8, size=n)
    r = np.clip(r, 0.05, 5.5)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    z = rng.normal(0.0, 0.05, size=n)
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta), z]).astype(np.float64)
    mass = np.full(n, 100.0 / n, dtype=np.float64)

    enclosed = 80.0 * (r**3) / ((r**2 + 1.4**2) ** 1.5) + 35.0
    v = np.sqrt(G * enclosed / np.maximum(r, 0.08))
    vel = np.column_stack([-np.sin(theta) * v, np.cos(theta) * v, rng.normal(0, 0.01, size=n)]).astype(np.float64)
    return pos, vel, mass


def local_acceleration(pos, mass, start, end):
    local_pos = pos[start:end]
    diff = pos[None, :, :] - local_pos[:, None, :]
    dist2 = np.sum(diff * diff, axis=2) + EPS2
    idx = np.arange(start, end)
    dist2[np.arange(end - start), idx] = np.inf
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    acc = G * np.sum(diff * (mass[None, :, None] * inv_dist3[:, :, None]), axis=1)
    return acc


def split_range(n, rank, size):
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return start, start + count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        pos, vel, mass = init_bodies(args.n, args.seed)
    else:
        pos = np.empty((args.n, 3), dtype=np.float64)
        vel = np.empty((args.n, 3), dtype=np.float64)
        mass = np.empty(args.n, dtype=np.float64)

    comm.Bcast(pos, root=0)
    comm.Bcast(vel, root=0)
    comm.Bcast(mass, root=0)

    start, end = split_range(args.n, rank, size)
    local_count = end - start

    if rank == 0:
        print("Demo MPI directo N-body")
        print(f"Procesos MPI: {size}")
        print(f"Cuerpos: {args.n}")
        print(f"Pasos: {args.steps}")

    print(f"rank={rank} cuerpos_asignados={local_count} range=[{start}, {end})", flush=True)

    comm.Barrier()
    t0 = time.perf_counter()

    counts = np.array([split_range(args.n, r, size)[1] - split_range(args.n, r, size)[0] for r in range(size)], dtype=np.int32)
    displs = np.array([split_range(args.n, r, size)[0] * 3 for r in range(size)], dtype=np.int32)
    counts3 = counts * 3

    for _ in range(args.steps):
        local_acc = local_acceleration(pos, mass, start, end)
        vel[start:end] += local_acc * DT
        pos[start:end] += vel[start:end] * DT

        sendbuf_pos = pos[start:end].reshape(-1)
        sendbuf_vel = vel[start:end].reshape(-1)
        comm.Allgatherv(sendbuf_pos, [pos.reshape(-1), counts3, displs, MPI.DOUBLE])
        comm.Allgatherv(sendbuf_vel, [vel.reshape(-1), counts3, displs, MPI.DOUBLE])

    comm.Barrier()
    elapsed = time.perf_counter() - t0

    times = comm.gather(elapsed, root=0)
    if rank == 0:
        print(f"Tiempo máximo entre ranks: {max(times):.6f} s")
        print(f"Tiempo mínimo entre ranks: {min(times):.6f} s")
        print(f"Tiempo promedio entre ranks: {sum(times)/len(times):.6f} s")
        print("Tiempo paralelo efectivo = tiempo máximo entre ranks")


if __name__ == "__main__":
    main()
