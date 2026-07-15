"""
hola_mpi.py — Test de humo del clúster

Confirma que MPI funciona y que los procesos se comunican. Cada proceso (rank)
imprime quién es y en qué computador corre.

Uso local (simula 3 procesos en tu PC):
    mpiexec -n 3 python scripts/hola_mpi.py

Uso en el clúster (los 3 PCs por sus IP de Tailscale):
    mpiexec -hosts 3 <IP-PC1> <IP-PC2> <IP-PC3> -n 3 python scripts/hola_mpi.py

Éxito = ves una línea por cada proceso, cada una con su número y su host.
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()

print(f"Hola desde el rank {rank} de {size} — corriendo en el host '{host}'", flush=True)

# pequeña prueba de comunicación: el rank 0 saluda y todos responden
comm.Barrier()
if rank == 0:
    print(f"\nLos {size} procesos se comunicaron correctamente. Clúster OK.", flush=True)
