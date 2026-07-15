"""
diagnostico.py — Revisa la salud de CADA PC del cluster y dice quien falla

Cada proceso revisa SU propia maquina (version de Python, librerias, GPU, rutas)
y le manda el informe al maestro, que imprime una tabla con el estado de los 3
PCs. Si algo falla, dice EXACTAMENTE en que PC y cual es el problema.

Uso local (simula 3 procesos):
    mpiexec -n 3 python scripts/diagnostico.py

Uso en el cluster real:
    mpiexec -hosts 3 <IP-PC1> <IP-PC2> <IP-PC3> -n 3 python scripts/diagnostico.py

Salida: tabla con OK / FALLA por cada PC y el detalle de cada problema.
"""

import os
import platform
import sys
import traceback


def revisar_paquete(nombre):
    """Comprueba si una libreria esta instalada y devuelve su version."""
    try:
        mod = __import__(nombre)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except Exception as e:
        return False, str(e)[:60]


def revisar_gpu():
    """Detecta la GPU y el backend disponible (CUDA o OpenCL)."""
    # CUDA via torch
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    # OpenCL via pyopencl (PC3 AMD)
    try:
        import pyopencl as cl
        disp = [d.name for p in cl.get_platforms() for d in p.get_devices()]
        if disp:
            return True, f"OpenCL: {disp[0]}"
    except Exception:
        pass
    return False, "sin GPU detectada (se usara CPU)"


def revisar_este_pc():
    """Reune el informe de ESTA maquina. Nunca lanza excepcion: si algo peta,
    lo captura y lo reporta como problema."""
    informe = {"host": platform.node(), "problemas": [], "datos": {}}

    try:
        # 1. Version de Python (debe ser 3.11.x en los 3)
        v = sys.version_info
        informe["datos"]["python"] = f"{v.major}.{v.minor}.{v.micro}"
        if (v.major, v.minor) != (3, 11):
            informe["problemas"].append(
                f"Python {v.major}.{v.minor} — se esperaba 3.11 (debe coincidir en los 3 PCs)")

        # 2. Librerias necesarias
        for paq in ["numpy", "mpi4py"]:
            ok, info = revisar_paquete(paq)
            informe["datos"][paq] = info if ok else "FALTA"
            if not ok:
                informe["problemas"].append(f"falta la libreria '{paq}' -> pip install {paq}")

        # 3. GPU / backend
        gpu_ok, gpu_info = revisar_gpu()
        informe["datos"]["gpu"] = gpu_info
        if not gpu_ok:
            informe["problemas"].append("no se detecto GPU (funcionara, pero lento en CPU)")

        # 4. El codigo del proyecto esta presente?
        aqui = os.path.dirname(os.path.abspath(__file__))
        informe["datos"]["ruta"] = aqui
        for archivo in ["simular_universo.py", "generar_universos.py"]:
            if not os.path.exists(os.path.join(aqui, archivo)):
                informe["problemas"].append(
                    f"falta el archivo '{archivo}' -> haz 'git pull' para tener el mismo codigo")

        # 5. Se puede escribir la carpeta de salida?
        salida = os.path.join(os.path.dirname(aqui), "dataset")
        try:
            os.makedirs(salida, exist_ok=True)
            prueba = os.path.join(salida, ".permiso_test")
            with open(prueba, "w") as f:
                f.write("ok")
            os.remove(prueba)
        except Exception as e:
            informe["problemas"].append(f"no se puede escribir en dataset/ -> {e}")

        # 6. La simulacion arranca de verdad?
        try:
            sys.path.insert(0, aqui)
            from simular_universo import simular
            simular(64, 2, seed=0)      # universo minusculo, solo para probar
            informe["datos"]["simulacion"] = "OK"
        except Exception as e:
            informe["datos"]["simulacion"] = "FALLA"
            informe["problemas"].append(f"la simulacion no corre -> {type(e).__name__}: {e}")

    except Exception:
        informe["problemas"].append("error inesperado revisando este PC:\n" + traceback.format_exc())

    return informe


def main():
    try:
        from mpi4py import MPI
    except Exception as e:
        print(f"ERROR CRITICO en {platform.node()}: no se pudo importar mpi4py -> {e}")
        print("Solucion: instala MS-MPI y luego 'pip install mpi4py'")
        return

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    informe = revisar_este_pc()
    informe["rank"] = rank
    todos = comm.gather(informe, root=0)

    if rank != 0:
        return

    print("\n" + "=" * 66)
    print("  DIAGNOSTICO DEL CLUSTER — estado de cada PC")
    print("=" * 66)

    con_fallas = []
    for inf in todos:
        estado = "OK" if not inf["problemas"] else "PROBLEMAS"
        marca = "[OK]" if not inf["problemas"] else "[!!]"
        print(f"\n{marca} rank {inf['rank']}  —  host '{inf['host']}'  —  {estado}")
        for clave, valor in inf["datos"].items():
            print(f"       {clave:12s}: {valor}")
        if inf["problemas"]:
            con_fallas.append(inf)
            for p in inf["problemas"]:
                print(f"       -> PROBLEMA: {p}")

    print("\n" + "=" * 66)
    if not con_fallas:
        print(f"  TODO OK — los {size} PCs estan listos para trabajar.")
    else:
        print(f"  {len(con_fallas)} de {size} PCs tienen problemas:")
        for inf in con_fallas:
            print(f"    - rank {inf['rank']} ({inf['host']}): {len(inf['problemas'])} problema(s)")
        print("\n  Arregla esos PCs y vuelve a correr este diagnostico.")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
