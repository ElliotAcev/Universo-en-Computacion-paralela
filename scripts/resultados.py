"""
resultados.py — Resumen de las metricas del proyecto para la presentacion

Junta en una sola salida los tres resultados clave:
  1. CLUSTER   : speedup de la generacion distribuida (3 procesos vs 1).
  2. BALANCEO  : mejora del reparto aprendido vs el ingenuo (Modelo 1).
  3. SURROGATE : aceleracion del emulador neuronal vs la simulacion (Modelo 2).

Uso:
    python scripts/resultados.py
"""

import glob
import os
import time
import numpy as np


def medir_cluster(carpeta="dataset"):
    """El speedup MEDIDO en la corrida de referencia (120 universos, 3 PCs).

    Sale de datos_medidos.py, que guarda los tiempos por rank de la ejecucion
    real junto con el comando que la produjo.

    OJO con cluster_stats.npz: lo reescribe CUALQUIER corrida, hasta un test de
    6 universos. Ya paso una vez y se llevo por delante los datos buenos. Por eso
    la corrida de referencia se transcribe a datos_medidos.py y el npz solo se
    usa para avisar si la ultima corrida fue otra."""
    import datos_medidos as D

    r = D.resumen()
    info = {
        "procesos": D.LOTE["procesos"],
        "universos": D.LOTE["universos"],
        "n": D.LOTE["n"],
        "steps": D.LOTE["steps"],
        "tiempos_por_rank": D.APRENDIDO["tiempos_rank"],
        "reparto": D.APRENDIDO["reparto"],
        "serie_s": D.APRENDIDO["t_serie"],
        "paralelo_s": D.APRENDIDO["t_paralelo"],
        "speedup": D.APRENDIDO["speedup"],
        "eficiencia": r["eficiencia_aprendido"],
        "ultima_corrida": None,
    }

    ruta = os.path.join(carpeta, "cluster_stats.npz")
    if os.path.exists(ruta):
        d = np.load(ruta)
        if int(d["universos"]) != D.LOTE["universos"]:
            info["ultima_corrida"] = (int(d["universos"]), float(d["speedup"]))
    return info


def medir_balanceo():
    """Reparto ingenuo vs aprendido: los tiempos REALES de las dos corridas.

    No se modela nada: son dos ejecuciones de 120 universos en los mismos 3 PCs
    en las que lo unico que cambio fue el reparto. Ver datos_medidos.py.

    (Antes esta funcion estimaba el reparto con vel=[1.00, 0.60, 0.40], unos
    pesos que me invente ANTES de medir y que resultaron falsos: la AMD no era
    0.40 sino 0.632. Reportaba un 40% de mejora que no le constaba a nadie.)"""
    import datos_medidos as D

    r = D.resumen()
    return {
        "ingenuo_reparto": D.INGENUO["reparto"],
        "aprendido_reparto": D.APRENDIDO["reparto"],
        "t_ingenuo": D.INGENUO["t_paralelo"],
        "t_aprendido": D.APRENDIDO["t_paralelo"],
        "speedup_ingenuo": D.INGENUO["speedup"],
        "speedup_aprendido": D.APRENDIDO["speedup"],
        "ocio_ingenuo": r["ocio_ingenuo"],
        "ocio_aprendido": r["ocio_aprendido"],
        "mejora_pct": r["mejora_speedup_pct"],
        "mejora_tiempo_pct": r["mejora_tiempo_pct"],
    }


def medir_surrogate(carpeta="dataset"):
    """Aceleracion del surrogate (inferencia) vs la simulacion real."""
    modelo_path = os.path.join(carpeta, "surrogate.pt")
    if not os.path.exists(modelo_path):
        return None
    try:
        import torch
        from entrenar_surrogate import SurrogateCNN
    except Exception:
        return None

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = SurrogateCNN().to(disp)
    modelo.load_state_dict(torch.load(modelo_path, map_location=disp))
    modelo.eval()

    x = torch.zeros(1, 1, 8, 8, 8, device=disp)
    with torch.no_grad():
        for _ in range(10):
            modelo(x)
        if disp == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            modelo(x)
        if disp == "cuda":
            torch.cuda.synchronize()
        t_inf = (time.perf_counter() - t0) / 100

    # ── Contra que se compara: simular UN universo, aqui y ahora ──
    #
    # NO se promedia el 'tiempo_s' del dataset. Esos ficheros son de dos corridas
    # distintas (unos en GPU a ~0.25 s, otros de una tanda vieja en CPU a ~19 s):
    # la fisica es la misma, pero mezclar sus cronometros da una media que no
    # corresponde a ninguna maquina real. Promediarlos daba un "13.000x" absurdo.
    #
    # Se mide en vivo, en la MISMA GPU que acaba de correr la inferencia, con la
    # misma configuracion con la que se entreno el surrogate. Comparacion justa.
    ejemplo = sorted(glob.glob(f"{carpeta}/universo_seed*.npz"))
    if not ejemplo:
        return None
    meta = np.load(ejemplo[0])
    n, steps = int(meta["n"]), int(meta["steps"])

    from simular_universo import simular
    from compute import create_backend

    # Sin backend explicito, simular() cae a NumPy (CPU) y estariamos comparando
    # una GPU contra una CPU: el surrogate saldria ~100x mejor de lo que es.
    bk = create_backend("auto")

    simular(n=n, steps=5, seed=999, backend=bk)   # calentar (kernel + VRAM)
    reps, t0 = 3, time.perf_counter()
    for k in range(reps):
        simular(n=n, steps=steps, seed=900 + k, backend=bk)
    t_sim = (time.perf_counter() - t0) / reps

    return {
        "dispositivo": disp,
        "backend_sim": type(bk).__name__,
        "n": n,
        "steps": steps,
        "inferencia_ms": t_inf * 1000,
        "simulacion_ms": t_sim * 1000,
        "aceleracion": t_sim / t_inf,
    }


def barra(titulo):
    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def main():
    print("\n" + "#" * 60)
    print("#  EL NACIMIENTO DEL UNIVERSO — RESULTADOS DEL PROYECTO")
    print("#  Computacion Paralela y Distribuida + IA")
    print("#" * 60)

    barra("1. CLUSTER — Generacion distribuida (speedup MEDIDO)")
    c = medir_cluster()
    print(f"  Procesos MPI             : {c['procesos']} (3 PCs, 3 casas, via Tailscale)")
    print(f"  Universos generados      : {c['universos']} (N={c['n']}, {c['steps']} pasos)")
    for r, (t, u) in enumerate(zip(c["tiempos_por_rank"], c["reparto"])):
        print(f"    rank {r}: {u:2} universos en {t:5.1f} s")
    print(f"  Trabajo total (1 proceso): {c['serie_s']:.1f} s")
    print(f"  Tiempo real del lote     : {c['paralelo_s']:.1f} s  (manda el mas lento)")
    print(f"  >> SPEEDUP MEDIDO        : {c['speedup']:.2f}x  "
          f"(eficiencia {c['eficiencia']*100:.1f}%)")
    if c["ultima_corrida"]:
        u, s = c["ultima_corrida"]
        print(f"\n  (nota: la ultima corrida guardada en cluster_stats.npz fue de {u} "
              f"universos\n   a {s:.2f}x — una prueba, no la corrida de referencia)")

    barra("2. BALANCEO — Modelo 1 (reparto entre GPUs desiguales)")
    b = medir_balanceo()
    print(f"  {'':32}  {'INGENUO':>10}  {'APRENDIDO':>10}")
    print(f"  Reparto (4060/3050/AMD)         : "
          f"{'/'.join(map(str, b['ingenuo_reparto'])):>10}  "
          f"{'/'.join(map(str, b['aprendido_reparto'])):>10}")
    print(f"  Tiempo del lote                 : "
          f"{b['t_ingenuo']:>9.1f}s  {b['t_aprendido']:>9.1f}s")
    print(f"  Speedup                         : "
          f"{b['speedup_ingenuo']:>9.2f}x  {b['speedup_aprendido']:>9.2f}x")
    print(f"  Ocio (el rapido esperando)      : "
          f"{b['ocio_ingenuo']:>9.1f}s  {b['ocio_aprendido']:>9.1f}s")
    print(f"  >> MEJORA : +{b['mejora_pct']:.1f}% de speedup "
          f"(-{b['mejora_tiempo_pct']:.1f}% de tiempo por lote)")

    barra("3. SURROGATE — Modelo 2 (emulador neuronal)")
    s = medir_surrogate()
    if s:
        print(f"  Universo                 : N={s['n']}, {s['steps']} pasos")
        print(f"  Los dos, en la misma GPU : {s['dispositivo']} / {s['backend_sim']}")
        print(f"  Simulacion real          : {s['simulacion_ms']:8.1f} ms")
        print(f"  Inferencia del surrogate : {s['inferencia_ms']:8.2f} ms")
        print(f"  >> ACELERACION           : ~{s['aceleracion']:.0f}x mas rapido")
    else:
        print("  (sin modelo: entrena el surrogate primero)")

    print("\n" + "#" * 60)
    print("#  Fin del resumen")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
