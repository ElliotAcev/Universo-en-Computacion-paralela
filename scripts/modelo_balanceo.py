"""
modelo_balanceo.py — MODELO 1 de IA: balanceo de carga del clúster

Problema: las 3 GPUs son desiguales (RTX 4060, RTX 3050, AMD OpenCL). Hay que
generar un LOTE de universos para el dataset del surrogate. Si repartimos los
universos a partes iguales, la GPU lenta se retrasa y las demás la esperan: el
lote entero tarda lo que tarda la más lenta.

Solución (este script):
  1. APRENDE un modelo del coste de un universo: mide cuánto tarda el cálculo
     N-body para distintos tamaños N y ajusta  tiempo ~ k * N^2  (regresión,
     porque el N-body directo es O(N^2)).
  2. REPARTE el lote de universos entre las GPUs proporcional a su velocidad,
     de modo que las tres terminen a la vez, y lo compara con el reparto
     ingenuo (a partes iguales).

Uso:
    python scripts/modelo_balanceo.py --universos 60 --n 1000
"""

import argparse
import time
import numpy as np

from simular_universo import aceleraciones, BOX

# Velocidad relativa de cada GPU (1.0 = la más rápida).
#
# MEDIDAS REALES en el clúster (2026-07-16, N=1500, steps=200, backend CUDA):
#   RTX 4060: 0.43 s/universo  -> 1.00 (referencia)
#   RTX 3050: 0.66 s/universo  -> 0.65
# Validado en hardware real: con reparto 10/10 el speedup fue 1.65x; aplicando
# estos pesos (reparto 12/8) subio a 1.96x -> 98% de eficiencia paralela.
#
# El peso de la AMD sigue siendo una ESTIMACION (aun no se ha medido).
GPUS = {
    "PC1-RTX4060 (CUDA)": 1.00,    # medido
    "PC2-RTX3050 (CUDA)": 0.65,    # medido
    "PC3-AMD-RX (OpenCL)": 0.40,   # estimado, pendiente de medir
}


def medir_tiempos(tamanos, repes=3):
    """Mide el tiempo de un paso de fuerzas para varios N (dataset del modelo)."""
    rng = np.random.default_rng(0)
    Ns, Ts = [], []
    print("Midiendo tiempos de calculo (aprendiendo el modelo de coste)...")
    for n in tamanos:
        pos = rng.uniform(0, BOX, size=(n, 3))
        masa = np.full(n, 1.0 / n)
        mejor = min(_cronometrar(pos, masa) for _ in range(repes))
        Ns.append(n)
        Ts.append(mejor)
        print(f"  N={n:5d}  ->  {mejor*1000:8.2f} ms/paso")
    return np.array(Ns, float), np.array(Ts, float)


def _cronometrar(pos, masa):
    t0 = time.perf_counter()
    aceleraciones(pos, masa)
    return time.perf_counter() - t0


def ajustar_modelo(Ns, Ts):
    """Ajusta tiempo ~ k * N^2 (el N-body directo es O(N^2)). Devuelve k y R^2."""
    x = Ns ** 2
    k = np.sum(x * Ts) / np.sum(x * x)          # mínimos cuadrados sin intercepto
    pred = k * x
    ss_res = np.sum((Ts - pred) ** 2)
    ss_tot = np.sum((Ts - Ts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return k, r2


def repartir_proporcional(universos, gpus):
    """Reparte el nº de universos proporcional a la velocidad de cada GPU."""
    vel = np.array(list(gpus.values()), float)
    frac = vel / vel.sum()
    reparto = np.floor(frac * universos).astype(int)
    reparto[np.argmax(vel)] += universos - reparto.sum()   # el resto a la más rápida
    return reparto


def tiempo_del_lote(reparto, coste_universo, gpus):
    """El lote termina cuando acaba la GPU más lenta. Coste por universo es
    'coste_universo' en la GPU de referencia; en la GPU i se divide por su velocidad."""
    vel = np.array(list(gpus.values()), float)
    tiempos = np.array(reparto, float) * coste_universo / vel
    return tiempos.max(), tiempos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universos", type=int, default=60, help="nº total de universos del lote")
    ap.add_argument("--n", type=int, default=1000, help="tamaño N de cada universo")
    args = ap.parse_args()

    # 1. Aprender el modelo de coste tiempo(N)
    tamanos = [500, 1000, 1500, 2000, 2500, 3000]
    Ns, Ts = medir_tiempos(tamanos)
    k, r2 = ajustar_modelo(Ns, Ts)
    print(f"\nModelo aprendido:  tiempo_por_paso ~ {k:.3e} * N^2   (R^2 = {r2:.4f})")

    # coste de un universo (aprox.: tiempo de un paso a ese N, como referencia relativa)
    coste = k * args.n ** 2
    print(f"Coste estimado de 1 universo de N={args.n}: {coste*1000:.2f} ms/paso (referencia)")

    # 2. Comparar reparto ingenuo vs aprendido
    nombres = list(GPUS.keys())
    U = args.universos

    ingenuo = repartir_igual(U, len(GPUS))
    t_ing, det_ing = tiempo_del_lote(ingenuo, coste, GPUS)

    aprendido = repartir_proporcional(U, GPUS)
    t_apr, det_apr = tiempo_del_lote(aprendido, coste, GPUS)

    print(f"\n=== Repartir {U} universos entre 3 GPUs ===\n")
    print(f"{'GPU':22s} {'ingenuo (u / tiempo)':>24s} {'aprendido (u / tiempo)':>26s}")
    for i, nom in enumerate(nombres):
        print(f"{nom:22s} {ingenuo[i]:4d} / {det_ing[i]:6.2f} s"
              f"        {aprendido[i]:4d} / {det_apr[i]:6.2f} s")

    print(f"\nTiempo total del lote (manda la GPU mas lenta):")
    print(f"  Reparto ingenuo   : {t_ing:7.2f} s")
    print(f"  Reparto aprendido : {t_apr:7.2f} s")
    mejora = (1 - t_apr / t_ing) * 100
    print(f"  MEJORA            : {mejora:5.1f}% mas rapido")


def repartir_igual(universos, n_gpus):
    base = universos // n_gpus
    reparto = np.full(n_gpus, base)
    reparto[0] += universos - reparto.sum()
    return reparto


if __name__ == "__main__":
    main()
