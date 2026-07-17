"""
datos_medidos.py — Los numeros REALES del proyecto, con su procedencia.

Una sola fuente de verdad. Todo lo que hay aqui salio de una ejecucion real en
el cluster de 3 PCs; cada bloque lleva el comando exacto que lo produjo para que
cualquiera pueda repetirlo y comprobarlo.

REGLA: aqui NO se estima, NO se extrapola y NO se redondea "a favor". Si un dato
no se ha medido, no esta en este archivo. Este proyecto ya se comio dos veces el
mismo error (un speedup calculado como serie/3 que siempre daba 3.00x, y el
surrogate comparandose contra 16 s hardcodeados de otra maquina). De ahi la regla.

Lo que NO esta aqui porque se mide en el momento:
  - Kernel CPU/CUDA/OpenCL  -> scripts/test_backends.py
  - Surrogate (inferencia)  -> scripts/resultados.py

EL CLUSTER
  PC1  DESKTOP-L9RKQCI  RTX 4060   CUDA (kernel propio)   100.95.179.127
  PC2  DESKTOP-4074RG9  RTX 3050   CUDA (kernel propio)   100.118.37.20
  PC3  DESKTOP-7P0N9DF  AMD gfx1032 (RX 6600 XT)  OpenCL  100.100.88.23
  Unidos por Tailscale (VPN de malla), cada uno en una casa distinta.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  1. MODO A — generacion distribuida por lotes (el que se usa en el proyecto)
# ═══════════════════════════════════════════════════════════════════════════
#
# Las dos corridas son IDENTICAS salvo el reparto: mismo N, mismos pasos, mismo
# numero de universos, mismos 3 PCs. Lo unico que cambia es como se reparte el
# trabajo. Por eso la comparacion es limpia: aisla el efecto del Modelo 1.

# Reparto INGENUO: 40/40/40, "cada uno lo mismo".
#   mpiexec -env MPICH_NETMASK 100.64.0.0/255.192.0.0 -hosts 3 <3 IPs> \
#           python scripts/generar_universos.py --total 120 --n 3000 --steps 300 \
#                  --backend auto
INGENUO = {
    "reparto":        [40, 40, 40],       # universos por rank
    "tiempos_rank":   [8.6, 9.8, 13.6],   # s que tardo cada rank
    "t_paralelo":     13.6,               # manda el mas lento
    "t_serie":        32.0,               # si lo hiciera 1 solo proceso (~suma)
    "speedup":        2.36,
}

# Reparto APRENDIDO: 47/42/31, proporcional a la velocidad MEDIDA de cada GPU.
# Los pesos 4.65/4.08/2.94 son universos/s medidos (ver modelo_balanceo.py).
#   ... --total 120 --n 3000 --steps 300 --backend auto --pesos 4.65,4.08,2.94
APRENDIDO = {
    "reparto":        [47, 42, 31],
    "tiempos_rank":   [10.1, 10.5, 10.2],
    "t_paralelo":     10.5,
    "t_serie":        30.8,
    "speedup":        2.94,
}

# Comun a las dos corridas
LOTE = {
    "universos": 120,
    "n":         3000,
    "steps":     300,
    "procesos":  3,
    "recogida_mb":       8.6,   # lo que devuelven los ranks por MPI
    "recogida_s_ingenuo": 4.6,
    "recogida_s_aprendido": 3.9,
}

RANKS = ["PC1 · RTX 4060", "PC2 · RTX 3050", "PC3 · AMD RX 6600 XT"]


# ═══════════════════════════════════════════════════════════════════════════
#  2. MODO B — un universo repartido entre nodos (implementado, medido, descartado)
# ═══════════════════════════════════════════════════════════════════════════
#
# Descomposicion de dominio: cada nodo lleva un trozo del MISMO universo y todos
# se intercambian posiciones cada paso (Allgatherv). La gravedad es de largo
# alcance: cada particula siente a TODAS, asi que no bastan particulas fantasma
# de frontera. Hay que compartirlo todo, cada paso.
#
#   mpiexec ... python scripts/nbody_distribuido.py --n <N> --steps <S> --backend auto
#
# Escenarios: mismo codigo, misma red Tailscale; "directo" es cuando Tailscale
# consiguio conexion P2P (49 ms) en vez de pasar por un relay DERP.
# Los tiempos son POR RANK. Importa: en la salida del script, "calculo (max)" y
# "comunicacion (max)" son maximos de ranks DISTINTOS, asi que no suman el total
# y apilarlos en un grafico enganaria. Por rank si cuadra: calculo + red = total.
MODO_B = [
    {
        "etiqueta": "N=19.683\n(relay DERP)",
        "n": 19683, "pasos": 50, "datos_mb_paso": 0.5,
        # rank: (calculo_s, red_s, total_s)
        "ranks": [(1.54, 19.16, 20.72), (1.84, 18.76, 20.70), (0.24, 20.36, 20.71)],
    },
    {
        "etiqueta": "N=148.877\n(relay DERP)",
        "n": 148877, "pasos": 20, "datos_mb_paso": 3.6,
        "ranks": [(5.99, 59.33, 65.33), (7.58, 57.46, 65.30), (2.23, 62.58, 65.32)],
    },
    {
        "etiqueta": "N=148.877\n(directo P2P, 49 ms)",
        "n": 148877, "pasos": 20, "datos_mb_paso": 3.6,
        "ranks": [(6.06, 50.18, 56.25), (7.12, 49.02, 56.24), (2.24, 53.79, 56.25)],
    },
]

# La conclusion, con numeros:
#   - La conexion directa (49 ms de latencia, sin relay) solo mejoro un 14%.
#     Si el problema fuera la LATENCIA, la mejora habria sido enorme. No lo es.
#   - El tiempo de red escala O(N) igual que los datos: 0.5 MB -> 3.6 MB (7.2x)
#     hizo pasar la red de 392.6 ms/paso a 3129.2 ms/paso (8.0x). Eso es
#     ANCHO DE BANDA, no latencia.
#   - En Allgatherv todos suben a todos -> manda la subida domestica (~10 Mbps).
MODO_B_CONCLUSION = "ancho de banda (~10 Mbps de subida), no latencia"


def modo_b_pct_red():
    """Que porcentaje del tiempo se va en red, por escenario (min-max entre ranks)."""
    out = []
    for esc in MODO_B:
        pcts = [red / total * 100 for _, red, total in esc["ranks"]]
        out.append((esc["etiqueta"], min(pcts), max(pcts)))
    return out


def resumen():
    """Comprueba que los numeros derivados cuadran con los medidos."""
    # Por rank, calculo + red ~= total. No cuadra al milimetro: queda un resto de
    # 0.1-0.5 s (<1%) de arranque y E/S que el script no cronometra por separado.
    # Se tolera un 2%; si algun dia se dispara, es que se transcribio algo mal.
    for esc in MODO_B:
        for calc, red, total in esc["ranks"]:
            assert abs((calc + red) - total) < 0.02 * total, esc["etiqueta"]

    for nombre, d in (("ingenuo", INGENUO), ("aprendido", APRENDIDO)):
        assert sum(d["reparto"]) == LOTE["universos"], nombre
        # el tiempo del lote lo marca el rank mas lento
        assert abs(max(d["tiempos_rank"]) - d["t_paralelo"]) < 0.05, nombre
        # speedup = trabajo en serie / tiempo real
        calc = d["t_serie"] / d["t_paralelo"]
        assert abs(calc - d["speedup"]) < 0.02, f"{nombre}: {calc:.2f} vs {d['speedup']}"

    p = LOTE["procesos"]
    return {
        "eficiencia_ingenuo":   INGENUO["speedup"] / p,          # 0.787
        "eficiencia_aprendido": APRENDIDO["speedup"] / p,        # 0.980
        # +24.6%: mejora del SPEEDUP
        "mejora_speedup_pct": (APRENDIDO["speedup"] / INGENUO["speedup"] - 1) * 100,
        # -22.8%: reduccion del TIEMPO (es el mismo hecho visto al reves)
        "mejora_tiempo_pct":  (1 - APRENDIDO["t_paralelo"] / INGENUO["t_paralelo"]) * 100,
        # Ocio: lo que el rank mas rapido pasa esperando al mas lento
        "ocio_ingenuo":   max(INGENUO["tiempos_rank"]) - min(INGENUO["tiempos_rank"]),
        "ocio_aprendido": max(APRENDIDO["tiempos_rank"]) - min(APRENDIDO["tiempos_rank"]),
    }


if __name__ == "__main__":
    r = resumen()
    print("Datos medidos — comprobacion")
    print(f"  Eficiencia  ingenuo   : {r['eficiencia_ingenuo']*100:.1f}%")
    print(f"  Eficiencia  aprendido : {r['eficiencia_aprendido']*100:.1f}%")
    print(f"  Mejora del speedup    : +{r['mejora_speedup_pct']:.1f}%")
    print(f"  Reduccion del tiempo  : -{r['mejora_tiempo_pct']:.1f}%")
    print(f"  Ocio ingenuo          : {r['ocio_ingenuo']:.1f} s")
    print(f"  Ocio aprendido        : {r['ocio_aprendido']:.1f} s")
    print("\n  Todos los asserts pasaron: los derivados cuadran con lo medido.")
