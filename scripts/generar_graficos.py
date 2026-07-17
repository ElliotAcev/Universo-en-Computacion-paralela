"""
generar_graficos.py — Los graficos de la presentacion, a partir de datos REALES.

Todo lo que se dibuja aqui sale de scripts/datos_medidos.py, que a su vez solo
contiene mediciones de ejecuciones reales en el cluster de 3 PCs (con el comando
que las produjo anotado al lado).

Antes este script tenia hardcodeados los tiempos de una demo vieja en CPU con
1/2/4 procesos — una configuracion que en el cluster de 3 PCs nunca existio.
Las curvas de speedup de la presentacion no cuadraban con lo que se decia en voz
alta. De ahi que ahora los datos vivan en un solo sitio.

Uso:
    python scripts/generar_graficos.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_medidos as D

OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(parents=True, exist_ok=True)

R = D.resumen()          # lanza los asserts: si algo no cuadra, no se generan graficos
COL_ING, COL_APR, COL_IDEAL = "#c44e52", "#4c9f70", "#8899aa"
COL_CALC, COL_RED = "#4c72b0", "#c44e52"


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT / nombre, dpi=180)
    plt.close(fig)
    print(f"  {nombre}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. Speedup — el titular del cluster
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
etiquetas = ["Ideal\n(3 procesos)", "Reparto ingenuo\n40/40/40", "Reparto aprendido\n47/42/31"]
valores = [3.0, D.INGENUO["speedup"], D.APRENDIDO["speedup"]]
barras = ax.bar(etiquetas, valores, color=[COL_IDEAL, COL_ING, COL_APR], width=0.55)
for b, v in zip(barras, valores):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}x",
            ha="center", fontweight="bold", fontsize=12)
ax.axhline(3.0, ls="--", lw=1, color=COL_IDEAL, alpha=0.8)
ax.set_ylabel("Speedup")
ax.set_ylim(0, 3.5)
ax.set_title(f"Speedup MEDIDO — {D.LOTE['universos']} universos "
             f"(N={D.LOTE['n']}, {D.LOTE['steps']} pasos) en 3 GPUs")
ax.grid(True, axis="y", alpha=0.3)
ax.text(0.5, 0.04, "Mismo trabajo, mismos 3 PCs: lo unico que cambia es el reparto",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555")
guardar(fig, "speedup_mpi.png")


# ═══════════════════════════════════════════════════════════════════════════
#  2. Eficiencia paralela
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
effs = [R["eficiencia_ingenuo"] * 100, R["eficiencia_aprendido"] * 100]
barras = ax.bar(["Reparto ingenuo", "Reparto aprendido"], effs,
                color=[COL_ING, COL_APR], width=0.5)
for b, v in zip(barras, effs):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
            ha="center", fontweight="bold", fontsize=13)
ax.axhline(100, ls="--", lw=1, color=COL_IDEAL)
ax.text(1.45, 101, "ideal 100%", fontsize=9, color="#555", va="bottom", ha="right")
ax.set_ylabel("Eficiencia paralela")
ax.set_ylim(0, 115)
ax.set_title("Eficiencia paralela — el Modelo 1 recupera el 20% que se perdia")
ax.grid(True, axis="y", alpha=0.3)
guardar(fig, "eficiencia_mpi.png")


# ═══════════════════════════════════════════════════════════════════════════
#  3. El balanceo, rank a rank — POR QUE funciona el Modelo 1
# ═══════════════════════════════════════════════════════════════════════════
# El grafico mas importante: se ve el ocio. Con 40/40/40 la 4060 termina en 8.6 s
# y luego espera 5 s de brazos cruzados a que la AMD acabe. Un cluster va a la
# velocidad del mas lento, asi que darle a todos lo mismo REGALA tiempo.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
y = np.arange(3)

for ax, datos, titulo, color in (
    (a1, D.INGENUO,   "Ingenuo: 40/40/40", COL_ING),
    (a2, D.APRENDIDO, "Aprendido: 47/42/31", COL_APR),
):
    t = datos["tiempos_rank"]
    lento = max(t)
    ax.barh(y, t, color=color, height=0.55, zorder=3)
    for i, ti in enumerate(t):
        ocio = lento - ti
        if ocio > 0.05:      # el hueco hasta que acaba el ultimo = tiempo perdido
            ax.barh(i, ocio, left=ti, color="#ddd", height=0.55, zorder=3,
                    hatch="///", edgecolor="#bbb")
            # Con el reparto aprendido el ocio es tan pequeno que la etiqueta no
            # cabe dentro de la barra: se saca fuera para que se lea.
            if ocio > 1.5:
                ax.text(ti + ocio / 2, i, f"ocio\n{ocio:.1f}s", ha="center",
                        va="center", fontsize=8, color="#777", zorder=4)
            else:
                ax.text(ti + ocio + 0.25, i, f"ocio {ocio:.1f}s", ha="left",
                        va="center", fontsize=8, color="#777", zorder=4)
        ax.text(ti / 2, i, f"{datos['reparto'][i]} univ · {ti:.1f}s",
                ha="center", va="center", color="white", fontweight="bold",
                fontsize=9, zorder=4)
    ax.axvline(lento, ls="--", lw=1.5, color="#333", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(D.RANKS)
    ax.set_xlabel("Tiempo (s)")
    ax.set_xlim(0, 15)
    ax.set_title(f"{titulo}\nlote = {lento:.1f} s", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.invert_yaxis()

fig.suptitle("Un cluster va a la velocidad del mas lento: repartir por igual REGALA tiempo",
             fontsize=12, fontweight="bold")
guardar(fig, "balanceo_ranks.png")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Modo B — el muro del ancho de banda
# ═══════════════════════════════════════════════════════════════════════════
# Se dibuja rank a rank. Los "max" que imprime nbody_distribuido.py son maximos
# de ranks distintos y NO suman el total; apilarlos daria porcentajes falsos.
# Por rank si cuadra exactamente: calculo + red = total.
fig, ax = plt.subplots(figsize=(10, 6))
ancho, sep = 0.22, 1.0
pos, etiq_pc = [], []
for i, esc in enumerate(D.MODO_B):
    for j in range(3):
        pos.append(i * sep + (j - 1) * ancho)
        etiq_pc.append(f"PC{j+1}")

for k, (xi, esc_j) in enumerate(zip(pos, [(e, j) for e in D.MODO_B for j in range(3)])):
    esc, j = esc_j
    calc, red, _ = esc["ranks"][j]
    ax.bar(xi, calc, ancho, color=COL_CALC, zorder=3,
           label="Calculo (la GPU trabajando)" if k == 0 else None)
    ax.bar(xi, red, ancho, bottom=calc, color=COL_RED, zorder=3,
           label="Red (esperando datos)" if k == 0 else None)
    ax.text(xi, calc + red + 1.0, f"{red/(calc+red)*100:.0f}%", ha="center",
            fontsize=8, color=COL_RED, fontweight="bold")

ax.set_xticks(pos)
ax.set_xticklabels(etiq_pc, fontsize=8, color="#666")
ax.set_ylabel("Tiempo (s)")
ax.set_ylim(0, 78)
ax.set_title("Modo B: repartir UN universo entre 3 casas — el muro del ancho de banda",
             pad=12)
ax.legend(loc="upper left")
ax.grid(True, axis="y", alpha=0.3, zorder=0)

# El nombre del escenario, debajo de su trio de barras
for i, esc in enumerate(D.MODO_B):
    ax.annotate(esc["etiqueta"], xy=(i * sep, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -32), textcoords="offset points",
                ha="center", va="top", fontsize=9.5, fontweight="bold")

fig.text(0.5, 0.035,
         "La conexion directa P2P (49 ms) solo mejoro un 14% -> el cuello NO es la latencia.\n"
         "Los datos x7.2 hicieron la red x8.0: es ANCHO DE BANDA (~10 Mbps de subida domestica).\n"
         "En Allgatherv todos suben a todos, cada paso.",
         ha="center", fontsize=9, style="italic", color="#555")
fig.subplots_adjust(bottom=0.30, top=0.91)
fig.savefig(OUT / "modo_b_muro.png", dpi=180)
plt.close(fig)
print("  modo_b_muro.png")


# ═══════════════════════════════════════════════════════════════════════════
#  5. Concepto inicial (NO es la simulacion)
# ═══════════════════════════════════════════════════════════════════════════
# Galaxia sintetica de matplotlib. Se conserva a proposito para ensenar de donde
# arranco el proyecto, pero el titulo deja claro que NO es un resultado: la
# galaxia de verdad la renderiza universo.exe con 60.000 particulas y fisica real.
rng = np.random.default_rng(42)
n, arms = 2500, 3
r = np.clip(rng.gamma(2.0, 0.85, n), 0.05, 5.8)
theta = (2 * np.pi * rng.integers(0, arms, n) / arms) + 1.35 * r + rng.normal(0, 0.22, n)
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(r * np.cos(theta), r * np.sin(theta), rng.normal(0, 0.07 + 0.012 * r, n),
           s=np.clip(12 / (r + 0.45), 1, 8), alpha=0.7)
ax.scatter([0], [0], [0], s=80, marker="o")
ax.set_title("Punto de partida: boceto sintetico de galaxia espiral\n"
             "(NO es la simulacion — la galaxia real la renderiza universo.exe)",
             fontsize=10)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-2, 2)
ax.view_init(elev=27, azim=38)
guardar(fig, "galaxia_3d_preview.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Y la tabla, con los mismos datos
# ═══════════════════════════════════════════════════════════════════════════
csv = Path(__file__).resolve().parents[1] / "resultados_mpi.csv"
with open(csv, "w", encoding="utf-8") as f:
    f.write("# Generacion distribuida de universos — Modo A (reparto por lotes)\n")
    f.write("# MEDIDO en 3 PCs reales (RTX 4060 + RTX 3050 + AMD RX 6600 XT) via Tailscale.\n")
    f.write(f"# {D.LOTE['universos']} universos, N={D.LOTE['n']}, {D.LOTE['steps']} pasos.\n")
    f.write("# Las dos filas son la MISMA carga; solo cambia el reparto.\n")
    f.write("reparto,universos_rank0,universos_rank1,universos_rank2,"
            "t_rank0_s,t_rank1_s,t_rank2_s,t_lote_s,t_serie_s,speedup,eficiencia,ocio_s\n")
    for nombre, d, ocio in (("ingenuo", D.INGENUO, R["ocio_ingenuo"]),
                            ("aprendido", D.APRENDIDO, R["ocio_aprendido"])):
        f.write(f"{nombre},{d['reparto'][0]},{d['reparto'][1]},{d['reparto'][2]},"
                f"{d['tiempos_rank'][0]},{d['tiempos_rank'][1]},{d['tiempos_rank'][2]},"
                f"{d['t_paralelo']},{d['t_serie']},{d['speedup']:.2f},"
                f"{d['speedup']/D.LOTE['procesos']:.3f},{ocio:.1f}\n")
print("  resultados_mpi.csv")

print(f"\nListo. Graficos en {OUT}")
print(f"  Speedup    : {D.INGENUO['speedup']:.2f}x -> {D.APRENDIDO['speedup']:.2f}x "
      f"(+{R['mejora_speedup_pct']:.1f}%)")
print(f"  Eficiencia : {R['eficiencia_ingenuo']*100:.1f}% -> {R['eficiencia_aprendido']*100:.1f}%")
print(f"  Ocio       : {R['ocio_ingenuo']:.1f}s -> {R['ocio_aprendido']:.1f}s")
