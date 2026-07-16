"""
zoom_galaxia.py — La narrativa completa: Big Bang -> telarana -> zoom a la galaxia

Genera una animacion en dos actos:
  ACTO 1: el universo evoluciona desde el estado casi uniforme (Big Bang) hasta
          formar la telarana cosmica.
  ACTO 2: la camara localiza el HALO MAS DENSO y hace zoom hasta el, que es
          donde se formaria una galaxia como la que simula la app CUDA.

Asi se enlaza la simulacion cosmologica (Python) con el motor de galaxia
(main_cuda_referencia.cu): "haz zoom a este nodo y ahi esta tu galaxia".

Uso:
    python scripts/zoom_galaxia.py --n 6000 --steps 400 --backend auto --guardar zoom.gif
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from simular_universo import simular, contraste_densidad, BOX, G, EPS2
from compute import create_backend


def encontrar_halo(pos, celdas=12):
    """Encuentra el halo mas denso: la celda con mas particulas, y devuelve el
    centro de masa de las particulas de su alrededor. Es un buscador de halos
    sencillo, en la linea de un FoF casero."""
    idx = np.floor(pos / BOX * celdas).astype(int) % celdas
    plano = (idx[:, 0] * celdas + idx[:, 1]) * celdas + idx[:, 2]
    conteo = np.bincount(plano, minlength=celdas ** 3)
    mejor = int(np.argmax(conteo))

    # centro aproximado de esa celda
    cz = mejor % celdas
    cy = (mejor // celdas) % celdas
    cx = mejor // (celdas * celdas)
    centro = (np.array([cx, cy, cz]) + 0.5) / celdas * BOX

    # refinamos con el centro de masa de lo que hay cerca (imagen minima)
    d = pos - centro
    d -= BOX * np.round(d / BOX)
    cerca = np.sum(d * d, axis=1) < (BOX / celdas) ** 2
    if cerca.sum() > 0:
        centro = centro + d[cerca].mean(axis=0)
    return centro, int(conteo[mejor])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--frames", type=int, default=50, help="fotogramas del acto 1 (evolucion)")
    ap.add_argument("--zoom-frames", type=int, default=40, help="fotogramas del acto 2 (zoom)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--guardar", type=str, default="", help="archivo .gif o .mp4")
    args = ap.parse_args()

    backend = create_backend(args.backend, {"G": G, "EPS2": EPS2, "BOX": BOX})
    print(f"Backend: {backend.nombre} [{backend.dispositivo}]")
    print(f"Simulando el universo ({args.n} particulas, {args.steps} pasos)...")

    res = simular(args.n, args.steps, args.seed, n_snapshots=args.frames, backend=backend)
    snaps = list(res["snapshots"]) + [res["pos_final"]]
    print(f"Listo en {res['tiempo_s']:.1f}s | contraste final: {contraste_densidad(res['pos_final']):.2f}")

    centro, poblacion = encontrar_halo(res["pos_final"])
    print(f"Halo mas denso encontrado en {np.round(centro, 3)} con ~{poblacion} particulas")

    # --- figura ---
    fig = plt.figure(figsize=(9, 9), facecolor="#05060d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05060d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    puntos = ax.scatter([], [], [], s=3, c="#8ab0ff", alpha=0.9, edgecolors="none")
    titulo = ax.set_title("", color="white", fontsize=13, pad=6)

    def densidad_local(pos, radio=0.07):
        celdas = max(4, int(BOX / radio))
        idx = np.floor(pos / BOX * celdas).astype(int) % celdas
        plano = (idx[:, 0] * celdas + idx[:, 1]) * celdas + idx[:, 2]
        conteo = np.bincount(plano, minlength=celdas ** 3)
        return conteo[plano].astype(float)

    total = len(snaps) + args.zoom_frames

    def frame(i):
        if i < len(snaps):
            # ---- ACTO 1: evolucion, vista completa ----
            pos = snaps[i]
            medio = BOX / 2
            radio = BOX / 2
            avance = i / max(1, len(snaps) - 1)
            texto = f"ACTO 1 — El nacimiento del universo ({avance*100:3.0f}%)\ncontraste: {contraste_densidad(pos):.2f}"
            cen = np.array([medio, medio, medio])
            azim = 25 + i * 60 / len(snaps)
        else:
            # ---- ACTO 2: zoom al halo mas denso ----
            pos = snaps[-1]
            t = (i - len(snaps)) / max(1, args.zoom_frames - 1)
            suave = t * t * (3 - 2 * t)                  # arranque y frenada suaves
            radio = (BOX / 2) * (1 - suave) + 0.06 * suave
            cen = np.array([BOX / 2] * 3) * (1 - suave) + centro * suave
            texto = ("ACTO 2 — Zoom al halo\n"
                     "aqui se formaria una galaxia como la del motor CUDA")
            azim = 85 + t * 40

        d = densidad_local(pos)
        d = d / max(d.max(), 1)
        puntos._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        puntos.set_sizes(2 + 30 * d ** 1.5)
        puntos.set_color(matplotlib.colormaps["magma"](0.25 + 0.75 * d ** 0.6))
        titulo.set_text(texto)

        ax.set_xlim(cen[0] - radio, cen[0] + radio)
        ax.set_ylim(cen[1] - radio, cen[1] + radio)
        ax.set_zlim(cen[2] - radio, cen[2] + radio)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
            eje.set_pane_color((0.02, 0.025, 0.06, 1.0))
            eje.line.set_color((0, 0, 0, 0))
        ax.grid(False)
        ax.view_init(elev=22, azim=azim)
        return puntos, titulo

    anim = FuncAnimation(fig, frame, frames=total, interval=70, blit=False)

    if args.guardar:
        print(f"Guardando {args.guardar} ({total} fotogramas, puede tardar)...")
        if args.guardar.endswith(".gif"):
            anim.save(args.guardar, writer=PillowWriter(fps=15), savefig_kwargs={"facecolor": "#05060d"})
        else:
            anim.save(args.guardar, fps=20, dpi=110, savefig_kwargs={"facecolor": "#05060d"})
        print(f"Guardado: {args.guardar}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
