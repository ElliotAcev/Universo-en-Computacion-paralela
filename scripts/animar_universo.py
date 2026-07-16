"""
animar_universo.py — Pelicula del nacimiento y evolucion del universo

Simula un universo guardando snapshots intermedios y genera una ANIMACION que
muestra como la gravedad va tejiendo la telarana cosmica desde el estado casi
uniforme del Big Bang hasta los cumulos y filamentos.

Uso (rapido, con GPU):
    python scripts/animar_universo.py --n 3000 --steps 400 --frames 80 --backend auto

    # Guardar como GIF (no necesita nada extra):
    python scripts/animar_universo.py --guardar universo.gif

    # Guardar como MP4 (necesita ffmpeg instalado):
    python scripts/animar_universo.py --guardar universo.mp4

Salida: una animacion girando alrededor del universo mientras evoluciona.
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from simular_universo import simular, BOX, contraste_densidad
from compute import create_backend
from simular_universo import G, EPS2


def main():
    ap = argparse.ArgumentParser(description="Animacion del nacimiento del universo")
    ap.add_argument("--n", type=int, default=3000, help="numero de particulas")
    ap.add_argument("--steps", type=int, default=400, help="pasos de simulacion")
    ap.add_argument("--frames", type=int, default=80, help="fotogramas de la animacion")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", type=str, default="auto", help="auto|cpu|cuda|opencl")
    ap.add_argument("--guardar", type=str, default="", help="archivo .gif o .mp4 (vacio = mostrar en pantalla)")
    ap.add_argument("--girar", action="store_true", help="girar la camara mientras evoluciona")
    args = ap.parse_args()

    backend = create_backend(args.backend, {"G": G, "EPS2": EPS2, "BOX": BOX})
    print(f"Backend: {backend.nombre} [{backend.dispositivo}]")
    print(f"Simulando {args.n} particulas, {args.steps} pasos, {args.frames} fotogramas...")

    res = simular(args.n, args.steps, args.seed, n_snapshots=args.frames, backend=backend)
    snaps = res["snapshots"]
    print(f"Listo en {res['tiempo_s']:.1f}s — {len(snaps)} fotogramas capturados")
    print(f"Contraste: {contraste_densidad(snaps[0]):.2f} (inicio) -> "
          f"{contraste_densidad(res['pos_final']):.2f} (final)")

    # --- montaje de la animacion ---
    fig = plt.figure(figsize=(9, 9), facecolor="#05060d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05060d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)

    puntos = ax.scatter([], [], [], s=3, c="#8ab0ff", alpha=0.85, edgecolors="none")
    titulo = ax.set_title("", color="white", fontsize=13, pad=6)

    def vecinos(pos, radio=0.07):
        """Cuenta vecinos cercanos de cada particula (con rejilla, rapido).
        Sirve para pintar mas brillantes las zonas densas."""
        celdas = max(4, int(BOX / radio))
        idx = np.floor(pos / BOX * celdas).astype(int) % celdas
        plano = (idx[:, 0] * celdas + idx[:, 1]) * celdas + idx[:, 2]
        conteo = np.bincount(plano, minlength=celdas ** 3)
        return conteo[plano].astype(float)

    def estilo():
        ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.set_zlim(0, BOX)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
            eje.set_pane_color((0.02, 0.025, 0.06, 1.0))
            eje.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.grid(False)

    def frame(i):
        pos = snaps[i]
        puntos._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

        # brillo y tamano segun la densidad local -> los cumulos destacan
        d = vecinos(pos)
        d = d / max(d.max(), 1)
        puntos.set_sizes(2 + 26 * d ** 1.5)
        puntos.set_color(matplotlib.colormaps["magma"](0.25 + 0.75 * d ** 0.6))

        avance = i / max(1, len(snaps) - 1)
        titulo.set_text(f"El nacimiento del universo — {avance*100:3.0f}%\n"
                        f"contraste de densidad: {contraste_densidad(pos):.2f}")
        if args.girar:
            ax.view_init(elev=22, azim=25 + i * 90 / len(snaps))
        estilo()
        return puntos, titulo

    estilo()
    ax.view_init(elev=22, azim=45)
    anim = FuncAnimation(fig, frame, frames=len(snaps), interval=60, blit=False)

    if args.guardar:
        print(f"Guardando en {args.guardar} (puede tardar)...")
        if args.guardar.endswith(".gif"):
            anim.save(args.guardar, writer=PillowWriter(fps=15), savefig_kwargs={"facecolor": "#05060d"})
        else:
            anim.save(args.guardar, fps=20, dpi=110, savefig_kwargs={"facecolor": "#05060d"})
        print(f"Guardado: {args.guardar}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
