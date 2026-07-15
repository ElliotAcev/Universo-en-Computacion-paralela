"""
ver_universo.py — Visualiza un universo simulado (antes vs después)

Muestra las partículas al inicio (casi uniforme) y al final (telaraña cósmica:
cúmulos y filamentos formados por la gravedad).

Uso:
    python scripts/ver_universo.py --archivo dataset/universo_seed01.npz
    python scripts/ver_universo.py --archivo dataset/universo_seed01.npz --guardar
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archivo", type=str, default="dataset/universo_seed01.npz")
    ap.add_argument("--guardar", action="store_true", help="guardar PNG en vez de mostrar")
    args = ap.parse_args()

    d = np.load(args.archivo)
    ini = d["pos_inicial"]
    fin = d["pos_final"]

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Nacimiento del universo — la gravedad teje la telaraña cósmica", color="white")
    fig.patch.set_facecolor("#05060d")

    for i, (pos, titulo) in enumerate([(ini, "Inicial (casi uniforme)"),
                                       (fin, "Evolucionado (telaraña cósmica)")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=2, c="#8ab0ff", alpha=0.5, edgecolors="none")
        ax.set_title(titulo, color="white")
        ax.set_facecolor("#05060d")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.02, 0.02, 0.05, 1.0))
            pane.label.set_color("#5c688a")
        ax.tick_params(colors="#5c688a")

    plt.tight_layout()
    if args.guardar:
        salida = args.archivo.replace(".npz", ".png")
        plt.savefig(salida, dpi=120, facecolor="#05060d")
        print(f"Guardado: {salida}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
