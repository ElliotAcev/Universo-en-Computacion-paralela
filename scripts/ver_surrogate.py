"""
ver_surrogate.py — Compara la prediccion del surrogate con la simulacion real

Para un universo del dataset, muestra tres rejillas de densidad (proyectadas en 2D):
  - Inicial            (el Big Bang)
  - Final REAL         (lo que dio la simulacion N-body)
  - Final PREDICHO     (lo que adivina el surrogate en milisegundos)

Asi se ve, de un vistazo, que el surrogate reproduce la telarana sin simular.

Uso:
    python scripts/ver_surrogate.py --seed 5 --grid 8
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from entrenar_surrogate import SurrogateCNN, densidad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--modelo", type=str, default="dataset/surrogate.pt")
    ap.add_argument("--guardar", action="store_true")
    args = ap.parse_args()

    d = np.load(f"dataset/universo_seed{args.seed:02d}.npz")
    dens_ini = densidad(d["pos_inicial"], args.grid)
    dens_fin = densidad(d["pos_final"], args.grid)

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = SurrogateCNN().to(disp)
    modelo.load_state_dict(torch.load(args.modelo, map_location=disp))
    modelo.eval()
    with torch.no_grad():
        x = torch.tensor(dens_ini)[None, None].to(disp)
        pred = modelo(x).cpu().numpy()[0, 0]

    # proyeccion 2D (suma a lo largo de un eje) para verlo como imagen
    def proj(v):
        return v.sum(axis=2)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#05060d")
    fig.suptitle(f"Surrogate vs simulacion real — universo seed {args.seed}", color="white")

    paneles = [
        (proj(dens_ini), "Inicial (Big Bang)"),
        (proj(dens_fin), "Final REAL (simulado)"),
        (proj(pred), "Final PREDICHO (surrogate)"),
    ]
    for ax, (img, titulo) in zip(axs, paneles):
        ax.imshow(img, cmap="magma", interpolation="bilinear")
        ax.set_title(titulo, color="white", fontsize=11)
        ax.set_facecolor("#05060d")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    if args.guardar:
        salida = f"dataset/surrogate_seed{args.seed:02d}.png"
        plt.savefig(salida, dpi=120, facecolor="#05060d")
        print(f"Guardado: {salida}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
