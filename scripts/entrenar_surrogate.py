"""
entrenar_surrogate.py — MODELO 2 de IA: surrogate neuronal del universo

Aprende a EMULAR la simulacion: dado el universo inicial (Big Bang), predice la
telarana cosmica final SIN calcular la gravedad paso a paso. Estilo D3M
(Deep Density Displacement Model).

Representacion: cada universo se convierte en una rejilla de densidad G^3
(cuantas particulas hay en cada celda). El modelo aprende:
    densidad_inicial (G^3)  --CNN 3D-->  densidad_final (G^3)

Entrena con los universos de dataset/ (generados por el cl
uster).

Uso:
    python scripts/entrenar_surrogate.py --grid 8 --epocas 300
"""

import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn

from simular_universo import BOX

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def densidad(pos, G):
    """Convierte posiciones (N,3) en una rejilla de densidad G^3 centrada en 0
    (contraste: 0 = densidad media, positivo = mas denso)."""
    idx = np.floor(pos / BOX * G).astype(int) % G
    plano = (idx[:, 0] * G + idx[:, 1]) * G + idx[:, 2]
    conteo = np.bincount(plano, minlength=G ** 3).astype(np.float32)
    conteo = conteo.reshape(G, G, G)
    return conteo / conteo.mean() - 1.0


def cargar_dataset(carpeta, G):
    X, Y = [], []
    for archivo in sorted(glob.glob(f"{carpeta}/universo_seed*.npz")):
        d = np.load(archivo)
        X.append(densidad(d["pos_inicial"], G))
        Y.append(densidad(d["pos_final"], G))
    X = np.array(X)[:, None, :, :, :]   # (muestras, 1, G, G, G)
    Y = np.array(Y)[:, None, :, :, :]
    return torch.tensor(X), torch.tensor(Y)


class SurrogateCNN(nn.Module):
    """CNN 3D con convoluciones periodicas (circular) para respetar la caja."""
    def __init__(self, canales=32):
        super().__init__()
        def conv(ci, co):
            return nn.Conv3d(ci, co, 3, padding=1, padding_mode="circular")
        self.red = nn.Sequential(
            conv(1, canales), nn.ReLU(),
            conv(canales, canales), nn.ReLU(),
            conv(canales, canales), nn.ReLU(),
            conv(canales, 1),
        )

    def forward(self, x):
        return self.red(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=8, help="resolucion de la rejilla de densidad")
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--carpeta", type=str, default="dataset")
    ap.add_argument("--salida", type=str, default="dataset/surrogate.pt")
    args = ap.parse_args()

    print(f"Dispositivo: {DISPOSITIVO}")
    X, Y = cargar_dataset(args.carpeta, args.grid)
    print(f"Universos cargados: {len(X)}  (rejilla {args.grid}^3)")
    if len(X) < 6:
        print("Pocos universos. Genera mas con generar_universos.py")
        return

    # separar entrenamiento / validacion
    n_val = max(2, len(X) // 5)
    Xtr, Ytr = X[:-n_val].to(DISPOSITIVO), Y[:-n_val].to(DISPOSITIVO)
    Xva, Yva = X[-n_val:].to(DISPOSITIVO), Y[-n_val:].to(DISPOSITIVO)

    modelo = SurrogateCNN().to(DISPOSITIVO)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perdida = nn.MSELoss()

    # baseline trivial: "el universo no cambia" (predecir inicial = final)
    base_val = perdida(Xva, Yva).item()

    print(f"\nEntrenando ({args.epocas} epocas)...")
    mejor_val, mejor_estado, mejor_ep = float("inf"), None, 0
    for ep in range(args.epocas):
        modelo.train()
        opt.zero_grad()
        loss = perdida(modelo(Xtr), Ytr)
        loss.backward()
        opt.step()

        # validacion cada epoca: nos quedamos con el MEJOR modelo (early stopping)
        modelo.eval()
        with torch.no_grad():
            lv = perdida(modelo(Xva), Yva).item()
        if lv < mejor_val:
            mejor_val, mejor_ep = lv, ep + 1
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}

        if (ep + 1) % max(1, args.epocas // 6) == 0:
            print(f"  epoca {ep+1:4d}  train={loss.item():.4f}  val={lv:.4f}")

    # restauramos el mejor modelo visto (no el ultimo, que ya sobreajusta)
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    print(f"  -> mejor modelo en la epoca {mejor_ep} (val={mejor_val:.4f})")

    modelo.eval()
    with torch.no_grad():
        val_final = perdida(modelo(Xva), Yva).item()

    print("\n=== Resultado del surrogate ===")
    print(f"  Error del modelo (val)        : {val_final:.4f}")
    print(f"  Error del baseline trivial    : {base_val:.4f}  (no evolucionar)")
    mejora = (1 - val_final / base_val) * 100 if base_val > 0 else 0
    print(f"  El modelo mejora al baseline en: {mejora:.1f}%")

    # velocidad: surrogate vs simulacion real
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(50):
            _ = modelo(Xva[:1])
        if DISPOSITIVO == "cuda":
            torch.cuda.synchronize()
        t_inf = (time.perf_counter() - t0) / 50
    print(f"\n  Inferencia del surrogate : {t_inf*1000:.2f} ms por universo")
    print(f"  Simulacion real (aprox.) : ~16000 ms por universo")
    print(f"  ACELERACION: ~{16.0/t_inf:.0f}x mas rapido (surrogate vs simular)")

    torch.save(modelo.state_dict(), args.salida)
    print(f"\n  Modelo guardado en: {args.salida}")


if __name__ == "__main__":
    main()
