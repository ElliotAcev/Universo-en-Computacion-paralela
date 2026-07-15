"""
simular_universo.py — Simulador N-body del nacimiento del universo (mono-nodo)

Genera un universo casi uniforme con pequeñas fluctuaciones, deja que la gravedad
forme la telaraña cósmica (cúmulos y filamentos), y guarda snapshots del estado.

Es la base del proyecto:
  - Produce el "estado inicial -> estado evolucionado" que entrena el SURROGATE.
  - Mide el tiempo de cálculo, dato que usará el MODELO DE BALANCEO.
  - Cada universo es independiente, así que luego se reparten entre los 3 PCs.

Uso:
    python scripts/simular_universo.py --n 3000 --steps 300 --seed 1 --guardar
    python scripts/simular_universo.py --n 2000 --steps 200 --snapshots 20

Salida:
    dataset/universo_seedXX.npz  con:
        pos_inicial (N,3), pos_final (N,3), y opcionalmente snapshots (T,N,3)
"""

import argparse
import os
import time
import numpy as np

# --- Parámetros físicos (unidades de simulación, no del mundo real) ---
G = 1.0            # constante gravitacional
EPS2 = 2.5e-3      # softening al cuadrado (evita fuerzas infinitas al chocar)
DT = 0.0025        # paso de tiempo
BOX = 1.0          # tamaño de la caja cúbica [0, BOX]^3


# --- Parámetros cosmológicos de las condiciones iniciales ---
NS = 0.96          # índice espectral primordial (ΛCDM: casi invariante de escala)
GAMMA = 0.30       # parámetro de forma del espectro (dónde está el 'turnover')
AMPLITUD = 0.18    # cuán fuertes son las fluctuaciones iniciales (fracción de celda)
VEL_FACTOR = 6.0   # velocidades del modo creciente (proporcionales al desplazamiento)


def espectro_potencia(k):
    """Espectro de potencia ΛCDM: P(k) = k^ns * T(k)^2, con la función de
    transferencia BBKS. Es el 'patrón' estadístico real de las fluctuaciones
    primordiales medido en el fondo cósmico de microondas (CMB)."""
    k = np.maximum(k, 1e-10)
    q = k / GAMMA
    T = (np.log(1 + 2.34 * q) / (2.34 * q) *
         (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25)
    return k ** NS * T ** 2


def condiciones_iniciales(n, seed=0):
    """Big Bang realista mediante la aproximación de ZEL'DOVICH (el método que
    usan los generadores cosmológicos MUSIC / N-GenIC):

      1. Se crea un campo de densidad gaussiano en el espacio de Fourier con
         amplitud dada por el espectro de potencia ΛCDM (fluctuaciones del CMB).
      2. Se calcula el campo de desplazamiento (gradiente del potencial).
      3. Se desplazan las partículas desde una rejilla perfecta según ese campo.

    Resultado: un universo casi uniforme pero con estructura CORRELACIONADA
    (protofilamentos), no ruido blanco. La gravedad la amplifica hasta la
    telaraña cósmica.
    """
    rng = np.random.default_rng(seed)
    M = int(round(n ** (1 / 3)))
    n = M ** 3  # ajustamos n a un cubo perfecto

    # rejilla lagrangiana (posiciones sin perturbar)
    lin = np.arange(M) / M * BOX
    qx, qy, qz = np.meshgrid(lin, lin, lin, indexing="ij")

    # ruido blanco -> a Fourier
    wk = np.fft.fftn(rng.standard_normal((M, M, M)))

    # vectores de onda de la caja periódica
    k1d = 2 * np.pi * np.fft.fftfreq(M, d=BOX / M)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0  # evita división por cero en el modo medio

    # campo de densidad con el espectro ΛCDM
    delta_k = wk * np.sqrt(espectro_potencia(np.sqrt(k2)))
    delta_k[0, 0, 0] = 0.0  # densidad media cero

    # desplazamiento de Zel'dovich: psi_k = i k delta_k / k^2
    psix = np.fft.ifftn(1j * kx * delta_k / k2).real
    psiy = np.fft.ifftn(1j * ky * delta_k / k2).real
    psiz = np.fft.ifftn(1j * kz * delta_k / k2).real

    # normalizamos la fuerza de la perturbación a una fracción de celda
    sigma = np.sqrt(psix ** 2 + psiy ** 2 + psiz ** 2).std()
    escala = AMPLITUD * (BOX / M) / max(sigma, 1e-12)
    psix *= escala; psiy *= escala; psiz *= escala

    pos = np.column_stack([(qx + psix).ravel(), (qy + psiy).ravel(), (qz + psiz).ravel()])
    pos %= BOX  # envolver en la caja periódica

    # velocidades del modo creciente (proporcionales al desplazamiento)
    vel = np.column_stack([psix.ravel(), psiy.ravel(), psiz.ravel()]) * VEL_FACTOR

    masa = np.full(n, 1.0 / n, dtype=np.float64)
    return pos.astype(np.float64), vel.astype(np.float64), masa, n


def aceleraciones(pos, masa):
    """Gravedad N-body directa O(N^2) en una CAJA PERIÓDICA.

    Usa la convención de 'imagen mínima': para cada par se toma la copia más
    cercana a través de los bordes que se envuelven. Esto evita que todo
    colapse a un único punto y deja que se formen filamentos y cúmulos
    locales -> la telaraña cósmica.
    """
    diff = pos[None, :, :] - pos[:, None, :]          # (N,N,3)
    diff -= BOX * np.round(diff / BOX)                 # imagen mínima (periódico)
    dist2 = np.sum(diff * diff, axis=2) + EPS2         # (N,N)
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    np.fill_diagonal(inv_dist3, 0.0)                   # nadie se atrae a sí mismo
    acc = G * np.einsum("ijk,ij,j->ik", diff, inv_dist3, masa)
    return acc


def contraste_densidad(pos, celdas=8):
    """Mide cuánto se ha agrupado la materia: reparte las partículas en una
    rejilla y devuelve la desviación relativa del nº por celda.
    0 = perfectamente uniforme; sube al formarse cúmulos/filamentos."""
    idx = np.floor(pos / BOX * celdas).astype(int) % celdas
    plano = (idx[:, 0] * celdas + idx[:, 1]) * celdas + idx[:, 2]
    conteo = np.bincount(plano, minlength=celdas ** 3).astype(float)
    return conteo.std() / conteo.mean()


def simular(n, steps, seed, n_snapshots=0, backend=None):
    """Evoluciona el universo con integración leapfrog (KDK).

    'backend' calcula la gravedad: CPU (NumPy), CUDA o OpenCL. Si es None se usa
    la version NumPy de este modulo (para que el script siga funcionando solo).
    """
    pos, vel, masa, n = condiciones_iniciales(n, seed)
    pos_inicial = pos.copy()

    calc_acc = backend.aceleraciones if backend is not None else aceleraciones

    guardar_cada = max(1, steps // n_snapshots) if n_snapshots > 0 else 0
    snapshots = []

    t0 = time.perf_counter()
    acc = calc_acc(pos, masa)
    for paso in range(steps):
        vel += 0.5 * DT * acc          # medio empujón
        pos += DT * vel                # mover
        pos %= BOX                     # envolver en la caja periódica
        acc = calc_acc(pos, masa)
        vel += 0.5 * DT * acc          # otro medio empujón
        if guardar_cada and paso % guardar_cada == 0:
            snapshots.append(pos.copy())
    elapsed = time.perf_counter() - t0

    resultado = {
        "pos_inicial": pos_inicial,
        "pos_final": pos,
        "n": n,
        "steps": steps,
        "seed": seed,
        "tiempo_s": elapsed,
    }
    if snapshots:
        resultado["snapshots"] = np.array(snapshots)
    return resultado


def main():
    ap = argparse.ArgumentParser(description="Simulador del nacimiento del universo (N-body)")
    ap.add_argument("--n", type=int, default=3000, help="nº de partículas (se ajusta a un cubo perfecto)")
    ap.add_argument("--steps", type=int, default=300, help="pasos de tiempo")
    ap.add_argument("--seed", type=int, default=0, help="semilla (cada semilla = un universo distinto)")
    ap.add_argument("--snapshots", type=int, default=0, help="nº de snapshots intermedios a guardar")
    ap.add_argument("--guardar", action="store_true", help="guardar resultado en dataset/")
    ap.add_argument("--salida", type=str, default="dataset", help="carpeta de salida")
    args = ap.parse_args()

    print(f"Simulando universo: seed={args.seed}  N~{args.n}  pasos={args.steps}")
    res = simular(args.n, args.steps, args.seed, args.snapshots)

    # cuánto se agrupó la materia (contraste de densidad en rejilla)
    c_ini = contraste_densidad(res["pos_inicial"])
    c_fin = contraste_densidad(res["pos_final"])
    print(f"  N real (cubo)         : {res['n']}")
    print(f"  Tiempo de calculo     : {res['tiempo_s']:.3f} s")
    print(f"  Contraste inicial     : {c_ini:.3f}")
    print(f"  Contraste final       : {c_fin:.3f}  (mayor = mas telarana cosmica)")

    if args.guardar:
        os.makedirs(args.salida, exist_ok=True)
        ruta = os.path.join(args.salida, f"universo_seed{args.seed:02d}.npz")
        # guardamos arrays y metadatos (tiempo_s lo usa el modelo de balanceo)
        np.savez_compressed(ruta, **res)
        print(f"  Guardado en        : {ruta}")


if __name__ == "__main__":
    main()
