"""
visual_galaxia_3d.py

Demo visual 3D para respaldar el avance.
No es la parte principal de paralelización: la paralelización está en mpi_nbody_demo.py y en main_cuda_referencia.cu.

Ejecutar:
    python .\scripts\visual_galaxia_3d.py --n 1200 --black-hole --trail
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

G = 1.0


def init_galaxy(n, black_hole=False, seed=42):
    rng = np.random.default_rng(seed)
    r = rng.gamma(shape=2.0, scale=0.85, size=n)
    r = np.clip(r, 0.1, 6.0)
    arms = 3
    arm = rng.integers(0, arms, size=n)
    theta = 2 * np.pi * arm / arms + 1.35 * r + rng.normal(0, 0.22, size=n)
    pos = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        rng.normal(0, 0.06 + 0.015 * r, size=n),
    ]).astype(float)

    galaxy_mass = 120.0
    bh_mass = 80.0 if black_hole else 0.0
    enclosed = galaxy_mass * (r**3) / ((r**2 + 1.6**2) ** 1.5) + bh_mass
    v = np.sqrt(G * enclosed / np.maximum(r, 0.12))
    vel = np.column_stack([
        -np.sin(theta) * v,
        np.cos(theta) * v,
        rng.normal(0, 0.02, size=n),
    ]).astype(float)
    return pos, vel


def disk_acceleration(pos, black_hole=False, bh_mass=80.0, eps=0.15):
    # Potencial externo suavizado de disco + agujero negro central.
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    galaxy_mass = 120.0
    a, b = 1.8, 0.25
    R2 = x * x + y * y
    B = np.sqrt(z * z + b * b)
    C = a + B
    denom = (R2 + C * C) ** 1.5 + 1e-12
    acc = np.column_stack([
        -G * galaxy_mass * x / denom,
        -G * galaxy_mass * y / denom,
        -G * galaxy_mass * C * z / (B * denom + 1e-12),
    ])

    if black_hole:
        r2 = np.sum(pos * pos, axis=1) + eps * eps
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        acc += -G * bh_mass * pos * inv_r3[:, None]
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--black-hole", action="store_true")
    parser.add_argument("--trail", action="store_true")
    args = parser.parse_args()

    pos, vel = init_galaxy(args.n, args.black_hole)
    alive = np.ones(args.n, dtype=bool)
    capture_radius = 0.12

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Galaxia 3D con masa central" + (" y agujero negro" if args.black_hole else ""))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    radius = np.linalg.norm(pos[:, :2], axis=1)
    sizes = np.clip(12 / (radius + 0.4), 1.0, 9.0)
    scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, alpha=0.8, depthshade=True)
    ax.scatter([0], [0], [0], s=90, marker="o")
    info = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    lines = []
    trail_ids = np.linspace(0, args.n - 1, min(18, args.n), dtype=int)
    history = {int(i): [] for i in trail_ids}
    if args.trail:
        for _ in trail_ids:
            line, = ax.plot([], [], [], linewidth=0.7, alpha=0.45)
            lines.append(line)

    stats = {"captured": 0}
    dt = args.dt
    start_time = time.perf_counter()

    for frame in range(args.steps):
        acc0 = disk_acceleration(pos, args.black_hole)
        vel[alive] += 0.5 * dt * acc0[alive]
        pos[alive] += dt * vel[alive]
        acc1 = disk_acceleration(pos, args.black_hole)
        vel[alive] += 0.5 * dt * acc1[alive]
        vel[alive] *= 0.9999

        if args.black_hole:
            dist = np.linalg.norm(pos, axis=1)
            captured = alive & (dist < capture_radius)
            if np.any(captured):
                stats["captured"] += int(np.sum(captured))
                alive[captured] = False
                pos[captured] = np.nan
                vel[captured] = 0.0

        visible = alive & np.isfinite(pos[:, 0])
        scat.set_offsets(pos[visible, :2])
        scat.set_3d_properties(pos[visible, 2], zdir='z')
        scat.set_sizes(sizes[visible])
        ax.view_init(elev=24, azim=frame * 0.2)
        fps = frame / max(time.perf_counter() - start_time, 1e-9)
        info.set_text(f"N={args.n} | vivas={int(np.sum(alive))} | capturadas={stats['captured']} | FPS≈{fps:.1f}")

        if args.trail:
            for line, idx in zip(lines, trail_ids):
                i = int(idx)
                if alive[i] and np.isfinite(pos[i, 0]):
                    history[i].append(pos[i].copy())
                    if len(history[i]) > 70:
                        history[i].pop(0)
                if history[i]:
                    h = np.array(history[i])
                    line.set_data(h[:, 0], h[:, 1])
                    line.set_3d_properties(h[:, 2])

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
