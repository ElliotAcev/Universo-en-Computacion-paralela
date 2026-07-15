from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Datos obtenidos en la demo MPI del usuario.
processes = np.array([1, 2, 4])
times_3000 = np.array([8.344851, 4.448759, 2.330754])
times_5000 = np.array([22.437188, 13.853276, 9.176025])
speedup_3000 = times_3000[0] / times_3000
speedup_5000 = times_5000[0] / times_5000
ideal = processes.astype(float)

plt.figure(figsize=(8, 5))
plt.plot(processes, ideal, marker="o", label="Speedup ideal")
plt.plot(processes, speedup_3000, marker="o", label="N = 3000")
plt.plot(processes, speedup_5000, marker="o", label="N = 5000")
plt.xticks(processes)
plt.xlabel("Procesos MPI")
plt.ylabel("Speedup")
plt.title("Speedup medido en demo MPI N-body")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "speedup_mpi.png", dpi=180)
plt.close()

# Imagen de galaxia 3D: vista estática para respaldar avance visual.
rng = np.random.default_rng(42)
n = 2500
arms = 3
r = rng.gamma(shape=2.0, scale=0.85, size=n)
r = np.clip(r, 0.05, 5.8)
arm_id = rng.integers(0, arms, size=n)
theta = (2 * np.pi * arm_id / arms) + 1.35 * r + rng.normal(0.0, 0.22, size=n)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = rng.normal(0, 0.07 + 0.012 * r, size=n)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, s=np.clip(12 / (r + 0.45), 1, 8), alpha=0.7)
ax.scatter([0], [0], [0], s=80, marker="o")
ax.set_title("Vista 3D inicial: galaxia espiral con masa central")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-2, 2)
ax.view_init(elev=27, azim=38)
plt.tight_layout()
plt.savefig(OUT / "galaxia_3d_preview.png", dpi=180)
plt.close()

# Barras de eficiencia.
eff_3000 = speedup_3000 / processes
eff_5000 = speedup_5000 / processes
xpos = np.arange(len(processes))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(xpos - width/2, eff_3000, width, label="N = 3000")
plt.bar(xpos + width/2, eff_5000, width, label="N = 5000")
plt.xticks(xpos, [str(p) for p in processes])
plt.ylim(0, 1.1)
plt.xlabel("Procesos MPI")
plt.ylabel("Eficiencia")
plt.title("Eficiencia paralela MPI")
plt.grid(True, axis="y", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "eficiencia_mpi.png", dpi=180)
plt.close()

print("Gráficos generados en", OUT)
