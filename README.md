# El Nacimiento del Universo — Simulación N-body distribuida con IA

Proyecto final del curso **Computación Paralela y Distribuida (UTEM)**.

Simulamos la **formación de estructura cósmica** —desde un universo casi uniforme hasta la telaraña de galaxias— repartiendo el cálculo gravitacional entre un **clúster de 3 GPUs heterogéneas** en distintas ubicaciones geográficas, y coronamos el pipeline con **dos modelos de inteligencia artificial**: uno que optimiza el reparto de carga y otro que emula la simulación física. El recorrido termina con un **zoom en tiempo real** hasta una galaxia espiral renderizada en GPU.

---

## De qué trata

El problema de N-cuerpos gravitacional (calcular cómo se atraen millones de partículas) es tan costoso que no cabe en un solo computador. El proyecto lo aborda en tres capas:

1. **Paralelismo masivo (GPU):** cada partícula se calcula en miles de hilos con CUDA / OpenCL.
2. **Paralelismo distribuido (clúster):** la simulación se reparte entre 3 PCs conectados por red virtual.
3. **Inteligencia artificial:** dos modelos aprenden a acelerar y optimizar el proceso.

**Narrativa:** Big Bang → la gravedad teje la telaraña cósmica sobre el clúster → se hace zoom a un nodo denso → aparece una galaxia espiral realista, simulada en tiempo real.

---

## Componentes del proyecto

### 1. Motor de galaxia en tiempo real (CUDA + OpenGL)
Aplicación de escritorio interactiva que simula una galaxia espiral con física N-body en la GPU y la renderiza en tiempo real con cámara libre.
- Física N-body O(N²) con *shared-memory tiling* e integración *leapfrog* simpléctica.
- Interoperabilidad **CUDA–OpenGL** (los datos nunca vuelven a la CPU).
- Galaxia realista: 2 brazos espirales, bulbo, agujero negro supermasivo central, polvo y nebulosas tipo Andrómeda, *bloom* HDR.
- ~200.000 cuerpos en tiempo real sobre una RTX 4060.
- Código: `scripts/main_cuda_referencia.cu` · Compilación: **[COMPILAR.md](COMPILAR.md)**

### 2. Motor cosmológico distribuido (MPI + CUDA/OpenCL)
Simulación del nacimiento del universo repartida entre los 3 PCs mediante **descomposición de dominio**: cada nodo posee una región del espacio e intercambia solo las fronteras.
- Reparto y comunicación con **MPI** (MS-MPI).
- Cómputo por GPU: **CUDA** en las NVIDIA, **OpenCL** en la AMD.
- Snapshots del estado guardados en **HDF5** para reproducir la evolución de forma fluida.

### 3. Modelo de IA #1 — Balanceo de carga
Como las 3 GPUs son desiguales y de dos familias distintas, un modelo aprende a repartir el trabajo por **tiempo medido** en vez de por número de partículas, minimizando el tiempo de cada paso.
- Implementado con **scikit-learn** (regresión de tiempos por GPU).
- Balanceo dinámico: los dominios se reasignan según la carga.

### 4. Modelo de IA #2 — Surrogate neuronal
Una red neuronal aprende a **emular el resultado del N-body** sin calcular todas las fuerzas, acelerando la simulación varios órdenes de magnitud.
- Implementado con **PyTorch** (CNN 3D sobre rejilla de densidad).
- Entrenado con los snapshots generados por el propio clúster (dataset propio).

---

## Herramientas utilizadas

| Área | Tecnología |
|------|------------|
| Cómputo GPU (NVIDIA) | **CUDA 12.6** (C++) |
| Cómputo GPU (AMD) | **OpenCL** / PyOpenCL |
| Render en tiempo real | **OpenGL** (GLFW + GLEW) |
| Cómputo distribuido | **MPI** (Microsoft MPI) + mpi4py |
| Red del clúster | **Tailscale** (VPN de malla entre PCs remotos) |
| IA · balanceo | **scikit-learn** |
| IA · surrogate | **PyTorch** |
| Datos / snapshots | **HDF5** (h5py), NumPy, SciPy |
| Visualización / gráficos | Matplotlib |
| Build (Windows) | VS2022 Build Tools + vcpkg |

---

## El clúster (3 PCs heterogéneos)

| PC | GPU | Backend | Rol |
|----|-----|---------|-----|
| **PC1** | RTX 4060 | CUDA | Maestro: coordina, entrena la IA, render y demo |
| **PC2** | RTX 3050 | CUDA | Trabajador: genera universos (lote grande) |
| **PC3** | AMD Radeon RX | OpenCL | Trabajador: genera universos (lote ligero) |

Los 3 están en ubicaciones distintas y se ven como una misma red gracias a Tailscale. El diseño es **tolerante a la latencia**: el cómputo pesado se reparte por lotes y los resultados se reproducen de forma fluida, en vez de sincronizar cada frame por internet.

---

## Estructura del repositorio

```
scripts/
  main_cuda_referencia.cu   Motor de galaxia CUDA + OpenGL (tiempo real)
  mpi_nbody_demo.py         Demo N-body distribuida con MPI (mide speedup)
  visual_galaxia_3d.py      Visualización 3D del modelo
  generar_graficos.py       Genera los gráficos de resultados
resultados_mpi.csv          Tiempos, speedup y eficiencia medidos
assets/                     Gráficos de speedup, eficiencia y preview 3D
COMPILAR.md                 Guía de dependencias y compilación (CUDA/OpenGL)
```

---

## Cómo ejecutar

### Motor de galaxia (CUDA + OpenGL)
Ver la guía completa en **[COMPILAR.md](COMPILAR.md)**. Resumen:
```powershell
nvcc scripts\main_cuda_referencia.cu -o galaxy.exe -arch=sm_89 ^
  -I <vcpkg>\include -L <vcpkg>\lib -lglfw3dll -lglew32 -lopengl32
.\galaxy.exe
```
Controles: **WASD** mover · **mouse** mirar · **scroll** velocidad · **SHIFT** turbo · **P** pausa.

### Demo distribuida (MPI)
```powershell
pip install mpi4py numpy
mpiexec -n 1 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
mpiexec -n 2 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
mpiexec -n 4 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
```

### Visualización 3D
```powershell
pip install numpy matplotlib
python .\scripts\visual_galaxia_3d.py --n 1200 --black-hole --trail
```

---

## Resultados

- **Speedup del clúster:** la generación distribuida escala de forma casi lineal con los 3 PCs (ver `resultados_mpi.csv` y `assets/speedup_mpi.png`).
- **Balanceo:** el reparto aprendido reduce el tiempo por paso frente al reparto ingenuo en el clúster heterogéneo.
- **Surrogate:** la red emula la simulación varios órdenes de magnitud más rápido que el cálculo N-body directo.

Optimización algorítmica: **Barnes-Hut** reduce el costo de O(N²) hacia O(N log N); **CUDA/OpenCL** aportan la aceleración por GPU y **MPI** la distribución entre nodos.
