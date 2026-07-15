# Avance — Simulación de galaxias y gravedad

Este paquete contiene una presentación HTML y archivos de respaldo para presentar el avance del trabajo final de Computación Paralela y Distribuida.

## Archivos principales

- `presentacion_avance_galaxias.html`  
  Presentación navegable en navegador. Usa flechas izquierda/derecha.

- `resultados_mpi.csv`  
  Tabla con tiempos, speedup y eficiencia obtenidos en la demo MPI.

- `assets/speedup_mpi.png`  
  Gráfico de speedup MPI.

- `assets/eficiencia_mpi.png`  
  Gráfico de eficiencia paralela.

- `assets/galaxia_3d_preview.png`  
  Imagen de respaldo visual para mostrar la galaxia 3D.

- `scripts/mpi_nbody_demo.py`  
  Demo MPI del problema N-body directo. Paraleliza el cálculo de aceleraciones repartiendo cuerpos entre ranks.

- `scripts/visual_galaxia_3d.py`  
  Demo visual 3D. No se usa como métrica de speedup; sirve para mostrar la galaxia.

- `scripts/main_cuda_referencia.cu`  
  Archivo CUDA de referencia entregado por el usuario. Sirve como base para la versión GPU + OpenGL.

- `scripts/generar_graficos.py`  
  Script que genera los gráficos de respaldo a partir de los resultados.

## Cómo abrir la presentación

Abrir directamente:

```powershell
start .\presentacion_avance_galaxias.html
```

O hacer doble clic sobre `presentacion_avance_galaxias.html`.

## Cómo ejecutar la demo MPI

Instalar dependencia:

```powershell
pip install mpi4py numpy
```

Ejecutar con 4 procesos:

```powershell
mpiexec -n 4 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
```

Ejecutar comparación básica:

```powershell
mpiexec -n 1 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
mpiexec -n 2 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
mpiexec -n 4 python .\scripts\mpi_nbody_demo.py --n 3000 --steps 20
```

## Cómo ejecutar la visualización 3D

Instalar dependencias:

```powershell
pip install numpy matplotlib
```

Ejecutar:

```powershell
python .\scripts\visual_galaxia_3d.py --n 1200 --black-hole --trail
```

## Qué se debe decir en la presentación

La paralelización principal está en el cálculo gravitacional. En MPI, los cuerpos se distribuyen entre procesos y cada rank calcula la aceleración de un subconjunto de cuerpos. La visualización 3D no es la parte paralela principal; se usa como evidencia visual del modelo.

Barnes-Hut se plantea como optimización algorítmica para reducir el costo desde `O(N²)` hacia aproximadamente `O(N log N)`. CUDA se plantea como aceleración GPU para ejecutar miles de threads en paralelo y mostrar una galaxia 3D en tiempo real.
