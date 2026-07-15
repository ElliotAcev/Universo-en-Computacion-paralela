# Cómo compilar y ejecutar la simulación de galaxia (CUDA + OpenGL)

Guía para compilar `scripts/main_cuda_referencia.cu` en Windows. Esta receta es la que **funciona** — sigue los pasos en orden.

---

## 1. Requisitos previos

| Herramienta | Versión | Nota |
|-------------|---------|------|
| **Visual Studio 2022 Build Tools** | 2022 | ⚠️ **NO usar VS2026** (es incompatible con CUDA 12.6, da error de cudafe++) |
| **CUDA Toolkit (nvcc)** | 12.6 | El compilador de NVIDIA |
| **GPU NVIDIA** | Compatible con `sm_89` | Ej. RTX 4060 (Ada Lovelace). Ajustar la arquitectura si tu GPU es distinta |
| **vcpkg** | cualquiera | Para instalar las librerías gráficas |

---

## 2. Instalar las librerías (glfw3 y glew) con vcpkg

El proyecto necesita **GLFW** (ventana/entrada) y **GLEW** (extensiones OpenGL). Se instalan con vcpkg — **no** están en este repo porque son dependencias externas.

```powershell
# 1. Clonar vcpkg (fuera de la carpeta del proyecto, p. ej. en C:\)
git clone https://github.com/microsoft/vcpkg C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat

# 2. Instalar las dos librerías (versión x64)
.\vcpkg install glfw3:x64-windows glew:x64-windows
```

Anota la ruta donde vcpkg dejó los headers y libs:
```
C:\vcpkg\installed\x64-windows\include   <- headers (glfw3, glew)
C:\vcpkg\installed\x64-windows\lib       <- .lib
C:\vcpkg\installed\x64-windows\bin       <- glfw3.dll, glew32.dll
```

---

## 3. Preparar el entorno de compilación

CUDA necesita el compilador de Visual Studio. Abre una terminal y carga las variables de VS2022:

```powershell
# Ajusta la ruta según tu instalación de Build Tools
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

> Si aparece un warning tipo *"vswhere.exe not recognized"*, **ignóralo**: es inofensivo y la compilación igual tiene éxito.

---

## 4. Compilar

```powershell
# Si el ejecutable anterior está corriendo, ciérralo primero (queda bloqueado):
taskkill /F /IM galaxy.exe

# Compilar con nvcc (ajusta las rutas de vcpkg si son distintas)
nvcc scripts\main_cuda_referencia.cu -o galaxy.exe ^
  -arch=sm_89 ^
  -I C:\vcpkg\installed\x64-windows\include ^
  -L C:\vcpkg\installed\x64-windows\lib ^
  -lglfw3dll -lglew32 -lopengl32
```

> **`-arch=sm_89`** es para RTX 4060 (Ada Lovelace). Si tu GPU es otra, cambia el número:
> - RTX 30xx (Ampere) → `sm_86`
> - RTX 20xx (Turing) → `sm_75`
> - GTX 16xx → `sm_75`

Copia los DLL junto al ejecutable (o ten la carpeta `bin` de vcpkg en el PATH):
```powershell
copy C:\vcpkg\installed\x64-windows\bin\glfw3.dll .
copy C:\vcpkg\installed\x64-windows\bin\glew32.dll .
```

---

## 5. Ejecutar

```powershell
.\galaxy.exe
```

### Controles
- **WASD** — mover la cámara
- **Mouse (arrastrar)** — mirar alrededor
- **Scroll** — velocidad de vuelo
- **SHIFT** — turbo
- **P** — pausa

---

## 6. Rendimiento esperado (RTX 4060, solo física)

| N (partículas) | FPS |
|----------------|-----|
| 50.000 | 117 |
| 100.000 | 33 |
| 150.000 | 15 |
| 200.000 | 8 |

Para ~10 FPS fluidos: N ≈ 150–180k. El valor por defecto es 200.000.

---

## Problemas comunes

| Síntoma | Causa / solución |
|---------|------------------|
| Error de cudafe++ / ACCESS_VIOLATION | Estás usando VS2026 → instala **VS2022 Build Tools** |
| `nvcc no reconocido` | No cargaste `vcvars64.bat`, o CUDA no está en el PATH |
| Falta `glfw3.dll` / `glew32.dll` | Copia los DLL desde `vcpkg\...\bin` junto al `.exe` |
| No compila por `sm_89` | Tu GPU es distinta → cambia `-arch` (ver tabla arriba) |
| El `.exe` no se regenera | Ciérralo con `taskkill /F /IM galaxy.exe` antes de recompilar |
