# Cómo conectar tu PC al clúster

Guía para montar un PC como **nodo del clúster** que simula el nacimiento del universo. Si estás leyendo esto, es porque ya clonaste el repositorio. 👍

> ¿Te trabas en algún paso? Abre **Claude Code** en tu PC, muéstrale este archivo y pídele ayuda con ese paso concreto.

---

## Qué hace tu PC

Tu computadora **calcula universos** que le manda el maestro (PC1, el de Elliot).

- **No** tienes que programar nada.
- **No** tienes que lanzar comandos raros: el maestro dispara todo desde su PC.
- **No** tienes que enviarle archivos: los resultados vuelven solos por la red.

Tu única tarea: **dejar el PC preparado y encendido**, con una ventana corriendo.

| PC | GPU | Rol |
|----|-----|-----|
| PC1 | RTX 4060 | 🧠 Maestro: reparte y junta resultados |
| PC2 | RTX 3050 | 💪 Trabajador |
| PC3 | AMD Radeon | 💪 Trabajador (usa OpenCL) |

---

## 1️⃣ Tailscale — conecta tu PC con los demás

Como están en casas distintas, este programa los une por internet como si fueran una misma red local.

1. Acepta la **invitación** que te llegó por correo de parte de Elliot.
2. Descarga Tailscale de **https://tailscale.com/download** e instálalo.
3. Inicia sesión **con la cuenta con la que aceptaste la invitación**.
4. Copia tu dirección en la red y **pásasela a Elliot**:
```powershell
tailscale ip -4
```
Te dará algo tipo `100.x.y.z`.

---

## 2️⃣ Python 3.11 y las librerías

Instala **Python 3.11** desde [python.org](https://www.python.org/downloads/) (marca "Add Python to PATH").

Luego, en la carpeta del proyecto:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Tu GPU (elige SOLO la línea de tu tarjeta)
```powershell
# Si tienes NVIDIA (RTX 3050, 4060...):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Si tienes AMD Radeon:
pip install pyopencl
#   (necesitas los drivers AMD Adrenalin, que ya traen OpenCL)
```

> No pasa nada si esto falla: el programa usa la CPU automáticamente, solo que más lento.

---

## 3️⃣ MS-MPI — el que recibe las órdenes del maestro

1. Descarga de **https://www.microsoft.com/en-us/download/details.aspx?id=105289**
2. Instala **los dos archivos**: `msmpisetup.exe` y `msmpisdk.msi`.
3. Abre PowerShell y deja **esta ventana abierta** mientras trabajen (no la cierres):
```powershell
smpd -d
```

> ⚠️ **El error nº 1:** cuando Windows pregunte por el **Firewall**, dale **Permitir** a `mpiexec.exe` y `smpd.exe`. Si no, el maestro no podrá conectarse a tu PC.

---

## 4️⃣ Mantén el código actualizado

El código **no viaja por la red**: cada PC usa su propia copia. Si Elliot cambia algo, actualiza antes de trabajar:
```powershell
git pull
```
> Si los PCs tienen versiones distintas del código, la simulación puede fallar o descoordinarse.

---

## ✅ ¿Cómo sé que estoy listo?

Repasa esto y avísale a Elliot:

- [ ] Tailscale conectado (haz `ping <IP-de-Elliot>` y que responda).
- [ ] Le pasaste tu IP (`tailscale ip -4`).
- [ ] La ventana con `smpd -d` está abierta y corriendo.
- [ ] El Firewall permite `mpiexec.exe` y `smpd.exe`.
- [ ] `pip install -r requirements.txt` terminó sin errores.
- [ ] Hiciste `git pull` (tienes la última versión).

Cuando esté todo ✅, dile: **"listo"**. Él lanzará el diagnóstico.

---

## 🔍 El diagnóstico (lo lanza Elliot)

Elliot correrá `diagnostico.py`, que revisa **tu PC** y le dice exactamente qué te falta (versión de Python, librerías, GPU, permisos...). Si algo está mal, aparecerá tu nombre de host y el problema concreto.

Después lanzará la prueba de humo (`hola_mpi.py`): si en su pantalla aparece una línea con **el nombre de tu PC**, ya eres parte del clúster. 🎉

---

## 💡 Qué pasa cuando empiece el trabajo de verdad

1. Elliot lanza el comando desde su PC.
2. Tu PC recibe un lote de universos y **tu GPU los calcula** (unos segundos cada uno).
3. Los resultados **vuelven solos** al PC de Elliot por la red.

Verás mensajes en tu ventana indicando el progreso. Tú no haces nada más que tener el PC encendido.

---

📌 **Ante cualquier duda, el coordinador es Elliot (PC1).**
