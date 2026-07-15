"""
compute — Backends de calculo para el N-body (CPU / CUDA / OpenCL)

Cada PC del cluster tiene hardware distinto:
    PC1 RTX 4060  -> CUDA
    PC2 RTX 3050  -> CUDA
    PC3 AMD Radeon-> OpenCL

Con backend="auto" cada PC DETECTA su hardware y elige el backend por si mismo,
asi el mismo comando funciona en los tres.

IMPORTANTE: la deteccion comprueba que exista un DISPOSITIVO usable, no solo que
la libreria se pueda importar (importar pyopencl no significa que haya GPU).

Uso:
    from compute import create_backend
    backend = create_backend("auto")      # o "cpu" / "cuda" / "opencl"
    acc = backend.aceleraciones(pos, masa)
"""

import warnings


def _probar_cuda(config):
    from .cuda import CUDABackend
    return CUDABackend(config)


def _probar_opencl(config):
    from .opencl import OpenCLBackend
    return OpenCLBackend(config)


def _cpu(config):
    from .cpu import CPUBackend
    return CPUBackend(config)


def create_backend(backend_name="auto", config=None):
    """Devuelve el backend pedido. Con 'auto' prueba CUDA -> OpenCL -> CPU.

    Cada intento CONSTRUYE el backend de verdad (compila el kernel, reserva el
    dispositivo). Si algo falla, pasa al siguiente. Nunca lanza excepcion:
    en el peor caso devuelve CPU, que siempre funciona.
    """
    config = config or {}
    nombre = (backend_name or "auto").lower()

    if nombre == "cpu":
        return _cpu(config)
    if nombre == "cuda":
        return _probar_cuda(config)
    if nombre == "opencl":
        return _probar_opencl(config)
    if nombre != "auto":
        raise ValueError(f"backend desconocido: {backend_name!r} (usa auto/cpu/cuda/opencl)")

    # --- auto-deteccion ---
    for intento, fabrica in (("CUDA", _probar_cuda), ("OpenCL", _probar_opencl)):
        try:
            return fabrica(config)
        except Exception as e:
            warnings.warn(f"{intento} no disponible aqui ({type(e).__name__}: {e}); probando el siguiente")
    return _cpu(config)
