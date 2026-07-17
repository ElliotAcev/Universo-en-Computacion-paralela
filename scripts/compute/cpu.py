"""Backend CPU (NumPy). Siempre disponible: es el plan B de cualquier PC."""

import numpy as np


class CPUBackend:
    nombre = "CPU (NumPy)"
    dispositivo = "CPU"

    def __init__(self, config=None):
        config = config or {}
        self.G = config.get("G", 1.0)
        self.eps2 = config.get("EPS2", 2.5e-3)
        self.box = config.get("BOX", 1.0)

    def aceleraciones(self, pos, masa):
        """Gravedad N-body O(N^2) en caja periodica (convencion de imagen minima)."""
        return self.aceleraciones_rango(pos, masa, 0, len(masa))

    def aceleraciones_rango(self, pos, masa, inicio, cuantas):
        """Acelera SOLO las particulas [inicio, inicio+cuantas), pero contra TODAS.

        Es lo que necesita el modo distribuido (Modo B): cada nodo calcula su
        trozo del universo. Ojo: la gravedad es de largo alcance, asi que hace
        falta la posicion de todas las particulas, no solo las del trozo.
        """
        locales = pos[inicio:inicio + cuantas]
        diff = pos[None, :, :] - locales[:, None, :]
        diff -= self.box * np.round(diff / self.box)
        dist2 = np.sum(diff * diff, axis=2) + self.eps2
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        # una particula no se atrae a si misma
        idx = np.arange(cuantas)
        inv_dist3[idx, inicio + idx] = 0.0
        return self.G * np.einsum("ijk,ij,j->ik", diff, inv_dist3, masa)
