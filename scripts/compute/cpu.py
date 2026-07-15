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
        diff = pos[None, :, :] - pos[:, None, :]
        diff -= self.box * np.round(diff / self.box)
        dist2 = np.sum(diff * diff, axis=2) + self.eps2
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        np.fill_diagonal(inv_dist3, 0.0)
        return self.G * np.einsum("ijk,ij,j->ik", diff, inv_dist3, masa)
