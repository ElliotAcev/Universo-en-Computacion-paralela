"""Backend CUDA (GPU NVIDIA) usando PyTorch.

Aprovechamos que PyTorch ya esta instalado para entrenar el surrogate: sus
tensores en GPU nos dan el N-body acelerado sin escribir CUDA a mano.

Se usa en PC1 (RTX 4060) y PC2 (RTX 3050).
"""

import numpy as np


class CUDABackend:
    nombre = "CUDA (PyTorch)"

    def __init__(self, config=None):
        import torch  # si no esta instalado -> excepcion -> auto prueba OpenCL
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("no hay GPU CUDA utilizable en este PC")

        config = config or {}
        self.G = config.get("G", 1.0)
        self.eps2 = config.get("EPS2", 2.5e-3)
        self.box = config.get("BOX", 1.0)
        self.dev = torch.device("cuda")
        self.dispositivo = torch.cuda.get_device_name(0)

        # prueba real: si la GPU no puede operar, fallamos AQUI (no en mitad del lote)
        _ = (torch.zeros(8, device=self.dev) + 1).sum().item()

    def aceleraciones(self, pos, masa):
        t = self.torch
        p = t.as_tensor(pos, dtype=t.float32, device=self.dev)
        m = t.as_tensor(masa, dtype=t.float32, device=self.dev)

        diff = p[None, :, :] - p[:, None, :]
        diff -= self.box * t.round(diff / self.box)          # imagen minima
        d2 = (diff * diff).sum(-1) + self.eps2
        inv3 = d2.pow(-1.5)
        inv3.fill_diagonal_(0.0)                              # nadie se atrae a si mismo
        acc = self.G * t.einsum("ijk,ij,j->ik", diff, inv3, m)
        return acc.cpu().numpy().astype(np.float64)
