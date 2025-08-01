"""Backend that uses torch.compile only."""

import torch
import torch.nn as nn

from ..compiler import BackendCompiler, BackendRegistry


@BackendRegistry.register("torch-compile")
class TorchCompileCompiler(BackendCompiler):
    def compile(self) -> nn.Module:
        """Compile the model using torch.compile."""
        return torch.compile(self.gm, dynamic=True)
