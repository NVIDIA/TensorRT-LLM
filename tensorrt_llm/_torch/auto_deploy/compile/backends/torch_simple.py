"""A simple no-op backend."""

import torch.nn as nn

from ..compiler import BackendCompiler, BackendRegistry


@BackendRegistry.register("torch-simple")
class TorchCompiler(BackendCompiler):
    def compile(self) -> nn.Module:
        return self.gm
