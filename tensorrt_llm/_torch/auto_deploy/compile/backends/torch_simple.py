"""A simple no-op backend."""

import torch.nn as nn

from ..compiler import CompileBackendRegistry, CompilerBackend


@CompileBackendRegistry.register("torch-simple")
class TorchCompiler(CompilerBackend):
    def compile(self) -> nn.Module:
        return self.model
