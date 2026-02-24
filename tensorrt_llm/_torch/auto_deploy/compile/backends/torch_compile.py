"""Backend that uses torch.compile only."""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

from ..compiler import CompileBackendRegistry, CompilerBackend


@CompileBackendRegistry.register("torch-compile")
class TorchCompileCompiler(CompilerBackend):
    def __init__(self, *args_for_init, **kwargs_for_init):
        super().__init__(*args_for_init, **kwargs_for_init)
        ad_logger.info(f"Torch Dynamo cache size limit {torch._dynamo.config.cache_size_limit=}")

    def compile(self) -> nn.Module:
        """Compile the model using torch.compile."""
        return torch.compile(self.model, dynamic=True)
