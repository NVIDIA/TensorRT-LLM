"""Mixed backend with torch.compile + Cudagraph."""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

from ..compiler import CompileBackendRegistry
from .torch_cudagraph import TorchCudagraphCompiler


@CompileBackendRegistry.register("torch-opt")
class TorchOptCompiler(TorchCudagraphCompiler):
    """Compiler that uses both torch.compile and CUDA graphs."""

    def __init__(self, *args_for_init, **kwargs_for_init):
        super().__init__(*args_for_init, **kwargs_for_init)
        torch._dynamo.config.recompile_limit = max(
            len(self.cuda_graph_batch_sizes), torch._dynamo.config.recompile_limit
        )
        ad_logger.info(
            f"Setting Torch Dynamo recompile limit {torch._dynamo.config.recompile_limit=}; "
            f"{torch._dynamo.config.cache_size_limit=}"
        )

    def compile(self) -> nn.Module:
        self.model = torch.compile(self.model, dynamic=True)
        return super().compile()
