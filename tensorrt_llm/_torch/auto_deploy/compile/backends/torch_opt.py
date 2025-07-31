"""Mixed backend with torch.compile + Cudagraph."""

import torch

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

from ..compiler import BackendRegistry
from .torch_cudagraph import CapturedGraph, TorchCudagraphCompiler


@BackendRegistry.register("torch-opt")
class TorchOptCompiler(TorchCudagraphCompiler):
    """Compiler that uses both torch.compile and CUDA graphs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch._dynamo.config.recompile_limit = max(
            len(self.cuda_graph_batch_sizes), torch._dynamo.config.recompile_limit
        )
        ad_logger.info(f"Setting recompile limit to {torch._dynamo.config.recompile_limit}")

        # Global torch config, set the torch compile cache to fix up to llama 405B
        torch._dynamo.config.cache_size_limit = 20
        ad_logger.info(f"Setting cache size limit to {torch._dynamo.config.cache_size_limit}")

    def _init_captured_graph(self, gm, in_spec, out_spec) -> CapturedGraph:
        gm = torch.compile(gm, dynamic=True)
        return super()._init_captured_graph(gm, in_spec, out_spec)
