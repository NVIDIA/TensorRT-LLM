"""Mixed backend with torch.compile + Cudagraph."""

import torch

from ..compiler import BackendRegistry
from .torch_cudagraph import CapturedGraph, TorchCudagraphCompiler


@BackendRegistry.register("torch-opt")
class TorchOptCompiler(TorchCudagraphCompiler):
    """Compiler that uses both torch.compile and CUDA graphs."""

    def _init_captured_graph(self, gm, in_spec, out_spec) -> CapturedGraph:
        gm = torch.compile(gm, dynamic=True)
        return super()._init_captured_graph(gm, in_spec, out_spec)
