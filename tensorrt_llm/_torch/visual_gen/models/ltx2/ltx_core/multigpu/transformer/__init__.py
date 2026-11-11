"""
Multi-GPU transformer utilities for LTX models.
This module provides utilities for running LTX transformer models across multiple GPUs
using tiled data parallelism.
"""

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.multigpu.transformer.tiled_data_parallel import (
    TiledDataParallelModelWrapper,
)

__all__ = [
    "TiledDataParallelModelWrapper",
]
