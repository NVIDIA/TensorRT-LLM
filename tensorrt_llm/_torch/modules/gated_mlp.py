from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..custom_op import IS_FLASHINFER_AVAIABLE
from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from .linear import Linear, WeightMode, WeightsLoadingConfig


def swiglu(x):
    if IS_FLASHINFER_AVAIABLE:
        # WAR for flashinfer activation since it does not support custom op properly
        from ..custom_op import flashinfer_silu_and_mul
        return flashinfer_silu_and_mul(x)
    else:
        gate, x = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class GatedMLP(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 intermediate_size: int,
                 bias: bool,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 dtype: Optional[torch.dtype] = None,
                 config: Optional[ModelConfig] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation

        config = config or ModelConfig()
        tp_rank = config.mapping.tp_rank
        tp_size = config.mapping.tp_size

        self.gate_up_proj = Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias=bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
            quant_config=config.get_quant_config(),
        )
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.ROW),
            quant_config=config.get_quant_config(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == F.silu:
            return self.down_proj(swiglu(self.gate_up_proj(x)))
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not yet implemented for fused GatedMLP"
            )
