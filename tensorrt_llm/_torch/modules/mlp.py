from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ..model_config import ModelConfig
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig


class MLP(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 intermediate_size: int,
                 bias: bool,
                 activation: Callable[[torch.Tensor], torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None,
                 config: Optional[ModelConfig] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation

        config = config or ModelConfig()
        self.up_proj = Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            dtype=dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.VANILLA),
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights,
        )
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))
