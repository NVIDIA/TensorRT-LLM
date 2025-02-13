from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from .linear import Linear, WeightMode, WeightsLoadingConfig


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
        tp_rank = config.mapping.tp_rank
        tp_size = config.mapping.tp_size
        gpus_per_node = config.mapping.gpus_per_node
        self.up_proj = Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.VANILLA),
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
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node),
            quant_config=config.get_quant_config(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))
