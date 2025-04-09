from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from .linear import Linear, WeightMode, WeightsLoadingConfig


class MLP(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 intermediate_size: int,
                 bias: bool,
                 activation: Callable[[torch.Tensor], torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None,
                 config: Optional[ModelConfig] = None,
                 layer_idx: Optional[int] = None):

        super().__init__()
        self.layer_idx = layer_idx
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
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=config.mapping.pp_size,
                parallel_rank=config.mapping.rank),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.VANILLA),
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights)

        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=config.mapping.pp_size,
                parallel_rank=config.mapping.rank),
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights)

        self.up_lora = LoraLayer([LoraModuleType.MLP_H_TO_4H],
                                 [self.intermediate_size])
        self.down_lora = LoraLayer([LoraModuleType.MLP_4H_TO_H],
                                   [self.hidden_size])

    def forward(
        self,
        x: torch.Tensor,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:

        x_up = self.up_proj(x)
        if lora_params is not None:
            assert self.layer_idx is not None, "layer_idx is required for lora"
            x_up_lora = self.up_lora(x, lora_params, self.layer_idx)
            if x_up_lora is not None:
                x_up = x_up + x_up_lora

        x_act = self.activation(x_up)
        x_down = self.down_proj(x_act)

        if lora_params is not None:
            x_down_lora = self.down_lora(x_act, lora_params, self.layer_idx)
            if x_down_lora is not None:
                x_down = x_down + x_down_lora

        return x_down
