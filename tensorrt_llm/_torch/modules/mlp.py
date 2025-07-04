from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig


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
        self.up_lora = LoraLayer(
            [LoraModuleType.MLP_H_TO_4H],
            [self.intermediate_size // config.mapping.tp_size])

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
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.up_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

        self.down_lora = LoraLayer([LoraModuleType.MLP_4H_TO_H],
                                   [self.hidden_size])
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.down_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

    def forward(
        self,
        x: torch.Tensor,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        if lora_params is not None:
            return self.forward_lora(x, lora_params=lora_params)

        x_up = self.up_proj(x)
        x_act = self.activation(x_up)
        x_down = self.down_proj(x_act)

        return x_down

    def forward_lora(
        self,
        x: torch.Tensor,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        assert lora_params is not None

        x_up = self.up_proj(x)

        assert self.layer_idx is not None, "layer_idx is required for lora"
        x_up_lora = self.up_lora(x, lora_params, self.layer_idx)
        if x_up_lora is not None:
            x_up = x_up + x_up_lora

        x_act = self.activation(x_up)
        x_down = self.down_proj(x_act,
                                lora_params=lora_params,
                                layer_idx=self.layer_idx)

        return x_down
