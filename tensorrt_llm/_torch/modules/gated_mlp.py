from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType

from ..custom_ops import IS_FLASHINFER_AVAIABLE
from ..distributed import AllReduceParams, ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from ..utils import Fp4QuantizedTensor
from .linear import Linear, WeightMode, WeightsLoadingConfig


def swiglu(x):
    if IS_FLASHINFER_AVAIABLE:
        # WAR for flashinfer activation since it does not support custom op properly
        from ..custom_ops import flashinfer_silu_and_mul
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
                 config: Optional[ModelConfig] = None,
                 overridden_tp_size: Optional[int] = None,
                 is_expert: bool = False,
                 layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation

        config = config or ModelConfig()
        if overridden_tp_size is not None:
            assert config.mapping.tp_size % overridden_tp_size == 0
            tp_rank = config.mapping.tp_rank % overridden_tp_size
            tp_size = overridden_tp_size
            # "Misuse" pp_size here to perform all-reduce within smaller groups
            pp_size = config.mapping.pp_size * config.mapping.tp_size // overridden_tp_size
        else:
            tp_rank = config.mapping.tp_rank
            tp_size = config.mapping.tp_size
            pp_size = config.mapping.pp_size
        gpus_per_node = config.mapping.gpus_per_node

        self.gate_up_proj = Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias=bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=pp_size,
                parallel_rank=config.mapping.rank),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
            quant_config=config.get_quant_config(),
            is_expert=is_expert,
            skip_create_weights=config.skip_create_weights,
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
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=pp_size,
                parallel_rank=config.mapping.rank),
            quant_config=config.get_quant_config(),
            is_expert=is_expert,
            skip_create_weights=config.skip_create_weights,
        )

        # These two modules are mutually exclusive - either splitted_gate_up_lora or fused_gate_up_lora will be used,
        # but never both at the same time. splitted_gate_up_lora handles gate and up separately while fused_gate_up_lora
        # handles them as a single fused operation.
        self.splitted_gate_up_lora = LoraLayer(
            [LoraModuleType.MLP_H_TO_4H, LoraModuleType.MLP_GATE],
            [self.intermediate_size, self.intermediate_size])
        self.fused_gate_up_lora = LoraLayer([LoraModuleType.MLP_GATE_UP],
                                            [2 * self.intermediate_size])
        self.down_lora = LoraLayer([LoraModuleType.MLP_4H_TO_H],
                                   [self.hidden_size])

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        min_latency_mode: Optional[bool] = False,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        if self.activation == F.silu:
            h1 = self.gate_up_proj(x)
            if lora_params is not None:
                assert self.layer_idx is not None, "layer_idx is required for lora"
                h1_lora = self.splitted_gate_up_lora(x, lora_params,
                                                     self.layer_idx)
                if h1_lora is not None:
                    h1 = h1 + h1_lora

                h1_lora = self.fused_gate_up_lora(x, lora_params,
                                                  self.layer_idx)

                if h1_lora is not None:
                    h1 = h1 + h1_lora

            h2 = swiglu(h1)
            output = self.down_proj(h2,
                                    all_reduce_params=final_all_reduce_params)
            if lora_params is not None:
                output_lora = self.down_lora(h2, lora_params, self.layer_idx)
                if output_lora is not None:
                    output = output + output_lora

            return output
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not yet implemented for fused GatedMLP"
            )
