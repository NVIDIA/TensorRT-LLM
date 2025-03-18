from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..custom_op import IS_FLASHINFER_AVAIABLE
from ..distributed import AllReduceParams, ParallelConfig, TensorParallelMode
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
                 config: Optional[ModelConfig] = None,
                 use_dp: bool = False,
                 is_expert: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation

        config = config or ModelConfig()
        if use_dp:
            tp_rank = 0
            tp_size = 1
        else:
            tp_rank = config.mapping.tp_rank
            tp_size = config.mapping.tp_size
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
                pipeline_parallel_size=config.mapping.pp_size,
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
                pipeline_parallel_size=config.mapping.pp_size,
                parallel_rank=config.mapping.rank),
            quant_config=config.get_quant_config(),
            is_expert=is_expert,
            skip_create_weights=config.skip_create_weights,
        )

    def forward(
        self,
        x: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        if self.activation == F.silu:
            return self.down_proj(swiglu(self.gate_up_proj(x)),
                                  all_reduce_params=final_all_reduce_params)
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not yet implemented for fused GatedMLP"
            )
