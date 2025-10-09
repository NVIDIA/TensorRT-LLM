from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import Fp4QuantizedTensor
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .swiglu import swiglu


class GatedMLP(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
        overridden_tp_size: Optional[int] = None,
        reduce_output: bool = True,
        layer_idx: Optional[int] = None,
        use_cute_dsl_blockscaling_mm: bool = False,
        disable_deep_gemm: bool = False,
    ):

        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation

        config = config or ModelConfig()
        self.mapping = config.mapping
        if overridden_tp_size is not None:
            assert config.mapping.tp_size % overridden_tp_size == 0
            tp_size = overridden_tp_size
            # "Misuse" pp_size here to perform all-reduce within smaller groups
            pp_size = config.mapping.pp_size * config.mapping.tp_size // overridden_tp_size
            mapping = Mapping(
                world_size=tp_size * pp_size,
                rank=self.mapping.rank,
                gpus_per_node=self.mapping.gpus_per_node,
                tp_size=tp_size,
                pp_size=pp_size,
            )
        else:
            mapping = config.mapping

        self.gate_up_proj = Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
            quant_config=config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
            disable_deep_gemm=disable_deep_gemm,
        )

        self.down_lora = LoraLayer([LoraModuleType.MLP_4H_TO_H],
                                   [self.hidden_size])

        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            reduce_output=reduce_output,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.down_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
            disable_deep_gemm=disable_deep_gemm,
        )

        # These two modules are mutually exclusive - either splitted_gate_up_lora or fused_gate_up_lora will be used,
        # but never both at the same time. splitted_gate_up_lora handles gate and up separately while fused_gate_up_lora
        # handles them as a single fused operation.
        self.splitted_gate_up_lora = LoraLayer(
            [LoraModuleType.MLP_H_TO_4H, LoraModuleType.MLP_GATE], [
                self.intermediate_size // mapping.tp_size,
                self.intermediate_size // mapping.tp_size
            ])
        self.fused_gate_up_lora = LoraLayer(
            [LoraModuleType.MLP_GATE_UP],
            [2 * self.intermediate_size // mapping.tp_size])

    def _apply_activation(self, x, *, has_lora: bool = False):
        if self.activation == F.silu:
            if self.down_proj.has_fp8_qdq or self.down_proj.has_w4a8_nvfp4_fp8:
                if has_lora:
                    # NOTE: This is a WAR, since LoRA grouped_gemm does not support FP8 yet.
                    # TODO: Remove this path when LoRA grouped_gemm supports FP8
                    # see: cpp/tensorrt_llm/thop/loraOp.cpp::lora_grouped_gemm
                    logger.warning(
                        f"GatedMLP._apply_activation: LoRA path active; forcing non-FP8 activation dtype bf16/fp16, layer_idx={self.layer_idx}"
                    )
                    return swiglu(x)
                else:
                    return swiglu(x,
                                  quant_scale=self.down_proj.input_scale,
                                  quant_type=torch.float8_e4m3fn)
            else:
                return swiglu(x)
        elif callable(self.activation):
            return self.activation(x)
        elif self.activation is None:
            return x
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not yet implemented for fused GatedMLP"
            )

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if bool(lora_params):
            return self.forward_lora(x, all_rank_num_tokens,
                                     final_all_reduce_params, lora_params)

        h1 = self.gate_up_proj(x)
        h2 = self._apply_activation(h1)
        output = self.down_proj(h2,
                                all_reduce_params=final_all_reduce_params,
                                layer_idx=self.layer_idx)
        return output

    def forward_lora(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        assert lora_params is not None
        assert self.layer_idx is not None, "layer_idx is required for lora"

        h1 = self.gate_up_proj(x)

        h1_lora = self.splitted_gate_up_lora(x, lora_params, self.layer_idx)

        if h1_lora is not None:
            h1 = h1 + h1_lora

        h1_lora = self.fused_gate_up_lora(x, lora_params, self.layer_idx)
        if h1_lora is not None:
            h1 = h1 + h1_lora

        h2 = self._apply_activation(h1, has_lora=True)
        output = self.down_proj(h2,
                                all_reduce_params=final_all_reduce_params,
                                lora_params=lora_params,
                                layer_idx=self.layer_idx)

        return output
