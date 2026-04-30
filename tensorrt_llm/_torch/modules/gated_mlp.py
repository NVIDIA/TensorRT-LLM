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
        use_custom_cublas_mm: bool = False,
        is_shared_expert: bool = False,
        swiglu_limit: Optional[float] = None,
    ):

        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.use_cute_dsl_blockscaling_mm = use_cute_dsl_blockscaling_mm
        self.swiglu_limit = float(swiglu_limit) if swiglu_limit is not None else None

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

        # Calculate local intermediate size after tensor parallel sharding
        tp_size = mapping.tp_size
        local_intermediate_size = self.intermediate_size // tp_size

        gateup_shard_indices_mapping = {
            'gate': (0, local_intermediate_size),
            'up': (local_intermediate_size, local_intermediate_size),
        }

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
            fused_weight_shard_indices_mapping=gateup_shard_indices_mapping,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        if is_shared_expert:
            down_type = LoraModuleType.SHARED_EXPERT_4H_TO_H
            h_to_4h_type = LoraModuleType.SHARED_EXPERT_H_TO_4H
            gate_type = LoraModuleType.SHARED_EXPERT_GATE
        else:
            down_type = LoraModuleType.MLP_4H_TO_H
            h_to_4h_type = LoraModuleType.MLP_H_TO_4H
            gate_type = LoraModuleType.MLP_GATE

        self.down_lora = LoraLayer([down_type], [self.hidden_size])

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
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        # These two modules are mutually exclusive - either splitted_gate_up_lora or fused_gate_up_lora will be used,
        # but never both at the same time. splitted_gate_up_lora handles gate and up separately while fused_gate_up_lora
        # handles them as a single fused operation.
        self.splitted_gate_up_lora = LoraLayer([h_to_4h_type, gate_type], [
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
                    return swiglu(x, swiglu_limit=self.swiglu_limit)
                else:
                    return swiglu(x,
                                  quant_scale=self.down_proj.input_scale,
                                  quant_type=torch.float8_e4m3fn,
                                  swiglu_limit=self.swiglu_limit)
            else:
                return swiglu(x, swiglu_limit=self.swiglu_limit)
        elif callable(self.activation):
            return self.activation(x)
        elif self.activation is None:
            return x
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not yet implemented for fused GatedMLP"
            )

    def _can_fuse_gate_up_swiglu(self):
        """Check if fused GEMM + SwiGLU path is available.

        Returns True when all conditions are met:
        - CuteDSL blockscaling mode is enabled (implies Blackwell + CuteDSL)
        - Activation is SwiGLU (F.silu)
        - gate_up_proj uses NVFP4 quantization
        - gate_up_proj has no bias (bias not supported in fused kernel)
        """
        return (self.use_cute_dsl_blockscaling_mm and self.activation == F.silu
                and self.gate_up_proj.has_nvfp4
                and not self.gate_up_proj.has_bias)

    def _can_fuse_gate_up_swiglu_fp4out(self):
        """Check if fused GEMM + SwiGLU with FP4 output path is available.

        Extends _can_fuse_gate_up_swiglu with additional conditions:
        - down_proj also uses NVFP4 (so it can consume FP4 input)
        - down_proj has static input_scale (needed as norm_const for SFC)
        - down_proj does not use dynamic quantization
        """
        if not self._can_fuse_gate_up_swiglu():
            return False
        if not self.down_proj.has_nvfp4:
            return False
        if self.down_proj.force_dynamic_quantization:
            return False
        if self.down_proj.input_scale is None:
            return False
        return True

    def _fused_gate_up_swiglu(self, x, fp4_out=False):
        """Fused FC1 GEMM + SwiGLU using CuteDSL dense kernel.

        Bypasses the separate gate_up_proj GEMM and swiglu Triton kernel,
        fusing them into a single CuteDSL kernel with SwiGLU in the epilogue.

        Args:
            x: Input tensor or Fp4QuantizedTensor.
            fp4_out: If True, produce FP4 output with scale factors,
                eliminating the bf16→fp4 requantization between FC1 and FC2.

        Returns:
            If fp4_out is False: Output tensor [m, intermediate_size/tp] in bfloat16.
            If fp4_out is True: Fp4QuantizedTensor with fused SwiGLU output.
        """
        module = self.gate_up_proj

        # Handle multi-dimensional inputs (e.g., 3D: batch, seq, hidden)
        original_shape = None
        if not isinstance(x, (tuple, Fp4QuantizedTensor)) and x.dim() > 2:
            original_shape = x.shape
            x = x.reshape(-1, x.shape[-1])

        # Get quantized inputs from Linear's NVFP4 pipeline
        act_fp4, act_sf, alpha = module.quant_method._input_prepare(module, x)

        if fp4_out:
            # FC2's input_scale serves as norm_const for SFC quantization
            global_sf = self.down_proj.input_scale
            fp4_output, out_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell(
                act_fp4, module.weight, act_sf, module.weight_scale, alpha,
                global_sf)
            if original_shape is not None:
                fp4_output = fp4_output.reshape(*original_shape[:-1],
                                                fp4_output.shape[-1])
            return Fp4QuantizedTensor(fp4_output, out_sf)

        # BF16 output path
        output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
            act_fp4, module.weight, act_sf, module.weight_scale, alpha,
            module.dtype)

        # Trim padding if weight was padded beyond logical out_features
        expected_out = module.out_features // 2
        if output.shape[-1] > expected_out:
            output = output[..., :expected_out].contiguous()

        if original_shape is not None:
            output = output.reshape(*original_shape[:-1], output.shape[-1])

        return output

    # Minimum M dimension for the fp4out CuTe DSL kernel.
    # Below this, the kernel's SFC epilogue may write out-of-bounds
    # because the CTA tile height exceeds the output allocation.
    _FP4OUT_MIN_M = 128

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

        if self._can_fuse_gate_up_swiglu_fp4out():
            # Get token count for minimum-M check
            if isinstance(x, (tuple, Fp4QuantizedTensor)):
                m = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
            else:
                m = x.reshape(
                    -1, x.shape[-1]).shape[0] if x.dim() > 2 else x.shape[0]
            h2 = self._fused_gate_up_swiglu(x,
                                            fp4_out=m >= GatedMLP._FP4OUT_MIN_M)
        elif self._can_fuse_gate_up_swiglu():
            h2 = self._fused_gate_up_swiglu(x)
        else:
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
