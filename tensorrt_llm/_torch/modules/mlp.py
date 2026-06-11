from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import Fp4QuantizedTensor, gelu_tanh, relu2
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig


class MLP(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
        layer_idx: Optional[int] = None,
        reduce_output: bool = True,
        overridden_tp_size: Optional[int] = None,
        lora_up_module_type: LoraModuleType = LoraModuleType.MLP_H_TO_4H,
        lora_down_module_type: LoraModuleType = LoraModuleType.MLP_4H_TO_H,
        use_custom_cublas_mm: bool = False,
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

        self.up_lora = LoraLayer([lora_up_module_type],
                                 [self.intermediate_size // mapping.tp_size])

        self.up_proj = Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.VANILLA),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.up_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        self.down_lora = LoraLayer([lora_down_module_type], [self.hidden_size])
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.down_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        self._use_fused_relu2_quant = False

    def create_weights(self):
        self.up_proj.create_weights()
        self.down_proj.create_weights()

        has_nvfp4 = hasattr(self.down_proj,
                            'has_nvfp4') and self.down_proj.has_nvfp4
        has_kernel = hasattr(torch.ops.trtllm, 'fused_relu2_quantize')
        has_scale = hasattr(self.down_proj, 'input_scale')
        is_relu2 = self.activation is relu2
        # The fused relu2+fp4_quantize kernel body is guarded by
        # ``__CUDA_ARCH__ >= 1000`` (see fusedActivationQuant.cu). On pre-SM100
        # GPUs the kernel is a no-op, so fall back to unfused relu2 → separate
        # quantize in the downstream linear layer.
        is_sm100_or_later = get_sm_version() >= 100

        self._use_fused_relu2_quant = (has_nvfp4 and has_kernel and has_scale
                                       and is_relu2 and is_sm100_or_later)

    # Minimum M for the fp4out CuTe DSL GELU kernel; below this its SFC epilogue
    # can write out-of-bounds (CTA tile height > output rows), so fall back to eager.
    _FP4OUT_MIN_M = 128

    def forward(
        self,
        x: torch.Tensor,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        if lora_params is not None:
            return self.forward_lora(x, lora_params=lora_params)

        # Fuse up_proj GEMM + bias + GELU(tanh) + NVFP4-quant into the GEMM
        # epilogue (mirrors GatedMLP's SwiGLU fp4out path) when eligible.
        if self._can_fuse_gelu_fp4out():
            m = x.reshape(-1, x.shape[-1]).shape[0]
            if m >= MLP._FP4OUT_MIN_M:
                # Helper returns a rank-preserving Fp4QuantizedTensor; the NVFP4
                # down_proj flattens/unflattens 3D inputs (see linear.py).
                return self.down_proj(self._apply_fused_gelu_fp4out(x))

        x_up = self.up_proj(x)

        if self._use_fused_relu2_quant:
            x_act = self._fused_relu2_quant(x_up)
        else:
            x_act = self.activation(x_up)

        x_down = self.down_proj(x_act)

        return x_down

    def _can_fuse_gelu_fp4out(self) -> bool:
        """Eligible for the fused up-GEMM + bias + GELU(tanh) + NVFP4 epilogue op.

        Mirrors GatedMLP._can_fuse_gate_up_swiglu_fp4out: needs NVFP4 up/down
        projections, a static down_proj input_scale (the SFC norm_const), the
        GELU(tanh) activation, the Blackwell CuteDSL op, and SM 100/103.
        """
        return (self.activation is gelu_tanh and hasattr(
            torch.ops.trtllm, "cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell")
                and get_sm_version() in (100, 103)
                and getattr(self.up_proj, "has_nvfp4", False)
                and getattr(self.down_proj, "has_nvfp4", False)
                and not self.down_proj.force_dynamic_quantization
                and self.down_proj.input_scale is not None)

    def _apply_fused_gelu_fp4out(self, x: torch.Tensor) -> Fp4QuantizedTensor:
        """Fused up-GEMM + bias + GELU(tanh) + NVFP4 quant -> down_proj input.

        Preserves input rank: a [B, S, H] input returns a [B, S, H'/2]-packed
        Fp4QuantizedTensor so the NVFP4 down_proj unflattens its output correctly.
        """
        module = self.up_proj
        original_shape = None
        if x.dim() > 2:
            original_shape = x.shape
            x = x.reshape(-1, x.shape[-1])

        act_fp4, act_sf, alpha = module.quant_method._input_prepare(module, x)
        fp4_output, out_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell(
            act_fp4, module.weight, act_sf, module.weight_scale, alpha,
            self.down_proj.input_scale, module.bias)

        if original_shape is not None:
            fp4_output = fp4_output.reshape(*original_shape[:-1],
                                            fp4_output.shape[-1])
        return Fp4QuantizedTensor(fp4_output, out_sf)

    def _fused_relu2_quant(self, x: torch.Tensor) -> Fp4QuantizedTensor:
        x_flat = x.view(-1, x.shape[-1])

        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        if x_flat.dtype not in (torch.float16, torch.bfloat16):
            x_flat = x_flat.to(torch.bfloat16)

        fp4_tensor, sf_tensor = torch.ops.trtllm.fused_relu2_quantize(
            x_flat, self.down_proj.input_scale, 16)

        return Fp4QuantizedTensor(
            fp4_tensor=fp4_tensor,
            scaling_factor=sf_tensor,
            is_sf_swizzled=True,
        )

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
