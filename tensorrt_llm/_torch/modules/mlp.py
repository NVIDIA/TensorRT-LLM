# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from typing import Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import Fp4QuantizedTensor, gelu_tanh, relu2
from .linear import (Linear, TensorParallelMode, UnquantizedLinearMethod,
                     WeightMode, WeightsLoadingConfig,
                     is_static_nvfp4_input_eligible)


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
        self._use_fused_gelu = False
        self._use_fused_gelu_fp4out = False

    def create_weights(self):
        self.up_proj.create_weights()
        self.down_proj.create_weights()

        has_static_nvfp4_input = is_static_nvfp4_input_eligible(self.down_proj)
        has_kernel = hasattr(torch.ops.trtllm, 'fused_relu2_quantize')
        is_relu2 = self.activation is relu2
        # The fused relu2+fp4_quantize kernel body is guarded by
        # ``__CUDA_ARCH__ >= 1000`` (see fusedActivationQuant.cu). On pre-SM100
        # GPUs the kernel is a no-op, so fall back to unfused relu2 → separate
        # quantize in the downstream linear layer.
        is_sm100_or_later = get_sm_version() >= 100

        self._use_fused_relu2_quant = (has_static_nvfp4_input and has_kernel
                                       and is_relu2 and is_sm100_or_later)

        # Static eligibility for the fused GELU(tanh) CuteDSL epilogue (mirrors
        # GatedMLP); the runtime quant_method check is deferred to first forward.
        self._use_fused_gelu, self._use_fused_gelu_fp4out = (
            self._nvfp4_gelu_fusion_eligibility())

    # Minimum M for the fp4out CuTe DSL GELU kernel; below this its SFC epilogue
    # can write out-of-bounds (CTA tile height > output rows), so fall back to
    # the (still fused) bf16-out path.
    _FP4OUT_MIN_M = 128

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        if lora_params is not None:
            return self.forward_lora(x, lora_params=lora_params)

        # Fuse up_proj GEMM + bias + GELU(tanh) (+ NVFP4-quant) into the GEMM
        # epilogue (mirrors GatedMLP). Static eligibility can go stale: quant_method
        # may be downgraded to unquantized after create_weights (e.g. LTX-2
        # quant-exclusion), so re-check the NVFP4 _input_prepare at runtime (a
        # torch.compile trace-time guard, not a per-step cost); else fall back to eager.
        if (self._use_fused_gelu
                and self.up_proj.has_nvfp4_activation_quantization
                and hasattr(getattr(self.up_proj, "quant_method", None),
                            "_input_prepare")):
            if (self._use_fused_gelu_fp4out
                    and is_static_nvfp4_input_eligible(self.down_proj)
                    and hasattr(getattr(self.down_proj, "quant_method", None),
                                "_input_prepare")):
                m = self._token_count(x)
                return self.down_proj(
                    self._fused_gelu(x, fp4_out=m >= MLP._FP4OUT_MIN_M))
            return self.down_proj(self._fused_gelu(x))

        if self._unquantized_gelu_fusion_eligible(x):
            x_act = self._fused_up_proj_gelu(x)
        else:
            x_up = self.up_proj(x)
            # Weight loading may replace the quantization method after
            # create_weights(), so do not rely on the cached eligibility alone.
            if (self._use_fused_relu2_quant
                    and is_static_nvfp4_input_eligible(self.down_proj)):
                x_act = self._fused_relu2_quant(x_up)
            else:
                x_act = self.activation(x_up)

        x_down = self.down_proj(x_act)

        return x_down

    def _unquantized_gelu_fusion_eligible(self, x: torch.Tensor) -> bool:
        """Whether the unquantized up projection can use the cuBLASLt GELU
        epilogue without bypassing Linear post-processing or another GEMM
        backend.

        Opt-in: the epilogue applies GELU to the fp32 GEMM accumulator whereas
        eager rounds the GEMM output to bf16 first. The per-layer difference is
        at bf16-ULP level, but it is a different rounding trajectory and
        compounds across residual blocks x denoising steps in diffusion
        transformers (LTX-2, Wan), whose reference outputs predate the fusion.
        """
        if os.environ.get("TRTLLM_MLP_FUSE_GELU_EPILOGUE", "0") != "1":
            return False
        up_proj = self.up_proj
        return (self.activation is gelu_tanh
                and hasattr(torch, "_addmm_activation")
                and isinstance(x, torch.Tensor) and x.dim() >= 2 and x.is_cuda
                and x.dtype == torch.bfloat16 and not torch.is_grad_enabled()
                and type(up_proj.quant_method) is UnquantizedLinearMethod
                and up_proj.bias is not None
                and up_proj.weight.dtype == torch.bfloat16
                and not up_proj.gather_output
                and not up_proj.use_custom_cublas_mm
                and not up_proj.use_cute_dsl_bf16_gemm)

    def _fused_up_proj_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bf16 up projection with a fused GELU(tanh) epilogue."""
        input_shape = x.shape
        # VisualGen diffusion transformers commonly process long token
        # sequences. Folding GELU into the up-projection avoids an extra
        # full-tensor read/write and standalone activation kernel at large M.
        output = torch._addmm_activation(self.up_proj.bias,
                                         x.reshape(-1, input_shape[-1]),
                                         self.up_proj.weight.t(),
                                         use_gelu=True)
        return output.reshape(*input_shape[:-1], output.shape[-1])

    def _nvfp4_gelu_fusion_eligibility(self) -> Tuple[bool, bool]:
        """Return (bf16_out_ok, fp4_out_ok) static eligibility for the fused
        GELU(tanh) epilogue (mirrors GatedMLP's SwiGLU paths). Requires the
        Blackwell CuteDSL op(s), SM 100/103, a local (not gathered) NVFP4
        up_proj output; fp4-out builds on bf16-out and also needs an NVFP4
        down_proj with a static input_scale and no forced dynamic quantization.
        The runtime quant_method check is applied in forward (quant_method can
        be downgraded after this).
        """
        if (self.activation is not gelu_tanh or self.up_proj.gather_output
                or get_sm_version() not in (100, 103) or not getattr(
                    self.up_proj, "has_nvfp4_activation_quantization", False)):
            return False, False
        bf16_ok = hasattr(torch.ops.trtllm,
                          "cute_dsl_nvfp4_dense_gemm_gelu_blackwell")
        fp4_ok = (bf16_ok and hasattr(
            torch.ops.trtllm, "cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell")
                  and is_static_nvfp4_input_eligible(self.down_proj))
        return bf16_ok, fp4_ok

    @staticmethod
    def _token_count(x: Union[torch.Tensor, tuple, Fp4QuantizedTensor]) -> int:
        # Row count M = product of leading dims (B*S for [B, S, ...]); keeps the
        # fp4-out switch correct for rank-3 / pre-quantized inputs.
        t = x.fp4_tensor if isinstance(
            x, Fp4QuantizedTensor) else (x[0] if isinstance(x, tuple) else x)
        return t.reshape(-1,
                         t.shape[-1]).shape[0] if t.dim() > 2 else t.shape[0]

    def _fused_gelu(
        self,
        x: Union[torch.Tensor, tuple, Fp4QuantizedTensor],
        fp4_out: bool = False,
    ) -> Union[torch.Tensor, Fp4QuantizedTensor]:
        """Fused up-GEMM + bias + GELU(tanh) using the Blackwell CuteDSL kernel.

        Mirrors GatedMLP._fused_gate_up_swiglu for the non-gated GELU(tanh)
        activation. Accepts a plain tensor, a tuple, or a pre-quantized
        Fp4QuantizedTensor, and preserves input rank.

        Returns a bf16 tensor, or (when fp4_out) a rank-preserving
        Fp4QuantizedTensor that the NVFP4 down_proj consumes directly.
        """
        module = self.up_proj

        original_shape = None
        if not isinstance(x, (tuple, Fp4QuantizedTensor)) and x.dim() > 2:
            original_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
        elif isinstance(x, Fp4QuantizedTensor) and x.fp4_tensor.dim() > 2:
            # Fused GEMM needs a 2D mat1: flatten a rank-3 fp4 activation's data
            # to [M, D/2] (its SF is already flat-M, so reuse it); unflatten below.
            original_shape = x.fp4_tensor.shape
            x = Fp4QuantizedTensor(
                fp4_tensor=x.fp4_tensor.reshape(-1, x.fp4_tensor.shape[-1]),
                scaling_factor=x.scaling_factor,
                is_sf_swizzled=x.is_sf_swizzled,
            )

        act_fp4, act_sf, alpha = module.quant_method._input_prepare(module, x)

        if fp4_out:
            # down_proj's input_scale serves as norm_const for SFC quantization.
            fp4_output, out_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell(
                act_fp4, module.weight, act_sf, module.weight_scale, alpha,
                self.down_proj.input_scale, module.bias)
            if original_shape is not None:
                fp4_output = fp4_output.reshape(*original_shape[:-1],
                                                fp4_output.shape[-1])
            return Fp4QuantizedTensor(fp4_output, out_sf)

        # bf16-out path (down_proj re-quantizes its input itself).
        output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_blackwell(
            act_fp4, module.weight, act_sf, module.weight_scale, alpha,
            module.dtype, module.bias)
        # Non-gated: trim any padding beyond the logical out_features.
        if output.shape[-1] > module.out_features:
            output = output[..., :module.out_features].contiguous()
        if original_shape is not None:
            output = output.reshape(*original_shape[:-1], output.shape[-1])
        return output

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

        assert self.layer_idx is not None, "layer_idx is required for lora"
        x_up = LoraLayer.forward_with_base(
            lambda: self.up_proj(x),
            (self.up_lora, ),
            x,
            lora_params,
            self.layer_idx,
        )

        x_act = self.activation(x_up)
        x_down = self.down_proj(x_act,
                                lora_params=lora_params,
                                layer_idx=self.layer_idx)

        return x_down
