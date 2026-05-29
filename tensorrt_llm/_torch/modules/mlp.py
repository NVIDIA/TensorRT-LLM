from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping

from ..._utils import nvtx_range
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
        self._use_fused_gelu_tanh_quant = False

    def create_weights(self):
        self.up_proj.create_weights()
        self.down_proj.create_weights()

        has_nvfp4 = hasattr(self.down_proj,
                            'has_nvfp4') and self.down_proj.has_nvfp4
        has_kernel = hasattr(torch.ops.trtllm, 'fused_relu2_quantize')
        # NVFP4LinearMethod.create_weights always allocates input_scale as a
        # Parameter (linear.py:1295), but a layer that opts into dynamic quant
        # at load time will reset it to None (linear.py:821). The fused kernel
        # needs a real scalar tensor, so guard against the None case explicitly.
        has_scale = getattr(self.down_proj, 'input_scale', None) is not None
        is_relu2 = self.activation is relu2

        self._use_fused_relu2_quant = has_nvfp4 and has_kernel and has_scale and is_relu2

        # Static-only fast path for GELU(tanh) + NVFP4 down_proj. The Linear
        # NVFP4 path returns a stale `module.alpha` when handed an
        # Fp4QuantizedTensor (linear.py:1263-1270); under dynamic quant
        # `module.alpha` is never calibrated, so we gate on
        # `not force_dynamic_quantization` to ensure the GEMM sees a valid
        # calibrated alpha. NVFP4 layers do not set `has_static_input_scale`
        # (that flag is FP8-only), so `force_dynamic_quantization` plus the
        # `input_scale is not None` check above are the canonical signals.
        has_kernel_gelu = hasattr(torch.ops.trtllm, 'fused_gelu_tanh_quantize')
        is_gelu_tanh = self.activation is gelu_tanh
        not_dynamic = not getattr(self.down_proj,
                                  "force_dynamic_quantization", False)

        self._use_fused_gelu_tanh_quant = (has_nvfp4 and has_kernel_gelu
                                           and has_scale and not_dynamic
                                           and is_gelu_tanh)

    def forward(
        self,
        x: torch.Tensor,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        if lora_params is not None:
            return self.forward_lora(x, lora_params=lora_params)

        x_up = self.up_proj(x)

        if self._use_fused_relu2_quant:
            # Distinct NVTX label so the chunk-1 fused activation+quant fast
            # path is visible in nsys traces (separate from the auto layerwise
            # marker on `MLP.forward`).
            with nvtx_range("relu2+NVFP4 fused", color="green"):
                x_act = self._fused_relu2_quant(x_up)
        elif self._use_fused_gelu_tanh_quant:
            with nvtx_range("gelu_tanh+NVFP4 fused", color="green"):
                x_act = self._fused_gelu_tanh_quant(x_up)
        else:
            x_act = self.activation(x_up)

        x_down = self.down_proj(x_act)

        return x_down

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

    def _fused_gelu_tanh_quant(self, x: torch.Tensor) -> Fp4QuantizedTensor:
        x_flat = x.view(-1, x.shape[-1])

        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        if x_flat.dtype not in (torch.float16, torch.bfloat16):
            x_flat = x_flat.to(torch.bfloat16)

        fp4_tensor, sf_tensor = torch.ops.trtllm.fused_gelu_tanh_quantize(
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
