"""Unit-loading NVFP4 conversion and companion-aware LoRA fusion."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import KeyValueOperation, SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import FuseRule
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.convert import (
    _is_ltx2_fv_nvfp4_layer,
    quantize_bf16_weight_to_nvfp4,
    scale_shape,
    swap_convert_linears,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.kernels import require_nvfp4
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.types import ActScale
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.policy import QuantizationPolicy

_FLOAT_INPUTS = frozenset({torch.bfloat16, torch.float16, torch.float32})


def _selected(key: str, value: torch.Tensor) -> bool:
    return (
        key.endswith(".weight")
        and _is_ltx2_fv_nvfp4_layer(key.removesuffix(".weight"))
        and value.ndim == 2
        and value.dtype in _FLOAT_INPUTS
    )


def _quantize_weight(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return quantize_bf16_weight_to_nvfp4(value if value.is_cuda else value.to("cuda"))


def _convert(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    value = inputs[key]
    if not _selected(key, value):
        return {key: value}
    if value.shape[1] % 32:
        raise ValueError(f"NVFP4 weight {key!r} input dimension must be divisible by 32")
    packed, scales, decode = _quantize_weight(value)
    prefix = key.removesuffix(".weight")
    return {key: packed, f"{prefix}.weight_scale": scales, f"{prefix}.weight_scale_2": decode}


def _convert_abstract(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    value = inputs[key]
    if not _selected(key, value):
        return {key: value}
    rows, width = value.shape
    if width % 32:
        raise ValueError(f"NVFP4 weight {key!r} input dimension must be divisible by 32")
    prefix = key.removesuffix(".weight")
    return {
        key: torch.empty((rows, width // 2), dtype=torch.uint8, device="meta"),
        f"{prefix}.weight_scale": torch.empty(scale_shape(rows, width), dtype=torch.uint8, device="meta"),
        f"{prefix}.weight_scale_2": torch.empty((), dtype=torch.float32, device="meta"),
    }


def convert_stage() -> SDOps:
    """Convert eligible transformer weights and emit NVFP4 scale companions."""
    return (
        SDOps("nvfp4-online-convert")
        .with_matching()
        .with_kv_operation(
            KeyValueOperation(_convert, _convert_abstract, preserve_output_device=True),
            key_prefix="transformer_blocks.",
            key_suffix=".weight",
        )
    )


def convert_swap_op(*, act_scale: ActScale = ActScale.FIXED_1) -> ModuleOps:
    """Replace eligible transformer linears with online NVFP4 modules."""
    if act_scale not in (ActScale.FIXED_1, ActScale.DYNAMIC):
        raise ValueError("online NVFP4 conversion needs FIXED_1 or DYNAMIC activation scale")
    return ModuleOps(
        f"nvfp4-online-swap:act={act_scale.value}",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=lambda model: swap_convert_linears(model, act_scale=act_scale),
    )


def _patch(
    key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if weight.dtype != torch.uint8:
        if not weight.is_floating_point():
            raise ValueError(f"{key!r} is not an NVFP4 or floating weight")
        return {key: (weight.to(delta.dtype) + delta).to(weight.dtype)}
    prefix = key.removesuffix(".weight")
    scale_key, decode_key = f"{prefix}.weight_scale", f"{prefix}.weight_scale_2"
    original_device = weight.device
    scale = clean[scale_key]
    weight_decode = clean[decode_key]
    compute_device = torch.device("cuda") if torch.cuda.is_available() else original_device
    packed = weight.to(compute_device)
    scales = scale.to(compute_device)
    decode = weight_decode.to(device=compute_device, dtype=torch.float32).reshape(())
    decoded = require_nvfp4().dequantize_nvfp4(packed, decode, scales, out_dtype=torch.bfloat16)
    merged = decoded + delta.to(device=compute_device, dtype=torch.bfloat16)
    new_packed, new_scales, new_decode = quantize_bf16_weight_to_nvfp4(merged)
    return {
        key: new_packed.to(original_device),
        scale_key: new_scales.to(original_device),
        decode_key: new_decode.to(original_device),
    }


def _logical_weight_shape(shape: torch.Size, dtype: torch.dtype) -> torch.Size:
    return torch.Size((shape[0], shape[1] * 2)) if dtype == torch.uint8 and len(shape) == 2 else shape


NVFP4_FUSE_RULE = FuseRule("nvfp4", torch.bfloat16, _patch, (".weight_scale", ".weight_scale_2"), _logical_weight_shape)


def build_convert_policy(*, act_scale: ActScale = ActScale.FIXED_1) -> QuantizationPolicy:
    """Bundle online NVFP4 conversion, shell replacement, and LoRA fusion."""
    return QuantizationPolicy(
        sd_ops=(convert_stage(),),
        module_ops=(convert_swap_op(act_scale=act_scale),),
        fuse_rule=NVFP4_FUSE_RULE,
    )
