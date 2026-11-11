"""Blockwise FP8/FP6 loading stages and companion-aware LoRA fusion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import KeyValueOperation, SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.units import FuseRule
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.policy import QuantizationPolicy

_BLOCK = 128
_EXCLUDED = (
    "patchify_proj",
    "adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "caption_projection",
    "proj_out",
    "audio_patchify_proj",
    "audio_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    "audio_caption_projection",
    "audio_proj_out",
    "to_gate_logits",
    "scale_shift_table",
)
_FLOAT_INPUTS = frozenset({torch.bfloat16, torch.float16, torch.float32})


class BlockwiseQuantizedWeight(NamedTuple):
    """Packed blockwise weight payload and its per-block scale tensor."""

    weight: torch.Tensor
    scale: torch.Tensor


def _kernels():  # noqa: ANN202
    """Load external CUDA kernels only when concrete quantization or a swap runs."""
    try:
        import blockwise_cpp  # noqa: F401, PLC0415
        import ops_cpp  # noqa: F401, PLC0415
        from ltx_kernels.blockwise.linear import BlockwiseFP6Linear, BlockwiseFP8Linear  # noqa: PLC0415
        from ltx_kernels.blockwise.ops.fp6_pack import fp6_pack_tensor, fp6_unpack_tensor  # noqa: PLC0415
        from ltx_kernels.blockwise.ops.quantize import (  # noqa: PLC0415
            blockwise_dequantize,
            fp6_blockwise_quantize_weights_torch,
            fp8_blockwise_quantize_weights_torch,
        )
    except ImportError as error:
        raise RuntimeError("blockwise FP8/FP6 needs built ltx-kernels CUDA extensions") from error
    return (
        BlockwiseFP6Linear,
        BlockwiseFP8Linear,
        fp6_pack_tensor,
        fp6_unpack_tensor,
        blockwise_dequantize,
        fp6_blockwise_quantize_weights_torch,
        fp8_blockwise_quantize_weights_torch,
    )


def _eligible(key: str, value: torch.Tensor) -> bool:
    return not any(excluded in key for excluded in _EXCLUDED) and value.ndim == 2 and value.dtype in _FLOAT_INPUTS


def _quantize_weight(value: torch.Tensor, *, fp6: bool) -> BlockwiseQuantizedWeight:
    _fp6_linear, _fp8_linear, pack, _unpack, _dequantize, quantize_fp6, quantize_fp8 = _kernels()
    compute = value if value.is_cuda or not torch.cuda.is_available() else value.to("cuda")
    quantized, scales = (quantize_fp6 if fp6 else quantize_fp8)(compute)
    payload = pack(quantized) if fp6 else quantized
    return BlockwiseQuantizedWeight(payload, scales)


def _quantize_concrete(fp6: bool) -> Callable:
    def transform(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value = inputs[key]
        if not _eligible(key, value):
            return {key: value}
        if value.shape[0] % _BLOCK or value.shape[1] % _BLOCK:
            raise ValueError(f"blockwise weight {key!r} dimensions must be multiples of {_BLOCK}")
        result = _quantize_weight(value, fp6=fp6)
        return {key: result.weight, f"{key.removesuffix('.weight')}.weight_scale": result.scale}

    return transform


def _quantize_abstract(fp6: bool) -> Callable:
    def transform(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value = inputs[key]
        if not _eligible(key, value):
            return {key: value}
        out_features, in_features = value.shape
        if out_features % _BLOCK or in_features % _BLOCK:
            raise ValueError(f"blockwise weight {key!r} dimensions must be multiples of {_BLOCK}")
        shape = (out_features, in_features // 4 * 3) if fp6 else value.shape
        dtype = torch.uint8 if fp6 else torch.float8_e4m3fn
        return {
            key: torch.empty(shape, dtype=dtype, device="meta"),
            f"{key.removesuffix('.weight')}.weight_scale": torch.empty(
                (out_features // _BLOCK, in_features // _BLOCK), dtype=torch.float32, device="meta"
            ),
        }

    return transform


def _bias_to_fp32(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    value = inputs[key]
    return {key: value if any(excluded in key for excluded in _EXCLUDED) else value.float()}


def build_sd_ops(*, fp6: bool) -> SDOps:
    """Build the checkpoint transform for blockwise FP8 or packed FP6 weights."""
    stage = SDOps("blockwise-fp6" if fp6 else "blockwise-fp8")
    stage = stage.with_kv_operation(
        KeyValueOperation(_quantize_concrete(fp6), _quantize_abstract(fp6), preserve_output_device=True),
        key_prefix="transformer_blocks.",
        key_suffix=".weight",
    )
    return stage.with_kv_operation(_bias_to_fp32, key_prefix="transformer_blocks.", key_suffix=".bias")


def _replace_linears(model: torch.nn.Module, *, fp6: bool) -> torch.nn.Module:
    fp6_linear, fp8_linear, *_ = _kernels()
    linear_class = fp6_linear if fp6 else fp8_linear
    replacements: list[tuple[torch.nn.Module, str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if "transformer_block" not in name or not isinstance(module, torch.nn.Linear):
            continue
        if any(excluded in name for excluded in ("to_gate_logits", "scale_shift_table")):
            continue
        parent_name, _, attribute = name.rpartition(".")
        replacements.append((model.get_submodule(parent_name), attribute, module))
    for parent, attribute, linear in replacements:
        setattr(parent, attribute, linear_class.from_linear(linear, False))
    return model


def module_op(*, fp6: bool) -> ModuleOps:
    """Replace eligible transformer linears with the matching blockwise layer."""
    return ModuleOps(
        "blockwise-fp6-swap" if fp6 else "blockwise-fp8-swap",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=lambda model: _replace_linears(model, fp6=fp6),
    )


def _dequantize(weight: torch.Tensor, scale: torch.Tensor, *, fp6: bool) -> torch.Tensor:
    _fp6_linear, _fp8_linear, _pack, unpack, dequantize, _quantize_fp6, _quantize_fp8 = _kernels()
    if fp6:
        original_width = weight.shape[-1] * 4 // 3
        weight = unpack(weight, original_width).view(torch.float8_e4m3fn)
    scales = scale.repeat_interleave(_BLOCK, dim=0)
    return dequantize(weight.unsqueeze(0), scales).view(weight.shape)


def _blockwise_patch(fp6: bool) -> Callable:
    def patch(
        key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        quantized = weight.dtype == torch.uint8 if fp6 else weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if not quantized:
            if not weight.is_floating_point():
                raise ValueError(f"{key!r} has an unsupported blockwise payload")
            return {key: (weight.to(delta.dtype) + delta).to(weight.dtype)}
        scale_key = f"{key.removesuffix('.weight')}.weight_scale"
        scale = clean[scale_key].to(weight.device)
        _fp6_linear, _fp8_linear, pack, _unpack, _dequantize_kernel, quantize_fp6, quantize_fp8 = _kernels()
        merged = _dequantize(weight, scale, fp6=fp6) + delta.to(weight.device)
        if fp6:
            new_weight, new_scale = quantize_fp6(merged)
            payload = pack(new_weight.view(torch.uint8))
        else:
            compute = merged.to("cuda") if torch.cuda.is_available() else merged
            new_weight, new_scale = quantize_fp8(compute)
            payload = new_weight
        return {key: payload.to(weight.device), scale_key: new_scale.to(weight.device)}

    return patch


FP8_FUSE_RULE = FuseRule("blockwise-fp8", torch.bfloat16, _blockwise_patch(False), (".weight_scale",))


def _fp6_logical_shape(shape: torch.Size, dtype: torch.dtype) -> torch.Size:
    if dtype != torch.uint8 or len(shape) != 2:
        return shape
    if shape[1] % 3:
        raise ValueError("FP6 packed width must be divisible by 3")
    return torch.Size((shape[0], shape[1] * 4 // 3))


FP6_FUSE_RULE = FuseRule(
    "blockwise-fp6", torch.bfloat16, _blockwise_patch(True), (".weight_scale",), _fp6_logical_shape
)


def build_policy(*, fp6: bool) -> QuantizationPolicy:
    """Bundle blockwise checkpoint transforms, shell replacement, and LoRA fusion."""
    return QuantizationPolicy(
        sd_ops=(build_sd_ops(fp6=fp6),),
        module_ops=(module_op(fp6=fp6),),
        fuse_rule=FP6_FUSE_RULE if fp6 else FP8_FUSE_RULE,
    )
