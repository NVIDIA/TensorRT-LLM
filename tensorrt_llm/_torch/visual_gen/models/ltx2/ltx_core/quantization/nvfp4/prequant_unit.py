"""Header-selected prequantized NVFP4 loading for weight units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.checkpoint import CheckpointReader
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.linear import NVFP4Linear
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.types import ActScale
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.unit import NVFP4_FUSE_RULE
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.policy import QuantizationPolicy

_AMBIGUOUS = ""


def _layer_index(layers: Mapping[str, int]) -> dict[str, str]:
    """Map every dotted suffix of each discovered layer path to that layer; a shared suffix maps to ``""``.
    Layers are discovered under raw checkpoint names, while model modules and renamed stage keys
    may drop a prefix such as ``model.diffusion_model.``.
    """
    index: dict[str, str] = {}
    for path in layers:
        parts = path.split(".")
        for start in range(len(parts)):
            suffix = ".".join(parts[start:])
            index[suffix] = path if index.get(suffix, path) == path else _AMBIGUOUS
    return index


def _matching_layer(name: str, index: Mapping[str, str]) -> str | None:
    match = index.get(name)
    if match == _AMBIGUOUS:
        raise ValueError(f"ambiguous NVFP4 layer {name!r}")
    return match


def discover_layers(paths: Sequence[Path], *, act_scale: ActScale = ActScale.STATIC) -> dict[str, int]:
    """Find and validate packed NVFP4 weights, scale companions, and activation metadata."""
    with CheckpointReader.open(paths) as reader:
        _, metadata = reader.scan()
    layers: dict[str, int] = {}
    for key, (shape, dtype) in metadata.items():
        if not key.endswith(".weight"):
            continue
        layer = key.removesuffix(".weight")
        scale = metadata.get(f"{layer}.weight_scale")
        decode = metadata.get(f"{layer}.weight_scale_2")
        if (scale is None) != (decode is None):
            raise ValueError(f"{layer!r} needs both .weight_scale and .weight_scale_2")
        if scale is None or decode is None:
            continue
        if dtype != torch.uint8 or scale[1] != torch.float8_e4m3fn or decode != (torch.Size(), torch.float32):
            raise ValueError(f"{layer!r} has an invalid NVFP4 weight/scale dtype or decode shape")
        if len(shape) != 2 or len(scale[0]) != 2 or shape[1] % 16:
            raise ValueError(f"{layer!r} has an invalid NVFP4 packed or block-scale shape")
        rows, packed_width = shape
        full_width = packed_width * 2
        if (
            full_width % 32
            or scale[0][0] < rows
            or scale[0][0] % 128
            or scale[0][1] != ((full_width // 16 + 3) // 4 * 4)
        ):
            raise ValueError(f"{layer!r} has an invalid NVFP4 scale layout")
        if act_scale == ActScale.STATIC:
            if metadata.get(f"{layer}.input_scale") != (torch.Size(), torch.float32):
                raise ValueError(f"act_scale=STATIC requires scalar FP32 .input_scale on {layer!r}")
        elif act_scale not in (ActScale.FIXED_1, ActScale.DYNAMIC):
            raise ValueError(f"unsupported NVFP4 activation scale: {act_scale!r}")
        layers[layer] = scale[0][0]
    if not layers:
        raise ValueError("NVFP4 prequant requires at least one packed weight/scale triple")
    return layers


def _stage_name(layers: dict[str, int], act_scale: ActScale) -> str:
    names = "\0".join(f"{name}:{rows}" for name, rows in sorted(layers.items()))
    digest = sha256(names.encode()).hexdigest()[:12]
    return f"nvfp4-prequant:act={act_scale.value}:layers={digest}"


def prequant_stage(layers: dict[str, int], *, act_scale: ActScale) -> SDOps:
    """Pass packed weights; reinterpret FP8 scale bytes as uint8 for the CUDA kernel."""
    index = _layer_index(layers)

    def transform(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value = inputs[key]
        if key.endswith(".weight") and _matching_layer(key.removesuffix(".weight"), index):
            if value.dtype != torch.uint8:
                raise ValueError(f"expected packed U8 NVFP4 weight for {key!r}")
        elif key.endswith(".weight_scale") and _matching_layer(key.removesuffix(".weight_scale"), index):
            if value.dtype != torch.float8_e4m3fn:
                raise ValueError(f"expected FP8 NVFP4 block scales for {key!r}")
            value = value.view(torch.uint8).contiguous()
        elif key.endswith(".weight_scale_2") and _matching_layer(key.removesuffix(".weight_scale_2"), index):
            value = value.float().reshape(())
        elif key.endswith(".input_scale") and _matching_layer(key.removesuffix(".input_scale"), index):
            if act_scale != ActScale.STATIC:
                return {}
            value = value.float().reshape(())
        return {key: value}

    stage = SDOps(_stage_name(layers, act_scale)).with_matching()
    for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
        stage = stage.with_kv_operation(transform, key_suffix=suffix)
    return stage


def prequant_swap_op(layers: dict[str, int], *, act_scale: ActScale) -> ModuleOps:
    """Replace selected model linears with modules sized for packed checkpoint scales."""
    index = _layer_index(layers)

    def swap(model: nn.Module) -> nn.Module:
        replacements: list[tuple[nn.Module, str, nn.Linear, int]] = []
        for name, layer in model.named_modules():
            if type(layer) is not nn.Linear:
                continue
            selected = _matching_layer(name, index)
            if selected is None:
                continue
            parent_name, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attribute, layer, layers[selected]))
        for parent, attribute, linear, rows in replacements:
            setattr(
                parent,
                attribute,
                NVFP4Linear(
                    linear.in_features,
                    linear.out_features,
                    bias=linear.bias is not None,
                    act_scale=act_scale,
                    weight_scale_rows=rows,
                    device=linear.weight.device,
                ),
            )
        return model

    return ModuleOps(
        f"{_stage_name(layers, act_scale)}:swap",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=swap,
    )


def build_policy(paths: Sequence[Path], *, act_scale: ActScale = ActScale.STATIC) -> QuantizationPolicy:
    """Build the header-selected prequantized NVFP4 load and LoRA-fusion policy."""
    layers = discover_layers(paths, act_scale=act_scale)
    return QuantizationPolicy(
        sd_ops=(prequant_stage(layers, act_scale=act_scale),),
        module_ops=(prequant_swap_op(layers, act_scale=act_scale),),
        fuse_rule=NVFP4_FUSE_RULE,
    )
