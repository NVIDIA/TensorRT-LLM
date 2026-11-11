"""Convert-policy helpers: meta swap + load-time BF16→NVFP4 ``SDOps``.
FV allowlist: 12 Linear suffixes per transformer block. Deliberately excluded:
``to_gate_logits`` (N=32 FP4 GEMM unsupported), the full audio branch, and
``adaln_single.linear``.
"""

from __future__ import annotations

import re

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.kernels import require_nvfp4
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.linear import NVFP4Linear
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.types import ActScale

_LTX2_FV_NVFP4_SUFFIXES: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_out.0",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "ff.net.0.proj",
    "ff.net.2",
)

# ``(?:\._orig_mod)?``: load keys may include compile's remap insert.
_LAYER_PATTERN = re.compile(
    r"^transformer_blocks\.\d+(?:\._orig_mod)?\.(" + "|".join(re.escape(s) for s in _LTX2_FV_NVFP4_SUFFIXES) + r")$"
)


def _is_ltx2_fv_nvfp4_layer(name: str) -> bool:
    """True iff ``name`` is ``transformer_blocks.<int>.<one of FV suffixes>``."""
    return _LAYER_PATTERN.match(name) is not None


def scale_shape(rows: int, width: int) -> tuple[int, int]:
    """Return the cuBLAS-tiled NVFP4 block-scale shape for a 1x16 operand."""
    return ((rows + 127) // 128 * 128, ((width // 16) + 3) // 4 * 4)


def quantize_bf16_weight_to_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 ``[out, in]`` → packed weight, uint8 block scales, decode scale."""
    orig_device = weight.device
    w = weight.detach()
    if w.device.type != "cuda":
        w = w.to(device="cuda", dtype=torch.bfloat16)
    elif w.dtype != torch.bfloat16:
        w = w.to(dtype=torch.bfloat16)

    nvfp4 = require_nvfp4()
    decode = nvfp4.per_tensor_decode_scale(w)
    weight_fp4, weight_scale = nvfp4.quantize_nvfp4(w, decode)
    return (
        weight_fp4.to(device=orig_device),
        weight_scale.to(device=orig_device),
        decode.to(device=orig_device),
    )


def swap_convert_linears(
    model: nn.Module,
    *,
    act_scale: ActScale = ActScale.FIXED_1,
) -> nn.Module:
    """Meta-device swap: replace FV-allowlisted ``nn.Linear`` with ``NVFP4Linear`` shells."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, NVFP4Linear):
            continue
        if not _is_ltx2_fv_nvfp4_layer(name):
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr_name = model, name
        replacements.append((parent, attr_name, module))

    for parent, attr_name, linear in replacements:
        setattr(parent, attr_name, NVFP4Linear.from_linear(linear, act_scale=act_scale))
    return model
