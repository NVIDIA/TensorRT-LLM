"""NVFP4 Linear via ``ltx_kernels.nvfp4`` (cast + prequant policies).
The implementation modules stay importable without the CUDA extension. Unit
policies can inspect checkpoint metadata on other devices; public policy
builders check for the extension when used.
"""

from __future__ import annotations

from pathlib import Path

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.kernels import require_nvfp4
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.types import ActScale
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.policy import QuantizationPolicy

__all__ = [
    "ActScale",
    "build_nvfp4_cast_policy",
    "build_nvfp4_prequant_policy",
]


def build_nvfp4_cast_policy(*, act_scale: ActScale = ActScale.FIXED_1) -> QuantizationPolicy:
    """Build online BF16-to-NVFP4 conversion and fusion operations."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.unit import build_convert_policy  # noqa: PLC0415

    require_nvfp4()
    return build_convert_policy(act_scale=act_scale)


def build_nvfp4_prequant_policy(
    checkpoint_path: str,
    *,
    act_scale: ActScale = ActScale.STATIC,
) -> QuantizationPolicy:
    """Build header-selected prequantized NVFP4 loading and fusion operations."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.quantization.nvfp4.prequant_unit import build_policy  # noqa: PLC0415

    require_nvfp4()
    return build_policy((Path(checkpoint_path),), act_scale=act_scale)
