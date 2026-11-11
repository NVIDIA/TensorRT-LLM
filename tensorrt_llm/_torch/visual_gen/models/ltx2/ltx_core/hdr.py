"""ACEScct working-space compress/decompress and sRGB EOTF.
ACEScct maps linear ACEScg HDR [0, ∞) <-> a compressed [0, 1] signal.
"""

from __future__ import annotations

import enum

import torch
from torch import Tensor

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.color.primaries import Primaries

__all__ = [
    "TransferEncoding",
    "srgb_eotf_to_linear",
    "to_acescct_working_space",
    "to_hdr_linear",
]


class TransferEncoding(enum.Enum):
    """Transfer encoding of an HDR tensor (scene-linear vs already-compressed log)."""

    LINEAR = "linear"
    LOG = "log"


_ACESCCT_A_LIN = 10.5402377416545
_ACESCCT_B_LIN = 0.0729055341958355
_ACESCCT_X_BRK = 0.0078125
_ACESCCT_Y_BRK = 0.155251141552511
_ACESCCT_LOG_M = 17.52
_ACESCCT_LOG_B = 9.72

# IEC 61966-2-1 sRGB EOTF
_SRGB_A = 0.055
_SRGB_LINEAR_THRESHOLD = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_GAMMA = 2.4


def _compress(hdr: Tensor) -> Tensor:
    """Compress linear ACEScg HDR [0, ∞) → ACEScct [0, 1]."""
    x = torch.clamp(hdr, min=0.0)
    log_part = (torch.log2(torch.clamp(x, min=1e-12)) + _ACESCCT_LOG_B) / _ACESCCT_LOG_M
    lin_part = _ACESCCT_A_LIN * x + _ACESCCT_B_LIN
    return torch.clamp(torch.where(x > _ACESCCT_X_BRK, log_part, lin_part), 0.0, 1.0)


def _decompress(ct: Tensor) -> Tensor:
    """Decompress ACEScct [0, 1] → linear ACEScg HDR [0, ∞)."""
    ct = torch.clamp(ct, 0.0, 1.0)
    lin_from_log = torch.pow(2.0, ct * _ACESCCT_LOG_M - _ACESCCT_LOG_B)
    lin_from_lin = (ct - _ACESCCT_B_LIN) / _ACESCCT_A_LIN
    return torch.where(ct > _ACESCCT_Y_BRK, lin_from_log, lin_from_lin)


def to_acescct_working_space(
    video: Tensor,
    source_primaries: Primaries = Primaries.REC709,
) -> Tensor:
    """Compress scene-linear HDR into the ACEScct [0, 1] working space."""
    linear_acescg = source_primaries.to(Primaries.AP1, video.float())
    return _compress(linear_acescg.clamp(min=0.0))


def to_hdr_linear(
    working: Tensor,
    out_primaries: Primaries = Primaries.REC709,
) -> Tensor:
    """Decompress ACEScct [0, 1] to scene-linear HDR in ``out_primaries``."""
    linear_acescg = _decompress(working.float()).clamp(min=0.0)
    out = Primaries.AP1.to(out_primaries, linear_acescg)
    if out_primaries is Primaries.AP0:
        return out
    return out.clamp(min=0.0)


def srgb_eotf_to_linear(srgb: Tensor) -> Tensor:
    """sRGB-encoded [0, 1] -> display-linear Rec.709/sRGB (IEC 61966-2-1 EOTF)."""
    x = torch.clamp(srgb.float(), 0.0, 1.0)
    return torch.where(
        x <= _SRGB_LINEAR_THRESHOLD,
        x / _SRGB_LINEAR_SLOPE,
        torch.pow((x + _SRGB_A) / (1.0 + _SRGB_A), _SRGB_GAMMA),
    )
