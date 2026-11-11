"""Scene-linear HDR frames to BT.2020 / HLG / 10-bit YUV conversion.
HLG = Hybrid Log-Gamma (ITU-R BT.2100 / ARIB STD-B67).
Pipeline: scene-linear RGB -> Rec.2020 -> map diffuse white -> HLG OETF ->
YUV420P10. OETF constants / scalar inverse come from ``colour-science``;
the tensor OETF stays in torch for GPU conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from colour.models import oetf_inverse_BT2100_HLG
from colour.models.rgb.transfer_functions.arib_std_b67 import CONSTANTS_ARIBSTDB67

from tensorrt_llm._torch.visual_gen.models.ltx_core.color.primaries import Primaries
from tensorrt_llm._torch.visual_gen.models.ltx_core.color.yuv import ColorRange, ColorSpace, rgb_to_yuv420p10

# ARIB STD-B67 / BT.2100 HLG OETF constants (colour-science).
_HLG_A = float(CONSTANTS_ARIBSTDB67["a"])
_HLG_B = float(CONSTANTS_ARIBSTDB67["b"])
_HLG_C = float(CONSTANTS_ARIBSTDB67["c"])


def hlg_inverse_oetf(v: float) -> float:
    """Inverse HLG OETF: signal → scene-linear (colour ``oetf_inverse_BT2100_HLG``)."""
    return float(np.asarray(oetf_inverse_BT2100_HLG(v), dtype=np.float64))


def _hlg_oetf(x: torch.Tensor) -> torch.Tensor:
    """HLG OETF on scene-linear ``x`` (torch; matches colour ``oetf_BT2100_HLG``)."""
    return torch.where(
        x <= 1.0 / 12.0,
        torch.sqrt((3.0 * x).clamp(min=0.0)),
        _HLG_A * torch.log((12.0 * x - _HLG_B).clamp(min=1e-12)) + _HLG_C,
    ).clamp(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class HlgGpuConverter:
    """GPU linear-HDR ``[F,H,W,3]`` → planar uint16 YUV420P10 (BT.2020 / HLG / limited)."""

    prim_mat: torch.Tensor
    white_x: float
    roll_k: float

    @classmethod
    def create(
        cls,
        primaries: Primaries = Primaries.REC709,
        white_signal: float = 0.75,
        rolloff_k: float | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> HlgGpuConverter:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        white_x = hlg_inverse_oetf(white_signal)
        roll_k = rolloff_k if rolloff_k is not None else white_x / (1.0 - white_x)
        return cls(
            prim_mat=primaries.matrix_to_rec2020.to(device=device, dtype=dtype),
            white_x=float(white_x),
            roll_k=float(roll_k),
        )

    def to_hlg_signal_(self, rgb_linear_chw: torch.Tensor) -> torch.Tensor:
        """``(*, 3, H, W)`` scene-linear → HLG signal ``[0, 1]``."""
        lin = torch.nan_to_num(
            torch.einsum("...chw,dc->...dhw", rgb_linear_chw, self.prim_mat).clamp(min=0.0),
            nan=0.0,
            neginf=0.0,
        )
        # Diffuse white → white_x; highlights roll toward 1.0.
        x = torch.where(
            lin <= 1.0,
            lin * self.white_x,
            1.0 - (1.0 - self.white_x) * torch.exp(-self.roll_k * (lin - 1.0)),
        )
        return _hlg_oetf(x)

    def __call__(self, frames_fhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[F,H,W,3]`` scene-linear → planar uint16 Y, U, V on the same device."""
        if frames_fhwc.ndim != 4 or frames_fhwc.shape[-1] != 3:
            raise ValueError(f"expected [F,H,W,3], got {tuple(frames_fhwc.shape)}")
        x = frames_fhwc.to(device=self.prim_mat.device, dtype=self.prim_mat.dtype)
        hlg = self.to_hlg_signal_(x.movedim(-1, -3).contiguous())
        return rgb_to_yuv420p10(hlg, ColorSpace.BT_2020_NCL, ColorRange.MPEG)
