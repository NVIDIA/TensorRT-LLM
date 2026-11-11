"""RMSNorm of the DiffVAE deterministic blocks: the fused row kernel or the uint32-safe norm, chosen per call."""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae import decode_workspace
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.ops import apply_rms_norm_uint32_safe
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.row_rms_norm import row_rms_norm_, row_rms_norm_supported

# Rows of a Q/K volume (or of a block input) normalised per chunk when the row kernel does not apply.
_NORM_CHUNK_ROWS: int = 1 << 22


class DetRMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` (same parameters and state dict) for the det blocks' block and Q/K norms.
    Traced, it is the uint32-safe functional norm (:func:`~ltx_core.model.video_vae.ops.rms_norm_uint32_safe`),
    so compiled graphs keep the functional op. Eager, it goes through the fused row kernel
    (:mod:`~ltx_core.model.video_vae.transformer.row_rms_norm`; ATen's arithmetic, the fp32 sum of squares may
    round differently, 1 bf16 ulp on ~2e-6 of the elements) where that applies, else the uint32-safe norm
    row chunk by row chunk; :meth:`forward` writes into a decode-workspace phase buffer when one is active,
    :meth:`forward_` back into its input.
    """

    def _fused(self, rows: torch.Tensor) -> bool:
        """Whether the fused row kernel serves this norm on 2-D ``rows``."""
        if self.weight is None or self.eps is None or tuple(self.normalized_shape) != (rows.shape[-1],):
            return False
        return bool(row_rms_norm_supported(rows, self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``norm(x)`` over the last dim, into a workspace phase buffer when one is active, else a fresh tensor."""
        if torch.compiler.is_compiling() or not x.is_contiguous():
            return apply_rms_norm_uint32_safe(self, x)
        rows_in = x.view(-1, x.shape[-1])
        fused = self._fused(rows_in)
        if not decode_workspace.workspace_active():
            if not fused:
                return apply_rms_norm_uint32_safe(self, x)
            out = torch.empty_like(x)
            row_rms_norm_(rows_in, self.weight, float(self.eps), out=out.view(-1, x.shape[-1]))
            return out
        out = decode_workspace.phase_empty(tuple(x.shape), dtype=x.dtype, device=x.device)
        rows_out = out.view(-1, x.shape[-1])
        if fused:
            row_rms_norm_(rows_in, self.weight, float(self.eps), out=rows_out)
            return out
        for r0 in range(0, rows_in.shape[0], _NORM_CHUNK_ROWS):
            rows_out[r0 : r0 + _NORM_CHUNK_ROWS].copy_(
                apply_rms_norm_uint32_safe(self, rows_in[r0 : r0 + _NORM_CHUNK_ROWS])
            )
        return out

    def forward_(self, x: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
        """``norm(x)`` (then ``* scale``, a second bf16 rounding as ``x.mul_(scale)`` after the norm is) written
        back into a contiguous ``x``, RMSNorm being per row; a new tensor for a non-contiguous ``x``. Eager only."""
        if not x.is_contiguous():
            y = apply_rms_norm_uint32_safe(self, x)
            return y if scale is None else y.mul_(scale)
        rows = x.view(-1, x.shape[-1])
        if self._fused(rows):
            row_rms_norm_(rows, self.weight, float(self.eps), scale=scale)
            return x
        for r0 in range(0, rows.shape[0], _NORM_CHUNK_ROWS):
            chunk = rows[r0 : r0 + _NORM_CHUNK_ROWS]
            chunk.copy_(apply_rms_norm_uint32_safe(self, chunk))
        return x if scale is None else x.mul_(scale)
