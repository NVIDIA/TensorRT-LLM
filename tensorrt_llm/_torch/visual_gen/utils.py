"""Utility functions for visual generation pipelines."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping


@torch.compile
def postprocess_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """Post-process video tensor from VAE decoder output to final format.

    This is a more efficient implementation than using VideoProcessor for single-batch cases,
    as it avoids loop overhead and processes the entire batch with vectorized operations.

    Args:
        video: Video tensor in (B, C, T, H, W) format from VAE decoder

    Returns:
        Post-processed video tensor in (B, T, H, W, C) uint8 format.

    Note:
        Assumes video values are in [-1, 1] range (standard VAE decoder output).
    """
    # Convert to (B, T, H, W, C) format
    video = video.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)

    # Normalize to [0, 1] range
    video = (video / 2 + 0.5).clamp(0, 1)

    # Convert to uint8
    video = (video * 255).round().to(torch.uint8)

    return video


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class SequenceSharder:
    """Block-shard / all-gather a tensor along its sequence dimension.

    A single ``SequenceSharder`` collapses the per-model
    ``if attn2d / elif ulysses / elif ring / else`` dispatch and the hand-rolled
    shard / gather of hidden states + RoPE into one model-agnostic helper.

    Built from a :class:`VisualGenMapping` via :meth:`from_vgm`, the sharder
    uses ``vgm.seq_size / seq_rank / seq_group`` so the same call sites work
    uniformly for ring, attn2d, ulysses, and ring + ulysses.

    Models call ``shard(...)`` / ``gather(...)`` / ``shard_rope(...)`` directly;
    when the sharder is inactive (``size == 1`` or runtime-disabled) every
    method is a no-op pass-through so the call sites do not need an
    ``if is_active`` guard.

    The sharder is intentionally model-agnostic: dimensions are passed
    explicitly at every call site and no model-specific shape conventions
    leak in (the sole exception is :meth:`shard_rope`, which infers the
    seq axis from a ``seq_len`` argument).
    """

    def __init__(self, size: int, rank: int, group: Optional[ProcessGroup]):
        self._size = size
        self._rank = rank
        self._group = group
        self._enabled = size > 1

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_vgm(
        cls,
        vgm: Optional[VisualGenMapping],
        *,
        num_attention_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ) -> "SequenceSharder":
        """Build a sharder from a :class:`VisualGenMapping`.

        Uses ``(cp_size * ulysses_size, seq_rank, seq_group)`` so the same
        sharder works for ring, attn2d, ulysses, and ring + ulysses.

        Validates head divisibility only when Ulysses is part of the seq
        group — ring and attn2d shard the sequence axis and have no
        head-count constraint.
        """
        if vgm is None:
            return cls(size=1, rank=0, group=None)

        size = vgm.seq_size
        rank = vgm.seq_rank
        group = vgm.seq_group

        if size > 1 and vgm.ulysses_size > 1:
            for label, count in (
                ("num_attention_heads", num_attention_heads),
                ("num_kv_heads", num_kv_heads),
            ):
                if count is not None and count % vgm.ulysses_size != 0:
                    raise ValueError(
                        f"{label}={count} must be divisible by ulysses_size={vgm.ulysses_size}"
                    )

        return cls(size=size, rank=rank, group=group)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._enabled and self._size > 1

    @property
    def size(self) -> int:
        return self._size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def group(self) -> Optional[ProcessGroup]:
        return self._group

    def disable(self) -> None:
        """Run as if ``size == 1``.

        Used by LTX2's stage-2 single-rank execution path where the
        non-primary workers have already exited.
        """
        self._enabled = False

    def enable(self) -> None:
        """Re-enable sharding after :meth:`disable` (no-op if size == 1)."""
        self._enabled = self._size > 1

    # ------------------------------------------------------------------
    # Shard
    # ------------------------------------------------------------------
    def shard(
        self,
        tensor: Optional[torch.Tensor],
        dim: int = 1,
        *,
        expected_seq_len: Optional[int] = None,
        pad_to_multiple: bool = False,
    ) -> Optional[torch.Tensor]:
        """Contiguous block-shard ``tensor`` along ``dim``.

        Returns ``tensor`` unchanged when:
          * the sharder is inactive (``size == 1`` or runtime-disabled),
          * ``tensor is None``,
          * ``expected_seq_len`` is given and ``tensor.shape[dim]`` doesn't
            match — used by LTX2 to skip dataclass fields whose seq axis
            doesn't line up with the field being sharded.

        When ``pad_to_multiple`` is ``True``, the sequence dim is right-padded
        with zeros to a multiple of ``size`` before sharding.  The matching
        :meth:`gather` call must then pass ``unpad_to`` to slice the padding
        back off.
        """
        if tensor is None or not self.is_active:
            return tensor

        seq_len = tensor.shape[dim]
        if expected_seq_len is not None and seq_len != expected_seq_len:
            return tensor

        if pad_to_multiple and seq_len % self._size != 0:
            pad = self._size - (seq_len % self._size)
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad
            tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)
            seq_len = tensor.shape[dim]

        if seq_len % self._size != 0:
            raise ValueError(
                f"Sequence length ({seq_len}) along dim {dim} is not "
                f"divisible by SequenceSharder.size ({self._size}). "
                f"Pass pad_to_multiple=True or adjust input dimensions."
            )

        chunk = seq_len // self._size
        start = self._rank * chunk
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(start, start + chunk)
        return tensor[tuple(idx)]

    def shard_rope(
        self,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Shard a ``(cos, sin)`` RoPE pair along whichever axis equals ``seq_len``.

        Handles both ``[B, S, D]`` (dim 1) and ``[B, H, S, D]`` (dim 2)
        RoPE layouts that occur across visual_gen models.  Returns the
        pair unchanged when the seq axis cannot be inferred unambiguously
        (e.g. ``seq_len == head_dim``); callers can fall back to explicit
        ``shard(..., dim=...)`` calls in that case.
        """
        if rope is None or not self.is_active:
            return rope

        cos, sin = rope
        candidates = [d for d, n in enumerate(cos.shape) if n == seq_len]
        if len(candidates) != 1:
            return rope
        d = candidates[0]
        return (self.shard(cos, dim=d), self.shard(sin, dim=d))

    # ------------------------------------------------------------------
    # Gather
    # ------------------------------------------------------------------
    def gather(
        self,
        tensor: torch.Tensor,
        dim: int = 1,
        *,
        unpad_to: Optional[int] = None,
    ) -> torch.Tensor:
        """All-gather ``tensor`` along ``dim``.

        No-op when sharder is inactive.  ``unpad_to`` slices the gathered
        tensor's ``dim`` back to the given length; pair with
        ``shard(..., pad_to_multiple=True)`` to round-trip through padding.
        """
        if not self.is_active:
            return tensor

        tensor = tensor.contiguous()
        parts = [torch.empty_like(tensor) for _ in range(self._size)]
        dist.all_gather(parts, tensor, group=self._group)
        out = torch.cat(parts, dim=dim)

        if unpad_to is not None:
            idx = [slice(None)] * out.ndim
            idx[dim] = slice(0, unpad_to)
            out = out[tuple(idx)]
        return out
