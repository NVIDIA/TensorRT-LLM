"""Deferred stage-4 context inject: sequential upsample then context_proj (W-chunked).
Eager in-place Python W-loop — stays outside ChunkedCompile's ``forward_attn_mlp``
graph so Inductor never sees inject. Runs ``upsamples[3].proj`` (pixel-shuffle)
then ``context_proj``; no fused Linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _upsample_then_ctx(
    feat: torch.Tensor,
    up_w: torch.Tensor,
    up_b: torch.Tensor,
    ctx_w: torch.Tensor,
    ctx_b: torch.Tensor | None,
    stride: tuple[int, int, int],
    *,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """``context_proj(pixel_shuffle(upsample_proj(feat)))``."""
    p1, p2, p3 = stride
    up = F.linear(feat, up_w, up_b)
    up = rearrange(
        up,
        "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
        p1=p1,
        p2=p2,
        p3=p3,
    )
    if p1 == 2 and drop_leading_frame:
        up = up[:, 1:, :, :, :]
    return F.linear(up, ctx_w, ctx_b)


def deferred(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample_proj: nn.Linear,
    context_proj: nn.Linear,
    stride: tuple[int, int, int],
    *,
    w_chunks: int,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """Chunk stage-4 feat along W, upsample+``context_proj`` each slab, ``add_`` into ``x``.
    Returns ``x`` (mutated).
    """
    assert upsample_proj.bias is not None
    p3 = stride[2]
    up_w, up_b = upsample_proj.weight, upsample_proj.bias
    ctx_w, ctx_b = context_proj.weight, context_proj.bias

    if w_chunks <= 1:
        x.add_(
            _upsample_then_ctx(
                stage4_feat,
                up_w,
                up_b,
                ctx_w,
                ctx_b,
                stride,
                drop_leading_frame=drop_leading_frame,
            )
        )
        return x

    feat_chunks = torch.chunk(stage4_feat, w_chunks, dim=3)
    w_hi = x.shape[3]
    lo = 0
    for feat_chunk in feat_chunks:
        hi = min(w_hi, lo + feat_chunk.shape[3] * p3)
        ctx = _upsample_then_ctx(
            feat_chunk,
            up_w,
            up_b,
            ctx_w,
            ctx_b,
            stride,
            drop_leading_frame=drop_leading_frame,
        )
        x[:, :, :, lo:hi, :].add_(ctx[:, :, :, : hi - lo, :])
        lo = hi
    return x


def deferred_keyframes(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample_proj: nn.Linear,
    context_proj: nn.Linear,
    stride: tuple[int, int, int],
    *,
    w_chunks: int,
) -> torch.Tensor:
    """Keyframe counterpart of :func:`deferred`. ``x``/``stage4_feat`` are ``(B, P, H, W, C)``.
    Each plane is its own ``T=1`` clip, so folding the plane axis into the batch turns this
    into exactly the video case and the whole W-chunk / halo policy is inherited rather than
    re-derived -- which is what keeps the two streams' W phases comparable downstream.
    ``drop_leading_frame`` is pinned ``True``: a temporal stride of 2 expands ``T=1`` to 2 and
    the drop takes it back to 1, keeping phase 1. Passing ``False`` (as tiled video does for
    non-origin tiles) would invent a second temporal plane per keyframe.
    The reshape is a view of a contiguous tensor, so :func:`deferred`'s in-place ``add_``
    writes through to ``x``. Returns ``x`` (mutated).
    """
    if x.shape[:2] != stage4_feat.shape[:2]:
        raise ValueError(
            f"keyframe x and stage-4 feat must share (B, P), got {tuple(x.shape[:2])} vs {tuple(stage4_feat.shape[:2])}"
        )
    if not x.is_contiguous():
        # A non-contiguous x makes the fold a copy instead of a view, and the inject would be
        # silently dropped rather than failing. Refuse instead of quietly decoding without it.
        raise ValueError("keyframe x must be contiguous channels-last so the plane fold stays a view")
    folded = x.shape[0] * x.shape[1]
    deferred(
        x.reshape(folded, 1, *x.shape[2:]),
        stage4_feat.reshape(folded, 1, *stage4_feat.shape[2:]),
        upsample_proj,
        context_proj,
        stride,
        w_chunks=w_chunks,
        drop_leading_frame=True,
    )
    return x
