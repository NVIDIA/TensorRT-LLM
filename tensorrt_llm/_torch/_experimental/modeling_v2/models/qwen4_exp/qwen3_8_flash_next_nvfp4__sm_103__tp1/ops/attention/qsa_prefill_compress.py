# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA block compression of the paged index-key side cache: completed groups to anchors."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

#: The mRoPE sections the checkpoint declares; the target passes `None` today
#: (`_QSA_MROPE_SECTION`) and these are certified, on the NVFP4 trunk-plane cell,
#: for when it flips.
MROPE_SECTION = (11, 11, 10)

#: This checkpoint's index geometry and the engine's QSA metadata: index head
#: 128, rotary width 64 (32 cos/sin pairs), compress ratio 4, rms eps 1e-6,
#: theta 1e7; `slots` x `width` is `qsa_block_table`. `pages` is the pool.
_CKPT = dict(
    head_dim=128,
    tokens_per_block=32,
    slots=32,
    width=260,
    compress_ratio=4,
    eps=1e-6,
    mrope_section=None,
    logical_dtype=torch.int64,
    request_dtype=torch.int32,
    layout="plain",
)

#: 32 requests of one MTP7 verify block each (8 rows, two completed groups), at starts that put
#: some blocks across a page edge and some inside one.
_MTP7_32 = tuple((29 + 61 * i, 8) for i in range(32))


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


def _mrope(segments, pages, **overrides) -> dict:
    """Prefill from 0 under mRoPE (11, 11, 10)."""
    return _spec(segments, pages, mrope_section=MROPE_SECTION, **overrides)


class QsaPrefillCompress(OpWrapper):
    """Replace each completed group's anchor slot with its pooled, normed, rotated key.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_prefill_compress`: one Triton launch, grid `(rows,)`. A row
    with `(L + 1) % cr == 0` (L = its logical position) is the anchor of the
    group `L - cr + 1 .. L`; every other row does nothing. Each member is
    resolved through `block_table` on its own, so groups may straddle pages::

        pooled = bf16(sum_f32(raw keys of the group) / cr)
        y      = f32(pooled) * rsqrt(mean(f32(pooled)^2) + eps) * (1 + k_norm_weight)
        y      = cos_sin.dtype(y)
        (t, h, w) = position_cache at the group's FIRST member
        pair q takes h if q % 3 == 1 and q < 3 * s1, w if q % 3 == 2 and
          q < 3 * s2, else t   (t for every pair when mrope_section is None)
        NeoX-rotate y[:D/2] by cos_sin[pos[q], q], cos_sin[pos[q], R + q];
          y[D/2:] passes through (R = cos_sin.shape[1] // 2 = D / 4)
        index_cache[anchor] = bf16(y)

    Returns `index_cache`, written in place at anchor slots only; everything
    else, including the non-anchor members' raw keys, is untouched.

    Not idempotent: the anchor slot is the group's last input and its only
    output, so a second call over the same rows re-pools a compressed anchor
    (measured: all 48 of 48 anchors moved, up to 2.94).

    Fusion boundary: pool, Gemma norm, position gather, rotation, store. The
    raw-key and position stores (`qsa_paged_index_store`, which must run
    first), the table and metadata, and the main K/V pool are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="prefill 2 x 128 from 0, mrope_section None: the target's call",
            spec=_spec(((0, 128),) * 2, pages=8),
        ),
        Cell(
            why="decode MTP7: 32 requests x 8 rows, two anchors each, None",
            spec=_spec(_MTP7_32, pages=1100, slots=40, rounding_flips=1),
        ),
        # Under an NVFP4 KV pool the position page shares each physical page with the index
        # pages, so both caches are strided views of one allocation: index page stride
        # 12 x 4096 + 192 = 49344 on the trunk, position strides (24672, 3, 1).
        Cell(
            why="NVFP4 trunk planes: 12 index roles and the position page share each page",
            spec=_mrope(((0, 64),) * 2, pages=4, layout="shared_page", roles=12),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_logical_positions` (int64) and `qsa_req_idx_per_token` (int32)
    sliced to the step's rows, `qsa_block_table` unsliced, the layer's
    `get_index_k_buffer(i)` and the shared `get_qsa_position_buffer()`, the
    indexer's `k_layernorm` weight (bf16 [128]) and eps, a float32
    `[max_positions, 64]` cos|sin table built at load, and `mrope_section`
    None. It runs on every step, decode and prefill rows alike.

    Preconditions:
      - every member's raw key and the first member's coordinates are in the
        caches before the call; this call stores neither.
      - every logical position is non-negative. `L = -1` satisfies the anchor
        test and addresses positions -4..-1: measured, an illegal memory
        access that poisons the process. A value, so not guarded; pad with 0
        (measured an exact no-op).
      - `logical_positions`, `request_indices`, `k_norm_weight` and the last
        dim of `cos_sin` are read with an assumed unit stride while the
        callable checks only shape: each was measured accepted and wrong
        (max abs 4.72, 6.20, 0.80, 7.16). `is_valid` refuses them.
    The callable rejects loudly (not pinned by a test): metadata shape/dtype mismatches, a non-int32
    or non-2-D table, more than one index head, a position plane with fewer
    than 3 axes or not int32, a non-bf16 cache, page sizes that disagree with
    `tokens_per_block`, a malformed cos|sin table, `4 R != D`, a non-2^k `D`
    or `compress_ratio`, a norm weight of the wrong length, CPU operands, and
    `mrope_section` that is not 3 entries summing to `R`. `rows == 0` returns
    before launching.

    `cos_sin`'s dtype is the rotation's precision: bf16 is accepted and was
    measured 1.4% of elements off by up to 1.56e-2 -- not certified. Text
    rows reduce mRoPE to plain RoPE bit for bit (measured, not pinned by a
    test). Plain `@triton.jit`, no
    autotune; CUDA-graph capturable at a fixed row count (captured at 256
    rows, 32 MTP7 verify blocks).

    Under an NVFP4 KV pool both caches are strided views of one per-slot
    allocation (each physical page holds the index roles, then the position
    page); the trunk-plane cell checks the other index roles and the position
    page untouched.

    The kernel's `tl.rsqrt` is the hardware approximation, an fp32 ulp from the
    reference's: every cell is bitwise the reference but the 32-block MTP7 one, where
    1 of 8192 anchor elements rounds one bf16 step away.

    Hard limits (read from the kernel, not probed): positions, requests and
    page ids are widened to int64 before any address arithmetic, so no int32
    boundary inside the call. Certified extent: positions to 1928 through a
    [40, 260] table over 1100 pages.

    Not certified: `compress_ratio` other than 4, `D` other than 128, a bf16
    or fp16 `cos_sin`, positions past 1928 or the table's rows, mRoPE outside
    the NVFP4 trunk-plane cell, the NVFP4 draft planes (one index role a page), multiple devices,
    `tokens_per_block` other than 32 (no group straddles a page), eps other
    than 1e-6, int32 positions or int64 request indices, a strided view of the
    index cache, position plane or table other than the NVFP4 pool's shared
    page, two rows naming one anchor in a launch, and `rows == 0`.
    """

    def __call__(
        self,
        *,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        tokens_per_block: int,
        compress_ratio: int,
        mrope_section: Optional[Tuple[int, int, int]],
    ) -> torch.Tensor:
        _qsa_kernels.triton_qsa_prefill_compress(
            logical_positions=logical_positions,
            request_indices=request_indices,
            block_table=block_table,
            index_cache=index_cache,
            position_cache=position_cache,
            k_norm_weight=k_norm_weight,
            cos_sin=cos_sin,
            eps=eps,
            tokens_per_block=tokens_per_block,
            compress_ratio=compress_ratio,
            mrope_section=mrope_section,
        )
        return index_cache

    def reference(
        self,
        *,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        tokens_per_block: int,
        compress_ratio: int,
        mrope_section: Optional[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """The index cache after the call, as a fresh tensor, from the pre-call state."""
        out = index_cache.clone()
        logical = logical_positions.long()
        anchored = (logical + 1) % compress_ratio == 0
        anchor = logical[anchored]
        if anchor.numel() == 0:
            return out
        device = index_cache.device
        request = request_indices.long()[anchored]
        members = (
            anchor[:, None] - (compress_ratio - 1) + torch.arange(compress_ratio, device=device)
        )
        pages = block_table.long()[request[:, None], members // tokens_per_block]
        slots = members % tokens_per_block

        head_dim = index_cache.shape[3]
        raw = index_cache[pages, slots, 0].float()  # [anchors, cr, D]
        pooled = (raw.sum(dim=1) / compress_ratio).to(torch.bfloat16).float()
        scale = torch.rsqrt(pooled.mul(pooled).sum(dim=-1, keepdim=True) / head_dim + eps)
        normed = (pooled * scale * (k_norm_weight.float() + 1.0)).to(cos_sin.dtype)

        t, h, w = position_cache[pages[:, 0], slots[:, 0], :3].long().unbind(-1)
        pairs_n = cos_sin.shape[1] // 2
        pairs = torch.arange(pairs_n, device=device)
        if mrope_section is None:
            pos = t[:, None].expand(-1, pairs_n)
        else:
            pos = torch.where(
                (pairs % 3 == 1) & (pairs < 3 * mrope_section[1]),
                h[:, None],
                torch.where(
                    (pairs % 3 == 2) & (pairs < 3 * mrope_section[2]), w[:, None], t[:, None]
                ),
            )
        cos, sin = cos_sin[pos, pairs], cos_sin[pos, pairs_n + pairs]
        first, second = normed[:, :pairs_n], normed[:, pairs_n : 2 * pairs_n]
        rotated = torch.cat(
            [first * cos - second * sin, second * cos + first * sin, normed[:, 2 * pairs_n :]],
            dim=-1,
        )
        out[pages[:, -1], slots[:, -1], 0] = rotated.to(torch.bfloat16)
        return out

    def is_valid(
        self,
        *,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        tokens_per_block: int,
        compress_ratio: int,
        mrope_section: Optional[Tuple[int, int, int]],
    ) -> None:
        # Read by the kernel with an assumed unit stride; the callable checks
        # only their shapes. Each measured accepted and silently wrong.
        assert logical_positions.is_contiguous(), "logical_positions must be contiguous"
        assert request_indices.is_contiguous(), "request_indices must be contiguous"
        assert k_norm_weight.is_contiguous(), "k_norm_weight must be contiguous"
        # `ndim != 2` is the callable's own loud error; leave it to the callable.
        assert cos_sin.ndim != 2 or cos_sin.stride(1) == 1, (
            "cos_sin must have unit stride along its last dimension"
        )


qsa_prefill_compress = QsaPrefillCompress()
