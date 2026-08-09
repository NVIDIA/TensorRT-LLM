# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry point for the MiniMax-M3 MoE gate projection.

The gate multiplies BF16 hidden states by an FP32 router weight. cuBLAS cannot
take those two dtypes together, so the model casts the activation to FP32
first -- at 16k tokens that materialises a 400MB temporary and then feeds a
CUDA-core SGEMM, which between them run at about 4% of what the memory system
can deliver.

This path hands the BF16 activation straight to the tensor cores and puts the
precision back on the weight side, where it is cheap: the FP32 weight is
rewritten as a sum of ``terms`` BF16 tensors, each one holding what the previous
rounding threw away. The GEMM accumulates all of them into a single FP32
accumulator, so the activation is never widened and never copied.

Each extra term costs one more tensor-core pass over a 3MB weight and buys about
eight mantissa bits:

===== ================== ==============================================
terms weight mantissa    max rel error at 32 tokens
===== ================== ==============================================
1     8 bits             6.1e-3; worse than the TF32 path, so not an option
2     ~16 bits           1.1e-5; 67x better than TF32, 12x short of FP32
3     ~24 bits           7.3e-6; only 1.5x better than two, for 0.5us more
===== ================== ==============================================

Two is the pick. That the third term barely moves the error is the useful
result: past ~16 weight bits the accumulation order dominates, so more terms
buy tensor-core passes and nothing else. The reference is FP64 over the same
BF16 activation, so the activation's own width is not an error term here --
it is the input, which is why weight bits keep paying up to that point.

Two BF16 passes cost about 24us of tensor-core time at 16k tokens against a
33us floor for reading the activation once, so the accurate configuration is
still bandwidth-bound and effectively free next to the one-term version.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from .minimax_m3_gate_gemm import MiniMaxM3GateGemmKernel
from .utils import make_ptr

#: ``(use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)``.
Tactic = Tuple[bool, Tuple[int, int], Tuple[int, int]]

DEFAULT_TERMS = 2

_TORCH_TO_CUTLASS = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
}


def split_weight(weight: torch.Tensor, terms: int = DEFAULT_TERMS) -> torch.Tensor:
    """Rewrite an FP32 ``[N, K]`` weight as ``terms`` stacked BF16 terms.

    Term ``i`` is the BF16 rounding of everything the first ``i`` terms failed
    to represent, so the terms sum back to the original weight to within the
    last one's rounding. Stacking them along N lets a single GEMM sweep all of
    them against one activation tile.

    The router weight is a frozen parameter, so in production this runs once at
    load rather than per call.
    """
    if weight.dtype != torch.float32:
        raise ValueError(f"router weight must be fp32, got {weight.dtype}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")

    residual = weight
    pieces = []
    for _ in range(terms):
        piece = residual.to(torch.bfloat16)
        pieces.append(piece)
        residual = residual - piece.to(torch.float32)
    return torch.cat(pieces, dim=0).contiguous()


#: Elements of K consumed by one MMA tile, by operand width. The tcgen05 MMA
#: instruction covers 32 bytes of K per row and the mainloop runs four of them
#: per tile, so a BF16 operand advances 64 elements at a time and a TF32 one 32.
_K_TILE_ELEMS = {torch.bfloat16: 64, torch.float32: 32}


def split_k_is_supported(k: int, split_k: int, ab_dtype: torch.dtype) -> bool:
    """Whether K divides evenly into ``split_k`` runs of whole MMA tiles.

    The kernel gives every partition the same number of K tiles and does no
    predication at the seams, so an uneven split would quietly drop the
    remainder rather than run slowly.
    """
    if split_k < 1:
        return False
    if split_k == 1:
        return True
    tile = _K_TILE_ELEMS.get(ab_dtype)
    if tile is None or k % tile:
        return False
    return (k // tile) % split_k == 0


def default_split_k(num_tokens: int) -> int:
    """How many CTAs to put on the K dimension, from the microbenchmark sweep.

    Below a thousand tokens the output is one or two tiles wide, so the whole
    grid is one or two CTAs and the mainloop walks all 96 K tiles serially in
    each of them. That, not bandwidth, is the small-M floor: a narrower N tile
    spreads the same weight over more CTAs and changes nothing, and a deeper K
    tile shortens the loop and changes nothing, but partitioning K takes the
    kernel from 8.9us to 2.8us at 32 tokens.

    Four is well short of what the kernel alone would like -- it keeps falling
    to 32 partitions -- because the partials go back through memory and Torch
    cannot reduce them for less than about 1.8us however the sum is phrased.
    Past four the reduction grows faster than the mainloop shrinks. Reducing
    inside the kernel does not lift this: both a last-CTA fixup and a barrier
    between the partitions were measured and neither beat the Torch pass, for
    the reasons in the kernel's module docstring.
    """
    if num_tokens >= 4096:
        return 1
    if num_tokens >= 2048:
        return 2
    return 4


def fold_is_supported(tactic: Tactic, stacked_n: int) -> bool:
    """Whether the epilogue can sum the weight terms for this tile shape.

    The epilogue folds by pairing accumulator subtiles a fixed distance apart,
    which holds only if one MMA tile covers every term -- otherwise a tile
    holds a single term and there is nothing in it to fold against. On top of
    that, a 2-CTA instruction whose M tile is 128 leaves each CTA with 64
    accumulator rows, and the subtile ordering that pairing relies on no longer
    holds; the same 64-row tile is fine as a 1-CTA instruction. Verified
    against an FP64 reference by ``test_minimax_m3_gate_gemm.py``.
    """
    use_2cta, (tiler_m, _), _ = tactic
    if tactic[1][1] != stacked_n:
        return False
    return not (use_2cta and tiler_m < 256)


def default_tactic(num_tokens: int) -> Tactic:
    """Tile and cluster shape, from the sweep in the microbenchmark's ``--tune``.

    N is at most 256 even with both weight terms, so the grid is essentially
    ``M / tile_m`` CTAs and the tile shape is really a choice about how few
    CTAs the work is allowed to land on.

    A tile spanning all of N lets the epilogue fold the terms in registers,
    which saves a second pass over the output, but it also forces the whole 3MB
    weight through however many CTAs the token count affords. Below 8192 tokens
    that is too few and the narrow tile wins outright despite the extra pass:
    at 32 tokens the wide tile takes 28.5us against 12.1us, because one CTA
    cannot stream 3MB any faster. Above it the ordering flips and folding is
    worth 5us at 16384. So the threshold is not about occupancy in the usual
    sense; it is the point where there are finally enough rows to spread the
    weight load across.

    Clustering along M multicasts the weight to the CTAs sharing it and helps
    everywhere. Below ~1024 tokens none of these shapes is the right tool: the
    kernel sits at a flat ~12us floor that has nothing to do with the work it
    is doing -- see the microbenchmark's notes on split-K.
    """
    if num_tokens >= 16384:
        return (True, (256, 256), (2, 1))  # folds in the epilogue
    if num_tokens >= 8192:
        return (True, (128, 256), (2, 1))  # also folds
    return (True, (128, 128), (2, 1))


class GateGemmRunner:
    """Compiles and caches the gate GEMM, one compilation per tactic.

    ``wrapper`` takes M, N and K as runtime values, so a single compiled kernel
    serves the whole token sweep and only the tile shape forces a recompile.
    """

    _cache: dict = {}

    @classmethod
    def _compiled(
        cls,
        tactic: Tactic,
        c_dtype: torch.dtype,
        ab_dtype: torch.dtype,
        fold: int,
        split_k: int,
    ):
        key = (tactic, c_dtype, ab_dtype, fold, split_k)
        hit = cls._cache.get(key)
        if hit is not None:
            return hit

        use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
        kernel = MiniMaxM3GateGemmKernel(
            cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            fold_terms=fold,
            split_k=split_k,
        )
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        # Shapes are dynamic; these placeholders only fix dtypes and rank.
        ab_cutlass = _TORCH_TO_CUTLASS[ab_dtype]
        a_stub = torch.empty((1, 128, 128), dtype=ab_dtype, device="cuda")
        c_stub = torch.empty((128, 128, 1), dtype=c_dtype, device="cuda")
        compiled = cute.compile(
            kernel.wrapper,
            128,
            128,
            128,
            1,
            make_ptr(ab_cutlass, a_stub.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_cutlass, a_stub.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            cute.runtime.from_dlpack(c_stub).mark_layout_dynamic(leading_dim=1),
            max_active_clusters=max_active_clusters,
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--opt-level 2",
        )
        cls._cache[key] = compiled
        return compiled

    @classmethod
    def run(
        cls,
        x: torch.Tensor,
        w_split: torch.Tensor,
        out: torch.Tensor,
        tactic: Optional[Tactic] = None,
        fold: int = 1,
        split_k: int = 1,
    ) -> None:
        """``out = x @ w_split.T``, with ``out`` holding one column block per term.

        ``out`` is ``[num_tokens, n]`` when ``split_k`` is 1 and ``[split_k,
        num_tokens, n]`` otherwise, one slice per K partition for the caller to
        sum. Either way it is handed to the kernel as an ``(m, n, l)`` tensor,
        which is what makes split-K cheap to express: the partition index rides
        in as the batch coordinate, the TMA store lands each partial in its own
        slice, and the epilogue needs no notion of splitting at all.

        Passing FP32 operands instead of BF16 selects the TF32 tensor cores, via
        the TMA descriptor's ``internal_type``. That is the comparison point for
        the multi-term BF16 path rather than a usable configuration, since an
        FP32 activation is exactly the copy this kernel exists to avoid.
        """
        m, k = x.shape
        n = w_split.shape[0]
        tactic = tactic or default_tactic(m)
        if not split_k_is_supported(k, split_k, x.dtype):
            raise ValueError(f"K={k} does not divide into {split_k} whole-tile partitions")
        # (m, n, l), n contiguous. The partials arrive as [l, m, n] so that the
        # reduction over l reads them contiguously.
        c_mnl = out.unsqueeze(-1) if split_k == 1 else out.permute(1, 2, 0)

        ab_cutlass = _TORCH_TO_CUTLASS[x.dtype]
        compiled = cls._compiled(tactic, out.dtype, x.dtype, fold, split_k)
        compiled(
            m,
            n,
            k,
            1,
            make_ptr(ab_cutlass, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_cutlass, w_split.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            cute.runtime.from_dlpack(c_mnl).mark_layout_dynamic(leading_dim=1),
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def gate_gemm(
    hidden_states: torch.Tensor,
    w_split: torch.Tensor,
    terms: int = DEFAULT_TERMS,
    tactic: Optional[Tactic] = None,
    fused: bool = True,
    split_k: Optional[int] = None,
) -> torch.Tensor:
    """Router logits from BF16 hidden states and a pre-split BF16 weight.

    ``w_split`` comes from :func:`split_weight`. Returns ``[num_tokens,
    num_experts]`` FP32.

    Two ways of getting the terms back together, picked by shape. When the tile
    spans all of N the epilogue folds them in registers and only the leading
    ``num_experts`` columns are ever stored -- the buffer is still allocated
    full width because the TMA store descriptor is derived from it, but the
    rest is untouched address space rather than traffic. Otherwise the terms
    are stored side by side and summed in a second pass, which is also where
    the K partitions are summed when K is split.
    """
    num_tokens = hidden_states.shape[0]
    stacked_n = w_split.shape[0]
    num_experts = stacked_n // terms
    tactic = tactic or default_tactic(num_tokens)
    if split_k is None:
        split_k = default_split_k(num_tokens)
    if not split_k_is_supported(hidden_states.shape[1], split_k, hidden_states.dtype):
        split_k = 1
    fused = fused and terms > 1 and fold_is_supported(tactic, stacked_n)

    if split_k > 1:
        # Folding in the epilogue would only shrink the partials, not remove
        # the reduction, so the second pass may as well sum both at once.
        partials = torch.empty(
            (split_k, num_tokens, stacked_n), dtype=torch.float32, device=hidden_states.device
        )
        GateGemmRunner.run(hidden_states, w_split, partials, tactic, fold=1, split_k=split_k)
        return partials.view(split_k, num_tokens, terms, num_experts).sum(dim=(0, 2))

    partials = torch.empty(
        (num_tokens, stacked_n), dtype=torch.float32, device=hidden_states.device
    )
    GateGemmRunner.run(hidden_states, w_split, partials, tactic, fold=terms if fused else 1)
    if fused or terms == 1:
        return partials[:, :num_experts]
    return partials.view(num_tokens, terms, num_experts).sum(dim=1)
