# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode fused sigmoid TopK + routing maps for the MTP3 / MTP5 decode widths
(65..256 rows): ``moe_topk_routing_push`` with more token-warps per CTA and experts that may
span two 128-row tiles.

``moe_topk_routing_push`` serves 9..64 rows: one warp per token, 8 token-warps per 256-thread
CTA, at most the 8-CTA portable cluster, so 64 rows is its ceiling; and with at most 64 rows
every active local expert holds exactly one tile. MTP3 / MTP5 verify steps and draft step 0
carry 128 / 192 rows (32 sequences x 4 / 6), which the context path served with the
three-launch prefill top-k (histogram, prefix, scatter). Here:

* **More token-warps, same cluster.** ``WARPS`` token-warps per CTA (16, 24 or 32: 512, 768
  or 1024 threads) on the 8-CTA cluster serve up to 128, 192 or 256 rows; one warp still
  owns one token. ``nvvm.reqntid`` (from the static block size) caps the registers.
* **Or a 16-CTA cluster** (``cluster=16``, non-portable): half the token-warps per CTA (8,
  12 or 16) on twice the SMs. The selection is issue-bound per SM, so it ends earlier;
  everything else -- each token's warp code, the counts, the tile layout -- is the 8-CTA
  kernel's, the prefix only summing 16 ranks' histograms.
* **Optionally a trimmed winner lookup** (``trim``, ``_select_topk_trim``): the same rounds,
  winners and order, but each winner's unbiased sigmoid value fetched once after the eight
  rounds (eight shuffles from the owner lanes) instead of a head search and a shuffle per
  round. Same values, so the same weights bit for bit.

B300, router + top-k behind an all-reduce-like PDL producer (1 GPU, graph-replayed, 8
cold layers): 21.6 -> 20.5 us per layer at 128 rows on the 16-CTA cluster, 20.3 with the
trimmed lookup; 23.1 -> 21.3 / 21.0 at 192.
* **The push kernel's selection.** ``moe_topk_routing_push._select_topk_lanevec`` with its
  one-redux rounds: winners, their order, the unbiased sigmoid values, the weight
  butterfly and the per-expert counts are that kernel's -- and the prefill top-k's, whose
  ``moe_topk_routing._select_topk_and_count`` they equal -- bit for bit.
* **The push hand-off.** Each CTA ``st.async``'s its 64-bin histogram into rank 0's
  ``[8, 64]`` slab; rank 0 prefixes and ``st.async``'s each rank's expert bases back. One
  cluster barrier orders the barrier initialisation, off the critical path.
* **Tiles, not experts.** Rank 0's prefix counts ``ceil(c / 128)`` tiles per local expert of
  ``c`` rows (two above 128 rows, at most 256 rows a step) by two ballots per warp. An
  expert's rows stay contiguous across its adjacent tiles, so its base is its first tile's
  first row and each tile's limit an absolute row bound: the canonical layout of
  ``cute_dsl_moe_routing.reference_maps``, which both grouped GEMM pairs read.

PDL roles and the optional ``expert_hist`` (rank 0 writes the step's per-local-expert token
counts for the next step's L2 prefetch) are ``moe_topk_routing_push``'s. No fused output
clear: the trunk's decode steps clear on the quantize or accumulate onto the shared expert.
The row count is a runtime argument: one compile per ``WARPS`` serves its whole range.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from . import moe_topk_routing as _base
from . import moe_topk_routing_push as _push

_WARP = _base._WARP
_LOCAL_EXPERTS = _base._LOCAL_EXPERTS
_EXPERTS = _base._EXPERTS
_TOP_K = _base._TOP_K
_TILE = _base._TILE
_VEC = _push._VEC
_FULL_MASK = _base._FULL_MASK
#: The portable cluster: every CTA of it holds ``WARPS`` tokens.
_CLUSTER_CTAS = 8
#: The cluster sizes the kernel compiles for: the portable 8 and the non-portable 16.
CLUSTERS = (8, 16)
#: Token-warps per CTA on the 8-CTA cluster, by row range: 65..128, 129..192, 193..256 (half
#: on the 16-CTA one).
WARP_COUNTS = (16, 24, 32)
MIN_TOKENS = _base._MAX_TOKENS + 1
MAX_TOKENS = _CLUSTER_CTAS * WARP_COUNTS[-1]
assert MAX_TOKENS <= 2 * _TILE, "an expert takes at most two tiles"

OUTPUT_ORDER = _push.OUTPUT_ORDER

TopKRoutingFn = Callable[..., None]
_CACHE: Dict[Tuple, TopKRoutingFn] = {}


def warps_for_tokens(num_tokens: int, cluster: int = _CLUSTER_CTAS) -> int:
    """Token-warps per CTA for ``num_tokens`` (65..256) on a ``cluster``-CTA cluster: 16, 24
    or 32 on 8 CTAs, 8, 12 or 16 on 16. The row range of a count is the same on both."""
    if not MIN_TOKENS <= num_tokens <= MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [{MIN_TOKENS}, {MAX_TOKENS}], got {num_tokens}")
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {CLUSTERS}, got {cluster}")
    return 8 * math.ceil(num_tokens / (8 * _CLUSTER_CTAS)) * _CLUSTER_CTAS // cluster


def max_tiles(num_tokens: int) -> int:
    """The tile records ``num_tokens`` rows can produce: one per active local expert (at most
    64) plus one per expert past 128 rows."""
    expanded = int(num_tokens) * _TOP_K
    filled = min(_LOCAL_EXPERTS, expanded)
    return filled + (expanded - filled) // _TILE


@cute.jit
def _select_topk_trim(
    logits: cute.Tensor,
    logit_splits: cutlass.Constexpr,
    window: cutlass.Constexpr,
    bias_vals,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    expert_count,
    token: cutlass.Int32,
    lane: cutlass.Int32,
    valid_token,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
):
    """``moe_topk_routing_push._select_topk_lanevec`` (one-redux rounds) with the winners'
    unbiased sigmoid values fetched once after the eight rounds: winner ``e`` sits in lane
    ``e / 8``, register ``e % 8``, so eight shuffles of the lanes' values give lane ``k`` its
    winner's. Same winners, order, values, weight butterfly and counts, bit for bit. The
    count histogram covers ``window`` experts from ``local_expert_offset`` (64: this EP
    rank's local experts, as the push kernel; 256 from 0: every expert, as
    ``moe_decode_topk_routing_rep._select_topk_global``). Returns this lane's (expert in the
    window or -1, histogram slot)."""
    keys = cutlass.Array(cutlass.Uint64, _VEC)
    raw_scores = cutlass.Array(cutlass.Float32, _VEC)
    my_raw = cutlass.Float32(0.0)
    my_selected_idx = cutlass.Int32(-1)
    for v in cutlass.range_constexpr(_VEC):
        keys[v] = cutlass.Uint64(0)
        raw_scores[v] = cutlass.Float32(0.0)
    if valid_token:
        lptr = logits.iterator.raw_ptr()
        row = token * cutlass.Int32(_EXPERTS * logit_splits) + lane * cutlass.Int32(_VEC)
        parts = []
        for split in cutlass.range_constexpr(logit_splits):
            lo = prims.load_ext(
                lptr + row + cutlass.Int32(split * _EXPERTS), dtype=cutlass.Float32, count=4
            )
            hi = prims.load_ext(
                lptr + row + cutlass.Int32(split * _EXPERTS + 4), dtype=cutlass.Float32, count=4
            )
            parts.append((lo, hi))
        for v in cutlass.range_constexpr(_VEC):
            half = 0 if v < 4 else 1
            idx = v if v < 4 else v - 4
            logit = cutlass.Float32(parts[0][half][idx])
            for split in cutlass.range_constexpr(1, logit_splits):
                logit = logit + cutlass.Float32(parts[split][half][idx])
            raw = cutlass.Float32(0.5) * cute.math.tanh(
                cutlass.Float32(0.5) * logit
            ) + cutlass.Float32(0.5)
            raw_scores[v] = raw
            keys[v] = _base._pack_score_index(
                raw + bias_vals[v], lane * cutlass.Int32(_VEC) + cutlass.Int32(v)
            )
        _base._sort8_key_desc(keys)
        head_idx = cutlass.Int32(65535) - cutlass.Int32(keys[0] & cutlass.Uint64(0xFFFF))
        for k in cutlass.range_constexpr(_TOP_K):
            score_key = cutlass.Uint32(keys[0] >> cutlass.Uint64(32))
            warp_score_key = cutlass.Uint32(
                prims.redux_sync(cutlass.Int32(score_key), prims.ReductionKind.UMAX, _FULL_MASK)
            )
            is_max = score_key == warp_score_key
            ball = cutlass.Uint32(prims.vote_sync(_FULL_MASK, is_max, prims.VoteSync.BALLOT))
            owner = cutlass.Int32(cute.arch.popc(ball - cutlass.Uint32(1)))
            if cutlass.Int32(cute.arch.popc(ball)) != cutlass.Int32(1):
                # Exact score tie: the served tie-break, smallest expert index (largest inverse index).
                inverse_idx = cutlass.Uint32(0)
                if is_max:
                    inverse_idx = cutlass.Uint32(keys[0] & cutlass.Uint64(0xFFFF))
                winner_inverse_idx = cutlass.Uint32(
                    prims.redux_sync(
                        cutlass.Int32(inverse_idx), prims.ReductionKind.UMAX, _FULL_MASK
                    )
                )
                cand = cutlass.Int32(0)
                if is_max and inverse_idx == winner_inverse_idx:
                    cand = cutlass.Int32(1)
                ball2 = cutlass.Uint32(
                    prims.vote_sync(_FULL_MASK, cand != cutlass.Int32(0), prims.VoteSync.BALLOT)
                )
                owner = cutlass.Int32(cute.arch.popc(ball2 - cutlass.Uint32(1)))
            winner_idx = cute.arch.shuffle_sync(head_idx, owner)
            if cutlass.const_expr(k < _TOP_K - 1):
                if lane == owner:
                    for v in cutlass.range_constexpr(_VEC - 1):
                        keys[v] = keys[v + 1]
                    keys[_VEC - 1] = cutlass.Uint64(0)
                    head_idx = cutlass.Int32(65535) - cutlass.Int32(
                        keys[0] & cutlass.Uint64(0xFFFF)
                    )
            if lane == cutlass.Int32(k):
                my_selected_idx = winner_idx
        # Lane k (< 8) takes winner k's value from its owner lane and register.
        src = (my_selected_idx >> cutlass.Int32(3)) & cutlass.Int32(_WARP - 1)
        reg = my_selected_idx & cutlass.Int32(_VEC - 1)
        for v in cutlass.range_constexpr(_VEC):
            value = cute.arch.shuffle_sync(raw_scores[v], src)
            if lane < cutlass.Int32(_TOP_K) and reg == cutlass.Int32(v):
                my_raw = value
    weight_sum = _base._warp_sum_f32(my_raw)
    my_local = cutlass.Int32(-1)
    my_slot = cutlass.Int32(0)
    if valid_token and lane < cutlass.Int32(_TOP_K):
        expert = my_selected_idx
        expanded = token * cutlass.Int32(_TOP_K) + lane
        topk_indices[expanded] = expert
        topk_weights[expanded] = (
            my_raw * routed_scaling_factor / (weight_sum + cutlass.Float32(1.0e-20))
        )
        local = expert - local_expert_offset
        if local >= cutlass.Int32(0) and local < cutlass.Int32(window):
            my_local = local
            my_slot = prims.atomicrmw("add", expert_count.subview(local), 1)
    return my_local, my_slot


@cute.kernel
def fused_topk_routing_push_wide_kernel(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_tokens: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
    WARPS: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    CLUSTER: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    expert_hist: Optional[cute.Tensor] = None,
):
    """One CTA of the ``CLUSTER``-CTA cluster owns ``WARPS`` tokens, one warp owns a token;
    push hand-off (module doc). ``TRIM``: the trimmed winner lookup (``_select_topk_trim``)."""
    tidx, _, _ = cute.arch.thread_idx()
    rank, _, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)
    count = cutlass.Array(
        cutlass.Int32, _LOCAL_EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    all_counts = cutlass.Array(
        cutlass.Int32,
        CLUSTER * _LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    base = cutlass.Array(
        cutlass.Int32, _LOCAL_EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    warp_totals = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)
    count_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    base_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if tidx < cutlass.Int32(_LOCAL_EXPERTS):
        count.subview(tidx).store(cutlass.Int32(0))
    if tidx == cutlass.Int32(0):
        prims.mbarrier_init(count_bar.subview(0), 1)
        prims.mbarrier_init(base_bar.subview(0), 1)
        prims.fence_mbarrier_init()
        # Armed now: a peer's complete_tx may land before this CTA reaches its wait.
        if rank == cutlass.Int32(0):
            prims.mbarrier_arrive_expect_tx(count_bar.subview(0), CLUSTER * _LOCAL_EXPERTS * 4)
        prims.mbarrier_arrive_expect_tx(base_bar.subview(0), _LOCAL_EXPERTS * 4)
    # Router-independent: this lane's 8 contiguous bias values, before the PDL wait.
    bias_vals = cutlass.Array(cutlass.Float32, _VEC)
    for v in cutlass.range_constexpr(_VEC):
        bias_vals[v] = cutlass.Float32(routing_bias[lane * cutlass.Int32(_VEC) + cutlass.Int32(v)])
    prims.barrier_cta_sync(0)
    # Orders every CTA's barrier initialisation before any peer's st.async complete_tx.
    prims.barrier_cluster_arrive()
    token = rank * cutlass.Int32(WARPS) + warp
    valid = token < num_tokens
    if cutlass.const_expr(WAIT_FOR_ROUTER_PDL):
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SIGNAL_GG1_PDL):
        cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(TRIM):
        my_local, my_slot = _select_topk_trim(
            logits,
            LOGIT_SPLITS,
            _LOCAL_EXPERTS,
            bias_vals,
            topk_weights,
            topk_indices,
            count,
            token,
            lane,
            valid,
            local_expert_offset,
            routed_scaling_factor,
        )
    else:
        my_local, my_slot = _push._select_topk_lanevec(
            logits,
            LOGIT_SPLITS,
            True,
            bias_vals,
            topk_weights,
            topk_indices,
            count,
            token,
            lane,
            valid,
            local_expert_offset,
            routed_scaling_factor,
        )
    prims.barrier_cta_sync(0)  # this CTA's histogram atomics are complete
    prims.barrier_cluster_wait()  # every CTA initialised its barriers
    if tidx < cutlass.Int32(_LOCAL_EXPERTS):
        _compat48.store_async(
            prims.mapa(all_counts, cutlass.Int32(0)).subview(
                rank * cutlass.Int32(_LOCAL_EXPERTS) + tidx
            ),
            count.subview(tidx).load(),
            prims.mapa(count_bar.subview(0), cutlass.Int32(0)),
        )
    if rank == cutlass.Int32(0) and tidx < cutlass.Int32(_LOCAL_EXPERTS):
        while not prims.mbarrier_try_wait_parity(count_bar.subview(0), 0):
            pass
        rank_prefix = cutlass.Array(cutlass.Int32, CLUSTER)
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(CLUSTER):
            rank_prefix[r] = total
            total = total + all_counts.subview(cutlass.Int32(r * _LOCAL_EXPERTS) + tidx).load()
        # ceil(total / 128) tiles, 0..2: one ballot for "any row", one for "past 128 rows".
        one = total != cutlass.Int32(0)
        two = total > cutlass.Int32(_TILE)
        prefix_one, count_one = _base._ballot_exclusive(one)
        prefix_two, count_two = _base._ballot_exclusive(two)
        if lane == cutlass.Int32(_WARP - 1):
            warp_totals.subview(warp).store(count_one + count_two)
        prims.barrier_cta_sync(1, thread_count=_LOCAL_EXPERTS)
        first_warp_total = warp_totals.subview(0).load()
        tile = prefix_one + prefix_two
        if warp == cutlass.Int32(1):
            tile = tile + first_warp_total
        expert_base = tile * cutlass.Int32(_TILE)
        for r in cutlass.range_constexpr(CLUSTER):
            _compat48.store_async(
                prims.mapa(base, cutlass.Int32(r)).subview(tidx),
                expert_base + rank_prefix[r],
                prims.mapa(base_bar.subview(0), cutlass.Int32(r)),
            )
        if one:
            first_rows = total
            if two:
                first_rows = cutlass.Int32(_TILE)
            tile_idx_to_expert_idx[tile] = tidx
            tile_idx_to_mn_limit[tile] = expert_base + first_rows
        if two:
            tile_idx_to_expert_idx[tile + cutlass.Int32(1)] = tidx
            tile_idx_to_mn_limit[tile + cutlass.Int32(1)] = expert_base + total
        if cutlass.const_expr(expert_hist is not None):
            # the decode MoE L2 prefetch: this step's count of local expert `tidx` (zero included)
            expert_hist[tidx] = total
        if tidx == cutlass.Int32(0):
            tiles = first_warp_total + warp_totals.subview(1).load()
            num_non_exiting_tiles[0] = tiles
            total_num_padded_tokens[0] = tiles * cutlass.Int32(_TILE)
    if (valid and lane < cutlass.Int32(_TOP_K)) or tidx == cutlass.Int32(0):
        # Thread 0 waits even when this CTA owns no valid token: rank 0 still writes its bases.
        while not prims.mbarrier_try_wait_parity(base_bar.subview(0), 0):
            pass
    if valid and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        if my_local >= cutlass.Int32(0):
            permuted = base.subview(my_local).load() + my_slot
            expanded_idx_to_permuted_idx[expanded] = permuted
            permuted_idx_to_expanded_idx[permuted] = expanded
        else:
            expanded_idx_to_permuted_idx[expanded] = cutlass.Int32(-1)


fused_topk_routing_push_wide_kernel.set_name_prefix(
    "glm52_moe_topk_routing_push_wide", remove_cutlass_symbol=True
)


@cute.jit
def launch_fused_topk_routing_push_wide(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_tokens: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
    WARPS: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    CLUSTER: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
    expert_hist: Optional[cute.Tensor] = None,
):
    """Launch the ``CLUSTER``-CTA cluster with ``WARPS`` token-warps per CTA."""
    fused_topk_routing_push_wide_kernel(
        logits,
        routing_bias,
        topk_weights,
        topk_indices,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        num_tokens,
        local_expert_offset,
        routed_scaling_factor,
        WARPS,
        LOGIT_SPLITS,
        WAIT_FOR_ROUTER_PDL,
        SIGNAL_GG1_PDL,
        CLUSTER,
        TRIM,
        expert_hist,
    ).launch(
        grid=(CLUSTER, 1, 1),
        block=(WARPS * _WARP, 1, 1),
        cluster=(CLUSTER, 1, 1),
        use_pdl=WAIT_FOR_ROUTER_PDL,
        stream=stream,
    )


def _validate(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    outputs: Dict[str, "torch.Tensor"],
) -> Tuple[int, int, int]:
    """`(device, num_tokens, logit_splits)`; the outputs need only hold the rows' entries."""
    if logits.ndim == 2:
        num_tokens, experts = logits.shape
        logit_splits = 1
    elif logits.ndim == 3:
        num_tokens, logit_splits, experts = logits.shape
        if logit_splits not in (2, 4, 8):
            raise ValueError("split logits must have 2, 4, or 8 K partitions")
    else:
        raise ValueError("logits must be [T,256] or [T,2|4|8,256]")
    num_tokens = int(num_tokens)
    warps_for_tokens(num_tokens)
    if experts != _EXPERTS:
        raise ValueError(f"logits must contain {_EXPERTS} experts, got {experts}")
    if logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous():
        raise ValueError("logits must be contiguous CUDA FP32")
    if logits.data_ptr() % 16:
        raise ValueError("the top-k loads 128-bit logits vectors: logits must be 16-byte aligned")
    if (
        tuple(routing_bias.shape) != (_EXPERTS,)
        or routing_bias.dtype != torch.float32
        or not routing_bias.is_contiguous()
    ):
        raise ValueError(f"routing_bias must be contiguous CUDA FP32 [{_EXPERTS}]")
    tiles = max_tiles(num_tokens)
    need = {
        "topk_weights": (num_tokens * _TOP_K, torch.float32),
        "topk_indices": (num_tokens * _TOP_K, torch.int32),
        "tile_idx_to_expert_idx": (tiles, torch.int32),
        "tile_idx_to_mn_limit": (tiles, torch.int32),
        "expanded_idx_to_permuted_idx": (num_tokens * _TOP_K, torch.int32),
        "permuted_idx_to_expanded_idx": (tiles * _TILE, torch.int32),
        "total_num_padded_tokens": (1, torch.int32),
        "num_non_exiting_tiles": (1, torch.int32),
    }
    for name in OUTPUT_ORDER:
        size, dtype = need[name]
        t = outputs[name]
        if t.dtype != dtype or not t.is_contiguous() or t.numel() < size:
            raise ValueError(f"{name} must be contiguous {dtype} holding at least {size}")
        if t.device != logits.device:
            raise ValueError("all routing tensors must share one CUDA device")
    device = logits.device.index
    return (torch.cuda.current_device() if device is None else device), num_tokens, logit_splits


def compile_fused_topk_routing_wide(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    outputs: Dict[str, "torch.Tensor"],
    *,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    expert_hist: Optional["torch.Tensor"] = None,
    cluster: int = _CLUSTER_CTAS,
    trim: bool = False,
) -> TopKRoutingFn:
    """Compile (once per device / cluster / lookup / token-warp count / logit planes / PDL
    roles / counts) the handle for ``logits``' row range (``warps_for_tokens``). It takes
    ``(logits, bias, *outputs in OUTPUT_ORDER, num_tokens, local_expert_offset,
    routed_scaling_factor, stream[, expert_hist])``; any row count of the same range runs on
    it."""
    device, num_tokens, logit_splits = _validate(logits, routing_bias, outputs)
    _push._check_hist(expert_hist, logits.device)
    warps = warps_for_tokens(num_tokens, cluster)
    key = (device, int(cluster), bool(trim), warps, logit_splits, bool(wait_for_router_pdl)) + (
        bool(signal_gg1_pdl),
        *((True,) if expert_hist is not None else ()),
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        as_t = _base._as_tensor
        _CACHE[key] = cute.compile(
            launch_fused_topk_routing_push_wide,
            as_t(logits.reshape(-1), assumed_align=16),
            as_t(routing_bias),
            *(as_t(outputs[name]) for name in OUTPUT_ORDER),
            cutlass.Int32(num_tokens),
            cutlass.Int32(0),
            cutlass.Float32(1.0),
            warps,
            logit_splits,
            bool(wait_for_router_pdl),
            bool(signal_gg1_pdl),
            int(cluster),
            bool(trim),
            cutlass_torch.current_stream(),
            *((as_t(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_fused_topk_routing_wide(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    outputs: Dict[str, "torch.Tensor"],
    *,
    local_expert_offset: int,
    routed_scaling_factor: float,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    expert_hist: Optional["torch.Tensor"] = None,
    cluster: int = _CLUSTER_CTAS,
    trim: bool = False,
) -> None:
    """Compile if needed and launch on the current stream without allocating."""
    if local_expert_offset not in (0, 64, 128, 192):
        raise ValueError(
            f"local_expert_offset must select one contiguous EP4 shard, got {local_expert_offset}"
        )
    compiled = compile_fused_topk_routing_wide(
        logits,
        routing_bias,
        outputs,
        wait_for_router_pdl=wait_for_router_pdl,
        signal_gg1_pdl=signal_gg1_pdl,
        expert_hist=expert_hist,
        cluster=cluster,
        trim=trim,
    )
    as_t = _base._as_tensor
    compiled(
        as_t(logits.reshape(-1), assumed_align=16),
        as_t(routing_bias),
        *(as_t(outputs[name]) for name in OUTPUT_ORDER),
        cutlass.Int32(int(logits.shape[0])),
        cutlass.Int32(local_expert_offset),
        cutlass.Float32(routed_scaling_factor),
        cutlass_torch.current_stream(),
        *((as_t(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
    )
