# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode fused sigmoid TopK + routing maps for the MTP3 / MTP5 wide decode rows
(65..256) with replicated hot experts.

At 128 / 192 decode rows the routed experts' grouped GEMMs stream one expert's weights per
active local expert, so a rank's MoE time follows its count of distinct active experts, and
under EP4 every rank waits in the post-MoE all-reduce for the rank with the most. On real
routing (32 MMLU prompts, teacher-forced, 75 layers; 32 requests x 4 / 6 consecutive rows)
the busiest rank holds 2.66 / 2.09 more active experts than the mean per layer; moving the
layer's 8 hottest experts whole to the least loaded ranks cuts the busiest rank's count by
1.46 / 1.23 experts (offline greedy, the decode replica policy).

This kernel is ``moe_topk_routing_push_wide``'s launch (one 8- or 16-CTA cluster, ``WARPS``
token-warps per CTA, one warp per token, the push hand-off, experts past 128 rows on two
adjacent tiles) with ``moe_decode_topk_routing_rep``'s assignment:

* each CTA claims slots in a 256-bin *global* histogram (``_select_topk_global``: the push
  kernel's selection and weights bit for bit; or, ``trim``, the wide kernel's trimmed winner
  lookup over the same 256 bins) and pushes it to cluster rank 0;
* rank 0 (256 threads, one global expert each) counts per EP rank the active experts that
  are not hot, assigns each active hot expert (``hot_ids`` order) to the EP rank with the
  fewest active experts so far (ties: lowest rank) -- every rank derives the same
  assignment from the same TopK, with no communication;
* this EP rank's slots are its active home experts that are not hot (``e - offset``) and
  the hot experts assigned to it (``64 + h``, whatever their home); a slot past 128 rows
  takes two tiles. Tiles follow slot order: two bitmasks over the ``64 + NREP`` slots ("any
  row", "past 128 rows"), the tile of a slot = popc of both below it;
* rank 0 pushes every CTA a 256-entry base (-1: another rank's), and each winner maps
  through it.

Every (token, k) is mapped on exactly one EP rank; the EP all-reduce sums the ranks' FC2
partials as before, only the rank computing a hot expert's partial changes (bf16 partial
grouping: not bitwise against the home mapping end to end). ``expert_hist`` keeps the wide
kernel's meaning (this step's count of each home expert). ``hot_ids`` entries < 0 are unused
slots: an all -1 list is the home mapping. Tile capacity ``max_tiles(T, NREP)``; FC1 / FC2
run on stacks ``64 + NREP`` deep with the replica weights in those slots.
"""

from typing import Callable, Dict, Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from . import moe_decode_topk_routing_rep as _rep
from . import moe_topk_routing as _base
from . import moe_topk_routing_push_wide as _wide

_WARP = _base._WARP
_FULL_MASK = _base._FULL_MASK
_EXPERTS = _base._EXPERTS
_LOCAL_EXPERTS = _base._LOCAL_EXPERTS
_TOP_K = _base._TOP_K
_TILE = _base._TILE
_VEC = _rep._VEC
_CLUSTER_CTAS = _wide._CLUSTER_CTAS
CLUSTERS = _wide.CLUSTERS
#: The slot bitmasks are 4 x 32 bits over the 64 home + NREP replica slots.
MAX_NREP = 64
MIN_TOKENS = _wide.MIN_TOKENS
MAX_TOKENS = _wide.MAX_TOKENS
WARP_COUNTS = _wide.WARP_COUNTS
warps_for_tokens = _wide.warps_for_tokens
OUTPUT_ORDER = _wide.OUTPUT_ORDER

assert min(WARP_COUNTS) * _CLUSTER_CTAS // max(CLUSTERS) * _WARP >= _EXPERTS, (
    "rank 0 maps one global expert per thread"
)

TopKRoutingFn = Callable[..., None]
_CACHE: Dict[Tuple, TopKRoutingFn] = {}


def max_tiles(num_tokens: int, nrep: int) -> int:
    """The tile records ``num_tokens`` rows can produce on one rank: one per active slot (at
    most ``64 + nrep``) plus one per slot past 128 rows."""
    expanded = int(num_tokens) * _TOP_K
    filled = min(_LOCAL_EXPERTS + int(nrep), expanded)
    return filled + (expanded - filled) // _TILE


@cute.jit
def _popc_below(w0, w1, w2, w3, ext: cutlass.Int32) -> cutlass.Int32:
    """Set bits of the 128-bit mask ``(w0, w1, w2, w3)`` below bit ``ext``."""
    word = ext >> cutlass.Int32(5)
    below = (cutlass.Uint32(1) << cutlass.Uint32(ext & cutlass.Int32(31))) - cutlass.Uint32(1)
    m0 = cutlass.Uint32(0xFFFFFFFF)
    if word == cutlass.Int32(0):
        m0 = below
    m1 = cutlass.Uint32(0)
    if word > cutlass.Int32(1):
        m1 = cutlass.Uint32(0xFFFFFFFF)
    if word == cutlass.Int32(1):
        m1 = below
    m2 = cutlass.Uint32(0)
    if word > cutlass.Int32(2):
        m2 = cutlass.Uint32(0xFFFFFFFF)
    if word == cutlass.Int32(2):
        m2 = below
    m3 = cutlass.Uint32(0)
    if word == cutlass.Int32(3):
        m3 = below
    return (
        cutlass.Int32(cute.arch.popc(w0 & m0))
        + cutlass.Int32(cute.arch.popc(w1 & m1))
        + cutlass.Int32(cute.arch.popc(w2 & m2))
        + cutlass.Int32(cute.arch.popc(w3 & m3))
    )


@cute.kernel
def fused_topk_routing_push_wide_rep_kernel(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    hot_ids: cute.Tensor,
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
    NREP: cutlass.Constexpr,
    CLUSTER: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    expert_hist: Optional[cute.Tensor] = None,
):
    """One CTA of the ``CLUSTER``-CTA cluster owns ``WARPS`` tokens, one warp owns a token;
    rank 0 assigns (module doc). ``TRIM``: the trimmed winner lookup."""
    tidx, _, _ = cute.arch.thread_idx()
    rank, _, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)
    count = cutlass.Array(cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128)
    all_counts = cutlass.Array(
        cutlass.Int32, CLUSTER * _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    base = cutlass.Array(cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128)
    hot = cutlass.Array(cutlass.Int32, NREP, space=cutlass.AddressSpace.smem, alignment=16)
    hot_active = cutlass.Array(cutlass.Int32, NREP, space=cutlass.AddressSpace.smem, alignment=16)
    assign = cutlass.Array(cutlass.Int32, NREP, space=cutlass.AddressSpace.smem, alignment=16)
    warp_cold = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    one_bits = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    two_bits = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    count_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    base_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if tidx < cutlass.Int32(_EXPERTS):
        count.subview(tidx).store(cutlass.Int32(0))
    if tidx < cutlass.Int32(NREP):
        # Router-independent layer constants, read before the PDL wait.
        hot.subview(tidx).store(cutlass.Int32(hot_ids[tidx]))
        hot_active.subview(tidx).store(cutlass.Int32(0))
    if tidx < cutlass.Int32(4):
        one_bits.subview(tidx).store(cutlass.Int32(0))
        two_bits.subview(tidx).store(cutlass.Int32(0))
    if tidx == cutlass.Int32(0):
        prims.mbarrier_init(count_bar.subview(0), 1)
        prims.mbarrier_init(base_bar.subview(0), 1)
        prims.fence_mbarrier_init()
        # Armed now: a peer's complete_tx may land before this CTA reaches its wait.
        if rank == cutlass.Int32(0):
            prims.mbarrier_arrive_expect_tx(count_bar.subview(0), CLUSTER * _EXPERTS * 4)
        prims.mbarrier_arrive_expect_tx(base_bar.subview(0), _EXPERTS * 4)
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
        my_expert, my_slot = _wide._select_topk_trim(
            logits,
            LOGIT_SPLITS,
            _EXPERTS,
            bias_vals,
            topk_weights,
            topk_indices,
            count,
            token,
            lane,
            valid,
            cutlass.Int32(0),
            routed_scaling_factor,
        )
    else:
        my_expert, my_slot = _rep._select_topk_global(
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
            cutlass.Int32(0),
            routed_scaling_factor,
        )
    prims.barrier_cta_sync(0)  # this CTA's 256-bin histogram atomics are complete
    prims.barrier_cluster_wait()  # every CTA initialised its barriers
    if tidx < cutlass.Int32(_EXPERTS):
        _compat48.store_async(
            prims.mapa(all_counts, cutlass.Int32(0)).subview(rank * cutlass.Int32(_EXPERTS) + tidx),
            count.subview(tidx).load(),
            prims.mapa(count_bar.subview(0), cutlass.Int32(0)),
        )
    if rank == cutlass.Int32(0) and tidx < cutlass.Int32(_EXPERTS):
        # One global expert e = tidx per thread; warps 2r, 2r + 1 cover EP rank r's 64.
        while not prims.mbarrier_try_wait_parity(count_bar.subview(0), 0):
            pass
        rank_prefix = cutlass.Array(cutlass.Int32, CLUSTER)
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(CLUSTER):
            rank_prefix[r] = total
            total = total + all_counts.subview(cutlass.Int32(r * _EXPERTS) + tidx).load()
        active = total != cutlass.Int32(0)
        my_hot = cutlass.Int32(-1)
        for h in cutlass.range_constexpr(NREP):
            if hot.subview(h).load() == tidx:
                my_hot = cutlass.Int32(h)
        cold_i = cutlass.Int32(0)
        if active:
            if my_hot < cutlass.Int32(0):
                cold_i = cutlass.Int32(1)
        cold_ball = cutlass.Uint32(
            prims.vote_sync(_FULL_MASK, cold_i != cutlass.Int32(0), prims.VoteSync.BALLOT)
        )
        if lane == cutlass.Int32(0):
            warp_cold.subview(warp).store(cutlass.Int32(cute.arch.popc(cold_ball)))
        if active and my_hot >= cutlass.Int32(0):
            hot_active.subview(my_hot).store(cutlass.Int32(1))
        prims.barrier_cta_sync(1, thread_count=_EXPERTS)
        if tidx == cutlass.Int32(0):
            # Deterministic greedy, identical on every EP rank (same TopK, same totals).
            n0 = warp_cold.subview(0).load() + warp_cold.subview(1).load()
            n1 = warp_cold.subview(2).load() + warp_cold.subview(3).load()
            n2 = warp_cold.subview(4).load() + warp_cold.subview(5).load()
            n3 = warp_cold.subview(6).load() + warp_cold.subview(7).load()
            for h in cutlass.range_constexpr(NREP):
                a = cutlass.Int32(-1)
                if hot_active.subview(h).load() != cutlass.Int32(0):
                    a = cutlass.Int32(0)
                    best = n0
                    if n1 < best:
                        a = cutlass.Int32(1)
                        best = n1
                    if n2 < best:
                        a = cutlass.Int32(2)
                        best = n2
                    if n3 < best:
                        a = cutlass.Int32(3)
                        best = n3
                    if a == cutlass.Int32(0):
                        n0 = n0 + cutlass.Int32(1)
                    if a == cutlass.Int32(1):
                        n1 = n1 + cutlass.Int32(1)
                    if a == cutlass.Int32(2):
                        n2 = n2 + cutlass.Int32(1)
                    if a == cutlass.Int32(3):
                        n3 = n3 + cutlass.Int32(1)
                assign.subview(h).store(a)
        prims.barrier_cta_sync(1, thread_count=_EXPERTS)
        my_ep = local_expert_offset >> cutlass.Int32(6)
        ext = cutlass.Int32(-1)
        if active:
            if my_hot >= cutlass.Int32(0):
                if assign.subview(my_hot).load() == my_ep:
                    ext = cutlass.Int32(_LOCAL_EXPERTS) + my_hot
            elif (tidx >> cutlass.Int32(6)) == my_ep:
                ext = tidx - local_expert_offset
        two = ext >= cutlass.Int32(0) and total > cutlass.Int32(_TILE)
        if ext >= cutlass.Int32(0):
            prims.atomicrmw(
                "or",
                one_bits.subview(ext >> cutlass.Int32(5)),
                cutlass.Int32(1) << (ext & cutlass.Int32(31)),
            )
        if two:
            prims.atomicrmw(
                "or",
                two_bits.subview(ext >> cutlass.Int32(5)),
                cutlass.Int32(1) << (ext & cutlass.Int32(31)),
            )
        prims.barrier_cta_sync(1, thread_count=_EXPERTS)
        w0 = cutlass.Uint32(one_bits.subview(0).load())
        w1 = cutlass.Uint32(one_bits.subview(1).load())
        w2 = cutlass.Uint32(one_bits.subview(2).load())
        w3 = cutlass.Uint32(one_bits.subview(3).load())
        v0 = cutlass.Uint32(two_bits.subview(0).load())
        v1 = cutlass.Uint32(two_bits.subview(1).load())
        v2 = cutlass.Uint32(two_bits.subview(2).load())
        v3 = cutlass.Uint32(two_bits.subview(3).load())
        tile = cutlass.Int32(0)
        expert_base = cutlass.Int32(-1)
        if ext >= cutlass.Int32(0):
            # A slot's first tile: the slots below it, plus one more per slot below past 128 rows.
            tile = _popc_below(w0, w1, w2, w3, ext) + _popc_below(v0, v1, v2, v3, ext)
            expert_base = tile * cutlass.Int32(_TILE)
        for r in cutlass.range_constexpr(CLUSTER):
            value = cutlass.Int32(-1)
            if ext >= cutlass.Int32(0):
                value = expert_base + rank_prefix[r]
            _compat48.store_async(
                prims.mapa(base, cutlass.Int32(r)).subview(tidx),
                value,
                prims.mapa(base_bar.subview(0), cutlass.Int32(r)),
            )
        if ext >= cutlass.Int32(0):
            first_rows = total
            if two:
                first_rows = cutlass.Int32(_TILE)
            tile_idx_to_expert_idx[tile] = ext
            tile_idx_to_mn_limit[tile] = expert_base + first_rows
        if two:
            tile_idx_to_expert_idx[tile + cutlass.Int32(1)] = ext
            tile_idx_to_mn_limit[tile + cutlass.Int32(1)] = expert_base + total
        if cutlass.const_expr(expert_hist is not None):
            # The wide kernel's counts: this step's rows of each home expert (zero included).
            if (tidx >> cutlass.Int32(6)) == my_ep:
                expert_hist[tidx - local_expert_offset] = total
        if tidx == cutlass.Int32(0):
            tiles = (
                cutlass.Int32(cute.arch.popc(w0))
                + cutlass.Int32(cute.arch.popc(w1))
                + cutlass.Int32(cute.arch.popc(w2))
                + cutlass.Int32(cute.arch.popc(w3))
                + cutlass.Int32(cute.arch.popc(v0))
                + cutlass.Int32(cute.arch.popc(v1))
                + cutlass.Int32(cute.arch.popc(v2))
                + cutlass.Int32(cute.arch.popc(v3))
            )
            num_non_exiting_tiles[0] = tiles
            total_num_padded_tokens[0] = tiles * cutlass.Int32(_TILE)
    if (valid and lane < cutlass.Int32(_TOP_K)) or tidx == cutlass.Int32(0):
        # Thread 0 waits even when this CTA owns no valid token: rank 0 still writes its bases.
        while not prims.mbarrier_try_wait_parity(base_bar.subview(0), 0):
            pass
    if valid and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        permuted = cutlass.Int32(-1)
        if my_expert >= cutlass.Int32(0):
            b = base.subview(my_expert).load()
            if b >= cutlass.Int32(0):
                permuted = b + my_slot
                permuted_idx_to_expanded_idx[permuted] = expanded
        expanded_idx_to_permuted_idx[expanded] = permuted


fused_topk_routing_push_wide_rep_kernel.set_name_prefix(
    "glm52_moe_topk_routing_push_wide_rep", remove_cutlass_symbol=True
)


@cute.jit
def launch_fused_topk_routing_push_wide_rep(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    hot_ids: cute.Tensor,
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
    NREP: cutlass.Constexpr,
    CLUSTER: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
    expert_hist: Optional[cute.Tensor] = None,
):
    """Launch the ``CLUSTER``-CTA cluster with ``WARPS`` token-warps per CTA."""
    fused_topk_routing_push_wide_rep_kernel(
        logits,
        routing_bias,
        hot_ids,
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
        NREP,
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


def _validate(logits, routing_bias, hot_ids, outputs, nrep):
    """`(device, num_tokens, logit_splits)`; the outputs need only hold the rows' entries."""
    if not 0 < int(nrep) <= MAX_NREP:
        raise ValueError(f"nrep must be in [1, {MAX_NREP}], got {nrep}")
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
    if tuple(hot_ids.shape) != (int(nrep),) or hot_ids.dtype != torch.int32:
        raise ValueError(f"hot_ids must be int32 [{nrep}]")
    if not hot_ids.is_contiguous() or hot_ids.data_ptr() % 16:
        raise ValueError("hot_ids must be contiguous and 16-byte aligned")
    tiles = max_tiles(num_tokens, nrep)
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


def compile_fused_topk_routing_wide_rep(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    hot_ids: "torch.Tensor",
    outputs: Dict[str, "torch.Tensor"],
    *,
    nrep: int,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    expert_hist: Optional["torch.Tensor"] = None,
    cluster: int = _CLUSTER_CTAS,
    trim: bool = False,
) -> TopKRoutingFn:
    """Compile (once per device / cluster / lookup / token-warp count / logit planes / PDL
    roles / nrep / counts) the handle for ``logits``' row range (``warps_for_tokens``). It takes
    ``(logits, bias, hot_ids, *outputs in OUTPUT_ORDER, num_tokens, local_expert_offset,
    routed_scaling_factor, stream[, expert_hist])``; any row count of the range runs on it."""
    device, num_tokens, logit_splits = _validate(logits, routing_bias, hot_ids, outputs, nrep)
    _wide._push._check_hist(expert_hist, logits.device)
    warps = warps_for_tokens(num_tokens, cluster)
    key = (
        device,
        int(cluster),
        bool(trim),
        warps,
        logit_splits,
        bool(wait_for_router_pdl),
        bool(signal_gg1_pdl),
        int(nrep),
        expert_hist is not None,
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        as_t = _base._as_tensor
        _CACHE[key] = cute.compile(
            launch_fused_topk_routing_push_wide_rep,
            as_t(logits.reshape(-1), assumed_align=16),
            as_t(routing_bias),
            as_t(hot_ids, assumed_align=16),
            *(as_t(outputs[name]) for name in OUTPUT_ORDER),
            cutlass.Int32(num_tokens),
            cutlass.Int32(0),
            cutlass.Float32(1.0),
            warps,
            logit_splits,
            bool(wait_for_router_pdl),
            bool(signal_gg1_pdl),
            int(nrep),
            int(cluster),
            bool(trim),
            cutlass_torch.current_stream(),
            *((as_t(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_fused_topk_routing_wide_rep(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    hot_ids: "torch.Tensor",
    outputs: Dict[str, "torch.Tensor"],
    *,
    nrep: int,
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
    compiled = compile_fused_topk_routing_wide_rep(
        logits,
        routing_bias,
        hot_ids,
        outputs,
        nrep=nrep,
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
        as_t(hot_ids, assumed_align=16),
        *(as_t(outputs[name]) for name in OUTPUT_ORDER),
        cutlass.Int32(int(logits.shape[0])),
        cutlass.Int32(local_expert_offset),
        cutlass.Float32(routed_scaling_factor),
        cutlass_torch.current_stream(),
        *((as_t(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
    )
