# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode fused sigmoid TopK + routing maps with a push-based cluster hand-off.

Same contract, outputs and host signature as ``moe_topk_routing`` (whose block
path, fused output zeroing and reference this module reuses); only the cluster
path (``num_tokens`` > warps per CTA, i.e. 9..64 decode rows at 256 threads)
is re-implemented, for latency:

* **Cluster hand-off by push, not by two cluster barriers.** In
  ``moe_topk_routing`` every CTA publishes its 64-bin local-expert histogram
  with a cluster barrier, rank 0 reads the eight histograms over DSMEM,
  computes the per-rank prefix, writes every rank's expert bases back over
  DSMEM and a second cluster barrier publishes them. Here each CTA
  ``st.async``'s its histogram into rank 0's ``[CTAS, 64]`` slab, completing
  on an mbarrier rank 0 armed for ``CTAS * 256`` bytes; rank 0 computes the
  prefix in the same rank order and ``st.async``'s each rank's bases into that
  rank, completing on its base barrier (armed for 256 bytes). Every CTA waits
  on its base barrier before its permutation-map writes and before exiting
  (rank 0 may still be writing into it). The one remaining cluster barrier
  (arrive after barrier init, wait before the first push) only orders barrier
  initialisation before any remote ``complete_tx`` and is off the critical
  path. Measured (fused decode router, B300, T=64, graph-replayed chain behind a
  PDL router): top-k end - router end 3.49 -> 2.94 us.
* **Lane-contiguous expert ownership.** A lane owns experts ``8*lane+v``, so
  each split-K partial row is two 128-bit loads per lane; the per-expert sum
  stays ``p0 + p1 + ... `` in split order. The routing bias (8 scalar loads)
  is read before the PDL wait.
* **One redux per selection round** (``ONE_REDUX``). Each round still takes
  the warp-max ``(score, -expert)`` key; the owner lane of the max score is
  found by ballot, and only an exact score tie (ballot popcount > 1) runs the
  second redux on the inverse expert index, as ``moe_topk_routing`` always
  does. The 8 rounds were 0.87 us of the 2.66 us selection phase.

Winners, their order, the unbiased sigmoid values, the weight-sum butterfly
(lane k holds winner k), the per-expert counts and therefore the tile maps and
counters are those of ``moe_topk_routing`` bit for bit; within an expert the
permutation rows follow the shared-memory atomic claim order, which the routing
contract leaves unspecified (``compare`` checks row sets and the inverse map).

PDL roles are unchanged: ``WAIT_FOR_ROUTER_PDL`` waits immediately before the
first logits load; ``SIGNAL_GG1_PDL`` admits GG1 right after that wait (GG1's
own wait protects every routing-metadata read).

Optional ``expert_hist`` (the decode MoE L2 prefetch): an int32 ``[64]`` tensor into which
rank 0 also writes the step's per-local-expert token counts (the totals it
already sums for the tile maps). The next decode step's L2 prefetch of the same
layer (``moe_decode_l2_prefetch``) ranks experts by it. Absent (``None``, the
default) the kernel is the one without it. Nothing else changes either way.
"""

from typing import Callable, Dict, Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from . import moe_topk_routing as _base

_BLOCK = _base._BLOCK
_WARP = _base._WARP
_WARPS = _base._WARPS
_FULL_MASK = _base._FULL_MASK
_EXPERTS = _base._EXPERTS
_TOP_K = _base._TOP_K
_LOCAL_EXPERTS = _base._LOCAL_EXPERTS
_TILE = _base._TILE
_VEC = 8  # experts per lane: 256 experts / 32 lanes
assert _EXPERTS == _VEC * _WARP

# One-redux selection rounds (bitwise identical either way; 0 keeps the served two-redux rounds).
_ONE_REDUX = True

OUTPUT_ORDER = _base.OUTPUT_ORDER
allocate_outputs = _base.allocate_outputs
cluster_ctas_for_tokens = _base.cluster_ctas_for_tokens

TopKRoutingFn = Callable[..., None]
_CACHE: Dict[Tuple[int, int, int, bool, bool, bool], TopKRoutingFn] = {}


@cute.jit
def _select_topk_lanevec(
    logits: cute.Tensor,
    logit_splits: cutlass.Constexpr,
    one_redux: cutlass.Constexpr,
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
    """One token's TopK with lane-contiguous experts; returns this lane's (local expert, histogram slot)."""
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
                lptr + row + cutlass.Int32(split * _EXPERTS),
                dtype=cutlass.Float32,
                count=4,
            )
            hi = prims.load_ext(
                lptr + row + cutlass.Int32(split * _EXPERTS + 4),
                dtype=cutlass.Float32,
                count=4,
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
        if cutlass.const_expr(one_redux):
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
                            cutlass.Int32(inverse_idx),
                            prims.ReductionKind.UMAX,
                            _FULL_MASK,
                        )
                    )
                    cand = cutlass.Int32(0)
                    if is_max and inverse_idx == winner_inverse_idx:
                        cand = cutlass.Int32(1)
                    ball2 = cutlass.Uint32(
                        prims.vote_sync(_FULL_MASK, cand != cutlass.Int32(0), prims.VoteSync.BALLOT)
                    )
                    owner = cutlass.Int32(cute.arch.popc(ball2 - cutlass.Uint32(1)))
                head_raw = cutlass.Float32(0.0)
                for v in cutlass.range_constexpr(_VEC):
                    if lane * cutlass.Int32(_VEC) + cutlass.Int32(v) == head_idx:
                        head_raw = raw_scores[v]
                winner_idx = cute.arch.shuffle_sync(head_idx, owner)
                winner_raw = cute.arch.shuffle_sync(head_raw, owner)
                if lane == owner:
                    for v in cutlass.range_constexpr(_VEC - 1):
                        keys[v] = keys[v + 1]
                    keys[_VEC - 1] = cutlass.Uint64(0)
                    head_idx = cutlass.Int32(65535) - cutlass.Int32(
                        keys[0] & cutlass.Uint64(0xFFFF)
                    )
                if lane == cutlass.Int32(k):
                    my_raw = winner_raw
                    my_selected_idx = winner_idx
        else:
            for k in cutlass.range_constexpr(_TOP_K):
                score_key = cutlass.Uint32(keys[0] >> cutlass.Uint64(32))
                warp_score_key = cutlass.Uint32(
                    prims.redux_sync(cutlass.Int32(score_key), prims.ReductionKind.UMAX, _FULL_MASK)
                )
                inverse_idx = cutlass.Uint32(0)
                if score_key == warp_score_key:
                    inverse_idx = cutlass.Uint32(keys[0] & cutlass.Uint64(0xFFFF))
                winner_inverse_idx = cutlass.Uint32(
                    prims.redux_sync(
                        cutlass.Int32(inverse_idx), prims.ReductionKind.UMAX, _FULL_MASK
                    )
                )
                winner_idx = cutlass.Int32(65535) - cutlass.Int32(winner_inverse_idx)
                winner_raw_local = cutlass.Float32(0.0)
                if score_key == warp_score_key and inverse_idx == winner_inverse_idx:
                    for v in cutlass.range_constexpr(_VEC):
                        if lane * cutlass.Int32(_VEC) + cutlass.Int32(v) == winner_idx:
                            winner_raw_local = raw_scores[v]
                    for v in cutlass.range_constexpr(_VEC - 1):
                        keys[v] = keys[v + 1]
                    keys[_VEC - 1] = cutlass.Uint64(0)
                winner_raw = cute.arch.shuffle_sync(
                    winner_raw_local, winner_idx >> cutlass.Int32(3)
                )
                if lane == cutlass.Int32(k):
                    my_raw = winner_raw
                    my_selected_idx = winner_idx
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
        if local >= cutlass.Int32(0) and local < cutlass.Int32(_LOCAL_EXPERTS):
            my_local = local
            my_slot = prims.atomicrmw("add", expert_count.subview(local), 1)
    return my_local, my_slot


@cute.kernel
def fused_topk_routing_push_kernel(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    zero_output: cute.Tensor,
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
    CLUSTER_CTAS: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    ONE_REDUX: cutlass.Constexpr,
    expert_hist: Optional[cute.Tensor] = None,
):
    """Cluster path: one CTA owns ``_WARPS`` tokens, one warp owns a token; push-based hand-off (module doc)."""
    tidx, _, _ = cute.arch.thread_idx()
    rank, _, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)
    count = cutlass.Array(
        cutlass.Int32, _LOCAL_EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    all_counts = cutlass.Array(
        cutlass.Int32,
        CLUSTER_CTAS * _LOCAL_EXPERTS,
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
            prims.mbarrier_arrive_expect_tx(count_bar.subview(0), CLUSTER_CTAS * _LOCAL_EXPERTS * 4)
        prims.mbarrier_arrive_expect_tx(base_bar.subview(0), _LOCAL_EXPERTS * 4)
    # Router-independent: this lane's 8 contiguous bias values, before the PDL wait.
    bias_vals = cutlass.Array(cutlass.Float32, _VEC)
    for v in cutlass.range_constexpr(_VEC):
        bias_vals[v] = cutlass.Float32(routing_bias[lane * cutlass.Int32(_VEC) + cutlass.Int32(v)])
    prims.barrier_cta_sync(0)
    # Orders every CTA's barrier initialisation before any peer's st.async complete_tx (waited below).
    prims.barrier_cluster_arrive()
    token = rank * cutlass.Int32(_WARPS) + warp
    valid = token < num_tokens
    if cutlass.const_expr(WAIT_FOR_ROUTER_PDL):
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SIGNAL_GG1_PDL):
        cute.arch.griddepcontrol_launch_dependents()
    my_local, my_slot = _select_topk_lanevec(
        logits,
        LOGIT_SPLITS,
        ONE_REDUX,
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
    prims.barrier_cluster_wait()  # every CTA initialised its barriers (it arrived long before)
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
        rank_prefix = cutlass.Array(cutlass.Int32, CLUSTER_CTAS)
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(CLUSTER_CTAS):
            rank_prefix[r] = total
            total = total + all_counts.subview(cutlass.Int32(r * _LOCAL_EXPERTS) + tidx).load()
        active = total != cutlass.Int32(0)
        local_prefix, local_total = _base._ballot_exclusive(active)
        if lane == cutlass.Int32(_WARP - 1):
            warp_totals.subview(warp).store(local_total)
        prims.barrier_cta_sync(1, thread_count=_LOCAL_EXPERTS)
        first_warp_total = warp_totals.subview(0).load()
        tile = local_prefix
        if warp == cutlass.Int32(1):
            tile = tile + first_warp_total
        expert_base = tile * cutlass.Int32(_TILE)
        for r in cutlass.range_constexpr(CLUSTER_CTAS):
            _compat48.store_async(
                prims.mapa(base, cutlass.Int32(r)).subview(tidx),
                expert_base + rank_prefix[r],
                prims.mapa(base_bar.subview(0), cutlass.Int32(r)),
            )
        if active:
            tile_idx_to_expert_idx[tile] = tidx
            tile_idx_to_mn_limit[tile] = expert_base + total
        if cutlass.const_expr(expert_hist is not None):
            # the decode MoE L2 prefetch: this step's count of local expert `tidx` (zero included)
            expert_hist[tidx] = total
        if tidx == cutlass.Int32(0):
            active_experts = first_warp_total + warp_totals.subview(1).load()
            num_non_exiting_tiles[0] = active_experts
            total_num_padded_tokens[0] = active_experts * cutlass.Int32(_TILE)
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


fused_topk_routing_push_kernel.set_name_prefix(
    "glm52_moe_topk_routing_push", remove_cutlass_symbol=True
)


@cute.jit
def launch_fused_topk_routing_push(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    zero_output: cute.Tensor,
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
    CLUSTER_CTAS: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    ONE_REDUX: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
    expert_hist: Optional[cute.Tensor] = None,
):
    """Launch the push cluster path (same runtime arguments as ``moe_topk_routing``'s launcher,
    plus ``expert_hist`` last when the step's per-expert counts are also written)."""
    fused_topk_routing_push_kernel(
        logits,
        routing_bias,
        zero_output,
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
        CLUSTER_CTAS,
        LOGIT_SPLITS,
        WAIT_FOR_ROUTER_PDL,
        SIGNAL_GG1_PDL,
        ONE_REDUX,
        expert_hist,
    ).launch(
        grid=(CLUSTER_CTAS, 1, 1),
        block=(_BLOCK, 1, 1),
        cluster=(CLUSTER_CTAS, 1, 1),
        use_pdl=WAIT_FOR_ROUTER_PDL,
        stream=stream,
    )


def uses_push_path(num_tokens: int, fuse_zero: bool = False) -> bool:
    """Whether this module's own kernel serves ``num_tokens`` (else ``moe_topk_routing``'s does)."""
    return not fuse_zero and cluster_ctas_for_tokens(num_tokens) > 1


def compile_fused_topk_routing(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    zero_output: Optional["torch.Tensor"],
    outputs: Dict[str, "torch.Tensor"],
    *,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    fuse_zero: bool = False,
    expert_hist: Optional["torch.Tensor"] = None,
) -> TopKRoutingFn:
    """``moe_topk_routing.compile_fused_topk_routing`` with the push cluster path.

    ``expert_hist`` (push path only): compile the form that also writes the
    per-local-expert counts; its handle then takes the int32 ``[64]`` tensor
    after the stream.

    The block path (<= warps-per-CTA tokens) and fused output zeroing stay on
    ``moe_topk_routing``. The returned handle takes the same runtime arguments
    either way: ``(logits, bias, zero_or_logits, *outputs, num_tokens,
    local_expert_offset, routed_scaling_factor, stream)``.
    """
    device, num_tokens, logit_splits = _base._validate_serving_inputs(
        logits, routing_bias, zero_output, outputs, fuse_zero
    )
    if not uses_push_path(num_tokens, fuse_zero):
        if expert_hist is not None:
            raise ValueError("expert_hist is written by the push cluster path only (9..64 tokens)")
        return _base.compile_fused_topk_routing(
            logits,
            routing_bias,
            zero_output,
            outputs,
            wait_for_router_pdl=wait_for_router_pdl,
            signal_gg1_pdl=signal_gg1_pdl,
            fuse_zero=fuse_zero,
        )
    _check_hist(expert_hist, logits.device)
    if logits.data_ptr() % 16:
        raise ValueError("push top-k loads 128-bit logits vectors: logits must be 16-byte aligned")
    key = (
        device,
        num_tokens,
        logit_splits,
        wait_for_router_pdl,
        signal_gg1_pdl,
        _ONE_REDUX,
    ) + ((True,) if expert_hist is not None else ())
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        cluster_ctas = cluster_ctas_for_tokens(num_tokens)
        tensor_args = (
            _base._as_tensor(logits.reshape(-1), assumed_align=16),
            _base._as_tensor(routing_bias),
            _base._as_tensor(logits.reshape(-1), assumed_align=16),
            *(_base._as_tensor(outputs[name]) for name in OUTPUT_ORDER),
        )
        _CACHE[key] = cute.compile(
            launch_fused_topk_routing_push,
            *tensor_args,
            cutlass.Int32(num_tokens),
            cutlass.Int32(0),
            cutlass.Float32(1.0),
            cluster_ctas,
            logit_splits,
            wait_for_router_pdl,
            signal_gg1_pdl,
            _ONE_REDUX,
            cutlass_torch.current_stream(),
            *(
                (_base._as_tensor(expert_hist, assumed_align=16),)
                if expert_hist is not None
                else ()
            ),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def _check_hist(expert_hist: Optional["torch.Tensor"], device) -> None:
    if expert_hist is None:
        return
    if (
        expert_hist.dtype != torch.int32
        or tuple(expert_hist.shape) != (_LOCAL_EXPERTS,)
        or not expert_hist.is_contiguous()
        or expert_hist.data_ptr() % 16
        or expert_hist.device != device
    ):
        raise ValueError(
            f"expert_hist must be a contiguous 16-byte aligned int32 [{_LOCAL_EXPERTS}] tensor on the logits' device"
        )


def run_fused_topk_routing(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    zero_output: Optional["torch.Tensor"],
    outputs: Dict[str, "torch.Tensor"],
    *,
    local_expert_offset: int,
    routed_scaling_factor: float,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    fuse_zero: bool = False,
    expert_hist: Optional["torch.Tensor"] = None,
) -> None:
    """``moe_topk_routing.run_fused_topk_routing`` with the push cluster path.

    ``expert_hist``: also write the step's per-local-expert counts (push path only;
    ``None`` = the kernel without it)."""
    num_tokens = int(logits.shape[0])
    if not uses_push_path(num_tokens, fuse_zero):
        if expert_hist is not None:
            raise ValueError("expert_hist is written by the push cluster path only (9..64 tokens)")
        _base.run_fused_topk_routing(
            logits,
            routing_bias,
            zero_output,
            outputs,
            local_expert_offset=local_expert_offset,
            routed_scaling_factor=routed_scaling_factor,
            wait_for_router_pdl=wait_for_router_pdl,
            signal_gg1_pdl=signal_gg1_pdl,
            fuse_zero=fuse_zero,
        )
        return
    if local_expert_offset not in (0, 64, 128, 192):
        raise ValueError(
            f"local_expert_offset must select one contiguous EP4 shard, got {local_expert_offset}"
        )
    compiled = compile_fused_topk_routing(
        logits,
        routing_bias,
        zero_output,
        outputs,
        wait_for_router_pdl=wait_for_router_pdl,
        signal_gg1_pdl=signal_gg1_pdl,
        fuse_zero=fuse_zero,
        expert_hist=expert_hist,
    )
    compiled(
        _base._as_tensor(logits.reshape(-1), assumed_align=16),
        _base._as_tensor(routing_bias),
        _base._as_tensor(logits.reshape(-1), assumed_align=16),
        *(_base._as_tensor(outputs[name]) for name in OUTPUT_ORDER),
        cutlass.Int32(num_tokens),
        cutlass.Int32(local_expert_offset),
        cutlass.Float32(routed_scaling_factor),
        cutlass_torch.current_stream(),
        *((_base._as_tensor(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
    )
