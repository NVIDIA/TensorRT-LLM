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
#: The portable cluster: every CTA of it holds ``WARPS`` tokens.
_CLUSTER_CTAS = 8
#: Token-warps per CTA, by row range: 65..128, 129..192, 193..256.
WARP_COUNTS = (16, 24, 32)
MIN_TOKENS = _base._MAX_TOKENS + 1
MAX_TOKENS = _CLUSTER_CTAS * WARP_COUNTS[-1]
assert MAX_TOKENS <= 2 * _TILE, "an expert takes at most two tiles"

OUTPUT_ORDER = _push.OUTPUT_ORDER

TopKRoutingFn = Callable[..., None]
_CACHE: Dict[Tuple[int, int, int, bool, bool, bool], TopKRoutingFn] = {}


def warps_for_tokens(num_tokens: int) -> int:
    """Token-warps per CTA for ``num_tokens`` (65..256): 16, 24 or 32."""
    if not MIN_TOKENS <= num_tokens <= MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [{MIN_TOKENS}, {MAX_TOKENS}], got {num_tokens}")
    return 8 * math.ceil(num_tokens / (8 * _CLUSTER_CTAS))


def max_tiles(num_tokens: int) -> int:
    """The tile records ``num_tokens`` rows can produce: one per active local expert (at most
    64) plus one per expert past 128 rows."""
    expanded = int(num_tokens) * _TOP_K
    filled = min(_LOCAL_EXPERTS, expanded)
    return filled + (expanded - filled) // _TILE


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
    expert_hist: Optional[cute.Tensor] = None,
):
    """One CTA owns ``WARPS`` tokens, one warp owns a token; push hand-off (module doc)."""
    tidx, _, _ = cute.arch.thread_idx()
    rank, _, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)
    count = cutlass.Array(
        cutlass.Int32, _LOCAL_EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    all_counts = cutlass.Array(
        cutlass.Int32,
        _CLUSTER_CTAS * _LOCAL_EXPERTS,
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
            prims.mbarrier_arrive_expect_tx(
                count_bar.subview(0), _CLUSTER_CTAS * _LOCAL_EXPERTS * 4
            )
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
        rank_prefix = cutlass.Array(cutlass.Int32, _CLUSTER_CTAS)
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(_CLUSTER_CTAS):
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
        for r in cutlass.range_constexpr(_CLUSTER_CTAS):
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
    stream: cuda_driver.CUstream,
    expert_hist: Optional[cute.Tensor] = None,
):
    """Launch the 8-CTA cluster with ``WARPS`` token-warps per CTA."""
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
        expert_hist,
    ).launch(
        grid=(_CLUSTER_CTAS, 1, 1),
        block=(WARPS * _WARP, 1, 1),
        cluster=(_CLUSTER_CTAS, 1, 1),
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
) -> TopKRoutingFn:
    """Compile (once per device / token-warp count / logit planes / PDL roles / counts) the
    handle for ``logits``' row range (``warps_for_tokens``). It takes ``(logits, bias,
    *outputs in OUTPUT_ORDER, num_tokens, local_expert_offset, routed_scaling_factor,
    stream[, expert_hist])``; any row count of the same range runs on it."""
    device, num_tokens, logit_splits = _validate(logits, routing_bias, outputs)
    _push._check_hist(expert_hist, logits.device)
    warps = warps_for_tokens(num_tokens)
    key = (device, warps, logit_splits, bool(wait_for_router_pdl), bool(signal_gg1_pdl)) + (
        (True,) if expert_hist is not None else ()
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
