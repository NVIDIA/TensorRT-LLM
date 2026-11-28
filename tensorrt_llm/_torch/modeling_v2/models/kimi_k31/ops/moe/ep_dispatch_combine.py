# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed-MoE token exchange under attention DP (pure EP over the attention-DP group).

`K31EpAllToAll` is the one-sided NVLink all-to-all: the ``torch.ops.trtllm.moe_a2a_*`` kernels on one MNNVL
workspace that every MoE layer of the process shares. `dispatch` sends each token's payload (the MXFP8 latent
``[T, 3584]``, its UE8M0 scales ``[T, 112]``, int32 expert ids ``[T, 16]`` and BF16 routing weights ``[T, 16]``)
to the ranks owning one of its experts and returns this rank's receive views ``[ep * T_max, ...]``; `combine` sums
the MoE output rows ``[ep * T_max, 3584]`` back onto this rank's tokens. The workspace layout and size, the kernels
and their arguments are the engine one-sided strategy's for K3.1. A step takes the sentinel transport (receivers
poll the data itself) when every EP rank can run it and the busiest rank's tokens fit its token limit, else the fence
transport (completion flags behind a system-scope fence); the choice is the engine strategy's "auto", with its
default token limit, and reads no environment.

`K31EpAllGather` is the engine's AllGather + ReduceScatter exchange: the rows of every rank gathered in rank order
(one grouped NCCL all-gather of the four payloads), and the MoE output summed across ranks with each rank keeping
its own rows. `create_ep_all_to_all` picks between the two: the one-sided all-to-all wherever it can be built, else
the all-gather (MPI session; routing refuses attention DP under Ray).
"""

from __future__ import annotations

import os
from typing import List, NamedTuple, Optional, Sequence, Tuple

import pynvml
import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import register_moe_comm_timeout_proxy
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: The receive row of the expert ids (payload order: activation, scales, expert ids, routing weights).
EXPERT_ID_PAYLOAD_INDEX = 2
#: Expert id of a receive row this rank does not serve (the TRTLLM-Gen runner skips -1 only).
INVALID_EXPERT_ID = -1

_MAX_RANKS = int(_tllm_internal.thop.MOE_A2A_MAX_RANKS)
_WORKSPACE_SIZE_INDEX = int(_tllm_internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)
_RECV_COUNTERS_OFFSET_INDEX = int(_tllm_internal.thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX)
_COMBINE_INPUT_OFFSET_INDEX = int(_tllm_internal.thop.MOE_A2A_COMBINE_INPUT_OFFSET_INDEX)

#: The sentinel payload format of the K3.1 exchange: the MXFP8 latent with its UE8M0 scales.
SENTINEL_PAYLOAD_FORMAT = "MXFP8"
#: The engine strategy's default sentinel token limit: never below this many tokens per rank (its GB300 measurements
#: found the sentinel transport faster up to here in every configuration), ...
_SENTINEL_MIN_DEFAULT_MAX_TOKENS = 159
#: ... and above it, the ``ep_size * limit`` rows a sentinel step polls kept within this many bytes per rank.
_SENTINEL_MAX_DEFAULT_REGION_BYTES = 64 << 20


def _per_rank(
    stands_for: str, counts: Sequence[int], why: str, real: Optional[Sequence[int]] = None
) -> Cell:
    """``counts``: the per-rank counts the exchange is given; ``real``: with DP padding, each rank's own tokens
    (the rest of its ``counts`` rows are the scheduler's zero padding)."""
    spec = dict(
        world=len(counts), all_rank_num_tokens=tuple(counts), use_dp_padding=real is not None
    )
    if real is not None:
        spec["real_num_tokens"] = tuple(real)
    return Cell(why=f"{stands_for}: {why}", spec=spec)


#: Per-rank token counts of one exchange. Four ranks (one GB300 tray) stand for DEP16: the kernels' per-row work
#: does not depend on the group size, and the 896 experts split evenly at both (224 / 56 per rank).
_EXCHANGE_CELLS: tuple[Cell, ...] = (
    _per_rank("DEP16 decode", (1, 1, 1, 1), "one token per rank (a sentinel step)"),
    _per_rank("DEP16 decode", (64, 64, 64, 64), "a 64-token step"),
    _per_rank(
        "DEP16 above the fused-EP cap",
        (257, 257, 257, 257),
        "first per-rank count past 256 (a fence step)",
    ),
    _per_rank("DEP16 prefill", (2048, 2048, 2048, 2048), "a 2048-token MoE chunk per rank"),
    _per_rank("DEP16 uneven", (37, 5, 64, 1), "ragged per-rank counts, no padding"),
    _per_rank(
        "DEP16 idle rank",
        (300, 300, 300, 300),
        "one rank without tokens, every rank padded to the largest count",
        real=(300, 0, 211, 17),
    ),
)


# ----------------------------------------------------------------------------------------------------------------
# Kernel entries
# ----------------------------------------------------------------------------------------------------------------


class MoeA2ADispatch(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_dispatch` without EPLB statistics or a rank mask.

    Each token of this rank goes once to every rank owning one of its experts (expert ``e`` lives on rank
    ``e // (num_experts / ep_size)``); the payload rows land in the destination's receive block of this source rank
    at the slot the destination's send counter hands out. Returns the per-payload receive views
    ``[ep_size, runtime_max_tokens_per_rank, cols]`` of the workspace, the combine-payload offset and an empty EPLB
    tensor. On a sentinel step (``use_sentinel``) the rows are not compacted: row ``[src, t]`` holds token ``t`` of
    rank ``src``, and the kernel writes ``invalid_token_expert_id`` into the expert ids (``expert_id_payload_index``)
    of the rows without a token routed here; on a fence step `MoeA2ASanitizeExpertIds` does that for the rows past
    each source block's count.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _EXCHANGE_CELLS
    note = """
    The row order inside one source block follows the order in which the source's CTAs claim destination slots
    (an atomic counter per destination), so it can differ between two calls with the same input; the set of rows of
    a block and the combine result do not. Rows past a block's count keep whatever the workspace held.
    """

    def __call__(
        self,
        token_selected_experts: torch.Tensor,
        payloads: List[torch.Tensor],
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        num_experts: int,
        use_sentinel: bool,
        expert_id_payload_index: Optional[int],
        invalid_token_expert_id: Optional[int],
    ) -> Tuple[List[torch.Tensor], int, torch.Tensor]:
        return torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            payloads,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            ep_size,
            top_k,
            num_experts,
            eplb_local_stats=None,
            expert_id_payload_index=expert_id_payload_index,
            invalid_token_expert_id=invalid_token_expert_id,
            enable_rank_mask=False,
            active_rank_mask=None,
            use_sentinel=use_sentinel,
        )

    def reference(
        self,
        payloads_in_rank_order: Sequence[Sequence[torch.Tensor]],
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ) -> List[List[torch.Tensor]]:
        """Per source rank, its payload rows this rank receives, in the source's token order.

        Collective: takes every rank's payloads (expert ids at `EXPERT_ID_PAYLOAD_INDEX`) in rank order. A fence step
        stores these rows first in the source's block, in slot-claim order (see `note`); a sentinel step at the
        token's own row of the block.
        """
        per_rank = num_experts // ep_size
        received = []
        for payloads in payloads_in_rank_order:
            ids = payloads[EXPERT_ID_PAYLOAD_INDEX]
            here = ((ids >= 0) & (torch.div(ids, per_rank, rounding_mode="floor") == ep_rank)).any(
                1
            )
            received.append([p[here] for p in payloads])
        return received


class MoeA2ASanitizeExpertIds(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_sanitize_expert_ids`: ``invalid_expert_id`` into every expert id of the receive rows
    past each source block's count (the counts of the last dispatch, read from the workspace)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _EXCHANGE_CELLS

    def __call__(
        self,
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        invalid_expert_id: int,
    ) -> None:
        torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
            expert_ids, workspace, metainfo, ep_rank, invalid_expert_id
        )

    def reference(
        self,
        expert_ids: torch.Tensor,
        recv_counts: Sequence[int],
        invalid_expert_id: int,
    ) -> torch.Tensor:
        """``expert_ids [ep_size, T, top_k]`` with the rows from ``recv_counts[s]`` on of block ``s`` invalid (the
        counts come from the workspace in the kernel)."""
        out = expert_ids.clone()
        for s, count in enumerate(recv_counts):
            out[s, count:] = invalid_expert_id
        return out


class MoeA2ACombine(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_combine` without a rank mask: each of this rank's ``local_num_tokens`` tokens sums
    the MoE output rows of its distinct destination ranks (from the routing the last dispatch stored).

    ``payload [ep_size, runtime_max_tokens_per_rank, hidden]`` is copied into the workspace first, unless
    ``payload_in_workspace`` (it is the `MoeA2AGetCombinePayloadTensor` view, already in place). Returns
    ``[local_num_tokens, hidden]`` in the payload dtype. A sentinel step's combine (``use_sentinel``, after a sentinel
    dispatch with the same token counts) reads the BF16 payload in place and receives each row as it lands.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _EXCHANGE_CELLS

    def __call__(
        self,
        payload: torch.Tensor,
        local_num_tokens: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool,
        use_sentinel: bool,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_combine(
            payload,
            local_num_tokens,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            ep_size,
            top_k,
            combine_payload_offset,
            payload_in_workspace,
            use_low_precision=use_low_precision,
            enable_rank_mask=False,
            active_rank_mask=None,
            use_sentinel=use_sentinel,
        )

    def reference(
        self,
        contributions: torch.Tensor,
        expert_ids: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """The sum, in FP32 rounded once, over each token's distinct destination ranks of their MoE output row.

        Collective: ``contributions [ep_size, n, hidden]`` holds, for each destination rank, the MoE output row it
        computed for each of this rank's ``n`` tokens (``expert_ids [n, top_k]``). The kernel's own sum order
        (first-occurrence ranks, pairwise) can differ from this one in the last bit.
        """
        ep_size = contributions.shape[0]
        ranks = torch.div(expert_ids.long(), num_experts // ep_size, rounding_mode="floor")
        hit = torch.stack([(ranks == r).any(1) for r in range(ep_size)])
        return (contributions.float() * hit.unsqueeze(-1)).sum(0).to(contributions.dtype)


class MoeA2AGetCombinePayloadTensor(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_get_combine_payload_tensor`: the view ``[ep_size * runtime_max_tokens_per_rank,
    hidden_size]`` of this rank's combine-input region of the workspace, for the MoE output to be written in place
    (`MoeA2ACombine` with ``payload_in_workspace`` then skips its copy)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = tuple(
        Cell(
            why=f"{why}: the BF16 MoE output of {t} rows per rank",
            spec=dict(world=4, runtime_max=t),
        )
        for t, why in (
            (1, "decode"),
            (300, "a step with an idle rank, padded"),
            (4096, "the workspace's capacity"),
        )
    )

    def __call__(
        self,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        runtime_max_tokens_per_rank: int,
        dtype: torch.dtype,
        hidden_size: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            workspace, metainfo, ep_rank, ep_size, runtime_max_tokens_per_rank, dtype, hidden_size
        )

    def reference(
        self,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        runtime_max_tokens_per_rank: int,
        dtype: torch.dtype,
        hidden_size: int,
    ) -> torch.Tensor:
        """The same bytes of this rank's workspace, from the layout's combine-input offset."""
        start = int(metainfo[_COMBINE_INPUT_OFFSET_INDEX])
        rows = ep_size * runtime_max_tokens_per_rank
        nbytes = rows * hidden_size * torch.empty((), dtype=dtype).element_size()
        return workspace[ep_rank, start : start + nbytes].view(dtype).view(rows, hidden_size)


class MoeA2AInitialize(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_initialize`: checks the planned layout and resets this rank's control region
    (flags and counters) of the workspace, then synchronizes the EP group (collective). Returns nothing; the result is
    this rank's workspace bytes up to the layout's size (see `reference`). A layout with sentinel regions
    (`moe_a2a_sentinel_extend_workspace_layout`) also has them armed with the transport's sentinel words."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="the workspace of the K3.1 exchange over one tray (4 ranks, 4096 tokens per rank)",
            spec=dict(world=4, max_tokens=4096),
        ),
    )

    def __call__(
        self, workspace: torch.Tensor, metainfo: torch.Tensor, ep_rank: int, ep_size: int
    ) -> None:
        torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, ep_rank, ep_size)

    def reference(
        self, workspace: torch.Tensor, metainfo: torch.Tensor, ep_rank: int, ep_size: int
    ) -> torch.Tensor:
        """This rank's workspace bytes after the call, for a layout without sentinel regions: zero up to the layout's
        size, except the receive counters (``2 x ep_size`` int32 at their offset), set to all ones (-1)."""
        size = int(metainfo[_WORKSPACE_SIZE_INDEX])
        expected = torch.zeros(size, dtype=torch.uint8, device=workspace.device)
        start = int(metainfo[_RECV_COUNTERS_OFFSET_INDEX])
        expected[start : start + 2 * ep_size * 4] = 0xFF
        return expected


#: The native layout planner's alignments: payload regions and cache lines.
_WORKSPACE_ALIGNMENT = int(_tllm_internal.thop.MOE_A2A_WORKSPACE_ALIGNMENT)
_CACHELINE_ALIGNMENT = 128


def _layout_index(name: str) -> int:
    return int(getattr(_tllm_internal.thop, f"MOE_A2A_{name}_INDEX"))


class MoeA2AGetWorkspaceLayout(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_get_workspace_layout` (host): the int64 metainfo of a workspace with these region
    sizes (control buffers, then the dispatch and combine-input regions), its total size per rank at
    ``MOE_A2A_WORKSPACE_SIZE_INDEX``; no sentinel regions (`moe_a2a_sentinel_extend_workspace_layout` appends them)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = tuple(
        Cell(
            why=f"K3.1 exchange over {ep} ranks, {tokens} tokens per rank",
            spec=dict(ep=ep, max_tokens=tokens),
        )
        for ep, tokens in ((16, 8192), (4, 4096))
    )

    def __call__(
        self,
        ep_size: int,
        max_num_tokens_per_rank: int,
        top_k: int,
        dispatch_payload_bytes: int,
        combine_input_bytes: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size,
            max_num_tokens_per_rank,
            top_k,
            dispatch_payload_bytes,
            combine_input_bytes,
            None,
        )

    def reference(
        self,
        ep_size: int,
        max_num_tokens_per_rank: int,
        top_k: int,
        dispatch_payload_bytes: int,
        combine_input_bytes: int,
    ) -> torch.Tensor:
        """The planner's layout: the regions in order, each at its alignment, the payload capacities rounded up to
        the workspace alignment (no EPLB statistics, no sentinel regions)."""

        def up(value: int, alignment: int) -> int:
            return (value + alignment - 1) // alignment * alignment

        fields = [0] * int(_tllm_internal.thop.MOE_A2A_NUM_METAINFO_FIELDS)
        offset = 0

        def reserve(name: str, nbytes: int, alignment: int = 4) -> None:
            nonlocal offset
            offset = up(offset, alignment)
            fields[_layout_index(f"{name}_OFFSET")] = offset
            offset += nbytes

        dispatch_bytes = up(dispatch_payload_bytes, _WORKSPACE_ALIGNMENT)
        combine_bytes = up(combine_input_bytes, _WORKSPACE_ALIGNMENT)
        rank_counts = ep_size * 4
        routes = max_num_tokens_per_rank * top_k * 4
        reserve("FLAG_VAL", 4)
        reserve("LOCAL_TOKEN_COUNTER", 4)
        reserve("SEND_COUNTERS", rank_counts)
        reserve("RECV_COUNTERS", 2 * rank_counts)
        reserve("DISPATCH_COMPLETION_FLAGS", rank_counts, _CACHELINE_ALIGNMENT)
        reserve("TOPK_TARGET_RANKS", routes, _CACHELINE_ALIGNMENT)
        reserve("TOPK_TARGET_INDICES", routes, _CACHELINE_ALIGNMENT)
        reserve("EPLB_GATHERED_STATS", 0, _CACHELINE_ALIGNMENT)
        reserve("DISPATCH_PAYLOAD", dispatch_bytes, _WORKSPACE_ALIGNMENT)
        fields[_layout_index("DISPATCH_PAYLOAD_SIZE")] = dispatch_bytes
        reserve("COMBINE_COMPLETION_FLAGS", rank_counts, _CACHELINE_ALIGNMENT)
        reserve("COMBINE_INPUT", combine_bytes, _WORKSPACE_ALIGNMENT)
        fields[_layout_index("COMBINE_INPUT_SIZE")] = combine_bytes
        fields[_layout_index("MAX_NUM_TOKENS")] = max_num_tokens_per_rank
        fields[_layout_index("TOP_K")] = top_k
        fields[_layout_index("EP_SIZE")] = ep_size
        fields[_layout_index("EPLB_STATS_NUM_EXPERTS")] = 0
        fields[_layout_index("WORKSPACE_SIZE")] = offset
        return torch.tensor(fields, dtype=torch.int64, device="cpu")


def moe_a2a_sentinel_extend_workspace_layout(
    metainfo: torch.Tensor, sentinel_max_tokens: int, payload_format: str, hidden_size: int
) -> None:
    """Appends the sentinel transport's regions (round state, landing, combine landing and receive rows for
    ``sentinel_max_tokens`` tokens per rank of ``payload_format`` rows) to a planner layout, in place (host). Host
    setup with nothing to compute."""
    torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
        metainfo, sentinel_max_tokens, _sentinel_format_id(payload_format), hidden_size
    )


def moe_a2a_sentinel_row_geometry(payload_format: str, hidden_size: int, top_k: int) -> List[int]:
    """(hidden-state row bytes, scale row bytes, dispatch blocks per token slot, workspace bytes per (source rank,
    token) row) of a sentinel payload format (host); raises for a hidden size the kernels do not take. Host setup with
    nothing to compute."""
    return list(
        torch.ops.trtllm.moe_a2a_sentinel_row_geometry(
            _sentinel_format_id(payload_format), hidden_size, top_k
        )
    )


def moe_a2a_sentinel_available() -> bool:
    """Whether the loaded kernels carry the sentinel transport for this GPU (host). Host setup with nothing to
    compute."""
    return bool(torch.ops.trtllm.moe_a2a_sentinel_available())


def _sentinel_format_id(payload_format: str) -> int:
    return int(getattr(_tllm_internal.thop, f"MOE_A2A_SENTINEL_FORMAT_{payload_format}"))


def moe_a2a_set_timeout(timeout_sec: int) -> None:
    """The in-kernel peer-wait timeout of the all-to-all launches issued after it, process-wide (host); 0 restores
    the native default. Host setup with nothing to compute."""
    torch.ops.trtllm.moe_a2a_set_timeout(timeout_sec)


class AllgatherList(OpWrapper):
    """`torch.ops.trtllm.allgather_list` (MPI session): every rank's rows of each tensor concatenated in rank order,
    all tensors in one NCCL group. ``sizes`` is ``None`` when every rank holds the same row count, else each rank's
    row count in rank order. Inputs ``[rows, cols]`` contiguous; returns ``[sum(rows), cols]`` per tensor."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _EXCHANGE_CELLS

    def __call__(
        self, inputs: List[torch.Tensor], sizes: Optional[List[int]], group: List[int]
    ) -> List[torch.Tensor]:
        return torch.ops.trtllm.allgather_list(inputs, sizes, group)

    def reference(
        self, inputs_in_rank_order: Sequence[Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Collective: takes every rank's tensors in rank order."""
        return [torch.cat(list(column), dim=0) for column in zip(*inputs_in_rank_order)]


moe_a2a_dispatch = MoeA2ADispatch()
moe_a2a_sanitize_expert_ids = MoeA2ASanitizeExpertIds()
moe_a2a_combine = MoeA2ACombine()
moe_a2a_get_combine_payload_tensor = MoeA2AGetCombinePayloadTensor()
moe_a2a_initialize = MoeA2AInitialize()
moe_a2a_get_workspace_layout = MoeA2AGetWorkspaceLayout()
allgather_list = AllgatherList()


class _OneSidedTimeoutProxy:
    """The engine MoE communication timeout guard's handle on the all-to-all kernels' process-wide peer-wait
    timeout: the guard sets the serving budget at registration and the warmup budget around the engine's warmup."""

    name = "NVLinkOneSided"

    def set_timeout_seconds(self, seconds: Optional[int]) -> None:
        # The op restores its native default for 0.
        moe_a2a_set_timeout(0 if seconds is None else seconds)


_TIMEOUT_PROXY = _OneSidedTimeoutProxy()


def _local_one_sided_mismatch(mapping: Mapping) -> Optional[str]:
    """Why the one-sided all-to-all cannot serve this EP group, or None: the platform has no NVLink fabric with every
    link up (the engine strategy's ``is_platform_supported`` probe; a failing NVML query counts as none), or the group
    is not one its kernels take (pure EP, at most ``_MAX_RANKS`` ranks)."""
    try:
        fabric = MnnvlMemory.supports_mnnvl()
    except pynvml.NVMLError as e:
        return f"the NVLink fabric probe failed: {e!r}"
    if not fabric:
        return "no NVLink fabric with every link up"
    if mapping.world_size != mapping.moe_ep_size:
        return "it serves pure EP only"
    if mapping.moe_ep_size > _MAX_RANKS:
        return f"it serves at most {_MAX_RANKS} EP ranks"
    return None


def _one_sided_unavailable(mapping: Mapping) -> Optional[str]:
    """Agree on the exchange before any rank enters its setup collectives."""
    reasons = MnnvlMemory.get_comm(mapping).allgather(_local_one_sided_mismatch(mapping))
    return next((reason for reason in reasons if reason is not None), None)


# ----------------------------------------------------------------------------------------------------------------
# The sentinel transport: the engine strategy's "auto" selection
# ----------------------------------------------------------------------------------------------------------------


class SentinelRow(NamedTuple):
    """Row geometry of a sentinel payload format, from the native op."""

    hidden_bytes: int  # one token's hidden states
    scale_bytes: int  # one token's scales (0 for BF16)
    blocks_per_token: int  # dispatch blocks per token slot
    bytes_per_row: int  # sentinel workspace bytes per (source rank, token) row


def sentinel_row(payload_format: str, hidden_size: int, top_k: int) -> SentinelRow:
    """Raises if the kernels do not take rows of ``hidden_size`` in ``payload_format``."""
    return SentinelRow(*map(int, moe_a2a_sentinel_row_geometry(payload_format, hidden_size, top_k)))


def default_sentinel_max_tokens(ep_size: int, num_sms: int, row: SentinelRow) -> int:
    """The sentinel token limit: at least ``_SENTINEL_MIN_DEFAULT_MAX_TOKENS``; else the largest step whose
    dispatch grid stays below one full wave of resident blocks, at most the kernels' own limit, lowered while the
    sentinel regions (``ep_size * limit`` rows) would exceed ``_SENTINEL_MAX_DEFAULT_REGION_BYTES`` per rank. The
    caller caps it by the workspace's tokens per rank."""
    thop = _tllm_internal.thop
    below_one_wave = (
        int(thop.MOE_A2A_SENTINEL_DISPATCH_BLOCKS_PER_SM) * num_sms - 1
    ) // row.blocks_per_token
    by_region_bytes = _SENTINEL_MAX_DEFAULT_REGION_BYTES // (ep_size * row.bytes_per_row)
    return max(
        min(below_one_wave, int(thop.MOE_A2A_SENTINEL_MAX_TOKENS_PER_RANK), by_region_bytes),
        _SENTINEL_MIN_DEFAULT_MAX_TOKENS,
    )


def sentinel_unsupported_reason(
    *,
    ep_size: int,
    top_k: int,
    num_slots: int,
    hidden_size: int,
    dtype: torch.dtype,
    payload_format: str,
    use_low_precision_combine: bool,
) -> Optional[str]:
    """Why this rank cannot use the sentinel transport, or ``None`` if it can. A reason, never an exception, so that
    every rank reaches the collective vote."""
    if not moe_a2a_sentinel_available():
        return "its kernels are not built for this GPU (SM100, SM103 and SM107 only)"
    if ep_size > int(_tllm_internal.thop.MOE_A2A_SENTINEL_MAX_RANKS):
        return f"it supports at most {int(_tllm_internal.thop.MOE_A2A_SENTINEL_MAX_RANKS)} EP ranks"
    if num_slots % ep_size != 0:
        return "it needs the expert slots to split evenly across EP ranks"
    if dtype != torch.bfloat16:
        return "it combines BF16 expert outputs only"
    if use_low_precision_combine:
        return "it has no FP8 combine"
    try:
        sentinel_row(payload_format, hidden_size, top_k)
    except RuntimeError as error:  # the native row check (TORCH_CHECK)
        return f"it does not carry hidden size {hidden_size} in {payload_format}: {error}"
    return None


def _pad128(nbytes: int) -> int:
    return (nbytes + 127) // 128 * 128


def workspace_layout(
    ep_size: int,
    max_num_tokens_per_rank: int,
    top_k: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The workspace metainfo without sentinel regions: the dispatch region holds raw activations or a quantized
    activation with FP32-per-16 scales, plus the expert ids and routing weights (4 bytes per slot each); the combine
    input holds the output at ``max(dtype size, 2)`` bytes."""
    tokens = ep_size * max_num_tokens_per_rank
    activations = _pad128(tokens * hidden_size * dtype.itemsize)
    quantized = _pad128(tokens * hidden_size) + _pad128(tokens * ((hidden_size + 15) // 16) * 4)
    dispatch_bytes = max(activations, quantized) + 2 * _pad128(tokens * top_k * 4)
    return moe_a2a_get_workspace_layout(
        ep_size,
        max_num_tokens_per_rank,
        top_k,
        dispatch_bytes,
        tokens * hidden_size * max(dtype.itemsize, 2),
    )


# ----------------------------------------------------------------------------------------------------------------
# The exchanges
# ----------------------------------------------------------------------------------------------------------------


def _cached_one_sided_mismatch(mapping: Mapping, workspace_cache: dict) -> Optional[str]:
    key = ("availability", tuple(mapping.tp_group))
    if key not in workspace_cache:
        workspace_cache[key] = _one_sided_unavailable(mapping)
    return workspace_cache[key]


class K31EpAllToAll:
    """The one-sided NVLink all-to-all of one MoE layer; the workspace is shared by every layer with the same
    layout (one per process for K3.1)."""

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        hidden_size: int,
        dtype: torch.dtype,
        use_low_precision_combine: bool = False,
        *,
        workspace_cache: Optional[dict] = None,
    ) -> None:
        """The sentinel transport carries the small steps of the rows `dispatch` sends in
        :data:`SENTINEL_PAYLOAD_FORMAT`. Collective over the EP group on the first instance of a layout (the
        sentinel vote and the workspace allocation)."""
        if workspace_cache is None:
            workspace_cache = {}
        if (reason := _cached_one_sided_mismatch(mapping, workspace_cache)) is not None:
            raise RuntimeError(f"the one-sided all-to-all is not available here: {reason}")
        self.mapping = mapping
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        # The completion-flag budget is process-wide, so every instance shares one proxy.
        register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)
        self.num_slots = num_slots
        self.top_k = top_k
        self.use_low_precision_combine = use_low_precision_combine

        metainfo = workspace_layout(
            self.ep_size, max_num_tokens_per_rank, top_k, hidden_size, dtype
        )
        shape = (
            self.ep_rank,
            self.ep_size,
            num_slots,
            top_k,
            max_num_tokens_per_rank,
            hidden_size,
            dtype,
        )
        shape += (use_low_precision_combine, tuple(metainfo.tolist()))
        state = workspace_cache.get(shape)
        if state is None:
            limit, layout = self._vote_sentinel(
                metainfo.clone(),
                max_num_tokens_per_rank,
                hidden_size,
                dtype,
                SENTINEL_PAYLOAD_FORMAT,
            )
            metainfo = torch.tensor(layout, dtype=torch.int64, device="cpu")
            MnnvlMemory.initialize()
            memory = MnnvlMemory(mapping, int(metainfo[_WORKSPACE_SIZE_INDEX]))
            workspace = memory.as_torch_strided_tensor(torch.uint8)
            moe_a2a_initialize(workspace, metainfo, self.ep_rank, self.ep_size)
            state = workspace_cache[shape] = (memory, workspace, metainfo, limit)
        self._memory, self.workspace, self.metainfo, self.sentinel_max_tokens = state
        self._sentinel_row = (
            sentinel_row(SENTINEL_PAYLOAD_FORMAT, hidden_size, top_k)
            if self.sentinel_max_tokens
            else None
        )
        self.workspace_size_per_rank = int(self.metainfo[_WORKSPACE_SIZE_INDEX])
        self.invalid_token_expert_id = INVALID_EXPERT_ID
        # (local token count, runtime max tokens per rank, combine payload offset, sentinel step) of the pending
        # dispatch.
        self._pending: Optional[Tuple[int, int, int, bool]] = None

    def _vote_sentinel(
        self,
        metainfo: torch.Tensor,
        max_num_tokens_per_rank: int,
        hidden_size: int,
        dtype: torch.dtype,
        payload_format: str,
    ) -> Tuple[int, Tuple[int, ...]]:
        """The collective sentinel vote: (token limit, 0 when off; the layout, with the sentinel regions when on).
        Every rank votes its support and SM count; a rank without support turns the transport off everywhere."""
        reason = sentinel_unsupported_reason(
            ep_size=self.ep_size,
            top_k=self.top_k,
            num_slots=self.num_slots,
            hidden_size=hidden_size,
            dtype=dtype,
            payload_format=payload_format,
            use_low_precision_combine=self.use_low_precision_combine,
        )
        num_sms = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            if reason is None
            else None
        )
        votes = MnnvlMemory.get_comm(self.mapping).allgather((reason is None, num_sms))
        if not all(supported for supported, _ in votes):
            message = (
                "Kimi K3.1 MoE exchange: the sentinel transport is off, every step takes the fence transport: "
                f"{reason or 'another EP rank does not support it'}"
            )
            logger.warning_once(message, key=message)
            return 0, tuple(metainfo.tolist())
        row = sentinel_row(payload_format, hidden_size, self.top_k)
        max_tokens = min(
            default_sentinel_max_tokens(self.ep_size, min(sms for _, sms in votes), row),
            max_num_tokens_per_rank,
        )
        moe_a2a_sentinel_extend_workspace_layout(metainfo, max_tokens, payload_format, hidden_size)
        message = (
            f"Kimi K3.1 MoE exchange: the sentinel transport carries {payload_format} steps of up to {max_tokens} "
            "tokens per rank"
        )
        logger.info_once(message, key=message)
        return max_tokens, tuple(metainfo.tolist())

    def _use_sentinel(
        self,
        runtime_max_tokens_per_rank: int,
        hidden_states: torch.Tensor,
        hidden_states_sf: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> bool:
        """Whether the sentinel transport carries this step: few enough tokens on the busiest rank and payloads in
        the voted format (the activation and scale row bytes, byte-sized activation values, 4-byte or BF16 routing
        weights). Depends only on values every rank passes alike, so all ranks take the same transport."""
        row = self._sentinel_row
        if row is None or runtime_max_tokens_per_rank > self.sentinel_max_tokens:
            return False

        def row_bytes(tensor: torch.Tensor) -> int:
            return tensor.shape[1] * tensor.element_size() if tensor.dim() == 2 else -1

        if (row_bytes(hidden_states), row_bytes(hidden_states_sf)) != (
            row.hidden_bytes,
            row.scale_bytes,
        ):
            return False
        if hidden_states.element_size() != 1:
            return False
        return (
            token_final_scales.dim() == 2
            and token_final_scales.shape[1] == self.top_k
            and (
                token_final_scales.element_size() == 4 or token_final_scales.dtype == torch.bfloat16
            )
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: torch.Tensor,
        token_selected_slots: torch.Tensor,
        token_final_scales: torch.Tensor,
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns this rank's receive views ``[ep_size * T, cols]`` (``T = max(all_rank_num_tokens)``) of the
        activation, scales, expert ids (``-1`` on rows without a token routed here) and routing weights.
        ``use_dp_padding`` is not read: padded ranks already pass ``T`` rows."""
        if self._pending is not None:
            raise RuntimeError("dispatch called twice without an intervening combine")
        runtime_max = max(all_rank_num_tokens)
        payloads = [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales]
        use_sentinel = self._use_sentinel(
            runtime_max, hidden_states, hidden_states_sf, token_final_scales
        )
        recv, combine_payload_offset, _ = moe_a2a_dispatch(
            token_selected_slots,
            payloads,
            self.workspace,
            self.metainfo,
            runtime_max,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self.num_slots,
            use_sentinel,
            EXPERT_ID_PAYLOAD_INDEX if use_sentinel else None,
            self.invalid_token_expert_id if use_sentinel else None,
        )
        self._pending = (
            token_selected_slots.size(0),
            runtime_max,
            int(combine_payload_offset),
            use_sentinel,
        )
        if not use_sentinel:
            moe_a2a_sanitize_expert_ids(
                recv[EXPERT_ID_PAYLOAD_INDEX],
                self.workspace,
                self.metainfo,
                self.ep_rank,
                self.invalid_token_expert_id,
            )
        x, x_sf, ids, weights = (r.view(-1, r.shape[-1]) for r in recv)
        return x, x_sf, ids, weights

    def combine_payload(self, hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
        """The workspace view ``[ep_size * T, hidden_size]`` (``T`` of the pending dispatch) the MoE output of the
        dispatched rows can be written into; `combine` of it with ``payload_in_workspace`` then skips the copy."""
        if self._pending is None:
            raise RuntimeError("combine_payload called before a dispatch")
        return moe_a2a_get_combine_payload_tensor(
            self.workspace,
            self.metainfo,
            self.ep_rank,
            self.ep_size,
            self._pending[1],
            dtype,
            hidden_size,
        )

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        all_rank_max_num_tokens: Optional[int] = None,
        payload_in_workspace: bool = False,
    ) -> torch.Tensor:
        """``final_hidden_states [ep_size * T, hidden]`` (the MoE output of the dispatched rows) -> this rank's
        ``[local_num_tokens, hidden]``, over the transport the dispatch took. ``all_rank_max_num_tokens`` is not
        read: ``T`` comes from the dispatch. ``payload_in_workspace``: ``final_hidden_states`` is the
        `combine_payload` view, already in place."""
        if self._pending is None:
            raise RuntimeError("combine called before a dispatch")
        local_num_tokens, runtime_max, combine_payload_offset, use_sentinel = self._pending
        self._pending = None
        payload = final_hidden_states.view(self.ep_size, runtime_max, final_hidden_states.shape[-1])
        return moe_a2a_combine(
            payload,
            local_num_tokens,
            self.workspace,
            self.metainfo,
            runtime_max,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            combine_payload_offset,
            payload_in_workspace,
            self.use_low_precision_combine,
            use_sentinel,
        )


class K31EpAllGather:
    """The AllGather + ReduceScatter exchange of one MoE layer over the attention-DP group (MPI session)."""

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        hidden_size: int,
        dtype: torch.dtype,
        use_low_precision_combine: bool = False,
    ) -> None:
        """Only ``mapping`` is read; the other arguments match `K31EpAllToAll`."""
        self.mapping = mapping
        self._sizes: Optional[List[int]] = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: torch.Tensor,
        token_selected_slots: torch.Tensor,
        token_final_scales: torch.Tensor,
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every rank's rows in rank order, ``[sum(all_rank_num_tokens), cols]`` (``[ep_size * T, cols]`` with
        ``use_dp_padding``, where every rank passes ``T`` rows)."""
        sizes = None if use_dp_padding else list(all_rank_num_tokens)
        payloads = [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales]
        if sizes is not None:
            assert len(sizes) == len(self.mapping.tp_group)
            assert all(p.shape[0] == sizes[self.mapping.tp_rank] for p in payloads)
        gathered = allgather_list(
            [p.contiguous().view(-1, p.shape[-1]) for p in payloads], sizes, self.mapping.tp_group
        )
        self._sizes = sizes
        x, x_sf, ids, weights = (g.view(-1, p.shape[-1]) for g, p in zip(gathered, payloads))
        return x, x_sf, ids, weights

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        all_rank_max_num_tokens: Optional[int] = None,
        payload_in_workspace: bool = False,
    ) -> torch.Tensor:
        """The sum of every rank's ``final_hidden_states`` over the dispatched rows; this rank keeps its own rows
        (``all_rank_max_num_tokens`` and ``payload_in_workspace``, always False here, are not read)."""
        sizes = self._sizes
        if sizes is not None:
            assert len(sizes) == len(self.mapping.tp_group)
            assert final_hidden_states.shape[0] == sum(sizes)
        hidden = final_hidden_states.shape[-1]
        out = reducescatter(
            final_hidden_states.contiguous().view(-1, hidden), sizes, self.mapping.tp_group
        )
        return out.view(-1, hidden)


def create_ep_all_to_all(
    mapping: Mapping,
    num_slots: int,
    top_k: int,
    max_num_tokens_per_rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_low_precision_combine: bool = False,
    *,
    workspace_cache: Optional[dict] = None,
) -> Optional[K31EpAllToAll | K31EpAllGather]:
    """The token exchange of K3.1's routed MoE under attention DP, or ``None`` without attention DP.

    ``TRTLLM_FORCE_COMM_METHOD`` = ``NVLINK_ONE_SIDED`` / ``ALLGATHER`` forces one (routing refuses other values);
    otherwise the one-sided all-to-all, and where the platform cannot serve it (``_one_sided_unavailable``: no NVLink
    fabric spanning the group, e.g. HGX nodes, or a group its kernels do not take) the all-gather + reduce-scatter
    exchange, logged. Where the platform serves it, a failure to build it raises: the layer never silently runs on
    the other exchange.
    """
    # The timeout budgets are read (an invalid value raises) before any exchange is tried: the fallback below must
    # not hide a configuration error.
    register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)
    if not mapping.enable_attention_dp or mapping.dp_size == 1:
        return None
    args = (
        mapping,
        num_slots,
        top_k,
        max_num_tokens_per_rank,
        hidden_size,
        dtype,
        use_low_precision_combine,
    )
    forced = os.environ.get("TRTLLM_FORCE_COMM_METHOD", "").upper()
    if forced == "ALLGATHER":
        return K31EpAllGather(*args)
    if workspace_cache is None:
        workspace_cache = {}
    unavailable = _cached_one_sided_mismatch(mapping, workspace_cache)
    if unavailable is None:
        return K31EpAllToAll(*args, workspace_cache=workspace_cache)
    if forced == "NVLINK_ONE_SIDED":
        raise RuntimeError(f"the one-sided all-to-all is not available here: {unavailable}")
    logger.info_once(
        "Kimi K3.1 MoE exchange: all-gather + reduce-scatter (the one-sided all-to-all is not available here: "
        f"{unavailable})",
        key="k31_ep_allgather_fallback",
    )
    return K31EpAllGather(*args)


__all__ = [
    "K31EpAllGather",
    "K31EpAllToAll",
    "SENTINEL_PAYLOAD_FORMAT",
    "allgather_list",
    "create_ep_all_to_all",
    "default_sentinel_max_tokens",
    "moe_a2a_combine",
    "moe_a2a_dispatch",
    "moe_a2a_get_combine_payload_tensor",
    "moe_a2a_get_workspace_layout",
    "moe_a2a_initialize",
    "moe_a2a_sanitize_expert_ids",
    "moe_a2a_sentinel_available",
    "moe_a2a_sentinel_extend_workspace_layout",
    "moe_a2a_sentinel_row_geometry",
    "moe_a2a_set_timeout",
    "sentinel_row",
    "sentinel_unsupported_reason",
    "workspace_layout",
]
