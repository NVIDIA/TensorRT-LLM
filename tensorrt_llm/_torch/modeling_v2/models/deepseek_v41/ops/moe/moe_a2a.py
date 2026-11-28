# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The expert all-to-all over NVLink symmetric memory: the engine's one-sided `moe_comm` kernels, as stock's context
role runs them under attention DP (`NVLinkOneSided` with its "auto" sync protocol, no EPLB, no rank mask): the sentinel
transport (`moeA2ASentinelDispatchKernel`, `moeA2ASentinelCombineKernel`) for steps whose busiest rank has at most the
workspace's `sentinel_max_rows` rows, the fence transport (`moeA2ADispatchKernel`, `moeA2ASanitizeExpertIdsKernel`,
`moeA2ACombineKernel`) for larger ones.

Under attention DP every rank routes and quantizes its own rows. `moe_a2a_dispatch` puts each row into the
symmetric receive buffer of every rank that holds one of its experts (once per rank, not per expert), so a rank
receives `[ep_size, max_rows, ...]` rows of which the first `count[src]` per source are live; the rest have their
expert ids set to -1 (no expert); a sentinel step does not compact (row `src * max_rows + t` is token `t` of source
`src`, -1 ids where that token is not routed here). The experts run over the received rows (each rank its own window
of experts, non-local ids contribute nothing) and `moe_a2a_combine` sums every row's partial results back on the
rank that owns the row; the experts may write their output straight into the workspace's combine region
(`moe_a2a_combine_payload`), so the combine does not stage it.

`moe_a2a_workspace` is a build-time helper of the owning target, not an entry: the **collective** allocation of
the symmetric workspace (every EP rank calls it, at build time, in the same order) and the layout every call
addresses. The target keeps the returned `MoeA2AWorkspace` and passes its fields to every call (layers run one
after another, so they share it). It also holds the outcome of stock's sentinel vote (`NVLinkOneSided._vote_sentinel`
under the default "auto" protocol): every rank votes whether its GPU runs the sentinel kernels for these rows and its SM
count, and where all can, the step limit is stock's `default_sentinel_max_tokens` and the sentinel regions are appended
to the layout. Dispatch and combine take that limit and pick the transport from the step's `max_rows`, which every rank
passes alike, so every rank takes the same one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.bindings import internal as _tllm_internal

INVALID_EXPERT = -1  # the id a dead receive row carries: `moe_sort` routes it nowhere
_ALIGN = 1024  # per-payload slack of the receive region (the op aligns each payload's start to a cache line)
_SENTINEL_FORMAT = "NVFP4"  # the sentinel payload format of the rows this family dispatches (stock's for NVFP4 layers)
_SF_VEC = 16  # NVFP4 elements per scale byte


def _pad(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


@dataclass
class MoeA2AWorkspace:
    """The symmetric workspace `[ep_size, bytes_per_rank]` uint8, its layout (`metainfo`), the MNNVL allocation
    that backs it (kept alive with it), the per-rank row capacity it was sized for, and the largest step the sentinel
    transport carries (`sentinel_max_rows`, 0 where the vote turned it off)."""

    workspace: torch.Tensor
    metainfo: torch.Tensor
    memory: object
    ep_rank: int
    ep_size: int
    top_k: int
    max_rows: int
    sentinel_max_rows: int


def _sentinel_vote(
    mapping, metainfo: torch.Tensor, max_rows: int, top_k: int, hidden: int, num_experts: int
) -> int:
    """Stock's sentinel vote (`NVLinkOneSided._vote_sentinel`, "auto" protocol) for NVFP4 rows of `hidden`: the step
    limit, with the sentinel regions appended to `metainfo`, or 0 (fence only) when a rank cannot run the sentinel
    kernels. The reasons are stock's `_sentinel_unsupported_reason` for this family's fixed settings (BF16 expert
    outputs, no FP8 combine, no watchdog or rank mask). Collective: every rank votes."""
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
        default_sentinel_max_tokens,
        sentinel_row,
    )

    thop = _tllm_internal.thop
    ep = mapping.moe_ep_size
    ok = bool(torch.ops.trtllm.moe_a2a_sentinel_available()) and ep <= int(
        thop.MOE_A2A_SENTINEL_MAX_RANKS
    )
    ok = ok and num_experts % ep == 0
    row = None
    if ok:
        try:
            row = sentinel_row(_SENTINEL_FORMAT, hidden, top_k)
        except RuntimeError:  # the native row check: the kernels do not take this hidden size
            ok = False
    sms = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        if ok
        else None
    )
    votes = MnnvlMemory.get_comm(mapping).allgather((ok, sms))
    if not all(v[0] for v in votes):
        return 0
    limit = min(default_sentinel_max_tokens(ep, min(v[1] for v in votes), row), max_rows)
    torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
        metainfo, limit, int(getattr(thop, f"MOE_A2A_SENTINEL_FORMAT_{_SENTINEL_FORMAT}")), hidden
    )
    return limit


def moe_a2a_workspace(
    mapping,
    max_rows: int,
    top_k: int,
    row_bytes: Sequence[int],
    combine_row_bytes: int,
    hidden: int,
    num_experts: int,
) -> MoeA2AWorkspace:
    """Allocate and initialize the workspace for dispatches of payloads of `row_bytes` bytes per row each (NVFP4 rows
    of `hidden` and their scales first, then ids and weights) and combines of `combine_row_bytes` per row, at most
    `max_rows` rows per rank, with the sentinel regions where every rank votes for them (`_sentinel_vote`).

    **Collective** over the expert-parallel group: the vote gathers every rank's, the MNNVL allocation exchanges
    handles and the initialization ends in a barrier, so every rank calls it once, at build time, never inside a
    forward or a graph capture.
    """
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    assert list(row_bytes[:2]) == [hidden // 2, hidden // _SF_VEC], (
        "NVFP4 rows of `hidden` and their scales first"
    )
    ep, rank = mapping.moe_ep_size, mapping.moe_ep_rank
    rows = ep * max_rows
    dispatch = sum(_pad(rows * b) for b in row_bytes)
    combine = _pad(rows * combine_row_bytes)
    metainfo = torch.ops.trtllm.moe_a2a_get_workspace_layout(
        ep, max_rows, top_k, dispatch, combine, None
    )
    MnnvlMemory.initialize()
    sentinel_max_rows = _sentinel_vote(mapping, metainfo, max_rows, top_k, hidden, num_experts)
    thop = _tllm_internal.thop
    size = int(metainfo[int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    memory = MnnvlMemory(mapping, size)
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, rank, ep)  # ends in a barrier
    return MoeA2AWorkspace(
        workspace, metainfo, memory, rank, ep, top_k, max_rows, sentinel_max_rows
    )


def moe_a2a_combine_payload(
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_rows: int,
    hidden: int,
) -> torch.Tensor:
    """The combine region of this rank's slice as `[ep_size * max_rows, hidden]` bf16: the experts write their output
    here (`cute_dsl_moe_nvfp4(..., out=)`), so neither combine stages it. Valid after this step's dispatch."""
    return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, metainfo, ep_rank, ep_size, max_rows, torch.bfloat16, hidden
    ).view(ep_size * max_rows, hidden)


class MoeA2ADispatch(OpWrapper):
    """`(ids [n, top_k] int32, payloads [n, *] each, ...)` -> `(received payloads [ep_size * max_rows, *] each, combine
    offset)`.

    Row `t` of this rank goes to every rank whose expert window `[r * E / ep, (r + 1) * E / ep)` holds one of
    `ids[t]` (an id of -1 goes nowhere); every payload of a row travels with it. `max_rows` is the largest row count of
    any rank this step (the same on every rank), and it picks the transport as stock's `use_sentinel_for_dispatch`
    does: the sentinel one up to `sentinel_max_rows` (the workspace's), else the fence one.

    - Fence: the rows from source `s` are packed into `[s * max_rows, s * max_rows + count)` in an unspecified order,
      and the expert ids of the dead rows past `count` are set to -1 (`moe_a2a_sanitize_expert_ids`).
    - Sentinel: row `s * max_rows + t` holds token `t` of source `s`; a row whose token is not routed here (or that
      `s` does not have) has -1 in every id slot. NVFP4 codes arrive canonicalized (a -0 nibble as +0, the same value).

    Either way the experts skip the rows with -1 ids. The returned tensors are views of the workspace, valid until the
    next dispatch; the offset goes to `moe_a2a_combine`, which must get the same `max_rows` and `sentinel_max_rows`.

    `payloads` must include `ids` itself (at `ids_index`): it is the id payload that is received and sanitized. A
    sentinel step needs the payloads in stock's order and format: NVFP4 codes `[n, H / 2]` and scales `[n, H / 16]`
    (bytes), `ids` at index 2, then 4-byte or bf16 weights `[n, top_k]` (the workspace's sentinel layout).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep2 context step: EP2, 2048 rows per rank, top-6 of 384, NVFP4 rows (H 5120), scales, ids, weights",
            spec=dict(ep=2, rows=(2048, 2048), top_k=6, experts=384, hidden=5120),
        ),
        Cell(
            why="dep2 uneven ranks (the decoder half of a bounded replay): 300 and 1 rows",
            spec=dict(ep=2, rows=(300, 1), top_k=6, experts=384, hidden=5120),
        ),
        Cell(
            why="dep2 rows whose experts all sit on one rank, and rows routed nowhere (-1)",
            spec=dict(ep=2, rows=(64, 64), top_k=6, experts=384, hidden=5120, skew=True),
        ),
        Cell(
            why="dep2 decoder pass graph, one prompt's 140-row replay window per rank (the bucket both ranks replay)",
            spec=dict(ep=2, rows=(140, 140), top_k=6, experts=384, hidden=5120),
        ),
        Cell(
            why="dep2 decoder pass graph, three windows per rank (420-row bucket)",
            spec=dict(ep=2, rows=(420, 420), top_k=6, experts=384, hidden=5120, skew=True),
        ),
    )
    note = """
    Two transports, as stock's "auto" protocol. Fence (steps past `sentinel_max_rows`): the dispatch kernel writes
    into the peers' receive buffers and publishes a completion flag behind a system fence; the sanitize kernel waits on
    the received counts. Sentinel (smaller steps): one dispatch and one combine kernel; every 16-byte vector carries its
    own arrival (a receiver polls the data for the armed sentinel word), so neither waits for a flag or a fence, and the
    dispatch fills the ids of rows without a token itself. A sentinel dispatch must be followed by its combine (the
    native op checks it). Under a CUDA graph the host arguments (row counts) are baked in, as for any collective. The op
    allocates nothing and syncs nothing on the host.
    """

    def __call__(
        self,
        ids: torch.Tensor,
        payloads: List[torch.Tensor],
        ids_index: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        max_rows: int,
        ep_rank: int,
        ep_size: int,
        num_experts: int,
        sentinel_max_rows: int,
    ) -> Tuple[List[torch.Tensor], int]:
        top_k = ids.shape[1]
        sentinel = max_rows <= sentinel_max_rows
        recv, combine_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            ids, payloads, workspace, metainfo, max_rows, ep_rank, ep_size, top_k, num_experts,
            eplb_local_stats=None, expert_id_payload_index=ids_index if sentinel else None,
            invalid_token_expert_id=INVALID_EXPERT if sentinel else None,
            enable_rank_mask=False, active_rank_mask=None, use_sentinel=sentinel,
        )  # fmt: skip
        if not sentinel:
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                recv[ids_index], workspace, metainfo, ep_rank, INVALID_EXPERT
            )
        return [t.view(ep_size * max_rows, t.shape[-1]) for t in recv], int(combine_offset)

    def reference(
        self,
        all_ids: Sequence[torch.Tensor],
        all_payloads: Sequence[Sequence[torch.Tensor]],
        ids_index: int,
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ) -> List[List[torch.Tensor]]:
        """What this rank receives, per source rank: the payload rows (in source order) of the rows that hold an
        expert of this rank's window. The collective exception (see `OpWrapper.reference`): it takes every rank's
        `ids` and `payloads` in rank order. The op's order within a source is unspecified, so a test compares the
        received live rows to these as sets (on a sentinel step a live row also sits at its token's index)."""
        lo = (num_experts * ep_rank) // ep_size
        hi = (num_experts * (ep_rank + 1)) // ep_size
        out = []
        for ids, payloads in zip(all_ids, all_payloads):
            keep = ((ids >= lo) & (ids < hi)).any(dim=1)
            out.append([p[keep] for p in payloads])
        return out

    def is_valid(
        self,
        ids: torch.Tensor,
        payloads: List[torch.Tensor],
        ids_index: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        max_rows: int,
        ep_rank: int,
        ep_size: int,
        num_experts: int,
        sentinel_max_rows: int,
    ) -> None:
        n, top_k = ids.shape
        assert ids.dtype == torch.int32 and ids.is_contiguous()
        assert payloads[ids_index] is ids, "the id payload is the one the sanitize pass rewrites"
        assert 0 < n <= max_rows, (
            "every rank dispatches at least one row (a dummy row routed nowhere if it has none)"
        )
        for p in payloads:
            assert p.dim() == 2 and p.shape[0] == n and p.is_contiguous() and p.data_ptr() % 16 == 0
        assert num_experts % ep_size == 0, (
            "this target's windows are equal (the kernel itself takes uneven ones)"
        )
        regions = int(metainfo[int(_tllm_internal.thop.MOE_A2A_SENTINEL_MAX_NUM_TOKENS_INDEX)])
        assert 0 <= sentinel_max_rows <= regions, (
            "the workspace's sentinel limit (`MoeA2AWorkspace.sentinel_max_rows`), or 0 for the fence transport only"
        )
        if max_rows <= sentinel_max_rows:  # stock's `use_sentinel_for_dispatch` row checks
            xq, xsf, w = payloads[0], payloads[1], payloads[3] if len(payloads) > 3 else None
            hidden = 2 * xq.shape[1]
            assert ids_index == 2 and len(payloads) in (3, 4), (
                "codes, scales, ids, weights (stock's order)"
            )
            assert (
                xq.element_size() == 1
                and xsf.element_size() == 1
                and xsf.shape[1] == hidden // _SF_VEC
            )
            assert w is None or (
                w.shape[1] == top_k and (w.element_size() == 4 or w.dtype == torch.bfloat16)
            )


moe_a2a_dispatch = MoeA2ADispatch()


class MoeA2ACombine(OpWrapper):
    """`partial [ep_size * max_rows, H]` bf16 (the experts' output over the received rows) -> `[n, H]` bf16: row `t`
    of this rank is the sum of the partial results its row got on every rank it was dispatched to (zero for a row
    routed nowhere). `n`, `max_rows`, `combine_offset` and `sentinel_max_rows` are the matching dispatch's: they pick
    the same transport. The partial rows of rows the experts skipped (-1 ids) are not read."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = MoeA2ADispatch.CELLS
    note = """
    Fence: the partials are first staged into the combine region of the workspace unless they already are that region
    (`in_workspace`, `moe_a2a_combine_payload`), then each rank reads its rows' partials from every peer and sums
    them in fp32, rounding once to bf16. Sentinel (after a sentinel dispatch): each rank pushes the partials of the rows
    it received into their owners' landing rows, and each owner polls its rows' partials and sums them in the fence
    combine's order (fp32, rounded once), so both transports give the same bits. Allocates its output only and syncs
    nothing on the host.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        n: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        max_rows: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_offset: int,
        sentinel_max_rows: int,
        in_workspace: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_combine(
            partial.view(ep_size, max_rows, partial.shape[-1]), n, workspace, metainfo, max_rows, ep_rank, ep_size,
            top_k, combine_offset, in_workspace, use_low_precision=False, enable_rank_mask=False,
            active_rank_mask=None, use_sentinel=max_rows <= sentinel_max_rows,
        )  # fmt: skip

    def reference(self, all_contributions: Sequence[torch.Tensor]) -> torch.Tensor:
        """The fp32 sum of the partial rows this rank's rows got on each rank (zeros where a row was not sent), in
        rank order, rounded once. The collective exception: a test regenerates every rank's contribution."""
        acc = torch.zeros(
            all_contributions[0].shape, dtype=torch.float32, device=all_contributions[0].device
        )
        for c in all_contributions:
            acc += c.float()
        return acc.to(torch.bfloat16)

    def is_valid(
        self,
        partial: torch.Tensor,
        n: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        max_rows: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_offset: int,
        sentinel_max_rows: int,
        in_workspace: bool = False,
    ) -> None:
        assert partial.dtype == torch.bfloat16 and partial.is_contiguous()
        assert partial.shape[0] == ep_size * max_rows, "the dispatch's receive layout"
        assert 0 < n <= max_rows
        own = workspace[ep_rank, combine_offset:].data_ptr()
        assert (partial.data_ptr() == own) == in_workspace, (
            "in_workspace says whether partial is the combine region"
        )


moe_a2a_combine = MoeA2ACombine()
