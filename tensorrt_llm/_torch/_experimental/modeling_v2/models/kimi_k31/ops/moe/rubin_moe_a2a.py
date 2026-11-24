# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DEP16 / DEP12 context MoE all-to-all at op level: one-sided NVLink dispatch and combine over
a workspace built once.

``rubin_moe_a2a_workspace`` (runtime-helper builder, collective) allocates the MNNVL workspace with the
engine's layout and initializes it; ``rubin_moe_a2a_dispatch`` and ``rubin_moe_a2a_combine`` are
the two entries, each one engine op call with every argument explicit. They run the same ops,
with the same arguments, as the engine ``NVLinkOneSided`` strategy, without its per-call state,
environment reads, host watchdog or checkpoint hooks. ``check_a2a_lever_env`` asserts at
construction that the engine strategy's environment variables are unset or equal.
``a2a_expert_window`` is the expert window the dispatch routes to each EP rank, which the target's
weights must hold.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.math_utils import pad_up

#: The engine strategy's automatic counted-write (CFT) token limits, dispatch and combine.
CFT_MAX_BATCH = 128
#: The engine strategy's environment variables; check_a2a_lever_env states the value each must have.
A2A_LEVER_ENV = (
    "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE",
)
#: Payload bytes per token must be a multiple of this for counted writes.
_CFT_ALIGNMENT_BYTES = 16
#: The id the kernels give a token slot that routes nowhere (hard-coded in the kernels).
INVALID_EXPERT_ID = -1


class RubinA2aWorkspace(NamedTuple):
    """One rank's all-to-all workspace, built by ``rubin_moe_a2a_workspace``."""

    workspace: torch.Tensor  # uint8 [ep_size, bytes_per_rank], every rank's region mapped
    metainfo: torch.Tensor  # int64 CPU: the layout moe_a2a_get_workspace_layout returns
    ep_rank: int
    ep_size: int
    top_k: int
    num_experts: int
    max_num_tokens_per_rank: int
    hidden_size: int
    can_use_cft: bool  # counted writes available (build, driver, device) and asked for
    use_low_precision_combine: bool  # FP8 combine wire (sizes the counted-write receive region)
    memory: Any  # the MNNVL allocation behind ``workspace``; kept alive here


def dispatch_issued() -> None:
    """Run the engine's ``run_after_next_dispatch`` callbacks: a target calls it right after its
    ``rubin_moe_a2a_dispatch`` is queued, so work they queue (side-stream traffic placed in the
    experts' window) starts after the dispatch, as under the engine's MoE schedulers."""
    from tensorrt_llm._torch.moe.fused_moe import moe_scheduler

    moe_scheduler.dispatch_issued()


def a2a_expert_window(num_experts: int, ep_size: int, ep_rank: int) -> tuple[int, int]:
    """``(first expert, expert count)`` of the window the dispatch routes to EP rank ``ep_rank``.

    The kernel's rule (``compute_target_rank_id``, moeAlltoAllKernels.cu): contiguous windows, the
    first ``num_experts % ep_size`` ranks one expert wider than the rest. Of 896 experts, EP16
    gives every rank 56; EP12 gives ranks 0-7 75 and ranks 8-11 74.
    """
    base, wide = divmod(num_experts, ep_size)
    assert base > 0 and 0 <= ep_rank < ep_size, (num_experts, ep_size, ep_rank)
    return ep_rank * base + min(ep_rank, wide), base + int(ep_rank < wide)


def a2a_expert_owner(expert_ids: torch.Tensor, num_experts: int, ep_size: int) -> torch.Tensor:
    """The EP rank whose ``a2a_expert_window`` holds each of ``expert_ids`` (int64, same shape)."""
    base, wide = divmod(num_experts, ep_size)
    split = wide * (base + 1)  # the first expert of the narrower windows
    ids = expert_ids.long()
    return torch.where(ids < split, ids // (base + 1), wide + (ids - split) // base)


def workspace_payload_bytes(
    ep_size: int,
    top_k: int,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    can_use_cft: bool,
    use_low_precision_combine: bool,
) -> tuple[int, int, int]:
    """``(dispatch, combine input, combine receive)`` payload bytes per rank, as the engine
    strategy sizes them (``NVLinkOneSided._make_workspace_layout``, no EPLB, no extra payload)."""
    tokens = ep_size * max_num_tokens
    itemsize = torch.empty((), dtype=dtype).element_size()
    activations = pad_up(tokens * hidden_size * itemsize, 128)
    quantized = pad_up(tokens * hidden_size, 128) + pad_up(
        tokens * ((hidden_size + 15) // 16) * 4, 128
    )
    dispatch = max(activations, quantized) + 2 * pad_up(tokens * top_k * 4, 128)
    combine_element = max(itemsize, 2)
    combine_input = tokens * hidden_size * combine_element
    combine_recv = (
        tokens * hidden_size * (1 if use_low_precision_combine else combine_element)
        if can_use_cft
        else 0
    )
    return dispatch, combine_input, combine_recv


def workspace_layout(
    ep_size: int,
    top_k: int,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    can_use_cft: bool,
    use_low_precision_combine: bool,
) -> torch.Tensor:
    """The workspace layout (``metainfo``, int64 CPU) for these shapes."""
    payloads = workspace_payload_bytes(
        ep_size, top_k, max_num_tokens, hidden_size, dtype, can_use_cft, use_low_precision_combine
    )
    return torch.ops.trtllm.moe_a2a_get_workspace_layout(
        ep_size, max_num_tokens, top_k, *payloads, None, can_use_cft
    )


def rubin_moe_a2a_workspace(
    mapping,
    *,
    num_experts: int,
    top_k: int,
    max_num_tokens_per_rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_low_precision_combine: bool,
    force_cft: Optional[bool],
) -> RubinA2aWorkspace:
    """Allocate and initialize this rank's all-to-all workspace (runtime-helper builder).

    Collective over the EP group (MNNVL handle exchange and the initialize barrier): every rank
    calls it once, in ``post_load_weights``, never during capture. ``max_num_tokens_per_rank``
    bounds ``runtime_max_tokens_per_rank`` of every call; ``hidden_size`` / ``dtype`` are the
    activation the MoE sees (the latent, BF16). ``use_low_precision_combine``: the combine's wire,
    FP8 or BF16, which sizes the counted-write receive region; the workspace keeps it for the
    combine calls. ``force_cft``: None decides counted writes from the build, the driver and the
    device, as the engine does when its variable is unset; False never uses them. Registers the
    engine's warmup-aware timeout for the all-to-all kernels, as the engine strategy does.
    """
    from tensorrt_llm._torch.distributed.mnnvl_memory import CftMnnvlMemory, MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
        _TIMEOUT_PROXY,
        NVLinkOneSided,
        select_cft_counted_writes,
    )
    from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import (
        register_moe_comm_timeout_proxy,
    )

    if mapping.has_cp_helix():
        raise ValueError("the MoE all-to-all covers the TP group, not a helix CP group")
    ep_size, ep_rank = mapping.moe_ep_size, mapping.moe_ep_rank
    if mapping.world_size != ep_size:
        raise ValueError("the one-sided all-to-all serves pure EP only (world size = EP size)")
    if ep_size > NVLinkOneSided.MAX_RANKS or top_k > NVLinkOneSided.MAX_TOP_K:
        raise ValueError(f"EP {ep_size} / top-k {top_k} exceed the all-to-all kernels' limits")
    register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)
    can_use_cft = select_cft_counted_writes(force_cft)
    metainfo = workspace_layout(
        ep_size,
        top_k,
        max_num_tokens_per_rank,
        hidden_size,
        dtype,
        can_use_cft,
        use_low_precision_combine,
    )
    size = int(metainfo[_workspace_size_index()])
    MnnvlMemory.initialize()
    memory = (CftMnnvlMemory if can_use_cft else MnnvlMemory)(mapping, size)
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, ep_rank, ep_size)
    if can_use_cft:
        torch.ops.trtllm.moe_a2a_cft_initialize(
            workspace, memory.local_mem_handle, int(workspace.size(1)), ep_rank, ep_size
        )
    return RubinA2aWorkspace(
        workspace=workspace,
        metainfo=metainfo,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
        num_experts=num_experts,
        max_num_tokens_per_rank=max_num_tokens_per_rank,
        hidden_size=hidden_size,
        can_use_cft=can_use_cft,
        use_low_precision_combine=use_low_precision_combine,
        memory=memory,
    )


def _workspace_size_index() -> int:
    from tensorrt_llm.bindings import internal as _internal

    return int(_internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)


def a2a_use_cft(
    can_use_cft: bool, force_cft: Optional[bool], max_batch: Optional[int], runtime_max_tokens: int
) -> bool:
    """Whether a call uses counted writes: the engine's rule (forced, or up to ``max_batch``
    tokens per rank) on a workspace that has them. A step fact from the row count."""
    if not can_use_cft:
        return False
    if force_cft is not None:
        return force_cft
    return max_batch is None or runtime_max_tokens <= max_batch


#: DEP16 context, rank 0 of 4 emulated ranks of the 16: the latent payload after NVFP4 quantization
#: (3584 / 2 bytes + 224 scale bytes), top 16 over 896 experts.
_DEP16 = dict(hidden=3584, top_k=16, num_experts=896, ranks=4)
#: DEP12's expert windows on 4 emulated ranks: 299 experts split 75 / 75 / 75 / 74, the kernels'
#: uneven split, with DEP12's payload and top 16 (fewer ranks than top-k, as at EP12, so the
#: dispatch routes rank-compact).
_DEP12_WINDOWS = dict(hidden=3584, top_k=16, num_experts=299, ranks=4)


class RubinMoeA2aDispatch(OpWrapper):
    """One-sided dispatch of the post-quant MoE inputs to the ranks that own their experts.

    Sends ``hidden_states`` (NVFP4 ``[T, H/2]``), ``hidden_states_sf`` (linear scales
    ``[T, H/16]``), ``token_selected_experts`` (int32 ``[T, top_k]``) and ``token_final_scales``
    (FP32 ``[T, top_k]``) through ``ws``; every rank receives ``[ep_size * runtime_max, ...]`` rows
    of each, slot ids of foreign experts set to -1 (in the dispatch with counted writes, else one
    sanitize launch, as the engine strategy). ``runtime_max_tokens_per_rank`` is the step's
    ``max(all_rank_num_tokens)``, the same on every rank.

    Returns ``(x, x_sf, ids, weights, combine_payload_offset, combine_payload)``: the four
    received tensors (views into the workspace), the offset ``rubin_moe_a2a_combine`` takes, and
    ``combine_payload``, the workspace region (``[ep_size * runtime_max, combine_hidden]`` of
    ``combine_dtype``) the expert output can be written into so the combine copies nothing.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 context chunk: 512 tokens per rank, fence path",
            spec=dict(rows=512, cft=False, **_DEP16),
        ),
        Cell(
            why="DEP16 context, uneven ranks (64 / 1 / 0 / 17 tokens): sanitized ids, empty rank",
            spec=dict(rows=(64, 1, 0, 17), cft=False, **_DEP16),
        ),
        Cell(
            why="DEP16 context, 96 tokens per rank: counted writes where the platform has them",
            spec=dict(rows=96, cft=None, **_DEP16),
        ),
        Cell(
            why="DEP12 windows (75 / 75 / 75 / 74 experts), 512 tokens per rank, fence path",
            spec=dict(rows=512, cft=False, **_DEP12_WINDOWS),
        ),
        Cell(
            why="DEP12 windows, uneven ranks (64 / 1 / 0 / 17 tokens), fence path",
            spec=dict(rows=(64, 1, 0, 17), cft=False, **_DEP12_WINDOWS),
        ),
    )

    note = """
    Collective: every rank of the EP group calls it, in the same order, with the same
    ``runtime_max_tokens_per_rank`` and ``use_cft_counted_writes``. The engine strategy's state
    machine (dispatch before combine, once each) is the caller's: dep16's MoE block calls the pair
    in order within one MoE layer. No rank-mask and no EPLB statistics: the flat targets run
    neither.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        ws: RubinA2aWorkspace,
        runtime_max_tokens_per_rank: int,
        use_cft_counted_writes: bool,
        invalid_token_expert_id: int,
        combine_dtype: torch.dtype,
        combine_hidden: int,
    ) -> tuple:
        payloads = [hidden_states, hidden_states_sf, token_selected_experts, token_final_scales]
        recv, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            payloads,
            ws.workspace,
            ws.metainfo,
            runtime_max_tokens_per_rank,
            ws.ep_rank,
            ws.ep_size,
            ws.top_k,
            ws.num_experts,
            None,
            use_cft_counted_writes,
            2 if use_cft_counted_writes else None,
            invalid_token_expert_id if use_cft_counted_writes else None,
            False,
            None,
        )
        if not use_cft_counted_writes:
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                recv[2], ws.workspace, ws.metainfo, ws.ep_rank, invalid_token_expert_id
            )
        combine_payload = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            ws.workspace,
            ws.metainfo,
            ws.ep_rank,
            ws.ep_size,
            runtime_max_tokens_per_rank,
            combine_dtype,
            combine_hidden,
        )
        x, x_sf, ids, weights = (t.view(-1, t.shape[-1]) for t in recv)
        return x, x_sf, ids, weights, int(combine_payload_offset), combine_payload

    def reference(
        self,
        hidden_states: list,
        hidden_states_sf: list,
        token_selected_experts: list,
        token_final_scales: list,
        ws: Any,
        runtime_max_tokens_per_rank: int,
        use_cft_counted_writes: bool,
        invalid_token_expert_id: int,
        combine_dtype: torch.dtype,
        combine_hidden: int,
    ) -> list:
        """The collective exception: the four payloads of every rank in rank order, and ``ws``
        any object with ``ep_size`` and ``num_experts``. Returns, per receiving rank ``r``, a dict
        ``{(s, t): (x, x_sf, ids, weights)}`` of every token ``t`` of source rank ``s`` with an
        expert in ``r``'s window (``a2a_expert_window``), its rows unchanged (all top-k ids). The
        kernels pack a source's tokens into ``[s * runtime_max, s * runtime_max + count)`` in an
        unspecified order and set the ids of the slots past ``count`` to
        ``invalid_token_expert_id``; the test compares those rows as a set."""
        out = []
        for r in range(ws.ep_size):
            got = {}
            for s in range(ws.ep_size):
                owner = a2a_expert_owner(token_selected_experts[s], ws.num_experts, ws.ep_size)
                for t in range(token_selected_experts[s].shape[0]):
                    if bool((owner[t] == r).any()):
                        got[(s, t)] = (
                            hidden_states[s][t],
                            hidden_states_sf[s][t],
                            token_selected_experts[s][t],
                            token_final_scales[s][t],
                        )
            out.append(got)
        return out

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        ws: RubinA2aWorkspace,
        runtime_max_tokens_per_rank: int,
        use_cft_counted_writes: bool,
        invalid_token_expert_id: int,
        combine_dtype: torch.dtype,
        combine_hidden: int,
    ) -> None:
        assert token_selected_experts.dtype == torch.int32, "expert ids are routed as int32"
        # The kernels hard-code -1 as the id of a slot that routes nowhere.
        assert invalid_token_expert_id == INVALID_EXPERT_ID, "the kernels' invalid id is -1"
        assert runtime_max_tokens_per_rank <= ws.max_num_tokens_per_rank, (
            f"{runtime_max_tokens_per_rank} tokens per rank exceed the workspace's "
            f"{ws.max_num_tokens_per_rank}"
        )
        assert not use_cft_counted_writes or ws.can_use_cft, "counted writes on a fence workspace"
        assert combine_hidden == ws.hidden_size, "the combine payload is the MoE activation width"


class RubinMoeA2aCombine(OpWrapper):
    """One-sided combine: each rank pulls (fence) or receives (counted writes) the expert outputs
    of its tokens from every rank and sums them, ``[local_num_tokens, H]`` BF16.

    ``payload`` is ``[ep_size * runtime_max, H]`` or ``[ep_size, runtime_max, H]``: the dispatch's
    ``combine_payload`` when ``payload_in_workspace`` (nothing is copied), else any tensor of
    that shape. ``combine_payload_offset`` and ``runtime_max_tokens_per_rank`` are the dispatch's;
    ``use_low_precision`` sends FP8 over NVLink (the output stays BF16): the workspace's
    ``use_low_precision_combine``, the wire its counted-write receive region was sized for.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 context chunk: 512 tokens per rank, low-precision combine from the workspace",
            spec=dict(rows=512, cft=False, low_precision=True, in_workspace=True, **_DEP16),
        ),
        Cell(
            why="DEP16 context chunk: 512 tokens per rank, BF16 combine from the workspace",
            spec=dict(rows=512, cft=False, low_precision=False, in_workspace=True, **_DEP16),
        ),
        Cell(
            why="uneven ranks (64 / 1 / 0 / 17 tokens), BF16 combine of a caller tensor",
            spec=dict(
                rows=(64, 1, 0, 17), cft=False, low_precision=False, in_workspace=False, **_DEP16
            ),
        ),
        Cell(
            why="96 tokens per rank: counted writes where the platform has them",
            spec=dict(rows=96, cft=None, low_precision=True, in_workspace=True, **_DEP16),
        ),
        Cell(
            why="96 tokens per rank, BF16 combine: counted writes where the platform has them",
            spec=dict(rows=96, cft=None, low_precision=False, in_workspace=True, **_DEP16),
        ),
        Cell(
            why="DEP12 windows (75 / 75 / 75 / 74 experts), 96 tokens per rank: counted writes "
            "where the platform has them",
            spec=dict(rows=96, cft=None, low_precision=True, in_workspace=True, **_DEP12_WINDOWS),
        ),
    )

    note = """
    Collective, as the dispatch, and right after the dispatch whose offset it takes. The FP8 wire
    format rounds each partial sum term to FP8 before the sum, so the low-precision combine is
    banded against the BF16 one (the test states the band).
    """

    def __call__(
        self,
        payload: torch.Tensor,
        local_num_tokens: int,
        ws: RubinA2aWorkspace,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool,
        use_cft_counted_writes: bool,
    ) -> torch.Tensor:
        if payload.dim() == 2:
            payload = payload.view(ws.ep_size, runtime_max_tokens_per_rank, payload.shape[-1])
        return torch.ops.trtllm.moe_a2a_combine(
            payload,
            local_num_tokens,
            ws.workspace,
            ws.metainfo,
            runtime_max_tokens_per_rank,
            ws.ep_rank,
            ws.ep_size,
            ws.top_k,
            combine_payload_offset,
            payload_in_workspace,
            use_low_precision,
            use_cft_counted_writes,
            False,
            None,
        )

    def reference(
        self,
        payload: list,
        local_num_tokens: list,
        ws: Any,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool,
        use_cft_counted_writes: bool,
    ) -> list:
        """The collective exception: ``payload[q]`` is rank ``q``'s expert output as a dict
        ``{(r, t): row}`` keyed by the source rank and token of each row it received, and
        ``local_num_tokens`` every rank's count, in rank order. Rank ``r``'s token ``t`` is the
        FP32 sum over ``q`` of ``payload[q][(r, t)]`` (a rank that did not receive it adds
        nothing). The FP8 wire format of ``use_low_precision`` is the test's band."""
        out = []
        for r in range(ws.ep_size):
            rows = []
            for t in range(local_num_tokens[r]):
                parts = [p[(r, t)].float() for p in payload if (r, t) in p]
                rows.append(torch.stack(parts).sum(0) if parts else None)
            width = next((row.shape[-1] for row in rows if row is not None), 0)
            out.append(
                torch.stack([row if row is not None else torch.zeros(width) for row in rows])
                if rows
                else torch.zeros(0, width)
            )
        return out

    def is_valid(
        self,
        payload: torch.Tensor,
        local_num_tokens: int,
        ws: RubinA2aWorkspace,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool,
        use_cft_counted_writes: bool,
    ) -> None:
        assert local_num_tokens <= runtime_max_tokens_per_rank, "more local tokens than dispatched"
        assert payload.numel() == ws.ep_size * runtime_max_tokens_per_rank * payload.shape[-1], (
            "the payload must hold ep_size x runtime_max rows"
        )
        assert not use_cft_counted_writes or ws.can_use_cft, "counted writes on a fence workspace"
        if use_cft_counted_writes:
            wire = 1 if use_low_precision else payload.element_size()
            # The op refuses a misaligned CFT dispatch payload but not a combine one.
            assert payload.shape[-1] * wire % _CFT_ALIGNMENT_BYTES == 0, (
                "counted-write combine rows must be 16-byte aligned"
            )


rubin_moe_a2a_dispatch = RubinMoeA2aDispatch()
rubin_moe_a2a_combine = RubinMoeA2aCombine()


def check_a2a_lever_env(environ: Mapping[str, str], *, force_cft: Optional[bool]) -> None:
    """Raise unless the engine all-to-all variables in ``environ`` are unset or equal to what the
    flat target runs: the automatic workspace size, ``force_cft`` (None: the variable unset), and
    the engine's default limits of ``CFT_MAX_BATCH`` tokens for counted writes."""
    want = {
        "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB": None,
        "TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": None if force_cft is None else str(int(force_cft)),
        "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH": str(CFT_MAX_BATCH),
        "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE": str(CFT_MAX_BATCH),
    }
    for name in A2A_LEVER_ENV:
        raw = environ.get(name)
        if raw is None:
            continue
        if want[name] is None or raw.strip() != want[name]:
            raise ValueError(
                f"{name}={raw} is not what the flat target runs; the flat all-to-all does not "
                "read it: unset it"
            )
