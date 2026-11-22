# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DEP16 context MoE all-to-all at op level: one-sided NVLink dispatch and combine over a
workspace built once.

``rubin_moe_a2a_workspace`` (runtime-helper builder, collective) allocates the MNNVL workspace with the
engine's layout and initializes it; ``rubin_moe_a2a_dispatch`` and ``rubin_moe_a2a_combine`` are
the two entries, each one engine op call with every argument explicit. They run the same ops,
with the same arguments, as the engine ``NVLinkOneSided`` strategy, without its per-call state,
environment reads, host watchdog or checkpoint hooks. Every step takes the fence transport: the
combine optionally goes over NVLink as FP8, which the sentinel transport does not carry, so the engine
strategy takes the fence transport for it as well.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.math_utils import pad_up

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
    memory: Any  # the MNNVL allocation behind ``workspace``; kept alive here
    use_low_precision_combine: bool = False


def workspace_payload_bytes(
    ep_size: int,
    top_k: int,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple[int, int]:
    """``(dispatch, combine input)`` payload bytes per rank, as the engine strategy sizes them
    (``NVLinkOneSided._make_workspace_layout``, no EPLB, no extra payload)."""
    tokens = ep_size * max_num_tokens
    itemsize = torch.empty((), dtype=dtype).element_size()
    activations = pad_up(tokens * hidden_size * itemsize, 128)
    quantized = pad_up(tokens * hidden_size, 128) + pad_up(
        tokens * ((hidden_size + 15) // 16) * 4, 128
    )
    dispatch = max(activations, quantized) + 2 * pad_up(tokens * top_k * 4, 128)
    combine_input = tokens * hidden_size * max(itemsize, 2)
    return dispatch, combine_input


def workspace_layout(
    ep_size: int,
    top_k: int,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The workspace layout (``metainfo``, int64 CPU) for these shapes (no sentinel regions)."""
    payloads = workspace_payload_bytes(ep_size, top_k, max_num_tokens, hidden_size, dtype)
    return torch.ops.trtllm.moe_a2a_get_workspace_layout(
        ep_size, max_num_tokens, top_k, *payloads, None
    )


def rubin_moe_a2a_workspace(
    mapping,
    *,
    num_experts: int,
    top_k: int,
    max_num_tokens_per_rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_low_precision_combine: bool = False,
) -> RubinA2aWorkspace:
    """Allocate and initialize this rank's all-to-all workspace (runtime-helper builder).

    Collective over the EP group (MNNVL handle exchange and the initialize barrier): every rank
    calls it once, in ``post_load_weights``, never during capture. ``max_num_tokens_per_rank``
    bounds ``runtime_max_tokens_per_rank`` of every call; ``hidden_size`` / ``dtype`` are the
    activation the MoE sees (the latent, BF16). Registers the engine's warmup-aware timeout for
    the all-to-all kernels, as the engine strategy does.
    """
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import (
        register_moe_comm_timeout_proxy,
    )
    from tensorrt_llm.bindings import internal as _internal

    # The family's one-sided all-to-all timeout proxy (the same op call as the engine strategy's).
    from ....ops.moe.ep_dispatch_combine import _TIMEOUT_PROXY

    if mapping.has_cp_helix():
        raise ValueError("the MoE all-to-all covers the TP group, not a helix CP group")
    ep_size, ep_rank = mapping.moe_ep_size, mapping.moe_ep_rank
    if mapping.world_size != ep_size:
        raise ValueError("the one-sided all-to-all serves pure EP only (world size = EP size)")
    thop = _internal.thop
    if ep_size > int(thop.MOE_A2A_MAX_RANKS) or top_k > int(thop.MOE_A2A_MAX_TOP_K):
        raise ValueError(f"EP {ep_size} / top-k {top_k} exceed the all-to-all kernels' limits")
    register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)
    metainfo = workspace_layout(ep_size, top_k, max_num_tokens_per_rank, hidden_size, dtype)
    size = int(metainfo[_workspace_size_index()])
    MnnvlMemory.initialize()
    memory = MnnvlMemory(mapping, size)
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, ep_rank, ep_size)
    return RubinA2aWorkspace(
        workspace=workspace,
        metainfo=metainfo,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
        num_experts=num_experts,
        max_num_tokens_per_rank=max_num_tokens_per_rank,
        hidden_size=hidden_size,
        memory=memory,
        use_low_precision_combine=use_low_precision_combine,
    )


def _workspace_size_index() -> int:
    from tensorrt_llm.bindings import internal as _internal

    return int(_internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)


#: DEP16 context, rank 0 of 4 emulated ranks of the 16: the latent payload after NVFP4 quantization
#: (3584 / 2 bytes + 224 scale bytes), top 16 over 896 experts.
_DEP16 = dict(hidden=3584, top_k=16, num_experts=896, ranks=4)


class RubinMoeA2aDispatch(OpWrapper):
    """One-sided dispatch of the post-quant MoE inputs to the ranks that own their experts.

    Sends ``hidden_states`` (NVFP4 ``[T, H/2]``), ``hidden_states_sf`` (linear scales
    ``[T, H/16]``), ``token_selected_experts`` (int32 ``[T, top_k]``) and ``token_final_scales``
    (FP32 ``[T, top_k]``) through ``ws``; every rank receives ``[ep_size * runtime_max, ...]`` rows
    of each, slot ids of foreign experts set to -1 (one sanitize launch after the fence dispatch, as
    the engine strategy). ``runtime_max_tokens_per_rank`` is the step's ``max(all_rank_num_tokens)``,
    the same on every rank.

    Returns ``(x, x_sf, ids, weights, combine_payload_offset, combine_payload)``: the four
    received tensors (views into the workspace), the offset ``rubin_moe_a2a_combine`` takes, and
    ``combine_payload``, the workspace region (``[ep_size * runtime_max, combine_hidden]`` of
    ``combine_dtype``) the expert output can be written into so the combine copies nothing.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 context chunk: 512 tokens per rank",
            spec=dict(rows=512, **_DEP16),
        ),
        Cell(
            why="DEP16 context, uneven ranks (64 / 1 / 0 / 17 tokens): sanitized ids, empty rank",
            spec=dict(rows=(64, 1, 0, 17), **_DEP16),
        ),
        Cell(
            why="DEP16 context, 96 tokens per rank: a short chunk",
            spec=dict(rows=96, **_DEP16),
        ),
    )

    note = """
    Collective: every rank of the EP group calls it, in the same order, with the same
    ``runtime_max_tokens_per_rank``. The engine strategy's state
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
            expert_id_payload_index=None,
            invalid_token_expert_id=None,
            enable_rank_mask=False,
            active_rank_mask=None,
            use_sentinel=False,
        )
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
        invalid_token_expert_id: int,
        combine_dtype: torch.dtype,
        combine_hidden: int,
    ) -> list:
        """The collective exception: the four payloads of every rank in rank order, and ``ws``
        any object with ``ep_size`` and ``num_experts``. Returns, per receiving rank ``r``, a dict
        ``{(s, t): (x, x_sf, ids, weights)}`` of every token ``t`` of source rank ``s`` with an
        expert on ``r`` (experts ``[r E / ep, (r + 1) E / ep)``), its rows unchanged (all top-k
        ids). The kernels pack a source's tokens into ``[s * runtime_max, s * runtime_max +
        count)`` in an unspecified order and set the ids of the slots past ``count`` to
        ``invalid_token_expert_id``; the test compares those rows as a set."""
        per_rank = ws.num_experts // ws.ep_size
        out = []
        for r in range(ws.ep_size):
            got = {}
            for s in range(ws.ep_size):
                owner = token_selected_experts[s].long() // per_rank
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
        assert combine_hidden == ws.hidden_size, "the combine payload is the MoE activation width"


class RubinMoeA2aCombine(OpWrapper):
    """One-sided combine: each rank pulls the expert outputs of its tokens from every rank and sums
    them, ``[local_num_tokens, H]`` BF16.

    ``payload`` is ``[ep_size * runtime_max, H]`` or ``[ep_size, runtime_max, H]``: the dispatch's
    ``combine_payload`` in the workspace (nothing is copied). ``combine_payload_offset`` and
    ``runtime_max_tokens_per_rank`` are the dispatch's. The combine uses BF16 or FP8 on the wire.
    The output is BF16.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 context chunk: 512 tokens per rank",
            spec=dict(rows=512, **_DEP16),
        ),
        Cell(
            why="uneven ranks (64 / 1 / 0 / 17 tokens): an empty rank gets an empty output",
            spec=dict(rows=(64, 1, 0, 17), **_DEP16),
        ),
        Cell(
            why="96 tokens per rank: a short chunk",
            spec=dict(rows=96, **_DEP16),
        ),
    )

    note = """
    All ranks call combine after the corresponding dispatch, using its payload offset.
    With low-precision combine enabled, each contribution is rounded to FP8 before summation.
    """

    def __call__(
        self,
        payload: torch.Tensor,
        local_num_tokens: int,
        ws: RubinA2aWorkspace,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
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
            True,  # payload_in_workspace
            use_low_precision=ws.use_low_precision_combine,
            enable_rank_mask=False,
            active_rank_mask=None,
            use_sentinel=False,
        )

    def reference(
        self,
        payload: list,
        local_num_tokens: list,
        ws: Any,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
    ) -> list:
        """The collective exception: ``payload[q]`` is rank ``q``'s expert output as a dict
        ``{(r, t): row}`` keyed by the source rank and token of each row it received, and
        ``local_num_tokens`` every rank's count, in rank order. Rank ``r``'s token ``t`` is the
        FP32 sum over ``q`` of ``payload[q][(r, t)]`` (a rank that did not receive it adds
        nothing)."""
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
    ) -> None:
        assert local_num_tokens <= runtime_max_tokens_per_rank, "more local tokens than dispatched"
        assert payload.numel() == ws.ep_size * runtime_max_tokens_per_rank * payload.shape[-1], (
            "the payload must hold ep_size x runtime_max rows"
        )


rubin_moe_a2a_dispatch = RubinMoeA2aDispatch()
rubin_moe_a2a_combine = RubinMoeA2aCombine()
