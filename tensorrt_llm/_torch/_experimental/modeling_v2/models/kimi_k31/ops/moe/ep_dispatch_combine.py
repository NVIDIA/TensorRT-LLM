# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed-MoE token exchange under attention DP (pure EP over the attention-DP group).

`K31EpAllToAll` is the one-sided NVLink all-to-all: the ``torch.ops.trtllm.moe_a2a_*`` kernels on one MNNVL
workspace that every MoE layer of the process shares. `dispatch` sends each token's payload (the MXFP8 latent
``[T, 3584]``, its UE8M0 scales ``[T, 112]``, int32 expert ids ``[T, 16]`` and BF16 routing weights ``[T, 16]``)
to the ranks owning one of its experts and returns this rank's receive views ``[ep * T_max, ...]``; `combine` sums
the MoE output rows ``[ep * T_max, 3584]`` back onto this rank's tokens. The workspace layout and size, the
counted-writes (CFT) choice, the kernels and their arguments are the engine one-sided strategy's for K3.1.

`K31EpAllGather` is the engine's AllGather + ReduceScatter exchange: the rows of every rank gathered in rank order
(one grouped NCCL all-gather of the four payloads), and the MoE output summed across ranks with each rank keeping
its own rows. `create_ep_all_to_all` picks between the two: the one-sided all-to-all wherever it can be built, else
the all-gather (MPI session; routing refuses attention DP under Ray).
"""

from __future__ import annotations

import os
import re
from typing import List, NamedTuple, Optional, Sequence, Tuple

import pynvml
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch.distributed.mnnvl_memory import CftMnnvlMemory, MnnvlMemory, cuda
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import register_moe_comm_timeout_proxy
from tensorrt_llm.bindings import BuildInfo
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

FORCE_CFT_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT"
CFT_MAX_BATCH_FOR_DISPATCH_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH"
CFT_MAX_BATCH_FOR_COMBINE_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE"
_CFT_DEFAULT_MAX_BATCH = 128
_CFT_ALIGNMENT_BYTES = 16
_CFT_MIN_DRIVER_BRANCH = 615


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
    _per_rank("DEP16 decode", (1, 1, 1, 1), "one token per rank (counted writes under 128 rows)"),
    _per_rank("DEP16 decode", (64, 64, 64, 64), "a 64-token step"),
    _per_rank(
        "DEP16 above the fused-EP cap",
        (257, 257, 257, 257),
        "first per-rank count past 256 (fence)",
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
    tensor. With counted writes the kernel also writes ``invalid_token_expert_id`` into the expert ids of the
    unfilled rows (``expert_id_payload_index``); without them `MoeA2ASanitizeExpertIds` does.
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
        use_cft_counted_writes: bool,
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
            None,
            use_cft_counted_writes,
            expert_id_payload_index,
            invalid_token_expert_id,
            False,
            None,
        )

    def reference(
        self,
        payloads_in_rank_order: Sequence[Sequence[torch.Tensor]],
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ) -> List[List[torch.Tensor]]:
        """Per source rank, its payload rows this rank receives, in the source's token order.

        Collective: takes every rank's payloads (expert ids at `EXPERT_ID_PAYLOAD_INDEX`) in rank order. The kernel
        stores these rows first in the source's block, in slot-claim order (see `note`).
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
    ``[local_num_tokens, hidden]`` in the payload dtype.
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
        use_cft_counted_writes: bool,
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
            use_low_precision,
            use_cft_counted_writes,
            False,
            None,
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
    this rank's workspace bytes up to the layout's size (see `reference`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="the workspace of the K3.1 exchange over one tray (4 ranks, 4096 tokens per rank)",
            spec=dict(world=4, max_tokens=4096, cft=False),
        ),
    )

    def __call__(
        self, workspace: torch.Tensor, metainfo: torch.Tensor, ep_rank: int, ep_size: int
    ) -> None:
        torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, ep_rank, ep_size)

    def reference(
        self, workspace: torch.Tensor, metainfo: torch.Tensor, ep_rank: int, ep_size: int
    ) -> torch.Tensor:
        """This rank's workspace bytes after the call: zero up to the layout's size, except the receive counters
        (``2 x ep_size`` int32 at their offset), set to all ones (-1)."""
        size = int(metainfo[_WORKSPACE_SIZE_INDEX])
        expected = torch.zeros(size, dtype=torch.uint8, device=workspace.device)
        start = int(metainfo[_RECV_COUNTERS_OFFSET_INDEX])
        expected[start : start + 2 * ep_size * 4] = 0xFF
        return expected


def moe_a2a_cft_initialize(
    workspace: torch.Tensor,
    workspace_mem_handle: int,
    workspace_size_per_rank: int,
    ep_rank: int,
    ep_size: int,
) -> None:
    """Binds a logical endpoint to this rank's workspace allocation and exchanges the endpoints across the EP group
    (collective; once per workspace, counted writes only). Host setup with nothing to compute."""
    torch.ops.trtllm.moe_a2a_cft_initialize(
        workspace, workspace_mem_handle, workspace_size_per_rank, ep_rank, ep_size
    )


def moe_a2a_cft_release(workspace: torch.Tensor, ep_rank: int) -> None:
    """Destroys the logical endpoint bound to this workspace (idempotent). Host setup with nothing to compute."""
    torch.ops.trtllm.moe_a2a_cft_release(workspace, ep_rank)


#: The native layout planner's alignments: payload regions, cache lines, counted-write counters.
_WORKSPACE_ALIGNMENT = int(_tllm_internal.thop.MOE_A2A_WORKSPACE_ALIGNMENT)
_CACHELINE_ALIGNMENT = 128
_CFT_COUNTER_STRIDE = 256


def _layout_index(name: str) -> int:
    return int(getattr(_tllm_internal.thop, f"MOE_A2A_{name}_INDEX"))


class MoeA2AGetWorkspaceLayout(OpWrapper):
    """`torch.ops.trtllm.moe_a2a_get_workspace_layout` (host): the int64 metainfo of a workspace with these region
    sizes (control buffers, then the dispatch, combine-input and combine-receive regions), its total size per rank
    at ``MOE_A2A_WORKSPACE_SIZE_INDEX``."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = tuple(
        Cell(
            why=f"K3.1 exchange over {ep} ranks, {tokens} tokens per rank, counted writes {'on' if cft else 'off'}",
            spec=dict(ep=ep, max_tokens=tokens, cft=cft),
        )
        for ep, tokens in ((16, 8192), (4, 4096))
        for cft in (False, True)
    )

    def __call__(
        self,
        ep_size: int,
        max_num_tokens_per_rank: int,
        top_k: int,
        dispatch_payload_bytes: int,
        combine_input_bytes: int,
        combine_recv_bytes: int,
        can_use_cft_counted_writes: bool,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size,
            max_num_tokens_per_rank,
            top_k,
            dispatch_payload_bytes,
            combine_input_bytes,
            combine_recv_bytes,
            None,
            can_use_cft_counted_writes,
        )

    def reference(
        self,
        ep_size: int,
        max_num_tokens_per_rank: int,
        top_k: int,
        dispatch_payload_bytes: int,
        combine_input_bytes: int,
        combine_recv_bytes: int,
        can_use_cft_counted_writes: bool,
    ) -> torch.Tensor:
        """The planner's layout: the regions in order, each at its alignment, the payload capacities rounded up to
        the workspace alignment, the counted-write regions only with counted writes (no EPLB statistics)."""

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
        recv_bytes = up(combine_recv_bytes, _WORKSPACE_ALIGNMENT)
        rank_counts = ep_size * 4
        slots = ep_size * max_num_tokens_per_rank
        routes = max_num_tokens_per_rank * top_k * 4
        cft = can_use_cft_counted_writes
        reserve("FLAG_VAL", 4)
        reserve("LOCAL_TOKEN_COUNTER", 4)
        reserve("SEND_COUNTERS", rank_counts)
        reserve("RECV_COUNTERS", 2 * rank_counts)
        reserve("DISPATCH_COMPLETION_FLAGS", rank_counts, _CACHELINE_ALIGNMENT)
        if cft:
            reserve(
                "DISPATCH_COUNTED_WRITE_COUNTERS",
                ep_size * _CFT_COUNTER_STRIDE,
                _CFT_COUNTER_STRIDE,
            )
            reserve("DISPATCH_COUNTER_BASELINE", ep_size * 8, _CACHELINE_ALIGNMENT)
        reserve("TOPK_TARGET_RANKS", routes, _CACHELINE_ALIGNMENT)
        reserve("TOPK_TARGET_INDICES", routes, _CACHELINE_ALIGNMENT)
        reserve("EPLB_GATHERED_STATS", 0, _CACHELINE_ALIGNMENT)
        reserve("DISPATCH_PAYLOAD", dispatch_bytes, _WORKSPACE_ALIGNMENT)
        fields[_layout_index("DISPATCH_PAYLOAD_SIZE")] = dispatch_bytes
        reserve("COMBINE_COMPLETION_FLAGS", rank_counts, _CACHELINE_ALIGNMENT)
        if cft:
            reserve(
                "COMBINE_COUNTED_WRITE_COUNTERS", slots * _CFT_COUNTER_STRIDE, _CFT_COUNTER_STRIDE
            )
            reserve("COMBINE_COUNTER_BASELINE", slots * 8, _CACHELINE_ALIGNMENT)
        reserve("COMBINE_INPUT", combine_bytes, _WORKSPACE_ALIGNMENT)
        fields[_layout_index("COMBINE_INPUT_SIZE")] = combine_bytes
        if cft:
            reserve("COMBINE_RECV", recv_bytes, _WORKSPACE_ALIGNMENT)
            fields[_layout_index("COMBINE_RECV_SIZE")] = recv_bytes
        fields[_layout_index("MAX_NUM_TOKENS")] = max_num_tokens_per_rank
        fields[_layout_index("TOP_K")] = top_k
        fields[_layout_index("EP_SIZE")] = ep_size
        fields[_layout_index("EPLB_STATS_NUM_EXPERTS")] = 0
        fields[_layout_index("CFT_ENABLED")] = int(cft)
        fields[_layout_index("WORKSPACE_SIZE")] = offset
        return torch.tensor(fields, dtype=torch.int64)


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

#: What building the one-sided all-to-all raises where the platform cannot serve it (the strategy factory then takes
#: the all-gather exchange): the fabric / NVLink probe (its NVML queries: NVMLError) and the rank checks here
#: (RuntimeError); the MNNVL allocation's CUDA driver calls, handle export / import and size checks (RuntimeError,
#: AssertionError), its MPI or process-group communicator (ValueError) and the handle exchange's timeout
#: (TimeoutError, an OSError); the native layout / initialization ops (RuntimeError).
_ONE_SIDED_BUILD_ERRORS = (RuntimeError, ValueError, AssertionError, OSError, pynvml.NVMLError)


# ----------------------------------------------------------------------------------------------------------------
# Counted writes (CFT): the engine's selection
# ----------------------------------------------------------------------------------------------------------------


def _force_cft() -> Optional[bool]:
    return {"0": False, "1": True}.get(os.environ.get(FORCE_CFT_ENV))


def _cft_max_batch(env_name: str) -> int:
    value = os.environ.get(env_name)
    if value is None:
        return _CFT_DEFAULT_MAX_BATCH
    try:
        threshold = int(value)
    except ValueError as e:
        raise ValueError(f"{env_name} must be an integer") from e
    if threshold < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return threshold


class CftOptions(NamedTuple):
    """The counted-writes knobs: ``force`` (``TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT``: True / False / None) and, when
    not forced, the dispatch / combine row thresholds."""

    force: Optional[bool]
    max_batch_for_dispatch: Optional[int]
    max_batch_for_combine: Optional[int]


def read_cft_options() -> CftOptions:
    """The counted-writes knobs from the environment; a malformed threshold raises ``ValueError``."""
    force = _force_cft()
    if force is not None:
        return CftOptions(force, None, None)
    return CftOptions(
        None,
        _cft_max_batch(CFT_MAX_BATCH_FOR_DISPATCH_ENV),
        _cft_max_batch(CFT_MAX_BATCH_FOR_COMBINE_ENV),
    )


def _driver_version() -> Optional[str]:
    try:
        try:
            pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError_Uninitialized:
            pynvml.nvmlInit()
        value = pynvml.nvmlSystemGetDriverVersion()
    except pynvml.NVMLError as error:
        logger.warning_once(
            f"CFT counted writes disabled: NVML driver version query failed ({error}). Using fence.",
            key="moe_a2a_cft_driver_query_failed",
        )
        return None
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _cft_device_unsupported_reason() -> Optional[str]:
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return f"SM{major}{minor} requires SM100 or newer"
    try:
        attributes = (
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
        )
    except AttributeError:
        return "CUDA Python bindings do not expose Logical Endpoint capabilities"
    device = torch.cuda.current_device()
    for attribute in attributes:
        status, supported = cuda.cuDeviceGetAttribute(attribute, device)
        if status != cuda.CUresult.CUDA_SUCCESS:
            return f"querying {attribute.name} failed with {status.name}"
        if not supported:
            return f"device does not support {attribute.name}"
    return None


def cft_supported(force_cft: Optional[bool]) -> bool:
    """Whether this process can use counted writes: a CUDA 13.4+ build, a 615+ driver (unless forced off), SM100+
    with fabric handles and logical endpoints. Otherwise the exchange uses fence-based flags."""
    if not BuildInfo.CFT_COUNTED_WRITES:
        logger.warning_once(
            "CFT counted writes disabled: this build has no logical endpoint support (CUDA < 13.4). Using fence.",
            key="moe_a2a_cft_not_compiled",
        )
        return False
    if force_cft is False:
        return False
    driver = _driver_version()
    match = re.match(r"^(\d+)(?:\.|$)", driver.strip()) if driver else None
    if not (match and int(match.group(1)) >= _CFT_MIN_DRIVER_BRANCH):
        if driver is not None:
            logger.warning_once(
                f"CFT counted writes disabled: NVIDIA driver {driver} is below {_CFT_MIN_DRIVER_BRANCH}.00. "
                "Using fence.",
                key=f"moe_a2a_cft_driver_unsupported_{driver}",
            )
        return False
    reason = _cft_device_unsupported_reason()
    if reason is not None:
        logger.warning_once(
            f"CFT counted writes disabled: {reason}. Using fence.",
            key=f"moe_a2a_cft_device_unsupported_{reason}",
        )
        return False
    return True


def _use_cft(
    can_use_cft: bool, force_cft: Optional[bool], max_batch: Optional[int], runtime_max_tokens: int
) -> bool:
    if not can_use_cft:
        return False
    if force_cft is not None:
        return force_cft
    return max_batch is None or runtime_max_tokens <= max_batch


def _rows_aligned(what: str, row_bytes: Sequence[int]) -> bool:
    for index, nbytes in enumerate(row_bytes):
        if nbytes % _CFT_ALIGNMENT_BYTES != 0:
            logger.warning_once(
                f"CFT counted writes disabled: {what} payload {index} has {nbytes} bytes per token, which is not "
                f"{_CFT_ALIGNMENT_BYTES}-byte aligned. Using fence.",
                key=f"moe_a2a_cft_{what}_alignment_{index}_{nbytes}",
            )
            return False
    return True


def _pad128(nbytes: int) -> int:
    return (nbytes + 127) // 128 * 128


def workspace_layout(
    ep_size: int,
    max_num_tokens_per_rank: int,
    top_k: int,
    hidden_size: int,
    dtype: torch.dtype,
    can_use_cft_counted_writes: bool,
    use_low_precision_combine: bool,
) -> torch.Tensor:
    """The workspace metainfo: the dispatch region holds raw activations or a quantized activation with FP32-per-16
    scales, plus the expert ids and routing weights (4 bytes per slot each); the combine input holds the output at
    ``max(dtype size, 2)`` bytes; the combine receive region exists with counted writes only."""
    tokens = ep_size * max_num_tokens_per_rank
    activations = _pad128(tokens * hidden_size * dtype.itemsize)
    quantized = _pad128(tokens * hidden_size) + _pad128(tokens * ((hidden_size + 15) // 16) * 4)
    dispatch_bytes = max(activations, quantized) + 2 * _pad128(tokens * top_k * 4)
    combine_element_size = max(dtype.itemsize, 2)
    combine_recv_element_size = 1 if use_low_precision_combine else combine_element_size
    return moe_a2a_get_workspace_layout(
        ep_size,
        max_num_tokens_per_rank,
        top_k,
        dispatch_bytes,
        tokens * hidden_size * combine_element_size,
        tokens * hidden_size * combine_recv_element_size if can_use_cft_counted_writes else 0,
        can_use_cft_counted_writes,
    )


# ----------------------------------------------------------------------------------------------------------------
# The exchanges
# ----------------------------------------------------------------------------------------------------------------


class K31EpAllToAll:
    """The one-sided NVLink all-to-all of one MoE layer; the workspace is shared by every layer with the same
    layout (one per process for K3.1)."""

    #: layout key -> {"memory", "workspace", "metainfo", "cft", "ep_rank"}.
    _WORKSPACES: dict = {}
    #: Layout of the workspace the process's logical endpoints are bound to (one per process).
    _CFT_OWNER: Optional[tuple] = None

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        hidden_size: int,
        dtype: torch.dtype,
        use_low_precision_combine: bool = False,
        cft_options: Optional[CftOptions] = None,
    ) -> None:
        """``cft_options``: the counted-writes knobs (:func:`read_cft_options` where not given)."""
        if not MnnvlMemory.supports_mnnvl():
            raise RuntimeError("the one-sided all-to-all needs an NVLink fabric with every link up")
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        if mapping.world_size != self.ep_size:
            raise RuntimeError("the one-sided all-to-all serves pure EP only")
        if self.ep_size > _MAX_RANKS:
            raise RuntimeError(f"the one-sided all-to-all supports at most {_MAX_RANKS} EP ranks")
        # The completion-flag budget is process-wide, so every instance shares one proxy.
        register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)
        self.num_slots = num_slots
        self.top_k = top_k
        self.use_low_precision_combine = use_low_precision_combine
        if cft_options is None:
            cft_options = read_cft_options()
        self.force_cft = cft_options.force
        can_use_cft = cft_supported(self.force_cft)
        cft_shape = (self.ep_size, max_num_tokens_per_rank, num_slots, top_k, hidden_size, dtype)
        cft_shape += (use_low_precision_combine,)
        owner = K31EpAllToAll._CFT_OWNER
        if can_use_cft and owner is not None and owner != cft_shape:
            logger.warning_once(
                "CFT counted writes disabled for this workspace: the logical endpoints are bound to a workspace "
                "with another layout. Using fence.",
                key="moe_a2a_cft_workspace_already_bound",
            )
            can_use_cft = False
        self.can_use_cft_counted_writes = can_use_cft
        self.cft_max_batch_for_dispatch = cft_options.max_batch_for_dispatch
        self.cft_max_batch_for_combine = cft_options.max_batch_for_combine

        metainfo = workspace_layout(
            self.ep_size,
            max_num_tokens_per_rank,
            top_k,
            hidden_size,
            dtype,
            can_use_cft,
            use_low_precision_combine,
        )
        self.workspace_size_per_rank = int(metainfo[_WORKSPACE_SIZE_INDEX])
        MnnvlMemory.initialize()
        key = (self.workspace_size_per_rank, self.ep_rank, *cft_shape, can_use_cft)
        key += (tuple(metainfo.tolist()),)
        state = K31EpAllToAll._WORKSPACES.get(key)
        if state is None:
            memory = (CftMnnvlMemory if can_use_cft else MnnvlMemory)(
                mapping, self.workspace_size_per_rank
            )
            workspace = memory.as_torch_strided_tensor(torch.uint8)
            moe_a2a_initialize(workspace, metainfo, self.ep_rank, self.ep_size)
            if can_use_cft:
                moe_a2a_cft_initialize(
                    workspace,
                    memory.local_mem_handle,
                    int(workspace.size(1)),
                    self.ep_rank,
                    self.ep_size,
                )
                K31EpAllToAll._CFT_OWNER = cft_shape
            state = dict(
                memory=memory,
                workspace=workspace,
                metainfo=metainfo,
                cft=can_use_cft,
                ep_rank=self.ep_rank,
            )
            K31EpAllToAll._WORKSPACES[key] = state
        self.workspace = state["workspace"]
        self.metainfo = state["metainfo"]
        self.invalid_token_expert_id = INVALID_EXPERT_ID
        # (local token count, runtime max tokens per rank, combine payload offset) of the pending dispatch.
        self._pending: Optional[Tuple[int, int, int]] = None

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
        activation, scales, expert ids (``-1`` on unfilled rows) and routing weights. ``use_dp_padding`` is not
        read: padded ranks already pass ``T`` rows."""
        if self._pending is not None:
            raise RuntimeError("dispatch called twice without an intervening combine")
        runtime_max = max(all_rank_num_tokens)
        payloads = [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales]
        use_cft = _use_cft(
            self.can_use_cft_counted_writes,
            self.force_cft,
            self.cft_max_batch_for_dispatch,
            runtime_max,
        ) and _rows_aligned("dispatch", [p.shape[1] * p.element_size() for p in payloads])
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
            use_cft,
            EXPERT_ID_PAYLOAD_INDEX if use_cft else None,
            self.invalid_token_expert_id if use_cft else None,
        )
        self._pending = (token_selected_slots.size(0), runtime_max, int(combine_payload_offset))
        if not use_cft:
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
        ``[local_num_tokens, hidden]``. ``all_rank_max_num_tokens`` is not read: ``T`` comes from the dispatch.
        ``payload_in_workspace``: ``final_hidden_states`` is the `combine_payload` view, already in place."""
        if self._pending is None:
            raise RuntimeError("combine called before a dispatch")
        local_num_tokens, runtime_max, combine_payload_offset = self._pending
        self._pending = None
        payload = final_hidden_states.view(self.ep_size, runtime_max, final_hidden_states.shape[-1])
        wire_element_size = 1 if self.use_low_precision_combine else payload.element_size()
        use_cft = _use_cft(
            self.can_use_cft_counted_writes,
            self.force_cft,
            self.cft_max_batch_for_combine,
            runtime_max,
        ) and _rows_aligned("combine", [payload.shape[-1] * wire_element_size])
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
            use_cft,
        )

    @classmethod
    def release_workspaces(cls) -> None:
        """Release every workspace (on every rank, after the last exchange): the logical endpoints first, then the
        memory once the last view of it is gone."""
        torch.cuda.synchronize()
        for state in cls._WORKSPACES.values():
            if state["cft"]:
                moe_a2a_cft_release(state["workspace"], state["ep_rank"])
        cls._WORKSPACES.clear()
        cls._CFT_OWNER = None


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
) -> Optional[K31EpAllToAll | K31EpAllGather]:
    """The token exchange of K3.1's routed MoE under attention DP, or ``None`` without attention DP.

    ``TRTLLM_FORCE_COMM_METHOD`` = ``NVLINK_ONE_SIDED`` / ``ALLGATHER`` forces one (routing refuses other values);
    otherwise the one-sided all-to-all, and where it cannot be built (no NVLink fabric spanning the group, e.g. HGX
    nodes), the all-gather + reduce-scatter exchange.
    """
    # The timeout budgets are read (an invalid value raises) before any exchange is tried, and so are the
    # counted-writes knobs under attention DP: the fallback below must not hide a configuration error.
    register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)
    if not mapping.enable_attention_dp or mapping.dp_size == 1:
        return None
    cft_options = read_cft_options()
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
    if forced == "NVLINK_ONE_SIDED":
        return K31EpAllToAll(*args, cft_options=cft_options)
    try:
        return K31EpAllToAll(*args, cft_options=cft_options)
    except _ONE_SIDED_BUILD_ERRORS as e:
        logger.info_once(
            "Kimi K3.1 MoE exchange: all-gather + reduce-scatter (the one-sided all-to-all is not available here: "
            f"{e!r})",
            key="k31_ep_allgather_fallback",
        )
    return K31EpAllGather(*args)


__all__ = [
    "K31EpAllGather",
    "K31EpAllToAll",
    "allgather_list",
    "cft_supported",
    "create_ep_all_to_all",
    "moe_a2a_cft_initialize",
    "moe_a2a_cft_release",
    "moe_a2a_combine",
    "moe_a2a_dispatch",
    "moe_a2a_get_combine_payload_tensor",
    "moe_a2a_get_workspace_layout",
    "moe_a2a_initialize",
    "moe_a2a_sanitize_expert_ids",
    "moe_a2a_set_timeout",
    "workspace_layout",
]
