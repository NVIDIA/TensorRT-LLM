# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501


"""
NVLINK One-Sided AllToAll Communication Strategy

This module implements the NVLINK one-sided comm AllToAll throughput communication method for MoE.

NVLINK One-Sided supports post-quant dispatch.
"""

import os
import sys
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    ActiveRankMaskSnapshot,
    AlltoAllWatchdog,
    AlltoAllWatchdogCoordinator,
    AlltoAllWatchdogTimeout,
    EPGroupHealthLike,
    reject_rank_mask_cuda_graph_capture,
)
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlCheckpointCommunicator, MnnvlMemory
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.quantization.mode import QuantAlgo

from ..moe_comm_timeout_guard import register_moe_comm_timeout_proxy
from .base import Communication

SYNC_PROTOCOL_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_SYNC_PROTOCOL"
# How a rank learns that a step's rows have arrived: "fence" (completion flags behind a
# system-scope fence), "sentinel" (receivers poll the data itself) or "auto" (sentinel for the
# steps it can carry, fence for the others). Same names as FlashInfer's ``sync_protocol``.
SYNC_PROTOCOLS = ("auto", "fence", "sentinel")
SENTINEL_MAX_TOKENS_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_SENTINEL_MAX_TOKENS"
# Sentinel payload formats (the MOE_A2A_SENTINEL_FORMAT_<name> constants).
SENTINEL_PAYLOAD_FORMATS = ("BF16", "MXFP8", "FP8_BLOCK128", "NVFP4")
# Default token limit of the sentinel transport (default_sentinel_max_tokens). Up to 159 tokens per
# rank the transport was faster than the fence transport in every configuration measured (GB300,
# EP 4 to 64, hidden 3584 to 7168, every payload format), so the default never goes below.
_SENTINEL_MIN_DEFAULT_MAX_TOKENS = 159
# Above it the default keeps the ep_size * limit rows a sentinel step polls, copies out and re-arms
# within this: a speed bound, not a memory budget; the largest that kept every GB300 case faster.
_SENTINEL_MAX_DEFAULT_REGION_BYTES = 64 << 20


class NVLinkOneSidedConfigError(ValueError):
    """An invalid NVLinkOneSided environment setting. Raised on every EP rank alike, and not
    treated by the communication factory as "NVLinkOneSided unavailable"."""


# CFT counted writes are gone; these variables shipped in v1.4.0rc1 / rc2 and are no longer read.
# FORCE_CFT=0 pinned the fence path, so it still selects fence (with a warning) for one release.
_REMOVED_FORCE_CFT_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT"
_REMOVED_CFT_ENVS = (
    _REMOVED_FORCE_CFT_ENV,
    "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE",
)
_WORKSPACE_MB_ENV = "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB"


def _sentinel_env() -> tuple[str | None, ...]:
    """The raw settings the sentinel vote reads; every EP rank must see the same ones."""
    return tuple(
        os.environ.get(name)
        for name in (
            SYNC_PROTOCOL_ENV,
            SENTINEL_MAX_TOKENS_ENV,
            _WORKSPACE_MB_ENV,
            *_REMOVED_CFT_ENVS,
        )
    )


def get_sync_protocol() -> str:
    """``TRTLLM_NVLINK_ONE_SIDED_A2A_SYNC_PROTOCOL``: "auto" (also when unset), "fence" or "sentinel".

    Raises:
        NVLinkOneSidedConfigError: If the value is none of these.
    """
    for name in _REMOVED_CFT_ENVS:
        if os.environ.get(name) is not None and not (
            name == _REMOVED_FORCE_CFT_ENV and os.environ.get(name) == "0"
        ):
            tllm_logger.warning_once(
                f"{name} is no longer read: CFT counted writes were removed from NVLinkOneSided. "
                f"Use {SYNC_PROTOCOL_ENV}=fence|sentinel|auto.",
                key=f"nvlink_one_sided_removed_{name}",
            )
    value = os.environ.get(SYNC_PROTOCOL_ENV, "").strip().lower()
    if not value and os.environ.get(_REMOVED_FORCE_CFT_ENV) == "0":
        tllm_logger.warning_once(
            f"{_REMOVED_FORCE_CFT_ENV}=0 is deprecated and will stop working in a future release; "
            f"it now selects the fence path, as {SYNC_PROTOCOL_ENV}=fence does.",
            key="nvlink_one_sided_force_cft_0",
        )
        return "fence"
    value = value or "auto"
    if value not in SYNC_PROTOCOLS:
        raise NVLinkOneSidedConfigError(
            f"{SYNC_PROTOCOL_ENV} must be one of {', '.join(SYNC_PROTOCOLS)}, got {value!r}"
        )
    return value


def get_sentinel_max_tokens() -> int | None:
    """``TRTLLM_NVLINK_ONE_SIDED_A2A_SENTINEL_MAX_TOKENS``: a positive token limit, or None when unset.

    Raises:
        NVLinkOneSidedConfigError: If the value is not a positive integer.
    """
    env_value = os.environ.get(SENTINEL_MAX_TOKENS_ENV, "").strip()
    if not env_value:
        return None
    try:
        max_tokens = int(env_value)
    except ValueError as e:
        raise NVLinkOneSidedConfigError(
            f"{SENTINEL_MAX_TOKENS_ENV} must be a positive integer, got {env_value!r}"
        ) from e
    if max_tokens <= 0:
        raise NVLinkOneSidedConfigError(
            f"{SENTINEL_MAX_TOKENS_ENV} must be a positive integer, got {env_value!r} "
            f"(use {SYNC_PROTOCOL_ENV}=fence to turn the sentinel transport off)"
        )
    return max_tokens


def sentinel_payload_format(quant_algo: QuantAlgo | None) -> str | None:
    """Sentinel payload format of the rows a MoE layer of ``quant_algo`` dispatches, or None when
    the transport carries no such rows.

    NVFP4 (and its AWQ / ARC calibrations) and the MXFP8-activation formats dispatch quantized rows
    with their block scales; FP8 per-tensor activations (FP8, W4A8_MXFP4_FP8, W4A8_NVFP4_FP8) travel
    as e4m3 bytes without block scales, a format the kernels do not carry. The others dispatch BF16
    activations (FP8 block-scale backends quantize after dispatch). Pass the layer's own quant algo:
    a mixed-precision checkpoint overrides it per layer.
    """
    if quant_algo in (QuantAlgo.NVFP4, QuantAlgo.NVFP4_AWQ, QuantAlgo.NVFP4_ARC):
        return "NVFP4"
    if quant_algo in (QuantAlgo.MXFP8, QuantAlgo.W4A8_MXFP4_MXFP8):
        return "MXFP8"
    if quant_algo in (QuantAlgo.FP8, QuantAlgo.W4A8_MXFP4_FP8, QuantAlgo.W4A8_NVFP4_FP8):
        return None
    return "BF16"


class SentinelRow(NamedTuple):
    """Row geometry of a sentinel payload format, from the native op."""

    hidden_bytes: int  # one token's hidden states
    scale_bytes: int  # one token's scales (0 for BF16)
    blocks_per_token: int  # dispatch blocks per token slot
    bytes_per_row: int  # sentinel workspace bytes per (source rank, token) row


def sentinel_row(payload_format: str, hidden_size: int, top_k: int) -> SentinelRow:
    """Raises if the kernels do not take rows of ``hidden_size`` in ``payload_format``."""
    format_id = int(getattr(_tllm_internal.thop, f"MOE_A2A_SENTINEL_FORMAT_{payload_format}"))
    return SentinelRow(
        *map(int, torch.ops.trtllm.moe_a2a_sentinel_row_geometry(format_id, hidden_size, top_k))
    )


def default_sentinel_max_tokens(ep_size: int, num_sms: int, row: SentinelRow) -> int:
    """Default token limit of the sentinel transport, in this order of precedence:

    1. at least ``_SENTINEL_MIN_DEFAULT_MAX_TOKENS`` (159);
    2. else the largest step whose dispatch grid stays below one full wave of
       resident blocks (from a full wave on, the fence transport catches up),
       at most the kernels' 512;
    3. lowered while the sentinel regions (landing, combine and receive rows,
       ``ep_size * limit`` of them) would exceed
       ``_SENTINEL_MAX_DEFAULT_REGION_BYTES`` per rank.

    The caller caps it by ``max_num_tokens_per_rank``.
    """
    thop = _tllm_internal.thop
    below_one_wave = (
        int(thop.MOE_A2A_SENTINEL_DISPATCH_BLOCKS_PER_SM) * num_sms - 1
    ) // row.blocks_per_token
    by_region_bytes = _SENTINEL_MAX_DEFAULT_REGION_BYTES // (ep_size * row.bytes_per_row)
    return max(
        min(below_one_wave, int(thop.MOE_A2A_SENTINEL_MAX_TOKENS_PER_RANK), by_region_bytes),
        _SENTINEL_MIN_DEFAULT_MAX_TOKENS,
    )


def _sentinel_unsupported_reason(
    *,
    ep_size: int,
    top_k: int,
    num_slots: int,
    hidden_size: int,
    dtype: torch.dtype,
    payload_format: str | None,
    use_low_precision_combine: bool,
    fault_tolerant: bool,
) -> str | None:
    """Why this rank cannot use the sentinel transport, or ``None`` if it can.

    Every unsupported configuration is a reason, not an exception, so that every rank reaches the
    collective sentinel vote.
    """
    thop = _tllm_internal.thop
    if payload_format is None:
        return "the layer dispatches FP8 per-tensor rows, which it does not carry"
    if not torch.ops.trtllm.moe_a2a_sentinel_available():
        return "its kernels are not built for this GPU (SM100, SM103 and SM107 only)"
    if ep_size > int(thop.MOE_A2A_SENTINEL_MAX_RANKS):
        return f"it supports at most {int(thop.MOE_A2A_SENTINEL_MAX_RANKS)} EP ranks"
    if num_slots % ep_size != 0:
        return "it needs the expert slots to split evenly across EP ranks"
    if dtype != torch.bfloat16:
        return "it combines BF16 expert outputs only"
    if use_low_precision_combine:
        return "it has no FP8 combine"
    if fault_tolerant:
        return "the AlltoAll watchdog and rank-mask recovery observe fence completion flags only"
    if payload_format not in SENTINEL_PAYLOAD_FORMATS:
        return f"it does not carry the payload format {payload_format}"
    try:
        sentinel_row(payload_format, hidden_size, top_k)
    except RuntimeError as error:  # the native row check (TORCH_CHECK)
        return f"it does not carry hidden size {hidden_size} in {payload_format}: {error}"
    return None


class _NVLinkOneSidedTimeoutProxy:
    """Applies MoE communication timeouts to the process-wide completion-flag budget."""

    name = "NVLinkOneSided"

    def set_timeout_seconds(self, seconds: int | None) -> None:
        # The op restores its native default for 0.
        torch.ops.trtllm.moe_a2a_set_timeout(0 if seconds is None else seconds)


_TIMEOUT_PROXY = _NVLinkOneSidedTimeoutProxy()


class NVLinkOneSided(Communication):
    """
    NVLINK one-sided comm AllToAll strategy for throughput scenarios.

    This implementation utilizes symmetric memory to enable peer-to-peer access between GPUs over NVLink.
    The kernels only take the role as one side of the communication: the dispatch kernel puts the data
    into peer ranks' symmetric memory from local buffer, while the combine kernel gets the data from peer
    ranks' symmetric memory and reduces the data into local buffer. It is the most efficient implementation
    by now, but requires symmetric memory size proportional to `max_num_tokens * n_ranks`, which may not
    scale well for very large-scale parallelization.
    """

    # Constants from C++ (must match moeAlltoAllKernels.h)
    MAX_RANKS = int(_tllm_internal.thop.MOE_A2A_MAX_RANKS)
    MAX_TOP_K = int(_tllm_internal.thop.MOE_A2A_MAX_TOP_K)
    MAX_PAYLOADS = int(_tllm_internal.thop.MOE_A2A_MAX_PAYLOADS)

    # Shared workspaces/memory across the process, keyed by payload layout.
    _WORKSPACES: Dict[Tuple[object, ...], dict] = {}
    _WORKSPACE_REFCOUNTS: Dict[Tuple[object, ...], int] = {}
    # Outcome of the sentinel vote per layout and settings: (token limit, payload format, layout
    # after the vote). Layers sharing a layout vote once; every rank builds its layers in the same
    # order, so every rank hits or misses alike.
    _SENTINEL_VOTES: Dict[Tuple[object, ...], Tuple[int, Optional[str], Tuple[int, ...]]] = {}
    _WORKSPACE: dict | None = None

    # MetaInfo indices - initialized from C++ constants
    FLAG_VAL_OFFSET_INDEX = None
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = None
    SEND_COUNTERS_OFFSET_INDEX = None
    RECV_COUNTERS_OFFSET_INDEX = None
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = None
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = None
    EPLB_GATHERED_STATS_OFFSET_INDEX = None
    DISPATCH_PAYLOAD_OFFSET_INDEX = None
    DISPATCH_PAYLOAD_SIZE_INDEX = None
    COMBINE_INPUT_OFFSET_INDEX = None
    COMBINE_INPUT_SIZE_INDEX = None
    WORKSPACE_SIZE_INDEX = None

    @staticmethod
    def _make_workspace_layout(
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        eplb_stats_num_experts: Optional[int],
        extra_payload_bytes_per_token: int,
    ) -> torch.Tensor:
        """Plan per-rank control buffers and independently sized payload regions."""
        if hidden_size <= 0 or extra_payload_bytes_per_token < 0:
            raise ValueError("hidden_size must be positive and extra payload size non-negative")
        tokens = ep_size * max_num_tokens
        # The wrapper accepts raw activations or quantized activations plus scales.
        # A FP32 scale per 16 elements bounds the supported block-scale formats;
        # accounting for its separate alignment also covers very small payloads.
        activations = pad_up(tokens * hidden_size * dtype.itemsize, 128)
        quantized = pad_up(tokens * hidden_size, 128) + pad_up(
            tokens * ((hidden_size + 15) // 16) * 4, 128
        )
        dispatch_bytes = (
            max(activations, quantized)
            + 2 * pad_up(tokens * top_k * 4, 128)
            + pad_up(tokens * extra_payload_bytes_per_token, 128)
        )
        # MoE may write its original-dtype output directly into the input region,
        # even when the communication wire format is FP8.
        combine_element_size = max(dtype.itemsize, 2)
        combine_input_bytes = tokens * hidden_size * combine_element_size
        return torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size,
            max_num_tokens,
            top_k,
            dispatch_bytes,
            combine_input_bytes,
            eplb_stats_num_experts,
        )

    @staticmethod
    def get_aux_data_size(
        ep_size: int,
        max_num_tokens: int,
        eplb_stats_num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> int:
        """Control-buffer bytes; omitted top_k reserves the native routing limit."""
        layout = torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size,
            max_num_tokens,
            NVLinkOneSided.MAX_TOP_K if top_k is None else top_k,
            0,
            0,
            eplb_stats_num_experts,
        )
        return int(layout[_tllm_internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX])

    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        if cls.FLAG_VAL_OFFSET_INDEX is None:
            thop = _tllm_internal.thop
            cls.FLAG_VAL_OFFSET_INDEX = int(thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX)
            cls.LOCAL_TOKEN_COUNTER_OFFSET_INDEX = int(
                thop.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX
            )
            cls.SEND_COUNTERS_OFFSET_INDEX = int(thop.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX)
            cls.RECV_COUNTERS_OFFSET_INDEX = int(thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX)
            cls.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = int(
                thop.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX
            )
            cls.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = int(
                thop.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX
            )
            cls.EPLB_GATHERED_STATS_OFFSET_INDEX = int(
                thop.MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX
            )
            cls.DISPATCH_PAYLOAD_OFFSET_INDEX = int(thop.MOE_A2A_DISPATCH_PAYLOAD_OFFSET_INDEX)
            cls.DISPATCH_PAYLOAD_SIZE_INDEX = int(thop.MOE_A2A_DISPATCH_PAYLOAD_SIZE_INDEX)
            cls.COMBINE_INPUT_OFFSET_INDEX = int(thop.MOE_A2A_COMBINE_INPUT_OFFSET_INDEX)
            cls.COMBINE_INPUT_SIZE_INDEX = int(thop.MOE_A2A_COMBINE_INPUT_SIZE_INDEX)
            cls.WORKSPACE_SIZE_INDEX = int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        payload_in_workspace: bool = False,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        num_experts: Optional[int] = None,
        use_low_precision_combine: bool = False,
        ep_group_health: EPGroupHealthLike | None = None,
        alltoall_watchdog_timeout_s: Optional[float] = None,
        alltoall_watchdog_poll_interval_s: float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
        alltoall_watchdog_on_timeout: Optional[Callable[[AlltoAllWatchdogTimeout], None]] = None,
        sentinel_payload_format: Optional[str] = "BF16",
    ) -> None:
        """
        Initialize NVLinkOneSided with workspace allocation.

        Steps take the fence transport: payloads, then a system fence and a
        completion flag per peer.

        Small steps can take the sentinel transport instead (moeAlltoAllSentinel.h):
        one dispatch and one combine kernel in which every pushed 16-byte vector
        carries its own arrival, so neither direction waits for a completion
        flag or a system fence. Where every EP rank supports it, it takes every
        step without EPLB statistics whose busiest rank sends at most
        ``sentinel_max_tokens`` tokens and whose payloads are in the instance's
        ``sentinel_payload_format``; the fence transport takes the other
        steps. It needs SM100, SM103 or SM107, at most 64 EP ranks, BF16 expert
        outputs without FP8 combine, expert slots that split evenly across ranks,
        and no AlltoAll watchdog or rank-mask recovery (which observe the fence
        completion flags). TRTLLM_NVLINK_ONE_SIDED_A2A_SYNC_PROTOCOL=fence turns it
        off and =sentinel asks for it (a warning names the reason when it stays
        off); unset or "auto", it is on. TRTLLM_NVLINK_ONE_SIDED_A2A_SENTINEL_MAX_TOKENS sets the token limit,
        which otherwise is the largest step whose dispatch grid stays within one
        wave of resident blocks and whose sentinel regions (``ep_size * limit``
        landing, combine and receive rows, appended to the workspace) stay within
        64 MiB per rank, clamped to 159..512 tokens and to
        ``max_num_tokens_per_rank``. Every rank votes, so a rank where it is off
        turns it off everywhere. A sentinel step places the rows it receives from
        rank ``src`` for its token ``t`` at ``src * runtime_max_tokens_per_rank + t``
        and gives rows of tokens not routed here invalid expert IDs. Payload
        words equal to the transport's sentinel arrive rewritten to the same
        value (NaN stays NaN), so callers that send other bytes in BF16 rows must
        turn it off.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            num_slots: Number of routing slots (token_selected_experts values are in [0, num_slots)).
                Note: The terminology is mapped to `num_experts` in this class and the kernels.
            top_k: Number of experts per token
            max_num_tokens_per_rank: Maximum number of tokens per rank (for workspace allocation)
            payload_in_workspace: If True, final_hidden_states is already in workspace
            hidden_size: Hidden dimension size (optional, for auto workspace calculation)
            dtype: Data type (optional, for auto workspace calculation)
            num_experts: (Optional) Number of experts for EPLB stats (must be <= num_slots). DO NOT provide this parameter if EPLB is not enabled.
                Note: The terminology is mapped to `eplb_stats_num_experts` in this class and the kernels.
            use_low_precision_combine: If True, quantize the combine payload to FP8 for NVLink
                transfer (halves NVLink bandwidth usage, output precision is preserved).
                Corresponds to model_config.use_low_precision_moe_combine.
            ep_group_health: Optional read-only committed EP membership. When present, rank-mask handling is
                enabled in the CUDA kernels, and its mask defines the peers expected by the watchdog. Timeout
                detection never mutates it. CUDA graphs are rejected until membership-scoped recapture lands.
            alltoall_watchdog_timeout_s: Optional timeout for the host-side AlltoAll watchdog. If None, the
                watchdog is disabled.
            alltoall_watchdog_poll_interval_s: Poll interval for the watchdog thread.
            alltoall_watchdog_on_timeout: Optional callback invoked when the watchdog reports suspects.
            sentinel_payload_format: Dispatch payload format whose small steps the sentinel transport carries,
                one of SENTINEL_PAYLOAD_FORMATS (see sentinel_payload_format()), or None when the layer
                dispatches rows the transport does not carry (no sentinel regions then). Needs
                hidden_size and dtype.
        """
        super().__init__(mapping)
        if mapping.has_cp_helix():
            raise ValueError(
                "NVLinkOneSided does not support Helix context parallelism because "
                "its MNNVL communicator covers only the tensor-parallel group"
            )

        if self.mapping.world_size != self.ep_size:
            raise RuntimeError("Currently NVLinkOneSided only supports pure EP for MoE.")
        # Check the kernel rank cap before allocating the (large) symmetric workspace.
        if self.ep_size > self.MAX_RANKS:
            raise RuntimeError(
                f"NVLinkOneSided supports at most {self.MAX_RANKS} EP ranks, got ep_size={self.ep_size}."
            )
        # The completion-flag budget is process-wide, so every instance shares one proxy.
        register_moe_comm_timeout_proxy(_TIMEOUT_PROXY)

        # Store needed parameters
        self.num_experts = num_slots
        self.top_k = top_k
        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self.payload_in_workspace = payload_in_workspace
        self.use_low_precision_combine = use_low_precision_combine
        if num_experts is not None:
            assert num_experts > 0 and num_experts <= num_slots, (
                "num_experts must be in (0, num_slots]"
            )
            tllm_logger.info(
                "NVLinkOneSided AlltoAll: EPLB is enabled, with num_slots="
                f"{num_slots} and num_experts={num_experts}"
            )
        self.enable_eplb = num_experts is not None
        self.eplb_stats_num_experts = num_experts
        # Initialize constants from C++
        self._init_constants()

        # Get workspace size
        auto_workspace_size = None
        metainfo = None
        if hidden_size is not None and dtype is not None:
            metainfo = self._make_workspace_layout(
                self.ep_size,
                self.top_k,
                max_num_tokens_per_rank,
                hidden_size,
                dtype,
                self.eplb_stats_num_experts,
                0,
            )
        # Largest step of the sentinel transport; 0 when this instance does not use it.
        self.sentinel_max_tokens = 0
        self._sentinel_format: str | None = None
        self._sentinel_row: SentinelRow | None = None
        if metainfo is not None:
            self._setup_sentinel(
                metainfo,
                hidden_size,
                dtype,
                sentinel_payload_format,
                fault_tolerant=ep_group_health is not None
                or alltoall_watchdog_timeout_s is not None,
            )
            auto_workspace_size = int(metainfo[self.WORKSPACE_SIZE_INDEX])
        workspace_mb_env = os.environ.get(_WORKSPACE_MB_ENV)
        if workspace_mb_env:
            self.workspace_size_per_rank = int(workspace_mb_env) * 1024 * 1024
            msg = f"NVLinkOneSided: Forcing workspace size to {self.workspace_size_per_rank} bytes (TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB={workspace_mb_env})."
            if auto_workspace_size is not None:
                msg += f"Automatically calculated workspace size is {auto_workspace_size} bytes."
                if self.workspace_size_per_rank < auto_workspace_size:
                    raise ValueError(
                        f"Workspace override is too small: {self.workspace_size_per_rank} bytes, "
                        f"layout requires {auto_workspace_size} bytes per rank"
                    )
            tllm_logger.warning(msg)
        elif auto_workspace_size is not None:
            self.workspace_size_per_rank = auto_workspace_size
        else:
            tllm_logger.warning(
                "NVLinkOneSided: hidden_size and dtype are not provided (which are required for calculating workspace size)."
                "Using default workspace size 2048MB."
            )
            self.workspace_size_per_rank = 2048 * 1024 * 1024

        if metainfo is None:
            # Without a model shape, distribute the explicit/default byte budget
            # conservatively. The resulting boundaries are still fixed at initialization.
            alignment = int(_tllm_internal.thop.MOE_A2A_WORKSPACE_ALIGNMENT)
            control_bytes = self.get_aux_data_size(
                self.ep_size,
                max_num_tokens_per_rank,
                self.eplb_stats_num_experts,
                self.top_k,
            )
            # Split the rest between the dispatch payload and combine input regions.
            capacity = (self.workspace_size_per_rank - control_bytes) // 2 // alignment * alignment
            if capacity <= 0:
                raise ValueError("Workspace byte budget leaves no room for payloads")
            metainfo = torch.ops.trtllm.moe_a2a_get_workspace_layout(
                self.ep_size,
                max_num_tokens_per_rank,
                self.top_k,
                capacity,
                capacity,
                self.eplb_stats_num_experts,
            )
        # Fixed region capacities are shared by allocation, native bounds checks,
        # and workspace-backed views. Runtime shapes only pack tokens inside them.
        MnnvlMemory.initialize()
        self._workspace_key = (
            self.workspace_size_per_rank,
            self.max_num_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.eplb_stats_num_experts,
            self.num_experts,
            self.top_k,
            hidden_size,
            dtype,
            self.use_low_precision_combine,
            tuple(metainfo.tolist()),
        )

        workspace_state = NVLinkOneSided._WORKSPACES.get(self._workspace_key)
        workspace_created = workspace_state is None
        if workspace_state is None:
            tllm_logger.info(
                f"NVLinkOneSided: Allocating workspace with size {self.workspace_size_per_rank} bytes."
                f"ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, top_k: {self.top_k}, max_num_tokens_per_rank: {self.max_num_tokens_per_rank}"
            )
            mnnvl_mem = MnnvlMemory(mapping, self.workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, self.ep_rank, self.ep_size)
            workspace_state = {
                "workspace_size_per_rank": self.workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "eplb_stats_num_experts": self.eplb_stats_num_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "use_low_precision_combine": self.use_low_precision_combine,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
            }
        else:
            expected_workspace_state = {
                "workspace_size_per_rank": self.workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "eplb_stats_num_experts": self.eplb_stats_num_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "use_low_precision_combine": self.use_low_precision_combine,
            }
            for key, expected_value in expected_workspace_state.items():
                assert workspace_state[key] == expected_value, (
                    f"reuse workspace with different {key}"
                )

        self._destroyed = False
        self._workspace_registered = False
        self._workspace_state = workspace_state
        self.mnnvl_mem = workspace_state["mnnvl_mem"]
        self.workspace = workspace_state["workspace"]
        self.max_num_tokens_per_rank = workspace_state["max_num_tokens_per_rank"]
        self.ep_group_health = ep_group_health
        # Keep the kernel specialization stable for this communicator's lifetime.
        self._rank_mask_enabled = ep_group_health is not None
        if alltoall_watchdog_timeout_s is None and self.ep_group_health is not None:
            alltoall_watchdog_timeout_s = DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
        flag_val_offset_index = self.FLAG_VAL_OFFSET_INDEX
        dispatch_flags_offset_index = self.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX
        combine_flags_offset_index = self.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX
        if (
            flag_val_offset_index is None
            or dispatch_flags_offset_index is None
            or combine_flags_offset_index is None
        ):
            raise RuntimeError("MoE All-to-All metadata indices are not initialized")
        self._workspace_lifecycle = _MnnvlAlltoAllWorkspaceLifecycle.get_or_create(
            workspace_state=workspace_state,
            memory=self.mnnvl_mem,
            workspace=self.workspace,
            metainfo=workspace_state["metainfo"],
            metainfo_index={
                "FLAG_VAL_OFFSET_INDEX": flag_val_offset_index,
                "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": dispatch_flags_offset_index,
                "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": combine_flags_offset_index,
            },
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            health=self.ep_group_health,
        )
        self._workspace_lifecycle.register(
            self,
            watchdog_timeout_s=alltoall_watchdog_timeout_s,
            watchdog_poll_interval_s=alltoall_watchdog_poll_interval_s,
            watchdog_on_timeout=alltoall_watchdog_on_timeout,
        )
        if workspace_created:
            NVLinkOneSided._WORKSPACES[self._workspace_key] = workspace_state
        NVLinkOneSided._WORKSPACE = workspace_state
        NVLinkOneSided._WORKSPACE_REFCOUNTS[self._workspace_key] = (
            NVLinkOneSided._WORKSPACE_REFCOUNTS.get(self._workspace_key, 0) + 1
        )
        self._workspace_registered = True

        # Initialize dispatch state
        self._dispatch_state = {"phase": "idle"}

        # Invalid token expert ID (default to -1), the kernels in TRTLLM-gen is hard-code to support -1 only.
        self.invalid_token_expert_id: int = -1

    def _setup_sentinel(
        self,
        metainfo: torch.Tensor,
        hidden_size: int,
        dtype: torch.dtype,
        payload_format: Optional[str],
        fault_tolerant: bool,
    ) -> None:
        """Vote on the sentinel transport and, where every rank takes it, append its regions to ``metainfo``.

        Collective over the EP group, once per layout and settings (see ``_SENTINEL_VOTES``).
        """
        key = (
            self.ep_rank,
            self.ep_size,
            self.num_experts,
            self.top_k,
            self.max_num_tokens_per_rank,
            hidden_size,
            dtype,
            payload_format,
            fault_tolerant,
            self.use_low_precision_combine,
            tuple(metainfo.tolist()),
            _sentinel_env(),
        )
        vote = NVLinkOneSided._SENTINEL_VOTES.get(key)
        if vote is None:
            vote = self._vote_sentinel(
                metainfo.clone(), hidden_size, dtype, payload_format, fault_tolerant
            )
            NVLinkOneSided._SENTINEL_VOTES[key] = vote
        max_tokens, payload_format, layout = vote
        metainfo.copy_(torch.tensor(layout, dtype=metainfo.dtype))
        if max_tokens:
            self.sentinel_max_tokens = max_tokens
            self._sentinel_format = payload_format
            self._sentinel_row = sentinel_row(payload_format, hidden_size, self.top_k)

    def _vote_sentinel(
        self,
        metainfo: torch.Tensor,
        hidden_size: int,
        dtype: torch.dtype,
        payload_format: Optional[str],
        fault_tolerant: bool,
    ) -> Tuple[int, Optional[str], Tuple[int, ...]]:
        """The collective sentinel vote: (token limit, 0 when off; payload format; layout).

        Every rank votes its own support and SM count (for the default token limit) along with its
        raw settings, and parses the settings only after seeing that every rank has the same ones,
        so an invalid or mismatched setting raises on every rank instead of stranding the others in
        the collective.
        """
        reason = _sentinel_unsupported_reason(
            ep_size=self.ep_size,
            top_k=self.top_k,
            num_slots=self.num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            payload_format=payload_format,
            use_low_precision_combine=self.use_low_precision_combine,
            fault_tolerant=fault_tolerant,
        )
        num_sms = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            if reason is None
            else None
        )
        votes = MnnvlMemory.get_comm(self.mapping).allgather(
            (reason is None, num_sms, _sentinel_env())
        )
        if len({vote[2] for vote in votes}) > 1:
            raise NVLinkOneSidedConfigError(
                "NVLinkOneSided settings must be the same on every EP rank "
                f"({SYNC_PROTOCOL_ENV}, {SENTINEL_MAX_TOKENS_ENV}, {_WORKSPACE_MB_ENV} and the removed "
                f"CFT variables), got {sorted({repr(vote[2]) for vote in votes})}"
            )
        protocol = get_sync_protocol()
        max_tokens_request = get_sentinel_max_tokens()
        off = (0, None, tuple(metainfo.tolist()))
        if protocol == "fence":
            return off
        if not all(vote[0] for vote in votes):
            log = tllm_logger.warning_once if protocol == "sentinel" else tllm_logger.info_once
            message = (
                "NVLinkOneSided AlltoAll: sentinel transport disabled: "
                f"{reason or 'another EP rank does not support it'}"
            )
            log(message, key=message)
            return off

        thop = _tllm_internal.thop
        row = sentinel_row(payload_format, hidden_size, self.top_k)
        cap = int(thop.MOE_A2A_SENTINEL_MAX_TOKENS_PER_RANK)
        if max_tokens_request is None:
            max_tokens = default_sentinel_max_tokens(
                self.ep_size, min(vote[1] for vote in votes), row
            )
        else:
            if max_tokens_request > cap:
                tllm_logger.warning_once(
                    f"{SENTINEL_MAX_TOKENS_ENV}={max_tokens_request} exceeds the sentinel "
                    f"transport's {cap} tokens per rank; using {cap}",
                    key="nvlink_one_sided_sentinel_max_tokens_clamped",
                )
            max_tokens = min(max_tokens_request, cap)
        max_tokens = min(max_tokens, self.max_num_tokens_per_rank)
        size_before = int(metainfo[self.WORKSPACE_SIZE_INDEX])
        torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
            metainfo,
            max_tokens,
            int(getattr(thop, f"MOE_A2A_SENTINEL_FORMAT_{payload_format}")),
            hidden_size,
        )
        size_after = int(metainfo[self.WORKSPACE_SIZE_INDEX])
        workspace_mb = os.environ.get(_WORKSPACE_MB_ENV)
        if workspace_mb and int(workspace_mb) * (1 << 20) < size_after:
            # The same setting on every rank (checked above), so every rank turns it off alike.
            message = (
                f"NVLinkOneSided AlltoAll: sentinel transport disabled: {_WORKSPACE_MB_ENV}="
                f"{workspace_mb} leaves no room for its {(size_after - size_before) / (1 << 20):.1f} "
                "MiB of regions per rank"
            )
            tllm_logger.warning_once(message, key=message)
            return off
        message = (
            f"NVLinkOneSided AlltoAll: sentinel transport for {payload_format} steps of up to "
            f"{max_tokens} tokens per rank, {(size_after - size_before) / (1 << 20):.1f} MiB "
            "of workspace per rank"
        )
        tllm_logger.info_once(message, key=message)
        return max_tokens, payload_format, tuple(metainfo.tolist())

    def use_sentinel_for_dispatch(
        self,
        runtime_max_tokens_per_rank: int,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        eplb_local_stats: Optional[torch.Tensor],
    ) -> bool:
        """Whether the sentinel transport carries this step: few enough tokens, no
        EPLB statistics and payloads in the instance's format.

        Depends only on values every rank passes alike, so all ranks take the
        same transport.
        """
        row = self._sentinel_row
        if (
            row is None
            or runtime_max_tokens_per_rank > self.sentinel_max_tokens
            or eplb_local_stats is not None
        ):
            return False

        def row_bytes(tensor: Optional[torch.Tensor]) -> int:
            if tensor is None:
                return 0
            return tensor.shape[1] * tensor.element_size() if tensor.dim() == 2 else -1

        if (row_bytes(hidden_states), row_bytes(hidden_states_sf)) != (
            row.hidden_bytes,
            row.scale_bytes,
        ):
            return False
        # Senders rewrite payload words equal to a sentinel the way the format
        # keeps its values (NaN stays NaN), so the dtype must be the format's.
        if self._sentinel_format == "BF16":
            if hidden_states.dtype != torch.bfloat16:
                return False
        elif hidden_states.element_size() != 1:  # quantized values are bytes
            return False
        # Routing weights travel as 4-byte words; bf16 weights (TRTLLM-Gen) as the fp32 words
        # of their values, which the receiver narrows back exactly.
        return token_final_scales is None or (
            token_final_scales.dim() == 2
            and token_final_scales.shape[1] == self.top_k
            and (
                token_final_scales.element_size() == 4 or token_final_scales.dtype == torch.bfloat16
            )
        )

    @property
    def moe_a2a_metainfo(self) -> torch.Tensor:
        return self._require_workspace_lifecycle().metainfo

    @property
    def _watchdog_coordinator(self) -> AlltoAllWatchdogCoordinator:
        return self._require_workspace_lifecycle().coordinator

    @property
    def _alltoall_watchdog(self) -> AlltoAllWatchdog | None:
        return self._require_workspace_lifecycle().watchdog_for(self)

    def _require_workspace_lifecycle(self) -> _MnnvlAlltoAllWorkspaceLifecycle:
        lifecycle = self._workspace_lifecycle
        if lifecycle is None:
            raise RuntimeError("NVLinkOneSided workspace has been destroyed")
        return lifecycle

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if NVLINK one-sided comm is supported on current hardware.
        """
        return MnnvlMemory.supports_mnnvl()

    def supports_post_quant_dispatch(self) -> bool:
        """
        NVLINK one-sided comm supports post-quant dispatch.
        """
        return True

    @classmethod
    def _release_workspace(cls, workspace_key: Tuple[object, ...]) -> None:
        """Release a shared workspace after its last communicator is destroyed."""
        workspace_state = cls._WORKSPACES.get(workspace_key)
        if workspace_state is None:
            return

        cls._WORKSPACES.pop(workspace_key, None)
        if cls._WORKSPACE is workspace_state:
            cls._WORKSPACE = None
        workspace_state.clear()

    def destroy(self):
        """Release this instance's reference after all ranks finish using the workspace."""
        if getattr(self, "_destroyed", False):
            return

        self._destroyed = True
        lifecycle = getattr(self, "_workspace_lifecycle", None)
        if lifecycle is not None and getattr(self, "_workspace_registered", False):
            lifecycle.unregister(self)
        workspace_key = getattr(self, "_workspace_key", None)
        if workspace_key is None or not getattr(self, "_workspace_registered", False):
            self._workspace_lifecycle = None
            return
        self._workspace_registered = False

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        refcount = NVLinkOneSided._WORKSPACE_REFCOUNTS.get(workspace_key, 0) - 1
        if refcount > 0:
            NVLinkOneSided._WORKSPACE_REFCOUNTS[workspace_key] = refcount
        else:
            NVLinkOneSided._WORKSPACE_REFCOUNTS.pop(workspace_key, None)
            NVLinkOneSided._release_workspace(workspace_key)

        self.mnnvl_mem = None
        self.workspace = None
        self._workspace_state = None
        self._workspace_lifecycle = None
        self._dispatch_state = {"phase": "destroyed"}

    def __del__(self) -> None:
        if sys.is_finalizing():
            return
        # Finalizers cannot safely perform collective cache eviction because
        # their order is nondeterministic across ranks.
        lifecycle = getattr(self, "_workspace_lifecycle", None)
        if lifecycle is not None and getattr(self, "_workspace_registered", False):
            lifecycle.unregister(self)
        workspace_key = getattr(self, "_workspace_key", None)
        if workspace_key is not None and getattr(self, "_workspace_registered", False):
            refcount = NVLinkOneSided._WORKSPACE_REFCOUNTS.get(workspace_key, 0) - 1
            if refcount > 0:
                NVLinkOneSided._WORKSPACE_REFCOUNTS[workspace_key] = refcount
            else:
                NVLinkOneSided._WORKSPACE_REFCOUNTS.pop(workspace_key, None)
        self._workspace_registered = False
        self._workspace_lifecycle = None

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if NVLINK one-sided comm is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
        """
        return True

    def _require_mapped(self) -> None:
        if not self.mnnvl_mem.mapped:
            raise RuntimeError("Native MoE All-to-All workspace handles are unmapped")

    def checkpoint_resource_key(self) -> int:
        """Identify wrappers sharing the same MNNVL workspace lifecycle."""
        return id(self._require_workspace_lifecycle())

    def checkpoint_prepare(self) -> None:
        """Collectively detach handles after every shared owner is idle."""
        self._require_workspace_lifecycle().checkpoint_prepare()

    def checkpoint_restore(
        self,
        comm: MnnvlCheckpointCommunicator | None = None,
    ) -> None:
        """Collectively restore handles and all shared frontend state.

        Args:
            comm: An mpi4py-like communicator exposing ``Get_rank()``,
                ``Get_size()``, ``allgather()``, and ``barrier()``. Its local
                rank and size must match the communicator used for the
                original allocation. Every rank must call this method
                symmetrically.
        """
        if comm is None:
            comm = self.mnnvl_mem.comm
        if comm is None:
            raise RuntimeError("MNNVL workspace communicator is not initialized")

        # The native op re-validates the planned layout and resets control state;
        # the layout itself is unchanged by a restore.
        def reinitialize_frontend() -> torch.Tensor:
            torch.ops.trtllm.moe_a2a_initialize(
                self.workspace, self.moe_a2a_metainfo, self.ep_rank, self.ep_size
            )
            return self.moe_a2a_metainfo

        self._require_workspace_lifecycle().checkpoint_restore(comm, reinitialize_frontend)

    def _mnnvl_checkpoint_is_idle(self) -> bool:
        return self._dispatch_state.get("phase") == "idle"

    def _mnnvl_checkpoint_reset(self) -> None:
        self._dispatch_state = {"phase": "idle"}

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Dispatch phase: scatter/send data to different ranks

        Args:
            hidden_states: Input tensor [local_num_tokens, hidden_size]
            hidden_states_sf: Input scaling factor [local_num_tokens, sf_size]
            token_selected_slots: Selected expert slots [local_num_tokens, top_k]
            token_final_scales: Router weights [local_num_tokens, top_k]
            all_rank_num_tokens: Token counts per rank [ep_size]
            use_dp_padding: Whether to use DP padding (optional)
            **kwargs: Strategy-specific arguments. In rank-mask mode, ``active_rank_mask`` may override the
                committed membership for dispatch. Without an override, the committed mask and generation are
                captured together; combine reuses that mask and fails closed if the generation changes first.
                The masked kernel rejects inactive routes before remote access; that sentinel is an internal abort
                artifact, not valid model output.

        Returns:
            Tuple of (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)
            Each tensor has shape [ep_size, max_tokens_per_rank, ...]
        """
        self._require_mapped()
        if self._dispatch_state.get("phase") == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")
        reject_rank_mask_cuda_graph_capture(self._rank_mask_enabled)

        # Calculate runtime_max_tokens_per_rank from all_rank_num_tokens
        runtime_max_tokens_per_rank = max(all_rank_num_tokens)
        eplb_local_stats = kwargs.get("eplb_local_stats")
        # The sentinel transport takes every step it can carry: its combine receives and
        # reduces each token in the same kernel as it lands.
        use_sentinel = self.use_sentinel_for_dispatch(
            runtime_max_tokens_per_rank,
            hidden_states,
            hidden_states_sf,
            token_final_scales,
            eplb_local_stats,
        )

        # Build payloads list - match TRTLLMGen baseline order for optimal performance
        # Order: [hidden_states, hidden_states_sf (optional), token_selected_slots, token_final_scales (optional)]

        payloads = []
        payloads.append(hidden_states)
        if hidden_states_sf is not None:
            payloads.append(hidden_states_sf)

        payloads.append(token_selected_slots)
        if token_final_scales is not None:
            payloads.append(token_final_scales)
        expert_id_payload_index = len(payloads) - (2 if token_final_scales is not None else 1)

        if eplb_local_stats is not None:
            assert self.enable_eplb, "eplb_local_stats provided but enable_eplb is False"
            assert eplb_local_stats.dim() == 1, "eplb_local_stats must be a 1D tensor"
            assert eplb_local_stats.size(0) == self.eplb_stats_num_experts, (
                "eplb_local_stats size must match eplb_stats_num_experts"
            )
        requested_active_rank_mask = kwargs.get("active_rank_mask")
        if not self._rank_mask_enabled and requested_active_rank_mask is not None:
            raise ValueError("active_rank_mask requires committed EP group health")
        active_rank_mask_snapshot = self._watchdog_coordinator.capture_active_rank_mask(
            requested_active_rank_mask
        )
        active_rank_mask = active_rank_mask_snapshot.active_rank_mask

        recv_buffers, combine_payload_offset, eplb_gathered_stats = (
            torch.ops.trtllm.moe_a2a_dispatch(
                token_selected_slots,
                payloads,
                self.workspace,
                self.moe_a2a_metainfo,
                runtime_max_tokens_per_rank,
                self.ep_rank,
                self.ep_size,
                self.top_k,
                self.num_experts,
                eplb_local_stats,
                # A sentinel dispatch fills the expert IDs of rows without a token itself.
                expert_id_payload_index=expert_id_payload_index if use_sentinel else None,
                invalid_token_expert_id=int(self.invalid_token_expert_id) if use_sentinel else None,
                enable_rank_mask=self._rank_mask_enabled,
                active_rank_mask=active_rank_mask,
                use_sentinel=use_sentinel,
            )
        )
        # A sentinel round writes no completion flags for the watchdog to observe.
        if not use_sentinel:
            self._watchdog_coordinator.watch_collective(
                self._alltoall_watchdog, "dispatch", active_rank_mask
            )
        if eplb_gathered_stats.numel() == 0:
            eplb_gathered_stats = None
        self._dispatch_state["eplb_gathered_stats"] = eplb_gathered_stats
        self._dispatch_state["combine_payload_offset"] = int(combine_payload_offset)
        self._dispatch_state["local_num_tokens"] = token_selected_slots.size(0)
        self._dispatch_state["runtime_max_tokens_per_rank"] = runtime_max_tokens_per_rank
        self._dispatch_state["active_rank_mask_snapshot"] = active_rank_mask_snapshot
        self._dispatch_state["sentinel"] = use_sentinel
        self._dispatch_state["phase"] = "dispatched"

        # Extract results from recv_buffers
        # Payload order matches input:
        # [hidden_states, hidden_states_sf (optional), token_selected_slots, token_final_scales (optional)]
        hidden_states_recv = recv_buffers[0]
        if hidden_states_sf is not None:
            hidden_states_sf_recv = recv_buffers[1]
            token_selected_slots_recv = recv_buffers[2]
            token_final_scales_recv = recv_buffers[3] if token_final_scales is not None else None
        else:
            hidden_states_sf_recv = None
            token_selected_slots_recv = recv_buffers[1]
            token_final_scales_recv = recv_buffers[2] if token_final_scales is not None else None

        if not use_sentinel:
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                token_selected_slots_recv,
                self.workspace,
                self.moe_a2a_metainfo,
                self.ep_rank,
                int(self.invalid_token_expert_id),
            )

        # Flatten 3D tensors to 2D for compatibility with MoE backends
        # recv_buffers have shape [ep_size, max_tokens_per_rank, ...], flatten to [ep_size * max_tokens_per_rank, ...]
        hidden_states_recv = hidden_states_recv.view(-1, hidden_states_recv.shape[-1])
        if hidden_states_sf_recv is not None:
            hidden_states_sf_recv = hidden_states_sf_recv.view(-1, hidden_states_sf_recv.shape[-1])
        token_selected_slots_recv = token_selected_slots_recv.view(
            -1, token_selected_slots_recv.shape[-1]
        )
        if token_final_scales_recv is not None:
            token_final_scales_recv = token_final_scales_recv.view(
                -1, token_final_scales_recv.shape[-1]
            )

        return (
            hidden_states_recv,
            hidden_states_sf_recv,
            token_selected_slots_recv,
            token_final_scales_recv,
        )

    def get_eplb_gathered_statistics(self) -> Optional[torch.Tensor]:
        """
        Return gathered EPLB statistics from the last dispatch, if available.
        """
        assert self.enable_eplb, "EPLB is not enabled"
        return self._dispatch_state.get("eplb_gathered_stats")

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Combine phase: gather/receive data from different ranks

        Args:
            final_hidden_states: Output from MoE computation
                Shape: [ep_size, max_tokens_per_rank, hidden_size] or
                       [ep_size * max_tokens_per_rank, hidden_size] (will be reshaped)
            **kwargs: Strategy-specific arguments. In rank-mask mode, ``active_rank_mask`` must match the mask
                captured by dispatch for this collective when supplied. A committed-generation change since
                dispatch aborts the collective epoch.

        Returns:
            Combined output tensor [local_num_tokens, hidden_size]

        """
        self._require_mapped()
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")
        reject_rank_mask_cuda_graph_capture(self._rank_mask_enabled)

        local_num_tokens = self._dispatch_state.get("local_num_tokens")
        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        runtime_max_tokens_per_rank = self._dispatch_state.get("runtime_max_tokens_per_rank")

        if (
            local_num_tokens is None
            or combine_payload_offset is None
            or runtime_max_tokens_per_rank is None
        ):
            raise RuntimeError("combine called but dispatch state is missing")

        # Reshape if needed (handle case where input is flattened)
        if final_hidden_states.dim() == 2:
            # Flattened: [ep_size * max_tokens_per_rank, hidden_size]
            # Reshape to: [ep_size, max_tokens_per_rank, hidden_size]
            hidden_size = final_hidden_states.shape[-1]
            final_hidden_states = final_hidden_states.view(
                self.ep_size, runtime_max_tokens_per_rank, hidden_size
            )
        elif final_hidden_states.dim() == 3:
            # Already shaped: [ep_size, max_tokens_per_rank, hidden_size]
            pass
        else:
            raise ValueError(
                f"final_hidden_states must be 2D or 3D, got {final_hidden_states.dim()}D"
            )
        active_rank_mask_snapshot = self._dispatch_state.get("active_rank_mask_snapshot")
        if not isinstance(active_rank_mask_snapshot, ActiveRankMaskSnapshot):
            raise RuntimeError("combine called but dispatch rank-mask snapshot is missing")
        requested_active_rank_mask = kwargs.get("active_rank_mask")
        if not self._rank_mask_enabled and requested_active_rank_mask is not None:
            raise ValueError("active_rank_mask requires committed EP group health")
        active_rank_mask = self._watchdog_coordinator.active_rank_mask_for_combine(
            active_rank_mask_snapshot,
            requested_active_rank_mask,
        )
        # A sentinel dispatch is followed by a sentinel combine.
        use_sentinel = bool(self._dispatch_state.get("sentinel", False))
        output = torch.ops.trtllm.moe_a2a_combine(
            final_hidden_states,
            int(local_num_tokens),
            self.workspace,
            self.moe_a2a_metainfo,
            int(runtime_max_tokens_per_rank),
            self.ep_rank,
            self.ep_size,
            self.top_k,
            int(combine_payload_offset),
            bool(self.payload_in_workspace),
            use_low_precision=bool(self.use_low_precision_combine),
            enable_rank_mask=self._rank_mask_enabled,
            active_rank_mask=active_rank_mask,
            use_sentinel=use_sentinel,
        )
        if not use_sentinel:
            self._watchdog_coordinator.watch_collective(
                self._alltoall_watchdog, "combine", active_rank_mask
            )

        # Reset state for next round
        self.reset_state()

        return output

    def reset_state(self) -> None:
        """Reset the dispatch/combine state machine to ``idle``.

        Safe to call between forward passes (or from an error handler) to
        recover from a forward that called ``dispatch`` but did not reach
        ``combine`` — e.g. because an OOM aborted the forward. Without this,
        the next ``dispatch`` would raise ``dispatch called twice``.

        It resets this Python state machine only. After a sentinel dispatch
        without its combine, the native op still holds the round open, and the
        peers' rows of that round are still in flight; only re-initializing
        the workspace (``moe_a2a_initialize``, collective) recovers it. The
        in-tree caller (the warmup OOM handler) runs only without peers.
        """
        self._dispatch_state = {"phase": "idle"}

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used
        as the output of MoE kernel to avoid extra copy.
        Passing the returned tensor to combine lets the C++ op detect workspace ownership.

        Args:
            runtime_max_tokens_per_rank: Runtime max tokens per rank
            hidden_size: Hidden dimension size
            dtype: Data type

        Returns:
            Tensor view into combine input [ep_size * runtime_max_tokens_per_rank, hidden_size]
        """
        self._require_mapped()
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        if combine_payload_offset is None:
            raise RuntimeError("combine_payload_offset not found in dispatch state")

        result = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            self.moe_a2a_metainfo,
            int(self.ep_rank),
            int(self.ep_size),
            int(runtime_max_tokens_per_rank),
            dtype,
            int(hidden_size),
        )

        return result
