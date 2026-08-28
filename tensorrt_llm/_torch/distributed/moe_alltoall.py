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
"""
MoE All-to-All Operations

This module provides a high-level interface for MoE all-to-all dispatch and combine operations
with proper workspace management and synchronization.
"""

# ruff: noqa: E501

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from tensorrt_llm._mnnvl_utils import CftMnnvlMemory, MnnvlMemory
from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S, ActiveRankMaskSnapshot,
    AlltoAllWatchdog, AlltoAllWatchdogCoordinator, AlltoAllWatchdogTimeout,
    EPGroupHealthLike, reject_rank_mask_cuda_graph_capture)
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up

_CFT_DEFAULT_MAX_BATCH_FOR_DISPATCH = 128
_CFT_MAX_BATCH_FOR_DISPATCH_ENV = "TRTLLM_MOE_A2A_CFT_MAX_BATCH_FOR_DISPATCH"
# CFT combine wins at small/medium batch and ties/regresses at large batch, so
# it is gated by the same per-call token-count threshold as dispatch.
_CFT_DEFAULT_MAX_BATCH_FOR_COMBINE = 128
_CFT_MAX_BATCH_FOR_COMBINE_ENV = "TRTLLM_MOE_A2A_CFT_MAX_BATCH_FOR_COMBINE"
FORCE_CFT_ENV = "TRTLLM_MOE_A2A_FORCE_CFT"
_CFT_ALIGNMENT_BYTES = 16


def get_force_cft() -> bool | None:
    value = os.environ.get(FORCE_CFT_ENV)
    if value == "0":
        return False
    if value == "1":
        return True
    return None


def should_use_cft(
    can_use_cft: bool,
    force_cft: bool | None,
    max_batch: int | None,
    runtime_max_tokens_per_rank: int,
) -> bool:
    if not can_use_cft:
        return False
    if force_cft is not None:
        return force_cft
    if max_batch is None:
        return True
    return runtime_max_tokens_per_rank <= max_batch


def _use_cft_for_dispatch_payloads(use_cft: bool,
                                   payloads: list[torch.Tensor]) -> bool:
    if not use_cft:
        return False
    for payload_index, payload in enumerate(payloads):
        bytes_per_token = payload.shape[1] * payload.element_size()
        if bytes_per_token % _CFT_ALIGNMENT_BYTES != 0:
            tllm_logger.warning_once(
                "CFT counted writes disabled: dispatch payload "
                f"{payload_index} has {bytes_per_token} bytes per token, which is not "
                f"{_CFT_ALIGNMENT_BYTES}-byte aligned. Falling back to fence-based dispatch.",
                key=
                f"moe_a2a_cft_dispatch_alignment_{payload_index}_{bytes_per_token}",
            )
            return False
    return True


def _use_cft_for_combine_payload(use_cft: bool, payload: torch.Tensor,
                                 use_low_precision: bool) -> bool:
    if not use_cft:
        return False
    wire_element_size = 1 if use_low_precision else payload.element_size()
    bytes_per_token = payload.shape[-1] * wire_element_size
    if bytes_per_token % _CFT_ALIGNMENT_BYTES != 0:
        tllm_logger.warning_once(
            "CFT counted writes disabled: combine payload has "
            f"{bytes_per_token} bytes per token, which is not "
            f"{_CFT_ALIGNMENT_BYTES}-byte aligned. Falling back to fence-based combine.",
            key=f"moe_a2a_cft_combine_alignment_{bytes_per_token}",
        )
        return False
    return True


def _get_cft_max_batch(env_name: str, default: int) -> int:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default
    try:
        threshold = int(env_value)
    except ValueError as e:
        raise ValueError(f"{env_name} must be an integer") from e
    if threshold < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return threshold


def _get_cft_max_batch_for_dispatch() -> int | None:
    return _get_cft_max_batch(_CFT_MAX_BATCH_FOR_DISPATCH_ENV,
                              _CFT_DEFAULT_MAX_BATCH_FOR_DISPATCH)


def _get_cft_max_batch_for_combine() -> int | None:
    return _get_cft_max_batch(_CFT_MAX_BATCH_FOR_COMBINE_ENV,
                              _CFT_DEFAULT_MAX_BATCH_FOR_COMBINE)


@dataclass
class _A2AState:
    phase: str = "idle"  # idle | dispatched
    local_num_tokens: int | None = None
    combine_payload_offset: int | None = None
    eplb_gathered_stats: torch.Tensor | None = None
    active_rank_mask_snapshot: ActiveRankMaskSnapshot | None = None


class MoeAlltoAll:
    """
    Manages MoE All-to-All operations with proper workspace allocation and synchronization.

    This class encapsulates the dispatch and combine operations, managing workspace memory
    and auxiliary data structures needed for cross-GPU communication.
    """

    # Shared workspace/memory across the process, separated by handle type.
    _WORKSPACES: Dict[bool, dict] = {}

    _METAINFO_INDEX: Dict[str, int] | None = None

    @staticmethod
    def get_aux_data_size(
        ep_size: int,
        max_num_tokens: int,
        eplb_stats_num_experts: Optional[int] = None,
        can_use_cft_counted_writes: bool = False,
    ) -> int:
        return torch.ops.trtllm.moe_a2a_get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts,
            can_use_cft_counted_writes)

    @staticmethod
    def calculate_required_workspace_size(
            ep_size: int,
            top_k: int,
            max_num_tokens: int,
            hidden_size: int,
            dtype: torch.dtype,
            eplb_stats_num_experts: Optional[int] = None,
            extra_payload_bytes_per_token: int = 0,
            can_use_cft_counted_writes: bool = False) -> int:
        element_size = dtype.itemsize

        # Auxiliary data size
        workspace_size = MoeAlltoAll.get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts,
            can_use_cft_counted_writes)

        # Dispatch needs workspace for [ep_size, max_tokens] tokens,
        # but due to the variety of quantization recipes, we cannot know the exact size, so we conservatively estimate assuming no quantization.
        # Meanwhile, we consider the alignment requirement as in moeA2ADispatchOp and moeA2ACombineOp.
        # (Unquantized) token hidden states
        workspace_size += ep_size * max_num_tokens * hidden_size * element_size
        workspace_size = pad_up(workspace_size, 128)
        # token_selected_experts
        workspace_size += ep_size * max_num_tokens * top_k * 4
        workspace_size = pad_up(workspace_size, 128)
        # token_final_scales
        workspace_size += ep_size * max_num_tokens * top_k * 4
        workspace_size = pad_up(workspace_size, 128)
        # extra payload bytes per token
        workspace_size += ep_size * max_num_tokens * extra_payload_bytes_per_token
        workspace_size = pad_up(workspace_size, 128)

        # Required workspace for combine [ep_size, max_tokens] tokens
        workspace_size += ep_size * max_num_tokens * hidden_size * element_size
        workspace_size = pad_up(workspace_size, 128)

        # CFT combine: dedicated combine RECEIVE region C (peer pushes land here;
        # prepareCombine never touches it -> no proxy aliasing). Same size as the combine region.
        if can_use_cft_counted_writes:
            workspace_size += ep_size * max_num_tokens * hidden_size * element_size
            workspace_size = pad_up(workspace_size, 128)

        return workspace_size

    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        # TODO: Can we avoid such code duplication?
        if cls._METAINFO_INDEX is None:
            thop = _tllm_internal.thop
            cls._METAINFO_INDEX = {
                "FLAG_VAL_OFFSET_INDEX":
                int(thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX),
                "LOCAL_TOKEN_COUNTER_OFFSET_INDEX":
                int(thop.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX),
                "SEND_COUNTERS_OFFSET_INDEX":
                int(thop.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX),
                "RECV_COUNTERS_OFFSET_INDEX":
                int(thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX),
                "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX":
                int(thop.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX),
                "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX":
                int(thop.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX),
                "DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX":
                int(thop.MOE_A2A_DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX),
                "TOPK_TARGET_RANKS_OFFSET_INDEX":
                int(thop.MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX),
                "TOPK_SEND_INDICES_OFFSET_INDEX":
                int(thop.MOE_A2A_TOPK_SEND_INDICES_OFFSET_INDEX),
                "EPLB_GATHERED_STATS_OFFSET_INDEX":
                int(thop.MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX),
                "PAYLOAD_DATA_OFFSET_INDEX":
                int(thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX),
                "NUM_METAINFO_FIELDS":
                int(thop.MOE_A2A_NUM_METAINFO_FIELDS),
            }

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: int,
        top_k: int,
        num_slots: int,
        workspace_size_per_rank: int,
        num_experts: Optional[int] = None,
        can_use_cft_counted_writes: bool = False,
        ep_group_health: Optional[EPGroupHealthLike] = None,
        alltoall_watchdog_timeout_s: Optional[float] = None,
        alltoall_watchdog_poll_interval_s:
        float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
        alltoall_watchdog_on_timeout: Optional[Callable[
            [AlltoAllWatchdogTimeout], None]] = None,
    ) -> None:
        """
        Initialize MoeAlltoAll with workspace allocation.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            max_num_tokens: Maximum number of tokens supported. Should be ModelConfig.max_num_tokens.
            workspace_size_per_rank: Size of workspace per rank in bytes
            num_slots: Number of routing slots (token_selected_experts values are in [0, num_slots)).
                Note: The terminology is mapped to `num_experts` in this class and the kernels.
            num_experts: (Optional) Number of experts for EPLB stats (must be <= num_slots). DO NOT provide this parameter if EPLB is not enabled.
                Note: The terminology is mapped to `eplb_stats_num_experts` in this class and the kernels.
            can_use_cft_counted_writes: If True, allow CFT handle-based counted
                writes (fabric.try_put.counted via Logical Endpoints) for dispatch.
                Requires sm_100+ (Blackwell or later), a build against CUDA 13.4+, an
                NVLink fabric, and a driver exporting the CUDA logical endpoint API.
            ep_group_health: Optional read-only committed EP membership. When present, rank-mask handling is
                enabled in the CUDA kernels, and its mask defines the peers expected by the watchdog. Timeout
                detection never mutates it. CUDA graphs are rejected until membership-scoped recapture lands.
            alltoall_watchdog_timeout_s: Optional timeout for the host-side AlltoAll watchdog. If None, the
                watchdog is disabled.
            alltoall_watchdog_poll_interval_s: Poll interval for the watchdog thread.
            alltoall_watchdog_on_timeout: Optional callback invoked when the watchdog reports suspects.
        """
        # Check for environment variable override
        workspace_mb_env = os.environ.get("TRTLLM_MOE_A2A_WORKSPACE_MB")
        if workspace_mb_env:
            workspace_size_env = int(workspace_mb_env) * 1024 * 1024
            tllm_logger.warning(
                f"Overriding automatically calculated workspace_size_per_rank ({workspace_size_per_rank} bytes) with "
                f"TRTLLM_MOE_A2A_WORKSPACE_MB={workspace_mb_env} ({workspace_size_env} bytes)."
                f"Automatically calculated workspace_size_per_rank is conservatively large, please only consider overriding it if you have a specific reason."
            )
            workspace_size_per_rank = workspace_size_env

        # Initialize constants from C++
        self._init_constants()

        # Initialize or reuse workspace
        MnnvlMemory.initialize()

        self.workspace_size_per_rank = workspace_size_per_rank
        self.max_num_tokens = max_num_tokens
        self.ep_size = mapping.moe_ep_size
        self.ep_rank = mapping.moe_ep_rank

        self.top_k = top_k
        self.num_experts = num_slots

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive int")
        if not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError("num_slots must be a positive int")

        if num_experts is not None:
            assert num_experts > 0 and num_experts <= num_slots, "num_experts must be in (0, num_slots]"
            tllm_logger.info(
                "NVLinkOneSided AlltoAll: EPLB is enabled, with num_slots="
                f"{num_slots} and num_experts={num_experts}")
        self.enable_eplb = num_experts is not None
        self.eplb_stats_num_experts = num_experts
        self._force_cft = get_force_cft()
        if self._force_cft is False:
            can_use_cft_counted_writes = False
        self.can_use_cft_counted_writes = can_use_cft_counted_writes
        if self._force_cft is None:
            self.cft_max_batch_for_dispatch = _get_cft_max_batch_for_dispatch()
            self.cft_max_batch_for_combine = _get_cft_max_batch_for_combine()
        else:
            self.cft_max_batch_for_dispatch = None
            self.cft_max_batch_for_combine = None

        workspace_key = self.can_use_cft_counted_writes
        workspace_entry = self._WORKSPACES.get(workspace_key)
        memory_cls = CftMnnvlMemory if self.can_use_cft_counted_writes else MnnvlMemory

        if workspace_entry is None:
            tllm_logger.info(
                f"NVLinkOneSided AlltoAll: Allocating workspace with size {workspace_size_per_rank} bytes. ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, max_num_tokens: {self.max_num_tokens}"
            )
            mnnvl_mem = memory_cls(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = torch.ops.trtllm.moe_a2a_initialize(
                workspace, self.ep_rank, self.ep_size, self.max_num_tokens,
                self.eplb_stats_num_experts, self.can_use_cft_counted_writes)
            workspace_entry = {
                "workspace_size_per_rank": workspace_size_per_rank,
                "max_num_tokens": self.max_num_tokens,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "eplb_stats_num_experts": self.eplb_stats_num_experts,
                "can_use_cft_counted_writes": self.can_use_cft_counted_writes,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
                "cft_initialized": False,
            }
            MoeAlltoAll._WORKSPACES[workspace_key] = workspace_entry
        else:
            assert workspace_entry[
                "workspace_size_per_rank"] == workspace_size_per_rank, "mistakenly reusing workspace with different workspace_size_per_rank"
            assert workspace_entry[
                "max_num_tokens"] == self.max_num_tokens, "mistakenly reusing workspace with different max_num_tokens"
            assert workspace_entry[
                "ep_rank"] == self.ep_rank, "mistakenly reusing workspace with different ep_rank"
            assert workspace_entry[
                "ep_size"] == self.ep_size, "mistakenly reusing workspace with different ep_size"
            assert workspace_entry[
                "eplb_stats_num_experts"] == self.eplb_stats_num_experts, (
                    "reuse workspace with different eplb_stats_num_experts")
            assert workspace_entry[
                "can_use_cft_counted_writes"] == self.can_use_cft_counted_writes, "reuse workspace with different CFT mode"

        self.mnnvl_mem = workspace_entry["mnnvl_mem"]
        self.workspace = workspace_entry["workspace"]
        self.metainfo = workspace_entry["metainfo"]
        # Internal state
        self._state: _A2AState = _A2AState()
        self.ep_group_health = ep_group_health
        # Keep the kernel specialization stable for this communicator's lifetime.
        self._rank_mask_enabled = ep_group_health is not None
        workspace_state = workspace_entry
        self._workspace_state = workspace_state
        metainfo_index = self._METAINFO_INDEX
        assert metainfo_index is not None
        self._watchdog_coordinator = AlltoAllWatchdogCoordinator(
            workspace_state=workspace_state,
            workspace=self.workspace,
            metainfo=self.metainfo,
            metainfo_index=metainfo_index,
            ep_rank=self.ep_rank,
            health=self.ep_group_health,
        )
        self._destroyed = False
        self._alltoall_watchdog: AlltoAllWatchdog | None = None
        if (alltoall_watchdog_timeout_s is None
                and self.ep_group_health is not None):
            alltoall_watchdog_timeout_s = DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
        if alltoall_watchdog_timeout_s is not None:
            self._alltoall_watchdog = self._watchdog_coordinator.acquire_watchdog(
                ep_size=self.ep_size,
                timeout_s=alltoall_watchdog_timeout_s,
                poll_interval_s=alltoall_watchdog_poll_interval_s,
                on_timeout=alltoall_watchdog_on_timeout,
            )

    def destroy(self) -> None:
        """Stop background watchdog resources owned by this wrapper."""
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        watchdog = getattr(self, "_alltoall_watchdog", None)
        if watchdog is not None:
            self._watchdog_coordinator.release_watchdog(watchdog)
            self._alltoall_watchdog = None

    def __del__(self) -> None:
        if not sys.is_finalizing():
            self.destroy()

    def use_cft_for_dispatch(self, runtime_max_tokens_per_rank: int) -> bool:
        return should_use_cft(self.can_use_cft_counted_writes, self._force_cft,
                              self.cft_max_batch_for_dispatch,
                              runtime_max_tokens_per_rank)

    def use_cft_for_combine(self, runtime_max_tokens_per_rank: int) -> bool:
        return should_use_cft(self.can_use_cft_counted_writes, self._force_cft,
                              self.cft_max_batch_for_combine,
                              runtime_max_tokens_per_rank)

    def cft_initialize(self) -> None:
        """
        Initialize CFT Logical Endpoints by binding the LE to the MNNVL workspace.
        Must be called once before the first dispatch when can_use_cft_counted_writes=True.
        """
        if not self.can_use_cft_counted_writes:
            raise ValueError(
                "cft_initialize called but can_use_cft_counted_writes is False")
        torch.ops.trtllm.moe_a2a_cft_initialize(
            self.workspace,
            self.mnnvl_mem.local_mem_handle,
            int(self.workspace.size(1)),
            self.ep_rank,
            self.ep_size,
        )
        tllm_logger.info(
            f"CFT LE initialized (workspace-bound): ep_rank={self.ep_rank}, ep_size={self.ep_size}"
        )

    def dispatch(self,
                 token_selected_experts: torch.Tensor,
                 input_payloads: list[torch.Tensor],
                 runtime_max_tokens_per_rank: int,
                 invalid_token_expert_id: Optional[int] = None,
                 expert_id_payload_index: Optional[int] = None,
                 eplb_local_stats: Optional[torch.Tensor] = None,
                 active_rank_mask: Optional[torch.Tensor] = None):
        """
        Perform MoE all-to-all dispatch operation.

        Args:
            token_selected_experts: [local_num_tokens, top_k] tensor of expert indices
            input_payloads: List of tensors to dispatch, each has shape [local_num_tokens, payload_num_elements_per_token]
            runtime_max_tokens_per_rank: Maximum of the number of tokens of each DP rank's local batch.
            invalid_token_expert_id: If not None, set the token_selected_experts of the invalid tokens to this expert id. This is used to notify the MoE to skip these tokens for GroupGEMM.
            expert_id_payload_index: The index of token_selected_experts in the input_payloads. Must be provided if invalid_token_expert_id is not None.
            eplb_local_stats: (Optional) [num_experts] tensor containing local statistics for EPLB
            active_rank_mask: Optional uint64 CPU tensor overriding committed membership in rank-mask mode. When
                omitted, the committed mask and generation are captured together. Combine reuses that mask and
                fails closed if the committed generation changes first. The masked kernel rejects inactive routes
                before remote access; that sentinel is an internal abort artifact, not valid model output.

        Returns:
            recv_tensors: List of tensors received, each has shape [ep_size, max_tokens_per_rank, payload_num_elements_per_token]
        """
        assert self._state.phase == "idle", "dispatch called twice without an intervening combine"
        reject_rank_mask_cuda_graph_capture(self._rank_mask_enabled)
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, "runtime_max_tokens_per_rank must not exceed max_num_tokens"
        can_use_cft_for_dispatch = self.use_cft_for_dispatch(
            runtime_max_tokens_per_rank)
        can_use_cft_for_dispatch = _use_cft_for_dispatch_payloads(
            can_use_cft_for_dispatch, input_payloads)
        # Auto-initialize CFT LEs on first dispatch only
        if self.can_use_cft_counted_writes and not self._workspace_state.get(
                'cft_initialized', False):
            self.cft_initialize()
            self._workspace_state['cft_initialized'] = True
        if eplb_local_stats is not None:
            assert self.enable_eplb, "eplb_local_stats provided but enable_eplb is False"
            assert eplb_local_stats.dim(
            ) == 1, "eplb_local_stats must be a 1D tensor"
            assert eplb_local_stats.size(
                0
            ) == self.eplb_stats_num_experts, "eplb_local_stats size must match eplb_stats_num_experts"
        can_fuse_sanitize = (can_use_cft_for_dispatch
                             and invalid_token_expert_id is not None
                             and expert_id_payload_index is not None)

        requested_active_rank_mask = active_rank_mask
        if (not self._rank_mask_enabled
                and requested_active_rank_mask is not None):
            raise ValueError(
                "active_rank_mask requires committed EP group health")
        active_rank_mask_snapshot = self._watchdog_coordinator.capture_active_rank_mask(
            requested_active_rank_mask)
        active_rank_mask = active_rank_mask_snapshot.active_rank_mask
        recv_tensors, combine_payload_offset, eplb_gathered_stats = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            input_payloads,
            self.workspace,
            self.metainfo,
            runtime_max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self.num_experts,
            eplb_local_stats,
            can_use_cft_for_dispatch,
            expert_id_payload_index if can_fuse_sanitize else None,
            invalid_token_expert_id if can_fuse_sanitize else None,
            self._rank_mask_enabled,
            active_rank_mask,
        )
        self._watchdog_coordinator.watch_collective(self._alltoall_watchdog,
                                                    "dispatch",
                                                    active_rank_mask)
        if eplb_gathered_stats.numel() == 0:
            eplb_gathered_stats = None

        # Update state together after successful dispatch
        self._state.local_num_tokens = token_selected_experts.size(0)
        self._state.combine_payload_offset = combine_payload_offset
        self._state.eplb_gathered_stats = eplb_gathered_stats
        self._state.active_rank_mask_snapshot = active_rank_mask_snapshot
        self._state.phase = "dispatched"

        if invalid_token_expert_id is not None and not can_fuse_sanitize:
            assert expert_id_payload_index is not None, "expert_id_payload_index must be provided if invalid_token_expert_id is not None"
            # Sanitize expert IDs for invalid tokens directly on the recv tensor payload
            recv_token_selected_experts = recv_tensors[expert_id_payload_index]
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                recv_token_selected_experts,
                self.workspace,
                self.metainfo,
                self.ep_rank,
                invalid_token_expert_id,
            )

        return recv_tensors

    def combine(
        self,
        payload,
        runtime_max_tokens_per_rank: int,
        payload_in_workspace: bool = False,
        use_low_precision_combine: bool = False,
        active_rank_mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform MoE all-to-all combine operation.

        Args:
            payload: [ep_size, max_tokens_per_rank, num_elements_per_token] tensor to combine. The dtype must be float32, bfloat16 or float16.
            runtime_max_tokens_per_rank: Maximum of the number of tokens of each DP rank's local batch.
            payload_in_workspace: If True, 'payload' is a view into 'workspace' at 'combine_payload_offset' and no staging copy is needed. If False, the op stages 'payload' into the workspace region before combining. Callers that cannot direct the MoE kernel's output into the workspace must leave this False.
            use_low_precision_combine: If True, quantize the combine payload to FP8 for NVLink transfer (halves NVLink bandwidth usage, output precision is preserved).
            active_rank_mask: Optional uint64 CPU tensor. In rank-mask mode, it must match the mask captured by
                dispatch for this collective when supplied. A committed-generation change since dispatch aborts
                the collective epoch.

        Returns:
            combined_output: [local_num_tokens, num_elements_per_token] tensor of combined results
        """
        assert self._state.phase == "dispatched", "combine called before a successful dispatch"
        reject_rank_mask_cuda_graph_capture(self._rank_mask_enabled)
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, "runtime_max_tokens_per_rank must not exceed max_num_tokens"

        active_rank_mask_snapshot = self._state.active_rank_mask_snapshot
        assert active_rank_mask_snapshot is not None
        requested_active_rank_mask = active_rank_mask
        if (not self._rank_mask_enabled
                and requested_active_rank_mask is not None):
            raise ValueError(
                "active_rank_mask requires committed EP group health")
        active_rank_mask = self._watchdog_coordinator.active_rank_mask_for_combine(
            active_rank_mask_snapshot, requested_active_rank_mask)
        use_cft_for_combine = _use_cft_for_combine_payload(
            self.use_cft_for_combine(runtime_max_tokens_per_rank), payload,
            use_low_precision_combine)
        output = torch.ops.trtllm.moe_a2a_combine(
            payload, self._state.local_num_tokens, self.workspace,
            self.metainfo, runtime_max_tokens_per_rank, self.ep_rank,
            self.ep_size, self.top_k, self._state.combine_payload_offset,
            payload_in_workspace, use_low_precision_combine,
            use_cft_for_combine, self._rank_mask_enabled, active_rank_mask)
        self._watchdog_coordinator.watch_collective(self._alltoall_watchdog,
                                                    "combine", active_rank_mask)

        # Reset state for next round
        self.reset_state()

        return output

    def reset_state(self) -> None:
        """Reset the dispatch/combine state machine to ``idle``.

        Safe to call between forward passes (or from an error handler) to
        recover from a forward that called ``dispatch`` but did not reach
        ``combine`` — e.g. because an OOM aborted the forward. Without this,
        the next ``dispatch`` would fire the assert at line 239.
        """
        self._state = _A2AState()

    def get_combine_payload_tensor_in_workspace(
            self, runtime_max_tokens_per_rank: int, hidden_size: int,
            dtype: torch.dtype) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used as the output of MoE kernel to avoid extra copy.
        Passing the returned tensor to combine lets the C++ op detect workspace ownership.
        """
        if self._state.phase != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            self.ep_rank,
            self.ep_size,
            runtime_max_tokens_per_rank,
            self._state.combine_payload_offset,
            dtype,
            hidden_size,
        )
