# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
AllGather + ReduceScatter Communication Strategy

This module implements the AllGather + ReduceScatter communication method for MoE.
This is the default fallback strategy that always works.

AllGather ALWAYS supports post-quant dispatch (quantize → allgather)
"""

from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.distributed import allgather, reducescatter
from tensorrt_llm._torch.distributed.nccl_fault_tolerance import (
    NCCL_FAULT_TOLERANCE_ENABLED,
    reconfigure_nccl_group,
    resolve_nccl_group,
)
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.mapping import Mapping

from .base import Communication


class _ActiveGroupMapping:
    """Read-only mapping view for a survivor-only NCCL communicator.

    The model-wide :class:`Mapping` remains immutable because TP/PP users may
    still be inspecting it while the MoE fallback switches membership.
    Attributes unrelated to the TP communicator delegate to the original
    mapping.
    """

    def __init__(self, mapping: Mapping, group: List[int], rank: int) -> None:
        self._mapping = mapping
        self._group = tuple(group)
        self._rank = rank

    @property
    def tp_group(self) -> List[int]:
        return list(self._group)

    @property
    def tp_size(self) -> int:
        return len(self._group)

    @property
    def tp_rank(self) -> int:
        return self._rank

    def __getattr__(self, name):
        return getattr(self._mapping, name)


class AllGatherReduceScatter(Communication):
    def __init__(
        self,
        mapping: Mapping,
    ):
        super().__init__(mapping)

        # Initialize dispatch state
        self._dispatch_state = {}
        self._original_group = tuple(mapping.tp_group)
        self._active_local_ranks = tuple(range(len(self._original_group)))
        self._active_global_group = self._original_group
        self._active_mapping = mapping
        self._raw_nccl_fault_tolerance = NCCL_FAULT_TOLERANCE_ENABLED and not mpi_disabled()

    @staticmethod
    def is_platform_supported() -> bool:
        """
        AllGather + ReduceScatter is always supported as the fallback strategy
        """
        return True

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if AllGather is feasible for the given workload at runtime.

        AllGather is always available as fallback, so this always returns True.
        """
        return True

    def abort_and_reinit(self, active_ranks: List[int], generation: Optional[int] = None) -> None:
        """Abort the current NCCL communicator and rebuild it for survivors.

        ``active_ranks`` contains global/world rank IDs from the original TP
        communication group.  A global-rank API is intentional: when MoE TP is
        enabled, one TP communicator spans several MoE-EP slices, so an
        EP-local rank number is ambiguous without the failure coordinator's
        slice identity.  The coordinator must translate its health snapshot to
        global ranks before calling this method.

        Reconfiguration is monotonic: a removed rank cannot be reintroduced by
        this Phase-1 API. Replacement-rank joins belong to process-group
        reconstruction. The failure coordinator must pass a shared monotonic
        recovery-event generation for same-membership recovery and advance it
        for every distinct attempt, including transport-only retries. Passing
        it for every recovery also deduplicates callbacks on every survivor.
        """
        if not NCCL_FAULT_TOLERANCE_ENABLED:
            raise RuntimeError(
                "NCCL error: communicator reinitialization requires TLLM_FAULT_TOLERANCE_MODE=1"
            )
        if not self._raw_nccl_fault_tolerance:
            raise NotImplementedError(
                "AllGatherReduceScatter fault tolerance requires the raw NCCL/MPI path; "
                "ProcessGroup reconstruction is not implemented"
            )

        if not active_ranks:
            raise ValueError("active_ranks must not be empty")
        if len(active_ranks) != len(set(active_ranks)):
            raise ValueError("active_ranks must not contain duplicates")
        requested = {int(rank) for rank in active_ranks}
        if not requested.issubset(self._original_group):
            raise ValueError(
                "active_ranks must be global ranks from the original TP group "
                f"{list(self._original_group)}, got {active_ranks}"
            )
        if self.mapping.rank not in requested:
            raise ValueError(f"current world rank {self.mapping.rank} is not active")

        active_global_group = tuple(rank for rank in self._original_group if rank in requested)

        # Commit Python metadata only after the coordinated native rebuild
        # succeeds. The old transport is terminal once native recovery starts,
        # even if construction of its replacement fails.
        reconfigure_nccl_group(
            self._original_group,
            active_global_group,
            torch.ops.trtllm.nccl_comm_abort_and_reinit,
            generation,
        )
        self._refresh_active_mapping()

    def _refresh_active_mapping(self) -> None:
        """Synchronize this layer with process-wide survivor membership."""
        active_global_group = tuple(resolve_nccl_group(self._original_group))
        if active_global_group == self._active_global_group:
            return
        global_rank = self.mapping.rank
        if global_rank not in active_global_group:
            raise RuntimeError(
                f"NCCL error: world rank {global_rank} is not in the active communicator"
            )
        active_global_ranks = set(active_global_group)
        active_local_ranks = tuple(
            index
            for index, world_rank in enumerate(self._original_group)
            if world_rank in active_global_ranks
        )
        active_mapping = _ActiveGroupMapping(
            self.mapping,
            list(active_global_group),
            active_global_group.index(global_rank),
        )
        self._active_local_ranks = active_local_ranks
        self._active_mapping = active_mapping
        # Publish the value used by the fast-path guard last. A concurrent
        # reader that observes this generation must also observe its mapping.
        self._active_global_group = active_global_group

    def check_async_error(self) -> None:
        """Raise a classifier-friendly exception for a watchdog failure."""
        if not NCCL_FAULT_TOLERANCE_ENABLED:
            raise RuntimeError(
                "NCCL error: async-error monitoring requires TLLM_FAULT_TOLERANCE_MODE=1"
            )
        if not self._raw_nccl_fault_tolerance:
            raise NotImplementedError(
                "AllGatherReduceScatter fault tolerance cannot inspect ProcessGroup errors; "
                "the raw NCCL/MPI watchdog is not active"
            )
        self._refresh_active_mapping()
        error = torch.ops.trtllm.nccl_comm_get_async_error(list(self._active_global_group))
        if error is not None:
            raise RuntimeError(f"NCCL error: communicator was aborted: {error}")

    def _active_sizes(self, all_rank_num_tokens: List[int]) -> List[int]:
        self._refresh_active_mapping()
        if len(all_rank_num_tokens) != len(self._original_group):
            raise ValueError(
                "all_rank_num_tokens must remain indexed by the original communicator "
                f"({len(self._original_group)} entries), got {len(all_rank_num_tokens)}"
            )
        if self._active_global_group == self._original_group:
            return all_rank_num_tokens
        return [all_rank_num_tokens[rank] for rank in self._active_local_ranks]

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
        AllGather dispatch (always post-quant dispatch)
        """
        if self._raw_nccl_fault_tolerance:
            if use_dp_padding:
                self._refresh_active_mapping()
                sizes = None
            else:
                sizes = self._active_sizes(all_rank_num_tokens)
            mapping = self._active_mapping
        else:
            sizes = None if use_dp_padding else all_rank_num_tokens
            mapping = self.mapping

        hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = allgather(
            [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales],
            mapping,
            dim=0,
            sizes=sizes,
        )

        # Store sizes for combine
        self._dispatch_state["sizes"] = sizes

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        ReduceScatter combine phase
        """
        if self._raw_nccl_fault_tolerance:
            self._refresh_active_mapping()
            mapping = self._active_mapping
        else:
            mapping = self.mapping
        outputs = reducescatter(
            final_hidden_states,
            mapping,
            dim=0,
            sizes=self._dispatch_state.get("sizes"),
        )
        return outputs
