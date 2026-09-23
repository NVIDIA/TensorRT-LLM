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
NVLINK Two-Sided AllToAll Communication Strategy

This module implements the NVLINK two-sided comm AllToAll communication method for MoE.

NVLINK Two-Sided supports post-quant dispatch for all quantization modes.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from weakref import WeakSet

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlCheckpointCommunicator, MnnvlMemory
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _collect_active_ranks
from tensorrt_llm.mapping import Mapping

from .base import Communication


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: torch.Tensor
    send_rank_count_cumsum: torch.Tensor
    send_rank_local_indices: torch.Tensor
    recv_rank_count_cumsum: torch.Tensor
    recv_rank_local_indices: torch.Tensor
    backward_recv_rank_local_indices: torch.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_prepare_workspace: MnnvlMemory = None
    moe_workspace_tensor: torch.Tensor = None
    moe_prepare_workspace_tensor: torch.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor

        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(
            mapping.moe_ep_size
        )
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(torch.uint64)
        torch.ops.trtllm.moe_initialize_workspace(
            MnnvlMoe.moe_workspace_tensor, mapping.moe_ep_rank, mapping.moe_ep_size
        )
        torch.cuda.synchronize()
        MnnvlMoe.moe_workspace.comm.barrier()
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def get_moe_prepare_workspace(mapping: Mapping):
        if MnnvlMoe.moe_prepare_workspace_tensor is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_prepare_workspace_tensor
        workspace_size_per_rank = torch.ops.trtllm.get_moe_prepare_workspace_size_per_rank(
            mapping.moe_ep_size
        )
        MnnvlMoe.moe_prepare_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_prepare_workspace_tensor = (
            MnnvlMoe.moe_prepare_workspace.as_torch_strided_tensor(torch.uint64)
        )
        return MnnvlMoe.moe_prepare_workspace_tensor

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: torch.Tensor, expert_count: int, ep_size: int
    ):
        assert expert_count % ep_size == 0, "expert_count should be divisible by ep_size"
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare_without_allgather(
        expert_ids: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ):
        (
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            gathered_expert_statics,
        ) = torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size
        # Looks like we don't need this.
        local_gather_indices = None

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            local_token_allocation_count,
        )

        return alltoall_info, gathered_expert_statics

    @staticmethod
    def mnnvl_moe_expert_static_allgather(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
    ):
        gathered_expert_ids = torch.ops.trtllm.mnnvl_moe_expert_static_allgather(
            expert_ids, workspace, ep_rank, ep_size, expert_count
        )
        return gathered_expert_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cumsum: Optional[torch.Tensor],
        gathered_expert_ids: torch.Tensor,
        gathered_scales: Optional[torch.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = torch.ops.trtllm.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size

        local_expert_ids = torch.empty(
            local_token_allocation_count, top_k, dtype=torch.int32, device=torch.device("cuda")
        )
        if gathered_scales is None:
            local_scales = None
        else:
            local_scales = torch.empty(
                local_token_allocation_count,
                top_k,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )

        torch.ops.trtllm.moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: Union[torch.Tensor, List[Optional[torch.Tensor]]],
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # Convert single tensor to list for unified handling
        is_single_tensor = not isinstance(x, list)
        if is_single_tensor:
            assert x.dim() == 2, "only 2D tensor supported, please reshape."
            x = [x]

        assert len(x) > 0, "Empty tensor list not supported"

        # Filter out None values
        valid_list = [tensor is not None for tensor in x]
        valid_tensors = [tensor for tensor in x if tensor is not None]

        if len(valid_tensors) == 0:
            # All tensors are None, return list of None
            result = [None] * len(x)
        else:
            first_dim = None
            for tensor in valid_tensors:
                # Validate dimensions of valid tensors
                assert tensor.dim() == 2, "only 2D tensor supported, please reshape."
                if first_dim is None:
                    first_dim = tensor.shape[0]
                else:
                    assert tensor.shape[0] == first_dim, (
                        f"All tensors must have the same first dimension, got {tensor.shape[0]} vs {first_dim}"
                    )

            # Process only valid tensors
            output_tensors = torch.ops.trtllm.moe_comm(
                valid_tensors,
                alltoall_info.send_rank_count_cumsum,
                alltoall_info.send_rank_local_indices,
                alltoall_info.recv_rank_count_cumsum,
                alltoall_info.recv_rank_local_indices,
                workspace,
                alltoall_info.local_token_allocation_count,
                ep_rank,
                ep_size,
            )

            # Restore None positions in output
            idx = 0
            result = []
            for is_valid in valid_list:
                if is_valid:
                    result.append(output_tensors[idx])
                    idx += 1
                else:
                    result.append(None)

        # If input was a single tensor, return a single tensor
        if is_single_tensor:
            result = result[0]

        return result

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
        use_low_precision_combine: bool = False,
        do_reduce: bool = True,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensors = torch.ops.trtllm.moe_comm(
            [x],
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            token_count * top_k,
            ep_rank,
            ep_size,
            [True],
            use_low_precision_combine,
        )
        output_tensor = output_tensors[0].reshape(token_count, top_k, x.shape[1])
        if do_reduce:
            return torch.sum(output_tensor, dim=1, keepdim=False)
        else:
            return output_tensor


class NVLinkTwoSided(Communication):
    """
    NVLINK two-sided comm AllToAll strategy.
    This implementation utilizes symmetric memory to enable peer-to-peer access between GPUs over NVLink.
    The kernel takes the role as both sender and receiver: as the sender, it puts the data into a FIFO
    quene in peer ranks' symmetric memory; as the receiver, it gets the data from the FIFO quene to the
    local buffer. This communication model is akin to NCCL's collective operations.
    The required symmetric memory size is proportional to the communication channels opened.
    """

    _INSTANCES: WeakSet = WeakSet()

    def __init__(
        self,
        mapping: Mapping,
        num_experts: int,
        num_slots: int,
        top_k: int = 1,
        use_low_precision_combine: bool = False,
        alltoall_result_do_sum: bool = False,
    ):
        super().__init__(mapping)
        if mapping.has_cp_helix():
            raise ValueError(
                "NVLinkTwoSided does not support Helix context parallelism because "
                "its MNNVL communicator covers only the tensor-parallel group"
            )

        # Store needed parameters
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.top_k = top_k

        self.use_low_precision_combine = use_low_precision_combine
        self.alltoall_result_do_sum = alltoall_result_do_sum
        # Read from environment variable, same as wideEP
        self.enable_postquant_alltoall = (
            os.environ.get("TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1") == "1"
        )

        # Invalid token expert ID (default to -1), the kernels in TRTLLM-gen is hard-coded to support -1 only.
        # CutlassFusedMoE kernels support any invalid value.
        self.invalid_token_expert_id: int = -1

        # Initialize NVLINK workspaces
        MnnvlMemory.initialize()
        self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(mapping)
        self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(mapping)

        # Initialize dispatch state
        self._dispatch_state = {}
        self._INSTANCES.add(self)

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if NVLINK two-sided comm is supported on current hardware.
        """
        return MnnvlMemory.supports_mnnvl()

    def supports_post_quant_dispatch(self) -> bool:
        """
        NVLINK two-sided comm supports post-quant for all modes.
        """
        return self.enable_postquant_alltoall

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if NVLINK two-sided comm is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
        """
        return True

    def checkpoint_resource_key(self) -> int:
        """Identify the process-global TRT-native two-sided workspaces."""
        return id(MnnvlMoe)

    def checkpoint_prepare(self) -> None:
        """Detach TRT-native two-sided workspaces after global quiescence."""
        workspaces = (MnnvlMoe.moe_workspace, MnnvlMoe.moe_prepare_workspace)
        if all(workspace is None or not workspace.mapped for workspace in workspaces):
            MnnvlMoe.checkpoint_prepare()
            return
        local_clients_idle = not any(instance._dispatch_state for instance in self._INSTANCES)
        workspace = MnnvlMoe.moe_workspace
        assert workspace is not None
        comm = workspace.comm
        if comm is None:
            raise RuntimeError("MNNVL workspace communicator is not initialized")
        try:
            active_ranks = _collect_active_ranks(
                comm,
                local_clients_idle=local_clients_idle,
                expected_size=self.ep_size,
            )
        except TimeoutError:
            for candidate in workspaces:
                if candidate is not None:
                    candidate.checkpoint_fail_closed()
            raise
        if active_ranks:
            raise RuntimeError(
                f"Cannot checkpoint during an active MoE All-to-All phase on ranks {active_ranks}"
            )
        MnnvlMoe.checkpoint_prepare()

    def checkpoint_restore(
        self,
        comm: MnnvlCheckpointCommunicator | None = None,
    ) -> None:
        """Restore TRT-native two-sided workspaces and protocol state.

        Args:
            comm: An mpi4py-like communicator exposing ``Get_rank()``,
                ``Get_size()``, ``allgather()``, and ``barrier()``. Its local
                rank and size must match the communicator used for the
                original allocations. Every rank must call this method
                symmetrically.
        """
        workspace = MnnvlMoe.moe_workspace or MnnvlMoe.moe_prepare_workspace
        if comm is None and workspace is not None:
            comm = workspace.comm
        if comm is None:
            raise RuntimeError("MNNVL workspace communicator is not initialized")
        restore_required = any(
            workspace is not None and not workspace.mapped
            for workspace in (MnnvlMoe.moe_workspace, MnnvlMoe.moe_prepare_workspace)
        )
        MnnvlMoe.checkpoint_restore(comm)
        if not restore_required:
            return
        for instance in self._INSTANCES:
            instance._dispatch_state = {}

    def prepare_dispatch(
        self,
        token_selected_slots: torch.Tensor,
        all_rank_num_tokens: List[int],
        local_statistic_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        NVLINK two-sided comm prepare dispatch: gather EPLB statistics and prepare alltoall_info.
        """
        MnnvlMoe.require_mapped()
        all_rank_max_num_tokens = max(all_rank_num_tokens)
        top_k = token_selected_slots.shape[1]

        # Call NVLINK prepare to get alltoall_info and gather EPLB statistics
        alltoall_info, gathered_local_statistic_tensor = (
            MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                token_selected_slots,
                local_statistic_tensor,
                self.alltoall_prepare_workspace,
                all_rank_max_num_tokens,
                self.ep_rank,
                self.ep_size,
                self.num_experts,
                self.num_slots,
                top_k,
            )
        )

        # Store alltoall_info in dispatch_state for use in dispatch()
        self._dispatch_state["alltoall_info"] = alltoall_info

        return gathered_local_statistic_tensor

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
        NVLINK two-sided comm dispatch (post-quant, uses alltoall_info from prepare_dispatch).
        """
        MnnvlMoe.require_mapped()
        # Read alltoall_info from dispatch_state (set by prepare_dispatch)
        alltoall_info = self._dispatch_state.get("alltoall_info")
        if alltoall_info is None:
            raise ValueError(
                "NVLinkTwoSided dispatch requires prepare_dispatch() to be called first"
            )

        all_rank_max_num_tokens = max(all_rank_num_tokens)
        original_token_count = hidden_states.shape[0]  # Store for combine
        top_k = token_selected_slots.shape[1]

        # Dispatch quantized data using AllToAll
        hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = (
            MnnvlMoe.mnnvl_moe_alltoallv(
                [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales],
                alltoall_info,
                self.alltoall_workspace,
                self.ep_rank,
                self.ep_size,
            )
        )

        # Set expert IDs after alltoall
        torch.ops.trtllm.memset_expert_ids(
            token_selected_slots,
            alltoall_info.recv_rank_count_cumsum,
            all_rank_max_num_tokens,
            top_k,
            self.invalid_token_expert_id,
            self.ep_size,
        )

        # Store original_token_count for combine (alltoall_info already stored in prepare_dispatch)
        self._dispatch_state["original_token_count"] = original_token_count

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        NVLINK two-sided comm combine - reads from self._dispatch_state.
        """
        MnnvlMoe.require_mapped()
        if isinstance(final_hidden_states, list):
            final_hidden_states = final_hidden_states[0]

        final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
            final_hidden_states,
            self._dispatch_state["alltoall_info"],
            self.alltoall_workspace,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=self.top_k,
            token_count=self._dispatch_state["original_token_count"],
            use_low_precision_combine=self.use_low_precision_combine,
            do_reduce=self.alltoall_result_do_sum,
        )

        self._dispatch_state = {}
        return final_hidden_states
