# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
DeepEP Communication Strategy

This module implements the DeepEP (Deep Expert Parallelism) communication method for MoE.
DeepEP supports both pre-quant and post-quant dispatch modes.
"""

import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.fused_moe.deep_ep_utils import buffer_pool, deep_ep_installed
from tensorrt_llm._utils import local_mpi_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .base import Communication


class DeepEP(Communication):
    """
    DeepEP strategy supporting both pre-quant and post-quant dispatch

    """

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        hidden_size: int,
        weight_dtype: torch.dtype,
        quant_config: QuantConfig,
        expert_size_per_partition: int = 0,
        use_cuda_graph: bool = False,
    ):
        super().__init__(mapping)

        # Check if DeepEP is feasible for the given number of ranks
        if not self._is_deepep_feasible(mapping.moe_ep_size):
            raise RuntimeError(
                f"DeepEP is not feasible for {mapping.moe_ep_size} ranks. "
                f"DeepEP supports: "
                f"1) Intranode: 2, 4, or 8 ranks; "
                f"2) Internode: 2, 4, 8, or 16 nodes with 8 ranks per node."
            )

        # Store needed parameters
        self.num_slots = num_slots
        self.hidden_size = hidden_size
        self.weight_dtype = weight_dtype
        self.quant_config = quant_config

        self.expert_size_per_partition = expert_size_per_partition
        self.use_cuda_graph = use_cuda_graph
        self.enable_postquant_alltoall = (
            os.environ.get("TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1") == "1"
        )

        # Initialize DeepEP buffer
        self.deep_ep_buffer = buffer_pool.get_buffer(mapping)
        self.deep_ep_buffer.reserve(hidden_size, weight_dtype)

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if DeepEP is supported on the current platform
        """
        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") != "1":
            return False
        return deep_ep_installed

    @staticmethod
    def _is_deepep_feasible(num_ranks: int) -> bool:
        """
        Check if DeepEP is feasible for the given number of ranks

        DeepEP supports two modes:
        1. Intranode: Single node with 2, 4, or 8 ranks
        2. Internode: 2, 4, 8, or 16 nodes with 8 ranks per node
        """
        NUM_INTRANODE_SUPPORTED_RANKS = {2, 4, 8}
        REQUIRED_LOCAL_MPI_SIZE = 8
        NUM_INTERNODE_SUPPORTED_RDMA_RANKS = {2, 4, 8, 16}
        mpi_size = local_mpi_size()

        # Intranode cases
        if num_ranks == mpi_size and num_ranks in NUM_INTRANODE_SUPPORTED_RANKS:
            return True

        # Internode cases
        if mpi_size != REQUIRED_LOCAL_MPI_SIZE:
            return False
        num_rdma_nodes = num_ranks // mpi_size
        return num_rdma_nodes in NUM_INTERNODE_SUPPORTED_RDMA_RANKS

    def supports_post_quant_dispatch(self) -> bool:
        """
        DeepEP supports post-quant dispatch only for nvfp4
        """
        has_nvfp4 = self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4()

        return self.enable_postquant_alltoall and has_nvfp4

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if DeepEP is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and weight dtype compatibility.
        """
        if num_chunks > 1:
            return False
        if self.weight_dtype != torch.bfloat16:
            return False
        return True

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
        DeepEP dispatch
        """
        all_rank_max_num_tokens = max(all_rank_num_tokens)

        if not self.supports_post_quant_dispatch():
            # Pre-quant dispatch (unquantized data)
            (
                hidden_states,
                recv_topk_idx,
                token_final_scales,
                num_recv_tokens_per_expert_list,
                deep_ep_handle,
            ) = self.deep_ep_buffer.dispatch(
                hidden_states,
                token_selected_slots,
                token_final_scales,
                self.num_slots,
                self.expert_size_per_partition * self.ep_rank,
                all_rank_max_num_tokens,
                self.ep_size,
                self.use_cuda_graph,
            )

            padded, hidden_states, _, token_selected_slots, token_final_scales = (
                self._pad_empty_recv_tensors(hidden_states, None, recv_topk_idx, token_final_scales)
            )

            # Store dispatch state for combine
            self._dispatch_state = {
                "deep_ep_handle": deep_ep_handle,
                "padded": padded,
            }

        else:
            # Post-quant dispatch (quantized data, nvfp4 only)

            if hidden_states_sf is not None:
                # Adapter between `hidden_states_sf` and DeepEP
                # TODO: remove the adapter by adding dtype support to DeepEP
                sf_dtype = hidden_states_sf.dtype
                hidden_states_sf = hidden_states_sf.view(torch.float32)

            (
                (hidden_states, hidden_states_sf),
                recv_topk_idx,
                token_final_scales,
                num_recv_tokens_per_expert_list,
                deep_ep_handle,
            ) = self.deep_ep_buffer.dispatch(
                (hidden_states, hidden_states_sf),
                token_selected_slots,
                token_final_scales,
                self.num_slots,
                self.expert_size_per_partition * self.ep_rank,
                all_rank_max_num_tokens,
                self.ep_size,
                self.use_cuda_graph,
            )

            padded, hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = (
                self._pad_empty_recv_tensors(
                    hidden_states, hidden_states_sf, recv_topk_idx, token_final_scales
                )
            )

            if hidden_states_sf is not None:
                hidden_states_sf = hidden_states_sf.view(sf_dtype)

            # Store dispatch state for combine
            self._dispatch_state = {
                "deep_ep_handle": deep_ep_handle,
                "padded": padded,
            }

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        DeepEP combine - reads from self._dispatch_state
        """
        deep_ep_handle = self._dispatch_state["deep_ep_handle"]
        padded = self._dispatch_state["padded"]

        final_hidden_states = self._unpad_tensors(padded, final_hidden_states)
        final_hidden_states = self.deep_ep_buffer.combine(final_hidden_states, deep_ep_handle)

        return final_hidden_states

    def _pad_empty_recv_tensors(
        self,
        x: torch.Tensor,
        x_sf: Optional[torch.Tensor],
        recv_topk_idx: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> Tuple[bool, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Pad empty recv tensors to avoid zero-size tensor issues
        """
        if x.shape[0] == 0:
            padded = True
            x = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
            if x_sf is not None:
                x_sf = torch.zeros((1, x_sf.shape[1]), dtype=x_sf.dtype, device=x_sf.device)
            recv_topk_idx = torch.full(
                (1, recv_topk_idx.shape[1]),
                self.num_slots,
                dtype=recv_topk_idx.dtype,
                device=recv_topk_idx.device,
            )
            token_final_scales = torch.ones(
                (1, token_final_scales.shape[1]),
                dtype=token_final_scales.dtype,
                device=token_final_scales.device,
            )
        else:
            padded = False
        return padded, x, x_sf, recv_topk_idx, token_final_scales

    def _unpad_tensors(self, padded: bool, final_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Unpad tensors if they were padded in dispatch
        """
        if padded:
            final_hidden_states = final_hidden_states[:0]
        return final_hidden_states
