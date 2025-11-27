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
Communication Method Factory for MoE

Factory for creating and selecting the best communication method based on
hardware support and configuration.
"""

import os
from typing import Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import local_mpi_size

from .allgather_reducescatter import AllGatherReduceScatter
from .base import Communication
from .deep_ep import DeepEP
from .deep_ep_low_latency import DeepEPLowLatency
from .mnnvl_latency import MnnvlLatency
from .mnnvl_throughput import MNNVLThroughput


def is_high_throughput() -> bool:
    """
    Check if high throughput mode is enabled
    """
    return True


def is_deepep_feasible(num_ranks: int) -> bool:
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


class CommunicationFactory:
    """
    Factory for creating MoE communication methods

    Selects the best communication method based on:
    - Hardware support (MNNVL, DeepEP)
    - Configuration settings
    - Workload characteristics
    """

    @staticmethod
    def create_strategy(
        model_config: ModelConfig,
        num_experts: int,
        num_slots: int,
        top_k: int,
        expert_size_per_partition: int,
        payload_in_workspace: bool = False,
        alltoall_result_do_sum: bool = False,
    ) -> Optional[Communication]:
        """
        Create the best communication method for the given configuration

        Selection priority:
        1. Force method (if specified via TRTLLM_FORCE_ALLTOALL_METHOD env)
        2. MNNVL (if hardware supports)
           - Selects latency or throughput backend based on TRTLLM_MOE_ALLTOALL_BACKEND env
           - Default: "mnnvllatency", alternative: "mnnvlthroughput"
        3. DeepEP / DeepEPLowLatency (if enabled and hardware supports)
        4. AllGather + ReduceScatter (fallback, always works)

        Args:
            model_config: Model configuration containing mapping, quant_config, max_num_tokens, etc.
            num_experts: Total number of experts
            num_slots: Total number of expert slots
            top_k: Number of experts per token
            expert_size_per_partition: Number of experts per partition (required for DeepEP)
            payload_in_workspace: If True, final_hidden_states is already in workspace (for MNNVLThroughput)
            alltoall_result_do_sum: If True, sum the alltoall results (for MnnvlLatency)

        Returns:
            The selected communication method, or None if attention does not use DP

        Note:
            Most parameters are extracted from model_config. Only MoE-specific parameters
            (num_experts, num_slots, top_k, expert_size_per_partition) need to be provided separately.
        """
        # Extract parameters from model_config
        mapping = model_config.mapping
        hidden_size = model_config.pretrained_config.hidden_size
        weight_dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        max_num_tokens = model_config.max_num_tokens
        moe_max_num_tokens = model_config.moe_max_num_tokens
        use_cuda_graph = model_config.use_cuda_graph
        use_low_precision_combine = model_config.use_low_precision_moe_combine

        # If attention does not use data parallelism (either uses TP or single card), no MoE communication is needed
        if (not mapping.enable_attention_dp) or mapping.dp_size == 1:
            return None

        # If no attention DP, or if MoE TP is enabled, use AllGather + ReduceScatter
        # AlltoAll cannot support MoE TP
        if mapping.moe_tp_size != 1:
            return AllGatherReduceScatter(mapping)

        # Check if forced method is specified via environment variable
        force_method = os.environ.get("TRTLLM_FORCE_ALLTOALL_METHOD")

        if force_method is not None:
            # Validate platform support for forced method
            method_upper = force_method.upper()
            if method_upper in ["MNNVLLATENCY", "MNNVLTHROUGHPUT"]:
                if not MnnvlLatency.is_platform_supported():
                    raise RuntimeError(
                        f"Forced method '{force_method}' is not supported on this platform. "
                        "MNNVLLATENCY and MNNVLTHROUGHPUT require compatible hardware."
                    )
            elif method_upper in ["DEEPEP", "DEEPEPLOWLATENCY"]:
                if not DeepEP.is_platform_supported(mapping):
                    raise RuntimeError(
                        f"Forced method '{force_method}' is not supported on this platform. "
                        "DeepEP requires compatible hardware and TRTLLM_CAN_USE_DEEP_EP=1."
                    )

            return CommunicationFactory._create_forced_method(
                force_method,
                model_config,
                num_experts,
                num_slots,
                top_k,
                expert_size_per_partition,
                payload_in_workspace,
                alltoall_result_do_sum,
            )

        # Try MNNVL first (highest priority)
        if MnnvlLatency.is_platform_supported():
            if is_high_throughput():
                # Currently, MNNVLThroughput shows better performance at all scenarios
                return MNNVLThroughput(
                    mapping,
                    num_experts,
                    top_k,
                    max_num_tokens_per_rank=max_num_tokens,
                    payload_in_workspace=payload_in_workspace,
                )
            else:
                return MnnvlLatency(
                    mapping,
                    num_experts,
                    num_slots,
                    top_k,
                    use_low_precision_combine,
                    alltoall_result_do_sum=alltoall_result_do_sum,
                )

        # Try DeepEP
        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") == "1":
            if weight_dtype == torch.bfloat16:
                if DeepEP.is_platform_supported(mapping) and is_deepep_feasible(
                    mapping.moe_ep_size
                ):
                    return DeepEP(
                        mapping,
                        num_slots,
                        hidden_size,
                        weight_dtype,
                        quant_config,
                        expert_size_per_partition,
                        use_cuda_graph,
                    )
                else:
                    # Use DeepEP Low Latency as fallback (when not feasible or not supported)
                    return DeepEPLowLatency(
                        mapping,
                        num_slots,
                        hidden_size,
                        weight_dtype,
                        quant_config,
                        expert_size_per_partition,
                        max_num_tokens,
                        use_low_precision_combine,
                        moe_max_num_tokens,
                    )

        # Fallback to AllGather + ReduceScatter
        return AllGatherReduceScatter(mapping)

    @staticmethod
    def _create_forced_method(
        method: str,
        model_config: ModelConfig,
        num_experts: int,
        num_slots: int,
        top_k: int,
        expert_size_per_partition: int,
        payload_in_workspace: bool,
        alltoall_result_do_sum: bool,
    ) -> Communication:
        """Create a specific method (for debugging/testing)"""
        # Extract parameters from model_config
        mapping = model_config.mapping
        hidden_size = model_config.pretrained_config.hidden_size
        weight_dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        max_num_tokens = model_config.max_num_tokens
        moe_max_num_tokens = model_config.moe_max_num_tokens
        use_cuda_graph = model_config.use_cuda_graph
        use_low_precision_combine = model_config.use_low_precision_moe_combine

        method = method.upper()

        if method == "MNNVLLATENCY":
            return MnnvlLatency(
                mapping,
                num_experts,
                num_slots,
                top_k,
                use_low_precision_combine,
                alltoall_result_do_sum=alltoall_result_do_sum,
            )
        elif method == "MNNVLTHROUGHPUT":
            # MNNVLThroughput requires max_num_tokens_per_rank
            # max_num_tokens is per-rank value (as passed from callers like cutlass)
            return MNNVLThroughput(
                mapping,
                num_experts,
                top_k,
                max_num_tokens_per_rank=max_num_tokens,
                payload_in_workspace=payload_in_workspace,
            )
        elif method == "DEEPEP":
            return DeepEP(
                mapping,
                num_slots,
                hidden_size,
                weight_dtype,
                quant_config,
                expert_size_per_partition,
                use_cuda_graph,
            )
        elif method == "DEEPEPLOWLATENCY":
            return DeepEPLowLatency(
                mapping,
                num_slots,
                hidden_size,
                weight_dtype,
                quant_config,
                expert_size_per_partition,
                max_num_tokens,
                use_low_precision_combine,
                moe_max_num_tokens,
            )
        elif method == "ALLGATHER":
            return AllGatherReduceScatter(mapping)
        else:
            raise ValueError(f"Unknown communication method: {method}")
