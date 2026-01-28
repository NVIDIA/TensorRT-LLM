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
from tensorrt_llm.logger import logger

from .allgather_reducescatter import AllGatherReduceScatter
from .base import Communication
from .deep_ep import DeepEP
from .deep_ep_low_latency import DeepEPLowLatency
from .nvlink_one_sided import NVLinkOneSided
from .nvlink_two_sided import NVLinkTwoSided


class CommunicationFactory:
    """
    Factory for creating MoE communication methods

    Selects the best communication method based on:
    - Hardware support (NVLINK, DeepEP)
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
        alltoall_result_do_sum: bool = True,
    ) -> Optional[Communication]:
        """
        Create the best communication method for the given configuration

        Selection priority (using try-catch mechanism):
        1. Force method (if specified via TRTLLM_FORCE_COMM_METHOD env)
        2. Auto-selection (tries in order):
           - NVLinkOneSided (highest priority for throughput)
           - NVLinkTwoSided (high priority for latency)
           - DeepEP (if enabled via TRTLLM_CAN_USE_DEEP_EP)
           - DeepEPLowLatency (if enabled via TRTLLM_CAN_USE_DEEP_EP)
           - AllGather + ReduceScatter (fallback, always works)

        Args:
            model_config: Model configuration containing mapping, quant_config, max_num_tokens, etc.
            num_experts: Total number of experts
            num_slots: Total number of expert slots
            top_k: Number of experts per token
            expert_size_per_partition: Number of experts per partition (required for DeepEP)
            payload_in_workspace: If True, final_hidden_states is already in workspace (for NVLinkOneSided)
            alltoall_result_do_sum: If True, sum the alltoall results (for NVLinkTwoSided)
            # TODO: Need a way to indicate whether EPLB is enabled.

        Returns:
            The selected communication method, or None if attention does not use DP

        Note:
            Most parameters are extracted from model_config. Only MoE-specific parameters
            (num_experts, num_slots, top_k, expert_size_per_partition) need to be provided separately.
        """
        # Extract parameters from model_config
        mapping = model_config.mapping
        hidden_size = model_config.pretrained_config.hidden_size
        act_dtype = model_config.torch_dtype
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
        force_method = os.environ.get("TRTLLM_FORCE_COMM_METHOD")

        if force_method is not None:
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

        # Auto-selection: Try strategies in priority order using try-catch
        # Priority: NVLinkOneSided > NVLinkTwoSided > DeepEP > DeepEPLowLatency > AllGather

        try:
            enable_eplb = model_config.moe_load_balancer is not None
            strategy = NVLinkOneSided(
                mapping,
                num_slots,
                top_k,
                max_num_tokens,
                payload_in_workspace,
                hidden_size=hidden_size,
                dtype=act_dtype,
                num_experts=num_experts if enable_eplb else None,
            )
            logger.info("Selected communication strategy: NVLinkOneSided")
            return strategy
        except RuntimeError as e:
            logger.debug(f"NVLinkOneSided not available: {e}")

        try:
            strategy = NVLinkTwoSided(
                mapping,
                num_experts,
                num_slots,
                top_k,
                use_low_precision_combine,
                alltoall_result_do_sum=alltoall_result_do_sum,
            )
            logger.info("Selected communication strategy: NVLinkTwoSided")
            return strategy
        except RuntimeError as e:
            logger.debug(f"NVLinkTwoSided not available: {e}")

        # Try DeepEP (if enabled and weight dtype is bfloat16)
        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") == "1" and act_dtype == torch.bfloat16:
            try:
                strategy = DeepEP(
                    mapping,
                    num_slots,
                    hidden_size,
                    act_dtype,
                    quant_config,
                    expert_size_per_partition,
                    use_cuda_graph,
                )
                logger.info("Selected communication strategy: DeepEP")
                return strategy
            except RuntimeError as e:
                logger.debug(f"DeepEP not available: {e}")

            # Try DeepEPLowLatency as fallback when DeepEP is not available
            try:
                strategy = DeepEPLowLatency(
                    mapping,
                    num_slots,
                    hidden_size,
                    act_dtype,
                    quant_config,
                    expert_size_per_partition,
                    max_num_tokens,
                    use_low_precision_combine,
                    moe_max_num_tokens,
                )
                logger.info("Selected communication strategy: DeepEPLowLatency")
                return strategy
            except RuntimeError as e:
                logger.debug(f"DeepEPLowLatency not available: {e}")

        # Fallback to AllGather + ReduceScatter (always works)
        strategy = AllGatherReduceScatter(mapping)
        logger.info("Selected communication strategy: AllGatherReduceScatter (fallback)")
        return strategy

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
        """
        Create a specific method (for debugging/testing)

        Raises:
            RuntimeError: If the forced method is not supported on this platform
            ValueError: If method name is unknown
        """
        # Extract parameters from model_config
        mapping = model_config.mapping
        hidden_size = model_config.pretrained_config.hidden_size
        act_dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        max_num_tokens = model_config.max_num_tokens
        moe_max_num_tokens = model_config.moe_max_num_tokens
        use_cuda_graph = model_config.use_cuda_graph
        use_low_precision_combine = model_config.use_low_precision_moe_combine

        method = method.upper()

        # Create strategy - will raise RuntimeError if platform not supported
        if method in ["NVLINK_TWO_SIDED"]:
            return NVLinkTwoSided(
                mapping,
                num_experts,
                num_slots,
                top_k,
                use_low_precision_combine,
                alltoall_result_do_sum=alltoall_result_do_sum,
            )
        elif method in ["NVLINK_ONE_SIDED"]:
            enable_eplb = model_config.moe_load_balancer is not None
            return NVLinkOneSided(
                mapping,
                num_slots,
                top_k,
                max_num_tokens,
                payload_in_workspace,
                hidden_size=hidden_size,
                dtype=act_dtype,
                num_experts=num_experts if enable_eplb else None,
            )
        elif method == "DEEPEP":
            return DeepEP(
                mapping,
                num_slots,
                hidden_size,
                act_dtype,
                quant_config,
                expert_size_per_partition,
                use_cuda_graph,
            )
        elif method == "DEEPEPLOWLATENCY":
            return DeepEPLowLatency(
                mapping,
                num_slots,
                hidden_size,
                act_dtype,
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
