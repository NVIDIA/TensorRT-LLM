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
Communication Method Factory for MoE

Factory for creating and selecting the best communication method based on
hardware support and configuration.
"""

import os
from typing import Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.logger import logger

from ..wide_ep_ft import get_wide_ep_ft_options
from .allgather_reducescatter import AllGatherReduceScatter
from .base import Communication
from .deep_ep import DeepEP
from .deep_ep_low_latency import DeepEPLowLatency
from .nccl_ep import NcclEP
from .nvlink_one_sided import NVLinkOneSided
from .nvlink_two_sided import NVLinkTwoSided
from .nvlink_two_sided_flashinfer import NVLinkTwoSidedFlashinfer

# Temporary NCCL-EP v0.1.0 limitation. The LL combine kernel derives its
# warp-group count and dynamic-SMEM requirement here:
# https://github.com/NVIDIA/nccl/blob/nccl-ep-v0.1.0/contrib/nccl_ep/device/low_latency.cu#L1990-L2025
# Its v0.1.0 group initialization limits LL execution to 14 warp groups:
# https://github.com/NVIDIA/nccl/blob/nccl-ep-v0.1.0/contrib/nccl_ep/nccl_ep.cc#L1302-L1314
# TODO: Remove this compatibility check after upgrading to NCCL-EP v0.2,
# which removes the v0.1.0 LL-combine launch limitation.
_NCCL_EP_V0_1_LL_MAX_WARP_GROUPS = 14


def _get_nccl_ep_ll_combine_smem_requirement(
    num_slots: int, hidden_size: int, num_device_sms: int
) -> int | None:
    """Return the NCCL-EP LL combine dynamic-SMEM requirement in bytes."""
    num_warp_groups = (num_slots + num_device_sms - 1) // num_device_sms
    if num_warp_groups > _NCCL_EP_V0_1_LL_MAX_WARP_GROUPS:
        return None
    num_warps_per_group = 32 // num_warp_groups

    num_warps = num_warp_groups * num_warps_per_group
    num_meta_bytes = hidden_size // 128 * 4
    num_send_tma_bytes = 32 * 16 * 4 + 16
    smem_send_size = num_warps * (3 * num_send_tma_bytes + num_meta_bytes)

    num_recv_tma_bytes = 16 + hidden_size * 2
    smem_recv_size = 2 * (3 * num_recv_tma_bytes + hidden_size * 2 + 3 * num_meta_bytes * 3)
    return max(smem_send_size, smem_recv_size)


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
        use_flashinfer: bool = False,
        hidden_size: Optional[int] = None,
        communication_method: Optional[str] = None,
    ) -> Optional[Communication]:
        """
        Create the best communication method for the given configuration

        Selection priority (using try-catch mechanism):
        1. Force method (if specified via TRTLLM_FORCE_COMM_METHOD env)
        2. Auto-selection (tries in order):
           - NVLinkOneSided (highest priority for throughput)
           - NVLinkTwoSided (high priority for latency)
           - NcclEP (if nccl-ep is available)
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
            hidden_size: Actual MoE activation dimension (the A2A payload width).
                For latent-MoE models this is moe_latent_size, not pretrained_config.hidden_size.
                Falls back to pretrained_config.hidden_size when not provided.
            communication_method: Optional model-selected communication method.
                ``TRTLLM_FORCE_COMM_METHOD`` takes precedence when set.
            # TODO: Need a way to indicate whether EPLB is enabled.

        Returns:
            The selected communication method, or None if attention does not use DP

        Note:
            Most parameters are extracted from model_config. Only MoE-specific parameters
            (num_experts, num_slots, top_k, expert_size_per_partition) need to be provided separately.
        """
        # Extract parameters from model_config
        mapping = model_config.mapping
        if hidden_size is None:
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

        # A forced method comes either from the environment, which wins, or from the
        # model-selected argument. Keep the source with the value so the log below can
        # name the one the reader can actually go and change.
        env_method = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
        if env_method is not None:
            force_method, force_source = env_method, "TRTLLM_FORCE_COMM_METHOD"
        else:
            force_method, force_source = communication_method, "communication_method"

        if force_method is not None:
            strategy = CommunicationFactory._create_forced_method(
                force_method,
                model_config,
                num_experts,
                num_slots,
                top_k,
                expert_size_per_partition,
                payload_in_workspace,
                alltoall_result_do_sum,
                use_flashinfer,
                hidden_size=hidden_size,
            )
            logger.info(
                f"Selected communication strategy: {strategy.__class__.__name__} "
                f"({force_source}={force_method})"
            )
            return strategy

        # Auto-selection: Try strategies in priority order using try-catch
        # Priority: NVLinkOneSided > NVLinkTwoSided > NcclEP > DeepEP > DeepEPLowLatency > AllGather

        try:
            enable_eplb = model_config.moe_load_balancer is not None
            ep_group_health, watchdog_timeout_s, watchdog_poll_interval_s = get_wide_ep_ft_options(
                model_config
            )
            strategy = NVLinkOneSided(
                mapping,
                num_slots,
                top_k,
                max_num_tokens,
                payload_in_workspace,
                hidden_size=hidden_size,
                dtype=act_dtype,
                num_experts=num_experts if enable_eplb else None,
                use_low_precision_combine=use_low_precision_combine,
                ep_group_health=ep_group_health,
                alltoall_watchdog_timeout_s=watchdog_timeout_s,
                alltoall_watchdog_poll_interval_s=watchdog_poll_interval_s,
            )
            logger.info("Selected communication strategy: NVLinkOneSided")
            return strategy
        except Exception as e:
            logger.info(f"NVLinkOneSided not available: {e}")

        # Non-divisible EP: NVLinkTwoSided and DeepEP require num_experts % ep_size == 0.
        if num_experts % mapping.moe_ep_size != 0:
            logger.info(
                f"Non-divisible EP (num_experts={num_experts}, ep_size={mapping.moe_ep_size}): "
                "falling back to AllGatherReduceScatter"
            )
            return AllGatherReduceScatter(mapping)

        try:
            if use_flashinfer:
                strategy = NVLinkTwoSidedFlashinfer(
                    mapping,
                    num_experts,
                    num_slots,
                    top_k,
                    use_low_precision_combine,
                    alltoall_result_do_sum=alltoall_result_do_sum,
                )
            else:
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
        except Exception as e:
            logger.info(f"NVLinkTwoSided not available: {e}")

        # Try NCCL EP (rank-major LL). Falls through to DeepEP/AllGather if
        # prerequisites are not met or libnccl_ep.so is not available.
        nccl_ep_unavailable_reason = CommunicationFactory._get_nccl_ep_unavailable_reason(
            act_dtype,
            quant_config,
            num_slots,
            hidden_size,
            max_num_tokens,
            moe_max_num_tokens,
            top_k,
        )
        if nccl_ep_unavailable_reason is None:
            try:
                strategy = NcclEP(
                    mapping,
                    num_slots,
                    hidden_size,
                    max_num_tokens,
                    moe_max_num_tokens,
                    top_k=top_k,
                )
                logger.info("Selected communication strategy: NcclEP")
                return strategy
            except RuntimeError as e:
                logger.debug(f"NcclEP not available: {e}")
        else:
            logger.debug(f"NcclEP not available: {nccl_ep_unavailable_reason}")

        # Try DeepEP (if enabled and weight dtype is bfloat16)
        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "1") == "1" and act_dtype == torch.bfloat16:
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
            except Exception as e:
                logger.info(f"DeepEP not available: {e}")

            # Try DeepEPLowLatency as fallback when DeepEP is not available
            try:
                if top_k > DeepEPLowLatency.MAX_TOP_K:
                    raise ValueError(
                        f"top_k={top_k} exceeds the low-latency kernels' "
                        f"compile-time cap MAX_TOP_K={DeepEPLowLatency.MAX_TOP_K} "
                        "(kNumMaxTopK in internode_ll.cu); the kernel-side "
                        "EP_HOST_ASSERT would abort on the first dispatch/combine"
                    )
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
            except Exception as e:
                logger.info(f"DeepEPLowLatency not available: {e}")

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
        use_flashinfer: bool,
        hidden_size: Optional[int] = None,
    ) -> Communication:
        """
        Create a specific method (for debugging/testing)

        Raises:
            RuntimeError: If the forced method is not supported on this platform
            ValueError: If method name is unknown
        """
        # Extract parameters from model_config
        mapping = model_config.mapping
        if hidden_size is None:
            hidden_size = model_config.pretrained_config.hidden_size
        act_dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        max_num_tokens = model_config.max_num_tokens
        moe_max_num_tokens = model_config.moe_max_num_tokens
        use_cuda_graph = model_config.use_cuda_graph
        use_low_precision_combine = model_config.use_low_precision_moe_combine

        method = method.upper()

        # Whitelist check: non-divisible EP only supports NVLinkOneSided and AllGather.
        _NONDIVISIBLE_EP_ALLOWED = {"NVLINK_ONE_SIDED", "ALLGATHER"}
        if num_experts % mapping.moe_ep_size != 0 and method not in _NONDIVISIBLE_EP_ALLOWED:
            raise ValueError(
                f"Communication method '{method}' requires num_experts % ep_size == 0, "
                f"but got num_experts={num_experts}, ep_size={mapping.moe_ep_size}. "
                f"Allowed methods for non-divisible EP: {sorted(_NONDIVISIBLE_EP_ALLOWED)}"
            )

        # Create strategy - will raise RuntimeError if platform not supported
        if method in ["NVLINK_TWO_SIDED"]:
            if use_flashinfer:
                return NVLinkTwoSidedFlashinfer(
                    mapping,
                    num_experts,
                    num_slots,
                    top_k,
                    use_low_precision_combine,
                    alltoall_result_do_sum=alltoall_result_do_sum,
                )
            else:
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
            ep_group_health, watchdog_timeout_s, watchdog_poll_interval_s = get_wide_ep_ft_options(
                model_config
            )
            return NVLinkOneSided(
                mapping,
                num_slots,
                top_k,
                max_num_tokens,
                payload_in_workspace,
                hidden_size=hidden_size,
                dtype=act_dtype,
                num_experts=num_experts if enable_eplb else None,
                use_low_precision_combine=use_low_precision_combine,
                ep_group_health=ep_group_health,
                alltoall_watchdog_timeout_s=watchdog_timeout_s,
                alltoall_watchdog_poll_interval_s=watchdog_poll_interval_s,
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
            if top_k > DeepEPLowLatency.MAX_TOP_K:
                raise ValueError(
                    f"DeepEPLowLatency supports top_k <= {DeepEPLowLatency.MAX_TOP_K}, got {top_k}."
                )
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
        elif method == "NCCL_EP":
            nccl_ep_unavailable_reason = CommunicationFactory._get_nccl_ep_unavailable_reason(
                act_dtype,
                quant_config,
                num_slots,
                hidden_size,
                max_num_tokens,
                moe_max_num_tokens,
                top_k,
            )
            if nccl_ep_unavailable_reason is not None:
                raise ValueError(nccl_ep_unavailable_reason)
            return NcclEP(
                mapping,
                num_slots,
                hidden_size,
                max_num_tokens,
                moe_max_num_tokens,
                top_k=top_k,
            )
        elif method == "ALLGATHER":
            return AllGatherReduceScatter(mapping)
        else:
            raise ValueError(f"Unknown communication method: {method}")

    @staticmethod
    def _get_nccl_ep_unavailable_reason(
        act_dtype: torch.dtype,
        quant_config,
        num_slots: int,
        hidden_size: int,
        max_num_tokens: int,
        moe_max_num_tokens: Optional[int],
        top_k: int,
    ) -> Optional[str]:
        if act_dtype != torch.bfloat16:
            return f"NcclEP requires act_dtype=torch.bfloat16, got {act_dtype}."
        if quant_config is not None:
            quant_mode = getattr(quant_config, "layer_quant_mode", None)
            if quant_mode is not None and quant_mode.has_any_quant(exclude_kv_cache=True):
                return "NcclEP v0.1 does not support quantized MoE communication."
        if num_slots <= 0 or hidden_size <= 0 or max_num_tokens <= 0:
            return (
                "NcclEP requires positive num_slots, hidden_size, and max_num_tokens, got "
                f"{num_slots=}, {hidden_size=}, {max_num_tokens=}."
            )
        if moe_max_num_tokens is not None and moe_max_num_tokens <= 0:
            return (
                f"NcclEP requires moe_max_num_tokens > 0 when provided, got {moe_max_num_tokens}."
            )
        if top_k <= 0 or top_k > num_slots:
            return f"NcclEP requires 0 < top_k <= num_slots, got {top_k=}, {num_slots=}."
        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            required_smem = _get_nccl_ep_ll_combine_smem_requirement(
                num_slots, hidden_size, device_properties.multi_processor_count
            )
            max_dynamic_smem = getattr(
                device_properties,
                "shared_memory_per_block_optin",
                device_properties.shared_memory_per_block,
            )
            if required_smem is None:
                return (
                    "NcclEP low-latency combine requires at most "
                    f"{_NCCL_EP_V0_1_LL_MAX_WARP_GROUPS} expert warp groups, got "
                    f"{num_slots=} and {device_properties.multi_processor_count=}."
                )
            if required_smem > max_dynamic_smem:
                return (
                    "NcclEP low-latency combine requires "
                    f"{required_smem} bytes of dynamic shared memory, but the current device "
                    f"supports only {max_dynamic_smem} bytes."
                )
        return None
