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
ConfigurableMoE: Composition-based Configurable MoE Module

This module provides a universal MoE execution flow using composition pattern:
- MoE Backend: Pluggable computation backend (Cutlass, TRTLLMGen, etc.)
- Communication Strategy: Pluggable communication (AllGather, AllToAll, etc.)
- EPLB: Optional load balancing (can be toggled on/off)

Design Principles:
1. Use composition instead of inheritance for flexibility
2. Backend declares its capabilities (separated vs fused routing)
3. ConfigurableMoE adapts flow based on backend capabilities
4. Unified EPLB integration for backends that support it
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.expert_statistic import ExpertStatistic
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._torch.modules.fused_moe.routing import BaseMoeRoutingMethod
from tensorrt_llm._torch.utils import AuxStreamType, EventType, Fp4QuantizedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from .communication import (
    AllGatherReduceScatter,
    Communication,
    CommunicationFactory,
    DeepEP,
    DeepEPLowLatency,
    NVLinkOneSided,
    NVLinkTwoSided,
)
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_deepgemm import DeepGemmFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE


class ConfigurableMoE(MoE):
    """
    Configurable MoE layer using composition pattern with automatic configuration

    This class orchestrates the MoE execution flow by composing:
    - moe_backend: Existing FusedMoE implementation (CutlassFusedMoE, CuteDslFusedMoE, etc.)
                   Note: Current FusedMoE implementations are used as backends (transitional).
                         Future will have dedicated MoEBackend interface.
    - Communication: Handles distributed communication (auto-selected)
    - EPLB (optional): Handles expert parallel load balancing (auto-detected)

    Args:
        routing_method: Routing method for token-to-expert assignment
        num_experts: Total number of experts
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        dtype: Data type for weight
        reduce_results: Whether to reduce results
        model_config: Model configuration
        aux_stream_dict: Auxiliary CUDA streams for overlap
        weight_loading_mode: Weight loading mode
        layer_idx: Layer index
        **kwargs: Additional arguments
            - backend_type: Backend type ('cutlass', 'trtllm_gen_min_latency', etc.)
                           Default: 'cutlass'
            - tune_max_num_tokens: Max tokens for profiling (passed to backend)
            - Other backend-specific arguments

    Key Attributes:
        - backend: MoE computation backend (auto-created attribute)
        - comm: Communication strategy (auto-created attribute, can be None)
        - layer_load_balancer: EPLB instance (auto-detected, optional)

    Auto-Detection:
        - EPLB: Enabled if get_moe_load_balancer() is not None
        - Backend: Defaults to CutlassMoEBackend, override via backend_type
        - Communication: Auto-selected based on hardware (NVLINK > DeepEP > AllGather)
    """

    @classmethod
    def can_implement(
        cls,
        quant_algo,
        dtype_activation: torch.dtype = torch.bfloat16,
        gptoss_style: bool = False,
    ):
        """
        ConfigurableMoE is a wrapper class that delegates to specific backends.

        To check capability, query the specific backend class directly:
        - CutlassFusedMoE.can_implement(quant_algo, dtype_activation, gptoss_style)
        - TRTLLMGenFusedMoE.can_implement(quant_algo, dtype_activation, gptoss_style)
        - etc.

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation data type
            gptoss_style: Whether gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled

        Returns:
            Tuple[bool, Optional[str]]: Always returns (False, reason)
        """
        del quant_algo, dtype_activation, gptoss_style  # Unused - wrapper class
        return False, (
            "ConfigurableMoE is a wrapper class. "
            "Query the specific backend (CutlassFusedMoE, TRTLLMGenFusedMoE, etc.) directly."
        )

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode=None,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        override_quant_config: Optional["QuantConfig"] = None,
        **kwargs,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,  # ConfigurableMoE needs correct layer_idx for EPLB initialization
            **kwargs,
        )

        # Store model_config and aux_stream_dict for later use (e.g., backend setter)
        self.model_config = model_config
        self.aux_stream_dict = aux_stream_dict

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # ========== Create MoE Backend (Default: Cutlass) ==========
        from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend, get_moe_cls

        # Get MoE backend class based on override_quant_config or model_config
        moe_cls = get_moe_cls(model_config, override_quant_config=override_quant_config)

        # Call create_moe_backend with all necessary parameters
        # init_load_balancer=False: Prevents backend from registering itself with load balancer
        # without_comm=True: Prevents backend from initializing communication (ConfigurableMoE handles it)
        # skip_create_weights_in_init=True: Prevents backend from creating weights in __init__
        #   because backend uses layer_idx=None and may have different expert assignments
        #   We will create weights after syncing attributes from ConfigurableMoE
        tmp_skip_create_weights_in_init = model_config.skip_create_weights_in_init
        model_config._frozen = False
        model_config.skip_create_weights_in_init = True
        model_config._frozen = True

        backend = create_moe_backend(
            moe_cls=moe_cls,
            routing_method=routing_method,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            reduce_results=self.reduce_results,
            model_config=model_config,
            aux_stream_dict=self.aux_stream_dict,
            weight_loading_mode=self.weight_loading_mode,
            bias=kwargs.get("bias", False),
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            layer_idx=None,
            swiglu_alpha=kwargs.get("swiglu_alpha"),
            swiglu_beta=kwargs.get("swiglu_beta"),
            swiglu_limit=kwargs.get("swiglu_limit"),
            init_load_balancer=False,
            without_comm=True,
            activation_type=self.activation_type,
        )

        self.validate_backend(backend)
        self.backend = backend

        # Sync critical attributes from ConfigurableMoE to backend
        # ConfigurableMoE's super().__init__() was called with real layer_idx and initialized load balancer.
        # Backend was created with init_load_balancer=False and without_comm=True to avoid
        # duplicate initialization. Now sync all attributes from ConfigurableMoE to backend.
        if self.backend is not None:
            self.backend.layer_idx = self.layer_idx
            self.backend.layer_idx_str = self.layer_idx_str
            self.backend.num_slots = self.num_slots
            self.backend.layer_load_balancer = self.layer_load_balancer
            self.backend.repeat_count = self.repeat_count
            self.backend.repeat_idx = self.repeat_idx
            self.backend.initial_local_expert_ids = self.initial_local_expert_ids
            self.backend.initial_global_assignments = self.initial_global_assignments
            self.backend.slot_start = self.slot_start
            self.backend.slot_end = self.slot_end
            self.backend.expert_size_per_partition = self.expert_size_per_partition

        # Create weights here, because the backend needs the layer_load_balancer info to create weights
        model_config._frozen = False
        model_config.skip_create_weights_in_init = tmp_skip_create_weights_in_init
        model_config._frozen = True
        if not model_config.skip_create_weights_in_init:
            self.backend.create_weights()

        # ========== Create Communication Strategy ==========
        self.comm = self._create_comm_strategy_auto()

        # ========== Chunking Configuration ==========
        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size

        # Auxiliary stream for chunking overlap
        if self.moe_max_num_tokens < default_moe_max_num_tokens:
            self.aux_stream = (
                aux_stream_dict[AuxStreamType.MoeChunkingOverlap]
                if aux_stream_dict is not None
                else torch.cuda.Stream()
            )
            self.event_dict = {
                key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeChunkingOverlap]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        # Validate configuration
        self.validate_config()

        # Mark as _weights_removed to skip ConfigurableMoE's post_load_weights in model_loader
        # The backend's post_load_weights will be called directly by model_loader
        # This avoids duplicate post_load_weights calls (once for ConfigurableMoE, once for backend)
        # TODO: in the future, all the weights related work should be done only in backend.
        self._weights_removed = True

    def _supports_load_balancer(self) -> bool:
        """Check if this MoE implementation supports load balancer."""
        # During initialization, backend might not be created yet
        # Return True by default (most backends support it), backend will validate later
        if not hasattr(self, "backend") or self.backend is None:
            return self.use_dp and self.parallel_size > 1
        return self.backend._supports_load_balancer()

    def validate_config(self):
        """
        Validate configuration parameters

        Validates:
        - apply_router_weight_on_input: Only supports top-1 routing
        """
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, (
                "apply_router_weight_on_input only supports top-1 routing"
            )

    def _create_comm_strategy(self, model_config: ModelConfig) -> Optional[Communication]:
        """
        Create communication strategy based on configuration

        Default: None (will use factory to auto-select when needed)
        Auto-selects best strategy based on hardware and configuration

        """
        # Communication strategy is None by default
        # Will be created lazily in determine_communication_method() when first needed
        # For now, return None and create on-demand
        return None

    def _get_quant_config_dict(self, model_config: ModelConfig) -> Optional[Dict]:
        """
        Extract quantization configuration from model_config

        """
        if model_config.quant_config is None:
            return None

        quant_mode = model_config.quant_config.layer_quant_mode
        return {
            "has_fp8_qdq": quant_mode.has_fp8_qdq()
            if hasattr(quant_mode, "has_fp8_qdq")
            else False,
            "has_nvfp4": quant_mode.has_nvfp4() if hasattr(quant_mode, "has_nvfp4") else False,
            "has_w4afp8": quant_mode.is_int4_weight_only_per_group()
            if hasattr(quant_mode, "is_int4_weight_only_per_group")
            else False,
            "has_fp8_block_scales": quant_mode.has_fp8_block_scales()
            if hasattr(quant_mode, "has_fp8_block_scales")
            else False,
        }

    def calculate_num_chunks(self, all_rank_num_tokens: List[int]) -> int:
        """
        Calculate how many chunks are needed

        """
        num_rows = sum(all_rank_num_tokens)
        return (num_rows + self.moe_max_num_tokens - 1) // self.moe_max_num_tokens

    def split_chunk(self, split_token_num: int, split_num_chunks: int) -> List[int]:
        """
        Split token count into multiple chunks as evenly as possible

        """
        val_div = split_token_num // split_num_chunks
        val_mod = split_token_num % split_num_chunks
        split_chunk_size_list = [val_div + 1] * val_mod + [val_div] * (split_num_chunks - val_mod)
        return split_chunk_size_list

    def determine_communication_method(
        self, all_rank_num_tokens: List[int], num_chunks: int
    ) -> None:
        """
        Determine and setup communication method with automatic fallback

        This method:
        1. Returns early if comm is None or already AllGather (nothing to validate)
        2. Validates if current AllToAll strategy can be used for given workload
        3. Falls back to AllGather if current strategy cannot be used (logs info message)

        After calling this method, use enable_alltoall to check which method is active.

        Args:
            all_rank_num_tokens: Token counts per rank
            num_chunks: Number of chunks

        Side effects:
            - May switch self.comm to AllGather if current strategy cannot be used

        Note: This method does NOT create strategy if None (creation happens lazily elsewhere).
              It only validates and potentially falls back existing AllToAll strategies.

        """

        # Early return if nothing to validate:
        # - None: Atten is TP or single rank, no communication needed
        # - AllGather: Already using fallback strategy, no validation needed
        if self.comm is None or isinstance(self.comm, AllGatherReduceScatter):
            return

        # Check if current strategy can be used
        feasible_workload = self.comm.is_workload_feasible(all_rank_num_tokens, num_chunks)

        if not feasible_workload:
            # Current comm cannot be used, fallback to AllGather
            all_rank_max_num_tokens = max(all_rank_num_tokens)
            logger.info(
                f"Communication strategy {self.comm.__class__.__name__} "
                f"cannot be used (num_chunks={num_chunks}, max_num_tokens={all_rank_max_num_tokens}). "
                f"Falling back to AllGatherReduceScatter."
            )

            # Switch to AllGather (always works)
            self.comm = AllGatherReduceScatter(mapping=self.mapping)

    def _create_comm_strategy_auto(self) -> Communication:
        """
        Auto-create the best communication strategy based on hardware and configuration

        Uses factory to select optimal strategy.

        """
        return CommunicationFactory.create_strategy(
            model_config=self.model_config,
            num_experts=self.num_experts,
            num_slots=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            expert_size_per_partition=self.expert_size_per_partition,
            payload_in_workspace=False,  # ConfigurableMoE does not use workspace output for now
            # Currently the TRTLLMGEN reduce sum internally.
            # Keep updated with more supported backends.
            alltoall_result_do_sum=True,
        )

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Universal forward implementation framework

        Flow:
        1. Handle padding
        2. Calculate chunk count and determine communication method
        3. Execute MoE computation (single or multiple chunks)
        4. Handle output truncation and EPLB repeat
        """
        # TODO: to clarify whether the output_dtype is needed.
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
        else:
            output_dtype = x.dtype
        # ========== Step 1: Handle padding ==========
        if all_rank_num_tokens is None:
            all_rank_num_tokens = [x.shape[0]]

        all_rank_max_num_tokens = max(all_rank_num_tokens)

        if use_dp_padding:
            all_rank_num_tokens_padded = [all_rank_max_num_tokens] * len(all_rank_num_tokens)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens

        # ========== Step 2: Determine communication method ==========
        num_chunks = self.calculate_num_chunks(all_rank_num_tokens_padded)

        # Determine and setup communication strategy (may fallback to AllGather)
        self.determine_communication_method(all_rank_num_tokens_padded, num_chunks)

        # ========== Step 3: Execute MoE computation ==========
        if num_chunks == 1:
            # Single chunk case
            outputs = self._forward_single_chunk(
                x,
                router_logits,
                output_dtype,
                all_rank_num_tokens_padded,
                use_dp_padding,
                do_finalize,
            )
        else:
            # Multiple chunks case
            outputs = self._forward_multiple_chunks(
                x,
                router_logits,
                num_chunks,
                output_dtype,
                all_rank_num_tokens_padded,
                use_dp_padding,
                do_finalize,
            )

        # ========== Step 4: Handle output truncation and EPLB repeat ==========
        if self.use_dp and self.parallel_size > 1:
            outputs = outputs[: all_rank_num_tokens[self.mapping.tp_rank]]

        # EPLB repeat logic
        self.repeat_idx = (self.repeat_idx + 1) % self.repeat_count

        return outputs

    def _prepare_workspace_deepgemm(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens: List[int],
    ) -> Optional[torch.Tensor]:
        """
        Prepare workspace for DeepGemmFusedMoE backend.

        Args:
            x: Input tensor
            all_rank_num_tokens: List of token counts for all ranks (used when use_dp is True)

        Returns:
            Workspace tensor or None if not using DeepGemmFusedMoE
        """
        if not isinstance(self.backend, DeepGemmFusedMoE):
            return None

        # Calculate the number of rows
        num_rows = x.shape[0]
        if self.use_dp and self.comm is not None:
            # When using communication, dispatch will create tensors with shape:
            # [ep_size * max_tokens_per_rank, ...] due to padding for balanced distribution
            # So we need to allocate workspace based on this size
            num_rows = self.mapping.moe_ep_size * max(all_rank_num_tokens)

        workspaces = self.backend.get_workspaces([num_rows])
        return workspaces[0]

    def _forward_single_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool],
        do_finalize: bool = True,
    ) -> torch.Tensor:
        """
        Single chunk execution path

        """
        # Calculate EPLB flags (first call or last call)
        is_first_call = self.repeat_idx == 0
        is_last_call = self.repeat_idx == self.repeat_count - 1

        # ========== Create workspace for DeepGemmFusedMoE ==========
        workspace = self._prepare_workspace_deepgemm(x, all_rank_num_tokens)

        # Execute unified flow (handles both separated and fused routing)
        outputs = self._forward_chunk_impl(
            x,
            router_logits,
            output_dtype,
            all_rank_num_tokens,
            use_dp_padding,
            is_first_call,
            is_last_call,
            do_finalize,
            workspace=workspace,
        )

        return outputs

    def _forward_chunk_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: List[int],
        use_dp_padding: bool,
        is_first_call: bool,
        is_last_call: bool,
        do_finalize: bool = True,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Unified execution flow for all backends

        Flow (based on EPLB_in_MOE[1].html):
        1. [EPLB] Start wait GPU stage (first call only, if enabled)
        2. Apply routing (only if backend supports routing separation)
        3. [EPLB] Update statistics and route (only if EPLB enabled)
        4. Quantization and Communication (adaptive ordering)
        5. MoE computation (backend)
        6. [EPLB] Start CPU stage (last call only, if enabled)
        7. Communication combine
        8. [EPLB] Done CPU stage (last call only, if enabled)

        - Separated routing: fused_moe_wide_ep.py:456-780, fused_moe_cutlass.py:236-443
        - Fused routing: fused_moe_trtllm_gen.py
        """

        # ========== Step 1: EPLB - Start wait GPU stage ==========
        self._load_balancer_start_wait_gpu_stage(is_first_call)

        # ========== Step 2: Apply routing (only if backend supports load balancer) ==========

        if self.backend._supports_load_balancer():
            # Separated routing: ConfigurableMoE calls routing_method
            token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)

            # Convert to standard dtypes for consistency with other MoE implementations
            token_selected_experts = token_selected_experts.to(torch.int32)

            assert token_selected_experts.shape[1] == self.routing_method.experts_per_token
            assert token_selected_experts.shape == token_final_scales.shape
            # CutlassFusedMoE expects float32, while TRTLLMGenFusedMoE uses bfloat16
            if isinstance(self.backend, CutlassFusedMoE):
                assert token_final_scales.dtype == torch.float32
            assert token_selected_experts.dtype == torch.int32

            # Convert token_final_scales to bfloat16 if needed (TRTLLMGen backend requires it)
            if token_final_scales is not None and isinstance(self.backend, TRTLLMGenFusedMoE):
                token_final_scales = token_final_scales.to(torch.bfloat16)

            # Apply router weight on input if enabled
            if self.apply_router_weight_on_input:
                assert x.dtype != torch.float8_e4m3fn, (
                    "Current workaround for apply_router_weight_on_input does not support fp8 input"
                )
                x = x * token_final_scales.to(x.dtype)
                # TODO: remove this once we have correct fusedmoe kernel ready
                # Check if using DeepEP strategies (they don't support token_final_scales=None)
                if isinstance(self.comm, (DeepEP, DeepEPLowLatency)):
                    # DeepEP doesn't support token_final_scales is None
                    token_final_scales = torch.ones_like(token_final_scales)
                else:
                    token_final_scales = None

        else:
            # Fused routing: Backend handles routing internally
            # EPLB must NOT be enabled for fused routing backends
            assert not self._using_load_balancer(), (
                f"EPLB is enabled but backend {self.backend.__class__.__name__} "
                f"has fused routing (does not support routing separation)"
            )

            # For fused routing, we don't have token_selected_experts yet
            # Will be handled by backend.run_moe_with_routing() later
            token_selected_experts = None
            token_final_scales = None

        # ========== Step 3: EPLB - Update statistics and route ==========
        # Only executed if backend supports routing separation AND EPLB is enabled
        if self.layer_load_balancer and token_selected_experts is not None:
            self._load_balancer_done_wait_gpu_stage(is_first_call)

            # Update EPLB statistics (method depends on communication strategy)
            # Use base class method: ignore_allreduce=True for NVLINK two-sided/one-sided (uses local stats only)
            ignore_allreduce = (
                self._is_using_nvlink_two_sided() or self._is_using_nvlink_one_sided()
            )
            self._load_balancer_update_statistic(
                token_selected_experts,
                is_first_call,
                is_last_call,
                ignore_allreduce=ignore_allreduce,
            )

            # EPLB routing: expert IDs -> slot IDs
            token_selected_slots = self._load_balancer_route(token_selected_experts, self.use_dp)
        else:
            token_selected_slots = token_selected_experts

        if token_selected_slots is not None:
            ExpertStatistic.set_layer(self.layer_idx)
            ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)
        token_selected_slots = get_calibrator().maybe_collect_or_replay_slots(
            self.num_slots, token_selected_slots
        )

        # ========== Step 3.5: Communication Prepare Phase (BEFORE quantization) ==========
        # NVLINK two-sided has a prepare phase to gather EPLB statistics

        local_statistic_tensor_for_dispatch = None
        eplb_dispatch_kwargs = {}
        should_update_eplb_after_dispatch = False
        # Only NVLINK two-sided needs prepare_dispatch
        if self._is_using_nvlink_two_sided():
            # Get local statistic info if this is the last call and EPLB is enabled
            local_statistic_tensor = None
            if is_last_call:
                local_statistic_tensor = self._load_balancer_get_local_statistic_tensor()

            # Call prepare_dispatch (gathers statistics for NVLINK two-sided)
            # prepare_dispatch stores alltoall_info in _dispatch_state and returns gathered_stats
            gathered_stats = self.comm.prepare_dispatch(
                token_selected_slots, all_rank_num_tokens, local_statistic_tensor
            )

            # Update EPLB with gathered statistics (if available)
            if gathered_stats is not None:
                gathered_stats = gathered_stats.view((self.mapping.moe_ep_size, self.num_experts))
                self._load_balancer_update_statistic_with_gathered_statistic(gathered_stats)
        # TODO: The abstract does not work well as NVLinkTwoSided gathers EPLB stats in prepare_dispatch,
        # while NVLinkOneSided gathers EPLB stats in dispatch.
        elif self._is_using_nvlink_one_sided():
            if self.layer_load_balancer and is_last_call:
                local_statistic_tensor_for_dispatch = (
                    self._load_balancer_get_local_statistic_tensor()
                )
            if local_statistic_tensor_for_dispatch is not None:
                eplb_dispatch_kwargs["eplb_local_stats"] = local_statistic_tensor_for_dispatch
                should_update_eplb_after_dispatch = True

        # ========== Step 4 & 5: Quantization and Communication Dispatch ==========
        # Order depends on whether strategy supports post-quant dispatch
        if self.comm is not None:
            # Check if we should use post-quant dispatch
            # supports_post_quant_dispatch checks strategy capability for the current quant mode
            supports_post_quant = self.comm.supports_post_quant_dispatch()

            # Call dummy_allreduce before allgather for load balancing debug
            if self.enable_dummy_allreduce:
                self.dummy_allreduce()

            if supports_post_quant:
                # ===== Post-quant flow: Quantize → Dispatch =====

                # Step 4a: Quantization FIRST
                x, x_sf = self.backend.quantize_input(x)

                # Step 4b: Dispatch AFTER quantization
                # Get pre_quant_scale for W4AFP8 if available (only DeepEPLowLatency needs it)
                # Other strategies will ignore this via **kwargs, so it's safe to pass unconditionally
                dispatch_kwargs = dict(eplb_dispatch_kwargs)
                if hasattr(self, "quant_scales") and self.quant_scales is not None:
                    if hasattr(self.quant_scales, "pre_quant_scale_1"):
                        dispatch_kwargs["pre_quant_scale"] = self.quant_scales.pre_quant_scale_1
                x, x_sf, token_selected_slots, token_final_scales = self.comm.dispatch(
                    hidden_states=x,
                    hidden_states_sf=x_sf,
                    token_selected_slots=token_selected_slots,
                    token_final_scales=token_final_scales,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                    **dispatch_kwargs,
                )
                if should_update_eplb_after_dispatch:
                    gathered_stats = self.comm.get_eplb_gathered_statistics()
                    self._load_balancer_update_statistic_with_gathered_statistic(gathered_stats)
            else:
                # ===== Pre-quant flow: Dispatch → Quantize =====

                # Step 4a: Dispatch FIRST (unquantized data)
                x, x_sf, token_selected_slots, token_final_scales = self.comm.dispatch(
                    hidden_states=x,
                    hidden_states_sf=None,  # Not quantized yet
                    token_selected_slots=token_selected_slots,
                    token_final_scales=token_final_scales,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                )

                # Step 4b: Quantization AFTER dispatch
                x, x_sf = self.backend.quantize_input(x)
        else:
            # No communication, just quantize
            # (use non-post-quant-comm path for TRTLLMGenFusedMoE)
            x, x_sf = self.backend.quantize_input(x, post_quant_comm=False)

        # ========== Step 6: MoE Computation ==========

        # Call unified run_moe interface with common parameters
        # If EPLB is enabled, token_selected_slots represents expert slots
        # Otherwise, token_selected_experts represents expert IDs
        final_hidden_states = self.backend.run_moe(
            x=x,
            token_selected_experts=token_selected_slots,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            **self._get_backend_kwargs(
                router_logits, do_finalize, all_rank_num_tokens, output_dtype, x, workspace
            ),
        )

        # ========== Step 8: EPLB - Start CPU stage ==========
        self._load_balancer_start_set_cpu_stage(is_last_call)

        # ========== Step 9: Communication - Combine ==========
        if self.comm is not None:
            if self.enable_dummy_allreduce:
                self.dummy_allreduce()
            # Use unified combine interface (reads dispatch state from strategy)
            final_hidden_states = self.comm.combine(final_hidden_states)
        else:
            # For non-comm case, It should be attention TP or single rank.
            # only check if allreduce is needed
            if self.parallel_size > 1 and self.reduce_results:
                final_hidden_states = self.all_reduce(final_hidden_states)
        # ========== Step 10: EPLB - Done CPU stage ==========
        self._load_balancer_done_set_cpu_stage(is_last_call)

        return final_hidden_states

    def _prepare_workspaces_for_chunk(
        self,
        all_rank_num_tokens_list: List[Optional[List[int]]],
        chunk_size_list: List[int],
        use_multi_stream: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare workspaces for chunked execution with DeepGemmFusedMoE backend.
        This will also be used for alltoall communication in the future.

        Args:
            all_rank_num_tokens_list: List of token counts per rank for each chunk (None if not using DP)
            chunk_size_list: List of chunk sizes
            use_multi_stream: Whether to use multi-stream execution (requires workspace_1)

        Returns:
            Tuple of (workspace_0, workspace_1), where workspace_1 is None if not using multi-stream
        """
        workspace_0 = None
        workspace_1 = None

        if not isinstance(self.backend, DeepGemmFusedMoE):
            return workspace_0, workspace_1

        # Always need at least workspace_0
        chunk_size_0 = (
            self.mapping.moe_ep_size * max(all_rank_num_tokens_list[0])
            if self.use_dp and all_rank_num_tokens_list[0] is not None
            else chunk_size_list[0]
        )
        workspace_chunk_sizes = [chunk_size_0]

        # Add workspace_1 if using multi-stream for alternating between streams
        # Reuse chunk_size_0 since it's always >= chunk_size_1 (first chunk is largest)
        if use_multi_stream:
            workspace_chunk_sizes.append(chunk_size_0)

        workspaces = self.backend.get_workspaces(workspace_chunk_sizes)
        workspace_0 = workspaces[0]
        if use_multi_stream:
            workspace_1 = workspaces[1]

        return workspace_0, workspace_1

    def _forward_multiple_chunks(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        num_chunks: int,
        output_dtype: Optional[torch.dtype],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool],
        do_finalize: bool = True,
    ) -> torch.Tensor:
        """
        Multiple chunks execution path with auxiliary stream for overlapping

        Same as original implementation - chunking logic is backend-agnostic

        """
        # ========== Chunk preparation ==========
        if self.use_dp:
            # When using DP: need all ranks' token counts for reducescatter
            all_rank_chunk_size_list = [
                self.split_chunk(val, num_chunks) for val in all_rank_num_tokens
            ]
            all_rank_num_tokens_list = [
                [val[idx_chunk] for val in all_rank_chunk_size_list]
                for idx_chunk in range(num_chunks)
            ]
            chunk_size_list = all_rank_chunk_size_list[self.rank]

            # For alltoall, replace 0 with 1 (avoid empty tensor)
            if self.enable_alltoall:
                all_rank_num_tokens_list = [
                    [1 if val == 0 else val for val in val_list]
                    for val_list in all_rank_num_tokens_list
                ]
        else:
            # When not using DP: only need current rank's input size
            all_rank_num_tokens_list = [None] * num_chunks
            chunk_size_list = self.split_chunk(x.shape[0], num_chunks)

        x_list = x.split(chunk_size_list)
        router_logits_list = router_logits.split(chunk_size_list)

        # Determine if we need multiple streams for overlapped execution
        use_multi_stream = not self.enable_alltoall and self.aux_stream is not None

        # ========== Setup auxiliary stream ==========
        if use_multi_stream:
            self.event_dict[EventType.Main].record()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.Main].wait()

        # ========== Create workspace for DeepGemmFusedMoE ==========
        workspace_0, workspace_1 = self._prepare_workspaces_for_chunk(
            all_rank_num_tokens_list, chunk_size_list, use_multi_stream
        )

        # ========== Padding empty chunk ==========
        chunked_used = torch.ones(num_chunks, dtype=torch.bool)
        if self.use_dp:
            # For empty chunk, will use chunk 0 instead. The current split heuristic
            # ensures that if an empty chunk exists, Chunk 0 contains exactly one token.
            assert x_list[0].numel() != 0, "chunk 0 shouldn't be empty"
            x_list = list(x_list)
            router_logits_list = list(router_logits_list)
            for idx_chunk in range(num_chunks):
                _x = x_list[idx_chunk]
                if _x.numel() == 0:
                    chunked_used[idx_chunk] = False
                    x_list[idx_chunk] = x_list[0]
                    router_logits_list[idx_chunk] = router_logits_list[0]
                    all_rank_num_tokens_list[idx_chunk][self.mapping.tp_rank] = (
                        all_rank_num_tokens_list[0][self.mapping.tp_rank]
                    )
            x_list = tuple(x_list)
            router_logits_list = tuple(router_logits_list)

        # ========== Execute chunking with overlap ==========
        outputs_list = []
        for idx_chunk, (x_chunk, router_logits_chunk) in enumerate(zip(x_list, router_logits_list)):
            # Calculate EPLB's first/last call
            is_first_call = idx_chunk == 0 and self.repeat_idx == 0
            is_last_call = idx_chunk == num_chunks - 1 and self.repeat_idx == self.repeat_count - 1

            if use_multi_stream:
                # Alternate between main stream and auxiliary stream
                # Each stream processes complete chunks (forward + reducescatter)
                if idx_chunk % 2 == 0:
                    # Even chunk: execute on auxiliary stream
                    with torch.cuda.stream(self.aux_stream):
                        outputs = self._forward_chunk_impl(
                            x_chunk,
                            router_logits_chunk,
                            output_dtype,
                            all_rank_num_tokens_list[idx_chunk],
                            use_dp_padding,
                            is_first_call,
                            is_last_call,
                            do_finalize,
                            workspace=workspace_0,
                        )
                else:
                    # Odd chunk: execute on main stream
                    outputs = self._forward_chunk_impl(
                        x_chunk,
                        router_logits_chunk,
                        output_dtype,
                        all_rank_num_tokens_list[idx_chunk],
                        use_dp_padding,
                        is_first_call,
                        is_last_call,
                        do_finalize,
                        workspace=workspace_1,
                    )
            else:
                # No overlap
                outputs = self._forward_chunk_impl(
                    x_chunk,
                    router_logits_chunk,
                    output_dtype,
                    all_rank_num_tokens_list[idx_chunk],
                    use_dp_padding,
                    is_first_call,
                    is_last_call,
                    do_finalize,
                    workspace=workspace_0,
                )

            if chunked_used[idx_chunk]:
                outputs_list.append(outputs)

        # ========== Wait for auxiliary stream to complete ==========
        if use_multi_stream:
            # Wait for auxiliary stream to complete all its chunks
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.MoeChunkingOverlap].record()
            self.event_dict[EventType.MoeChunkingOverlap].wait()

        # ========== Concatenate outputs from all chunks ==========
        outputs = torch.cat(outputs_list)

        return outputs

    # ========== Backend Validation ==========

    def validate_backend(self, backend: MoE):
        """
        Validate MOE backend.

        It validates that:
        1. Backend is not None
        2. If EPLB is enabled, backend must support routing separation

        Args:
            backend: MoEBackend instance to set

        Raises:
            ValueError: If backend is incompatible with current configuration

        Note: EPLB initialization is done in __init__, not in setter.
              Setter only validates compatibility.
        """
        if backend is None:
            raise ValueError("Backend cannot be None")

        # Validate EPLB compatibility
        if self._using_load_balancer() and not backend._supports_load_balancer():
            raise ValueError(
                f"EPLB is enabled but backend {backend.__class__.__name__} "
                f"does not support load balancer. "
                f"Either disable EPLB or use a backend that supports load balancer."
            )

    # ========== Helper Methods ==========

    def _is_using_nvlink_two_sided(self) -> bool:
        """Check if using NVLinkTwoSided communication strategy"""
        return isinstance(self.comm, NVLinkTwoSided)

    def _is_using_nvlink_one_sided(self) -> bool:
        """Check if using NVLinkOneSided communication strategy"""
        return isinstance(self.comm, NVLinkOneSided)

    def _get_nvlink_onesided_moe_output(
        self,
        all_rank_num_tokens: Optional[List[int]],
        output_dtype: Optional[torch.dtype],
    ) -> Optional[torch.Tensor]:
        """
        Get workspace output buffer for NVLinkOneSided communication backend.

        This method handles moe_output allocation for both CutlassFusedMoE and TRTLLMGenFusedMoE
        when using NVLinkOneSided communication strategy.

        Args:
            all_rank_num_tokens: Token counts per rank
            output_dtype: Output data type

        Returns:
            moe_output tensor if NVLinkOneSided is used and backend supports it, None otherwise
        """
        if not isinstance(self.comm, NVLinkOneSided):
            return None

        if not self.backend.supports_moe_output_in_alltoall_workspace():
            # Ensure payload_in_workspace is False if backend doesn't support it
            self.comm.payload_in_workspace = False
            return None

        # Determine workspace dtype and whether backend supports workspace output
        workspace_dtype = output_dtype
        if isinstance(self.backend, TRTLLMGenFusedMoE):
            # TRTLLMGen specific configuration
            self.comm.invalid_token_expert_id = -1
            workspace_dtype = torch.bfloat16

        # Calculate runtime max tokens per rank
        assert all_rank_num_tokens is not None, (
            "all_rank_num_tokens must be provided for NVLinkOneSided backend"
        )
        runtime_max_tokens_per_rank = max(all_rank_num_tokens)

        # Get workspace-backed output tensor
        moe_output = self.comm.get_combine_payload_tensor_in_workspace(
            runtime_max_tokens_per_rank, self.hidden_size, workspace_dtype
        )

        # Dynamically enable payload_in_workspace for this forward pass
        self.comm.payload_in_workspace = True
        return moe_output

    def _get_backend_kwargs(
        self,
        router_logits: Optional[torch.Tensor] = None,
        do_finalize: bool = True,
        all_rank_num_tokens: Optional[List[int]] = None,
        output_dtype: Optional[torch.dtype] = None,
        x: Optional[torch.Tensor] = None,
        workspace: Optional[dict] = None,
    ) -> Dict:
        """
        Get backend-specific keyword arguments for run_moe

        Returns backend-specific parameters that are not part of the common run_moe interface.
        Different backends need different parameters - this method provides them via kwargs.

        TODO: This is not finalized, will be updated later.
        Common kwargs (multiple backends):
            - cluster_size, cluster_rank: Cutlass, DeepGemm
            - min_latency_mode: Cutlass, WideEP, DeepGemm
            - use_fused_finalize: Cutlass, WideEP
            - tuner_num_tokens, tuner_top_k: Cutlass, WideEP

        Backend-specific kwargs:
            - Cutlass: swizzled_input_sf, enable_alltoall, output_tensor
            - WideEP: swizzled_input_sf (fixed False), use_all_to_all
            - DeepGemm: workspace, permutation tensors
            - TRTLLMGen: router_logits, do_finalize, moe_output

        Args:
            router_logits: Router logits tensor (for TRTLLMGen backend)
            do_finalize: Whether to finalize output (for TRTLLMGen backend)
            all_rank_num_tokens: Token counts per rank (for TRTLLMGen backend moe_output)
            output_dtype: Output data type
            x: Input tensor (for calculating tuner_num_tokens in Cutlass)

        Returns:
            Dict: Backend-specific keyword arguments
        """
        kwargs = {}

        # Common parameters for Cutlass and DeepGemm
        if self.backend.__class__ in (CutlassFusedMoE, DeepGemmFusedMoE, CuteDslFusedMoE):
            pass

        # Cutlass-specific parameters
        if self.backend.__class__ == CutlassFusedMoE:
            # Determine if scaling factors are swizzled based on communication flow
            # In post-quant communication (quantize -> dispatch), scaling factors are not swizzled
            # In pre-quant communication (dispatch -> quantize), scaling factors are swizzled
            supports_post_quant = self.comm is not None and self.comm.supports_post_quant_dispatch()
            kwargs["is_sf_swizzled"] = not supports_post_quant
            kwargs["output_dtype"] = output_dtype

            # Prepare additional information for profiling in case padding is applied when using alltoall.
            # Only the non-alltoall case is considered for profiling in the warmup phase.
            # Therefore, to get the correct tactics during the actual inference, the inputs to the tuner
            # should be the same as when not using alltoall.
            kwargs["enable_alltoall"] = self.enable_alltoall
            if self.enable_alltoall:
                if all_rank_num_tokens is not None:
                    kwargs["tuner_num_tokens"] = sum(all_rank_num_tokens)
                else:
                    kwargs["tuner_num_tokens"] = (
                        x.shape[0] * self.mapping.tp_size if x is not None else None
                    )
                kwargs["tuner_top_k"] = self.routing_method.top_k

            # Get moe_output for NVLinkOneSided backend
            kwargs["moe_output"] = self._get_nvlink_onesided_moe_output(
                all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
            )

        # CuteDSL-specific parameters
        elif self.backend.__class__ == CuteDslFusedMoE:
            kwargs["enable_alltoall"] = self.enable_alltoall

            # Get moe_output for NVLinkOneSided backend
            kwargs["moe_output"] = self._get_nvlink_onesided_moe_output(
                all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
            )

        # DeepGemm-specific parameters
        elif self.backend.__class__ == DeepGemmFusedMoE:
            if workspace is not None:
                kwargs["workspace"] = workspace

        # TRTLLMGen-specific parameters
        elif self.backend.__class__ == TRTLLMGenFusedMoE:
            # Determine router_logits based on whether routing has been done
            # If backend doesn't support load balancer, routing is done before communication
            # In that case, router_logits should be None (routing already done)
            router_logits_arg = None
            if not self.backend._supports_load_balancer():
                # For fused routing backends, router_logits is only needed if routing hasn't been done yet
                router_logits_arg = router_logits

            kwargs["router_logits"] = router_logits_arg
            kwargs["do_finalize"] = do_finalize

            # Get moe_output for NVLinkOneSided backend
            kwargs["moe_output"] = self._get_nvlink_onesided_moe_output(
                all_rank_num_tokens=all_rank_num_tokens, output_dtype=output_dtype
            )

        return kwargs

    def create_weights(self):
        """
        Create weights - delegated to backend

        """
        assert hasattr(self.backend, "create_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement create_weights()"
        )
        return self.backend.create_weights()

    def load_weights(self, weights: List[Dict]):
        """
        Load weights - delegated to backend

        """
        assert hasattr(self.backend, "load_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement load_weights()"
        )
        return self.backend.load_weights(weights)

    def post_load_weights(self):
        """
        Post load weights processing - delegated to backend

        """
        assert hasattr(self.backend, "post_load_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement post_load_weights()"
        )
        return self.backend.post_load_weights()

    def process_weights_after_loading(self):
        """
        Process weights after loading - delegated to backend

        """
        assert hasattr(self.backend, "process_weights_after_loading"), (
            f"Backend {self.backend.__class__.__name__} must implement process_weights_after_loading()"
        )
        return self.backend.process_weights_after_loading()

    def pre_reload_weights(self):
        """
        Pre reload weights - delegated to backend
        """
        assert hasattr(self.backend, "pre_reload_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement pre_reload_weights()"
        )
        return self.backend.pre_reload_weights()

    # ========== Communication and Quantization Properties ==========

    @property
    def enable_alltoall(self):
        """
        Check if alltoall is enabled

        This delegates to the communication strategy to determine if alltoall is available.

        """
        if self.comm is None:
            return False
        # Simplified check - AllGather strategy means no alltoall
        return not isinstance(self.comm, AllGatherReduceScatter)

    @property
    def _weights_created(self):
        """Check if weights have been created (required for quantization properties)"""
        assert hasattr(self.backend, "_weights_created"), (
            f"Backend {self.backend.__class__.__name__} must have _weights_created attribute"
        )
        return self.backend._weights_created

    # ========== Explicit Backend Attribute Proxies ==========
    # These properties delegate to backend for commonly accessed attributes
    # TODO: Unify the property access to backend in ConfigurableMoE.
    # At the same time, we need to keep the existing test cases working.

    @property
    def quant_method(self):
        """Delegate quant_method to backend"""
        return getattr(self.backend, "quant_method", None)

    @property
    def w3_w1_weight(self):
        """Delegate w3_w1_weight to backend"""
        return getattr(self.backend, "w3_w1_weight", None)

    @property
    def w2_weight(self):
        """Delegate w2_weight to backend"""
        return getattr(self.backend, "w2_weight", None)

    @property
    def has_nvfp4(self):
        """Delegate has_nvfp4 to backend"""
        return getattr(self.backend, "has_nvfp4", False)

    def forward_fake(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Fake forward for shape inference during torch.compile

        Delegates to backend's forward_fake if available, otherwise calls parent's forward_fake

        Args:
            x: Input tensor
            router_logits: Router logits for expert selection
            do_finalize: Whether to finalize MoE output
            output_dtype: Output data type
            all_rank_num_tokens: Token counts per rank
            use_dp_padding: Whether to use data parallel padding
            **kwargs: Additional arguments

        Returns:
            Empty tensor(s) with correct shape for torch.compile
        """
        if hasattr(self.backend, "forward_fake"):
            # Backend has forward_fake, delegate to it
            return self.backend.forward_fake(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )
        else:
            # Backend doesn't have forward_fake, use parent's implementation
            return super().forward_fake(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )
