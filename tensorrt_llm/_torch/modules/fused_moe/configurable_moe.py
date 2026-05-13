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

from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.interface import MoE, MoESchedulerKind
from tensorrt_llm._torch.modules.fused_moe.routing import BaseMoeRoutingMethod
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager
from tensorrt_llm._torch.utils import AuxStreamType, EventType, Fp4QuantizedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from .communication import AllGatherReduceScatter, Communication, CommunicationFactory
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .moe_scheduler import MoEScheduler, create_moe_scheduler

# Attributes that ConfigurableMoE owns (computed in MoE.__init__ from real
# layer_idx + load balancer) and must be mirrored onto the backend after
# the backend was constructed with layer_idx=None / init_load_balancer=False.
# Adding a new EPLB-derived attribute? Append it here so the sync stays
# in one place and __init__ does not silently drift.
_BACKEND_SYNC_ATTRS = (
    "layer_idx",
    "layer_idx_str",
    "num_slots",
    "layer_load_balancer",
    "repeat_count",
    "repeat_idx",
    "initial_local_expert_ids",
    "initial_global_assignments",
    "slot_start",
    "slot_end",
    "expert_size_per_partition",
)


class ConfigurableMoE(MoE):
    """
    Configurable MoE layer using composition pattern with automatic configuration

    This class orchestrates the MoE execution flow by composing:
    - moe_backend: Existing FusedMoE implementation used as a pluggable backend.
                   Currently supported backends (see ``create_moe.get_moe_cls``):
                   CutlassFusedMoE, TRTLLMGenFusedMoE, DeepGemmFusedMoE,
                   CuteDslFusedMoE, DenseGEMMFusedMoE, MegaMoEDeepGemm.
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
            - tune_max_num_tokens: Max tokens for profiling (passed to backend)
            - Other backend-specific arguments

    Key Attributes:
        - backend: MoE computation backend (auto-created attribute)
        - comm: Communication strategy (auto-created attribute, can be None)
        - layer_load_balancer: EPLB instance (auto-detected, optional)

    Auto-Detection:
        - EPLB: Enabled if get_moe_load_balancer() is not None
        - Backend: Selected by ``model_config.moe_backend`` via ``create_moe.get_moe_cls``;
                   defaults to CutlassFusedMoE when the requested backend is unsupported
                   for the active quant/SM config.
        - Communication: Auto-selected based on hardware (NVLINK > DeepEP > AllGather);
                         skipped entirely for FUSED_COMM backends (e.g. MegaMoEDeepGemm).
    """

    @classmethod
    def can_implement(
        cls,
        quant_algo,
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ):
        """
        ConfigurableMoE is a wrapper class that delegates to specific backends.

        To check capability, query the specific backend class directly:
        - CutlassFusedMoE.can_implement(quant_algo, dtype_activation, swiglu_gptoss_style)
        - TRTLLMGenFusedMoE.can_implement(quant_algo, dtype_activation, swiglu_gptoss_style)
        - etc.

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation data type
            swiglu_gptoss_style: Whether swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled

        Returns:
            Tuple[bool, Optional[str]]: Always returns (False, reason)
        """
        del quant_algo, dtype_activation, swiglu_gptoss_style  # Unused - wrapper class
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

        # ========== Create MoE Backend (selected by model_config.moe_backend) ==========
        self._create_and_sync_backend(
            model_config=model_config,
            routing_method=routing_method,
            override_quant_config=override_quant_config,
            **kwargs,
        )

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

        # ========== Optional DWDP integration ==========
        self.dwdp_manager = get_global_dwdp_manager()
        self.dwdp_handle_collector = None
        self.dwdp_rank = None
        self.enable_dwdp = False
        if self.dwdp_manager is not None and self._should_enable_dwdp():
            self.enable_dwdp = True
            self.dwdp_handle_collector = self.dwdp_manager.add_layer(
                layer_idx=self.layer_idx,
            )
            self.dwdp_rank = self.dwdp_manager.dwdp_rank
            self.backend.dwdp_handle_collector = self.dwdp_handle_collector

        # Mark as _weights_removed to skip ConfigurableMoE's post_load_weights in model_loader
        # The backend's post_load_weights will be called directly by model_loader
        # This avoids duplicate post_load_weights calls (once for ConfigurableMoE, once for backend)
        # TODO: in the future, all the weights related work should be done only in backend.
        self._weights_removed = True

        # ========== Create forward scheduler (ExternalComm / FusedComm) ==========
        # Constructed last so the scheduler may safely read any wrapper state
        # (comm, aux_stream, event_dict, moe_max_num_tokens, dwdp_*) at init
        # time without ordering surprises. Selection is based on
        # ``backend.scheduler_kind`` set on the backend class.
        self.scheduler: MoEScheduler = create_moe_scheduler(self)

    @staticmethod
    @contextmanager
    def _temporarily_skip_weight_creation(model_config: ModelConfig):
        """Force ``model_config.skip_create_weights_in_init = True`` for the duration.

        The backend is constructed with ``layer_idx=None`` and an unset load
        balancer, so weight allocation must be deferred until ConfigurableMoE
        has synced the real EPLB-derived attributes onto the backend (see
        ``_BACKEND_SYNC_ATTRS``). The flag is also flipped through the
        ``_frozen`` Pydantic guard, hence the bracketing dance. Using a
        contextmanager guarantees the original state is restored even if
        backend construction raises.
        """
        previous = model_config.skip_create_weights_in_init
        model_config._frozen = False
        model_config.skip_create_weights_in_init = True
        model_config._frozen = True
        try:
            yield
        finally:
            model_config._frozen = False
            model_config.skip_create_weights_in_init = previous
            model_config._frozen = True

    def _create_and_sync_backend(
        self,
        *,
        model_config: ModelConfig,
        routing_method: BaseMoeRoutingMethod,
        override_quant_config: Optional["QuantConfig"],
        **kwargs,
    ) -> None:
        """Build the MoE backend, mirror EPLB attrs, then create weights.

        Why this dance:
        - ``init_load_balancer=False`` / ``without_comm=True``: the backend
          would otherwise re-register itself with the load balancer and
          initialize its own communication; ConfigurableMoE owns both.
        - ``layer_idx=None``: the wrapper passes the real ``layer_idx`` to
          ``MoE.__init__`` to drive load-balancer setup. The backend
          receives ``None`` so its own EPLB hooks no-op until we sync the
          real values via ``_BACKEND_SYNC_ATTRS`` below.
        - ``skip_create_weights_in_init=True`` (via contextmanager): weights
          depend on ``layer_load_balancer`` / ``initial_local_expert_ids``
          / etc., which only become known after the sync. Defer weight
          creation to the explicit ``backend.create_weights()`` call below.
        """
        from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend, get_moe_cls

        moe_cls = get_moe_cls(model_config, override_quant_config=override_quant_config)

        with self._temporarily_skip_weight_creation(model_config):
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
        self.use_flashinfer = getattr(self.backend, "use_flashinfer", False)

        # Mirror wrapper-owned EPLB / layer-id state onto the backend so any
        # backend code path that reads e.g. ``self.layer_load_balancer`` or
        # ``self.num_slots`` sees the real values resolved by MoE.__init__.
        if self.backend is not None:
            for attr in _BACKEND_SYNC_ATTRS:
                setattr(self.backend, attr, getattr(self, attr))

        # Sync done -- now the backend has enough info to allocate weight
        # tensors with the right shard / slot count.
        if not model_config.skip_create_weights_in_init:
            self.backend.create_weights()

    def _supports_load_balancer(self) -> bool:
        """Check if this MoE implementation supports load balancer.

        ``MoE.__init__`` can query this before ``ConfigurableMoE`` has
        created ``self.backend``. In that initialization window, fall back to
        the wrapper-level DP/parallelism condition; ``validate_backend`` runs
        after backend construction and enforces the backend-specific answer.
        """
        # During initialization, backend might not be created yet.
        # Backend-specific support is checked later by validate_backend.
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

    def _should_enable_dwdp(self) -> bool:
        # DWDP is currently supported only for CuteDslFusedMoE with NVFP4 quantization.
        if not isinstance(self.backend, CuteDslFusedMoE):
            return False

        quant_config = getattr(self.backend, "quant_config", None)
        if quant_config is None:
            quant_config = getattr(self.model_config, "quant_config", None)
        if quant_config is None:
            return False

        quant_mode = getattr(quant_config, "layer_quant_mode", None)
        return bool(
            quant_mode is not None and hasattr(quant_mode, "has_nvfp4") and quant_mode.has_nvfp4()
        )

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
        Calculate how many chunks are needed.

        Uses ep_size * max(all_rank_num_tokens) when A2A communication is active,
        because the A2A recv buffer is shaped [ep_size, max_tokens_per_rank, hidden]
        regardless of how tokens are distributed across ranks. This matches the
        actual memory footprint of the MoE GEMM workspace.
        """
        if self.use_dp and self.comm is not None:
            num_rows = self.mapping.moe_ep_size * max(all_rank_num_tokens)
        else:
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
            all_rank_max_num_tokens = max(all_rank_num_tokens)
            logger.info(
                f"Communication strategy {self.comm.__class__.__name__} "
                f"cannot be used (num_chunks={num_chunks}, max_num_tokens={all_rank_max_num_tokens}). "
                f"Falling back to AllGatherReduceScatter."
            )

            self.comm.destroy()
            self.comm = AllGatherReduceScatter(mapping=self.mapping)

    def destroy(self):
        """Release communication resources.

        Must be called on ALL ranks before the module is discarded.
        DeepEP Buffer.__del__ calls intranode::barrier (a collective op);
        without an explicit, synchronous release, non-deterministic GC
        timing across ranks causes some to enter the barrier while others
        proceed, resulting in an indefinite hang.

        Prefer using ConfigurableMoE as a context manager (``with``) so
        that destroy() is called automatically on scope exit.
        """
        if self.comm is not None:
            self.comm.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.destroy()

    def _create_comm_strategy_auto(self) -> Communication:
        """
        Auto-create the best communication strategy based on hardware and configuration

        Uses factory to select optimal strategy. Backends whose fused kernel
        owns cross-rank exchange (``scheduler_kind=FUSED_COMM``) skip
        host-side comm entirely; layering Communication.dispatch / combine
        on top of the fused exchange would double-count traffic and break
        the in-kernel NVLink barrier semantics.
        """
        if self.backend.scheduler_kind == MoESchedulerKind.FUSED_COMM:
            return None
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
            use_flashinfer=self.use_flashinfer,
            hidden_size=self.hidden_size,
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
        """Forward entry point.

        Acts as a thin wrapper that:

        1. Validates / fills ``output_dtype``.
        2. Delegates the per-pass execution to ``self.scheduler`` (chosen
           once at init time from ``backend.scheduler_kind``).
        3. Records DWDP compute/prefetch (per layer, not per chunk).
        4. Advances the EPLB ``repeat_idx``.

        DP-padding handling and chunking live in the scheduler.
        """
        del kwargs

        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
        else:
            output_dtype = x.dtype

        outputs = self.scheduler.forward(
            x,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        # DWDP: record compute and trigger next prefetch (per-layer, not per-chunk).
        # Owned at the wrapper because schedulers must not run it twice (external-comm
        # might enter via single- or multi-chunk paths).
        if self.enable_dwdp:
            self.dwdp_manager.record_compute_and_prefetch_next(self.layer_idx)

        # EPLB repeat counter: advance once per forward, regardless of chunk count.
        # Schedulers are forbidden from rotating ``repeat_idx`` themselves.
        self.repeat_idx = (self.repeat_idx + 1) % self.repeat_count

        return outputs

    # ========== Backend Validation ==========

    def validate_backend(self, backend: MoE):
        """Validate MoE backend compatibility with this ConfigurableMoE.

        Generic checks (always run):
          1. ``backend`` is not None.
          2. If EPLB is enabled, the backend must support routing
             separation (``backend._supports_load_balancer()``).

        Backend-specific checks are delegated to
        ``backend.validate_configurable_moe(self)``; backends with extra
        constraints (e.g. fused-comm backends rejecting dynamic
        EPLB) override that hook. EPLB / num_slots / ep_size are already
        populated on ``self`` by ``MoE.__init__`` -> ``_init_load_balancer``
        before this is called, so backends may inspect them directly.
        """
        if backend is None:
            raise ValueError("Backend cannot be None")

        if self._using_load_balancer() and not backend._supports_load_balancer():
            raise ValueError(
                f"EPLB is enabled but backend {backend.__class__.__name__} "
                f"does not support load balancer. "
                f"Either disable EPLB or use a backend that supports load balancer."
            )

        backend.validate_configurable_moe(self)

    def create_weights(self):
        """
        Create weights - delegated to backend

        """
        assert hasattr(self.backend, "create_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement create_weights()"
        )
        return self.backend.create_weights()

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        """
        Load weights - delegated to backend

        """
        assert hasattr(self.backend, "load_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement load_weights()"
        )
        return self.backend.load_weights(weights, allow_partial_loading)

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
