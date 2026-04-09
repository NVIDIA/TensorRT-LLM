# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from typing import Any, Dict, List, Optional, Union

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils import fp4_utils

from ...autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...distributed import allgather
from ...memory_buffer_utils import get_memory_buffers
from ...model_config import ModelConfig
from ...utils import (
    AuxStreamType,
    EventType,
    Fp4QuantizedTensor,
    deep_gemm_tuning_buckets,
    prev_deep_gemm_bucket,
    swizzle_sf,
    unswizzle_sf,
)
from .green_context import create_sm_only_gc_streams
from .interface import MoE, MoEWeightLoadingMode
from .quantization import NVFP4CuteDslFusedMoEMethod
from .routing import BaseMoeRoutingMethod


@torch.compile(options={"max-autotune": True})
def gen_fc2_alpha_fused(
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    alpha: Optional[torch.Tensor],
    alpha_max: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate fc2 alpha values, optionally normalized for FC1 alpha_post fusion.

    Instead of:
        1. zeros() -> scatter_() -> multiply with alpha (operates on large [N, E] tensor)

    We do:
        1. Gather alpha values for selected experts (small [N, top_k] tensor)
        2. Multiply scales with gathered alpha (small tensor operation)
        3. Optionally normalize by alpha_max for FC1 alpha_post fusion
        4. Scatter to output (single write to large tensor)

    This reduces memory bandwidth by avoiding read-modify-write on the large tensor.

    Args:
        token_selected_experts: Expert indices for each token [num_tokens, top_k]
        token_final_scales: Final scaling factors [num_tokens, top_k]
        alpha: Per-expert alpha values [expert_size]
        alpha_max: Max alpha value for normalization (optional)
        output: Pre-allocated output buffer [num_tokens, expert_size] (optional).
                If None, a new tensor will be allocated (not compatible with CUDA graph).
    """
    # Pre-compute scaled values on small tensor [num_tokens, top_k]
    if alpha is not None:
        # Gather alpha for selected experts: alpha[expert_idx] for each selection
        gathered_alpha = alpha[token_selected_experts.long()]  # [num_tokens, top_k]
        scaled_values = token_final_scales * gathered_alpha
    else:
        scaled_values = token_final_scales

    # Normalize by alpha_max for FC1 alpha_post fusion
    if alpha_max is not None:
        scaled_values = scaled_values / alpha_max

    # Use pre-allocated output or create new tensor
    if output is not None:
        output.zero_()
        fc2_alpha = output
    else:
        assert alpha is not None, (
            "alpha must be provided when output buffer is not pre-allocated, "
            "since expert_size cannot be inferred from token_final_scales alone"
        )
        num_tokens = token_selected_experts.shape[0]
        expert_size = alpha.shape[0]
        fc2_alpha = torch.zeros(
            [num_tokens, expert_size],
            dtype=torch.float32,
            device=token_selected_experts.device,
        )

    return fc2_alpha.scatter_(1, token_selected_experts.long(), scaled_values)


class DenseGEMMGCSMRunner(TunableRunner):
    """TunableRunner that sweeps the FC1 SM count for the Green Context overlap path.

    Each tactic is an integer representing the requested SM count assigned to FC1.
    The router partition receives the remaining SMs.  GreenContext stream pairs for
    all candidate SM splits are pre-created in ``DenseGEMMFusedMoE.__init__`` and
    stored in ``moe_layer._gc_stream_pool``.
    """

    tuning_config_cache: dict = {}

    def __init__(self, moe_layer: "DenseGEMMFusedMoE") -> None:
        super().__init__()
        self.moe = moe_layer

    def unique_id(self):
        moe = self.moe
        return (
            moe.num_experts,
            moe.hidden_size,
            moe.intermediate_size_per_partition,
            moe.expert_size_per_partition,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        candidates = list(self.moe._gc_stream_pool.keys())
        # print(
        #     f"[DenseGEMMGCSMRunner] get_valid_tactics: num_tokens={inputs[0].shape[0]}, "
        #     f"fc1_sms candidates={candidates}"
        # )
        return candidates

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0,
                        0,
                        # Use the same fine-grained bucket grid as the inner
                        # GEMM runners so that the outer sm-split tactic is
                        # tuned at the same resolution as the inner tactics.
                        deep_gemm_tuning_buckets,
                        # Floor lookup: map actual num_tokens to the largest
                        # cached bucket strictly smaller than x.  This ensures
                        # the cached tactic was profiled with fewer tokens than
                        # the actual workload (conservative choice).
                        prev_deep_gemm_bucket,
                    ),
                ),
                constraint_specs=(
                    # x_sf: shape[0] tracks num_tokens (pre-swizzle, 2-D)
                    ConstraintSpec(1, 0, lambda shapes: shapes[0][0]),
                    # router_input: shape[0] tracks num_tokens
                    ConstraintSpec(3, 0, lambda shapes: shapes[0][0]),
                ),
                use_cold_l2_cache=True,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        *,
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        x, x_sf, router_weight_t, router_input = inputs
        num_tokens = x.shape[0]
        pool = self.moe._gc_stream_pool
        # do_preparation is called by the autotuner with tactic=-1 before the
        # actual sweep; use the first pool entry so the kernel compiles.
        fc1_sms = tactic if tactic in pool else next(iter(pool))
        if do_preparation:
            # Two-phase inner-runner warm-up before the outer CUDA-graph capture.
            #
            # Phase A — full-SM tuning (once, cheap in the outer dimension):
            #   Call _warm_inner_runners_full_sm() with sm_budget=0 so the inner
            #   autotuners (CuteDSLNVFP4DenseGemmSwigluRunner and
            #   CuteDSLBf16BlackwellGemmRunner) profile all token-buckets once
            #   with the full hardware SM count and cache the best tactic.
            #   Because sm_budget is excluded from both runners' unique_id(), this
            #   single entry is shared across every fc1_sms split.
            #
            # Phase B — per-candidate kernel-compilation warm-up (S passes, fast):
            #   Iterate over all fc1_sms candidates as before. Since the inner
            #   autotuner caches are now warm (Phase A), choose_one() returns the
            #   cached full-SM tactic without re-profiling.  The only work done is
            #   kernel compilation for the specific (tactic, max_active_clusters)
            #   combination corresponding to each split, which is required before
            #   CUDA-graph capture (cute.compile() is not graph-safe).
            #
            # Net cost: O(inner_buckets × inner_tactics) + O(S) instead of
            #           O(S × inner_buckets × inner_tactics).
            # print(
            #     f"[DenseGEMMGCSMRunner] [preparation] Phase A: warming inner runners "
            #     f"with full SM, num_tokens={num_tokens}"
            # )
            try:
                self.moe._warm_inner_runners_full_sm(x, x_sf, router_weight_t, router_input)
            except Exception as e:
                logger.warning(
                    f"[DenseGEMMGCSMRunner] full-SM inner warm-up failed, "
                    f"num_tokens={num_tokens}: {e}"
                )

            for cand_fc1_sms in pool:
                # print(
                #     f"[DenseGEMMGCSMRunner] [preparation] Phase B: kernel compilation "
                #     f"fc1_sms={cand_fc1_sms}, num_tokens={num_tokens}"
                # )
                try:
                    self.moe._run_moe_nvfp4_gc_impl(
                        x, x_sf, router_weight_t, router_input, cand_fc1_sms
                    )
                except Exception as e:
                    # Some candidates may be incompatible with the hardware GC
                    # topology (e.g. unsupported cluster shape for a given SM
                    # count).  Catch and log the failure so the outer autotuner
                    # can still profile all remaining candidates; this candidate
                    # will be naturally rejected during the outer profiling sweep.
                    logger.warning(
                        f"[DenseGEMMGCSMRunner] preparation failed for "
                        f"fc1_sms={cand_fc1_sms}, num_tokens={num_tokens}: {e}"
                    )
        else:
            pass
            # print(
            #     f"[DenseGEMMGCSMRunner] fc1_sms={fc1_sms}, "
            #     f"num_tokens={num_tokens}"
            # )
        return self.moe._run_moe_nvfp4_gc_impl(x, x_sf, router_weight_t, router_input, fc1_sms)


class DenseGEMMFusedMoE(MoE):
    """CuteDSL DenseGEMM flow of fused mixture of experts (MoE) Layer — Green Context variant.

    This backend uses CuTe DSL dense GEMM kernels with fused SwiGLU for MoE
    computation. It supports NVFP4 quantization only and is restricted to
    SM100/SM103 (Blackwell) architectures.

    Unlike CutlassFusedMoE which uses per-expert scattered GEMM, DenseGEMM
    packs all experts into a single dense matrix and uses standard GEMM operations,
    which can be more efficient for small token counts (min-latency scenarios).

    This variant (gc) uses GreenContext for hardware-level SM isolation so that
    FC1 and the Router GEMM truly run in parallel without SM contention. The SM
    split follows the same formula as the smp variant (raw SM count), but enforcement
    is via hardware GreenContext partitions rather than soft sm_budget kernel hints.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    # Memory buffer pool for CUDA graph compatibility
    buffers = get_memory_buffers()

    # DenseGEMM only supports SM100 and SM103 (Blackwell CuTe DSL kernels).
    _SUPPORTED_SM_VERSIONS = (100, 103)

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> tuple:
        """Check if DenseGEMMFusedMoE can implement the given configuration.

        DenseGEMMFusedMoE supports:
        - NVFP4 quantization only
        - SM100/SM103 (Blackwell) only
        - SwiGLU activation only (swiglu_gptoss_style not supported)
        """
        from tensorrt_llm._utils import get_sm_version

        from .interface import _warn_and_return

        sm_version = get_sm_version()
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            return _warn_and_return(
                f"DenseGEMMFusedMoE requires SM {cls._SUPPORTED_SM_VERSIONS}, got SM{sm_version}"
            )

        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"DenseGEMMFusedMoE only supports NVFP4 quantization (got quant_algo={quant_algo})"
            )

        if swiglu_gptoss_style:
            return _warn_and_return("DenseGEMMFusedMoE does not support swiglu_gptoss_style")

        return (True, None)

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
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type=None,
    ):
        # DenseGEMM CuTe DSL kernels only support SM100 and SM103.
        from tensorrt_llm._utils import get_sm_version

        from ...utils import ActivationType

        sm_version = get_sm_version()
        assert sm_version in self._SUPPORTED_SM_VERSIONS, (
            f"DenseGEMMFusedMoE only supports SM {self._SUPPORTED_SM_VERSIONS} "
            f"(got SM {sm_version}). The CuTe DSL kernels require Blackwell architecture."
        )

        # DenseGEMM kernel hardcodes SwiGLU fusion — reject other activation types
        # before calling super().__init__() to fail fast with a clear message.
        if activation_type is None:
            activation_type = ActivationType.Swiglu
        assert activation_type == ActivationType.Swiglu, (
            f"DenseGEMMFusedMoE only supports SwiGLU activation "
            f"(got activation_type={activation_type}). "
            f"The FC1 kernel fuses SwiGLU into the GEMM epilogue."
        )

        # FC2 DenseGEMM kernel tiles K dimension with MMA tile size 256.
        # weight_per_expert (= intermediate_size) must be 256-aligned so that
        # expert boundaries align with MMA tile boundaries.
        _MMA_TILE_K = 256
        assert intermediate_size % _MMA_TILE_K == 0, (
            f"DenseGEMMFusedMoE requires intermediate_size to be a multiple of "
            f"{_MMA_TILE_K} (got intermediate_size={intermediate_size}). "
            f"FC2 kernel cannot correctly split alpha_scale at expert boundaries "
            f"when weight_per_expert is not MMA tile-K aligned."
        )

        # Call MoE base class directly (not CutlassFusedMoE).
        # Note: `without_comm` and `apply_router_weight_on_input` are accepted
        # for API compatibility with create_moe_backend() but are not passed to
        # MoE.__init__() since DenseGEMM does not use alltoall communication.
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        # Environment variable to control fc2_alpha fusion into FC1's alpha_post.
        # Default: disabled (0). Set to "1" to enable fusion (known accuracy issue under TP).
        self.use_fused_fc2_alpha = os.environ.get("TRTLLM_MOE_FUSED_FC2_ALPHA", "0") == "1"

        # Pre-register fc2_alpha_max buffer for fused fc2_alpha optimization.
        # Populated in load_weights() with max(fc2_alpha).
        self.register_buffer("fc2_alpha_max", torch.zeros(1, dtype=torch.float32))

        device_id = torch.cuda.current_device()
        num_sms = torch.cuda.get_device_properties(device_id).multi_processor_count
        self.num_sms = num_sms
        print(f"DenseGEMMFusedMoE initializing on device {device_id} with {num_sms} SMs")

        # Initialize GreenContext SM partitions and events for FC1/Router overlap.
        # Each GreenContext provides hardware-level SM isolation so that FC1 and
        # the Router GEMM truly run in parallel without SM contention.
        # Uses PyTorch torch.cuda.GreenContext path: SM isolation only,
        # workqueue remains shared across GreenContexts.

        self._gc_cleanup = None
        if not self.use_fused_fc2_alpha:
            # Build a pool of GC stream pairs for all SM-split candidates so the
            # autotuner can profile each split and pick the best one.
            # Sweep fc1_sms from 104 to num_sms (exclusive) with step 8.
            # Step 8 matches smCoscheduledAlignment on Blackwell so every
            # candidate maps to a distinct hardware SM partition; non-aligned
            # values would be rounded to the same partition as the next
            # multiple of 8, making duplicate or incompatible GC configs.
            _alignment = 8  # smCoscheduledAlignment on Blackwell (SM100/SM103)
            _start = 108  # first sensible fc1_sms (leaves at least 1 SM for Router)
            _candidates = sorted(range(_start, num_sms, _alignment))

            self._gc_stream_pool: Dict[int, tuple] = {}
            self._gc_cleanup_fns: List = []
            for cand_fc1_sms in _candidates:
                cand_router_sms = max(1, num_sms - cand_fc1_sms)
                try:
                    fc1_s, router_s, cleanup = create_sm_only_gc_streams(
                        cand_fc1_sms, cand_router_sms, device_id
                    )
                    self._gc_stream_pool[cand_fc1_sms] = (fc1_s, router_s)
                    self._gc_cleanup_fns.append(cleanup)
                except Exception as e:
                    logger.warning(
                        f"skipping fc1_sms={cand_fc1_sms} (router_sms={cand_router_sms}): {e}"
                    )

            self.fc1_gc = None  # managed by Driver API
            self.router_gc = None
            # Pick the router stream from the first successfully-created candidate.
            _first_valid = next(iter(self._gc_stream_pool))
            router_gc_stream = self._gc_stream_pool[_first_valid][1]
        else:
            self._gc_stream_pool = {}
            self._gc_cleanup_fns = []
            self.fc1_gc = None
            self.router_gc = None
            router_gc_stream = torch.cuda.Stream()

        # Initialize auxiliary stream and events for gen_fc2_alpha_fused overlap with fc1
        if self.aux_stream_dict is None:
            self.aux_stream_dict = aux_stream_dict if aux_stream_dict is not None else {}
        if AuxStreamType.MoeFc2Alpha not in self.aux_stream_dict:
            self.aux_stream_dict[AuxStreamType.MoeFc2Alpha] = router_gc_stream
        self.event_dict = {}
        for key in [EventType.Main, EventType.MoeFc2Alpha]:
            self.event_dict[key] = torch.cuda.Event(enable_timing=False)
        self.fc1_done_event = torch.cuda.Event(enable_timing=False)

        # Weight creation
        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def __del__(self) -> None:
        """Release raw CUDA Driver handles held by all GreenContext stream pairs."""
        for cleanup in getattr(self, "_gc_cleanup_fns", []):
            try:
                cleanup()
            except Exception:
                pass  # best-effort; avoid raising in __del__
        # Legacy single-cleanup fallback (kept for safety).
        cleanup = getattr(self, "_gc_cleanup", None)
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass

    def _supports_load_balancer(self) -> bool:
        """DenseGEMMFusedMoE supports load balancer."""
        return True

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True
        ):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
            raise ValueError(
                f"{self.__class__.__name__} only supports NVFP4 quantization, "
                f"got {self.quant_config.quant_mode}."
            )
        raise ValueError(
            f"{self.__class__.__name__} requires quantization (NVFP4), "
            f"but no quantization config was provided."
        )

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        kargs = {}
        if "allow_partial_loading" in inspect.getfullargspec(self.quant_method.load_weights).args:
            kargs["allow_partial_loading"] = allow_partial_loading
        self.quant_method.load_weights(self, weights, self.weight_loading_mode, **kargs)

        # Transpose w2_weight layout: (E, H, ...) -> (H, E, ...) for dense GEMM.
        # NOTE: .contiguous() on the transposed view allocates a full-size temporary,
        # temporarily doubling peak memory.  An in-place multi-dim transpose is not
        # feasible without complex cycle-following, and this runs only once during
        # weight loading, so the trade-off is acceptable.
        w2_transposed = self.w2_weight.transpose(0, 1).contiguous()
        self.w2_weight.reshape([-1]).copy_(w2_transposed.reshape([-1]), non_blocking=True)
        del w2_transposed
        if self.has_any_quant:
            if self.has_nvfp4:
                self._transform_w2_weight_scale_for_min_latency()
                # Compute fc2_alpha_max for fused fc2_alpha optimization
                self.fc2_alpha_max.copy_(torch.max(self.fc2_alpha).reshape(1), non_blocking=True)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} only supports nvfp4 quantization, "
                    f"got {self.quant_config.quant_mode}."
                )

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)

    def _transform_w2_weight_scale_for_min_latency(self):
        """Transform w2_weight_scale for minimum latency path optimization."""
        # Calculate padded dimensions
        nrows = fp4_utils.pad_up(self.hidden_size, 128)
        ncols = fp4_utils.pad_up(
            self.intermediate_size_per_partition // self.scaling_vector_size, 4
        )

        # Clone and convert weight scale to uint8
        w2_weight_scale = self.w2_weight_scale.clone().view(torch.uint8)

        # Unswizzle the scale factor
        w2_weight_scale = unswizzle_sf(
            w2_weight_scale,
            self.hidden_size * self.expert_size_per_partition,
            self.intermediate_size_per_partition,
        )

        # Reshape and transpose for min latency layout
        w2_weight_scale = w2_weight_scale.reshape([self.expert_size_per_partition, nrows, ncols])
        w2_weight_scale = w2_weight_scale.transpose(0, 1).reshape(
            nrows, self.expert_size_per_partition * ncols
        )

        # Swizzle back with new layout
        w2_weight_scale = swizzle_sf(
            w2_weight_scale,
            self.hidden_size,
            self.expert_size_per_partition * self.intermediate_size_per_partition,
        )

        # Copy back to original tensor
        self.w2_weight_scale.copy_(
            w2_weight_scale.view(self.w2_weight_scale.dtype).view(self.w2_weight_scale.shape),
            non_blocking=True,
        )

    def quantize_input(
        self, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
    ):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation.

        Args:
            x: Input tensor to quantize
            post_quant_comm:
                If True, quantize for post-quant communication path.
                If False, quantize for non-communication path

        Returns: (x, x_sf) where x_sf is already reshaped to 2D if needed

        For quantization methods that produce scaling factors:
        - x_sf is reshaped from 1D to 2D: [num_elements] -> [batch_size, ceil_div(hidden_size, scaling_vector_size)]
        - The 2D shape is required for proper handling in alltoall/allgather operations
        - scaling_vector_size is typically the group size for block-wise quantization
        """
        x_sf = None
        if self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, (
                    "Fp4QuantizedTensor should not be swizzled before communication"
                )
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x_row = x.shape[0]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False, False
                )
        else:
            raise ValueError(
                f"{self.__class__.__name__} only supports nvfp4 quantization, "
                f"got {self.quant_config.quant_mode}."
            )

        if x_sf is not None:
            x_sf = x_sf.view(x_row, -1)

        return x, x_sf

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        router_weight_t: Optional[torch.Tensor] = None,
        router_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run MoE computation with NVFP4 quantization using Green Context stream overlap.

        Args:
            x: Input tensor
            token_selected_experts: Must be None (routing done internally)
            token_final_scales: Must be None (routing done internally)
            x_sf: Input scale factors
            enable_alltoall: Whether alltoall communication is enabled
            router_weight_t: Router weight matrix [hidden, num_experts] (transposed)
            router_input: Optional separate router input; defaults to x if None

        Note:
            The implementation is controlled by TRTLLM_MOE_FUSED_FC2_ALPHA env var (default: disabled).
            When enabled, fc2_alpha is fused into FC1's alpha_post with scalar fc2_alpha_max in FC2.
            When disabled, uses Green Context for hardware-level SM isolation so that FC1 and
            Router GEMM run in parallel on separate SM partitions.
        """
        assert self.has_nvfp4
        assert token_selected_experts is None and token_final_scales is None, (
            "DenseGEMMFusedMoE only supports internal routing path. "
            "Expected token_selected_experts/token_final_scales to be None."
        )
        assert router_weight_t is not None, (
            "DenseGEMMFusedMoE internal routing requires router_weight_t."
        )
        if router_input is None:
            router_input = x
        assert router_input.ndim == 2 and router_weight_t.ndim == 2, (
            "router_input and router_weight_t must both be rank-2 tensors."
        )
        assert router_input.shape[1] == router_weight_t.shape[0], (
            "DenseGEMMFusedMoE internal routing shape mismatch: "
            f"router_input.shape={tuple(router_input.shape)}, "
            f"router_weight_t.shape={tuple(router_weight_t.shape)}"
        )

        num_tokens = x.shape[0]

        if self.use_fused_fc2_alpha:
            # Fused path: router GEMM and FC1 run sequentially on main stream (no overlap).
            # No SM budget applied — both kernels use all available SMs for maximum throughput.
            capture_graph = torch.cuda.is_current_stream_capturing()
            fc2_alpha_buffer = DenseGEMMFusedMoE.buffers.get_buffer(
                (num_tokens, self.expert_size_per_partition),
                dtype=torch.float32,
                buffer_name="fc2_alpha",
                reserve_buffer=capture_graph,
            )
            m, n = router_input.shape[0], router_weight_t.shape[1]
            router_logits = torch.empty(m, n, dtype=torch.float32, device=router_input.device)
            torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell(
                router_input.contiguous(),
                router_weight_t.t().contiguous(),
                router_logits,
            )
            token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
            token_selected_experts = token_selected_experts.to(torch.int32)
            assert token_final_scales is not None
            assert token_final_scales.dtype == torch.float32

            # Append fused shared experts (always active, scale=1.0) if any.
            # Shared experts are the experts beyond the routed experts range
            # (expert_size_per_partition > num_routing_experts).
            num_routing_experts = router_weight_t.shape[1]
            if self.expert_size_per_partition > num_routing_experts:
                n_shared = self.expert_size_per_partition - num_routing_experts
                M_tok = router_input.shape[0]
                shared_ids = (
                    torch.arange(
                        num_routing_experts,
                        self.expert_size_per_partition,
                        dtype=torch.int32,
                        device=x.device,
                    )
                    .unsqueeze(0)
                    .expand(M_tok, n_shared)
                )
                shared_scales = torch.ones(M_tok, n_shared, dtype=torch.float32, device=x.device)
                token_selected_experts = torch.cat([token_selected_experts, shared_ids], dim=1)
                token_final_scales = torch.cat([token_final_scales, shared_scales], dim=1)

            # New implementation: fuse fc2_alpha into FC1's alpha_post
            x_sf = swizzle_sf(x_sf, num_tokens, self.hidden_size)

            # Generate normalized fc2_alpha for FC1 alpha_post fusion
            fc2_alpha_normalized = gen_fc2_alpha_fused(
                token_selected_experts,
                token_final_scales,
                self.fc2_alpha,
                self.fc2_alpha_max,  # Normalize by max for FC1 alpha_post
                fc2_alpha_buffer,  # Pre-allocated buffer
            )

            # FC1: GEMM + SwiGLU with post-SwiGLU alpha scaling (fused fc2_alpha)
            fc1_output, fc1_output_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
                x,
                self.w3_w1_weight.view(torch.uint8),
                x_sf,
                self.w3_w1_weight_scale,
                self.fc31_alpha,
                fc2_alpha_normalized,  # Pass normalized fc2_alpha as alpha_post
                self.fc2_input_scale,
                expert_count=self.expert_size_per_partition,
                weight_per_expert=2 * self.intermediate_size_per_partition,
                output_dtype=torch.float4_e2m1fn_x2,
                scaling_vector_size=self.scaling_vector_size,
            )

            # FC2: Standard nvfp4_gemm with scalar alpha = fc2_alpha_max
            final_hidden_states = torch.ops.trtllm.nvfp4_gemm(
                fc1_output.view(torch.uint8),
                self.w2_weight.view(torch.uint8).reshape(self.hidden_size, -1),
                fc1_output_sf.view(torch.uint8).reshape(-1),
                self.w2_weight_scale.view(torch.uint8),
                self.fc2_alpha_max,
                torch.bfloat16,
                to_userbuffers=False,
                allowed_backends="cutlass,cublaslt,cutedsl,cuda_core",
            )
        else:
            # Green Context path: use AutoTuner to find the optimal FC1 SM split.
            # Pass unswizzled x_sf so the constraint spec (shape[0] == num_tokens)
            # remains valid; swizzle_sf is applied inside _run_moe_nvfp4_gc_impl.
            tuner = AutoTuner.get()
            runner = DenseGEMMGCSMRunner(self)
            inputs = [x, x_sf, router_weight_t, router_input]
            _, best_tactic = tuner.choose_one(
                "DenseGEMMFusedMoE::run_moe_nvfp4_gc",
                [runner],
                runner.get_tuning_config(),
                inputs,
            )
            final_hidden_states = runner(inputs, tactic=best_tactic)

        return final_hidden_states

    def _warm_inner_runners_full_sm(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        router_weight_t: torch.Tensor,
        router_input: torch.Tensor,
    ) -> None:
        """Warm FC1 and Router GEMM autotuner caches with the full SM budget.

        This must be called once before the outer profiling loop in
        DenseGEMMGCSMRunner.do_preparation.  Because sm_budget is excluded from
        both inner runners' unique_id(), a single full-SM tuning pass populates
        the shared cache entry used by every GC SM split.  Subsequent calls with
        any sm_budget find a cache hit and skip profiling entirely, reducing the
        total tune cost from O(S × inner) to O(inner + S).

        Args:
            x: NVFP4-quantised activations (packed), unswizzled x_sf expected.
            x_sf: Unswizzled scale factors for x (2-D: [num_tokens, hidden/16]).
            router_weight_t: Router weight matrix [hidden, num_experts].
            router_input: BF16 input to the router GEMM [num_tokens, hidden].
        """
        num_tokens = x.shape[0]
        # swizzle_sf is normally applied inside _run_moe_nvfp4_gc_impl; do it
        # here because we are calling the FC1 op directly.
        x_sf_swizzled = swizzle_sf(x_sf, num_tokens, self.hidden_size)

        # FC1: sm_budget=0 → unconstrained (bypasses GC auto-detect, full SM).
        torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
            x,
            self.w3_w1_weight.view(torch.uint8),
            x_sf_swizzled,
            self.w3_w1_weight_scale,
            self.fc31_alpha,
            None,  # alpha_post: not needed for cache warm-up
            self.fc2_input_scale,
            expert_count=self.expert_size_per_partition,
            weight_per_expert=2 * self.intermediate_size_per_partition,
            output_dtype=torch.float4_e2m1fn_x2,
            scaling_vector_size=self.scaling_vector_size,
            sm_budget=0,  # full SM: sm_budget <= 0 → max_active_clusters_hw
        )

        # Router GEMM: sm_budget=0 → unconstrained (stream has no GC, so
        # auto-detect would also return 0; passing 0 explicitly is cleaner).
        m, n = router_input.shape[0], router_weight_t.shape[1]
        router_logits = torch.empty(m, n, dtype=torch.float32, device=router_input.device)
        torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell(
            router_input.contiguous(),
            router_weight_t.t().contiguous(),
            router_logits,
            sm_budget=0,  # full SM
        )

    def _run_moe_nvfp4_gc_impl(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        router_weight_t: torch.Tensor,
        router_input: torch.Tensor,
        fc1_sms: int,
    ) -> torch.Tensor:
        """Execute the GC overlap path with the given FC1 SM partition.

        Args:
            x: NVFP4-quantised activations (already packed).
            x_sf: Swizzled scaling factors for ``x``.
            router_weight_t: Router weight matrix ``[hidden, num_experts]``.
            router_input: Input fed to the router GEMM (same shape as ``x``
                before quantisation, i.e. ``[num_tokens, hidden_size]``).
            fc1_sms: Requested SM count for the FC1 GreenContext partition.
                     Must be a key in ``self._gc_stream_pool``.

        Returns:
            ``final_hidden_states`` tensor of shape ``[num_tokens, hidden_size]``.
        """
        fc1_gc_stream, router_gc_stream = self._gc_stream_pool[fc1_sms]
        num_tokens = x.shape[0]

        x_sf = swizzle_sf(x_sf, num_tokens, self.hidden_size)

        capture_graph = torch.cuda.is_current_stream_capturing()
        fc2_alpha_buffer = DenseGEMMFusedMoE.buffers.get_buffer(
            (num_tokens, self.expert_size_per_partition),
            dtype=torch.float32,
            buffer_name="fc2_alpha",
            reserve_buffer=capture_graph,
        )

        # Fork: record on main stream so both GC streams can wait on it.
        self.event_dict[EventType.Main].record()

        router_sms = max(1, self.num_sms - fc1_sms)
        router_sms = (
            (router_sms + 7) // 8 * 8
        )  # Round up to nearest multiple of 8 for better GC compatibility
        fc1_sms = self.num_sms - router_sms  # Recalculate fc1_sms to reflect any rounding

        with torch.cuda.stream(fc1_gc_stream):
            self.event_dict[EventType.Main].wait()
            fc1_output, fc1_output_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
                x,
                self.w3_w1_weight.view(torch.uint8),
                x_sf,
                self.w3_w1_weight_scale,
                self.fc31_alpha,
                None,  # alpha_post: no post-SwiGLU scaling in this path
                self.fc2_input_scale,
                expert_count=self.expert_size_per_partition,
                weight_per_expert=2 * self.intermediate_size_per_partition,
                output_dtype=torch.float4_e2m1fn_x2,
                scaling_vector_size=self.scaling_vector_size,
            )
            self.fc1_done_event.record()

        with torch.cuda.stream(router_gc_stream):
            self.event_dict[EventType.Main].wait()
            m, n = router_input.shape[0], router_weight_t.shape[1]
            router_logits = torch.empty(m, n, dtype=torch.float32, device=router_input.device)
            _router_input_c = router_input.contiguous()
            _router_weight_c = router_weight_t.t().contiguous()
            torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell(
                _router_input_c,
                _router_weight_c,
                router_logits,
                # sm_budget=router_sms
            )
            token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
            token_selected_experts = token_selected_experts.to(torch.int32)
            # Append fused shared experts (always active, scale=1.0) if any.
            num_routing_experts = router_weight_t.shape[1]
            if self.expert_size_per_partition > num_routing_experts:
                n_shared = self.expert_size_per_partition - num_routing_experts
                M_tok = router_input.shape[0]
                shared_ids = (
                    torch.arange(
                        num_routing_experts,
                        self.expert_size_per_partition,
                        dtype=torch.int32,
                        device=x.device,
                    )
                    .unsqueeze(0)
                    .expand(M_tok, n_shared)
                )
                shared_scales = torch.ones(M_tok, n_shared, dtype=torch.float32, device=x.device)
                token_selected_experts = torch.cat([token_selected_experts, shared_ids], dim=1)
                token_final_scales = torch.cat([token_final_scales, shared_scales], dim=1)
            fc2_alpha = gen_fc2_alpha_fused(
                token_selected_experts,
                token_final_scales,
                self.fc2_alpha,
                output=fc2_alpha_buffer,
            )
            self.event_dict[EventType.MoeFc2Alpha].record()

        # Join: main stream waits for both FC1 output and fc2_alpha.
        self.fc1_done_event.wait()
        self.event_dict[EventType.MoeFc2Alpha].wait()

        # FC2: input k = expert_count * intermediate_size (after SwiGLU halving)
        result = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_fc2_blackwell(
            fc1_output,
            self.w2_weight.view(torch.uint8).reshape(self.hidden_size, -1),
            fc1_output_sf.reshape(-1),
            self.w2_weight_scale,
            fc2_alpha,
            expert_count=self.expert_size_per_partition,
            weight_per_expert=self.intermediate_size_per_partition,
            output_dtype=torch.bfloat16,
            scaling_vector_size=self.scaling_vector_size,
        )
        return result

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run MoE computation with DenseGEMM backend (NVFP4 only).

        Args:
            x: Input hidden states (pre-quantized to NVFP4)
            token_selected_experts: Must be None (routing done internally via router_weight_t)
            token_final_scales: Must be None (routing done internally)
            x_sf: Input scale factors for NVFP4
            enable_alltoall: Whether alltoall communication is enabled.
            **kwargs: Must contain 'router_weight_t'; optionally 'router_input'.

        Returns:
            final_hidden_states tensor.
        """
        assert self.has_nvfp4, (
            f"{self.__class__.__name__} only supports nvfp4 quantization, "
            f"got {self.quant_config.quant_mode}."
        )
        return self.run_moe_nvfp4(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            enable_alltoall=enable_alltoall,
            router_weight_t=kwargs.get("router_weight_t"),
            router_input=kwargs.get("router_input"),
        )

    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        # Currently, the default path is that ConfigurableMoE calls DenseGEMMFusedMoE.run_moe.
        # This forward_chunk method is a reference implementation of the legacy path.
        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        assert token_selected_experts.shape[1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        x, x_sf = self.quantize_input(x)

        if self.use_dp and self.parallel_size > 1:
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens,
            )

        x = self.run_moe(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            enable_alltoall=False,
        )
        return x

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
        assert do_finalize, "DenseGEMMFusedMoE does not support do_finalize=False"

        is_first_call = self.repeat_idx == 0
        is_last_call = self.repeat_idx == self.repeat_count - 1

        outputs = self.forward_chunk(
            x,
            router_logits,
            output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            repeating_info=(is_first_call, is_last_call),
        )
        outputs = self.reducescatter_or_allreduce(
            outputs,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        if self.use_dp and self.parallel_size > 1:
            rank = self.parallel_rank
            outputs = outputs[: all_rank_num_tokens[rank]]
        self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1
        return outputs
