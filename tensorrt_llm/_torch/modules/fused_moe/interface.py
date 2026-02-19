import os
import weakref
from abc import abstractmethod
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Union, final

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...distributed.ops import reducescatter


def _warn_and_return(reason: str) -> Tuple[bool, Optional[str]]:
    """
    Log a warning and return (False, reason) for can_implement() checks.

    This is a common utility function used by all MoE backend implementations
    to provide consistent logging and return values when a configuration
    is not supported.

    Args:
        reason: The reason why the configuration is not supported.

    Returns:
        Tuple[bool, Optional[str]]: Always returns (False, reason)
    """
    logger.warning(reason)
    return False, reason


from ...model_config import ModelConfig
from ...utils import (ActivationType, AuxStreamType, Fp4QuantizedTensor,
                      get_model_extra_attrs, is_gated_activation,
                      is_torch_compiling)
from .routing import BaseMoeRoutingMethod


class MoEWeightLoadingMode(Enum):
    # Gate and up projection are not fused
    VANILLA = 0
    # Gate and up projection are fused
    FUSED_GATE_UP_PROJ = 1
    # Custom W4A8 weights from examples/quantization/quantize_mixed_precision_moe.py
    W4A8_CUSTOM = 2


# The type of alltoall method
class AlltoallMethodType(IntEnum):
    # Not available
    NotEnabled = 0
    # NVLink One-Sided
    NVLinkOneSided = 1
    # NVLink Two-Sided
    NVLinkTwoSided = 2
    # DeepEP intranode or internode: CUDA Graphs are supported, IBGDA is required by internode
    DeepEP = 3
    # DeepEP low latency: CUDA Graphs are supported, IBGDA is required
    DeepEPLowLatency = 4


def extract_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs are not set"

    moe_layers = extra_attrs.get("moe_layers", None)
    assert moe_layers is not None, "No MoE layers registered"
    moe_layer_ref = moe_layers.get(layer_idx)
    assert moe_layer_ref is not None, f"Cannot find MoE layer for layer_idx={layer_idx}"
    moe_layer = moe_layer_ref() if callable(moe_layer_ref) else None
    assert moe_layer is not None, f"MoE layer for layer_idx={layer_idx!r} is no longer alive"

    return moe_layer


@torch.library.custom_op("trtllm::moe_custom_op", mutates_args=())
def moe_custom_op(
    layer_idx: str,
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    is_swizzled: bool,
    router_logits: torch.Tensor,
    do_finalize: bool,
    output_dtype: Optional[torch.dtype],
    all_rank_num_tokens: Optional[List[int]],
    use_dp_padding: Optional[bool],
) -> List[torch.Tensor]:
    moe_layer = extract_extra_attrs(layer_idx)

    hidden_states = x if x_sf is None else Fp4QuantizedTensor(
        x, x_sf, is_swizzled)

    res = moe_layer.forward_impl(
        hidden_states,
        router_logits,
        do_finalize=do_finalize,
        output_dtype=output_dtype,
        all_rank_num_tokens=all_rank_num_tokens,
        use_dp_padding=use_dp_padding,
    )

    if do_finalize:
        return [res]
    else:
        return res


@moe_custom_op.register_fake
def _(
    layer_idx,
    x,
    x_sf,
    is_swizzled,
    router_logits,
    do_finalize,
    output_dtype,
    all_rank_num_tokens,
    use_dp_padding,
):
    moe_layer = extract_extra_attrs(layer_idx)
    hidden_states = x if x_sf is None else Fp4QuantizedTensor(
        x, x_sf, is_swizzled)
    res = moe_layer.forward_fake(
        hidden_states,
        router_logits,
        do_finalize=do_finalize,
        output_dtype=output_dtype,
        all_rank_num_tokens=all_rank_num_tokens,
        use_dp_padding=use_dp_padding,
    )

    if do_finalize:
        return [res]
    else:
        return res


class MoE(nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer interface.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
    """

    @classmethod
    @abstractmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if this MoE backend can implement the given quantization algorithm.

        NOTE: This is a TRANSITIONAL interface. In the future, this method will be moved
        to the MoEBackend interface as part of the backend abstraction layer. During this
        transition period, it remains in the MoE base class to maintain compatibility.

        This method checks both:
        1. Whether the backend supports the specified quantization algorithm
        2. Whether the current platform (SM version) supports the backend and quantization

        Each backend MUST override this method to provide accurate capability information.

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation data type.
            gptoss_style: Whether gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled.

        Returns:
            Tuple[bool, Optional[str]]: (can_implement, skip_reason)
                - can_implement: True if the backend can implement this configuration
                - skip_reason: None if can_implement is True, otherwise a string explaining why not
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement can_implement method")

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
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        activation_type: ActivationType = ActivationType.Swiglu,
        init_load_balancer: bool = True,
    ):
        from ...distributed import AllReduce

        super().__init__()
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_loading_mode = weight_loading_mode
        self.bias = bias
        self.dtype = dtype
        self.reduce_results = reduce_results
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx) if layer_idx is not None else None
        self.activation_type = int(activation_type)
        # Note:
        # - for gated activations, there should be with gate and up projections, so the intermediate size should be expanded by 2.
        # - for non-gated activations, there is only one up projection (no gate projection), so the intermediate size should not be expanded.
        self.is_gated_activation = is_gated_activation(activation_type)
        self.intermediate_size_expand_ratio = 2 if self.is_gated_activation else 1

        self._register_layer(model_config)

        # could be modified later
        self.quant_config = model_config.quant_config

        self.cluster_rank = model_config.mapping.moe_cluster_rank
        self.cluster_size = model_config.mapping.moe_cluster_size
        self.smart_router = True if self.cluster_size > 1 else False

        self.rank = model_config.mapping.rank

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank

        self.moe_backend = model_config.moe_backend
        self.use_dp = model_config.mapping.enable_attention_dp

        # All ranks participate in allreduce regardless of EP/TP combination
        self.mapping = model_config.mapping
        self.parallel_rank = self.mapping.tp_rank
        self.parallel_size = self.mapping.tp_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.all_reduce = None
        # Debug function for eliminating imbalance during performance analysis.
        self.enable_dummy_allreduce = os.environ.get(
            "TRTLLM_ENABLE_DUMMY_ALLREDUCE", "0") == "1"
        if not self.use_dp and self.mapping.tp_size > 1:
            self.all_reduce = AllReduce(
                mapping=self.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=self.dtype)
        elif self.enable_dummy_allreduce:
            from tensorrt_llm.functional import AllReduceStrategy
            self.all_reduce = AllReduce(mapping=self.mapping,
                                        strategy=AllReduceStrategy.NCCL,
                                        dtype=self.dtype)

        # Initialize load balancer related attributes
        if init_load_balancer:
            self._init_load_balancer(model_config, aux_stream_dict)
        else:
            # When init_load_balancer=False, initialize minimal attributes
            # These will be synced from the parent wrapper (e.g., ConfigurableMoE) later
            self.aux_stream_dict = aux_stream_dict
            self.layer_load_balancer = None
            self.repeat_idx = 0
            self.repeat_count = 1
            self.expert_size_per_partition = self.num_experts // self.ep_size
            self.num_slots = self.num_experts
            self.slot_start = self.ep_rank * self.expert_size_per_partition
            self.slot_end = self.slot_start + self.expert_size_per_partition
            self.initial_local_expert_ids = list(
                range(self.slot_start, self.slot_end))
            self.initial_global_assignments = list(range(self.num_experts))
            self.allreduce = None

    def _init_load_balancer(
        self,
        model_config: ModelConfig,
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
    ):
        """Initialize load balancer related attributes."""
        from .moe_load_balancer import get_moe_load_balancer

        # Store aux_stream_dict for load balancer
        self.aux_stream_dict = aux_stream_dict

        # Initialize load balancer attributes
        self.layer_load_balancer = None
        self.repeat_idx = 0
        self.repeat_count = 1

        # Get global load balancer instance
        moe_load_balancer = get_moe_load_balancer()
        moe_load_balancer_config = model_config.moe_load_balancer

        # Calculate initial expert assignments
        init_expert_size_per_partition = (
            moe_load_balancer_config.num_local_slots
            if moe_load_balancer_config else self.num_experts // self.ep_size)

        self.initial_global_assignments = [
            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
            self.num_experts for ep_rank in range(self.ep_size)
            for local_slot_id in range(init_expert_size_per_partition)
        ]

        # Setup load balancer if available
        if moe_load_balancer:
            assert self._supports_load_balancer()
            assert self.use_dp and self.parallel_size > 1, "Load Balancer should be only used with ADP and EP > 1"
            assert moe_load_balancer_config is not None
            top_k = self.routing_method.experts_per_token
            self.expert_size_per_partition = moe_load_balancer_config.num_local_slots

            # Add this layer to the load balancer
            aux_stream = getattr(self, '_get_load_balancer_aux_stream',
                                 lambda: None)()
            self.layer_load_balancer = moe_load_balancer.add_layer(
                self.num_experts,
                top_k,
                self.expert_size_per_partition,
                aux_stream=aux_stream)

            self.repeat_count = self.layer_load_balancer.get_repeat_count()

            # Handle initial global assignments
            loaded_initial_global_assignments = (
                moe_load_balancer_config.get_layer_initial_global_assignments(
                    self.layer_idx))
            self.num_slots = moe_load_balancer_config.num_slots

            if loaded_initial_global_assignments is not None:
                assert isinstance(loaded_initial_global_assignments, list)
                assert len(loaded_initial_global_assignments) == self.num_slots
                assert self.num_slots >= self.num_experts
                assert set(loaded_initial_global_assignments) == set(
                    range(self.num_experts))
                self.initial_global_assignments = loaded_initial_global_assignments

            self.layer_load_balancer.set_initial_weight_assignments(
                self.initial_global_assignments)

            from tensorrt_llm.logger import logger
            logger.info(
                f"MoE load balancer enabled. num_experts = {self.num_experts}, "
                f"num_slots = {self.num_slots}, ep_size = {self.ep_size}")
            logger.info(
                f"initial_global_assignments (layer {self.layer_idx}) = {self.initial_global_assignments}"
            )
        else:
            # Fallback when no load balancer
            assert self.num_experts % self.ep_size == 0
            self.expert_size_per_partition = self.num_experts // self.ep_size
            self.num_slots = self.num_experts

        # Calculate slot boundaries
        self.slot_start = self.ep_rank * self.expert_size_per_partition
        self.slot_end = self.slot_start + self.expert_size_per_partition
        self.initial_local_expert_ids = self.initial_global_assignments[
            self.slot_start:self.slot_end]
        assert len(
            self.initial_local_expert_ids) == self.expert_size_per_partition

        # Setup AllReduce for dynamic routing if needed
        if self._using_dynamic_load_balancer():
            from tensorrt_llm.functional import AllReduceStrategy

            from ...distributed import AllReduce
            self.allreduce = AllReduce(mapping=model_config.mapping,
                                       strategy=AllReduceStrategy.NCCL)
        else:
            self.allreduce = None

    def _add_raw_shared_weights_for_unmap(self,
                                          weight_tensors: List[torch.Tensor]):
        if self._using_dynamic_load_balancer():
            self.layer_load_balancer._add_raw_host_weight_for_unmap(
                weight_tensors)

    def _supports_load_balancer(self) -> bool:
        """Check if this MoE implementation supports load balancer.

        Subclasses can override this to indicate load balancer support.
        """
        return False

    def _using_load_balancer(self) -> bool:
        """Check if this MoE is using load balancer."""
        return self.layer_load_balancer is not None

    def _using_dynamic_load_balancer(self) -> bool:
        """Check if this MoE is using dynamic load balancer."""
        if self.layer_load_balancer:
            return self.layer_load_balancer.is_dynamic_routing()
        return False

    def _get_load_balancer_aux_stream(self) -> Optional[torch.cuda.Stream]:
        """Get auxiliary stream for load balancer from aux_stream_dict.

        Returns the MoeBalancer stream from aux_stream_dict if available.
        """
        if self.aux_stream_dict is not None:
            return self.aux_stream_dict.get(AuxStreamType.MoeBalancer)
        return None

    def _load_balancer_start_wait_gpu_stage(self, is_first_call: bool):
        """Start waiting for GPU stage in load balancer."""
        if self._using_dynamic_load_balancer() and is_first_call:
            self.layer_load_balancer.start_wait_gpu_stage()

    def _load_balancer_done_wait_gpu_stage(self, is_first_call: bool):
        """Mark GPU wait stage as done in load balancer."""
        if self._using_dynamic_load_balancer() and is_first_call:
            self.layer_load_balancer.done_wait_gpu_stage()

    def _load_balancer_update_statistic(self,
                                        token_selected_experts: torch.Tensor,
                                        is_first_call: bool,
                                        is_last_call: bool,
                                        ignore_allreduce: bool = False):
        """
        Update load balancer statistics.

        Args:
            token_selected_experts: The selected experts of all tokens, has shape of [tokenCount * topK]
            is_first_call: Whether this is the first call for the same weights
            is_last_call: Whether this is the last call for the same weights
            ignore_allreduce: Whether to ignore allreduce, if True, only update local statistics, need call _load_balancer_get_local_statistic_tensor to get the local statistic tensor and then do external allgather and then call _load_balancer_update_statistic_with_gathered_statistic to update the global statistics. NVLINKTwoSided supports this.
        """
        if self._using_dynamic_load_balancer():
            if ignore_allreduce:
                self.layer_load_balancer.update_local_statistic(
                    token_selected_experts,
                    is_first_stage=is_first_call,
                    is_last_stage=is_last_call)
            else:
                self.layer_load_balancer.update_statistic_with_local_ids(
                    token_selected_experts,
                    is_first_stage=is_first_call,
                    is_last_stage=is_last_call,
                    allreduce=self.allreduce)

    def _load_balancer_route(self, token_selected_experts: torch.Tensor,
                             use_dp: bool) -> torch.Tensor:
        """Route tokens using load balancer."""
        if self.layer_load_balancer:
            return self.layer_load_balancer.route(token_selected_experts,
                                                  use_dp)
        else:
            return token_selected_experts

    def _load_balancer_start_set_cpu_stage(self, is_last_call: bool):
        """Start CPU stage in load balancer."""
        if self._using_dynamic_load_balancer() and is_last_call:
            self.layer_load_balancer.start_set_cpu_stage()

    def _load_balancer_done_set_cpu_stage(self, is_last_call: bool):
        """Mark CPU stage as done in load balancer."""
        if self._using_dynamic_load_balancer() and is_last_call:
            self.layer_load_balancer.done_set_cpu_stage()

    def _load_balancer_get_local_statistic_tensor(self):
        """Get local statistic tensor from load balancer."""
        if self._using_dynamic_load_balancer():
            return self.layer_load_balancer.get_local_statistic_tensor()
        return None

    def _load_balancer_update_statistic_with_gathered_statistic(
            self, gathered_statistic):
        """Update load balancer with gathered statistics."""
        if self._using_dynamic_load_balancer():
            self.layer_load_balancer.update_statistic_with_gathered_statistic(
                gathered_statistic)

    def register_parameter_weight_slot_fn(self, weight_name: str,
                                          local_slot_id: int):
        """Register parameter weight slot function for load balancer."""
        if not self._using_dynamic_load_balancer():
            return

        assert hasattr(
            self, weight_name), f"MoE doesn't have weight attr: {weight_name}"
        weight_tensor = getattr(self, weight_name).data[local_slot_id]
        self.layer_load_balancer.register_weight_slot(local_slot_id,
                                                      weight_name,
                                                      weight_tensor)

    def register_to_fix_weight_fn(self, weight_name: str):
        """Register weight fixing function for load balancer."""
        if not self._using_dynamic_load_balancer():
            return

        assert hasattr(
            self, weight_name), f"MoE doesn't have weight attr: {weight_name}"
        param = getattr(self, weight_name)
        weight_tensor = param.detach()
        assert isinstance(
            weight_tensor,
            torch.Tensor), f'weight {weight_name} should be a tensor'
        assert weight_tensor.is_contiguous(), (
            f'weight {weight_name} should be contiguous, '
            f'shape={weight_tensor.shape}, strides={weight_tensor.stride()}')
        assert weight_tensor.numel() * weight_tensor.element_size(
        ) == weight_tensor.untyped_storage().size(), (
            f'weight {weight_name} shape={weight_tensor.shape} '
            f'storage_size = {weight_tensor.untyped_storage().size()}, '
            f'numel={weight_tensor.numel()}, eltsize={weight_tensor.element_size()}, '
            f'dtype={weight_tensor.dtype}')
        self.layer_load_balancer.make_tensor_host_accessible(weight_tensor)
        param.data = weight_tensor

    def register_all_parameter_slot_and_to_fix_weight_fns(
            self, weight_and_tensor_dict: Dict[str, torch.Tensor]):
        """Register all parameter slot and weight fixing functions for load balancer."""
        if not self._using_dynamic_load_balancer():
            return

        # Register weight functions for each local slot
        for local_slot_id, expert_id in enumerate(
                self.initial_local_expert_ids):
            for weight_name in weight_and_tensor_dict:
                self.layer_load_balancer.add_register_weight_fn(
                    self.register_parameter_weight_slot_fn,
                    (weight_name, local_slot_id))

        # Register weight migration functions
        for weight_name in weight_and_tensor_dict:
            self.layer_load_balancer.add_to_migrate_weight_fn(
                self.register_to_fix_weight_fn, (weight_name, ))

        # Setup host tensor sharing
        local_shared_load_expert_ids = self.layer_load_balancer.get_load_expert_ids(
        )
        for expert_id in range(self.num_experts):
            for weight_name, weight_tensor in weight_and_tensor_dict.items():
                if expert_id in local_shared_load_expert_ids:
                    local_slot_id = local_shared_load_expert_ids.index(
                        expert_id)
                    self.layer_load_balancer.host_tensor_sharer.share_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor[local_slot_id])
                else:
                    self.layer_load_balancer.host_tensor_sharer.pre_register_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor.dtype,
                        weight_tensor[0].shape)

    def _register_layer(self, model_config: ModelConfig):
        self.register_to_config = False
        if model_config is not None and self.layer_idx_str is not None:
            if "moe_layers" not in model_config.extra_attrs:
                model_config.extra_attrs["moe_layers"] = {}
            assert self.layer_idx_str not in model_config.extra_attrs["moe_layers"], \
                f"Duplicate MoE layer for layer_idx={self.layer_idx_str}"
            model_config.extra_attrs["moe_layers"][
                self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

    @abstractmethod
    def create_weights(self):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        """
        Args:
            weights: List of weight dictionaries to load.
            allow_partial_loading: Whether to enable partial loading for module parameters.
                When True, weights are loaded without applying quantization transformations.
                When False (default), weights are loaded and quantized together.
        """
        raise NotImplementedError

    def post_load_weights(self):
        pass

    def process_weights_after_loading(self):
        """
        Apply quantization processing to loaded weights.

        When allow_partial_loading=True is used in load_weights(), this method
        must be called separately to complete the loading setup.
        """
        if hasattr(self.quant_method, 'process_weights_after_loading'):
            self.quant_method.process_weights_after_loading(self)

    def pre_reload_weights(self):
        """
        Prepare tensors for weight reloading by reverting them to their original creation shape.
        """
        assert hasattr(
            self.quant_method, 'pre_reload_weights'
        ), "pre_reload_weights is not supported for this quant method"
        if self._using_load_balancer():
            raise NotImplementedError(
                "Weight reloading is not compatible with Expert Parallel Load Balancer (EPLB). "
            )
        self.quant_method.pre_reload_weights(self)

    @abstractmethod
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict]:
        """
        Quantize input tensor - unified interface for all MoE backends

        NOTE: This is a temporary interface. In the future, this method should be moved
        to the MoEBackend interface as part of the backend abstraction layer.

        This method handles quantization of input tensors before MoE computation.
        All MoE backend implementations must override this method to implement their
        specific quantization logic.

        Args:
            x: Input tensor [num_tokens, hidden_size] or Fp4QuantizedTensor
            **kwargs: Backend-specific arguments (e.g., token_selected_experts, workspace, etc.)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]] or Dict:
                (quantized_x, scaling_factors)
                where scaling_factors should be reshaped to 2D if applicable

        Examples:
            Simple backends (Cutlass, WideEP, TRTLLMGen):
                return x_quantized, x_sf  # x_sf is 2D or None
        """
        raise NotImplementedError

    @abstractmethod
    def run_moe(
        self,
        # ========== Common parameters (all backends use) ==========
        x: torch.Tensor,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        # ========== Backend-specific parameters (via kwargs) ==========
        **kwargs
    ) -> torch.Tensor:
        """
        Unified MoE computation interface

        NOTE: This is a TEMPORARY interface. In the future, this method should be moved
        to the MoEBackend interface as part of the backend abstraction layer.

        This method performs the core MoE computation. Different backends will implement
        their specific computation logic while following this unified interface.

        Common parameters (all backends use):
            x: Input activations [num_tokens, hidden_size]
            token_selected_experts: Expert IDs [num_tokens, top_k] (used by DeepGemm/TRTLLMGen).
                                    If EPLB is enabled, this represents expert slots [num_tokens, top_k].
            token_final_scales: Routing weights [num_tokens, top_k]
            x_sf: Input scale factor (for quantization, if applicable)

        Backend-specific parameters (passed via kwargs, obtained from _get_backend_kwargs()):
            TODO: This is not finalized, will be updated later.

        Returns:
            torch.Tensor: MoE computation result [num_tokens, hidden_size]
        """
        raise NotImplementedError

    @abstractmethod
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
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

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
        is_nvfp4_input = isinstance(x, Fp4QuantizedTensor)
        assert do_finalize, "Default forward_fake does not support do_finalize=False"
        data_type = output_dtype if is_nvfp4_input else x.dtype
        num_tokens = all_rank_num_tokens[
            self.mapping.tp_rank] if all_rank_num_tokens else x.shape[0]
        hidden_size = x.shape[1] * (2 if is_nvfp4_input else 1)
        return x.new_empty((num_tokens, hidden_size), dtype=data_type)

    # Sub class is not allowed to override forward.
    # This is universal interface for all MoE backends
    @final
    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.register_to_config and is_torch_compiling():
            hidden_states = x.fp4_tensor if isinstance(
                x, Fp4QuantizedTensor) else x
            x_sf = x.scaling_factor if isinstance(x,
                                                  Fp4QuantizedTensor) else None
            is_swizzled = x.is_sf_swizzled if isinstance(
                x, Fp4QuantizedTensor) else False

            res = moe_custom_op(
                self.layer_idx_str,
                hidden_states,
                x_sf,
                is_swizzled,
                router_logits,
                do_finalize,
                output_dtype,
                all_rank_num_tokens,
                use_dp_padding,
            )
            if do_finalize:
                return res[0]
            else:
                return res
        else:
            return self.forward_impl(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )

    @property
    def has_any_quant(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True)

    # The following three properties are common enough to warrant inclusion in the interface.
    @property
    def has_fp8_qdq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq(
        )

    @property
    def has_deepseek_fp8_block_scales(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_block_scales(
        )

    @property
    def has_nvfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4(
        )

    @property
    def has_w4a8_nvfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8(
        )

    @property
    def has_w4a8_mxfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8(
        )

    @property
    def has_w4a8_mxfp4_mxfp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8(
        )

    @property
    def has_w4a16_mxfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a16_mxfp4(
        )

    @property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return False

    @property
    def expand_intermediate_size_per_partition(self):
        return self.intermediate_size_per_partition * self.intermediate_size_expand_ratio

    def supports_moe_output_in_alltoall_workspace(self):
        """ Supports moe_output in alltoall workspace
        """
        return False

    def reducescatter_or_allreduce(
        self,
        inputs,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ):
        """
        Common helper for TP and EP in subclasses of the MoE module.
        """
        outputs = inputs
        if self.parallel_size > 1 and not self.enable_alltoall:
            if self.use_dp:
                outputs = reducescatter(
                    inputs,
                    self.mapping,
                    dim=0,
                    sizes=None if use_dp_padding else all_rank_num_tokens)
            elif self.reduce_results:
                outputs = self.all_reduce(inputs)
        return outputs

    def dummy_allreduce(self):
        assert self.enable_dummy_allreduce and self.all_reduce is not None, "Dummy allreduce is not enabled"
        """
        Debug function for eliminating imbalance during performance analysis.
        Creates a small dummy tensor and performs allreduce to synchronize processes
        and eliminate timing imbalances for more accurate profiling measurements.
        """
        dummy_tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
        dummy_tensor = self.all_reduce(dummy_tensor)
        return dummy_tensor
