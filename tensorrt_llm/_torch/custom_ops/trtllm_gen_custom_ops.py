from dataclasses import dataclass, replace
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.modules.fused_moe.routing import (
    ROUTING_METHOD_TYPE_TO_CLASS, RoutingMethodType)
from tensorrt_llm._torch.utils import (Fp4QuantizedTensor, fp4_utils,
                                       get_last_power_of_2_num_tokens_buckets,
                                       last_positive_power_of_2,
                                       next_positive_power_of_2)

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)


def prepare_dummy_topk_and_hook(
    topk_weights: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    routing_logits: Optional[torch.Tensor],
    routing_method_type: int,
    base_tuning_config: TuningConfig,
    top_k: int,
    num_experts: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    routed_scaling_factor: Optional[float],
    hidden_states_index: int = 2,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, TuningConfig]:
    """
    Prepare dummy topk tensors and input pre-hook for AutoTuner profiling.

    This function handles attention DP scenarios where topk_weights/topk_ids are pre-computed.
    It creates dummy tensors to prevent the routing kernel from being called during profiling,
    and provides a hook to dynamically adjust tensor shapes when AutoTuner tries different
    token counts.

    NOTE: whether or not MoE accepts routing_logits or topk_id/topk_weights, ALWAYS start with dummy
    routing_logits then calculate the dummy topk_id/topk_weights according to model routing_method.
    This has found to more closely mirror the actual expert distribution and thus result in better
    e2e performance.

    Args:
        topk_weights: Pre-computed topk weights (None for normal routing scenario)
        topk_ids: Pre-computed topk ids (None for normal routing scenario)
        hidden_states: Hidden states tensor (used for shape and device)
        routing_logits: Routing logits (None if not provided)
        base_tuning_config: Base tuning config to add hook to
        top_k: Number of top experts to select
        num_experts: Total number of experts
        hidden_states_index: Index of hidden_states in input_tensors list (default: 2)

    Returns:
        Tuple of (routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook)
    """

    # NOTE: This prevents auto-tuning related code from being executed in actual runs
    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        return routing_logits, topk_weights, topk_ids, base_tuning_config

    if routing_logits is None:
        routing_logits_for_tuner = torch.randn(hidden_states.shape[0],
                                               num_experts,
                                               dtype=torch.bfloat16,
                                               device=hidden_states.device)
    else:
        routing_logits_for_tuner = routing_logits

    # Determine if we need dummy topk tensors (attention DP scenario)
    need_dummy_topk = (topk_weights is not None or topk_ids is not None)

    # Get routing method
    routing_cls_kwargs = {}
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        routing_cls_kwargs.update({
            'n_group':
            n_group,
            'topk_group':
            topk_group,
            'routed_scaling_factor':
            routed_scaling_factor,
            'is_fused':
            False,  # fuse_routing_kernel
            'callable_e_score_correction_bias':
            lambda: torch.randn(
                num_experts, dtype=torch.bfloat16, device=hidden_states.device)
        })
    routing_method = ROUTING_METHOD_TYPE_TO_CLASS[routing_method_type](
        top_k=top_k, **routing_cls_kwargs)

    # Create dummy topk tensors for attention DP scenario
    if need_dummy_topk:
        # Attention DP: topk is pre-computed, no routing needed
        topk_ids_for_tuner, topk_weights_for_tuner = routing_method.apply(
            routing_logits_for_tuner)
        topk_weights_for_tuner = topk_weights_for_tuner.to(torch.bfloat16)
        # Don't pass routing_logits to avoid C++ warning about all three being provided
        routing_logits_for_tuner = None
    else:
        # Normal routing: need routing_logits, topk will be computed by kernel
        topk_weights_for_tuner = topk_weights
        topk_ids_for_tuner = topk_ids
        assert topk_weights_for_tuner is None
        assert topk_ids_for_tuner is None

    # Define hook to recreate dummy tensors when shape changes during profiling
    def recreate_dummy_topk_if_needed(
            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Recreate dummy topk tensors if token count changed during profiling."""
        current_num_tokens = inputs[hidden_states_index].shape[0]
        # Recreate routing logits if token count changed
        if inputs[0] is None or inputs[0].shape[0] != current_num_tokens:
            routing_logits_for_tuner = torch.randn(
                current_num_tokens,
                num_experts,
                dtype=torch.bfloat16,
                device=inputs[hidden_states_index].device)

        # Only recreate if we originally created dummies
        if need_dummy_topk:
            # Check if shape changed
            if inputs[-1] is not None and inputs[-1].shape[
                    0] != current_num_tokens:
                # Recreate with new shape
                topk_ids_for_tuner, topk_weights_for_tuner = routing_method.apply(
                    routing_logits_for_tuner)
                inputs[-1] = topk_ids_for_tuner
                inputs[-2] = topk_weights_for_tuner.to(torch.bfloat16)
            # Note: routing_logits is None in attention DP, no need to adjust
            assert inputs[0] is None

        return inputs

    # Add inputs_pre_hook to handle shape changes during profiling
    tuning_config_with_hook = replace(
        base_tuning_config, inputs_pre_hook=recreate_dummy_topk_if_needed)

    return routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook


def calculate_tile_tokens_dim(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    imbalance_factor: float = 1.0,
    max_tile_tokens_dim: int = 128,
) -> int:
    # We use the num_tokens after round mapping to generate tile tokens dim.
    # This is to keep the tuning tokens aligned with the tokens used for inference.
    num_tokens = min(next_positive_power_of_2(num_tokens), 4096)

    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = num_tokens * top_k // num_experts
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)

    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # For 128-256 tokens per expert, use 192 tokens per CTA tile.
    if num_tokens_per_expert > 128 and num_tokens_per_expert < 256:
        tile_tokens_dim = 192
    # Cap to 8-max_tile_tokens_dim tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), max_tile_tokens_dim)

    return tile_tokens_dim


@dataclass(frozen=True)
class FP4BlockScaleMoEInputs:

    routing_logits: Optional[torch.Tensor]
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: torch.Tensor
    gemm1_alpha: torch.Tensor
    gemm1_beta: torch.Tensor
    gemm1_clamp_limit: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: torch.Tensor
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class FP4BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int, do_finalize: bool, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_method_type = routing_method_type
        self.do_finalize = do_finalize
        self.act_type = act_type

        self.tuning_config = FP4BlockScaleMoERunner.get_tuning_config(
            self.num_experts // self.local_num_experts)

    # The unique_id is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing type does not matter
    def unique_id(self):
        return (self.top_k, self.intermediate_size, self.local_num_experts,
                self.act_type)

    def get_runner(self):
        instance_key = (self.act_type, )
        if instance_key not in FP4BlockScaleMoERunner.runner_dict:
            FP4BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP4BlockScaleMoERunner(
                    self.act_type)
        return FP4BlockScaleMoERunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: List[int] = [-1, -1],
    ) -> torch.Tensor:
        assert isinstance(tactic, list)

        args = FP4BlockScaleMoEInputs(*inputs)
        kernel_runner = self.get_runner()

        return kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.hidden_states_scale, args.gemm1_weights,
            args.gemm1_weights_scale, args.gemm1_bias, args.gemm1_alpha,
            args.gemm1_beta, args.gemm1_clamp_limit, args.gemm2_weights,
            args.gemm2_weights_scale, args.gemm2_bias,
            args.output1_scale_scalar, args.output1_scale_gate_scalar,
            args.output2_scale_scalar, self.num_experts, self.top_k,
            self.n_group, self.topk_group, self.intermediate_size,
            self.local_expert_offset, self.local_num_experts,
            self.routed_scaling_factor, self.routing_method_type,
            self.do_finalize, tactic, args.topk_weights, args.topk_ids)

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile,
                          **kwargs) -> List[List[int]]:

        args = FP4BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]

        # The hidden size is actually 2 * hidden_size because we pack 2x e2m1
        # into 1 byte.
        hidden_size = args.hidden_states.shape[1] * 2

        kernel_runner = self.get_runner()

        tactics = kernel_runner.get_valid_configs(
            self.top_k,
            hidden_size,
            self.intermediate_size,
            self.local_num_experts,
            num_tokens,
        )

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls,
                                 ep_size: int) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        def round_rule(x: int) -> int:
            value = last_positive_power_of_2(x) // ep_size
            return min(max(1, value), MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX,
                                   TUNED_DIM,
                                   m_values,
                                   map_to_tuning_buckets=round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_to_num_tokens(shapes: Tuple[torch.Size]) -> int:
            HIDDEN_STATES_IDX = 2
            NUM_TOKENS_DIM = 0

            num_tokens = shapes[HIDDEN_STATES_IDX][NUM_TOKENS_DIM]

            return num_tokens

        def _constrain_fp4_linear_layout(shapes: Tuple[torch.Size]) -> int:
            HIDDEN_STATES_IDX = 2
            NUM_TOKENS_DIM = 0
            HIDDEN_SIZE_DIM = 1

            num_tokens = shapes[HIDDEN_STATES_IDX][NUM_TOKENS_DIM]

            # The hidden size is actually 2 * hidden_size because we pack 2x e2m1
            hidden_size = shapes[HIDDEN_STATES_IDX][HIDDEN_SIZE_DIM] * 2

            sf_linear_size = num_tokens * (hidden_size // 16)

            return sf_linear_size

        HIDDEN_STATES_SCALE_IDX = 3
        CONSTRAINED_HS_SCALE_DIM = 0

        constraint_hidden_states_scale = ConstraintSpec(
            HIDDEN_STATES_SCALE_IDX, CONSTRAINED_HS_SCALE_DIM,
            _constrain_fp4_linear_layout)

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_RL_DIM = 0
        TOPK_WEIGHTS_IDX = 16
        TOPK_IDS_IDX = 17

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_to_num_tokens)
        constraint_topk_weights = ConstraintSpec(TOPK_WEIGHTS_IDX,
                                                 CONSTRAINED_RL_DIM,
                                                 _constrain_to_num_tokens)
        constraint_topk_ids = ConstraintSpec(TOPK_IDS_IDX, CONSTRAINED_RL_DIM,
                                             _constrain_to_num_tokens)

        constraint_specs_tuple = (
            constraint_hidden_states_scale,
            constraint_routing_logits,
            constraint_topk_weights,
            constraint_topk_ids,
        )

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, ep_size: int) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(ep_size)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp4_block_scale_moe_runner", mutates_args=())
def fp4_block_scale_moe_runner(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: torch.Tensor,
        gemm1_alpha: torch.Tensor,
        gemm1_beta: torch.Tensor,
        gemm1_clamp_limit: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = 0,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None) -> List[torch.Tensor]:

    tuner = AutoTuner.get()
    kernel_runner = FP4BlockScaleMoERunner(
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        act_type,
    )

    # Prepare dummy topk tensors and hook for AutoTuner profiling
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook = \
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
        )

    # Build input_tensors_for_tuner
    input_tensors_for_tuner = [
        routing_logits_for_tuner,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        topk_weights_for_tuner,  # Dummy if need_dummy_topk, else actual value
        topk_ids_for_tuner,  # Dummy if need_dummy_topk, else actual value
    ]

    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::fp4_block_scale_moe_runner",
        [kernel_runner],
        tuning_config_with_hook,
        input_tensors_for_tuner,
    )

    input_tensors = input_tensors_for_tuner
    input_tensors[
        0] = routing_logits  # replace dummy routing logits with actual routing logits
    input_tensors[-2] = topk_weights  # replace dummy topk_weights with actual
    input_tensors[-1] = topk_ids  # replace dummy topk_ids with actual
    return kernel_runner(input_tensors,
                         tactic=[-1, -1] if best_tactic == -1 else best_tactic)


def fp4_block_scale_fake_output_without_finalize(
    hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
    num_experts: int,
    top_k: int,
    routing_bias: Optional[torch.Tensor],
):
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] * (2 if isinstance(
        hidden_states, Fp4QuantizedTensor) else 1)

    # Use the max possible tile tokens dimension
    tile_tokens_dim = 128

    expanded_row_count = num_tokens * top_k
    max_padding_required = (tile_tokens_dim - 1) * num_experts
    max_num_padded_tokens = fp4_utils.pad_up(
        expanded_row_count + max_padding_required, tile_tokens_dim)
    wt_dtype = routing_bias.dtype if routing_bias is not None else torch.bfloat16
    return [
        hidden_states.new_empty((max_num_padded_tokens, hidden_size),
                                dtype=torch.bfloat16),
        hidden_states.new_empty((num_tokens, top_k), dtype=wt_dtype),
        hidden_states.new_empty((num_tokens, top_k), dtype=torch.int32)
    ]


@fp4_block_scale_moe_runner.register_fake
def _(routing_logits,
      routing_bias,
      hidden_states,
      hidden_states_scale,
      gemm1_weights,
      gemm1_weights_scale,
      gemm1_bias,
      gemm1_alpha,
      gemm1_beta,
      gemm1_clamp_limit,
      gemm2_weights,
      gemm2_weights_scale,
      gemm2_bias,
      output1_scale_scalar,
      output1_scale_gate_scalar,
      output2_scale_scalar,
      num_experts,
      top_k,
      n_group,
      topk_group,
      intermediate_size,
      local_expert_offset,
      local_num_experts,
      routed_scaling_factor,
      routing_method_type,
      do_finalize,
      act_type,
      topk_weights: Optional[torch.Tensor] = None,
      topk_ids: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    if do_finalize:
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1] * 2
        return [
            hidden_states.new_empty((num_tokens, hidden_size),
                                    dtype=torch.bfloat16)
        ]

    return fp4_block_scale_fake_output_without_finalize(
        hidden_states,
        num_experts,
        top_k,
        routing_bias,
    )


@dataclass(frozen=True)
class FP8BlockScaleMoEInputs:

    routing_logits: Optional[torch.Tensor]
    routing_bias: torch.Tensor
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class FP8BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
    ):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_method_type = routing_method_type
        self.act_type = 0
        self.tuning_config = FP8BlockScaleMoERunner.get_tuning_config(
            self.num_experts // self.local_num_experts)

    # The unique_id is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def unique_id(self):
        return (self.top_k, self.intermediate_size, self.local_num_experts,
                self.act_type)

    def get_runner(self):
        instance_key = ()
        if instance_key not in FP8BlockScaleMoERunner.runner_dict:
            FP8BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8BlockScaleMoERunner()
        return FP8BlockScaleMoERunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: List[int] = [-1, -1],
    ) -> torch.Tensor:

        args = FP8BlockScaleMoEInputs(*inputs)

        kernel_runner = self.get_runner()

        return kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.hidden_states_scale, args.gemm1_weights,
            args.gemm1_weights_scale, args.gemm2_weights,
            args.gemm2_weights_scale, self.num_experts, self.top_k,
            self.n_group, self.topk_group, self.intermediate_size,
            self.local_expert_offset, self.local_num_experts,
            self.routed_scaling_factor, self.routing_method_type, tactic,
            args.topk_weights, args.topk_ids)

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile,
                          **kwargs) -> List[List[int]]:

        args = FP8BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        kernel_runner = self.get_runner()

        tactics = kernel_runner.get_valid_configs(
            self.top_k,
            hidden_size,
            self.intermediate_size,
            self.local_num_experts,
            num_tokens,
        )

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls,
                                 ep_size: int) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        def round_rule(x: int) -> int:
            value = last_positive_power_of_2(x) // ep_size
            return min(max(1, value), MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX,
                                   TUNED_DIM,
                                   m_values,
                                   map_to_tuning_buckets=round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_to_num_tokens(shapes: Tuple[torch.Size]) -> int:
            num_tokens = shapes[2][0]

            return num_tokens

        HS_SCALE_IDX = 3
        CONSTRAINED_HS_SCALE_DIM = 1

        constraint_hidden_states_scale = ConstraintSpec(
            HS_SCALE_IDX, CONSTRAINED_HS_SCALE_DIM, _constrain_to_num_tokens)

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_RL_DIM = 0
        TOPK_WEIGHTS_IDX = 8
        TOPK_IDS_IDX = 9

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_to_num_tokens)
        constraint_topk_weights = ConstraintSpec(TOPK_WEIGHTS_IDX,
                                                 CONSTRAINED_RL_DIM,
                                                 _constrain_to_num_tokens)
        constraint_topk_ids = ConstraintSpec(TOPK_IDS_IDX, CONSTRAINED_RL_DIM,
                                             _constrain_to_num_tokens)

        constraint_specs_tuple = (
            constraint_hidden_states_scale,
            constraint_routing_logits,
            constraint_topk_weights,
            constraint_topk_ids,
        )

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, ep_size: int) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(ep_size)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp8_block_scale_moe_runner", mutates_args=())
def fp8_block_scale_moe_runner(routing_logits: Optional[torch.Tensor],
                               routing_bias: torch.Tensor,
                               hidden_states: torch.Tensor,
                               hidden_states_scale: torch.Tensor,
                               gemm1_weights: torch.Tensor,
                               gemm1_weights_scale: torch.Tensor,
                               gemm2_weights: torch.Tensor,
                               gemm2_weights_scale: torch.Tensor,
                               num_experts: int,
                               top_k: int,
                               n_group: Optional[int],
                               topk_group: Optional[int],
                               intermediate_size: int,
                               local_expert_offset: int,
                               local_num_experts: int,
                               routed_scaling_factor: Optional[float],
                               routing_method_type: int,
                               topk_weights: Optional[torch.Tensor] = None,
                               topk_ids: Optional[torch.Tensor] = None,
                               act_type: int = 0) -> torch.Tensor:

    tuner = AutoTuner.get()
    kernel_runner = FP8BlockScaleMoERunner(
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        act_type,
    )

    # Prepare dummy topk tensors and hook for AutoTuner profiling
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook = \
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
        )

    input_tensors_for_tuner = [
        routing_logits_for_tuner,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        topk_weights_for_tuner,
        topk_ids_for_tuner,
    ]

    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::fp8_block_scale_moe_runner",
        [kernel_runner],
        tuning_config_with_hook,
        input_tensors_for_tuner,
    )

    input_tensors = input_tensors_for_tuner
    input_tensors[
        0] = routing_logits  # replace dummy routing logits with actual routing logits
    input_tensors[-2] = topk_weights  # replace dummy topk_weights with actual
    input_tensors[-1] = topk_ids  # replace dummy topk_ids with actual
    return kernel_runner(input_tensors,
                         tactic=[-1, -1] if best_tactic == -1 else best_tactic)


@fp8_block_scale_moe_runner.register_fake
def _(routing_logits: torch.Tensor,
      routing_bias: torch.Tensor,
      hidden_states: torch.Tensor,
      hidden_states_scale: torch.Tensor,
      gemm1_weights: torch.Tensor,
      gemm1_weights_scale: torch.Tensor,
      gemm2_weights: torch.Tensor,
      gemm2_weights_scale: torch.Tensor,
      num_experts: int,
      top_k: int,
      n_group: Optional[int],
      topk_group: Optional[int],
      intermediate_size: int,
      local_expert_offset: int,
      local_num_experts: int,
      routed_scaling_factor: Optional[float],
      routing_method_type: int,
      topk_weights: Optional[torch.Tensor] = None,
      topk_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] * 2

    return hidden_states.new_empty((num_tokens, hidden_size),
                                   dtype=torch.bfloat16)


@dataclass(frozen=True)
class MxE4m3MxE2m1BlockScaleMoEInputs:
    routing_logits: Optional[torch.Tensor]
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm1_clamp_limit: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class MxE4m3MxE2m1BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 valid_hidden_size: int, valid_intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.valid_hidden_size = valid_hidden_size
        self.valid_intermediate_size = valid_intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_method_type = routing_method_type
        self.act_type = act_type

        self.tuning_config = MxE4m3MxE2m1BlockScaleMoERunner.get_tuning_config(
            self.num_experts // self.local_num_experts)

    # The unique_id is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def unique_id(self):
        return (
            self.top_k,
            self.intermediate_size,
            self.valid_hidden_size,
            self.valid_intermediate_size,
            self.local_num_experts,
            self.act_type,
        )

    def get_runner(self):
        instance_key = (self.act_type, True)
        if instance_key not in MxE4m3MxE2m1BlockScaleMoERunner.runner_dict:
            MxE4m3MxE2m1BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(
                    self.act_type, True)
        return MxE4m3MxE2m1BlockScaleMoERunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: List[int] = [-1, -1],
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert isinstance(tactic, list)

        args = MxE4m3MxE2m1BlockScaleMoEInputs(*inputs)

        kernel_runner = self.get_runner()
        return kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.hidden_states_scale, args.gemm1_weights,
            args.gemm1_weights_scale, args.gemm1_bias, args.gemm1_alpha,
            args.gemm1_beta, args.gemm1_clamp_limit, args.gemm2_weights,
            args.gemm2_weights_scale, args.gemm2_bias, None, None, None,
            self.num_experts, self.top_k, self.n_group, self.topk_group,
            self.intermediate_size, self.valid_hidden_size,
            self.valid_intermediate_size, self.local_expert_offset,
            self.local_num_experts, self.routed_scaling_factor,
            self.routing_method_type, tactic, args.topk_weights, args.topk_ids,
            output)

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile,
                          **kwargs) -> List[List[int]]:

        args = MxE4m3MxE2m1BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        kernel_runner = self.get_runner()

        tactics = kernel_runner.get_valid_configs(
            self.top_k,
            hidden_size,
            self.intermediate_size,
            self.local_num_experts,
            num_tokens,
            self.valid_hidden_size or hidden_size,
            self.valid_intermediate_size or self.intermediate_size,
        )

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls,
                                 ep_size: int) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        def round_rule(x: int) -> int:
            value = last_positive_power_of_2(x) // ep_size
            return min(max(1, value), MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX,
                                   TUNED_DIM,
                                   m_values,
                                   map_to_tuning_buckets=round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_hidden_states_scale(shapes: Tuple[torch.Size]) -> int:
            # hidden_states dim 0 and dim 1
            num_tokens = shapes[2][0]
            hidden_size = shapes[2][1]

            SF_BLOCK_SIZE = 32

            # Linear fp4 sf layout is just rows * columns
            size = num_tokens * (hidden_size // SF_BLOCK_SIZE)

            return size

        def _constrain_routing_logits(shapes: Tuple[torch.Size]) -> int:
            # hidden_states dim 0 and dim 1
            num_tokens = shapes[2][0]

            return num_tokens

        HIDDEN_STATE_SCALE_IDX = 3
        CONSTRAINED_HS_DIM = 0

        constraint_hidden_states_scale = ConstraintSpec(
            HIDDEN_STATE_SCALE_IDX, CONSTRAINED_HS_DIM,
            _constrain_hidden_states_scale)

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_RL_DIM = 0
        TOPK_WEIGHTS_IDX = 13
        TOPK_IDS_IDX = 14

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_routing_logits)
        constraint_topk_weights = ConstraintSpec(TOPK_WEIGHTS_IDX,
                                                 CONSTRAINED_RL_DIM,
                                                 _constrain_routing_logits)
        constraint_topk_ids = ConstraintSpec(TOPK_IDS_IDX, CONSTRAINED_RL_DIM,
                                             _constrain_routing_logits)

        constraint_specs_tuple = (constraint_hidden_states_scale,
                                  constraint_routing_logits,
                                  constraint_topk_weights, constraint_topk_ids)

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, ep_size: int) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(ep_size)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::mxe4m3_mxe2m1_block_scale_moe_runner",
                         mutates_args=())
def mxe4m3_mxe2m1_block_scale_moe_runner(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        valid_hidden_size: Optional[int],
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None) -> torch.Tensor:

    tuner = AutoTuner.get()
    kernel_runner = MxE4m3MxE2m1BlockScaleMoERunner(
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        valid_hidden_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        act_type,
    )

    # Prepare dummy topk tensors and hook for AutoTuner profiling
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook = \
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
        )

    input_tensors_for_tuner = [
        routing_logits_for_tuner,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        topk_weights_for_tuner,
        topk_ids_for_tuner,
    ]

    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::mxe4m3_mxe2m1_block_scale_moe_runner",
        [kernel_runner],
        tuning_config_with_hook,
        input_tensors_for_tuner,
    )

    input_tensors = input_tensors_for_tuner
    input_tensors[
        0] = routing_logits  # replace dummy routing logits with actual routing logits
    input_tensors[-2] = topk_weights  # replace dummy topk_weights with actual
    input_tensors[-1] = topk_ids  # replace dummy topk_ids with actual
    result = kernel_runner(
        input_tensors,
        tactic=[-1, -1] if best_tactic == -1 else best_tactic,
        output=output)
    # When output is provided, the result is written in-place to output.
    # Return empty tensor to avoid aliasing constraint violation in PyTorch 2.9.1+
    # (custom op output cannot be the same tensor as input).
    # Callers should use output directly when they provide it.
    if output is not None:
        return torch.empty(0, device=result.device, dtype=result.dtype)
    return result


@dataclass(frozen=True)
class E4m3MxE2m1BlockScaleMoEInputs:
    routing_logits: Optional[torch.Tensor]
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm1_clamp_limit: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: Optional[torch.Tensor]
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class E4m3MxE2m1BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 valid_hidden_size: int, valid_intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.valid_hidden_size = valid_hidden_size
        self.valid_intermediate_size = valid_intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_method_type = routing_method_type
        self.act_type = act_type

        self.tuning_config = E4m3MxE2m1BlockScaleMoERunner.get_tuning_config(
            self.num_experts // self.local_num_experts)

    # The unique_id is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def unique_id(self):
        return (
            self.top_k,
            self.intermediate_size,
            self.valid_hidden_size,
            self.valid_intermediate_size,
            self.local_num_experts,
            self.act_type,
        )

    def get_runner(self):
        instance_key = (self.act_type, False)
        if instance_key not in E4m3MxE2m1BlockScaleMoERunner.runner_dict:
            E4m3MxE2m1BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(
                    self.act_type, False)
        return E4m3MxE2m1BlockScaleMoERunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: List[int] = [-1, -1],
    ) -> torch.Tensor:
        assert isinstance(tactic, list)

        args = E4m3MxE2m1BlockScaleMoEInputs(*inputs)

        kernel_runner = self.get_runner()
        return kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states, None,
            args.gemm1_weights, args.gemm1_weights_scale, args.gemm1_bias,
            args.gemm1_alpha, args.gemm1_beta, args.gemm1_clamp_limit,
            args.gemm2_weights, args.gemm2_weights_scale, args.gemm2_bias,
            args.output1_scale_scalar, args.output1_scale_gate_scalar,
            args.output2_scale_scalar, self.num_experts, self.top_k,
            self.n_group, self.topk_group, self.intermediate_size,
            self.valid_hidden_size, self.valid_intermediate_size,
            self.local_expert_offset, self.local_num_experts,
            self.routed_scaling_factor, self.routing_method_type, tactic,
            args.topk_weights, args.topk_ids, None
        )  # TODO: Currently user provided output is only supported in w4a8_mxfp4_mxfp8

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile,
                          **kwargs) -> List[List[int]]:

        args = E4m3MxE2m1BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        kernel_runner = self.get_runner()

        tactics = kernel_runner.get_valid_configs(
            self.top_k,
            hidden_size,
            self.intermediate_size,
            self.local_num_experts,
            num_tokens,
            self.valid_hidden_size or hidden_size,
            self.valid_intermediate_size or self.intermediate_size,
        )

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls,
                                 ep_size: int) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        def round_rule(x: int) -> int:
            value = last_positive_power_of_2(x) // ep_size
            return min(max(1, value), MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX,
                                   TUNED_DIM,
                                   m_values,
                                   map_to_tuning_buckets=round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_routing_logits(shapes: Tuple[torch.Size]) -> int:
            # hidden_states dim 0 and dim 1
            num_tokens = shapes[2][0]

            return num_tokens

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_RL_DIM = 0
        TOPK_WEIGHTS_IDX = 15
        TOPK_IDS_IDX = 16

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_routing_logits)
        constraint_topk_weights = ConstraintSpec(TOPK_WEIGHTS_IDX,
                                                 CONSTRAINED_RL_DIM,
                                                 _constrain_routing_logits)
        constraint_topk_ids = ConstraintSpec(TOPK_IDS_IDX, CONSTRAINED_RL_DIM,
                                             _constrain_routing_logits)

        constraint_specs_tuple = (constraint_routing_logits,
                                  constraint_topk_weights, constraint_topk_ids)

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, ep_size: int) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(ep_size)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::e4m3_mxe2m1_block_scale_moe_runner",
                         mutates_args=())
def e4m3_mxe2m1_block_scale_moe_runner(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        valid_hidden_size: Optional[int],
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None) -> torch.Tensor:

    tuner = AutoTuner.get()
    kernel_runner = E4m3MxE2m1BlockScaleMoERunner(
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        valid_hidden_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        act_type,
    )

    # Prepare dummy topk tensors and hook for AutoTuner profiling
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook = \
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
        )

    input_tensors_for_tuner = [
        routing_logits_for_tuner,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        topk_weights_for_tuner,
        topk_ids_for_tuner,
    ]

    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::e4m3_mxe2m1_block_scale_moe_runner",
        [kernel_runner],
        tuning_config_with_hook,
        input_tensors_for_tuner,
    )

    # Replace dummy tensors with actual ones for final execution
    input_tensors = input_tensors_for_tuner
    input_tensors[
        0] = routing_logits  # replace dummy routing logits with actual routing logits
    input_tensors[-2] = topk_weights  # replace dummy topk_weights with actual
    input_tensors[-1] = topk_ids  # replace dummy topk_ids with actual
    return kernel_runner(input_tensors,
                         tactic=[-1, -1] if best_tactic == -1 else best_tactic)


@dataclass(frozen=True)
class Bf16MxE2m1BlockScaleMoEInputs:
    routing_logits: Optional[torch.Tensor]
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm1_clamp_limit: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class Bf16MxE2m1BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 valid_hidden_size: int, valid_intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.valid_hidden_size = valid_hidden_size
        self.valid_intermediate_size = valid_intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_method_type = routing_method_type
        self.act_type = act_type

        self.tuning_config = Bf16MxE2m1BlockScaleMoERunner.get_tuning_config(
            self.num_experts // self.local_num_experts)

    # The unique_id is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def unique_id(self):
        return hash((
            self.top_k,
            self.intermediate_size,
            self.valid_hidden_size,
            self.valid_intermediate_size,
            self.local_num_experts,
            self.act_type,
        ))

    def get_runner(self):
        instance_key = (self.act_type, )
        if instance_key not in Bf16MxE2m1BlockScaleMoERunner.runner_dict:
            Bf16MxE2m1BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.Bf16MxE2m1BlockScaleMoERunner(
                    self.act_type)
        return Bf16MxE2m1BlockScaleMoERunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: List[int] = [-1, -1],
    ) -> torch.Tensor:
        assert isinstance(tactic, list)

        args = Bf16MxE2m1BlockScaleMoEInputs(*inputs)

        kernel_runner = self.get_runner()

        return kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.gemm1_weights, args.gemm1_weights_scale, args.gemm1_bias,
            args.gemm1_alpha, args.gemm1_beta, args.gemm1_clamp_limit,
            args.gemm2_weights, args.gemm2_weights_scale, args.gemm2_bias,
            self.num_experts, self.top_k, self.n_group, self.topk_group,
            self.intermediate_size, self.valid_hidden_size,
            self.valid_intermediate_size, self.local_expert_offset,
            self.local_num_experts, self.routed_scaling_factor,
            self.routing_method_type, tactic, args.topk_weights, args.topk_ids)

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile,
                          **kwargs) -> List[List[int]]:

        args = Bf16MxE2m1BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        kernel_runner = self.get_runner()

        tactics = kernel_runner.get_valid_configs(
            self.top_k,
            hidden_size,
            self.intermediate_size,
            self.valid_hidden_size or hidden_size,
            self.valid_intermediate_size or self.intermediate_size,
            self.local_num_experts,
            num_tokens,
        )

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls,
                                 ep_size: int) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        def round_rule(x: int) -> int:
            value = last_positive_power_of_2(x) // ep_size
            return min(max(1, value), MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX,
                                   TUNED_DIM,
                                   m_values,
                                   map_to_tuning_buckets=round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_routing_logits(shapes: Tuple[torch.Size]) -> int:
            # hidden_states dim 0 and dim 1
            num_tokens = shapes[2][0]

            return num_tokens

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_DIM = 0
        TOPK_WEIGHTS_IDX = 12
        TOPK_IDS_IDX = 13

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_DIM,
                                                   _constrain_routing_logits)
        constraint_topk_weights = ConstraintSpec(TOPK_WEIGHTS_IDX,
                                                 CONSTRAINED_DIM,
                                                 _constrain_routing_logits)
        constraint_topk_ids = ConstraintSpec(TOPK_IDS_IDX, CONSTRAINED_DIM,
                                             _constrain_routing_logits)

        constraint_specs_tuple = (constraint_routing_logits,
                                  constraint_topk_weights, constraint_topk_ids)

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, ep_size: int) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(ep_size)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::bf16_mxe2m1_block_scale_moe_runner",
                         mutates_args=())
def bf16_mxe2m1_block_scale_moe_runner(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        valid_hidden_size: Optional[int],
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None) -> torch.Tensor:

    tuner = AutoTuner.get()
    kernel_runner = Bf16MxE2m1BlockScaleMoERunner(
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        valid_hidden_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        act_type,
    )

    # Prepare dummy topk tensors and hook for AutoTuner profiling
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook = \
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
        )

    input_tensors_for_tuner = [
        routing_logits_for_tuner,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        topk_weights_for_tuner,
        topk_ids_for_tuner,
    ]

    # Choose best tactic using autotuner
    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::bf16_mxe2m1_block_scale_moe_runner",
        [kernel_runner],
        tuning_config_with_hook,
        input_tensors_for_tuner,
    )

    # Replace dummy tensors with actual ones for final execution
    input_tensors = input_tensors_for_tuner
    input_tensors[
        0] = routing_logits  # replace dummy routing logits with actual routing logits
    input_tensors[-2] = topk_weights  # replace dummy topk_weights with actual
    input_tensors[-1] = topk_ids  # replace dummy topk_ids with actual
    return kernel_runner(input_tensors,
                         tactic=[-1, -1] if best_tactic == -1 else best_tactic)


@dataclass(frozen=True)
class FP8FP4BlockScaleMoEInputs:

    routing_logits: Optional[torch.Tensor]
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class FP8FP4BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int, do_finalize: bool, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_method_type = routing_method_type
        self.do_finalize = do_finalize
        self.act_type = act_type

        self.tuning_config = FP8FP4BlockScaleMoERunner.get_tuning_config(
            self.num_experts // self.local_num_experts)

    def unique_id(self):
        return (
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.act_type,
        )

    def get_runner(self):
        instance_key = (self.act_type, )
        if instance_key not in FP8FP4BlockScaleMoERunner.runner_dict:
            FP8FP4BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8FP4BlockScaleMoERunner(
                    self.act_type)
        return FP8FP4BlockScaleMoERunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: List[int] = [-1, -1],
    ) -> torch.Tensor:
        assert isinstance(tactic, list)

        args = FP8FP4BlockScaleMoEInputs(*inputs)
        kernel_runner = self.get_runner()

        return kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.gemm1_weights, args.gemm1_weights_scale, args.gemm2_weights,
            args.gemm2_weights_scale, args.output1_scale_scalar,
            args.output1_scale_gate_scalar, args.output2_scale_scalar,
            self.num_experts, self.top_k, self.n_group, self.topk_group,
            self.intermediate_size, self.local_expert_offset,
            self.local_num_experts, self.routed_scaling_factor,
            self.routing_method_type, self.do_finalize, tactic,
            args.topk_weights, args.topk_ids)

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile,
                          **kwargs) -> List[List[int]]:

        args = FP8FP4BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]

        hidden_size = args.hidden_states.shape[1]

        kernel_runner = self.get_runner()

        tactics = kernel_runner.get_valid_configs(
            self.top_k,
            hidden_size,
            self.intermediate_size,
            self.local_num_experts,
            num_tokens,
        )

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls,
                                 ep_size: int) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        def round_rule(x: int) -> int:
            value = last_positive_power_of_2(x) // ep_size
            return min(max(1, value), MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX,
                                   TUNED_DIM,
                                   m_values,
                                   map_to_tuning_buckets=round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_to_num_tokens(shapes: Tuple[torch.Size]) -> int:
            HIDDEN_STATES_IDX = 2
            NUM_TOKENS_DIM = 0

            num_tokens = shapes[HIDDEN_STATES_IDX][NUM_TOKENS_DIM]

            return num_tokens

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_RL_DIM = 0
        TOPK_WEIGHTS_IDX = 10
        TOPK_IDS_IDX = 11

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_to_num_tokens)
        constraint_topk_weights = ConstraintSpec(TOPK_WEIGHTS_IDX,
                                                 CONSTRAINED_RL_DIM,
                                                 _constrain_to_num_tokens)
        constraint_topk_ids = ConstraintSpec(TOPK_IDS_IDX, CONSTRAINED_RL_DIM,
                                             _constrain_to_num_tokens)

        constraint_specs_tuple = (constraint_routing_logits,
                                  constraint_topk_weights, constraint_topk_ids)

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, ep_size: int) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(ep_size)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp8_fp4_block_scale_moe_runner",
                         mutates_args=())
def fp8_fp4_block_scale_moe_runner(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None) -> List[torch.Tensor]:

    tuner = AutoTuner.get()
    kernel_runner = FP8FP4BlockScaleMoERunner(
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        act_type,
    )

    # Prepare dummy topk tensors and hook for AutoTuner profiling
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook = \
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
        )

    input_tensors_for_tuner = [
        routing_logits_for_tuner,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        topk_weights_for_tuner,
        topk_ids_for_tuner,
    ]

    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::fp8_fp4_block_scale_moe_runner",
        [kernel_runner],
        tuning_config_with_hook,
        input_tensors_for_tuner,
    )

    # Replace dummy tensors with actual ones for final execution
    input_tensors = input_tensors_for_tuner
    input_tensors[
        0] = routing_logits  # replace dummy routing logits with actual routing logits
    input_tensors[-2] = topk_weights  # replace dummy topk_weights with actual
    input_tensors[-1] = topk_ids  # replace dummy topk_ids with actual
    return kernel_runner(input_tensors,
                         tactic=[-1, -1] if best_tactic == -1 else best_tactic)


@fp8_fp4_block_scale_moe_runner.register_fake
def _(routing_logits,
      routing_bias,
      hidden_states,
      gemm1_weights,
      gemm1_weights_scale,
      gemm2_weights,
      gemm2_weights_scale,
      output1_scale_scalar,
      output1_scale_gate_scalar,
      output2_scale_scalar,
      num_experts,
      top_k,
      n_group,
      topk_group,
      intermediate_size,
      local_expert_offset,
      local_num_experts,
      routed_scaling_factor,
      routing_method_type,
      do_finalize,
      act_type,
      topk_weights: Optional[torch.Tensor] = None,
      topk_ids: Optional[torch.Tensor] = None) -> List[torch.Tensor]:

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    if do_finalize:
        return [
            hidden_states.new_empty((num_tokens, hidden_size),
                                    dtype=torch.bfloat16)
        ]

    tile_tokens_dim = calculate_tile_tokens_dim(num_tokens,
                                                num_experts,
                                                top_k,
                                                max_tile_tokens_dim=64)
    expanded_row_count = num_tokens * top_k
    max_padding_required = (tile_tokens_dim - 1) * num_experts
    max_num_padded_tokens = fp4_utils.pad_up(
        expanded_row_count + max_padding_required, tile_tokens_dim)
    wt_dtype = routing_bias.dtype if routing_bias is not None else torch.bfloat16
    return [
        hidden_states.new_empty((max_num_padded_tokens, hidden_size),
                                dtype=torch.bfloat16),
        hidden_states.new_empty((num_tokens, top_k), dtype=wt_dtype),
        hidden_states.new_empty((num_tokens, top_k), dtype=torch.int32)
    ]
