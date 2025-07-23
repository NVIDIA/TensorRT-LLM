from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.utils import (fp4_utils,
                                       get_last_power_of_2_num_tokens_buckets,
                                       last_positive_power_of_2,
                                       next_positive_power_of_2)

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)


def calculate_tile_tokens_dim(num_tokens: int, num_experts: int,
                              top_k: int) -> int:
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = num_tokens * top_k // num_experts

    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim


@dataclass(frozen=True)
class FP4BlockScaleMoEInputs:

    routing_logits: torch.Tensor
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor


class FP4BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float], tile_tokens_dim: int,
                 routing_method_type: int, do_finalize: bool):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.tile_tokens_dim = tile_tokens_dim
        self.routing_method_type = routing_method_type
        self.do_finalize = do_finalize

        FP4BlockScaleMoERunner.tuning_config = FP4BlockScaleMoERunner.get_tuning_config(
        )

        instance_key = (
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
        )

        if instance_key not in FP4BlockScaleMoERunner.runner_dict:
            FP4BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP4BlockScaleMoERunner(
                    tile_tokens_dim)

        self.kernel_runner = FP4BlockScaleMoERunner.runner_dict[instance_key]

    # The hash is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2
    # so the routing type does not matter
    def __hash__(self):
        return hash((
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
        ))

    # __eq__ and __hash__ must agree
    def __eq__(self, other):
        if not isinstance(other, FP4BlockScaleMoERunner):
            return False

        return (self.top_k == other.top_k
                and self.intermediate_size == other.intermediate_size
                and self.local_num_experts == other.local_num_experts
                and self.tile_tokens_dim == other.tile_tokens_dim)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:

        args = FP4BlockScaleMoEInputs(*inputs)

        return self.kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.hidden_states_scale, args.gemm1_weights,
            args.gemm1_weights_scale, args.gemm2_weights,
            args.gemm2_weights_scale, args.output1_scale_scalar,
            args.output1_scale_gate_scalar, args.output2_scale_scalar,
            self.num_experts, self.top_k, self.n_group, self.topk_group,
            self.intermediate_size, self.local_expert_offset,
            self.local_num_experts, self.routed_scaling_factor,
            self.routing_method_type, self.do_finalize, tactic)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        args = FP4BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]

        # The hidden size is actually 2 * hidden_size because we pack 2x e2m1
        # into 1 byte.
        hidden_size = args.hidden_states.shape[1] * 2

        tactics = self.kernel_runner.get_valid_configs(self.top_k, hidden_size,
                                                       self.intermediate_size,
                                                       self.local_num_experts,
                                                       num_tokens)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        round_rule = lambda x: min(last_positive_power_of_2(x),
                                   MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX, TUNED_DIM, m_values,
                                   round_rule), )

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

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_to_num_tokens)

        constraint_specs_tuple = (
            constraint_hidden_states_scale,
            constraint_routing_logits,
        )

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp4_block_scale_moe_runner", mutates_args=())
def fp4_block_scale_moe_runner(routing_logits: torch.Tensor,
                               routing_bias: Optional[torch.Tensor],
                               hidden_states: torch.Tensor,
                               hidden_states_scale: torch.Tensor,
                               gemm1_weights: torch.Tensor,
                               gemm1_weights_scale: torch.Tensor,
                               gemm2_weights: torch.Tensor,
                               gemm2_weights_scale: torch.Tensor,
                               output1_scale_scalar: torch.Tensor,
                               output1_scale_gate_scalar: torch.Tensor,
                               output2_scale_scalar: torch.Tensor,
                               num_experts: int, top_k: int,
                               n_group: Optional[int],
                               topk_group: Optional[int],
                               intermediate_size: int, local_expert_offset: int,
                               local_num_experts: int,
                               routed_scaling_factor: Optional[float],
                               routing_method_type: int,
                               do_finalize: bool) -> List[torch.Tensor]:

    tuner = AutoTuner.get()

    num_tokens = hidden_states.shape[0]
    tile_tokens_dim = calculate_tile_tokens_dim(num_tokens, num_experts, top_k)

    kernel_runner = FP4BlockScaleMoERunner(
        num_experts, top_k, n_group, topk_group, intermediate_size,
        local_expert_offset, local_num_experts, routed_scaling_factor,
        tile_tokens_dim, routing_method_type, do_finalize)

    inputs = [
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
    ]

    _, best_tactic = tuner.choose_one(
        "trtllm::fp4_block_scale_moe_runner",
        [kernel_runner],
        kernel_runner.tuning_config,
        inputs,
    )

    return kernel_runner(inputs, tactic=best_tactic)


@fp4_block_scale_moe_runner.register_fake
def _(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
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
) -> List[torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] * 2
    if do_finalize:
        return [
            hidden_states.new_empty((num_tokens, hidden_size),
                                    dtype=torch.bfloat16)
        ]

    tile_tokens_dim = calculate_tile_tokens_dim(num_tokens, num_experts, top_k)

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


@dataclass(frozen=True)
class FP8BlockScaleMoEInputs:

    routing_logits: torch.Tensor
    routing_bias: torch.Tensor
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor


class FP8BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: int,
                 topk_group: int, intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: float, tile_tokens_dim: int,
                 routing_method_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.tile_tokens_dim = tile_tokens_dim
        self.routing_method_type = routing_method_type

        FP8BlockScaleMoERunner.tuning_config = FP8BlockScaleMoERunner.get_tuning_config(
        )

        instance_key = (
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
        )

        if instance_key not in FP8BlockScaleMoERunner.runner_dict:
            FP8BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8BlockScaleMoERunner(
                    tile_tokens_dim)

        self.kernel_runner = FP8BlockScaleMoERunner.runner_dict[instance_key]

    # The hash is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def __hash__(self):
        return hash((
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
        ))

    # __eq__ and __hash__ must agree
    def __eq__(self, other):
        if not isinstance(other, FP8BlockScaleMoERunner):
            return False

        return (self.top_k == other.top_k
                and self.intermediate_size == other.intermediate_size
                and self.local_num_experts == other.local_num_experts
                and self.tile_tokens_dim == other.tile_tokens_dim)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:

        args = FP8BlockScaleMoEInputs(*inputs)

        return self.kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.hidden_states_scale, args.gemm1_weights,
            args.gemm1_weights_scale, args.gemm2_weights,
            args.gemm2_weights_scale, self.num_experts, self.top_k,
            self.n_group, self.topk_group, self.intermediate_size,
            self.local_expert_offset, self.local_num_experts,
            self.routed_scaling_factor, self.routing_method_type, tactic)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        args = FP8BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        tactics = self.kernel_runner.get_valid_configs(self.top_k, hidden_size,
                                                       self.intermediate_size,
                                                       self.local_num_experts,
                                                       num_tokens)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0

        MAX_PROFILE_BUCKET = 4096

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        round_rule = lambda x: min(last_positive_power_of_2(x),
                                   MAX_PROFILE_BUCKET)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX, TUNED_DIM, m_values,
                                   round_rule), )

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

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_to_num_tokens)

        constraint_specs_tuple = (
            constraint_hidden_states_scale,
            constraint_routing_logits,
        )

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp8_block_scale_moe_runner", mutates_args=())
def fp8_block_scale_moe_runner(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    routing_method_type: int,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    num_tokens = hidden_states.shape[0]
    tile_tokens_dim = calculate_tile_tokens_dim(num_tokens, num_experts, top_k)

    kernel_runner = FP8BlockScaleMoERunner(num_experts, top_k, n_group,
                                           topk_group, intermediate_size,
                                           local_expert_offset,
                                           local_num_experts,
                                           routed_scaling_factor,
                                           tile_tokens_dim, routing_method_type)

    inputs = [
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    ]

    _, best_tactic = tuner.choose_one(
        "trtllm::fp8_block_scale_moe_runner",
        [kernel_runner],
        kernel_runner.tuning_config,
        inputs,
    )

    return kernel_runner(inputs, tactic=best_tactic)


@fp8_block_scale_moe_runner.register_fake
def _(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    routing_method_type: int,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] * 2

    return hidden_states.new_empty((num_tokens, hidden_size),
                                   dtype=torch.bfloat16)


@dataclass(frozen=True)
class MxE4m3MxE2m1BlockScaleMoEInputs:
    routing_logits: torch.Tensor
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: Optional[torch.Tensor]


class MxE4m3MxE2m1BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 hidden_size_output: int, local_expert_offset: int,
                 local_num_experts: int, routed_scaling_factor: Optional[float],
                 tile_tokens_dim: int, routing_method_type: int, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.hidden_size_output = hidden_size_output
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.tile_tokens_dim = tile_tokens_dim
        self.routing_method_type = routing_method_type
        self.act_type = act_type

        MxE4m3MxE2m1BlockScaleMoERunner.tuning_config = MxE4m3MxE2m1BlockScaleMoERunner.get_tuning_config(
        )

        instance_key = (
            self.top_k,
            self.intermediate_size,
            self.hidden_size_output,
            self.local_num_experts,
            self.tile_tokens_dim,
            self.act_type,
        )

        if instance_key not in MxE4m3MxE2m1BlockScaleMoERunner.runner_dict:
            MxE4m3MxE2m1BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(
                    self.tile_tokens_dim, self.act_type, True)

        self.kernel_runner = MxE4m3MxE2m1BlockScaleMoERunner.runner_dict[
            instance_key]

    # The hash is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def __hash__(self):
        return hash((
            self.top_k,
            self.intermediate_size,
            self.hidden_size_output,
            self.local_num_experts,
            self.tile_tokens_dim,
            self.act_type,
        ))

    # __eq__ and __hash__ must agree
    def __eq__(self, other):
        if not isinstance(other, MxE4m3MxE2m1BlockScaleMoERunner):
            return False

        return (self.top_k == other.top_k
                and self.intermediate_size == other.intermediate_size
                and self.hidden_size_output == other.hidden_size_output
                and self.local_num_experts == other.local_num_experts
                and self.tile_tokens_dim == other.tile_tokens_dim
                and self.act_type == other.act_type)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:

        args = MxE4m3MxE2m1BlockScaleMoEInputs(*inputs)

        return self.kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.hidden_states_scale, args.gemm1_weights,
            args.gemm1_weights_scale, args.gemm1_bias, args.gemm1_alpha,
            args.gemm1_beta, args.gemm2_weights, args.gemm2_weights_scale,
            args.gemm2_bias, None, None, None, self.num_experts, self.top_k,
            self.n_group, self.topk_group, self.intermediate_size,
            self.hidden_size_output, self.local_expert_offset,
            self.local_num_experts, self.routed_scaling_factor,
            self.routing_method_type, tactic)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        args = MxE4m3MxE2m1BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        tactics = self.kernel_runner.get_valid_configs(self.top_k, hidden_size,
                                                       self.intermediate_size,
                                                       self.local_num_experts,
                                                       num_tokens)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0

        m_values = get_last_power_of_2_num_tokens_buckets(4096)
        round_rule = lambda x: min(last_positive_power_of_2(x), 4096)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX, TUNED_DIM, m_values,
                                   round_rule), )

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

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_routing_logits)

        constraint_specs_tuple = (constraint_hidden_states_scale,
                                  constraint_routing_logits)

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::mxe4m3_mxe2m1_block_scale_moe_runner",
                         mutates_args=())
def mxe4m3_mxe2m1_block_scale_moe_runner(
        routing_logits: torch.Tensor, routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor, gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor], gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor], gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor, gemm2_bias: Optional[torch.Tensor],
        num_experts: int, top_k: int, n_group: Optional[int],
        topk_group: Optional[int], intermediate_size: int,
        hidden_size_output: int, local_expert_offset: int,
        local_num_experts: int, routed_scaling_factor: Optional[float],
        tile_tokens_dim: int, routing_method_type: int,
        act_type: int) -> torch.Tensor:

    tuner = AutoTuner.get()

    kernel_runner = MxE4m3MxE2m1BlockScaleMoERunner(
        num_experts, top_k, n_group, topk_group, intermediate_size,
        hidden_size_output, local_expert_offset, local_num_experts,
        routed_scaling_factor, tile_tokens_dim, routing_method_type, act_type)

    input_tensors = [
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
    ]

    _, best_tactic = tuner.choose_one(
        "trtllm::mxe4m3_mxe2m1_block_scale_moe_runner",
        [kernel_runner],
        kernel_runner.tuning_config,
        input_tensors,
    )

    return kernel_runner(input_tensors, tactic=best_tactic)


@dataclass(frozen=True)
class E4m3MxE2m1BlockScaleMoEInputs:
    routing_logits: torch.Tensor
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: Optional[torch.Tensor]
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor


class E4m3MxE2m1BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float], tile_tokens_dim: int,
                 routing_method_type: int, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.tile_tokens_dim = tile_tokens_dim
        self.routing_method_type = routing_method_type
        self.act_type = act_type

        E4m3MxE2m1BlockScaleMoERunner.tuning_config = E4m3MxE2m1BlockScaleMoERunner.get_tuning_config(
        )

        instance_key = (
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
            self.act_type,
        )

        if instance_key not in E4m3MxE2m1BlockScaleMoERunner.runner_dict:
            E4m3MxE2m1BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(
                    self.tile_tokens_dim, self.act_type, False)

        self.kernel_runner = E4m3MxE2m1BlockScaleMoERunner.runner_dict[
            instance_key]

    # The hash is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def __hash__(self):
        return hash((
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
            self.act_type,
        ))

    # __eq__ and __hash__ must agree
    def __eq__(self, other):
        if not isinstance(other, E4m3MxE2m1BlockScaleMoERunner):
            return False

        return (self.top_k == other.top_k
                and self.intermediate_size == other.intermediate_size
                and self.local_num_experts == other.local_num_experts
                and self.tile_tokens_dim == other.tile_tokens_dim
                and self.act_type == other.act_type)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:

        args = E4m3MxE2m1BlockScaleMoEInputs(*inputs)

        return self.kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states, None,
            args.gemm1_weights, args.gemm1_weights_scale, args.gemm1_bias,
            args.gemm1_alpha, args.gemm1_beta, args.gemm2_weights,
            args.gemm2_weights_scale, args.gemm2_bias,
            args.output1_scale_scalar, args.output1_scale_gate_scalar,
            args.output2_scale_scalar, self.num_experts, self.top_k,
            self.n_group, self.topk_group, self.intermediate_size, None,
            self.local_expert_offset, self.local_num_experts,
            self.routed_scaling_factor, self.routing_method_type, tactic)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        args = E4m3MxE2m1BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        tactics = self.kernel_runner.get_valid_configs(self.top_k, hidden_size,
                                                       self.intermediate_size,
                                                       self.local_num_experts,
                                                       num_tokens)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0

        m_values = get_last_power_of_2_num_tokens_buckets(4096)
        round_rule = lambda x: min(last_positive_power_of_2(x), 4096)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX, TUNED_DIM, m_values,
                                   round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_routing_logits(shapes: Tuple[torch.Size]) -> int:
            # hidden_states dim 0 and dim 1
            num_tokens = shapes[2][0]

            return num_tokens

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_RL_DIM = 0

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_RL_DIM,
                                                   _constrain_routing_logits)

        constraint_specs_tuple = (constraint_routing_logits, )

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::e4m3_mxe2m1_block_scale_moe_runner",
                         mutates_args=())
def e4m3_mxe2m1_block_scale_moe_runner(
        routing_logits: torch.Tensor, routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor, gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor, gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor], gemm1_beta: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor, gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor], output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor, num_experts: int, top_k: int,
        n_group: Optional[int], topk_group: Optional[int],
        intermediate_size: int, local_expert_offset: int,
        local_num_experts: int, routed_scaling_factor: Optional[float],
        tile_tokens_dim: int, routing_method_type: int,
        act_type: int) -> torch.Tensor:

    tuner = AutoTuner.get()

    kernel_runner = E4m3MxE2m1BlockScaleMoERunner(
        num_experts, top_k, n_group, topk_group, intermediate_size,
        local_expert_offset, local_num_experts, routed_scaling_factor,
        tile_tokens_dim, routing_method_type, act_type)

    input_tensors = [
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
    ]

    _, best_tactic = tuner.choose_one(
        "trtllm::e4m3_mxe2m1_block_scale_moe_runner",
        [kernel_runner],
        kernel_runner.tuning_config,
        input_tensors,
    )

    return kernel_runner(input_tensors, tactic=best_tactic)


@dataclass(frozen=True)
class Bf16MxE2m1BlockScaleMoEInputs:
    routing_logits: torch.Tensor
    routing_bias: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_bias: Optional[torch.Tensor]
    gemm1_alpha: Optional[torch.Tensor]
    gemm1_beta: Optional[torch.Tensor]
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm2_bias: Optional[torch.Tensor]


class Bf16MxE2m1BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, num_experts: int, top_k: int, n_group: Optional[int],
                 topk_group: Optional[int], intermediate_size: int,
                 local_expert_offset: int, local_num_experts: int,
                 routed_scaling_factor: Optional[float], tile_tokens_dim: int,
                 routing_method_type: int, act_type: int):

        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.intermediate_size = intermediate_size
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = local_num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.tile_tokens_dim = tile_tokens_dim
        self.routing_method_type = routing_method_type
        self.act_type = act_type

        Bf16MxE2m1BlockScaleMoERunner.tuning_config = Bf16MxE2m1BlockScaleMoERunner.get_tuning_config(
        )

        instance_key = (
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
            self.act_type,
        )

        if instance_key not in Bf16MxE2m1BlockScaleMoERunner.runner_dict:
            Bf16MxE2m1BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.Bf16MxE2m1BlockScaleMoERunner(
                    self.tile_tokens_dim, self.act_type)

        self.kernel_runner = Bf16MxE2m1BlockScaleMoERunner.runner_dict[
            instance_key]

    # The hash is used by the autotuner to get the cache key, so we hash on members
    # that influence tactic validity here. e.g. we are tuning FC1 and FC2 so the routing
    # type does not matter
    def __hash__(self):
        return hash((
            self.top_k,
            self.intermediate_size,
            self.local_num_experts,
            self.tile_tokens_dim,
            self.act_type,
        ))

    # __eq__ and __hash__ must agree
    def __eq__(self, other):
        if not isinstance(other, Bf16MxE2m1BlockScaleMoERunner):
            return False

        return (self.top_k == other.top_k
                and self.intermediate_size == other.intermediate_size
                and self.local_num_experts == other.local_num_experts
                and self.tile_tokens_dim == other.tile_tokens_dim
                and self.act_type == other.act_type)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:

        args = Bf16MxE2m1BlockScaleMoEInputs(*inputs)

        return self.kernel_runner.run_moe(
            args.routing_logits, args.routing_bias, args.hidden_states,
            args.gemm1_weights, args.gemm1_weights_scale, args.gemm1_bias,
            args.gemm1_alpha, args.gemm1_beta, args.gemm2_weights,
            args.gemm2_weights_scale, args.gemm2_bias, self.num_experts,
            self.top_k, self.n_group, self.topk_group, self.intermediate_size,
            self.local_expert_offset, self.local_num_experts,
            self.routed_scaling_factor, self.routing_method_type, tactic)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        args = Bf16MxE2m1BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        tactics = self.kernel_runner.get_valid_configs(self.top_k, hidden_size,
                                                       self.intermediate_size,
                                                       self.local_num_experts,
                                                       num_tokens)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0

        m_values = get_last_power_of_2_num_tokens_buckets(4096)
        round_rule = lambda x: min(last_positive_power_of_2(x), 4096)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX, TUNED_DIM, m_values,
                                   round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:

        def _constrain_routing_logits(shapes: Tuple[torch.Size]) -> int:
            # hidden_states dim 0 and dim 1
            num_tokens = shapes[2][0]

            return num_tokens

        ROUTER_LOGITS_IDX = 0
        CONSTRAINED_DIM = 0

        constraint_routing_logits = ConstraintSpec(ROUTER_LOGITS_IDX,
                                                   CONSTRAINED_DIM,
                                                   _constrain_routing_logits)

        constraint_specs_tuple = (constraint_routing_logits, )

        return constraint_specs_tuple

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::bf16_mxe2m1_block_scale_moe_runner",
                         mutates_args=())
def bf16_mxe2m1_block_scale_moe_runner(
        routing_logits: torch.Tensor, routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor, gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor, gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor], gemm1_beta: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor, gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor], num_experts: int, top_k: int,
        n_group: Optional[int], topk_group: Optional[int],
        intermediate_size: int, local_expert_offset: int,
        local_num_experts: int, routed_scaling_factor: Optional[float],
        tile_tokens_dim: int, routing_method_type: int,
        act_type: int) -> torch.Tensor:

    tuner = AutoTuner.get()

    kernel_runner = Bf16MxE2m1BlockScaleMoERunner(
        num_experts, top_k, n_group, topk_group, intermediate_size,
        local_expert_offset, local_num_experts, routed_scaling_factor,
        tile_tokens_dim, routing_method_type, act_type)

    input_tensors = [
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
    ]

    _, best_tactic = tuner.choose_one(
        "trtllm::bf16_mxe2m1_block_scale_moe_runner",
        [kernel_runner],
        kernel_runner.tuning_config,
        input_tensors,
    )

    return kernel_runner(input_tensors, tactic=best_tactic)
