from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import torch

from tensorrt_llm._torch.utils import last_positive_power_of_2

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)


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
    num_experts: int
    top_k: int
    n_group: int
    topk_group: int
    intermediate_size: int
    local_expert_offset: int
    local_num_experts: int
    routed_scaling_factor: float
    routing_method_type: int


class FP8BlockScaleMoERunner(TunableRunner):

    runner_dict = dict()
    tuning_config = None

    def __init__(self, tile_tokens_dim: int):
        self.tile_tokens_dim = tile_tokens_dim

        FP8BlockScaleMoERunner.tuning_config = FP8BlockScaleMoERunner.get_tuning_config(
        )

        instance_key = (tile_tokens_dim, )

        if instance_key not in FP8BlockScaleMoERunner.runner_dict:
            FP8BlockScaleMoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8BlockScaleMoERunner(
                    tile_tokens_dim)

        self.kernel_runner = FP8BlockScaleMoERunner.runner_dict[instance_key]

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
            args.gemm2_weights_scale, args.num_experts, args.top_k,
            args.n_group, args.topk_group, args.intermediate_size,
            args.local_expert_offset, args.local_num_experts,
            args.routed_scaling_factor, args.routing_method_type, tactic)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        args = FP8BlockScaleMoEInputs(*inputs)

        num_tokens = args.hidden_states.shape[0]
        hidden_size = args.hidden_states.shape[1]

        tactics = self.kernel_runner.get_valid_configs(args.top_k, hidden_size,
                                                       args.intermediate_size,
                                                       args.local_num_experts,
                                                       num_tokens)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0

        m_values = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        round_rule = lambda x: last_positive_power_of_2(x)

        specs = (DynamicTensorSpec(HIDDEN_STATES_IDX, TUNED_DIM, m_values,
                                   round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls) -> Tuple[ConstraintSpec, ...]:
        return ()

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp8_block_scale_moe_runner", mutates_args=())
def fp8_block_scale_moe_runner(routing_logits: torch.Tensor,
                               routing_bias: torch.Tensor,
                               hidden_states: torch.Tensor,
                               hidden_states_scale: torch.Tensor,
                               gemm1_weights: torch.Tensor,
                               gemm1_weights_scale: torch.Tensor,
                               gemm2_weights: torch.Tensor,
                               gemm2_weights_scale: torch.Tensor,
                               num_experts: int, top_k: int, n_group: int,
                               topk_group: int, intermediate_size: int,
                               local_expert_offset: int, local_num_experts: int,
                               routed_scaling_factor: float,
                               tile_tokens_dim: int,
                               routing_method_type: int) -> torch.Tensor:

    tuner = AutoTuner.get()

    kernel_runner = FP8BlockScaleMoERunner(tile_tokens_dim)

    inputs = [
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
    ]

    _, best_tactic = tuner.choose_one(
        "trtllm::fp8_block_scale_moe_runner",
        [kernel_runner],
        kernel_runner.tuning_config,
        inputs,
    )

    return kernel_runner(inputs, tactic=best_tactic)
