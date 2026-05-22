import os
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.utils import (ActType_TrtllmGen, Fp4QuantizedTensor,
                                       fp4_utils,
                                       get_last_power_of_2_num_tokens_buckets,
                                       last_positive_power_of_2,
                                       next_positive_power_of_2)

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)

_MOE_AUTOTUNE_DUMMY_DISTRIBUTION_ENV = (
    "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION")
# Distribution names control the *shape* of the autotune dummy topk
# (random vs balanced). Whether the dummy targets the local expert shard
# or all experts is decided by the caller via the `use_dp` argument:
#
#   use_dp=True — mimics load-balanced DEP (Attention DP + MoE EP).
#   The system globally holds `ep_size * runtime_max_tokens_per_rank`
#   tokens which A2A perfectly distributes across the ep_size ranks,
#   so each rank's autotune dummy carries `runtime_max_tokens_per_rank`
#   rows whose top_k slots all live in the local expert shard
#   `[local_expert_offset, local_expert_offset + local_num_experts)`.
#   m_local per local expert = `num_tokens * top_k / local_num_experts`.
#
#   use_dp=False — pure EP (no Attention DP). The system never holds
#   `ep_size * runtime_max_tokens_per_rank` tokens; each rank sees its
#   own `runtime_max_tokens_per_rank` rows whose top_k slots span global
#   experts, so only ~1/ep_size of slots hit local experts. m_local per
#   local expert = `num_tokens * top_k / num_experts`.
#
# The same `use_dp` flag also drives `round_rule` in each runner's
# `get_dynamic_tensor_specs`. The autotuner sees `x` = the kernel m-dim
# of the input buffer (= the inflated A2A recv pool when use_dp=True,
# = per-rank tokens when use_dp=False) and must map it to one of the
# m_values profile buckets. The mapping:
#
#   1. Round to last power of 2.
#   2. If use_dp=True, deflate by ep_size to assume the avg-case
#      perfectly-distributed per-rank load `runtime_max_tokens_per_rank`.
#   3. If the result still exceeds MAX_PROFILE_BUCKET (8192) and the
#      caller appended a `tune_max_num_tokens` bucket past it, route
#      there — the caller added that bucket precisely so the kernel
#      gets timed at runtime scale instead of being clamped to 8192.
#   4. Otherwise clamp into the small pow2 ladder.
_BALANCED = "balanced"
_RANDOM = "random"


def prepare_dummy_topk_and_hook(
    topk_weights: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    routing_logits: Optional[torch.Tensor],
    routing_method_type: int,
    base_tuning_config: TuningConfig,
    top_k: int,
    num_experts: int,
    local_num_experts: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    routed_scaling_factor: Optional[float],
    hidden_states_index: int = 2,
    local_expert_offset: int = 0,
    use_dp: bool = False,
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
        local_num_experts: Number of experts owned by this rank (for the
            balanced dummy-topk stride; ignored by the random distribution)
        hidden_states_index: Index of hidden_states in input_tensors list (default: 2)
        local_expert_offset: First global expert id owned by this rank;
            used to translate local-shard indices back to global ids
            when `use_dp=True` (default 0).
        use_dp: When True, the dummy mimics the load-balanced DEP
            (Attention DP + MoE EP) regime — every top_k slot targets the
            local expert shard. When False (default), the dummy spans
            all global experts (pure-EP regime). See the module-level
            distribution comment for the full framing and the
            corresponding profile-bucket math driven by `round_rule`.

    Returns:
        Tuple of (routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config_with_hook)
    """

    # NOTE: This prevents auto-tuning related code from being executed in actual runs
    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        return routing_logits, topk_weights, topk_ids, base_tuning_config

    need_dummy_topk = (topk_weights is not None or topk_ids is not None)
    autotune_distribution = os.environ.get(_MOE_AUTOTUNE_DUMMY_DISTRIBUTION_ENV,
                                           _RANDOM).lower()
    supported_distributions = {_BALANCED, _RANDOM}
    if autotune_distribution not in supported_distributions:
        raise ValueError(
            f"Unsupported {_MOE_AUTOTUNE_DUMMY_DISTRIBUTION_ENV}={autotune_distribution!r}; "
            f"expected one of {sorted(supported_distributions)}")
    is_balanced = autotune_distribution == _BALANCED
    # Caller-supplied flag: under Attention DP + MoE EP each rank's autotune
    # dummy carries `runtime_max_tokens_per_rank` rows whose top_k slots
    # all target the local expert shard (the perfectly-distributed
    # post-A2A state). Under pure EP (use_dp=False) the dummy spans all
    # experts and only ~1/ep_size of slots hit local.
    is_local = use_dp

    def make_balanced_dummy_topk(
            num_tokens: int,
            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic round-robin topk_ids of shape [num_tokens, top_k].

        `is_local` selects between two MoE deployment regimes:

          * `is_local=True` — DEP (Data + Expert Parallel + Attention
            DP). Mimics the load-balanced post-A2A layout: the global
            pool of `ep_size * runtime_max_tokens_per_rank` tokens is
            perfectly distributed across the ep_size ranks, so each
            rank sees `runtime_max_tokens_per_rank` rows whose top_k
            slots all live in the local expert shard
            `[local_expert_offset, local_expert_offset + local_num_experts)`.
            m_local per local expert = `num_tokens * top_k / local_num_experts`.
            Pure EP without Attention DP never has that many tokens in
            the system, so this regime is specific to DEP.

          * `is_local=False` — pure EP. Round-robin over all
            `num_experts`; only ~1/ep_size of slots hit local experts.
            m_local per local expert = `num_tokens * top_k / num_experts`.

        Formula (both regimes):
            n_target = local_num_experts if is_local else num_experts
            topk_ids[t, k] = (t + k * stride) % n_target  (+ local_expert_offset if is_local)

        Stride is picked so each row's `top_k` entries stay distinct;
        the assertion guards against degenerate (n_target < top_k) configs
        that would produce in-row duplicates.
        """
        n_target = local_num_experts if is_local else num_experts
        assert n_target >= top_k, (
            f"make_balanced_dummy_topk requires n_target>={top_k}; "
            f"got n_target={n_target}, is_local={is_local}, "
            f"num_experts={num_experts}, local_num_experts={local_num_experts}")
        stride = max(1, min(local_num_experts, n_target // top_k))
        if stride * top_k > n_target:
            stride = max(1, n_target // top_k)
        base = torch.arange(top_k, device=device, dtype=torch.int32) * stride
        token_idx = torch.arange(num_tokens, device=device,
                                 dtype=torch.int32).unsqueeze(1)
        topk_ids = (base + token_idx) % n_target
        if is_local:
            topk_ids = topk_ids + local_expert_offset
        topk_weights = torch.ones(num_tokens,
                                  top_k,
                                  dtype=torch.bfloat16,
                                  device=device)
        return topk_weights, topk_ids

    def make_selected_dummy_topk(
            num_tokens: int, device: torch.device,
            logits: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_balanced:
            return make_balanced_dummy_topk(num_tokens, device)
        # Random regime: `is_local` decides whether the dummy topk
        # targets only local experts (DEP) or spans all experts (pure EP).
        # The local path bypasses `routing_method.apply` and does a plain
        # `torch.topk` on per-rank logits, since the model's routing
        # method emits global indices and would have to be re-projected.
        return make_routing_dummy_topk(num_tokens, device,
                                       None if is_local else logits)

    def make_routing_method():

        # Lazy import to avoid circular import: fused_moe imports from this module.
        from tensorrt_llm._torch.modules.fused_moe.routing import (
            ROUTING_METHOD_TYPE_TO_CLASS, RoutingMethodType)

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
                lambda: torch.randn(num_experts,
                                    dtype=torch.bfloat16,
                                    device=hidden_states.device)
            })
        if routing_method_type == RoutingMethodType.MiniMax2:
            routing_cls_kwargs.update({
                'callable_e_score_correction_bias':
                lambda: torch.randn(num_experts,
                                    dtype=torch.bfloat16,
                                    device=hidden_states.device),
                'num_experts':
                num_experts,
            })
        if routing_method_type == RoutingMethodType.SigmoidRenorm:
            routing_cls_kwargs.update({
                'num_experts': num_experts,
            })
        return ROUTING_METHOD_TYPE_TO_CLASS[routing_method_type](
            top_k=top_k, **routing_cls_kwargs)

    def make_routing_dummy_topk(
        num_tokens: int,
        device: torch.device,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random topk dummy.

        `is_local` selects between two MoE deployment regimes (see the
        module-level distribution comment):

          * `is_local=True` (DEP) — generate logits over
            `local_num_experts`, take a plain `torch.topk`, then offset
            indices into the local expert shard. We deliberately bypass
            `routing_method.apply` here: the model's routing operates
            on the global `num_experts` and emits global indices, while
            the post-A2A dummy must fall entirely inside the local
            shard. CAVEAT: routing methods that group experts at global
            scale (e.g. DeepSeekV3 with `n_group` / `topk_group`,
            Llama4) won't have their group structure reflected, and
            grouped methods can be ill-defined when `local_num_experts`
            is smaller than the group cardinality. For a tuning dummy
            the distributional difference is acceptable; runtime still
            uses the real routing method.

          * `is_local=False` (pure EP) — generate logits over
            `num_experts` and run them through the model's routing method
            so the per-expert distribution matches what real inference
            would observe.
        """
        if is_local:
            assert local_num_experts >= top_k, (
                f"random_local requires local_num_experts >= top_k; "
                f"got local_num_experts={local_num_experts}, top_k={top_k}")
            if (logits is None or logits.shape[0] != num_tokens
                    or logits.shape[-1] != local_num_experts):
                logits = torch.randn(num_tokens,
                                     local_num_experts,
                                     dtype=torch.bfloat16,
                                     device=device)
            # Plain topk over local logits — bypasses the model's
            # routing_method on purpose (see docstring caveat re:
            # grouped routings like DeepSeekV3 / Llama4).
            topk_ids = torch.topk(logits.float(), top_k, dim=-1).indices.to(
                torch.int32) + local_expert_offset
            topk_weights = torch.ones(num_tokens,
                                      top_k,
                                      dtype=torch.bfloat16,
                                      device=device)
            return topk_weights, topk_ids
        routing_method = make_routing_method()
        if logits is None or logits.shape[0] != num_tokens:
            logits = torch.randn(num_tokens,
                                 num_experts,
                                 dtype=torch.bfloat16,
                                 device=device)
        topk_ids, topk_weights = routing_method.apply(logits)
        return topk_weights.to(torch.bfloat16), topk_ids

    if routing_logits is None:
        routing_logits_for_tuner = torch.randn(hidden_states.shape[0],
                                               num_experts,
                                               dtype=torch.bfloat16,
                                               device=hidden_states.device)
    else:
        routing_logits_for_tuner = routing_logits

    # Determine if we need dummy topk tensors (attention DP scenario)
    need_dummy_topk = (topk_weights is not None or topk_ids is not None)

    # Create dummy topk tensors for attention DP scenario
    if need_dummy_topk:
        # Attention DP: topk is pre-computed, no routing needed
        topk_weights_for_tuner, topk_ids_for_tuner = make_selected_dummy_topk(
            hidden_states.shape[0], hidden_states.device,
            routing_logits_for_tuner)
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

        # Only recreate if we originally created dummies
        if need_dummy_topk:
            # Check if shape changed
            if inputs[-1] is not None and inputs[-1].shape[
                    0] != current_num_tokens:
                # Recreate with new shape
                topk_weights_for_tuner, topk_ids_for_tuner = make_selected_dummy_topk(
                    current_num_tokens, inputs[hidden_states_index].device,
                    None)
                inputs[-1] = topk_ids_for_tuner
                inputs[-2] = topk_weights_for_tuner
            # Note: routing_logits is None in attention DP, no need to adjust
            assert inputs[0] is None
        # Recreate routing logits if token count changed
        elif inputs[0] is None or inputs[0].shape[0] != current_num_tokens:
            inputs[0] = torch.randn(current_num_tokens,
                                    num_experts,
                                    dtype=torch.bfloat16,
                                    device=inputs[hidden_states_index].device)

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

    def __init__(self,
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
                 tune_max_num_tokens: int = 8192,
                 use_dp: bool = False):

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
            self.num_experts // self.local_num_experts,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp)

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
        output: Optional[torch.Tensor] = None,
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
            self.do_finalize, tactic, args.topk_weights, args.topk_ids, output)

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
    def get_dynamic_tensor_specs(
            cls,
            ep_size: int,
            tune_max_num_tokens: int = 8192,
            use_dp: bool = False) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 8192

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        if tune_max_num_tokens > MAX_PROFILE_BUCKET:
            m_values = tuple(m_values) + (tune_max_num_tokens, )

        def round_rule(x: int) -> int:
            # See the module-level distribution comment for the use_dp
            # / ep_size deflation and tune_max_num_tokens fall-through.
            deflated = x // ep_size if use_dp else x
            if (deflated > MAX_PROFILE_BUCKET
                    and tune_max_num_tokens > MAX_PROFILE_BUCKET):
                return tune_max_num_tokens
            value = last_positive_power_of_2(deflated)
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
    def get_tuning_config(cls,
                          ep_size: int,
                          tune_max_num_tokens: int = 8192,
                          use_dp: bool = False) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(
            ep_size, tune_max_num_tokens, use_dp)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs,
                                     tune_max_num_tokens=max(
                                         8192, tune_max_num_tokens))

        return tuning_config


@torch.library.custom_op("trtllm::fp4_block_scale_moe_runner", mutates_args=())
def fp4_block_scale_moe_runner(routing_logits: Optional[torch.Tensor],
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
                               act_type: int = ActType_TrtllmGen.SwiGlu.value,
                               topk_weights: Optional[torch.Tensor] = None,
                               topk_ids: Optional[torch.Tensor] = None,
                               output: Optional[torch.Tensor] = None,
                               tune_max_num_tokens: int = 8192,
                               use_dp: bool = False) -> List[torch.Tensor]:

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
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
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
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
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
    result = kernel_runner(
        input_tensors,
        tactic=[-1, -1] if best_tactic == -1 else best_tactic,
        output=output)
    # When output is provided and do_finalize=True, the result is written in-place to output.
    # Return empty tensor to avoid aliasing constraint violation in PyTorch 2.9.1+
    # (custom op output cannot be the same tensor as input).
    # Callers should use output directly when they provide it.
    if output is not None and do_finalize:
        return [torch.empty(0, device=output.device, dtype=output.dtype)]
    return result


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
      topk_ids: Optional[torch.Tensor] = None,
      output: Optional[torch.Tensor] = None,
      tune_max_num_tokens: int = 8192,
      use_dp: bool = False) -> List[torch.Tensor]:
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
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
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
            self.num_experts // self.local_num_experts,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp)

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
        output: Optional[torch.Tensor] = None,
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
            args.topk_weights, args.topk_ids, output)

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
    def get_dynamic_tensor_specs(
            cls,
            ep_size: int,
            tune_max_num_tokens: int = 8192,
            use_dp: bool = False) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 8192

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        if tune_max_num_tokens > MAX_PROFILE_BUCKET:
            m_values = tuple(m_values) + (tune_max_num_tokens, )

        def round_rule(x: int) -> int:
            # See the module-level distribution comment for the use_dp
            # / ep_size deflation and tune_max_num_tokens fall-through.
            deflated = x // ep_size if use_dp else x
            if (deflated > MAX_PROFILE_BUCKET
                    and tune_max_num_tokens > MAX_PROFILE_BUCKET):
                return tune_max_num_tokens
            value = last_positive_power_of_2(deflated)
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
    def get_tuning_config(cls,
                          ep_size: int,
                          tune_max_num_tokens: int = 8192,
                          use_dp: bool = False) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(
            ep_size, tune_max_num_tokens, use_dp)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs,
                                     tune_max_num_tokens=max(
                                         8192, tune_max_num_tokens))

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
                               act_type: int = 0,
                               output: Optional[torch.Tensor] = None,
                               tune_max_num_tokens: int = 8192,
                               use_dp: bool = False) -> torch.Tensor:

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
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
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
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
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
      topk_ids: Optional[torch.Tensor] = None,
      act_type: int = 0,
      output: Optional[torch.Tensor] = None,
      tune_max_num_tokens: int = 8192,
      use_dp: bool = False) -> torch.Tensor:
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

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 n_group: Optional[int],
                 topk_group: Optional[int],
                 intermediate_size: int,
                 valid_hidden_size: int,
                 valid_intermediate_size: int,
                 local_expert_offset: int,
                 local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int,
                 act_type: int,
                 tune_max_num_tokens: int = 8192,
                 use_dp: bool = False):

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
            self.num_experts // self.local_num_experts,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp)

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
    def get_dynamic_tensor_specs(
            cls,
            ep_size: int,
            tune_max_num_tokens: int = 8192,
            use_dp: bool = False) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 8192

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        if tune_max_num_tokens > MAX_PROFILE_BUCKET:
            m_values = tuple(m_values) + (tune_max_num_tokens, )

        def round_rule(x: int) -> int:
            # See the module-level distribution comment for the use_dp
            # / ep_size deflation and tune_max_num_tokens fall-through.
            deflated = x // ep_size if use_dp else x
            if (deflated > MAX_PROFILE_BUCKET
                    and tune_max_num_tokens > MAX_PROFILE_BUCKET):
                return tune_max_num_tokens
            value = last_positive_power_of_2(deflated)
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
    def get_tuning_config(cls,
                          ep_size: int,
                          tune_max_num_tokens: int = 8192,
                          use_dp: bool = False) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(
            ep_size, tune_max_num_tokens, use_dp)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs,
                                     tune_max_num_tokens=max(
                                         8192, tune_max_num_tokens))

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
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False) -> torch.Tensor:

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
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
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
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
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

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 n_group: Optional[int],
                 topk_group: Optional[int],
                 intermediate_size: int,
                 valid_hidden_size: int,
                 valid_intermediate_size: int,
                 local_expert_offset: int,
                 local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int,
                 act_type: int,
                 tune_max_num_tokens: int = 8192,
                 use_dp: bool = False):

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
            self.num_experts // self.local_num_experts,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp)

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
        output: Optional[torch.Tensor] = None,
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
            args.topk_weights, args.topk_ids, output)

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
    def get_dynamic_tensor_specs(
            cls,
            ep_size: int,
            tune_max_num_tokens: int = 8192,
            use_dp: bool = False) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 8192

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        if tune_max_num_tokens > MAX_PROFILE_BUCKET:
            m_values = tuple(m_values) + (tune_max_num_tokens, )

        def round_rule(x: int) -> int:
            # See the module-level distribution comment for the use_dp
            # / ep_size deflation and tune_max_num_tokens fall-through.
            deflated = x // ep_size if use_dp else x
            if (deflated > MAX_PROFILE_BUCKET
                    and tune_max_num_tokens > MAX_PROFILE_BUCKET):
                return tune_max_num_tokens
            value = last_positive_power_of_2(deflated)
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
    def get_tuning_config(cls,
                          ep_size: int,
                          tune_max_num_tokens: int = 8192,
                          use_dp: bool = False) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(
            ep_size, tune_max_num_tokens, use_dp)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs,
                                     tune_max_num_tokens=max(
                                         8192, tune_max_num_tokens))

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
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False) -> torch.Tensor:

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
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
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
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
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

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 n_group: Optional[int],
                 topk_group: Optional[int],
                 intermediate_size: int,
                 valid_hidden_size: int,
                 valid_intermediate_size: int,
                 local_expert_offset: int,
                 local_num_experts: int,
                 routed_scaling_factor: Optional[float],
                 routing_method_type: int,
                 act_type: int,
                 tune_max_num_tokens: int = 8192,
                 use_dp: bool = False):

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
            self.num_experts // self.local_num_experts,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp)

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
        output: Optional[torch.Tensor] = None,
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
            self.routing_method_type, tactic, args.topk_weights, args.topk_ids,
            output)

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
    def get_dynamic_tensor_specs(
            cls,
            ep_size: int,
            tune_max_num_tokens: int = 8192,
            use_dp: bool = False) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 8192

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        if tune_max_num_tokens > MAX_PROFILE_BUCKET:
            m_values = tuple(m_values) + (tune_max_num_tokens, )

        def round_rule(x: int) -> int:
            # See the module-level distribution comment for the use_dp
            # / ep_size deflation and tune_max_num_tokens fall-through.
            deflated = x // ep_size if use_dp else x
            if (deflated > MAX_PROFILE_BUCKET
                    and tune_max_num_tokens > MAX_PROFILE_BUCKET):
                return tune_max_num_tokens
            value = last_positive_power_of_2(deflated)
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
    def get_tuning_config(cls,
                          ep_size: int,
                          tune_max_num_tokens: int = 8192,
                          use_dp: bool = False) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(
            ep_size, tune_max_num_tokens, use_dp)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs,
                                     tune_max_num_tokens=max(
                                         8192, tune_max_num_tokens))

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
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False) -> torch.Tensor:

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
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
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
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
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

    def __init__(self,
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
                 tune_max_num_tokens: int = 8192,
                 use_dp: bool = False):

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
            self.num_experts // self.local_num_experts,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp)

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
        output: Optional[torch.Tensor] = None,
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
            args.topk_weights, args.topk_ids, output)

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
    def get_dynamic_tensor_specs(
            cls,
            ep_size: int,
            tune_max_num_tokens: int = 8192,
            use_dp: bool = False) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0
        MAX_PROFILE_BUCKET = 8192

        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)
        if tune_max_num_tokens > MAX_PROFILE_BUCKET:
            m_values = tuple(m_values) + (tune_max_num_tokens, )

        def round_rule(x: int) -> int:
            # See the module-level distribution comment for the use_dp
            # / ep_size deflation and tune_max_num_tokens fall-through.
            deflated = x // ep_size if use_dp else x
            if (deflated > MAX_PROFILE_BUCKET
                    and tune_max_num_tokens > MAX_PROFILE_BUCKET):
                return tune_max_num_tokens
            value = last_positive_power_of_2(deflated)
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
    def get_tuning_config(cls,
                          ep_size: int,
                          tune_max_num_tokens: int = 8192,
                          use_dp: bool = False) -> TuningConfig:

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs(
            ep_size, tune_max_num_tokens, use_dp)
        constraint_specs = cls.get_constraint_specs()

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs,
                                     tune_max_num_tokens=max(
                                         8192, tune_max_num_tokens))

        return tuning_config


@torch.library.custom_op("trtllm::fp8_fp4_block_scale_moe_runner",
                         mutates_args=())
def fp8_fp4_block_scale_moe_runner(routing_logits: Optional[torch.Tensor],
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
                                   topk_ids: Optional[torch.Tensor] = None,
                                   output: Optional[torch.Tensor] = None,
                                   tune_max_num_tokens: int = 8192,
                                   use_dp: bool = False) -> List[torch.Tensor]:

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
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
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
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
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
    result = kernel_runner(
        input_tensors,
        tactic=[-1, -1] if best_tactic == -1 else best_tactic,
        output=output)
    # When output is provided and do_finalize=True, the result is written in-place to output.
    # Return empty tensor to avoid aliasing constraint violation in PyTorch 2.9.1+
    # (custom op output cannot be the same tensor as input).
    # Callers should use output directly when they provide it.
    if output is not None and do_finalize:
        return [torch.empty(0, device=output.device, dtype=output.dtype)]
    return result


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
      topk_ids: Optional[torch.Tensor] = None,
      output: Optional[torch.Tensor] = None,
      tune_max_num_tokens: int = 8192,
      use_dp: bool = False) -> List[torch.Tensor]:

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
