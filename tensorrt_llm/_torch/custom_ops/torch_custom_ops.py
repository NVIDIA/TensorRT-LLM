from functools import lru_cache
from typing import List, Optional, Tuple

import torch

from ..attention_backend.interface import AttentionInputType
from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)
from ..utils import (compute_swizzled_sf_shape, fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)


# Used to WAR an issue in torch.bmm that it would break the graph when the out is not contiguous.
@torch.library.custom_op("trtllm::bmm_out", mutates_args=("out", ))
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(a, b, out=out)


class MoERunner(TunableRunner):
    # avoid overhead of creating a new runner in forward pass
    runner_dict = dict()
    # TODO: only profile for min_latency_mode = False due to the error in the moe_kernels
    tuning_config = TuningConfig(dynamic_tensor_specs=(
        DynamicTensorSpec(0, 0, get_last_power_of_2_num_tokens_buckets(8192),
                          lambda x: min(last_positive_power_of_2(x), 8192)),
        DynamicTensorSpec(3, 0, (0, ), lambda x: x),
    ))

    def __init__(
        self,
        x_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        output_dtype: torch.dtype,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        cluster_size: int,
        cluster_rank: int,
        use_deepseek_fp8_block_scale: bool,
        use_w4a8_group_scaling: bool,
    ):
        self.x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.output_dtype = output_dtype
        self.top_k = top_k
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.cluster_size = cluster_size
        self.cluster_rank = cluster_rank
        self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale
        self.use_w4a8_group_scaling = use_w4a8_group_scaling

        instance_key = (x_dtype, weight_dtype, output_dtype,
                        use_deepseek_fp8_block_scale, use_w4a8_group_scaling)

        if instance_key not in MoERunner.runner_dict:
            MoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FusedMoeRunner(
                    x_dtype, weight_dtype, output_dtype,
                    use_deepseek_fp8_block_scale, use_w4a8_group_scaling)
        self.fused_moe_runner = MoERunner.runner_dict[instance_key]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        x, _, _, min_latency_mode_tensor = inputs
        min_latency_mode = min_latency_mode_tensor.size(0) == 1
        m = x.shape[0]

        # Only profile m <= 128 for min latency mode = True
        # Profile all valid buckets for min latency mode = False
        # TODO: min_latency_mode = True will cause the following error:
        # Cannot profile configuration 4: Cutlass GEMM Tactic
        # [TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
        # Should be fixed in the moe_kernels in the future.
        invalid = (m > 128 and
                   min_latency_mode) or (m <= 128 and min_latency_mode and
                                         (not self.weight_dtype == torch.int64))

        return [] if invalid else list(
            range(self.fused_moe_runner.get_tactic_num()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        gemm_idx: int = 0,
        tactic: int = -1,
        do_preparation: bool = False,
    ):
        x, fc1_expert_weights, fc2_expert_weights, min_latency_mode_tensor = inputs
        min_latency_mode = min_latency_mode_tensor.size(0) == 1
        # determine if we should use min latency mode according to the profiled seq len
        self.fused_moe_runner.run_gemm_profile(
            x,
            fc1_expert_weights,
            fc2_expert_weights,
            self.top_k,
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
            self.cluster_size,
            self.cluster_rank,
            min_latency_mode,
            gemm_idx,
            tactic,
            do_preparation,
        )

    @classmethod
    @lru_cache(maxsize=None)
    def refine_tuning_config(cls, tune_max_num_tokens: int):
        cls.tuning_config = TuningConfig(dynamic_tensor_specs=(
            DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets(
                    tune_max_num_tokens), lambda x: min(
                        last_positive_power_of_2(x), tune_max_num_tokens)),
            DynamicTensorSpec(3, 0, (0, ), lambda x: x),
        ))


@torch.library.custom_op("trtllm::fused_moe", mutates_args=())
def fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4a8_group_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
) -> List[torch.Tensor]:

    tuner = AutoTuner.get()
    MoERunner.refine_tuning_config(tune_max_num_tokens)

    # TODO: set min_latency_mode always to False due to the error in the moe_kernels
    min_latency_tensor = torch.empty(0)

    # allocate workspace for profiling
    moe_runner = MoERunner(
        x_dtype=input.dtype,
        weight_dtype=fc1_expert_weights.dtype,
        output_dtype=output_dtype,
        top_k=token_selected_experts.size(1),
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cluster_size=cluster_size,
        cluster_rank=cluster_rank,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4a8_group_scaling=use_w4a8_group_scaling,
    )

    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1",
        [moe_runner],
        MoERunner.tuning_config,
        [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
        gemm_idx=1,
    )

    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2",
        [moe_runner],
        MoERunner.tuning_config,
        [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
        gemm_idx=2,
    )

    run_moe = moe_runner.fused_moe_runner.run_moe_min_latency if min_latency_mode else moe_runner.fused_moe_runner.run_moe
    output = run_moe(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        quant_scales,
        input_sf,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        min_latency_mode,
        [gemm_tactic_1, gemm_tactic_2],
    )

    return output if min_latency_mode else [output]


@torch.library.register_fake("trtllm::fused_moe")
def _(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4a8_group_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
):
    seq_len = input.shape[0]
    hidden_size = fc2_expert_weights.shape[1]

    if min_latency_mode:
        num_experts_on_rank = fc2_expert_weights.shape[0]
        output_shape = [seq_len * num_experts_on_rank, hidden_size]
        experts_to_token_score_shape = [num_experts_on_rank, seq_len]
        active_expert_global_ids_shape = [num_experts_on_rank]
        return [
            input.new_empty(output_shape, dtype=output_dtype),
            input.new_empty([1], dtype=torch.int32),
            input.new_empty(experts_to_token_score_shape, dtype=torch.float32),
            input.new_empty(active_expert_global_ids_shape, dtype=torch.int32),
        ]
    else:
        return [input.new_empty([seq_len, hidden_size], dtype=output_dtype)]


class NVFP4GemmRunner(TunableRunner):
    runner_dict = dict()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, get_last_power_of_2_num_tokens_buckets,
        last_positive_power_of_2), ),
                                 constraint_specs=(ConstraintSpec(
                                     2, 0, fp4_scale_infer_shape), ))

    def __init__(
        self,
        sf_use_ue8m0: bool,
        to_userbuffers: bool,
        output_dtype: torch.dtype,
    ):
        self.sf_use_ue8m0 = sf_use_ue8m0
        self.output_dtype = output_dtype
        self.to_userbuffers = to_userbuffers
        if output_dtype not in NVFP4GemmRunner.runner_dict:
            NVFP4GemmRunner.runner_dict[
                output_dtype] = torch.classes.trtllm.FP4GemmRunner(output_dtype)
        self.nvfp4_gemm_runner = NVFP4GemmRunner.runner_dict[output_dtype]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        return list(range(self.nvfp4_gemm_runner.get_num_configs()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:
        mat1, mat2, mat1_scale, mat2_scale, global_scale = inputs
        return self.nvfp4_gemm_runner.run_gemm(
            mat1,
            mat2,
            mat1_scale,
            mat2_scale,
            global_scale,
            self.sf_use_ue8m0,
            self.to_userbuffers,
            tactic,
        )


@torch.library.custom_op("trtllm::nvfp4_gemm", mutates_args=())
def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    sf_use_ue8m0: bool,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    # allocate workspace for profiling
    nvfp4_gemm_runner = NVFP4GemmRunner(sf_use_ue8m0, to_userbuffers,
                                        output_dtype)

    _, best_tactic = tuner.choose_one(
        "trtllm::nvfp4_gemm::gemm",
        [nvfp4_gemm_runner],
        NVFP4GemmRunner.tuning_config,
        [act_fp4, weight, act_sf, weight_scale, alpha],
    )

    return nvfp4_gemm_runner(
        inputs=[act_fp4, weight, act_sf, weight_scale, alpha],
        tactic=best_tactic)


@nvfp4_gemm.register_fake
def _(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    sf_use_ue8m0: bool,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:
    return act_fp4.new_empty((act_fp4.size(0), weight.size(0)),
                             dtype=output_dtype)


class FP8BatchedGemmRunner(TunableRunner):
    runner_dict = dict()
    tuning_config = None

    def __init__(self, output_dtype: torch.dtype, use_deep_seek_fp8: bool,
                 low_latency_kernel: bool, tile_size: int,
                 epilogue_tile_m: int):

        self.output_dtype = output_dtype
        self.use_deep_seek_fp8 = use_deep_seek_fp8
        self.low_latency_kernel = low_latency_kernel
        self.tile_size = tile_size
        self.epilogue_tile_m = epilogue_tile_m
        FP8BatchedGemmRunner.tuning_config = FP8BatchedGemmRunner.get_tuning_config(
            use_deep_seek_fp8, tile_size)

        instance_key = (output_dtype, use_deep_seek_fp8, low_latency_kernel,
                        tile_size, epilogue_tile_m)

        if instance_key not in FP8BatchedGemmRunner.runner_dict:
            FP8BatchedGemmRunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8BatchedGemmRunner(
                    output_dtype, use_deep_seek_fp8, low_latency_kernel,
                    tile_size, epilogue_tile_m)

        self.kernel_runner = FP8BatchedGemmRunner.runner_dict[instance_key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the batched GEMM operation with the given inputs and tactic.
        """

        mat1, mat2, dq_sfs_a, dq_sfs_b, scale_c = inputs

        out_tensors = self.kernel_runner.run_batched_gemm(
            mat1,
            mat2,
            dq_sfs_a,
            dq_sfs_b,
            scale_c,
            tactic,
        )

        return out_tensors

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:

        mat1, mat2, _, _, _ = inputs

        b = mat1.shape[0]
        m = mat1.shape[1]
        n = mat2.shape[1]
        k = mat1.shape[2]

        tactics = self.kernel_runner.get_valid_configs(b, m, n, k)

        return tactics

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        """Get the dynamic tensor specs for use with the AutoTuner."""

        # These indices correspond to the 0th input tensor and it's first dimension
        # i.e. we are tuning M where the first input tensor is of shape [B, M, K]

        MAT1_IDX = 0
        TUNED_DIM = 1

        # Starting at 8 as M % tile size == 0 is required
        m_values = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
        round_rule = last_positive_power_of_2

        specs = (DynamicTensorSpec(MAT1_IDX, TUNED_DIM, m_values, round_rule), )

        return specs

    @classmethod
    def get_constraint_specs(cls, use_deep_seek_fp8: bool,
                             tile_size: int) -> Tuple[ConstraintSpec, ...]:
        """Get the constraint specs for the dynamic tensors for use with the AutoTuner.
        """

        # When using deepseek fp8, the dq_sfs_a and dq_sfs_b tensors are expected to
        # have specific dimensions. As we are only tuning M, we need only constrain
        # dimension 1 of dq_sfs_a
        if not use_deep_seek_fp8:
            constraint_dq_sfs_a = ()
        else:

            def _constrain_dq_sfs_a_dim1(shapes: Tuple[torch.Size]) -> int:
                b = shapes[0][0]
                m = shapes[0][1]

                m_padded = (m + tile_size - 1) // tile_size
                result = m_padded * tile_size * b

                return result

            SFS_A_IDX = 2
            CONSTRAINED_DIM = 1

            constraint_dq_sfs_a = (ConstraintSpec(SFS_A_IDX, CONSTRAINED_DIM,
                                                  _constrain_dq_sfs_a_dim1), )

        return constraint_dq_sfs_a

    @classmethod
    @lru_cache(maxsize=None)
    def get_tuning_config(cls, use_deep_seek_fp8: bool,
                          tile_size: int) -> TuningConfig:
        """Get the tuning configuration for the AutoTuner."""

        dynamic_tensor_specs = cls.get_dynamic_tensor_specs()
        constraint_specs = cls.get_constraint_specs(use_deep_seek_fp8,
                                                    tile_size)

        tuning_config = TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs,
                                     constraint_specs=constraint_specs)

        return tuning_config


@torch.library.custom_op("trtllm::fp8_batched_gemm_trtllmgen", mutates_args=())
def fp8_batched_gemm_trtllmgen(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    tile_size: int,
    use_deep_seek_fp8: Optional[bool] = False,
    low_latency: Optional[bool] = False,
    epilogue_tile_m: Optional[int] = 0,
    dq_sfs_a: Optional[torch.Tensor] = None,
    dq_sfs_b: Optional[torch.Tensor] = None,
    scale_c: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.half
) -> Tuple[torch.Tensor, torch.Tensor]:

    kernel_runner = FP8BatchedGemmRunner(output_dtype=out_dtype,
                                         use_deep_seek_fp8=use_deep_seek_fp8,
                                         low_latency_kernel=low_latency,
                                         tile_size=tile_size,
                                         epilogue_tile_m=epilogue_tile_m)

    tuner = AutoTuner.get()

    inputs = [mat1, mat2, dq_sfs_a, dq_sfs_b, scale_c]

    _, best_tactic = tuner.choose_one(
        "trtllm::fp8_batched_gemm_trtllmgen::batched_gemm",
        [kernel_runner],
        FP8BatchedGemmRunner.tuning_config,
        inputs,
    )

    return kernel_runner(
        inputs=inputs,
        tactic=best_tactic,
    )


# Allows the tunable TRTLLM-Gen FP8 batched GEMM to be
# used with torch.compile
@fp8_batched_gemm_trtllmgen.register_fake
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    tile_size: int,
    use_deep_seek_fp8: Optional[bool] = False,
    low_latency: Optional[bool] = False,
    epilogue_tile_m: Optional[int] = 0,
    dq_sfs_a: Optional[torch.Tensor] = None,
    dq_sfs_b: Optional[torch.Tensor] = None,
    scale_c: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    b = mat1.size(0)
    m = mat1.size(1)
    n = mat2.size(1)

    fake_out = mat1.new_empty((b, m, n), dtype=out_dtype)

    if use_deep_seek_fp8:
        ds_fp8_quant_block_size = 128
        dim0_size = n // ds_fp8_quant_block_size
        dim1_size = b * m
        fake_dq_sfs_c = torch.empty((dim0_size, dim1_size), dtype=torch.float32)
    else:
        fake_dq_sfs_c = torch.empty((0, 0), dtype=torch.float32)

    return (fake_out, fake_dq_sfs_c)


@torch.library.custom_op("trtllm::attention", mutates_args=())
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    workspace: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    context_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    cache_indirection: Optional[torch.Tensor],
    kv_scale_orig_quant: Optional[torch.Tensor],
    kv_scale_quant_orig: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    rotary_inv_freq: Optional[torch.Tensor],
    rotary_cos_sin: Optional[torch.Tensor],
    latent_cache: Optional[torch.Tensor],
    q_pe: Optional[torch.Tensor],
    block_ids_per_seq: Optional[torch.Tensor],
    is_fused_qkv: bool,
    update_kv_cache: bool,
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: Optional[int],
    max_num_requests: int,
    max_context_length: int,
    attention_window_size: int,
    sink_token_length: int,
    beam_width: int,
    mask_type: int,
    quant_mode: int,
    q_scaling: float,
    position_embedding_type: int,
    rotary_embedding_dim: int,
    rotary_embedding_base: float,
    rotary_embedding_scale_type: int,
    rotary_embedding_scale: float,
    rotary_embedding_short_m_scale: float,
    rotary_embedding_long_m_scale: float,
    rotary_embedding_max_positions: int,
    rotary_embedding_original_max_positions: int,
    use_paged_context_fmha: bool,
    attention_input_type: Optional[int],
    is_mla_enable: bool,
    q_lora_rank: Optional[int],
    kv_lora_rank: Optional[int],
    qk_nope_head_dim: Optional[int],
    qk_rope_head_dim: Optional[int],
    v_head_dim: Optional[int],
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    mla_context_paged_kv: Optional[torch.Tensor],
    mla_context_kv_cache_block_offsets: Optional[torch.Tensor],
    attention_chunk_size: Optional[int],
) -> List[torch.Tensor]:
    num_tokens = q.size(0)
    attention_input_type = (AttentionInputType(attention_input_type)
                            if attention_input_type is not None else
                            AttentionInputType.mixed)
    is_gen_only = attention_input_type == AttentionInputType.generation_only
    v_head_size = head_size if not is_mla_enable else kv_lora_rank if is_gen_only else v_head_dim
    if out_dtype is None:
        out_dtype = q.dtype

    if out_dtype == torch.uint8:
        num_nvfp4_elements_per_container = 2
        scaling_vector_size = 16
        size_per_token = num_heads * v_head_size
        output_act = q.new_empty(
            (num_tokens, size_per_token // num_nvfp4_elements_per_container),
            dtype=torch.uint8)
        # Create a sf (scaling factors) tensor for NVFP4 (use INT8 as the container dtype).
        output_sf = q.new_empty(compute_swizzled_sf_shape(
            num_tokens, size_per_token // scaling_vector_size),
                                dtype=torch.uint8)
    else:
        output_act = q.new_empty((num_tokens, num_heads * v_head_size),
                                 dtype=out_dtype)
        # NOTE(tizheng): Does this introduce overhead?
        output_sf = torch.empty(())  # Create a placeholder, which is not used.

    torch.ops.trtllm.attention_inplace(
        q, k, v, output_act, output_sf, out_dtype, workspace, sequence_length,
        host_past_key_value_lengths, context_lengths, host_context_lengths,
        host_request_types, kv_cache_block_offsets, host_kv_cache_block_offsets,
        host_kv_cache_pool_pointers, host_kv_cache_pool_mapping,
        cache_indirection, kv_scale_orig_quant, kv_scale_quant_orig, out_scale,
        rotary_inv_freq, rotary_cos_sin, latent_cache, q_pe, block_ids_per_seq,
        is_fused_qkv, update_kv_cache, predicted_tokens_per_seq, layer_idx,
        num_heads, num_kv_heads, head_size, tokens_per_block, max_num_requests,
        max_context_length, attention_window_size, sink_token_length,
        beam_width, mask_type, quant_mode, q_scaling, position_embedding_type,
        rotary_embedding_dim, rotary_embedding_base,
        rotary_embedding_scale_type, rotary_embedding_scale,
        rotary_embedding_short_m_scale, rotary_embedding_long_m_scale,
        rotary_embedding_max_positions, rotary_embedding_original_max_positions,
        use_paged_context_fmha, attention_input_type, is_mla_enable,
        q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        v_head_dim, mrope_rotary_cos_sin, mrope_position_deltas,
        mla_context_paged_kv, mla_context_kv_cache_block_offsets,
        attention_chunk_size)
    return output_act, output_sf


@attention.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    workspace: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    context_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    cache_indirection: Optional[torch.Tensor],
    kv_scale_orig_quant: Optional[torch.Tensor],
    kv_scale_quant_orig: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    rotary_inv_freq: Optional[torch.Tensor],
    rotary_cos_sin: Optional[torch.Tensor],
    latent_cache: Optional[torch.Tensor],
    q_pe: Optional[torch.Tensor],
    block_ids_per_seq: Optional[torch.Tensor],
    is_fused_qkv: bool,
    update_kv_cache: bool,
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: Optional[int],
    max_num_requests: int,
    max_context_length: int,
    attention_window_size: int,
    sink_token_length: int,
    beam_width: int,
    mask_type: int,
    quant_mode: int,
    q_scaling: float,
    position_embedding_type: int,
    rotary_embedding_dim: int,
    rotary_embedding_base: float,
    rotary_embedding_scale_type: int,
    rotary_embedding_scale: float,
    rotary_embedding_short_m_scale: float,
    rotary_embedding_long_m_scale: float,
    rotary_embedding_max_positions: int,
    rotary_embedding_original_max_positions: int,
    use_paged_context_fmha: bool,
    attention_input_type: Optional[int],
    is_mla_enable: bool,
    q_lora_rank: Optional[int],
    kv_lora_rank: Optional[int],
    qk_nope_head_dim: Optional[int],
    qk_rope_head_dim: Optional[int],
    v_head_dim: Optional[int],
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    mla_context_paged_kv: Optional[torch.Tensor],
    mla_context_kv_cache_block_offsets: Optional[torch.Tensor],
    attention_chunk_size: Optional[int],
) -> List[torch.Tensor]:
    num_tokens = q.size(0)
    attention_input_type = (AttentionInputType(attention_input_type)
                            if attention_input_type is not None else
                            AttentionInputType.mixed)
    if out_dtype is None:
        out_dtype = q.dtype
    is_gen_only = attention_input_type == AttentionInputType.generation_only
    v_head_size = head_size if not is_mla_enable else kv_lora_rank if is_gen_only else v_head_dim

    if out_dtype == torch.uint8:
        num_nvfp4_elements_per_container = 2
        scaling_vector_size = 16
        size_per_token = num_heads * v_head_size
        output_act = q.new_empty(
            (num_tokens, size_per_token // num_nvfp4_elements_per_container),
            dtype=torch.uint8)
        # Create a sf (scaling factors) tensor for NVFP4 (use INT8 as the container dtype).
        output_sf = q.new_empty(compute_swizzled_sf_shape(
            num_tokens, size_per_token // scaling_vector_size),
                                dtype=torch.uint8)
    else:
        output_act = q.new_empty((num_tokens, num_heads * v_head_size),
                                 dtype=out_dtype)
        output_sf = torch.empty(())  # Create a placeholder, which is not used.

    return output_act, output_sf
