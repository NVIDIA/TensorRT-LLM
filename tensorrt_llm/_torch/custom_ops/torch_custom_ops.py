from functools import lru_cache
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import nvtx
import torch
from cuda import cuda
from cutlass.cute.runtime import from_dlpack

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
# from ..attention_backend.interface import AttentionInputType
# from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
#                          OptimizationProfile, TunableRunner, TuningConfig)
# from ..utils import (compute_swizzled_sf_shape, fp4_scale_infer_shape,
#                      get_last_power_of_2_num_tokens_buckets,
#                      last_positive_power_of_2)
from tensorrt_llm._torch.autotuner import (AutoTuner, ConstraintSpec,
                                           DynamicTensorSpec,
                                           OptimizationProfile, TunableRunner,
                                           TuningConfig)
# from tensorrt_llm._torch.custom_ops.cute_dsl_kernels.blackwell.blockwise_gemm import \
#     BlockwiseGemmKernel
from tensorrt_llm._torch.custom_ops.cute_dsl_kernels.blackwell.blockwise_gemm_release import \
    BlockwiseGemmKernel
# from tensorrt_llm._torch.custom_ops.cute_dsl_kernels.blackwell.continuous_offset_grouped_gemm import \
#     BlockwiseContiguousGroupedGemmKernel
from tensorrt_llm._torch.custom_ops.cute_dsl_kernels.blackwell.continuous_offset_grouped_gemm_release import \
    BlockwiseContiguousGroupedGemmKernel
from tensorrt_llm._torch.utils import (fp4_scale_infer_shape,
                                       get_last_power_of_2_num_tokens_buckets,
                                       last_positive_power_of_2)
from tensorrt_llm._utils import get_sm_version


# Used to WAR an issue in torch.bmm that it would break the graph when the out is not contiguous.
@torch.library.custom_op("trtllm::bmm_out", mutates_args=("out", ))
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(a, b, out=out)


class MoERunner(TunableRunner):
    # avoid overhead of creating a new runner in forward pass
    runner_dict = dict()
    tuning_config = TuningConfig(dynamic_tensor_specs=(
        DynamicTensorSpec(0, 0, get_last_power_of_2_num_tokens_buckets(8192),
                          lambda x: min(last_positive_power_of_2(x), 8192)), ))

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
        use_w4_group_scaling: bool,
        use_mxfp8_act_scaling: bool,
        min_latency_mode: bool,
        use_fused_finalize: bool,
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
        # The best tactic is estimated as if alltoall is disabled
        self.enable_alltoall = False
        self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale
        self.use_w4_group_scaling = use_w4_group_scaling
        self.use_mxfp8_act_scaling = use_mxfp8_act_scaling
        self.min_latency_mode = min_latency_mode
        self.use_fused_finalize = use_fused_finalize

        instance_key = (x_dtype, weight_dtype, output_dtype,
                        use_deepseek_fp8_block_scale, use_w4_group_scaling,
                        use_mxfp8_act_scaling)

        if instance_key not in MoERunner.runner_dict:
            MoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FusedMoeRunner(
                    x_dtype, weight_dtype, output_dtype,
                    use_deepseek_fp8_block_scale, use_w4_group_scaling,
                    use_mxfp8_act_scaling, use_fused_finalize)
        self.fused_moe_runner = MoERunner.runner_dict[instance_key]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        return range(self.fused_moe_runner.get_tactic_num())

    def forward(
        self,
        inputs: List[torch.Tensor],
        gemm_idx: int = 0,
        tactic: int = -1,
        do_preparation: bool = False,
    ):
        x, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases = inputs
        self.fused_moe_runner.run_gemm_profile(
            x,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            self.top_k,
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
            self.cluster_size,
            self.cluster_rank,
            self.enable_alltoall,
            self.min_latency_mode,
            gemm_idx,
            tactic,
            do_preparation,
        )

    @classmethod
    @lru_cache(maxsize=None)
    def refine_tuning_config(cls, tune_max_num_tokens: int):
        cls.tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets(
                    tune_max_num_tokens), lambda x: min(
                        last_positive_power_of_2(x), tune_max_num_tokens)), ))


@torch.library.custom_op("trtllm::fused_moe", mutates_args=())
def fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,
    fc2_expert_biases: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    swizzled_input_sf: bool = True,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_fused_finalize: bool = True,
    tune_max_num_tokens: int = 8192,
    tuner_num_tokens: Optional[int] = None,
    tuner_top_k: Optional[int] = None,
) -> List[torch.Tensor]:

    tuner = AutoTuner.get()
    MoERunner.refine_tuning_config(tune_max_num_tokens)

    # Only the non-alltoall case is considered for profiling in the warmup phase.
    # Therefore, to get the correct tactics during the actual inference, the inputs to the tuner should be the same as when not using alltoall.
    if enable_alltoall:
        assert tuner_num_tokens is not None
        assert tuner_top_k is not None
        tuner_input = input[:tuner_num_tokens]
    else:
        assert tuner_num_tokens is None
        assert tuner_top_k is None
        tuner_input = input
        tuner_top_k = token_selected_experts.size(1)

    # allocate workspace for profiling
    moe_runner = MoERunner(
        x_dtype=input.dtype,
        weight_dtype=fc1_expert_weights.dtype,
        output_dtype=output_dtype,
        top_k=tuner_top_k,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cluster_size=cluster_size,
        cluster_rank=cluster_rank,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4_group_scaling=use_w4_group_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        min_latency_mode=min_latency_mode,
        use_fused_finalize=use_fused_finalize,
    )

    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1",
        [moe_runner],
        MoERunner.tuning_config,
        [
            tuner_input, fc1_expert_weights, fc1_expert_biases,
            fc2_expert_weights, fc2_expert_biases
        ],
        gemm_idx=1,
    )

    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2",
        [moe_runner],
        MoERunner.tuning_config,
        [
            tuner_input, fc1_expert_weights, fc1_expert_biases,
            fc2_expert_weights, fc2_expert_biases
        ],
        gemm_idx=2,
    )

    run_moe = moe_runner.fused_moe_runner.run_moe_min_latency if min_latency_mode else moe_runner.fused_moe_runner.run_moe
    output = run_moe(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc1_expert_biases,
        fc2_expert_weights,
        fc2_expert_biases,
        quant_scales,
        input_sf,
        swizzled_input_sf,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        cluster_size,
        cluster_rank,
        enable_alltoall,
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
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,
    fc2_expert_biases: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    swizzled_input_sf: bool = True,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_fused_finalize: bool = True,
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


class FP8RowwiseGemmRunner(TunableRunner):
    runner_dict = dict()
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(
            0, 0, get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2), ),
        constraint_specs=(
            ConstraintSpec(2, 0, lambda shapes: shapes[0][0]),
            ConstraintSpec(3, 0, lambda shapes: shapes[1][0]),
        ))

    def __init__(
        self,
        to_userbuffers: bool,
        output_dtype: torch.dtype,
    ):
        self.to_userbuffers = to_userbuffers
        self.output_dtype = output_dtype
        instance_key = (output_dtype, )
        if instance_key not in FP8RowwiseGemmRunner.runner_dict:
            FP8RowwiseGemmRunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8RowwiseGemmRunner(
                    output_dtype)
        self.fp8_rowwise_gemm_runner = FP8RowwiseGemmRunner.runner_dict[
            instance_key]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        return list(range(self.fp8_rowwise_gemm_runner.get_num_configs()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:
        mat1, mat2, mat1_scale, mat2_scale = inputs
        return self.fp8_rowwise_gemm_runner.run_gemm(
            mat1,
            mat2,
            mat1_scale,
            mat2_scale,
            self.to_userbuffers,
            tactic,
        )


@torch.library.custom_op("trtllm::fp8_rowwise_gemm", mutates_args=())
def fp8_rowwise_gemm(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    # allocate workspace for profiling
    fp8_rowwise_gemm_runner = FP8RowwiseGemmRunner(to_userbuffers, output_dtype)

    _, best_tactic = tuner.choose_one(
        "trtllm::fp8_rowwise_gemm::gemm",
        [fp8_rowwise_gemm_runner],
        FP8RowwiseGemmRunner.tuning_config,
        [act, weight, act_scale, weight_scale],
    )

    return fp8_rowwise_gemm_runner(
        inputs=[act, weight, act_scale, weight_scale], tactic=best_tactic)


@fp8_rowwise_gemm.register_fake
def _(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:
    return act.new_empty((act.size(0), weight.size(0)), dtype=output_dtype)


class FP4GemmRunner(TunableRunner):
    runner_dict = dict()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, get_last_power_of_2_num_tokens_buckets,
        last_positive_power_of_2), ),
                                 constraint_specs=(ConstraintSpec(
                                     2, 0, fp4_scale_infer_shape), ))

    def __init__(
        self,
        fp4_gemm_type: fp4_utils.FP4GemmType,
        to_userbuffers: bool,
        output_dtype: torch.dtype,
    ):
        self.fp4_gemm_type = fp4_gemm_type
        self.output_dtype = output_dtype
        self.to_userbuffers = to_userbuffers
        instance_key = (output_dtype, int(fp4_gemm_type))
        if instance_key not in FP4GemmRunner.runner_dict:
            FP4GemmRunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP4GemmRunner(
                    output_dtype, int(fp4_gemm_type))
        self.fp4_gemm_runner = FP4GemmRunner.runner_dict[instance_key]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        return list(range(self.fp4_gemm_runner.get_num_configs()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:
        mat1, mat2, mat1_scale, mat2_scale, global_scale = inputs
        return self.fp4_gemm_runner.run_gemm(
            mat1,
            mat2,
            mat1_scale,
            mat2_scale,
            global_scale,
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
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    # allocate workspace for profiling
    nvfp4_gemm_runner = FP4GemmRunner(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                                      to_userbuffers, output_dtype)

    _, best_tactic = tuner.choose_one(
        "trtllm::fp4_gemm::gemm",
        [nvfp4_gemm_runner],
        FP4GemmRunner.tuning_config,
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


@torch.library.custom_op("trtllm::w4a8_mxfp4_fp8_gemm", mutates_args=())
def w4a8_mxfp4_fp8_gemm(
    act_fp8: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    # allocate workspace for profiling
    w4a8_mxfp4_fp8_gemm_runner = FP4GemmRunner(
        fp4_utils.FP4GemmType.W4A8_MXFP4_MXFP8, to_userbuffers, output_dtype)

    _, best_tactic = tuner.choose_one(
        "trtllm::w4a8_mxfp4_fp8_gemm::gemm",
        [w4a8_mxfp4_fp8_gemm_runner],
        FP4GemmRunner.tuning_config,
        [act_fp8, weight, act_sf, weight_scale, alpha],
    )

    return w4a8_mxfp4_fp8_gemm_runner(
        inputs=[act_fp8, weight, act_sf, weight_scale, alpha],
        tactic=best_tactic)


@w4a8_mxfp4_fp8_gemm.register_fake
def _(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:
    return act_fp8.new_empty((act_fp8.size(0), weight.size(0)),
                             dtype=output_dtype)


class WeightOnlyQuantGemmRunner(TunableRunner):
    runner_dict = dict()
    tuning_config = TuningConfig(dynamic_tensor_specs=(
        DynamicTensorSpec(0, 0, get_last_power_of_2_num_tokens_buckets,
                          last_positive_power_of_2), ))

    def __init__(
        self,
        activation_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        output_dtype: torch.dtype,
        to_userbuffers: bool,
    ):
        self.output_dtype = output_dtype
        self.to_userbuffers = to_userbuffers
        instance_key = (activation_dtype, weight_dtype)
        if instance_key not in WeightOnlyQuantGemmRunner.runner_dict:
            WeightOnlyQuantGemmRunner.runner_dict[
                instance_key] = torch.classes.trtllm.WeightOnlyQuantGemmRunner(
                    activation_dtype, weight_dtype)
        self.weight_only_quant_gemm_runner = WeightOnlyQuantGemmRunner.runner_dict[
            instance_key]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        return list(range(self.weight_only_quant_gemm_runner.get_num_configs()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:
        activation, weight, weight_scale = inputs
        return self.weight_only_quant_gemm_runner.run_gemm(
            activation,
            weight,
            weight_scale,
            tactic,
            self.to_userbuffers,
            self.output_dtype,
        )


@torch.library.custom_op("trtllm::weight_only_quant_gemm", mutates_args=())
def weight_only_quant_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    # allocate workspace for profiling
    weight_only_quant_gemm_runner = WeightOnlyQuantGemmRunner(
        activation.dtype, weight_dtype, output_dtype, to_userbuffers)

    _, best_tactic = tuner.choose_one(
        "trtllm::weight_only_quant_gemm::gemm",
        [weight_only_quant_gemm_runner],
        WeightOnlyQuantGemmRunner.tuning_config,
        [activation, weight, weight_scale],
    )

    return weight_only_quant_gemm_runner(
        inputs=[activation, weight, weight_scale], tactic=best_tactic)


@weight_only_quant_gemm.register_fake
def _(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_type: torch.dtype,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype = None,
    to_userbuffers: bool = False,
) -> torch.Tensor:
    dtype = output_dtype if output_dtype is not None else activation.dtype
    return activation.new_empty((activation.size(0), weight.size(1)),
                                dtype=dtype)


class FinegrainedMixedDtypeGemm(TunableRunner):
    _runner_dict = dict()
    MAX_SUPPORTED_SM_VERSION = 90

    def __init__(self, activation_dtype: torch.dtype, output_dtype: torch.dtype,
                 quant_mode: int):
        instance_key = (activation_dtype, output_dtype, quant_mode)
        if instance_key not in FinegrainedMixedDtypeGemm._runner_dict:
            FinegrainedMixedDtypeGemm._runner_dict[
                instance_key] = torch.classes.trtllm.finegrainedMixedDtypeGemmRunner(
                    activation_dtype, output_dtype, quant_mode)
        self._finegrained_mixed_dtype_gemm_runner = FinegrainedMixedDtypeGemm._runner_dict[
            instance_key]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        return list(
            range(self._finegrained_mixed_dtype_gemm_runner.get_num_configs()))

    def forward(self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs) -> torch.Tensor:

        if get_sm_version() > self.MAX_SUPPORTED_SM_VERSION:
            raise ValueError(
                f"SM version {get_sm_version()} is not supported for W4A16 GEMM"
            )

        activation, weights_packed, scales = inputs

        alpha = 1.0 if kwargs.get("alpha") is None else kwargs["alpha"]

        return self._finegrained_mixed_dtype_gemm_runner.run_gemm(
            activation, weights_packed, scales, kwargs["group_size"], tactic,
            kwargs["bias"], kwargs["zeros"], alpha)


@torch.library.custom_op("trtllm::finegrained_mixed_dtype_gemm",
                         mutates_args=())
def finegrained_mixed_dtype_gemm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        has_zero_point: bool,
        output_dtype: torch.dtype,
        alpha: Optional[float] = None,
        bias: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None) -> torch.Tensor:

    assert not has_zero_point or zeros is not None, "Expected 'zeros' tensor when has_zero_point is True"

    tuner = AutoTuner.get()

    tuning_config = TuningConfig(dynamic_tensor_specs=(
        # For tensor index 0 (input A), tune dimension 0 (M dimension)
        DynamicTensorSpec(0, 0, (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32,
                                 16, 8, 4, 2, 1), last_positive_power_of_2), ))

    # NOTE: qunant_mode equals 0 it means we use scale only (FINEGRAINED_SCALE_ONLY), zeros is not used, else we use scale and zero point
    quant_mode = 1 if has_zero_point else 0
    if quant_mode == 0:
        assert zeros is None, "When quant_mode is 0 (FINEGRAINED_SCALE_ONLY), zeros must be None"

    finegrained_mixed_dtype_gemm_runner = FinegrainedMixedDtypeGemm(
        input.dtype, output_dtype, quant_mode)

    kwargs = {
        "group_size": group_size,
        "zeros": zeros,
        "bias": bias,
        "alpha": alpha
    }

    _, best_tactic = tuner.choose_one(
        "trtllm::finegrained_mixed_dtype_gemm::gemm",
        [finegrained_mixed_dtype_gemm_runner], tuning_config,
        [input, weight, scales], **kwargs)

    return finegrained_mixed_dtype_gemm_runner(inputs=[input, weight, scales],
                                               tactic=best_tactic,
                                               **kwargs)


@finegrained_mixed_dtype_gemm.register_fake
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    has_zero_point: bool,
    output_dtype: torch.dtype,
    alpha: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    zeros: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # For a typical GEMM: input [M, K] @ weight [K, N] -> output [M, N]
    # Weight is typically packed, so we need to infer the output dimension
    M = input.size(0)
    # Assuming weight is packed and the output dimension can be inferred from weight.size(1)
    N = weight.size(1) if weight.dim() > 1 else weight.size(0)
    return input.new_empty((M, N), dtype=output_dtype)


def get_event(event_idx: int):
    from ..utils import get_model_extra_attrs
    extra_attrs = get_model_extra_attrs()
    assert "events" in extra_attrs, "Missing Event Book"
    return extra_attrs["events"]()[event_idx]


def get_stream(stream_id: int):
    from ..utils import get_model_extra_attrs
    extra_attrs = get_model_extra_attrs()
    if stream_id == 0:
        return extra_attrs["global_stream"]
    assert "aux_streams" in extra_attrs, "Missing Aux Streams"
    return extra_attrs["aux_streams"]()[stream_id - 1]


@torch.library.custom_op("trtllm::set_stream", mutates_args=())
def set_stream(stream_id: int) -> None:
    stream = get_stream(stream_id)
    assert stream is not None
    torch.cuda.set_stream(stream)


@torch.library.custom_op("trtllm::record_event", mutates_args=())
def record_event(event_idx: int) -> None:
    event = get_event(event_idx)
    event.record()


@torch.library.custom_op("trtllm::wait_event", mutates_args=())
def wait_event(event_idx: int) -> None:
    event = get_event(event_idx)
    event.wait()


@torch.library.custom_op("trtllm::record_stream", mutates_args=())
def record_stream(tensor: torch.Tensor, stream_id: int) -> None:
    stream = get_stream(stream_id)
    assert stream is not None
    tensor.record_stream(stream)


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@cute.jit
def permute(tensor, perm):
    """
    General dimension permutation function.
    Args:
      tensor: Input tensor.
      perm: A tuple indicating the new order of dimensions, e.g., (1,2,0) means permuting dimensions 0,1,2 to 1,2,0.
    """
    layout = tensor.layout
    shapes = cute.shape(layout)  # Get original shape
    strides = layout.stride  # Get original strides

    # Rearrange shape and stride according to perm
    new_shapes = tuple(shapes[p] for p in perm)
    new_strides = tuple(strides[p] for p in perm)

    # Create new layout and tensor
    new_layout = cute.make_layout(new_shapes, stride=new_strides)
    return cute.make_tensor(tensor.iterator, new_layout)


@cute.jit
def append_ones_wrapper(a: cute.Tensor):
    # return cute.append_ones(tensor)
    a_layout = a.layout
    a_layout = cute.append(a_layout,
                           cute.make_layout(1, stride=1),
                           up_to_rank=3)
    new_a = cute.make_tensor(a.iterator, a_layout)
    return new_a


class CuteDSLFp8BlackwellLinear(TunableRunner):
    kernel_dict = dict()

    def __init__(self):
        super().__init__()

    def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            use_2cta_instrs: bool = False,
            mma_tiler_mn: Tuple[int, int] = (128, 128),
            cluster_shape_mn: Tuple[int, int] = (1, 1),
            **kwargs,
    ) -> List[int]:
        m = inputs[0].shape[0]
        n = inputs[1].shape[0]
        k = inputs[0].shape[1]
        l = 1
        # m,k
        a_major = "k"
        # n, k
        b_major = "k"
        # m, n
        c_major = "n"
        is_valid = BlockwiseGemmKernel.can_implement(
            cutlass.Float8E4M3FN,  # ab_dtype,
            cutlass.Float32,  # acc_dtype,
            cutlass.BFloat16,  # c_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
        )
        if is_valid:
            return [0]
        else:
            return []

    def forward(
        self,
        inputs: List[torch.Tensor],
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        tactic: int = -1,
    ) -> torch.Tensor:
        """Performs fp8 blockwise (deepgemm like) operation using CuTe DSL.

        :param a (inputs[0]): Input tensor of shape (m, k)
        :type a: torch.Tensor, type: fp8
        :param b (inputs[1]): Weight tensor of shape (n, k)
        :type b: torch.Tensor, type: fp8
        :param a_sf (inputs[2]): Input scale tensor of shape (k//128, m).
        :type a_sf: torch.Tensor, type: fp32
        :param b_sf (inputs[3]): Weight scale tensor of shape (n, k//128)
        :type b_sf: torch.Tensor, type: fp32

        :return: Output tensor of shape (m, n)
        :rtype: torch.Tensor, type: bf16
        """
        # before opt
        """
        a, b, a_sf, b_sf = inputs
        m, n, k = a.shape[0], b.shape[0], a.shape[1]
        w_n, w_k = b_sf.shape[0], b_sf.shape[1]
        c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")

        # torch_tensor -> cute.tensor
        a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
        b_tmp = b.as_strided((n, k, 1), (k, 1, n * k)).view(torch.uint8)
        c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))
        weight_scale_tmp = b_sf.as_strided((w_n, w_k, 1), (w_k, 1, w_n * w_k))
        # [xx, m]
        input_scale_tmp = a_sf.permute(1, 0).as_strided(
            (m, w_k, 1), (1, m, w_k * m)
        )

        mA = from_dlpack(a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mB = from_dlpack(b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mC = from_dlpack(c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mA.element_type = cutlass.Float8E4M3FN
        mB.element_type = cutlass.Float8E4M3FN

        # TODO: mSFA is column major
        mSFA = from_dlpack(input_scale_tmp, assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        )
        mSFB = from_dlpack(weight_scale_tmp, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        """

        a, b, a_sf, b_sf = inputs
        m, n = a.shape[0], b.shape[0]
        c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")

        a_tmp = a.view(torch.uint8)
        b_tmp = b.view(torch.uint8)
        mA = from_dlpack(a_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mB = from_dlpack(b_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mC = from_dlpack(c, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mA.element_type = cutlass.Float8E4M3FN
        mB.element_type = cutlass.Float8E4M3FN

        # TODO: mSFA is column major
        mSFA = from_dlpack(a_sf,
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mSFB = from_dlpack(b_sf,
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)

        # get stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        cache_key = (
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
        )
        if cache_key not in CuteDSLFp8BlackwellLinear.kernel_dict:
            gemm = BlockwiseGemmKernel(
                cutlass.Float32,  # acc_dtype,
                use_2cta_instrs=use_2cta_instrs,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
            )
            # Compute max active clusters on current device
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1])

            @cute.jit
            def mm_permute_wrapper(
                a: cute.Tensor,
                b: cute.Tensor,
                c: cute.Tensor,
                a_sf: cute.Tensor,
                b_sf: cute.Tensor,
                max_active_clusters: cutlass.Constexpr,
                stream: cuda.CUstream,
            ):
                a = cute.append_ones(a)
                b = cute.append_ones(b)
                c = cute.append_ones(c)
                b_sf = cute.append_ones(b_sf)
                a_sf = permute(a_sf, (1, 0))
                a_sf = cute.append_ones(a_sf)
                gemm(a, b, c, a_sf, b_sf, max_active_clusters, stream)

            # max_active_clusters = 148
            compiled_gemm = cute.compile(
                # gemm,
                mm_permute_wrapper,
                mA,
                mB,
                mC,
                mSFA,
                mSFB,
                max_active_clusters,
                stream,
            )
            CuteDSLFp8BlackwellLinear.kernel_dict[cache_key] = compiled_gemm
        else:
            compiled_gemm = CuteDSLFp8BlackwellLinear.kernel_dict[cache_key]

        # launch gemm kernel
        compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)
        return c


# a/b: fp8, scale: fp32, output: bf16
@torch.library.custom_op("trtllm::cute_dsl_fp8_gemm_blackwell",
                         mutates_args=(),
                         device_types="cuda")
# @autotuner.tuning_config(
#     name="trtllm::cute_dsl_fp8_gemm_blackwell::gemm",
#     dynamic_tensor_specs=(DynamicTensorSpec(
#         0, 0, get_last_power_of_2_num_tokens_buckets,
#         last_positive_power_of_2), ),
#     constraint_specs=(ConstraintSpec(
#         2, 1, cute_dsl_fp8_linear_scale_infer_shape_blackwell), ),
#     configs={
#         "use_2cta_instrs": [False],
#         "mma_tiler_mn": [(128, 128)],
#         "cluster_shape_mn": [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1),
#                              (4, 4)],
#     },
# )
def cute_dsl_fp8_gemm_blackwell(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # tuner = AutoTuner.get()
    # # allocate workspace for profiling
    # cute_dsl_fp8_gemm_blackwell_runner = CuteDSLFp8BlackwellLinear()
    # _, best_tactic, best_config = tuner.choose_one(
    #     "trtllm::cute_dsl_fp8_gemm_blackwell::gemm",
    #     [cute_dsl_fp8_gemm_blackwell_runner],
    #     [input, weight, input_scale, weight_scale],
    # )
    # return cute_dsl_fp8_gemm_blackwell_runner(
    #     inputs=[input, weight, input_scale, weight_scale],
    #     tactic=best_tactic,
    #     **best_config,
    # )

    cute_dsl_fp8_gemm_blackwell_runner = CuteDSLFp8BlackwellLinear()
    return cute_dsl_fp8_gemm_blackwell_runner(
        inputs=[input, weight, input_scale, weight_scale],
        tactic=0,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )


@torch.library.register_fake("trtllm::cute_dsl_fp8_gemm_blackwell")
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    # [m, k]
    shape = [i for i in mat_a.shape]
    # [n, k]
    shape[-1] = mat_b.shape[-2]
    # output is fixed as bf16
    ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
    return ret


class CuteDSLFp8BlackwellBmm(TunableRunner):
    kernel_dict = dict()

    def __init__(self):
        super().__init__()

    def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            use_2cta_instrs: bool = False,
            mma_tiler_mn: Tuple[int, int] = (128, 128),
            cluster_shape_mn: Tuple[int, int] = (1, 1),
            **kwargs,
    ) -> List[int]:
        # [b, m, k]
        l, m, k = inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]
        # [b, n, k]
        n = inputs[1].shape[1]
        # m,k
        a_major = "k"
        # n, k
        b_major = "k"
        # m, n
        c_major = "n"
        is_valid = BlockwiseGemmKernel.can_implement(
            cutlass.Float8E4M3FN,  # ab_dtype,
            cutlass.Float32,  # acc_dtype,
            cutlass.BFloat16,  # c_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
        )
        if is_valid:
            return [0]
        else:
            return []

    def forward(
            self,
            inputs: List[torch.Tensor],
            use_2cta_instrs: bool = False,
            mma_tiler_mn: Tuple[int, int] = (128, 128),
            cluster_shape_mn: Tuple[int, int] = (1, 1),
            tactic: int = -1,
    ) -> None:
        """Performs linear operation using cute-dsl with autotuning.

        :param a: Input tensor of shape (M, K)
        :type a: torch.Tensor, type: fp8
        :param b: Weight tensor of shape (N, K)
        :type b: torch.Tensor, type: fp8
        :param a_sf: Input scale tensor of shape (P). P is computed by the following formula:
            P = (div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128)/sizeof(float)
        :type a_sf: torch.Tensor, type: fp32
        :param b_sf: Weight scale tensor of shape (w_n, w_k)
        :type b_sf: torch.Tensor, type: fp32

        :return: Output tensor of shape (M, N)
        :rtype: torch.Tensor, type: bf16
        """
        a, b, a_sf, b_sf, c = inputs
        l, m, n, k = a.shape[0], a.shape[1], b.shape[1], b.shape[2]
        w_n, w_k = b_sf.shape[1], b_sf.shape[2]
        # if c.dtype != torch.bfloat16:
        #     assert False, "c.dtype != bf16"
        # if c.shape != (l, m, n):
        #     assert False, "c.shape != (l, m, n)"

        # """
        # torch_tensor -> cute.tensor
        a_tmp = a.permute(1, 2, 0).view(torch.uint8)
        b_tmp = b.permute(1, 2, 0).view(torch.uint8)
        c_tmp = c.permute(1, 2, 0)
        weight_scale_tmp = b_sf.permute(1, 2, 0)
        # NO: [l, w_k, m] -> [m, w_k, l], (2, 1, 0)
        # div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128/sizeof(float);
        # input: [m, b, n]
        # output: [b, m, n], scales: [b, n/128, padding(m)]
        with nvtx.annotate("bmm input_scale_tmp", color="green"):
            m_padded = pad_up(m, 4)
            input_scale_tmp = a_sf[0:m_padded * w_k * l]
            input_scale_tmp = input_scale_tmp.reshape(l, -1, m_padded)
            input_scale_tmp = (
                input_scale_tmp[:l, :w_k, :m].contiguous().permute(2, 1, 0))
            # after optimization
            # input_scale_tmp = a_sf.permute(2, 1, 0)
        with nvtx.annotate("bmm from_dlpack", color="gray"):
            mA = from_dlpack(
                a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mB = from_dlpack(
                b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mC = from_dlpack(
                c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mA.element_type = cutlass.Float8E4M3FN
            mB.element_type = cutlass.Float8E4M3FN

            # Note: mSFA is column major
            mSFA = from_dlpack(
                input_scale_tmp,
                assumed_align=16).mark_layout_dynamic(leading_dim=0)
            mSFB = from_dlpack(
                weight_scale_tmp,
                assumed_align=16).mark_layout_dynamic(leading_dim=1)
        # """
        # a_tmp = a.view(torch.uint8)
        # b_tmp = b.view(torch.uint8)
        # mA = from_dlpack(
        #     a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        # mB = from_dlpack(
        #     b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        # mC = from_dlpack(
        #     c, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        # mA.element_type = cutlass.Float8E4M3FN
        # mB.element_type = cutlass.Float8E4M3FN

        # # Note: mSFA is column major
        # mSFA = from_dlpack(
        #     a_sf, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        # mSFB = from_dlpack(
        #     b_sf, assumed_align=16).mark_layout_dynamic(leading_dim=2)

        # get stream
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        cache_key = (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
        if cache_key not in CuteDSLFp8BlackwellBmm.kernel_dict:
            with nvtx.annotate("bgemm, compile cache miss", color="yellow"):
                gemm = BlockwiseGemmKernel(
                    cutlass.Float32,  # acc_dtype,
                    use_2cta_instrs=use_2cta_instrs,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                @cute.jit
                def bmm_permute_wrapper(
                    a: cute.Tensor,
                    b: cute.Tensor,
                    c: cute.Tensor,
                    a_sf: cute.Tensor,
                    b_sf: cute.Tensor,
                    max_active_clusters: cutlass.Constexpr,
                    stream: cuda.CUstream,
                ):
                    a = permute(a, (1, 2, 0))
                    b = permute(b, (1, 2, 0))
                    c = permute(c, (1, 2, 0))
                    a_sf = permute(a_sf, (2, 1, 0))
                    b_sf = permute(b_sf, (1, 2, 0))
                    gemm(a, b, c, a_sf, b_sf, max_active_clusters, stream)

                # max_active_clusters = 148
                compiled_gemm = cute.compile(
                    gemm,
                    # bmm_permute_wrapper,
                    mA,
                    mB,
                    mC,
                    mSFA,
                    mSFB,
                    max_active_clusters,
                    stream,
                )
                CuteDSLFp8BlackwellBmm.kernel_dict[cache_key] = compiled_gemm
        else:
            compiled_gemm = CuteDSLFp8BlackwellBmm.kernel_dict[cache_key]

        # launch gemm kernel
        compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)


# a/b: fp8, scale: fp32, out: bf16
@torch.library.custom_op("trtllm::cute_dsl_fp8_bmm_blackwell",
                         mutates_args=(),
                         device_types="cuda")
# @autotuner.tuning_config(
#     name="trtllm::cute_dsl_fp8_bmm_blackwell::gemm",
#     dynamic_tensor_specs=(DynamicTensorSpec(
#         0, 1, get_last_power_of_2_num_tokens_buckets,
#         last_positive_power_of_2), ),
#     # constraint_specs=(
#     #     ConstraintSpec(2, 0, cute_dsl_fp8_bmm_scale_infer_shape_blackwell),
#     # ),
#     constraint_specs=(ConstraintSpec(
#         2, 2, cute_dsl_fp8_bmm_scale_infer_shape_blackwell), ),
#     configs={
#         "use_2cta_instrs": [False],
#         "mma_tiler_mn": [(128, 128)],
#         "cluster_shape_mn": [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1),
#                              (4, 4)],
#         # 'cluster_shape_mn': [(1, 1)],
#     },
# )
def cute_dsl_fp8_bmm_blackwell(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # tuner = AutoTuner.get()

    # # allocate workspace for profiling
    # cute_dsl_fp8_bmm_blackwell_runner = CuteDSLFp8BlackwellBmm()

    # _, best_tactic, best_config = tuner.choose_one(
    #     "trtllm::cute_dsl_fp8_bmm_blackwell::gemm",
    #     [cute_dsl_fp8_bmm_blackwell_runner],
    #     [input, weight, input_scale, weight_scale, out],
    # )
    # cute_dsl_fp8_bmm_blackwell_runner(
    #     inputs=[input, weight, input_scale, weight_scale, out],
    #     tactic=best_tactic,
    #     **best_config,
    # )

    cute_dsl_fp8_bmm_blackwell_runner = CuteDSLFp8BlackwellBmm()
    cute_dsl_fp8_bmm_blackwell_runner(
        inputs=[input, weight, input_scale, weight_scale, out],
        tactic=0,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )


@torch.library.register_fake("trtllm::cute_dsl_fp8_bmm_blackwell")
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    l, m, k = mat_a.shape[0], mat_a.shape[1], mat_a.shape[2]
    n = mat_b.shape[1]
    if out.dtype != torch.bfloat16:
        assert False, "out.dtype != bf16"
    if out.shape != (l, m, n):
        assert False, "out.shape != (l, m, n)"


class CuteDSLFp8BlackwellGroupGemm(TunableRunner):
    kernel_dict = dict()

    # stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def __init__(self):
        super().__init__()

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        # use_tma_store: bool = True,
        **kwargs,
    ) -> List[int]:
        # [m, k]
        m, k = inputs[0].shape[0], inputs[0].shape[1]
        # [group_num, n, k]
        group_num, n, k = inputs[1].shape[0], inputs[1].shape[1], inputs[
            1].shape[2]
        # m,k
        a_major = "k"
        # n, k
        b_major = "k"
        # m, n
        c_major = "n"
        is_valid = BlockwiseContiguousGroupedGemmKernel.can_implement(
            cutlass.Float8E4M3FN,  # ab_dtype,ab_dtype,
            cutlass.Float32,  # acc_dtype,
            cutlass.BFloat16,  # c_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            # use_tma_store,
            m,
            n,
            k,
            group_num,
            a_major,
            b_major,
            c_major,
        )
        if is_valid:
            return [0]
        else:
            return []

    def forward(
        self,
        inputs: List[torch.Tensor],
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        # use_tma_store: bool = True,
        tactic: int = -1,
    ) -> torch.Tensor:
        """Performs linear operation using cute-dsl with autotuning.

        :param a: Input tensor of shape (M, K)
        :type a: torch.Tensor, type: fp8
        :param b: Weight tensor of shape (N, K)
        :type b: torch.Tensor, type: fp8
        :param a_sf: Input scale tensor of shape (P). P is computed by the following formula:
            P = (div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128)/sizeof(float)
        :type a_sf: torch.Tensor, type: fp32
        :param b_sf: Weight scale tensor of shape (w_n, w_k)
        :type b_sf: torch.Tensor, type: fp32

        :return: Output tensor of shape (M, N)
        :rtype: torch.Tensor, type: bf16
        """
        # before opt
        # t_start= time()
        """
        with nvtx.annotate("gemm, empty", color="red"):
            a, b, a_sf, b_sf, group_offset = inputs
            m, k = a.shape[0], a.shape[1]
            num_group, n, b_k = b.shape[0], b.shape[1], b.shape[2]
            # assert b_k == k, "b_k must be equal to k"

            num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]
            c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
            # assert n%128 == 0, "n must be divisible by 128"
            # assert k%128 == 0, "k must be divisible by 128"

        # torch_tensor -> cute.tensor
        with nvtx.annotate("gemm, tmp", color="green"):
            a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
            b_tmp = b.permute(1, 2, 0).view(torch.uint8)
            c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))
            b_sf_tmp = b_sf.permute(1, 2, 0)
            input_scale_tmp = a_sf.permute(1, 0).as_strided(
                (m, w_k, 1), (1, m, w_k * m)
            )
            # OK: renmoved
            # group_offset_tmp = group_offset.to(torch.int32)

        with nvtx.annotate("gemm, from_dlpack", color="gray"):
            mA = from_dlpack(a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mB = from_dlpack(b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mC = from_dlpack(c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mA.element_type = cutlass.Float8E4M3FN
            mB.element_type = cutlass.Float8E4M3FN

            mSFB = from_dlpack(b_sf_tmp, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            mSFA = from_dlpack(input_scale_tmp, assumed_align=16).mark_layout_dynamic(
                leading_dim=0
            )
            # group_offset_cute_tensor = from_dlpack(
            #     group_offset_tmp).mark_layout_dynamic()
            group_offset_cute_tensor = from_dlpack(group_offset).mark_layout_dynamic()
            print(f"limin: mA.shape = {mA.shape}, mA.stride = {mA.stride}")
            print(f"limin: mB.shape = {mB.shape}, mB.stride = {mB.stride}")
            print(f"limin: mC.shape = {mC.shape}, mC.stride = {mC.stride}")
            print(f"limin: mSFB.shape = {mSFB.shape}, mSFB.stride = {mSFB.stride}")
            print(f"limin: mSFA.shape = {mSFA.shape}, mSFA.stride = {mSFA.stride}")
            print(f"limin: group_offset_cute_tensor.shape = {group_offset_cute_tensor.shape}, group_offset_cute_tensor.stride = {group_offset_cute_tensor.stride}")
        """

        a, b, a_sf, b_sf, group_offset = inputs
        m, n = a.shape[0], b.shape[1]
        # assert b_k == k, "b_k must be equal to k"
        c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
        # assert n%128 == 0, "n must be divisible by 128"
        # assert k%128 == 0, "k must be divisible by 128"

        a_tmp = a.view(torch.uint8)
        b_tmp = b.view(torch.uint8)

        mA = from_dlpack(a_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mB = from_dlpack(b_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mC = from_dlpack(c, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mA.element_type = cutlass.Float8E4M3FN
        mB.element_type = cutlass.Float8E4M3FN

        mSFB = from_dlpack(b_sf,
                           assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mSFA = from_dlpack(a_sf,
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)
        group_offset_cute_tensor = from_dlpack(
            group_offset).mark_layout_dynamic()
        # print(f"limin: mA.shape = {mA.shape}, mA.stride = {mA.stride}")
        # print(f"limin: mB.shape = {mB.shape}, mB.stride = {mB.stride}")
        # print(f"limin: mC.shape = {mC.shape}, mC.stride = {mC.stride}")
        # print(f"limin: mSFB.shape = {mSFB.shape}, mSFB.stride = {mSFB.stride}")
        # print(f"limin: mSFA.shape = {mSFA.shape}, mSFA.stride = {mSFA.stride}")
        # print(f"limin: group_offset_cute_tensor.shape = {group_offset_cute_tensor.shape}, group_offset_cute_tensor.stride = {group_offset_cute_tensor.stride}")

        # get stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        cache_key = (
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            # use_tma_store
        )
        if cache_key not in CuteDSLFp8BlackwellGroupGemm.kernel_dict:
            with nvtx.annotate("gemm, compile cache miss", color="yellow"):
                gemm = BlockwiseContiguousGroupedGemmKernel(
                    cutlass.Float32,  # acc_dtype,
                    use_2cta_instrs=use_2cta_instrs,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    # use_tma_store=use_tma_store,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                @cute.jit
                def group_gemm_permute_wrapper(
                    a: cute.Tensor,
                    b: cute.Tensor,
                    c: cute.Tensor,
                    a_sf: cute.Tensor,
                    b_sf: cute.Tensor,
                    group_offset: cute.Tensor,
                    max_active_clusters: cutlass.Constexpr,
                    stream: cuda.CUstream,
                ):
                    # a = cute.append_ones(a)
                    # c = cute.append_ones(c)
                    a = append_ones_wrapper(a)
                    c = append_ones_wrapper(c)
                    b = permute(b, (1, 2, 0))
                    b_sf = permute(b_sf, (1, 2, 0))

                    a_sf = permute(a_sf, (1, 0))
                    # a_sf = cute.append_ones(a_sf)
                    a_sf = append_ones_wrapper(a_sf)
                    # print(f"limin-compile: a = {a}")
                    # print(f"limin-compile: b = {b}")
                    # print(f"limin-compile: c = {c}")
                    # print(f"limin-compile: a_sf = {a_sf}")
                    # print(f"limin-compile: b_sf = {b_sf}")
                    # print(f"limin-compile: group_offset = {group_offset}")
                    # cute.printf(f"limin: a = {a.shape}, a.stride = {a.stride}")
                    # cute.printf(f"limin: b = {b.shape}, b.stride = {b.stride}")
                    # cute.printf(f"limin: c = {c.shape}, c.stride = {c.stride}")
                    # cute.printf(f"limin: a_sf = {a_sf.shape}, a_sf.stride = {a_sf.stride}")
                    # cute.printf(f"limin: b_sf = {b_sf.shape}, b_sf.stride = {b_sf.stride}")
                    # cute.printf(f"limin: group_offset = {group_offset.shape}, group_offset.stride = {group_offset.stride}")
                    # cute.print_tensor(a)
                    # cute.print_tensor(b)
                    # cute.print_tensor(c)
                    # cute.print_tensor(a_sf)
                    # cute.print_tensor(b_sf)
                    # cute.print_tensor(group_offset)

                    gemm(a, b, c, a_sf, b_sf, group_offset, max_active_clusters,
                         stream)

                compiled_gemm = cute.compile(
                    # gemm,
                    group_gemm_permute_wrapper,
                    mA,
                    mB,
                    mC,
                    mSFA,
                    mSFB,
                    group_offset_cute_tensor,
                    max_active_clusters,
                    stream,
                )
                CuteDSLFp8BlackwellGroupGemm.kernel_dict[
                    cache_key] = compiled_gemm
        else:
            compiled_gemm = CuteDSLFp8BlackwellGroupGemm.kernel_dict[cache_key]

        # launch gemm kernel
        compiled_gemm(mA, mB, mC, mSFA, mSFB, group_offset_cute_tensor, stream)
        return c


# ### a/b: fp8, scale: fp32 -> bf16
@torch.library.custom_op("trtllm::cute_dsl_fp8_group_gemm_blackwell",
                         mutates_args=(),
                         device_types="cuda")
# # @autotuner.tuning_config(
# #     name="trtllm::cute_dsl_fp8_group_gemm_blackwell::group_gemm",
# #     # dynamic_tensor_specs=(DynamicTensorSpec(
# #     #     0, 0,
# #     #     # get_last_power_of_2_num_tokens_buckets,
# #     #     (128,),
# #     #     last_positive_power_of_2), ),
# #     # constraint_specs=(ConstraintSpec(
# #     #     2, 1, cute_dsl_fp8_linear_scale_infer_shape_blackwell), ),
# #     configs={
# #         'use_2cta_instrs': [False],
# #         'mma_tiler_mn': [(128, 128)],
# #         'cluster_shape_mn': [(1, 1)],
# #     },
# # )
def cute_dsl_fp8_group_gemm_blackwell(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_offset: torch.Tensor,
) -> torch.Tensor:
    # tuner = AutoTuner.get()

    # # allocate workspace for profiling
    # cute_dsl_fp8_group_gemm_blackwell_runner = CuteDSLFp8BlackwellGroupGemm()

    # _, best_tactic, best_config = tuner.choose_one(
    #     "trtllm::cute_dsl_fp8_group_gemm_blackwell::group_gemm",
    #     [cute_dsl_fp8_group_gemm_blackwell_runner],
    #     [input, weight, input_scale, weight_scale, group_offset],
    # )

    # return cute_dsl_fp8_group_gemm_blackwell_runner(
    #     inputs=[input, weight, input_scale, weight_scale, group_offset],
    #     tactic=best_tactic,
    #     **best_config)

    cute_dsl_fp8_group_gemm_blackwell_runner = CuteDSLFp8BlackwellGroupGemm()
    return cute_dsl_fp8_group_gemm_blackwell_runner(
        inputs=[input, weight, input_scale, weight_scale, group_offset],
        tactic=0,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        # use_tma_store=True,
    )


@torch.library.register_fake("trtllm::cute_dsl_fp8_group_gemm_blackwell")
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_offset: torch.Tensor,
) -> torch.Tensor:
    m, k = mat_a.shape[0], mat_a.shape[1]
    num_group, n, k = mat_b.shape[0], mat_b.shape[1], mat_b.shape[2]
    return mat_a.new_empty((m, n), dtype=torch.bfloat16)
