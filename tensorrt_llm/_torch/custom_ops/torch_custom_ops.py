from functools import lru_cache
from typing import List, Mapping, Optional, Tuple

import torch
import triton  # type: ignore[import]

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm import deep_gemm
from tensorrt_llm._utils import get_sm_version

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)
from ..modules.multi_stream_utils import do_multi_stream
from ..modules.swiglu import silu_and_mul_kernel
from ..utils import (fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)


# Used to WAR an issue in torch.bmm that it would break the graph when the out is not contiguous.
@torch.library.custom_op("trtllm::bmm_out", mutates_args=("out", ))
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(a, b, out=out)


class MoERunner(TunableRunner):
    # avoid overhead of creating a new runner in forward pass
    runner_dict = dict()
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(
            0, 0, get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2), ),
        tune_max_num_tokens=8192,
    )

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
        use_int8_woq_per_channel: bool,
        use_mxfp8_act_scaling: bool,
        min_latency_mode: bool,
        use_fused_finalize: bool,
        unpadded_hidden_size: Optional[int] = None,
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
        self.use_int8_woq_per_channel = use_int8_woq_per_channel
        self.use_mxfp8_act_scaling = use_mxfp8_act_scaling
        self.min_latency_mode = min_latency_mode
        self.use_fused_finalize = use_fused_finalize
        self.unpadded_hidden_size = unpadded_hidden_size if unpadded_hidden_size is not None else 0

        instance_key = (x_dtype, weight_dtype, output_dtype,
                        use_deepseek_fp8_block_scale, use_w4_group_scaling,
                        use_int8_woq_per_channel, use_mxfp8_act_scaling)

        if instance_key not in MoERunner.runner_dict:
            MoERunner.runner_dict[
                instance_key] = torch.classes.trtllm.FusedMoeRunner(
                    x_dtype, weight_dtype, output_dtype,
                    use_deepseek_fp8_block_scale, use_w4_group_scaling,
                    use_int8_woq_per_channel, use_mxfp8_act_scaling,
                    use_fused_finalize)
        self.fused_moe_runner = MoERunner.runner_dict[instance_key]

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        return range(self.fused_moe_runner.get_tactic_num(kwargs["gemm_idx"]))

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
            self.unpadded_hidden_size,
        )


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
    use_int8_woq_per_channel: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_fused_finalize: bool = True,
    tune_max_num_tokens: int = 8192,
    tuner_num_tokens: Optional[int] = None,
    tuner_top_k: Optional[int] = None,
    unpadded_hidden_size: Optional[int] = None,
) -> List[torch.Tensor]:

    tuner = AutoTuner.get()

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
        use_int8_woq_per_channel=use_int8_woq_per_channel,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        min_latency_mode=min_latency_mode,
        use_fused_finalize=use_fused_finalize,
        unpadded_hidden_size=unpadded_hidden_size,
    )

    MoERunner.tuning_config.tune_max_num_tokens = tune_max_num_tokens

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
        unpadded_hidden_size,
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
    use_int8_woq_per_channel: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_fused_finalize: bool = True,
    tune_max_num_tokens: int = 8192,
    tuner_num_tokens: Optional[int] = None,
    tuner_top_k: Optional[int] = None,
    unpadded_hidden_size: Optional[int] = None,
):
    seq_len = input.shape[0]
    if use_int8_woq_per_channel:
        # Note: The weight shape for INT8 weight only quantization is different, i.e.,
        # fc2_expert_weights: [num_experts, inter_size, hidden_size]
        hidden_size = fc2_expert_weights.shape[2]
    else:
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

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
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

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
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


class CublasLtFP4GemmRunner(TunableRunner):
    """CublasLt-based FP4 GEMM runner with auto-tuning support.

    Uses cuBLASLt's heuristic to discover and tune across multiple algorithms.
    """
    runner_dict = dict()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, get_last_power_of_2_num_tokens_buckets,
        last_positive_power_of_2), ),
                                 constraint_specs=(ConstraintSpec(
                                     2, 0, fp4_scale_infer_shape), ))

    def __init__(
        self,
        to_userbuffers: bool,
        output_dtype: torch.dtype,
    ):
        self.output_dtype = output_dtype
        self.to_userbuffers = to_userbuffers
        instance_key = (output_dtype, )

        if instance_key not in CublasLtFP4GemmRunner.runner_dict:
            CublasLtFP4GemmRunner.runner_dict[
                instance_key] = torch.classes.trtllm.CublasLtFP4GemmRunner(
                    output_dtype)

        self.cublaslt_runner = CublasLtFP4GemmRunner.runner_dict[instance_key]

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        """Get all valid tactics (algorithms) from cuBLASLt heuristic."""
        mat1, mat2, mat1_scale, mat2_scale, alpha = inputs
        # Pass actual scale tensors to avoid creating dummy tensors in C++
        num_algos = self.cublaslt_runner.get_num_heuristic_algos(
            mat1, mat2, mat1_scale, mat2_scale)
        return list(range(num_algos))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:
        mat1, mat2, mat1_scale, mat2_scale, alpha = inputs
        result = self.cublaslt_runner.run_gemm(
            mat1,
            mat2,
            mat1_scale,
            mat2_scale,
            alpha,
            self.to_userbuffers,
            tactic,
        )
        return result


@torch.library.custom_op("trtllm::nvfp4_gemm_cublaslt", mutates_args=())
def nvfp4_gemm_cublaslt(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    to_userbuffers: bool = False,
) -> torch.Tensor:
    """cuBLASLt-based NVFP4 GEMM with heuristic-based auto-tuning."""
    tuner = AutoTuner.get()

    # Use CublasLt runner with heuristic-based tuning
    nvfp4_gemm_runner = CublasLtFP4GemmRunner(to_userbuffers, output_dtype)

    runner_type = type(nvfp4_gemm_runner).__name__
    op_key = f"trtllm::fp4_gemm_cublaslt::gemm::{runner_type}"

    _, best_tactic = tuner.choose_one(
        op_key,
        [nvfp4_gemm_runner],
        nvfp4_gemm_runner.tuning_config,
        [act_fp4, weight, act_sf, weight_scale, alpha],
    )

    result = nvfp4_gemm_runner(
        inputs=[act_fp4, weight, act_sf, weight_scale, alpha],
        tactic=best_tactic)

    return result


@nvfp4_gemm_cublaslt.register_fake
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
    """CUTLASS-based NVFP4 GEMM with auto-tuning."""
    tuner = AutoTuner.get()

    # Use Cutlass runner with predefined configs
    nvfp4_gemm_runner = FP4GemmRunner(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                                      to_userbuffers, output_dtype)

    runner_type = type(nvfp4_gemm_runner).__name__
    _, best_tactic = tuner.choose_one(
        f"trtllm::fp4_gemm::gemm::{runner_type}",
        [nvfp4_gemm_runner],
        nvfp4_gemm_runner.tuning_config,
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

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:

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

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
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

    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
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


def fp8_swap_ab_gen_tuning_buckets(x: int):
    buckets = tuple(range(8, 128, 8))
    if x >= 128:
        buckets += tuple(range(128, x, 128))
    return buckets


class fp8SwapABGemmRunner(TunableRunner):
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(
            0, 0, fp8_swap_ab_gen_tuning_buckets), ),
        tune_max_num_tokens=4096,
    )

    def __init__(self, output_dtype: torch.dtype, disable_ue8m0_cast: bool):
        self.output_dtype = output_dtype
        self.disable_ue8m0_cast = disable_ue8m0_cast

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        # Encode swap_ab as False (0) and True (1). Currently enabled when GEMM m <= 128.
        input, _, _ = inputs
        return [0, 1] if input.shape[0] <= 128 else [0]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
    ) -> torch.Tensor:
        input, weight, weight_scale = inputs
        a, a_sf = fp8_utils.per_token_quant_and_transform(input)
        output = torch.empty(
            (input.size(0), weight.size(0)),
            device=input.device,
            dtype=self.output_dtype,
        )

        forward_func = deep_gemm.fp8_gemm_ntt if tactic == 1 else deep_gemm.fp8_gemm_nt
        forward_func(
            (a, a_sf),
            (weight, weight_scale),
            output,
            disable_ue8m0_cast=self.disable_ue8m0_cast,
        )
        return output


@torch.library.custom_op("trtllm::fp8_swap_ab_gemm", mutates_args=())
def fp8_swap_ab_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    disable_ue8m0_cast: bool = False,
    tune_max_num_tokens: int = 4096,
) -> torch.Tensor:
    tuner = AutoTuner.get()
    fp8_swap_ab_gemm_runner = fp8SwapABGemmRunner(
        output_dtype,
        disable_ue8m0_cast,
    )
    fp8SwapABGemmRunner.tuning_config.tune_max_num_tokens = tune_max_num_tokens
    _, best_tactic = tuner.choose_one(
        "trtllm::fp8_swap_ab_gemm",
        [fp8_swap_ab_gemm_runner],
        fp8SwapABGemmRunner.tuning_config,
        [input, weight, weight_scale],
    )
    return fp8_swap_ab_gemm_runner(
        inputs=[input, weight, weight_scale],
        tactic=best_tactic,
    )


@fp8_swap_ab_gemm.register_fake
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    disable_ue8m0_cast: bool = False,
    tune_max_num_tokens: int = 4096,
) -> torch.Tensor:
    return input.new_empty((input.size(0), weight.size(0)), dtype=output_dtype)


@torch.library.custom_op("trtllm::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor,
                 scale: Optional[torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o_dtype = dtype or x.dtype
    o = torch.empty((b, d), dtype=o_dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    silu_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        o_scale_ptr=scale,
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        BLOCK_SIZE=1024,
        HAS_O_SCALE=scale is not None,
    )

    return o


@silu_and_mul.register_fake
def _(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o_dtype = dtype or x.dtype
    return x.new_empty((b, d), dtype=o_dtype)


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
    if not do_multi_stream():
        return
    stream = get_stream(stream_id)
    assert stream is not None
    torch.cuda.set_stream(stream)


@torch.library.custom_op("trtllm::record_event", mutates_args=())
def record_event(event_idx: int) -> None:
    if not do_multi_stream():
        return
    event = get_event(event_idx)
    event.record()


@torch.library.custom_op("trtllm::wait_event", mutates_args=())
def wait_event(event_idx: int) -> None:
    if not do_multi_stream():
        return
    event = get_event(event_idx)
    event.wait()


@torch.library.custom_op("trtllm::record_stream", mutates_args=())
def record_stream(tensor: torch.Tensor, stream_id: int) -> None:
    if not do_multi_stream():
        return
    stream = get_stream(stream_id)
    assert stream is not None
    tensor.record_stream(stream)
