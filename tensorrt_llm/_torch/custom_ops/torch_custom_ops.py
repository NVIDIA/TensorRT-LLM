import sys
from functools import lru_cache
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cuda import cuda
from cutlass.cute.runtime import from_dlpack

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from .. import autotuner
from ..attention_backend.interface import AttentionInputType
from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)
from ..utils import (compute_swizzled_sf_shape, fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)

sys.path.append(
    '/home/lmin/scratch/trt-dkg/cutlass_ir/compiler/python/examples/hopper')
from blockwise_gemm import HopperBlockwiseGemmKernel


# Used to WAR an issue in torch.bmm that it would break the graph when the out is not contiguous.
@torch.library.custom_op("trtllm::bmm_out", mutates_args=("out", ))
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(a, b, out=out)


class MoERunner(TunableRunner):
    # avoid overhead of creating a new runner in forward pass
    runner_dict = dict()

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
        enable_alltoall: bool,
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
        self.enable_alltoall = enable_alltoall
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
            self.enable_alltoall,
            min_latency_mode,
            gemm_idx,
            tactic,
            do_preparation,
        )

    @classmethod
    @lru_cache(maxsize=None)
    def refine_tuning_config(cls, tune_max_num_tokens: int):
        # TODO: Remove this and put it automatically in the autotuner
        # User can decide if tune_max_num_tokens is needed or not
        tuning_config = TuningConfig(
            name=("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"),
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    0, 0,
                    get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens),
                    lambda x: min(last_positive_power_of_2(x),
                                  tune_max_num_tokens)),
                DynamicTensorSpec(3, 0, (0, ), lambda x: x),
            ),
        )
        AutoTuner.get().register_tuning_config(tuning_config)


@torch.library.custom_op("trtllm::fused_moe", mutates_args=())
@autotuner.tuning_config(
    name=("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"),
    dynamic_tensor_specs=(
        DynamicTensorSpec(0, 0, get_last_power_of_2_num_tokens_buckets(8192),
                          lambda x: min(last_positive_power_of_2(x), 8192)),
        DynamicTensorSpec(3, 0, (0, ), lambda x: x),
    ),
)
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
    enable_alltoall: bool = False,
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
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4a8_group_scaling=use_w4a8_group_scaling,
    )

    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1",
        [moe_runner],
        [input, fc1_expert_weights, fc2_expert_weights, min_latency_tensor],
        gemm_idx=1,
    )

    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2",
        [moe_runner],
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


class FP4GemmRunner(TunableRunner):
    runner_dict = dict()

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
@autotuner.tuning_config(
    name="trtllm::nvfp4_gemm::gemm",
    dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, get_last_power_of_2_num_tokens_buckets,
        last_positive_power_of_2), ),
    constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
)
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
        [act_fp4, weight, act_sf, weight_scale, alpha],
    )

    return nvfp4_gemm_runner(
        inputs=[act_fp4, weight, act_sf, weight_scale, alpha],
        tactic=best_tactic,
    )


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

    def __init__(self, output_dtype: torch.dtype, use_deep_seek_fp8: bool,
                 low_latency_kernel: bool):

        self.output_dtype = output_dtype
        self.use_deep_seek_fp8 = use_deep_seek_fp8
        self.low_latency_kernel = low_latency_kernel

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        tile_size: int = 8,
        epilogue_tile_m: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the batched GEMM operation with the given inputs and tactic.
        """

        if tactic == -1:
            # use the right default value for epilogue_tile_m
            # This is not tunable due to
            epilogue_tile_m = 64 if self.use_deep_seek_fp8 else 128

        mat1, mat2, dq_sfs_a, dq_sfs_b, scale_c = inputs
        kernel_runner = self.get_runner(self.output_dtype,
                                        self.use_deep_seek_fp8,
                                        self.low_latency_kernel, tile_size,
                                        epilogue_tile_m)

        out_tensors = kernel_runner.run_batched_gemm(
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
        tile_size: int,
        epilogue_tile_m: int,
    ) -> List[int]:

        valid_epilogue_tile_m = (epilogue_tile_m == 64
                                 and self.use_deep_seek_fp8) or (
                                     epilogue_tile_m == 128
                                     and not self.use_deep_seek_fp8)
        m = inputs[0].shape[1]
        valid_tile_size = (m % tile_size == 0)
        valid = valid_epilogue_tile_m and valid_tile_size
        if not valid:
            return []

        mat1, mat2, _, _, _ = inputs

        b = mat1.shape[0]
        m = mat1.shape[1]
        n = mat2.shape[1]
        k = mat1.shape[2]

        kernel_runner = self.get_runner(self.output_dtype,
                                        self.use_deep_seek_fp8,
                                        self.low_latency_kernel, tile_size,
                                        epilogue_tile_m)

        tactics = kernel_runner.get_valid_configs(b, m, n, k)

        return tactics

    def get_runner(self, output_dtype: torch.dtype, use_deep_seek_fp8: bool,
                   low_latency_kernel: bool, tile_size: int,
                   epilogue_tile_m: int):
        instance_key = (output_dtype, use_deep_seek_fp8, low_latency_kernel,
                        tile_size, epilogue_tile_m)

        if instance_key not in FP8BatchedGemmRunner.runner_dict:
            FP8BatchedGemmRunner.runner_dict[
                instance_key] = torch.classes.trtllm.FP8BatchedGemmRunner(
                    output_dtype, use_deep_seek_fp8, low_latency_kernel,
                    tile_size, epilogue_tile_m)

        return FP8BatchedGemmRunner.runner_dict[instance_key]

    @classmethod
    def constrain_dq_sfs_a_dim1(cls, shapes: Tuple[torch.Size]) -> int:
        b = shapes[0][0]
        m = shapes[0][1]

        return m * b


@torch.library.custom_op("trtllm::fp8_batched_gemm_trtllmgen", mutates_args=())
@autotuner.tuning_config(
    name="trtllm::fp8_batched_gemm_trtllmgen::batched_gemm",
    dynamic_tensor_specs=(DynamicTensorSpec(
        0, 1, (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        last_positive_power_of_2), ),
    constraint_specs=(ConstraintSpec(
        2, 1, FP8BatchedGemmRunner.constrain_dq_sfs_a_dim1), ),
    configs={
        'tile_size': [8],
        'epilogue_tile_m': [64, 128]
    },
)
def fp8_batched_gemm_trtllmgen(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    use_deep_seek_fp8: Optional[bool] = False,
    low_latency: Optional[bool] = False,
    dq_sfs_a: Optional[torch.Tensor] = None,
    dq_sfs_b: Optional[torch.Tensor] = None,
    scale_c: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.half
) -> Tuple[torch.Tensor, torch.Tensor]:

    kernel_runner = FP8BatchedGemmRunner(output_dtype=out_dtype,
                                         use_deep_seek_fp8=use_deep_seek_fp8,
                                         low_latency_kernel=low_latency)

    tuner = AutoTuner.get()

    inputs = [mat1, mat2, dq_sfs_a, dq_sfs_b, scale_c]

    _, best_tactic, best_config = tuner.choose_one(
        "trtllm::fp8_batched_gemm_trtllmgen::batched_gemm",
        [kernel_runner],
        inputs,
    )

    return kernel_runner(
        inputs=inputs,
        tactic=best_tactic,
        **best_config,
    )


# Allows the tunable TRTLLM-Gen FP8 batched GEMM to be
# used with torch.compile
@fp8_batched_gemm_trtllmgen.register_fake
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    use_deep_seek_fp8: Optional[bool] = False,
    low_latency: Optional[bool] = False,
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


def div_up(x: int, y: int) -> int:
    return ((x + y - 1) // y)


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def cute_dsl_fp8_scale_infer_shape(input_shapes: List[List[int]]):
    """Calculate the dimensions of the fp4 scale tensor.
    """
    m, k = input_shapes[0]
    scale_shape = pad_up(pad_up(m, 4) * div_up(k, 128) * 4, 128) // 4
    return scale_shape


class DummyHopperBlockwiseGemmKernel:

    def __init__(
        self,
        acc_dtype: torch.dtype,
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
    ):
        self.acc_dtype = acc_dtype
        self.tile_shape_mnk = tile_shape_mnk
        self.cluster_shape_mnk = cluster_shape_mnk

    def __call__(self, mA, mB, mC, mSFA, mSFB):
        # print("[DummyHopperBlockwiseGemmKernel] Run inference")
        # print(f"[DummyHopperBlockwiseGemmKernel] mA.shape = {mA.shape}")
        # print(f"[DummyHopperBlockwiseGemmKernel] mB.shape = {mB.shape}")
        # print(f"[DummyHopperBlockwiseGemmKernel] mC.shape = {mC.shape}")
        # print(f"[DummyHopperBlockwiseGemmKernel] mSFA.shape = {mSFA.shape}")
        # print(f"[DummyHopperBlockwiseGemmKernel] mSFB.shape = {mSFB.shape}")
        # print(f"[DummyHopperBlockwiseGemmKernel] self.acc_dtype = {self.acc_dtype}")
        # print(f"[DummyHopperBlockwiseGemmKernel] self.tile_shape_mnk = {self.tile_shape_mnk}")
        # print(f"[DummyHopperBlockwiseGemmKernel] self.cluster_shape_mnk = {self.cluster_shape_mnk}")
        pass


class CuteDSLFp8Linear(TunableRunner):
    kernel_dict = dict()

    def __init__(self):
        super().__init__()

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        # Each config corresponds to a single generated kernel in this case.
        return [0]

    def forward(
        self,
        inputs: List[torch.Tensor],
        # acc_dtype: torch.dtype = torch.bfloat16,
        tile_shape_mnk: Tuple[int, int, int] = (128, 128, 128),
        cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1),
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
        a, b, a_sf, b_sf = inputs
        m, n, k = a.shape[0], b.shape[0], a.shape[1]
        w_n, w_k = b_sf.shape[0], b_sf.shape[1]
        c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
        # print(f"limin: m = {m}, n = {n}, k = {k}, w_n = {w_n}, w_k = {w_k}")
        # print("limin: a.dtype = ", a.dtype)
        # print("limin: b.dtype = ", b.dtype)
        # print("limin: a_sf.dtype = ", a_sf.dtype)
        # print("limin: b_sf.dtype = ", b_sf.dtype)
        # print(f"limin: a.shape = {a.shape}, a.stride = {a.stride()}")
        # print(f"limin: b.shape = {b.shape}, b.stride = {b.stride()}")
        # print(f"limin: a_sf.shape = {a_sf.shape}, a_sf.stride = {a_sf.stride()}")
        # print(f"limin: b_sf.shape = {b_sf.shape}, b_sf.stride = {b_sf.stride()}")

        # torch_tensor -> cute.tensor
        a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
        b_tmp = b.as_strided((n, k, 1), (k, 1, n * k)).view(torch.uint8)
        c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))

        weight_scale_tmp = b_sf.as_strided((w_n, w_k, 1), (w_k, 1, w_n * w_k))

        m_padded = pad_up(m, 4)
        input_scale_tmp = a_sf[0:m_padded * w_k]
        # print(f"limin: 0, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
        input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
        # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
        input_scale_tmp = input_scale_tmp[:w_k, :m_padded].contiguous().permute(
            1, 0)
        # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
        input_scale_tmp = input_scale_tmp.as_strided(
            (m_padded, w_k, 1), (1, m_padded, m_padded * w_k))
        # print(f"limin: 3, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")

        mA = from_dlpack(a_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mB = from_dlpack(b_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mC = from_dlpack(c_tmp,
                         assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mA.element_type = cutlass.Float8E4M3FN
        mB.element_type = cutlass.Float8E4M3FN

        # TODO: mSFA is column major
        mSFA = from_dlpack(input_scale_tmp,
                           assumed_align=16).mark_layout_dynamic(leading_dim=0)
        mSFB = from_dlpack(weight_scale_tmp,
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)

        # print(f"limin: mA.shape = {mA.shape}, mA.stride = {mA.stride}")
        # print(f"limin: mB.shape = {mB.shape}, mB.stride = {mB.stride}")
        # print(f"limin: mC.shape = {mC.shape}, mC.stride = {mC.stride}")
        # print(f"limin: mSFA.shape = {mSFA.shape}, mSFA.stride = {mSFA.stride}")
        # print(f"limin: mSFB.shape = {mSFB.shape}, mSFB.stride = {mSFB.stride}")

        # gemm = HopperBlockwiseGemmKernel(
        #     # acc_dtype,  # acc_dtype,
        #     cutlass.Float32,
        #     tile_shape_mnk=tile_shape_mnk,
        #     cluster_shape_mnk=cluster_shape_mnk,
        # )

        # get stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        cache_key = (tile_shape_mnk, cluster_shape_mnk)
        if cache_key not in CuteDSLFp8Linear.kernel_dict:
            gemm = HopperBlockwiseGemmKernel(
                # acc_dtype,  # acc_dtype,
                cutlass.Float32,
                tile_shape_mnk=tile_shape_mnk,
                cluster_shape_mnk=cluster_shape_mnk,
            )
            # compiled_gemm = gemm
            compiled_gemm = cute.compile(
                gemm,
                mA,
                mB,
                mC,
                mSFA,
                mSFB,
                stream,
            )
            CuteDSLFp8Linear.kernel_dict[cache_key] = compiled_gemm
        else:
            compiled_gemm = CuteDSLFp8Linear.kernel_dict[cache_key]

        # launch gemm kernel
        compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)

        return c


### a/b: fp8, scale: fp32 -> bf16
@torch.library.custom_op("trtllm::cute_dsl_fp8_gemm",
                         mutates_args=(),
                         device_types="cuda")
@autotuner.tuning_config(
    name="trtllm::cute_dsl_fp8_gemm::gemm",
    dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, get_last_power_of_2_num_tokens_buckets,
        last_positive_power_of_2), ),
    constraint_specs=(ConstraintSpec(2, 0, cute_dsl_fp8_scale_infer_shape), ),
    configs={
        # 'acc_dtype': [torch.float],
        'tile_shape_mnk': [(64, 128, 128), (128, 128, 128)],
        'cluster_shape_mnk': [(1, 1, 1), (2, 2, 1), (1, 4, 1), (4, 4, 1)]
    },
)
def cute_dsl_fp8_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    tuner = AutoTuner.get()

    # allocate workspace for profiling
    cute_dsl_fp8_gemm_runner = CuteDSLFp8Linear()

    _, best_tactic, best_config = tuner.choose_one(
        "trtllm::cute_dsl_fp8_gemm::gemm",
        [cute_dsl_fp8_gemm_runner],
        [input, weight, input_scale, weight_scale],
    )

    return cute_dsl_fp8_gemm_runner(
        inputs=[input, weight, input_scale, weight_scale],
        tactic=best_tactic,
        **best_config)


@torch.library.register_fake("trtllm::cute_dsl_fp8_gemm")
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
    # TODO: output is fixed as bf16?
    ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
    return ret


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
