from typing import Dict, List, Optional

import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from ..autotuner import AutoTuner, TunableRunner, TuningConfig
from ..utils import (get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2, next_positive_power_of_2)


# Used to WAR an issue in torch.bmm that it would break the graph when the out is not contiguous.
@torch.library.custom_op("trtllm::bmm_out", mutates_args=("out", ))
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(a, b, out=out)


class MoERunner(TunableRunner):
    # avoid overhead of creating a new runner in forward pass
    _runner_dict: Dict[str, torch.classes.trtllm.FusedMoeRunner] = dict()

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
        use_fp8_block_scaling: bool,
    ):
        self.x_dtype = x_dtype
        self.weight_dtype = weight_dtype
        self.output_dtype = output_dtype
        self.top_k = top_k
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.use_fp8_block_scaling = use_fp8_block_scaling

        instance_key = (x_dtype, weight_dtype, output_dtype,
                        use_fp8_block_scaling)

        if instance_key not in MoERunner._runner_dict:
            MoERunner._runner_dict[
                instance_key] = torch.classes.trtllm.FusedMoeRunner(
                    x_dtype, weight_dtype, output_dtype, use_fp8_block_scaling)
        self._fused_moe_runner = MoERunner._runner_dict[instance_key]
        self._is_nvfp4 = weight_dtype == torch.int64

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
    ) -> List[int]:
        x, fc2_expert_weights, min_latency_mode_tensor = inputs
        min_latency_mode = min_latency_mode_tensor.size(0) == 1
        m = x.shape[0]

        # Only profile m <= 128 for min latency mode = True
        # Profile all valid buckets for min latency mode = False
        # TODO: min_latency_mode = True will cause the following error:
        # Cannot profile configuration 4: Cutlass GEMM Tactic
        # [TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
        # Should be fixed in the moe_kernels in the future.
        invalid = (m > 128
                   and min_latency_mode) or (m <= 128 and min_latency_mode and
                                             (not self._is_nvfp4))

        return [] if invalid else list(
            range(self._fused_moe_runner.get_tactic_num()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        gemm_idx: int = 0,
        tactic: int = -1,
        do_preparation: bool = False,
    ):
        x, fc2_expert_weights, min_latency_mode_tensor = inputs
        min_latency_mode = min_latency_mode_tensor.size(0) == 1
        # determine if we should use min latency mode according to the profiled seq len
        self._fused_moe_runner.run_gemm_profile(
            x,
            fc2_expert_weights,
            self.top_k,
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
            min_latency_mode,
            gemm_idx,
            tactic,
            do_preparation,
        )


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
    use_fp8_block_scaling: bool = False,
    min_latency_mode: bool = False,
) -> List[torch.Tensor]:

    tuner = AutoTuner.get()

    # TODO: only profile for min_latency_mode = False due to the error in the moe_kernels
    tuning_config = TuningConfig(dynamic_tensors=(
        # input, dim 0, all valid buckets, map a seq_len to power of 2 bucket index
        (0, 0, ((16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4,
                 2, 1), next_positive_power_of_2)),
        # min_latency_tensor, dim 0, (0 for False, 1 for True), map to it self
        (2, 0, ((0, ), lambda x: x)),
    ))

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
        use_fp8_block_scaling=use_fp8_block_scaling,
    )

    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1",
        [moe_runner],
        tuning_config,
        [input, fc2_expert_weights, min_latency_tensor],
        gemm_idx=1,
    )

    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2",
        [moe_runner],
        tuning_config,
        [input, fc2_expert_weights, min_latency_tensor],
        gemm_idx=2,
    )

    run_moe = moe_runner._fused_moe_runner.run_moe_min_latency if min_latency_mode else moe_runner._fused_moe_runner.run_moe
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
    use_fp8_block_scaling: bool = False,
    min_latency_mode: bool = False,
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
    _runner_dict = dict()

    def __init__(
        self,
        sf_use_ue8m0: bool,
        to_userbuffers: bool,
        output_dtype: torch.dtype,
    ):
        self.sf_use_ue8m0 = sf_use_ue8m0
        self.output_dtype = output_dtype
        self.to_userbuffers = to_userbuffers
        if output_dtype not in NVFP4GemmRunner._runner_dict:
            NVFP4GemmRunner._runner_dict[
                output_dtype] = torch.classes.trtllm.FP4GemmRunner(output_dtype)
        self._nvfp4_gemm_runner = NVFP4GemmRunner._runner_dict[output_dtype]

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
    ) -> List[int]:
        return list(range(self._nvfp4_gemm_runner.get_num_configs()))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
    ) -> torch.Tensor:
        mat1, mat2, mat1_scale, mat2_scale, global_scale = inputs
        return self._nvfp4_gemm_runner.run_gemm(
            mat1,
            mat2,
            mat1_scale,
            mat2_scale,
            global_scale,
            self.sf_use_ue8m0,
            self.to_userbuffers,
            tactic,
        )


def fp4_scale_dims(input_shapes: List[torch.Tensor], sf_vec_size: int = 16):
    """Calculate the dimensions of the fp4 scale tensor.

    The shape of act_fp4 determines the dimensions of the fp4 scale tensor. And due to the first dimension of act_fp4 is dynamic and will be tuned in Autotuner, we should always keep these associated dimensions aligned.
    """
    out_shape, scale_shape = fp4_utils.get_fp4_shape(input_shapes[0],
                                                     sf_vec_size)
    return scale_shape * 2


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

    tuning_config = TuningConfig(
        dynamic_tensors=((0, 0, (get_last_power_of_2_num_tokens_buckets,
                                 last_positive_power_of_2)), ),
        constraints=((2, 0, fp4_scale_dims), ),
    )

    # allocate workspace for profiling
    nvfp4_gemm_runner = NVFP4GemmRunner(sf_use_ue8m0, to_userbuffers,
                                        output_dtype)

    _, best_tactic = tuner.choose_one(
        "trtllm::nvfp4_gemm::gemm",
        [nvfp4_gemm_runner],
        tuning_config,
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
