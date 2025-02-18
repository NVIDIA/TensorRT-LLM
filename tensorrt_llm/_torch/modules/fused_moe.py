from typing import Dict, List, NamedTuple, Optional

import torch
from torch import nn

from ...quantization.utils.fp4_utils import float4_sf_dtype
from ..model_config import ModelConfig
from .linear import ParallelConfig, TensorParallelMode, load_weight_shard

# The declarations aligns with moe_kernels.h
# pack inputs into int64, e.g. 4 x bf16 input values
FUSED_MOE_NVFP4_INPUT_DTYPE = torch.int64
# pack weights into int64, e.g. 16 x nvfp4 weight values
FUSED_MOE_NVFP4_WEIGHT_DTYPE = torch.int64
# pack weight block scales into int32, e.g. 4 x fp8 weight values
FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32

# The declarations aligns with moe_kernels.h MOEExpertScaleNormalizationMode
from enum import IntEnum


class MOEExpertScaleNormalizationMode(IntEnum):
    NONE = 0
    RENORMALIZE = 1
    SPARSE_MIXER = 2
    DEVICE_LIMITED = 3
    DEVICE_LIMITED_RENORM = 4


def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1

    return 1 << (x - 1).bit_length()


def get_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = next_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = 1
    while m <= max_num_tokens:
        num_token_buckets.append(m)
        m *= 2

    return num_token_buckets


class FusedMoE(nn.Module):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        tune_max_num_tokens (int): Maximum number of tokens for performance tuning.
        model_config (ModelConfig): Configuration object for the model.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        tune_max_num_tokens: int = 8192,
        model_config: ModelConfig = ModelConfig(),
        normalization_mode: int = MOEExpertScaleNormalizationMode.RENORMALIZE,
    ):
        from tensorrt_llm._torch.distributed import AllReduce

        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dtype = dtype
        self.reduce_results = reduce_results
        # could be modified later
        self.quant_config = model_config.quant_config

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank

        # All ranks participate in allreduce regardless of EP/TP combination
        self.parallel_config = ParallelConfig(
            tensor_parallel_rank=model_config.mapping.tp_rank,
            tensor_parallel_size=model_config.mapping.tp_size,
            gpus_per_node=model_config.mapping.gpus_per_node)

        self.all_reduce = AllReduce(self.parallel_config)

        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.expert_size_per_partition = num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.expert_size_per_partition
        self.expert_end = min(
            self.expert_start + self.expert_size_per_partition,
            self.num_experts)

        self.tune_max_num_tokens = tune_max_num_tokens
        self.has_been_profiled = False

        self.normalization_mode = normalization_mode
        self._weights_created = False
        if not model_config.skip_create_weights:
            self.create_weights()

    def create_weights(self):
        if self._weights_created:
            return
        device = torch.device('cuda')
        weight_dtype = self.dtype
        w3_w1_weight_shape = (self.expert_size_per_partition,
                              self.intermediate_size_per_partition * 2,
                              self.hidden_size)
        w2_weight_shape = (
            self.expert_size_per_partition,
            self.hidden_size,
            self.intermediate_size_per_partition,
        )

        if self.quant_config and self.quant_config.quant_mode.has_any_quant():
            qc = self.quant_config
            if qc.quant_mode.has_fp8_qdq():
                weight_dtype = torch.float8_e4m3fn

                fc31_dequant = nn.Parameter(torch.empty(
                    self.expert_size_per_partition,
                    dtype=torch.float32,
                    device=device),
                                            requires_grad=False)
                self.register_parameter("fc31_dequant", fc31_dequant)

                fc2_dequant = nn.Parameter(torch.empty(
                    self.expert_size_per_partition,
                    dtype=torch.float32,
                    device=device),
                                           requires_grad=False)
                self.register_parameter("fc2_dequant", fc2_dequant)

                fc2_quant = nn.Parameter(torch.tensor(1.,
                                                      dtype=torch.float32,
                                                      device=device),
                                         requires_grad=False)
                self.register_parameter("fc2_quant", fc2_quant)

                fc31_input_dequant = nn.Parameter(torch.tensor(
                    1., dtype=torch.float32, device=device),
                                                  requires_grad=False)
                self.register_parameter("fc31_input_dequant",
                                        fc31_input_dequant)

            elif qc.quant_mode.has_fp8_block_scales():
                weight_dtype = torch.float8_e4m3fn

                w3_w1_weight_scaling_factor = nn.Parameter(torch.empty(
                    (self.expert_size_per_partition, w3_w1_weight_shape[1] //
                     128, w3_w1_weight_shape[2] // 128),
                    dtype=torch.float32,
                    device=device),
                                                           requires_grad=False)
                self.register_parameter("w3_w1_weight_scaling_factor",
                                        w3_w1_weight_scaling_factor)

                w2_weight_scaling_factor = nn.Parameter(torch.empty(
                    (self.expert_size_per_partition, w2_weight_shape[1] // 128,
                     w2_weight_shape[2] // 128),
                    dtype=torch.float32,
                    device=device),
                                                        requires_grad=False)
                self.register_parameter("w2_weight_scaling_factor",
                                        w2_weight_scaling_factor)

            elif qc.quant_mode.has_nvfp4():
                weight_dtype = FUSED_MOE_NVFP4_WEIGHT_DTYPE
                scaling_vector_size = 16
                # Divide by 16 because we use int64 to pack 16 fp4 values
                w3_w1_weight_shape = (self.expert_size_per_partition,
                                      self.intermediate_size_per_partition * 2,
                                      self.hidden_size // 16)
                w2_weight_shape = (self.expert_size_per_partition,
                                   self.hidden_size,
                                   self.intermediate_size_per_partition // 16)

                # Divide by 4 because we use int32 to pack 4 fp8 values
                # column parallel
                w3_w1_weight_scale = nn.Parameter(torch.ones(
                    self.expert_size_per_partition,
                    self.intermediate_size_per_partition * 2,
                    self.hidden_size // scaling_vector_size // 4,
                    dtype=FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE,
                    device=device),
                                                  requires_grad=False)
                self.register_parameter("w3_w1_weight_scale",
                                        w3_w1_weight_scale)

                # row parallel
                w2_weight_scale = nn.Parameter(
                    torch.ones(self.expert_size_per_partition,
                               self.hidden_size,
                               self.intermediate_size_per_partition //
                               scaling_vector_size // 4,
                               dtype=FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE,
                               device=device),
                    requires_grad=False)
                self.register_parameter("w2_weight_scale", w2_weight_scale)

                fc31_input_scale = nn.Parameter(torch.tensor(
                    1., dtype=torch.float32, device=device),
                                                requires_grad=False)
                self.register_parameter("fc31_input_scale", fc31_input_scale)

                fc2_input_scale = nn.Parameter(torch.tensor(1.,
                                                            dtype=torch.float32,
                                                            device=device),
                                               requires_grad=False)
                self.register_parameter("fc2_input_scale", fc2_input_scale)

                fc31_alpha = nn.Parameter(torch.ones(
                    self.expert_size_per_partition,
                    dtype=torch.float32,
                    device=device),
                                          requires_grad=False)
                self.register_parameter("fc31_alpha", fc31_alpha)

                fc2_alpha = nn.Parameter(torch.ones(
                    self.expert_size_per_partition,
                    dtype=torch.float32,
                    device=device),
                                         requires_grad=False)
                self.register_parameter("fc2_alpha", fc2_alpha)

            else:
                # TODO: support other quant mode
                raise ValueError(
                    f"unsupported quantization mode: {qc.quant_mode}")

        # Fused gate_up_proj (column parallel)
        w3_w1_weight = nn.Parameter(torch.empty(w3_w1_weight_shape,
                                                dtype=weight_dtype,
                                                device=device),
                                    requires_grad=False)
        self.register_parameter("w3_w1_weight", w3_w1_weight)

        # down_proj (row parallel)
        w2_weight = nn.Parameter(torch.empty(w2_weight_shape,
                                             dtype=weight_dtype,
                                             device=device),
                                 requires_grad=False)
        self.register_parameter("w2_weight", w2_weight)
        self._weights_created = True

    def forward(self, x: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype

        quant_scales = None
        use_fp8_block_scaling = False

        if self.quant_config and self.quant_config.quant_mode.has_any_quant():
            if self.quant_config.quant_mode.has_fp8_qdq():
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
                quant_scales = FusedMoEQuantScalesFP8(
                    fc1_dequant=self.fc31_dequant,
                    fc2_quant=self.fc2_quant,
                    fc2_dequant=self.fc2_dequant,
                    fc1_input_dequant=self.fc31_input_dequant,
                )
            elif self.quant_config.quant_mode.has_nvfp4():
                # nvfp4 input quantization happens inside the custom op
                x = x.view(FUSED_MOE_NVFP4_INPUT_DTYPE)
                quant_scales = FusedMoEQuantScalesNVFP4(
                    fc1_act_global=self.fc31_input_scale,
                    fc1_weight_block=self.w3_w1_weight_scale,
                    fc1_global=self.fc31_alpha,
                    fc2_act_global=self.fc2_input_scale,
                    fc2_weight_block=self.w2_weight_scale,
                    fc2_global=self.fc2_alpha,
                )
            elif self.quant_config.quant_mode.has_fp8_block_scales():
                quant_scales = FusedMoEQuantScalesFP8BlockScales(
                    fc_weight_scales=self.w3_w1_weight_scaling_factor,
                    proj_weight_scales=self.w2_weight_scaling_factor,
                )
                use_fp8_block_scaling = True

            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        if not self.has_been_profiled:
            self.profiler = torch.classes.trtllm.FusedMoeProfiler.get_instance(
                x.dtype, self.w3_w1_weight.dtype, output_dtype,
                use_fp8_block_scaling)
            self.profiler.run_profile(
                self.w2_weight, self.top_k, self.tp_size, self.tp_rank,
                self.ep_size, self.ep_rank,
                get_power_of_2_num_tokens_buckets(self.tune_max_num_tokens))
            self.has_been_profiled = True

        profile_ids = self.profiler.get_profile_ids(
            next_positive_power_of_2(x.shape[0]), self.w2_weight, self.top_k,
            self.num_experts)

        final_hidden_states = torch.ops.trtllm.fused_moe(
            x,
            router_logits.float(),
            self.w3_w1_weight,
            self.w2_weight,
            output_dtype,
            self.top_k,
            quant_scales=quant_scales,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            profile_ids=profile_ids,
            normalization_mode=self.normalization_mode,
            use_fp8_block_scaling=use_fp8_block_scaling,
        )

        if self.reduce_results and self.parallel_config.tensor_parallel_size > 1:
            final_hidden_states = self.all_reduce(final_hidden_states)

        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        def load_expert_w3_w1_weight(w1_weight, w3_weight,
                                     dst_w3_w1_weight: torch.Tensor):
            w1_weight_shard = load_weight_shard(w1_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)
            w3_weight_shard = load_weight_shard(w3_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)

            expert_w3_weight = dst_w3_w1_weight.narrow(
                dim=0, start=0, length=self.intermediate_size_per_partition)
            expert_w3_weight.copy_(w3_weight_shard.view(expert_w3_weight.dtype))

            expert_w1_weight = dst_w3_w1_weight.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition)
            expert_w1_weight.copy_(w1_weight_shard.view(expert_w1_weight.dtype))

        def load_expert_w2_weight(w2_weight, dst_w2_weight: torch.Tensor):
            w2_weight_shard = load_weight_shard(w2_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.ROW)
            dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype))

        for expert_id in range(self.expert_start, self.expert_end):
            w1_weight = weights[f"{expert_id}.w1.weight"]
            w3_weight = weights[f"{expert_id}.w3.weight"]
            w2_weight = weights[f"{expert_id}.w2.weight"]

            expert_idx = expert_id - self.expert_start

            load_expert_w3_w1_weight(w1_weight, w3_weight,
                                     self.w3_w1_weight.data[expert_idx])
            load_expert_w2_weight(w2_weight, self.w2_weight.data[expert_idx])

        if self.quant_config and self.quant_config.quant_mode.has_any_quant():
            if self.quant_config.quant_mode.has_fp8_qdq():
                self._load_fp8_qdq_scales(weights)
            elif self.quant_config.quant_mode.has_nvfp4():
                self._load_nvfp4_scales(weights)
            elif self.quant_config.quant_mode.has_fp8_block_scales():
                self._load_fp8_block_scales_scales(weights)
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

    def _load_fp8_block_scales_scales(self, weights: Dict):
        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.ROW)
            for expert_id in range(self.expert_start, self.expert_end)
        ]

        w2_scales = torch.stack(all_w2_scales)
        self.w2_weight_scaling_factor.data.copy_(w2_scales)

        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in range(self.expert_start, self.expert_end)
        ]

        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.weight_scale_inv"],
                              self.tp_size, self.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in range(self.expert_start, self.expert_end)
        ]

        w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)
        self.w3_w1_weight_scaling_factor.data.copy_(w3_w1_scales)

    def _load_fp8_qdq_scales(self, weights: Dict):
        # Step1: Load input scales.
        def load_expert_fc31_input_scale_fp8_qdq(
                w1_input_scale, w3_input_scale,
                dst_fc31_input_scale: torch.Tensor):
            dst_fc31_input_scale.copy_(
                max(w1_input_scale[...].reshape([]),
                    w3_input_scale[...].reshape([])))

        def load_expert_fc2_input_scale_fp8_qdq(
                w2_input_scale, dst_fc2_input_scale: torch.Tensor):
            dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

        tmp_fc31_input_scale = torch.empty(self.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(self.num_experts, dtype=torch.float32)
        for expert_id in range(self.num_experts):
            w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
            w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
            w2_input_scale = weights[f"{expert_id}.w2.input_scale"]

            load_expert_fc31_input_scale_fp8_qdq(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])

            load_expert_fc2_input_scale_fp8_qdq(w2_input_scale,
                                                tmp_fc2_input_scale[expert_id])

        # max_fc31_input_scale is the maximum of all w1 input scales and w3 input scales.
        # It's used to quantize fc31 input inside the MOE op
        max_fc31_input_scale = tmp_fc31_input_scale.max()
        # max_fc2_input_scale is the maximum of all w2 input scales.
        max_fc2_input_scale = tmp_fc2_input_scale.max()

        # Step2: Load weight scales and requantize w3_w1_weight.
        tmp_w3_w1_weight_scale = torch.empty(self.expert_size_per_partition,
                                             dtype=torch.float32)
        tmp_w2_weight_scale = torch.empty(self.expert_size_per_partition,
                                          dtype=torch.float32)

        def load_expert_w3_w1_weight_scale_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale: torch.Tensor):
            w1_weight_scale = w1_weight_scale[...].reshape([])
            w3_weight_scale = w3_weight_scale[...].reshape([])
            dst_w3_w1_weight_scale.copy_(max(w1_weight_scale, w3_weight_scale))

        def requantize_expert_w3_w1_weight_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight: torch.Tensor):
            w1_weight_scale = w1_weight_scale[...].reshape([])
            w3_weight_scale = w3_weight_scale[...].reshape([])
            max_w3_w1_weight_scale = max(w1_weight_scale, w3_weight_scale)

            w3_weight = dst_w3_w1_weight.narrow(
                dim=0, start=0, length=self.intermediate_size_per_partition).to(
                    dtype=self.dtype)
            w1_weight = dst_w3_w1_weight.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition).to(
                    dtype=self.dtype)
            dequant_w3_weight = w3_weight * w3_weight_scale
            dequant_w1_weight = w1_weight * w1_weight_scale
            requant_w3_weight = (dequant_w3_weight / max_w3_w1_weight_scale).to(
                torch.float8_e4m3fn)
            requant_w1_weight = (dequant_w1_weight / max_w3_w1_weight_scale).to(
                torch.float8_e4m3fn)

            dst_w3_w1_weight.narrow(
                dim=0, start=0,
                length=self.intermediate_size_per_partition).copy_(
                    requant_w3_weight)
            dst_w3_w1_weight.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition).copy_(
                    requant_w1_weight)

        def load_expert_w2_weight_scale_fp8(w2_weight_scale,
                                            dst_w2_weight_scale: torch.Tensor):
            dst_w2_weight_scale.copy_(w2_weight_scale[...].reshape([]))

        for expert_id in range(self.expert_start, self.expert_end):
            w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
            w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
            w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]

            expert_idx = expert_id - self.expert_start

            load_expert_w3_w1_weight_scale_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                tmp_w3_w1_weight_scale[expert_idx])

            requantize_expert_w3_w1_weight_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                self.w3_w1_weight.data[expert_idx])

            load_expert_w2_weight_scale_fp8(w2_weight_scale,
                                            tmp_w2_weight_scale[expert_idx])

        # Step3: calculate and store final loaded weights
        self.fc31_dequant.data.copy_(tmp_w3_w1_weight_scale *
                                     max_fc31_input_scale)
        self.fc2_quant.data.copy_(max_fc2_input_scale.reciprocal())
        self.fc2_dequant.data.copy_(tmp_w2_weight_scale * max_fc2_input_scale)
        self.fc31_input_dequant.data.copy_(max_fc31_input_scale)

    def _load_nvfp4_scales(self, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(self.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(self.num_experts, dtype=torch.float32)

        def load_expert_fc31_input_scale_nvfp4(
                w1_input_scale, w3_input_scale,
                dst_fc31_input_scale: torch.Tensor):
            w1_input_scale = w1_input_scale[...].reshape([])
            w3_input_scale = w3_input_scale[...].reshape([])
            assert torch.allclose(
                w1_input_scale,
                w3_input_scale), "w1_input_scale != w3_input_scale"
            dst_fc31_input_scale.copy_(w1_input_scale)

        def load_expert_fc2_input_scale_nvfp4(
                w2_input_scale, dst_fc2_input_scale: torch.Tensor):
            dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

        for expert_id in range(self.num_experts):
            w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
            w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
            w2_input_scale = weights[f"{expert_id}.w2.input_scale"]

            load_expert_fc31_input_scale_nvfp4(w1_input_scale, w3_input_scale,
                                               tmp_fc31_input_scale[expert_id])
            load_expert_fc2_input_scale_nvfp4(w2_input_scale,
                                              tmp_fc2_input_scale[expert_id])

        # fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
        self.fc31_input_scale.data.copy_(
            tmp_fc31_input_scale.max().reciprocal())
        # fc2_input_scale is the reciprocal of the maximum of all w2 input scales.
        self.fc2_input_scale.data.copy_(tmp_fc2_input_scale.max().reciprocal())

        # Step2: Load weight block scales and alphas.
        def load_expert_w3_w1_weight_scale_nvfp4(
                w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale: torch.Tensor):
            w1_weight_scale = load_weight_shard(w1_weight_scale, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)
            w3_weight_scale = load_weight_shard(w3_weight_scale, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)

            dst_w3_weight_scale = dst_w3_w1_weight_scale.narrow(
                dim=0, start=0, length=self.intermediate_size_per_partition)
            dst_w3_weight_scale.copy_(
                w3_weight_scale.view(dst_w3_weight_scale.dtype))

            dst_w1_weight_scale = dst_w3_w1_weight_scale.narrow(
                dim=0,
                start=self.intermediate_size_per_partition,
                length=self.intermediate_size_per_partition)
            dst_w1_weight_scale.copy_(
                w1_weight_scale.view(dst_w1_weight_scale.dtype))

            orig_shape = dst_w3_w1_weight_scale.shape
            dst_w3_w1_weight_scale.copy_(
                torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                    dst_w3_w1_weight_scale.cpu().view(float4_sf_dtype)).view(
                        FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE).reshape(
                            orig_shape))

        def load_expert_w2_weight_scale_nvfp4(
                w2_weight_scale, dst_w2_weight_scale: torch.Tensor):
            w2_weight_scale = load_weight_shard(w2_weight_scale, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.ROW)
            dst_w2_weight_scale.copy_(
                w2_weight_scale.view(dst_w2_weight_scale.dtype))

            orig_shape = dst_w2_weight_scale.shape
            dst_w2_weight_scale.copy_(
                torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                    dst_w2_weight_scale.cpu().view(float4_sf_dtype)).view(
                        FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE).reshape(
                            orig_shape))

        def load_expert_fc31_alpha_nvfp4(w1_weight_scale_2, w3_weight_scale_2,
                                         final_fc31_input_scale: torch.Tensor,
                                         dst_fc31_alpha: torch.Tensor):
            w1_weight_scale_2 = w1_weight_scale_2[...].reshape([])
            w3_weight_scale_2 = w3_weight_scale_2[...].reshape([])
            assert torch.allclose(
                w1_weight_scale_2,
                w3_weight_scale_2), "w1_weight_scale_2 != w3_weight_scale_2"

            w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2
            dst_fc31_alpha.copy_(
                1.0 / (final_fc31_input_scale * w3_w1_weight_scale_2))

        def load_expert_fc2_alpha_nvfp4(w2_weight_scale_2,
                                        final_fc2_input_scale: torch.Tensor,
                                        dst_w2_alpha: torch.Tensor):
            w2_weight_scale_2 = 1.0 / w2_weight_scale_2[...].reshape([])
            dst_w2_alpha.copy_(1.0 /
                               (final_fc2_input_scale * w2_weight_scale_2))

        for expert_id in range(self.expert_start, self.expert_end):
            w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
            w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
            w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
            w1_weight_scale_2 = weights[f"{expert_id}.w1.weight_scale_2"]
            w3_weight_scale_2 = weights[f"{expert_id}.w3.weight_scale_2"]
            w2_weight_scale_2 = weights[f"{expert_id}.w2.weight_scale_2"]

            expert_idx = expert_id - self.expert_start

            load_expert_w3_w1_weight_scale_nvfp4(
                w1_weight_scale, w3_weight_scale,
                self.w3_w1_weight_scale.data[expert_idx])
            load_expert_w2_weight_scale_nvfp4(
                w2_weight_scale, self.w2_weight_scale.data[expert_idx])

            load_expert_fc31_alpha_nvfp4(w1_weight_scale_2, w3_weight_scale_2,
                                         self.fc31_input_scale.data,
                                         self.fc31_alpha.data[expert_idx])
            load_expert_fc2_alpha_nvfp4(w2_weight_scale_2,
                                        self.fc2_input_scale.data,
                                        self.fc2_alpha.data[expert_idx])


class FusedMoEQuantScalesFP8(NamedTuple):
    fc1_dequant: torch.Tensor
    fc2_quant: torch.Tensor
    fc2_dequant: torch.Tensor
    fc1_input_dequant: torch.Tensor


class FusedMoEQuantScalesNVFP4(NamedTuple):
    fc1_act_global: torch.Tensor
    fc1_weight_block: torch.Tensor
    # fc1_global_scale = 1.0 / (fc1_weight_global_scale * fc1_act_global_scale)
    fc1_global: torch.Tensor

    fc2_act_global: torch.Tensor
    fc2_weight_block: torch.Tensor
    # fc2_global_scale = 1.0 / (fc2_weight_global_scale * fc2_act_global_scale)
    fc2_global: torch.Tensor


class FusedMoEQuantScalesFP8BlockScales(NamedTuple):
    fc_weight_scales: torch.Tensor
    proj_weight_scales: torch.Tensor
