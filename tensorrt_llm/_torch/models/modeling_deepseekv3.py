# --------------------------------------------------
# Portions of this code were derived from DeepSeek‑V3:
#   https://github.com/deepseek-ai/DeepSeek-V3
#
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.utils import enable_llm_debug

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduce, allgather)
from ..model_config import ModelConfig
from ..models.modeling_utils import ModelConfig
from ..modules.attention import MLA
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import BaseMoeRoutingMethod, FusedMoE
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import MTPEagleWorker, MTPSpecMetadata, MTPWorker
from ..utils import (AuxStreamType, EventType, Fp4QuantizedTensor,
                     disable_fp4_allgather)
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, register_auto_model)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor,
                   s: torch.Tensor,
                   block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(
    ), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),
                         triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


class DeepseekV3MTPHead(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(self, hidden_states: torch.Tensor, lm_head: Linear,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        if attn_metadata is not None:
            last_tokens = torch.cumsum(
                attn_metadata.seq_lens_cuda,
                dim=0,
                dtype=torch.long,
            ) - 1
            last_token_hidden_states = hidden_states[last_tokens]
        else:
            last_token_hidden_states = hidden_states[-1].unsqueeze(0)

        logits = lm_head(last_token_hidden_states)
        return logits


class DeepseekV3Attention(MLA):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config
        predicted_tokens_per_seq = model_config.spec_config.num_nextn_predict_layers + 1 if model_config.spec_config is not None else 1
        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         qk_rope_head_dim=config.qk_rope_head_dim,
                         qk_nope_head_dim=config.qk_nope_head_dim,
                         q_lora_rank=config.q_lora_rank,
                         kv_lora_rank=config.kv_lora_rank,
                         v_head_dim=config.v_head_dim,
                         predicted_tokens_per_seq=predicted_tokens_per_seq,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=False,
                         pos_embd_params=PositionalEmbeddingParams(
                             type=PositionEmbeddingType.yarn,
                             rope=RopeParams.from_config(config),
                             is_neox=False,
                         ),
                         layer_idx=layer_idx,
                         dtype=config.torch_dtype,
                         config=model_config,
                         aux_stream=aux_stream)


class Deepseekv3RoutingImpl():

    def __init__(
        self,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        is_fused: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.topk_group = topk_group
        self.n_group = n_group
        self.routed_scaling_factor = routed_scaling_factor
        self.is_fused = is_fused

    def noaux_tc(self, logits, e_score_correction_bias):
        n_group = self.n_group
        scores = F.sigmoid(logits)
        scores_with_bias = scores + e_score_correction_bias
        scores_shape = list(scores_with_bias.shape)

        if enable_llm_debug():
            has_nan = torch.isnan(scores_with_bias).any()
            if has_nan:
                warnings.warn(
                    "Detected NAN in the tensor scores_with_bias. Please check if it matches the expectation."
                )

        if not self.is_fused:
            group_scores = torch.sum(torch.topk(
                scores_with_bias.view(scores_shape[:-1] +
                                      [n_group, scores_shape[-1] // n_group]),
                k=2,
                dim=-1,
                largest=True,
                sorted=True)[0],
                                     dim=-1)
            _, group_idx = torch.topk(group_scores,
                                      k=self.topk_group,
                                      dim=-1,
                                      largest=True,
                                      sorted=True)
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(-1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(
                scores_shape[:-1] +
                [n_group, scores_shape[-1] // n_group]).reshape(scores_shape)
            scores_with_bias = scores_with_bias * score_mask
            _, topk_idx = torch.topk(scores_with_bias,
                                     k=self.top_k,
                                     dim=-1,
                                     largest=True,
                                     sorted=True)
            new_mask = torch.zeros_like(scores)
            new_mask.scatter_(-1, topk_idx, 1)
            scores = scores * new_mask
            score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
            scores = scores / score_sum * \
                self.routed_scaling_factor
            topk_values, topk_indices = torch.topk(scores,
                                                   k=self.top_k,
                                                   dim=-1,
                                                   largest=True)
            return topk_values, topk_indices
        else:
            topk_values, topk_indices = torch.ops.trtllm.noaux_tc_op(
                scores, scores_with_bias, n_group, self.topk_group, self.top_k,
                self.routed_scaling_factor)
            return topk_values, topk_indices

    def apply(
        self, logits: torch.Tensor, e_score_correction_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_values, topk_indices = self.noaux_tc(logits,
                                                  e_score_correction_bias)
        return topk_indices.to(torch.int32), topk_values.to(torch.float32)


class DeepseekV3Gate(BaseMoeRoutingMethod):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        dtype: Optional[torch.dtype] = None,
        fuse_routing_kernel: bool = True,
        apply_routing: bool = False,
        moe_backend: str = 'CUTLASS',
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.moe_backend = moe_backend
        if moe_backend == 'TRTLLM':
            bias_dtype = torch.bfloat16
        else:
            bias_dtype = torch.float32

        self.e_score_correction_bias = nn.Parameter(torch.empty(
            (num_experts), dtype=bias_dtype),
                                                    requires_grad=False)

        assert not apply_routing, "DeepseekV3Gate routing is called inside MoE"

        # TODO: e_score_correction_bias belongs in this gate class but is required by the routing impl.
        # To avoid weight-loading issues, we treat this gate as the BaseMoeRoutingMethod and dispatch to the routing impl.
        # This is a temporary hack that should be refactored later.
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_fused=fuse_routing_kernel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router gemm
        logits = torch.ops.trtllm.cublas_mm(hidden_states,
                                            self.weight.t(),
                                            bias=None,
                                            out_dtype=torch.float32)
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

        self.e_score_correction_bias.copy_(
            weights[0]["e_score_correction_bias"][:].to(
                self.e_score_correction_bias.dtype))

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # topk routing
        return self.routing_impl.apply(logits, self.e_score_correction_bias)

    @property
    def routing_method(self) -> BaseMoeRoutingMethod:
        return self

    def get_experts_per_token(self):
        return self.routing_impl.top_k


class Deepseekv3MoE(nn.Module):

    def __init__(self,
                 *,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 intermediate_size: int,
                 shared_expert_intermediate_size: int,
                 aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig()):
        from ..distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.use_dp = model_config.mapping.enable_attention_dp
        self.enable_alltoall = Deepseekv3MoE.should_enable_alltoall(
            model_config, top_k)
        if self.enable_alltoall:
            MnnvlMemory.initialize()
        self.gate = DeepseekV3Gate(
            hidden_size,
            num_experts,
            top_k=top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=dtype,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=model_config.moe_backend)
        self.experts = FusedMoE(
            num_experts=num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=
            False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
            model_config=model_config,
            aux_stream=aux_stream_dict[AuxStreamType.MoeChunkingOverlap],
            enable_alltoall=self.enable_alltoall)

        self.mapping = model_config.mapping

        # FIXME: incompatible with mixed quantization mode (including excluding modules from quantization)
        block_size = 1
        if model_config.quant_config and model_config.quant_config.group_size is not None:
            block_size = model_config.quant_config.group_size

        shared_tp_size, self.shared_output_scale = self._compute_shared_expert_tp_size(
            shared_expert_intermediate_size, block_size)

        self.shared_experts = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=shared_tp_size,
            reduce_output=False)

        self.allreduce = AllReduce(self.mapping)
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeShared]
        }

    def _compute_shared_expert_tp_size(self, intermediate_size: int,
                                       block_size: int) -> int:
        """
        In the case of Deepseek-R1, the TP size of MLP is capped by intermediate_size // block_size.
        For example, when the intermediate_size is 2048 and block scaling size is 128,
        TP sizes are limited to {1, 2, 4, 8, 16} because of 2048/128 = 16.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            int: The computed tp_size.
        """

        assert intermediate_size % block_size == 0, "intermediate_size must be divisible by block_size."

        shared_output_scale = None
        # The block scale size is 128, which requires shared_expert_intermediate_size to be divisible by 128.
        if self.use_dp:
            # If using attention DP, the shared experts also use DP instead of TP.
            shared_tp_size = 1
        else:
            # Due to the restriction of block scale size (i.e., 128), the supported TP sizes only include 1, 2, 4, 8, and 16.
            # The math.gcd operation ensures that shared_tp_size falls in the supported TP sizes.
            shared_tp_size = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )
            # If shared_tp_size has been overridden, the output of shared experts needs to be scaled down accordingly before all-reduce.
            if shared_tp_size != self.mapping.tp_size:
                shared_output_scale = shared_tp_size / self.mapping.tp_size

        return shared_tp_size, shared_output_scale

    @staticmethod
    def should_enable_alltoall(model_config: ModelConfig, top_k: int) -> bool:
        if not model_config.mapping.enable_attention_dp:
            return False

        if model_config.mapping.tp_size == 1:
            return False

        if not MnnvlMemory.supports_mnnvl():
            return False

        if os.environ.get("TRTLLM_MOE_DISABLE_ALLTOALLV", "0") == "1":
            return False

        if model_config.mapping.moe_ep_size <= top_k:
            return False

        return True

    def compute_routed_output(self, hidden_states, hidden_states_fp4,
                              all_rank_num_tokens, cutlass_min_latency_mode):
        # max-throughput
        use_dp_padding = False
        if self.use_dp and self.mapping.tp_size > 1:
            # FP4 all_gather moves this bf16 allgather in to after topk and fp4 quantization
            # to reduce allreduce BW
            if disable_fp4_allgather() and not self.enable_alltoall:
                hidden_states = allgather(hidden_states,
                                          self.mapping,
                                          dim=0,
                                          sizes=all_rank_num_tokens)
            elif not self.experts.is_cutlass() or (not self.experts.has_fp8_qdq
                                                   and self.experts.has_nvfp4):
                # Use padding when not using the cutlass path or when x_sf in self.experts is not None
                use_dp_padding = True
                max_num_token = max(all_rank_num_tokens)
                hidden_states = torch.nn.functional.pad(
                    hidden_states,
                    (0, 0, 0, max_num_token - hidden_states.shape[0]))

        router_logits = self.gate(hidden_states)

        routed_output = self.experts(hidden_states_fp4 or hidden_states,
                                     router_logits,
                                     cutlass_min_latency_mode,
                                     output_dtype=hidden_states.dtype,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=use_dp_padding)

        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Optional[Fp4QuantizedTensor] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        cutlass_min_latency_mode: Optional[bool] = False,
    ) -> torch.Tensor:
        if cutlass_min_latency_mode:
            assert not self.use_dp

        def _compute_shared_output():
            shared_output = self.shared_experts(hidden_states_fp4
                                                or hidden_states)
            if self.shared_output_scale is not None:
                shared_output *= self.shared_output_scale
            return shared_output

        def _compute_routed_output():
            routed_output = self.compute_routed_output(
                hidden_states, hidden_states_fp4, all_rank_num_tokens,
                cutlass_min_latency_mode)
            return routed_output

        shared_output, routed_output = maybe_execute_in_parallel(
            _compute_shared_output, _compute_routed_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared], self.aux_stream)

        if cutlass_min_latency_mode:
            return [shared_output, *routed_output]
        else:
            assert shared_output.size() == routed_output.size(
            ), f'unmatched tensor shape'
            final_hidden_states = shared_output + routed_output
            if not self.use_dp and self.mapping.tp_size > 1:
                final_hidden_states = self.allreduce(
                    final_hidden_states,
                    all_reduce_params=final_all_reduce_params)

            return final_hidden_states


class DeepseekV3DecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.mapping = model_config.mapping
        mapping = self.mapping

        self.self_attn = DeepseekV3Attention(
            model_config,
            layer_idx=layer_idx,
            aux_stream=aux_stream_dict[AuxStreamType.Attention])
        self.enable_attention_dp = mapping.enable_attention_dp

        self.mlp_tp_size = mapping.tp_size

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get(
            "TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        # FIXME: incompatible with mixed quantization mode (including excluding modules from quantization)
        self.is_nvfp4 = model_config.quant_config.layer_quant_mode.has_nvfp4()
        has_tp = mapping.has_tp()

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):

            self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
            self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION

            self.mlp = Deepseekv3MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.moe_intermediate_size *
                self.num_shared_experts,
                dtype=config.torch_dtype,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict)
        else:
            block_size = 1
            if model_config.quant_config and model_config.quant_config.group_size is not None:
                block_size = model_config.quant_config.group_size
            self.mlp_tp_size = self._compute_mlp_tp_size(
                config.intermediate_size, block_size)

            has_mlp_tp = self.mlp_tp_size > 1
            self.fusion_config.PRE_MLP_FUSION = self.enable_fusion and has_mlp_tp and self.is_nvfp4
            self.fusion_config.POST_MLP_FUSION = self.enable_fusion and has_mlp_tp

            self.mlp = GatedMLP(hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                bias=False,
                                dtype=config.torch_dtype,
                                config=model_config,
                                overridden_tp_size=self.mlp_tp_size,
                                reduce_output=True)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.fusion_config.PRE_MLP_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.layer_idx = layer_idx
        self.allreduce = AllReduce(self.mapping, dtype=config.torch_dtype)
        self.moe_allreduce = MoEAllReduce(self.mapping)
        self.next_layer_layernorm: RMSNorm = None

    def _compute_mlp_tp_size(self, intermediate_size: int,
                             block_size: int) -> int:
        """
        For DeepSeek‑R1, MLP TP size is limited by intermediate_size // block_size
        and must also be multiples of gpus_per_node to avoid expensive inter‑node allreduce.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            int: The computed tp_size.
        """

        assert intermediate_size % block_size == 0, "intermediate_size must be divisible by block_size."
        if self.enable_attention_dp:
            # If using attention DP, the MLP also uses DP instead of TP.
            mlp_tp_size = 1
        else:
            # The two math.gcd operations ensure that mlp_tp_size falls in the candidate TP sizes.
            tp = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )
            mlp_tp_size = math.gcd(
                tp,
                self.mapping.gpus_per_node,
            ) if tp > self.mapping.gpus_per_node else tp  # Avoid costly inter-node TP
        return mlp_tp_size

    def _enable_min_latency_mode(self, num_tokens: int):
        return (num_tokens <= 128 and self.fusion_config.POST_MOE_FUSION
                and self.is_nvfp4 and self.model_config.moe_backend == 'CUTLASS'
                and not self.mapping.is_multi_node()
                and self.allreduce.mnnvl_allreduce is None)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        if isinstance(self.mlp, Deepseekv3MoE):
            return self.forward_MoE(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )
        else:
            assert isinstance(self.mlp, GatedMLP)
            return self.forward_mlp(
                hidden_states=hidden_states,
                residual=residual,
            )

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
    ) -> torch.Tensor:

        def _run_MoE(hidden_states, hidden_states_fp4):
            return self.mlp(
                hidden_states,
                hidden_states_fp4,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(
                    enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                          or self.mapping.tp_size == 1)),
                cutlass_min_latency_mode=cutlass_min_latency_mode,
            )

        cutlass_min_latency_mode = self._enable_min_latency_mode(
            hidden_states.shape[0])

        if cutlass_min_latency_mode:
            assert self.fusion_config.PRE_MOE_FUSION and self.fusion_config.POST_MOE_FUSION
            assert self.model_config.moe_backend == 'CUTLASS'

            hidden_states, hidden_states_act, hidden_states_sf, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.
                    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.mlp.experts.fc31_input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
            hidden_states_fp4 = Fp4QuantizedTensor(hidden_states_act,
                                                   hidden_states_sf)

            hidden_states = _run_MoE(hidden_states, hidden_states_fp4)

            shared_output = hidden_states[0]
            hidden_states_activated_experts = hidden_states[1]
            num_activated_experts_per_node = hidden_states[2]
            experts_to_token_score = hidden_states[3]

            # MoE_finalize is fused into allreduce
            hidden_states, residual = self.moe_allreduce(
                residual,
                self.next_layer_layernorm.weight,
                device_num_experts=num_activated_experts_per_node,
                scale_input=experts_to_token_score,
                active_experts_token_input=hidden_states_activated_experts,
                token_input=shared_output,
                eps=self.next_layer_layernorm.variance_epsilon,
            )
        else:
            if self.fusion_config.PRE_MOE_FUSION:
                # moe_backend can be either CUTLASS or TRTLLM here
                # TODO: unify the two min-latency MoE backends by enabling quant fusion
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ))
            else:
                # No fusion
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)

            hidden_states = _run_MoE(hidden_states, hidden_states_fp4=None)

            if self.fusion_config.POST_MOE_FUSION:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                if self.next_layer_layernorm is not None:
                    hidden_states, residual = self.next_layer_layernorm(
                        hidden_states, residual)

        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:

        if self.fusion_config.PRE_MLP_FUSION:
            act_fp4, act_sf, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.mlp.gate_up_proj.input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ),
            )
            hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
        else:
            # No fusion
            # We need to add twoshot allreduce here to avoid modifying MLA logic
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.POST_MLP_FUSION or self.mlp_tp_size == 1)),
        )

        if self.fusion_config.POST_MLP_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                ),
            )
        else:
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)

        return hidden_states, residual


class DeepseekV3MTP(DeepseekV3DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__(model_config, layer_idx, aux_stream_dict)
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.enorm = RMSNorm(hidden_size=config.hidden_size,
                             eps=config.rms_norm_eps,
                             dtype=config.torch_dtype)

        self.hnorm = RMSNorm(hidden_size=config.hidden_size,
                             eps=config.rms_norm_eps,
                             dtype=config.torch_dtype)

        self.eh_proj = Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
        )

        self.shared_head = DeepseekV3MTPHead(model_config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        spec_metadata: MTPSpecMetadata,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs_embeds = self.enorm(embed_tokens(input_ids))
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.concat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.eh_proj(hidden_states)

        # Input layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        # MTP Layer Must have sparse MOE
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ),
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # MoE
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=spec_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
        )

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.shared_head.norm.weight,
                    eps=self.shared_head.norm.variance_epsilon,
                ),
            )
        else:
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)

        logits = self.shared_head(hidden_states, lm_head, attn_metadata).float()

        return hidden_states, logits


class DeepseekV3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        aux_stream_list = [torch.cuda.Stream() for _ in range(2)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(model_config, layer_idx,
                                   self.aux_stream_dict)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for decoder_layer in self.layers[:self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        return hidden_states


@register_auto_model("DeepseekV3ForCausalLM")
class DeepseekV3ForCausalLM(DecoderModelForCausalLM[DeepseekV3Model,
                                                    PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(DeepseekV3Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

        self.model_nextn = 0
        if model_config.spec_config is not None:
            model_nextn = model_config.spec_config.num_nextn_predict_layers
            ckpt_nextn = self.config.num_nextn_predict_layers
            self.num_hidden_layers = self.config.num_hidden_layers
            assert ckpt_nextn > 0, "There is not MTP modules in the checkpoint."
            if ckpt_nextn == 1:
                mtp_layer = DeepseekV3MTP(model_config, self.num_hidden_layers,
                                          self.model.aux_stream_dict)
                self.model.layers.append(mtp_layer)
                self.epilogue.append(mtp_layer)
                self.mtp_worker = MTPEagleWorker(model_config.spec_config)
            else:
                # TODO: fix the accuracy issue and remove this assert.
                assert False, "Cannot support num_nextn_predict_layers>1 in checkpoint now. Will fix it soon"
                mtp_layers = nn.ModuleList([
                    DeepseekV3MTP(model_config,
                                  layer_idx + self.num_hidden_layers,
                                  self.model.aux_stream_dict)
                    for layer_idx in range(model_nextn)
                ])
                self.model.layers.extend(mtp_layers)
                self.epilogue.extend(mtp_layers)
                self.mtp_worker = MTPWorker(model_config.spec_config)
                # modify the QuantConfig to support duplicated mtp layers
                if model_config.quant_config.exclude_modules is not None:
                    extend_exclude_modules = []
                    for model_mtp_idx in range(
                            self.num_hidden_layers,
                            self.num_hidden_layers + model_nextn):
                        ckpt_mtp_idx = (model_mtp_idx - self.num_hidden_layers
                                        ) % ckpt_nextn + self.num_hidden_layers
                        model_prefix = f"model.layers.{model_mtp_idx}"
                        ckpt_prefix = f"model.layers.{ckpt_mtp_idx}"
                        for exclude_module in model_config.quant_config.exclude_modules:
                            if ckpt_prefix in exclude_module and model_prefix not in exclude_module:
                                extend_exclude_modules.append(
                                    exclude_module.replace(
                                        ckpt_prefix, model_prefix))
                    self.model_config.quant_config.exclude_modules.extend(
                        extend_exclude_modules)
            self.epilogue.append(self.mtp_worker)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[MTPSpecMetadata] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata.num_generations_per_batch = self.model_nextn + 1
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        if spec_metadata and spec_metadata.spec_dec_mode.is_mtp():
            # get logits
            logits = self.logits_processor.forward(
                hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )
            # get accepted tokens and next draft tokens
            return self.mtp_worker(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                logits=logits,
                lm_head=self.lm_head,
                embed_tokens=self.model.embed_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                mtp_layers=self.model.layers[self.num_hidden_layers:])
        else:
            logits = self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                return_context_logits,
            )
            return logits

    def load_weights(self, weights: Dict):

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        def rename_moe_weight(weights: Dict, rename_rules: Dict):
            result = {}
            for key, value in weights.items():
                new_key = key
                for old, new in rename_rules.items():
                    new_key = new_key.replace(old, new)
                result[new_key] = value
            return result

        ## Prepare weights for TP
        def split(v, tp_size, idx, dim=0):
            if tp_size == 1:
                return v
            if len(v.shape) == 1:
                return torch.chunk(v, tp_size)[idx].contiguous()
            else:
                return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()

        def split_matrix_tp(v, tensor_parallel, rank, dim):
            return split(v, tensor_parallel, rank, dim=dim)

        def load_kv_b_proj_and_k_b_proj_trans(module_name: str,
                                              is_scale: bool) -> torch.Tensor:
            weight_name = "weight" if not is_scale else "weight_scale_inv"
            local_qk_nope_head_dim = qk_nope_head_dim if not is_scale else qk_nope_head_dim // 128
            local_v_head_dim = v_head_dim if not is_scale else v_head_dim // 128
            local_kv_lora_rank = kv_lora_rank if not is_scale else kv_lora_rank // 128

            kv_b_proj = weights[f"{module_name}.{weight_name}"][:].unflatten(
                0,
                [
                    num_heads,
                    local_qk_nope_head_dim + local_v_head_dim,
                ],
            )

            if not self.model_config.mapping.enable_attention_dp:
                kv_b_proj = split_matrix_tp(kv_b_proj, tp_size, tp_rank, 0)
            k_nope_weight, v_weight = kv_b_proj.split(
                [local_qk_nope_head_dim, local_v_head_dim],
                dim=1,
            )
            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_nope_weight_trans = k_nope_weight.transpose(2, 1)

            kv_b_proj = torch.concat([
                k_nope_weight.reshape(local_num_heads * local_qk_nope_head_dim,
                                      local_kv_lora_rank),
                v_weight.reshape(local_num_heads * local_v_head_dim,
                                 local_kv_lora_rank)
            ],
                                     dim=0)

            return kv_b_proj, k_nope_weight_trans

        def check_weight_dtype(module_name: str, dtype):
            weight_name = "weight"
            w_dtype = weights[f"{module_name}.{weight_name}"].dtype
            return w_dtype == dtype

        def load_kv_b_proj_and_k_b_proj_trans_dequant(
                module_name: str) -> torch.Tensor:
            weight_name = "weight"
            local_qk_nope_head_dim = qk_nope_head_dim
            local_v_head_dim = v_head_dim
            local_kv_lora_rank = kv_lora_rank

            kv_b_proj = weights[f"{module_name}.{weight_name}"][:].cuda()

            weight_name = "weight_scale_inv"
            kv_b_proj_scale = weights[f"{module_name}.{weight_name}"][:].cuda()

            kv_b_proj = weight_dequant(kv_b_proj, kv_b_proj_scale)
            kv_b_proj = kv_b_proj.unflatten(
                0,
                [
                    num_heads,
                    local_qk_nope_head_dim + local_v_head_dim,
                ],
            )
            if not self.model_config.mapping.enable_attention_dp:
                kv_b_proj = split_matrix_tp(kv_b_proj, tp_size, tp_rank, 0)
            k_nope_weight, v_weight = kv_b_proj.split(
                [local_qk_nope_head_dim, local_v_head_dim],
                dim=1,
            )
            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_nope_weight_trans = k_nope_weight.transpose(2, 1)

            kv_b_proj = torch.concat([
                k_nope_weight.reshape(local_num_heads * local_qk_nope_head_dim,
                                      local_kv_lora_rank),
                v_weight.reshape(local_num_heads * local_v_head_dim,
                                 local_kv_lora_rank)
            ],
                                     dim=0)

            return kv_b_proj, k_nope_weight_trans

        def split_kv_b_proj(kv_b_proj: torch.Tensor,
                            is_scale: bool) -> torch.Tensor:
            local_qk_nope_head_dim = qk_nope_head_dim if not is_scale else qk_nope_head_dim // 128
            local_v_head_dim = v_head_dim if not is_scale else v_head_dim // 128

            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_b_proj, v_b_proj = kv_b_proj.split([
                local_num_heads * local_qk_nope_head_dim,
                local_num_heads * local_v_head_dim
            ],
                                                 dim=0)
            k_b_proj = k_b_proj.view(
                [local_num_heads, local_qk_nope_head_dim, -1])
            v_b_proj = v_b_proj.view([local_num_heads, local_v_head_dim, -1])

            return k_b_proj, v_b_proj

        is_lite = self.config.q_lora_rank is None
        num_heads = self.config.num_attention_heads
        qk_nope_head_dim = self.config.qk_nope_head_dim
        v_head_dim = self.config.v_head_dim
        kv_lora_rank = self.config.kv_lora_rank

        tp_rank = self.model_config.mapping.tp_rank
        tp_size = self.model_config.mapping.tp_size

        params_map = {'gate_up_proj': ['gate_proj', 'up_proj']}
        all_named_modules = dict(self.named_modules())

        for name, module in tqdm(all_named_modules.items(),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                names = name.split('.')
                parent_module_name = '.'.join(names[:-1])
                if "model.layers" in name and int(
                        names[2]) >= self.config.num_hidden_layers:
                    mtp_layer_idx = int(
                        names[2]) - self.config.num_hidden_layers
                    names[2] = str(mtp_layer_idx %
                                   self.config.num_nextn_predict_layers +
                                   self.config.num_hidden_layers)
                    name = '.'.join(names)
                if names[-1] == "kv_b_proj":
                    # TODO: remove weight_dequant after enabling fp8_bmm
                    dequant_kv_b_proj = self.model_config.quant_config.is_module_excluded_from_quantization(
                        names[-1])
                    if dequant_kv_b_proj:
                        kv_b_proj, k_b_proj_trans = load_kv_b_proj_and_k_b_proj_trans_dequant(
                            name)
                    else:
                        kv_b_proj, k_b_proj_trans = load_kv_b_proj_and_k_b_proj_trans(
                            name, is_scale=False)
                    module.weight.data.copy_(
                        kv_b_proj.reshape(module.weight.shape))

                    attn_module = all_named_modules[parent_module_name]
                    _, v_b_proj = split_kv_b_proj(module.weight.data,
                                                  is_scale=False)
                    attn_module.v_b_proj = nn.Parameter(v_b_proj,
                                                        requires_grad=False)

                    attn_module.k_b_proj_trans.data.copy_(
                        k_b_proj_trans.reshape(
                            attn_module.k_b_proj_trans.shape))

                    if getattr(module, "weight_scale",
                               None) is not None and not dequant_kv_b_proj:
                        kv_b_proj_scale, k_b_proj_trans_scale = load_kv_b_proj_and_k_b_proj_trans(
                            name, is_scale=True)
                        module.weight_scale.copy_(
                            kv_b_proj_scale.reshape(module.weight_scale.shape))
                        attn_module.k_b_proj_trans_scale.copy_(
                            k_b_proj_trans_scale.reshape(
                                attn_module.k_b_proj_trans_scale.shape))

                        _, v_b_proj_scale = split_kv_b_proj(
                            module.weight_scale.data, is_scale=True)
                        attn_module.v_b_proj_scale = nn.Parameter(
                            v_b_proj_scale, requires_grad=False)

                elif names[-1] == "fused_a":
                    fused_a = weights[
                        f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"][:]
                    if not is_lite:
                        q_a_proj = weights[
                            f"{'.'.join(names[:-1])}.q_a_proj.weight"][:]
                        fused_a = torch.cat([q_a_proj, fused_a], dim=0)

                    if f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_inv" in weights:
                        fused_a_scale = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_inv"]
                        if not is_lite:
                            q_a_proj_scale = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.weight_scale_inv"][:]
                            fused_a_scale = torch.cat(
                                [q_a_proj_scale, fused_a_scale], dim=0)

                        module.weight_scale.data.copy_(fused_a_scale)

                    module.weight.data.copy_(fused_a)
                elif names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        module_weights.append(
                            filter_weights('.'.join(names[:-1] + [new_name]),
                                           weights))
                    module.load_weights(weights=module_weights)
                elif names[-1] == "experts":
                    module_weights = filter_weights(name, weights)
                    module_weights = rename_moe_weight(module_weights, {
                        "down_proj": "w2",
                        "up_proj": "w3",
                        "gate_proj": "w1",
                    })
                    module.load_weights(weights=[module_weights])
                elif names[-1] == "self_attn":
                    continue
                elif names[-1] == "next_layer_layernorm":
                    continue
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module.named_parameters():
                            p.data.copy_(module_weights[n][:])

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
