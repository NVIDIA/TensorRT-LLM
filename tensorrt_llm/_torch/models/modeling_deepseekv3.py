import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, DeepseekAllReduce,
                                             ParallelConfig, allgather,
                                             reducescatter)
from tensorrt_llm.functional import PositionEmbeddingType

from ...llmapi.utils import enable_llm_debug
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..models.modeling_utils import MissingLayer, ModelConfig, support_pp
from ..modules.attention import MLA
from ..modules.decoder_layer import DecoderLayer
from ..modules.fused_moe import BaseMoeRoutingMethod, FusedMoE
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from ..pipeline_interface import PipelineInterface
from ..pyexecutor.cuda_graph_runner import is_graph_capturing
from ..speculative import MTPEagleWorker, MTPSpecMetadata, MTPWorker
from ..utils import Fp4QuantizedTensor, disable_fp4_allgather
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, register_auto_model)


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


class DeepseekV3RotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 config: PretrainedConfig,
                 device: Optional[torch.device] = None):
        head_dim = config.hidden_size // config.num_attention_heads
        super().__init__(config,
                         head_dim=head_dim,
                         num_attention_heads=config.num_attention_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         device=device,
                         rope_type="default")


class DeepseekV3Attention(MLA):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config
        if model_config.fuse_pos_embd:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.yarn,
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = None

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
                         rotary_emb=DeepseekV3RotaryEmbedding(config),
                         pos_embd_params=pos_embd_params,
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
        is_thop: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.topk_group = topk_group
        self.n_group = n_group
        self.routed_scaling_factor = routed_scaling_factor
        self.is_thop = is_thop

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

        if self.is_thop == False:
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


class Deepseekv3Gate(BaseMoeRoutingMethod):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        dtype: Optional[torch.dtype] = None,
        is_thop: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.empty(
            (num_experts), dtype=torch.float32),
                                                    requires_grad=False)

        # TODO: e_score_correction_bias makes sense to live in this gate class, but it is needed for the routing impl
        #       So we don't run into issues with weight loading, we make this gate object the BaseMoeRoutingMethod
        #       and then dispatch to the routing impl for the actual implementation.
        #       This is a bit of a hack and we should clean this up in the future.
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_thop=is_thop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.ops.trtllm.cublas_mm(hidden_states,
                                            self.weight.t(),
                                            bias=None,
                                            out_dtype=torch.float32)
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

        self.e_score_correction_bias.copy_(
            weights[0]["e_score_correction_bias"][:].to(torch.float32))

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
                 aux_stream: torch.cuda.Stream,
                 dtype: Optional[torch.dtype] = None,
                 tune_max_num_tokens: int = 8192,
                 model_config: ModelConfig = ModelConfig()):
        from tensorrt_llm._torch.distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.use_dp = model_config.mapping.enable_attention_dp
        self.gate = Deepseekv3Gate(
            hidden_size,
            num_experts,
            top_k=top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=dtype)
        self.experts = FusedMoE(
            num_experts=num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=
            False,  # In both low latency and attention dp scenarios, FusedMoE needs not to do allreduce inside op.
            tune_max_num_tokens=tune_max_num_tokens,
            model_config=model_config)

        self.shared_output_scale = None
        if self.use_dp:
            shared_tp_size = 1
        else:
            assert shared_expert_intermediate_size % 128 == 0
            shared_tp_size = math.gcd(
                shared_expert_intermediate_size // 128,
                model_config.mapping.tp_size,
            )
            if shared_tp_size != model_config.mapping.tp_size:
                self.shared_output_scale = shared_tp_size / model_config.mapping.tp_size
        self.shared_experts = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=shared_tp_size,
            is_expert=True)

        self.parallel_config = ParallelConfig(
            tensor_parallel_rank=model_config.mapping.tp_rank,
            tensor_parallel_size=model_config.mapping.tp_size,
            gpus_per_node=model_config.mapping.gpus_per_node,
            pipeline_parallel_size=model_config.mapping.pp_size,
            parallel_rank=model_config.mapping.rank)
        self.all_reduce = AllReduce(self.parallel_config)
        self.aux_stream = aux_stream
        self.moe_event = [torch.cuda.Event(), torch.cuda.Event()]

    def reduce_scatter(self, input_tensor, all_rank_num_tokens):
        world_size = self.parallel_config.tensor_parallel_size
        rank = self.parallel_config.tensor_parallel_rank
        if world_size == 1:
            return input_tensor
        dst_tensor = input_tensor
        outputs = reducescatter(dst_tensor, self.parallel_config, scatter_dim=0)
        depad_tensors = outputs[:all_rank_num_tokens[rank]]
        return depad_tensors

    def compute_routed_output(self, hidden_states, all_rank_num_tokens):
        if self.use_dp and self.parallel_config.tensor_parallel_size > 1:
            max_num_token = max(all_rank_num_tokens)
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, max_num_token - hidden_states.shape[0]))
            if disable_fp4_allgather():
                hidden_states = allgather(hidden_states,
                                          self.parallel_config,
                                          gather_dim=0)
        router_logits = self.gate(hidden_states)
        routed_output = self.experts(hidden_states, router_logits)
        if self.use_dp:
            routed_output = self.reduce_scatter(routed_output,
                                                all_rank_num_tokens)
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        # Only enable multi-stream for cuda graph since switch stream has extra host overhead
        # This design is mainly for low latency use case. Need to improve for max throughput use case.
        do_multi_stream = is_graph_capturing()
        if do_multi_stream:
            self.moe_event[0].record()
        shared_output = self.shared_experts(hidden_states)
        if self.shared_output_scale is not None:
            shared_output *= self.shared_output_scale
        if do_multi_stream:
            with torch.cuda.stream(self.aux_stream):
                self.moe_event[0].wait()
                routed_output = self.compute_routed_output(
                    hidden_states, all_rank_num_tokens)
                self.moe_event[1].record()
            self.moe_event[1].wait()
        else:
            routed_output = self.compute_routed_output(hidden_states,
                                                       all_rank_num_tokens)
        assert shared_output.size() == routed_output.size(
        ), f'unmatched tensor shape'
        final_hidden_states = shared_output + routed_output
        if not self.use_dp and self.parallel_config.tensor_parallel_size > 1:
            final_hidden_states = self.all_reduce(
                final_hidden_states, all_reduce_params=final_all_reduce_params)

        return final_hidden_states


class DeepseekV3DecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.self_attn = DeepseekV3Attention(model_config,
                                             layer_idx=layer_idx,
                                             aux_stream=aux_stream)
        self.fusion_config = EagerFusionConfig()
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.mlp_tp_size = model_config.mapping.tp_size

        self.enable_fusion = os.environ.get(
            "TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED", "0") == "0"

        pp_layer_offset = model_config.mapping.pp_layers_torch(
            config.num_hidden_layers)[0]
        global_layer_idx = pp_layer_offset + layer_idx

        if (config.n_routed_experts is not None
                and global_layer_idx >= config.first_k_dense_replace
                and global_layer_idx % config.moe_layer_freq == 0):
            self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and model_config.mapping.has_tp(
            ) and not self.enable_attention_dp
            self.fusion_config.POST_MOE_FUSION = self.enable_fusion and model_config.mapping.has_tp(
            ) and not self.enable_attention_dp and not model_config.mapping.has_pp(
            )
            self.mlp = Deepseekv3MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.moe_intermediate_size *
                self.num_shared_experts,
                dtype=config.torch_dtype,
                tune_max_num_tokens=config.max_position_embeddings // 4,
                model_config=model_config,
                aux_stream=aux_stream)
        else:
            if self.enable_attention_dp:
                self.mlp_tp_size = 1
            else:
                assert config.intermediate_size % 128 == 0
                self.mlp_tp_size = math.gcd(
                    math.gcd(
                        config.intermediate_size // 128,
                        model_config.mapping.tp_size,
                    ),
                    model_config.mapping.
                    gpus_per_node,  # Avoid costly inter-node TP
                )
            self.fusion_config.PRE_MLP_FUSION = self.enable_fusion and model_config.mapping.has_tp(
            ) and model_config.quant_config.layer_quant_mode.has_nvfp4(
            ) and not self.enable_attention_dp
            self.fusion_config.POST_MLP_FUSION = self.enable_fusion and self.mlp_tp_size > 1 and not self.enable_attention_dp and not model_config.mapping.has_pp(
            )
            self.mlp = GatedMLP(hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                bias=False,
                                dtype=config.torch_dtype,
                                config=model_config,
                                overridden_tp_size=self.mlp_tp_size,
                                is_expert=False)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.parallel_config = ParallelConfig(
            tensor_parallel_rank=model_config.mapping.tp_rank,
            tensor_parallel_size=model_config.mapping.tp_size,
            gpus_per_node=model_config.mapping.gpus_per_node,
            pipeline_parallel_size=model_config.mapping.pp_size,
            parallel_rank=model_config.mapping.rank)
        self.layer_idx = layer_idx
        self.all_reduce = AllReduce(self.parallel_config)
        self.next_layer_layernorm: RMSNorm = None

        self.deepseek_allreduce_disabled = os.environ.get(
            "TRTLLM_DEEPSEEK_ALLREDUCE_FUSION_DISABLED", "0") == "1"
        if model_config.mapping.is_multi_node():
            self.deepseek_allreduce_disabled = True

        if not self.deepseek_allreduce_disabled:
            self.deepseek_allreduce = DeepseekAllReduce(self.parallel_config)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        # deepseek allreduce kernel is better when m < 512
        using_prev_fusion = self.deepseek_allreduce_disabled or hidden_states.size(
            0) >= 512

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.PRE_MOE_FUSION or self.fusion_config.
                PRE_MLP_FUSION or self.parallel_config.tensor_parallel_size == 1
                or self.enable_attention_dp)),
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
            # Custom AR Fusion for DeepseekV3
            if using_prev_fusion:
                # Custom AR Fusion for DeepseekV3
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ))
            else:
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states,
                    [residual, self.post_attention_layernorm.weight],
                    self.post_attention_layernorm.variance_epsilon,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                )
        elif self.fusion_config.PRE_MLP_FUSION:
            # Custom AR Fusion for DeepseekV3 with quant_fp4
            if using_prev_fusion:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ))
                act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                    hidden_states, self.mlp.gate_up_proj.input_scale,
                    self.mlp.gate_up_proj.scaling_vector_size, False)
            else:
                act_fp4, act_sf, residual = self.deepseek_allreduce(
                    hidden_states,
                    [
                        residual, self.post_attention_layernorm.weight,
                        self.mlp.gate_up_proj.input_scale
                    ],
                    self.post_attention_layernorm.variance_epsilon,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                )
            hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)

        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.POST_MOE_FUSION
                or self.fusion_config.POST_MLP_FUSION or self.mlp_tp_size == 1
                or self.enable_attention_dp)),
        )

        if self.fusion_config.POST_MOE_FUSION:
            if using_prev_fusion:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states,
                    [residual, self.next_layer_layernorm.weight],
                    self.next_layer_layernorm.variance_epsilon,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                )
        elif self.fusion_config.POST_MLP_FUSION:

            if using_prev_fusion:
                # Custom AR Fusion for DeepseekV3
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states,
                    [residual, self.next_layer_layernorm.weight],
                    self.next_layer_layernorm.variance_epsilon,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                )

        else:
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)

        return hidden_states, residual


class DeepseekV3MTP(DeepseekV3DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__(model_config, layer_idx, aux_stream)
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
            skip_create_weights=model_config.skip_create_weights,
        )

        self.shared_head = DeepseekV3MTPHead(model_config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        embed_tokens: nn.Embedding,
        attn_metadata: AttentionMetadata,
        spec_metadata: MTPSpecMetadata,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # deepseek allreduce kernel is better when m < 512
        using_prev_fusion = self.deepseek_allreduce_disabled or hidden_states.size(
            0) >= 512

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
            all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.PRE_MOE_FUSION or self.parallel_config.
                tensor_parallel_size == 1 or self.enable_attention_dp)),
            **kwargs,
        )

        # MTP Layer Must have sparse MOE
        if self.fusion_config.PRE_MOE_FUSION:
            # Custom AR Fusion for DeepseekV3
            if using_prev_fusion:
                # Custom AR Fusion for DeepseekV3
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ))
            else:
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states,
                    [residual, self.post_attention_layernorm.weight],
                    self.post_attention_layernorm.variance_epsilon,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
        # Fully Connected
        hidden_states = self.mlp(
            hidden_states,
            spec_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.POST_MOE_FUSION or self.parallel_config.
                tensor_parallel_size == 1 or self.enable_attention_dp)),
        )

        if self.fusion_config.POST_MOE_FUSION:
            if using_prev_fusion:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.shared_head.norm.weight,
                        eps=self.shared_head.norm.variance_epsilon,
                    ))
            else:
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states,
                    [residual, self.shared_head.norm.weight],
                    self.shared_head.norm.variance_epsilon,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                )
        else:
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)

        logits = self.shared_head(hidden_states, lm_head, attn_metadata).float()

        return hidden_states, logits


@support_pp
class DeepseekV3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         dtype=config.torch_dtype)

        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(model_config, layer_idx, self.aux_stream)
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
        residual = hidden_states

        hidden_states = self.layers[0].input_layernorm(hidden_states)
        for decoder_layer in self.layers[:self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual)

        return hidden_states

    def _pp_forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
    ) -> torch.Tensor:
        if self.pp_rank != 0:
            if pipeline_interface is None:
                raise ValueError(
                    "pipeline_interface is required for non-first pp rank.")
            hidden_states, residual = pipeline_interface
        else:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds
            residual = hidden_states

        local_decoder_layers = ([
            self.layers[layer_id] for layer_id in self.pp_layer_list
        ] if self.pp_size > 1 else self.layers)

        if self.pp_rank == 0:
            hidden_states = local_decoder_layers[0].input_layernorm(
                hidden_states)
        else:
            hidden_states, residual = local_decoder_layers[0].input_layernorm(
                hidden_states, residual)

        for decoder_layer in local_decoder_layers:
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual)

        if not self.pp_rank == self.pp_size - 1:
            return PipelineInterface(hidden_states, residual)
        else:
            return hidden_states


@register_auto_model("DeepseekV3ForCausalLM")
class DeepseekV3ForCausalLM(DecoderModelForCausalLM[DeepseekV3Model,
                                                    PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(DeepseekV3Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

        assert not (
            model_config.mapping.has_pp()
            and model_config.mapping.enable_attention_dp
        ), "Pipeline parallelism and attention DP cannot be used together"

        self.model_nextn = 0
        if model_config.spec_config is not None:
            model_nextn = model_config.spec_config.num_nextn_predict_layers
            ckpt_nextn = self.config.num_nextn_predict_layers
            self.num_hidden_layers = self.config.num_hidden_layers
            assert ckpt_nextn > 0, "There is not MTP modules in the checkpoint."
            if ckpt_nextn == 1:
                mtp_layer = DeepseekV3MTP(model_config, self.num_hidden_layers,
                                          self.model.aux_stream)
                self.model.layers.append(mtp_layer)
                self.mtp_worker = MTPEagleWorker(model_config.spec_config)
            else:
                # TODO: fix the accuracy issue and remove this assert.
                assert False, "Cannot support num_nextn_predict_layers>1 in checkpoint now. Will fix it soon"
                mtp_layers = nn.ModuleList([
                    DeepseekV3MTP(model_config,
                                  layer_idx + self.num_hidden_layers,
                                  self.model.aux_stream)
                    for layer_idx in range(model_nextn)
                ])
                self.model.layers.extend(mtp_layers)
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

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[MTPSpecMetadata] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata.num_generations_per_batch = self.model_nextn + 1
        if self._supports_pp and self.pp_size > 1:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pipeline_interface=pipeline_interface,
            )

            # No need to compute logits for non-last PP ranks
            if self.pp_rank < self.pp_size - 1:
                return output
            else:
                hidden_states = output
        else:
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
            # get accepetd tokens and next draft tokens
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

                    if getattr(module, "weight_scale", None) is not None:
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
            elif not isinstance(self.model.layers[idx + 1], MissingLayer):
                # layers[idx + 1] is MissingLayer for last layer in pp rank
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
