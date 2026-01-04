import os
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
from transformers import GptOssConfig

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           allgather)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from ..modules.fused_moe import (MoE, MoEWeightLoadingMode,
                                 RenormalizeMoeRoutingMethod, TritonFusedMoE,
                                 create_moe)
from ..modules.fused_moe.routing import (get_cached_perfect_router_logits,
                                         precompute_common_perfect_router_logits
                                         )
# isort: on
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import Fp4QuantizedTensor
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, filter_weights, register_auto_model

# Use TinyGEMM when the number of tokens is not larger than this threshold
MIN_LATENCY_TINYGEMM_NUM_TOKENS = 128


class AttentionBlock(Attention):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int = 0,
        reduce_output: bool = True,
        use_custom_cublas_mm: bool = False,
    ):
        pretrained_config = config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams(
                dim=pretrained_config.head_dim,
                theta=pretrained_config.rope_theta,
                scale_type=RotaryScalingType.yarn,
                scale=pretrained_config.rope_scaling['factor'],
                max_positions=pretrained_config.max_position_embeddings,
                original_max_positions=pretrained_config.
                rope_scaling['original_max_position_embeddings'],
                beta_fast=pretrained_config.rope_scaling['beta_fast'],
                beta_slow=pretrained_config.rope_scaling['beta_slow'],
                duplicate_data=False),
            is_neox=False,
        )

        super().__init__(
            hidden_size=pretrained_config.hidden_size,
            num_attention_heads=pretrained_config.num_attention_heads,
            num_key_value_heads=pretrained_config.num_key_value_heads,
            max_position_embeddings=pretrained_config.max_position_embeddings,
            bias=True,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=pretrained_config.torch_dtype,
            dense_bias=True,
            config=config,
            q_scaling=1.0,
            attention_chunk_size=None,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        # Only apply sliding window to every other layer
        self.sliding_window = pretrained_config.sliding_window if layer_idx % 2 == 0 else None

        self.sinks = Parameter(
            torch.empty(pretrained_config.num_attention_heads // self.tp_size,
                        dtype=torch.float32))
        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor | Fp4QuantizedTensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attention_window_size = self.sliding_window
        attn_output = super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            attention_window_size=attention_window_size,
            attention_sinks=self.sinks.data,
            **kwargs)
        return attn_output, residual

    def load_weights(self, weights: Dict):
        sinks = weights[0]['sinks'][self.num_heads *
                                    self.tp_rank:self.num_heads *
                                    (self.tp_rank + 1)]
        self.sinks.data = sinks.to(torch.float32).to("cuda")


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int,
        reduce_results: bool = True,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__()

        self.config = config  # Store config as instance variable
        pretrained_config = config.pretrained_config
        self.num_experts = pretrained_config.num_local_experts
        moe_load_balancer_config = config.moe_load_balancer
        self.num_slots = moe_load_balancer_config.num_slots if moe_load_balancer_config and moe_load_balancer_config.num_slots else self.num_experts

        self.layer_idx = layer_idx
        self.enable_attention_dp = config.mapping.enable_attention_dp
        self.mapping = config.mapping

        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

        self.gate = Linear(
            in_features=pretrained_config.hidden_size,
            out_features=pretrained_config.num_local_experts,
            bias=True,
            dtype=pretrained_config.torch_dtype,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        self.routing_method = RenormalizeMoeRoutingMethod(
            top_k=pretrained_config.num_experts_per_tok,
            output_dtype=torch.bfloat16
            if config.moe_backend.upper() == "TRTLLM" else torch.float32)

        self.swiglu_alpha = torch.tensor(
            [1.702] * (self.num_slots // config.mapping.moe_ep_size),
            dtype=torch.float32).cuda()
        self.swiglu_beta = torch.tensor(
            [1.0] * (self.num_slots // config.mapping.moe_ep_size),
            dtype=torch.float32).cuda()
        self.swiglu_limit = torch.tensor(
            [7.0] * (self.num_slots // config.mapping.moe_ep_size),
            dtype=torch.float32).cuda()
        # Prepare MoE creation parameters
        moe_params = {
            'routing_method': self.routing_method,
            'num_experts': pretrained_config.num_local_experts,
            'hidden_size': pretrained_config.hidden_size,
            'intermediate_size': pretrained_config.intermediate_size,
            'dtype': pretrained_config.torch_dtype,
            'reduce_results': not self.enable_attention_dp and reduce_results,
            'model_config': config,
            'weight_loading_mode': MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            'bias': True,
            'swiglu_alpha': self.swiglu_alpha,
            'swiglu_beta': self.swiglu_beta,
            'swiglu_limit': self.swiglu_limit
        }

        self.experts = create_moe(**moe_params)

        # Perfect router caching - precompute common logits if enabled
        if os.environ.get('ENABLE_PERFECT_ROUTER', '0') == '1':
            precompute_common_perfect_router_logits(
                num_experts=pretrained_config.num_local_experts,
                experts_per_token=pretrained_config.num_experts_per_tok,
                moe_ep_size=config.mapping.moe_ep_size,
                dtype=pretrained_config.torch_dtype)

    @staticmethod
    def swiglu(x, alpha: float = 1.702):
        """
        This function is not really used in self.forward(), it's kept here for reference.
        It's implemented as part of the MoE kernels.
        """
        # Note we add an extra bias of 1 to the linear layer
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        return out_glu * (x_linear + 1)

    def _create_ideal_expert_load_balanced_logits(
            self, num_tokens: int, num_experts: int,
            device: torch.device) -> torch.Tensor:
        """
        Create ideal logits that produce GPU-aware load balanced expert assignment.
         This method now uses the global cache to access precomputed logits to optimize performance.
        """
        pretrained_config = self.config.pretrained_config

        # Use global cached logits
        return get_cached_perfect_router_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=pretrained_config.experts_per_token,
            moe_ep_size=self.config.mapping.moe_ep_size,
            device=device,
            dtype=pretrained_config.torch_dtype)

    def compute_gate_output(self,
                            x: torch.Tensor,
                            lora_params: Optional[dict] = None) -> torch.Tensor:
        # Skip tinygemm2 optimization when LoRA is active (tinygemm2 doesn't support LoRA)
        use_tinygemm = (get_sm_version() in [90, 100, 103]
                        and x.shape[0] <= MIN_LATENCY_TINYGEMM_NUM_TOKENS
                        and (lora_params is None or not bool(lora_params)))

        if use_tinygemm:
            weight = self.gate.weight
            bias = self.gate.bias
            g = torch.ops.trtllm.tinygemm2(x, weight, bias)
        else:
            g = self.gate(x, lora_params=lora_params, layer_idx=self.layer_idx)
        return g

    def forward_normal(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x = x.view(-1, hidden_dim)

        # t = self.norm(x) was done in the parent block
        t = x

        g = self.compute_gate_output(t, lora_params=lora_params)
        # Use ideal load balanced logits if enabled, otherwise use gate output
        if os.environ.get('ENABLE_PERFECT_ROUTER', '0') == '1':
            # WARNING: This discards the learned gate output and uses ideal logits for perfect load balancing
            # Only use this for testing load balancing strategies, not for actual inference
            num_tokens, num_experts = g.shape
            g = self._create_ideal_expert_load_balanced_logits(
                num_tokens=num_tokens, num_experts=num_experts, device=x.device)

        # When attention_dp is not enabled, don't pass those parameters
        expert_output = self.experts(x=t, router_logits=g)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward_attn_dp(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x = x.view(-1, hidden_dim)

        # t = self.norm(x) was done in the parent block
        t = x

        # Get attention_dp parameters
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.mapping.tp_size > 1 and all_rank_num_tokens is not None:
            if (isinstance(self.experts, (TritonFusedMoE))):
                t = allgather(t, self.mapping, dim=0, sizes=all_rank_num_tokens)

        g = self.compute_gate_output(t, lora_params=lora_params)
        # Use ideal load balanced logits if enabled, otherwise use gate output
        if os.environ.get('ENABLE_PERFECT_ROUTER', '0') == '1':
            # WARNING: This discards the learned gate output and uses ideal logits for perfect load balancing
            # Only use this for testing load balancing strategies, not for actual inference
            # The gate is still computed to maintain realistic performance measurement
            num_tokens, num_experts = g.shape
            g = self._create_ideal_expert_load_balanced_logits(
                num_tokens=num_tokens, num_experts=num_experts, device=x.device)

        # Let CutlassFusedMoE and TRTLLMGenFusedMoE handle allgather internally
        # Pass the normalized tensor (t) as input to experts, not x
        expert_output = self.experts(x=t,
                                     router_logits=g,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=False)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.enable_attention_dp:
            return self.forward_attn_dp(x, attn_metadata, residual, lora_params)
        else:
            return self.forward_normal(x, residual, lora_params)


class TransformerBlock(DecoderLayer):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        mapping = config.mapping
        self.enable_attn_dp = mapping.enable_attention_dp
        self.is_tp = mapping.has_tp() and not self.enable_attn_dp

        pretrained_config = config.pretrained_config
        self.input_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        self.attn = AttentionBlock(config,
                                   layer_idx,
                                   reduce_output=False,
                                   use_custom_cublas_mm=use_custom_cublas_mm)

        self.post_attention_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        self.mlp = MLPBlock(config,
                            layer_idx,
                            reduce_results=False,
                            use_custom_cublas_mm=use_custom_cublas_mm)

        self.mapping = config.mapping

        self.next_layer_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        # setup for tp
        self.allreduce = None
        if self.is_tp:
            self.allreduce = AllReduce(
                mapping=config.mapping,
                strategy=config.allreduce_strategy,
                dtype=config.pretrained_config.torch_dtype)

    def forward_normal(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        x, residual = self.attn(position_ids,
                                hidden_states,
                                attn_metadata,
                                residual=residual,
                                lora_params=lora_params,
                                **kwargs)
        x, residual = self.post_attention_layernorm(x, residual)

        x, residual = self.mlp(x,
                               attn_metadata,
                               residual,
                               lora_params=lora_params)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, x,
                                                      residual)

        x, residual = self.next_layer_layernorm(x, residual)
        return x, residual

    def forward_tp(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        x, residual = self.attn(
            position_ids,
            hidden_states,
            attn_metadata,
            residual=residual,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            lora_params=lora_params,
            **kwargs)

        x, residual = self.allreduce(
            x,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
                trigger_completion_at_end=False,
            ))

        x, residual = self.mlp(x,
                               attn_metadata,
                               residual,
                               lora_params=lora_params)

        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            # In eagle3 mode, we capture the value in the boundary of decoder layer.
            # If fusing rms in the next layer, the value is not correct. Thus, if
            # this layer will be captured, we should not fuse the rms in the next
            # layer.
            x = self.allreduce(x,
                               all_reduce_params=AllReduceParams(
                                   fusion_op=AllReduceFusionOp.NONE,
                                   trigger_completion_at_end=False,
                               ))
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, x,
                                                      residual)
            x, residual = self.next_layer_layernorm(x, residual)
        else:
            x, residual = self.allreduce(
                x,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ))

        return x, residual

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.is_tp:
            _forward = self.forward_tp
        else:
            _forward = self.forward_normal
        return _forward(position_ids,
                        hidden_states,
                        attn_metadata,
                        residual,
                        spec_metadata=spec_metadata,
                        lora_params=lora_params,
                        **kwargs)


class Transformer(DecoderModel):

    def __init__(self, model_config: ModelConfig[GptOssConfig]):
        super().__init__(model_config)
        config = self.model_config

        # Triton MoE kernels require installing Triton main branch,
        # which may be incompatible with torch.compile due to version mismatch.
        enable_torch_compile_for_embedding = model_config.moe_backend != "TRITON"

        # Use custom cublas since we need LUT to tune the perf.
        prop = torch.cuda.get_device_properties(0)
        sm_version = prop.major * 10 + prop.minor
        self.use_custom_cublas_mm = sm_version == 121

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embedding = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                enable_torch_compile_for_embedding=
                enable_torch_compile_for_embedding)
        else:
            self.embedding = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                enable_torch_compile_for_embedding=
                enable_torch_compile_for_embedding,
                use_custom_cublas_mm=self.use_custom_cublas_mm,
            )
        # For modeling_speculative, different name expected
        self.embed_tokens = self.embedding
        self.block = nn.ModuleList([
            TransformerBlock(
                model_config,
                layer_idx,
                use_custom_cublas_mm=self.use_custom_cublas_mm,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        hidden_states = inputs_embeds or self.embedding(input_ids)

        residual = None
        for block in self.block:
            hidden_states, residual = block(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )

        return hidden_states


@register_auto_model("GptOssForCausalLM")
class GptOssForCausalLM(SpecDecOneEngineForCausalLM[Transformer, GptOssConfig]):

    params_map = {
        # TRTLLM module name : GptOss module name
        "qkv_proj": "qkv",
        "o_proj": "out",
        "lm_head": "unembedding",
    }

    hf_params_map = {
        # TRTLLM module name : HuggingFace module name
        "embedding": "embed_tokens",
        # Order matters for attn.norm and attn.
        'attn.norm': 'input_layernorm',
        'attn': 'self_attn',
        'mlp.norm': 'post_attention_layernorm',
        'block': 'layers',
        'gate': 'router',
    }

    def __init__(
        self,
        model_config: ModelConfig[GptOssConfig],
    ):
        # Map config to HF format.
        if hasattr(model_config.pretrained_config, 'num_experts'):
            model_config.pretrained_config.num_local_experts = model_config.pretrained_config.num_experts
            model_config.pretrained_config.num_experts_per_tok = model_config.pretrained_config.experts_per_token
            model_config.pretrained_config.rope_scaling = {
                'factor':
                model_config.pretrained_config.rope_scaling_factor,
                'beta_fast':
                model_config.pretrained_config.rope_ntk_beta,
                'beta_slow':
                model_config.pretrained_config.rope_ntk_alpha,
                'original_max_position_embeddings':
                model_config.pretrained_config.initial_context_length,
            }
        if model_config.pretrained_config.torch_dtype is None:
            model_config.pretrained_config.torch_dtype = torch.bfloat16

        assert model_config.mapping.pp_size == 1, "Pipeline parallelism is not supported."

        super().__init__(
            Transformer(model_config),
            model_config=model_config,
        )

    def __post_init__(self):
        # Do not call super().__post_init__()
        params_map_reverse = {v: k for k, v in self.params_map.items()}

        quant_config = self.model_config.quant_config
        if quant_config.exclude_modules:
            if quant_config.quant_algo == "NVFP4":
                quant_config.exclude_modules = [
                    'block.*.attn.qkv',
                    'block.*.attn.out',
                    'block.*.mlp.gate',
                    'embedding',
                    'unembedding',
                ]

            for i, module in enumerate(quant_config.exclude_modules):
                names = module.split(".")
                if names[-1] in params_map_reverse:
                    names[-1] = params_map_reverse[names[-1]]
                prefix = [] if names[0] == "model" else ["model"]
                quant_config.exclude_modules[i] = '.'.join(prefix + names)

        super().apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def load_weights(self, weights: Dict):
        is_nvfp4 = self.model_config.quant_config.quant_mode.has_nvfp4()

        if is_nvfp4:
            self.load_nvfp4_weights(weights)
        else:
            self.load_hf_weights(weights)

    def post_load_weights(self):
        for idx, layer in enumerate(
                self.model.block[:self.config.num_hidden_layers]):
            if idx == 0:
                layer.input_layernorm = layer.attn.norm

            layer.post_attention_layernorm = layer.mlp.norm

            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.block[idx + 1].attn.norm

    def load_hf_weights(self, weights: Dict):
        num_expert = self.config.num_local_experts

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue

            module_weights = {}
            for k, v in self.hf_params_map.items():
                name = name.replace(k, v)

            # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
            # Currently saved MoE weights don't include 'backend' in their names.
            # After MoE refactoring, ConfigurableMoE now has a backend submodule,
            # and weights loading is done in the backend, so module name includes '.backend'.
            # We need to use parent module name (without .backend) to match saved weight names.
            # After MoE refactoring is fully complete, all paths will follow this branch.
            names = name.split('.')
            if names[-1] == "backend" and isinstance(module, MoE):
                # Backend is under experts module (ConfigurableMoE wrapper)
                name = '.'.join(names[:-1])

            module_weights = filter_weights(name, weights)

            if isinstance(module, MoE):
                try:
                    # For BF16 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights['gate_up_proj']
                    gate, up = gate_up_weight[:, :, ::2], gate_up_weight[:, :,
                                                                         1::2]
                    gate_up_weight = torch.cat([gate, up], dim=-1)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate, up = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate, up], dim=-1)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight.to(self.model.dtype)[i, :, :]
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj'][i, :, :].to(
                                self.model.dtype) for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ]
                    }
                except:
                    # For MXFP4 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights[
                        'gate_up_proj_blocks'].flatten(-2, -1)
                    gate_weight, up_weight = gate_up_weight[:, ::
                                                            2, :], gate_up_weight[:,
                                                                                  1::
                                                                                  2, :]
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=-2)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate_bias, up_bias = gate_up_bias[:, ::
                                                      2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate_bias, up_bias], dim=-1)
                    gate_up_weight_scale = module_weights['gate_up_proj_scales']
                    gate_weight_scale, up_weight_scale = gate_up_weight_scale[:, ::
                                                                              2, :], gate_up_weight_scale[:,
                                                                                                          1::
                                                                                                          2, :]
                    gate_up_weight_scale = torch.cat(
                        [gate_weight_scale, up_weight_scale], dim=-2)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj_blocks'].flatten(
                                -2, -1)[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ],
                        'gate_up_proj_weight_scale': [
                            gate_up_weight_scale[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj_weight_scale': [
                            module_weights['down_proj_scales']
                            [i, :, :].transpose(0, 1) for i in range(num_expert)
                        ]
                    }

                    if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                        for i in range(num_expert):
                            moe_weights[
                                f"{i}.w1.weight_scale_inv"] = gate_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w3.weight_scale_inv"] = up_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w2.weight_scale_inv"] = module_weights[
                                    'down_proj_scales'][i, :, :]

                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                if 'qkv' in name:
                    # For qkv_proj
                    q_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'q_proj'), weights)
                    k_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'k_proj'), weights)
                    v_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'v_proj'), weights)
                    module.load_weights(
                        weights=[q_weight_bias, k_weight_bias, v_weight_bias])
                else:
                    # For o_proj, sinks.
                    module.load_weights(weights=[module_weights])
            else:
                # Load four LN weights (attn.norm, mlp.norm, input_layernorm, post_attention_layernorm).
                if 'next_layer_layernorm' in name:
                    continue

                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])

    def load_nvfp4_weights(self, weights: Dict):
        num_expert = self.config.num_local_experts

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue

            module_weights = {}
            for k, v in self.hf_params_map.items():
                name = name.replace(k, v)

            names = name.split('.')
            if names[-1] == "backend" and isinstance(module, MoE):
                # Backend is under experts module (ConfigurableMoE wrapper)
                name = '.'.join(names[:-1])

            module_weights = filter_weights(name, weights)

            if isinstance(module, MoE):
                assert getattr(module, "quant_config", None) is not None and \
                   module.quant_config.quant_mode.has_nvfp4()
                gate_up = module_weights.get('gate_up_proj', None)
                down = module_weights.get('down_proj', None)
                gate_up_bias = module_weights.get('gate_up_proj_bias', None)
                down_bias = module_weights.get('down_proj_bias', None)

                def deinterleave(tensor):
                    g, u = tensor[..., ::2], tensor[..., 1::2]
                    return torch.cat([g, u], dim=-1)

                gate_up = deinterleave(gate_up)
                gate_up_bias = deinterleave(gate_up_bias)

                # Only fp32 bias is supported for NVFP4 MoE.
                if gate_up_bias.dtype != torch.float32:
                    gate_up_bias = gate_up_bias.to(torch.float32)
                if down_bias.dtype != torch.float32:
                    down_bias = down_bias.to(torch.float32)

                moe_weights = {}
                if gate_up is not None:
                    moe_weights['gate_up_proj'] = [
                        gate_up[i, :, :] for i in range(num_expert)
                    ]
                if down is not None:
                    moe_weights['down_proj'] = [
                        down[i, :, :] for i in range(num_expert)
                    ]
                if gate_up_bias is not None:
                    moe_weights['gate_up_proj.bias'] = [
                        gate_up_bias[i, :] for i in range(num_expert)
                    ]
                if down_bias is not None:
                    moe_weights['down_proj.bias'] = [
                        down_bias[i, :] for i in range(num_expert)
                    ]

                # Per-expert block scales (transpose to expected layout)
                if 'gate_up_proj_weight_scale' in module_weights:
                    gu_ws = module_weights['gate_up_proj_weight_scale']
                    gu_ws = deinterleave(gu_ws)
                    moe_weights['gate_up_proj_weight_scale'] = [
                        gu_ws[i, :, :] for i in range(num_expert)
                    ]
                if 'down_proj_weight_scale' in module_weights:
                    dp_ws = module_weights['down_proj_weight_scale']
                    moe_weights['down_proj_weight_scale'] = [
                        dp_ws[i, :, :] for i in range(num_expert)
                    ]

                # Module-level globals for NVFP4 loaders
                for src_key in [
                        'gate_up_proj_weight_scale_2',
                        'down_proj_weight_scale_2',
                        'gate_up_proj_input_scale',
                        'down_proj_input_scale',
                ]:
                    if src_key in module_weights:
                        moe_weights[src_key] = module_weights[src_key]

                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                if 'qkv' in name:
                    # For qkv_proj
                    q_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'q_proj'), weights)
                    k_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'k_proj'), weights)
                    v_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'v_proj'), weights)
                    module.load_weights(
                        weights=[q_weight_bias, k_weight_bias, v_weight_bias])
                else:
                    # For o_proj, sinks.
                    module.load_weights(weights=[module_weights])
            else:
                # Load four LN weights (attn.norm, mlp.norm, input_layernorm, post_attention_layernorm).
                if 'next_layer_layernorm' in name:
                    continue

                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])
