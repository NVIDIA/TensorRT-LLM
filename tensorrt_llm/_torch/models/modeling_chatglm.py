"""ChatGLM (THUDM ChatGLM3-6B) for the PyTorch backend.

ChatGLM3-6B is a dense, text-only decoder that is architecturally very close to
a Llama/Qwen2 decoder, with three source-specific contracts that this module
preserves exactly:

* Partial, interleaved (GPT-J style) rotary embedding. Only the first
  ``kv_channels / 2`` (= 64) dimensions of each 128-dim head are rotated, and
  the rotation pairs *adjacent* elements (``x[..., 0], x[..., 1]``) rather than
  splitting the head in half. This maps to TensorRT-LLM's unfused
  ``RotaryEmbedding`` with ``is_neox=False`` and a rotary ``dim`` of 64.
* A single fused ``query_key_value`` projection stored in ``[all Q, grouped K,
  grouped V]`` order with a bias, consumed here as GQA/MQA with
  ``num_key_value_heads=2`` (multi_query_group_num). The output projection and
  both MLP projections have no bias (``add_bias_linear=false``).
* SwiGLU MLP where ``dense_h_to_4h`` stores the SiLU gate as the first half and
  the up-projection as the second half.

The HF checkpoint advertises ``architectures=["ChatGLMModel"]`` and
``model_type="chatglm"``; config field names are normalized to the standard
TensorRT-LLM names in ``pyexecutor/config_utils.py`` at load time.
"""

from typing import Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


class ChatGLMAttention(Attention):
    """ChatGLM self-attention: fused QKV (with bias) + partial interleaved RoPE."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # ChatGLM applies rotary embedding to only the first half of each head
        # (partial_rotary_factor == 0.5) and rotates adjacent pairs (GPT-J /
        # interleaved), so is_neox must be False. RoPE is applied unfused so the
        # exact HF math is reproduced; the TRTLLM attention op still owns
        # masking, paged KV cache and CUDA-graph capture.
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 0.5))
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams(
                dim=rotary_dim,
                theta=getattr(config, "rope_theta", 10000.0),
                max_positions=config.max_position_embeddings,
            ),
            is_neox=False,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            # add_qkv_bias=true -> fused QKV projection carries a bias.
            bias=True,
            pos_embd_params=pos_embd_params,
            # Keep RoPE unfused to guarantee exact parity with HF's partial
            # interleaved rotary; the layer scaling coefficient cancels in the
            # source so the default 1/sqrt(head_dim) scale is correct.
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            # add_bias_linear=false -> output projection has no bias.
            dense_bias=False,
            config=model_config,
        )


class ChatGLMDecoderLayer(DecoderLayer):
    """Pre-norm decoder block (input norm -> attn -> post norm -> SwiGLU MLP).

    ChatGLM sets apply_residual_connection_post_layernorm=false, so residuals
    are taken from the pre-norm inputs, which is exactly the standard fused
    RMSNorm(add_residual) contract used across TensorRT-LLM dense decoders.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        self.self_attn = ChatGLMAttention(model_config, layer_idx=layer_idx)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            # add_bias_linear=false -> MLP projections have no bias.
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        return hidden_states, residual


class ChatGLMModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                ChatGLMDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# The checkpoint advertises architectures=["ChatGLMModel"], so register this
# causal-LM wrapper under that exact architecture string.
@register_auto_model("ChatGLMModel")
class ChatGLMForCausalLM(DecoderModelForCausalLM[ChatGLMModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            ChatGLMModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict, weight_mapper=None):
        """Rename ChatGLM's fused, sequence-major HF weights to the standard
        TensorRT-LLM per-projection names, then reuse the shared loader (which
        fuses q/k/v -> qkv_proj and gate/up -> gate_up_proj with the correct TP
        sharding and KV-head duplication)."""
        config = self.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        q_dim = config.num_attention_heads * head_dim
        kv_dim = config.num_key_value_heads * head_dim
        ffn = config.intermediate_size

        converted: Dict[str, torch.Tensor] = {}

        def rename_layer(i: int) -> None:
            src = f"transformer.encoder.layers.{i}."
            dst = f"model.layers.{i}."
            # Fused [Q, grouped K, grouped V] -> separate q/k/v; the shared
            # loader re-fuses and shards them (K/V duplicated for TP>num_kv).
            qkv_w = weights[src + "self_attention.query_key_value.weight"]
            qkv_b = weights[src + "self_attention.query_key_value.bias"]
            q_w, k_w, v_w = qkv_w.split([q_dim, kv_dim, kv_dim], dim=0)
            q_b, k_b, v_b = qkv_b.split([q_dim, kv_dim, kv_dim], dim=0)
            converted[dst + "self_attn.q_proj.weight"] = q_w
            converted[dst + "self_attn.q_proj.bias"] = q_b
            converted[dst + "self_attn.k_proj.weight"] = k_w
            converted[dst + "self_attn.k_proj.bias"] = k_b
            converted[dst + "self_attn.v_proj.weight"] = v_w
            converted[dst + "self_attn.v_proj.bias"] = v_b
            converted[dst + "self_attn.o_proj.weight"] = weights[
                src + "self_attention.dense.weight"
            ]
            # dense_h_to_4h stores [SiLU gate, up]; split into gate/up.
            gate_up = weights[src + "mlp.dense_h_to_4h.weight"]
            gate_w, up_w = gate_up.split([ffn, ffn], dim=0)
            converted[dst + "mlp.gate_proj.weight"] = gate_w
            converted[dst + "mlp.up_proj.weight"] = up_w
            converted[dst + "mlp.down_proj.weight"] = weights[src + "mlp.dense_4h_to_h.weight"]
            converted[dst + "input_layernorm.weight"] = weights[src + "input_layernorm.weight"]
            converted[dst + "post_attention_layernorm.weight"] = weights[
                src + "post_attention_layernorm.weight"
            ]

        for i in range(config.num_hidden_layers):
            rename_layer(i)

        converted["model.embed_tokens.weight"] = weights[
            "transformer.embedding.word_embeddings.weight"
        ]
        converted["model.norm.weight"] = weights["transformer.encoder.final_layernorm.weight"]
        converted["lm_head.weight"] = weights["transformer.output_layer.weight"]

        # transformer.rotary_pos_emb.inv_freq is a derived RoPE buffer, not a
        # learned parameter; the rotary cache is recomputed internally, so it is
        # intentionally not loaded.

        super().load_weights(converted)
