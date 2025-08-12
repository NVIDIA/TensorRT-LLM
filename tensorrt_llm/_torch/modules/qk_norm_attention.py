from typing import Optional

import torch

from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PositionEmbeddingType, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm


class QKNormRoPEAttention(Attention):
    """
    QKNormRoPEAttention is a custom attention layer that applies QK norm and RoPE to the input tensor.
    It is used in the Qwen3 model.
    It is a subclass of Attention, and overrides the apply_rope method to apply QK norm and RoPE.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: Optional[int] = None,
        fuse_qk_norm_rope: bool = True,
    ):
        config = model_config.pretrained_config

        if getattr(config, "rope_scaling", None) is not None:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string(
                    config.rope_scaling["type"]),
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        self.fuse_qk_norm_rope = fuse_qk_norm_rope

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            rope_fusion=not self.
            fuse_qk_norm_rope,  # If fuse_qk_norm_rope is true, do not apply fused RoPE in attention OP, and self.rotary_emb will be skipped in the overridden apply_rope.
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=config.attention_bias,
            config=model_config,
        )

        self.q_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=1e-6,
                              dtype=config.torch_dtype,
                              has_weights=True)
        self.k_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=1e-6,
                              dtype=config.torch_dtype,
                              has_weights=True)
        self.aux_stream = torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.q_norm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.k_norm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def apply_qk_norm_rope(self, qkv, position_ids):
        torch.ops.trtllm.fused_qk_norm_rope(
            qkv, self.num_heads, self.num_key_value_heads,
            self.num_key_value_heads, self.head_dim,
            self.q_norm.variance_epsilon, self.q_norm.weight,
            self.k_norm.weight, self.pos_embd_params.rope.theta,
            self.pos_embd_params.is_neox, position_ids.view(-1))
        return qkv, None, None

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        The apply_rope method is called in the forward method of the Attention class.
        The apply_rope method is overridden in this class to apply QK norm and RoPE to the input tensor.
        """
        # Qwen3 applies QK norm before RoPE.
        if not self.fuse_qk_norm_rope:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.apply_qk_norm(q, k)
            return super().apply_rope(q, k, v, position_ids)

        assert k is None and v is None, "The input should be a concatenated qkv tensor to apply_qk_norm_rope"
        qkv = q
        return self.apply_qk_norm_rope(qkv, position_ids)
