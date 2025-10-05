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

from typing import Dict, Iterable, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from .modeling_deepseekv3 import DeepseekV3DecoderLayer, DeepseekV3ForCausalLM
from .modeling_utils import _load_weights_impl, register_auto_model


class GLMAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        fuse_qk_norm_rope: bool = True,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams.from_config(config),
            is_neox=True,
        )

        self.fuse_qk_norm_rope = fuse_qk_norm_rope

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            rope_fusion=not self.fuse_qk_norm_rope,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
        )
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(hidden_size=self.head_dim,
                                  eps=1e-6,
                                  dtype=config.torch_dtype,
                                  has_weights=True)
            self.k_norm = RMSNorm(hidden_size=self.head_dim,
                                  eps=1e-6,
                                  dtype=config.torch_dtype,
                                  has_weights=True)
        self.aux_stream = aux_stream
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
        if not self.use_qk_norm:
            return super().apply_rope(q, k, v, position_ids)
        if not self.fuse_qk_norm_rope:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.apply_qk_norm(q, k)
            return super().apply_rope(q, k, v, position_ids)

        assert k is None and v is None, (
            "The input should be a concatenated qkv tensor to apply_qk_norm_rope"
        )
        qkv = q
        return self.apply_qk_norm_rope(qkv, position_ids)


def _replace_attention_with_glm(layers: Iterable[nn.Module]) -> None:
    for layer in layers:
        if isinstance(layer, DeepseekV3DecoderLayer):
            aux_stream = getattr(layer.self_attn, "aux_stream", None)
            fuse_rope = getattr(layer.self_attn, "fuse_qk_norm_rope", True)
            layer.self_attn = GLMAttention(layer.model_config,
                                           layer_idx=layer.layer_idx,
                                           fuse_qk_norm_rope=fuse_rope,
                                           aux_stream=aux_stream)


@register_auto_model("Glm4MoeForCausalLM")
class Glm4MoeForCausalLM(DeepseekV3ForCausalLM):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        _replace_attention_with_glm(self.model.layers)
        if hasattr(self, "draft_model") and self.draft_model is not None:
            _replace_attention_with_glm(getattr(self.draft_model, "layers", []))
            _replace_attention_with_glm(
                getattr(self.draft_model, "mtp_layers", []))
        if hasattr(self, "epilogue"):
            _replace_attention_with_glm(self.epilogue)

    def load_weights(self, weights: Dict):
        _load_weights_impl(
            self,
            weights,
            params_map={
                r'(?!.*shared_experts)(?=.*experts?)(.*?)up_proj(.*)':
                r'\1w3\2',
                r'(?!.*shared_experts)(?=.*experts?)(.*?)down_proj(.*)':
                r'\1w2\2',
                r'(?!.*shared_experts)(?=.*experts?)(.*?)gate_proj(.*)':
                r'\1w1\2',
            })
