from typing import Optional

import torch
from torch import nn

from ..attention_backend import AttentionMetadata, TrtllmAttention
from ..attention_backend.interface import PositionalEmbeddingParams
from ..attention_backend.utils import create_attention
from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from .linear import Linear, WeightMode, WeightsLoadingConfig
from .rotary_embedding import RotaryEmbedding


class Attention(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 max_position_embeddings: int,
                 bias: bool,
                 pos_embd_params: Optional[PositionalEmbeddingParams] = None,
                 rotary_emb: Optional[RotaryEmbedding] = None,
                 layer_idx: Optional[int] = None,
                 dtype: torch.dtype = None,
                 dense_bias: Optional[bool] = None,
                 config: Optional[ModelConfig] = None):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

        # tensor parallel
        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
        )
        self.o_proj = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.ROW),
            quant_config=config.get_quant_config(),
        )

        self.attn = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=pos_embd_params,
            quant_config=config.get_quant_config(),
        )
        self.rotary_emb = rotary_emb

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        is_fused_qkv = False
        if isinstance(self.attn, TrtllmAttention):
            is_fused_qkv = True

        if is_fused_qkv:
            if self.pos_embd_params is None and position_ids is not None:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                    dim=-1)
                q, k = self.rotary_emb(
                    position_ids,
                    [q.contiguous(), k.contiguous()], attn_metadata)
                qkv = torch.concat(
                    [q.contiguous(),
                     k.contiguous(),
                     v.contiguous()], dim=-1)

            out_scale = None
            if self.o_proj.has_fp8_qdq or self.o_proj.has_nv_fp4:
                out_scale = self.o_proj.inv_input_scale
            attn_output = self.attn.forward(
                qkv,
                None,
                None,
                attn_metadata,
                out_scale=out_scale,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            if self.pos_embd_params is None and position_ids is not None:
                q, k = self.rotary_emb(
                    position_ids,
                    [q.contiguous(), k.contiguous()], attn_metadata)

            attn_output = self.attn.forward(q.contiguous(), k.contiguous(),
                                            v.contiguous(), attn_metadata)

        attn_output = self.o_proj(attn_output)

        return attn_output
