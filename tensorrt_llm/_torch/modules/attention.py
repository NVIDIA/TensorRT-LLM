from typing import Optional

import torch
from torch import nn

from ..attention_backend import AttentionMetadata, TrtllmAttention
from ..attention_backend.interface import PositionalEmbeddingParams
from ..attention_backend.utils import create_attention
from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from .linear import Linear, WeightMode, WeightsLoadingConfig
from .rms_norm import RMSNorm
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
        gpus_per_node = config.mapping.gpus_per_node
        if config.mapping.enable_attention_dp:
            tp_size = 1
            tp_rank = 0

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
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node),
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
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node),
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


class MLA(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 qk_nope_head_dim: int,
                 qk_rope_head_dim: int,
                 v_head_dim: int,
                 q_lora_rank: int,
                 kv_lora_rank: int,
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
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

        # tensor parallel
        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        gpus_per_node = config.mapping.gpus_per_node
        if config.mapping.enable_attention_dp:
            tp_size = 1
            tp_rank = 0

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size

        rms_norm_eps = config.pretrained_config.rms_norm_eps

        if not self.is_lite:
            self.fused_a = Linear(hidden_size,
                                  self.q_lora_rank + self.kv_lora_rank +
                                  self.qk_rope_head_dim,
                                  bias=bias,
                                  dtype=dtype,
                                  quant_config=config.get_quant_config())

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)
        else:
            self.fused_a = Linear(hidden_size,
                                  self.kv_lora_rank + self.qk_rope_head_dim,
                                  bias=bias,
                                  dtype=dtype,
                                  quant_config=config.get_quant_config())
        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

        quant_config = config.get_quant_config()
        quant_mode = quant_config.quant_mode

        if quant_mode.has_fp8_block_scales():
            self.kv_b_proj_scale = nn.Parameter(torch.empty(
                (int(self.num_heads * self.qk_nope_head_dim / 128 * 2),
                 int(self.kv_lora_rank / 128)),
                dtype=torch.float32),
                                                requires_grad=False)

            self.k_b_proj_trans_scale = nn.Parameter(torch.empty(
                (int(self.num_heads * self.kv_lora_rank / 128),
                 int(self.qk_nope_head_dim / 128)),
                dtype=torch.float32),
                                                     requires_grad=False)

            self.q_b_proj_scale = nn.Parameter(torch.empty(
                (int(self.num_heads *
                     (self.qk_nope_head_dim + self.qk_rope_head_dim) / 128),
                 int(self.q_lora_rank / 128)),
                dtype=torch.float32),
                                               requires_grad=False)
        else:
            self.kv_b_proj_scale = None
            self.k_b_proj_trans_scale = None
            self.q_b_proj_scale = None

        if quant_mode.has_fp8_block_scales():
            mla_weight_dtype = torch.float8_e4m3fn
        else:
            mla_weight_dtype = dtype

        self.kv_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads * self.qk_nope_head_dim * 2, self.kv_lora_rank),
                dtype=mla_weight_dtype))
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads * self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype))

        self.q_b_proj = nn.Parameter(
            torch.empty((self.num_heads *
                         (self.qk_nope_head_dim + self.qk_rope_head_dim),
                         self.q_lora_rank),
                        dtype=mla_weight_dtype))

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim * tp_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node),
            quant_config=config.get_quant_config(),
        )

        self.attn = create_attention(config.attn_backend,
                                     self.layer_idx,
                                     self.num_heads,
                                     self.head_dim,
                                     self.num_key_value_heads,
                                     pos_embd_params=pos_embd_params,
                                     quant_config=config.get_quant_config(),
                                     is_mla_enable=True,
                                     q_lora_rank=self.q_lora_rank,
                                     kv_lora_rank=self.kv_lora_rank,
                                     qk_nope_head_dim=self.qk_nope_head_dim,
                                     qk_rope_head_dim=self.qk_rope_head_dim,
                                     v_head_dim=self.v_head_dim)
        self.rotary_emb = rotary_emb

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_lite:
            compressed_kv, k_pe = self.fused_a(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = compressed_kv.contiguous()
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            input_qkv = torch.concat([hidden_states, compressed_kv, k_pe],
                                     dim=-1)
        else:
            compressed_q, compressed_kv, k_pe = self.fused_a(
                hidden_states).split([
                    self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
                ], -1)
            compressed_q = compressed_q.contiguous()
            compressed_kv = compressed_kv.contiguous()
            compressed_q = self.q_a_layernorm(compressed_q)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            input_qkv = torch.concat([compressed_q, compressed_kv, k_pe],
                                     dim=-1)

        out_scale = getattr(self.o_proj, "inv_input_scale", None)
        attn_output = self.attn.forward(
            input_qkv,
            None,
            None,
            attn_metadata,
            out_scale=out_scale,
            q_b_proj=self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            k_b_proj_trans=self.k_b_proj_trans,
            q_b_proj_scale=self.q_b_proj_scale,
            kv_b_proj_scale=self.kv_b_proj_scale,
            k_b_proj_trans_scale=self.k_b_proj_trans_scale,
        )

        attn_output = self.o_proj(attn_output)

        return attn_output
