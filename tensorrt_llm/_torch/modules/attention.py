from typing import Optional

import torch
from torch import nn

from ..attention_backend import (AttentionInputType, AttentionMetadata,
                                 TrtllmAttention)
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import create_attention
from ..distributed import AllReduceParams, ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from .linear import Linear, WeightMode, WeightsLoadingConfig
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding


class Attention(nn.Module):

    def __init__(
        self,
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
        config: Optional[ModelConfig] = None,
    ):
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
                tensor_parallel_size=tp_size,
                tensor_parallel_rank=tp_rank,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=config.mapping.pp_size,
                parallel_rank=config.mapping.rank),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights,
        )
        self.o_proj = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                tensor_parallel_rank=tp_rank,
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=config.mapping.pp_size,
                parallel_rank=config.mapping.rank),
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights,
        )
        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend
        self.pos_embd_params = pos_embd_params
        self.rotary_emb = rotary_emb

        if not config.skip_create_weights:
            self.create_weights()

    def create_weights(self):
        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params,
            quant_config=self.quant_config,
        )

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
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
            if self.o_proj.has_fp8_qdq or self.o_proj.has_nv_fp4 or self.o_proj.has_fp8_block_scales:
                out_scale = self.o_proj.inv_input_scale
            attn_output = self.attn.forward(qkv,
                                            None,
                                            None,
                                            attn_metadata,
                                            out_scale=out_scale,
                                            attention_mask=attention_mask,
                                            mrope_config=mrope_config)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            if self.pos_embd_params is None and position_ids is not None:
                q, k = self.rotary_emb(
                    position_ids,
                    [q.contiguous(), k.contiguous()], attn_metadata)

            attn_output = self.attn.forward(q.contiguous(),
                                            k.contiguous(),
                                            v.contiguous(),
                                            attn_metadata,
                                            attention_mask=attention_mask,
                                            mrope_config=mrope_config)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MLA(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream: Optional[torch.cuda.Stream] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
    ):
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
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
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

        row_parallel_config = ParallelConfig(
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=tp_size,
            tensor_parallel_mode=TensorParallelMode.ROW,
            gpus_per_node=gpus_per_node,
            pipeline_parallel_size=config.mapping.pp_size,
            parallel_rank=config.mapping.rank,
        )
        col_parallel_config = ParallelConfig(
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=tp_size,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gpus_per_node=gpus_per_node,
            pipeline_parallel_size=config.mapping.pp_size,
            parallel_rank=config.mapping.rank,
        )

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size

        rms_norm_eps = config.pretrained_config.rms_norm_eps
        quant_config = config.get_quant_config()
        quant_mode = quant_config.quant_mode

        if not self.is_lite:
            self.fused_a = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights=config.skip_create_weights,
                use_custom_cublas_mm=True)

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)

            self.q_b_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads *
                (self.qk_nope_head_dim + self.qk_rope_head_dim),
                bias=bias,
                dtype=dtype,
                parallel_config=col_parallel_config,
                quant_config=quant_config,
                skip_create_weights=config.skip_create_weights)
        else:
            self.fused_a = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights=config.skip_create_weights,
                use_custom_cublas_mm=True)

            self.q_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads *
                (self.qk_nope_head_dim + self.qk_rope_head_dim),
                bias=bias,
                dtype=dtype,
                parallel_config=col_parallel_config,
                quant_config=quant_config,
                skip_create_weights=config.skip_create_weights)
            self.q_b_proj = self.q_proj

        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

        if quant_mode.has_fp8_block_scales():
            mla_weight_dtype = torch.float8_e4m3fn
        else:
            mla_weight_dtype = dtype

        self.kv_b_proj = Linear(self.kv_lora_rank,
                                tp_size * self.num_heads *
                                (self.qk_nope_head_dim + self.v_head_dim),
                                bias=bias,
                                dtype=dtype,
                                parallel_config=col_parallel_config,
                                quant_config=quant_config,
                                skip_create_weights=config.skip_create_weights)
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        if quant_mode.has_fp8_block_scales():
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim * tp_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            parallel_config=row_parallel_config,
            quant_config=quant_config,
            skip_create_weights=config.skip_create_weights,
        )

        self.mha = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.num_key_value_heads,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
        )

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            1,  # num_kv_heads
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
        )
        self.rotary_emb = rotary_emb
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        if self.is_lite:
            compressed_kv, k_pe = self.fused_a(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            compressed_q = hidden_states
        else:
            compressed_q, compressed_kv, k_pe = self.fused_a(
                hidden_states).split([
                    self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
                ], -1)
            do_multi_stream = torch.cuda.is_current_stream_capturing(
            ) and self.aux_stream is not None
            if do_multi_stream:
                self.ln_events[0].record()
                compressed_kv = self.kv_a_layernorm(compressed_kv)
                with torch.cuda.stream(self.aux_stream):
                    self.ln_events[0].wait()
                    compressed_q = self.q_a_layernorm(compressed_q)
                    self.ln_events[1].record()
                self.ln_events[1].wait()
            else:
                compressed_q = self.q_a_layernorm(compressed_q)
                compressed_kv = self.kv_a_layernorm(compressed_kv)

        q = self.q_b_proj(compressed_q)

        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        assert q.shape[
            0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]

            attn_output_context = self.forward_context(q_ctx, compressed_kv_ctx,
                                                       k_pe_ctx, attn_metadata)
        else:
            attn_output_context = None

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]

            attn_output_gen = self.forward_generation(q_gen, compressed_kv_gen,
                                                      k_pe_gen, attn_metadata)
        else:
            attn_output_gen = None

        # merge context and gen batches
        if attn_output_context is not None and attn_output_gen is not None:
            assert (
                len(attn_output_context.shape) == 2
            ), f"attn_output_context must be rank 2, not {len(attn_output_context.shape)}"
            assert (
                len(attn_output_gen.shape) == 2
            ), f"attn_output_gen must be rank 2, not {len(attn_output_gen.shape)}"
            attn_output = torch.cat([attn_output_context, attn_output_gen],
                                    dim=0)
        elif attn_output_gen is None:
            attn_output = attn_output_context
        else:
            attn_output = attn_output_gen

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        k = torch.empty_like(q).view(
            -1, self.num_heads, (self.qk_nope_head_dim + self.qk_rope_head_dim))
        k[..., :self.qk_nope_head_dim] = k_nope.view(-1, self.num_heads,
                                                     self.qk_nope_head_dim)
        k = k.view(
            -1,
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        # Concat q(including q_pe), k + k_pe, v together as input_qkv
        input_qkv = torch.cat([q, k, v], dim=-1)

        out_scale = getattr(self.o_proj, "inv_input_scale", None)
        attn_output = self.mha.forward(
            input_qkv,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
            out_scale=out_scale,
        )

        return attn_output

    def forward_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

        q_nope, q_pe = q.view([
            -1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        ]).split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            q_nope, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                q_nope)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            torch.ops.trtllm.fp8_block_scaling_bmm_out(
                q_nope, self.k_b_proj_trans, q_nope_scales,
                self.k_b_proj_trans_scale, q_nope_out)
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        fused_q = fused_q.view([
            num_tokens,
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        out_scale = getattr(self.o_proj, "inv_input_scale", None)
        attn_out_latent = self.mqa.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
        )
        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        attn_output = torch.empty([num_tokens, self.num_heads, self.v_head_dim],
                                  dtype=attn_out_latent.dtype,
                                  device=attn_out_latent.device)

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            attn_out_latent, attn_out_latent_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                attn_out_latent)

            torch.ops.trtllm.fp8_block_scaling_bmm_out(
                attn_out_latent, self.v_b_proj, attn_out_latent_scales,
                self.v_b_proj_scale, attn_output.transpose(0, 1))
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        # [seq, num_heads * v_head_dim]
        attn_output_flatten = attn_output.flatten(1, 2)

        return attn_output_flatten


class VanillaMLA(nn.Module):

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
                 config: Optional[ModelConfig] = None,
                 aux_stream: Optional[torch.cuda.Stream] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = hidden_size
        self.n_heads = num_attention_heads
        self.n_local_heads = num_attention_heads // config.mapping.tp_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        gpus_per_node = config.mapping.gpus_per_node
        device = torch.device('cuda')

        row_parallel_config = ParallelConfig(
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=tp_size,
            tensor_parallel_mode=TensorParallelMode.ROW,
            gpus_per_node=gpus_per_node,
        )
        col_parallel_config = ParallelConfig(
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=tp_size,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gpus_per_node=gpus_per_node,
        )

        # quantization
        quant_config = config.get_quant_config()
        quant_mode = quant_config.quant_mode

        if quant_mode.has_fp8_block_scales():
            self.mla_weight_dtype = torch.float8_e4m3fn
        else:
            self.mla_weight_dtype = dtype

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        if self.is_lite:
            self.wq = Linear(
                self.dim,
                self.n_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                parallel_config=col_parallel_config,
                quant_config=config.get_quant_config(),
                skip_create_weights=config.skip_create_weights,
            )
        else:
            self.wq_a = Linear(self.dim,
                               self.q_lora_rank,
                               bias=bias,
                               dtype=dtype,
                               quant_config=config.get_quant_config(),
                               skip_create_weights=config.skip_create_weights)
            self.q_norm = RMSNorm(hidden_size=self.q_lora_rank,
                                  eps=config.pretrained_config.rms_norm_eps,
                                  dtype=dtype)
            self.wq_b = Linear(
                self.q_lora_rank,
                self.n_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                parallel_config=col_parallel_config,
                quant_config=config.get_quant_config(),
                skip_create_weights=config.skip_create_weights,
            )
        self.wkv_a = Linear(self.dim,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                            bias=bias,
                            dtype=dtype,
                            quant_config=config.get_quant_config(),
                            skip_create_weights=config.skip_create_weights)
        self.kv_norm = RMSNorm(hidden_size=self.kv_lora_rank,
                               eps=config.pretrained_config.rms_norm_eps,
                               dtype=dtype)
        if quant_mode.has_fp8_block_scales():
            self.wkv_b = None
            self.kv_b_proj = Linear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=bias,
                dtype=dtype,
                parallel_config=col_parallel_config,
                quant_config=config.get_quant_config(),
                skip_create_weights=config.skip_create_weights,
            )
            self.k_b_proj_trans = nn.Parameter(
                torch.empty(
                    (self.n_heads // tp_size, self.kv_lora_rank,
                     self.qk_nope_head_dim),
                    dtype=self.mla_weight_dtype,
                    device=device,
                ))
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.n_heads // tp_size,
                        int(self.kv_lora_rank / 128),
                        int(self.qk_nope_head_dim / 128),
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.v_b_proj = None  # view into self.kv_b_proj.weight
            self.v_b_proj_scale = None  # view into self.kv_b_proj.weight_scale
        else:
            self.wkv_b = Linear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=bias,
                dtype=dtype,
                parallel_config=col_parallel_config,
                quant_config=config.get_quant_config(),
                skip_create_weights=config.skip_create_weights,
            )
            self.k_b_proj_trans = None
            self.k_b_proj_trans_scale = None
            self.v_b_proj = None
            self.v_b_proj_scale = None

        self.wo = Linear(
            self.n_heads * self.v_head_dim,
            self.dim,
            bias=self.dense_bias,
            dtype=dtype,
            parallel_config=row_parallel_config,
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights,
        )

        # rope
        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        self.softmax_scale = self.qk_head_dim**-0.5
        rope_scaling = getattr(config.pretrained_config, 'rope_scaling', None)
        self.rope_params = {
            "qk_rope_head_dim": config.pretrained_config.qk_rope_head_dim,
            "rope_theta": config.pretrained_config.rope_theta,
        }
        if rope_scaling is not None:
            self.rope_params.update({
                "beta_fast":
                rope_scaling.get("beta_fast", 32),
                "beta_slow":
                rope_scaling.get("beta_slow", 1),
                "original_seq_len":
                rope_scaling.get("original_max_position_embeddings", 1024),
                "rope_factor":
                rope_scaling.get("factor", 1.0),
            })
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            scaling_factor = rope_scaling.get("factor", 1.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.freqs_cis = None

    def apply_rotary_emb(self, x: torch.Tensor,
                         freqs_cis: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(x.size(0), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(2)
        return y.to(dtype)

    def precompute_freqs_cis(
        self,
        max_seq_len: int,
        qk_rope_head_dim: int,
        beta_fast: int = 32,
        beta_slow: int = 1,
        original_seq_len: int = 4096,
        rope_factor: float = 40,
        rope_theta: float = 10000,
    ) -> torch.Tensor:
        dim = qk_rope_head_dim
        seqlen = max_seq_len
        beta_fast = beta_fast
        beta_slow = beta_slow
        base = rope_theta
        factor = rope_factor

        import math

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(
                max_seq_len /
                (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(
                find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(
                find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) -
                           min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        freqs = 1.0 / (base
                       **(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if seqlen > original_seq_len:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                              original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def _single_request_update_kv_cache(self, kv, k_pe, kv_cache_tensor,
                                        cache_idx, start_pos, end_pos):
        kv_cache = kv_cache_tensor[cache_idx,
                                   0, :, :, :self.kv_lora_rank].squeeze()
        pe_cache = kv_cache_tensor[cache_idx, 0, :, :,
                                   self.kv_lora_rank:].squeeze()
        kv_cache[start_pos:end_pos] = kv
        pe_cache[start_pos:end_pos] = k_pe
        return kv_cache[:end_pos, :], pe_cache[:end_pos, :]

    def _single_request_forward(self, x: torch.Tensor,
                                kv_cache_tensor: torch.Tensor, start_pos: int,
                                cache_idx: int):
        seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        # rope param
        if end_pos > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(max_seq_len=end_pos,
                                                       **self.rope_params).to(
                                                           x.device)
        # get mask
        mask = None
        if seqlen > 1:
            mask = torch.full((end_pos, end_pos), float("-inf"),
                              device='cuda').triu_(1)
            mask = mask[-seqlen:]
        # proj
        if self.is_lite:
            q = self.wq(x)
        else:
            qnorm = self.q_norm(self.wq_a(x))
            q = self.wq_b(qnorm)
        # q rope
        q = q.view(seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.apply_rotary_emb(q_pe, self.freqs_cis[start_pos:end_pos])
        # kv proj a
        kv = self.wkv_a(x)
        # kv rope
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                               dim=-1)
        k_pe = self.apply_rotary_emb(k_pe.unsqueeze(1),
                                     self.freqs_cis[start_pos:end_pos])
        # q_nope proj
        if self.mla_weight_dtype == torch.bfloat16:
            wkv_b = self.wkv_b.weight.view(self.n_local_heads, -1,
                                           self.kv_lora_rank)
            q_nope = torch.einsum("shd,hdc->shc", q_nope,
                                  wkv_b[:, :self.qk_nope_head_dim])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            q_nope, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                q_nope)
            q_nope = torch.ops.trtllm.fp8_block_scaling_bmm(
                q_nope, self.k_b_proj_trans, q_nope_scales,
                self.k_b_proj_trans_scale)
            q_nope = q_nope.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # update kv cache
        # Fake QDQ
        q_nope = q_nope.to(torch.float8_e4m3fn).to(torch.bfloat16)
        q_pe = q_pe.to(torch.float8_e4m3fn).to(torch.bfloat16)

        kv_states, pe_states = self._single_request_update_kv_cache(
            self.kv_norm(kv), k_pe.squeeze(1), kv_cache_tensor, cache_idx,
            start_pos, end_pos)

        kv_states = kv_states.to(torch.float8_e4m3fn).to(torch.bfloat16)
        pe_states = pe_states.to(torch.float8_e4m3fn).to(torch.bfloat16)
        # attention
        scores = (torch.einsum("shc,tc->sht", q_nope, kv_states) + torch.einsum(
            "shr,tr->sht", q_pe, pe_states)) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # Fake QDQ
        scores = (scores * 448).to(torch.float8_e4m3fn).to(torch.bfloat16) / 448
        x = torch.einsum("sht,tc->shc", scores, kv_states)
        # v proj
        if self.mla_weight_dtype == torch.bfloat16:
            x = torch.einsum("shc,hdc->shd", x, wkv_b[:, -self.v_head_dim:])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            x, x_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                x)
            x = torch.ops.trtllm.fp8_block_scaling_bmm(x, self.v_b_proj,
                                                       x_scales,
                                                       self.v_b_proj_scale)
            x = x.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # proj
        x = self.wo(x.flatten(1))
        return x

    def dummy_forward(self, x: torch.Tensor):
        seqlen, hidden_dim = x.size()
        end_pos = seqlen
        # rope param
        if self.freqs_cis is None or seqlen > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(max_seq_len=seqlen,
                                                       **self.rope_params).to(
                                                           x.device)
        # get mask
        mask = None
        if seqlen > 1:
            mask = torch.full((end_pos, end_pos), float("-inf"),
                              device='cuda').triu_(1)
            mask = mask[-seqlen:]
        # proj
        if self.is_lite:
            q = self.wq(x)
        else:
            q = self.wq_a(x.view(-1, hidden_dim))
            q = self.q_norm(q).type_as(x)
            q = self.wq_b(q).view(seqlen, -1)
        # q rope
        q = q.view(seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.apply_rotary_emb(q_pe, self.freqs_cis[0:end_pos])
        # kv proj a
        kv = self.wkv_a(x.view(-1, hidden_dim)).view(seqlen, -1)
        # kv rope
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                               dim=-1)
        k_pe = self.apply_rotary_emb(k_pe.unsqueeze(1),
                                     self.freqs_cis[0:end_pos])
        # q_nope proj
        if self.mla_weight_dtype == torch.bfloat16:
            wkv_b = self.wkv_b.weight.view(self.n_local_heads, -1,
                                           self.kv_lora_rank)
            q_nope = torch.einsum("shd,hdc->shc", q_nope,
                                  wkv_b[:, :self.qk_nope_head_dim])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            q_nope, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                q_nope)
            q_nope = torch.ops.trtllm.fp8_block_scaling_bmm(
                q_nope, self.k_b_proj_trans, q_nope_scales,
                self.k_b_proj_trans_scale)
            q_nope = q_nope.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # get kv and pe states
        kv_states = self.kv_norm(kv).type(self.dtype)
        pe_states = k_pe.squeeze(1)
        # attention
        scores = (torch.einsum("shc,tc->sht", q_nope, kv_states) + torch.einsum(
            "shr,tr->sht", q_pe, pe_states)) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("sht,tc->shc", scores, kv_states)
        # v proj
        if self.mla_weight_dtype == torch.bfloat16:
            x = torch.einsum("shc,hdc->shd", x, wkv_b[:, -self.v_head_dim:])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            x, x_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                x)
            x = torch.ops.trtllm.fp8_block_scaling_bmm(x, self.v_b_proj,
                                                       x_scales,
                                                       self.v_b_proj_scale)
            x = x.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # proj
        x = self.wo(x.flatten(1))
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        attn_metadata: Optional[AttentionMetadata],
        **kwargs,
    ) -> torch.Tensor:

        if attn_metadata is None or attn_metadata.kv_cache_manager is None:
            return self.dummy_forward(hidden_states)

        max_seq_len = attn_metadata.kv_cache_manager.max_seq_len
        if self.freqs_cis is None or max_seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(max_seq_len=max_seq_len,
                                                       **self.rope_params).to(
                                                           hidden_states.device)

        past_seen_tokens = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [
            block_ids[0] for block_ids in attn_metadata.block_ids_per_seq
        ]
        kv_cache_tensor = attn_metadata.kv_cache_manager.get_buffers(
            self.layer_idx)

        assert len(cache_indices) == len(past_seen_tokens)
        assert len(cache_indices) == attn_metadata.seq_lens.nelement()

        offset = 0
        attn_outputs = []
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            single_hidden_state = hidden_states[offset:offset + seq_len, :]
            past_seen_token = past_seen_tokens[i]
            cache_idx = cache_indices[i]
            attn_output = self._single_request_forward(single_hidden_state,
                                                       kv_cache_tensor,
                                                       past_seen_token,
                                                       cache_idx)
            attn_outputs.append(attn_output)
            offset += seq_len

        attn_output = torch.cat(attn_outputs, dim=0).contiguous()
        return attn_output
