# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-specific orchestration for the generic MLA module."""

import os
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.sparse.mla import (
    forward_context_sparse_mla,
    forward_generation_sparse_mla,
)
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._utils import is_sm_100f


def _fp8_block_scaling_bmm_out(*args, **kwargs):
    from tensorrt_llm._torch.modules.attention import fp8_block_scaling_bmm_out

    return fp8_block_scaling_bmm_out(*args, **kwargs)


def deepseek_v4_q_b_layernorm(self, q: torch.Tensor) -> torch.Tensor:
    assert q.dim() == 2 and q.shape[1] == self.num_heads_tp * self.qk_head_dim
    return torch.ops.trtllm.deepseek_v4_q_norm(
        q, self.num_heads_tp, self.qk_head_dim, float(self.q_b_layernorm.variance_epsilon)
    )


def deepseek_v4_o_proj(
    self, attn_out_latent: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    num_tokens = attn_out_latent.shape[0]
    attn_out_latent = attn_out_latent.view(num_tokens, self.num_heads_tp, -1)

    # When o_a_proj is FP8 and the cute_dsl FP8 BMM is enabled on SM100,
    # fuse the inverse-RoPE into the FP8-quant epilogue (vLLM-ported
    # Triton kernel) and call cute_dsl_fp8_bmm_blackwell directly. Saves
    # one BF16 read+write of the latent vs the
    # mla_rope_inplace + fp8_batched_quantize_1x128_permute102 pair.
    fused_inv_rope_fp8 = (
        self.o_a_proj.dtype == torch.float8_e4m3fn
        and self.use_cute_dsl_blockscaling_bmm
        and is_sm_100f()
    )
    if fused_inv_rope_fp8:
        heads_per_group = self.num_heads_tp // self.n_local_groups
        attn_fp8, attn_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
            attn_out_latent,
            position_ids.view(-1),
            self.inverse_rotary_emb.rotary_cos_sin,
            self.n_local_groups,
            heads_per_group,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            128,
            self.inverse_rotary_emb.is_neox,
        )
        o_lora = torch.empty(
            [num_tokens, self.n_local_groups, self.o_lora_rank],
            device=attn_out_latent.device,
            dtype=self.dtype,
        )
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(
            attn_fp8, self.o_a_proj, attn_scale, self.o_a_proj_scale, o_lora.transpose(0, 1)
        )
        o_lora = o_lora.flatten(1)
        return self.o_b_proj(o_lora)

    # Fused in-place inverse RoPE on the rope portion of each head
    torch.ops.trtllm.mla_rope_inplace(
        attn_out_latent,
        position_ids.view(-1),
        self.inverse_rotary_emb.rotary_cos_sin,
        self.num_heads_tp,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
        True,
        self.inverse_rotary_emb.is_neox,
    )

    # Output projections
    o_lora = torch.empty(
        [num_tokens, self.n_local_groups, self.o_lora_rank],
        device=attn_out_latent.device,
        dtype=attn_out_latent.dtype,
    )
    if self.o_a_proj.dtype == torch.bfloat16:
        # dim = head_dim * num_head // num_group
        # [num_groups, num_tokens, dim] x [num_groups, dim, o_lora_rank]
        # -> [num_groups, num_tokens, o_lora_rank]
        torch.ops.trtllm.bmm_out(
            attn_out_latent.view(num_tokens, self.n_local_groups, -1).transpose(0, 1),
            self.o_a_proj.transpose(1, 2),
            o_lora.transpose(0, 1),
        )
    elif self.o_a_proj.dtype == torch.float8_e4m3fn:
        _fp8_block_scaling_bmm_out(
            attn_out_latent.view(num_tokens, self.n_local_groups, -1),
            self.o_a_proj,
            self.o_a_proj_scale,
            o_lora.transpose(0, 1),
            self.o_a_proj_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.o_a_proj.dtype}.")
    o_lora = o_lora.flatten(1)
    output = self.o_b_proj(o_lora)
    return output


def forward_impl_with_deepseek_v4(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
) -> None:
    """
    Forward pass for the MLA module with DeepSeek-V4 (always in MQA mode).

    Args:
        position_ids (Optional[torch.IntTensor]): The position IDs.
        hidden_states (torch.Tensor): The hidden states.
        attn_metadata (AttentionMetadata): The attention metadata.

    Returns:
        torch.Tensor: The output tensor.
    """
    assert self.mha is None and self.mqa is not None, "DeepSeek-V4 is only supported in MQA mode"
    # split q, k, v into context and gen batches
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens

    hidden_states = hidden_states[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    q, kv = self.kv_a_proj_with_mqa(hidden_states).split(
        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], -1
    )

    q, kv = maybe_execute_in_parallel(
        lambda: self.q_a_layernorm(q),
        lambda: self.kv_a_layernorm(kv),
        self.ln_events[0],
        self.ln_events[1],
        self.aux_stream,
    )
    compressed_kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
    qr = q
    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

    # TRTLLM_MLA_EXTRA_OVERLAP=1 reorders the V4 attention prologue so the
    # outer compressor and the ratio-4 indexer can execute concurrently
    # with q_b_proj + q_b_layernorm. The indexer is launched on a
    # dedicated stream and still uses a different aux stream for its
    # internal q-proj/weights-proj split.
    _v4_extra_overlap = (
        os.environ.get("TRTLLM_MLA_EXTRA_OVERLAP", "1") == "1"
        and self.compressor is not None
        and self.aux_stream is not None
    )

    def _q_branch():
        q_proj = self.q_b_proj(q)
        return deepseek_v4_q_b_layernorm(self, q_proj)

    def _compressor_branch():
        self.compressor(hidden_states, attn_metadata)
        return None

    def _indexer_branch():
        return self.indexer(
            qr,
            hidden_states,
            attn_metadata,
            position_ids,
        )

    topk_indices = None
    indexer_ran = False
    if _v4_extra_overlap:
        use_indexer_overlap = (
            do_multi_stream() and self.indexer is not None and self.indexer_stream is not None
        )
        if use_indexer_overlap:
            self.dsv4_overlap_start_event.record()

            with torch.cuda.stream(self.aux_stream):
                self.dsv4_overlap_start_event.wait()
                _compressor_branch()
                self.dsv4_compressor_event.record()

            with torch.cuda.stream(self.indexer_stream):
                self.dsv4_overlap_start_event.wait()
                topk_indices = _indexer_branch()
                indexer_ran = True
                self.dsv4_indexer_event.record()

            q = _q_branch()
            self.dsv4_compressor_event.wait()
            self.dsv4_indexer_event.wait()
        else:
            q, _ = maybe_execute_in_parallel(
                _q_branch,
                _compressor_branch,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
    else:
        q = _q_branch()
        if self.compressor is not None:
            self.compressor(hidden_states, attn_metadata)

    if self.indexer is not None:
        if not indexer_ran:
            topk_indices = _indexer_branch()

    assert q.shape[0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

    assert output is not None, "output must be provided"

    if num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        topk_indices_ctx = topk_indices[:num_ctx_tokens, :] if topk_indices is not None else None
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        ctx_position_ids = position_ids[..., :num_ctx_tokens] if position_ids is not None else None
        if self.apply_rotary_emb:
            assert ctx_position_ids is not None
            k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)

        forward_context_sparse_mla(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            position_ids=ctx_position_ids,
            latent_cache=latent_cache_ctx,
            topk_indices=topk_indices_ctx,
        )

    if num_generations > 0:
        q_gen = q[num_ctx_tokens:, ...]
        topk_indices_gen = (
            topk_indices[num_ctx_tokens:num_tokens, :] if topk_indices is not None else None
        )
        compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
        k_pe_gen = k_pe[num_ctx_tokens:, ...]
        latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
        gen_position_ids = (
            position_ids[..., num_ctx_tokens:num_tokens] if position_ids is not None else None
        )
        if self.apply_rotary_emb:
            assert gen_position_ids is not None
            k_pe_gen = self.apply_rope(q_gen, k_pe_gen, gen_position_ids)

        forward_generation_sparse_mla(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            position_ids=gen_position_ids,
            latent_cache=latent_cache_gen,
            topk_indices=topk_indices_gen,
        )
