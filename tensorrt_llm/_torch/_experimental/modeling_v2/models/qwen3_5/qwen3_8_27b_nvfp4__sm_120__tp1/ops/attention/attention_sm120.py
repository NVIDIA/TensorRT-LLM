# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention of the 16 attention layers on the target's own kernels
(``native_ops.attention_sm120``, built from ``csrc/attentionSm120Op.cpp``).

It takes the fused ``[Q | K | V]`` rows ``qkv_gemma_rmsnorm_rope_gate`` wrote (RoPE and the
q/k norms applied) and, for one engine step:

* context rows: K/V appended to the fp8 paged cache and Q quantized to fp8 (TensorRT-LLM's
  exported ``invokeQKVPreprocessing``), then the target's SM120 e4m3 head-256 causal
  paged-KV flash kernel (``csrc/fmha``: base-2 softmax, last q-tile split along KV for 1-3
  contexts, and on request the leading q-tiles left unwritten);
* generation rows: the new token's K/V appended, then the target's XQA decode kernel
  (``csrc/xqa``, compiled ahead of time) with the multi-block count capped at one CTA wave
  and the shared-prefix pass from ``shared_prefix_min_batch`` requests.

``AttentionSm120Buffers`` owns the op's workspace and its semaphores (zeroed once, kept for
the life of the KV cache binding: the decode kernel leaves them at zero).
"""

from __future__ import annotations

import torch

from ... import native_ops


class AttentionSm120Buffers:
    """The op's workspace for up to ``max_tokens`` context tokens and ``max_seqs`` sequences
    per step, and its semaphores."""

    def __init__(self, max_tokens: int, max_seqs: int, device: torch.device) -> None:
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.workspace = torch.empty(
            native_ops.attention_sm120_workspace_bytes(max_tokens, max_seqs),
            dtype=torch.uint8,
            device=device,
        )
        self.semaphores = torch.zeros(
            native_ops.attention_sm120_semaphore_words(max_seqs), dtype=torch.int32, device=device
        )


def attention_sm120(
    qkv: torch.Tensor,
    output: torch.Tensor,
    *,
    num_contexts: int,
    num_ctx_tokens: int,
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    context_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    layer_idx: int,
    tokens_per_block: int,
    max_seq_len: int,
    kv_scale_orig_quant: torch.Tensor,
    kv_scale_quant_orig: torch.Tensor,
    buffers: AttentionSm120Buffers,
    q_scaling: float,
    skip_leading_q_tiles: bool,
    shared_prefix_min_batch: int,
) -> None:
    """Write the attention of ``qkv`` ([tokens, 8192] bf16) into ``output`` ([tokens, 6144])."""
    native_ops.attention_sm120(
        qkv,
        output,
        num_contexts,
        num_ctx_tokens,
        sequence_length,
        host_past_key_value_lengths,
        context_lengths,
        host_context_lengths,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        layer_idx,
        tokens_per_block,
        max_seq_len,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        buffers.workspace,
        buffers.max_tokens,
        buffers.max_seqs,
        buffers.semaphores,
        q_scaling,
        skip_leading_q_tiles,
        shared_prefix_min_batch,
    )
