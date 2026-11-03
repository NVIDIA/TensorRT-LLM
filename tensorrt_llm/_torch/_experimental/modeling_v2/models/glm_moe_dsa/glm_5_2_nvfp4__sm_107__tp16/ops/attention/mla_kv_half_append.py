# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV half of an MLA generation step: kv_a RMSNorm, K RoPE and the FP8 paged-cache append."""

from __future__ import annotations

import math

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The latent row: kv_lora_rank 512 normalized columns followed by qk_rope_head_dim 64 K-RoPE
#: columns.
KV_LORA_RANK = 512
ROPE_DIM = 64
#: Where the latent starts in fused-A's output row (after q_lora_rank 2048 q_a columns).
LATENT_OFFSET = 2048
#: The E4M3 range the kernel saturates to.
_E4M3_MAX = 448.0
#: log2(e) as the kernel's fp32 constant.
_LOG2E = 1.4426950408889634

#: GLM-5.2's pool page, kv_a_layernorm epsilon, 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)
#: and rope_theta.
_GLM = dict(
    tokens_per_block=64,
    kv_norm_eps=1e-5,
    softmax_scale=1.0 / math.sqrt(256.0),
    rope_theta=8_000_000.0,
)


class GlmMlaKvHalfAppend(OpWrapper):
    """One generation step's MLA KV half over a paged FP8 latent cache.

    Wraps ``torch.ops.trtllm.glm_mla_kv_half_append``. Row ``t`` of ``latent`` is token
    ``t % seq_len_q`` of generation sequence ``s = t // seq_len_q`` and sits at position
    ``p = sequence_length[num_contexts + s] - seq_len_q + t % seq_len_q``. Its cache row is::

        e4m3(rmsnorm(latent[t, :512], kv_norm_weight, kv_norm_eps))   # x * (rstd * w), fp32
        e4m3(gptj_rope(latent[t, 512:576], position p))                # pairs (2i, 2i + 1), fp32

    saturated to +-448 at unit scale, written to layer ``layer_idx`` of the paged pool at
    position ``p`` through the same block offsets, pool pointers and pool mapping as
    ``mla_rope_generation``. A negative position is skipped. The call also fills
    ``mla_bmm1_scale = [softmax_scale, softmax_scale * log2(e)]`` and ``mla_bmm2_scale = [1]``,
    the scales the generation attention reads. It returns nothing.

    Values reach E4M3 straight from fp32: unlike ``mla_rope_generation``'s KV half, neither
    the normalized latent nor the rotated K is rounded to BF16 first.

    Fusion boundary: the RMSNorm of the latent, the RoPE of K, the quantization, the append
    and the scales are inside. kv_a_proj, the query and the attention are not.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="verify and MTP step 0: 8 rows of fused-A's 2624-column output across a page edge",
            spec=dict(
                seq_len_q=8,
                cached=(4_090,),
                latent_width=2624,
                num_layers=4,
                layer_idx=1,
                trigger_at_entry=True,
                **_GLM,
            ),
        ),
        Cell(
            why="full-indexer layers: the latent inside fused-A's 2784-column output (wk|wp rows)",
            spec=dict(
                seq_len_q=8,
                cached=(4_090,),
                latent_width=2784,
                num_layers=4,
                layer_idx=0,
                trigger_at_entry=True,
                **_GLM,
            ),
        ),
        Cell(
            why="MTP draft steps 1-6: one row at the MTP layer, the last layer of the pool",
            spec=dict(
                seq_len_q=1,
                cached=(4_100,),
                latent_width=2624,
                num_layers=4,
                layer_idx=3,
                trigger_at_entry=True,
                **_GLM,
            ),
        ),
        Cell(
            why="two sequences (max_batch_size 2): per-sequence offsets and positions, end trigger",
            spec=dict(
                seq_len_q=8,
                cached=(700, 4_090),
                latent_width=2624,
                num_layers=4,
                layer_idx=2,
                trigger_at_entry=False,
                **_GLM,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. What the target passes:
      - `latent`: fused-A's output columns [2048, 2624) -- the raw kv_a_proj row, not
        normalized; `kv_norm_weight` / `kv_norm_eps` the kv_a_layernorm weight and epsilon.
      - `rotary_cos_sin`: the MLA rope table (`RopeParams(..., duplicate_data=True)`, GPT-J
        pairs), covering every position the cache can hold (`max_seq_len`).
      - `sequence_length`, `num_contexts`, `kv_cache_block_offsets`: the attention metadata's
        `kv_lens_cuda_runtime`, `num_contexts` and `kv_cache_block_offsets`;
        `host_kv_cache_pool_pointers` / `host_kv_cache_pool_mapping` the cache manager's
        `kv_cache_pool_pointers` / `kv_cache_pool_mapping`; `layer_idx` the local layer;
        `attention_window_size` the metadata's `max_seq_len`.
      - `mla_bmm1_scale` ([2] fp32) and `mla_bmm2_scale` ([1] fp32): buffers allocated once,
        read by `glm_sparse_mla_flash_decode`.

    The cache carries no scaling factor: GLM-5.2's FP8 cache is written and read at 1.0, as
    the PyTorch MLA path writes it on the context server. The pool must hold plain 576-byte
    E4M3 rows (`kv_cache_dtype: fp8`); the inline-scale `fp8_ds_mla` layout has 656-byte rows,
    which the paged-cache arguments do not reveal, so the target asserts the KV dtype at
    construction.

    PDL: the kernel waits on its grid dependency before any global read. With
    `trigger_at_entry` its dependents may launch at its entry (the flash decode that gathers
    earlier steps' rows before its wait), otherwise after its stores; every dependent must
    wait on its own grid dependency before reading the appended rows or the scales.

    Not certified: a mixed batch (`num_contexts` > 0), more than two sequences, a ragged batch,
    pages other than 64 tokens. Rewriting a prepared step rewrites the same slots.
    """

    def __call__(
        self,
        latent: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        kv_norm_eps: float,
        rotary_cos_sin: torch.Tensor,
        sequence_length: torch.Tensor,
        num_contexts: int,
        seq_len_q: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        layer_idx: int,
        tokens_per_block: int,
        attention_window_size: int,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        softmax_scale: float,
        trigger_at_entry: bool = False,
    ) -> None:
        torch.ops.trtllm.glm_mla_kv_half_append(
            latent,
            kv_norm_weight,
            kv_norm_eps,
            rotary_cos_sin,
            sequence_length,
            num_contexts,
            seq_len_q,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            layer_idx,
            tokens_per_block,
            attention_window_size,
            mla_bmm1_scale,
            mla_bmm2_scale,
            softmax_scale,
            trigger_at_entry,
        )

    def reference(
        self,
        latent: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        kv_norm_eps: float,
        rotary_cos_sin: torch.Tensor,
        sequence_length: torch.Tensor,
        num_contexts: int,
        seq_len_q: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        layer_idx: int,
        tokens_per_block: int,
        attention_window_size: int,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        softmax_scale: float,
        trigger_at_entry: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(rows, positions, bmm1, bmm2)``: what the op is supposed to write, and where.

        ``rows`` [T, 576] E4M3 is the cache row of each latent row, ``positions`` [T] int64 its
        position in its sequence. The paged-cache arguments (block offsets, pool pointers and
        mapping, layer, page size, window) only say where a row lands, and the test reads it
        back there; ``mla_bmm1_scale`` / ``mla_bmm2_scale`` are where the scales land and
        ``trigger_at_entry`` when the dependents launch, so all of them are ignored here.

        fp32 throughout, as the kernel: the RMSNorm multiplies ``x * (rstd * w)``, the RoPE
        pairs consecutive columns (GPT-J), and each value is saturated and rounded to E4M3
        once.
        """
        num_tokens = latent.shape[0]
        device = latent.device
        token = torch.arange(num_tokens, device=device)
        seq = token // seq_len_q
        positions = sequence_length[num_contexts + seq].long() - seq_len_q + token % seq_len_q

        x = latent[:, :KV_LORA_RANK].float()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + kv_norm_eps)
        nope = x * (rstd * kv_norm_weight.float())

        pairs = (
            latent[:, KV_LORA_RANK : KV_LORA_RANK + ROPE_DIM]
            .float()
            .reshape(num_tokens, ROPE_DIM // 2, 2)
        )
        table = rotary_cos_sin.reshape(-1, ROPE_DIM, 2)[positions.clamp(min=0), : ROPE_DIM // 2]
        cos, sin = table[..., 0], table[..., 1]
        roped = torch.stack(
            (cos * pairs[..., 0] - sin * pairs[..., 1], cos * pairs[..., 1] + sin * pairs[..., 0]),
            dim=-1,
        ).reshape(num_tokens, ROPE_DIM)

        rows = torch.cat((nope, roped), dim=-1).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        scale = torch.tensor(softmax_scale, dtype=torch.float32, device=device)
        log2e = torch.tensor(_LOG2E, dtype=torch.float32, device=device)
        bmm1 = torch.stack((scale, scale * log2e))
        bmm2 = torch.ones(1, dtype=torch.float32, device=device)
        return rows, positions, bmm1, bmm2

    def is_valid(
        self,
        latent: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        kv_norm_eps: float,
        rotary_cos_sin: torch.Tensor,
        sequence_length: torch.Tensor,
        num_contexts: int,
        seq_len_q: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        layer_idx: int,
        tokens_per_block: int,
        attention_window_size: int,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        softmax_scale: float,
        trigger_at_entry: bool = False,
    ) -> None:
        """The lengths the op derives positions from without checking.

        Only ``latent``'s row count, ``sequence_length``, ``num_contexts``, ``seq_len_q``,
        ``rotary_cos_sin``'s size, ``kv_cache_block_offsets``' width and ``tokens_per_block``
        are read.
        """
        num_generations = latent.shape[0] // seq_len_q
        if num_generations == 0:
            return
        lens = sequence_length[num_contexts : num_contexts + num_generations].long()
        # A sequence shorter than its step leaves rows at negative positions, which the kernel
        # skips: those tokens never reach the cache.
        assert bool((lens >= seq_len_q).all()), (
            "sequence_length must count this step's seq_len_q tokens; a shorter sequence "
            "leaves its rows unwritten"
        )
        last_position = int(lens.max()) - 1
        # The table is read at the position without a bound.
        num_positions = rotary_cos_sin.numel() // (2 * ROPE_DIM)
        assert last_position < num_positions, (
            f"rotary_cos_sin covers {num_positions} positions; position {last_position} "
            "reads past it"
        )
        # A block past the offsets' width reads another sequence's offsets and writes its page.
        width = kv_cache_block_offsets.shape[-1]
        assert last_position // tokens_per_block < width, (
            f"kv_cache_block_offsets has {width} blocks per sequence; position {last_position} "
            "needs more"
        )


mla_kv_half_append = GlmMlaKvHalfAppend()
