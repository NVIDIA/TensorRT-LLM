# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoPE the new context tokens and append their latent rows to the paged MLA cache."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128


class MlaRopeAppendPagedKvAssignQ(OpWrapper):
    """RoPE each new context token's `q_pe` (in `q`) and `k_pe` (in
    `latent_cache`) in place at its absolute position, then append the token's
    latent row `[compressed_kv | rope(k_pe)]` to the paged MLA KV cache.

    Two buffers are written and one is the paged pool itself, which is why the
    cells below check what landed in the cache rather than a return value: the
    op returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Hy4-Flash's MLA geometry: 512-wide compressed_kv, 64-wide k_pe, and a
    # 192-wide qk_nope ahead of each head's rope tail. The sparse layers run 32
    # heads, the sliding-window layers 64, and `attn_tp` cuts them: 8/16 at
    # tp4, 16/32 at the disaggregated context engine's tp2 (`dims.py`'s `tep2`),
    # so the cells vary head count over 64/32/16/8 and hold the latent geometry
    # fixed. The core KV cache is bf16 (quant_mode 0); the kv_fp8 variant's
    # context call runs the e4m3 pool (128).
    _HY4 = dict(
        nope_size=192,
        rope_size=64,
        lora_size=512,
        residual_dim=0,
        tokens_per_block=32,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="swa context call: 64 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=64, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="swa context call: a 5120-token prefill chunk over a bf16 pool",
            spec=dict(head_num=64, cached=[0], new=[5120], quant_mode=0, **_HY4),
        ),
        Cell(
            why="sparse context call: 32 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=32, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="fp8 pool: two sequences, the second starting on a 512-token cached prefix",
            spec=dict(
                head_num=64,
                cached=[0, 512],
                new=[512, 512],
                quant_mode=QM_FP8_KV_CACHE,
                **_HY4,
            ),
        ),
        Cell(
            why="swa at tp4: 16 heads per rank, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=16, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="sparse at tp4: 8 heads per rank, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=8, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="swa at the ctx engine's tp2: 32 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=32, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="sparse at the ctx engine's tp2: 16 heads, a 512-token prefill chunk over a bf16 pool",
            spec=dict(head_num=16, cached=[0], new=[512], quant_mode=0, **_HY4),
        ),
        Cell(
            why="the ctx engine's prefill chunk bound at tp2: 32 heads, 5120 new tokens, bf16 pool",
            spec=dict(head_num=32, cached=[0], new=[5120], quant_mode=0, **_HY4),
        ),
        Cell(
            why="envelope's largest row: 8192 new tokens in one context call, 64 heads, bf16",
            spec=dict(head_num=64, cached=[0], new=[8192], quant_mode=0, **_HY4),
        ),
        Cell(
            why="envelope's largest row at the ctx engine's tp2: 32 heads, 8192 new tokens, bf16",
            spec=dict(head_num=32, cached=[0], new=[8192], quant_mode=0, **_HY4),
        ),
    )

    note = """
    Hy4-Flash's context prologue at sm_103. `q` is the unabsorbed MLA query --
    each head is `[nope 192 | rope 64]` wide -- and the latent row is
    `[compressed_kv 512 | k_pe 64]`, page 32. Head counts are the engine's:
    the sparse layers run 32 (8 per rank at tp4), the sliding-window layers 64
    (16 at tp4); the disaggregated deployment's context engine runs attn_tp 2
    (16 sparse / 32 SWA per rank, `dims.py`'s `tep2`), which the four `at the
    ctx engine's tp2` cells hold.

    The RoPE table is the duplicated `(cos, sin)` float2 layout the MLA backend
    builds (`RopeParams(..., duplicate_data=True)`): 64 pairs per position, of
    which the kernel reads the first 32. Hy4's rope is interleaved mRoPE over
    three position planes -- sections [12, 10, 10] interleaved at stride 3 --
    at base 1e7 on the sparse layers and 1e4 on the sliding-window ones; with
    text-only positions the three planes coincide and the table is the plain
    duplicated one, which is what every cell above uses. The test also drives a
    table whose planes differ and checks the kernel's rotation against the HF
    `apply_rotary_pos_emb`, so the interleaved layout is certified, not assumed.

    `quant_mode=8192` (`NVFP4_KV_CACHE`) is **not** certified here: this
    target's KV cache is bf16 or e4m3, and the op's NVFP4 branch
    (`quantizeAndWriteMlaFp4`) needs an NVFP4 MLA pool with a block-scale pool
    that no engine path in this run builds. The row that asks for it is
    certified at 128 instead, the value `TrtllmAttention` passes for the kv_fp8
    variant. The optional 22nd argument (`kv_cache_local_slots`, the Helix
    local-slot remap) is also not driven: this entry omits it, as the shared
    catalog entry does.

    Two more values are outside the cells. `layer_idx` is the engine's local
    layer index (0..41) and picks the layer's slab inside the pool; every case
    here drives a one-layer pool at 0, so a multi-layer pool's per-layer offset
    is not certified. The input dtype is bf16 only -- the engine's activations
    -- so the op's fp16 and fp32 paths are not driven.

    The ctx engine's own arguments size two cells: `max_seq_len` 5120 is the
    largest single-sequence prefill one context call carries there (the 5120-row
    cell) and `max_num_tokens` 8192 the largest row count one call carries (the
    8192-row cells, the pool's default bound as well). `attention_window_size` is
    driven at the pool's max sequence length in every cell, while the target hands
    a windowed layer its own window -- 512 on the sliding-window layers
    (`dims.py`'s `LayerGeom.window`); the window sizes the block array's window
    fields and the reader's cyclic remap, and the sibling generation prologue
    measures the append to be window-independent
    (`test_window_does_not_reach_the_append`), but this op's windowed value is not
    driven here: not certified.

    No `is_valid` beyond `residual_dim`. What else could go wrong here is either
    rejected by the op or is a property of what an earlier call wrote -- a write
    scale that disagrees with the read scale appends plausible bytes, and
    nothing visible at this call site can tell. The test drives that directly.
    """

    def __call__(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        num_contexts: int,
        cu_ctx_cached_kv_lens: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_input_uncached_seq_len: int,
        cos_sin_cache: torch.Tensor,
        head_num: int,
        nope_size: int,
        rope_size: int,
        lora_size: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        residual_dim: int,
        layer_idx: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_contexts,
            cu_ctx_cached_kv_lens,
            cu_seq_lens,
            max_input_uncached_seq_len,
            cos_sin_cache,
            head_num,
            nope_size,
            rope_size,
            lora_size,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            residual_dim,
            layer_idx,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
        )
        return q

    def reference(
        self,
        latent_row: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_size: int,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        store_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        """The latent row this op is supposed to append, fp32 then stored once.

        The same documented exception to mirroring as the other paged-cache
        entries: this op's effect is on the pool, and its arguments only say
        where. `latent_row` is `[compressed_kv | k_pe]` before the rotation, and
        `cos` / `sin` are the halves of the rope table already selected for each
        row's absolute position; the test reads both out of the environment,
        which is the only place they exist.

        The pairing is **GPT-J interleaved** -- consecutive elements form a pair
        -- not the split-half neox form. The two are indistinguishable by shape
        and give different answers, which is what makes stating it here worth
        more than stating it in prose.

        `compressed_kv` passes through untouched. The scale is applied once,
        before the store, so the stored bytes are a function of the row rather
        than of the order the kernel visited it in.

        The rotated pair is rounded to the input dtype before the scale and the
        store: the kernel rotates into a register vector of its input type
        (`rotary_embedding_transform` returns `__nv_bfloat162`) and quantizes
        from there. Converting the fp32 rotation straight to e4m3 rounds once
        where the kernel rounds twice.
        """
        row = latent_row.float()
        lora = row[..., : row.shape[-1] - rope_size]
        pe = row[..., row.shape[-1] - rope_size :]
        pairs = pe.reshape(*pe.shape[:-1], rope_size // 2, 2)
        roped = torch.empty_like(pairs)
        roped[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        roped[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        roped = roped.to(latent_row.dtype).float()
        out = torch.cat([lora, roped.reshape(pe.shape)], dim=-1)
        if kv_scale_orig_quant is not None:
            out = out * kv_scale_orig_quant[0]
        return out.to(store_dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        num_contexts: int,
        cu_ctx_cached_kv_lens: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_input_uncached_seq_len: int,
        cos_sin_cache: torch.Tensor,
        head_num: int,
        nope_size: int,
        rope_size: int,
        lora_size: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        residual_dim: int,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        # `residual_dim` must be 0 or `rope_size`, and the op rejects non-zero
        # unless the KV pool is FP4. Every caller here runs a bf16 or fp8-e4m3
        # pool, so 0 is the only legal value and a non-zero one would be read
        # against a pool layout that does not exist.
        assert residual_dim == 0, (
            f"residual_dim must be 0 for a bf16 or fp8 KV pool; got {residual_dim}. "
            "A non-zero value is only legal against an FP4 pool"
        )


mla_rope_append_paged_kv_assign_q = MlaRopeAppendPagedKvAssignQ()
