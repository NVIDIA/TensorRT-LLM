# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoPE the new context tokens and append their latent rows to the paged MLA cache."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128

#: Half an e4m3 ulp. e4m3 carries 3 explicit mantissa bits, so a
#: round-to-nearest lands within 2**-3 relative.
E4M3_ULP_RTOL = 2.0**-3

#: How much of the appended output may miss bit-exactness before the gate calls
#: it drift rather than rounding. See `compare` for why this is not zero and why
#: it has never been approached.
E4M3_MAX_INEXACT_FRACTION = 1e-3


class _MlaRopeAppendPagedKvAssignQ(OpWrapper):
    """RoPE each new context token's `q_pe` (in `q`) and `k_pe` (in
    `latent_cache`) in place at its absolute position, then append the token's
    latent row `[compressed_kv | rope(k_pe)]` to the paged MLA KV cache.

    Two buffers are written and one is the paged pool itself, which is why the
    cells below check what landed in the cache rather than a return value: the
    op returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1's context call is the only caller. Its latent geometry is
    # fixed by the checkpoint -- 512-wide compressed_kv, 64-wide k_pe, 128
    # heads -- so the cells vary what a step actually varies: how much of each
    # sequence was already cached, how many tokens are new, and the page size.
    _R1 = dict(
        head_num=128,
        nope_size=128,
        rope_size=64,
        lora_size=512,
        residual_dim=0,
        quant_mode=QM_FP8_KV_CACHE,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="production context call: fp8 pool, page 32, mixed cached lengths",
            spec=dict(cached=[0, 32, 100], new=[37, 64, 300], tokens_per_block=32, **_R1),
        ),
        Cell(
            why="page 64, the size a tuned MLA target opts into",
            spec=dict(cached=[0, 32, 100], new=[37, 64, 300], tokens_per_block=64, **_R1),
        ),
        Cell(
            why="a single new token on top of a long cached prefix -- the decode-adjacent shape",
            spec=dict(cached=[511], new=[1], tokens_per_block=64, **_R1),
        ),
    )

    note = """
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
    ) -> None:
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
        """
        row = latent_row.float()
        lora = row[..., : row.shape[-1] - rope_size]
        pe = row[..., row.shape[-1] - rope_size :]
        pairs = pe.reshape(*pe.shape[:-1], rope_size // 2, 2)
        roped = torch.empty_like(pairs)
        roped[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        roped[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        out = torch.cat([lora, roped.reshape(pe.shape)], dim=-1)
        if kv_scale_orig_quant is not None:
            out = out * kv_scale_orig_quant[0]
        return out.to(store_dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Half an e4m3 ulp, *and* a floor on how much may miss bit-exactness.

        The rotation is where kernel and reference can disagree: both evaluate
        `x*cos -+ y*sin`, and they may do it in different orders. e4m3's coarse
        rounding absorbs that completely in practice -- every appended byte
        measured on this architecture, across every scale and every cell, came
        back bit-exact, so neither half of this gate has ever been approached.

        Both halves are kept anyway, and both are measured rather than assumed:
        the test runs four deliberately wrong mirrors of the same bytes and
        asserts each blows the half it is able to blow. A tolerance alone would
        pass a mirror that is uniformly a little wrong; a bit-exact demand alone
        would fail the first time a reordering did survive the rounding.
        """
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out.float(), ref.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
        inexact = int(
            (out.reshape(-1).view(torch.uint8) != ref.reshape(-1).view(torch.uint8)).sum()
        )
        allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * out.numel()))
        assert inexact <= allowed, (
            f"not bit-exact enough: {inexact}/{out.numel()} bytes differ (allowed {allowed})"
        )

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


mla_rope_append_paged_kv_assign_q = _MlaRopeAppendPagedKvAssignQ()
