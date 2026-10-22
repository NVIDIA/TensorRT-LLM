# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context step's MLA rope, latent-KV append and fp8 query, in one CuTe DSL launch."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_prefill_mla_rope_append_fp8q_h16_kv512_rope64_bf16_fp8 as _kernel,
)

NUM_HEADS = _kernel.NUM_HEADS
KV_LORA_RANK = _kernel.KV_LORA_RANK
ROPE_DIM = _kernel.ROPE_DIM
#: One fused-query head and one latent row: `kv_lora + rope` = 576.
LATENT_DIM = _kernel.HEAD_SIZE
#: The projected query head `q_pe` is a view of: `qk_nope + rope` = 256.
QK_HEAD_DIM = _kernel.QK_HEAD_DIM
#: Context requests one launch takes.
MAX_REQUESTS = _kernel.MAX_REQUESTS
#: e4m3's largest finite value; the kernel's conversion saturates to it.
_E4M3_MAX = 448.0


@dataclass(frozen=True)
class CompiledMlaRopeAppendPrefill:
    """The kernel `compile` built: the page size and device it is specialised for."""

    device_index: int
    tokens_per_block: int
    kernel: object


def compile(  # noqa: A001 -- the kernel module's own name for this step
    *, tokens_per_block: int, device_index: int
) -> CompiledMlaRopeAppendPrefill:
    """Compile the kernel for `tokens_per_block`; return the handle the op takes as
    `compiled`. Call once at load.

    The specialization is this target's layout: `q_pe` the rope tail of the
    `[T, 16, 256]` q_b output, a packed `[T, 576]` latent. Token and request
    counts are symbolic, so one compile serves every context step. Costs one CuTe
    DSL compile and one launch on a one-token, one-block case.
    """
    tpb, dev = int(tokens_per_block), torch.device("cuda", int(device_index))
    kernel = _kernel.compile(tokens_per_block=tpb, device_index=int(device_index))
    # The first call of a compiled function binds and loads its module: done here,
    # not on the first served step.
    bf16 = torch.bfloat16
    q = torch.zeros([1, NUM_HEADS, QK_HEAD_DIM], dtype=bf16, device=dev)
    runner = _kernel.prepare_runner(
        torch.zeros([1, NUM_HEADS, LATENT_DIM], dtype=bf16, device=dev),
        q[..., QK_HEAD_DIM - ROPE_DIM :],
        torch.zeros([1, LATENT_DIM], dtype=bf16, device=dev),
        torch.empty([1, NUM_HEADS, LATENT_DIM], dtype=torch.uint8, device=dev),
        torch.zeros([tpb * LATENT_DIM], dtype=torch.uint8, device=dev),
        torch.zeros([1, 2, 1], dtype=torch.int32, device=dev),
        torch.tensor([0, 1], dtype=torch.int32, device=dev),
        torch.zeros([1], dtype=torch.int32, device=dev),
        torch.zeros([1, ROPE_DIM, 2], dtype=torch.float32, device=dev),
        tokens_per_block=tpb,
    )
    dataclasses.replace(runner, compiled=kernel)()
    torch.cuda.synchronize(dev)
    return CompiledMlaRopeAppendPrefill(int(device_index), tpb, kernel)


def _positions(cu_seqlens: torch.Tensor, cached_lens: torch.Tensor, tokens: int):
    """`(request, position)` of every token, int64, as the kernel locates them."""
    b = int(cached_lens.shape[0])
    cu = cu_seqlens[: b + 1].to(torch.int64)
    tok = torch.arange(tokens, device=cu.device)
    req = torch.searchsorted(cu[1:b], tok, right=True)
    return req, tok - cu[req] + cached_lens.to(torch.int64)[req]


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J interleaved rotation of the bf16 pairs in `x`'s last dim, as the kernel
    rounds it: one product of each output in fp32, fused into the other (an fp32
    FMA, formed exactly in fp64 and rounded once), then bf16 round-to-nearest.
    `cos` / `sin` broadcast against `x[..., 0::2]`."""
    xe, xo = x[..., 0::2].float(), x[..., 1::2].float()
    even = (cos.double() * xe.double() - (sin * xo).double()).float()
    odd = (sin.double() * xe.double() + (cos * xo).double()).float()
    return torch.stack([even, odd], -1).reshape(x.shape).to(torch.bfloat16)


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to e4m3 the way the kernel converts: nearest, saturating."""
    return x.float().clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


class CuteDslMlaRopeAppendPrefill(OpWrapper):
    """RoPE the context rows, append their latent rows to the fp8 paged pool, and
    write the fp8 query, in one launch.

    `fused_q` is `[T, 16, 576]` bf16: the absorbed `q_nope` already in its first
    512 columns of each head; the last 64 are written with `rope(q_pe)`. `q_pe`
    is `[T, 16, 64]` bf16, the rope tail of the `[T, 16, 256]` projected query
    (a strided view, read only). `latent_cache` is `[T, 576]` bf16
    `[compressed_kv | k_pe]`, packed; its `k_pe` is rotated in place. The latent
    row `[compressed_kv | rope(k_pe)]` is stored as e4m3 into the pool, and
    `quant_q_buffer` (`[T, 16, 576]` e4m3 or uint8) receives `e4m3(fused_q)`,
    unscaled, for all 576 columns.

    Token `t` of request `b` sits at position `t - cu_seqlens[b] + cached_lens[b]`
    (int32 `[B + 1]` and `[B]`, the new-row prefix sum and the cached-prefix
    lengths). Its pool bytes start at
    `(layer_in_pool + kv_cache_block_offsets[b, 0, pos // tpb]) * tpb * 576 +
    (pos % tpb) * 576` of `kv_pool` (bytes), the KV block array's addressing:
    `kv_cache_block_offsets` is the int32 `[B, 2, max_blocks]` slice of the
    engine's offsets for the context requests. `rotary_cos_sin` is the fp32
    `[positions, 64, 2]` table (any contiguous shape), pair `p` at `[pos, p]`.

    Rotation is GPT-J interleaved, rounded to bf16, and the e4m3 values are
    conversions of those bf16 values, as the stock rope/append op and cast do.
    Returns `(fused_q, latent_cache, quant_q_buffer, kv_pool)`, all written in
    place.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target's context call: 16 heads, page 32, fp8 pool. The cells vary
    # what a step varies: one request or several (the kernel locates each
    # token's request by scanning the prefix sum), a cached prefix (positions
    # and pool rows start past 0), a step whose row count is not a multiple of
    # the kernel's 32-token tile, and a long chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request of 77 rows, no cache: a partial last 32-token tile",
            spec=dict(new=[77], cached=[0], layers=2, layer=1),
        ),
        Cell(
            why="three requests, two with cached prefixes that end mid-block, so a "
            "request's first new row lands mid-page",
            spec=dict(new=[37, 64, 300], cached=[0, 32, 100], layers=3, layer=2),
        ),
        Cell(
            why="a 4100-row chunk past index_topk, layer 0 of a one-layer pool",
            spec=dict(new=[4100], cached=[0], layers=1, layer=0),
        ),
        Cell(
            why="one new row on a 511-token prefix next to a fresh 9-row request",
            spec=dict(new=[1, 9], cached=[511, 0], layers=2, layer=0),
        ),
    )

    note = """
    The target compiles one handle at load for its page size and takes this
    op on the trunk's context rows when the step has at most MAX_REQUESTS (64)
    context requests and the layer's pool base is the primary pool with no
    secondary pool; otherwise it runs the stock `mla_rope_append_paged_kv_assign_q`
    plus a cast. The MTP layer's context rows take the stock op.

    The kernel validates every call on the host (dtypes, shapes, alignment,
    devices, request count). It keys its compile on the `q_pe` and latent
    strides, so this op refuses any layout but the compiled one rather than
    compile inside a call. What it does not check is in `is_valid`: a prefix
    sum that does not end at T, a position past the rope table, and a
    position whose block is past the offsets' width or unallocated (the kernel
    skips that row's append silently).
    """

    def __call__(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        kv_pool: torch.Tensor,
        layer_in_pool: int,
        kv_cache_block_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cached_lens: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        tokens_per_block: int,
        compiled: CompiledMlaRopeAppendPrefill,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if fused_q.device.index != compiled.device_index:
            raise ValueError(
                f"the kernel was compiled for device {compiled.device_index}, not {fused_q.device}"
            )
        if int(tokens_per_block) != compiled.tokens_per_block:
            raise ValueError(
                f"the kernel was compiled for {compiled.tokens_per_block} tokens per block, "
                f"not {tokens_per_block}"
            )
        # Another layout selects a specialization nothing compiled at load.
        if tuple(q_pe.stride()[:2]) != (NUM_HEADS * QK_HEAD_DIM, QK_HEAD_DIM) or (
            latent_cache.stride(0) != LATENT_DIM
        ):
            raise ValueError(
                "q_pe must be the rope tail of a contiguous [T, 16, 256] query and "
                "latent_cache a packed [T, 576]"
            )
        base = kv_pool.data_ptr() + int(layer_in_pool) * int(tokens_per_block) * LATENT_DIM
        runner = _kernel.prepare_runner(
            fused_q,
            q_pe,
            latent_cache,
            quant_q_buffer.view(torch.uint8),
            base,
            kv_cache_block_offsets,
            cu_seqlens,
            cached_lens,
            rotary_cos_sin,
            tokens_per_block=int(tokens_per_block),
        )
        dataclasses.replace(runner, compiled=compiled.kernel)()
        return fused_q, latent_cache, quant_q_buffer, kv_pool

    def reference(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        kv_pool: torch.Tensor,
        layer_in_pool: int,
        kv_cache_block_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cached_lens: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        tokens_per_block: int,
        compiled: CompiledMlaRopeAppendPrefill,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(fused_q, latent_cache, quant_q_buffer, kv_pool)` as the op leaves them,
        as new tensors; nothing passed in is written. `quant_q_buffer` is returned
        as uint8 bytes and `kv_pool` as a flat uint8 byte copy. `compiled` is
        ignored.

        Bit-exact by construction: the rotation reproduces the kernel's fp32
        rounding (see `_rope`), and every other value is a copy or a single
        rounding.
        """
        t, tpb = int(fused_q.shape[0]), int(tokens_per_block)
        req, pos = _positions(cu_seqlens, cached_lens, t)
        table = rotary_cos_sin.reshape(-1, ROPE_DIM, 2).float()[pos, : ROPE_DIM // 2]
        cos, sin = table[..., 0], table[..., 1]
        q_out = fused_q.clone()
        q_out[..., KV_LORA_RANK:] = _rope(q_pe, cos[:, None, :], sin[:, None, :])
        lat_out = latent_cache.clone()
        lat_out[:, KV_LORA_RANK:] = _rope(latent_cache[:, KV_LORA_RANK:], cos, sin)
        pool = kv_pool.reshape(-1).view(torch.uint8).clone()
        block = kv_cache_block_offsets[req, 0, pos // tpb].to(torch.int64)
        start = (int(layer_in_pool) + block) * tpb * LATENT_DIM + (pos % tpb) * LATENT_DIM
        cols = torch.arange(LATENT_DIM, device=pool.device)
        pool[start[:, None] + cols[None, :]] = _e4m3(lat_out)
        return q_out, lat_out, _e4m3(q_out), pool

    def is_valid(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        kv_pool: torch.Tensor,
        layer_in_pool: int,
        kv_cache_block_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cached_lens: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        tokens_per_block: int,
        compiled: CompiledMlaRopeAppendPrefill,
    ) -> None:
        """Reads the prefix sum and offsets back to the host: never on the serving path."""
        t, tpb = int(fused_q.shape[0]), int(tokens_per_block)
        b = int(cached_lens.shape[0])
        assert quant_q_buffer.dtype in (torch.float8_e4m3fn, torch.uint8), (
            f"quant_q_buffer must be e4m3 or its uint8 bytes, got {quant_q_buffer.dtype}"
        )
        # The kernel finds a token's request by scanning the boundaries: a token
        # past `cu[B]` lands in the last request at a position past its rows.
        cu = cu_seqlens[: b + 1].cpu().long()
        assert int(cu[0]) == 0 and int(cu[-1]) == t and bool((cu[1:] >= cu[:-1]).all()), (
            f"cu_seqlens must rise from 0 to the {t} rows, got {cu.tolist()}"
        )
        req, pos = (v.cpu() for v in _positions(cu_seqlens, cached_lens, t))
        positions = rotary_cos_sin.numel() // (ROPE_DIM * 2)
        assert int(pos.max()) < positions, (
            f"a position past the rope table's {positions} rows; the kernel reads past it"
        )
        # A block past the offsets' width is skipped silently; a -1 block would
        # write before the layer's base.
        offsets = kv_cache_block_offsets.cpu().long()
        blk = pos // tpb
        assert int(blk.max()) < offsets.shape[2], "a token's block is past the offsets' width"
        block = offsets[req, 0, blk]
        assert bool((block >= 0).all()), "a token's block is unallocated (-1)"
        end = (int(layer_in_pool) + int(block.max()) + 1) * tpb * LATENT_DIM
        nbytes = kv_pool.numel() * kv_pool.element_size()
        assert end <= nbytes, f"an append past the pool's {nbytes} bytes"


cute_dsl_mla_rope_append_prefill = CuteDslMlaRopeAppendPrefill()
