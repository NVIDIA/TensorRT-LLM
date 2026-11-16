# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context step's MLA rope, latent-KV append and fp8 query, in one CuTe DSL launch."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_prefill_mla_rope_append_fp8q_h16_kv512_rope64_bf16_fp8 as _kernel,
)
from ...kernels.common.cuda_stream import DIET as _HOST_DIET

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
#: The latent row strides `compile` builds: a packed `[T, 576]` latent, and the latent
#: tail of the `[T, 2048 + 576]` `q_a | kv_a | k_pe` projection row, where the in-place
#: kv_a norm leaves it. The kernel reads either through its row stride, so the second
#: needs no packing copy.
Q_LORA_RANK = 2048
LATENT_STRIDES = (LATENT_DIM, Q_LORA_RANK + LATENT_DIM)


@dataclass(frozen=True)
class CompiledMlaRopeAppendPrefill:
    """The kernels `compile` built: the page size and device they are specialised for,
    one kernel per latent row stride in `LATENT_STRIDES`, and the same per stride with
    `skip_q_base` (only the rope and latent-append groups launch), with `q_in_place`
    (rope(q_pe) back into `q_pe`, no fp8 query) and with `q_in_place` on an e4m3 `q_pe`
    (`q_e4m3`, the fp8 context attention's query)."""

    device_index: int
    tokens_per_block: int
    kernels: Mapping[int, object]
    kernels_skip_q_base: Mapping[int, object] = dataclasses.field(default_factory=dict)
    kernels_q_in_place: Mapping[int, object] = dataclasses.field(default_factory=dict)
    kernels_q_e4m3: Mapping[int, object] = dataclasses.field(default_factory=dict)
    #: CTAs the `q_e4m3` kernels split a token tile's RoPE work over (0: one CTA).
    rope_par: int = 0


#: The `q_e4m3` mode's build: a token tile's RoPE work over 2 CTAs, loads issued before the
#: request search, packed rounding, vector c_kv copy loads. Same output bytes as one CTA;
#: -1.0 .. -1.9 us per launch at 7-8k rows on B300.
ROPE_PAR = 2


def compile(  # noqa: A001 -- the kernel module's own name for this step
    *, tokens_per_block: int, device_index: int, rope_par: int = ROPE_PAR
) -> CompiledMlaRopeAppendPrefill:
    """Compile the kernel for `tokens_per_block`; return the handle the op takes as
    `compiled`. Call once at load.

    The specialization is this target's layout: `q_pe` the rope tail of the
    `[T, 16, 256]` q_b output, and a latent of either stride in `LATENT_STRIDES`.
    The kernel keys its compile on the latent stride and on the mode (`skip_q_base`,
    `q_in_place`, `q_e4m3`), so each pair is a separate kernel. The `q_in_place` modes
    run all of a token tile's RoPE groups in one CTA (`merge_rope`). Token and request
    counts are symbolic, so one compile per pair serves every context step. Costs one
    CuTe DSL compile and one launch on a one-token, one-block case per pair. `rope_par`
    (2 or 4; 0 for one CTA) builds the `q_e4m3` mode's kernels with the tile split.
    """
    tpb, dev = int(tokens_per_block), torch.device("cuda", int(device_index))
    bf16 = torch.bfloat16
    tables: dict[str, dict] = {"plain": {}, "skip": {}, "in_place": {}, "e4m3": {}}
    for stride, mode in ((s, m) for s in LATENT_STRIDES for m in tables):
        skip, e4m3 = mode == "skip", mode == "e4m3"
        in_place = mode in ("in_place", "e4m3")
        par = int(rope_par) if e4m3 else 0
        q = torch.zeros(
            [1, NUM_HEADS, QK_HEAD_DIM], dtype=torch.float8_e4m3fn if e4m3 else bf16, device=dev
        )
        kernel = _kernel._compile_for(
            _kernel.PrefillRopeSpecialization(
                tokens_per_block=tpb,
                latent_cache_stride=stride,
                skip_q_base=skip,
                q_in_place=in_place,
                q_e4m3=e4m3,
                merge_rope=in_place,
                rope_par=par,
            ),
            int(device_index),
        )
        # The first call of a compiled function binds and loads its module: done here,
        # not on the first served step.
        row = torch.zeros([1, stride], dtype=bf16, device=dev)
        runner = _kernel.prepare_runner(
            None if in_place else torch.zeros([1, NUM_HEADS, LATENT_DIM], dtype=bf16, device=dev),
            q[..., QK_HEAD_DIM - ROPE_DIM :],
            row[:, stride - LATENT_DIM :],
            None
            if in_place
            else torch.empty([1, NUM_HEADS, LATENT_DIM], dtype=torch.uint8, device=dev),
            torch.zeros([tpb * LATENT_DIM], dtype=torch.uint8, device=dev),
            torch.zeros([1, 2, 1], dtype=torch.int32, device=dev),
            torch.tensor([0, 1], dtype=torch.int32, device=dev),
            torch.zeros([1], dtype=torch.int32, device=dev),
            torch.zeros([1, ROPE_DIM, 2], dtype=torch.float32, device=dev),
            tokens_per_block=tpb,
            skip_q_base=skip,
            q_in_place=in_place,
            q_e4m3=e4m3,
            merge_rope=in_place,
            rope_par=par,
        )
        dataclasses.replace(runner, compiled=kernel)()
        tables[mode][stride] = kernel
    torch.cuda.synchronize(dev)
    return CompiledMlaRopeAppendPrefill(
        int(device_index),
        tpb,
        tables["plain"],
        tables["skip"],
        tables["in_place"],
        tables["e4m3"],
        int(rope_par),
    )


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

    `skip_q_base=True`: the caller has already written `quant_q_buffer[..., :512]`
    (the fp8 q-base K-B writes those bytes itself); only the rope columns
    `[..., 512:]` of `quant_q_buffer` are written and `fused_q[..., :512]` is not
    read. The kernel launches only its rope and latent-append groups.

    `q_in_place=True` (the non-absorbed context attention): `fused_q` and
    `quant_q_buffer` are None, `rope(q_pe)` is written back into `q_pe` -- the q_b
    output is then that attention's query -- and no fp8 query is produced. The
    latent rotation and the pool append are the same code as the absorbed mode, so
    the decode cache bytes do not depend on the mode. Returns `(q_pe, latent_cache,
    None, kv_pool)`: the first value is the rotated query in both modes.

    `q_e4m3=True` (with `q_in_place`, the fp8 context attention): `q_pe` is the e4m3
    rope tail of the e4m3 `[T, 16, 256]` q_b output (its power-of-two scale already
    applied; rope is linear, so the scale needs no handling). The kernel widens the
    values exactly, rotates them with the same bf16-rounded arithmetic and writes them
    back as e4m3 (nearest, saturating). The `q_in_place` modes run all of a token
    tile's RoPE groups in one CTA; outputs are the same bytes as separate groups.
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
        Cell(
            why="the latent read in place from the projection row (stride 2624), "
            "as the target's context step passes it",
            spec=dict(
                new=[37, 64, 300], cached=[0, 32, 100], layers=3, layer=2, latent_stride=2624
            ),
        ),
        Cell(
            why="the in-place latent on a 4100-row chunk",
            spec=dict(new=[4100], cached=[0], layers=1, layer=0, latent_stride=2624),
        ),
        Cell(
            why="skip_q_base after the fp8 K-B: three requests with cached prefixes, "
            "the in-place latent",
            spec=dict(
                new=[37, 64, 300],
                cached=[0, 32, 100],
                layers=3,
                layer=2,
                latent_stride=2624,
                skip_q_base=True,
            ),
        ),
        Cell(
            why="skip_q_base on a 4100-row chunk, the in-place latent",
            spec=dict(
                new=[4100], cached=[0], layers=1, layer=0, latent_stride=2624, skip_q_base=True
            ),
        ),
        Cell(
            why="q_in_place for the non-absorbed attention: three requests with cached "
            "prefixes, the in-place latent",
            spec=dict(
                new=[37, 64, 300],
                cached=[0, 32, 100],
                layers=3,
                layer=2,
                latent_stride=2624,
                q_in_place=True,
            ),
        ),
        Cell(
            why="q_in_place on an 8 x 987-row step, the served context shape",
            spec=dict(
                new=[987] * 8,
                cached=[0] * 8,
                layers=2,
                layer=1,
                latent_stride=2624,
                q_in_place=True,
            ),
        ),
        Cell(
            why="q_e4m3 for the fp8 attention: three requests with cached prefixes, the "
            "in-place latent",
            spec=dict(
                new=[37, 64, 300],
                cached=[0, 32, 100],
                layers=3,
                layer=2,
                latent_stride=2624,
                q_in_place=True,
                q_e4m3=True,
            ),
        ),
        Cell(
            why="q_e4m3 on an 8 x 987-row step, a partial last tile in none of them",
            spec=dict(
                new=[987] * 8,
                cached=[0] * 8,
                layers=2,
                layer=1,
                latent_stride=2624,
                q_in_place=True,
                q_e4m3=True,
            ),
        ),
    )

    note = """
    The target compiles one handle at load for its page size and takes this
    op on the context rows (the trunk's and the MTP layer's) when the step has
    at most MAX_REQUESTS (64) context requests and the layer's pool base is the
    primary pool with no secondary pool; otherwise it runs the stock
    `mla_rope_append_paged_kv_assign_q` plus a cast. The latent is the
    projection row's tail, read in place (stride 2624): `k_pe` is rotated
    inside the projection output, which nothing reads after the append. On a
    dense-causal step the trunk layers call it with `q_in_place`: the q_b
    output becomes the non-absorbed attention's query (`cute_dsl_mha_prefill`);
    with the fp8 attention built, `q_in_place` and `q_e4m3` on the e4m3 q_b output.

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
        fused_q: torch.Tensor | None,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        quant_q_buffer: torch.Tensor | None,
        kv_pool: torch.Tensor,
        layer_in_pool: int,
        kv_cache_block_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cached_lens: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        tokens_per_block: int,
        compiled: CompiledMlaRopeAppendPrefill,
        skip_q_base: bool = False,
        q_in_place: bool = False,
        q_e4m3: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if q_pe.device.index != compiled.device_index:
            raise ValueError(
                f"the kernel was compiled for device {compiled.device_index}, not {q_pe.device}"
            )
        if int(tokens_per_block) != compiled.tokens_per_block:
            raise ValueError(
                f"the kernel was compiled for {compiled.tokens_per_block} tokens per block, "
                f"not {tokens_per_block}"
            )
        # Another layout selects a specialization nothing compiled at load.
        table = (
            compiled.kernels_q_e4m3
            if q_in_place and q_e4m3
            else compiled.kernels_q_in_place
            if q_in_place
            else compiled.kernels_skip_q_base
            if skip_q_base
            else compiled.kernels
        )
        kernel = table.get(int(latent_cache.stride(0)))
        if tuple(q_pe.stride()[:2]) != (NUM_HEADS * QK_HEAD_DIM, QK_HEAD_DIM) or kernel is None:
            raise ValueError(
                "q_pe must be the rope tail of a contiguous [T, 16, 256] query and "
                f"latent_cache a [T, 576] with a row stride in {sorted(compiled.kernels)}"
            )
        base = kv_pool.data_ptr() + int(layer_in_pool) * int(tokens_per_block) * LATENT_DIM
        quant_q_u8 = None if quant_q_buffer is None else quant_q_buffer.view(torch.uint8)
        rope_par = compiled.rope_par if q_in_place and q_e4m3 else 0
        signature, launched = None, False
        if _HOST_DIET and not torch.cuda.is_current_stream_capturing():
            # CTX-HOST-DIET: a call signature `prepare_runner` passed before launches the same
            # arguments without its contract walk (`run_accepted`).
            options = (
                int(tokens_per_block), bool(skip_q_base), bool(q_in_place), bool(q_e4m3),
                bool(q_in_place), False, rope_par,
            )  # fmt: skip
            signature = _kernel.run_accepted(
                kernel, fused_q, q_pe, latent_cache, quant_q_u8, base, kv_cache_block_offsets,
                cu_seqlens, cached_lens, rotary_cos_sin, options,
            )  # fmt: skip
            launched = signature is None
        if not launched:
            runner = _kernel.prepare_runner(
                fused_q,
                q_pe,
                latent_cache,
                quant_q_u8,
                base,
                kv_cache_block_offsets,
                cu_seqlens,
                cached_lens,
                rotary_cos_sin,
                tokens_per_block=int(tokens_per_block),
                skip_q_base=bool(skip_q_base),
                q_in_place=bool(q_in_place),
                q_e4m3=bool(q_e4m3),
                merge_rope=bool(q_in_place),
                rope_par=rope_par,
            )
            dataclasses.replace(runner, compiled=kernel)()
            if signature is not None:
                _kernel.accept(signature)
        if q_in_place:
            return q_pe, latent_cache, None, kv_pool
        return fused_q, latent_cache, quant_q_buffer, kv_pool

    def reference(
        self,
        fused_q: torch.Tensor | None,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        quant_q_buffer: torch.Tensor | None,
        kv_pool: torch.Tensor,
        layer_in_pool: int,
        kv_cache_block_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cached_lens: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        tokens_per_block: int,
        compiled: CompiledMlaRopeAppendPrefill,
        skip_q_base: bool = False,
        q_in_place: bool = False,
        q_e4m3: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """`(fused_q, latent_cache, quant_q_buffer, kv_pool)` as the op leaves them,
        as new tensors; nothing passed in is written. `quant_q_buffer` is returned
        as uint8 bytes and `kv_pool` as a flat uint8 byte copy; with `skip_q_base`
        its first 512 columns of each head are the bytes passed in. With
        `q_in_place` the first value is `rope(q_pe)` and the third None; with `q_e4m3`
        it is e4m3 (`rope` of the widened values, rounded to bf16, then to e4m3). `compiled`
        is ignored.

        Bit-exact by construction: the rotation reproduces the kernel's fp32
        rounding (see `_rope`), and every other value is a copy or a single
        rounding.
        """
        t, tpb = int(q_pe.shape[0]), int(tokens_per_block)
        req, pos = _positions(cu_seqlens, cached_lens, t)
        table = rotary_cos_sin.reshape(-1, ROPE_DIM, 2).float()[pos, : ROPE_DIM // 2]
        cos, sin = table[..., 0], table[..., 1]
        q_rot = _rope(q_pe.to(torch.bfloat16), cos[:, None, :], sin[:, None, :])
        if q_e4m3:
            q_rot = _e4m3(q_rot).view(torch.float8_e4m3fn)
        lat_out = latent_cache.clone()
        lat_out[:, KV_LORA_RANK:] = _rope(latent_cache[:, KV_LORA_RANK:], cos, sin)
        pool = kv_pool.reshape(-1).view(torch.uint8).clone()
        block = kv_cache_block_offsets[req, 0, pos // tpb].to(torch.int64)
        start = (int(layer_in_pool) + block) * tpb * LATENT_DIM + (pos % tpb) * LATENT_DIM
        cols = torch.arange(LATENT_DIM, device=pool.device)
        pool[start[:, None] + cols[None, :]] = _e4m3(lat_out)
        if q_in_place:
            return q_rot, lat_out, None, pool
        q_out = fused_q.clone()
        q_out[..., KV_LORA_RANK:] = q_rot
        quant = _e4m3(q_out)
        if skip_q_base:
            quant[..., :KV_LORA_RANK] = quant_q_buffer.view(torch.uint8)[..., :KV_LORA_RANK]
        return q_out, lat_out, quant, pool

    def is_valid(
        self,
        fused_q: torch.Tensor | None,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        quant_q_buffer: torch.Tensor | None,
        kv_pool: torch.Tensor,
        layer_in_pool: int,
        kv_cache_block_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cached_lens: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        tokens_per_block: int,
        compiled: CompiledMlaRopeAppendPrefill,
        skip_q_base: bool = False,
        q_in_place: bool = False,
        q_e4m3: bool = False,
    ) -> None:
        """Reads the prefix sum and offsets back to the host: never on the serving path."""
        t, tpb = int(q_pe.shape[0]), int(tokens_per_block)
        b = int(cached_lens.shape[0])
        assert not q_e4m3 or q_in_place, "q_e4m3 is a variant of q_in_place"
        assert q_pe.dtype == (torch.float8_e4m3fn if q_e4m3 else torch.bfloat16), (
            f"q_pe must be {'e4m3' if q_e4m3 else 'bf16'}, got {q_pe.dtype}"
        )
        if q_in_place:
            assert fused_q is None and quant_q_buffer is None and not skip_q_base, (
                "q_in_place writes no fp8 query: fused_q and quant_q_buffer must be None"
            )
        else:
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
