# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A full-attention layer's pre-attention prep and decode-attention preprocess, in one launch.

On a full-attention (NoPE) layer the built-in decode step runs two launches
between the QKV projection and the trtllm-gen generation FMHA: the
pre-attention prep (``preattn_prep``) and the attention op's generation
preprocess (``thop.trtllm_gen_generation_preprocess``, whose only kernel there
is ``applyBiasRopeUpdateKVCacheV2``). This entry is ``preattn_prep``'s NoPE path
with everything that kernel writes on this layer folded in, so the FMHA can be
launched straight after it:

* the step's K and V into the paged FP8 cache, ``e4m3_rn_satfinite(bf16(x *
  bf16(s)))`` at ``pool[block_table[kv][pos // P], kv_head, pos % P, :]`` --
  on a NoPE layer that kernel applies no rotation, so this is a cast and a
  store (the conversion and addressing are ``paged_decode_attention``'s, which
  are byte-identical to the built-in append);
* Q in E4M3, the same conversion, ``[rows, H, 64]`` into the buffer the FMHA
  reads its query from;
* the FMHA's two scale words: ``bmm1 = [s_q * s_k * bmm1_host, that * log2(e)]``
  and ``bmm2 = o_scale * s_v``, with ``s_q = s_k = s_v`` the cache's dequant
  scale, in the built-in kernel's operation order.

Nothing else that preprocess computes is read on this path (the decoder-info
kernel does not run for one sequence without a mask, the tile counter is not
passed, the cache views are host-side). The fused QKV operand and the gate are
still written, bit for bit ``preattn_prep``'s, so a step that ends up running
the built-in preprocess after all reads the same operand.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .paged_decode_attention import _to_cache_fp8
from .preattn_prep import HEAD_DIM, _inverse_rms, _load8, _normalize, _split_projection

#: ``log2(e)`` as the built-in kernel's ``kLog2e`` float constant (the reference's; the
#: kernel spells the literal out).
LOG2E = 1.4426950408889634
#: Query rows a decode step may carry: one accepted token and up to 7 drafts.
MAX_ROWS = 8


@triton.jit
def _store8_fp8(ptr, y0, y1, y2, y3, y4, y5, y6, y7, scale_bf16):
    """The eight BF16-valued elements one norm thread owns, as E4M3 at ``ptr + 0..7``."""
    tl.store(ptr + 0, _to_cache_fp8(y0, scale_bf16))
    tl.store(ptr + 1, _to_cache_fp8(y1, scale_bf16))
    tl.store(ptr + 2, _to_cache_fp8(y2, scale_bf16))
    tl.store(ptr + 3, _to_cache_fp8(y3, scale_bf16))
    tl.store(ptr + 4, _to_cache_fp8(y4, scale_bf16))
    tl.store(ptr + 5, _to_cache_fp8(y5, scale_bf16))
    tl.store(ptr + 6, _to_cache_fp8(y6, scale_bf16))
    tl.store(ptr + 7, _to_cache_fp8(y7, scale_bf16))


@triton.jit
def _globaltimer(anchor):
    """The SM's nanosecond global timer (``anchor`` only orders the read after its producer)."""
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [anchor], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _preattn_prep_append_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    q8_ptr,
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    seq_len_ptr,
    orig_quant_ptr,
    quant_orig_ptr,
    bmm1_ptr,
    bmm2_ptr,
    bmm1_host,
    o_scale,
    qkv_stride,
    qkv_out_stride,
    gate_out_stride,
    eps,
    q_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    P: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    TRIGGER_AT_ENTRY: tl.constexpr = False,
    TEST_DELAY_NS: tl.constexpr = 0,
):
    D: tl.constexpr = 64
    PAGE: tl.constexpr = NUM_KV_HEADS * P * D
    if LAUNCH_WITH_PDL:
        if TRIGGER_AT_ENTRY:
            # Release the dependent (the trtllm-gen FMHA) at entry: its launch and
            # prologue overlap this grid. It reads the cache, the E4M3 query and the
            # scale words only after its own wait, which returns when this grid has
            # completed and every store below is visible.
            tl.extra.cuda.gdc_launch_dependents()
        # Before any global read: the projection is the previous kernel's
        # output. Without the entry trigger a dependent launches when this grid
        # ends; either way it reads what this grid writes only after its own wait.
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    if TEST_DELAY_NS > 0:
        # Test-only (the ordering control): spin before the first store.
        start = _globaltimer(head)
        now = start
        while now - start < TEST_DELAY_NS:
            now = _globaltimer(head)
    is_q = head < NUM_Q_HEADS
    kv_head = head - NUM_Q_HEADS
    in_row = qkv_ptr + row * qkv_stride
    out_row = qkv_out_ptr + row * qkv_out_stride
    scale_bf16 = tl.load(orig_quant_ptr).to(tl.bfloat16)

    # The step's cache slot of this row: the step's rows are the last ``rows``
    # positions of the sequence (seq_lens[0] counts them).
    position = tl.load(seq_len_ptr) - tl.num_programs(0) + tl.program_id(0)
    slot = (kv_head * P + position % P) * D

    # The pass-through half first, so its load is in flight during the norm:
    # a query program copies its gate head, a key program its value head (and
    # stores it into the cache's V page).
    d = tl.arange(0, D)
    if is_q:
        if QKV_PERMUTED:
            gate_in = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * D + head * D
        else:
            gate_in = head * 2 * D + D
        gate = tl.load(in_row + gate_in + d)
        tl.store(gate_out_ptr + row * gate_out_stride + head * D + d, gate)
    else:
        if QKV_PERMUTED:
            v_in = NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
        else:
            v_in = 2 * NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
        value = tl.load(in_row + v_in + d)
        tl.store(out_row + (NUM_Q_HEADS + NUM_KV_HEADS) * D + kv_head * D + d, value)
        page_v = tl.load(bt_ptr + bt_stride_kv + position // P).to(tl.int64)
        tl.store(pool_ptr + page_v * PAGE + slot + d, _to_cache_fp8(value, scale_bf16))

    if QKV_PERMUTED:
        in_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    else:
        in_off = tl.where(is_q, head * 2 * D, 2 * NUM_Q_HEADS * D + kv_head * D)
    out_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    scale = tl.where(is_q, q_scale, 1.0)

    lane = (
        4 * tl.arange(0, 2)[:, None, None]
        + 2 * tl.arange(0, 2)[None, :, None]
        + tl.arange(0, 2)[None, None, :]
    )
    x0, x1, x2, x3, x4, x5, x6, x7 = _load8(in_row + in_off + lane * 8)
    inverse_rms = _inverse_rms(x0, x1, x2, x3, x4, x5, x6, x7, eps)
    y0 = _normalize(x0, inverse_rms, scale)
    y1 = _normalize(x1, inverse_rms, scale)
    y2 = _normalize(x2, inverse_rms, scale)
    y3 = _normalize(x3, inverse_rms, scale)
    y4 = _normalize(x4, inverse_rms, scale)
    y5 = _normalize(x5, inverse_rms, scale)
    y6 = _normalize(x6, inverse_rms, scale)
    y7 = _normalize(x7, inverse_rms, scale)

    out = out_row + out_off + lane * 8
    tl.store(out + 0, y0.to(tl.bfloat16))
    tl.store(out + 1, y1.to(tl.bfloat16))
    tl.store(out + 2, y2.to(tl.bfloat16))
    tl.store(out + 3, y3.to(tl.bfloat16))
    tl.store(out + 4, y4.to(tl.bfloat16))
    tl.store(out + 5, y5.to(tl.bfloat16))
    tl.store(out + 6, y6.to(tl.bfloat16))
    tl.store(out + 7, y7.to(tl.bfloat16))

    # The E4M3 copies of the BF16 values just stored: Q into the FMHA's query
    # buffer, K into the cache's K page.
    if is_q:
        q8 = q8_ptr + (row * NUM_Q_HEADS + head) * D + lane * 8
        _store8_fp8(q8, y0, y1, y2, y3, y4, y5, y6, y7, scale_bf16)
    else:
        page_k = tl.load(bt_ptr + position // P).to(tl.int64)
        k8 = pool_ptr + page_k * PAGE + slot + lane * 8
        _store8_fp8(k8, y0, y1, y2, y3, y4, y5, y6, y7, scale_bf16)

    if (row == 0) & (head == 0):
        # The FMHA's scale words, in the built-in kernel's operation order:
        # (s_q * s_k) * bmm1_host, then that * log2(e); o_scale * s_v.
        s = tl.load(quant_orig_ptr)
        bmm1 = s * s * bmm1_host
        tl.store(bmm1_ptr, bmm1)
        tl.store(bmm1_ptr + 1, bmm1 * 1.4426950408889634)
        tl.store(bmm2_ptr, o_scale * s)


class PreattnPrepAppend(OpWrapper):
    """``preattn_prep`` on a NoPE layer, plus the step's FP8 K/V append, E4M3 Q and FMHA scales.

    With ``H = num_q_heads``, ``G = num_kv_heads``, head width 64, ``M`` rows at
    positions ``L - M .. L - 1`` (``L = seq_lens[0]``), the projection row in
    checkpoint order (or, with ``qkv_permuted``, ``[q | k | v | gate]``) and
    ``s = kv_scale_orig_quant[0]``, ``u = kv_scale_quant_orig[0]``::

        q'    = bf16(bf16(rmsnorm(q)) * q_scale)        # weightless, per head
        k'    = bf16(rmsnorm(k))
        qkv_out  = [q' | k' | v]                         # [M, (H + 2G) * 64]
        gate_out = [gate_0 | ... | gate_{H-1}]           # [M, H * 64]
        e4m3(x)  = e4m3_rn_satfinite(bf16(x * bf16(s)))
        q_fp8[t, h, :]                                   = e4m3(q'[t, h])
        kv_pool[block_tables[0][p // P], g, p % P, :]    = e4m3(k'[t, g])   p = L - M + t
        kv_pool[block_tables[1][p // P], g, p % P, :]    = e4m3(v[t, g])
        bmm1_scale[0:2] = [u * u * bmm1_host_scale, that * log2(e)]
        bmm2_scale[0]   = o_scale * u

    ``qkv_out`` and ``gate_out`` are fresh and contiguous; ``q_fp8`` (at least
    ``M * H * 64`` bytes, any dtype), ``kv_pool``, ``bmm1_scale`` (fp32, at least
    2) and ``bmm2_scale`` (fp32, at least 1) are written in place.
    ``block_tables`` is the sequence's ``[2, pages]`` row of the TRT-LLM table
    (K page ids, then V page ids) into ``kv_pool [pages, G, P, 64]``.

    Fusion boundary: the QKV projection producing ``qkv`` is outside; the
    trtllm-gen generation FMHA that reads ``q_fp8``, the cache and the scales,
    and the output gate's norm and sigmoid, are outside.

    ``enable_pdl`` launches with programmatic dependent launch: the grid may
    start while the projection's kernel finishes and waits before its first
    global read; it triggers its dependents only by completing, unless
    ``trigger_at_entry`` (with ``enable_pdl``; ignored without it) releases them
    at entry, before its wait.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0] (device, read by the kernel), the
    sequence's K and V page ids, the layer's pool view, the two FP8 KV scales,
    the FMHA's host bmm1 scale and the output scale (1.0 for a BF16 output).
    Call site: the target's 16 full-attention layers on decode steps of one
    sequence and at most 8 rows (fast path full_attention_prep_append), whose
    trtllm-gen generation preprocess launch the target then skips; a caller
    may launch it with trigger_at_entry to release its dependents at entry. Grid (rows, H + G), one warp per program.

    Bit-identical, gated by its own tests: qkv_out and gate_out against
    preattn_prep and the built-in prep; the cache bytes, the E4M3 query bytes
    and both scale words against thop.trtllm_gen_generation_preprocess
    (applyBiasRopeUpdateKVCacheV2) run on the same projection, pool and
    tables, including saturating, NaN, zero and special heads; and the FMHA
    output on both paths, eager and in a captured graph.

    Loud preconditions (ValueError before any launch): BF16 projection with
    unit last stride and width 2 * (H + G) * 64, 1..8 rows, FP8 E4M3 pool
    [pages, G, P, 64] with P a power of two, int32 [2, pages] block table with
    unit last stride, int32 seq_lens, fp32 scales, a query buffer of at least
    M * H * 64 bytes.

    Not certified: RoPE layers (the full-attention layers are NoPE), more
    than one sequence per call, head width other than 64, NVFP4 or BF16 caches.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        q_fp8: torch.Tensor,
        bmm1_scale: torch.Tensor,
        bmm2_scale: torch.Tensor,
        bmm1_host_scale: float,
        o_scale: float = 1.0,
        enable_pdl: bool = False,
        trigger_at_entry: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, width = qkv.shape
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        if (
            qkv.dtype != torch.bfloat16
            or qkv.stride(-1) != 1
            or width != 2 * (num_q_heads + num_kv_heads) * HEAD_DIM
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or page & (page - 1) != 0
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 2
            or block_tables.shape[0] != 2
            or block_tables.stride(-1) != 1
            or seq_lens.dtype != torch.int32
            or kv_scale_orig_quant.dtype != torch.float32
            or kv_scale_quant_orig.dtype != torch.float32
            or bmm1_scale.dtype != torch.float32
            or bmm1_scale.numel() < 2
            or bmm2_scale.dtype != torch.float32
            or bmm2_scale.numel() < 1
            or not q_fp8.is_contiguous()
            or q_fp8.numel() * q_fp8.element_size() < rows * num_q_heads * HEAD_DIM
        ):
            raise ValueError(
                "preattn_prep_append takes a BF16 [1..8, 2 (H + G) 64] projection, an FP8 E4M3 "
                "pool [pages, G, P, 64] (P a power of two), an int32 [2, pages] block table, "
                "int32 seq_lens, fp32 scales and a contiguous query buffer of M H 64 bytes; got "
                f"qkv {qkv.dtype} {tuple(qkv.shape)} strides {qkv.stride()}, H={num_q_heads} "
                f"G={num_kv_heads}, pool {kv_pool.dtype} {tuple(kv_pool.shape)}, table "
                f"{block_tables.dtype} {tuple(block_tables.shape)}, seq_lens {seq_lens.dtype}, "
                f"scales {kv_scale_orig_quant.dtype}/{kv_scale_quant_orig.dtype}, bmm "
                f"{bmm1_scale.dtype}[{bmm1_scale.numel()}]/{bmm2_scale.dtype}"
                f"[{bmm2_scale.numel()}], query buffer {q_fp8.numel() * q_fp8.element_size()} bytes"
            )
        qkv_out = torch.empty(
            (rows, (num_q_heads + 2 * num_kv_heads) * HEAD_DIM), dtype=qkv.dtype, device=qkv.device
        )
        gate_out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=qkv.dtype, device=qkv.device)
        _preattn_prep_append_kernel[(rows, num_q_heads + num_kv_heads)](
            qkv,
            qkv_out,
            gate_out,
            q_fp8.view(torch.uint8).view(torch.float8_e4m3fn),
            kv_pool,
            block_tables,
            block_tables.stride(0),
            seq_lens,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            bmm1_scale,
            bmm2_scale,
            bmm1_host_scale,
            o_scale,
            qkv.stride(0),
            qkv_out.stride(0),
            gate_out.stride(0),
            eps,
            q_scale,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            QKV_PERMUTED=qkv_permuted,
            P=page,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=1,
            launch_pdl=enable_pdl,
            # Passed only when set: the default launch is the kernel without the switch.
            **({"TRIGGER_AT_ENTRY": True} if enable_pdl and trigger_at_entry else {}),
        )
        return qkv_out, gate_out

    def reference(
        self,
        qkv: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        q_fp8: torch.Tensor,
        bmm1_scale: torch.Tensor,
        bmm2_scale: torch.Tensor,
        bmm1_host_scale: float,
        o_scale: float = 1.0,
        enable_pdl: bool = False,
        trigger_at_entry: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """float64 between the rounding points the op computes, and the same in-place writes.

        The prep rounds to BF16 after the norm and after the query scale; the
        E4M3 conversions round the BF16 product with the BF16 scale, then to
        E4M3 with saturation. Those roundings are what the op computes, so the
        reference keeps them; between them it is exact to float64. The scale
        words are fp32 products in the kernel's order. ``enable_pdl`` and
        ``trigger_at_entry`` change how the op is launched and are ignored here.
        """
        rows = qkv.shape[0]
        page = kv_pool.shape[2]
        q, gate, k, v = _split_projection(qkv, num_q_heads, num_kv_heads, qkv_permuted)
        q_mul = torch.tensor(q_scale, dtype=torch.float32).item()

        def norm(x: torch.Tensor) -> torch.Tensor:
            x = x.double()
            return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(qkv.dtype)

        q = (norm(q).double() * q_mul).to(qkv.dtype)
        k = norm(k)
        s_bf16 = kv_scale_orig_quant.float()[0].to(torch.bfloat16).float()

        def e4m3(x: torch.Tensor) -> torch.Tensor:
            # E4M3 round-to-nearest with saturation: a finite overflow lands on
            # +-448 rather than torch's NaN; NaN stays NaN.
            y = (x.float() * s_bf16).to(torch.bfloat16).float()
            return y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)

        q8 = q_fp8.view(torch.uint8).reshape(-1)
        q8[: rows * num_q_heads * HEAD_DIM] = e4m3(q).reshape(-1)
        pool_bytes = kv_pool.view(torch.uint8)
        length = int(seq_lens[0])
        positions = torch.arange(length - rows, length, device=qkv.device)
        k_pages = block_tables[0, positions // page].long()
        v_pages = block_tables[1, positions // page].long()
        slots = positions % page
        for g in range(num_kv_heads):
            pool_bytes[k_pages, g, slots] = e4m3(k[:, g])
            pool_bytes[v_pages, g, slots] = e4m3(v[:, g])
        u = kv_scale_quant_orig.float()[0]
        bmm1 = u * u * torch.tensor(bmm1_host_scale, dtype=torch.float32, device=u.device)
        bmm1_scale[0] = bmm1
        bmm1_scale[1] = bmm1 * torch.tensor(LOG2E, dtype=torch.float32, device=u.device)
        bmm2_scale[0] = torch.tensor(o_scale, dtype=torch.float32, device=u.device) * u
        qkv_out = torch.cat([q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1)], dim=-1)
        return qkv_out, gate.reshape(rows, -1).contiguous()

    def is_valid(
        self,
        qkv: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        q_fp8: torch.Tensor,
        bmm1_scale: torch.Tensor,
        bmm2_scale: torch.Tensor,
        bmm1_host_scale: float,
        o_scale: float = 1.0,
        enable_pdl: bool = False,
        trigger_at_entry: bool = False,
    ) -> None:
        """Inputs the kernel takes and answers wrongly, with no error.

        A context shorter than the step's rows puts the step's keys at negative
        positions; a block table shorter than the context reads page ids past
        its end; a page id outside the pool writes outside it.
        """
        rows = qkv.shape[0]
        length = int(seq_lens[0])
        page = kv_pool.shape[2]
        if length < rows:
            raise ValueError(f"seq_lens[0] = {length} is shorter than the step's {rows} rows")
        last = (length - 1) // page
        if block_tables.shape[1] <= last:
            raise ValueError(
                f"a [2, {block_tables.shape[1]}] block table cannot address {length} keys "
                f"of {page}-token pages"
            )
        used = block_tables[:, (length - rows) // page : last + 1]
        if int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]:
            raise ValueError(
                f"page ids [{int(used.min())}, {int(used.max())}] outside the "
                f"{kv_pool.shape[0]}-page pool"
            )


preattn_prep_append = PreattnPrepAppend()
