# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer K-cache gather for prefill chunks (CuTe DSL).

The indexer gathers each prefill chunk's FP8 index keys and their FP32 scales out of the paged
indexer K cache. The launcher is :func:`k31_gather_k_cache_paged`: the same bytes as the
engine's ``trtllm::indexer_k_cache_gather_op``.

The cache is a uint8 view ``[pages, tokens_per_block, 1, bytes_per_token]`` whose tokens are
contiguous inside a page and whose pages sit ``page_stride`` bytes apart (one layer of a
multi-layer pool). Token ``t`` of the chunk reads ``head_dim`` key bytes at the flat byte index
``slot_mapping_fp8[k_token_start + t]`` and 4 scale bytes at ``slot_mapping_scale[...]``; a flat
index ``f`` addresses byte ``f % page_bytes`` of page ``f // page_bytes``. A token with either
slot negative is skipped (its output rows are left unwritten).

``head_dim / 16`` consecutive threads move one token's keys, 16 bytes each, with one 16-byte load
and store when the source address is 16-byte aligned (always, for the indexer's layout) and byte
by byte otherwise; the first of them moves the scale (one 4-byte word when aligned). A CTA of 256
threads covers ``4096 / head_dim`` tokens. Launches use programmatic dependent launch when
``TRTLLM_ENABLE_PDL`` is unset or ``1``: the kernel waits for its producer before it reads the
slot mappings; a grid of at most one wave also lets its dependent launch once its loads are
issued (a larger one leaves that to its exit).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

#: Bytes of one token's scale (one FP32).
SCALE_BYTES = 4
#: Payload bytes per token the kernel is instantiated for.
HEAD_DIMS = (64, 128, 256)
_THREADS = 256
_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _l(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _i(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_b64(addr, pred, *, loc=None, ip=None):
    """The 8-byte word at ``addr``, or -1 where ``pred`` is 0 (no memory access)."""
    return Int64(
        _asm(
            T.i64(),
            [_l(addr, loc=loc, ip=ip), _i(pred, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b64 $0, -1;\n"
            "@p ld.global.nc.b64 $0, [$1];\n}",
            "=l,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _copy_v4(src, dst, pred, *, loc=None, ip=None):
    """Copy 16 bytes, both addresses 16-byte aligned, where ``pred`` is nonzero."""
    _asm(
        None,
        [_l(src, loc=loc, ip=ip), _l(dst, loc=loc, ip=ip), _i(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\nsetp.ne.b32 p, $2, 0;\n"
        "@p ld.global.nc.v4.b32 {a, b, c, d}, [$0];\n"
        "@p st.global.v4.b32 [$1], {a, b, c, d};\n}",
        "l,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_b32(src, dst, pred, *, loc=None, ip=None):
    """Copy 4 bytes, both addresses 4-byte aligned, where ``pred`` is nonzero."""
    _asm(
        None,
        [_l(src, loc=loc, ip=ip), _l(dst, loc=loc, ip=ip), _i(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\n.reg .b32 a;\nsetp.ne.b32 p, $2, 0;\n"
        "@p ld.global.nc.b32 a, [$0];\n@p st.global.b32 [$1], a;\n}",
        "l,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_bytes(src, dst, count, pred, *, loc=None, ip=None):
    """Copy ``count`` bytes one at a time (any alignment) where ``pred`` is nonzero."""
    body = "{\n.reg .pred p;\n.reg .b32 a;\nsetp.ne.b32 p, $2, 0;\n"
    for i in range(count):
        body += f"@p ld.global.nc.u8 a, [$0+{i}];\n@p st.global.u8 [$1+{i}], a;\n"
    body += "}"
    _asm(
        None,
        [_l(src, loc=loc, ip=ip), _l(dst, loc=loc, ip=ip), _i(pred, loc=loc, ip=ip)],
        body,
        "l,l,r",
        loc=loc,
        ip=ip,
    )


class GatherKCachePagedKernel:
    """Tokens ``[k_token_start, k_token_start + num_k_tokens)`` of the slot mappings, ``head_dim``
    key bytes and 4 scale bytes each, ``head_dim / 16`` threads per token."""

    def __init__(self, head_dim: int, page_bytes: int, use_pdl: bool):
        if head_dim not in HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
        self.head_dim = head_dim
        self.page_bytes = page_bytes
        self.lanes = head_dim // 16
        self.tokens_per_cta = _THREADS // self.lanes
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        k_cache: cute.Tensor,
        slot_fp8: cute.Tensor,
        slot_scale: cute.Tensor,
        out_fp8: cute.Tensor,
        out_scale: cute.Tensor,
        k_token_start: Int64,
        page_stride: Int64,
        trigger: Int32,
        stream: cuda.CUstream,
    ):
        num_tokens = out_fp8.shape[0]
        self.kernel(
            k_cache.iterator.toint(),
            slot_fp8.iterator.toint(),
            slot_scale.iterator.toint(),
            out_fp8.iterator.toint(),
            out_scale.iterator.toint(),
            k_token_start,
            num_tokens,
            page_stride,
            trigger,
        ).launch(
            grid=[(num_tokens + self.tokens_per_cta - 1) // self.tokens_per_cta, 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        cache_addr: Int64,
        slot_fp8_addr: Int64,
        slot_scale_addr: Int64,
        out_fp8_addr: Int64,
        out_scale_addr: Int64,
        k_token_start: Int64,
        num_tokens: Int32,
        page_stride: Int64,
        trigger: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % self.lanes
        t = bidx * self.tokens_per_cta + tidx // self.lanes
        in_range = Int32(t < num_tokens)
        slot_off = (k_token_start + Int64(t)) * 8

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        f = _ld_b64(slot_fp8_addr + slot_off, in_range)
        fs = _ld_b64(slot_scale_addr + slot_off, in_range)
        ok = Int32(f >= Int64(0)) & Int32(fs >= Int64(0))

        # The keys: 16 bytes per lane.
        src = cache_addr + (f // self.page_bytes) * page_stride + f % self.page_bytes + lane * 16
        dst = out_fp8_addr + Int64(t) * self.head_dim + lane * 16
        aligned = Int32((src & Int64(15)) == Int64(0))
        _copy_v4(src, dst, ok & aligned)
        # The scale: one word, from the first lane of the token.
        src_s = cache_addr + (fs // self.page_bytes) * page_stride + fs % self.page_bytes
        dst_s = out_scale_addr + Int64(t) * SCALE_BYTES
        first = Int32(lane == 0)
        aligned_s = Int32((src_s & Int64(3)) == Int64(0))
        _copy_b32(src_s, dst_s, ok & first & aligned_s)
        if cutlass.const_expr(self.use_pdl):
            if trigger != 0:
                cute.arch.griddepcontrol_launch_dependents()

        # Unaligned sources (not produced by the indexer's own layout): byte copies.
        if (ok & (aligned ^ 1)) != 0:
            _copy_bytes(src, dst, 16, Int32(1))
        if (ok & first & (aligned_s ^ 1)) != 0:
            _copy_bytes(src_s, dst_s, SCALE_BYTES, Int32(1))


@functools.cache
def _compile(head_dim: int, page_bytes: int, use_pdl: bool):
    cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64(), 1),
    )
    slots = cute.sym_int()
    slot_fp8_fake = cute.runtime.make_fake_tensor(cutlass.Int64, (slots,), stride=(1,))
    slot_scale_fake = cute.runtime.make_fake_tensor(cutlass.Int64, (cute.sym_int(),), stride=(1,))
    tokens = cute.sym_int()
    out_fp8_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (tokens, head_dim), stride_order=(1, 0)
    )
    out_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (tokens, SCALE_BYTES), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = GatherKCachePagedKernel(head_dim, page_bytes, use_pdl)
    return cute.compile(
        kernel,
        cache_fake,
        slot_fp8_fake,
        slot_scale_fake,
        out_fp8_fake,
        out_scale_fake,
        Int64(0),
        Int64(0),
        Int32(0),
        stream_fake,
        options=_OPTIONS,
    )


#: CTAs up to which the grid is taken for one wave (about 7 per SM).
_ONE_WAVE_CTAS = 1024


def supports_layout(k_cache_4d: torch.Tensor, head_dim: int) -> bool:
    """Whether the kernel serves this ``[pages, tokens_per_block, 1, bytes_per_token]`` uint8 view
    (contiguous tokens within a page, any page stride) and payload width."""
    _, _, heads, per_token = k_cache_4d.shape
    return (
        heads == 1
        and k_cache_4d.dtype == torch.uint8
        and k_cache_4d.stride(3) == 1
        and k_cache_4d.stride(1) == per_token
        and head_dim in HEAD_DIMS
    )


def gather_k_cache_paged_into(
    k_cache_4d: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    use_pdl: bool | None = None,
    trigger: bool | None = None,
) -> None:
    """Gather ``out_fp8.shape[0]`` tokens into ``out_fp8`` (uint8 ``[N, head_dim]``) and
    ``out_scale`` (uint8 ``[N, 4]``), both contiguous. The layout is not re-checked (see
    :func:`supports_layout`); the slot mappings are contiguous int64. Compiles once per
    ``(head_dim, page bytes)`` on the first call."""
    num_tokens, head_dim = out_fp8.shape
    if num_tokens == 0:
        return
    _, tpb, _, per_token = k_cache_4d.shape
    if use_pdl is None:
        use_pdl = get_env_enable_pdl()
    if trigger is None:
        trigger = num_tokens * head_dim <= _ONE_WAVE_CTAS * _THREADS * 16
    compiled = _compile(head_dim, tpb * per_token, use_pdl)
    compiled(
        k_cache_4d,
        slot_mapping_fp8,
        slot_mapping_scale,
        out_fp8,
        out_scale,
        k_token_start,
        k_cache_4d.stride(0),
        int(trigger),
    )


def k31_gather_k_cache_paged(
    k_cache_4d: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    num_k_tokens: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``trtllm::indexer_k_cache_gather_op``'s outputs: ``(k uint8 [num_k_tokens, head_dim]`` viewed
    as FP8 E4M3, ``scale fp32 [num_k_tokens, 1])``, tokens ``[k_token_start, k_token_start +
    num_k_tokens)`` of the slot mappings (a negative slot skips the token: its rows are left
    unwritten, as there)."""
    if not supports_layout(k_cache_4d, head_dim):
        raise ValueError(
            "k31_gather_k_cache_paged takes a uint8 [pages, tokens_per_block, 1, bytes] view with "
            f"contiguous tokens and head_dim in {HEAD_DIMS}"
        )
    for slots in (slot_mapping_fp8, slot_mapping_scale):
        if slots.dtype != torch.int64 or slots.dim() != 1 or slots.stride(0) != 1:
            raise ValueError("k31_gather_k_cache_paged takes contiguous 1D int64 slot mappings")
    device = k_cache_4d.device
    out_fp8 = torch.empty(num_k_tokens, head_dim, dtype=torch.uint8, device=device)
    out_scale = torch.empty(num_k_tokens, SCALE_BYTES, dtype=torch.uint8, device=device)
    gather_k_cache_paged_into(
        k_cache_4d, slot_mapping_fp8, slot_mapping_scale, k_token_start, out_fp8, out_scale
    )
    return out_fp8.view(torch.float8_e4m3fn), out_scale.view(torch.float32).view(num_k_tokens, 1)


__all__ = [
    "GatherKCachePagedKernel",
    "HEAD_DIMS",
    "SCALE_BYTES",
    "gather_k_cache_paged_into",
    "k31_gather_k_cache_paged",
    "supports_layout",
]
