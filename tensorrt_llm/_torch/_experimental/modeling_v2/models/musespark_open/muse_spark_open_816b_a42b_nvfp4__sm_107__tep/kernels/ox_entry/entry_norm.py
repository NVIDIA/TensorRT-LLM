# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter's layer-0 input RMSNorm with an entry trigger (fast path ``drafter_folds``).

The DFlash 2 drafter's block decode starts with ``layers[0].input_layernorm``
of the noise block (``drafter_block_decode.block_decode``, the one norm of the
forward that no hop applies). The built-in call is flashinfer's CuTe-DSL
``RMSNormKernel``; it releases its dependents only when it ends, so layer 0's
coefficient GEMV (``attention_convs[0].prepare``, a PDL-launched split-K GEMV
whose weight warp streams before its wait) cannot start streaming weights
before the norm has finished. This kernel computes the same bits and executes
``griddepcontrol.launch_dependents`` at its entry.

Arithmetic, as ``RMSNormKernel`` compiles for an 8192-wide BF16 row (the
statements ``kernels/pf_pre/pf_pre.cu`` and ``ops/allreduce/drafter_hop.py``
document for this kernel and its fused-add sibling): one CTA of 128 threads
per row; thread ``t`` owns the 64 columns ``8 t + 1024 b + e``; one FMA chain
``s = fma(x, x, s)`` from ``+0`` over ``b`` then ``e`` ascending; a butterfly
over the warp's lanes (offsets 1, 2, 4, 8, 16); the same butterfly over the
four warp sums padded with ``+0``; ``rstd = rsqrt.approx.ftz(fma(s, 2^-13,
eps))``; ``y = (x * rstd) * (w + 0)``, rounded once to BF16. Every operation
is an inline PTX instruction (round to nearest, no contraction), so the
compiler cannot fuse or reorder them.

Ordering. The trigger only lets the dependent launch early; a dependent
launched with programmatic dependent launch still waits (``griddepcontrol.wait``)
for this grid's completion before it reads the output, as flashinfer's split-K
GEMV's activation warp does. The weight (a parameter, written at load time
only) is read before this kernel's own wait; the input only after it.

Part of the ``drafter_folds`` fast path (``install``). Value-identical: the same
statements on the same values.

``EntryGatherNorm``: the same kernel with the noise block's gather in front, for
the verify fold's prepare (``verify_fold/fold.py``), in place of
``replicated_embedding_gather_twice``
(``ops/embedding/replicated_gather.py``) and this norm. Row ``r`` reads its token
id ``ids[r]`` (int64) after the wait; an id outside ``[0, vocab)`` gives a zero
row, any other the table row with each BF16 half of zero magnitude written as
``+0``, as the gather kernel does. The kernel stores those bits twice (the noise
block and the residual copy, the gather's two outputs) and normalizes them with
the statements above; the drafter forward takes the normalized rows from the
drafter preamble (``drafter_block_decode.DrafterPreamble.normed``) instead of running layer
0's input norm. Unlike the norm alone, this kernel releases its dependent only
after its own ``griddepcontrol.wait`` (its predecessor is the verify fold, as for
the gather it replaces). Value-identical: the gather's bits, then the norm's
statements on them. Only an explicit ``install_gather`` call installs it.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

HIDDEN = 8192
THREADS = 128
WARPS = THREADS // 32
VECTOR = 8
BLOCKS = HIDDEN // (THREADS * VECTOR)
ROW_BYTES = HIDDEN * 2
#: Rows the drafter's block decode normalizes at layer 0 (one request of K + 1 = 8; up to 16 taken).
MAX_ROWS = 16

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    HAS_CUTE_DSL = False
else:
    HAS_CUTE_DSL = True


if HAS_CUTE_DSL:
    from ...ops.allreduce.drafter_hop import (
        _add_rn,
        _asm,
        _fma_rn,
        _i64,
        _load_x4,
        _load_x4_if,
        _mul_rn,
        _pack_bf16x2,
        _rsqrt_ftz,
        _shfl_bfly,
        _store_x4,
        _u32,
        _unpack_bf16x2,
    )

    @dsl_user_op
    def _load_s64(address, *, loc=None, ip=None):
        """One 8-byte global load (a token id), as a side-effecting statement."""
        return Int64(
            _asm(T.i64(), [_i64(address, loc, ip)], "ld.global.s64 $0, [$1];", "=l,l", loc, ip)
        )

    @dsl_user_op
    def _row_valid(token, vocab, *, loc=None, ip=None):
        """1 when ``0 <= token < vocab``, else 0."""
        return Uint32(
            _asm(
                T.i32(),
                [_i64(token, loc, ip), _i64(vocab, loc, ip)],
                "{\n.reg .pred p, q;\nsetp.ge.s64 p, $1, 0;\nsetp.lt.s64 q, $1, $2;\n"
                "and.pred p, p, q;\nselp.u32 $0, 1, 0, p;\n}",
                "=r,l,l",
                loc,
                ip,
                side_effects=False,
            )
        )

    @dsl_user_op
    def _row_source(table, token, valid, *, loc=None, ip=None):
        """``table + token * ROW_BYTES`` for a valid row, ``table`` otherwise (never read then)."""
        return Int64(
            _asm(
                T.i64(),
                [_i64(table, loc, ip), _i64(token, loc, ip), _u32(valid, loc, ip)],
                "{\n.reg .pred p;\n.reg .s64 t;\nsetp.ne.u32 p, $3, 0;\nselp.s64 t, $2, 0, p;\n"
                f"mul.lo.s64 t, t, {ROW_BYTES};\nadd.s64 $0, t, $1;\n}}",
                "=l,l,l,r",
                loc,
                ip,
                side_effects=False,
            )
        )

    @dsl_user_op
    def _pos_zero(word, *, loc=None, ip=None):
        """``word`` with each BF16 half of zero magnitude (``+0`` / ``-0``) written as ``+0``."""
        return Uint32(
            _asm(
                T.i32(),
                [_u32(word, loc, ip)],
                "{\n.reg .b32 t;\n.reg .pred p;\nmov.b32 $0, $1;\n"
                "and.b32 t, $1, 0x00007FFF;\nsetp.eq.u32 p, t, 0;\n@p and.b32 $0, $0, 0xFFFF0000;\n"
                "and.b32 t, $1, 0x7FFF0000;\nsetp.eq.u32 p, t, 0;\n@p and.b32 $0, $0, 0x0000FFFF;\n}",
                "=r,r",
                loc,
                ip,
                side_effects=False,
            )
        )

    class _EntryNormKernel:
        """flashinfer's 8192-wide BF16 ``RMSNormKernel``, one row per CTA, dependents released at entry.

        ``gather``: the row is the noise block's gather of token ``ids[row]`` from the table ``x``
        (``EntryGatherNorm``), stored to ``noise`` and ``residual`` before the norm; otherwise row ``row`` of ``x``.
        """

        def __init__(self, eps: float, pdl: bool, gather: bool = False):
            self.eps = float(eps)
            self.pdl = bool(pdl)
            self.gather = bool(gather)

        @cute.jit
        def __call__(
            self,
            x: Int64,
            weight: Int64,
            out: Int64,
            ids: Int64,
            vocab: Int64,
            noise: Int64,
            residual: Int64,
            rows: Int32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(x, weight, out, ids, vocab, noise, residual, dry).launch(
                grid=[rows, 1, 1],
                block=[THREADS, 1, 1],
                smem=WARPS * 4 + 64,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            x: Int64,
            weight: Int64,
            out: Int64,
            ids: Int64,
            vocab: Int64,
            noise: Int64,
            residual: Int64,
            dry: Int32,
        ):
            tid = cute.arch.thread_idx()[0]
            lane = tid % 32
            warp = tid // 32
            row = cute.arch.block_idx()[0]
            smem = utils.SmemAllocator()
            sums = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), byte_alignment=16)
            if dry == 0:
                if cutlass.const_expr(self.pdl and not self.gather):
                    cute.arch.griddepcontrol_launch_dependents()
                # The weight (a parameter) before the wait, the input after it.
                w = cute.make_rmem_tensor(cute.make_layout((BLOCKS, 4)), Uint32)
                for b in cutlass.range_constexpr(BLOCKS):
                    column_bytes = Int64(VECTOR * tid + THREADS * VECTOR * b) * 2
                    w0, w1, w2, w3 = _load_x4(weight + column_bytes)
                    w[b, 0] = w0
                    w[b, 1] = w1
                    w[b, 2] = w2
                    w[b, 3] = w3
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.pdl and self.gather):
                    # The gather variant releases its dependent only after its wait, as the gather it
                    # replaces does: the predecessor is the verify fold, and a GEMV released before the
                    # fold completes would hold its SMs (and stream its weights) during the sampler's
                    # last kernels.
                    cute.arch.griddepcontrol_launch_dependents()
                row_base = Int64(row) * ROW_BYTES
                v = cute.make_rmem_tensor(cute.make_layout((BLOCKS, 4)), Uint32)
                if cutlass.const_expr(self.gather):
                    # The gather variant: the gather's bits (zero row for an id outside the table, +0 for -0),
                    # stored as the noise block and its residual copy.
                    token = _load_s64(ids + Int64(row) * 8)
                    valid = _row_valid(token, vocab)
                    source = _row_source(x, token, valid)
                    for b in cutlass.range_constexpr(BLOCKS):
                        column_bytes = Int64(VECTOR * tid + THREADS * VECTOR * b) * 2
                        g0, g1, g2, g3 = _load_x4_if(source + column_bytes, valid)
                        v0 = _pos_zero(g0)
                        v1 = _pos_zero(g1)
                        v2 = _pos_zero(g2)
                        v3 = _pos_zero(g3)
                        _store_x4(noise + row_base + column_bytes, v0, v1, v2, v3)
                        _store_x4(residual + row_base + column_bytes, v0, v1, v2, v3)
                        v[b, 0] = v0
                        v[b, 1] = v1
                        v[b, 2] = v2
                        v[b, 3] = v3
                else:
                    for b in cutlass.range_constexpr(BLOCKS):
                        column_bytes = Int64(VECTOR * tid + THREADS * VECTOR * b) * 2
                        v0, v1, v2, v3 = _load_x4(x + row_base + column_bytes)
                        v[b, 0] = v0
                        v[b, 1] = v1
                        v[b, 2] = v2
                        v[b, 3] = v3
                # One FMA chain over b then e ascending, from +0.
                s = Float32(0.0)
                for b in cutlass.range_constexpr(BLOCKS):
                    for e in cutlass.range_constexpr(VECTOR):
                        value = _unpack_bf16x2(v[b, e // 2])[e % 2]
                        s = _fma_rn(value, value, s)
                for i in cutlass.range_constexpr(5):
                    s = _add_rn(s, _shfl_bfly(s, Uint32(1 << i)))
                if lane == 0:
                    sums[warp] = s
                cute.arch.barrier()
                row_sum = Float32(0.0)
                if lane < WARPS:
                    row_sum = sums[lane]
                for i in cutlass.range_constexpr(5):
                    row_sum = _add_rn(row_sum, _shfl_bfly(row_sum, Uint32(1 << i)))
                rstd = _rsqrt_ftz(_fma_rn(row_sum, Float32(1.0 / HIDDEN), Float32(self.eps)))
                for b in cutlass.range_constexpr(BLOCKS):
                    column_bytes = Int64(VECTOR * tid + THREADS * VECTOR * b) * 2
                    packed = []
                    for k in cutlass.range_constexpr(4):
                        lo_x, hi_x = _unpack_bf16x2(v[b, k])
                        lo_w, hi_w = _unpack_bf16x2(w[b, k])
                        lo = _mul_rn(_mul_rn(lo_x, rstd), _add_rn(lo_w, Float32(0.0)))
                        hi = _mul_rn(_mul_rn(hi_x, rstd), _add_rn(hi_w, Float32(0.0)))
                        packed.append(_pack_bf16x2(lo, hi))
                    _store_x4(out + row_base + column_bytes, *packed)

    @functools.lru_cache(maxsize=16)
    def _compiled(device_index: int, eps: float, pdl: bool, gather: bool = False):
        kernel = _EntryNormKernel(eps, pdl, gather)
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int32(1),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


def norm_reason(norm: nn.Module) -> Optional[str]:
    """Why ``norm`` is not the plain flashinfer RMSNorm this kernel reproduces, or None."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not HAS_CUTE_DSL:
        return "CuTe DSL is unavailable"
    if not IS_FLASHINFER_AVAILABLE:
        return "flashinfer (the built-in RMSNorm this kernel reproduces) is unavailable"
    if type(norm) is not RMSNorm:
        return f"the layer-0 input norm is {type(norm).__name__}, not the built-in RMSNorm"
    if norm.use_gemma or norm.use_cuda_tile or norm.return_hp_output:
        return "the layer-0 input norm is not the plain flashinfer RMSNorm"
    if norm.is_nvfp4 and norm.nvfp4_scale is not None:
        return "the layer-0 input norm quantizes its output"
    weight = getattr(norm, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (HIDDEN,):
        return f"the layer-0 input norm has no BF16 weight of {HIDDEN}"
    if not weight.is_cuda or not weight.is_contiguous() or weight.data_ptr() % 16:
        return "the layer-0 input norm weight is not a contiguous, 16-byte aligned CUDA tensor"
    return None


class EntryNorm:
    """The layer-0 input norm of one drafter: ``__call__(x)`` for ``takes(x)`` inputs."""

    def __init__(self, norm: nn.Module):
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        self.norm = norm
        self.weight = norm.weight
        self.eps = float(norm.variance_epsilon)
        self.device = self.weight.device
        # The built-in call (``flashinfer_rmsnorm``) launches with programmatic dependent launch
        # when ``TRTLLM_ENABLE_PDL`` enables it; this kernel takes the same launch attribute.
        self.pdl = bool(get_env_enable_pdl())
        self._kernel = _compiled(self.device.index, self.eps, self.pdl)
        dummy = self.weight.data_ptr()
        with torch.cuda.device(self.device):
            self._kernel(dummy, dummy, dummy, 0, 0, 0, 0, 1, 1, self._stream())
            torch.cuda.synchronize(self.device)
        #: Calls taken (a test counts them).
        self.calls = 0

    def _stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def takes(self, x: torch.Tensor) -> bool:
        """Whether ``x`` is a contiguous, aligned ``[1..16, 8192]`` BF16 tensor on this device."""
        return (
            x.dim() == 2
            and 1 <= x.shape[0] <= MAX_ROWS
            and x.shape[1] == HIDDEN
            and x.dtype == torch.bfloat16
            and x.device == self.device
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self._kernel(
            x.data_ptr(),
            self.weight.data_ptr(),
            out.data_ptr(),
            0,
            0,
            0,
            0,
            x.shape[0],
            0,
            self._stream(),
        )
        self.calls += 1
        return out


class EntryGatherNorm:
    """The noise block's gather, its residual copy and the layer-0 input norm in one kernel.

    ``__call__(table, ids)`` for ``takes(table, ids)`` inputs returns ``(noise, residual,
    normed)``: ``noise`` and ``residual`` are ``replicated_embedding_gather_twice(table, ids)``'s
    two outputs (``[*ids.shape, 8192]``), ``normed`` the norm of ``noise``'s rows
    (``[ids.numel(), 8192]``), each bit for bit.
    """

    def __init__(self, norm: nn.Module):
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        self.norm = norm
        self.weight = norm.weight
        self.eps = float(norm.variance_epsilon)
        self.device = self.weight.device
        # PDL like the built-in norm; the kernel waits before its first read of the ids.
        self.pdl = bool(get_env_enable_pdl())
        self._kernel = _compiled(self.device.index, self.eps, self.pdl, True)
        dummy = self.weight.data_ptr()
        with torch.cuda.device(self.device):
            self._kernel(dummy, dummy, dummy, dummy, 0, dummy, dummy, 1, 1, self._stream())
            torch.cuda.synchronize(self.device)
        #: Calls taken (a test counts them).
        self.calls = 0

    def _stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def takes(self, table: torch.Tensor, ids: torch.Tensor) -> bool:
        """A contiguous, aligned ``[V, 8192]`` BF16 table and 1..16 contiguous int64 ids on this device."""
        return (
            table.dim() == 2
            and table.shape[1] == HIDDEN
            and table.dtype == torch.bfloat16
            and table.device == self.device
            and table.is_contiguous()
            and table.data_ptr() % 16 == 0
            and ids.dtype == torch.int64
            and ids.device == self.device
            and ids.is_contiguous()
            and 1 <= ids.numel() <= MAX_ROWS
        )

    def __call__(self, table: torch.Tensor, ids: torch.Tensor):
        rows = ids.numel()
        noise = torch.empty((*ids.shape, HIDDEN), dtype=table.dtype, device=table.device)
        residual = torch.empty_like(noise)
        normed = torch.empty((rows, HIDDEN), dtype=table.dtype, device=table.device)
        self._kernel(
            table.data_ptr(),
            self.weight.data_ptr(),
            normed.data_ptr(),
            ids.data_ptr(),
            table.shape[0],
            noise.data_ptr(),
            residual.data_ptr(),
            rows,
            0,
            self._stream(),
        )
        self.calls += 1
        return noise, residual, normed


def install(draft_model: Optional[nn.Module]) -> Optional[EntryNorm]:
    """Give the drafter its entry-trigger layer-0 input norm. Idempotent; rank-local.

    Raises:
        ValueError: if the drafter's layer-0 input norm is not the plain flashinfer RMSNorm
            (the same on every rank: module classes, dtypes and shapes).
    """
    if draft_model is None:
        raise ValueError("[musespark_open] drafter entry norm: no drafter")
    existing = draft_model.__dict__.get("_ox_entry_norm")
    if isinstance(existing, EntryNorm):
        return existing
    norm = draft_model.model.layers[0].input_layernorm
    reason = norm_reason(norm)
    if reason is not None:
        raise ValueError(
            f"[musespark_open] drafter entry norm (drafter_folds) cannot run here: {reason}"
        )
    entry = EntryNorm(norm)
    draft_model.__dict__["_ox_entry_norm"] = entry
    logger.info(
        "[musespark_open] drafter entry norm: the drafter's layer-0 input norm releases its dependent at entry"
    )
    return entry


def install_gather(draft_model: Optional[nn.Module]) -> Optional[EntryGatherNorm]:
    """Give the drafter the fused noise gather + layer-0 input norm (``EntryGatherNorm``). Idempotent; rank-local.

    The verify fold's drafter_preamble prepare uses it (``_ox_entry_gather``) when the step takes the
    replicated gather; the drafter forward then takes the normalized rows from the preamble.

    Raises:
        ValueError: as ``install``.
    """
    if draft_model is None:
        raise ValueError("[musespark_open] drafter entry gather norm: no drafter")
    existing = draft_model.__dict__.get("_ox_entry_gather")
    if isinstance(existing, EntryGatherNorm):
        return existing
    norm = draft_model.model.layers[0].input_layernorm
    reason = norm_reason(norm)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter entry gather norm cannot run here: {reason}")
    entry = EntryGatherNorm(norm)
    draft_model.__dict__["_ox_entry_gather"] = entry
    logger.info(
        "[musespark_open] drafter entry gather norm: the noise gather, its residual copy and the drafter's layer-0 "
        "input norm run as one kernel"
    )
    return entry
