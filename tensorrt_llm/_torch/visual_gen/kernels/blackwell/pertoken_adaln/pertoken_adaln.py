# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Fused per-token AdaLN modulate for WAN (temb.ndim == 4 path).
#
# Provenance: the row-per-CTA scaffold (two-pass mean/var warp+CTA reduce,
# autovec 128-bit copies, symbolic-stride compile cache) is derived from
# SGLang's CuTe DSL ScaleResidualNormScaleShift kernel (Apache-2.0):
#   upstream repo: https://github.com/sgl-project/sglang
#   upstream path: python/sglang/kernels/ops/diffusion/cutedsl/
#                  scale_residual_norm_scale_shift.py (+ common/norm_fusion.py)
#   pinned SHA:    e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7
# reduce.py in this package is a vendored copy (see its header).
#
# WHY THIS KERNEL EXISTS (delta vs the SGLang op): the SGLang op's contract
# takes MATERIALIZED fp32 scale/shift tensors. At the WAN per-token site
# (temb.ndim == 4, e.g. [B=2, S=27280, 6, D=3072]) that forces writing and
# re-reading two to three extra [B, S, D] fp32 tensors (~670 MB each) per
# site: ~3x DRAM traffic. This kernel instead takes the RAW bf16 temb chunk
# views (shift/scale are DIFFERENT chunks of the [B, S, 6, D] temb: strides
# (S*6D, 6D, 1), stride(-1) == 1) plus the [D] fp32 rows of
# scale_shift_table, and fuses the fp32 table+chunk add INLINE in registers,
# so its DRAM traffic equals the analytic floor: read x + 2 bf16 chunks,
# write y (~1.34 GB at the production shape).
#
# Math (matches the eager WAN block exactly: fp32 accumulation, ONE final
# rounding at the store):
#   mod_shift = table_shift + shift_chunk.float()
#   mod_scale = table_scale + scale_chunk.float()
#   y = LayerNorm_noaffine(x.float()) * (1 + mod_scale) + mod_shift  -> x.dtype
#
# Residual variant (norm2/norm3 sites):
#   v       = x_in.float() * (table_gate + gate_chunk.float())    [if gated]
#   res_out = (residual.float() + v) -> x.dtype                   [stored]
#   ln_in   = res_out (post-rounding, matching the eager intermediate
#             .to(x.dtype) before the norm)
#   y = LN(ln_in.float()) [* weight + bias] [* (1+mod_scale) + mod_shift]
#       -> x.dtype
#
# torch.library custom ops (functional, register_fake):
#   trtllm::fused_pertoken_adaln
#   trtllm::fused_pertoken_adaln_residual
# Both are tagged torch.Tag.needs_fixed_stride_order: inside compiled
# regions inductor otherwise passes arbitrary-strided buffers to opaque
# custom ops, which eager testing never exercises.

from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from .reduce import cta_reduce_sum, warp_reduce_sum

WARP_SIZE = 32

TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_COMPILE_CACHE = {}

_REQUIRED_ALIGNMENT = 32

# Elements per thread. 8 -> one LDG.128 per bf16 operand per thread,
# num_warps = D/256 (12 warps at D=3072). 16 halves the CTA thread count
# (more ILP, shorter reduce tree) at double the registers; it measures
# within noise of 8 at the production shape, and vec=16 crashes the pinned
# cutlass-dsl 4.5.0 compiler for the residual config (gate + fp32 [D]
# weight/bias operands) — so the residual op hard-validates vec == 8, while
# the plain op accepts 8 or 16 (both compile).
DEFAULT_VEC = 8

# CuTe DSL specialization requires the tensor-or-scalar-sentinel operands in
# ``PerTokenAdaLN`` and its nested helpers to remain unannotated. Python union
# annotations are not valid CuTe argument types in the pinned cutlass-dsl 4.5.0;
# the decorator derives their concrete MLIR types from each compiled call.


def _to_fake_cute_arg(t: torch.Tensor | int | float) -> object:
    """Symbolic shapes/strides everywhere except the last dim (compile-time D)
    so one compilation covers any B/S and any D-divisible outer strides —
    including the temb chunk views with strides (S*6*D, 6*D, 1)."""
    if isinstance(t, torch.Tensor):
        D = t.shape[-1]
        dtype = TORCH_TO_CUTE_DTYPE[t.dtype]
        shape = (*(cute.sym_int() for _ in range(t.ndim - 1)), D)
        stride = (*(cute.sym_int(divisibility=D) for _ in range(t.ndim - 1)), 1)
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride, memspace=cute.AddressSpace.gmem, assumed_align=32
        )
    if isinstance(t, int):
        return cutlass.Int32(t)
    if isinstance(t, float):
        return cutlass.Float32(t)
    return t


@cute.jit
def _ln_stats_cta(
    num_warps: cutlass.Constexpr,
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    D: Union[cutlass.Int32, cutlass.Constexpr],
    eps: cutlass.Float32,
):
    """Two-pass LayerNorm statistics over a per-thread fragment (fp32
    accumulation; same mean-then-var order as torch). Returns (mean, rstd)
    as fp32 scalars — the caller applies them in fp32 so the result is
    rounded exactly ONCE at the final store."""
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        val += tXrX[idx].to(cutlass.Float32)
    val = warp_reduce_sum(val)
    val = cta_reduce_sum(val, num_warps, tidx)
    mean = val / D
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        d = tXrX[idx].to(cutlass.Float32) - mean
        val += d * d
    val = warp_reduce_sum(val)
    val = cta_reduce_sum(val, num_warps, tidx)
    rstd = cute.rsqrt(val / D + eps)
    return mean, rstd


class PerTokenAdaLN:
    """Row-per-CTA fused (gate/residual) + LayerNorm + per-token AdaLN
    modulate with the fp32 table+chunk add computed inline in registers.

    Grid: one CTA per (batch, seq) token row; block: num_warps * 32 threads;
    each thread owns `vec` consecutive elements of the D row (128-bit
    vectorized gmem copies via cute.autovec_copy). ALL gmem loads are issued
    before the reduction so chunk/table loads overlap the two CTA reduces.
    """

    @classmethod
    def make_hash_key(cls, *inputs: torch.Tensor | int | float) -> tuple[object, ...]:
        def _sig(val: torch.Tensor | int | float) -> object:
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(self, D: int, vec: int = DEFAULT_VEC) -> None:
        assert D % (WARP_SIZE * vec) == 0, f"D={D} must be a multiple of {WARP_SIZE * vec}"
        self.D = D
        self.vec = vec
        self.num_warps = D // (WARP_SIZE * vec)
        # cta_reduce_sum gathers the per-warp partials with a SINGLE warp
        # (one lane per partial), so > WARP_SIZE warps would silently drop
        # partials. The dispatch guards (_check_d: D <= 8192, vec in (8, 16))
        # already imply this; assert it here so a direct construction that
        # relaxes either bound fails loudly instead of corrupting the reduce.
        assert self.num_warps <= WARP_SIZE, (
            f"num_warps={self.num_warps} exceeds {WARP_SIZE}: cta_reduce_sum "
            f"reduces at most one partial per lane of warp 0 (D={D}, vec={vec})"
        )
        self.num_threads = self.num_warps * WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mY,
        mResOut,
        mRes,
        mX,
        mGateC,
        mTGate,
        mWeight,
        mBias,
        mShiftC,
        mScaleC,
        mTShift,
        mTScale,
        eps: cutlass.Float32 = cutlass.Float32(1e-6),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        B, S, _ = mX.shape
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        t_layout = cute.make_layout(self.num_threads)
        v_layout = cute.make_layout(self.vec)
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)

        self.kernel(
            mY,
            mResOut,
            mRes,
            mX,
            mGateC,
            mTGate,
            mWeight,
            mBias,
            mShiftC,
            mScaleC,
            mTShift,
            mTScale,
            tiled_copy,
            eps,
        ).launch(
            grid=[B * S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mY,
        mResOut,
        mRes,
        mX,
        mGateC,
        mTGate,
        mWeight,
        mBias,
        mShiftC,
        mScaleC,
        mTShift,
        mTScale,
        tiled_copy: cute.TiledCopy,
        eps: cutlass.Float32,
    ):
        _, S, _ = mX.shape
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bidx = cutlass.Int32(bid // S)  # batch index
        bidy = cutlass.Int32(bid % S)  # seq index
        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def slice_if(mV):
            """[B,S,D] (any D-divisible outer strides) or [D] -> per-thread
            gmem tile + rmem fragment. local_tile preserves base-pointer
            alignment for vectorized loads."""
            if cutlass.const_expr(isinstance(mV, cute.Tensor)):
                if cutlass.const_expr(len(mV.shape) == 1):
                    gV = mV
                else:
                    gV = cute.local_tile(mV, tiler=(1, 1, self.D), coord=(bidx, bidy, 0))
                    gV = gV[0, 0, None]
                tVgV = thr_copy.partition_S(gV)
                tVrV = cute.make_fragment_like(tVgV, tVgV.element_type)
                return tVgV, tVrV
            return mV, mV

        @cute.jit
        def copy_if(src, dst):
            if cutlass.const_expr(isinstance(src, cute.Tensor) and isinstance(dst, cute.Tensor)):
                cute.autovec_copy(src, dst)

        tXgX, tXrX = slice_if(mX)
        tRgR, tRrR = slice_if(mRes)
        tGgG, tGrG = slice_if(mGateC)
        tTGgTG, tTGrTG = slice_if(mTGate)
        tWgW, tWrW = slice_if(mWeight)
        tBgB, tBrB = slice_if(mBias)
        tSHgSH, tSHrSH = slice_if(mShiftC)
        tSCgSC, tSCrSC = slice_if(mScaleC)
        tTSHgTSH, tTSHrTSH = slice_if(mTShift)
        tTSCgTSC, tTSCrTSC = slice_if(mTScale)
        tROgRO, tROrRO = slice_if(mResOut)
        tYgY, tYrY = slice_if(mY)

        # Issue ALL gmem loads up front: the bf16 chunk + fp32 table loads
        # overlap the mean/var CTA reductions instead of serializing after
        # them.
        copy_if(tXgX, tXrX)
        copy_if(tRgR, tRrR)
        copy_if(tGgG, tGrG)
        copy_if(tTGgTG, tTGrTG)
        copy_if(tWgW, tWrW)
        copy_if(tBgB, tBrB)
        copy_if(tSHgSH, tSHrSH)
        copy_if(tSCgSC, tSCrSC)
        copy_if(tTSHgTSH, tTSHrTSH)
        copy_if(tTSCgTSC, tTSCrTSC)

        # Optional gate + residual accumulation, fp32, then ONE rounding to
        # the residual dtype (matches the eager `.to(x.dtype)` before the
        # norm).
        if cutlass.const_expr(isinstance(tRrR, cute.Tensor)):
            v = tXrX.load().to(cutlass.Float32)
            if cutlass.const_expr(isinstance(tGrG, cute.Tensor)):
                # gate row = table_gate (fp32) + gate_chunk (bf16->fp32), inline
                g = tGrG.load().to(cutlass.Float32)
                if cutlass.const_expr(isinstance(tTGrTG, cute.Tensor)):
                    g = tTGrTG.load() + g
                v = v * g
            v = tRrR.load().to(cutlass.Float32) + v
            tROrRO.store(v.to(tROrRO.element_type))
            copy_if(tROrRO, tROgRO)
            ln_src = tROrRO  # LN reads the ROUNDED residual (eager parity)
        else:
            ln_src = tXrX

        mean, rstd = _ln_stats_cta(self.num_warps, tidx, ln_src, self.D, eps)

        # Normalize + affine + per-token modulate, all in fp32 registers.
        n = (ln_src.load().to(cutlass.Float32) - mean) * rstd
        if cutlass.const_expr(isinstance(tWrW, cute.Tensor)):
            n = n * tWrW.load().to(cutlass.Float32)
        if cutlass.const_expr(isinstance(tBrB, cute.Tensor)):
            n = n + tBrB.load().to(cutlass.Float32)
        if cutlass.const_expr(isinstance(tSCrSC, cute.Tensor)):
            mod_scale = tSCrSC.load().to(cutlass.Float32)
            if cutlass.const_expr(isinstance(tTSCrTSC, cute.Tensor)):
                mod_scale = tTSCrTSC.load() + mod_scale
            n = n * (1.0 + mod_scale)
        if cutlass.const_expr(isinstance(tSHrSH, cute.Tensor)):
            mod_shift = tSHrSH.load().to(cutlass.Float32)
            if cutlass.const_expr(isinstance(tTSHrTSH, cute.Tensor)):
                mod_shift = tTSHrTSH.load() + mod_shift
            n = n + mod_shift
        # ONE rounding, at the final store.
        tYrY.store(n.to(tYrY.element_type))
        copy_if(tYrY, tYgY)


# ---------------------------------------------------------------------------
# Validation + dispatch
# ---------------------------------------------------------------------------
_ROW_DTYPES = (torch.bfloat16,)


def _validate_bsd(
    t: torch.Tensor,
    B: int,
    S: int,
    D: int,
    name: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    if t.dtype not in _ROW_DTYPES:
        raise ValueError(f"{name}: unsupported dtype {t.dtype}")
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {t.dtype}")
    if not t.is_cuda:
        raise ValueError(f"{name}: expected a CUDA tensor, got device {t.device}")
    if device is not None and t.device != device:
        raise ValueError(f"{name}: expected device {device}, got {t.device}")
    if t.shape != (B, S, D):
        raise ValueError(f"{name}: expected shape {(B, S, D)}, got {tuple(t.shape)}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name}: last dim must be contiguous (stride(-1)==1)")
    for i in range(t.ndim - 1):
        if t.stride(i) % D != 0:
            raise ValueError(f"{name}: stride({i})={t.stride(i)} not divisible by D={D}")
    if t.data_ptr() % _REQUIRED_ALIGNMENT != 0:
        raise ValueError(f"{name}: data pointer must be {_REQUIRED_ALIGNMENT}-byte aligned")


def _validate_row(t: Optional[torch.Tensor], D: int, name: str, device: torch.device) -> None:
    if t is None:
        return
    if t.dtype != torch.float32:
        raise ValueError(f"{name}: expected fp32, got {t.dtype}")
    if not t.is_cuda or t.device != device:
        raise ValueError(f"{name}: expected device {device}, got {t.device}")
    if t.shape != (D,):
        raise ValueError(f"{name}: expected shape ({D},), got {tuple(t.shape)}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name}: must be contiguous")
    if t.data_ptr() % _REQUIRED_ALIGNMENT != 0:
        raise ValueError(f"{name}: data pointer must be {_REQUIRED_ALIGNMENT}-byte aligned")


def _check_d(D: int, vec: int) -> None:
    if vec not in (8, 16):
        raise ValueError(f"vec={vec} not supported: expected 8 or 16")
    if D <= 0 or D % 256 != 0 or D > 8192:
        raise ValueError(f"D={D} not supported: must be a multiple of 256 and <= 8192")
    if D % (WARP_SIZE * vec) != 0:
        raise ValueError(f"D={D} not divisible by {WARP_SIZE * vec} (vec={vec})")


def _launch(
    torch_tensors: list[torch.Tensor | int | float],
    D: int,
    vec: int,
    eps: float,
    device: torch.device,
) -> None:
    with torch.cuda.device(device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        hash_key = (device, vec, *PerTokenAdaLN.make_hash_key(*torch_tensors))
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            if torch.cuda.get_device_capability(device) != (10, 0):
                raise ValueError("fused per-token AdaLN requires an SM100 GPU")
            kernel = PerTokenAdaLN(D, vec)
            fake_sig_args = [_to_fake_cute_arg(t) for t in torch_tensors]
            compiled_fn = cute.compile(kernel, *fake_sig_args, options="--enable-tvm-ffi")
            _COMPILE_CACHE[hash_key] = compiled_fn
        compiled_fn(*torch_tensors, eps, stream)


@torch.library.custom_op(
    "trtllm::fused_pertoken_adaln",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def fused_pertoken_adaln(
    x: torch.Tensor,
    shift_chunk: torch.Tensor,
    scale_chunk: torch.Tensor,
    table_shift: torch.Tensor,
    table_scale: torch.Tensor,
    eps: float = 1e-6,
    vec: int = DEFAULT_VEC,
) -> torch.Tensor:
    """y = LN_noaffine(x.float()) * (1 + (table_scale + scale_chunk.float()))
           + (table_shift + shift_chunk.float())  -> x.dtype

    Expects:
      - x:                        [B, S, D] bf16, stride(-1)==1
      - shift_chunk, scale_chunk: [B, S, D] views (e.g. temb[:, :, i, :] chunk
                                  views of a [B, S, 6, D] temb — strides
                                  (S*6*D, 6*D, 1)); stride(-1)==1, outer
                                  strides divisible by D
      - table_shift, table_scale: [D] fp32 contiguous rows of scale_shift_table
      - D: multiple of 256, <= 8192 (unpredicated LDG.128 vectorized loads)
    """
    if x.ndim != 3:
        raise ValueError(f"x: expected a 3D [B, S, D] tensor, got {x.ndim}D")
    B, S, D = x.shape
    if B == 0 or S == 0:
        raise ValueError(f"x: B and S must be nonzero, got shape {tuple(x.shape)}")
    _check_d(D, vec)
    _validate_bsd(x, B, S, D, "x")
    _validate_bsd(shift_chunk, B, S, D, "shift_chunk", x.dtype, x.device)
    _validate_bsd(scale_chunk, B, S, D, "scale_chunk", x.dtype, x.device)
    _validate_row(table_shift, D, "table_shift", x.device)
    _validate_row(table_scale, D, "table_scale", x.device)
    y = torch.empty_like(x)
    # Scalar placeholders for absent operands (CuTe DSL TVM-FFI backend does
    # not accept None); they generate no code (const_expr isinstance checks).
    torch_tensors = [y, 0, 0, x, 1, 1, 1, 0, shift_chunk, scale_chunk, table_shift, table_scale]
    _launch(torch_tensors, D, vec, eps, x.device)
    return y


@fused_pertoken_adaln.register_fake
def _fused_pertoken_adaln_fake(
    x: torch.Tensor,
    shift_chunk: torch.Tensor,
    scale_chunk: torch.Tensor,
    table_shift: torch.Tensor,
    table_scale: torch.Tensor,
    eps: float = 1e-6,
    vec: int = DEFAULT_VEC,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "trtllm::fused_pertoken_adaln_residual",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def fused_pertoken_adaln_residual(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate_chunk: Optional[torch.Tensor],
    table_gate: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    shift_chunk: Optional[torch.Tensor],
    scale_chunk: Optional[torch.Tensor],
    table_shift: Optional[torch.Tensor],
    table_scale: Optional[torch.Tensor],
    eps: float = 1e-6,
    vec: int = DEFAULT_VEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """residual_out = (residual.float()
                       + x.float() * (table_gate + gate_chunk.float())) -> dtype
       y = LN(residual_out.float()) [* weight + bias]
             [* (1 + (table_scale + scale_chunk.float()))
              + (table_shift + shift_chunk.float())]      -> dtype

    Covers the WAN per-token sites:
      norm2: gate_chunk+table_gate present, weight/bias [D] fp32 present,
             shift/scale absent (plain affine LN).
      norm3: gate absent (residual_out = residual + x), weight/bias absent,
             shift/scale chunks + table rows present.
    LN reads the ROUNDED residual_out (bit-matching the eager intermediate
    `.to(x.dtype)`), then normalizes/modulates in fp32 with one final
    rounding.
    """
    if x.ndim != 3:
        raise ValueError(f"x: expected a 3D [B, S, D] tensor, got {x.ndim}D")
    B, S, D = x.shape
    if B == 0 or S == 0:
        raise ValueError(f"x: B and S must be nonzero, got shape {tuple(x.shape)}")
    if vec != 8:
        # cutlass-dsl 4.5.0 (requirements.txt pin) crashes compiling the
        # residual configs at vec=16; vec=8 is also the measured-best at the
        # production shape.
        raise ValueError("fused_pertoken_adaln_residual supports vec=8 only")
    _check_d(D, vec)
    _validate_bsd(x, B, S, D, "x")
    _validate_bsd(residual, B, S, D, "residual", x.dtype, x.device)
    if (gate_chunk is None) != (table_gate is None):
        raise ValueError("gate_chunk and table_gate must be passed together")
    mod_operands = (shift_chunk, scale_chunk, table_shift, table_scale)
    if any(t is None for t in mod_operands) and not all(t is None for t in mod_operands):
        raise ValueError("shift_chunk/scale_chunk/table_shift/table_scale must be passed together")
    if gate_chunk is not None:
        _validate_bsd(gate_chunk, B, S, D, "gate_chunk", x.dtype, x.device)
        _validate_row(table_gate, D, "table_gate", x.device)
    _validate_row(weight, D, "weight", x.device)
    _validate_row(bias, D, "bias", x.device)
    if shift_chunk is not None:
        _validate_bsd(shift_chunk, B, S, D, "shift_chunk", x.dtype, x.device)
        _validate_bsd(scale_chunk, B, S, D, "scale_chunk", x.dtype, x.device)
        _validate_row(table_shift, D, "table_shift", x.device)
        _validate_row(table_scale, D, "table_scale", x.device)
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    torch_tensors = [
        y,
        residual_out,
        residual,
        x,
        1 if gate_chunk is None else gate_chunk,
        1 if table_gate is None else table_gate,
        1 if weight is None else weight,
        0 if bias is None else bias,
        0 if shift_chunk is None else shift_chunk,
        0 if scale_chunk is None else scale_chunk,
        0 if table_shift is None else table_shift,
        0 if table_scale is None else table_scale,
    ]
    _launch(torch_tensors, D, vec, eps, x.device)
    return y, residual_out


@fused_pertoken_adaln_residual.register_fake
def _fused_pertoken_adaln_residual_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate_chunk: Optional[torch.Tensor],
    table_gate: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    shift_chunk: Optional[torch.Tensor],
    scale_chunk: Optional[torch.Tensor],
    table_shift: Optional[torch.Tensor],
    table_scale: Optional[torch.Tensor],
    eps: float = 1e-6,
    vec: int = DEFAULT_VEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)
