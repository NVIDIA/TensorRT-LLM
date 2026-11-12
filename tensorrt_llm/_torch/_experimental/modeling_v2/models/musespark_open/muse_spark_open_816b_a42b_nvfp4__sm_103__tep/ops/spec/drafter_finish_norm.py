# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DFlash 2 drafter hop's finish convolution, residual add and RMSNorm in one launch.

After each all-reduced drafter projection (``o_proj``, ``down_proj``) the built-in block decode
runs two kernels on drafter blocks the fused drafter hop does not take (``drafter_hops``: 8 rows;
``wide_drafter_hops``: up to its row bound): the finish-side block convolution
(``ops/spec/dflash2_block_conv.py``, ``rows x 8`` programs) and flashinfer's CuTe-DSL fused
residual add + RMSNorm of the next norm (``FusedAddRMSNormKernel``, which the built-in
``RMSNorm`` reaches on this GPU; in place on both operands). This entry is the two in one
kernel, bit for bit: the fused drafter hop's kernel (``ops/allreduce/drafter_hop.py``, its
thread map, arithmetic and helpers) with the all-reduce transport left out -- rows ``t`` and
``t - 1`` of the already reduced projection are read from global memory.

* convolution: ``dflash2_block_conv``'s statements -- each coefficient ``base + delta``, tap
  product, mask product and running sum rounded to BF16 where it rounds, IEEE FP32 add /
  multiply in between, the masked tap-1 term multiplied by 0.0 (NaN and signed zeros propagate
  as the torch chain's do), row ``t - 1`` read across a block boundary and zero before row 0;
* add + RMSNorm: ``h = conv + r`` in FP32 (not rounded); per flashinfer thread ``f`` of the
  row's 128, one FMA chain ``s = fma(h, h, s)`` from ``+0`` over its 64 columns
  ``8 f + 1024 b + e`` (``b`` then ``e`` ascending); the butterfly over each warp (lane offsets
  1, 2, 4, 8, 16) and over the 4 warp sums padded with zeros;
  ``rstd = rsqrt.approx.ftz(fma(s, 2**-13, eps))``; ``y = (w + 0) * (h * rstd)``; the residual
  becomes ``bf16(h)``. Every floating-point operation is an inline PTX instruction.

Thread map (the hop's): a row is one cluster of 8 CTAs of 128 threads; CTA ``c`` owns
flashinfer threads ``16 c .. 16 c + 15``, its thread ``j`` block ``j mod 8`` of flashinfer
thread ``16 c + j / 8`` (8 columns at ``8 f + 1024 b``); the 16 chain sums of each CTA go to
every CTA of the cluster through distributed shared memory, and every CTA runs both butterflies
on the 128 sums. Every global load of a thread is issued before its first use.

Programmatic dependent launch: the norm weight and the base kernel (model parameters) are read
before ``griddepcontrol.wait``; the projection rows, the coefficients and the residual after
it. The dependents are released at entry (``entry_trigger``) or after the stores.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ..allreduce import drafter_hop as _hop

HIDDEN = _hop.HIDDEN
TAPS = 2
GROUP_SIZE = 16

if _hop.HAS_CUTE_DSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64, Uint32

    from ..allreduce.drafter_hop import (
        CLUSTER,
        OWNER_STRIDE,
        OWNERS,
        ROW_BYTES,
        SMEM_BYTES,
        SUMS_BYTES,
        THREADS,
        VECTOR,
        WARPS,
        _add_rn,
        _bf16_bits_to_f32,
        _float_bits,
        _fma_rn,
        _keep,
        _load_u16,
        _load_x4,
        _load_x4_if,
        _mul_rn,
        _pack_bf16x2,
        _round_bf16,
        _rsqrt_ftz,
        _shfl_bfly,
        _store_x4,
        _unpack_bf16x2,
    )

    class _FinishNormKernel:
        """Finish convolution + flashinfer add + RMSNorm, one row per cluster of 8 CTAs."""

        def __init__(self, eps: float, pdl: bool, entry_trigger: bool):
            self.eps = float(eps)
            self.pdl = pdl
            self.entry_trigger = entry_trigger

        @cute.jit
        def __call__(
            self,
            x: Int64,
            residual: Int64,
            delta: Int64,
            base: Int64,
            weight: Int64,
            output: Int64,
            rows: Int32,
            block: Int32,
            x_row: Int32,
            residual_row: Int32,
            delta_row: Int32,
            delta_tap: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                x,
                residual,
                delta,
                base,
                weight,
                output,
                rows,
                block,
                x_row,
                residual_row,
                delta_row,
                delta_tap,
            ).launch(
                grid=[rows * CLUSTER, 1, 1],
                block=[THREADS, 1, 1],
                cluster=[CLUSTER, 1, 1],
                smem=SMEM_BYTES,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            x: Int64,
            residual: Int64,
            delta: Int64,
            base: Int64,
            weight: Int64,
            output: Int64,
            rows: Int32,
            block: Int32,
            x_row: Int32,
            residual_row: Int32,
            delta_row: Int32,
            delta_tap: Int32,
        ):
            tid = cute.arch.thread_idx()[0]
            lane = tid % 32
            warp = tid // 32
            crank = cute.arch.block_idx()[0] % CLUSTER
            row = cute.arch.block_idx()[0] // CLUSTER
            owner = tid // CLUSTER
            block_b = tid % CLUSTER
            column = (crank * OWNERS + owner) * VECTOR + block_b * (THREADS * VECTOR)
            smem = utils.SmemAllocator()
            h_local = smem.allocate_tensor(
                Float32, cute.make_layout((OWNERS * OWNER_STRIDE,)), byte_alignment=16
            )
            chains = smem.allocate_tensor(Float32, cute.make_layout((OWNERS,)), byte_alignment=16)
            all_chains = smem.allocate_tensor(
                Float32, cute.make_layout((THREADS,)), byte_alignment=16
            )
            sums = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), byte_alignment=16)
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((1,)), byte_alignment=8)
            if tid == 0:
                cute.arch.mbarrier_init(bars.iterator, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator, SUMS_BYTES)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            cute.arch.cluster_arrive_relaxed()

            # Model parameters (written at load time only) before the wait.
            column_bytes = Int64(column) * 2
            w = _load_x4(weight + column_bytes)
            b0 = _load_x4(base + column_bytes)
            b1 = _load_x4(base + Int64(ROW_BYTES) + column_bytes)
            if cutlass.const_expr(self.pdl):
                if cutlass.const_expr(self.entry_trigger):
                    cute.arch.griddepcontrol_launch_dependents()
                cute.arch.griddepcontrol_wait()
            has_prev = Uint32(row != 0)
            x_offset = Int64(row) * Int64(x_row) * 2 + column_bytes
            current_row = _load_x4(x + x_offset)
            previous_row = _load_x4_if(x + x_offset - Int64(x_row) * 2, has_prev)
            residual_offset = Int64(row) * Int64(residual_row) * 2 + column_bytes
            r = _load_x4(residual + residual_offset)
            group = Int64(column // GROUP_SIZE)
            delta_address = delta + (Int64(row) * Int64(delta_row) + group) * 2
            d0 = _bf16_bits_to_f32(_load_u16(delta_address))
            d1 = _bf16_bits_to_f32(_load_u16(delta_address + Int64(delta_tap) * 2))
            # Peers' barriers are initialized before any distributed store below.
            cute.arch.cluster_wait()

            keep = _keep(row % block)
            h = cute.make_rmem_tensor(cute.make_layout((VECTOR,)), Float32)
            for e in cutlass.range_constexpr(VECTOR):
                k = e // 2
                current = _unpack_bf16x2(current_row[k])[e % 2]
                shifted = _unpack_bf16x2(previous_row[k])[e % 2]
                # The finish-side block convolution, rounded where dflash2_block_conv rounds.
                coefficient = _round_bf16(_add_rn(_unpack_bf16x2(b0[k])[e % 2], d0))
                acc = _round_bf16(_mul_rn(coefficient, current))
                coefficient = _round_bf16(_add_rn(_unpack_bf16x2(b1[k])[e % 2], d1))
                term = _round_bf16(_mul_rn(coefficient, shifted))
                term = _round_bf16(_mul_rn(term, keep))
                acc = _round_bf16(_add_rn(acc, term))
                # flashinfer's residual add.
                h[e] = _add_rn(acc, _unpack_bf16x2(r[k])[e % 2])

            # Block ``block_b`` of local flashinfer thread ``owner``: at [owner][block_b].
            for e in cutlass.range_constexpr(VECTOR):
                h_local[owner * OWNER_STRIDE + block_b * VECTOR + e] = h[e]
            cute.arch.barrier()
            # Local flashinfer thread ``tid``'s FMA chain over blocks 0..7, 8 columns each.
            if tid < OWNERS:
                chain = Float32(0.0)
                for k in cutlass.range_constexpr(CLUSTER * VECTOR):
                    value = h_local[tid * OWNER_STRIDE + k]
                    chain = _fma_rn(value, value, chain)
                chains[tid] = chain
            cute.arch.barrier()
            # The 16 sums of flashinfer threads 16 crank .. 16 crank + 15, to every CTA of the
            # row's cluster (this one included), 4 per store.
            if tid < OWNERS // 4:
                target = all_chains.iterator + crank * OWNERS + tid * 4
                for peer in cutlass.range_constexpr(CLUSTER):
                    cute.arch.store_async_dsmem(
                        target.align(16),
                        (
                            _float_bits(chains[4 * tid]),
                            _float_bits(chains[4 * tid + 1]),
                            _float_bits(chains[4 * tid + 2]),
                            _float_bits(chains[4 * tid + 3]),
                        ),
                        bars.iterator,
                        peer,
                    )
            cute.arch.mbarrier_wait(bars.iterator, 0)
            # Lane ``l`` of warp ``w`` holds flashinfer thread ``32 w + l``'s sum.
            sum_sq = all_chains[tid]
            for i in cutlass.range_constexpr(5):
                sum_sq = _add_rn(sum_sq, _shfl_bfly(sum_sq, Uint32(1 << i)))
            if lane == 0:
                sums[warp] = sum_sq
            cute.arch.barrier()
            row_sum = Float32(0.0)
            if lane < WARPS:
                row_sum = sums[lane]
            for i in cutlass.range_constexpr(5):
                row_sum = _add_rn(row_sum, _shfl_bfly(row_sum, Uint32(1 << i)))
            rms_rcp = _rsqrt_ftz(_fma_rn(row_sum, Float32(1.0 / HIDDEN), Float32(self.eps)))

            updated = []
            normed = []
            for k in cutlass.range_constexpr(4):
                lo_w, hi_w = _unpack_bf16x2(w[k])
                lo = _mul_rn(_add_rn(lo_w, Float32(0.0)), _mul_rn(h[2 * k], rms_rcp))
                hi = _mul_rn(_add_rn(hi_w, Float32(0.0)), _mul_rn(h[2 * k + 1], rms_rcp))
                normed.append(_pack_bf16x2(lo, hi))
                updated.append(_pack_bf16x2(h[2 * k], h[2 * k + 1]))
            _store_x4(residual + residual_offset, *updated)
            _store_x4(output + Int64(row) * (HIDDEN * 2) + column_bytes, *normed)
            if cutlass.const_expr(self.pdl and not self.entry_trigger):
                cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=64)
    def _compiled_kernel(device_index: int, eps: float, pdl: bool, entry_trigger: bool):
        kernel = _FinishNormKernel(eps, pdl, entry_trigger)
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *([Int64(0)] * 6),
                Int32(1),
                Int32(8),
                Int32(HIDDEN),
                Int32(HIDDEN),
                Int32(2 * TAPS * HIDDEN // GROUP_SIZE),
                Int32(HIDDEN // GROUP_SIZE),
                stream,
                options="--enable-tvm-ffi",
            )


def takes(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether this entry computes these operands (else the caller runs the built-in two)."""
    rows = x.shape[0] if x.dim() == 2 else -1
    return (
        _hop.HAS_CUTE_DSL
        and rows >= 1
        and all(t.dtype == torch.bfloat16 and t.is_cuda for t in (x, delta, base, residual, weight))
        and all(t.device == x.device for t in (delta, base, residual, weight))
        and tuple(x.shape) == (rows, HIDDEN)
        and tuple(residual.shape) == (rows, HIDDEN)
        and tuple(delta.shape) == (rows, TAPS, HIDDEN // GROUP_SIZE)
        and tuple(base.shape) == (TAPS, HIDDEN)
        and tuple(weight.shape) == (HIDDEN,)
        and x.stride(1) == 1
        and residual.stride(1) == 1
        and delta.stride(2) == 1
        and base.is_contiguous()
        and weight.is_contiguous()
        and all(t.data_ptr() % 16 == 0 for t in (base, weight))
        and all(t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0 for t in (x, residual))
        and not torch.is_grad_enabled()
    )


class DrafterFinishNorm(OpWrapper):
    """``y, residual <- rmsnorm(conv(x) + residual)``: the drafter hop without its all-reduce.

    Arguments: ``x`` the all-reduced projection output ``[rows, 8192]``; ``delta`` the finish
    coefficients ``[rows, 2, 512]`` (``coefficients[:, 1]`` of the convolution's ``prepare``,
    unit group stride); ``base`` the finish base kernel ``[2, 8192]`` (a parameter: read before
    the dependency wait); ``residual`` ``[rows, 8192]``, updated in place to
    ``bf16(conv(x) + residual)``; ``weight`` / ``eps`` the next RMSNorm's (a parameter);
    ``block_size`` rows per request; ``pdl`` launch with programmatic dependent launch;
    ``entry_trigger`` release the dependents at entry instead of after the stores. Returns the
    normed rows, a fresh tensor.

    Fusion boundary: the convolution and the add + norm; the all-reduce and the next GEMM stay
    outside.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: none. One CuTe-DSL kernel, compiled per (device, eps, pdl, trigger):
    grid `rows x 8` CTAs of 128 threads in clusters of 8 (one row per cluster), the fused
    drafter hop's thread map without its transport.

    Call site: `drafter_block_decode/block_decode.py` `windowed_block_decode`, both hops of
    every drafter layer, on blocks the fused drafter hop does not take, when
    `TRTLLM_MUSESPARK_DRAFTER_FINISH_NORM=1` (`FinishNorm`).

    Numerics: bitwise the built-in `dflash2_block_conv` then flashinfer's CuTe-DSL fused add +
    RMSNorm for a 8192-wide BF16 row (the fused drafter hop's arithmetic). `reference` is
    float64 between the rounding points.

    Hard limits
      hidden 8192, two taps over groups of 16, BF16, 16-byte aligned rows (`takes`).
    """

    def __call__(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        pdl: bool,
        entry_trigger: bool = False,
    ) -> torch.Tensor:
        rows = x.shape[0]
        out = torch.empty((rows, HIDDEN), dtype=x.dtype, device=x.device)
        if rows == 0:
            return out
        compiled = _compiled_kernel(x.get_device(), float(eps), bool(pdl), bool(entry_trigger))
        compiled(
            x.data_ptr(),
            residual.data_ptr(),
            delta.data_ptr(),
            base.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            rows,
            block_size,
            x.stride(0),
            residual.stride(0),
            delta.stride(0),
            delta.stride(1),
            cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
        )
        return out

    def reference(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        pdl: bool,
        entry_trigger: bool = False,
    ) -> torch.Tensor:
        """The rounding points, float64 in between; updates ``residual`` as the op does.

        ``pdl`` and ``entry_trigger`` change how the op launches, not what it computes; ignored.
        """
        rows = x.shape[0]
        xf = x.float()
        coef = (base.float().unsqueeze(0) + delta.float().repeat_interleave(GROUP_SIZE, -1)).to(
            torch.bfloat16
        )
        position = torch.arange(rows, device=x.device) % block_size
        acc = (coef[:, 0].float() * xf).to(torch.bfloat16).float()
        shifted = torch.zeros_like(xf)
        shifted[1:] = xf[:-1]
        term = (coef[:, 1].float() * shifted).to(torch.bfloat16).float()
        term = term * (position >= 1).float().unsqueeze(-1)
        acc = (acc + term).to(torch.bfloat16).double()
        s = acc + residual.double()
        normed = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + eps) * weight.double()
        residual.copy_(s.to(torch.bfloat16))
        return normed.to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        pdl: bool,
        entry_trigger: bool = False,
    ) -> None:
        assert takes(x, delta, base, residual, weight), (
            "drafter_finish_norm takes BF16 x / residual [rows, 8192] (16-byte aligned rows), "
            "delta [rows, 2, 512] with unit group stride, contiguous base [2, 8192] and weight "
            f"[8192]; got x {tuple(x.shape)} {x.dtype}, delta {tuple(delta.shape)} "
            f"{delta.stride()}, base {tuple(base.shape)}, residual {tuple(residual.shape)}, "
            f"weight {tuple(weight.shape)}"
        )
        assert block_size >= 1, f"block_size must be >= 1, got {block_size}"


drafter_finish_norm = DrafterFinishNorm()
