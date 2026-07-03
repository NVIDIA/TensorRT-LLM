# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous epilogue for the fused fc1+fc2 swap-AB MegaMoE kernel.

Per-thread RMEM tensors flow between the transpose / SwiGLU / quantize / fc2
store steps as bare ``cute.Tensor`` fragments; their thread distribution is a
fixed physical property of the surrounding atom sequence and is documented in
local comments.  A ``Contract`` is only kept where it earns its keep: the fc2
store-out mapping (``Fc2ProcessPipeline.store_out_mapping``) is evaluated at
runtime to drive which token/hidden cell each issue targets and therefore how
the per-token metadata is fetched.
"""

import dataclasses
from typing import (Any, Callable, ClassVar, List, Literal, Optional, Tuple,
                    Type, Union)

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64, T

from .common_moe_utils import cvt_f32_to_f8_to_f32
from .contract import Contract, FunctionMapping, Space, eval_function_mapping
from .fc1_fc2_fuse_sched import BlockPhase
from .flag_batch import GpuReleaseFlagBatchTracker
from .iket_compat import iket
from .megamoe_constants import (Fp8E4M3RcpLimit, Fp8E5M2RcpLimit, Fp32Max,
                                Nvfp4BlockSize, Nvfp4E2M1RcpLimit)
from .moe_persistent_scheduler import (MoESchedConsumer, MoESchedExtension,
                                       MoEWorkTileInfo)
from .ptx_helpers import cp_async_bulk_s2g as _cp_async_bulk_s2g
from .ptx_helpers import \
    cp_reduce_async_bulk_add_noftz_bf16_s2g as \
    _cp_reduce_async_bulk_add_noftz_bf16_s2g
from .ptx_helpers import \
    red_add_relaxed_sys_v2_bf16x2 as _red_add_relaxed_sys_v2_bf16x2
from .sym_buffer import SymBufferDeviceBase
from .token_comm import CombineFormat, TokenCommArgs, TokenSrcMetadata


@dataclasses.dataclass(frozen=True)
class QuantImpl:
    """Register-level block quantizer shared by the fc1 / fc2 epilogues.

    Returns ``(data_regs, sf_regs)`` ONLY: the caller pre-multiplies the topk
    weight / global scale into ``prequant_reg`` beforehand and owns the data /
    sf plane stores afterwards. ``sf_vec_direction`` selects the per-block amax
    reduction:

      * ``regs_in_thread``            -- a block is ``sf_vec`` contiguous regs of
        one thread; amax is thread-local (packed bf16x2 abs-max for combine,
        fp32 ``fmax`` for orthodox).
      * ``threads_with_the_same_reg`` -- a block is one reg across ``sf_vec`` warp
        lanes; amax is a warp CREDUX (full warp for vec=32, lane-predicated
        halves for vec=16) and is fp32-only, so bf16 is upconverted first.

    ``prequant_reg`` must be 1D with size divisible by ``sf_vec``. Combine inputs
    are bf16 (fc2's bf16 reorder regs); orthodox nvfp4 input is fp32 (swiglu).
    """

    quant_kind: Union[str, CombineFormat]
    sf_vec_direction: Literal["regs_in_thread", "threads_with_the_same_reg"]
    lane_idx: Optional[Any] = None  # Int32; only the across-lane path needs it

    _orthodox_kinds: ClassVar[Tuple[str, ...]] = ("nvfp4", "mxfp4",
                                                  "mxfp8_e4m3", "mxfp8_e5m2")

    # -- config / validation --------------------------------------------------

    def __post_init__(self):
        if isinstance(self.quant_kind, CombineFormat):
            if not self.quant_kind.is_quantized:
                raise ValueError(
                    f"QuantImpl combine path needs a quantized CombineFormat, "
                    f"got {self.quant_kind}.")
        elif self.quant_kind in self._orthodox_kinds:
            if self.quant_kind != "nvfp4":
                raise NotImplementedError(
                    f"orthodox {self.quant_kind} quant is reserved but not wired "
                    f"(only orthodox 'nvfp4' is implemented).")
        else:
            raise ValueError(
                f"quant_kind must be a CombineFormat or one of "
                f"{self._orthodox_kinds}, got {self.quant_kind!r}.")
        if self.sf_vec_direction not in ("regs_in_thread",
                                         "threads_with_the_same_reg"):
            raise ValueError(
                f"sf_vec_direction must be 'regs_in_thread' or "
                f"'threads_with_the_same_reg', got {self.sf_vec_direction!r}.")
        # Orthodox nvfp4 only sees the transposed (regs-in-thread) layout fc1
        # produces via its TMEM transpose.
        if not isinstance(self.quant_kind, CombineFormat) and (
                self.sf_vec_direction != "regs_in_thread"):
            raise NotImplementedError(
                "orthodox nvfp4 quant is regs_in_thread only.")
        if self.sf_vec_direction == "threads_with_the_same_reg" and (
                self.lane_idx is None):
            raise ValueError(
                "across-lane quant needs lane_idx for the CREDUX half-warp predicate."
            )

    @property
    def data_dtype(self):
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.act_dtype
        return cutlass.Float4E2M1FN  # orthodox nvfp4

    @property
    def scale_dtype(self):
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.scale_dtype
        return cutlass.Float8E4M3FN  # orthodox nvfp4 e4m3 sfc

    @property
    def sf_vec_size(self) -> int:
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.scale_block
        return Nvfp4BlockSize  # 16

    @property
    def _is_combine(self) -> bool:
        return isinstance(self.quant_kind, CombineFormat)

    @property
    def _data_rcp_limit(self) -> float:
        # 1 / max representable magnitude of the data element type.
        dt = self.data_dtype
        if dt is cutlass.Float4E2M1FN:
            return Nvfp4E2M1RcpLimit  # 1/6
        if dt is cutlass.Float8E4M3FN:
            return Fp8E4M3RcpLimit  # 1/448
        return Fp8E5M2RcpLimit  # 1/57344

    # -- dispatch -------------------------------------------------------------

    @cute.jit
    def __call__(self, prequant_reg: cute.Tensor, *, norm_const=None):
        if cutlass.const_expr(cute.size(prequant_reg) % self.sf_vec_size != 0):
            raise ValueError("prequant_reg size must be divisible by sf_vec.")
        # Combine quantizes fc2's bf16 reorder regs; orthodox nvfp4 the fp32 swiglu.
        expected_in = cutlass.BFloat16 if self._is_combine else cutlass.Float32
        if cutlass.const_expr(prequant_reg.element_type is not expected_in):
            raise TypeError(
                f"QuantImpl({self.quant_kind}) expects {expected_in} prequant "
                f"input, got {prequant_reg.element_type}.")
        if cutlass.const_expr(not self._is_combine):
            return self.nvfp4_quant_impl(prequant_reg, norm_const=norm_const)
        if cutlass.const_expr(self.data_dtype is cutlass.Float4E2M1FN):
            if cutlass.const_expr(self.sf_vec_direction == "regs_in_thread"):
                return self.nvfp4_combine_quant_regs_in_thread_impl(
                    prequant_reg)
            return self.nvfp4_combine_quant_threads_with_the_same_reg_impl(
                prequant_reg)
        if cutlass.const_expr(self.sf_vec_direction == "regs_in_thread"):
            return self.mxfp8_combine_quant_regs_in_thread_impl(prequant_reg)
        return self.mxfp8_combine_quant_threads_with_the_same_reg_impl(
            prequant_reg)

    # -- impls ----------------------------------------------------------------

    # regs_in_thread only; fc1 promises the vec direction via its TMEM transpose.
    @cute.jit
    def nvfp4_quant_impl(
        self,
        prequant_reg: cute.Tensor,
        *,
        norm_const: Optional[cutlass.Float32] = None,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # fp32 in -> e2m1 data + e4m3 sfc. Mirrors the prior nvfp4_quant scale
        # math (sfc -> capped/masked acc_scale); topk pre-mult + sf store are the
        # caller's job now.
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg), ),
                                     cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks, ), cutlass.Float8E4M3FN)
        in_blocks = cute.zipped_divide(prequant_reg,
                                       (vec, ))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        rcp_limit = cutlass.Float32(self._data_rcp_limit)
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            amax = self._amax_thread_fp32(block)
            if cutlass.const_expr(norm_const is not None):
                sfc_fp32 = amax * rcp_limit * norm_const
            else:
                sfc_fp32 = amax * rcp_limit
            sfc_e4m3 = sfc_fp32.to(cutlass.Float8E4M3FN)
            sfc_rt = cutlass.Float32(sfc_e4m3)
            if cutlass.const_expr(norm_const is not None):
                acc_scale = norm_const * cute.arch.rcp_approx(sfc_rt)
            else:
                acc_scale = cute.arch.rcp_approx(sfc_rt)
            acc_scale = cute.arch.fmin(acc_scale, Fp32Max)
            mask = cute.arch.fmin(sfc_rt * cutlass.Float32(1e30),
                                  cutlass.Float32(1.0))
            acc_scale = acc_scale * mask
            sf_values.append(sfc_e4m3)
            data_blocks.append(self._scale_to_e2m1_ssa(block, acc_scale))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.Float8E4M3FN))
        return data, sf

    @cute.jit
    def nvfp4_combine_quant_regs_in_thread_impl(
        self,
        prequant_reg: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # bf16 in -> e2m1 data + per-16 bf16 amax. amax found on bf16 (packed).
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg), ),
                                     cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks, ), cutlass.BFloat16)
        in_blocks = cute.zipped_divide(prequant_reg,
                                       (vec, ))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            amax = self._amax_thread_bf16(block)
            sf_values.append(amax)
            decode_scale = cutlass.Float32(amax) * cutlass.Float32(
                self._data_rcp_limit)
            data_blocks.append(
                self._scale_to_e2m1_ssa(block, self._enc_nvfp4(decode_scale)))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.BFloat16))
        return data, sf

    # Mapping: (lane_idx, selected_sf_idx) -> (token_64, hidden_32)
    # token_idx = lane_idx % 16 + selected_sf_idx * 16
    # hidden_idx = lane_idx // 16 * 16
    @cute.jit
    def nvfp4_combine_quant_threads_with_the_same_reg_impl(
        self,
        prequant_reg: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # UBLK has lane == hidden, so the warp's 32 lanes are 32 consecutive
        # hidden. A scale block = sf_vec hidden, so the lanes split along hidden
        # into 32 // sf_vec blocks of sf_vec lanes each (warp = blocks_per_warp *
        # lanes_per_block, the EP x TP split). Only the sf_vec lanes inside a
        # block share its CREDUX scale, so they pool the subtile tokens: sf_vec
        # == 32 pools the whole warp, sf_vec < 32 pools fewer (more per lane).
        lanes_per_block = self.sf_vec_size
        n_tokens = cute.size(prequant_reg)
        lane_in_block = self.lane_idx % cutlass.Int32(lanes_per_block)
        data = cute.make_rmem_tensor((n_tokens, ), cutlass.Float4E2M1FN)
        selected_sf = cute.make_rmem_tensor((n_tokens // lanes_per_block, ),
                                            cutlass.BFloat16)
        scaled_vec = cute.full((n_tokens, ), cutlass.Float32(0.0),
                               cutlass.Float32)
        for token_idx in cutlass.range_constexpr(n_tokens):
            value = cutlass.Float32(prequant_reg[token_idx])
            amax_bf16 = self._amax_lane(value).to(cutlass.BFloat16)
            slot = token_idx // lanes_per_block
            if (token_idx % lanes_per_block) == lane_in_block:
                selected_sf[slot] = amax_bf16
            else:
                selected_sf[slot] = selected_sf[slot]
            decode_scale = cutlass.Float32(amax_bf16) * cutlass.Float32(
                self._data_rcp_limit)
            scaled_value = value * self._enc_nvfp4(decode_scale)
            scaled_vec = cute.TensorSSA(
                vector.insert(
                    scaled_value.ir_value(),
                    scaled_vec.ir_value(),
                    [],
                    [token_idx],
                ),
                (n_tokens, ),
                cutlass.Float32,
            )
        self._store_packed_data(data, scaled_vec.to(cutlass.Float4E2M1FN))
        return data, selected_sf

    @cute.jit
    def mxfp8_combine_quant_regs_in_thread_impl(
        self,
        prequant_reg: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # bf16 in -> e4m3 data + per-32 e8m0. amax found on bf16 (packed).
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg), ),
                                     cutlass.Float8E4M3FN)
        sf = cute.make_rmem_tensor((n_blocks, ), cutlass.Float8E8M0FNU)
        in_blocks = cute.zipped_divide(prequant_reg,
                                       (vec, ))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            # widen the native-bf16 amax to fp32 for the e8m0 round-up math.
            scale_e8m0, scale_f32 = self._e8m0(
                cutlass.Float32(self._amax_thread_bf16(block)))
            sf_values.append(scale_e8m0)
            data_blocks.append(
                self._scale_to_e4m3_ssa(block, self._enc_mxfp8(scale_f32)))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.Float8E8M0FNU))
        return data, sf

    # Mapping: (lane_idx, selected_sf_idx) -> (token_64, hidden_32)
    # token_idx = lane_idx + selected_sf_idx * 32
    # hidden_idx = 0
    @cute.jit
    def mxfp8_combine_quant_threads_with_the_same_reg_impl(
        self,
        prequant_reg: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # UBLK has lane == hidden, so the warp's 32 lanes are 32 consecutive
        # hidden. A scale block = sf_vec hidden, so the lanes split along hidden
        # into 32 // sf_vec blocks of sf_vec lanes each (warp = blocks_per_warp *
        # lanes_per_block, the EP x TP split). Only the sf_vec lanes inside a
        # block share its CREDUX scale, so they pool the subtile tokens. mxfp8
        # sf_vec == 32 -> the whole warp is one block, all 32 lanes pool.
        lanes_per_block = self.sf_vec_size
        n_tokens = cute.size(prequant_reg)
        lane_in_block = self.lane_idx % cutlass.Int32(lanes_per_block)
        data = cute.make_rmem_tensor((n_tokens, ), cutlass.Float8E4M3FN)
        selected_sf = cute.make_rmem_tensor((n_tokens // lanes_per_block, ),
                                            cutlass.Float8E8M0FNU)
        scaled_vec = cute.full((n_tokens, ), cutlass.Float32(0.0),
                               cutlass.Float32)
        for token_idx in cutlass.range_constexpr(n_tokens):
            value = cutlass.Float32(prequant_reg[token_idx])
            scale_e8m0, scale_f32 = self._e8m0(self._amax_lane(value))
            slot = token_idx // lanes_per_block
            if (token_idx % lanes_per_block) == lane_in_block:
                selected_sf[slot] = scale_e8m0
            else:
                selected_sf[slot] = selected_sf[slot]
            scaled_value = value * self._enc_mxfp8(scale_f32)
            scaled_vec = cute.TensorSSA(
                vector.insert(
                    scaled_value.ir_value(),
                    scaled_vec.ir_value(),
                    [],
                    [token_idx],
                ),
                (n_tokens, ),
                cutlass.Float32,
            )
        self._store_packed_data(data, scaled_vec.to(cutlass.Float8E4M3FN))
        return data, selected_sf

    # -- shared sub-steps -----------------------------------------------------

    @cute.jit
    def _scale_to_e2m1_ssa(
        self,
        block: cute.Tensor,
        enc: cutlass.Float32,
    ) -> cute.TensorSSA:
        block_f32 = block.load().to(cutlass.Float32)
        enc_vec = cute.full_like(block_f32, enc, cutlass.Float32)
        return (block_f32 * enc_vec).to(cutlass.Float4E2M1FN)

    @cute.jit
    def _scale_to_e4m3_ssa(
        self,
        block: cute.Tensor,
        enc: cutlass.Float32,
    ) -> cute.TensorSSA:
        block_f32 = block.load().to(cutlass.Float32)
        enc_vec = cute.full_like(block_f32, enc, cutlass.Float32)
        return (block_f32 * enc_vec).to(cutlass.Float8E4M3FN)

    @cute.jit
    def _concat_blocks_ssa(self, blocks,
                           dtype: Type[cutlass.Numeric]) -> cute.TensorSSA:
        values = []
        for block_idx in cutlass.range_constexpr(len(blocks)):
            block = blocks[block_idx]
            for elem_idx in cutlass.range_constexpr(cute.size(block.shape)):
                values.append(block[elem_idx].ir_value())
        vec = vector.from_elements(
            T.vector(len(values), dtype.mlir_type),
            values,
        )
        return cute.TensorSSA(vec, (len(values), ), dtype)

    @cute.jit
    def _values_to_ssa(self, values,
                       dtype: Type[cutlass.Numeric]) -> cute.TensorSSA:
        vec = vector.from_elements(
            T.vector(len(values), dtype.mlir_type),
            [values[i].ir_value() for i in range(len(values))],
        )
        return cute.TensorSSA(vec, (len(values), ), dtype)

    @cute.jit
    def _concat_i32_blocks_ssa(self, blocks) -> cute.TensorSSA:
        values = []
        for block_idx in cutlass.range_constexpr(len(blocks)):
            packed_block = blocks[block_idx].bitcast(cutlass.Int32)
            for elem_idx in cutlass.range_constexpr(
                    cute.size(packed_block.shape)):
                values.append(packed_block[elem_idx].ir_value())
        vec = vector.from_elements(
            T.vector(len(values), cutlass.Int32.mlir_type),
            values,
        )
        return cute.TensorSSA(vec, (len(values), ), cutlass.Int32)

    @cute.jit
    def _store_packed_blocks(self, data: cute.Tensor, blocks) -> None:
        packed_data = cute.recast_tensor(data, cutlass.Int32)
        packed_data.store(self._concat_i32_blocks_ssa(blocks))

    @cute.jit
    def _store_packed_data(self, data: cute.Tensor,
                           data_ssa: cute.TensorSSA) -> None:
        packed_data = cute.recast_tensor(data, cutlass.Int32)
        packed_data.store(data_ssa.bitcast(cutlass.Int32))

    @cute.jit
    def _amax_thread_fp32(self, block: cute.Tensor) -> cutlass.Float32:
        # max.xorsign.abs reduces |.| in one op per element; the result sign is
        # the xor of the inputs (junk for an amax), so clear it at the end.
        def max_abs(lhs: cutlass.Float32,
                    rhs: cutlass.Float32) -> cutlass.Float32:
            return cutlass.Float32(
                llvm.inline_asm(
                    T.f32(),
                    [
                        cutlass.Float32(lhs).ir_value(),
                        cutlass.Float32(rhs).ir_value()
                    ],
                    "max.xorsign.abs.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                ))

        acc = block[0]
        for elem_idx in cutlass.range_constexpr(1, cute.size(block)):
            acc = max_abs(acc, block[elem_idx])
        mag_bits = cutlass.Int32(
            llvm.bitcast(
                T.i32(),
                cutlass.Float32(acc).ir_value())) & cutlass.Int32(0x7FFFFFFF)
        return cutlass.Float32(llvm.bitcast(T.f32(), mag_bits.ir_value()))

    @cute.jit
    def _amax_thread_bf16(self, block: cute.Tensor) -> cutlass.BFloat16:
        # Packed bf16x2 abs-max: tree-reduce the pairs, then fold the survivor's
        # two halves (high shifted into low). max.xorsign.abs leaves a junk sign,
        # so the low bf16 is masked before being read back. The amax is natively
        # bf16 -- exactly what the wire format stores.
        def max_abs(lhs: cutlass.Int32, rhs: cutlass.Int32) -> cutlass.Int32:
            return cutlass.Int32(
                llvm.inline_asm(
                    T.i32(),
                    [
                        cutlass.Int32(lhs).ir_value(),
                        cutlass.Int32(rhs).ir_value()
                    ],
                    "max.xorsign.abs.bf16x2 $0, $1, $2;",
                    "=r,r,r",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                ))

        pairs = cute.recast_tensor(block, cutlass.Int32)  # (vec/2,) bf16x2
        acc = cutlass.Int32(pairs[0])
        for pair_idx in cutlass.range_constexpr(1, cute.size(pairs)):
            acc = max_abs(acc, pairs[pair_idx])
        acc = max_abs(acc, acc >> cutlass.Int32(16))
        amax_bits = cute.make_rmem_tensor((1, ), cutlass.Int32)
        amax_bits[0] = acc & cutlass.Int32(0x7FFF)
        return cute.recast_tensor(amax_bits, cutlass.BFloat16)[0]

    @cute.jit
    def _amax_lane(self, v: cutlass.Float32) -> cutlass.Float32:
        if cutlass.const_expr(self.sf_vec_size == 32):
            return cute.arch.warp_redux_sync(v, "fmax", abs=True)
        first_half = (self.lane_idx % cutlass.Int32(32)) < cutlass.Int32(16)
        vsel = cutlass.Float32(0.0)
        if first_half:
            vsel = v
        amax = cute.arch.warp_redux_sync(vsel, "fmax", abs=True)
        if not first_half:
            amax = cute.arch.warp_redux_sync(v, "fmax", abs=True)
        return amax

    @cute.jit
    def _e8m0(
            self, amax: cutlass.Float32
    ) -> Tuple[cutlass.Float8E8M0FNU, cutlass.Float32]:
        candidate = amax * cutlass.Float32(self._data_rcp_limit)
        scale_f32 = cutlass.Float32(
            cvt_f32_to_f8_to_f32(candidate, cutlass.Float8E8M0FNU))
        return scale_f32.to(cutlass.Float8E8M0FNU), scale_f32

    @cute.jit
    def _enc_nvfp4(self, decode_scale: cutlass.Float32) -> cutlass.Float32:
        # rcp.approx.ftz with the fc1 cap+mask idiom (amax==0 -> 0, no inf*0 NaN).
        enc = cute.arch.fmin(cute.arch.rcp_approx(decode_scale), Fp32Max)
        mask = cute.arch.fmin(decode_scale * cutlass.Float32(1e30),
                              cutlass.Float32(1.0))
        return enc * mask

    @cute.jit
    def _enc_mxfp8(self, scale_f32: cutlass.Float32) -> cutlass.Float32:
        # Skip nan
        enc = cute.arch.fmin(cute.arch.rcp_approx(scale_f32), Fp32Max)
        mask = cute.arch.fmin(scale_f32 * cutlass.Float32(1e30),
                              cutlass.Float32(1.0))
        return enc * mask


# =============================================================================
# Region tag
# =============================================================================


class Region:
    """Codegen-time region tag for a 16x32 sub-region within a 32x32 tile."""

    Top = 0
    Bottom = 1


# =============================================================================
# TmemTranspose16x32
# =============================================================================


class _TmemTranspose16x32Core:
    """Physical implementation of the 16x32 -> 32x16 TMEM in-place transpose.

    The transpose is a fixed sequence of tcgen05 32-bit element atoms; each
    32-bit slot is opaque to it (an fp32 swiglu-fold value for fc1, or a packed
    ``(bf16, bf16)`` pair for fc2 -- the physical (lane_idx, elem_idx)
    distribution is identical either way).  The (thread, reg) -> (tmem_dp,
    tmem_col) input / output mapping is documented on the ``TmemTranspose16x32``
    subclass, which is the public entry point.

    Per-thread RMEM coordinate convention:

      - ``lane_idx`` -- warp lane id (= thread index within warp), in [0, 32).
      - ``elem_idx`` -- per-thread reg index, in [0, 16).
    """

    _PermR1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    _PermR3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    _PermR4 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)

    _TmemRowStride = 1 << 16
    _io_dtype = cutlass.Float32

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1), ),
            stride=(((_TmemTranspose16x32Core._TmemRowStride, 1), 0), ),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor,
                        num_regs: int,
                        offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset,
            cute.make_layout((((num_regs, ), 1), ), stride=(((1, ), 0), )),
        )

    @staticmethod
    def load_subtile_raw_acc(
        tmem_subtile_tensor: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """LDTM the entire 32-lane x 64-col raw acc region of one epi
        subtile into 4 independent (16,) fp32 RMEM tensors.

        Used by the overlap-acc unroll path in
        ``_run_fc{1,2}_task_tile`` to extract all raw acc data of the
        first 2 subtiles up front, so that the acc TMEM can be released
        to the next mma right after the first subtile's 4 LDTMs (instead
        of waiting for a full subtile body to complete).

        ``tmem_subtile_tensor`` is the (32 lanes, 64 cols) view onto a
        single epi subtile's acc TMEM region (already offset by
        ``warp_lane_offset + acc_stage_col_offset + subtile_col_offset``;
        see ``SwapABSwigluFp4Epilogue._subtile_local_tmem_tensor``).

        Returns a 4-tuple of (16,) fp32 RMEM tensors, each carrying the
        (lane_idx, elem_idx) input distribution documented on
        ``TmemTranspose16x32`` (physically identical for fc1 and fc2):

          [0] gate_lo / first-half top   -- subtile cols 0..31, lanes 0..15
          [1] up_lo   / first-half bot   -- subtile cols 0..31, lanes 16..31
          [2] raw_top / second-half top  -- subtile cols 32..63, lanes 0..15
          [3] raw_bot / second-half bot  -- subtile cols 32..63, lanes 16..31

        4 atom calls of ``Ld16x64bOp(Repetition.x16) Float32`` -- the same
        atom used by the per-subtile entry LDTM.  Each output is in the
        ``TmemTranspose16x32`` input distribution and can be fed straight
        into a transpose as ``reg_tensor`` (skip-R1.Load mode).
        """
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            _TmemTranspose16x32Core._io_dtype,
        )

        ptr = tmem_subtile_tensor.iterator
        half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride

        # 4 source 16-lane x 32-col views over the (32, 64) subtile region:
        #   first  half (cols 0..31): top  lanes 0..15  / bot lanes 16..31
        #   second half (cols 32..63): top lanes 0..15  / bot lanes 16..31
        # All offsets are Python ints (compile-time const) so cute can
        # const-fold them and infer the correct (>= 8 B / 2 col) ptr
        # alignment that the LDTM atom requires.  Using ``cutlass.Int32``
        # offsets here would wrap them as SSA values that cute treats as
        # alignment-unknown, tripping the atom's verifier.
        first_top_view = cute.make_tensor(
            ptr,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        first_bot_view = cute.make_tensor(
            ptr + half_lane_off,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        second_top_view = cute.make_tensor(
            ptr + 32,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        second_bot_view = cute.make_tensor(
            ptr + 32 + half_lane_off,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )

        first_top = cute.make_rmem_tensor((16, ),
                                          _TmemTranspose16x32Core._io_dtype)
        first_bot = cute.make_rmem_tensor((16, ),
                                          _TmemTranspose16x32Core._io_dtype)
        second_top = cute.make_rmem_tensor((16, ),
                                           _TmemTranspose16x32Core._io_dtype)
        second_bot = cute.make_rmem_tensor((16, ),
                                           _TmemTranspose16x32Core._io_dtype)

        cute.copy(
            atom_ld16x64,
            first_top_view,
            _TmemTranspose16x32Core._rmem_copy_view(first_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            first_bot_view,
            _TmemTranspose16x32Core._rmem_copy_view(first_bot, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_top_view,
            _TmemTranspose16x32Core._rmem_copy_view(second_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_bot_view,
            _TmemTranspose16x32Core._rmem_copy_view(second_bot, 16),
        )

        return (first_top, first_bot, second_top, second_bot)

    def __init__(
        self,
        tmem_ptr,
        region: int,
        reg_tensor: Optional[cute.Tensor] = None,
    ) -> None:
        # The whole transpose is built from 32-bit element atoms; _io_dtype
        # drives _src_regs / output / every LDTM/STTM atom below, so guard the
        # invariant once here (tautological today, defensive against future
        # dtype edits).
        if cutlass.const_expr(self._io_dtype.width != 32):
            raise TypeError(
                f"{type(self).__name__} requires a 32-bit _io_dtype (the "
                f"transpose uses 32-bit element atoms), got {self._io_dtype} "
                f"(width {self._io_dtype.width}).")

        half_lane_off = 16 * self._TmemRowStride
        if region == Region.Top:
            src_ptr = tmem_ptr
            dst_ptr = tmem_ptr
        elif region == Region.Bottom:
            src_ptr = tmem_ptr + half_lane_off
            dst_ptr = tmem_ptr + 16
        else:
            raise ValueError("region must be Region.Top or Region.Bottom")

        self.region = region

        self._tmem_src_full = cute.make_tensor(src_ptr,
                                               self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(dst_ptr,
                                               self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(dst_ptr,
                                              self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(dst_ptr + half_lane_off,
                                              self._tmem_layout(16, 16))

        self._atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            self._io_dtype,
        )
        self._atom_st16x128 = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8),
            self._io_dtype,
        )
        self._atom_st32x32 = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16),
            self._io_dtype,
        )
        self._atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2),
            self._io_dtype,
        )
        self._atom_ld16x128 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition.x4),
            self._io_dtype,
        )

        self._src_regs = cute.make_rmem_tensor((16, ), self._io_dtype)
        # ``output`` is a bare (16,) RMEM fragment; its (lane_idx, elem_idx)
        # distribution after all four rounds is the transpose output mapping
        # documented on ``TmemTranspose16x32``.
        self.output = cute.make_rmem_tensor((16, ), self._io_dtype)

        # skip-R1.Load mode: ``reg_tensor`` must already be in the transpose
        # input distribution (see ``TmemTranspose16x32`` / produced by
        # ``load_subtile_raw_acc``); we copy it in lieu of the R1 LDTM.
        # Weak entry guard (replaces the removed input contract): the transpose
        # atoms are 32-bit element atoms over exactly 16 regs/lane, so the fed
        # tensor must be a 32-bit element type (fp32 or packed bf16x2) of size 16.
        self._reg_tensor = reg_tensor
        if reg_tensor is not None:
            if cutlass.const_expr(reg_tensor.element_type.width != 32):
                raise TypeError(
                    f"{type(self).__name__} reg_tensor must be a 32-bit element "
                    f"type (fp32 or packed bf16x2), got element type "
                    f"{reg_tensor.element_type} (width {reg_tensor.element_type.width})."
                )
            if cutlass.const_expr(cute.size(reg_tensor) != 16):
                raise ValueError(
                    f"{type(self).__name__} reg_tensor must hold exactly 16 "
                    f"elements, got {cute.size(reg_tensor)}.")
            for r in range(16):
                self._src_regs[r] = reg_tensor[r]

    # -- R1 ------------------------------------------------------------------

    def r1_load(self) -> None:
        """LDTM src region -> ``_src_regs``.  No-op in skip-R1.Load mode."""
        if self._reg_tensor is not None:
            return
        cute.copy(
            self._atom_ld16x64,
            self._tmem_src_full,
            self._rmem_copy_view(self._src_regs, 16),
        )

    def r1_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(
            self._atom_st16x128,
            self._rmem_copy_view(self.output, 16),
            self._tmem_src_full,
        )

    # -- R2 ------------------------------------------------------------------

    def r2_load(self) -> None:
        cute.copy(
            self._atom_ld16x64,
            self._tmem_src_full,
            self._rmem_copy_view(self._src_regs, 16),
        )

    def r2_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self._src_regs, 16),
            self._tmem_dst_full,
        )

    # -- R3 ------------------------------------------------------------------

    def r3_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r3_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r3_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR3[r]]

    def r3_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output, 16),
            self._tmem_dst_full,
        )

    # -- R4 ------------------------------------------------------------------

    def r4_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r4_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r4_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR4[r]]

    def r4_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output, 16),
            self._tmem_dst_full,
        )

    def from_r1_perm_until_last_store(self) -> cute.Tensor:
        self.r1_perm()
        self.r1_store()
        self.r2_load()
        self.r2_store()
        self.r3_load_top()
        self.r3_load_bot()
        self.r3_perm()
        self.r3_store()
        self.r4_load_top()
        self.r4_load_bot()
        self.r4_perm()
        return self.output


class TmemTranspose16x32(_TmemTranspose16x32Core):
    """Public 16x32 -> 32x16 TMEM in-place transpose.

    The per-thread RMEM ``(lane_idx, elem_idx) -> (tmem_dp, tmem_col)`` mapping
    is fixed by the underlying atom sequence and is identical for fc1 (each slot
    is an fp32 swiglu-fold value, ``tmem_col`` = intermediate-output index) and
    fc2 (each slot is a packed bf16x2, ``tmem_col`` = hidden-pair index).  Only
    the ``tmem_col`` semantic name differs between the two uses; the physical
    distribution below is the single source of truth.

    Input distribution -- what each (lane_idx, elem_idx) reg holds on entry
    (i.e. straight after the 16-dp x 32-col source LDTM, or as fed in via
    ``reg_tensor`` / ``load_subtile_raw_acc`` for skip-R1.Load mode):

        tmem_dp  = elem_idx * 2 + (lane_idx // 2) % 2          # in [0, 32)
        tmem_col = (lane_idx % 2) * 8 + lane_idx // 4          # in [0, 16)

    Output distribution -- after all four rounds, the 32-dp x 16-col result has
    each lane owning one full dp-row of 16 cols:

        tmem_dp  = lane_idx                                    # in [0, 32)
        tmem_col = elem_idx                                    # in [0, 16)
    """


# =============================================================================
# TmemTranspose32x32Inplace
# =============================================================================


class TmemTranspose32x32Inplace:
    """fc1 epi 32x32 in-place TMEM transpose: two ``TmemTranspose16x32``
    sub-instances (``top`` = lanes 0..15, ``bot`` = lanes 16..31).

    Optional ``reg_tensor_top`` / ``reg_tensor_bot`` enable skip-R1.Load mode
    for both halves; they must be provided or omitted together.
    """

    def __init__(
        self,
        tmem_ptr,
        reg_tensor_top: Optional[cute.Tensor] = None,
        reg_tensor_bot: Optional[cute.Tensor] = None,
    ) -> None:
        if (reg_tensor_top is None) != (reg_tensor_bot is None):
            raise ValueError(
                "TmemTranspose32x32Inplace: reg_tensor_top and reg_tensor_bot "
                "must be provided or omitted together (both halves either "
                "skip-R1.Load or do R1.Load).")
        self.top = TmemTranspose16x32(tmem_ptr,
                                      Region.Top,
                                      reg_tensor=reg_tensor_top)
        self.bot = TmemTranspose16x32(tmem_ptr,
                                      Region.Bottom,
                                      reg_tensor=reg_tensor_bot)

    def from_r1_perm_until_last_store(self) -> Tuple[cute.Tensor, cute.Tensor]:
        self.bot.r1_perm()
        self.top.r1_perm()
        self.bot.r1_store()
        self.top.r1_store()

        self.bot.r2_load()
        self.top.r2_load()
        self.top.r2_store()
        self.bot.r2_store()

        self.top.r3_load_top()
        self.top.r3_load_bot()
        self.bot.r3_load_top()
        self.bot.r3_load_bot()
        self.top.r3_perm()
        self.bot.r3_perm()
        self.top.r3_store()
        self.bot.r3_store()

        self.top.r4_load_top()
        self.top.r4_load_bot()
        self.bot.r4_load_top()
        self.bot.r4_load_bot()
        self.top.r4_perm()
        self.bot.r4_perm()
        return self.top.output, self.bot.output


@dataclasses.dataclass(frozen=True)
class NvFp4OptinalEpiArgs:
    # MoE domain (experts), nvfp4 only?
    fc1_alpha: Optional[cute.Tensor]
    fc2_alpha: Optional[cute.Tensor]
    fc1_norm_const: Optional[cute.Tensor]
    # -----------------------------------
    # MoE domain (token, topk), deepgemm graph only? for transformer graph, we want reduce kernel to perform the score mul.
    topk_scores: Optional[cute.Tensor]


# TODO: Need to remove `Swiglu` and `Fp4` out of name, later this should be extended to other activations and dtypes.
class SwapABSwigluFp4Epilogue:
    """Autonomous epilogue for the swap-AB SwiGLU NVFP4 kernel.

    ``run()`` is the single entry point the kernel calls inside the epi
    warp body.  The kernel's responsibility is reduced to:

      - allocate / free TMEM and build ``acc_tensor``
      - construct the AB / acc pipelines
      - obtain the scheduler consumer

    Everything else (acc consumer state, task-tile loop, overlap rotation,
    early release, TMA store commit / drain, per-subtile dispatch) lives
    inside this class.
    """

    _EpilogueSyncWaitBarId = 1  # Arrive and wait only
    _EpilogueAsyncBarIdBase = 4  # Some arrive, the others arrive and wait
    _EpilogueFc1GateUpInterleave = 16
    _EpilogueTokenTileSize = 64  # Fundamentally the epi_tile_n
    _EpilogueFc1IntermediateGateUpTileSize = 128  # Fundamentally epi_tile_m
    _EpilogueFc1IntermediateDownTileSize = 64  # Fundamentally epi_tile_m // 2
    _EpilogueFc2HiddenTileSize = 128  # Fundamentally epi_tile_m
    _EpilogueWarpCnt = 4
    _TmemColsTotal = 512  # TODO: Remove this hardcode for future arch

    def __init__(
            self,
            *,
            mma_tiler_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            use_2cta_instrs: bool,
            sf_vec_size: int,
            fc1_output_dtype: Type[cutlass.Numeric],
            combine_format:
        CombineFormat,  # fc2 combine wire: act_dtype (data) + scale_dtype (sf)
            non_ubulk_fc2_store:
        bool,  # Whether epilogue warps use STG or UBLK in fc2
            in_kernel_fc2_reduce:
        bool,  # Whether epilogue warps reduce fc2 output to peer
            token_back_by_dispatch:
        bool = False,  # Whether epilogue warps store fc2 to local or peer
            acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
            fc1_output_sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
            allow_overlap_acc: bool = True,
            static_expert_shape: Optional[Tuple[
                int, int, int]] = None,  # [expert, intermediate, hidden]
            gate_up_clamp: Optional[float] = None,  # Swiglu style only
            epi_flag_batch: Optional[Tuple[int, int]] = (
                1, 1),  # (fc1, fc2) done-counter publish batch
    ) -> None:
        if fc1_output_dtype is not cutlass.Float4E2M1FN:
            raise NotImplementedError(
                "SwapABSwigluFp4Epilogue currently assumes fc1 output in "
                f"sC is NVFP4 Float4E2M1FN; got {fc1_output_dtype}. "
                "Changing this dtype requires redesigning the fixed 8KB "
                "shared epilogue scratch layout.")
        if token_back_by_dispatch and not non_ubulk_fc2_store:
            raise ValueError(
                "token_back_by_dispatch=True requires non_ubulk_fc2_store=True; "
                "bulk fc2 store is incompatible with dispatch-warp token back "
                "(STG is strictly more efficient for that pipeline).")
        if token_back_by_dispatch:
            in_kernel_fc2_reduce = False
        self.fc2_use_bulk = not non_ubulk_fc2_store
        self.reduce_topk_in_kernel = in_kernel_fc2_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.combine_format = combine_format
        self.fc1_output_dtype = fc1_output_dtype
        self.acc_dtype = acc_dtype
        self.fc1_output_sf_dtype = fc1_output_sf_dtype
        self.sf_vec_size = sf_vec_size
        # Swiglu gate/up clamp limit; None disables clamping.
        self.gate_up_clamp = gate_up_clamp
        # Done-counter publish batch granularity
        _fc1_eb, _fc2_eb = (1, 1) if epi_flag_batch is None else epi_flag_batch
        self.fc1_epi_flag_batch = max(1, min(32, int(_fc1_eb)))
        self.fc2_epi_flag_batch = max(1, min(32, int(_fc2_eb)))
        self.cluster_tile_intermediate_downproj = self._EpilogueFc1IntermediateDownTileSize * cluster_shape_mn[
            0]

        atom_thr_size = 2 if use_2cta_instrs else 1
        self.cta_tile_m = self._EpilogueFc2HiddenTileSize
        self.cta_tile_n = mma_tiler_mnk[1]
        self.cta_tile_k = mma_tiler_mnk[2]
        assert (mma_tiler_mnk[0] // atom_thr_size == self.cta_tile_m)
        assert (self.cta_tile_n % self._EpilogueTokenTileSize == 0)
        self.static_expert_shape = static_expert_shape
        self.acc_tmem_cols = self.cta_tile_n
        self.acc_sf_cols = (max(self.cta_tile_n // 128, 1) * self.cta_tile_k +
                            max(self.cta_tile_m // 128, 1) *
                            self.cta_tile_k) // self.sf_vec_size

        if static_expert_shape is not None and static_expert_shape[2] % (
                self.cta_tile_m * cluster_shape_mn[0]) == 0:
            self.fc2_hidden_needs_predicate: bool = False
        else:
            self.fc2_hidden_needs_predicate: bool = True

        if static_expert_shape is not None:
            intermediate_downproj = static_expert_shape[1] // 2
            self.intermediate_downproj: Optional[int] = intermediate_downproj
        else:
            self.intermediate_downproj: Optional[int] = None

        self.subtile_cnt = self.cta_tile_n // self._EpilogueTokenTileSize
        self.overlapping_accum = allow_overlap_acc and (
            self.acc_tmem_cols + self.acc_sf_cols > self._TmemColsTotal // 2)
        self.num_acc_stage = 2
        self.num_acc_pipeline_stages = 1 if self.overlapping_accum else self.num_acc_stage
        self.overlapped_tmem_cols = self._EpilogueTokenTileSize if self.overlapping_accum else 0
        assert (not self.overlapping_accum
                or self.overlapped_tmem_cols >= self.acc_sf_cols)
        self.epi_smem_bytes = 8 * 1024
        if self.fc1_output_dtype.width > 4:
            raise NotImplementedError(
                "Remember to adjust the smem size when switch to mxfp8 support")
        self.tmem_acc_layout_py_obj = ((self.cta_tile_m, self.cta_tile_n,
                                        self.num_acc_stage),
                                       (_TmemTranspose16x32Core._TmemRowStride,
                                        1, self.cta_tile_n -
                                        self.overlapped_tmem_cols))

    def get_epi_storage_type(self) -> Type:
        # This could be extended to take atoms space for the larger sf_vec_size quant.
        @cute.struct
        class EpilogueSharedStorage:
            # 256 byte alignment is for the swizzle start address.
            epi_smem: cute.struct.Align[cute.struct.MemRange[
                cutlass.Int8, self.epi_smem_bytes], 256]

        return EpilogueSharedStorage

    def fc1_staged_smem_layout(
        self,
        n_stages: int,
        without_stage_mode: bool = False
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        layout = sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self._EpilogueTokenTileSize,
             self._EpilogueFc1IntermediateDownTileSize),
            n_stages,
        )
        if without_stage_mode:
            return cute.select(layout, mode=[0, 1])
        return layout

    @cute.jit
    def run(
            self,
            epi_smem_storage,
            tmem_ptr: cute.Pointer,
            acc_pipeline,
            # ── Sched ────────────────────────────────────────────────────────
            sched_consumer: MoESchedConsumer,
            sched_ext: MoESchedExtension,
            # ── tensors ──────────────────────────────────
            tma_atom_fc1_output: cute.CopyAtom,
            fc1_output: cute.Tensor,  # Domain of fake (m, n, l)
            fc1_output_sf: cute.Tensor,  # Domain of fake (m, n, l)
            fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
            fc1_done_counter: cute.Tensor,  # 1D tensor
            tidx: cutlass.Int32,
            optional_epi_args:
        NvFp4OptinalEpiArgs = None,  # Epilogue optional runtime arguments.
            token_comm_args=None,  # Only valid when enable token communication
    ):
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = NvFp4OptinalEpiArgs(
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                topk_scores=None,
            )
        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(
                self.tmem_acc_layout_py_obj[0],
                stride=self.tmem_acc_layout_py_obj[1],
            ),
        )

        fc1_epi = SwapABFc1Epilogue(self, tidx, epi_smem_storage, sched_ext,
                                    tma_atom_fc1_output, fc1_output,
                                    fc1_output_sf, fc1_done_counter,
                                    optional_epi_args)
        fc2_epi = SwapABFc2Epilogue(self, tidx, epi_smem_storage, fc2_output,
                                    token_comm_args, optional_epi_args)

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages)
        wait_only_named_barrier = pipeline.NamedBarrier(
            barrier_id=self._EpilogueSyncWaitBarId,
            num_threads=32 * self._EpilogueWarpCnt,
        )
        is_odd_turn = cutlass.Int32(1)
        work_tile_info = sched_consumer.consume_work()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx % (self._EpilogueWarpCnt * 32),
        )

        while work_tile_info.is_valid_tile:
            if cutlass.const_expr(self.overlapping_accum):
                tmem_stage_idx = acc_consumer_state.phase
            else:
                tmem_stage_idx = acc_consumer_state.index
            tmem_acc_current = tmem_acc[None, None, tmem_stage_idx]
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                # The __call__ args should only take the while loop args, leave all loop irrevalent args to the init.
                fc1_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            else:
                # The __call__ args should only take the while loop args, leave all loop irrevalent args to the init.
                fc2_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            iket.range_pop()

            prev_work_tile_info = work_tile_info
            cur_was_linear1 = prev_work_tile_info.phase == cutlass.Int32(
                BlockPhase.Linear1)

            acc_consumer_state.advance()
            if cutlass.const_expr(self.overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn

            work_tile_info = sched_consumer.consume_work()

            # Drain fc1 TMA stores and sf stores before publishing the fc1-done counter.
            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            # _fence_rel_gpu()
            wait_only_named_barrier.arrive_and_wait()

            # Publish completion for the work tile snapshotted above.
            if cur_was_linear1:
                flag_tracker = fc1_epi.signal_fc1_done(prev_work_tile_info,
                                                       work_tile_info,
                                                       flag_tracker)
            else:
                flag_tracker = fc2_epi.signal_fc2_done(prev_work_tile_info,
                                                       work_tile_info,
                                                       flag_tracker)
        # Tail flush
        flag_tracker.fire()


class _ImmutableAfterInit:
    """Froze at the point calling `_freeze()`"""

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen_", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable after __init__ "
                f"(cannot set {name!r}).")
        object.__setattr__(self, name, value)

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen_", True)


# Device only object
class SwapABFc1Epilogue(_ImmutableAfterInit):

    def __init__(
        self,
        base: SwapABSwigluFp4Epilogue,
        tidx: cutlass.Int32,
        epi_smem_storage,
        sched_ext: MoESchedExtension,
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,  # fake (m,n,l) domain
        fc1_output_sf: cute.Tensor,  # fake (m,n,l) domain
        fc1_done_counter: cute.Tensor,  # 1D tensor
        optional_epi_args: NvFp4OptinalEpiArgs,
    ):
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        if cutlass.const_expr(base.fc1_output_dtype.width != 4):
            raise NotImplementedError(
                "Remember to adjust the swizzle and smem size.")
        # (token64, intermediate, stage)
        self.smem_tensor = cute.make_tensor(
            cute.recast_ptr(
                epi_smem_storage.epi_smem.data_ptr(),
                cute.make_swizzle(1, 4, 3),
                dtype=base.fc1_output_dtype,
            ),
            base.fc1_staged_smem_layout(base.subtile_cnt).outer,
        )
        self.sched_ext = sched_ext
        self.fc1_tma_atom = tma_atom_fc1_output
        self.fc1_output = fc1_output
        self.fc1_output_sf = fc1_output_sf
        self.fc1_done_counter = fc1_done_counter
        self.optional_epi_args = optional_epi_args
        self._freeze()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        # This object is a loop-invariant Python context wrapper, not a
        # dynamic value.  Keep it out of scf.while iter_args and reconstruct by
        # identity across region boundaries.  Any field that becomes a
        # loop-carried SSA value must be passed explicitly to __call__ instead
        # of being stored here.
        return []

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "SwapABFc1Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc1_done(self, work_tile_info, next_work_tile_info,
                        flag_tracker):
        # Only in-bound intermediate_downproj tiles signal; OOB -> null slot.
        if cutlass.const_expr(self.static_expert_shape is None
                              or self.intermediate_downproj %
                              self.cluster_tile_intermediate_downproj != 0):
            in_bound = (work_tile_info.tile_m_idx *
                        self._EpilogueFc1IntermediateDownTileSize
                        < self.fc1_output.shape[1])
        else:
            in_bound = True
        slot = (work_tile_info.cumulative_token_block_count +
                work_tile_info.tile_n_idx)
        flag_addr = Int64(0)
        if in_bound:
            flag_addr = (self.fc1_done_counter.iterator + slot).toint()
        return flag_tracker.accumulate(
            next_work_tile_info.phase,
            self.fc1_epi_flag_batch,
            flag_addr,
        )

    @cute.jit
    def __call__(
            self,
            work_tile_info: MoEWorkTileInfo,
            tmem_acc_tensor: cute.Tensor,  # (cta_tile_m, cta_tile_n)
            acc_pipeline,
            acc_consumer_state,
            is_odd_turn: cutlass.Int32):
        # (tokens_this_expert, intermediate_down, 1)
        real_fc1_output, _ = self.sched_ext.get_gmem_tensor(
            "c",
            self.fc1_output,
            work_tile_info,
        )
        # (tokens_this_expert, intermediate_down, 1)
        real_fc1_output_sf, _ = self.sched_ext.get_gmem_tensor(
            "sfc",
            self.fc1_output_sf,
            work_tile_info,
        )
        # subtile-irrevalent hoist out here.
        if cutlass.const_expr(self.optional_epi_args.fc1_alpha is not None):
            alpha_val = self.optional_epi_args.fc1_alpha[
                work_tile_info.expert_idx]
        else:
            alpha_val = None
        if cutlass.const_expr(
                self.optional_epi_args.fc1_norm_const is not None):
            norm_const = self.optional_epi_args.fc1_norm_const[
                work_tile_info.expert_idx]
        else:
            norm_const = None
        # (cta_tile_m, cta_tile_n) -> (epi_tile_m, epi_tile_n, iters)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc1IntermediateGateUpTileSize,
                              self._EpilogueTokenTileSize))[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("fc1_epi")
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        # Overlap path preloads two subtiles before releasing acc TMEM.
        unroll_tile_cnt = 2 if cutlass.const_expr(self.overlapping_accum) else 0
        remain_subtile_cnt = self.subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (cutlass.Int32(self.subtile_cnt) -
                                 is_odd_turn) % cutlass.Int32(self.subtile_cnt)
            subtile_idx_second = (cutlass.Int32(self.subtile_cnt + 1) -
                                  is_odd_turn) % cutlass.Int32(self.subtile_cnt)

            # preload_subtile_first: subtile_idx_first's raw PRE-transpose acc, LDTM'd by
            # all 128 epi threads into 4 reg tensors == the 4 quadrants of the subtile's
            # (128 tmem_dp x 64 tmem_col) footprint. Only these raw-TMEM offsets are
            # guaranteed:
            #   reg[0]/reg[1], reg[2]/reg[3] : top vs bot      -> 16 apart in tmem_dp
            #   reg[0]/reg[2], reg[1]/reg[3] : 1st vs 2nd half -> 32 apart in tmem_col
            # (so reg[0..1] = the first 128x32, reg[2..3] = the second 128x32 of the 128x64.)
            # The per-lane (lane_idx, elem_idx) -> (tmem_dp, tmem_col) layout INSIDE each
            # reg tensor is opaque -- do not assume it; it only becomes well-defined once
            # the tmem transpose consumes them.
            preload_subtile_first: Tuple[
                cute.Tensor, cute.Tensor, cute.Tensor,
                cute.Tensor] = _TmemTranspose16x32Core.load_subtile_raw_acc(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None,
                                                      subtile_idx_first])

            # Release acc to next MMA unconditionally.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            # preload_subtile_second: same 128 tmem_dp x 64 tmem_col footprint, but for
            # subtile_idx_second (the other token subtile, not the 2nd col-half). Same
            # quadrant/offset invariants and opaque per-lane layout as above.
            preload_subtile_second: Tuple[
                cute.Tensor, cute.Tensor, cute.Tensor,
                cute.Tensor] = _TmemTranspose16x32Core.load_subtile_raw_acc(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None,
                                                      subtile_idx_second])

            # Both unrolled subtiles borrow tmem_subtile_second as workspace.
            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(
                        self._EpilogueTokenTileSize) < valid_tokens:
                    self.run_subtile(
                        work_tile_info=work_tile_info,
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                            None, None, subtile_idx_second],
                        preload_acc=preload_pair[i],
                        fc1_output=real_fc1_output,
                        fc1_output_sf=real_fc1_output_sf,
                        alpha_val=alpha_val,
                        norm_const=norm_const,
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self.overlapping_accum):
                subtile_idx = (cutlass.Int32(real_i + self.subtile_cnt) -
                               is_odd_turn) % cutlass.Int32(self.subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)

            if subtile_idx * cutlass.Int32(
                    self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                        None, None, subtile_idx],
                    preload_acc=None,
                    fc1_output=real_fc1_output,
                    fc1_output_sf=real_fc1_output_sf,
                    alpha_val=alpha_val,
                    norm_const=norm_const,
                )

        # Non-overlap-path release: at the natural task-tile boundary.
        if cutlass.const_expr(not self.overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        work_tile_info: MoEWorkTileInfo,
        subtile_idx: cutlass.Int32,
        # (intermedaite_gateup_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        # Rmems preloaded from tmem, contract with downstream tmem trans. Do not assume mapping here.
        preload_acc: Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        # (tokens_this_expert, intermediate_down, 1)
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
        norm_const: Optional[cutlass.Float32],
    ):
        if cutlass.const_expr(self.optional_epi_args.topk_scores is not None):
            # This means we need to perform DeepGEMM computation graph, topk_score at fc1 pre-quant
            topk_score_tensor, _ = self.sched_ext.get_gmem_tensor(
                "topk",
                self.optional_epi_args.topk_scores,
                work_tile_info,
            )  # (tokens_this_expert)
        else:
            topk_score_tensor = None

        # Contract about the transposed acc (assume nvfp4 output):
        # (epi_tid, val_id) -> (token_idx, intermediate_down_idx)
        # token_idx = epi_tid % 32 + val_id // 16 * 32
        # intermediate_down_idx = val_id % 16 + epi_tid // 32 * 16
        # Each thread holds (intermediate_down_16, token_2):(1, 16)

        # Step -1: preload topk scores.
        current_two_token_idices = (
            work_tile_info.tile_n_idx * self.cta_tile_n +
            subtile_idx * self._EpilogueTokenTileSize + self.lane_idx,
            work_tile_info.tile_n_idx * self.cta_tile_n +
            subtile_idx * self._EpilogueTokenTileSize + self.lane_idx + 32)
        if cutlass.const_expr(topk_score_tensor is not None):
            topk_scores = (topk_score_tensor[current_two_token_idices[0]],
                           topk_score_tensor[current_two_token_idices[1]])
        else:
            topk_scores = None

        # Step 0: load tmem
        if cutlass.const_expr(preload_acc is not None):
            gate_token_0_32, up_token_0_32, gate_token_32_64, up_token_32_64 = preload_acc
        else:
            gate_token_0_32 = cute.make_rmem_tensor((16, ), cutlass.Float32)
            up_token_0_32 = cute.make_rmem_tensor((16, ), cutlass.Float32)
            gate_token_32_64 = cute.make_rmem_tensor((16, ), cutlass.Float32)
            up_token_32_64 = cute.make_rmem_tensor((16, ), cutlass.Float32)
            # Although hardcode is not right, but since the whole tmem transpose is too tricky, I have to hardcode...
            # (epi_tile_m, epi_tile_n) -> (warp_local_epi_tile_m, epi_tile_n)
            # tmem_subtile_tensor_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[(None, self.warp_idx), None]
            tmem_subtile_tensor_per_warp = cute.logical_divide(
                tmem_subtile_tensor, (32, None))[(None, 0), None]
            # (warp_local_epi_tile_m, epi_tile_n) -> (((16, 32), 1), (2, 2))
            tmem_subtile_tensor_in_first_load_view = cute.logical_divide(
                cute.zipped_divide(tmem_subtile_tensor_per_warp, (16, 32)),
                ((16, 32), 1))
            atom = cute.make_copy_atom(
                tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
                cutlass.Float32,
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(
                    tmem_subtile_tensor_in_first_load_view[None, 0]),
                wrap_into_copy_standard_layout(gate_token_0_32),
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(
                    tmem_subtile_tensor_in_first_load_view[None, 1]),
                wrap_into_copy_standard_layout(up_token_0_32),
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(
                    tmem_subtile_tensor_in_first_load_view[None, 2]),
                wrap_into_copy_standard_layout(gate_token_32_64),
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(
                    tmem_subtile_tensor_in_first_load_view[None, 3]),
                wrap_into_copy_standard_layout(up_token_32_64),
            )

        # Step 1: perform swiglu on the first part, interleave with the second's 32x32 tmem transpose.
        token_0_32_pre_quant_pre_trans = self.alpha_swiglu_clamp(
            gate_token_0_32, up_token_0_32, alpha_val)

        # gate_token_32_64 / up_token_32_64 are already in the transpose input
        # distribution (see TmemTranspose16x32 / load_subtile_raw_acc).
        token_32_64_tmem_trans = TmemTranspose32x32Inplace(
            tmem_subtile_tensor.iterator,
            reg_tensor_top=gate_token_32_64,
            reg_tensor_bot=up_token_32_64,
        )

        # Transpose output: each lane holds (token_1, intermediate_16); tmem_dp
        # = lane_idx (token), tmem_col = elem_idx (intermediate output idx).
        gate_token_32_64_trans_pre_act, up_token_32_64_trans_pre_act = token_32_64_tmem_trans.from_r1_perm_until_last_store(
        )

        token_32_64_pre_quant = self.alpha_swiglu_clamp(
            gate_token_32_64_trans_pre_act,
            up_token_32_64_trans_pre_act,
            alpha_val,
        )

        token_0_32_tmem_trans = TmemTranspose16x32(
            tmem_subtile_tensor.iterator,
            Region.Top,
            reg_tensor=token_0_32_pre_quant_pre_trans,
        )
        token_0_32_pre_quant = token_0_32_tmem_trans.from_r1_perm_until_last_store(
        )

        # Step 2: Quant
        self.nvfp4_quant(work_tile_info=work_tile_info,
                         two_token=(token_0_32_pre_quant,
                                    token_32_64_pre_quant),
                         topk_scores=topk_scores,
                         norm_const=norm_const,
                         intermediate_output_size=cute.size(fc1_output, 1),
                         fc1_output_sf=fc1_output_sf,
                         subtile_idx=subtile_idx)

        # Step 3: TMASTG
        # (token_64, intermeidate_64)
        fc1_smem = self.smem_tensor[None, None, subtile_idx]
        # (token, intermediate_down, l=1) -> (cta_token, cta_intermediate_down)
        fc1_gmem_cta_view = cute.flat_divide(
            fc1_output,
            (self.cta_tile_n, self.cta_tile_m // 2),
        )[None, None, work_tile_info.tile_n_idx, work_tile_info.tile_m_idx, 0]
        # (cta_token, cta_intermediate_down) -> (token_64, intermediate_64)
        fc1_gmem_subtile_view = cute.flat_divide(
            fc1_gmem_cta_view,
            (self._EpilogueTokenTileSize,
             self._EpilogueFc1IntermediateDownTileSize))[None, None,
                                                         subtile_idx, 0]
        tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
            self.fc1_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(fc1_smem, 0, 2),
            cute.group_modes(fc1_gmem_subtile_view, 0, 2),
        )

        subtile_bar_id = subtile_idx + cutlass.Int32(
            SwapABSwigluFp4Epilogue._EpilogueAsyncBarIdBase)
        tma_ready_to_read_smem_named_barrier = pipeline.NamedBarrier(
            barrier_id=subtile_bar_id,
            num_threads=self._EpilogueWarpCnt * 32,
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        if self.warp_idx == subtile_idx:
            tma_ready_to_read_smem_named_barrier.arrive_and_wait()
            with cute.arch.elect_one():
                # if work_tile_info.tile_m_idx * (self.cta_tile_m // 2) < cute.size(fc1_output, 1):
                cute.copy(self.fc1_tma_atom, tma_smem_src, tma_gmem_dst)
        else:
            tma_ready_to_read_smem_named_barrier.arrive()

    @cute.jit
    def alpha_swiglu_clamp(
        self,
        gate_rmem: cute.
        Tensor,  # Raw fc1 acc (pre-dequant); even-size 1D fp32 rmem
        up_rmem: cute.
        Tensor,  # Raw fc1 acc (pre-dequant); even-size 1D fp32 rmem
        alpha_val: Optional[cutlass.Float32]
    ) -> cute.Tensor:
        # ── Input contract checks (compile-time): fp32, 1D, even-count, rmem ──
        # Wrapped in const_expr so the DSL evaluates them at trace time and the
        # raise fires during compilation rather than emitting a runtime branch.
        for _name, _t in (("gate_rmem", gate_rmem), ("up_rmem", up_rmem)):
            if cutlass.const_expr(_t.element_type is not cutlass.Float32):
                raise TypeError(
                    f"alpha_swiglu_clamp: {_name} must be Float32, got {_t.element_type}"
                )
            if cutlass.const_expr(_t.memspace != AddressSpace.rmem):
                raise ValueError(
                    f"alpha_swiglu_clamp: {_name} must be a register (rmem) tensor, "
                    f"got address space {_t.memspace}")
            if cutlass.const_expr(cute.rank(_t) != 1):
                raise ValueError(
                    f"alpha_swiglu_clamp: {_name} must be 1D, got rank {cute.rank(_t)}"
                )
            if cutlass.const_expr(cute.size(_t) % 2 != 0):
                raise ValueError(
                    f"alpha_swiglu_clamp: {_name} element count must be even, got {cute.size(_t)}"
                )
        if cutlass.const_expr(cute.size(gate_rmem) != cute.size(up_rmem)):
            raise ValueError(
                "alpha_swiglu_clamp: gate_rmem and up_rmem must have equal size, got "
                f"{cute.size(gate_rmem)} vs {cute.size(up_rmem)}")

        # gate_rmem / up_rmem are the RAW fc1 fp32 accumulator (pre-dequant).
        # Order follows the NVFP4 -> fp32 -> SwiGLU contract and MUST be:
        #
        #   1. dequant:  gate = alpha * gate_raw ; up = alpha * up_raw
        #      (alpha = expert-wise global scale on the acc; None => alpha == 1.)
        #   2. clamp the DEQUANTED (real) values, gpt-oss ``_apply_gate`` style:
        #        gate = min(gate, +limit)           (upper bound only)
        #        up   = clamp(up, -limit, +limit)   (symmetric)
        #   3. swiglu:   out = up * gate * sigmoid(gate)
        #                sigmoid(x) = rcp(1 + exp2(-x * log2e))
        #
        # The symmetric up-clamp is a single ``min.xorsign.abs.f32`` (magnitude
        # min(|up|, limit), sign = sign(up)^sign(limit) = sign(up) since limit>=0);
        # the gate-clamp is a plain ``min.f32``. ``.xorsign.abs`` has no f32x2 form,
        # so dequant+clamp run scalar while the swiglu core stays packed f32x2.
        n = cute.size(gate_rmem)
        out = cute.make_rmem_tensor((n, ), cutlass.Float32)
        log2_e = 1.4426950408889634

        neg_log2e_pair = (
            cutlass.Float32(-log2_e),
            cutlass.Float32(-log2_e),
        )
        one_pair = (cutlass.Float32(1.0), cutlass.Float32(1.0))
        if cutlass.const_expr(self.gate_up_clamp is not None):
            limit = cutlass.Float32(self.gate_up_clamp)

        for i in cutlass.range_constexpr(0, n, 2):
            g0 = gate_rmem[i]
            g1 = gate_rmem[i + 1]
            u0 = up_rmem[i]
            u1 = up_rmem[i + 1]

            # 1) dequant raw acc to real values (skip entirely when alpha is None).
            if cutlass.const_expr(alpha_val is not None):
                alpha_pair = (alpha_val, alpha_val)
                g0, g1 = cute.arch.mul_packed_f32x2((g0, g1), alpha_pair)
                u0, u1 = cute.arch.mul_packed_f32x2((u0, u1), alpha_pair)

            # 2) clamp the real values (skip when no clamp configured).
            if cutlass.const_expr(self.gate_up_clamp is not None):
                # gate upper-clamp: min(gate, +limit)
                g0 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [g0.ir_value(), limit.ir_value()],
                        "min.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    ))
                g1 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [g1.ir_value(), limit.ir_value()],
                        "min.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    ))
                # up symmetric-clamp: clamp(up, -limit, +limit) in one instruction
                u0 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [u0.ir_value(), limit.ir_value()],
                        "min.xorsign.abs.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    ))
                u1 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [u1.ir_value(), limit.ir_value()],
                        "min.xorsign.abs.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    ))

            # 3) swiglu on the dequanted (and clamped) real values:
            #    out = up * gate * sigmoid(gate)
            ug = cute.arch.mul_packed_f32x2((u0, u1), (g0, g1))
            neg_g_log2e = cute.arch.mul_packed_f32x2((g0, g1), neg_log2e_pair)
            exp_pair = (
                cute.math.exp2(neg_g_log2e[0], fastmath=True),
                cute.math.exp2(neg_g_log2e[1], fastmath=True),
            )
            one_plus_exp = cute.arch.add_packed_f32x2(exp_pair, one_pair)
            sigmoid_pair = (
                cute.arch.rcp_approx(one_plus_exp[0]),
                cute.arch.rcp_approx(one_plus_exp[1]),
            )
            out_pair = cute.arch.mul_packed_f32x2(ug, sigmoid_pair)

            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]

        return out

    @cute.jit
    def nvfp4_quant(
        self,
        work_tile_info: MoEWorkTileInfo,
        two_token: Tuple[
            cute.Tensor, cute.
            Tensor],  # two rmem tensor, each fp32 @ (token_1, intermediate_16)
        topk_scores: Optional[Tuple[cutlass.Float32, cutlass.Float32]],
        norm_const: Optional[cutlass.Float32],
        intermediate_output_size: cutlass.Int32,
        fc1_output_sf: cute.
        Tensor,  # MoE domain (token_this_rank, intermediate_down, 1)
        subtile_idx: cutlass.Int32):
        # ``two_token`` are the two post-swiglu, transposed token rmem tensors;
        # each lane holds one token's ``sf_vec_size`` (=16, one NVFP4 SF block)
        # intermediate-output values.  half 0 -> token (lane), half 1 -> (lane+32).
        #
        # Per token (ported from PostSwigluHalf._gen_sfc_quantize + stg_sfc + r2s):
        #   1. (Path A) pre-multiply topk weight into the values, if present.
        #   2. absmax over the (weighted) block.
        #   3. sfc = absmax * (1/6) * norm_const  -> E4M3 scale factor.
        #   4. acc_scale = norm_const * rcp(sfc), capped at FP32_MAX, with a
        #      sfc==0 guard mask; scale the values by acc_scale.
        #   5. write the E4M3 sfc to fc1_output_sf[token, intermediate_idx, 0]
        #      (plain scalar store; predicated unless statically in-bound).
        #   6. cvt the scaled values to NVFP4 and STS.64 into this subtile's
        #      shared output stage.
        # norm_const is treated like alpha_val: None => behaves as 1.0 (factors
        # const-elided, not multiplied by 1.0).
        n = cute.size(two_token[0])
        # The core block quant (amax + e4m3 sfc + capped/masked acc_scale + e2m1
        # cvt) is QuantImpl's job; this method still owns the topk pre-multiply,
        # the sfc store, and the STS.64 into the shared output stage.
        quant = QuantImpl("nvfp4", "regs_in_thread")

        intermediate_idx = (work_tile_info.tile_m_idx * (self.cta_tile_m // 2) +
                            self.warp_idx * Nvfp4BlockSize)
        subtile_token_start = (work_tile_info.tile_n_idx * self.cta_tile_n +
                               subtile_idx * self._EpilogueTokenTileSize)
        token_idx_pair = (
            subtile_token_start + self.lane_idx,
            subtile_token_start + self.lane_idx + 32,
        )

        # This subtile's (token, intermediate) shared output stage, tiled into
        # (1, 16) blocks so each thread's 16 NVFP4 cells slice out directly
        # (zipped_divide + slice; avoids the ambiguous local_tile surface).
        smem_stage = self.smem_tensor[None, None, subtile_idx]
        # (token_64, intermediate_down_64) -> ((1, 16), (token_tile_size, warp_cnt))
        smem_tiled = cute.zipped_divide(smem_stage, (1, Nvfp4BlockSize))

        fp4_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float4E2M1FN,
            num_bits_per_copy=64,
        )

        for half in cutlass.range_constexpr(2):
            tok = two_token[half]

            # 1) topk-weight pre-multiply (Path A) into a weighted scratch.
            weighted = cute.make_rmem_tensor((n, ), cutlass.Float32)
            if cutlass.const_expr(topk_scores is not None):
                topk_pair = (topk_scores[half], topk_scores[half])
                for i in cutlass.range_constexpr(0, n, 2):
                    w0, w1 = cute.arch.mul_packed_f32x2((tok[i], tok[i + 1]),
                                                        topk_pair)
                    weighted[i] = w0
                    weighted[i + 1] = w1
            else:
                for i in cutlass.range_constexpr(0, n):
                    weighted[i] = tok[i]

            # 2) Core block quant: amax + e4m3 sfc + capped/masked acc_scale +
            #    e2m1 cvt. One 16-wide block -> one e2m1 reg tensor + one sfc.
            fp4_regs, sfc_regs = quant(weighted, norm_const=norm_const)
            sfc_e4m3 = sfc_regs[0]

            # 3) scale-factor store (predicate const-elided when statically
            #    in-bound, mirroring signal_fc1_done's intermediate predicate).
            if cutlass.const_expr(self.static_expert_shape is None
                                  or self.intermediate_downproj %
                                  self.cluster_tile_intermediate_downproj != 0):
                if intermediate_idx < intermediate_output_size:
                    fc1_output_sf[token_idx_pair[half], intermediate_idx,
                                  0] = sfc_e4m3
            else:
                fc1_output_sf[token_idx_pair[half], intermediate_idx,
                              0] = sfc_e4m3

            # 4) STS.64 the e2m1 into this subtile's shared output stage.
            # ((1, 16), (token_tile_size, warp_cnt)) -> (16)
            smem_thread_row = smem_tiled[(0, None), (self.lane_idx + 32 * half,
                                                     self.warp_idx)]
            cute.copy(
                fp4_copy_atom,
                cute.coalesce(fp4_regs),
                cute.coalesce(smem_thread_row),
            )


@dataclasses.dataclass(frozen=True)
class Fc2ProcessPipeline():
    tmem_acc_load: Callable
    f2fp: Callable
    post_f2fp_reorder: Callable
    store_function: Callable
    # Kept as a finer-grained, elem-level reading aid for the store-out layout
    # (never evaluated); ``store_out_mapping`` is the per-issue form that the
    # router actually evaluates at runtime to drive metadata / pointer math.
    fc2_cta_tile_contract: Contract
    store_out_mapping: Contract  # data plane, per-issue
    require_tmem_trans: bool
    # SF plane per-issue mapping; None for the bf16 (unquantized) paths.
    sf_store_out_mapping: Optional[Contract] = None


# Device only object
class SwapABFc2Epilogue(_ImmutableAfterInit):

    def __init__(
        self,
        base: SwapABSwigluFp4Epilogue,
        tidx: cutlass.Int32,
        epi_smem_storage,
        fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
        token_comm_args: TokenCommArgs,
        optional_epi_args: NvFp4OptinalEpiArgs,
    ):
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        self.fc2_output = fc2_output
        self.token_comm_args = token_comm_args
        self.optional_epi_args = optional_epi_args
        if cutlass.const_expr(base.fc2_use_bulk):
            wire_dtype = base.combine_format.act_dtype
            fc2_smem_rows = 32
            if cutlass.const_expr(fc2_smem_rows *
                                  base._EpilogueFc2HiddenTileSize *
                                  wire_dtype.width // 8 > base.epi_smem_bytes):
                raise ValueError("fc2 UBLK data smem exceeds epi_smem budget.")
            self.smem_tensor = cute.make_tensor(
                cute.recast_ptr(
                    epi_smem_storage.epi_smem.data_ptr(),
                    dtype=wire_dtype,
                ),
                cute.make_layout(
                    (fc2_smem_rows, base._EpilogueFc2HiddenTileSize),
                    stride=(base._EpilogueFc2HiddenTileSize, 1),
                ),
            )
            self.process_pipeline = make_fc2_ublk_process_pipeline(
                combine_format=base.combine_format,
                cta_token_tile_size=base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            )
        else:
            self.smem_tensor = None
            if cutlass.const_expr(base.reduce_topk_in_kernel):
                self.process_pipeline = make_fc2_redg_process_pipeline(
                    combine_format=base.combine_format,
                    cta_token_tile_size=base.cta_tile_n,
                    cta_hidden_tile_size=base.cta_tile_m,
                )
            else:
                self.process_pipeline = make_fc2_stg_process_pipeline(
                    combine_format=base.combine_format,
                    cta_token_tile_size=base.cta_tile_n,
                    cta_hidden_tile_size=base.cta_tile_m,
                )
        self._freeze()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        # See SwapABFc1Epilogue.__extract_mlir_values__: this helper carries
        # only loop-invariant Python context.  It intentionally serializes no
        # MLIR values, so changing it to store loop-carried state would be a
        # correctness bug.
        return []

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "SwapABFc2Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc2_done(self, work_tile_info, next_work_tile_info,
                        flag_tracker):
        publish: cutlass.Constexpr = (self.token_back_by_dispatch
                                      or self.combine_format.is_quantized)
        if cutlass.const_expr(publish):
            flag_addr = (self.token_comm_args.fc2_done_counter.iterator +
                         work_tile_info.expert_idx).toint()
        else:
            flag_addr = Int64(0)
        no_fire: cutlass.Constexpr = not publish
        return flag_tracker.accumulate(next_work_tile_info.phase,
                                       self.fc2_epi_flag_batch, flag_addr,
                                       no_fire)

    @cute.jit
    def _make_output_router(
        self,
        work_tile_info: MoEWorkTileInfo,
    ) -> "Fc2OutputRouter":
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row +
            work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n))
        hidden_base_this_cta_tile = (work_tile_info.tile_m_idx *
                                     cutlass.Int32(self.cta_tile_m))
        valid_hidden_this_cta_tile = (cutlass.Int32(self.fc2_output.shape[2]) -
                                      hidden_base_this_cta_tile)
        if valid_hidden_this_cta_tile < 0:
            valid_hidden_this_cta_tile = 0
        if valid_hidden_this_cta_tile > self._EpilogueFc2HiddenTileSize:
            valid_hidden_this_cta_tile = self._EpilogueFc2HiddenTileSize

        metadata_u32 = None
        peer_rank_ptr_mapper = None
        data_token_base = task_tile_data_row_start
        if cutlass.const_expr(self.token_comm_args is not None
                              and not self.token_back_by_dispatch):
            metadata_u32 = cute.domain_offset(
                (task_tile_data_row_start, 0),
                cute.recast_tensor(
                    self.token_comm_args.token_src_metadata,
                    cutlass.Uint32,
                ),
            )
            peer_rank_ptr_mapper = self.token_comm_args.peer_rank_ptr_mapper
            data_token_base = None

        if cutlass.const_expr(self.combine_format.is_quantized):
            base_outputs = (self.fc2_output, self.token_comm_args.fc2_output_sf)
            token_bases = (data_token_base, task_tile_data_row_start)
            output_mappings = (
                self.process_pipeline.store_out_mapping,
                self.process_pipeline.sf_store_out_mapping,
            )
        else:
            base_outputs = self.fc2_output
            token_bases = data_token_base
            output_mappings = self.process_pipeline.store_out_mapping

        return Fc2OutputRouter(
            metadata=metadata_u32,
            token_bases=token_bases,
            base_outputs=base_outputs,
            hidden_base_this_cta_tile=hidden_base_this_cta_tile,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden_this_cta_tile=valid_hidden_this_cta_tile,
            reduce_topk_in_kernel=self.reduce_topk_in_kernel,
            output_mappings=output_mappings,
            epi_tid=self.tidx,
            combine_format=self.combine_format,
        ).prefetch()

    @cute.jit
    def __call__(
        self,
        work_tile_info: MoEWorkTileInfo,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
    ):
        # subtile-irrelevant hoist: fc2 alpha scales raw fc2 accumulators before f2fp.
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[
                work_tile_info.expert_idx]
        else:
            alpha_val = None
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        fc2_output_router = self._make_output_router(work_tile_info)
        # (cta_tile_m, cta_tile_n) -> (epi_tile_m, epi_tile_n, iters)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc2HiddenTileSize,
                              self._EpilogueTokenTileSize))[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        iket.range_push("fc2_epi")
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        # Overlap path preloads two subtiles before releasing acc TMEM.
        unroll_tile_cnt = 2 if cutlass.const_expr(
            self.overlapping_accum
            and self.process_pipeline.require_tmem_trans) else 0
        remain_subtile_cnt = self.subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (cutlass.Int32(self.subtile_cnt) -
                                 is_odd_turn) % cutlass.Int32(self.subtile_cnt)
            subtile_idx_second = (cutlass.Int32(self.subtile_cnt + 1) -
                                  is_odd_turn) % cutlass.Int32(self.subtile_cnt)

            # preload_subtile_first: subtile_idx_first's raw PRE-transpose acc, LDTM'd by
            # all 128 epi threads into 4 reg tensors == the 4 quadrants of the subtile's
            # (128 tmem_dp x 64 tmem_col) footprint. Only these raw-TMEM offsets are
            # guaranteed:
            #   reg[0]/reg[1], reg[2]/reg[3] : top vs bot      -> 16 apart in tmem_dp
            #   reg[0]/reg[2], reg[1]/reg[3] : 1st vs 2nd half -> 32 apart in tmem_col
            # (so reg[0..1] = the first 128x32, reg[2..3] = the second 128x32 of the 128x64.)
            # The per-lane (lane_idx, elem_idx) -> (tmem_dp, tmem_col) layout INSIDE each
            # reg tensor is opaque -- do not assume it; it only becomes well-defined once
            # the tmem transpose consumes them.
            preload_subtile_first: Tuple[
                cute.Tensor, cute.Tensor, cute.Tensor,
                cute.Tensor] = _TmemTranspose16x32Core.load_subtile_raw_acc(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None,
                                                      subtile_idx_first])

            # Release acc to next MMA unconditionally.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            # preload_subtile_second: same 128 tmem_dp x 64 tmem_col footprint, but for
            # subtile_idx_second (the other token subtile, not the 2nd col-half). Same
            # quadrant/offset invariants and opaque per-lane layout as above.
            preload_subtile_second: Tuple[
                cute.Tensor, cute.Tensor, cute.Tensor,
                cute.Tensor] = _TmemTranspose16x32Core.load_subtile_raw_acc(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None,
                                                      subtile_idx_second])

            # Both unrolled subtiles borrow tmem_subtile_second as workspace.
            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(
                        self._EpilogueTokenTileSize) < valid_tokens:
                    self.run_subtile(
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                            None, None, subtile_idx_second],
                        preload_acc=preload_pair[i],
                        fc2_output_router=fc2_output_router,
                        alpha_val=alpha_val,
                        release_after_ldtm=False,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                    )

        if cutlass.const_expr(self.overlapping_accum and unroll_tile_cnt == 0):
            release_after_ldtm = True
        else:
            release_after_ldtm = False
        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            # for i in cutlass.range_constexpr(remain_subtile_cnt):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self.overlapping_accum):
                subtile_idx = (cutlass.Int32(real_i + self.subtile_cnt) -
                               is_odd_turn) % cutlass.Int32(self.subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)

            if subtile_idx * cutlass.Int32(
                    self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                        None, None, subtile_idx],
                    preload_acc=None,
                    fc2_output_router=fc2_output_router,
                    alpha_val=alpha_val,
                    release_after_ldtm=release_after_ldtm,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
                release_after_ldtm = False

        # Non-overlap-path release: at the natural task-tile boundary.
        if cutlass.const_expr(not self.overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        subtile_idx: cutlass.Int32,
        # (hidden_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        preload_acc: Optional[Tuple[cute.Tensor, cute.Tensor, cute.Tensor,
                                    cute.Tensor]],
        fc2_output_router: "Fc2OutputRouter",
        alpha_val: Optional[cutlass.Float32],
        release_after_ldtm: Union[cutlass.Boolean, bool],
        acc_pipeline,
        acc_consumer_state,
    ):
        process_pipeline = self.process_pipeline
        if cutlass.const_expr(preload_acc is None):
            loaded = process_pipeline.tmem_acc_load(
                tmem_subtile_tensor=tmem_subtile_tensor,
                epi=self,
            )
            if release_after_ldtm:
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
        else:
            loaded = preload_acc

        casted = process_pipeline.f2fp(
            *loaded,
            alpha_val=alpha_val,
        )
        # reorder returns a bare RMEM fragment in the store's expected pre-store
        # distribution; reorder + store are paired 1:1 inside the pipeline.
        pre_store = process_pipeline.post_f2fp_reorder(
            casted=casted,
            tmem_subtile_view=tmem_subtile_tensor,
        )
        process_pipeline.store_function(
            epi=self,
            subtile=pre_store,
            subtile_idx=subtile_idx,
            fc2_output_router=fc2_output_router,
        )


@dataclasses.dataclass(frozen=True)
class Fc2OutputRouter:
    # (token, 3), 3 -> rank_idx, token_idx, top_k
    # Later this will be changed to (token, 2), where top_k and rank_idx will be fused into 32bit.
    # If metadata is None then this is a local write.
    metadata: Optional[cute.Tensor]
    # token + possible sf
    token_bases: Union[Tuple[Optional[cutlass.Int32], cutlass.Int32],
                       Optional[cutlass.Int32]]
    base_outputs: Union[Tuple[cute.Tensor, cute.Tensor],
                        cute.Tensor]  # (token, topk, hidden)
    hidden_base_this_cta_tile: Union[cutlass.Int32, int]
    peer_rank_ptr_mapper: Optional[SymBufferDeviceBase]
    valid_tokens_this_cta_tile: cutlass.Int32
    valid_hidden_this_cta_tile: Union[cutlass.Int32, int]
    reduce_topk_in_kernel: bool
    # Per-issue (epi_tid, iter_idx) -> (token_cta_tile, hidden_cta_tile). Data
    # mapping, or (data mapping, sf mapping) when quantized.
    output_mappings: Union[Tuple[Contract, Contract], Contract]
    epi_tid: cutlass.Int32
    combine_format: CombineFormat
    # After metadata prefetch
    dst_ptrs: Optional[
        cute.
        Tensor] = None  # i64 x (copy_iters_this_thread_cta_tile), fundamentally the pointers.
    valid: Optional[cute.Tensor] = None  # (copy_iters_this_thread_cta_tile)

    @property
    def data_output(self) -> cute.Tensor:
        return self.base_outputs[0] if isinstance(self.base_outputs,
                                                  tuple) else self.base_outputs

    @property
    def sf_output(self) -> Optional[cute.Tensor]:
        # Present iff quantized; (pool_token, 1, hidden // sf_vec) rank-local.
        return self.base_outputs[1] if isinstance(self.base_outputs,
                                                  tuple) else None

    @property
    def data_token_base(self) -> Optional[cutlass.Int32]:
        return self.token_bases[0] if isinstance(self.token_bases,
                                                 tuple) else self.token_bases

    @property
    def sf_token_base(self) -> Optional[cutlass.Int32]:
        return self.token_bases[1] if isinstance(self.token_bases,
                                                 tuple) else None

    @property
    def data_mapping(self) -> Contract:
        return self.output_mappings[0] if isinstance(
            self.output_mappings, tuple) else self.output_mappings

    @property
    def sf_mapping(self) -> Optional[Contract]:
        return self.output_mappings[1] if isinstance(self.output_mappings,
                                                     tuple) else None

    def __post_init__(self) -> None:
        if (self.metadata is None) == (self.data_token_base is None):
            raise ValueError(
                "Fc2OutputRouter requires exactly one of metadata or "
                "a (data) token base.")
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputRouter requires peer_rank_ptr_mapper iff metadata is set."
            )
        if self.reduce_topk_in_kernel and self.metadata is None:
            raise ValueError(
                "Fc2OutputRouter reduce_topk_in_kernel requires metadata routing."
            )

    @cute.jit
    def prefetch(self) -> "Fc2OutputRouter":
        # Only the metadata (comm) path prefetches a pointer array: its
        # metadata-derived address has long-latency LDGs worth issuing early.
        # The local (no-comm) path computes its affine address on demand in
        # get_dst() -- no array, hence no runtime-indexed local-memory spill.
        if cutlass.const_expr(self.metadata is None):
            return self
        iter_axis = self.data_mapping.domain.names.index("iter_idx")
        copy_iters: cutlass.Constexpr[int] = self.data_mapping.domain.sizes[
            iter_axis]

        valid = cute.make_rmem_tensor((copy_iters, ), cutlass.Int32)
        dst_ptrs = cute.make_rmem_tensor((copy_iters, ), cutlass.Int64)

        # Compiler should be able to optimize the same token_copy_group's offset add. (Fundamental cse + strength_reduce)
        # We should check the SASS to ensure this happens.
        for iter_idx in cutlass.range_constexpr(copy_iters):
            coord = eval_function_mapping(
                self.data_mapping,
                epi_tid=self.epi_tid,
                iter_idx=iter_idx,
            )
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])

            valid[iter_idx] = cutlass.Int32(0)
            dst_ptrs[iter_idx] = cutlass.Int64(0)

            token_valid = token_in_tile < self.valid_tokens_this_cta_tile
            hidden_valid = hidden_in_tile < cutlass.Int32(
                self.valid_hidden_this_cta_tile)
            if token_valid and hidden_valid:
                valid[iter_idx] = cutlass.Int32(1)
                if cutlass.const_expr(self.metadata is None):
                    dst_tokens = self.data_token_base + token_in_tile
                    dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                    # Int64 token coord: dst_tokens*K*H overflows int32 once
                    # T*K*H exceeds 2^31 (data_output is (token, topk, hidden)).
                    dst_ptrs[iter_idx] = self.data_output[
                        Int64(dst_tokens), None, dst_hidden].iterator.toint()

                else:
                    md = TokenSrcMetadata.load(self.metadata.iterator.toint() +
                                               Int64(token_in_tile) *
                                               Int64(TokenSrcMetadata.nbytes))
                    dst_rank = md.src_rank
                    dst_token = md.src_token
                    dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                    if cutlass.const_expr(not self.reduce_topk_in_kernel):
                        dst_topk = md.src_topk
                    else:
                        dst_topk = 0
                    # Int64 token coord: domain_offset on (token, topk, hidden)
                    # computes dst_token*K*H, which overflows int32 once T*K*H > 2^31.
                    dst_ptrs[
                        iter_idx] = self.peer_rank_ptr_mapper.ptr_map_to_rank(
                            cute.domain_offset(
                                (Int64(dst_token), dst_topk, dst_hidden),
                                self.data_output).iterator,
                            dst_rank,
                            byte_align=32).toint()

        return dataclasses.replace(
            self,
            dst_ptrs=dst_ptrs,
            valid=valid,
        )

    @cute.jit
    def get_data_dst(
        self,
        iter_idx: Union[int, cutlass.Int32],
    ) -> Tuple[cute.Pointer, cutlass.Int32]:
        """Per-issue DATA destination: gmem pointer + validity predicate.

        The router owns ``data_output`` so the caller never re-assembles a
        pointer from a raw int; it just builds its own copy tensor (STG) or
        feeds the pointer to inline asm (REDG/UBLK).

        Alignment is unified at 32 B: only STG feeds this pointer to a real
        ``cute.copy`` (256 b vector store, genuinely 32 B aligned); REDG/UBLK
        only ``ptrtoint`` it for inline-asm issue, where the hint is inert.
        """
        if cutlass.const_expr(self.metadata is None):
            # no-comm: on-demand affine address (no prefetched array). The
            # invariant base hoists out of the caller's loop via CSE; a
            # constexpr iter folds the per-issue offset into the store.
            coord = eval_function_mapping(
                self.data_mapping,
                epi_tid=self.epi_tid,
                iter_idx=iter_idx,
            )
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
            pred = cutlass.Int32(0)
            addr = cutlass.Int64(0)
            if (token_in_tile < self.valid_tokens_this_cta_tile
                    and hidden_in_tile < cutlass.Int32(
                        self.valid_hidden_this_cta_tile)):
                pred = cutlass.Int32(1)
                dst_tokens = self.data_token_base + token_in_tile
                dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                # Int64 token coord: dst_tokens*K*H overflows int32 once T*K*H > 2^31.
                addr = self.data_output[Int64(dst_tokens), None,
                                        dst_hidden].iterator.toint()
        else:
            # comm: read the pointer / validity prefetched by prefetch().
            addr = self.dst_ptrs[iter_idx]
            pred = self.valid[iter_idx]
        ptr = cute.make_ptr(
            self.data_output.element_type,
            addr,
            AddressSpace.gmem,
            assumed_align=32,
        )
        return ptr, pred

    @cute.jit
    def get_sf_dst(
        self,
        iter_idx: Union[int, cutlass.Int32],
    ) -> Tuple[cute.Pointer, cutlass.Int32]:
        """Per-issue SF destination: rank-local gmem pointer + validity predicate.

        SF never goes to a peer (it is staged locally and pushed token-contiguously
        by the dispatch / standalone warps), so this is always the affine local
        address -- no metadata routing, no prefetch. ``sf_output`` is the broadcast
        plane ``(pool_token, 1, (sf_vec, hidden//sf_vec)):(., ., (0, 1))``, so the
        logical hidden coordinate folds to its scale block on indexing.
        """
        coord = eval_function_mapping(
            self.sf_mapping,
            epi_tid=self.epi_tid,
            iter_idx=iter_idx,
        )
        token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
        hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
        pred = cutlass.Int32(0)
        addr = cutlass.Int64(0)
        if (token_in_tile < self.valid_tokens_this_cta_tile and hidden_in_tile
                < cutlass.Int32(self.valid_hidden_this_cta_tile)):
            pred = cutlass.Int32(1)
            sf_row = self.sf_token_base + token_in_tile
            sf_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
            addr = self.sf_output[Int64(sf_row), None,
                                  sf_hidden].iterator.toint()
        # Per-block scale offsets are element-granular; claim the scale dtype's
        # natural element alignment (e8m0 1 B / bf16 2 B).
        sf_ptr = cute.make_ptr(
            self.sf_output.element_type,
            addr,
            AddressSpace.gmem,
            assumed_align=4,
        )
        return sf_ptr, pred


def make_fc2_stg_cta_store_out_contract(combine_format: CombineFormat,
                                        cta_token_tile_size: int,
                                        cta_hidden_tile_size: int):
    assert (cta_hidden_tile_size == 128)
    assert (cta_token_tile_size % 64 == 0)
    wire_dtype = combine_format.act_dtype
    assert wire_dtype.width in (4, 8,
                                16), "fc2 STG wire dtype must be fp4/fp8/bf16."
    elems_per_stg = min(256 // wire_dtype.width, 32)
    stgs_per_hidden32 = 32 // elems_per_stg
    fundamental_mapping = Contract(
        domain=Space(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                       (cta_token_tile_size, cta_hidden_tile_size)),
        mapping=FunctionMapping(
            lambda epi_tid, elem_idx: {
                "token_in_cta_tile": epi_tid % 32 + elem_idx // 32 * 32,
                "hidden_in_cta_tile": elem_idx % 32 + epi_tid // 32 * 32,
            }))
    store_out_mapping = Contract(
        domain=Space(("epi_tid", "iter_idx"),
                     (128, stgs_per_hidden32 * cta_token_tile_size // 32)),
        codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                       (cta_token_tile_size, cta_hidden_tile_size)),
        mapping=FunctionMapping(
            lambda epi_tid, iter_idx: {
                "token_in_cta_tile":
                epi_tid % 32 + iter_idx // stgs_per_hidden32 * 32,
                "hidden_in_cta_tile": (iter_idx % stgs_per_hidden32) *
                elems_per_stg + epi_tid // 32 * 32,
            }))
    sf_store_out_mapping = None
    if combine_format.is_quantized:

        def stg_sf_mapping(epi_tid, iter_idx):
            lane = epi_tid % 32
            warp = epi_tid // 32
            return {
                "token_in_cta_tile": lane + iter_idx * 32,
                "hidden_in_cta_tile": warp * 32,
            }

        sf_store_out_mapping = Contract(
            domain=Space(("epi_tid", "iter_idx"),
                         (128, cta_token_tile_size // 32)),
            codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                           (cta_token_tile_size, cta_hidden_tile_size)),
            mapping=FunctionMapping(stg_sf_mapping),
        )
    return store_out_mapping, sf_store_out_mapping, fundamental_mapping


def make_fc2_redg_cta_store_out_contract(combine_format: CombineFormat,
                                         cta_token_tile_size: int,
                                         cta_hidden_tile_size: int):
    assert (cta_hidden_tile_size == 128)
    assert (cta_token_tile_size % 64 == 0)
    # In-kernel reduce is bf16-only and never quantized, so there is no SF plane.
    assert (combine_format.act_dtype.width == 16)
    assert not combine_format.is_quantized
    fundamental_mapping = Contract(
        domain=Space(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                       (cta_token_tile_size, cta_hidden_tile_size)),
        mapping=FunctionMapping(
            lambda epi_tid, elem_idx: {
                "token_in_cta_tile":
                (((elem_idx // 4) // 16) * 64 + (((elem_idx // 4) % 16) // 8) *
                 32 + (((elem_idx // 4) % 8) // 4) * 16 + ((
                     (elem_idx // 4) % 4) % 2) * 8 + (epi_tid % 32) // 4),
                "hidden_in_cta_tile": (
                    (epi_tid // 32) * 32 + (epi_tid % 4) * 4 + ((
                        (elem_idx // 4) % 4) // 2) * 16 + elem_idx % 4),
            }))
    # SIMT REDG emits one 8B red.v2.bf16x2 per 4 hidden elements.  Each
    # 64-token subtile contributes two token rows per lane and 8 hidden
    # segments per token row.
    store_out_mapping = Contract(
        domain=Space(("epi_tid", "iter_idx"),
                     (128, cta_token_tile_size // 64 * 16)),
        codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                       (cta_token_tile_size, cta_hidden_tile_size)),
        mapping=FunctionMapping(
            lambda epi_tid, iter_idx: {
                "token_in_cta_tile": ((iter_idx // 16) * 64 + (
                    (iter_idx % 16) // 8) * 32 + ((iter_idx % 8) // 4) * 16 + (
                        (iter_idx % 4) % 2) * 8 + (epi_tid % 32) // 4),
                "hidden_in_cta_tile": ((epi_tid // 32) * 32 + (epi_tid % 4) * 4
                                       + ((iter_idx % 4) // 2) * 16),
            }))
    return store_out_mapping, None, fundamental_mapping


def make_fc2_ublk_store_out_contract(combine_format: CombineFormat,
                                     cta_token_tile_size: int,
                                     cta_hidden_tile_size: int):
    assert (cta_hidden_tile_size == 128)
    assert (cta_token_tile_size % 64 == 0)
    # UBLK pushes whole hidden rows by byte count, so the token/hidden mapping
    # is element-indexed and dtype-independent (wire dtype only sets copy bytes).
    assert combine_format.act_dtype.width in (
        4, 8, 16), "fc2 UBLK wire dtype must be fp4/fp8/bf16."
    assert (cta_token_tile_size <= 256)
    max_token_cta_tile = 256
    fundamental_mapping = Contract(
        domain=Space(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                       (max_token_cta_tile, cta_hidden_tile_size)),
        mapping=FunctionMapping(
            lambda epi_tid, elem_idx: {
                "token_in_cta_tile":
                elem_idx // cta_hidden_tile_size * 32 + epi_tid % 8 + epi_tid //
                32 * 8 + ((epi_tid % 32) // 8) * 64,
                "hidden_in_cta_tile":
                elem_idx % cta_hidden_tile_size,
            }))
    store_out_mapping = Contract(
        domain=Space(("epi_tid", "iter_idx"), (128, 2)),
        codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                       (max_token_cta_tile, cta_hidden_tile_size)),
        mapping=FunctionMapping(
            lambda epi_tid, iter_idx: {
                "token_in_cta_tile":
                iter_idx * 32 + epi_tid % 8 + epi_tid // 32 * 8 +
                ((epi_tid % 32) // 8) * 64,
                "hidden_in_cta_tile":
                0,
            }))
    # SF mapping: each lane owns one hidden across a 64-token subtile, and the
    # sf_vec lanes of a block share the CREDUX scale -- so they split the block's
    # tokens. sf_iter flattens (subtile, slot) to keep the whole cta-tile domain.
    sf_store_out_mapping = None
    if combine_format.is_quantized:
        sf_vec = combine_format.scale_block
        lanes_per_block = sf_vec
        subtile_tokens = SwapABSwigluFp4Epilogue._EpilogueTokenTileSize
        iters_per_subtile = subtile_tokens // lanes_per_block
        n_subtiles = cta_token_tile_size // subtile_tokens

        def ublk_sf_mapping(epi_tid, iter_idx):
            subtile_idx = iter_idx // iters_per_subtile
            slot = iter_idx % iters_per_subtile
            lane = epi_tid % 32
            warp = epi_tid // 32
            lane_in_block = lane % lanes_per_block
            block_in_warp = lane // lanes_per_block
            return {
                "token_in_cta_tile":
                subtile_idx * subtile_tokens + lane_in_block +
                slot * lanes_per_block,
                "hidden_in_cta_tile":
                warp * 32 + block_in_warp * sf_vec,
            }

        sf_store_out_mapping = Contract(
            domain=Space(("epi_tid", "iter_idx"),
                         (128, n_subtiles * iters_per_subtile)),
            codomain=Space(("token_in_cta_tile", "hidden_in_cta_tile"),
                           (max_token_cta_tile, cta_hidden_tile_size)),
            mapping=FunctionMapping(ublk_sf_mapping),
        )
    return store_out_mapping, sf_store_out_mapping, fundamental_mapping


# (...) -> ((atom_v, 1))
@cute.jit
def wrap_into_copy_standard_layout(tensor: cute.Tensor):
    tensor = cute.coalesce(cute.flatten(tensor))
    tensor = cute.append_ones(tensor, cute.rank(tensor) + 1)
    tensor = cute.group_modes(tensor, 0, cute.rank(tensor) - 1)
    tensor = cute.group_modes(tensor, 0, cute.rank(tensor))
    return tensor


@cute.jit
def fc2_f2fp(
    *tensors,
    alpha_val: Optional[cutlass.Float32] = None,
    **_,
) -> cute.Tensor:
    reorder_dtype = cutlass.BFloat16
    total_size = 0
    for t in tensors:
        total_size += cute.size(t)
    converted_acc = cute.make_rmem_tensor((total_size, ), reorder_dtype)
    elems_processed = 0
    for t in tensors:
        current_tensor_size = cute.size(t)
        dst = cute.make_tensor(
            converted_acc.iterator + elems_processed,
            cute.make_layout((current_tensor_size, )),
        )
        if cutlass.const_expr(alpha_val is None):
            dst.store(t.load().to(reorder_dtype))
        else:
            if cutlass.const_expr(current_tensor_size % 2 != 0):
                raise ValueError(
                    "fc2_f2fp expects even elements for each input tensor.")
            scaled = cute.make_rmem_tensor((current_tensor_size, ),
                                           cutlass.Float32)
            for i in cutlass.range_constexpr(0, current_tensor_size, 2):
                # scaled[i] = t[i] * alpha_val
                s0, s1 = cute.arch.mul_packed_f32x2((t[i], t[i + 1]),
                                                    (alpha_val, alpha_val))
                scaled[i] = s0
                scaled[i + 1] = s1
            dst.store(scaled.load().to(reorder_dtype))
        elems_processed += current_tensor_size
    return converted_acc


@cute.jit
def post_f2fp_reorder_identity(*, casted: cute.Tensor, **_):
    # UBLK: the f2fp output is already in the pre-store distribution (each lane
    # owns one hidden element across the 64 subtile tokens); no reorder needed.
    return casted


@cute.jit
def fc2_stg_tmem_acc_load(*, tmem_subtile_tensor: cute.Tensor, **_):
    return _TmemTranspose16x32Core.load_subtile_raw_acc(tmem_subtile_tensor)


@cute.jit
def fc2_ublk_tmem_acc_load(*, tmem_subtile_tensor: cute.Tensor, epi, **_):
    # UBLK consumes a warp-local 32-hidden x 64-token slice.  The caller passes
    # the CTA-level 128-hidden x 64-token subtile view, so select this epi
    # warp's hidden block before issuing LDTM.x64.
    tmem_subtile_per_warp = cute.logical_divide(
        tmem_subtile_tensor,
        (32, None),
    )[(None, epi.warp_idx), None]
    raw_regs = cute.make_rmem_tensor((64, ), cutlass.Float32)
    atom_ld32x32_x64 = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    cute.copy(
        atom_ld32x32_x64,
        wrap_into_copy_standard_layout(tmem_subtile_per_warp),
        wrap_into_copy_standard_layout(raw_regs),
    )
    return (raw_regs, )


@cute.jit
def fc2_stg_post_f2fp_reorder(
        *,
        casted: cute.Tensor,  # (subtile_cnt,)
        tmem_subtile_view: cute.Tensor,  # (epi_tile_m, epi_tile_n)
        **_):

    if cutlass.const_expr(cute.size(casted) != 64):
        raise NotImplementedError(
            "fc2 stg pass expects 64 fp32 regs in total before store reorder.")

    # casted (flat 64) = [h0_top, h0_bot, h1_top, h1_bot], 16 bf16 each.
    #
    # gather: interleave (top[i], bot[i]) -> bf16x2 slot i, both halves at once.
    #   read casted through (t, hidden, half) -> casted[t*16 + hidden + half*32]
    #   in (t fastest) order -> [top0,bot0,top1,bot1,...] per half = packed bf16x2.
    # scatter: de-interleave the transposed natural-hidden regs back to the
    #   STG pre-store order (token = lane + 32*(vid//32), hidden = vid % 32).
    gather_top_bot_map = ((2, 16, 2), (16, 1, 32))
    scatter_top_bot_map = ((16, 2, 2), (2, 1, 32))
    dtype = cutlass.BFloat16

    packed = cute.make_rmem_tensor((64, ), dtype)
    cute.autovec_copy(
        cute.composition(
            casted,
            cute.make_layout((gather_top_bot_map[0], ),
                             stride=(gather_top_bot_map[1], ))), packed)
    # Although this works...
    # packed.store(
    #     cute.make_tensor(
    #         casted.iterator,
    #         cute.make_layout(gather_top_bot_map[0], stride=gather_top_bot_map[1]),
    #     ).load()
    # )
    packed_i32 = cute.recast_tensor(packed,
                                    cutlass.Float32)  # (32,): 16 i32 per half

    # Reuse the 32-bit transpose: each i32 slot carries one packed bf16x2 pair.
    token_0_32_pre_scatter_back = TmemTranspose16x32(
        tmem_subtile_view.iterator,
        Region.Top,
        reg_tensor=cute.composition(packed_i32, (16, )),
    ).from_r1_perm_until_last_store()
    token_32_64_pre_scatter_back = TmemTranspose16x32(
        tmem_subtile_view.iterator + 32,
        Region.Top,
        reg_tensor=cute.composition(cute.domain_offset(16, packed_i32), (16, )),
    ).from_r1_perm_until_last_store()
    cute.autovec_copy(token_0_32_pre_scatter_back,
                      cute.zipped_divide(packed_i32, (16, ))[None, 0])
    cute.autovec_copy(token_32_64_pre_scatter_back,
                      cute.zipped_divide(packed_i32, (16, ))[None, 1])
    out = cute.make_rmem_tensor((64, ), dtype)
    cute.autovec_copy(
        cute.composition(
            packed,
            cute.make_layout((scatter_top_bot_map[0], ),
                             stride=(scatter_top_bot_map[1], ))), out)
    return out


@cute.jit
def fc2_redg_post_f2fp_reorder(
    *,
    casted: cute.Tensor,
    tmem_subtile_view: cute.Tensor,
    **_,
):
    # (epi_tid, elem_idx) -> (token_64, hidden_128), each thread hold token_2 x hidden_32
    natural = fc2_stg_post_f2fp_reorder(
        casted=casted,
        tmem_subtile_view=tmem_subtile_view,
    )
    core_matrix_reorder_sttm_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16),
        cutlass.Float32,
    )
    core_matrix_reorder_ldtm_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x2),
        cutlass.Float32,
    )
    # ((16, 2), token_32_group)
    natural_divided_by_token32_16dp = cute.logical_divide(
        cute.zipped_divide(natural, (32, )), (16, None))
    out = cute.make_rmem_tensor(natural_divided_by_token32_16dp.shape,
                                casted.dtype)
    out_as_i32 = cute.recast_tensor(out, cutlass.Float32)
    # (32, 64)
    tmem_subtile_warp_local = cute.flat_divide(
        tmem_subtile_view, (32, cute.size(tmem_subtile_view, 1)))[None, None, 0,
                                                                  0]
    # (16, 16, 16dp_group, token_32_groups). Note, this tmem can provide 2x cols since the original is bf16.
    tmem_subtile_divided_by_token_group_divided_by_16dp = cute.flat_divide(
        tmem_subtile_warp_local, (16, 16))
    for i in cutlass.range_constexpr(
            cute.size(natural_divided_by_token32_16dp, 1)):
        current_sttm_src = cute.recast_tensor(
            natural_divided_by_token32_16dp[None, i], cutlass.Float32)
        cute.copy(
            core_matrix_reorder_sttm_atom,
            wrap_into_copy_standard_layout(current_sttm_src),
            wrap_into_copy_standard_layout(
                tmem_subtile_divided_by_token_group_divided_by_16dp[None, None,
                                                                    None, i]))
        cute.copy(
            core_matrix_reorder_ldtm_atom,
            wrap_into_copy_standard_layout(
                tmem_subtile_divided_by_token_group_divided_by_16dp[None, None,
                                                                    0, i]),
            wrap_into_copy_standard_layout(out_as_i32[(None, 0), i]))
        cute.copy(
            core_matrix_reorder_ldtm_atom,
            wrap_into_copy_standard_layout(
                tmem_subtile_divided_by_token_group_divided_by_16dp[None, None,
                                                                    1, i]),
            wrap_into_copy_standard_layout(out_as_i32[(None, 1), i]))

    return cute.coalesce(out)


@cute.jit
def fc2_stg_store_function(
        *,
        epi,
        subtile: cute.Tensor,  # Always bf16 pre quant tesnor
        subtile_idx: cutlass.Int32,
        fc2_output_router: Fc2OutputRouter,
        **_):
    if cutlass.const_expr(epi.combine_format.is_quantized):
        data_subtile, sf_regs = QuantImpl(epi.combine_format,
                                          "regs_in_thread")(subtile)
    else:
        data_subtile = subtile
        sf_regs = None
    stg_width_elems: cutlass.Constexpr[int] = min(
        32, 256 // data_subtile.element_type.width)
    stg_bits: cutlass.Constexpr[
        int] = stg_width_elems * data_subtile.element_type.width
    copy_atom_vec = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Int32,
        num_bits_per_copy=stg_bits,
    )
    elems_per_thread: cutlass.Constexpr[int] = cute.size(data_subtile)
    if cutlass.const_expr(elems_per_thread % stg_width_elems != 0):
        raise ValueError(
            "fc2 STG store requires pre-store elems per thread to be divisible "
            f"by STG issue width, got {elems_per_thread} and {stg_width_elems}."
        )

    if cutlass.const_expr(sf_regs is not None):
        sf_scales_per_stg: cutlass.Constexpr[
            int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[
            int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = cutlass.Int32(subtile_idx) * cutlass.Int32(
                token_groups_per_subtile) + token_group
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(
                    sf_ptr, cute.make_layout((sf_scales_per_stg, )))
                for k in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[k] = sf_regs[token_group * sf_scales_per_stg + k]

    iters_per_subtile: cutlass.Constexpr[
        int] = elems_per_thread // stg_width_elems
    copy_src = cute.zipped_divide(data_subtile, (stg_width_elems, ))
    single_copy_layout = cute.make_layout(((stg_width_elems, 1), ),
                                          stride=((1, 0), ))
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(
        iters_per_subtile)
    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            src_i = cute.make_tensor(copy_src[None, local_iter].iterator,
                                     single_copy_layout)
            dst_i = cute.make_tensor(dst_ptr, single_copy_layout)
            cute.copy(copy_atom_vec, cute.recast_tensor(src_i, cutlass.Int32),
                      cute.recast_tensor(dst_i, cutlass.Int32))


@cute.jit
def fc2_ublk_store_function_impl(
    *,
    epi,
    subtile: cute.Tensor,  # Always bf16 pre-quant tensor
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    smem_tensor = epi.smem_tensor
    if cutlass.const_expr(smem_tensor is None):
        raise ValueError("fc2 UBLK store requires epi.smem_tensor.")
    quantized: cutlass.Constexpr[bool] = epi.combine_format.is_quantized
    if cutlass.const_expr(quantized):
        data_subtile, selected_sf = QuantImpl(
            epi.combine_format,
            "threads_with_the_same_reg",
            lane_idx=epi.lane_idx,
        )(subtile)
    else:
        data_subtile = subtile
        selected_sf = None

    smem_read_write_bar = pipeline.NamedBarrier(
        barrier_id=SwapABSwigluFp4Epilogue._EpilogueSyncWaitBarId,
        num_threads=SwapABSwigluFp4Epilogue._EpilogueWarpCnt * 32,
    )
    warp_idx = epi.warp_idx
    lane_idx = epi.lane_idx
    warp_hidden_base = cutlass.Int32(warp_idx * 32)

    regs_per_thread: cutlass.Constexpr[int] = cute.size(data_subtile)
    tokens_per_smem_slice: cutlass.Constexpr[int] = cute.size(smem_tensor,
                                                              mode=[0])
    if cutlass.const_expr(regs_per_thread % tokens_per_smem_slice != 0):
        raise ValueError(
            "fc2 UBLK store requires pre-store regs per thread to be divisible "
            f"by scratch token rows, got {regs_per_thread} and {tokens_per_smem_slice}."
        )
    loop_cnt: cutlass.Constexpr[int] = regs_per_thread // tokens_per_smem_slice

    # SF straight out (no smem): the sf_vec lanes of a block share the CREDUX
    # result, so each emits its slice -- one scale per slot. The sf mapping owns
    # the (lane, slot) -> (token, hidden) layout; selected_sf[slot] is this
    # lane's slot-th scale, aligned to sf_iter = subtile_idx*iters_per_subtile+slot.
    if cutlass.const_expr(quantized):
        # A scale block = sf_vec hidden = sf_vec lanes (UBLK), and those lanes
        # split the subtile's tokens, so each emits subtile_tokens // lanes_per_block.
        lanes_per_block: cutlass.Constexpr[int] = epi.combine_format.scale_block
        iters_per_subtile: cutlass.Constexpr[
            int] = epi._EpilogueTokenTileSize // lanes_per_block
        for slot in cutlass.range_constexpr(iters_per_subtile):
            sf_iter = cutlass.Int32(subtile_idx) * cutlass.Int32(
                iters_per_subtile) + slot
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((1, )))
                sf_dst[0] = selected_sf[slot]

    for token32_group_idx in cutlass.range_constexpr(loop_cnt):
        if cutlass.const_expr(token32_group_idx > 0):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            smem_read_write_bar.arrive_and_wait()

        # R2S transpose: each lane writes its hidden column's 32 token rows for
        # this group. (SF already went straight out above; only DATA transposes
        # through smem here.)
        for token_i in cutlass.range_constexpr(tokens_per_smem_slice):
            src_reg = token_i + tokens_per_smem_slice * token32_group_idx
            smem_tensor[token_i,
                        warp_hidden_base + lane_idx] = data_subtile[src_reg]

        cute.arch.fence_proxy("async.shared", space="cta")
        smem_read_write_bar.arrive_and_wait()

        # ublk_iter_idx is constexpr so get_dst indexes a constexpr slot (no spill).
        # Gate / scratch_row specialize the store-out mapping: lane_idx//8 picks the
        # subtile, warp_idx*8 + lane_idx%8 is the token's row within the 32-group.
        ublk_iter_idx = token32_group_idx
        # SF already went straight out above; here only the DATA row is pushed.
        dst_ptr, pred = fc2_output_router.get_data_dst(ublk_iter_idx)
        if pred != cutlass.Int32(0) and (lane_idx //
                                         cutlass.Int32(8)) == subtile_idx:
            scratch_row = warp_idx * cutlass.Int32(
                8) + lane_idx % cutlass.Int32(8)
            copy_elems = cutlass.Int32(128)
            if cutlass.const_expr(epi.fc2_hidden_needs_predicate):
                copy_elems = cutlass.Int32(
                    fc2_output_router.valid_hidden_this_cta_tile)
            # smem_tensor holds the combine wire dtype, so the bulk byte count
            # scales with the wire width, not the bf16 compute dtype.
            copy_bytes = copy_elems * epi.combine_format.act_dtype.width // 8

            src_row = cute.slice_(smem_tensor, (scratch_row, None))
            if cutlass.const_expr(epi.reduce_topk_in_kernel):
                _cp_reduce_async_bulk_add_noftz_bf16_s2g(
                    dst_ptr,
                    src_row.iterator,
                    copy_bytes,
                )
            else:
                _cp_async_bulk_s2g(
                    dst_ptr,
                    src_row.iterator,
                    copy_bytes,
                )

        cute.arch.cp_async_bulk_commit_group()

    # Drain the final bulk op before the fixed scratch tile is reused by the
    # next subtile / task tile.
    cute.arch.cp_async_bulk_wait_group(0, read=True)
    smem_read_write_bar.arrive_and_wait()


@cute.jit
def fc2_redg_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # Always bf16; in-kernel reduce never quantizes
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    redg_width_elems: cutlass.Constexpr[int] = 4
    elems_per_thread: cutlass.Constexpr[int] = cute.size(subtile)
    if cutlass.const_expr(elems_per_thread % redg_width_elems != 0):
        raise ValueError(
            "fc2 REDG store requires pre-store elems per thread to be divisible "
            f"by REDG issue width, got {elems_per_thread} and {redg_width_elems}."
        )
    iters_per_subtile: cutlass.Constexpr[
        int] = elems_per_thread // redg_width_elems
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(
        iters_per_subtile)
    subtile_by_redg_issue = cute.zipped_divide(subtile, (redg_width_elems, ))

    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            bf16x4 = subtile_by_redg_issue[None, local_iter]
            packed_bf16x2 = cute.recast_tensor(bf16x4, cutlass.Float32)
            _red_add_relaxed_sys_v2_bf16x2(
                dst_ptr,
                cutlass.Float32(packed_bf16x2[0]),
                cutlass.Float32(packed_bf16x2[1]),
            )


def make_fc2_stg_process_pipeline(
    *,
    combine_format: CombineFormat,
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = make_fc2_stg_cta_store_out_contract(
        combine_format,
        cta_token_tile_size,
        cta_hidden_tile_size,
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_stg_post_f2fp_reorder,
        store_function=fc2_stg_store_function,
        fc2_cta_tile_contract=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=True,
    )


def make_fc2_redg_process_pipeline(
    *,
    combine_format: CombineFormat,
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = make_fc2_redg_cta_store_out_contract(
        combine_format,
        cta_token_tile_size,
        cta_hidden_tile_size,
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_redg_post_f2fp_reorder,
        store_function=fc2_redg_store_function,
        fc2_cta_tile_contract=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=True,
    )


def make_fc2_ublk_process_pipeline(
    *,
    combine_format: CombineFormat,
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = make_fc2_ublk_store_out_contract(
        combine_format,
        cta_token_tile_size,
        cta_hidden_tile_size,
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_ublk_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=post_f2fp_reorder_identity,
        store_function=fc2_ublk_store_function_impl,
        fc2_cta_tile_contract=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=False,
    )
