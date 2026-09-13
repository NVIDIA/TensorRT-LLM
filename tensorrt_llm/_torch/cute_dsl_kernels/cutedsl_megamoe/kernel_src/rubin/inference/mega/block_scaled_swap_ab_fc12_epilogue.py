# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous Rubin epilogue for the fused FC1+FC2 swap-AB MegaMoE kernel.

Per-thread RMEM tensors flow between the transpose / SwiGLU / quantize / fc2
store steps as bare ``cute.Tensor`` fragments; their thread distribution is a
fixed physical property of the surrounding atom sequence and is documented in
local comments. FC2 store mappings are finite ``FunctionMapping`` objects
evaluated at runtime to drive metadata lookup and destination pointer math.
"""

import dataclasses
from typing import Any, Callable, ClassVar, List, Literal, Optional, Tuple, Type, Union

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

from .....api import (
    ImplDesc,
    KernelComponent,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....communication.token_protocol import TokenSrcMetadata
from .....helpers.constants import Fp8E4M3RcpLimit, Fp8E5M2RcpLimit, Fp32Max, Nvfp4E2M1RcpLimit
from .....helpers.cute_py_helpers import tcgen05_block_scaled_acc_dtype
from .....helpers.dsl_helpers import mark_alignment
from .....helpers.flag_batch import make_flag_batch_tracker
from .....helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    cp_reduce_async_bulk_add_bf16_s2g,
    cvt_f32_to_fp8_to_f32,
    movmatrix_b16,
    red_add_relaxed_sys_v2_bf16x2,
)
from .....helpers.smem_workspace import SmemRegion, SmemWorkspace
from .....quant_def import CombineFormat, QuantKind
from ....function_mapping import CoordinateSpace, FunctionMapping
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase, SwapAbFc12WorkTileInfo
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension


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
      * ``regs_in_pair_threads``      -- a block is ``sf_vec / 2`` regs in each of
        two paired warps; amax is thread-local over the half, then exchanged with
        the partner warp through SMEM. Orthodox mx only: fc1's TMEM transpose
        hands one warp exactly 16 intermediate values per token, so a 32-wide
        scale block necessarily straddles warps ``w`` and ``w ^ 1``.
    ``prequant_reg`` must be 1D with size divisible by the per-thread block share
    (``sf_vec``, or ``sf_vec / 2`` for the paired direction). Combine inputs are
    bf16 (fc2's bf16 reorder regs); orthodox input is fp32 (swiglu).
    """

    quant_kind: Union[QuantKind, CombineFormat]
    sf_vec_direction: Literal["regs_in_thread", "threads_with_the_same_reg", "regs_in_pair_threads"]
    lane_idx: Optional[Any] = None
    warp_idx: Optional[Any] = None
    pair_exchange_barrier: Optional[Any] = None

    _directions: ClassVar[Tuple[str, ...]] = (
        "regs_in_thread",
        "threads_with_the_same_reg",
        "regs_in_pair_threads",
    )

    # -- config / validation --------------------------------------------------

    def __post_init__(self):
        if isinstance(self.quant_kind, CombineFormat):
            if not self.quant_kind.is_quantized:
                raise ValueError(
                    f"QuantImpl combine path needs a quantized CombineFormat, got {self.quant_kind}."
                )
        elif not isinstance(self.quant_kind, QuantKind):
            raise ValueError(
                f"quant_kind must be a QuantKind or a CombineFormat, got {self.quant_kind!r}."
            )
        if self.sf_vec_direction not in self._directions:
            raise ValueError(
                f"sf_vec_direction must be one of {self._directions}, got {self.sf_vec_direction!r}."
            )
        if self.sf_vec_direction == "threads_with_the_same_reg" and self.lane_idx is None:
            raise ValueError("across-lane quant needs lane_idx for the CREDUX half-warp predicate.")
        if self.sf_vec_direction == "regs_in_pair_threads":
            if isinstance(self.quant_kind, CombineFormat):
                raise ValueError(
                    "The paired direction is orthodox-only; combine blocks never straddle warps."
                )
            if self.sf_vec_size % 2 != 0:
                raise ValueError(
                    f"The paired direction needs an even sf_vec, got {self.sf_vec_size}."
                )
            if self.lane_idx is None or self.warp_idx is None or self.pair_exchange_barrier is None:
                raise ValueError(
                    "The paired direction needs lane_idx, warp_idx and pair_exchange_barrier."
                )
        elif not isinstance(self.quant_kind, CombineFormat) and self.sf_vec_size != 16:
            # A wider orthodox block cannot be reduced inside one thread: see the paired direction.
            raise ValueError(f"Orthodox sf_vec {self.sf_vec_size} needs the paired direction.")

    @property
    def data_dtype(self):
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.act_dtype
        # Orthodox output feeds fc2 as its activation, so it is the kind's activation type.
        return self.quant_kind.activation_dtype

    @property
    def scale_dtype(self):
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.scale_dtype
        return self.quant_kind.sf_dtype

    @property
    def sf_vec_size(self) -> int:
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.scale_block
        return self.quant_kind.sf_vec_size

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

    @property
    def _block_share_per_thread(self) -> int:
        """How many of a block's elements this thread holds."""
        return (
            self.sf_vec_size // 2
            if self.sf_vec_direction == "regs_in_pair_threads"
            else self.sf_vec_size
        )

    # -- dispatch -------------------------------------------------------------

    @cute.jit
    def __call__(self, prequant_reg: cute.Tensor, *, norm_const=None, smem_intermediate=None):
        if cutlass.const_expr(cute.size(prequant_reg) % self._block_share_per_thread != 0):
            raise ValueError(
                "prequant_reg size must be divisible by this thread's share of a block."
            )
        # Combine quantizes fc2's bf16 reorder regs; orthodox the fp32 swiglu.
        expected_in = cutlass.BFloat16 if self._is_combine else cutlass.Float32
        if cutlass.const_expr(prequant_reg.element_type is not expected_in):
            raise TypeError(
                f"QuantImpl({self.quant_kind}) expects {expected_in} prequant input, got {prequant_reg.element_type}."
            )
        needs_smem = cutlass.const_expr(self.sf_vec_direction == "regs_in_pair_threads")
        if cutlass.const_expr(needs_smem != (smem_intermediate is not None)):
            raise ValueError(
                "smem_intermediate must be supplied for the paired direction and only for it."
            )
        if cutlass.const_expr(not self._is_combine):
            if cutlass.const_expr(self.sf_vec_direction == "regs_in_pair_threads"):
                return self.mx_quant_regs_in_pair_threads_impl(prequant_reg, smem_intermediate)
            return self.nvfp4_quant_impl(prequant_reg, norm_const=norm_const)
        if cutlass.const_expr(self.data_dtype is cutlass.Float4E2M1FN):
            if cutlass.const_expr(self.sf_vec_direction == "regs_in_thread"):
                return self.nvfp4_combine_quant_regs_in_thread_impl(prequant_reg)
            return self.nvfp4_combine_quant_threads_with_the_same_reg_impl(prequant_reg)
        if cutlass.const_expr(self.sf_vec_direction == "regs_in_thread"):
            return self.mxfp8_combine_quant_regs_in_thread_impl(prequant_reg)
        return self.mxfp8_combine_quant_threads_with_the_same_reg_impl(prequant_reg)

    # -- impls ----------------------------------------------------------------

    # regs_in_thread only; fc1 promises the vec direction via its TMEM transpose.
    @cute.jit
    def nvfp4_quant_impl(
        self, prequant_reg: cute.Tensor, *, norm_const: Optional[cutlass.Float32] = None
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # fp32 in -> e2m1 data + e4m3 sfc. Mirrors the prior nvfp4_quant scale
        # math (sfc -> capped/masked acc_scale); topk pre-mult + sf store are the
        # caller's job now.
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.Float8E4M3FN)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
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
            mask = cute.arch.fmin(sfc_rt * cutlass.Float32(1e30), cutlass.Float32(1.0))
            acc_scale = acc_scale * mask
            sf_values.append(sfc_e4m3)
            data_blocks.append(self._scale_to_data_ssa(block, acc_scale))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.Float8E4M3FN))
        return data, sf

    @cute.jit
    def mx_quant_regs_in_pair_threads_impl(
        self,
        prequant_reg: cute.Tensor,  # (token_2_intermeidate_x)
        smem_intermediate: cute.Tensor,  # (token_blocks, epi_threads)
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        half = self._block_share_per_thread
        n_blocks = cute.size(prequant_reg) // half
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), self.data_dtype)
        sf = cute.make_rmem_tensor((n_blocks,), self.scale_dtype)
        in_blocks = cute.zipped_divide(prequant_reg, (half,))  # ((half,), (n_blocks,))

        if cutlass.const_expr(cute.size(smem_intermediate, mode=[0]) != n_blocks):
            raise ValueError(
                f"The paired amax exchange needs {n_blocks} rows, got {cute.size(smem_intermediate, mode=[0])}."
            )
        if cutlass.const_expr(smem_intermediate.stride[0] != 1):
            # Thread-major would still be correct but would silently split the exchange into
            # per-block scalar accesses.
            raise ValueError(
                "The paired amax exchange must be block-major to stay one vector access."
            )

        exchange_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=n_blocks * cutlass.Float32.width,
        )
        # Which warps pair up is epilogue knowledge: warp w owns one contiguous run of
        # intermediate outputs, so a block that is twice that run wide joins adjacent warps.
        partner_warp = self.warp_idx ^ cutlass.Int32(1)
        own_thread = self.warp_idx * cutlass.Int32(32) + self.lane_idx
        partner_thread = partner_warp * cutlass.Int32(32) + self.lane_idx

        half_amax = cute.make_rmem_tensor((n_blocks,), cutlass.Float32)
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            half_amax[vec_block_idx] = self._amax_thread_fp32(in_blocks[None, vec_block_idx])
        cute.copy(
            exchange_atom,
            cute.coalesce(half_amax),
            cute.coalesce(smem_intermediate[None, own_thread]),
        )

        self.pair_exchange_barrier.arrive_and_wait()

        partner_amax = cute.make_rmem_tensor((n_blocks,), cutlass.Float32)
        cute.copy(
            exchange_atom,
            cute.coalesce(smem_intermediate[None, partner_thread]),
            cute.coalesce(partner_amax),
        )

        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block_amax = cute.arch.fmax(half_amax[vec_block_idx], partner_amax[vec_block_idx])
            scale_e8m0, scale_f32 = self._e8m0(block_amax)
            sf_values.append(scale_e8m0)
            data_blocks.append(
                self._scale_to_data_ssa(in_blocks[None, vec_block_idx], self._enc_mxfp8(scale_f32))
            )
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, self.scale_dtype))
        return data, sf

    @cute.jit
    def nvfp4_combine_quant_regs_in_thread_impl(
        self, prequant_reg: cute.Tensor
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # bf16 in -> e2m1 data + per-16 bf16 amax. amax found on bf16 (packed).
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.BFloat16)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            amax = self._amax_thread_bf16(block)
            sf_values.append(amax)
            decode_scale = cutlass.Float32(amax) * cutlass.Float32(self._data_rcp_limit)
            data_blocks.append(self._scale_to_data_ssa(block, self._enc_nvfp4(decode_scale)))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.BFloat16))
        return data, sf

    # Mapping: (lane_idx, selected_sf_idx) -> (token_64, hidden_32)
    # token_idx = lane_idx % 16 + selected_sf_idx * 16
    # hidden_idx = lane_idx // 16 * 16
    @cute.jit
    def nvfp4_combine_quant_threads_with_the_same_reg_impl(
        self, prequant_reg: cute.Tensor
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
        data = cute.make_rmem_tensor((n_tokens,), cutlass.Float4E2M1FN)
        selected_sf = cute.make_rmem_tensor((n_tokens // lanes_per_block,), cutlass.BFloat16)
        scaled_vec = cute.full((n_tokens,), cutlass.Float32(0.0), cutlass.Float32)
        for token_idx in cutlass.range_constexpr(n_tokens):
            value = cutlass.Float32(prequant_reg[token_idx])
            amax_bf16 = self._amax_lane(value).to(cutlass.BFloat16)
            slot = token_idx // lanes_per_block
            if (token_idx % lanes_per_block) == lane_in_block:
                selected_sf[slot] = amax_bf16
            else:
                selected_sf[slot] = selected_sf[slot]
            decode_scale = cutlass.Float32(amax_bf16) * cutlass.Float32(self._data_rcp_limit)
            scaled_value = value * self._enc_nvfp4(decode_scale)
            scaled_vec = cute.TensorSSA(
                vector.insert(scaled_value.ir_value(), scaled_vec.ir_value(), [], [token_idx]),
                (n_tokens,),
                cutlass.Float32,
            )
        self._store_packed_data(data, scaled_vec.to(cutlass.Float4E2M1FN))
        return data, selected_sf

    @cute.jit
    def mxfp8_combine_quant_regs_in_thread_impl(
        self, prequant_reg: cute.Tensor
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # bf16 in -> e4m3/e5m2 data + per-32 e8m0. amax found on bf16 (packed).
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), self.data_dtype)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.Float8E8M0FNU)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            # widen the native-bf16 amax to fp32 for the e8m0 round-up math.
            scale_e8m0, scale_f32 = self._e8m0(cutlass.Float32(self._amax_thread_bf16(block)))
            sf_values.append(scale_e8m0)
            data_blocks.append(self._scale_to_data_ssa(block, self._enc_mxfp8(scale_f32)))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.Float8E8M0FNU))
        return data, sf

    # Mapping: (lane_idx, selected_sf_idx) -> (token_64, hidden_32)
    # token_idx = lane_idx + selected_sf_idx * 32
    # hidden_idx = 0
    @cute.jit
    def mxfp8_combine_quant_threads_with_the_same_reg_impl(
        self, prequant_reg: cute.Tensor
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
        data = cute.make_rmem_tensor((n_tokens,), self.data_dtype)
        selected_sf = cute.make_rmem_tensor((n_tokens // lanes_per_block,), cutlass.Float8E8M0FNU)
        scaled_vec = cute.full((n_tokens,), cutlass.Float32(0.0), cutlass.Float32)
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
                vector.insert(scaled_value.ir_value(), scaled_vec.ir_value(), [], [token_idx]),
                (n_tokens,),
                cutlass.Float32,
            )
        self._store_packed_data(data, scaled_vec.to(self.data_dtype))
        return data, selected_sf

    # -- shared sub-steps -----------------------------------------------------

    @cute.jit
    def _scale_to_data_ssa(self, block: cute.Tensor, enc: cutlass.Float32) -> cute.TensorSSA:
        block_f32 = block.load().to(cutlass.Float32)
        enc_vec = cute.full_like(block_f32, enc, cutlass.Float32)
        return (block_f32 * enc_vec).to(self.data_dtype)

    @cute.jit
    def _concat_blocks_ssa(self, blocks, dtype: Type[cutlass.Numeric]) -> cute.TensorSSA:
        values = []
        for block_idx in cutlass.range_constexpr(len(blocks)):
            block = blocks[block_idx]
            for elem_idx in cutlass.range_constexpr(cute.size(block.shape)):
                values.append(block[elem_idx].ir_value())
        vec = vector.from_elements(T.vector(len(values), dtype.mlir_type), values)
        return cute.TensorSSA(vec, (len(values),), dtype)

    @cute.jit
    def _values_to_ssa(self, values, dtype: Type[cutlass.Numeric]) -> cute.TensorSSA:
        vec = vector.from_elements(
            T.vector(len(values), dtype.mlir_type),
            [values[i].ir_value() for i in range(len(values))],
        )
        return cute.TensorSSA(vec, (len(values),), dtype)

    @cute.jit
    def _concat_i32_blocks_ssa(self, blocks) -> cute.TensorSSA:
        values = []
        for block_idx in cutlass.range_constexpr(len(blocks)):
            packed_block = blocks[block_idx].bitcast(cutlass.Int32)
            for elem_idx in cutlass.range_constexpr(cute.size(packed_block.shape)):
                values.append(packed_block[elem_idx].ir_value())
        vec = vector.from_elements(T.vector(len(values), cutlass.Int32.mlir_type), values)
        return cute.TensorSSA(vec, (len(values),), cutlass.Int32)

    @cute.jit
    def _store_packed_blocks(self, data: cute.Tensor, blocks) -> None:
        packed_data = cute.recast_tensor(data, cutlass.Int32)
        packed_data.store(self._concat_i32_blocks_ssa(blocks))

    @cute.jit
    def _store_packed_data(self, data: cute.Tensor, data_ssa: cute.TensorSSA) -> None:
        packed_data = cute.recast_tensor(data, cutlass.Int32)
        packed_data.store(data_ssa.bitcast(cutlass.Int32))

    @cute.jit
    def _amax_thread_fp32(self, block: cute.Tensor) -> cutlass.Float32:
        # max.xorsign.abs reduces |.| in one op per element; the result sign is
        # the xor of the inputs (junk for an amax), so clear it at the end.
        def max_abs(lhs: cutlass.Float32, rhs: cutlass.Float32) -> cutlass.Float32:
            return cutlass.Float32(
                llvm.inline_asm(
                    T.f32(),
                    [cutlass.Float32(lhs).ir_value(), cutlass.Float32(rhs).ir_value()],
                    "max.xorsign.abs.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        acc = block[0]
        for elem_idx in cutlass.range_constexpr(1, cute.size(block)):
            acc = max_abs(acc, block[elem_idx])
        mag_bits = cutlass.Int32(
            llvm.bitcast(T.i32(), cutlass.Float32(acc).ir_value())
        ) & cutlass.Int32(0x7FFFFFFF)
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
                    [cutlass.Int32(lhs).ir_value(), cutlass.Int32(rhs).ir_value()],
                    "max.xorsign.abs.bf16x2 $0, $1, $2;",
                    "=r,r,r",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        pairs = cute.recast_tensor(block, cutlass.Int32)  # (vec/2,) bf16x2
        acc = cutlass.Int32(pairs[0])
        for pair_idx in cutlass.range_constexpr(1, cute.size(pairs)):
            acc = max_abs(acc, pairs[pair_idx])
        acc = max_abs(acc, acc >> cutlass.Int32(16))
        amax_bits = cute.make_rmem_tensor((1,), cutlass.Int32)
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
    def _e8m0(self, amax: cutlass.Float32) -> Tuple[cutlass.Float8E8M0FNU, cutlass.Float32]:
        candidate = amax * cutlass.Float32(self._data_rcp_limit)
        scale_f32 = cutlass.Float32(cvt_f32_to_fp8_to_f32(candidate, cutlass.Float8E8M0FNU))
        return scale_f32.to(cutlass.Float8E8M0FNU), scale_f32

    @cute.jit
    def _enc_nvfp4(self, decode_scale: cutlass.Float32) -> cutlass.Float32:
        # rcp.approx.ftz with the fc1 cap+mask idiom (amax==0 -> 0, no inf*0 NaN).
        enc = cute.arch.fmin(cute.arch.rcp_approx(decode_scale), Fp32Max)
        mask = cute.arch.fmin(decode_scale * cutlass.Float32(1e30), cutlass.Float32(1.0))
        return enc * mask

    @cute.jit
    def _enc_mxfp8(self, scale_f32: cutlass.Float32) -> cutlass.Float32:
        # Skip nan
        enc = cute.arch.fmin(cute.arch.rcp_approx(scale_f32), Fp32Max)
        mask = cute.arch.fmin(scale_f32 * cutlass.Float32(1e30), cutlass.Float32(1.0))
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
    32-bit slot is an fp32 SwiGLU-fold value for FC1. The (thread, reg) ->
    (tmem_dp, tmem_col) input / output mapping is documented on the
    ``TmemTranspose16x32`` subclass, which is the public entry point.

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
            (((num_lanes, num_cols), 1),),
            stride=(((_TmemTranspose16x32Core._TmemRowStride, 1), 0),),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor, num_regs: int, offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset, cute.make_layout((((num_regs,), 1),), stride=(((1,), 0),))
        )

    def __init__(self, tmem_ptr, region: int, reg_tensor: Optional[cute.Tensor] = None) -> None:
        # The whole transpose is built from 32-bit element atoms; _io_dtype
        # drives _src_regs / output / every LDTM/STTM atom below, so guard the
        # invariant once here (tautological today, defensive against future
        # dtype edits).
        if cutlass.const_expr(self._io_dtype.width != 32):
            raise TypeError(
                f"{type(self).__name__} requires a 32-bit _io_dtype (the "
                f"transpose uses 32-bit element atoms), got {self._io_dtype} "
                f"(width {self._io_dtype.width})."
            )

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

        self._tmem_src_full = cute.make_tensor(src_ptr, self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(dst_ptr, self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(dst_ptr, self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(dst_ptr + half_lane_off, self._tmem_layout(16, 16))

        self._atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), self._io_dtype
        )
        self._atom_st16x128 = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8), self._io_dtype
        )
        self._atom_st32x32 = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16), self._io_dtype
        )
        self._atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), self._io_dtype
        )
        self._atom_ld16x128 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), self._io_dtype
        )

        self._src_regs = cute.make_rmem_tensor((16,), self._io_dtype)
        # ``output`` is a bare (16,) RMEM fragment; its (lane_idx, elem_idx)
        # distribution after all four rounds is the transpose output mapping
        # documented on ``TmemTranspose16x32``.
        self.output = cute.make_rmem_tensor((16,), self._io_dtype)

        # skip-R1.Load mode: ``reg_tensor`` must already be in the transpose
        # input distribution documented by ``TmemTranspose16x32``; we copy it
        # in lieu of the R1 LDTM.
        # Weak entry guard (replaces the removed input contract): the transpose
        # atoms are 32-bit element atoms over exactly 16 regs/lane, so the fed
        # tensor must be a 32-bit element type of size 16.
        self._reg_tensor = reg_tensor
        if reg_tensor is not None:
            if cutlass.const_expr(reg_tensor.element_type.width != 32):
                raise TypeError(
                    f"{type(self).__name__} reg_tensor must be a 32-bit element "
                    f"type, got element type "
                    f"{reg_tensor.element_type} (width {reg_tensor.element_type.width})."
                )
            if cutlass.const_expr(cute.size(reg_tensor) != 16):
                raise ValueError(
                    f"{type(self).__name__} reg_tensor must hold exactly 16 elements, got {cute.size(reg_tensor)}."
                )
            for r in range(16):
                self._src_regs[r] = reg_tensor[r]

    # -- R1 ------------------------------------------------------------------

    def r1_load(self) -> None:
        """LDTM src region -> ``_src_regs``.  No-op in skip-R1.Load mode."""
        if self._reg_tensor is not None:
            return
        cute.copy(self._atom_ld16x64, self._tmem_src_full, self._rmem_copy_view(self._src_regs, 16))

    def r1_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(self._atom_st16x128, self._rmem_copy_view(self.output, 16), self._tmem_src_full)

    # -- R2 ------------------------------------------------------------------

    def r2_load(self) -> None:
        cute.copy(self._atom_ld16x64, self._tmem_src_full, self._rmem_copy_view(self._src_regs, 16))

    def r2_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self._src_regs, 16), self._tmem_dst_full)

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
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self.output, 16), self._tmem_dst_full)

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
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self.output, 16), self._tmem_dst_full)

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
    """FC1 16x32 -> 32x16 TMEM in-place transpose.

    The per-thread RMEM ``(lane_idx, elem_idx) -> (tmem_dp, tmem_col)`` mapping
    is fixed by the underlying atom sequence. Each slot is an fp32 SwiGLU-fold
    value and ``tmem_col`` is the intermediate-output index.

    Input distribution -- what each (lane_idx, elem_idx) reg holds on entry
    (i.e. straight after the 16-dp x 32-col source LDTM, or as fed in via
    ``reg_tensor`` for skip-R1.Load mode):

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
                "skip-R1.Load or do R1.Load)."
            )
        self.top = TmemTranspose16x32(tmem_ptr, Region.Top, reg_tensor=reg_tensor_top)
        self.bot = TmemTranspose16x32(tmem_ptr, Region.Bottom, reg_tensor=reg_tensor_bot)

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


class TmemTranspose32x64B16Movm:
    """FC2 warp-local 32-hidden x 64-token BF16 transpose using MOVM.

    Input is the flat ``[top, bottom]`` distribution produced by two
    16dp256bit accumulator loads followed by ``fc2_f2fp``. Output
    ``(lane_idx, elem_idx)`` coordinates are:

        token  = lane_idx + 32 * (elem_idx // 32)
        hidden = elem_idx % 32

    The fixed register permutation between MOVM and STTM is an SSA rename. It
    keeps each thread's two complete hidden-32 rows without any lane exchange.
    """

    _tmem_row_stride = 1 << 16
    _store_reg_source_indices = (
        0,
        2,
        1,
        3,
        16,
        18,
        17,
        19,
        8,
        10,
        9,
        11,
        24,
        26,
        25,
        27,
        4,
        6,
        5,
        7,
        20,
        22,
        21,
        23,
        12,
        14,
        13,
        15,
        28,
        30,
        29,
        31,
    )

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1),),
            stride=(((TmemTranspose32x64B16Movm._tmem_row_stride, 1), 0),),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor, num_regs: int, offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset, cute.make_layout((((num_regs,), 1),), stride=(((1,), 0),))
        )

    @cute.jit
    def __init__(self, tmem_ptr, reg_tensor: cute.Tensor) -> None:
        if cutlass.const_expr(reg_tensor.element_type is not cutlass.BFloat16):
            raise TypeError(
                f"{type(self).__name__} expects BF16 input after f2fp, got {reg_tensor.element_type}."
            )
        if cutlass.const_expr(cute.size(reg_tensor) != 64):
            raise ValueError(
                f"{type(self).__name__} expects 64 BF16 elements, got {cute.size(reg_tensor)}."
            )

        movm_words = movmatrix_b16(cute.recast_tensor(reg_tensor, cutlass.Int32))
        self._store_words = cute.make_rmem_tensor(movm_words.layout, movm_words.element_type)
        for store_reg in cutlass.range_constexpr(32):
            self._store_words[store_reg] = movm_words[self._store_reg_source_indices[store_reg]]

        half_lane_offset = 16 * self._tmem_row_stride
        self._tmem_top = cute.make_tensor(tmem_ptr, self._tmem_layout(16, 32))
        self._tmem_bottom = cute.make_tensor(tmem_ptr + half_lane_offset, self._tmem_layout(16, 32))
        self._tmem_full = cute.make_tensor(tmem_ptr, self._tmem_layout(32, 32))
        self._store_atom = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8), cutlass.Float32
        )
        self._load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
        )

    @cute.jit
    def __call__(self) -> cute.Tensor:
        movm_words_f32 = cute.recast_tensor(self._store_words, cutlass.Float32)
        cute.copy(self._store_atom, self._rmem_copy_view(movm_words_f32, 16), self._tmem_top)
        cute.copy(
            self._store_atom, self._rmem_copy_view(movm_words_f32, 16, offset=16), self._tmem_bottom
        )

        output_words = cute.make_rmem_tensor((32,), cutlass.Float32)
        cute.copy(self._load_atom, self._tmem_full, self._rmem_copy_view(output_words, 32))
        return cute.recast_tensor(output_words, cutlass.BFloat16)


@dataclasses.dataclass(frozen=True)
class GatedActEpilogueArgs:
    """Optional runtime tensors used by the gated-activation epilogue."""

    fc1_alpha: Optional[cute.Tensor]
    fc2_alpha: Optional[cute.Tensor]
    fc1_norm_const: Optional[cute.Tensor]
    # -----------------------------------
    # MoE domain (token, topk), deepgemm graph only? for transformer graph, we want reduce kernel to perform the
    # score mul.
    topk_scores: Optional[cute.Tensor]


class SwapABGatedActEpilogue(KernelComponent):
    """Autonomous epilogue for the swap-AB SwiGLU NVFP4 kernel.

    ``run()`` is the single entry point the kernel calls inside the epi
    warp body.  The kernel's responsibility is reduced to:

      - allocate / free TMEM and build ``acc_tensor``
      - construct the AB / acc pipelines
      - obtain the scheduler consumer

    Everything else (acc consumer state, task-tile loop, TMEM release,
    TMA store commit / drain, per-subtile dispatch) lives inside this class.
    """

    _EpilogueSyncWaitBarId = 1  # Arrive and wait only
    _EpilogueAsyncBarIdBase = 4  # Some arrive, the others arrive and wait
    _EpilogueFc1GateUpInterleave = 16
    _EpilogueTokenTileSize = 64  # Fundamentally the epi_tile_n
    _EpilogueFc1IntermediateGateUpTileSize = 128  # Fundamentally epi_tile_m
    _EpilogueFc1IntermediateDownTileSize = 64  # Fundamentally epi_tile_m // 2
    _EpilogueFc2HiddenTileSize = 128  # Fundamentally epi_tile_m
    _EpilogueWarpCnt = 4
    # One warp owns this many intermediate_down outputs per token: the TMEM transpose gives each
    # warp 32 accumulator rows, which the gate/up interleave halves.
    _EpilogueFc1IntermediateDownPerWarp = 16
    smem_scratch_overlay: ClassVar[str] = "rubin.swap_ab_gated_act_epilogue.scratch"
    fc1_staging_region: ClassVar[str] = "rubin.swap_ab_gated_act_epilogue.fc1_staging"
    fc2_staging_region: ClassVar[str] = "rubin.swap_ab_gated_act_epilogue.fc2_staging"
    _Fc2UblkFp8RowStrideBytes = 144

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            # The accumulator follows from the instruction family; see __init__.
            "quant_kind": str,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "combine_format": CombineFormat,
            "gate_up_clamp": Optional[float],
            # SiTU (Kimi K3) selects a different gated-activation core; optional so
            # existing SwiGLU descriptors stay valid unchanged. See _resolve_situ_betas.
            "situ_beta": OptionalRequirement(Optional[float]),
            "situ_linear_beta": OptionalRequirement(Optional[float]),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "fc2_use_bulk": bool,
            "communication_enabled": bool,
            "fc1_epi_flag_batch": int,
            "fc2_epi_flag_batch": int,
            "fc2_tma_stages": OptionalRequirement(int),
            "reduce_topk_in_kernel": OptionalRequirement(bool),
            "token_back_push_data": OptionalRequirement(bool),
        }

    @staticmethod
    def _resolve_situ_betas(problem_desc: ProblemDesc) -> Tuple[Optional[float], Optional[float]]:
        """Resolve the SiTU (Kimi K3) betas from the ProblemDesc; ``(None, None)`` means SwiGLU.

        Both betas must be given together, and SiTU excludes ``gate_up_clamp`` -- matching DeepGEMM's
        ``DG_HOST_ASSERT(not use_situ or not activation_clamp_opt.has_value())``. They are baked in at
        codegen time exactly like ``gate_up_clamp``, so the enclosing KernelClass must also fold them
        into its ``name()`` cache key.
        """
        beta = problem_desc.get("situ_beta")
        linear_beta = problem_desc.get("situ_linear_beta")
        if (beta is None) != (linear_beta is None):
            raise ValueError(
                f"situ_beta and situ_linear_beta must be set together, got {beta} and {linear_beta}."
            )
        if beta is None:
            return None, None
        if beta <= 0 or linear_beta <= 0:
            raise ValueError(
                f"SiTU beta parameters must be positive, got {beta} and {linear_beta}."
            )
        if problem_desc["gate_up_clamp"] is not None:
            raise ValueError(
                "SiTU does not support gate_up_clamp; the two activation variants are exclusive."
            )
        return float(beta), float(linear_beta)

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.hidden_size = problem_desc["hidden_size"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.situ_beta, self.situ_linear_beta = self._resolve_situ_betas(problem_desc)
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.communication_enabled = impl_desc["communication_enabled"]
        self.fc1_epi_flag_batch = impl_desc["fc1_epi_flag_batch"]
        self.fc2_epi_flag_batch = impl_desc["fc2_epi_flag_batch"]

        if self.communication_enabled:
            for field_name in ("reduce_topk_in_kernel", "token_back_push_data"):
                if field_name not in impl_desc:
                    raise KeyError(
                        f"Communication-enabled Epilogue requires ImplDesc field {field_name!r}."
                    )
        self.reduce_topk_in_kernel = impl_desc.get("reduce_topk_in_kernel", False)
        self.token_back_push_data = impl_desc.get("token_back_push_data", False)
        if not self.communication_enabled and (
            self.reduce_topk_in_kernel or self.token_back_push_data
        ):
            raise ValueError(
                "A communication-disabled Epilogue cannot enable communication policies."
            )

        # FC1 emits what FC2 consumes as its activation, in the kind's own scale format.
        self.fc1_output_dtype = self.quant_kind.activation_dtype
        self.fc1_output_sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        # A 32-wide scale block spans two epilogue warps; see QuantImpl's paired direction.
        self.needs_pair_amax_exchange = self.sf_vec_size > self._EpilogueFc1IntermediateDownPerWarp
        self.token_back_push_sf = self.communication_enabled and self.combine_format.is_quantized
        self.token_back_enabled = self.token_back_push_data or self.token_back_push_sf
        self.fc2_output_is_local = not self.communication_enabled or self.token_back_push_data
        self.fc2_use_tma = self.fc2_use_bulk and self.fc2_output_is_local
        self.fc2_use_ublk = self.fc2_use_bulk and not self.fc2_output_is_local
        if self.fc2_use_ublk and self.combine_format.act_dtype.width == 4:
            raise ValueError("FC2 UBLK does not support an FP4 combine payload.")
        if self.reduce_topk_in_kernel and self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("In-kernel top-k reduction requires a BF16 combine format.")
        self.reduce_topk_in_epilogue = self.reduce_topk_in_kernel and not self.token_back_push_data
        if (
            not 1 <= self.fc1_epi_flag_batch <= self._EpilogueWarpCnt
            or not 1 <= self.fc2_epi_flag_batch <= self._EpilogueWarpCnt
        ):
            raise ValueError(
                f"Asynchronous epilogue flag batch sizes must be in [1, {self._EpilogueWarpCnt}]."
            )
        self.cluster_tile_intermediate_downproj = (
            self._EpilogueFc1IntermediateDownTileSize * self.cluster_shape_mn[0]
        )

        atom_thr_size = 2 if self.use_2cta_instrs else 1
        self.cta_tile_m = self._EpilogueFc2HiddenTileSize
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.cta_tile_k = self.mma_tiler_mnk[2]
        assert self.mma_tiler_mnk[0] // atom_thr_size == self.cta_tile_m
        assert self.cta_tile_n % self._EpilogueTokenTileSize == 0
        tmem_plan = impl_desc["tmem_plan"]
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if tmem_plan.accumulator_stage_stride_columns != tmem_plan.accumulator_stage_columns:
            raise ValueError("Rubin MegaMoE does not support overlapping accumulator stages.")
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError(
                "Rubin MegaMoE requires one pipeline stage per disjoint accumulator stage."
            )
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.accumulator_shape = (
            self.cta_tile_m,
            self.cta_tile_n,
            tmem_plan.accumulator_stage_count,
        )
        self.accumulator_stride = (1 << 16, 1, tmem_plan.accumulator_stage_stride_columns)

        if (
            isinstance(self.hidden_size, int)
            and self.hidden_size % (self.cta_tile_m * self.cluster_shape_mn[0]) == 0
        ):
            self.fc2_hidden_needs_predicate: bool = False
        else:
            self.fc2_hidden_needs_predicate: bool = True

        if isinstance(self.intermediate_gateup_size, int):
            self.intermediate_downproj: Optional[int] = self.intermediate_gateup_size // 2
        else:
            self.intermediate_downproj: Optional[int] = None

        self.subtile_cnt = self.cta_tile_n // self._EpilogueTokenTileSize

        # One staging stage per token subtile, each an (epi_tile_n, epi_tile_m // 2) quantized tile.
        self.fc1_staging_stage_bytes = (
            self._EpilogueTokenTileSize
            * self._EpilogueFc1IntermediateDownTileSize
            * self.fc1_output_dtype.width
            // 8
        )
        self.fc1_staging_bytes = self.subtile_cnt * self.fc1_staging_stage_bytes
        # The amax exchange plane is (token chunk, epilogue thread), block-major so each thread's
        # column is one vector access. A thread holds one token per 32-lane chunk of the subtile.
        # These slots borrow the current subtile's staging stage rather than costing their own
        # bytes -- see the lifetime argument in fc1_quant.
        self.fc1_amax_token_chunks = self._EpilogueTokenTileSize // 32
        self.fc1_amax_slot_count = (
            self.fc1_amax_token_chunks * self._EpilogueWarpCnt * 32
            if self.needs_pair_amax_exchange
            else 0
        )
        if self.fc1_amax_slot_count * 4 > self.fc1_staging_stage_bytes:
            raise ValueError("The paired amax exchange does not fit in one fc1 staging stage.")

        requested_fc2_tma_stages = impl_desc.get("fc2_tma_stages")
        if (
            requested_fc2_tma_stages is not None
            and not 1 <= requested_fc2_tma_stages <= self.subtile_cnt
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {self.subtile_cnt}], got {requested_fc2_tma_stages}."
            )
        if self.fc2_use_bulk:
            single_stage_region = self._make_fc2_single_stage_region()
            if self.fc2_use_ublk and single_stage_region.nbytes % 16 != 0:
                raise ValueError("Each FC2 UBLK staging stage must occupy a multiple of 16 bytes.")
            if requested_fc2_tma_stages is not None:
                self.fc2_tma_stages = requested_fc2_tma_stages
            else:
                bf16_baseline_stages = min(2, self.subtile_cnt)
                bf16_stage_bytes = (
                    self._EpilogueTokenTileSize
                    * self._EpilogueFc2HiddenTileSize
                    * cutlass.BFloat16.width
                    // 8
                )
                available_staging_bytes = max(
                    self.fc1_staging_bytes, bf16_baseline_stages * bf16_stage_bytes
                )
                self.fc2_tma_stages = min(
                    self.subtile_cnt, available_staging_bytes // single_stage_region.nbytes
                )
            self.fc2_staging_spec: Optional[SmemRegion] = self._make_fc2_staging_region(
                self.fc2_tma_stages
            )
        else:
            self.fc2_tma_stages = 0
            self.fc2_staging_spec = None

    @classmethod
    def epilogue_sync_barrier(cls) -> pipeline.NamedBarrier:
        """The one barrier every epilogue rendezvous uses: all four warps, arrive-and-wait.

        Reused rather than split per purpose because the participant set is always the same 128
        threads and no two uses are ever in flight together -- the tile-boundary rendezvous sits
        outside the subtile loop, FC1's amax exchange and FC2's bulk-store handshake belong to
        different work tiles.
        """
        return pipeline.NamedBarrier(
            barrier_id=cls._EpilogueSyncWaitBarId, num_threads=32 * cls._EpilogueWarpCnt
        )

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Declare the epilogue scratch as one allocation shared by two exclusive lifetimes.

        FC1's staging tile and FC2's store tile belong to different work tiles, separated by the
        tile-boundary TMA drain and rendezvous in ``run()``, so they never coexist.
        """
        overlay = smem_workspace.create_overlay(self.smem_scratch_overlay)
        overlay.add_lifetime("fc1_staging").register_tensor(
            self.fc1_staging_region, cutlass.Int8, (self.fc1_staging_bytes,), byte_alignment=128
        )
        if self.fc2_staging_spec is not None:
            overlay.add_lifetime("fc2_staging").register_tensor(
                self.fc2_staging_region,
                self.fc2_staging_spec.dtype,
                self.fc2_staging_spec.shape,
                stride=self.fc2_staging_spec.stride,
                swizzle=self.fc2_staging_spec.swizzle,
                byte_alignment=self.fc2_staging_spec.byte_alignment,
            )

    def fc1_staged_smem_layout(
        self, n_stages: int, without_stage_mode: bool = False
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        layout = sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize),
            n_stages,
        )
        if without_stage_mode:
            return cute.select(layout, mode=[0, 1])
        return layout

    def fc2_tma_staged_smem_spec(self, n_stages: int) -> Tuple[Tuple, Tuple, Tuple[int, int, int]]:
        """Return the bank-conflict-free token-major FC2 TMA staging layout."""
        stage_stride = self._EpilogueTokenTileSize * self._EpilogueFc2HiddenTileSize
        wire_dtype = self.combine_format.act_dtype
        if wire_dtype is cutlass.BFloat16:
            shape = ((32, 2), (64, 2), n_stages)
            stride = ((64, 2048), (1, 4096), stage_stride if n_stages > 1 else 0)
            swizzle = (3, 4, 3)
        elif wire_dtype.width == 8:
            shape = ((32, 2), 128, n_stages)
            stride = ((128, 4096), 1, stage_stride if n_stages > 1 else 0)
            swizzle = (3, 4, 3)
        elif wire_dtype.width == 4:
            shape = ((32, 2), 128, n_stages)
            stride = ((128, 4096), 1, stage_stride if n_stages > 1 else 0)
            swizzle = (2, 4, 3)
        else:
            raise ValueError(f"Unsupported FC2 TMA staging dtype {wire_dtype}.")
        return shape, stride, swizzle

    def fc2_tma_staged_smem_layout(
        self, n_stages: int, without_stage_mode: bool = False
    ) -> cute.ComposedLayout:
        shape, stride, swizzle = self.fc2_tma_staged_smem_spec(n_stages)
        layout = cute.make_composed_layout(
            cute.make_swizzle(*swizzle), 0, cute.make_layout(shape, stride=stride)
        )
        if without_stage_mode:
            return cute.select(layout, mode=[0, 1])
        return layout

    def _make_fc2_single_stage_region(self) -> SmemRegion:
        if self.fc2_use_tma:
            shape, stride, swizzle = self.fc2_tma_staged_smem_spec(1)
            return SmemRegion(
                name="",
                kind="tensor",
                dtype=self.combine_format.act_dtype,
                shape=shape[:-1],
                stride=stride[:-1],
                swizzle=swizzle,
                byte_alignment=128,
            )

        row_stride_elements = (
            self._Fc2UblkFp8RowStrideBytes
            if self.combine_format.act_dtype.width == 8
            else self._EpilogueFc2HiddenTileSize
        )
        single_stage_region = SmemRegion(
            name="",
            kind="tensor",
            dtype=self.combine_format.act_dtype,
            shape=(self._EpilogueTokenTileSize, self._EpilogueFc2HiddenTileSize),
            stride=(row_stride_elements, 1),
            swizzle=None,
            byte_alignment=16,
        )
        return single_stage_region

    def _make_fc2_staging_region(self, stage_count: int) -> SmemRegion:
        single_stage_region = self._make_fc2_single_stage_region()
        return SmemRegion(
            name="",
            kind="tensor",
            dtype=single_stage_region.dtype,
            shape=(*single_stage_region.shape, stage_count),
            stride=(
                *single_stage_region.stride,
                single_stage_region.cosize if stage_count > 1 else 0,
            ),
            swizzle=single_stage_region.swizzle,
            byte_alignment=single_stage_region.byte_alignment,
        )

    def prepare_tma_store_params(
        self, fc1_output_template: cute.Tensor, fc2_output_template: cute.Tensor
    ) -> Tuple[cute.CopyAtom, cute.Tensor, Optional[cute.CopyAtom], Optional[cute.Tensor]]:
        """Build the FC1 TMA store and the optional local FC2 TMA store."""
        fc1_operation = cpasync.CopyBulkTensorTileS2GOp()
        fc1_smem_layout = self.fc1_staged_smem_layout(1, without_stage_mode=True)
        fc1_tile = (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize)
        fc1_atom, fc1_tensor = cpasync.make_tiled_tma_atom(
            fc1_operation, fc1_output_template, fc1_smem_layout, fc1_tile
        )

        if cutlass.const_expr(self.fc2_use_tma):
            # Keep tiled rest modes dynamic so a single static tile cannot collapse to stride zero.
            runtime_token_extent = cutlass.Int32(fc2_output_template.shape[0])
            runtime_hidden_extent = cutlass.Int32(fc2_output_template.shape[2])
            fc2_token_major_template = cute.make_tensor(
                fc2_output_template.iterator,
                cute.make_layout(
                    (runtime_token_extent, runtime_hidden_extent, cutlass.Int32(1)),
                    stride=(fc2_output_template.stride[0], fc2_output_template.stride[2], 0),
                ),
            )
            fc2_operation = cpasync.CopyBulkTensorTileS2GOp()
            fc2_smem_layout = self.fc2_tma_staged_smem_layout(1, without_stage_mode=True)
            fc2_tile = (self._EpilogueTokenTileSize, self._EpilogueFc2HiddenTileSize)
            fc2_atom, fc2_tensor = cpasync.make_tiled_tma_atom(
                fc2_operation, fc2_token_major_template, fc2_smem_layout, fc2_tile
            )
        else:
            fc2_atom = None
            fc2_tensor = None
        return fc1_atom, fc1_tensor, fc2_atom, fc2_tensor

    @cute.jit
    def run(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        # ── Sched ────────────────────────────────────────────────────────
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        # ── tensors ──────────────────────────────────
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,  # Domain of fake (m, n, l)
        fc1_output_sf: cute.Tensor,  # Domain of fake (m, n, l)
        tma_atom_fc2_output: Optional[cute.CopyAtom],
        fc2_tma_output: Optional[cute.Tensor],  # Domain (physical_token, hidden, l=1)
        fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
        fc1_done_counter: cute.Tensor,  # 1D tensor
        tidx: cutlass.Int32,
        token_src_metadata: Optional[cute.Tensor],
        fc2_done_counter: Optional[cute.Tensor],
        fc2_output_sf: Optional[cute.Tensor],
        peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
        optional_epi_args: Optional[GatedActEpilogueArgs] = None,
    ):
        if cutlass.const_expr(not smem_workspace.finalized):
            raise RuntimeError("SwapABGatedActEpilogue.run requires a finalized SmemWorkspace.")
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = GatedActEpilogueArgs(
                fc1_alpha=None, fc2_alpha=None, fc1_norm_const=None, topk_scores=None
            )
        fc1_staging_pointer = smem_workspace.ptr(self.fc1_staging_region, smem_base)
        if cutlass.const_expr(self.fc2_tma_stages > 0):
            fc2_smem_tensor = smem_workspace.tensor(self.fc2_staging_region, smem_base)
        else:
            fc2_smem_tensor = None
        if cutlass.const_expr(
            self.fc2_use_tma and (tma_atom_fc2_output is None or fc2_tma_output is None)
        ):
            raise ValueError("FC2 TMA store requires a TMA atom and token-major output tensor.")
        if cutlass.const_expr(
            self.communication_enabled
            and (token_src_metadata is None or peer_rank_ptr_mapper is None)
        ):
            raise ValueError(
                "Communication-enabled Epilogue requires token metadata and a peer pointer mapper."
            )
        if cutlass.const_expr(self.token_back_enabled and fc2_done_counter is None):
            raise ValueError("Token-back requires an FC2 done counter.")
        if cutlass.const_expr(self.token_back_push_sf and fc2_output_sf is None):
            raise ValueError("Quantized token-back requires an FC2 output scale tensor.")
        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(self.accumulator_shape, stride=self.accumulator_stride),
        )

        fc1_epi = SwapABFc1Epilogue(
            self,
            tidx,
            fc1_staging_pointer,
            kernel_extension,
            tma_atom_fc1_output,
            fc1_output,
            fc1_output_sf,
            fc1_done_counter,
            optional_epi_args,
        )
        fc2_epi = SwapABFc2Epilogue(
            self,
            tidx,
            fc2_smem_tensor,
            tma_atom_fc2_output,
            fc2_tma_output,
            fc2_output,
            token_src_metadata,
            fc2_done_counter,
            fc2_output_sf,
            peer_rank_ptr_mapper,
            optional_epi_args,
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_accumulator_pipeline_stages
        )
        wait_only_named_barrier = self.epilogue_sync_barrier()
        work_tile_info = sched_consumer.consume_work()

        flag_tracker = make_flag_batch_tracker(
            True,
            flag_address=Int64(0),
            accumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            thread_idx=tidx % (self._EpilogueWarpCnt * 32),
        )

        while work_tile_info.is_valid_tile:
            tmem_acc_current = tmem_acc[None, None, acc_consumer_state.index]
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                fc1_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
            else:
                fc2_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
            prev_work_tile_info = work_tile_info
            cur_was_linear1 = prev_work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)

            acc_consumer_state.advance()

            work_tile_info = sched_consumer.consume_work()

            # Every asynchronous store commits at issue; drain before completion or scratch reuse.
            cute.arch.cp_async_bulk_wait_group(0)
            # _fence_rel_gpu()
            wait_only_named_barrier.arrive_and_wait()

            # Publish completion for the work tile snapshotted above.
            if cur_was_linear1:
                flag_tracker = fc1_epi.signal_fc1_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
            else:
                flag_tracker = fc2_epi.signal_fc2_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
        # Tail flush
        flag_tracker.fire()


class _ImmutableAfterInit:
    """Froze at the point calling `_freeze()`"""

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen_", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable after __init__ (cannot set {name!r})."
            )
        object.__setattr__(self, name, value)

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen_", True)


# Device only object
class SwapABFc1Epilogue(_ImmutableAfterInit):
    def __init__(
        self,
        base: SwapABGatedActEpilogue,
        tidx: cutlass.Int32,
        staging_pointer: cute.Pointer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,  # fake (m,n,l) domain
        fc1_output_sf: cute.Tensor,  # fake (m,n,l) domain
        fc1_done_counter: cute.Tensor,  # 1D tensor
        optional_epi_args: GatedActEpilogueArgs,
    ):
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        # (token64, intermediate, stage). The swizzle travels with the layout instead of being
        # spelled out here, so an 8-bit fc1 output picks up its own atom without a code change.
        staged_layout = base.fc1_staged_smem_layout(base.subtile_cnt)
        self.smem_tensor = cute.make_tensor(
            cute.recast_ptr(staging_pointer, staged_layout.inner, dtype=base.fc1_output_dtype),
            staged_layout.outer,
        )
        # Kept unswizzled and untyped so fc1_quant can carve an fp32 amax-exchange view out of one
        # staging stage without going through the staging tensor's swizzle.
        self.staging_pointer = staging_pointer
        self.kernel_extension = kernel_extension
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

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "SwapABFc1Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc1_done(self, work_tile_info, next_work_tile_info, flag_tracker):
        # Only in-bound intermediate_downproj tiles signal; OOB -> null slot.
        needs_intermediate_guard = (
            self.intermediate_downproj is None
            or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
        )
        if cutlass.const_expr(needs_intermediate_guard):
            in_bound = (
                work_tile_info.tile_m_idx * self._EpilogueFc1IntermediateDownTileSize
                < self.fc1_output.shape[1]
            )
        else:
            in_bound = True
        slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
        flag_address = Int64(0)
        if in_bound:
            flag_address = (self.fc1_done_counter.iterator + slot).toint()
        return flag_tracker.accumulate(
            next_work_tile_info.phase, self.fc1_epi_flag_batch, flag_address
        )

    @cute.jit
    def __call__(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,  # (cta_tile_m, cta_tile_n)
        acc_pipeline,
        acc_consumer_state,
    ):
        # (tokens_this_expert, intermediate_down, 1)
        real_fc1_output, _ = self.kernel_extension.get_gmem_tensor(
            "c", self.fc1_output, work_tile_info
        )
        # (tokens_this_expert, intermediate_down, 1)
        real_fc1_output_sf, _ = self.kernel_extension.get_gmem_tensor(
            "sfc", self.fc1_output_sf, work_tile_info
        )
        # subtile-irrevalent hoist out here.
        if cutlass.const_expr(self.optional_epi_args.fc1_alpha is not None):
            alpha_val = self.optional_epi_args.fc1_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        if cutlass.const_expr(self.optional_epi_args.fc1_norm_const is not None):
            norm_const = self.optional_epi_args.fc1_norm_const[work_tile_info.expert_idx]
        else:
            norm_const = None
        # (cta_tile_m, cta_tile_n) -> (epi_tile_m, epi_tile_n, iters)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor,
            (self._EpilogueFc1IntermediateGateUpTileSize, self._EpilogueTokenTileSize),
        )[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state)
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        for subtile_idx in cutlass.range(self.subtile_cnt, unroll=1):
            if subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx],
                    fc1_output=real_fc1_output,
                    fc1_output_sf=real_fc1_output_sf,
                    alpha_val=alpha_val,
                    norm_const=norm_const,
                )

        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        subtile_idx: cutlass.Int32,
        # (intermedaite_gateup_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        # (tokens_this_expert, intermediate_down, 1)
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
        norm_const: Optional[cutlass.Float32],
    ):
        if cutlass.const_expr(self.optional_epi_args.topk_scores is not None):
            # This means we need to perform DeepGEMM computation graph, topk_score at fc1 pre-quant
            topk_score_tensor, _ = self.kernel_extension.get_gmem_tensor(
                "topk", self.optional_epi_args.topk_scores, work_tile_info
            )  # (tokens_this_expert)
        else:
            topk_score_tensor = None

        # Mapping of the transposed accumulator for orthodox NVFP4 output:
        # (epi_tid, val_id) -> (token_idx, intermediate_down_idx)
        # token_idx = epi_tid % 32 + val_id // 16 * 32
        # intermediate_down_idx = val_id % 16 + epi_tid // 32 * 16
        # Each thread holds (intermediate_down_16, token_2):(1, 16)

        # Step -1: preload topk scores.
        current_two_token_idices = (
            work_tile_info.tile_n_idx * self.cta_tile_n
            + subtile_idx * self._EpilogueTokenTileSize
            + self.lane_idx,
            work_tile_info.tile_n_idx * self.cta_tile_n
            + subtile_idx * self._EpilogueTokenTileSize
            + self.lane_idx
            + 32,
        )
        if cutlass.const_expr(topk_score_tensor is not None):
            topk_scores = (
                topk_score_tensor[current_two_token_idices[0]],
                topk_score_tensor[current_two_token_idices[1]],
            )
        else:
            topk_scores = None

        # Step 0: load tmem
        gate_token_0_32 = cute.make_rmem_tensor((16,), cutlass.Float32)
        up_token_0_32 = cute.make_rmem_tensor((16,), cutlass.Float32)
        gate_token_32_64 = cute.make_rmem_tensor((16,), cutlass.Float32)
        up_token_32_64 = cute.make_rmem_tensor((16,), cutlass.Float32)
        # Although hardcode is not right, but since the whole tmem transpose is too tricky, I have to hardcode...
        # (epi_tile_m, epi_tile_n) -> (warp_local_epi_tile_m, epi_tile_n)
        # tmem_subtile_tensor_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[(None,
        # self.warp_idx), None]
        tmem_subtile_tensor_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[
            (None, 0), None
        ]
        # (warp_local_epi_tile_m, epi_tile_n) -> (((16, 32), 1), (2, 2))
        tmem_subtile_tensor_in_first_load_view = cute.logical_divide(
            cute.zipped_divide(tmem_subtile_tensor_per_warp, (16, 32)), ((16, 32), 1)
        )
        atom = cute.make_copy_atom(tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32)
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 0]),
            wrap_into_copy_standard_layout(gate_token_0_32),
        )
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 1]),
            wrap_into_copy_standard_layout(up_token_0_32),
        )
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 2]),
            wrap_into_copy_standard_layout(gate_token_32_64),
        )
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 3]),
            wrap_into_copy_standard_layout(up_token_32_64),
        )

        # Step 1: perform swiglu on the first part, interleave with the second's 32x32 tmem transpose.
        token_0_32_pre_quant_pre_trans = self.alpha_swiglu_clamp(
            gate_token_0_32, up_token_0_32, alpha_val
        )

        # gate_token_32_64 / up_token_32_64 are already in the transpose input
        # distribution documented by TmemTranspose16x32.
        token_32_64_tmem_trans = TmemTranspose32x32Inplace(
            tmem_subtile_tensor.iterator,
            reg_tensor_top=gate_token_32_64,
            reg_tensor_bot=up_token_32_64,
        )

        # Transpose output: each lane holds (token_1, intermediate_16); tmem_dp
        # = lane_idx (token), tmem_col = elem_idx (intermediate output idx).
        gate_token_32_64_trans_pre_act, up_token_32_64_trans_pre_act = (
            token_32_64_tmem_trans.from_r1_perm_until_last_store()
        )

        token_32_64_pre_quant = self.alpha_swiglu_clamp(
            gate_token_32_64_trans_pre_act, up_token_32_64_trans_pre_act, alpha_val
        )

        token_0_32_tmem_trans = TmemTranspose16x32(
            tmem_subtile_tensor.iterator, Region.Top, reg_tensor=token_0_32_pre_quant_pre_trans
        )
        token_0_32_pre_quant = token_0_32_tmem_trans.from_r1_perm_until_last_store()

        # Step 2: Quant
        self.fc1_quant(
            work_tile_info=work_tile_info,
            two_token=(token_0_32_pre_quant, token_32_64_pre_quant),
            topk_scores=topk_scores,
            norm_const=norm_const,
            intermediate_output_size=cute.size(fc1_output, 1),
            fc1_output_sf=fc1_output_sf,
            subtile_idx=subtile_idx,
        )

        # Step 3: TMASTG
        # (token_64, intermeidate_64)
        fc1_smem = self.smem_tensor[None, None, subtile_idx]
        # (token, intermediate_down, l=1) -> (cta_token, cta_intermediate_down)
        fc1_gmem_cta_view = cute.flat_divide(fc1_output, (self.cta_tile_n, self.cta_tile_m // 2))[
            None, None, work_tile_info.tile_n_idx, work_tile_info.tile_m_idx, 0
        ]
        # (cta_token, cta_intermediate_down) -> (token_64, intermediate_64)
        fc1_gmem_subtile_view = cute.flat_divide(
            fc1_gmem_cta_view,
            (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize),
        )[None, None, subtile_idx, 0]
        tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
            self.fc1_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(fc1_smem, 0, 2),
            cute.group_modes(fc1_gmem_subtile_view, 0, 2),
        )

        subtile_bar_id = subtile_idx + cutlass.Int32(SwapABGatedActEpilogue._EpilogueAsyncBarIdBase)
        tma_ready_to_read_smem_named_barrier = pipeline.NamedBarrier(
            barrier_id=subtile_bar_id, num_threads=self._EpilogueWarpCnt * 32
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        if self.warp_idx == subtile_idx:
            tma_ready_to_read_smem_named_barrier.arrive_and_wait()
            with cute.arch.elect_one():
                # if work_tile_info.tile_m_idx * (self.cta_tile_m // 2) < cute.size(fc1_output, 1):
                cute.copy(self.fc1_tma_atom, tma_smem_src, tma_gmem_dst)
                cute.arch.cp_async_bulk_commit_group()
        else:
            tma_ready_to_read_smem_named_barrier.arrive()

    @cute.jit
    def alpha_swiglu_clamp(
        self,
        gate_rmem: cute.Tensor,  # Raw fc1 acc (pre-dequant); even-size 1D fp32 rmem
        up_rmem: cute.Tensor,  # Raw fc1 acc (pre-dequant); even-size 1D fp32 rmem
        alpha_val: Optional[cutlass.Float32],
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
                    f"alpha_swiglu_clamp: {_name} must be a register (rmem) tensor, got address space {_t.memspace}"
                )
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
                f"{cute.size(gate_rmem)} vs {cute.size(up_rmem)}"
            )

        # gate_rmem / up_rmem are the RAW fc1 fp32 accumulator (pre-dequant).
        # Order follows the NVFP4 -> fp32 -> SwiGLU contract and MUST be:
        #
        #   1. dequant:  gate = alpha * gate_raw ; up = alpha * up_raw
        #      (alpha = expert-wise global scale on the acc; None => alpha == 1.)
        #   2. clamp the DEQUANTED (real) values, gpt-oss ``_apply_gate`` style:
        #        gate = min(gate, +limit)           (upper bound only)
        #        up   = clamp(up, -limit, +limit)   (symmetric)
        #   3. gated activation, either SwiGLU or SiTU (mutually exclusive; SiTU has no clamp):
        #        SwiGLU: out = up * gate * sigmoid(gate)
        #        SiTU:   out = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
        #                sigmoid(x) = rcp(1 + exp2(-x * log2e))
        #
        # The symmetric up-clamp is a single ``min.xorsign.abs.f32`` (magnitude
        # min(|up|, limit), sign = sign(up)^sign(limit) = sign(up) since limit>=0);
        # the gate-clamp is a plain ``min.f32``. ``.xorsign.abs`` has no f32x2 form,
        # so dequant+clamp run scalar while the swiglu core stays packed f32x2.
        n = cute.size(gate_rmem)
        out = cute.make_rmem_tensor((n,), cutlass.Float32)
        log2_e = 1.4426950408889634

        neg_log2e_pair = (cutlass.Float32(-log2_e), cutlass.Float32(-log2_e))
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
                    )
                )
                g1 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [g1.ir_value(), limit.ir_value()],
                        "min.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
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
                    )
                )
                u1 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [u1.ir_value(), limit.ir_value()],
                        "min.xorsign.abs.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )

            # 3) gated activation on the dequanted (and clamped) real values.
            #    sigmoid(x) = rcp(1 + exp2(-x * log2e)) -- shared by both cores.
            def _sigmoid(p0, p1):
                neg = cute.arch.mul_packed_f32x2((p0, p1), neg_log2e_pair)
                e = (cute.math.exp2(neg[0], fastmath=True), cute.math.exp2(neg[1], fastmath=True))
                d = cute.arch.add_packed_f32x2(e, one_pair)
                return (cute.arch.rcp_approx(d[0]), cute.arch.rcp_approx(d[1]))

            sigmoid_pair = _sigmoid(g0, g1)

            if cutlass.const_expr(self.situ_beta is None):
                # SwiGLU: out = up * gate * sigmoid(gate)
                ug = cute.arch.mul_packed_f32x2((u0, u1), (g0, g1))
                out_pair = cute.arch.mul_packed_f32x2(ug, sigmoid_pair)
            else:
                # SiTU (Kimi K3), matching HF ``modeling_kimi.py``:
                #     situ_gate = beta        * tanh(gate / beta) * sigmoid(gate)
                #     situ_up   = linear_beta * tanh(up   / linear_beta)
                #     out       = situ_gate * situ_up
                #
                # ``tanh(z) = 2 * sigmoid(2z) - 1`` keeps the whole core on the packed f32x2 path --
                # there is no packed tanh, so calling one would force this loop back to scalar.
                #
                # beta * tanh(x/beta) = beta * (2*sigmoid(2x/beta) - 1) = 2*beta*sigmoid(2x/beta) - beta
                # so the reciprocals and the 2*beta factors fold at trace time.
                inv_2beta = cutlass.Float32(2.0 / self.situ_beta)
                two_beta = cutlass.Float32(2.0 * self.situ_beta)
                neg_beta = cutlass.Float32(-self.situ_beta)
                inv_2lbeta = cutlass.Float32(2.0 / self.situ_linear_beta)
                two_lbeta = cutlass.Float32(2.0 * self.situ_linear_beta)
                neg_lbeta = cutlass.Float32(-self.situ_linear_beta)

                gs = _sigmoid(*cute.arch.mul_packed_f32x2((g0, g1), (inv_2beta, inv_2beta)))
                tanh_g = cute.arch.add_packed_f32x2(
                    cute.arch.mul_packed_f32x2(gs, (two_beta, two_beta)), (neg_beta, neg_beta)
                )

                us = _sigmoid(*cute.arch.mul_packed_f32x2((u0, u1), (inv_2lbeta, inv_2lbeta)))
                tanh_u = cute.arch.add_packed_f32x2(
                    cute.arch.mul_packed_f32x2(us, (two_lbeta, two_lbeta)), (neg_lbeta, neg_lbeta)
                )

                situ_gate = cute.arch.mul_packed_f32x2(tanh_g, sigmoid_pair)
                out_pair = cute.arch.mul_packed_f32x2(situ_gate, tanh_u)

            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]

        return out

    @cute.jit
    def fc1_quant(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        two_token: Tuple[
            cute.Tensor, cute.Tensor
        ],  # two rmem tensor, each fp32 @ (token_1, intermediate_16)
        topk_scores: Optional[Tuple[cutlass.Float32, cutlass.Float32]],
        norm_const: Optional[cutlass.Float32],
        intermediate_output_size: cutlass.Int32,
        fc1_output_sf: cute.Tensor,  # MoE domain (token_this_rank, intermediate_down, 1)
        subtile_idx: cutlass.Int32,
    ):
        # ``two_token`` are the two post-swiglu, transposed token rmem tensors; each lane holds one
        # token's 16 intermediate-output values. half 0 -> token (lane), half 1 -> (lane+32).
        #
        # Those 16 values are a whole scale block at sf_vec 16 (nvfp4) but only half of one at
        # sf_vec 32 (mx), where the other half sits in the paired warp -- hence the two directions
        # below. Both are handed the two tokens in one call so the paired path needs a single
        # exchange barrier per subtile rather than one per token.
        #
        # Per token (ported from PostSwigluHalf._gen_sfc_quantize + stg_sfc + r2s):
        #   1. (Path A) pre-multiply topk weight into the values, if present.
        #   2. block quant -> data regs + one scale factor (QuantImpl's job).
        #   3. write the scale factor to fc1_output_sf[token, intermediate_idx, 0]
        #      (plain scalar store; predicated unless statically in-bound).
        #   4. STS the quantized values into this subtile's shared output stage.
        # norm_const is treated like alpha_val: None => behaves as 1.0 (factors const-elided, not
        # multiplied by 1.0). It only exists for nvfp4; an e8m0 scale absorbs the rescale itself.
        values_per_token = cute.size(two_token[0])
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            # The exchange slots live in THIS subtile's staging stage. That stage is dead right
            # now -- the previous tile's copy of it was drained at the tile boundary, and this
            # tile writes it only after the retire barrier below -- whereas every other stage may
            # still have a TMA store reading it. Borrowing any other stage would corrupt it.
            amax_exchange = mark_alignment(
                cute.make_tensor(
                    cute.recast_ptr(self.staging_pointer, dtype=cutlass.Float32)
                    + subtile_idx * cutlass.Int32(self.fc1_staging_stage_bytes // 4),
                    cute.make_layout(
                        (self.fc1_amax_token_chunks, self._EpilogueWarpCnt * 32),
                        stride=(1, self.fc1_amax_token_chunks),
                    ),
                ),
                16,
            )
            quant = QuantImpl(
                self.quant_kind,
                "regs_in_pair_threads",
                lane_idx=self.lane_idx,
                warp_idx=self.warp_idx,
                pair_exchange_barrier=SwapABGatedActEpilogue.epilogue_sync_barrier(),
            )
        else:
            amax_exchange = None
            quant = QuantImpl(self.quant_kind, "regs_in_thread")

        # Both warps of a pair derive the same scale, so only the even one stores it. The scale
        # plane folds any coordinate inside a block onto that block's slot, so the per-warp base
        # addresses the right entry for either vec size.
        intermediate_idx = (
            work_tile_info.tile_m_idx * (self.cta_tile_m // 2)
            + self.warp_idx * self._EpilogueFc1IntermediateDownPerWarp
        )
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            stores_scale_factor = self.warp_idx % 2 == 0
        else:
            stores_scale_factor = True
        subtile_token_start = (
            work_tile_info.tile_n_idx * self.cta_tile_n + subtile_idx * self._EpilogueTokenTileSize
        )
        token_idx_pair = (
            subtile_token_start + self.lane_idx,
            subtile_token_start + self.lane_idx + 32,
        )

        # This subtile's (token, intermediate) shared output stage, tiled into (1, 16) blocks so
        # each thread's cells slice out directly (zipped_divide + slice; avoids the ambiguous
        # local_tile surface).
        smem_stage = self.smem_tensor[None, None, subtile_idx]
        # (token_64, intermediate_down_64) -> ((1, 16), (token_tile_size, warp_cnt))
        smem_tiled = cute.zipped_divide(smem_stage, (1, values_per_token))
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=values_per_token * self.fc1_output_dtype.width,
        )

        # 1) topk-weight pre-multiply (Path A) into a weighted scratch, both tokens back to back.
        weighted = cute.make_rmem_tensor((2 * values_per_token,), cutlass.Float32)
        for half in cutlass.range_constexpr(2):
            tok = two_token[half]
            base = half * values_per_token
            if cutlass.const_expr(topk_scores is not None):
                topk_pair = (topk_scores[half], topk_scores[half])
                for i in cutlass.range_constexpr(0, values_per_token, 2):
                    w0, w1 = cute.arch.mul_packed_f32x2((tok[i], tok[i + 1]), topk_pair)
                    weighted[base + i] = w0
                    weighted[base + i + 1] = w1
            else:
                for i in cutlass.range_constexpr(0, values_per_token):
                    weighted[base + i] = tok[i]

        # 2) Core block quant. One scale factor per token either way; the paired direction spends
        #    its exchange barrier inside this call.
        data_regs, sf_regs = quant(weighted, norm_const=norm_const, smem_intermediate=amax_exchange)
        data_by_token = cute.zipped_divide(data_regs, (values_per_token,))

        # Retire the exchange: its slots are this stage's own bytes, so no thread may start
        # writing the stage until every thread has read its partner's amax. Sitting after the
        # quantization rather than before it puts the sync latency behind work that is already in
        # flight (the e8m0 path alone goes through MUFU).
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            SwapABGatedActEpilogue.epilogue_sync_barrier().arrive_and_wait()

        for half in cutlass.range_constexpr(2):
            # 3) scale-factor store (predicate const-elided when statically in-bound, mirroring
            #    signal_fc1_done's intermediate predicate).
            if stores_scale_factor:
                if cutlass.const_expr(
                    self.intermediate_downproj is None
                    or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
                ):
                    if intermediate_idx < intermediate_output_size:
                        fc1_output_sf[token_idx_pair[half], intermediate_idx, 0] = sf_regs[half]
                else:
                    fc1_output_sf[token_idx_pair[half], intermediate_idx, 0] = sf_regs[half]

            # 4) STS the quantized values into this subtile's shared output stage.
            # ((1, 16), (token_tile_size, warp_cnt)) -> (16)
            smem_thread_row = smem_tiled[(0, None), (self.lane_idx + 32 * half, self.warp_idx)]
            cute.copy(
                store_atom, cute.coalesce(data_by_token[None, half]), cute.coalesce(smem_thread_row)
            )


@dataclasses.dataclass(frozen=True)
class Fc2ProcessPipeline:
    tmem_acc_load: Callable
    f2fp: Callable
    post_f2fp_reorder: Callable
    store_function: Callable
    # Kept as a finer-grained, elem-level reading aid for the store-out layout
    # (never evaluated); ``store_out_mapping`` is the per-issue form that the
    # router actually evaluates at runtime to drive metadata / pointer math.
    fc2_cta_tile_mapping: FunctionMapping
    store_out_mapping: FunctionMapping
    # SF plane per-issue mapping; None for the bf16 (unquantized) paths.
    sf_store_out_mapping: Optional[FunctionMapping] = None


# Device only object
class SwapABFc2Epilogue(_ImmutableAfterInit):
    def __init__(
        self,
        base: SwapABGatedActEpilogue,
        tidx: cutlass.Int32,
        smem_tensor: Optional[cute.Tensor],
        tma_atom_fc2_output: Optional[cute.CopyAtom],
        fc2_tma_output: Optional[cute.Tensor],
        fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
        token_src_metadata: Optional[cute.Tensor],
        fc2_done_counter: Optional[cute.Tensor],
        fc2_output_sf: Optional[cute.Tensor],
        peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
        optional_epi_args: GatedActEpilogueArgs,
    ):
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        self.tma_atom_fc2_output = tma_atom_fc2_output
        self.fc2_tma_output = fc2_tma_output
        self.fc2_output = fc2_output
        self.token_src_metadata = token_src_metadata
        self.fc2_done_counter = fc2_done_counter
        self.fc2_output_sf = fc2_output_sf
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.optional_epi_args = optional_epi_args
        if cutlass.const_expr(base.fc2_use_tma):
            self.smem_tensor = smem_tensor
            self.process_pipeline = make_fc2_tma_process_pipeline(
                combine_format=base.combine_format,
                cta_token_tile_size=base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            )
        elif cutlass.const_expr(base.fc2_use_ublk):
            self.smem_tensor = smem_tensor
            self.process_pipeline = make_fc2_ublk_process_pipeline(
                combine_format=base.combine_format,
                cta_token_tile_size=base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            )
        else:
            self.smem_tensor = None
            if cutlass.const_expr(base.reduce_topk_in_epilogue):
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

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "SwapABFc2Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc2_done(self, work_tile_info, next_work_tile_info, flag_tracker):
        publish: cutlass.Constexpr = self.token_back_enabled
        flag_address = Int64(0)
        if cutlass.const_expr(publish):
            flag_address = (self.fc2_done_counter.iterator + work_tile_info.expert_idx).toint()
        no_fire: cutlass.Constexpr = not publish
        return flag_tracker.accumulate(
            next_work_tile_info.phase, self.fc2_epi_flag_batch, flag_address, no_fire
        )

    @cute.jit
    def _make_output_router(self, work_tile_info: SwapAbFc12WorkTileInfo) -> "Fc2OutputRouter":
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n)
        )
        hidden_base_this_cta_tile = work_tile_info.tile_m_idx * cutlass.Int32(self.cta_tile_m)
        valid_hidden_this_cta_tile = (
            cutlass.Int32(self.fc2_output.shape[2]) - hidden_base_this_cta_tile
        )
        if valid_hidden_this_cta_tile < 0:
            valid_hidden_this_cta_tile = 0
        if valid_hidden_this_cta_tile > self._EpilogueFc2HiddenTileSize:
            valid_hidden_this_cta_tile = self._EpilogueFc2HiddenTileSize

        metadata = None
        peer_rank_ptr_mapper = None
        data_token_base = task_tile_data_row_start
        if cutlass.const_expr(
            self.token_src_metadata is not None and not self.token_back_push_data
        ):
            metadata = cute.domain_offset((task_tile_data_row_start,), self.token_src_metadata)
            peer_rank_ptr_mapper = self.peer_rank_ptr_mapper
            data_token_base = None

        if cutlass.const_expr(self.combine_format.is_quantized):
            base_outputs = (self.fc2_output, self.fc2_output_sf)
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
            metadata=metadata,
            token_bases=token_bases,
            base_outputs=base_outputs,
            hidden_base_this_cta_tile=hidden_base_this_cta_tile,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden_this_cta_tile=valid_hidden_this_cta_tile,
            reduce_topk_in_epilogue=self.reduce_topk_in_epilogue,
            output_mappings=output_mappings,
            epi_tid=self.tidx,
            combine_format=self.combine_format,
        ).prefetch()

    @cute.jit
    def __call__(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
    ):
        # subtile-irrelevant hoist: fc2 alpha scales raw fc2 accumulators before f2fp.
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        fc2_output_router = self._make_output_router(work_tile_info)
        # (cta_tile_m, cta_tile_n) -> (epi_tile_m, epi_tile_n, iters)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc2HiddenTileSize, self._EpilogueTokenTileSize)
        )[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        for subtile_idx in cutlass.range(self.subtile_cnt, unroll=1):
            if subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    epilogue_iter_idx=subtile_idx,
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx],
                    fc2_output_router=fc2_output_router,
                    alpha_val=alpha_val,
                )

        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        epilogue_iter_idx: cutlass.Int32,
        subtile_idx: cutlass.Int32,
        # (hidden_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        fc2_output_router: "Fc2OutputRouter",
        alpha_val: Optional[cutlass.Float32],
    ):
        process_pipeline = self.process_pipeline
        loaded = process_pipeline.tmem_acc_load(tmem_subtile_tensor=tmem_subtile_tensor, epi=self)

        casted = process_pipeline.f2fp(*loaded, alpha_val=alpha_val)
        # reorder returns a bare RMEM fragment in the store's expected pre-store
        # distribution; reorder + store are paired 1:1 inside the pipeline.
        pre_store = process_pipeline.post_f2fp_reorder(
            casted=casted, tmem_subtile_view=tmem_subtile_tensor
        )
        process_pipeline.store_function(
            epi=self,
            subtile=pre_store,
            work_tile_info=work_tile_info,
            epilogue_iter_idx=epilogue_iter_idx,
            subtile_idx=subtile_idx,
            fc2_output_router=fc2_output_router,
        )


@dataclasses.dataclass(frozen=True)
class Fc2OutputRouter:
    # One packed i64 TokenSrcMetadata record per pool token. None means a local write.
    metadata: Optional[cute.Tensor]
    # token + possible sf
    token_bases: Union[Tuple[Optional[cutlass.Int32], cutlass.Int32], Optional[cutlass.Int32]]
    base_outputs: Union[Tuple[cute.Tensor, cute.Tensor], cute.Tensor]  # (token, topk, hidden)
    hidden_base_this_cta_tile: Union[cutlass.Int32, int]
    peer_rank_ptr_mapper: Optional[SymmetricBufferDevice]
    valid_tokens_this_cta_tile: cutlass.Int32
    valid_hidden_this_cta_tile: Union[cutlass.Int32, int]
    reduce_topk_in_epilogue: bool
    # Per-issue (epi_tid, iter_idx) -> (token_cta_tile, hidden_cta_tile). Data
    # mapping, or (data mapping, sf mapping) when quantized.
    output_mappings: Union[Tuple[FunctionMapping, FunctionMapping], FunctionMapping]
    epi_tid: cutlass.Int32
    combine_format: CombineFormat
    # After metadata prefetch
    dst_ptrs: Optional[cute.Tensor] = (
        None  # i64 x (copy_iters_this_thread_cta_tile), fundamentally the pointers.
    )
    valid: Optional[cute.Tensor] = None  # (copy_iters_this_thread_cta_tile)

    @property
    def data_output(self) -> cute.Tensor:
        return self.base_outputs[0] if isinstance(self.base_outputs, tuple) else self.base_outputs

    @property
    def sf_output(self) -> Optional[cute.Tensor]:
        # Present iff quantized; (pool_token, 1, hidden // sf_vec) rank-local.
        return self.base_outputs[1] if isinstance(self.base_outputs, tuple) else None

    @property
    def data_token_base(self) -> Optional[cutlass.Int32]:
        return self.token_bases[0] if isinstance(self.token_bases, tuple) else self.token_bases

    @property
    def sf_token_base(self) -> Optional[cutlass.Int32]:
        return self.token_bases[1] if isinstance(self.token_bases, tuple) else None

    @property
    def data_mapping(self) -> FunctionMapping:
        return (
            self.output_mappings[0]
            if isinstance(self.output_mappings, tuple)
            else self.output_mappings
        )

    @property
    def sf_mapping(self) -> Optional[FunctionMapping]:
        return self.output_mappings[1] if isinstance(self.output_mappings, tuple) else None

    def __post_init__(self) -> None:
        if (self.metadata is None) == (self.data_token_base is None):
            raise ValueError(
                "Fc2OutputRouter requires exactly one of metadata or a (data) token base."
            )
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError("Fc2OutputRouter requires peer_rank_ptr_mapper iff metadata is set.")
        if self.reduce_topk_in_epilogue and self.metadata is None:
            raise ValueError("Fc2OutputRouter reduction requires metadata routing.")

    @cute.jit
    def prefetch(self) -> "Fc2OutputRouter":
        # Only the metadata (comm) path prefetches a pointer array: its
        # metadata-derived address has long-latency LDGs worth issuing early.
        # The local (no-comm) path computes its affine address on demand in
        # get_dst() -- no array, hence no runtime-indexed local-memory spill.
        if cutlass.const_expr(self.metadata is None):
            return self
        copy_iters: cutlass.Constexpr[int] = self.data_mapping.domain.axis_size("iter_idx")

        valid = cute.make_rmem_tensor((copy_iters,), cutlass.Int32)
        dst_ptrs = cute.make_rmem_tensor((copy_iters,), cutlass.Int64)

        # Compiler should be able to optimize the same token_copy_group's offset add. (Fundamental cse +
        # strength_reduce)
        # We should check the SASS to ensure this happens.
        for iter_idx in cutlass.range_constexpr(copy_iters):
            coord = self.data_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])

            valid[iter_idx] = cutlass.Int32(0)
            dst_ptrs[iter_idx] = cutlass.Int64(0)

            token_valid = token_in_tile < self.valid_tokens_this_cta_tile
            hidden_valid = hidden_in_tile < cutlass.Int32(self.valid_hidden_this_cta_tile)
            if token_valid and hidden_valid:
                valid[iter_idx] = cutlass.Int32(1)
                if cutlass.const_expr(self.metadata is None):
                    dst_tokens = self.data_token_base + token_in_tile
                    dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                    # Int64 token coord: dst_tokens*K*H overflows int32 once
                    # T*K*H exceeds 2^31 (data_output is (token, topk, hidden)).
                    dst_ptrs[iter_idx] = self.data_output[
                        Int64(dst_tokens), None, dst_hidden
                    ].iterator.toint()

                else:
                    md = TokenSrcMetadata.load(
                        self.metadata.iterator.toint()
                        + Int64(token_in_tile) * Int64(TokenSrcMetadata.nbytes)
                    )
                    dst_rank = md.src_rank
                    dst_token = md.src_token
                    dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                    if cutlass.const_expr(not self.reduce_topk_in_epilogue):
                        dst_topk = md.src_topk
                    else:
                        dst_topk = 0
                    # Int64 token coord: domain_offset on (token, topk, hidden)
                    # computes dst_token*K*H, which overflows int32 once T*K*H > 2^31.
                    dst_ptrs[iter_idx] = self.peer_rank_ptr_mapper.map_pointer(
                        cute.domain_offset(
                            (Int64(dst_token), dst_topk, dst_hidden), self.data_output
                        ).iterator,
                        dst_rank,
                        byte_alignment=32,
                    ).toint()

        return dataclasses.replace(self, dst_ptrs=dst_ptrs, valid=valid)

    @cute.jit
    def get_data_dst(
        self, iter_idx: Union[int, cutlass.Int32]
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
            coord = self.data_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
            pred = cutlass.Int32(0)
            addr = cutlass.Int64(0)
            if token_in_tile < self.valid_tokens_this_cta_tile and hidden_in_tile < cutlass.Int32(
                self.valid_hidden_this_cta_tile
            ):
                pred = cutlass.Int32(1)
                dst_tokens = self.data_token_base + token_in_tile
                dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                # Int64 token coord: dst_tokens*K*H overflows int32 once T*K*H > 2^31.
                addr = self.data_output[Int64(dst_tokens), None, dst_hidden].iterator.toint()
        else:
            # comm: read the pointer / validity prefetched by prefetch().
            addr = self.dst_ptrs[iter_idx]
            pred = self.valid[iter_idx]
        ptr = cute.make_ptr(
            self.data_output.element_type, addr, AddressSpace.gmem, assumed_align=32
        )
        return ptr, pred

    @cute.jit
    def get_sf_dst(self, iter_idx: Union[int, cutlass.Int32]) -> Tuple[cute.Pointer, cutlass.Int32]:
        """Per-issue SF destination: rank-local gmem pointer + validity predicate.

        SF never goes to a peer (it is staged locally and pushed token-contiguously
        by the dispatch / standalone warps), so this is always the affine local
        address -- no metadata routing, no prefetch. ``sf_output`` is the broadcast
        plane ``(pool_token, 1, (sf_vec, hidden//sf_vec)):(., ., (0, 1))``, so the
        logical hidden coordinate folds to its scale block on indexing.
        """
        coord = self.sf_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
        token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
        hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
        pred = cutlass.Int32(0)
        addr = cutlass.Int64(0)
        if token_in_tile < self.valid_tokens_this_cta_tile and hidden_in_tile < cutlass.Int32(
            self.valid_hidden_this_cta_tile
        ):
            pred = cutlass.Int32(1)
            sf_row = self.sf_token_base + token_in_tile
            sf_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
            addr = self.sf_output[Int64(sf_row), None, sf_hidden].iterator.toint()
        # Per-block scale offsets are element-granular; claim the scale dtype's
        # natural element alignment (e8m0 1 B / bf16 2 B).
        sf_ptr = cute.make_ptr(
            self.sf_output.element_type, addr, AddressSpace.gmem, assumed_align=4
        )
        return sf_ptr, pred


def make_fc2_stg_cta_store_out_mapping(
    combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0
    wire_dtype = combine_format.act_dtype
    assert wire_dtype.width in (4, 8, 16), "fc2 STG wire dtype must be fp4/fp8/bf16."
    elems_per_stg = min(256 // wire_dtype.width, 32)
    stgs_per_hidden32 = 32 // elems_per_stg
    fundamental_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, elem_idx: {
            "token_in_cta_tile": epi_tid % 32 + elem_idx // 32 * 32,
            "hidden_in_cta_tile": elem_idx % 32 + epi_tid // 32 * 32,
        },
    )
    store_out_mapping = FunctionMapping(
        domain=CoordinateSpace(
            ("epi_tid", "iter_idx"), (128, stgs_per_hidden32 * cta_token_tile_size // 32)
        ),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": epi_tid % 32 + iter_idx // stgs_per_hidden32 * 32,
            "hidden_in_cta_tile": (iter_idx % stgs_per_hidden32) * elems_per_stg
            + epi_tid // 32 * 32,
        },
    )
    sf_store_out_mapping = None
    if combine_format.is_quantized:

        def stg_sf_mapping(epi_tid, iter_idx):
            lane = epi_tid % 32
            warp = epi_tid // 32
            return {"token_in_cta_tile": lane + iter_idx * 32, "hidden_in_cta_tile": warp * 32}

        sf_store_out_mapping = FunctionMapping(
            domain=CoordinateSpace(("epi_tid", "iter_idx"), (128, cta_token_tile_size // 32)),
            codomain=CoordinateSpace(
                ("token_in_cta_tile", "hidden_in_cta_tile"),
                (cta_token_tile_size, cta_hidden_tile_size),
            ),
            function=stg_sf_mapping,
        )
    return store_out_mapping, sf_store_out_mapping, fundamental_mapping


def make_fc2_redg_cta_store_out_mapping(
    combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0
    # In-kernel reduce is bf16-only and never quantized, so there is no SF plane.
    assert combine_format.act_dtype.width == 16
    assert not combine_format.is_quantized

    fundamental_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, elem_idx: {
            "token_in_cta_tile": (
                ((elem_idx // 4) // 16) * 64
                + (((elem_idx // 4) % 16) // 8) * 32
                + (((elem_idx // 4) % 8) // 4) * 16
                + (((elem_idx // 4) % 4) % 2) * 8
                + (epi_tid % 32) // 4
            ),
            "hidden_in_cta_tile": (
                (epi_tid // 32) * 32
                + (epi_tid % 4) * 4
                + (((elem_idx // 4) % 4) // 2) * 16
                + elem_idx % 4
            ),
        },
    )
    # SIMT REDG emits one 8B red.v2.bf16x2 per 4 hidden elements. Each
    # 64-token subtile contributes two token rows per lane and 8 hidden
    # segments per token row.
    store_out_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (128, cta_token_tile_size // 64 * 16)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": (
                (iter_idx // 16) * 64
                + ((iter_idx % 16) // 8) * 32
                + ((iter_idx % 8) // 4) * 16
                + ((iter_idx % 4) % 2) * 8
                + (epi_tid % 32) // 4
            ),
            "hidden_in_cta_tile": (
                (epi_tid // 32) * 32 + (epi_tid % 4) * 4 + ((iter_idx % 4) // 2) * 16
            ),
        },
    )
    return store_out_mapping, None, fundamental_mapping


def make_fc2_ublk_store_out_mapping(
    combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0
    assert combine_format.act_dtype.width in (8, 16), "fc2 UBLK wire dtype must be fp8 or bf16."
    assert cta_token_tile_size <= 256
    max_token_cta_tile = 256
    fundamental_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (max_token_cta_tile, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, elem_idx: {
            "token_in_cta_tile": elem_idx // cta_hidden_tile_size * 32
            + epi_tid % 8
            + epi_tid // 32 * 8
            + ((epi_tid % 32) // 8) * 64,
            "hidden_in_cta_tile": elem_idx % cta_hidden_tile_size,
        },
    )
    copy_iters = (cta_token_tile_size + 127) // 128
    store_out_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (128, copy_iters)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (max_token_cta_tile, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": (
                iter_idx * 128 + ((epi_tid % 32) // 16) * 64 + (epi_tid // 32) * 16 + epi_tid % 16
            ),
            "hidden_in_cta_tile": 0,
        },
    )
    if combine_format.is_quantized:
        _, sf_store_out_mapping, fundamental_mapping = make_fc2_stg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    else:
        sf_store_out_mapping = None
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
def fc2_f2fp(*tensors, alpha_val: Optional[cutlass.Float32] = None, **_) -> cute.Tensor:
    reorder_dtype = cutlass.BFloat16
    total_size = 0
    for t in tensors:
        total_size += cute.size(t)
    converted_acc = cute.make_rmem_tensor((total_size,), reorder_dtype)
    elems_processed = 0
    for t in tensors:
        current_tensor_size = cute.size(t)
        dst = cute.make_tensor(
            converted_acc.iterator + elems_processed, cute.make_layout((current_tensor_size,))
        )
        if cutlass.const_expr(alpha_val is None):
            dst.store(t.load().to(reorder_dtype))
        else:
            if cutlass.const_expr(current_tensor_size % 2 != 0):
                raise ValueError("fc2_f2fp expects even elements for each input tensor.")
            scaled = cute.make_rmem_tensor((current_tensor_size,), cutlass.Float32)
            for i in cutlass.range_constexpr(0, current_tensor_size, 2):
                # scaled[i] = t[i] * alpha_val
                s0, s1 = cute.arch.mul_packed_f32x2((t[i], t[i + 1]), (alpha_val, alpha_val))
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
    atom_ld16x256 = cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32)
    ptr = tmem_subtile_tensor.iterator
    half_lane_offset = 16 * TmemTranspose32x64B16Movm._tmem_row_stride
    top_view = cute.make_tensor(ptr, TmemTranspose32x64B16Movm._tmem_layout(16, 64))
    bottom_view = cute.make_tensor(
        ptr + half_lane_offset, TmemTranspose32x64B16Movm._tmem_layout(16, 64)
    )
    top = cute.make_rmem_tensor((32,), cutlass.Float32)
    bottom = cute.make_rmem_tensor((32,), cutlass.Float32)
    cute.copy(atom_ld16x256, top_view, TmemTranspose32x64B16Movm._rmem_copy_view(top, 32))
    cute.copy(atom_ld16x256, bottom_view, TmemTranspose32x64B16Movm._rmem_copy_view(bottom, 32))
    return top, bottom


@cute.jit
def fc2_ublk_tmem_acc_load(*, tmem_subtile_tensor: cute.Tensor, epi, **_):
    # UBLK consumes a warp-local 32-hidden x 64-token slice.  The caller passes
    # the CTA-level 128-hidden x 64-token subtile view, so select this epi
    # warp's hidden block before issuing LDTM.x64.
    tmem_subtile_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[
        (None, epi.warp_idx), None
    ]
    raw_regs = cute.make_rmem_tensor((64,), cutlass.Float32)
    atom_ld32x32_x64 = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32
    )
    cute.copy(
        atom_ld32x32_x64,
        wrap_into_copy_standard_layout(tmem_subtile_per_warp),
        wrap_into_copy_standard_layout(raw_regs),
    )
    return (raw_regs,)


@cute.jit
def fc2_stg_post_f2fp_reorder(
    *,
    casted: cute.Tensor,  # (subtile_cnt,)
    tmem_subtile_view: cute.Tensor,  # (epi_tile_m, epi_tile_n)
    **_,
):
    if cutlass.const_expr(cute.size(casted) != 64):
        raise NotImplementedError("fc2 stg pass expects 64 BF16 registers before store reorder.")
    return TmemTranspose32x64B16Movm(tmem_subtile_view.iterator, casted)()


@cute.jit
def fc2_redg_post_f2fp_reorder(*, casted: cute.Tensor, tmem_subtile_view: cute.Tensor, **_):
    # (epi_tid, elem_idx) -> (token_64, hidden_128), each thread hold token_2 x hidden_32
    natural = fc2_stg_post_f2fp_reorder(casted=casted, tmem_subtile_view=tmem_subtile_view)
    core_matrix_reorder_sttm_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
    )
    core_matrix_reorder_ldtm_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), cutlass.Float32
    )
    # ((16, 2), token_32_group)
    token_groups = cute.logical_divide(cute.zipped_divide(natural, (32,)), (16, None))
    out = cute.make_rmem_tensor(token_groups.shape, casted.dtype)
    out_as_i32 = cute.recast_tensor(out, cutlass.Float32)
    # (32, 64)
    tmem_warp = cute.flat_divide(tmem_subtile_view, (32, cute.size(tmem_subtile_view, 1)))[
        None, None, 0, 0
    ]
    # (16, 16, 16dp_group, token_32_groups). Note, this tmem can provide 2x cols since the original is bf16.
    tmem_groups = cute.flat_divide(tmem_warp, (16, 16))
    for group_idx in cutlass.range_constexpr(cute.size(token_groups, 1)):
        sttm_source = cute.recast_tensor(token_groups[None, group_idx], cutlass.Float32)
        sttm_destination = tmem_groups[None, None, None, group_idx]
        cute.copy(
            core_matrix_reorder_sttm_atom,
            wrap_into_copy_standard_layout(sttm_source),
            wrap_into_copy_standard_layout(sttm_destination),
        )
        cute.copy(
            core_matrix_reorder_ldtm_atom,
            wrap_into_copy_standard_layout(tmem_groups[None, None, 0, group_idx]),
            wrap_into_copy_standard_layout(out_as_i32[(None, 0), group_idx]),
        )
        cute.copy(
            core_matrix_reorder_ldtm_atom,
            wrap_into_copy_standard_layout(tmem_groups[None, None, 1, group_idx]),
            wrap_into_copy_standard_layout(out_as_i32[(None, 1), group_idx]),
        )
    return cute.coalesce(out)


@cute.jit
def fc2_stg_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # Always BF16 before quantization
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    if cutlass.const_expr(epi.combine_format.is_quantized):
        data_subtile, sf_regs = QuantImpl(epi.combine_format, "regs_in_thread")(subtile)
    else:
        data_subtile = subtile
        sf_regs = None
    stg_width_elems: cutlass.Constexpr[int] = min(32, 256 // data_subtile.element_type.width)
    stg_bits: cutlass.Constexpr[int] = stg_width_elems * data_subtile.element_type.width
    copy_atom_vec = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=stg_bits
    )
    elems_per_thread: cutlass.Constexpr[int] = cute.size(data_subtile)
    if cutlass.const_expr(elems_per_thread % stg_width_elems != 0):
        raise ValueError(
            "fc2 STG store requires pre-store elems per thread to be divisible "
            f"by STG issue width, got {elems_per_thread} and {stg_width_elems}."
        )

    if cutlass.const_expr(sf_regs is not None):
        sf_scales_per_stg: cutlass.Constexpr[int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = (
                cutlass.Int32(subtile_idx) * cutlass.Int32(token_groups_per_subtile) + token_group
            )
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[token_group * sf_scales_per_stg + scale_idx]

    iters_per_subtile: cutlass.Constexpr[int] = elems_per_thread // stg_width_elems
    copy_src = cute.zipped_divide(data_subtile, (stg_width_elems,))
    single_copy_layout = cute.make_layout(((stg_width_elems, 1),), stride=((1, 0),))
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(iters_per_subtile)
    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            src_i = cute.make_tensor(copy_src[None, local_iter].iterator, single_copy_layout)
            dst_i = cute.make_tensor(dst_ptr, single_copy_layout)
            cute.copy(
                copy_atom_vec,
                cute.recast_tensor(src_i, cutlass.Int32),
                cute.recast_tensor(dst_i, cutlass.Int32),
            )


@cute.jit
def fc2_tma_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # Always BF16 before quantization
    work_tile_info: SwapAbFc12WorkTileInfo,
    epilogue_iter_idx: cutlass.Int32,
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    if cutlass.const_expr(
        epi.smem_tensor is None or epi.tma_atom_fc2_output is None or epi.fc2_tma_output is None
    ):
        raise ValueError(
            "FC2 TMA store requires staged SMEM, a TMA atom, and a token-major output tensor."
        )

    if cutlass.const_expr(epi.combine_format.is_quantized):
        data_subtile, sf_regs = QuantImpl(epi.combine_format, "regs_in_thread")(subtile)
    else:
        data_subtile = subtile
        sf_regs = None

    if cutlass.const_expr(sf_regs is not None):
        sf_scales_per_stg: cutlass.Constexpr[int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = (
                cutlass.Int32(subtile_idx) * cutlass.Int32(token_groups_per_subtile) + token_group
            )
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[token_group * sf_scales_per_stg + scale_idx]

    vector_elements: cutlass.Constexpr[int] = 128 // data_subtile.element_type.width
    if cutlass.const_expr(
        cute.rank(data_subtile) != 1
        or cute.size(data_subtile) != 2 * 32
        or data_subtile.stride[0] != 1
        or 32 % vector_elements != 0
    ):
        raise ValueError(
            "FC2 TMA store requires a contiguous two-token by 32-hidden register fragment."
        )

    stage_idx = epilogue_iter_idx % cutlass.Int32(epi.fc2_tma_stages)
    smem_stage = epi.smem_tensor[None, None, stage_idx]
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), data_subtile.element_type, num_bits_per_copy=128
    )
    tiler_mn = (64, 128)
    layout_copy_tv = cute.make_layout(((32, 4), (32, 2)), stride=((1, 2048), (64, 32)))
    tiled_store = cute.make_tiled_copy(store_atom, layout_copy_tv, tiler_mn)
    thread_store = tiled_store.get_slice(epi.tidx)
    smem_partition = thread_store.partition_D(smem_stage)
    register_partition = cute.composition(data_subtile, cute.make_layout(smem_partition.shape))
    cute.copy(tiled_store, register_partition, smem_partition)

    token_tile_idx = (
        work_tile_info.cumulative_data_physical_row
        + work_tile_info.tile_n_idx * cutlass.Int32(epi.cta_tile_n)
        + subtile_idx * cutlass.Int32(epi._EpilogueTokenTileSize)
    ) // cutlass.Int32(epi._EpilogueTokenTileSize)
    tiled_output = cute.flat_divide(
        epi.fc2_tma_output, (epi._EpilogueTokenTileSize, epi._EpilogueFc2HiddenTileSize)
    )
    gmem_subtile = tiled_output[None, None, token_tile_idx, work_tile_info.tile_m_idx, 0]
    tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
        epi.tma_atom_fc2_output,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_stage, 0, 2),
        cute.group_modes(gmem_subtile, 0, 2),
    )

    epilogue_barrier = SwapABGatedActEpilogue.epilogue_sync_barrier()
    cute.arch.fence_proxy("async.shared", space="cta")
    epilogue_barrier.arrive_and_wait()
    if epi.warp_idx == cutlass.Int32(0):
        cute.copy(epi.tma_atom_fc2_output, tma_smem_src, tma_gmem_dst)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(epi.fc2_tma_stages - 1, read=True)
    epilogue_barrier.arrive_and_wait()


@cute.jit
def fc2_ublk_store_function_impl(
    *,
    epi,
    subtile: cute.Tensor,  # Always bf16 pre-quant tensor
    epilogue_iter_idx: cutlass.Int32,
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    smem_tensor = epi.smem_tensor
    if cutlass.const_expr(smem_tensor is None):
        raise ValueError("fc2 UBLK store requires epi.smem_tensor.")
    quantized: cutlass.Constexpr[bool] = epi.combine_format.is_quantized
    if cutlass.const_expr(quantized):
        data_subtile, sf_regs = QuantImpl(epi.combine_format, "regs_in_thread")(subtile)
        sf_scales_per_stg: cutlass.Constexpr[int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = (
                cutlass.Int32(subtile_idx) * cutlass.Int32(token_groups_per_subtile) + token_group
            )
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[token_group * sf_scales_per_stg + scale_idx]
    else:
        data_subtile = subtile

    smem_read_write_bar = SwapABGatedActEpilogue.epilogue_sync_barrier()
    warp_idx = epi.warp_idx
    lane_idx = epi.lane_idx
    stage_idx = epilogue_iter_idx % cutlass.Int32(epi.fc2_tma_stages)
    smem_stage = smem_tensor[None, None, stage_idx]

    if cutlass.const_expr(quantized):
        vector_elements: cutlass.Constexpr[int] = 128 // data_subtile.element_type.width
        if cutlass.const_expr(
            cute.rank(data_subtile) != 1
            or cute.size(data_subtile) != 2 * 32
            or data_subtile.stride[0] != 1
            or 32 % vector_elements != 0
        ):
            raise ValueError(
                "FC2 UBLK store requires a contiguous two-token by 32-hidden register fragment."
            )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), data_subtile.element_type, num_bits_per_copy=128
        )
        tiler_mn = (64, 128)
        layout_copy_tv = cute.make_layout(((32, 4), (32, 2)), stride=((1, 2048), (64, 32)))
        tiled_store = cute.make_tiled_copy(store_atom, layout_copy_tv, tiler_mn)
        thread_store = tiled_store.get_slice(epi.tidx)
        smem_partition = thread_store.partition_D(smem_stage)
        register_partition = cute.composition(data_subtile, cute.make_layout(smem_partition.shape))
        cute.copy(tiled_store, register_partition, smem_partition)
    else:
        if cutlass.const_expr(cute.size(data_subtile) != epi._EpilogueTokenTileSize):
            raise ValueError(
                "BF16 UBLK staging requires one register per token in the 64-token subtile."
            )
        warp_hidden_base = cutlass.Int32(warp_idx * 32)
        for token_idx in cutlass.range_constexpr(epi._EpilogueTokenTileSize):
            smem_stage[token_idx, warp_hidden_base + lane_idx] = data_subtile[token_idx]

    cute.arch.fence_proxy("async.shared", space="cta")
    smem_read_write_bar.arrive_and_wait()

    copy_elems = cutlass.Int32(epi._EpilogueFc2HiddenTileSize)
    if cutlass.const_expr(epi.fc2_hidden_needs_predicate):
        copy_elems = cutlass.Int32(fc2_output_router.valid_hidden_this_cta_tile)
    copy_bytes = copy_elems * epi.combine_format.act_dtype.width // 8
    scratch_row = warp_idx * cutlass.Int32(16) + lane_idx % cutlass.Int32(16)
    copy_iters: cutlass.Constexpr[int] = fc2_output_router.data_mapping.domain.axis_size("iter_idx")
    for ublk_iter_idx in cutlass.range_constexpr(copy_iters):
        owned_subtile = cutlass.Int32(ublk_iter_idx * 2) + lane_idx // cutlass.Int32(16)
        if owned_subtile == subtile_idx:
            dst_ptr, pred = fc2_output_router.get_data_dst(ublk_iter_idx)
            if pred != cutlass.Int32(0):
                src_row = cute.slice_(smem_stage, (scratch_row, None))
                if cutlass.const_expr(epi.reduce_topk_in_epilogue):
                    cp_reduce_async_bulk_add_bf16_s2g(dst_ptr, src_row.iterator, copy_bytes)
                else:
                    cp_async_bulk_s2g(dst_ptr, src_row.iterator, copy_bytes)

    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(epi.fc2_tma_stages - 1, read=True)
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
    iters_per_subtile: cutlass.Constexpr[int] = elems_per_thread // redg_width_elems
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(iters_per_subtile)
    subtile_by_redg_issue = cute.zipped_divide(subtile, (redg_width_elems,))

    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            bf16x4 = subtile_by_redg_issue[None, local_iter]
            packed_bf16x2 = cute.recast_tensor(bf16x4, cutlass.Float32)
            red_add_relaxed_sys_v2_bf16x2(
                dst_ptr, cutlass.Float32(packed_bf16x2[0]), cutlass.Float32(packed_bf16x2[1])
            )


def make_fc2_stg_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = (
        make_fc2_stg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_stg_post_f2fp_reorder,
        store_function=fc2_stg_store_function,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
    )


def make_fc2_tma_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = (
        make_fc2_stg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_stg_post_f2fp_reorder,
        store_function=fc2_tma_store_function,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
    )


def make_fc2_redg_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = (
        make_fc2_redg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_redg_post_f2fp_reorder,
        store_function=fc2_redg_store_function,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
    )


def make_fc2_ublk_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = make_fc2_ublk_store_out_mapping(
        combine_format, cta_token_tile_size, cta_hidden_tile_size
    )
    if combine_format.is_quantized:
        if combine_format.act_dtype.width != 8:
            raise ValueError("FC2 UBLK quantization requires an FP8 combine payload.")
        tmem_acc_load = fc2_stg_tmem_acc_load
        post_f2fp_reorder = fc2_stg_post_f2fp_reorder
    else:
        tmem_acc_load = fc2_ublk_tmem_acc_load
        post_f2fp_reorder = post_f2fp_reorder_identity
    return Fc2ProcessPipeline(
        tmem_acc_load=tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=post_f2fp_reorder,
        store_function=fc2_ublk_store_function_impl,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
    )


__all__ = [
    "Fc2OutputRouter",
    "Fc2ProcessPipeline",
    "GatedActEpilogueArgs",
    "QuantImpl",
    "SwapABGatedActEpilogue",
    "make_fc2_redg_process_pipeline",
    "make_fc2_stg_process_pipeline",
    "make_fc2_tma_process_pipeline",
    "make_fc2_ublk_process_pipeline",
]
