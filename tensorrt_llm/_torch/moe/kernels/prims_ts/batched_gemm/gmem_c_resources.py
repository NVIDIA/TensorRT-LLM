# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GMEM output resource for C (epilogue store)."""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm, nvvm as _nvvm_dialect
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental.primitives import nvvm_wrapper as _nvvm_wrapper

from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    producer_work,
    WorkAttr,
)

from .batched_gemm_config import (
    BatchedGemmConfig,
    ActKind,
    DType,
    SfLayout,
    SHUFFLED_MATRIX_ROW_GROUP,
    TMA_SWIZZLE_ATOM_BYTES,
    WARPS_PER_WARPGROUP,
    WARP_SIZE,
)
from .gmem_ab_resources import nonnegative_div, nonnegative_mod
from cutlass.experimental import primitives as prims
from flashinfer.cute_dsl.fp4_common import cvt_e2m1x8_f32

Constexpr = cutlass.Constexpr

# Named-barrier slots are part of the kernel-wide schedule.  Slots 7/8 belong
# to independent epilogue warpgroups; slot 9 synchronizes the complete
# epilogue task.  Naming them here prevents store paths from silently drifting.
_TMA_STORE_BARRIER_BASE = 7
_EPILOGUE_BARRIER_ID = 9
_FP4_VALUES_PER_PACKED_WORD = 8
_PACKED_WORD_BYTES = 4


def _convert_f32x2_to_e2m1x2(a, b):
    dst_type = _ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
    return cutlass.Int8(
        _nvvm_dialect.convert_f32x2_to_f4x2(
            Float32(a).ir_value(),
            Float32(b).ir_value(),
            dst_type,
        )
    )


def _convert_f32x2_to_e4m3x2(a, b):
    dst_type = _ir.TypeAttr.get(cutlass.Float8E4M3FN.mlir_type)
    return cutlass.Int16(
        _nvvm_dialect.convert_f32x2_to_f8x2(
            cutlass.Int16.mlir_type,
            Float32(a).ir_value(),
            Float32(b).ir_value(),
            dst_type,
            rnd=_nvvm_wrapper._to_dialect(
                prims.FPRoundingMode.RN, _nvvm_wrapper._FP_ROUNDING_MODE_TO_DIALECT
            ),
            sat=_nvvm_wrapper._to_dialect(
                prims.SaturationMode.SATFINITE,
                _nvvm_wrapper._SATURATION_MODE_TO_DIALECT,
            ),
        )
    )


@dsl_user_op
def _convert_f32x2_to_e4m3x2_and_back(a, b, *, loc=None, ip=None):
    """Pack two E4M3 scales and decode both with one native x2 conversion."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i16(), T.f32(), T.f32()]),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 packed;
            .reg .b32 h2;
            .reg .b16 h0, h1;
            cvt.rn.satfinite.e4m3x2.f32 packed, $4, $3;
            cvt.rn.f16x2.e4m3x2 h2, packed;
            mov.b32 {h0, h1}, h2;
            mov.b16 $0, packed;
            cvt.f32.f16 $1, h0;
            cvt.f32.f16 $2, h1;
        }
        """,
        "=h,=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Int16(llvm.extractvalue(T.i16(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _clamp_f32_trt(val, lower, upper, *, loc=None, ip=None):
    """TRTLLM-gen clamp with NaN-propagating finite bounds."""
    # `.NaN` propagates a NaN activation instead of selecting a finite bound.
    # Omitting `.ftz` preserves subnormal inputs.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(val).ir_value(loc=loc, ip=ip),
                Float32(lower).ir_value(loc=loc, ip=ip),
                Float32(upper).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f32 tmp;
                max.NaN.f32 tmp, $1, $2;
                min.NaN.f32 $0, tmp, $3;
            }
            """,
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clamp_upper_f32_trt(val, upper, *, loc=None, ip=None):
    """TRTLLM-gen upper clamp with NaN propagation."""
    # `.NaN` propagates a NaN activation instead of selecting the finite bound.
    # Omitting `.ftz` preserves subnormal inputs.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(val).ir_value(loc=loc, ip=ip),
                Float32(upper).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                min.NaN.f32 $0, $1, $2;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dataclass(kw_only=True)
class GmemCResource(MemoryResource):
    """GMEM output resource for BF16/FP16, plain-FP8, or quantized stores.

    TMA output paths obey one shared-memory ownership protocol:

    * Before staging, the issuing lane waits until the selected scratch stage
      is reusable; a barrier then holds every warp that can write that stage.
    * After staging, an SMEM fence plus the same writer barrier publishes a
      complete tile before the elected lane issues and commits the TMA store.
    * Stores normally remain asynchronous until the next scratch reuse.  An
      aliased A/B allocation is drained at ownership handoff because its next
      writer may be a different task, and the final group is drained once at
      epilogue-task tail.

    Bulk wait-group state is per issuing thread.  Consequently a wait without
    its following writer barrier is insufficient to protect SMEM reuse.
    """

    cfg: Constexpr[BatchedGemmConfig]
    c_tensor: Any = None  # cute.Tensor for C output
    sf_c_tensor: Any = None  # cute.Tensor for output scaling factors
    tma_c_desc: Any = None
    bias_tensor: Any = None
    scale_c_tensor: Any = None
    scale_gate_tensor: Any = None
    gemm1_alpha_tensor: Any = None
    gemm1_beta_tensor: Any = None
    gemm1_clamp_limit_tensor: Any = None
    per_token_sf_a_tensor: Any = None
    per_token_sf_b_tensor: Any = None
    route_map_view: Any = None
    tile_idx_view: Any = None
    mn_limit_view: Any = None
    total_num_padded_tokens_tensor: Any = None
    problem_m: Any = None
    problem_n: Any = None
    c_row_stride: Any = None
    sf_c_m_offset: Any = None
    gC: Any = None  # make_array_view of c_tensor
    gCBytes: Any = None
    gCInt16: Any = None
    gCInt32: Any = None
    gSfC: Any = None
    gSfCBytes: Any = None
    gSfCInt16: Any = None
    gBias: Any = None
    gScaleC: Any = None
    gScaleGate: Any = None
    gGemm1Alpha: Any = None
    gGemm1Beta: Any = None
    gGemm1ClampLimit: Any = None
    gPerTokenSfA: Any = None
    gPerTokenSfB: Any = None
    gTotalNumPaddedTokens: Any = None
    sC: Any = None
    sCInt16: Any = None
    sCInt32: Any = None
    sCFloat: Any = None
    sDsFp8Absmax: Any = None
    sBias: Any = None
    sPerTokenSfA: Any = None
    sPerTokenSfB: Any = None
    tile_scale_c: Any = None
    tile_scale_gate: Any = None
    tile_gemm1_alpha: Any = None
    tile_gemm1_beta: Any = None
    tile_gemm1_clamp_limit: Any = None
    tile_expert_idx: Any = None
    tile_token_limit: Any = None
    dsfp8_c_scale_stride: Any = None
    _alloc_sc: Constexpr[Optional[SmemAllocation]] = None
    _alloc_dsfp8_absmax: Constexpr[Optional[SmemAllocation]] = None
    _alloc_bias: Constexpr[Optional[SmemAllocation]] = None
    _alloc_per_token_sf_a: Constexpr[Optional[SmemAllocation]] = None
    _alloc_per_token_sf_b: Constexpr[Optional[SmemAllocation]] = None

    def __post_init__(self):
        if self._alloc_sc is None:
            self._alloc_sc = SmemAllocation(
                f"{self.name}_sc",
                size_bytes=self.cfg.num_bytes_c_smem_scratch,
                alignment=1024,
            )
        if (
            self._alloc_dsfp8_absmax is None
            and self.cfg.has_deepseek_fp8_c_scale
            and self.cfg.uses_deepseek_fp8_atomic_absmax
        ):
            self._alloc_dsfp8_absmax = SmemAllocation(
                f"{self.name}_dsfp8_absmax",
                size_bytes=self.cfg.epi_tile_n * self.cfg.dsfp8_row_groups_per_tile * 4,
                alignment=16,
            )
        if self._alloc_bias is None and self.cfg.has_bias_m:
            self._alloc_bias = SmemAllocation(
                f"{self.name}_bias",
                size_bytes=self.cfg.tile_m * 4,
                alignment=16,
            )
        if self._alloc_per_token_sf_a is None and self.cfg.has_per_token_sf_a:
            self._alloc_per_token_sf_a = SmemAllocation(
                f"{self.name}_per_token_sf_a",
                size_bytes=self.cfg.tile_m * 4,
                alignment=16,
            )
        if self._alloc_per_token_sf_b is None and self.cfg.has_per_token_sf_b:
            self._alloc_per_token_sf_b = SmemAllocation(
                f"{self.name}_per_token_sf_b",
                size_bytes=self.cfg.tile_n * 4,
                alignment=16,
            )
        if self.c_tensor is not None:
            self.gC = cutlass.make_array_view(self.c_tensor)
            if (
                self.cfg.uses_fp4_output_quant
                or self.cfg.uses_mxfp8_output_quant
                or self.cfg.uses_fp8_output
            ):
                self.gCBytes = cutlass.Array(self.gC.data_ptr(), dtype=cutlass.Int8)
            if self.cfg.uses_fp4_output_quant:
                self.gCInt32 = cutlass.Array(self.gC.data_ptr(), dtype=cutlass.Int32)
            if self.cfg.uses_fp8_output or self.cfg.uses_mxfp8_output_quant:
                self.gCInt16 = cutlass.Array(self.gC.data_ptr(), dtype=cutlass.Int16)
        if self.sf_c_tensor is not None:
            self.gSfC = cutlass.make_array_view(self.sf_c_tensor)
            if self.cfg.has_output_block_scaling:
                self.gSfCBytes = cutlass.Array(self.gSfC.data_ptr(), dtype=cutlass.Int8)
                self.gSfCInt16 = cutlass.Array(
                    self.gSfC.data_ptr(), dtype=cutlass.Int16
                )
        if self.bias_tensor is not None:
            self.gBias = cutlass.make_array_view(self.bias_tensor)
        if self.scale_c_tensor is not None:
            self.gScaleC = cutlass.make_array_view(self.scale_c_tensor)
        if self.scale_gate_tensor is not None:
            self.gScaleGate = cutlass.make_array_view(self.scale_gate_tensor)
        if self.gemm1_alpha_tensor is not None:
            self.gGemm1Alpha = cutlass.make_array_view(self.gemm1_alpha_tensor)
        if self.gemm1_beta_tensor is not None:
            self.gGemm1Beta = cutlass.make_array_view(self.gemm1_beta_tensor)
        if self.gemm1_clamp_limit_tensor is not None:
            self.gGemm1ClampLimit = cutlass.make_array_view(
                self.gemm1_clamp_limit_tensor
            )
        if self.per_token_sf_a_tensor is not None:
            self.gPerTokenSfA = cutlass.make_array_view(self.per_token_sf_a_tensor)
        if self.per_token_sf_b_tensor is not None:
            self.gPerTokenSfB = cutlass.make_array_view(self.per_token_sf_b_tensor)
        if self.total_num_padded_tokens_tensor is not None:
            self.gTotalNumPaddedTokens = cutlass.make_array_view(
                self.total_num_padded_tokens_tensor
            )

    def get_smem_requirements(self):
        requirements = [self._alloc_sc]
        if self._alloc_dsfp8_absmax is not None:
            requirements.append(self._alloc_dsfp8_absmax)
        if self._alloc_bias is not None:
            requirements.append(self._alloc_bias)
        if self._alloc_per_token_sf_a is not None:
            requirements.append(self._alloc_per_token_sf_a)
        if self._alloc_per_token_sf_b is not None:
            requirements.append(self._alloc_per_token_sf_b)
        return requirements

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        if cutlass.const_expr(
            self.cfg.uses_fp4_output_quant
            or self.cfg.uses_mxfp8_output_quant
            or self.cfg.uses_fp8_output
        ):
            sc_dtype = cutlass.Int8
            sc_elem_bytes = 1
        elif cutlass.const_expr(self.cfg.dtype_c == int(DType.BF16)):
            sc_dtype = cutlass.BFloat16
            sc_elem_bytes = 2
        elif cutlass.const_expr(self.cfg.dtype_c == int(DType.FP16)):
            sc_dtype = cutlass.Float16
            sc_elem_bytes = 2
        else:
            raise ValueError(f"Unsupported dtype_c output store: {self.cfg.dtype_c}")
        self.sC = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sc.offset,
            dtype=sc_dtype,
            shape=(self._alloc_sc.size_bytes // sc_elem_bytes,),
            addrspace=3,
        )
        self.sCFloat = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sc.offset,
            dtype=cutlass.Float32,
            shape=(self._alloc_sc.size_bytes // 4,),
            addrspace=3,
        )
        self.sCInt16 = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sc.offset,
            dtype=cutlass.Int16,
            shape=(self._alloc_sc.size_bytes // 2,),
            addrspace=3,
        )
        self.sCInt32 = cutlass.Array(
            context.smem_base.data_ptr() + self._alloc_sc.offset,
            dtype=cutlass.Int32,
            shape=(self._alloc_sc.size_bytes // 4,),
            addrspace=3,
        )
        if cutlass.const_expr(self._alloc_dsfp8_absmax is not None):
            self.sDsFp8Absmax = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc_dsfp8_absmax.offset,
                dtype=cutlass.Float32,
                shape=(self.cfg.epi_tile_n,),
                addrspace=3,
            )
        if cutlass.const_expr(self._alloc_bias is not None):
            self.sBias = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc_bias.offset,
                dtype=cutlass.Float32,
                shape=(self.cfg.tile_m,),
                addrspace=3,
            )
        if cutlass.const_expr(self._alloc_per_token_sf_a is not None):
            self.sPerTokenSfA = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc_per_token_sf_a.offset,
                dtype=cutlass.Float32,
                shape=(self.cfg.tile_m,),
                addrspace=3,
            )
        if cutlass.const_expr(self._alloc_per_token_sf_b is not None):
            self.sPerTokenSfB = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc_per_token_sf_b.offset,
                dtype=cutlass.Float32,
                shape=(self.cfg.tile_n,),
                addrspace=3,
            )
        self.tile_scale_c = Float32(1.0)
        self.tile_scale_gate = Float32(1.0)
        self.tile_gemm1_alpha = Float32(1.0)
        self.tile_gemm1_beta = Float32(
            1.0 if self.cfg.act_kind == int(ActKind.SITU) else 0.0
        )
        self.tile_gemm1_clamp_limit = Float32(0.0)
        self.tile_expert_idx = Int32(0)
        self.tile_token_limit = Int32(0)
        if cutlass.const_expr(self.cfg.uses_non_swap_c_tma_ring):
            # This counter survives persistent work-tile iterations.  Deriving
            # the stage from tile-local column indices can otherwise select a
            # stage that is still being read by a prior tile's TMA store.
            self._c_tma_call_idx = Int32(0)
        self.dsfp8_c_scale_stride = self.problem_n
        if cutlass.const_expr(
            self.cfg.has_deepseek_fp8_c_scale and self.gTotalNumPaddedTokens is not None
        ):
            self.dsfp8_c_scale_stride = self.gTotalNumPaddedTokens.load(
                idx=Int32(0), vector_size=1
            )[0]

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def reset_dsfp8_absmax(self, stage_info: StageInfo) -> None:
        """Clear the dedicated DeepSeek output-scale reduction scratch."""
        if cutlass.const_expr(self._alloc_dsfp8_absmax is not None):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx < Int32(self.cfg.epi_tile_n * self.cfg.dsfp8_row_groups_per_tile):
                self.sDsFp8Absmax.subview(tidx).store(Float32(0.0))
            prims.barrier_cta_sync(
                barrier_id=_EPILOGUE_BARRIER_ID,
                thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
            )

    def _c_pitch(self, output_m):
        """Row pitch of C in elements; defaults to this launch's output width."""
        if self.c_row_stride is not None:
            return self.c_row_stride
        return output_m

    def _sf_group_axis(self, output_m):
        """Extent used to compute the SF-C stride between outer-axis tiles."""
        if cutlass.const_expr(self.cfg.is_swap_ab):
            return self._c_pitch(output_m)
        return output_m

    def _use_swap_ab_quant_tma_store(self) -> bool:
        """Whether the swapAB quantized epilogue should stage C through TMA."""
        return self.cfg.use_tma_store

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def drain_store_tail(self, stage_info: StageInfo) -> None:
        """Retire the final C TMA group before the epilogue task exits.

        Bulk wait-group state belongs to each issuing thread.  Every epilogue
        thread executes the wait (a no-op for non-issuers), then the full
        epilogue group rendezvous so no thread can finish the kernel while a C
        store issued by another warpgroup is still reading shared memory.
        """
        del stage_info
        if cutlass.const_expr(self.cfg.use_tma_store):
            prims.cp_async_bulk_wait_group(0, read=True)
            prims.barrier_cta_sync(
                barrier_id=_EPILOGUE_BARRIER_ID,
                thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
            )

    @cute.jit
    def _quick_gelu(self, x, scale_gate):
        """QuickGELU approximation used for GeGLU."""
        neg_scaled = Float32(-1.702 * 1.4426950408889634) * x * scale_gate
        sigmoid = cute.math.rcp(Float32(1.0) + cute.math.exp2(neg_scaled))
        return x * sigmoid

    @cute.jit
    def _apply_gated_activation(self, linear, gate, scale_gate):
        """Apply gated activation for (linear, gate) pairs.

        Uses exp2 (native EX2 instruction) + rcp (native RCP) for fast
        sigmoid: sigmoid(x) = rcp(1 + exp2(-x * log2(e))).
        """
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.SITU)):
            linear = self._clamp_swiglu_oai_linear(linear)
            gate = self._clamp_swiglu_oai_gate(gate)
            linear_scaled = linear * scale_gate
            gate_scaled = gate * scale_gate
            left = self.tile_gemm1_beta * cute.math.tanh(
                linear_scaled / self.tile_gemm1_beta, approx=True
            )
            right = self.tile_gemm1_alpha * cute.math.tanh(
                gate_scaled / self.tile_gemm1_alpha, approx=True
            )
            neg_gate_log2e = Float32(-1.4426950408889634) * gate_scaled
            sigmoid_gate = cute.math.rcp(Float32(1.0) + cute.math.exp2(neg_gate_log2e))
            return left * right * sigmoid_gate

        if cutlass.const_expr(self.cfg.has_swiglu_oai_params):
            linear = self._clamp_swiglu_oai_linear(linear)
            gate = self._clamp_swiglu_oai_gate(gate)
            gate_scaled = gate * scale_gate
            neg_gate_log2e = (
                Float32(-1.4426950408889634) * self.tile_gemm1_alpha * gate_scaled
            )
            sigmoid_gate = cute.math.rcp(Float32(1.0) + cute.math.exp2(neg_gate_log2e))
            if cutlass.const_expr(self.cfg.has_gemm1_beta):
                linear = linear + self.tile_gemm1_beta
            return linear * gate_scaled * sigmoid_gate

        linear_scaled = linear * scale_gate
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.GEGLU)):
            return linear_scaled * self._quick_gelu(gate, scale_gate)
        # silu(gate) = gate * sigmoid(gate)
        neg_gate_log2e = Float32(-1.4426950408889634) * gate * scale_gate
        sigmoid_gate = cute.math.rcp(Float32(1.0) + cute.math.exp2(neg_gate_log2e))
        return linear_scaled * gate * sigmoid_gate

    @cute.jit
    def _fmul2(self, lhs0, lhs1, rhs0, rhs1):
        return prims.mul_packed_f32x2(
            (lhs0, lhs1),
            (rhs0, rhs1),
            ftz=True,
            rnd="rn",
        )

    @cute.jit
    def _fadd2(self, lhs0, lhs1, rhs0, rhs1):
        return prims.add_packed_f32x2(
            (lhs0, lhs1),
            (rhs0, rhs1),
            ftz=True,
            rnd="rn",
        )

    @cute.jit
    def _ffma2(self, lhs0, lhs1, rhs0, rhs1, add0, add1):
        return prims.fma_packed_f32x2(
            (lhs0, lhs1),
            (rhs0, rhs1),
            (add0, add1),
            ftz=True,
            rnd="rn",
        )

    @cute.jit
    def _fmax_ftz(self, lhs, rhs):
        """Native numeric fmax used by the FP4 output absmax reduction."""
        return prims.inline_ptx_hl(
            "max.ftz.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],
            read_only_args=[Float32(lhs), Float32(rhs)],
        )

    @cute.jit
    def _apply_gated_activation_pair(self, linear0, linear1, gate0, gate1, scale_gate):
        """Packed f32x2 SwiGLU for the two-row FP4 epilogue path."""
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.SITU)):
            linear0 = self._clamp_swiglu_oai_linear(linear0)
            linear1 = self._clamp_swiglu_oai_linear(linear1)
            gate0 = self._clamp_swiglu_oai_gate(gate0)
            gate1 = self._clamp_swiglu_oai_gate(gate1)

            linear_scaled0, linear_scaled1 = self._fmul2(
                linear0,
                linear1,
                scale_gate,
                scale_gate,
            )
            gate_scaled0, gate_scaled1 = self._fmul2(
                gate0,
                gate1,
                scale_gate,
                scale_gate,
            )

            inverse_beta = cute.math.rcp(self.tile_gemm1_beta, approx=True, ftz=True)
            left_arg0, left_arg1 = self._fmul2(
                linear_scaled0,
                linear_scaled1,
                inverse_beta,
                inverse_beta,
            )
            left0 = cute.math.tanh(left_arg0, approx=True)
            left1 = cute.math.tanh(left_arg1, approx=True)
            left0, left1 = self._fmul2(
                left0,
                left1,
                self.tile_gemm1_beta,
                self.tile_gemm1_beta,
            )

            inverse_alpha = cute.math.rcp(self.tile_gemm1_alpha, approx=True, ftz=True)
            right_arg0, right_arg1 = self._fmul2(
                gate_scaled0,
                gate_scaled1,
                inverse_alpha,
                inverse_alpha,
            )
            right0 = cute.math.tanh(right_arg0, approx=True)
            right1 = cute.math.tanh(right_arg1, approx=True)
            right0, right1 = self._fmul2(
                right0,
                right1,
                self.tile_gemm1_alpha,
                self.tile_gemm1_alpha,
            )

            neg0, neg1 = self._fmul2(
                gate_scaled0,
                gate_scaled1,
                Float32(-1.4426950408889634),
                Float32(-1.4426950408889634),
            )
            exp0 = cute.math.exp2(neg0, fastmath=True)
            exp1 = cute.math.exp2(neg1, fastmath=True)
            denom0, denom1 = self._fadd2(
                exp0,
                exp1,
                Float32(1.0),
                Float32(1.0),
            )
            sig0 = cute.math.rcp(denom0, approx=True, ftz=True)
            sig1 = cute.math.rcp(denom1, approx=True, ftz=True)
            left_right0, left_right1 = self._fmul2(
                left0,
                left1,
                right0,
                right1,
            )
            return self._fmul2(left_right0, left_right1, sig0, sig1)
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.GEGLU)):
            return (
                self._apply_gated_activation(linear0, gate0, scale_gate),
                self._apply_gated_activation(linear1, gate1, scale_gate),
            )
        if cutlass.const_expr(self.cfg.has_swiglu_oai_params):
            linear0 = self._clamp_swiglu_oai_linear(linear0)
            linear1 = self._clamp_swiglu_oai_linear(linear1)
            gate0 = self._clamp_swiglu_oai_gate(gate0)
            gate1 = self._clamp_swiglu_oai_gate(gate1)
            gate_scaled0, gate_scaled1 = self._fmul2(
                gate0,
                gate1,
                scale_gate,
                scale_gate,
            )
            neg0, neg1 = self._fmul2(
                gate_scaled0,
                gate_scaled1,
                Float32(-1.4426950408889634) * self.tile_gemm1_alpha,
                Float32(-1.4426950408889634) * self.tile_gemm1_alpha,
            )
            exp0 = cute.math.exp2(neg0, fastmath=True)
            exp1 = cute.math.exp2(neg1, fastmath=True)
            denom0, denom1 = self._fadd2(
                exp0,
                exp1,
                Float32(1.0),
                Float32(1.0),
            )
            sig0 = cute.math.rcp(denom0, approx=True, ftz=True)
            sig1 = cute.math.rcp(denom1, approx=True, ftz=True)
            linear_beta0, linear_beta1 = linear0, linear1
            if cutlass.const_expr(self.cfg.has_gemm1_beta):
                linear_beta0, linear_beta1 = self._fadd2(
                    linear0,
                    linear1,
                    self.tile_gemm1_beta,
                    self.tile_gemm1_beta,
                )
            gated0, gated1 = self._fmul2(gate_scaled0, gate_scaled1, sig0, sig1)
            return self._fmul2(linear_beta0, linear_beta1, gated0, gated1)

        linear_scaled0, linear_scaled1 = self._ffma2(
            linear0,
            linear1,
            scale_gate,
            scale_gate,
            Float32(0.0),
            Float32(0.0),
        )
        gate_scaled0, gate_scaled1 = self._fmul2(
            gate0,
            gate1,
            scale_gate,
            scale_gate,
        )
        neg0, neg1 = self._fmul2(
            gate_scaled0,
            gate_scaled1,
            Float32(-1.4426950408889634),
            Float32(-1.4426950408889634),
        )
        exp0 = cute.math.exp2(neg0, fastmath=True)
        exp1 = cute.math.exp2(neg1, fastmath=True)
        denom0, denom1 = self._fadd2(
            exp0,
            exp1,
            Float32(1.0),
            Float32(1.0),
        )
        sig0 = cute.math.rcp(denom0, approx=True, ftz=True)
        sig1 = cute.math.rcp(denom1, approx=True, ftz=True)
        gate_sig0, gate_sig1 = self._fmul2(gate0, gate1, sig0, sig1)
        return self._fmul2(linear_scaled0, linear_scaled1, gate_sig0, gate_sig1)

    @cute.jit
    def _expert_idx_for_tile(self, tile_coord_m, tile_coord_n):
        return self.tile_expert_idx

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @cute.jit
    def _token_limit_for_tile(self, tile_coord_m, tile_coord_n):
        return self.tile_token_limit

    @cute.jit
    def _bias_storage_row(self, m_row):
        # Bias uses the same physical row permutation as shuffleMatrix.
        block_size = Int32(self.cfg.shuffled_matrix_block_size)
        num_row_groups = block_size // Int32(SHUFFLED_MATRIX_ROW_GROUP)
        row_in_block = m_row % block_size
        return (
            (m_row // block_size) * block_size
            + (row_in_block % num_row_groups) * Int32(SHUFFLED_MATRIX_ROW_GROUP)
            + row_in_block // num_row_groups
        )

    @cute.jit
    def _load_bias_m(self, expert_idx, m_row, local_m_row):
        if cutlass.const_expr(self.cfg.has_bias_m and self.gBias is not None):
            if cutlass.const_expr(self.sBias is not None):
                return self.sBias.subview(local_m_row).load()
            bias_row = self._bias_storage_row(m_row)
            return self.gBias.load(
                idx=expert_idx * self.problem_m + bias_row,
                vector_size=1,
            )[0]
        return Float32(0.0)

    @cute.jit
    def _maybe_add_bias_m(self, val, expert_idx, m_row, local_m_row):
        if cutlass.const_expr(self.cfg.has_bias_m):
            val = val + self._load_bias_m(expert_idx, m_row, local_m_row)
        return val

    @cute.jit
    def _load_scale_c_global(self, expert_idx):
        if cutlass.const_expr(self.cfg.uses_global_scales and self.gScaleC is not None):
            return self.gScaleC.load(idx=expert_idx, vector_size=1)[0]
        return Float32(1.0)

    @cute.jit
    def _load_scale_gate_global(self, expert_idx):
        if cutlass.const_expr(
            self.cfg.uses_global_scales and self.gScaleGate is not None
        ):
            return self.gScaleGate.load(idx=expert_idx, vector_size=1)[0]
        return Float32(1.0)

    @cute.jit
    def _load_gemm1_alpha_global(self, expert_idx):
        if cutlass.const_expr(self.cfg.has_gemm1_alpha):
            return self.gGemm1Alpha.load(idx=expert_idx, vector_size=1)[0]
        return Float32(1.0)

    @cute.jit
    def _load_gemm1_beta_global(self, expert_idx):
        if cutlass.const_expr(self.cfg.has_gemm1_beta):
            return self.gGemm1Beta.load(idx=expert_idx, vector_size=1)[0]
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.SITU)):
            return Float32(1.0)
        return Float32(0.0)

    @cute.jit
    def _load_gemm1_clamp_limit_global(self, expert_idx):
        if cutlass.const_expr(self.cfg.has_gemm1_clamp_limit):
            return self.gGemm1ClampLimit.load(idx=expert_idx, vector_size=1)[0]
        return Float32(0.0)

    @cute.jit
    def _maybe_apply_scale_c(self, val, scale_c):
        if cutlass.const_expr(self.cfg.uses_global_scales):
            val = val * scale_c
        return val

    @cute.jit
    def _maybe_apply_scale_c_pair(self, val0, val1, scale_c):
        if cutlass.const_expr(self.cfg.uses_global_scales):
            return self._fmul2(val0, val1, scale_c, scale_c)
        return val0, val1

    @cute.jit
    def _apply_eltwise_act(self, val, scale_gate):
        """Apply a non-gated activation after the optional activation scale."""
        if cutlass.const_expr(self.cfg.uses_global_scales):
            val = val * scale_gate
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.RELU2)):
            val = cute.math.max(val, Float32(0.0))
            return val * val
        neg_val_log2e = Float32(-1.4426950408889634) * val
        sigmoid = cute.math.rcp(
            Float32(1.0) + cute.math.exp2(neg_val_log2e, fastmath=True),
            approx=True,
            ftz=True,
        )
        return val * sigmoid

    @cute.jit
    def _apply_eltwise_act_pair(self, val0, val1, scale_gate):
        if cutlass.const_expr(self.cfg.uses_global_scales):
            val0, val1 = self._fmul2(val0, val1, scale_gate, scale_gate)
        if cutlass.const_expr(self.cfg.act_kind == int(ActKind.RELU2)):
            val0 = cute.math.max(val0, Float32(0.0))
            val1 = cute.math.max(val1, Float32(0.0))
            return val0 * val0, val1 * val1
        neg_val0_log2e, neg_val1_log2e = self._fmul2(
            val0,
            val1,
            Float32(-1.4426950408889634),
            Float32(-1.4426950408889634),
        )
        exp0 = cute.math.exp2(neg_val0_log2e, fastmath=True)
        exp1 = cute.math.exp2(neg_val1_log2e, fastmath=True)
        denom0, denom1 = self._fadd2(
            exp0,
            exp1,
            Float32(1.0),
            Float32(1.0),
        )
        sigmoid0 = cute.math.rcp(denom0, approx=True, ftz=True)
        sigmoid1 = cute.math.rcp(denom1, approx=True, ftz=True)
        return self._fmul2(val0, val1, sigmoid0, sigmoid1)

    @cute.jit
    def _load_per_token_sf_a(self, local_m_row):
        if cutlass.const_expr(
            self.cfg.has_per_token_sf_a and self.sPerTokenSfA is not None
        ):
            return self.sPerTokenSfA.subview(local_m_row).load()
        return Float32(1.0)

    @cute.jit
    def _maybe_apply_per_token_sf_a(self, val, local_m_row):
        if cutlass.const_expr(self.cfg.has_per_token_sf_a):
            val = val * self._load_per_token_sf_a(local_m_row)
        return val

    @cute.jit
    def _load_per_token_sf_b(self, local_token_col):
        if cutlass.const_expr(
            self.cfg.has_per_token_sf_b and self.sPerTokenSfB is not None
        ):
            return self.sPerTokenSfB.subview(local_token_col).load()
        return Float32(1.0)

    @cute.jit
    def _maybe_apply_per_token_sf_b(self, val, local_token_col):
        if cutlass.const_expr(self.cfg.has_per_token_sf_b):
            val = val * self._load_per_token_sf_b(local_token_col)
        return val

    @cute.jit
    def _to_output_value(self, val):
        if cutlass.const_expr(self.cfg.uses_fp8_output):
            return val.to(cutlass.Float8E4M3FN)
        if cutlass.const_expr(self.cfg.uses_mxfp8_output_quant):
            return val.to(cutlass.Float8E4M3FN)
        if cutlass.const_expr(self.cfg.dtype_c == int(DType.BF16)):
            return val.to(cutlass.BFloat16)
        if cutlass.const_expr(self.cfg.dtype_c == int(DType.FP16)):
            return val.to(cutlass.Float16)
        raise ValueError(f"Unsupported dtype_c output conversion: {self.cfg.dtype_c}")

    @cute.jit
    def _scale_and_convert_output_vec(self, vec_f32, scale_c):
        """Apply the optional global scale and cast one contiguous C vector."""
        if cutlass.const_expr(self.cfg.uses_global_scales):
            vec_f32 = vec_f32 * cutlass.vector.full_like(vec_f32, scale_c)
        return self._to_output_value(vec_f32)

    @cute.jit
    def _clamp_swiglu_oai_linear(self, val):
        if cutlass.const_expr(self.cfg.has_gemm1_clamp_limit):
            limit = self.tile_gemm1_clamp_limit
            val = _clamp_f32_trt(val, -limit, limit)
        return val

    @cute.jit
    def _clamp_swiglu_oai_gate(self, val):
        if cutlass.const_expr(self.cfg.has_gemm1_clamp_limit):
            limit = self.tile_gemm1_clamp_limit
            val = _clamp_upper_f32_trt(val, limit)
        return val

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_epilogue_tile_state(self, stage_info: StageInfo) -> None:
        """Load epilogue-side scalar/tile data once at work-tile head."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        expert_idx = Int32(0)
        if cutlass.const_expr(self.tile_idx_view is not None):
            if cutlass.const_expr(self.cfg.is_swap_ab):
                token_tile = tile_coord_n
            else:
                token_tile = tile_coord_m
            token_cga = token_tile // Int32(self.cfg.token_cluster_dim)
            expert_idx = self.tile_idx_view.load(idx=token_cga, vector_size=1)[0]
        self.tile_expert_idx = expert_idx
        if cutlass.const_expr(self.cfg.is_swap_ab):
            token_limit = Int32(self.cfg.tile_n)
            token_rows = self.cfg.tile_n
        else:
            token_limit = Int32(self.cfg.tile_m)
            token_rows = self.cfg.tile_m
        if cutlass.const_expr(self.mn_limit_view is not None):
            if cutlass.const_expr(self.cfg.is_swap_ab):
                token_tile = tile_coord_n
            else:
                token_tile = tile_coord_m
            token_cga = token_tile // Int32(self.cfg.token_cluster_dim)
            token_limit = self._local_tile_limit(
                self.mn_limit_view.load(idx=token_cga, vector_size=1)[0],
                token_tile,
                token_rows,
            )
        self.tile_token_limit = token_limit
        self.tile_scale_c = self._load_scale_c_global(expert_idx)
        self.tile_scale_gate = self._load_scale_gate_global(expert_idx)
        if cutlass.const_expr(self.cfg.has_gemm1_activation_params):
            self.tile_gemm1_alpha = self._load_gemm1_alpha_global(expert_idx)
            self.tile_gemm1_beta = self._load_gemm1_beta_global(expert_idx)
            self.tile_gemm1_clamp_limit = self._load_gemm1_clamp_limit_global(
                expert_idx
            )

        if cutlass.const_expr(
            self.cfg.has_per_token_sf_a and self.sPerTokenSfA is not None
        ):
            tidx, _, _ = cute.arch.thread_idx()
            local_m_row = tidx
            if local_m_row < Int32(self.cfg.tile_m):
                m_row = tile_coord_m * Int32(self.cfg.tile_m) + local_m_row
                sf_idx = m_row
                sf_val = Float32(1.0)
                if m_row < self.problem_m:
                    # Unlike BiasType.M, the public per-token/per-channel scale
                    # tensor is kept in logical row order even when swapAB
                    # weights use the shuffled epilogue layout.
                    if cutlass.const_expr(
                        not self.cfg.is_swap_ab
                        and self.cfg.has_routed_act
                        and self.route_map_view is not None
                    ):
                        sf_idx = self.route_map_view.load(idx=m_row, vector_size=1)[0]
                    if cutlass.const_expr(self.gPerTokenSfA is not None):
                        sf_val = Float32(
                            self.gPerTokenSfA.load(idx=sf_idx, vector_size=1)[0]
                        )
                self.sPerTokenSfA.subview(local_m_row).store(sf_val)
            prims.barrier_cta_sync(
                barrier_id=6,
                thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
            )

        if cutlass.const_expr(
            self.cfg.has_per_token_sf_b and self.sPerTokenSfB is not None
        ):
            tidx, _, _ = cute.arch.thread_idx()
            local_col = tidx
            if local_col < Int32(self.cfg.tile_n):
                sf_val = Float32(1.0)
                if cutlass.const_expr(self.cfg.is_swap_ab):
                    token_base = tile_coord_n * Int32(self.cfg.tile_n)
                    token_idx = token_base + local_col
                    if local_col < token_limit:
                        if cutlass.const_expr(
                            self.cfg.has_routed_act and self.route_map_view is not None
                        ):
                            token_idx = self.route_map_view.load(
                                idx=token_idx, vector_size=1
                            )[0]
                        if cutlass.const_expr(self.gPerTokenSfB is not None):
                            sf_val = Float32(
                                self.gPerTokenSfB.load(idx=token_idx, vector_size=1)[0]
                            )
                else:
                    channel_idx = tile_coord_n * Int32(self.cfg.tile_n) + local_col
                    if channel_idx < self.problem_n:
                        if cutlass.const_expr(self.gPerTokenSfB is not None):
                            sf_val = Float32(
                                self.gPerTokenSfB.load(idx=channel_idx, vector_size=1)[
                                    0
                                ]
                            )
                self.sPerTokenSfB.subview(local_col).store(sf_val)
            prims.barrier_cta_sync(
                barrier_id=6,
                thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
            )

        if cutlass.const_expr(self.cfg.has_bias_m and self.sBias is not None):
            tidx, _, _ = cute.arch.thread_idx()
            local_m_row = tidx
            if local_m_row < Int32(self.cfg.tile_m):
                m_row = tile_coord_m * Int32(self.cfg.tile_m) + local_m_row
                bias_val = Float32(0.0)
                if cutlass.const_expr(self.gBias is not None):
                    if m_row < self.problem_m:
                        bias_row = self._bias_storage_row(m_row)
                        bias_val = self.gBias.load(
                            idx=expert_idx * self.problem_m + bias_row,
                            vector_size=1,
                        )[0]
                self.sBias.subview(local_m_row).store(bias_val)
            prims.barrier_cta_sync(
                barrier_id=_EPILOGUE_BARRIER_ID,
                thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
            )

    @producer_work
    @cute.jit
    def store_epilogue(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem: cutlass.Float32,
        t2r_rmem_1: cutlass.Float32,
        t2r_output_call_idx: Int32,
        subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        """Store FP32 accumulators through the configured C output path."""

        if self.gC is not None:
            tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
            warp_idx = cute.arch.warp_idx()
            if cutlass.const_expr(self.cfg.is_swap_ab):
                if cutlass.const_expr(
                    self.cfg.use_tile256_tmem_overlap
                    and self.cfg.warp_layout.num_epilogue_warps == 4
                ):
                    self._store_swap_ab_16x256b(
                        t2r_rmem,
                        t2r_rmem_1,
                        tile_coord_m,
                        tile_coord_n,
                        t2r_output_call_idx,
                        warp_idx,
                    )
                else:
                    self._store_swap_ab_16x256b(
                        t2r_rmem,
                        t2r_rmem_1,
                        tile_coord_m,
                        tile_coord_n,
                        subtile_idx,
                        warp_idx,
                    )
            else:
                if cutlass.const_expr(self.cfg.has_gated_epilogue):
                    # TmemC returns a group-major step index; convert it to
                    # this step's first output-fragment index.
                    fragment_call_idx = t2r_output_call_idx * Int32(
                        self.cfg.non_swap_epilogue_fragments_per_group
                    )
                elif cutlass.const_expr(self.cfg.uses_tmem_overlap_window):
                    fragment_call_idx = t2r_output_call_idx
                else:
                    fragment_call_idx = Int32(subtile_idx)
                self._store_non_swap_32x32b(
                    t2r_rmem,
                    t2r_rmem_1,
                    tile_coord_m,
                    tile_coord_n,
                    fragment_call_idx,
                    warp_idx,
                )
                if cutlass.const_expr(self.cfg.uses_non_swap_c_tma_ring):
                    self._c_tma_call_idx = self._c_tma_call_idx + Int32(1)

    @producer_work
    @cute.jit
    def store_non_swap_overlap(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem: cutlass.Float32,
        t2r_rmem_1: cutlass.Float32,
        t2r_output_call_idx: Int32,
    ) -> None:
        """Store the one or two fragments preloaded from shared TMEM columns."""
        if self.gC is not None:
            tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
            warp_idx = cute.arch.warp_idx()
            if cutlass.const_expr(self.cfg.has_gated_epilogue):
                # TmemC returns a group-major step index; convert it to
                # this step's first output-fragment index.
                fragment_call_idx = t2r_output_call_idx * Int32(
                    self.cfg.non_swap_epilogue_fragments_per_group
                )
            else:
                fragment_call_idx = t2r_output_call_idx
            self._store_non_swap_32x32b(
                t2r_rmem,
                t2r_rmem_1,
                tile_coord_m,
                tile_coord_n,
                fragment_call_idx,
                warp_idx,
            )
            if cutlass.const_expr(self.cfg.uses_non_swap_c_tma_ring):
                self._c_tma_call_idx = self._c_tma_call_idx + Int32(1)
            if cutlass.const_expr(
                not self.cfg.has_gated_epilogue
                and self.cfg.non_swap_tmem_overlap_loads == 2
            ):
                self._store_non_swap_32x32b(
                    t2r_rmem_1,
                    t2r_rmem_1,
                    tile_coord_m,
                    tile_coord_n,
                    fragment_call_idx + Int32(1),
                    warp_idx,
                )
                if cutlass.const_expr(self.cfg.uses_non_swap_c_tma_ring):
                    self._c_tma_call_idx = self._c_tma_call_idx + Int32(1)

    @cute.jit
    def _store_non_swap_32x32b(
        self,
        t2r_rmem,
        t2r_rmem_1,
        tile_coord_m,
        tile_coord_n,
        call_idx,
        warp_idx,
    ):
        """Non-swapAB: 32x32b T2R, row-major output."""
        epi_t2r_repx = self.cfg.non_swap_tmem_load_num_regs
        warp_in_epi = warp_idx - Int32(self.cfg.warp_layout.epilogue_warp_idx)
        warp_in_epi = cute.arch.make_warp_uniform(warp_in_epi)
        lane_id = cute.arch.lane_idx()

        warp_in_epi4 = warp_in_epi % Int32(WARPS_PER_WARPGROUP)
        warp_in_epi4 = cute.arch.make_warp_uniform(warp_in_epi4)
        warpgroup_idx = warp_in_epi // Int32(WARPS_PER_WARPGROUP)
        warpgroup_idx = cute.arch.make_warp_uniform(warpgroup_idx)
        row = (
            tile_coord_m * Int32(self.cfg.tile_m)
            + warp_in_epi4 * Int32(WARP_SIZE)
            + lane_id
        )
        row_in_tile = warp_in_epi4 * Int32(WARP_SIZE) + lane_id
        n = self.problem_n
        expert_idx = self._expert_idx_for_tile(tile_coord_m, tile_coord_n)
        scale_c = self.tile_scale_c
        scale_gate = self.tile_scale_gate
        token_limit = Int32(self.cfg.tile_m)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            token_limit = self._token_limit_for_tile(tile_coord_m, tile_coord_n)
        token_in_bounds = row_in_tile < token_limit
        # A physical CGA may be wider than the logical N grid. Every CTA must
        # still participate in the atom-local pipelines, but padded Y CTAs
        # must not store their out-of-range N tile. Without this predicate a
        # row-major vector store at col == problem_n aliases the next row and
        # races the valid Y=0 epilogue.
        output_tile_in_bounds = (tile_coord_n * Int32(self.cfg.tile_n)) < self.problem_n

        is_gated_act = cutlass.const_expr(
            (self.cfg.act_kind == int(ActKind.SWIGLU))
            or (self.cfg.act_kind == int(ActKind.GEGLU))
            or (self.cfg.act_kind == int(ActKind.SITU))
        )
        is_eltwise_act = cutlass.const_expr(self.cfg.has_eltwise_act)
        if cutlass.const_expr(is_gated_act):
            # Gated epilogue: adjacent columns are (gate, up) pairs.
            # Output has half the columns.
            output_n = n // Int32(2)
            col_base = tile_coord_n * Int32(self.cfg.tile_n // 2) + call_idx * Int32(
                epi_t2r_repx // 2
            )
            results_per_call = self.cfg.non_swap_gated_output_cols_per_call
            result_vals = []
            for pair_idx in cutlass.range_constexpr(results_per_call // 2):
                pi0 = pair_idx * 2
                pi1 = pi0 + 1
                # The host first interleaves (linear, gate) rows and then,
                # when requested, applies TRTLLM-gen's row shuffle.  Select
                # physical accumulator columns through that same source->dst
                # map so activation and stores remain in logical output order.
                linear0 = self._load_non_swap_accumulator(
                    t2r_rmem, t2r_rmem_1, pi0 * 2, epi_t2r_repx
                )
                gate0 = self._load_non_swap_accumulator(
                    t2r_rmem, t2r_rmem_1, pi0 * 2 + 1, epi_t2r_repx
                )
                linear1 = self._load_non_swap_accumulator(
                    t2r_rmem, t2r_rmem_1, pi1 * 2, epi_t2r_repx
                )
                gate1 = self._load_non_swap_accumulator(
                    t2r_rmem, t2r_rmem_1, pi1 * 2 + 1, epi_t2r_repx
                )
                linear0 = self._maybe_apply_per_token_sf_a(linear0, row_in_tile)
                gate0 = self._maybe_apply_per_token_sf_a(gate0, row_in_tile)
                linear1 = self._maybe_apply_per_token_sf_a(linear1, row_in_tile)
                gate1 = self._maybe_apply_per_token_sf_a(gate1, row_in_tile)
                sf_b_col0 = call_idx * Int32(epi_t2r_repx) + Int32(pi0 * 2)
                sf_b_col1 = call_idx * Int32(epi_t2r_repx) + Int32(pi1 * 2)
                linear0 = self._maybe_apply_per_token_sf_b(linear0, sf_b_col0)
                gate0 = self._maybe_apply_per_token_sf_b(gate0, sf_b_col0 + Int32(1))
                linear1 = self._maybe_apply_per_token_sf_b(linear1, sf_b_col1)
                gate1 = self._maybe_apply_per_token_sf_b(gate1, sf_b_col1 + Int32(1))
                linear0 = self._maybe_add_bias_m(linear0, expert_idx, row, row_in_tile)
                gate0 = self._maybe_add_bias_m(gate0, expert_idx, row, row_in_tile)
                linear1 = self._maybe_add_bias_m(linear1, expert_idx, row, row_in_tile)
                gate1 = self._maybe_add_bias_m(gate1, expert_idx, row, row_in_tile)
                result0, result1 = self._apply_gated_activation_pair(
                    linear0,
                    linear1,
                    gate0,
                    gate1,
                    scale_gate,
                )
                result_vals.append(result0)
                result_vals.append(result1)

            if cutlass.const_expr(self.cfg.has_output_block_scaling):
                sf_block = self.cfg.output_sf_block_size_c
                all_scaled_vals = []
                num_sf_blocks = self.cfg.non_swap_output_sf_blocks_per_call
                num_sf_pairs = num_sf_blocks // 2
                for sf_pair_idx in cutlass.range_constexpr(num_sf_pairs):
                    result_base0 = sf_pair_idx * 2 * sf_block
                    result_base1 = result_base0 + sf_block
                    block_absmax0 = cute.math.abs(result_vals[result_base0])
                    block_absmax1 = cute.math.abs(result_vals[result_base1])
                    for value_idx in cutlass.range_constexpr(1, sf_block):
                        block_absmax0 = cute.arch.fmax(
                            block_absmax0,
                            cute.math.abs(result_vals[result_base0 + value_idx]),
                        )
                        block_absmax1 = cute.arch.fmax(
                            block_absmax1,
                            cute.math.abs(result_vals[result_base1 + value_idx]),
                        )
                    # fmax filters NaNs unless every value is NaN; clamp so an
                    # all-NaN block yields SF=0 like the swap-AB reductions.
                    block_absmax0 = cute.math.max(block_absmax0, Float32(0.0))
                    block_absmax1 = cute.math.max(block_absmax1, Float32(0.0))
                    if cutlass.const_expr(self.cfg.uses_global_scales):
                        block_absmax0, block_absmax1 = self._fmul2(
                            block_absmax0,
                            block_absmax1,
                            scale_c,
                            scale_c,
                        )
                    sf_pre0, sf_pre1 = self._fmul2(
                        block_absmax0,
                        block_absmax1,
                        Float32(1.0 / 6.0),
                        Float32(1.0 / 6.0),
                    )
                    sf_packed, sf_rounded0, sf_rounded1 = (
                        _convert_f32x2_to_e4m3x2_and_back(sf_pre0, sf_pre1)
                    )
                    output_scale0 = cute.arch.fmin(
                        scale_c * cute.arch.rcp_approx(sf_rounded0),
                        Float32(3.402823466e38),
                    )
                    output_scale1 = cute.arch.fmin(
                        scale_c * cute.arch.rcp_approx(sf_rounded1),
                        Float32(3.402823466e38),
                    )
                    output_scale0 = output_scale0 * cute.arch.fmin(
                        sf_rounded0 * Float32(1.0e30), Float32(1.0)
                    )
                    output_scale1 = output_scale1 * cute.arch.fmin(
                        sf_rounded1 * Float32(1.0e30), Float32(1.0)
                    )
                    for value_idx in cutlass.range_constexpr(sf_block):
                        all_scaled_vals.append(
                            result_vals[result_base0 + value_idx] * output_scale0
                        )
                    for value_idx in cutlass.range_constexpr(sf_block):
                        all_scaled_vals.append(
                            result_vals[result_base1 + value_idx] * output_scale1
                        )

                    self._store_non_swap_output_sf_pair(
                        sf_packed,
                        row,
                        output_n,
                        col_base + Int32(result_base0),
                        token_in_bounds,
                        output_tile_in_bounds,
                    )

                if cutlass.const_expr(num_sf_blocks % 2 != 0):
                    result_base = num_sf_pairs * 2 * sf_block
                    block_absmax = cute.math.abs(result_vals[result_base])
                    for value_idx in cutlass.range_constexpr(1, sf_block):
                        block_absmax = cute.arch.fmax(
                            block_absmax,
                            cute.math.abs(result_vals[result_base + value_idx]),
                        )
                    block_absmax = cute.math.max(block_absmax, Float32(0.0))
                    if cutlass.const_expr(self.cfg.uses_global_scales):
                        block_absmax = block_absmax * scale_c

                    sf_e4m3 = (block_absmax * Float32(1.0 / 6.0)).to(
                        cutlass.Float8E4M3FN
                    )
                    sf_rounded = Float32(sf_e4m3)
                    output_scale = cute.arch.fmin(
                        scale_c * cute.arch.rcp_approx(sf_rounded),
                        Float32(3.402823466e38),
                    )
                    output_scale = output_scale * cute.arch.fmin(
                        sf_rounded * Float32(1.0e30), Float32(1.0)
                    )
                    for value_idx in cutlass.range_constexpr(sf_block):
                        all_scaled_vals.append(
                            result_vals[result_base + value_idx] * output_scale
                        )
                    self._store_non_swap_output_sf(
                        sf_e4m3.bitcast(cutlass.Int8),
                        row,
                        output_n,
                        col_base + Int32(result_base),
                        token_in_bounds,
                        output_tile_in_bounds,
                    )

                # Pack eight FP4 outputs per register. Both direct GMEM and
                # TMA-staged paths then emit the widest naturally aligned
                # vector available instead of scalar packed-byte stores.
                packed_i32_list = []
                num_packed_words = results_per_call // _FP4_VALUES_PER_PACKED_WORD
                for pack_idx in cutlass.range_constexpr(num_packed_words):
                    value_idx = pack_idx * _FP4_VALUES_PER_PACKED_WORD
                    packed_i32_list.append(
                        cvt_e2m1x8_f32(
                            all_scaled_vals[value_idx],
                            all_scaled_vals[value_idx + 1],
                            all_scaled_vals[value_idx + 2],
                            all_scaled_vals[value_idx + 3],
                            all_scaled_vals[value_idx + 4],
                            all_scaled_vals[value_idx + 5],
                            all_scaled_vals[value_idx + 6],
                            all_scaled_vals[value_idx + 7],
                        ).bitcast(cutlass.Int32)
                    )
                packed_i32 = tuple(packed_i32_list)
                if cutlass.const_expr(self.cfg.use_tma_store):
                    staged_vals_i32 = packed_i32
                    if not token_in_bounds:
                        staged_vals_i32 = tuple(
                            cutlass.Int32(0) for _ in range(num_packed_words)
                        )
                    store_barrier_id = self._tma_store_barrier_id(warpgroup_idx)
                    self._wait_for_tma_scratch_reuse(store_barrier_id)
                    self._stage_non_swap_fp4_tma(
                        staged_vals_i32,
                        row_in_tile,
                        warpgroup_idx,
                        num_packed_words,
                    )
                    self._publish_tma_scratch(store_barrier_id)
                    self._commit_non_swap_fp4_tma(
                        tile_coord_m,
                        tile_coord_n,
                        call_idx,
                        output_n,
                        output_tile_in_bounds,
                        warp_idx,
                        warpgroup_idx,
                    )
                else:
                    self._store_non_swap_fp4_gmem(
                        packed_i32,
                        row,
                        output_n,
                        col_base,
                        token_in_bounds,
                        output_tile_in_bounds,
                        num_packed_words,
                    )
            else:
                result_out_vals = []
                for result_idx in cutlass.range_constexpr(results_per_call):
                    result = self._maybe_apply_scale_c(result_vals[result_idx], scale_c)
                    result_out_vals.append(self._to_output_value(result))
                if cutlass.const_expr(
                    self.cfg.use_tma_store
                    and self.cfg.dtype_c in (int(DType.BF16), int(DType.FP16))
                ):
                    if cutlass.const_expr(self.cfg.dtype_c == int(DType.BF16)):
                        result_out_vec = cutlass.Vector.from_elements(
                            tuple(result_out_vals), dtype=cutlass.BFloat16
                        )
                    else:
                        result_out_vec = cutlass.Vector.from_elements(
                            tuple(result_out_vals), dtype=cutlass.Float16
                        )
                    self._store_non_swap_gated_16bit_tma(
                        result_out_vec,
                        row_in_tile,
                        col_base,
                        tile_coord_m,
                        tile_coord_n,
                        token_in_bounds,
                        output_tile_in_bounds,
                        warp_idx,
                        warpgroup_idx,
                    )
                else:
                    self._store_non_swap_gmem_vec(
                        tuple(result_out_vals),
                        row,
                        output_n,
                        col_base,
                        token_in_bounds,
                        output_tile_in_bounds,
                        results_per_call,
                    )
        else:
            col_base = tile_coord_n * Int32(self.cfg.tile_n) + call_idx * Int32(
                epi_t2r_repx
            )
            vec_f32 = t2r_rmem
            if cutlass.const_expr(self.cfg.use_shuffled_matrix):
                # A 32x32b load owns complete shuffle blocks (validated in
                # validate_config), so unshuffling is a register permutation;
                # output addresses and vectorized stores stay contiguous.
                vec_f32 = cutlass.Vector.from_elements(
                    tuple(
                        t2r_rmem[self._non_swap_shuffled_col_idx(vi)]
                        for vi in range(epi_t2r_repx)
                    ),
                    dtype=cutlass.Float32,
                )
            if cutlass.const_expr(self.cfg.has_per_token_sf_a):
                sf_a_vec = cutlass.vector.full_like(
                    vec_f32,
                    self._load_per_token_sf_a(row_in_tile),
                )
                vec_f32 = vec_f32 * sf_a_vec
            if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                sf_b_vals = [Float32(1.0)] * epi_t2r_repx
                for vi in cutlass.range_constexpr(epi_t2r_repx):
                    sf_b_col = call_idx * Int32(epi_t2r_repx) + Int32(vi)
                    sf_b_vals[vi] = self._load_per_token_sf_b(sf_b_col)
                sf_b_vec = cutlass.Vector.from_elements(
                    tuple(sf_b_vals), dtype=cutlass.Float32
                )
                vec_f32 = vec_f32 * sf_b_vec
            if cutlass.const_expr(self.cfg.has_bias_m):
                bias_vec = cutlass.vector.full_like(
                    vec_f32,
                    self._load_bias_m(expert_idx, row, row_in_tile),
                )
                vec_f32 = vec_f32 + bias_vec
            if cutlass.const_expr(is_eltwise_act):
                if cutlass.const_expr(self.cfg.uses_global_scales):
                    scale_gate_vec = cutlass.vector.full_like(vec_f32, scale_gate)
                    vec_f32 = vec_f32 * scale_gate_vec
                if cutlass.const_expr(self.cfg.act_kind == int(ActKind.RELU2)):
                    relu_vec = cute.math.max(
                        vec_f32, cutlass.vector.full_like(vec_f32, 0.0)
                    )
                    vec_f32 = relu_vec * relu_vec
                else:
                    if cutlass.const_expr(epi_t2r_repx % 2 != 0):
                        raise ValueError(
                            "Non-swap SiLU requires an even TMEM load width"
                        )
                    silu_vals = [Float32(0.0)] * epi_t2r_repx
                    for vi in cutlass.range_constexpr(epi_t2r_repx // 2):
                        pair_idx = vi * 2
                        silu_vals[pair_idx], silu_vals[pair_idx + 1] = (
                            self._apply_eltwise_act_pair(
                                vec_f32[pair_idx],
                                vec_f32[pair_idx + 1],
                                Float32(1.0),
                            )
                        )
                    vec_f32 = cutlass.Vector.from_elements(
                        tuple(silu_vals), dtype=cutlass.Float32
                    )
            vec_out = self._scale_and_convert_output_vec(vec_f32, scale_c)
            if cutlass.const_expr(
                self.cfg.use_tma_store and self.tma_c_desc is not None
            ):
                self._store_non_swap_tma_32x32b(
                    vec_out,
                    row_in_tile,
                    col_base,
                    tile_coord_m,
                    call_idx,
                    token_in_bounds,
                    output_tile_in_bounds,
                    warp_idx,
                )
            else:
                self._store_non_swap_gmem_vec(
                    vec_out,
                    row,
                    n,
                    col_base,
                    token_in_bounds,
                    output_tile_in_bounds,
                    epi_t2r_repx,
                )

    def _tma_store_barrier_id(self, warpgroup_idx):
        return Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx

    @cute.jit
    def _wait_for_tma_scratch_reuse(self, barrier_id):
        """Hold a warpgroup until its prior TMA read has released SMEM."""
        prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(
            barrier_id=barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )

    @cute.jit
    def _publish_tma_scratch(self, barrier_id):
        """Publish all warpgroup STS writes before its elected TMA issuer."""
        cute.arch.fence_view_async_shared()
        prims.barrier_cta_sync(
            barrier_id=barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )

    @cute.jit
    def _store_non_swap_gmem_vec(
        self,
        vec_out,
        row,
        output_n,
        col_base,
        token_in_bounds,
        output_tile_in_bounds,
        vector_elems: cutlass.Constexpr[int],
    ):
        """Store the widest legal contiguous vector, with a scalar N-tail."""
        row_in_bounds = output_tile_in_bounds
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            row_in_bounds = row_in_bounds & token_in_bounds
        if row_in_bounds:
            linear_idx = row * output_n + col_base
            element_bytes = self.cfg.dtype_c_bits // 8
            vector_bytes = element_bytes * vector_elems
            if col_base + Int32(vector_elems) <= output_n:
                packed_vec = tuple(
                    vec_out[elem_idx] for elem_idx in range(vector_elems)
                )
                self.gC.store(
                    packed_vec,
                    idx=linear_idx,
                    vector_size=vector_elems,
                    alignment=vector_bytes,
                )
            else:
                for elem_idx in cutlass.range_constexpr(vector_elems):
                    if col_base + Int32(elem_idx) < output_n:
                        self.gC.store(
                            vec_out[elem_idx],
                            idx=linear_idx + Int32(elem_idx),
                            vector_size=1,
                            alignment=element_bytes,
                        )

    @cute.jit
    def _store_non_swap_output_sf(
        self,
        sf_value,
        row,
        output_n,
        output_col,
        token_in_bounds,
        output_tile_in_bounds,
    ):
        """Store one non-swap output scale with the common OOB predicates."""
        row_in_bounds = output_tile_in_bounds
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            row_in_bounds = row_in_bounds & token_in_bounds
        if row_in_bounds & (output_col < output_n):
            sf_idx = self._sf_c_index(output_col, row, output_n)
            self.gSfCBytes.subview(sf_idx).store(sf_value)

    @cute.jit
    def _store_non_swap_output_sf_pair(
        self,
        sf_packed,
        row,
        output_n,
        output_col,
        token_in_bounds,
        output_tile_in_bounds,
    ):
        """Store two adjacent output scales together, with a one-block tail."""
        row_in_bounds = output_tile_in_bounds
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            row_in_bounds = row_in_bounds & token_in_bounds
        if row_in_bounds:
            sf_idx = self._sf_c_index(output_col, row, output_n)
            second_col = output_col + Int32(self.cfg.output_sf_block_size_c)
            if second_col < output_n:
                self.gSfCInt16.subview(sf_idx // Int32(2)).store(sf_packed)
            elif output_col < output_n:
                sf_first = cutlass.Int8(Int32(sf_packed) & Int32(0xFF))
                self.gSfCBytes.subview(sf_idx).store(sf_first)

    @cute.jit
    def _store_non_swap_fp4_gmem(
        self,
        packed_i32,
        row,
        output_n,
        col_base,
        token_in_bounds,
        output_tile_in_bounds,
        num_packed_words: cutlass.Constexpr[int],
    ):
        """Store FP4 output in maximally packed, 16-byte GMEM atoms."""
        row_in_bounds = output_tile_in_bounds
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            row_in_bounds = row_in_bounds & token_in_bounds
        if row_in_bounds:
            packed_row_base = (row * output_n + col_base) // Int32(
                _FP4_VALUES_PER_PACKED_WORD
            )
            if (
                col_base + Int32(num_packed_words * _FP4_VALUES_PER_PACKED_WORD)
                <= output_n
            ):
                words_per_vector = TMA_SWIZZLE_ATOM_BYTES // _PACKED_WORD_BYTES
                full_vectors = num_packed_words // words_per_vector
                for vector_idx in cutlass.range_constexpr(full_vectors):
                    word_base = vector_idx * words_per_vector
                    packed_vec = tuple(
                        packed_i32[word_base + word_idx]
                        for word_idx in range(words_per_vector)
                    )
                    self.gCInt32.store(
                        packed_vec,
                        idx=packed_row_base + Int32(word_base),
                        vector_size=words_per_vector,
                        alignment=TMA_SWIZZLE_ATOM_BYTES,
                    )
                tail_words = num_packed_words % words_per_vector
                if cutlass.const_expr(tail_words != 0):
                    word_base = full_vectors * words_per_vector
                    packed_tail = tuple(
                        packed_i32[word_base + word_idx]
                        for word_idx in range(tail_words)
                    )
                    self.gCInt32.store(
                        packed_tail,
                        idx=packed_row_base + Int32(word_base),
                        vector_size=tail_words,
                        alignment=tail_words * _PACKED_WORD_BYTES,
                    )
            else:
                for word_idx in cutlass.range_constexpr(num_packed_words):
                    output_col = col_base + Int32(
                        word_idx * _FP4_VALUES_PER_PACKED_WORD
                    )
                    if output_col + Int32(_FP4_VALUES_PER_PACKED_WORD) <= output_n:
                        self.gCInt32.store(
                            packed_i32[word_idx],
                            idx=packed_row_base + Int32(word_idx),
                            vector_size=1,
                            alignment=_PACKED_WORD_BYTES,
                        )

    @cute.jit
    def _non_swap_fp4_tma_smem_word_offset(self, m_local_row, n_local_col):
        """Map one packed FP4 word into the TMA-swizzled scratch tile."""
        row_bytes = Int32(self.cfg.tma_store_cols // 2)
        byte_offset = m_local_row * row_bytes + n_local_col // Int32(2)
        swizzle_mask = self._c_tma_smem_swizzle_mask(byte_offset // Int32(128))
        return (byte_offset ^ swizzle_mask) // Int32(_PACKED_WORD_BYTES)

    @cute.jit
    def _stage_non_swap_fp4_tma(
        self,
        packed_i32,
        row_in_tile,
        warpgroup_idx,
        num_packed_words: cutlass.Constexpr[int],
    ):
        """Stage FP4 as independently swizzled 16-byte SMEM atoms."""
        words_per_atom = TMA_SWIZZLE_ATOM_BYTES // _PACKED_WORD_BYTES
        group_word_offset = (
            warpgroup_idx
            * Int32(self.cfg.num_bytes_c_tma_store_per_group)
            // Int32(_PACKED_WORD_BYTES)
        )
        full_atoms = num_packed_words // words_per_atom
        for atom_idx in cutlass.range_constexpr(full_atoms):
            word_base = atom_idx * words_per_atom
            atom = tuple(
                packed_i32[word_base + word_idx] for word_idx in range(words_per_atom)
            )
            word_offset = self._non_swap_fp4_tma_smem_word_offset(
                row_in_tile,
                Int32(word_base * _FP4_VALUES_PER_PACKED_WORD),
            )
            self.sCInt32.store(
                atom,
                idx=group_word_offset + word_offset,
                vector_size=words_per_atom,
                alignment=TMA_SWIZZLE_ATOM_BYTES,
            )
        tail_words = num_packed_words % words_per_atom
        if cutlass.const_expr(tail_words != 0):
            word_base = full_atoms * words_per_atom
            tail = tuple(
                packed_i32[word_base + word_idx] for word_idx in range(tail_words)
            )
            word_offset = self._non_swap_fp4_tma_smem_word_offset(
                row_in_tile,
                Int32(word_base * _FP4_VALUES_PER_PACKED_WORD),
            )
            self.sCInt32.store(
                tail,
                idx=group_word_offset + word_offset,
                vector_size=tail_words,
                alignment=tail_words * _PACKED_WORD_BYTES,
            )

    def _non_swap_shuffled_col_idx(self, logical_col: int) -> int:
        """Map a logical batch-M column to its physical shuffled column."""
        if not self.cfg.use_shuffled_matrix:
            return logical_col
        block_size = self.cfg.shuffled_matrix_block_size
        indices = self.cfg.shuffled_matrix_row_indices
        return (logical_col // block_size) * block_size + indices[
            logical_col % block_size
        ]

    @cute.jit
    def _load_non_swap_accumulator(
        self, t2r_rmem, t2r_rmem_1, logical_col: int, fragment_regs: int
    ):
        """Load one logical column from one or two batch-M T2R fragments."""
        physical_col = self._non_swap_shuffled_col_idx(logical_col)
        if cutlass.const_expr(physical_col >= fragment_regs):
            return t2r_rmem_1[physical_col - fragment_regs]
        return t2r_rmem[physical_col]

    @cute.jit
    def _commit_non_swap_fp4_tma(
        self,
        tile_coord_m,
        tile_coord_n,
        call_idx,
        output_n,
        output_tile_in_bounds,
        warp_idx,
        warpgroup_idx,
    ):
        """Publish one 128-row by output-N FP4 warpgroup tile through B4X16."""
        store_barrier_id = self._tma_store_barrier_id(warpgroup_idx)
        group_leader = Int32(
            self.cfg.warp_layout.epilogue_warp_idx
        ) + warpgroup_idx * Int32(WARPS_PER_WARPGROUP)
        output_col_base = tile_coord_n * Int32(self.cfg.tile_n // 2) + call_idx * Int32(
            self.cfg.non_swap_tmem_load_num_regs // 2
        )
        smem_base = self.sC.subview(
            warpgroup_idx * Int32(self.cfg.num_bytes_c_tma_store_per_group)
        )
        should_store = (
            (warp_idx == group_leader)
            & prims.elect_sync()
            & output_tile_in_bounds
            & (output_col_base < output_n)
        )
        if should_store:
            prims.cp_async_bulk_tensor_global_shared_cta(
                self.tma_c_desc,
                smem_base,
                (output_col_base, tile_coord_m * Int32(self.cfg.tile_m)),
            )
            prims.cp_async_bulk_commit_group()
            if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                # The next writer can be an A/B producer running on different
                # threads.  Drain on the issuing lane before handing ownership
                # of the aliased allocation back to that task.
                prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )

    @cute.jit
    def _store_non_swap_gated_16bit_tma(
        self,
        result_out_vec,
        row_in_tile,
        output_col_base,
        tile_coord_m,
        tile_coord_n,
        token_in_bounds,
        output_tile_in_bounds,
        warp_idx,
        warpgroup_idx,
    ):
        """Accumulate one gated BF16/FP16 step and TMA-store a complete box.

        Each epilogue warpgroup owns a configuration-derived output tile.
        Keeping a separate scratch tile and named barrier per group also works
        for the rotated output order used by max TMEM overlap.
        """
        store_barrier_id = self._tma_store_barrier_id(warpgroup_idx)
        group_leader = Int32(
            self.cfg.warp_layout.epilogue_warp_idx
        ) + warpgroup_idx * Int32(WARPS_PER_WARPGROUP)
        step_output_cols = self.cfg.non_swap_gated_output_cols_per_call
        tma_store_cols = self.cfg.tma_store_cols
        calls_per_box = tma_store_cols // step_output_cols
        tile_output_col_base = tile_coord_n * Int32(self.cfg.tile_n // 2)
        local_output_col = output_col_base - tile_output_col_base
        step_in_box = (local_output_col % Int32(tma_store_cols)) // Int32(
            step_output_cols
        )
        box_elems = self.cfg.epi_tile_m * tma_store_cols
        smem_group_elem_offset = warpgroup_idx * Int32(
            box_elems * self.cfg.num_stages_c_smem
        )
        smem_stage_idx = Int32(0)
        if cutlass.const_expr(self.cfg.uses_non_swap_c_tma_ring):
            smem_stage_idx = (self._c_tma_call_idx // Int32(calls_per_box)) % Int32(
                self.cfg.num_stages_c_smem
            )
        smem_base = self.sC.subview(
            smem_group_elem_offset + smem_stage_idx * Int32(box_elems)
        )

        staged_vec = result_out_vec
        if token_in_bounds:
            pass
        else:
            staged_vec = cutlass.vector.full_like(result_out_vec, 0.0)

        # The first writer into a box waits until its selected stage is no
        # longer read by TMA. Later calls fill disjoint column slices.
        if step_in_box == Int32(0):
            prims.cp_async_bulk_wait_group(self.cfg.num_stages_c_smem - 1, read=True)
            prims.barrier_cta_sync(
                barrier_id=store_barrier_id,
                thread_count=self.cfg.epilogue_threads_per_warpgroup,
            )
        self._stage_non_swap_tma_vec(
            staged_vec,
            row_in_tile,
            step_in_box * Int32(step_output_cols),
            smem_base,
            step_output_cols,
        )

        if step_in_box == Int32(calls_per_box - 1):
            self._publish_tma_scratch(store_barrier_id)
            tma_output_col_base = output_col_base - step_in_box * Int32(
                step_output_cols
            )
            is_issuer = (warp_idx == group_leader) & prims.elect_sync()
            if is_issuer:
                if output_tile_in_bounds & (
                    tma_output_col_base < self.problem_n // Int32(2)
                ):
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_c_desc,
                        smem_base,
                        (tma_output_col_base, tile_coord_m * Int32(self.cfg.tile_m)),
                    )
                # Empty groups are intentional for padded output tiles.  They
                # keep wait_group(S - 1) aligned with the running stage ring.
                prims.cp_async_bulk_commit_group()

    @cute.jit
    def _store_non_swap_tma_32x32b(
        self,
        vec_out,
        row_in_tile,
        col_base,
        tile_coord_m,
        call_idx,
        token_in_bounds,
        output_tile_in_bounds,
        warp_idx,
    ):
        """Stage one non-swap epilogue subtile to SMEM and TMA-store it."""
        epi_t2r_repx = self.cfg.non_swap_tmem_load_num_regs
        tma_store_cols = self.cfg.tma_store_cols
        calls_per_tma = max(1, tma_store_cols // epi_t2r_repx)
        call_in_tma = call_idx % calls_per_tma
        smem_stage_offset = Int32(0)
        if cutlass.const_expr(self.cfg.num_stages_c_smem > 1):
            if cutlass.const_expr(self.cfg.uses_non_swap_c_tma_ring):
                tma_group_idx = self._c_tma_call_idx // Int32(calls_per_tma)
            else:
                tma_group_idx = call_idx // Int32(calls_per_tma)
            tma_smem_stage = tma_group_idx % Int32(self.cfg.num_stages_c_smem)
            smem_stage_stride = Int32(
                self.cfg.num_bytes_c_per_stage * 8 // self.cfg.dtype_c_bits
            )
            smem_stage_offset = tma_smem_stage * smem_stage_stride
        smem_box_base = self.sC.subview(smem_stage_offset)
        smem_col = Int32(call_in_tma * epi_t2r_repx)
        staged_vec = vec_out
        if token_in_bounds:
            pass
        else:
            staged_vec = cutlass.vector.full_like(vec_out, 0.0)
        # Keep the reuse guard immediately adjacent to every SMEM staging
        # call.  It is redundant for later disjoint slices of one TMA box, but
        # avoids making synchronization depend on a staged call index.  A
        # wait-group is local to the issuer, so the all-writer barrier must
        # follow it before any warp can overwrite the selected stage.
        prims.cp_async_bulk_wait_group(self.cfg.num_stages_c_smem - 1, read=True)
        prims.barrier_cta_sync(
            barrier_id=_EPILOGUE_BARRIER_ID,
            thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
        )
        if cutlass.const_expr(self.cfg.c_tma_swizzle_bytes >= 32):
            # Match the TMA descriptor's 32/64/128-byte swizzle for every C
            # format. Split the fragment into 16-byte atoms because TMA
            # swizzling permutes those atoms within each row; a wider vector
            # store could cross a swizzle boundary.
            self._stage_non_swap_tma_vec(
                staged_vec,
                row_in_tile,
                smem_col,
                smem_box_base,
                epi_t2r_repx,
            )
        else:
            smem_row = smem_box_base.subview(
                row_in_tile * Int32(tma_store_cols) + smem_col
            )
            smem_row.store(
                staged_vec,
                vector_size=epi_t2r_repx,
                alignment=((1 if self.cfg.uses_fp8_output else 2) * epi_t2r_repx),
            )

        if call_in_tma == Int32(calls_per_tma - 1):
            # Calls write disjoint column slices of one SMEM box. Publish and
            # issue TMA only after its final slice, independent of scheduling.
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(
                barrier_id=_EPILOGUE_BARRIER_ID,
                thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
            )
            tma_col_base = col_base - Int32(call_in_tma * epi_t2r_repx)
            is_issuer = (
                warp_idx == Int32(self.cfg.warp_layout.epilogue_warp_idx)
            ) & prims.elect_sync()
            if is_issuer:
                if output_tile_in_bounds:
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_c_desc,
                        smem_box_base,
                        (tma_col_base, tile_coord_m * Int32(self.cfg.tile_m)),
                    )
                prims.cp_async_bulk_commit_group()
                if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                    prims.cp_async_bulk_wait_group(0, read=True)
            if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                prims.barrier_cta_sync(
                    barrier_id=_EPILOGUE_BARRIER_ID,
                    thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
                )

    @cute.jit
    def _c_tma_smem_swizzle_mask(self, swizzle_row_128b):
        """XOR mask for the 128-byte SMEM band containing a C element."""
        if cutlass.const_expr(self.cfg.c_tma_swizzle_bytes % 128 == 0):
            return (swizzle_row_128b % Int32(8)) * Int32(16)
        if cutlass.const_expr(self.cfg.c_tma_swizzle_bytes % 64 == 0):
            return (swizzle_row_128b % Int32(4)) * Int32(16)
        if cutlass.const_expr(self.cfg.c_tma_swizzle_bytes % 32 == 0):
            return (swizzle_row_128b % Int32(2)) * Int32(16)
        return Int32(0)

    @cute.jit
    def _non_swap_tma_smem_element_offset(self, m_local_row, n_local_col):
        """Map a row-major C element into the TMA-swizzled scratch."""
        tma_store_cols = Int32(self.cfg.tma_store_cols)
        element_bytes = Int32(self.cfg.dtype_c_bits // 8)
        smem_offset_bytes = (m_local_row * tma_store_cols + n_local_col) * element_bytes
        # TMA derives the XOR row from 128-byte SMEM bands.  A logical row is
        # narrower than that for s32b/s64b, so using m_local_row directly
        # advances the swizzle pattern too quickly.
        swizzle_mask = self._c_tma_smem_swizzle_mask(smem_offset_bytes // Int32(128))
        return (smem_offset_bytes ^ swizzle_mask) // element_bytes

    @cute.jit
    def _stage_non_swap_tma_vec(
        self,
        vec_out,
        m_local_row,
        n_local_col,
        smem_box_base,
        vector_elems: cutlass.Constexpr[int],
    ):
        """Stage a C fragment as independently swizzled 16-byte atoms."""
        atom_elems = self.cfg.c_tma_smem_atom_elements
        full_atoms = vector_elems // atom_elems
        for atom_idx in cutlass.range_constexpr(full_atoms):
            atom_col = atom_idx * atom_elems
            atom = tuple(vec_out[atom_col + elem_idx] for elem_idx in range(atom_elems))
            smem_offset = self._non_swap_tma_smem_element_offset(
                m_local_row, n_local_col + Int32(atom_col)
            )
            smem_box_base.store(
                atom,
                idx=smem_offset,
                vector_size=atom_elems,
                alignment=TMA_SWIZZLE_ATOM_BYTES,
            )
        tail_elems = vector_elems % atom_elems
        if cutlass.const_expr(tail_elems != 0):
            atom_col = full_atoms * atom_elems
            tail = tuple(vec_out[atom_col + elem_idx] for elem_idx in range(tail_elems))
            smem_offset = self._non_swap_tma_smem_element_offset(
                m_local_row, n_local_col + Int32(atom_col)
            )
            smem_box_base.store(
                tail,
                idx=smem_offset,
                vector_size=tail_elems,
                alignment=(self.cfg.dtype_c_bits // 8) * tail_elems,
            )

    @cute.jit
    def _bf16_tma_smem_element_offset(self, m_local_row, n_local_col):
        box_rows = Int32(self.cfg.swap_tma_store_rows)
        row_block = m_local_row // box_rows
        row_in_block = m_local_row % box_rows
        box_stride = Int32(self.cfg.epi_tile_n) * box_rows
        linear = row_block * box_stride + n_local_col * box_rows + row_in_block
        smem_offset_bytes = linear * Int32(2)
        swizzle_mask = self._c_tma_smem_swizzle_mask(smem_offset_bytes // Int32(128))
        return (smem_offset_bytes ^ swizzle_mask) // Int32(2)

    @cute.jit
    def _bf16_gated_tma_smem_element_offset(self, m_local_row, n_local_col):
        return self._bf16_tma_smem_element_offset(m_local_row, n_local_col)

    @cute.jit
    def _tma_oob_c_coords_bf16(
        self,
        tile_coord_m,
        tile_coord_n,
        n_subtile_offset,
        row_offset,
        token_limit,
    ):
        m_base = tile_coord_m * Int32(self.cfg.tile_m) + row_offset
        n_base = tile_coord_n * Int32(self.cfg.tile_n)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            large_n = Int32(0x40000000)
            tile_n_i32 = Int32(self.cfg.tile_n)
            limit_mod = token_limit % tile_n_i32
            dist = (tile_n_i32 - limit_mod) % tile_n_i32
            return (
                m_base,
                n_subtile_offset + dist,
                large_n,
                n_base - dist + large_n,
            )
        return (m_base, n_base + n_subtile_offset)

    @cute.jit
    def _tma_oob_c_coords_bf16_gated(
        self,
        tile_coord_m,
        tile_coord_n,
        n_subtile_offset,
        row_offset,
        token_limit,
    ):
        m_base = tile_coord_m * Int32(self.cfg.tile_m // 2) + row_offset
        n_base = tile_coord_n * Int32(self.cfg.tile_n)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            large_n = Int32(0x40000000)
            tile_n_i32 = Int32(self.cfg.tile_n)
            limit_mod = token_limit % tile_n_i32
            dist = (tile_n_i32 - limit_mod) % tile_n_i32
            return (
                m_base,
                n_subtile_offset + dist,
                large_n,
                n_base - dist + large_n,
            )
        return (m_base, n_base + n_subtile_offset)

    @cute.jit
    def _stage_swap_ab_bf16_tma_value(
        self,
        val_out,
        m_local_row,
        n_local_col,
        output_in_bounds,
    ):
        smem_offset = self._bf16_tma_smem_element_offset(m_local_row, n_local_col)
        if output_in_bounds:
            self.sC.subview(smem_offset).store(val_out)
        else:
            self.sC.subview(smem_offset).store(self._to_output_value(Float32(0.0)))

    @cute.jit
    def _stage_swap_ab_bf16_tma_vec4(
        self,
        val0_out,
        val1_out,
        val2_out,
        val3_out,
        smem_offset,
    ):
        self.sC.store(
            (val0_out, val1_out, val2_out, val3_out),
            idx=smem_offset,
            vector_size=4,
            alignment=8,
        )

    @cute.jit
    def _stage_swap_ab_bf16_dsfp8_tma_pair(
        self,
        val0_out,
        val1_out,
        m_local_row,
        n_local_col,
        warpgroup_idx,
    ):
        smem_offset = self._bf16_tma_smem_element_offset(m_local_row, n_local_col)
        group_stride_elems = Int32(64 * self.cfg.epi_tile_n)
        self.sC.store(
            (val0_out, val1_out),
            idx=warpgroup_idx * group_stride_elems + smem_offset,
            vector_size=2,
            alignment=4,
        )

    @cute.jit
    def _stage_swap_ab_bf16_gated_tma_pair(
        self,
        val0_out,
        val1_out,
        m_local_row,
        n_local_col,
        output_in_bounds,
    ):
        smem_offset = self._bf16_gated_tma_smem_element_offset(m_local_row, n_local_col)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            self.sC.store(
                (val0_out, val1_out),
                idx=smem_offset,
                vector_size=2,
                alignment=4,
            )
            return
        if output_in_bounds:
            self.sC.store(
                (val0_out, val1_out),
                idx=smem_offset,
                vector_size=2,
                alignment=4,
            )
        else:
            zero_out = self._to_output_value(Float32(0.0))
            self.sC.store(
                (zero_out, zero_out),
                idx=smem_offset,
                vector_size=2,
                alignment=4,
            )

    @cute.jit
    def _commit_swap_ab_plain_tma(
        self,
        tile_coord_m,
        tile_coord_n,
        n_subtile_offset,
        token_limit,
        warp_idx,
        warpgroup_idx,
    ):
        cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_load()
        store_barrier_id = Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )
        warp_in_epi4 = warp_idx % Int32(WARPS_PER_WARPGROUP)
        warp_in_epi4 = cute.arch.make_warp_uniform(warp_in_epi4)
        should_store = (warp_in_epi4 == Int32(0)) & prims.elect_sync()
        c_element_bytes = self.cfg.dtype_c_bits // 8
        group_stride_elems = Int32(
            self.cfg.num_bytes_c_tma_store_per_group // c_element_bytes
        )
        smem_group_base = self.sC.subview(warpgroup_idx * group_stride_elems)
        # SwapAB's descriptor-fastest dimension is M. The complete N stripe is
        # stored at once; the row loop below emits the full M tile through the
        # warpgroup's 128-byte-or-smaller contiguous boxes.
        for pass_idx in cutlass.range_constexpr(self.cfg.tma_store_passes):
            n_slice_offset = Int32(pass_idx * self.cfg.tma_store_cols)
            slice_should_store = should_store
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                slice_should_store = slice_should_store & (
                    n_subtile_offset + n_slice_offset < token_limit
                )
            if slice_should_store:
                for row_pass_idx in cutlass.range_constexpr(
                    self.cfg.swap_tma_store_row_passes
                ):
                    row_offset = Int32(row_pass_idx * self.cfg.swap_tma_store_rows)
                    if cutlass.const_expr(self.cfg.uses_fp8_output):
                        smem_offset = self._fp8_nongated_tma_smem_byte_offset(
                            row_offset, n_slice_offset
                        )
                    else:
                        smem_offset = self._bf16_tma_smem_element_offset(
                            row_offset, n_slice_offset
                        )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_c_desc,
                        smem_group_base.subview(smem_offset),
                        self._tma_oob_c_coords_bf16(
                            tile_coord_m,
                            tile_coord_n,
                            n_subtile_offset + n_slice_offset,
                            row_offset,
                            token_limit,
                        ),
                    )
        commit_store = should_store
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            commit_store = commit_store & (n_subtile_offset < token_limit)
        if commit_store:
            prims.cp_async_bulk_commit_group()
            if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )

    @cute.jit
    def _tma_oob_c_coords_bf16_dsfp8(
        self,
        tile_coord_m,
        tile_coord_n,
        n_subtile_offset,
        warpgroup_idx,
        row_offset,
        token_limit,
    ):
        m_base = (tile_coord_m * Int32(2) + warpgroup_idx) * Int32(64) + row_offset
        n_base = tile_coord_n * Int32(self.cfg.tile_n)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            large_n = Int32(0x40000000)
            tile_n_i32 = Int32(self.cfg.tile_n)
            limit_mod = token_limit % tile_n_i32
            dist = (tile_n_i32 - limit_mod) % tile_n_i32
            return (
                m_base,
                n_subtile_offset + dist,
                large_n,
                n_base - dist + large_n,
            )
        return (m_base, n_base + n_subtile_offset)

    @cute.jit
    def _commit_swap_ab_bf16_dsfp8_tma(
        self,
        tile_coord_m,
        tile_coord_n,
        n_subtile_offset,
        token_limit,
        warp_idx,
        warpgroup_idx,
    ):
        cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_load()
        store_barrier_id = Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )
        warp_in_epi4 = warp_idx % Int32(WARPS_PER_WARPGROUP)
        warp_in_epi4 = cute.arch.make_warp_uniform(warp_in_epi4)
        should_store = (warp_in_epi4 == Int32(0)) & prims.elect_sync()
        group_stride_elems = Int32(64 * self.cfg.epi_tile_n)
        smem_group_base = self.sC.subview(warpgroup_idx * group_stride_elems)
        for pass_idx in cutlass.range_constexpr(self.cfg.tma_store_passes):
            n_slice_offset = Int32(pass_idx * self.cfg.tma_store_cols)
            slice_should_store = should_store
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                slice_should_store = slice_should_store & (
                    n_subtile_offset + n_slice_offset < token_limit
                )
            if slice_should_store:
                for row_pass_idx in cutlass.range_constexpr(
                    self.cfg.swap_tma_store_row_passes
                ):
                    row_offset = Int32(row_pass_idx * self.cfg.swap_tma_store_rows)
                    smem_offset = self._bf16_tma_smem_element_offset(
                        row_offset, n_slice_offset
                    )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_c_desc,
                        smem_group_base.subview(smem_offset),
                        self._tma_oob_c_coords_bf16_dsfp8(
                            tile_coord_m,
                            tile_coord_n,
                            n_subtile_offset + n_slice_offset,
                            warpgroup_idx,
                            row_offset,
                            token_limit,
                        ),
                    )
        commit_store = should_store
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            commit_store = commit_store & (n_subtile_offset < token_limit)
        if commit_store:
            prims.cp_async_bulk_commit_group()
            if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )

    @cute.jit
    def _commit_swap_ab_bf16_gated_tma(
        self,
        tile_coord_m,
        tile_coord_n,
        n_subtile_offset,
        token_limit,
        warp_idx,
    ):
        cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_load()
        prims.barrier_cta_sync(
            barrier_id=_TMA_STORE_BARRIER_BASE,
            thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
        )
        warp_in_epi4 = warp_idx % Int32(WARPS_PER_WARPGROUP)
        warp_in_epi4 = cute.arch.make_warp_uniform(warp_in_epi4)
        should_store = (warp_in_epi4 == Int32(0)) & prims.elect_sync()
        for pass_idx in cutlass.range_constexpr(self.cfg.tma_store_passes):
            n_slice_offset = Int32(pass_idx * self.cfg.tma_store_cols)
            slice_should_store = should_store
            if cutlass.const_expr(not self.cfg.use_tma_oob_opt):
                slice_should_store = slice_should_store & (
                    n_subtile_offset + n_slice_offset < token_limit
                )
            if slice_should_store:
                for row_pass_idx in cutlass.range_constexpr(
                    self.cfg.swap_tma_store_row_passes
                ):
                    row_offset = Int32(row_pass_idx * self.cfg.swap_tma_store_rows)
                    smem_offset = self._bf16_gated_tma_smem_element_offset(
                        row_offset, n_slice_offset
                    )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_c_desc,
                        self.sC.subview(smem_offset),
                        self._tma_oob_c_coords_bf16_gated(
                            tile_coord_m,
                            tile_coord_n,
                            n_subtile_offset + n_slice_offset,
                            row_offset,
                            token_limit,
                        ),
                    )
        commit_store = should_store
        if cutlass.const_expr(not self.cfg.use_tma_oob_opt):
            commit_store = commit_store & (n_subtile_offset < token_limit)
        if commit_store:
            prims.cp_async_bulk_commit_group()
            if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(
            barrier_id=_TMA_STORE_BARRIER_BASE,
            thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
        )

    @cute.jit
    def _reduce_fp4_absmax_m16(self, val):
        """Reduce a positive max value across the 8 lanes owning one M16 block.

        NaNs are filtered by the NaN-quiet max against 0.0 below
        (max(NaN, 0.0) = 0.0), so all-NaN local contributions reduce as zero.
        """
        if cutlass.const_expr(self.cfg.has_deepseek_fp8_c_scale):
            val = cute.arch.fmax(cute.arch.fmax(val, -val), Float32(0.0))
        else:
            val = cute.math.max(cute.math.abs(val), Float32(0.0))
        return self._reduce_positive_max_lanes(val, True)

    @cute.jit
    def _reduce_positive_max_lanes(self, val, include_xor16: cutlass.Constexpr[bool]):
        """Butterfly-reduce a non-negative maximum across 4 or 8 lane groups."""
        for lane_mask in (4, 8):
            other = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=val.bitcast(cutlass.Int32),
                offset=lane_mask,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
                return_value_and_is_valid=False,
            ).bitcast(cutlass.Float32)
            val = self._fmax_ftz(val, other)
        if cutlass.const_expr(include_xor16):
            other = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=val.bitcast(cutlass.Int32),
                offset=16,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
                return_value_and_is_valid=False,
            ).bitcast(cutlass.Float32)
            val = self._fmax_ftz(val, other)
        return val

    @cute.jit
    def _trunc_abs_float_to_pow2(self, val):
        val_i32 = Float32(val).bitcast(cutlass.Int32)
        return (val_i32 & Int32(0x7F800000)).bitcast(cutlass.Float32)

    @cute.jit
    def _scale_rcp_exp_only(self, val):
        val_i32 = Float32(val).bitcast(cutlass.Int32)
        return (Int32(0x7F000000) - val_i32).bitcast(cutlass.Float32)

    @cute.jit
    def _absmax_scratch_base(self, warpgroup_idx):
        scratch_base = Int32(0)
        if cutlass.const_expr(
            self.cfg.use_tma_store and not self.cfg.has_deepseek_fp8_c_scale
        ):
            scratch_base = Int32(
                self.cfg.num_bytes_c_tma_store_per_group
                * self.cfg.num_epilogue_warpgroups
                // 4
            )
        scratch_group_stride = Int32(max(1, self.cfg.epi_tile_n // 8) * 32)
        return scratch_base + warpgroup_idx * scratch_group_stride

    @cute.jit
    def _absmax_scratch_idx(self, warpgroup_idx, warp_in_epi4, lane_id, scale_slot):
        lane_col = (lane_id & Int32(3)) << Int32(1)
        scratch_warp_stride = Int32(max(1, self.cfg.epi_tile_n // 8) * 8)
        slot_col = (scale_slot >> Int32(1)) * Int32(8) + (scale_slot & Int32(1))
        return (
            self._absmax_scratch_base(warpgroup_idx)
            + warp_in_epi4 * scratch_warp_stride
            + slot_col
            + lane_col
        )

    @cute.jit
    def _absmax_scratch_pair_idx(
        self, warpgroup_idx, warp_in_epi4, lane_id, scale_pair
    ):
        lane_col = (lane_id & Int32(3)) << Int32(1)
        scratch_warp_stride = Int32(max(1, self.cfg.epi_tile_n // 8) * 8)
        return (
            self._absmax_scratch_base(warpgroup_idx)
            + warp_in_epi4 * scratch_warp_stride
            + scale_pair * Int32(8)
            + lane_col
        )

    @cute.jit
    def _write_absmax_scratch(
        self, val, warpgroup_idx, warp_in_epi4, lane_id, scale_slot
    ):
        """Phase 1: reduce M16 within warp and write to SMEM scratch.

        Call once per scale_slot BEFORE the batched bar.sync.
        """
        warp_absmax = self._reduce_fp4_absmax_m16(val)
        smem_idx = self._absmax_scratch_idx(
            warpgroup_idx, warp_in_epi4, lane_id, scale_slot
        )
        if (lane_id >> Int32(2)) == Int32(0):
            self.sCFloat.subview(smem_idx).store(warp_absmax)

    @cute.jit
    def _write_absmax_scratch_pair(
        self, val0, val1, warpgroup_idx, warp_in_epi4, lane_id, scale_pair
    ):
        warp_absmax0 = self._reduce_fp4_absmax_m16(val0)
        warp_absmax1 = self._reduce_fp4_absmax_m16(val1)
        smem_idx = self._absmax_scratch_pair_idx(
            warpgroup_idx, warp_in_epi4, lane_id, scale_pair
        )
        if (lane_id >> Int32(2)) == Int32(0):
            self.sCFloat.store(
                (warp_absmax0, warp_absmax1),
                smem_idx,
                vector_size=2,
                alignment=8,
            )

    @cute.jit
    def _atomic_dsfp8_absmax_scratch_pair(
        self, val0, val1, lane_id, scale_pair, row_group
    ):
        warp_absmax0 = self._reduce_fp4_absmax_m16(val0)
        warp_absmax1 = self._reduce_fp4_absmax_m16(val1)
        lane_col = (lane_id % Int32(4)) * Int32(2)
        scratch_idx = (
            row_group * Int32(self.cfg.epi_tile_n) + scale_pair * Int32(8) + lane_col
        )
        if (lane_id // Int32(4)) == Int32(0):
            cute.arch.atomic_fmax(
                ptr=self.sDsFp8Absmax.data_ptr(scratch_idx),
                val=warp_absmax0,
                sign_bit=False,
                sem="relaxed",
                scope="cta",
            )
            cute.arch.atomic_fmax(
                ptr=self.sDsFp8Absmax.data_ptr(scratch_idx + Int32(1)),
                val=warp_absmax1,
                sign_bit=False,
                sem="relaxed",
                scope="cta",
            )

    @cute.jit
    def _read_absmax_scratch(self, warpgroup_idx, warp_in_epi4, lane_id, scale_slot):
        """Phase 2: read own + partner warp's scratch and combine.

        Call once per scale_slot AFTER the batched bar.sync.
        """
        partner_warp = warp_in_epi4 ^ Int32(1)
        own_idx = self._absmax_scratch_idx(
            warpgroup_idx, warp_in_epi4, lane_id, scale_slot
        )
        partner_idx = self._absmax_scratch_idx(
            warpgroup_idx, partner_warp, lane_id, scale_slot
        )
        own_val = self.sCFloat.subview(own_idx).load()
        partner_val = self.sCFloat.subview(partner_idx).load()
        if cutlass.const_expr(self.cfg.has_deepseek_fp8_c_scale):
            return cute.arch.fmax(own_val, partner_val)
        return cute.math.max(own_val, partner_val)

    @cute.jit
    def _read_absmax_scratch_pair(
        self, warpgroup_idx, warp_in_epi4, lane_id, scale_pair
    ):
        partner_warp = warp_in_epi4 ^ Int32(1)
        own_idx = self._absmax_scratch_pair_idx(
            warpgroup_idx, warp_in_epi4, lane_id, scale_pair
        )
        partner_idx = self._absmax_scratch_pair_idx(
            warpgroup_idx, partner_warp, lane_id, scale_pair
        )
        own_vec = self.sCFloat.load(own_idx, vector_size=2, alignment=8)
        partner_vec = self.sCFloat.load(partner_idx, vector_size=2, alignment=8)
        if cutlass.const_expr(self.cfg.has_deepseek_fp8_c_scale):
            return (
                cute.arch.fmax(own_vec[0], partner_vec[0]),
                cute.arch.fmax(own_vec[1], partner_vec[1]),
            )
        return (
            self._fmax_ftz(own_vec[0], partner_vec[0]),
            self._fmax_ftz(own_vec[1], partner_vec[1]),
        )

    @cute.jit
    def _read_absmax_scratch_pair_all_warps(self, warpgroup_idx, lane_id, scale_pair):
        idx0 = self._absmax_scratch_pair_idx(
            warpgroup_idx, Int32(0), lane_id, scale_pair
        )
        idx1 = self._absmax_scratch_pair_idx(
            warpgroup_idx, Int32(1), lane_id, scale_pair
        )
        idx2 = self._absmax_scratch_pair_idx(
            warpgroup_idx, Int32(2), lane_id, scale_pair
        )
        idx3 = self._absmax_scratch_pair_idx(
            warpgroup_idx, Int32(3), lane_id, scale_pair
        )
        v0 = self.sCFloat.load(idx0, vector_size=2, alignment=8)
        v1 = self.sCFloat.load(idx1, vector_size=2, alignment=8)
        v2 = self.sCFloat.load(idx2, vector_size=2, alignment=8)
        v3 = self.sCFloat.load(idx3, vector_size=2, alignment=8)
        return (
            cute.arch.fmax(cute.arch.fmax(v0[0], v1[0]), cute.arch.fmax(v2[0], v3[0])),
            cute.arch.fmax(cute.arch.fmax(v0[1], v1[1]), cute.arch.fmax(v2[1], v3[1])),
        )

    @cute.jit
    def _dsfp8_c_scale_pair(self, warpgroup_idx, lane_id, scale_pair):
        block_abs0, block_abs1 = self._read_absmax_scratch_pair_all_warps(
            warpgroup_idx, lane_id, scale_pair
        )
        safe_abs0, safe_abs1 = self._fmul2(
            cute.arch.fmax(block_abs0, Float32(1.0e-12)),
            cute.arch.fmax(block_abs1, Float32(1.0e-12)),
            Float32(1.0),
            Float32(1.0),
        )
        q0 = Float32(448.0) * cute.math.rcp(safe_abs0, approx=True, ftz=True)
        q1 = Float32(448.0) * cute.math.rcp(safe_abs1, approx=True, ftz=True)
        dq0 = block_abs0 * Float32(1.0 / 448.0)
        dq1 = block_abs1 * Float32(1.0 / 448.0)
        return q0, dq0, q1, dq1

    @cute.jit
    def _dsfp8_c_scale_pair_two_epilogues(
        self, warpgroup_idx, warp_in_epi4, lane_id, scale_pair, row_group
    ):
        """Block absmax of one column pair over this warp's scale row group.

        At block 128 the group is the whole 128-row tile (all eight warps).
        At block 32 it is this warp and its partner ``warp_in_epi4 ^ 1``.
        """
        if cutlass.const_expr(self.cfg.uses_deepseek_fp8_atomic_absmax):
            lane_col = (lane_id % Int32(4)) * Int32(2)
            scratch_idx = (
                row_group * Int32(self.cfg.epi_tile_n)
                + scale_pair * Int32(8)
                + lane_col
            )
            block_abs0, block_abs1 = self.sDsFp8Absmax.load(
                scratch_idx,
                vector_size=2,
                alignment=8,
            )
        elif cutlass.const_expr(self.cfg.dsfp8_row_groups_per_tile > 1):
            block_abs0, block_abs1 = self._read_absmax_scratch_pair(
                warpgroup_idx, warp_in_epi4, lane_id, scale_pair
            )
        else:
            group0_abs0, group0_abs1 = self._read_absmax_scratch_pair_all_warps(
                Int32(0), lane_id, scale_pair
            )
            group1_abs0, group1_abs1 = self._read_absmax_scratch_pair_all_warps(
                Int32(1), lane_id, scale_pair
            )
            block_abs0 = cute.arch.fmax(group0_abs0, group1_abs0)
            block_abs1 = cute.arch.fmax(group0_abs1, group1_abs1)
        safe_abs0, safe_abs1 = self._fmul2(
            cute.arch.fmax(block_abs0, Float32(1.0e-12)),
            cute.arch.fmax(block_abs1, Float32(1.0e-12)),
            Float32(1.0),
            Float32(1.0),
        )
        q0 = Float32(448.0) * cute.math.rcp(safe_abs0, approx=True, ftz=True)
        q1 = Float32(448.0) * cute.math.rcp(safe_abs1, approx=True, ftz=True)
        dq0 = block_abs0 * Float32(1.0 / 448.0)
        dq1 = block_abs1 * Float32(1.0 / 448.0)
        return q0, dq0, q1, dq1

    @cute.jit
    def _dsfp8_c_scale_idx(self, m_row, n_col, output_m):
        if cutlass.const_expr(self.cfg.is_swap_ab):
            return (
                m_row // Int32(self.cfg.deepseek_fp8_block_size)
            ) * self.dsfp8_c_scale_stride + n_col
        return (n_col // Int32(128)) * output_m + m_row

    @cute.jit
    def _store_dsfp8_c_scale(
        self,
        dq_scale,
        m_row,
        n_col,
        output_m,
        token_in_bounds,
        output_in_bounds,
        lane_id,
        warp_in_epi4,
        warpgroup_idx,
    ):
        lane_stores = (lane_id // Int32(4)) == Int32(0)
        if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
            # The warpgroups split the tile rows: the first warp of each scale
            # row group stores that group's scale.
            warps_per_group = (
                self.cfg.warp_layout.num_epilogue_warps
                // self.cfg.dsfp8_row_groups_per_tile
            )
            warp_in_epi = warpgroup_idx * Int32(4) + warp_in_epi4
            should_store = (
                (warp_in_epi % Int32(warps_per_group)) == Int32(0)
            ) & lane_stores
        else:
            # The warpgroups split the tile columns: warp 0 of each stores the
            # scale of its own columns.
            should_store = (warp_in_epi4 == Int32(0)) & lane_stores
        if should_store:
            if output_in_bounds:
                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                    if token_in_bounds:
                        self.gSfC.store(
                            dq_scale,
                            idx=self._dsfp8_c_scale_idx(m_row, n_col, output_m),
                            vector_size=1,
                        )
                else:
                    self.gSfC.store(
                        dq_scale,
                        idx=self._dsfp8_c_scale_idx(m_row, n_col, output_m),
                        vector_size=1,
                    )

    @cute.jit
    def _reduce_mxfp8_absmax_m32(
        self, val, warpgroup_idx, warp_in_epi4, lane_id, scale_slot
    ):
        """Reduce MX output amax across the pair of M16 epilogue warps in one M32 block.

        Legacy path: per-call barrier pair.  Prefer the batched
        ``_write_absmax_scratch`` / ``_read_absmax_scratch`` split for
        throughput-critical epilogues.
        """
        self._write_absmax_scratch(
            val, warpgroup_idx, warp_in_epi4, lane_id, scale_slot
        )
        prims.barrier_cta_sync(
            barrier_id=_EPILOGUE_BARRIER_ID,
            thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
        )
        block_absmax = self._read_absmax_scratch(
            warpgroup_idx, warp_in_epi4, lane_id, scale_slot
        )
        prims.barrier_cta_sync(
            barrier_id=_EPILOGUE_BARRIER_ID,
            thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
        )
        return block_absmax

    @cute.jit
    def _sf_c_index(self, m_row, n_col, output_m):
        if cutlass.const_expr(self.cfg.is_swap_ab):
            m_row = m_row + cute.assume(self.sf_c_m_offset, divby=32)
        block_m = Int32(self.cfg.output_sf_block_size_c)
        group_m = block_m * Int32(4)
        sf_group_axis = self._sf_group_axis(output_m)
        if cutlass.const_expr(self.cfg.output_sf_block_size_c in (16, 32)):
            # These coordinates are non-negative.  Spell out the power-of-two
            # arithmetic so the DSL does not lower Python's signed // and %
            # semantics into correction branches.
            block_shift = 4 if self.cfg.output_sf_block_size_c == 16 else 5
            group_shift = block_shift + 2
            m_block = m_row >> Int32(block_shift)
            m_groups = (sf_group_axis + group_m - Int32(1)) >> Int32(group_shift)
            if cutlass.const_expr(self.cfg.sf_layout_c == int(SfLayout.R8c4)):
                return (
                    (n_col >> Int32(3)) * (m_groups * Int32(32))
                    + (m_block >> Int32(2)) * Int32(32)
                    + (n_col & Int32(7)) * Int32(4)
                    + (m_block & Int32(3))
                )
            return (
                (n_col >> Int32(7)) * (m_groups * Int32(512))
                + (m_block >> Int32(2)) * Int32(512)
                + (n_col & Int32(31)) * Int32(16)
                + ((n_col >> Int32(5)) & Int32(3)) * Int32(4)
                + (m_block & Int32(3))
            )
        m_block = m_row // block_m
        m_groups = (sf_group_axis + group_m - Int32(1)) // group_m
        if cutlass.const_expr(self.cfg.sf_layout_c == int(SfLayout.R8c4)):
            return (
                (n_col // Int32(8)) * (m_groups * Int32(32))
                + (m_block // Int32(4)) * Int32(32)
                + (n_col % Int32(8)) * Int32(4)
                + (m_block % Int32(4))
            )
        return (
            (n_col // Int32(128)) * (m_groups * Int32(512))
            + (m_block // Int32(4)) * Int32(512)
            + (n_col % Int32(32)) * Int32(16)
            + ((n_col % Int32(128)) // Int32(32)) * Int32(4)
            + (m_block % Int32(4))
        )

    @cute.jit
    def _fp4_tma_smem_byte_offset(self, m_local_row, n_local_col):
        box_rows = Int32(self.cfg.swap_tma_store_rows)
        row_block = m_local_row // box_rows
        row_in_block = m_local_row % box_rows
        box_stride = Int32(self.cfg.epi_tile_n) * box_rows
        linear = row_block * box_stride + n_local_col * box_rows + row_in_block
        byte_offset = linear // Int32(2)
        swizzle_mask = self._c_tma_smem_swizzle_mask(byte_offset // Int32(128))
        return byte_offset ^ swizzle_mask

    @cute.jit
    def _mxfp8_tma_smem_byte_offset(self, m_local_row, n_local_col):
        box_rows = Int32(self.cfg.swap_tma_store_rows)
        row_block = m_local_row // box_rows
        row_in_block = m_local_row % box_rows
        box_stride = Int32(self.cfg.epi_tile_n) * box_rows
        linear = row_block * box_stride + n_local_col * box_rows + row_in_block
        swizzle_mask = self._c_tma_smem_swizzle_mask(linear // Int32(128))
        return linear ^ swizzle_mask

    @cute.jit
    def _pack_swap_ab_fp8_gated_tma_pair(self, val0_f32, val1_f32):
        # NVVM's f8x2 conversion maps the second source to the low byte.  Store
        # row0 at the lower shared-memory byte to match trtllm-gen packing.
        return _convert_f32x2_to_e4m3x2(val1_f32, val0_f32)

    @cute.jit
    def _fp8_nongated_tma_smem_byte_offset(self, m_local_row, n_local_col):
        box_rows = Int32(self.cfg.swap_tma_store_rows)
        row_block = m_local_row // box_rows
        row_in_block = m_local_row % box_rows
        box_stride = Int32(self.cfg.epi_tile_n) * box_rows
        linear = row_block * box_stride + n_local_col * box_rows + row_in_block
        swizzle_mask = self._c_tma_smem_swizzle_mask(linear // Int32(128))
        return linear ^ swizzle_mask

    @cute.jit
    def _stage_swap_ab_fp8_tma_vec4(
        self,
        val0_f32,
        val1_f32,
        val2_f32,
        val3_f32,
        m_local_row,
        n_local_col,
    ):
        smem_offset = self._fp8_nongated_tma_smem_byte_offset(m_local_row, n_local_col)
        packed01 = self._pack_swap_ab_fp8_gated_tma_pair(val0_f32, val1_f32)
        packed23 = self._pack_swap_ab_fp8_gated_tma_pair(val2_f32, val3_f32)
        self.sCInt16.store(
            (packed01, packed23),
            idx=smem_offset // Int32(2),
            vector_size=2,
            alignment=4,
        )

    @cute.jit
    def _fp8_tma_smem_i16_offset(
        self,
        m_local_row,
        n_local_col,
        warpgroup_idx,
        tma_smem_stage,
    ):
        smem_stage_stride = Int32(
            self.cfg.num_bytes_c_tma_store_per_group * self.cfg.num_epilogue_warpgroups
        )
        smem_offset = (
            self._mxfp8_tma_smem_byte_offset(m_local_row, n_local_col)
            + tma_smem_stage * smem_stage_stride
            + warpgroup_idx * Int32(self.cfg.num_bytes_c_tma_store_per_group)
        )
        return smem_offset // Int32(2)

    @cute.jit
    def _fp8_store_dependency_xor16(
        self,
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
    ):
        return prims.inline_ptx_hl(
            "{\n"
            ".reg .u32 dep;\n"
            ".reg .u32 tmp;\n"
            "and.b32 dep, {$r0}, 65535;\n"
            "and.b32 tmp, {$r1}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r2}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r3}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r4}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r5}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r6}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r7}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r8}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r9}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r10}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r11}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r12}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r13}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r14}, 65535;\n"
            "xor.b32 dep, dep, tmp;\n"
            "and.b32 tmp, {$r15}, 65535;\n"
            "xor.b32 {$w0}, dep, tmp;\n"
            "}",
            write_only_types=[Int32],
            read_only_args=[
                Int32(v0),
                Int32(v1),
                Int32(v2),
                Int32(v3),
                Int32(v4),
                Int32(v5),
                Int32(v6),
                Int32(v7),
                Int32(v8),
                Int32(v9),
                Int32(v10),
                Int32(v11),
                Int32(v12),
                Int32(v13),
                Int32(v14),
                Int32(v15),
            ],
        )

    @cute.jit
    def _opaque_i32(self, value):
        return prims.inline_ptx_hl(
            "mov.u32 {$w0}, {$r0};",
            write_only_types=[Int32],
            read_only_args=[Int32(value)],
        )

    @cute.jit
    def _stage_swap_ab_fp8_gated_tma_packed_pair(
        self,
        packed,
        m_local_row,
        n_local_col,
        warpgroup_idx,
        tma_smem_stage,
        store_dep,
        output_in_bounds,
    ):
        smem_offset_i16 = (
            self._fp8_tma_smem_i16_offset(
                m_local_row, n_local_col, warpgroup_idx, tma_smem_stage
            )
            + store_dep
        )
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            self.sCInt16.subview(smem_offset_i16).store(packed)
            return
        if output_in_bounds:
            self.sCInt16.subview(smem_offset_i16).store(packed)
        else:
            self.sCInt16.subview(smem_offset_i16).store(cutlass.Int16(0))

    @cute.jit
    def _stage_swap_ab_fp8_gated_tma_pair(
        self,
        val0_f32,
        val1_f32,
        m_local_row,
        n_local_col,
        warpgroup_idx,
        tma_smem_stage,
        output_in_bounds,
    ):
        packed = self._pack_swap_ab_fp8_gated_tma_pair(val0_f32, val1_f32)
        self._stage_swap_ab_fp8_gated_tma_packed_pair(
            packed,
            m_local_row,
            n_local_col,
            warpgroup_idx,
            tma_smem_stage,
            Int32(0),
            output_in_bounds,
        )

    @cute.jit
    def _tma_oob_c_coords(
        self,
        tile_coord_m,
        tile_coord_n,
        call_idx,
        token_limit,
        warpgroup_idx,
        n_slice_offset,
        m_slice_offset,
    ):
        output_tile_m = self.cfg.tile_m
        if cutlass.const_expr(self.cfg.has_gated_epilogue):
            output_tile_m //= 2
        m_base = tile_coord_m * Int32(output_tile_m) + m_slice_offset
        n_base = tile_coord_n * Int32(self.cfg.tile_n)
        n_subtile_offset = call_idx * Int32(self.cfg.epi_tile_n)
        if cutlass.const_expr(self.cfg.warp_layout.num_epilogue_warps > 4):
            n_subtile_offset = call_idx * Int32(
                self.cfg.epi_tile_n * self.cfg.num_epilogue_warpgroups
            ) + warpgroup_idx * Int32(self.cfg.epi_tile_n)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            large_n = Int32(0x40000000)
            tile_n_i32 = Int32(self.cfg.tile_n)
            limit_mod = token_limit % tile_n_i32
            dist = (tile_n_i32 - limit_mod) % tile_n_i32
            return (
                m_base,
                n_subtile_offset + n_slice_offset + dist,
                large_n,
                n_base - dist + large_n,
            )
        return (m_base, n_base + n_subtile_offset + n_slice_offset)

    @cute.jit
    def _commit_swap_ab_packed_tma(
        self,
        tile_coord_m,
        tile_coord_n,
        call_idx,
        token_limit,
        warp_idx,
        warpgroup_idx,
        tma_smem_stage,
    ):
        cute.arch.fence_view_async_shared()
        # Uses named barrier 7 for epilogue TMA-store staging.
        # Barrier 4 is used by compact SFB STTM copy; sharing it lets the
        # epilogue and CopySfB warp-groups corrupt each other's rendezvous.
        store_barrier_id = Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )
        smem_stage_stride = Int32(
            self.cfg.num_bytes_c_tma_store_per_group * self.cfg.num_epilogue_warpgroups
        )
        smem_base = self.sC.subview(
            tma_smem_stage * smem_stage_stride
            + warpgroup_idx * Int32(self.cfg.num_bytes_c_tma_store_per_group)
        )
        group_leader = Int32(
            self.cfg.warp_layout.epilogue_warp_idx
        ) + warpgroup_idx * Int32(WARPS_PER_WARPGROUP)
        should_store = (warp_idx == group_leader) & prims.elect_sync()
        n_subtile_offset = call_idx * Int32(self.cfg.epi_tile_n)
        if cutlass.const_expr(self.cfg.warp_layout.num_epilogue_warps > 4):
            n_subtile_offset = call_idx * Int32(
                self.cfg.epi_tile_n * self.cfg.num_epilogue_warpgroups
            ) + warpgroup_idx * Int32(self.cfg.epi_tile_n)
        for pass_idx in cutlass.range_constexpr(self.cfg.tma_store_passes):
            n_slice_offset = Int32(pass_idx * self.cfg.tma_store_cols)
            slice_should_store = should_store
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                slice_should_store = slice_should_store & (
                    n_subtile_offset + n_slice_offset < token_limit
                )
            if slice_should_store:
                for row_pass_idx in cutlass.range_constexpr(
                    self.cfg.swap_tma_store_row_passes
                ):
                    m_slice_offset = Int32(row_pass_idx * self.cfg.swap_tma_store_rows)
                    if cutlass.const_expr(self.cfg.uses_fp4_output_quant):
                        smem_slice_offset = self._fp4_tma_smem_byte_offset(
                            m_slice_offset, n_slice_offset
                        )
                    else:
                        smem_slice_offset = self._mxfp8_tma_smem_byte_offset(
                            m_slice_offset, n_slice_offset
                        )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_c_desc,
                        smem_base.subview(smem_slice_offset),
                        self._tma_oob_c_coords(
                            tile_coord_m,
                            tile_coord_n,
                            call_idx,
                            token_limit,
                            warpgroup_idx,
                            n_slice_offset,
                            m_slice_offset,
                        ),
                    )
        if should_store:
            # Commit on every call, even when the token limit skipped every TMA
            # above: multi-stage callers reuse a scratch stage after
            # wait_group(stages - 1), which assumes one bulk group per call.
            # A skipped commit would leave the previous same-stage store in
            # flight while this stage is overwritten.
            prims.cp_async_bulk_commit_group()
            # An aliased A/B producer cannot wait on this lane's TMA group.
            # Drain only at that ownership handoff; disjoint C scratch instead
            # remains asynchronous until its next epilogue-stage reuse.
            if cutlass.const_expr(self.cfg.aliases_c_scratch_with_ab):
                prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(
            barrier_id=store_barrier_id,
            thread_count=self.cfg.epilogue_threads_per_warpgroup,
        )

    @cute.jit
    def _nvfp4_output_scale_pair(
        self,
        local_absmax0,
        local_absmax1,
        scale_c,
        include_xor16: cutlass.Constexpr[bool],
    ):
        """Compute two E4M3 block scales and their matching E2M1 multipliers."""
        local_absmax0 = cute.math.max(local_absmax0, Float32(0.0))
        local_absmax1 = cute.math.max(local_absmax1, Float32(0.0))
        if cutlass.const_expr(self.cfg.uses_global_scales):
            local_absmax0, local_absmax1 = self._fmul2(
                local_absmax0,
                local_absmax1,
                scale_c,
                scale_c,
            )
        block_absmax0 = self._reduce_positive_max_lanes(local_absmax0, include_xor16)
        block_absmax1 = self._reduce_positive_max_lanes(local_absmax1, include_xor16)
        sf0, sf1 = self._fmul2(
            block_absmax0,
            block_absmax1,
            Float32(1.0 / 6.0),
            Float32(1.0 / 6.0),
        )
        sf_packed2, sf_rounded0, sf_rounded1 = _convert_f32x2_to_e4m3x2_and_back(
            sf0, sf1
        )
        # The E4M3 round trip yields either zero or a value at least 2^-9, so
        # no positive scale needs an epsilon floor.  rcp(0) produces infinity
        # without trapping, and the following selects prevent it from escaping.
        output_scale0 = cute.math.rcp(sf_rounded0, approx=True, ftz=True)
        output_scale1 = cute.math.rcp(sf_rounded1, approx=True, ftz=True)
        if sf_rounded0 == Float32(0.0):
            output_scale0 = Float32(0.0)
        if sf_rounded1 == Float32(0.0):
            output_scale1 = Float32(0.0)
        if cutlass.const_expr(self.cfg.uses_global_scales):
            output_scale0, output_scale1 = self._fmul2(
                output_scale0,
                output_scale1,
                scale_c,
                scale_c,
            )

        # The packed conversion helper places sf0 in the low byte.  Return
        # scalar bytes in column order for the SF-C stores.
        sf_bits = Int32(sf_packed2)
        sf_packed0 = cutlass.Int8(sf_bits & Int32(0xFF))
        sf_packed1 = cutlass.Int8((sf_bits >> Int32(8)) & Int32(0xFF))
        return output_scale0, sf_packed0, output_scale1, sf_packed1

    @cute.jit
    def _mx_output_scale_pair_from_absmax(self, block_absmax0, block_absmax1, scale_c):
        """Compute two UE8M0 scales and matching MX output multipliers."""
        max_pow2_rcp = Float32(1.0 / 256.0)
        if cutlass.const_expr(self.cfg.uses_mxfp4_output_quant):
            max_pow2_rcp = Float32(1.0 / 4.0)
        sf0, sf1 = self._fmul2(
            self._trunc_abs_float_to_pow2(block_absmax0),
            self._trunc_abs_float_to_pow2(block_absmax1),
            max_pow2_rcp,
            max_pow2_rcp,
        )
        sf_rcp0 = self._scale_rcp_exp_only(sf0)
        sf_rcp1 = self._scale_rcp_exp_only(sf1)
        if sf0 == Float32(0.0):
            sf_rcp0 = Float32(0.0)
        if sf1 == Float32(0.0):
            sf_rcp1 = Float32(0.0)

        output_scale0 = sf_rcp0
        output_scale1 = sf_rcp1
        if cutlass.const_expr(self.cfg.uses_global_scales):
            output_scale0, output_scale1 = self._fmul2(
                output_scale0,
                output_scale1,
                scale_c,
                scale_c,
            )
        sf_packed0 = sf0.to(cutlass.Float8E8M0FNU).bitcast(cutlass.Int8)
        sf_packed1 = sf1.to(cutlass.Float8E8M0FNU).bitcast(cutlass.Int8)
        return output_scale0, sf_packed0, output_scale1, sf_packed1

    @cute.jit
    def _mx_output_scale_pair(self, local_absmax0, local_absmax1, scale_c):
        """Reduce two local M32 maxima and compute their MX output scales."""
        local_absmax0 = cute.math.max(local_absmax0, Float32(0.0))
        local_absmax1 = cute.math.max(local_absmax1, Float32(0.0))
        if cutlass.const_expr(self.cfg.uses_global_scales):
            local_absmax0, local_absmax1 = self._fmul2(
                local_absmax0,
                local_absmax1,
                scale_c,
                scale_c,
            )
        block_absmax0 = self._reduce_positive_max_lanes(local_absmax0, True)
        block_absmax1 = self._reduce_positive_max_lanes(local_absmax1, True)
        return self._mx_output_scale_pair_from_absmax(
            block_absmax0, block_absmax1, scale_c
        )

    @cute.jit
    def _store_swap_ab_fp4_vec4(
        self,
        result0,
        result1,
        result2,
        result3,
        output_scale,
        m_row0,
        n_col,
        output_m,
        token_in_bounds,
        warpgroup_idx,
        tma_smem_stage,
        m_local_row0,
        n_local_col,
    ):
        """Scale and pack four consecutive non-gated rows as two NVFP4 bytes."""
        scaled0, scaled1 = self._fmul2(result0, result1, output_scale, output_scale)
        scaled2, scaled3 = self._fmul2(result2, result3, output_scale, output_scale)
        packed01 = _convert_f32x2_to_e2m1x2(scaled1, scaled0)
        packed23 = _convert_f32x2_to_e2m1x2(scaled3, scaled2)
        if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
            smem_offset = (
                self._fp4_tma_smem_byte_offset(m_local_row0, n_local_col)
                + tma_smem_stage
                * Int32(
                    self.cfg.num_bytes_c_tma_store_per_group
                    * self.cfg.num_epilogue_warpgroups
                )
                + warpgroup_idx * Int32(self.cfg.num_bytes_c_tma_store_per_group)
            )
            self.sC.store(
                (packed01, packed23),
                idx=smem_offset,
                vector_size=2,
                alignment=2,
            )
        elif n_col < self.problem_n:
            flat_idx0 = n_col * self._c_pitch(output_m) + m_row0
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                if token_in_bounds:
                    if m_row0 < output_m:
                        self.gCBytes.store(
                            (packed01, packed23),
                            idx=flat_idx0 >> Int32(1),
                            vector_size=2,
                            alignment=2,
                        )
            else:
                if m_row0 < output_m:
                    self.gCBytes.store(
                        (packed01, packed23),
                        idx=flat_idx0 >> Int32(1),
                        vector_size=2,
                        alignment=2,
                    )

    @cute.jit
    def _store_swap_ab_mx_vec4(
        self,
        result0,
        result1,
        result2,
        result3,
        output_scale,
        m_row0,
        n_col,
        output_m,
        token_in_bounds,
        warpgroup_idx,
        tma_smem_stage,
        m_local_row0,
        n_local_col,
    ):
        """Scale and pack four consecutive non-gated rows as MXE4M3 bytes."""
        scaled0, scaled1 = self._fmul2(result0, result1, output_scale, output_scale)
        scaled2, scaled3 = self._fmul2(result2, result3, output_scale, output_scale)
        packed01 = self._pack_swap_ab_fp8_gated_tma_pair(scaled0, scaled1)
        packed23 = self._pack_swap_ab_fp8_gated_tma_pair(scaled2, scaled3)
        if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
            smem_offset = (
                self._mxfp8_tma_smem_byte_offset(m_local_row0, n_local_col)
                + tma_smem_stage
                * Int32(
                    self.cfg.num_bytes_c_tma_store_per_group
                    * self.cfg.num_epilogue_warpgroups
                )
                + warpgroup_idx * Int32(self.cfg.num_bytes_c_tma_store_per_group)
            )
            self.sCInt16.store(
                (packed01, packed23),
                idx=smem_offset >> Int32(1),
                vector_size=2,
                alignment=4,
            )
        elif n_col < self.problem_n:
            flat_idx0 = n_col * self._c_pitch(output_m) + m_row0
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                if token_in_bounds:
                    if m_row0 < output_m:
                        self.gCInt16.store(
                            (packed01, packed23),
                            idx=flat_idx0 >> Int32(1),
                            vector_size=2,
                            alignment=4,
                        )
            else:
                if m_row0 < output_m:
                    self.gCInt16.store(
                        (packed01, packed23),
                        idx=flat_idx0 >> Int32(1),
                        vector_size=2,
                        alignment=4,
                    )

    @cute.jit
    def _store_swap_ab_fp4_pair2(
        self,
        result00,
        result01,
        result10,
        result11,
        scale_c,
        m_row0,
        n_col0,
        output_m,
        token_in_bounds0,
        token_in_bounds1,
        output_in_bounds0,
        output_in_bounds1,
        warpgroup_idx,
        m_local_row0,
        n_local_col0,
    ):
        """Quantize and stage two adjacent NVFP4 output-column pairs.

        The two columns have independent M16 scales, but computing them
        together exposes packed multiply and E4M3x2 conversion instructions.
        C remains byte-stored because adjacent columns are not adjacent in the
        swapAB TMA scratch layout.
        """
        local_absmax0 = self._fmax_ftz(
            cute.math.abs(result00),
            cute.math.abs(result01),
        )
        local_absmax1 = self._fmax_ftz(
            cute.math.abs(result10),
            cute.math.abs(result11),
        )
        (
            output_scale0,
            sf_packed0,
            output_scale1,
            sf_packed1,
        ) = self._nvfp4_output_scale_pair(
            local_absmax0,
            local_absmax1,
            scale_c,
            True,
        )
        scaled00, scaled01 = self._fmul2(
            result00,
            result01,
            output_scale0,
            output_scale0,
        )
        scaled10, scaled11 = self._fmul2(
            result10,
            result11,
            output_scale1,
            output_scale1,
        )
        # NVVM's e2m1x2 conversion operand order maps the second source to the
        # low nibble.  Store row0 in the low nibble to match trtllm-gen packing.
        packed0 = _convert_f32x2_to_e2m1x2(scaled01, scaled00)
        packed1 = _convert_f32x2_to_e2m1x2(scaled11, scaled10)

        if cutlass.const_expr(self.cfg.use_tma_store):
            # Wide FC1 tiles use two epilogue warpgroups, each with a distinct
            # TMA-store scratch segment.  Keep their staged FP4 bytes disjoint.
            smem_group_offset = warpgroup_idx * Int32(
                self.cfg.num_bytes_c_tma_store_per_group
            )
            smem_offset0 = (
                self._fp4_tma_smem_byte_offset(m_local_row0, n_local_col0)
                + smem_group_offset
            )
            smem_offset1 = (
                self._fp4_tma_smem_byte_offset(m_local_row0, n_local_col0 + Int32(1))
                + smem_group_offset
            )
            self.sC.subview(smem_offset0).store(packed0)
            self.sC.subview(smem_offset1).store(packed1)
        else:
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                if output_in_bounds0 & token_in_bounds0:
                    flat_idx0 = n_col0 * self._c_pitch(output_m) + m_row0
                    self.gCBytes.subview(flat_idx0 // Int32(2)).store(packed0)
                if output_in_bounds1 & token_in_bounds1:
                    flat_idx1 = (n_col0 + Int32(1)) * self._c_pitch(output_m) + m_row0
                    self.gCBytes.subview(flat_idx1 // Int32(2)).store(packed1)
            else:
                if output_in_bounds0:
                    flat_idx0 = n_col0 * self._c_pitch(output_m) + m_row0
                    self.gCBytes.subview(flat_idx0 // Int32(2)).store(packed0)
                if output_in_bounds1:
                    flat_idx1 = (n_col0 + Int32(1)) * self._c_pitch(output_m) + m_row0
                    self.gCBytes.subview(flat_idx1 // Int32(2)).store(packed1)

        return sf_packed0, sf_packed1

    @cute.jit
    def _select_int8(self, values, selector):
        """Select one of one, two, four, or eight packed bytes with ``selp``."""

        if cutlass.const_expr(len(values) == 1):
            return values[0]

        select_bit0 = (selector & Int32(1)) != Int32(0)
        pair0 = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[Int32(values[1]), Int32(values[0]), select_bit0],
        )
        if cutlass.const_expr(len(values) == 2):
            return pair0.to(cutlass.Int8)

        select_bit1 = (selector & Int32(2)) != Int32(0)
        pair1 = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[Int32(values[3]), Int32(values[2]), select_bit0],
        )
        quad0 = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[pair1, pair0, select_bit1],
        )
        if cutlass.const_expr(len(values) == 4):
            return quad0.to(cutlass.Int8)

        pair2 = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[Int32(values[5]), Int32(values[4]), select_bit0],
        )
        pair3 = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[Int32(values[7]), Int32(values[6]), select_bit0],
        )
        select_bit2 = (selector & Int32(4)) != Int32(0)
        quad1 = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[pair3, pair2, select_bit1],
        )
        selected = prims.inline_ptx_hl(
            "selp.b32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Int32],
            read_only_args=[quad1, quad0, select_bit2],
        )
        return selected.to(cutlass.Int8)

    @cute.jit
    def _store_swap_ab_quant_sf_epi32(
        self,
        sf_values,
        m_row,
        n_tile_base,
        n_subtile_offset,
        token_limit,
        output_m,
        lane_id,
    ):
        """Store an epi-N32 quantization SF-C fragment with one warp store.

        Each group of four lanes selects one of the eight scale values, matching
        TRT-LLM Gen's ``threadIdxInGroup`` mapping.  The previous path emitted
        eight separately predicated stores per epilogue subtile.
        """

        selector = lane_id >> Int32(2)
        sf_packed = self._select_int8(sf_values, selector)
        n_local_col = (
            (selector >> Int32(1)) * Int32(8)
            + (lane_id & Int32(3)) * Int32(2)
            + (selector & Int32(1))
        )
        n_col = n_tile_base + n_local_col
        sf_idx = self._sf_c_index(m_row, n_col, output_m)
        output_in_bounds = (n_col < self.problem_n) & (m_row < output_m)
        if output_in_bounds:
            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                token_in_bounds = (n_subtile_offset + n_local_col) < token_limit
                if token_in_bounds:
                    self.gSfCBytes.subview(sf_idx).store(sf_packed)
            else:
                self.gSfCBytes.subview(sf_idx).store(sf_packed)

    @cute.jit
    def _store_swap_ab_quant_sf_epi64(
        self,
        sf_values,
        m_row,
        n_tile_base,
        n_subtile_offset,
        token_limit,
        output_m,
        lane_id,
    ):
        """Store an epi-N64 SF-C fragment as two coalesced warp stores."""

        selector = lane_id >> Int32(2)
        lane_col = (
            (selector >> Int32(1)) * Int32(8)
            + (lane_id & Int32(3)) * Int32(2)
            + (selector & Int32(1))
        )
        for half in cutlass.range_constexpr(2):
            value_base = half * 8
            sf_packed = self._select_int8(
                sf_values[value_base : value_base + 8], selector
            )
            n_local_col = lane_col + Int32(half * 32)
            n_col = n_tile_base + n_local_col
            sf_idx = self._sf_c_index(m_row, n_col, output_m)
            output_in_bounds = (n_col < self.problem_n) & (m_row < output_m)
            if output_in_bounds:
                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                    token_in_bounds = (n_subtile_offset + n_local_col) < token_limit
                    if token_in_bounds:
                        self.gSfCBytes.subview(sf_idx).store(sf_packed)
                else:
                    self.gSfCBytes.subview(sf_idx).store(sf_packed)

    @cute.jit
    def _store_swap_ab_nongated_nvfp4_sf(
        self,
        sf_values,
        m_tile_base,
        n_tile_base,
        n_subtile_offset,
        token_limit,
        output_m,
        lane_id,
        warp_in_epi4,
    ):
        """Store both M16 scale rows owned by one non-gated output warp."""

        lane_group = lane_id >> Int32(2)
        selector = lane_group & Int32(3)
        m_row = (
            m_tile_base
            + warp_in_epi4 * Int32(32)
            + (lane_group >> Int32(2)) * Int32(16)
        )
        num_sf_values = self.cfg.epi_tile_n // 4
        num_store_groups = (num_sf_values + 3) // 4
        for store_group in cutlass.range_constexpr(num_store_groups):
            sf_base = store_group * 4
            sf_values_in_group = min(4, num_sf_values - sf_base)
            sf_packed = self._select_int8(
                sf_values[sf_base : sf_base + sf_values_in_group], selector
            )
            sf_slot = Int32(sf_base) + selector
            sf_col_offset = (sf_slot // Int32(2)) * Int32(8) + sf_slot % Int32(2)
            n_local_col = (lane_id & Int32(3)) * Int32(2) + sf_col_offset
            n_col = n_tile_base + n_local_col
            if (
                (selector < Int32(sf_values_in_group))
                & (m_row < output_m)
                & (n_col < self.problem_n)
            ):
                sf_idx = self._sf_c_index(m_row, n_col, output_m)
                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                    token_in_bounds = (n_subtile_offset + n_local_col) < token_limit
                    if token_in_bounds:
                        self.gSfCBytes.subview(sf_idx).store(sf_packed)
                else:
                    self.gSfCBytes.subview(sf_idx).store(sf_packed)

    @cute.jit
    def _mx_output_write_absmax_phase(
        self,
        result0,
        result1,
        scale_c,
        lane_id,
        warpgroup_idx,
        warp_in_epi4,
        scale_slot,
    ):
        """Phase 1 of a batched MX output store: write local absmax scratch.

        Call for every (group, col_sub) BEFORE the batched bar.sync 9.
        """
        local_absmax = self._fmax_ftz(
            cute.math.abs(result0),
            cute.math.abs(result1),
        )
        if cutlass.const_expr(self.cfg.uses_global_scales):
            local_absmax = local_absmax * scale_c
        self._write_absmax_scratch(
            local_absmax, warpgroup_idx, warp_in_epi4, lane_id, scale_slot
        )

    @cute.jit
    def _mx_output_write_absmax_phase_pair(
        self,
        result00,
        result01,
        result10,
        result11,
        scale_c,
        lane_id,
        warpgroup_idx,
        warp_in_epi4,
        scale_pair,
    ):
        local_absmax0 = self._fmax_ftz(
            cute.math.abs(result00),
            cute.math.abs(result01),
        )
        local_absmax1 = self._fmax_ftz(
            cute.math.abs(result10),
            cute.math.abs(result11),
        )
        if cutlass.const_expr(self.cfg.uses_global_scales):
            local_absmax0, local_absmax1 = self._fmul2(
                local_absmax0,
                local_absmax1,
                scale_c,
                scale_c,
            )
        self._write_absmax_scratch_pair(
            local_absmax0,
            local_absmax1,
            warpgroup_idx,
            warp_in_epi4,
            lane_id,
            scale_pair,
        )

    @cute.jit
    def _mx_output_compute_scale_phase(
        self,
        scale_c,
        lane_id,
        warpgroup_idx,
        warp_in_epi4,
        scale_slot,
    ):
        """Phase 2a of a batched MX output store: compute UE8M0 scale.

        Call for every (group, col_sub) AFTER the batched bar.sync 9.
        """
        block_absmax = self._read_absmax_scratch(
            warpgroup_idx, warp_in_epi4, lane_id, scale_slot
        )
        max_pow2_rcp = Float32(1.0 / 256.0)
        if cutlass.const_expr(self.cfg.uses_mxfp4_output_quant):
            max_pow2_rcp = Float32(1.0 / 4.0)
        sf = self._trunc_abs_float_to_pow2(block_absmax) * max_pow2_rcp
        sf_rcp = self._scale_rcp_exp_only(sf)
        if sf == Float32(0.0):
            sf_rcp = Float32(0.0)

        output_scale = sf_rcp
        if cutlass.const_expr(self.cfg.uses_global_scales):
            output_scale = output_scale * scale_c
        sf_packed = sf.to(cutlass.Float8E8M0FNU).bitcast(cutlass.Int8)
        return output_scale, sf_packed

    @cute.jit
    def _mx_output_compute_scale_phase_pair(
        self,
        scale_c,
        lane_id,
        warpgroup_idx,
        warp_in_epi4,
        scale_pair,
    ):
        block_absmax0, block_absmax1 = self._read_absmax_scratch_pair(
            warpgroup_idx,
            warp_in_epi4,
            lane_id,
            scale_pair,
        )
        return self._mx_output_scale_pair_from_absmax(
            block_absmax0, block_absmax1, scale_c
        )

    @cute.jit
    def _mx_output_store_phase(
        self,
        result0,
        result1,
        output_scale,
        sf_packed,
        m_row0,
        n_col,
        output_m,
        token_in_bounds,
        output_in_bounds,
        lane_id,
        warpgroup_idx,
        warp_in_epi4,
        m_local_row0,
        n_local_col,
        store_sf_c,
    ):
        """Phase 2b of a batched MX output store: scale and store C/SF-C."""
        scaled0, scaled1 = self._fmul2(
            result0,
            result1,
            output_scale,
            output_scale,
        )

        if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
            if cutlass.const_expr(self.cfg.uses_mxfp4_output_quant):
                packed = _convert_f32x2_to_e2m1x2(scaled1, scaled0)
                smem_offset = self._fp4_tma_smem_byte_offset(m_local_row0, n_local_col)
                self.sC.subview(smem_offset).store(packed)
            else:
                packed = self._pack_swap_ab_fp8_gated_tma_pair(scaled0, scaled1)
                smem_offset0 = self._mxfp8_tma_smem_byte_offset(
                    m_local_row0, n_local_col
                ) + warpgroup_idx * Int32(self.cfg.num_bytes_c_tma_store_per_group)
                self.sCInt16.subview(smem_offset0 >> Int32(1)).store(packed)
        elif output_in_bounds:
            flat_idx0 = n_col * self._c_pitch(output_m) + m_row0
            if cutlass.const_expr(self.cfg.uses_mxfp4_output_quant):
                packed = _convert_f32x2_to_e2m1x2(scaled1, scaled0)
                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                    if token_in_bounds:
                        self.gCBytes.subview(flat_idx0 >> Int32(1)).store(packed)
                else:
                    self.gCBytes.subview(flat_idx0 >> Int32(1)).store(packed)
            else:
                fp8_0 = scaled0.to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
                fp8_1 = scaled1.to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                    if token_in_bounds:
                        self.gCBytes.subview(flat_idx0).store(fp8_0)
                        self.gCBytes.subview(flat_idx0 + Int32(1)).store(fp8_1)
                else:
                    self.gCBytes.subview(flat_idx0).store(fp8_0)
                    self.gCBytes.subview(flat_idx0 + Int32(1)).store(fp8_1)

        if ((lane_id >> Int32(2)) == Int32(0)) & (
            (warp_in_epi4 & Int32(1)) == Int32(0)
        ):
            if cutlass.const_expr(store_sf_c):
                sf_idx = self._sf_c_index(m_row0, n_col, output_m)
                if (n_col < self.problem_n) & (m_row0 < output_m):
                    if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                        if token_in_bounds:
                            self.gSfCBytes.subview(sf_idx).store(sf_packed)
                    else:
                        self.gSfCBytes.subview(sf_idx).store(sf_packed)
        return sf_packed

    @cute.jit
    def _store_swap_ab_nongated_quant(
        self,
        t2r_rmem,
        t2r_slice1,
        tile_coord_m,
        tile_coord_n,
        call_idx,
        warp_idx,
        warpgroup_idx,
        warp_in_epi4,
        lane_id,
        m_tile_base,
        n_tile_base,
        n_subtile_offset,
        base_tmem_col,
        base_row_idx,
        expert_idx,
        output_m,
        token_limit,
        scale_c,
        scale_gate,
        is_eltwise_act: cutlass.Constexpr[bool],
    ):
        """Quantize a non-gated swapAB fragment to NVFP4 or MXE4M3."""
        tma_stage_count = self.cfg.num_swap_ab_nongated_quant_tma_stages
        tma_smem_stage = call_idx % Int32(tma_stage_count)
        if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
            prims.cp_async_bulk_wait_group(tma_stage_count - 1, read=True)
            prims.barrier_cta_sync(
                barrier_id=Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx,
                thread_count=self.cfg.epilogue_threads_per_warpgroup,
            )

        sf_values = []
        for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
            col_off = Int32(group * 8)
            result0_vals = []
            result1_vals = []
            result2_vals = []
            result3_vals = []
            local_abs_vals = []
            for col_sub in cutlass.range_constexpr(2):
                tmem_col = base_tmem_col + col_off + Int32(col_sub)
                reg0 = group * 4 + col_sub
                reg1 = group * 4 + 2 + col_sub
                result0 = self._maybe_apply_per_token_sf_a(t2r_rmem[reg0], base_row_idx)
                result1 = self._maybe_apply_per_token_sf_a(
                    t2r_rmem[reg1], base_row_idx + Int32(1)
                )
                result2 = self._maybe_apply_per_token_sf_a(
                    t2r_slice1[reg0], base_row_idx + Int32(2)
                )
                result3 = self._maybe_apply_per_token_sf_a(
                    t2r_slice1[reg1], base_row_idx + Int32(3)
                )
                if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                    local_token_col = n_subtile_offset + tmem_col
                    token_sf = self._load_per_token_sf_b(local_token_col)
                    result0, result1 = self._fmul2(result0, result1, token_sf, token_sf)
                    result2, result3 = self._fmul2(result2, result3, token_sf, token_sf)
                result0 = self._maybe_add_bias_m(
                    result0,
                    expert_idx,
                    m_tile_base + base_row_idx,
                    base_row_idx,
                )
                result1 = self._maybe_add_bias_m(
                    result1,
                    expert_idx,
                    m_tile_base + base_row_idx + Int32(1),
                    base_row_idx + Int32(1),
                )
                result2 = self._maybe_add_bias_m(
                    result2,
                    expert_idx,
                    m_tile_base + base_row_idx + Int32(2),
                    base_row_idx + Int32(2),
                )
                result3 = self._maybe_add_bias_m(
                    result3,
                    expert_idx,
                    m_tile_base + base_row_idx + Int32(3),
                    base_row_idx + Int32(3),
                )
                if cutlass.const_expr(is_eltwise_act):
                    result0, result1 = self._apply_eltwise_act_pair(
                        result0, result1, scale_gate
                    )
                    result2, result3 = self._apply_eltwise_act_pair(
                        result2, result3, scale_gate
                    )
                local_abs01 = self._fmax_ftz(
                    cute.math.abs(result0), cute.math.abs(result1)
                )
                local_abs23 = self._fmax_ftz(
                    cute.math.abs(result2), cute.math.abs(result3)
                )
                local_abs_vals.append(self._fmax_ftz(local_abs01, local_abs23))
                result0_vals.append(result0)
                result1_vals.append(result1)
                result2_vals.append(result2)
                result3_vals.append(result3)

            if cutlass.const_expr(self.cfg.dtype_c == int(DType.E2M1)):
                (
                    output_scale0,
                    sf_packed0,
                    output_scale1,
                    sf_packed1,
                ) = self._nvfp4_output_scale_pair(
                    local_abs_vals[0],
                    local_abs_vals[1],
                    scale_c,
                    False,
                )
            else:
                (
                    output_scale0,
                    sf_packed0,
                    output_scale1,
                    sf_packed1,
                ) = self._mx_output_scale_pair(
                    local_abs_vals[0], local_abs_vals[1], scale_c
                )
            sf_values.append(sf_packed0)
            sf_values.append(sf_packed1)

            for col_sub in cutlass.range_constexpr(2):
                tmem_col = base_tmem_col + col_off + Int32(col_sub)
                n_col = n_tile_base + tmem_col
                token_in_bounds = (n_subtile_offset + tmem_col) < token_limit
                output_scale = output_scale0
                sf_packed = sf_packed0
                if col_sub == 1:
                    output_scale = output_scale1
                    sf_packed = sf_packed1
                m_row0 = m_tile_base + base_row_idx
                if cutlass.const_expr(self.cfg.dtype_c == int(DType.E2M1)):
                    self._store_swap_ab_fp4_vec4(
                        result0_vals[col_sub],
                        result1_vals[col_sub],
                        result2_vals[col_sub],
                        result3_vals[col_sub],
                        output_scale,
                        m_row0,
                        n_col,
                        output_m,
                        token_in_bounds,
                        warpgroup_idx,
                        tma_smem_stage,
                        base_row_idx,
                        tmem_col,
                    )
                else:
                    self._store_swap_ab_mx_vec4(
                        result0_vals[col_sub],
                        result1_vals[col_sub],
                        result2_vals[col_sub],
                        result3_vals[col_sub],
                        output_scale,
                        m_row0,
                        n_col,
                        output_m,
                        token_in_bounds,
                        warpgroup_idx,
                        tma_smem_stage,
                        base_row_idx,
                        tmem_col,
                    )
                if cutlass.const_expr(
                    self.cfg.dtype_c != int(DType.E2M1)
                    and self.cfg.epi_tile_n not in (32, 64)
                ):
                    lane_group = lane_id >> Int32(2)
                    sf_m_row = m_tile_base + warp_in_epi4 * Int32(32)
                    if (
                        (lane_group == Int32(0))
                        & (sf_m_row < output_m)
                        & (n_col < self.problem_n)
                    ):
                        sf_idx = self._sf_c_index(sf_m_row, n_col, output_m)
                        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                            if token_in_bounds:
                                self.gSfCBytes.subview(sf_idx).store(sf_packed)
                        else:
                            self.gSfCBytes.subview(sf_idx).store(sf_packed)

        if cutlass.const_expr(self.cfg.dtype_c == int(DType.E2M1)):
            self._store_swap_ab_nongated_nvfp4_sf(
                sf_values,
                m_tile_base,
                n_tile_base,
                n_subtile_offset,
                token_limit,
                output_m,
                lane_id,
                warp_in_epi4,
            )
        elif cutlass.const_expr(self.cfg.epi_tile_n in (32, 64)):
            sf_m_row = m_tile_base + warp_in_epi4 * Int32(32)
            if cutlass.const_expr(self.cfg.epi_tile_n == 32):
                self._store_swap_ab_quant_sf_epi32(
                    sf_values,
                    sf_m_row,
                    n_tile_base,
                    n_subtile_offset,
                    token_limit,
                    output_m,
                    lane_id,
                )
            else:
                self._store_swap_ab_quant_sf_epi64(
                    sf_values,
                    sf_m_row,
                    n_tile_base,
                    n_subtile_offset,
                    token_limit,
                    output_m,
                    lane_id,
                )

        if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
            self._commit_swap_ab_packed_tma(
                tile_coord_m,
                tile_coord_n,
                call_idx,
                token_limit,
                warp_idx,
                warpgroup_idx,
                tma_smem_stage,
            )

    @cute.jit
    def _store_swap_ab_16x256b(
        self, t2r_rmem, t2r_rmem_1, tile_coord_m, tile_coord_n, call_idx, warp_idx
    ):
        """SwapAB: 16x256b T2R, M-major output, element-by-element STG.

        16x256b register layout — 2 slices of ``tile_n / 8 * 4`` FP32 regs:
          slice0: TMEM rows 0..15
          slice1: TMEM rows 16..31

        Generated kernels first write the 16x256b fragment to a swizzled SMEM
        tile, then TMA-store that tile to C. This direct GMEM path preserves the
        same register grouping and uses logical row-major ArrayView indices; the
        C tensor layout maps those indices to column-major physical memory.
        """
        t2r_slice1 = t2r_rmem_1

        tidx, _, _ = cute.arch.thread_idx()
        warp_in_epi = warp_idx
        warpgroup_idx = nonnegative_div(warp_in_epi, 4)
        warpgroup_idx = cute.arch.make_warp_uniform(warpgroup_idx)
        warp_in_epi4 = nonnegative_mod(warp_in_epi, 4)
        warp_in_epi4 = cute.arch.make_warp_uniform(warp_in_epi4)
        lane_id = tidx & Int32(31)

        m_tile_base = tile_coord_m * Int32(self.cfg.tile_m)
        warpgroup_count = self.cfg.num_epilogue_warpgroups
        n_warpgroup_count = warpgroup_count
        n_warpgroup_idx = warpgroup_idx
        if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
            m_tile_base += warpgroup_idx * Int32(self.cfg.tile_m // 2)
            n_warpgroup_count = 1
            n_warpgroup_idx = Int32(0)
        n_subtile_offset = call_idx * Int32(
            self.cfg.epi_tile_n * n_warpgroup_count
        ) + n_warpgroup_idx * Int32(self.cfg.epi_tile_n)
        n_tile_base = tile_coord_n * Int32(self.cfg.tile_n) + n_subtile_offset
        expert_idx = self._expert_idx_for_tile(tile_coord_m, tile_coord_n)
        scale_c = self.tile_scale_c
        scale_gate = self.tile_scale_gate
        token_limit = Int32(self.cfg.tile_n)
        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
            token_limit = self._token_limit_for_tile(tile_coord_m, tile_coord_n)

        is_gated_act = cutlass.const_expr(
            (self.cfg.act_kind == int(ActKind.SWIGLU))
            or (self.cfg.act_kind == int(ActKind.GEGLU))
            or (self.cfg.act_kind == int(ActKind.SITU))
        )
        is_eltwise_act = cutlass.const_expr(self.cfg.has_eltwise_act)
        output_m = self.problem_m
        if cutlass.const_expr(is_gated_act):
            output_m = self.problem_m >> Int32(1)

        base_tmem_col = (lane_id & Int32(3)) * Int32(2)
        warp_row_stride = 16 if cutlass.const_expr(is_gated_act) else 32
        row_group_size = 2 if cutlass.const_expr(is_gated_act) else 4
        if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
            warp_row_stride = 16
            row_group_size = 2
        base_row_idx = warp_in_epi4 * Int32(warp_row_stride) + (
            lane_id >> Int32(2)
        ) * Int32(row_group_size)

        if cutlass.const_expr(
            self.cfg.has_deepseek_fp8_two_epilogue
            and (
                self.cfg.dtype_c == int(DType.BF16)
                or self.cfg.dtype_c == int(DType.FP16)
            )
            and self.cfg.use_tma_store
            and not is_gated_act
        ):
            # Each split-M warpgroup owns a disjoint scratch tile.  The lane
            # that issued its previous TMA waits, then all four writer warps
            # rendezvous before any of them starts overwriting that tile.
            prims.cp_async_bulk_wait_group(0, read=True)
            prims.barrier_cta_sync(
                barrier_id=Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx,
                thread_count=self.cfg.epilogue_threads_per_warpgroup,
            )
            for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                col_off = Int32(group * 8)
                for col_sub in cutlass.range_constexpr(2):
                    tmem_col = base_tmem_col + col_off + Int32(col_sub)
                    reg0 = group * 4 + col_sub
                    reg1 = group * 4 + 2 + col_sub
                    val0_f32 = self._maybe_add_bias_m(
                        t2r_rmem[reg0],
                        expert_idx,
                        m_tile_base + base_row_idx,
                        base_row_idx,
                    )
                    val1_f32 = self._maybe_add_bias_m(
                        t2r_rmem[reg1],
                        expert_idx,
                        m_tile_base + base_row_idx + Int32(1),
                        base_row_idx + Int32(1),
                    )
                    if cutlass.const_expr(is_eltwise_act):
                        val0_f32, val1_f32 = self._apply_eltwise_act_pair(
                            val0_f32, val1_f32, scale_gate
                        )
                    val0_f32 = self._maybe_apply_scale_c(val0_f32, scale_c)
                    val1_f32 = self._maybe_apply_scale_c(val1_f32, scale_c)
                    self._stage_swap_ab_bf16_dsfp8_tma_pair(
                        self._to_output_value(val0_f32),
                        self._to_output_value(val1_f32),
                        base_row_idx,
                        tmem_col,
                        warpgroup_idx,
                    )
            self._commit_swap_ab_bf16_dsfp8_tma(
                tile_coord_m,
                tile_coord_n,
                n_subtile_offset,
                token_limit,
                warp_idx,
                warpgroup_idx,
            )
            return

        # Process both slices
        if cutlass.const_expr(is_gated_act):
            if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
                # Protect the scratch stage before its first writer below.
                # Plain FP8 FC1 tileN=128/256 writes multiple epilogue
                # subtiles per C tile.  Use the existing two C scratch
                # stages so one TMA store can remain in flight while the
                # next subtile is staged into the other buffer.
                num_subtiles = self.cfg.tile_n // (
                    self.cfg.epi_tile_n * self.cfg.num_epilogue_warpgroups
                )
                can_overlap_stores = (
                    self.cfg.uses_fp8_output
                    and self.cfg.num_stages_c_smem > 1
                    and num_subtiles > 1
                )
                pending_groups = 1 if can_overlap_stores else 0
                ring_remainder = num_subtiles % self.cfg.num_stages_c_smem
                # TMA wait counts are immediates. Only an uneven ring boundary
                # needs a runtime choice before SMEM stage zero is overwritten.
                if cutlass.const_expr(can_overlap_stores and ring_remainder != 0):
                    if call_idx == Int32(0):
                        prims.cp_async_bulk_wait_group(0, read=True)
                    else:
                        prims.cp_async_bulk_wait_group(pending_groups, read=True)
                else:
                    prims.cp_async_bulk_wait_group(pending_groups, read=True)
                if cutlass.const_expr(
                    self.cfg.has_output_block_scaling or self.cfg.uses_fp8_output
                ):
                    # Packed output scratch is partitioned per warpgroup.
                    prims.barrier_cta_sync(
                        barrier_id=Int32(_TMA_STORE_BARRIER_BASE) + warpgroup_idx,
                        thread_count=self.cfg.epilogue_threads_per_warpgroup,
                    )
                else:
                    # The gated 16-bit path uses one tile shared by every
                    # epilogue writer warp.
                    prims.barrier_cta_sync(
                        barrier_id=_TMA_STORE_BARRIER_BASE,
                        thread_count=self.cfg.warp_layout.num_epilogue_warps
                        * WARP_SIZE,
                    )
            if cutlass.const_expr(self.cfg.has_output_block_scaling):
                # Quantized output paths collect values or scale bytes in
                # register lists before their batched reduction/conversion or
                # lane-redistributed SF-C store below.
                mx_result0_vals = []
                mx_result1_vals = []
                mx_output_scale_vals = []
                mx_sf_c_vals = []
                nvfp4_sf_c_vals = []
                for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                    col_off = Int32(group * 8)
                    nvfp4_result0_vals = []
                    nvfp4_result1_vals = []
                    for col_sub in cutlass.range_constexpr(2):
                        gate_idx = group * 4 + col_sub
                        up_idx = group * 4 + 2 + col_sub
                        tmem_col_even = base_tmem_col + col_off + Int32(col_sub)
                        m_local_row0 = base_row_idx
                        m_local_row1 = base_row_idx + Int32(1)

                        gate0 = t2r_rmem[gate_idx]
                        up0 = t2r_rmem[up_idx]
                        gate1 = t2r_slice1[gate_idx]
                        up1 = t2r_slice1[up_idx]
                        gate0 = self._maybe_apply_per_token_sf_a(
                            gate0, m_local_row0 * Int32(2)
                        )
                        up0 = self._maybe_apply_per_token_sf_a(
                            up0, m_local_row0 * Int32(2) + Int32(1)
                        )
                        gate1 = self._maybe_apply_per_token_sf_a(
                            gate1, m_local_row1 * Int32(2)
                        )
                        up1 = self._maybe_apply_per_token_sf_a(
                            up1, m_local_row1 * Int32(2) + Int32(1)
                        )
                        if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                            local_token_col = n_subtile_offset + tmem_col_even
                            token_sf = self._load_per_token_sf_b(local_token_col)
                            gate0, gate1 = self._fmul2(gate0, gate1, token_sf, token_sf)
                            up0, up1 = self._fmul2(up0, up1, token_sf, token_sf)
                        gate0 = self._maybe_add_bias_m(
                            gate0,
                            expert_idx,
                            m_tile_base + m_local_row0 * Int32(2),
                            m_local_row0 * Int32(2),
                        )
                        up0 = self._maybe_add_bias_m(
                            up0,
                            expert_idx,
                            m_tile_base + m_local_row0 * Int32(2) + Int32(1),
                            m_local_row0 * Int32(2) + Int32(1),
                        )
                        gate1 = self._maybe_add_bias_m(
                            gate1,
                            expert_idx,
                            m_tile_base + m_local_row1 * Int32(2),
                            m_local_row1 * Int32(2),
                        )
                        up1 = self._maybe_add_bias_m(
                            up1,
                            expert_idx,
                            m_tile_base + m_local_row1 * Int32(2) + Int32(1),
                            m_local_row1 * Int32(2) + Int32(1),
                        )
                        result0, result1 = self._apply_gated_activation_pair(
                            gate0,
                            gate1,
                            up0,
                            up1,
                            scale_gate,
                        )

                        m_row0 = (m_tile_base >> Int32(1)) + m_local_row0
                        m_row1 = m_row0 + Int32(1)
                        n_col = n_tile_base + tmem_col_even
                        token_in_bounds = (
                            n_subtile_offset + tmem_col_even
                        ) < token_limit
                        output_in_bounds = (m_row1 < output_m) & (
                            n_col < self.problem_n
                        )
                        if cutlass.const_expr(self.cfg.uses_mx_output_quant):
                            mx_result0_vals.append(result0)
                            mx_result1_vals.append(result1)
                        else:
                            nvfp4_result0_vals.append(result0)
                            nvfp4_result1_vals.append(result1)
                    if cutlass.const_expr(not self.cfg.uses_mx_output_quant):
                        n_local_col0 = base_tmem_col + col_off
                        n_col0 = n_tile_base + n_local_col0
                        token_in_bounds0 = (
                            n_subtile_offset + n_local_col0
                        ) < token_limit
                        token_in_bounds1 = (
                            n_subtile_offset + n_local_col0 + Int32(1)
                        ) < token_limit
                        m_row0 = m_tile_base // Int32(2) + base_row_idx
                        m_row1 = m_row0 + Int32(1)
                        output_in_bounds0 = (m_row1 < output_m) & (
                            n_col0 < self.problem_n
                        )
                        output_in_bounds1 = (m_row1 < output_m) & (
                            n_col0 + Int32(1) < self.problem_n
                        )
                        sf_packed0, sf_packed1 = self._store_swap_ab_fp4_pair2(
                            nvfp4_result0_vals[0],
                            nvfp4_result1_vals[0],
                            nvfp4_result0_vals[1],
                            nvfp4_result1_vals[1],
                            scale_c,
                            m_row0,
                            n_col0,
                            output_m,
                            token_in_bounds0,
                            token_in_bounds1,
                            output_in_bounds0,
                            output_in_bounds1,
                            warpgroup_idx,
                            base_row_idx,
                            n_local_col0,
                        )
                        nvfp4_sf_c_vals.append(sf_packed0)
                        nvfp4_sf_c_vals.append(sf_packed1)
                if cutlass.const_expr(self.cfg.uses_mx_output_quant):
                    # Phase 1: after all activation outputs are available,
                    # reduce/store adjacent MX scale slots as paired f32
                    # values. It gives the compiler a contiguous 64-bit SMEM exchange.
                    for group_write in cutlass.range_constexpr(
                        self.cfg.epi_tile_n // 8
                    ):
                        pair_idx_write = group_write * 2
                        self._mx_output_write_absmax_phase_pair(
                            mx_result0_vals[pair_idx_write],
                            mx_result1_vals[pair_idx_write],
                            mx_result0_vals[pair_idx_write + 1],
                            mx_result1_vals[pair_idx_write + 1],
                            scale_c,
                            lane_id,
                            warpgroup_idx,
                            warp_in_epi4,
                            Int32(group_write),
                        )
                    # Batched barrier: all absmax writes done above, now sync
                    # once instead of per-(group, col_sub). This replaces 2×N
                    # bar.sync calls with just 2.
                    prims.barrier_cta_sync(
                        barrier_id=_EPILOGUE_BARRIER_ID,
                        thread_count=self.cfg.warp_layout.num_epilogue_warps
                        * WARP_SIZE,
                    )
                    # Phase 2a: compute all SF/output scales first so the
                    # dependent multiply/convert/store block is packed.
                    for group2 in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                        (
                            output_scale0,
                            sf_packed0,
                            output_scale1,
                            sf_packed1,
                        ) = self._mx_output_compute_scale_phase_pair(
                            scale_c,
                            lane_id,
                            warpgroup_idx,
                            warp_in_epi4,
                            Int32(group2),
                        )
                        mx_output_scale_vals.append(output_scale0)
                        mx_sf_c_vals.append(sf_packed0)
                        mx_output_scale_vals.append(output_scale1)
                        mx_sf_c_vals.append(sf_packed1)
                    # Phase 2b: scale and store the C fragment.
                    for group2 in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                        col_off2 = Int32(group2 * 8)
                        for col_sub2 in cutlass.range_constexpr(2):
                            pair_idx2 = group2 * 2 + col_sub2
                            tmem_col_even2 = base_tmem_col + col_off2 + Int32(col_sub2)
                            m_local_row0_2 = base_row_idx
                            m_row0_2 = (m_tile_base >> Int32(1)) + m_local_row0_2
                            m_row1_2 = m_row0_2 + Int32(1)
                            n_col2 = n_tile_base + tmem_col_even2
                            token_ib2 = (
                                n_subtile_offset + tmem_col_even2
                            ) < token_limit
                            out_ib2 = (m_row1_2 < output_m) & (n_col2 < self.problem_n)
                            self._mx_output_store_phase(
                                mx_result0_vals[pair_idx2],
                                mx_result1_vals[pair_idx2],
                                mx_output_scale_vals[pair_idx2],
                                mx_sf_c_vals[pair_idx2],
                                m_row0_2,
                                n_col2,
                                output_m,
                                token_ib2,
                                out_ib2,
                                lane_id,
                                warpgroup_idx,
                                warp_in_epi4,
                                m_local_row0_2,
                                tmem_col_even2,
                                self.cfg.epi_tile_n != 32,
                            )
                    if cutlass.const_expr(self.cfg.epi_tile_n == 32):
                        # MX output scale reduction pairs adjacent M16 warps.
                        # Only the even warp owns the resulting M32 scale row.
                        # Select one of its eight scale registers per four-lane
                        # group and emit one coalesced store, matching Gen,
                        # instead of eight separately predicated stores.
                        if (warp_in_epi4 & Int32(1)) == Int32(0):
                            self._store_swap_ab_quant_sf_epi32(
                                mx_sf_c_vals,
                                (m_tile_base >> Int32(1)) + warp_in_epi4 * Int32(16),
                                n_tile_base,
                                n_subtile_offset,
                                token_limit,
                                output_m,
                                lane_id,
                            )
                    elif cutlass.const_expr(self.cfg.epi_tile_n == 64):
                        if (warp_in_epi4 & Int32(1)) == Int32(0):
                            self._store_swap_ab_quant_sf_epi64(
                                mx_sf_c_vals,
                                (m_tile_base >> Int32(1)) + warp_in_epi4 * Int32(16),
                                n_tile_base,
                                n_subtile_offset,
                                token_limit,
                                output_m,
                                lane_id,
                            )
                    prims.barrier_cta_sync(
                        barrier_id=_EPILOGUE_BARRIER_ID,
                        thread_count=self.cfg.warp_layout.num_epilogue_warps
                        * WARP_SIZE,
                    )
                else:
                    # Each lane owns epi_tile_n/4 M16 scales at column offsets
                    # [0, 1, 8, 9, ...] relative to its lane%4 base.  Move
                    # groups of eight register values across the lane/4 groups
                    # so all 32 lanes issue one static SF-C store per group.
                    # For the target epi_tile_n=64 kernel this is the same two-
                    # store mapping as TRTLLM-gen and replaces 16 separately
                    # predicated address/store sites.  The final predicate
                    # also supports smaller 8- and 32-column epilogues.
                    thread_group = lane_id // Int32(4)
                    num_sf_vals = self.cfg.epi_tile_n // 4
                    num_sf_store_groups = (num_sf_vals + 7) // 8
                    for store_half in cutlass.range_constexpr(num_sf_store_groups):
                        sf_base = store_half * 8
                        sf_packed = nvfp4_sf_c_vals[sf_base]
                        sf_vals_in_group = min(8, num_sf_vals - sf_base)
                        for sf_select in cutlass.range_constexpr(1, sf_vals_in_group):
                            if thread_group == Int32(sf_select):
                                sf_packed = nvfp4_sf_c_vals[sf_base + sf_select]

                        sf_slot = Int32(sf_base) + thread_group
                        sf_col_offset = (sf_slot // Int32(2)) * Int32(
                            8
                        ) + sf_slot % Int32(2)
                        sf_n_col = n_tile_base + base_tmem_col + sf_col_offset
                        sf_m_row = m_tile_base // Int32(2) + base_row_idx
                        sf_token_in_bounds = (
                            n_subtile_offset + base_tmem_col + sf_col_offset
                        ) < token_limit
                        sf_idx = self._sf_c_index(sf_m_row, sf_n_col, output_m)
                        if (
                            (thread_group < Int32(sf_vals_in_group))
                            & (sf_n_col < self.problem_n)
                            & (sf_m_row < output_m)
                        ):
                            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                                if sf_token_in_bounds:
                                    self.gSfCBytes.subview(sf_idx).store(sf_packed)
                            else:
                                self.gSfCBytes.subview(sf_idx).store(sf_packed)
                if cutlass.const_expr(self._use_swap_ab_quant_tma_store()):
                    self._commit_swap_ab_packed_tma(
                        tile_coord_m,
                        tile_coord_n,
                        call_idx,
                        token_limit,
                        warp_idx,
                        warpgroup_idx,
                        Int32(0),
                    )
            else:
                if cutlass.const_expr(
                    self.cfg.dtype_c == int(DType.BF16)
                    or self.cfg.dtype_c == int(DType.FP16)
                ):
                    for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                        col_off = Int32(group * 8)
                        for col_sub in cutlass.range_constexpr(2):
                            # Gated epilogue: adjacent TMEM columns are
                            # (gate, up) pairs. Process both 16x256b rows
                            # together so sigmoid uses packed f32x2 math.
                            gate_idx = group * 4 + col_sub
                            up_idx = group * 4 + 2 + col_sub
                            tmem_col_even = base_tmem_col + col_off + Int32(col_sub)
                            m_local_row0 = base_row_idx
                            m_local_row1 = base_row_idx + Int32(1)
                            pre_gate_row0 = m_tile_base + m_local_row0 * Int32(2)
                            pre_gate_row1 = m_tile_base + m_local_row1 * Int32(2)
                            pre_up_row0 = pre_gate_row0 + Int32(1)
                            pre_up_row1 = pre_gate_row1 + Int32(1)

                            gate0 = t2r_rmem[gate_idx]
                            up0 = t2r_rmem[up_idx]
                            gate1 = t2r_slice1[gate_idx]
                            up1 = t2r_slice1[up_idx]
                            gate0 = self._maybe_apply_per_token_sf_a(
                                gate0, m_local_row0 * Int32(2)
                            )
                            up0 = self._maybe_apply_per_token_sf_a(
                                up0, m_local_row0 * Int32(2) + Int32(1)
                            )
                            gate1 = self._maybe_apply_per_token_sf_a(
                                gate1, m_local_row1 * Int32(2)
                            )
                            up1 = self._maybe_apply_per_token_sf_a(
                                up1, m_local_row1 * Int32(2) + Int32(1)
                            )
                            if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                                local_token_col = n_subtile_offset + tmem_col_even
                                token_sf = self._load_per_token_sf_b(local_token_col)
                                gate0, gate1 = self._fmul2(
                                    gate0, gate1, token_sf, token_sf
                                )
                                up0, up1 = self._fmul2(up0, up1, token_sf, token_sf)
                            gate0 = self._maybe_add_bias_m(
                                gate0,
                                expert_idx,
                                pre_gate_row0,
                                m_local_row0 * Int32(2),
                            )
                            up0 = self._maybe_add_bias_m(
                                up0,
                                expert_idx,
                                pre_up_row0,
                                m_local_row0 * Int32(2) + Int32(1),
                            )
                            gate1 = self._maybe_add_bias_m(
                                gate1,
                                expert_idx,
                                pre_gate_row1,
                                m_local_row1 * Int32(2),
                            )
                            up1 = self._maybe_add_bias_m(
                                up1,
                                expert_idx,
                                pre_up_row1,
                                m_local_row1 * Int32(2) + Int32(1),
                            )
                            result0, result1 = self._apply_gated_activation_pair(
                                gate0,
                                gate1,
                                up0,
                                up1,
                                scale_gate,
                            )
                            result0, result1 = self._maybe_apply_scale_c_pair(
                                result0, result1, scale_c
                            )
                            result0_out = self._to_output_value(result0)
                            result1_out = self._to_output_value(result1)

                            m_row0 = m_tile_base // Int32(2) + m_local_row0
                            m_row1 = m_row0 + Int32(1)
                            n_col = n_tile_base + tmem_col_even
                            token_in_bounds = (
                                n_subtile_offset + tmem_col_even
                            ) < token_limit
                            output_in_bounds = (m_row1 < output_m) & (
                                n_col < self.problem_n
                            )
                            if cutlass.const_expr(self.cfg.use_tma_store):
                                self._stage_swap_ab_bf16_gated_tma_pair(
                                    result0_out,
                                    result1_out,
                                    m_local_row0,
                                    tmem_col_even,
                                    output_in_bounds,
                                )
                            else:
                                flat_idx0 = n_col * self._c_pitch(output_m) + m_row0
                                flat_idx1 = flat_idx0 + Int32(1)
                                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                                    if token_in_bounds & output_in_bounds:
                                        self.gC.store(
                                            result0_out,
                                            idx=flat_idx0,
                                            vector_size=1,
                                            alignment=2,
                                        )
                                        self.gC.store(
                                            result1_out,
                                            idx=flat_idx1,
                                            vector_size=1,
                                            alignment=2,
                                        )
                                else:
                                    if output_in_bounds:
                                        self.gC.store(
                                            result0_out,
                                            idx=flat_idx0,
                                            vector_size=1,
                                            alignment=2,
                                        )
                                        self.gC.store(
                                            result1_out,
                                            idx=flat_idx1,
                                            vector_size=1,
                                            alignment=2,
                                        )
                    if cutlass.const_expr(self.cfg.use_tma_store):
                        self._commit_swap_ab_bf16_gated_tma(
                            tile_coord_m,
                            tile_coord_n,
                            n_subtile_offset,
                            token_limit,
                            warp_idx,
                        )
                elif cutlass.const_expr(
                    self.cfg.uses_fp8_output and self.cfg.use_tma_store
                ):
                    tma_smem_stage = Int32(0)
                    if cutlass.const_expr(
                        self.cfg.num_stages_c_smem > 1
                        and self.cfg.tile_n
                        > self.cfg.epi_tile_n * self.cfg.num_epilogue_warpgroups
                    ):
                        tma_smem_stage = call_idx % Int32(self.cfg.num_stages_c_smem)
                    pair_count = (self.cfg.epi_tile_n // 8) * 2
                    token_sfs = []
                    if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                        for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                            col_off = Int32(group * 8)
                            local_token_col = n_subtile_offset + base_tmem_col + col_off
                            token_sf_pair = self.sPerTokenSfB.load(
                                idx=local_token_col,
                                vector_size=2,
                                alignment=8,
                            )
                            token_sfs.append(token_sf_pair[0])
                            token_sfs.append(token_sf_pair[1])

                    gate0_vals = []
                    gate1_vals = []
                    up0_vals = []
                    up1_vals = []
                    tmem_col_evens = []
                    for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                        col_off = Int32(group * 8)
                        for col_sub in cutlass.range_constexpr(2):
                            gate_idx = group * 4 + col_sub
                            up_idx = group * 4 + 2 + col_sub
                            tmem_col_even = base_tmem_col + col_off + Int32(col_sub)
                            gate0_vals.append(t2r_rmem[gate_idx])
                            gate1_vals.append(t2r_slice1[gate_idx])
                            up0_vals.append(t2r_rmem[up_idx])
                            up1_vals.append(t2r_slice1[up_idx])
                            tmem_col_evens.append(tmem_col_even)

                    for pair_idx in cutlass.range_constexpr(pair_count):
                        m_local_row0 = base_row_idx
                        m_local_row1 = base_row_idx + Int32(1)
                        gate0_vals[pair_idx] = self._maybe_apply_per_token_sf_a(
                            gate0_vals[pair_idx], m_local_row0 * Int32(2)
                        )
                        up0_vals[pair_idx] = self._maybe_apply_per_token_sf_a(
                            up0_vals[pair_idx], m_local_row0 * Int32(2) + Int32(1)
                        )
                        gate1_vals[pair_idx] = self._maybe_apply_per_token_sf_a(
                            gate1_vals[pair_idx], m_local_row1 * Int32(2)
                        )
                        up1_vals[pair_idx] = self._maybe_apply_per_token_sf_a(
                            up1_vals[pair_idx], m_local_row1 * Int32(2) + Int32(1)
                        )

                    if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                        for pair_idx in cutlass.range_constexpr(pair_count):
                            token_sf = token_sfs[pair_idx]
                            gate0_vals[pair_idx], gate1_vals[pair_idx] = self._fmul2(
                                gate0_vals[pair_idx],
                                gate1_vals[pair_idx],
                                token_sf,
                                token_sf,
                            )
                            up0_vals[pair_idx], up1_vals[pair_idx] = self._fmul2(
                                up0_vals[pair_idx],
                                up1_vals[pair_idx],
                                token_sf,
                                token_sf,
                            )

                    for pair_idx in cutlass.range_constexpr(pair_count):
                        m_local_row0 = base_row_idx
                        m_local_row1 = base_row_idx + Int32(1)
                        pre_gate_row0 = m_tile_base + m_local_row0 * Int32(2)
                        pre_gate_row1 = m_tile_base + m_local_row1 * Int32(2)
                        pre_up_row0 = pre_gate_row0 + Int32(1)
                        pre_up_row1 = pre_gate_row1 + Int32(1)
                        gate0_vals[pair_idx] = self._maybe_add_bias_m(
                            gate0_vals[pair_idx],
                            expert_idx,
                            pre_gate_row0,
                            m_local_row0 * Int32(2),
                        )
                        gate1_vals[pair_idx] = self._maybe_add_bias_m(
                            gate1_vals[pair_idx],
                            expert_idx,
                            pre_gate_row1,
                            m_local_row1 * Int32(2),
                        )
                        up0_vals[pair_idx] = self._maybe_add_bias_m(
                            up0_vals[pair_idx],
                            expert_idx,
                            pre_up_row0,
                            m_local_row0 * Int32(2) + Int32(1),
                        )
                        up1_vals[pair_idx] = self._maybe_add_bias_m(
                            up1_vals[pair_idx],
                            expert_idx,
                            pre_up_row1,
                            m_local_row1 * Int32(2) + Int32(1),
                        )

                    result0_vals = []
                    result1_vals = []
                    for pair_idx in cutlass.range_constexpr(pair_count):
                        result0, result1 = self._apply_gated_activation_pair(
                            gate0_vals[pair_idx],
                            gate1_vals[pair_idx],
                            up0_vals[pair_idx],
                            up1_vals[pair_idx],
                            scale_gate,
                        )
                        result0_vals.append(result0)
                        result1_vals.append(result1)

                    fp8_packed = []
                    for pair_idx in cutlass.range_constexpr(pair_count):
                        result0, result1 = self._maybe_apply_scale_c_pair(
                            result0_vals[pair_idx],
                            result1_vals[pair_idx],
                            scale_c,
                        )
                        fp8_packed.append(
                            self._pack_swap_ab_fp8_gated_tma_pair(result0, result1)
                        )

                    store_dep = Int32(0)
                    if cutlass.const_expr(
                        self.cfg.use_tma_oob_opt and len(fp8_packed) == 16
                    ):
                        store_dep_src = self._fp8_store_dependency_xor16(
                            fp8_packed[0],
                            fp8_packed[1],
                            fp8_packed[2],
                            fp8_packed[3],
                            fp8_packed[4],
                            fp8_packed[5],
                            fp8_packed[6],
                            fp8_packed[7],
                            fp8_packed[8],
                            fp8_packed[9],
                            fp8_packed[10],
                            fp8_packed[11],
                            fp8_packed[12],
                            fp8_packed[13],
                            fp8_packed[14],
                            fp8_packed[15],
                        )
                        store_dep = store_dep_src - self._opaque_i32(store_dep_src)

                    for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                        col_off = Int32(group * 8)
                        for col_sub in cutlass.range_constexpr(2):
                            result_idx = group * 2 + col_sub
                            m_local_row0 = base_row_idx
                            m_row0 = m_tile_base // Int32(2) + m_local_row0
                            m_row1 = m_row0 + Int32(1)
                            n_col = n_tile_base + tmem_col_evens[result_idx]
                            output_in_bounds = (m_row1 < output_m) & (
                                n_col < self.problem_n
                            )
                            self._stage_swap_ab_fp8_gated_tma_packed_pair(
                                fp8_packed[result_idx],
                                m_local_row0,
                                tmem_col_evens[result_idx],
                                warpgroup_idx,
                                tma_smem_stage,
                                store_dep,
                                output_in_bounds,
                            )
                    self._commit_swap_ab_packed_tma(
                        tile_coord_m,
                        tile_coord_n,
                        call_idx,
                        token_limit,
                        warp_idx,
                        warpgroup_idx,
                        tma_smem_stage,
                    )
                else:
                    for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                        col_off = Int32(group * 8)
                        for row_sub in cutlass.range_constexpr(2):
                            for col_sub in cutlass.range_constexpr(2):
                                # Gated epilogue: adjacent TMEM columns are
                                # (gate, up) pairs.  Combine and store once.
                                gate_idx = group * 4 + col_sub
                                up_idx = group * 4 + 2 + col_sub
                                gate = t2r_rmem[gate_idx]
                                up = t2r_rmem[up_idx]
                                if row_sub == 1:
                                    gate = t2r_slice1[gate_idx]
                                    up = t2r_slice1[up_idx]
                                tmem_col_even = base_tmem_col + col_off + Int32(col_sub)
                                m_local_row = base_row_idx + Int32(row_sub)
                                pre_gate_row = m_tile_base + m_local_row * Int32(2)
                                pre_up_row = pre_gate_row + Int32(1)
                                gate = self._maybe_apply_per_token_sf_a(
                                    gate, m_local_row * Int32(2)
                                )
                                up = self._maybe_apply_per_token_sf_a(
                                    up, m_local_row * Int32(2) + Int32(1)
                                )
                                local_token_col = n_subtile_offset + tmem_col_even
                                gate = self._maybe_apply_per_token_sf_b(
                                    gate, local_token_col
                                )
                                up = self._maybe_apply_per_token_sf_b(
                                    up, local_token_col
                                )
                                gate = self._maybe_add_bias_m(
                                    gate,
                                    expert_idx,
                                    pre_gate_row,
                                    m_local_row * Int32(2),
                                )
                                up = self._maybe_add_bias_m(
                                    up,
                                    expert_idx,
                                    pre_up_row,
                                    m_local_row * Int32(2) + Int32(1),
                                )
                                result = self._apply_gated_activation(
                                    gate, up, scale_gate
                                )
                                result = self._maybe_apply_scale_c(result, scale_c)
                                result_out = self._to_output_value(result)

                                m_row = m_tile_base // Int32(2) + m_local_row
                                n_col = n_tile_base + tmem_col_even
                                flat_idx = n_col * self._c_pitch(output_m) + m_row
                                token_in_bounds = (
                                    n_subtile_offset + tmem_col_even
                                ) < token_limit
                                output_in_bounds = (m_row < output_m) & (
                                    n_col < self.problem_n
                                )
                                if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                                    if token_in_bounds & output_in_bounds:
                                        self.gC.store(
                                            result_out,
                                            idx=flat_idx,
                                            vector_size=1,
                                            alignment=(
                                                1 if self.cfg.uses_fp8_output else 2
                                            ),
                                        )
                                else:
                                    if output_in_bounds:
                                        self.gC.store(
                                            result_out,
                                            idx=flat_idx,
                                            vector_size=1,
                                            alignment=(
                                                1 if self.cfg.uses_fp8_output else 2
                                            ),
                                        )
        elif cutlass.const_expr(self.cfg.has_deepseek_fp8_c_scale):
            if cutlass.const_expr(self.cfg.has_deepseek_fp8_two_epilogue):
                # Scale row group of this warp's rows (warp-uniform). Each
                # epilogue warp owns tile_m / num_epilogue_warps consecutive
                # rows, so a warp never straddles two groups.
                dsfp8_row_group = Int32(0)
                if cutlass.const_expr(self.cfg.dsfp8_row_groups_per_tile > 1):
                    rows_per_warp = (
                        self.cfg.tile_m // self.cfg.warp_layout.num_epilogue_warps
                    )
                    dsfp8_row_group = (
                        (warpgroup_idx * Int32(4) + warp_in_epi4) * Int32(rows_per_warp)
                    ) // Int32(self.cfg.deepseek_fp8_block_size)
                val0_vals = []
                val1_vals = []
                for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                    col_off = Int32(group * 8)
                    local_abs0 = Float32(0.0)
                    local_abs1 = Float32(0.0)
                    for col_sub in cutlass.range_constexpr(2):
                        tmem_col = base_tmem_col + col_off + Int32(col_sub)
                        reg0 = group * 4 + col_sub
                        reg1 = group * 4 + 2 + col_sub
                        val0_f32 = self._maybe_add_bias_m(
                            t2r_rmem[reg0],
                            expert_idx,
                            m_tile_base + base_row_idx,
                            base_row_idx,
                        )
                        val1_f32 = self._maybe_add_bias_m(
                            t2r_rmem[reg1],
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(1),
                            base_row_idx + Int32(1),
                        )
                        val0_f32 = self._maybe_apply_scale_c(val0_f32, scale_c)
                        val1_f32 = self._maybe_apply_scale_c(val1_f32, scale_c)
                        val0_vals.append(val0_f32)
                        val1_vals.append(val1_f32)
                        local_abs = cute.arch.fmax(
                            cute.arch.fmax(val0_f32, -val0_f32),
                            cute.arch.fmax(val1_f32, -val1_f32),
                        )
                        if col_sub == 0:
                            local_abs0 = local_abs
                        else:
                            local_abs1 = local_abs
                    if cutlass.const_expr(self.cfg.uses_deepseek_fp8_atomic_absmax):
                        self._atomic_dsfp8_absmax_scratch_pair(
                            local_abs0,
                            local_abs1,
                            lane_id,
                            Int32(group),
                            dsfp8_row_group,
                        )
                    else:
                        self._write_absmax_scratch_pair(
                            local_abs0,
                            local_abs1,
                            warpgroup_idx,
                            warp_in_epi4,
                            lane_id,
                            Int32(group),
                        )

                prims.barrier_cta_sync(
                    barrier_id=_EPILOGUE_BARRIER_ID,
                    thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
                )
                for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                    q0, dq0, q1, dq1 = self._dsfp8_c_scale_pair_two_epilogues(
                        warpgroup_idx,
                        warp_in_epi4,
                        lane_id,
                        Int32(group),
                        dsfp8_row_group,
                    )
                    col_off = Int32(group * 8)
                    for col_sub in cutlass.range_constexpr(2):
                        pair_idx = group * 2 + col_sub
                        tmem_col = base_tmem_col + col_off + Int32(col_sub)
                        n_col = n_tile_base + tmem_col
                        token_in_bounds = (n_subtile_offset + tmem_col) < token_limit
                        q_scale = q0
                        dq_scale = dq0
                        if col_sub == 1:
                            q_scale = q1
                            dq_scale = dq1
                        m_row0 = m_tile_base + base_row_idx
                        scale_in_bounds = (m_row0 < output_m) & (n_col < self.problem_n)
                        self._store_dsfp8_c_scale(
                            dq_scale,
                            m_row0,
                            n_col,
                            output_m,
                            token_in_bounds,
                            scale_in_bounds,
                            lane_id,
                            warp_in_epi4,
                            warpgroup_idx,
                        )
                        m_row1 = m_row0 + Int32(1)
                        flat_idx0 = n_col * self._c_pitch(output_m) + m_row0
                        output_pair_in_bounds = (m_row1 < output_m) & (
                            n_col < self.problem_n
                        )
                        packed = self._pack_swap_ab_fp8_gated_tma_pair(
                            val0_vals[pair_idx] * q_scale,
                            val1_vals[pair_idx] * q_scale,
                        )
                        if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                            if token_in_bounds & output_pair_in_bounds:
                                self.gCInt16.subview(flat_idx0 // Int32(2)).store(
                                    packed
                                )
                        else:
                            if output_pair_in_bounds:
                                self.gCInt16.subview(flat_idx0 // Int32(2)).store(
                                    packed
                                )
                prims.barrier_cta_sync(
                    barrier_id=_EPILOGUE_BARRIER_ID,
                    thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
                )
            else:
                val0_vals = []
                val1_vals = []
                val2_vals = []
                val3_vals = []
                for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                    col_off = Int32(group * 8)
                    local_abs0 = Float32(0.0)
                    local_abs1 = Float32(0.0)
                    for col_sub in cutlass.range_constexpr(2):
                        tmem_col = base_tmem_col + col_off + Int32(col_sub)
                        reg0 = group * 4 + col_sub
                        reg1 = group * 4 + 2 + col_sub

                        val0_f32 = t2r_rmem[reg0]
                        val1_f32 = t2r_rmem[reg1]
                        val2_f32 = t2r_slice1[reg0]
                        val3_f32 = t2r_slice1[reg1]
                        val0_f32 = self._maybe_apply_per_token_sf_a(
                            val0_f32, base_row_idx
                        )
                        val1_f32 = self._maybe_apply_per_token_sf_a(
                            val1_f32, base_row_idx + Int32(1)
                        )
                        val2_f32 = self._maybe_apply_per_token_sf_a(
                            val2_f32, base_row_idx + Int32(2)
                        )
                        val3_f32 = self._maybe_apply_per_token_sf_a(
                            val3_f32, base_row_idx + Int32(3)
                        )
                        val0_f32 = self._maybe_apply_per_token_sf_b(val0_f32, tmem_col)
                        val1_f32 = self._maybe_apply_per_token_sf_b(val1_f32, tmem_col)
                        val2_f32 = self._maybe_apply_per_token_sf_b(val2_f32, tmem_col)
                        val3_f32 = self._maybe_apply_per_token_sf_b(val3_f32, tmem_col)
                        val0_f32 = self._maybe_add_bias_m(
                            val0_f32,
                            expert_idx,
                            m_tile_base + base_row_idx,
                            base_row_idx,
                        )
                        val1_f32 = self._maybe_add_bias_m(
                            val1_f32,
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(1),
                            base_row_idx + Int32(1),
                        )
                        val2_f32 = self._maybe_add_bias_m(
                            val2_f32,
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(2),
                            base_row_idx + Int32(2),
                        )
                        val3_f32 = self._maybe_add_bias_m(
                            val3_f32,
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(3),
                            base_row_idx + Int32(3),
                        )
                        if cutlass.const_expr(is_eltwise_act):
                            val0_f32, val1_f32 = self._apply_eltwise_act_pair(
                                val0_f32, val1_f32, scale_gate
                            )
                            val2_f32, val3_f32 = self._apply_eltwise_act_pair(
                                val2_f32, val3_f32, scale_gate
                            )
                        val0_f32 = self._maybe_apply_scale_c(val0_f32, scale_c)
                        val1_f32 = self._maybe_apply_scale_c(val1_f32, scale_c)
                        val2_f32 = self._maybe_apply_scale_c(val2_f32, scale_c)
                        val3_f32 = self._maybe_apply_scale_c(val3_f32, scale_c)
                        val0_vals.append(val0_f32)
                        val1_vals.append(val1_f32)
                        val2_vals.append(val2_f32)
                        val3_vals.append(val3_f32)

                        local_abs = cute.arch.fmax(
                            cute.arch.fmax(val0_f32, -val0_f32),
                            cute.arch.fmax(val1_f32, -val1_f32),
                        )
                        local_abs = cute.arch.fmax(
                            local_abs,
                            cute.arch.fmax(
                                cute.arch.fmax(val2_f32, -val2_f32),
                                cute.arch.fmax(val3_f32, -val3_f32),
                            ),
                        )
                        if col_sub == 0:
                            local_abs0 = local_abs
                        else:
                            local_abs1 = local_abs
                    self._write_absmax_scratch_pair(
                        local_abs0,
                        local_abs1,
                        warpgroup_idx,
                        warp_in_epi4,
                        lane_id,
                        Int32(group),
                    )

                prims.barrier_cta_sync(
                    barrier_id=_EPILOGUE_BARRIER_ID,
                    thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
                )
                for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                    q0, dq0, q1, dq1 = self._dsfp8_c_scale_pair(
                        warpgroup_idx,
                        lane_id,
                        Int32(group),
                    )
                    col_off = Int32(group * 8)
                    for col_sub in cutlass.range_constexpr(2):
                        pair_idx = group * 2 + col_sub
                        tmem_col = base_tmem_col + col_off + Int32(col_sub)
                        n_col = n_tile_base + tmem_col
                        token_in_bounds = (n_subtile_offset + tmem_col) < token_limit
                        q_scale = q0
                        dq_scale = dq0
                        if col_sub == 1:
                            q_scale = q1
                            dq_scale = dq1

                        m_row0 = m_tile_base + base_row_idx
                        scale_in_bounds = (m_row0 < output_m) & (n_col < self.problem_n)
                        self._store_dsfp8_c_scale(
                            dq_scale,
                            m_row0,
                            n_col,
                            output_m,
                            token_in_bounds,
                            scale_in_bounds,
                            lane_id,
                            warp_in_epi4,
                            warpgroup_idx,
                        )

                        for row_sub in cutlass.range_constexpr(4):
                            m_local_row = base_row_idx + Int32(row_sub)
                            m_row = m_tile_base + m_local_row
                            val_f32 = val0_vals[pair_idx]
                            if row_sub == 1:
                                val_f32 = val1_vals[pair_idx]
                            elif row_sub == 2:
                                val_f32 = val2_vals[pair_idx]
                            elif row_sub == 3:
                                val_f32 = val3_vals[pair_idx]
                            val_out = self._to_output_value(val_f32 * q_scale)
                            flat_idx = n_col * self._c_pitch(output_m) + m_row
                            output_in_bounds = (m_row < output_m) & (
                                n_col < self.problem_n
                            )
                            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                                if token_in_bounds & output_in_bounds:
                                    self.gC.store(
                                        val_out,
                                        idx=flat_idx,
                                        vector_size=1,
                                        alignment=1,
                                    )
                            else:
                                if output_in_bounds:
                                    self.gC.store(
                                        val_out,
                                        idx=flat_idx,
                                        vector_size=1,
                                        alignment=1,
                                    )
                prims.barrier_cta_sync(
                    barrier_id=_EPILOGUE_BARRIER_ID,
                    thread_count=self.cfg.warp_layout.num_epilogue_warps * WARP_SIZE,
                )
        else:
            if cutlass.const_expr(self.cfg.has_output_block_scaling):
                self._store_swap_ab_nongated_quant(
                    t2r_rmem,
                    t2r_slice1,
                    tile_coord_m,
                    tile_coord_n,
                    call_idx,
                    warp_idx,
                    warpgroup_idx,
                    warp_in_epi4,
                    lane_id,
                    m_tile_base,
                    n_tile_base,
                    n_subtile_offset,
                    base_tmem_col,
                    base_row_idx,
                    expert_idx,
                    output_m,
                    token_limit,
                    scale_c,
                    scale_gate,
                    is_eltwise_act,
                )
                return
            if cutlass.const_expr(
                self.cfg.use_tma_store
                and not self.cfg.uses_fp8_output
                and self.cfg.swap_tma_store_offsets_are_col_linear
            ):
                group_stride_elems = Int32(
                    self.cfg.num_bytes_c_tma_store_per_group // 2
                )
                smem_group_offset = warpgroup_idx * group_stride_elems
                tma_smem_base_even = (
                    smem_group_offset
                    + self._bf16_tma_smem_element_offset(base_row_idx, base_tmem_col)
                )
                tma_smem_base_odd = (
                    smem_group_offset
                    + self._bf16_tma_smem_element_offset(
                        base_row_idx, base_tmem_col + Int32(1)
                    )
                )
            for group in cutlass.range_constexpr(self.cfg.epi_tile_n // 8):
                col_off = Int32(group * 8)
                for col_sub in cutlass.range_constexpr(2):
                    tmem_col = base_tmem_col + col_off + Int32(col_sub)
                    reg0 = group * 4 + col_sub
                    reg1 = group * 4 + 2 + col_sub
                    n_col = n_tile_base + tmem_col
                    if cutlass.const_expr(self.cfg.use_tma_store):
                        val0_f32 = t2r_rmem[reg0]
                        val1_f32 = t2r_rmem[reg1]
                        val2_f32 = t2r_slice1[reg0]
                        val3_f32 = t2r_slice1[reg1]
                        val0_f32 = self._maybe_apply_per_token_sf_a(
                            val0_f32, base_row_idx
                        )
                        val1_f32 = self._maybe_apply_per_token_sf_a(
                            val1_f32, base_row_idx + Int32(1)
                        )
                        val2_f32 = self._maybe_apply_per_token_sf_a(
                            val2_f32, base_row_idx + Int32(2)
                        )
                        val3_f32 = self._maybe_apply_per_token_sf_a(
                            val3_f32, base_row_idx + Int32(3)
                        )
                        if cutlass.const_expr(self.cfg.has_per_token_sf_b):
                            local_token_col = n_subtile_offset + tmem_col
                            token_sf = self._load_per_token_sf_b(local_token_col)
                            val0_f32, val1_f32 = self._fmul2(
                                val0_f32, val1_f32, token_sf, token_sf
                            )
                            val2_f32, val3_f32 = self._fmul2(
                                val2_f32, val3_f32, token_sf, token_sf
                            )
                        val0_f32 = self._maybe_add_bias_m(
                            val0_f32,
                            expert_idx,
                            m_tile_base + base_row_idx,
                            base_row_idx,
                        )
                        val1_f32 = self._maybe_add_bias_m(
                            val1_f32,
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(1),
                            base_row_idx + Int32(1),
                        )
                        val2_f32 = self._maybe_add_bias_m(
                            val2_f32,
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(2),
                            base_row_idx + Int32(2),
                        )
                        val3_f32 = self._maybe_add_bias_m(
                            val3_f32,
                            expert_idx,
                            m_tile_base + base_row_idx + Int32(3),
                            base_row_idx + Int32(3),
                        )
                        if cutlass.const_expr(is_eltwise_act):
                            val0_f32, val1_f32 = self._apply_eltwise_act_pair(
                                val0_f32, val1_f32, scale_gate
                            )
                            val2_f32, val3_f32 = self._apply_eltwise_act_pair(
                                val2_f32, val3_f32, scale_gate
                            )
                        val0_f32 = self._maybe_apply_scale_c(val0_f32, scale_c)
                        val1_f32 = self._maybe_apply_scale_c(val1_f32, scale_c)
                        val2_f32 = self._maybe_apply_scale_c(val2_f32, scale_c)
                        val3_f32 = self._maybe_apply_scale_c(val3_f32, scale_c)
                        if cutlass.const_expr(group == 0 and col_sub == 0):
                            # Overlap T2R and register arithmetic with the
                            # previous TMA read, then protect the first writer.
                            prims.cp_async_bulk_wait_group(0, read=True)
                            prims.barrier_cta_sync(
                                barrier_id=Int32(_TMA_STORE_BARRIER_BASE)
                                + warpgroup_idx,
                                thread_count=self.cfg.epilogue_threads_per_warpgroup,
                            )
                        if cutlass.const_expr(self.cfg.uses_fp8_output):
                            self._stage_swap_ab_fp8_tma_vec4(
                                val0_f32,
                                val1_f32,
                                val2_f32,
                                val3_f32,
                                base_row_idx,
                                tmem_col,
                            )
                        else:
                            if cutlass.const_expr(
                                self.cfg.swap_tma_store_offsets_are_col_linear
                            ):
                                if cutlass.const_expr(col_sub == 0):
                                    smem_offset = tma_smem_base_even
                                else:
                                    smem_offset = tma_smem_base_odd
                                smem_offset += Int32(
                                    group * self.cfg.swap_tma_store_col_group_step_elems
                                )
                            else:
                                group_stride_elems = Int32(
                                    self.cfg.num_bytes_c_tma_store_per_group // 2
                                )
                                smem_offset = (
                                    warpgroup_idx * group_stride_elems
                                    + self._bf16_tma_smem_element_offset(
                                        base_row_idx, tmem_col
                                    )
                                )
                            self._stage_swap_ab_bf16_tma_vec4(
                                self._to_output_value(val0_f32),
                                self._to_output_value(val1_f32),
                                self._to_output_value(val2_f32),
                                self._to_output_value(val3_f32),
                                smem_offset,
                            )
                    else:
                        for row_sub in cutlass.range_constexpr(4):
                            m_local_row = base_row_idx + Int32(row_sub)
                            m_row = m_tile_base + m_local_row
                            val_f32 = t2r_rmem[reg0]
                            if row_sub == 0:
                                val_f32 = t2r_rmem[reg0]
                            elif row_sub == 1:
                                val_f32 = t2r_rmem[reg1]
                            elif row_sub == 2:
                                val_f32 = t2r_slice1[reg0]
                            else:
                                val_f32 = t2r_slice1[reg1]
                            val_f32 = self._maybe_apply_per_token_sf_a(
                                val_f32, m_local_row
                            )
                            local_token_col = n_subtile_offset + tmem_col
                            val_f32 = self._maybe_apply_per_token_sf_b(
                                val_f32, local_token_col
                            )
                            val_f32 = self._maybe_add_bias_m(
                                val_f32, expert_idx, m_row, m_local_row
                            )
                            if cutlass.const_expr(is_eltwise_act):
                                val_f32 = self._apply_eltwise_act(val_f32, scale_gate)
                            val_f32 = self._maybe_apply_scale_c(val_f32, scale_c)
                            val_out = self._to_output_value(val_f32)
                            flat_idx = n_col * self._c_pitch(output_m) + m_row
                            token_in_bounds = (
                                n_subtile_offset + tmem_col
                            ) < token_limit
                            output_in_bounds = (m_row < output_m) & (
                                n_col < self.problem_n
                            )
                            if cutlass.const_expr(self.cfg.use_tma_oob_opt):
                                if token_in_bounds & output_in_bounds:
                                    self.gC.store(
                                        val_out,
                                        idx=flat_idx,
                                        vector_size=1,
                                        alignment=(
                                            1 if self.cfg.uses_fp8_output else 2
                                        ),
                                    )
                            else:
                                if output_in_bounds:
                                    self.gC.store(
                                        val_out,
                                        idx=flat_idx,
                                        vector_size=1,
                                        alignment=(
                                            1 if self.cfg.uses_fp8_output else 2
                                        ),
                                    )
            if cutlass.const_expr(self.cfg.use_tma_store):
                self._commit_swap_ab_plain_tma(
                    tile_coord_m,
                    tile_coord_n,
                    n_subtile_offset,
                    token_limit,
                    warp_idx,
                    warpgroup_idx,
                )
