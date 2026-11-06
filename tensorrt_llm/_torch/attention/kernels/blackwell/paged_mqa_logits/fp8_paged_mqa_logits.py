# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CuTe DSL FP8 paged MQA logits kernel for Blackwell (SM100).

Two opt-in flags on ``FP8MQALogitsKernel`` (and on ``_compile_fp8_kernel`` /
``_prepare_fp8_kernel_launch`` / ``fp8_paged_mqa_logits``); with both at their
defaults the traced kernel body, the launch and the ABI are unchanged.

  fused_weight_scale  The kernel consumes the weight-scale node's inputs (raw
                      FP32 weights ``[N, B]`` and the FP32 Q scale ``[N, B]``)
                      plus the runtime ``weight_scale_factor`` and computes
                      ``(raw * q_scale) * weight_scale_factor`` in FP32 at the
                      cold per-q weight register-cache fill (and in the SMEM
                      weight path).  Left-associated, two separately rounded
                      FP32 multiplies, exactly like the standalone weight-scale
                      kernel, so the logits are bit-identical to running that
                      kernel first.  The Q scale rides on the existing Q-stage
                      TMA barrier next to the weights (+128 B per q change).
                      Requires the FP32 epilogue.  The fused ABI is
                      ``FP8MQALogitsFusedWeightScaleKernel``; the accepted
                      ``FP8MQALogitsKernel`` ABI refuses the flag.
  pdl                 Programmatic Dependent Launch: the launch carries the
                      programmatic stream serialization attribute; the two TMA
                      warps issue one ``griddepcontrol.wait`` after the
                      pipeline init and before the first TMA load (Q / weights
                      / Q scale / K cache / block table are the only
                      predecessor-produced reads; the schedule-metadata and
                      context-length loads above the wait are host metadata).
                      Math, UMMA and epilogue roles never touch predecessor
                      data before a pipeline barrier that the TMA warps arm.
                      The math warps issue ``griddepcontrol.launch_dependents``
                      after their last logits store and the TMEM release; a
                      dependent grid still has to wait before reading logits.

  early_prologue      Opt-in (default off; refused together with ``pdl``): the
                      five pipelines are created with ``defer_init``, each TMA
                      warp initializes exactly the mbarriers it produces on and
                      issues the first Q / weight (/ Q-scale) copies and the
                      first KV tile of both groups BEFORE the CTA-wide init
                      sync (the block-table entries of the CTA's first task are
                      loaded meanwhile; the task loop skips that first KV issue
                      once); ``context_lens`` is warp-resident for batch <= 32
                      and later ``num_kv`` values are shuffles instead of
                      dependent global loads.  Consumers touch a barrier only
                      after the CTA-wide sync, so their ordering is unchanged
                      and the logits are bit-identical.  Every statement of the
                      variant is under ``if cutlass.const_expr(
                      self.early_prologue)`` with the previous code as the
                      ``else`` arm: the default build is the previous kernel.
                      The early copies read predecessor data, so the flag
                      cannot precede ``griddepcontrol.wait`` (hence not with
                      ``pdl``).

Architecture:
  - 384 threads: 256 math (2 WGs) + 128 specialized (2 TMA + 2 UMMA)
  - 1 TMA per KV block [128, 128], UMMA iterates 4x K=32
  - 2 warp groups process 2 KV blocks per iteration (kNumMathWarpGroups=2)
  - Q reloaded via TMA pipeline when q_idx (batch) changes
  - Persistent kernel: CTAs iterate through assigned (q_idx, kv_idx) pairs
  - Weights cached in registers: preloaded once per q_idx change (not per KV block)
  - KV Scales loaded via TMA to SMEM (separate pipeline per group, Math consumes)

Merged KV+Scale pipeline:
  - KV data and scales share a single TMA barrier per group
  - TMA loads both KV and Scale under one barrier (combined tx_count)
  - UMMA waits on merged barrier (for KV GEMM), does NOT release
  - Math waits on merged barrier (for scale read), Math releases

Fused KV layout:
  - KV data and scales stored contiguously per physical block:
    [num_phys_blocks, block_kv * (head_dim + 4)] bytes
  - Per block: [KV_all_tokens (block_kv * head_dim bytes)] [Scales (block_kv * 4 bytes)]
  - KV and Scale views are derived inside __call__ using CuTE pointer arithmetic

Scheduler:
  - schedule_meta[sm_idx] = (start_q_idx, start_kv_idx / kNumMathWarpGroups)
  - schedule_meta[sm_idx+1] = end boundary for this CTA
  - fetch_next_task pattern: each warp role independently advances (q_idx, kv_idx)
  - kv_idx in units of KV blocks, advances by kNumMathWarpGroups=2 per step

Dynamic shape support:
  - Model-constant dims (block_kv, head_dim, N, per_token) remain static for codegen
  - Runtime-varying dims (batch_size, num_phys_blocks, max_ctx, max_blocks_per_seq,
    num_ctas) are marked dynamic via mark_compact_shape_dynamic
  - Allows JIT cache reuse across different batch sizes / sequence lengths

Epilogue dtype flows (--acc_dtype / --epi_dtype):

  Flow 1: --acc_dtype Float32 --epi_dtype Float16
    Q(FP8) x K(FP8) -> MMA acc(FP32) -> TMEM(FP32)
      -> LDTM -> Reg(FP32) -> cvt FP16 -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  Flow 2: --acc_dtype Float16 --epi_dtype Float16
    Q(FP8) x K(FP8) -> MMA acc(FP16) -> TMEM(FP16, pack_16b)
      -> LDTM -> Reg(FP16) -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  --output_dtype: Float32 (default), Float16, BFloat16. Controls logits tensor
  dtype and final store conversion.
  Default: --acc_dtype Float32 --epi_dtype Float32 --output_dtype Float32

The module also contains a standalone correctness driver with input generation,
schedule construction, compilation, dispatch, and a PyTorch reference. Importing
the module does not execute the driver.
"""

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float16, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.arch import get_max_tmem_alloc_cols
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import BaseDSL, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .mqa_logits_util import atom_token_offset, atom_tokens_after, is_uneven_split

# CuTe DSL CUDA 13 validates rounding modes as string literals. The string
# form is also accepted by older wrappers, so keep it version-independent.
_RND_RN = "rn"


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc=None,
    ip=None,
) -> Int32:
    f16_ty = Float16.mlir_type
    i32_ty = Int32.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = vector.from_elements(
        vec2_f16,
        (Float16(a).ir_value(loc=loc, ip=ip), Float16(b).ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(i32_ty, v, loc=loc, ip=ip))


@dsl_user_op
def unpack_f16x2(
    packed: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float16, Float16]:
    f16_ty = Float16.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = llvm.bitcast(vec2_f16, Int32(packed).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    r0 = Float16(vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
    r1 = Float16(vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip))
    return r0, r1


@dsl_user_op
def fma_f16x2(
    a: Int32,
    b: Int32,
    c: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


def _subview(tensor, offset, count):
    """Load ``count`` contiguous elements of ``tensor`` starting at ``offset``.

    A view over the iterator rather than a reshape into groups: the offset is an
    element index, so it need not be a multiple of ``count``. Reshaping to
    ``(count, size // count)`` and selecting a mode would address groups
    instead, and silently read the wrong one whenever the offset did not divide.
    """
    return cute.make_tensor(tensor.iterator + offset, cute.make_layout(count)).load()


def _target_is_rubin() -> bool:
    """Whether the JIT target is a Rubin arch.

    There is no canonical "is Rubin" Arch API: Arch.BlackwellArchs() lumps the
    Rubin archs in with Blackwell, and Arch.is_family_of() only compares the
    major version and is suffix-sensitive. Within major 10 a minor version of 7
    or above is Rubin, so `minor >= 7` is the discriminator.
    """
    arch = BaseDSL._get_dsl().get_arch_enum()
    return arch.major == 10 and arch.minor >= 7


# Heads per next_n slot kept in the register weight cache, by atom and weight
# dtype (the remaining heads take the SMEM epilogue path). Must be a multiple
# of 4 (the epilogue FMA works on head groups of 4). Lowering a value is safe
# (more SMEM traffic); raising one needs a spill check (LDL/STL count in the
# SASS), to be redone when the arch, toolchain or epilogue changes. Atoms 1..4
# were tuned on 64 heads (fp32 atom 4: 40 spilled to shared memory, 36 is the
# largest spill-free width); atoms 5..8 (<= 32 heads) spread ~144 fp32
# registers over the slots. 2-byte weights pack two per register.
_W_IN_REG = {
    cutlass.Float32: {1: 64, 2: 52, 3: 52, 4: 36, 5: 28, 6: 24, 7: 20, 8: 16},
    cutlass.Float16: {1: 64, 2: 64, 3: 64, 4: 48, 5: 56, 6: 48, 7: 40, 8: 36},
}
# Atoms 9..16 are only reachable with <= 16 heads (N tile <= 256); the same
# 144-register budget spread over the slots, floor 4 (one epilogue head group).
for _a in range(9, 17):
    _W_IN_REG[cutlass.Float32][_a] = max(4, (144 // _a) // 4 * 4)
    _W_IN_REG[cutlass.Float16][_a] = max(4, (288 // _a) // 4 * 4)
# Head-count specific exceptions (the per-token epilogue state also grows with
# the head count): 128 heads, atom 2, fp32 spills at 52 and 48; 44 is the
# widest spill-free width.
_W_IN_REG_BY_HEADS = {(cutlass.Float32, 2, 128): 44}


def _max_num_w_in_reg(epi_dtype, atom: int, num_heads: int) -> int:
    return _W_IN_REG_BY_HEADS.get((epi_dtype, atom, num_heads), _W_IN_REG[epi_dtype][atom])


class FP8MQALogitsKernel:
    """FP8 paged MQA logits kernel for Blackwell (SM100).

    Each CTA processes a range of (q_idx, kv_split) pairs.
    A split = 2 consecutive KV blocks within a sequence (one per warp group).
    Q is shared between warp groups and reloaded when q_idx changes.
    """

    def __init__(
        self,
        block_kv: int = 128,
        phys_block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        next_n: int = 1,
        num_next_n_atoms: int = 1,
        atom: int | None = None,
        num_sms: int = 148,
        remove_kv_wait_in_epilogue: bool = False,
        early_tmem_copy: bool = False,
        smem_subpartition_opt: bool = False,
        max_kv_pipeline: bool = False,
        max_umma_pipeline: bool = False,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        use_flat_logits_view=None,
        use_two_level_task_loop=None,
        use_paired_pipeline_polls=None,
        use_relu_trick: bool = True,
        fused_weight_scale: bool = False,
        pdl: bool = False,
        early_prologue: bool = False,
    ):
        if (acc_dtype, epi_dtype) not in (
            (cutlass.Float32, cutlass.Float32),
            (cutlass.Float32, cutlass.Float16),
            (cutlass.Float16, cutlass.Float16),
        ):
            raise ValueError(
                "FP8MQALogitsKernel supports only (acc_dtype, epi_dtype) "
                "combinations (Float32, Float32), (Float32, Float16), and "
                f"(Float16, Float16); got ({acc_dtype}, {epi_dtype})"
            )
        if output_dtype not in (
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
        ):
            raise ValueError(
                "FP8MQALogitsKernel output_dtype must be Float32, Float16, "
                f"or BFloat16; got {output_dtype}"
            )

        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        if block_kv % phys_block_kv != 0:
            raise ValueError(
                f"block_kv={block_kv} must be divisible by phys_block_kv={phys_block_kv}"
            )
        self.num_blocks_per_mma = block_kv // phys_block_kv
        if self.num_blocks_per_mma > 4:
            raise ValueError(f"num_blocks_per_mma={self.num_blocks_per_mma} exceeds max 4")
        self.remove_kv_wait_in_epilogue = remove_kv_wait_in_epilogue
        self.early_tmem_copy = early_tmem_copy
        self.smem_subpartition_opt = smem_subpartition_opt
        # Per-arch optimization gates (None = auto). All consumed via
        # cutlass.const_expr in-kernel.
        # - use_flat_logits_view (flat logits view + carried row offset):
        #   regresses on Rubin — the backend schedules the UMMA
        #   consumer-release late in the math loop, throttling the MMA
        #   pipe per KV tile, growing with ctx. Pure win on Blackwell.
        #   Auto: off on Rubin.
        # - use_two_level_task_loop / use_paired_pipeline_polls: with the
        #   flat view off, both on wins on Rubin at nn<4; at nn=4 the
        #   baseline loop is the best achievable there. Auto: on except
        #   Rubin nn=4 (both fall back together so nn=4 compiles to the
        #   baseline task-loop form).
        # The Rubin values were tuned on exact instantiations (atom == tile);
        # with the uneven epilogue dispatch they cost up to 1.9x on Rubin, so the
        # auto values apply them to exact instantiations only.
        is_rubin = _target_is_rubin()
        uneven = is_uneven_split(next_n, num_next_n_atoms, atom)
        rubin_exact = is_rubin and not uneven
        if use_flat_logits_view is None:
            use_flat_logits_view = not rubin_exact
        if use_two_level_task_loop is None:
            use_two_level_task_loop = not (rubin_exact and next_n >= 4)
        if use_paired_pipeline_polls is None:
            use_paired_pipeline_polls = not (rubin_exact and next_n >= 4)
        self.use_flat_logits_view = use_flat_logits_view
        self.use_two_level_task_loop = use_two_level_task_loop
        self.use_paired_pipeline_polls = use_paired_pipeline_polls
        # relu(x) = (x+|x|)/2 (FADD2+abs, 2x the throughput of cute.max's
        # single FMNMX.NAN) vs the exact vector max. On by default, as in
        # DeepGEMM's SM100 epilogue; x+|x| overflows to inf for finite
        # x >= 2^127, which unbounded scales and weights can reach.
        self.use_relu_trick = use_relu_trick
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n = next_n  # logical next-token count (drives output row layout)
        # In-kernel atom split: next_n is decomposed into num_next_n_atoms atoms
        # of next_n_atom positions each (1 => direct). The MMA/TMEM N tile is
        # sized by the per-atom next_n; the logical next_n still drives out_row.
        # Uneven split (next_n % num_next_n_atoms != 0): the first atom_rem atoms
        # hold atom_base+1 tokens, the rest atom_base; the fixed tile width
        # `atom` (>= widest atom) must then be passed explicitly.
        self.num_next_n_atoms = num_next_n_atoms
        self.atom_base, self.atom_rem = divmod(next_n, num_next_n_atoms)
        self.next_n_atom = self.atom_base if atom is None else atom
        # "Uneven" = the tile holds padded slots: a remainder, OR a tile wider
        # than every atom's token count (e.g. 4 heads, next_n=5 -> one atom in
        # a 6-token tile, odd atoms being illegal N tiles there). Both use the
        # per-token-windowed views and the base/rem-driven epilogue; the exact
        # atom == tile path is untouched.
        self.uneven = uneven
        assert self.next_n_atom >= self.atom_base + (self.atom_rem > 0), (
            f"atom={atom} too narrow for next_n={next_n}/{num_next_n_atoms} atoms"
        )
        self.N = self.next_n_atom * num_heads
        self.num_sms = num_sms
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.epi_bytes = 2 if epi_dtype == cutlass.Float16 else 4
        # sW stage stride padded to 128-byte SMEM alignment for TMA bulk copy.
        # Without padding, e.g. fp16 + N=32 gives 64B per stage, so stage 1
        # at +64 would be misaligned (TMA requires 128-byte aligned SMEM dest).
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.epi_bytes
        # Fusion B: the FP32 Q scale gets its own 128-byte-aligned stage row so
        # its TMA copy is legal; it is only allocated when the fusion is on.
        if fused_weight_scale and epi_dtype != cutlass.Float32:
            raise ValueError(
                "fused_weight_scale reproduces I8's FP32 arithmetic and requires "
                f"epi_dtype=Float32; got {epi_dtype}"
            )
        self.fused_weight_scale = bool(fused_weight_scale)
        self.pdl = bool(pdl)
        # Opt-in early-TMA prologue (module docstring, "early_prologue"): the
        # TMA warps init their own mbarriers and issue the first copies before
        # the CTA-wide init sync.  Those copies read predecessor-produced data,
        # so they must not precede griddepcontrol.wait: the flag is refused
        # together with pdl.  Default off: the default build is the previous
        # kernel unchanged.
        if early_prologue and self.pdl:
            raise ValueError(
                "early_prologue issues predecessor-dependent TMA copies before "
                "the CTA-wide init sync and cannot be combined with pdl=True"
            )
        # The early prologue keeps context_lens warp-resident (lane l holds
        # ctx[min(l, batch_size - 1)]) and reads it as shuffle_sync(lane_ctx,
        # q_idx).  Both index by the kernel's batch_size, which is the Q /
        # weights L dim = batch * num_next_n_atoms q atoms, while context_lens
        # has only batch native rows: with num_next_n_atoms > 1 the lane load
        # runs past the end of context_lens and the shuffle returns ctx[qa]
        # instead of the atom's _atom_ctx_len(qa) (seq = qa // num_atoms minus
        # the stagger).  The GLM route always uses num_next_n_atoms == 1, so
        # the pair is refused rather than silently miscomputing num_kv.
        if early_prologue and num_next_n_atoms > 1:
            raise ValueError(
                "early_prologue keeps context_lens warp-resident indexed by the "
                "q-atom batch_size (batch * num_next_n_atoms) while context_lens "
                "has one row per sequence; it cannot be combined with "
                f"num_next_n_atoms={num_next_n_atoms} > 1"
            )
        self.early_prologue = bool(early_prologue)
        self.qscale_bytes = 4
        self.qscale_stage_stride = (
            (self.N * self.qscale_bytes + 127) // 128 * 128
        ) // self.qscale_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "num_heads // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        self.num_q_stages = 3  # 3 stages for Q pipelining across batch sequences

        # TMEM columns (arch-aware): each group needs N columns per UMMA stage,
        # so max_umma_stages = TMEM_COLS // (2 * N). Rubin exposes more than
        # Blackwell's 512, letting larger N still fit >= 2 UMMA stages.
        arch = BaseDSL._get_dsl().get_arch_enum()
        TMEM_COLS = get_max_tmem_alloc_cols(f"sm_{arch.major}{arch.minor}")
        # max_umma_pipeline (default OFF): opt-in second UMMA stage (double-buffer
        # the MMA accumulator in TMEM) when TMEM has room -- min(2, ...). On Rubin
        # only next_n=1/2 (N=64/128) fit 2 stages; next_n=3/4 stay at 1.
        # Kept OFF: an A/B sweep on Rubin (next_n=1..4) was neutral (mean ON/OFF
        # ~1.00 at next_n=1, ~0.99 at next_n=2; no meaningful win) -- the GEMM K is
        # tiny so MMA isn't the bottleneck and a second UMMA stage buys nothing.
        if max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // (2 * self.N))
        else:
            self.num_umma_stages = 1

        # max_kv_pipeline: opt-in deeper KV pipeline that spends the full SMEM
        # capacity on stages (arch-aware via get_smem_capacity_in_bytes: e.g.
        # Rubin's larger SMEM gives more stages than the default 3). Kept OFF
        # by default: an A/B sweep on Rubin (next_n=1..4) showed it is
        # neutral-to-slower everywhere (mean ON/OFF ~1.08 at next_n=1/2, ~1.01 at
        # next_n=3/4; never a meaningful win) -- this kernel is not KV-latency
        # starved at 3 stages, so extra stages only add SMEM/mbarrier/L2 cost.
        if max_kv_pipeline:
            smem_capacity = cutlass.memory.get_smem_capacity_in_bytes()
            # Reserve ~1 KB for barriers and misc
            SMEM_BUDGET = smem_capacity - 1024
            # KV+Scale per stage (×2 groups):
            #   2 * (block_kv * head_dim * 1B + block_kv * 4B)
            kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
            # Q+W per stage: Q is N * head_dim * 1B, W uses padded stride
            qw_per_stage = self.N * head_dim + self.w_stage_stride * self.epi_bytes
            if self.fused_weight_scale:
                qw_per_stage += self.qscale_stage_stride * self.qscale_bytes
            qw_total = qw_per_stage * self.num_q_stages
            self.num_kv_stages = (SMEM_BUDGET - qw_total) // kv_scale_per_stage
        else:
            self.num_kv_stages = 3

        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB),
        # avoiding sub-bank conflicts with UMMA reading sKV.
        # Layout: barriers(~256B) | sKV_0 | sKV_1 | sQ | [pad] | sW | sScales
        if self.smem_subpartition_opt:
            BOUNDARY = 128 * 1024
            used = (
                256
                + 2 * (block_kv * head_dim * self.num_kv_stages)
                + self.N * head_dim * self.num_q_stages
            )
            used = ((used + 127) // 128) * 128
            if used < BOUNDARY:
                self.smem_pad_bytes = ((BOUNDARY - used + 1023) // 1024) * 1024
            else:
                self.smem_pad_bytes = 0
        else:
            self.smem_pad_bytes = 0

        self.acc_dtype = acc_dtype
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _atom_seq(self, qa):
        """Native sequence (block_table / context_lens row) of a q_atom_idx.
        num_next_n_atoms == 1 => identity."""
        return qa // self.num_next_n_atoms

    def _atom_ctx_len(self, qa, context_lens):
        """Per-atom context length: atom i (= qa % num_atoms, 0 = oldest) sees
        ctx shortened by (num_atoms-1-i)*next_n_atom so the split's staggered
        causal limits match the unsplit next_n. num_atoms == 1 => ctx[qa]."""
        na = self.num_next_n_atoms
        if cutlass.const_expr(not self.uneven):
            return context_lens[qa // na] - (na - 1 - qa % na) * self.next_n_atom
        else:  # must match the host schedule builder (same atom_tokens_after)
            return context_lens[qa // na] - atom_tokens_after(
                qa, na, self.atom_base, self.atom_rem, self.next_n
            )

    def _atom_out_row_base(self, qa):
        """Output row base for a q_atom_idx: seq*next_n + atom*next_n_atom (the
        +t within the atom is added by the caller). num_atoms == 1 => qa*next_n."""
        na = self.num_next_n_atoms
        if cutlass.const_expr(not self.uneven):
            return (qa // na) * self.next_n + (qa % na) * self.next_n_atom
        else:
            return (qa // na) * self.next_n + atom_token_offset(
                qa, na, self.atom_base, self.atom_rem
            )

    def _q_tma_coord(self, qa):
        """TMA L-coordinate of the Q / weights / Q-scale stage for q_atom_idx.
        rem == 0: atom-tiled host views, L = atom row. rem > 0: per-token-strided
        host views (see _prepare_fp8_kernel_launch), L = flat token index.
        (A method, not a closure: the DSL cannot carry a closure across the
        producer's runtime-if regions.)
        """
        if cutlass.const_expr(not self.uneven):
            return qa
        else:
            return self._atom_seq(qa) * self.next_n + atom_token_offset(
                qa, self.num_next_n_atoms, self.atom_base, self.atom_rem
            )

    def _epi_tokens(self, n_valid, advance_umma, *ops):
        """Epilogue for one KV tile: token slots 0..n_valid-1 (see _epi_t_body)."""
        for t in range(n_valid):
            self._epi_t_body(t, n_valid, advance_umma, *ops)

    def _epi_t_body(
        self,
        t,
        n_valid,
        advance_umma,
        tTR,
        tiled_copy_ref,
        tTR_rAcc,
        w_cache,
        sW,
        sQScale,
        q_stage_local,
        NUM_W_IN_REG,
        umma_pipeline,
        umma_cons_state,
        scale_val,
        kv_pos,
        out_row_off,
        logits_stride0,
        mLogits_flat,
        mLogits_2d,
        q_idx,
        weight_scale_factor,
        packed_zero,
    ):
        """One epilogue token-slot `t` (LDTM -> ReLU+FMA over heads -> logits
        store) for one math warp group; shared by WG0/WG1. Every operand is an
        explicit argument (no closure: the rem>0 dispatch calls this from inside
        a staged runtime `if`, where the DSL forbids closure capture). `n_valid`
        is the atom's compile-time valid-token count and times the UMMA release.
        """
        num_heads, num_epi_subtiles = self.num_heads, self.num_epi_subtiles
        subtile_n = num_heads // num_epi_subtiles
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            pz = pack_f16x2(Float16(0.0), Float16(0.0))
            ps01 = pz
            ps23 = pz
        else:
            s0x = cutlass.Float32(0.0)
            s0y = cutlass.Float32(0.0)
            s1x = cutlass.Float32(0.0)
            s1y = cutlass.Float32(0.0)
        for i in range(num_epi_subtiles):
            # LDTM for sub-tiles 1..N-1
            # (sub-tile 0 handled above)
            if t > 0 or i > 0:
                cute.copy(
                    tiled_copy_ref,
                    tTR[
                        (
                            None,
                            None,
                            None,
                            0,
                            t * num_epi_subtiles + i,
                        )
                    ],
                    tTR_rAcc,
                )
                cute.arch.fence_view_async_tmem_load()
            # Release UMMA after last LDTM+fence
            if t == n_valid - 1 and i == num_epi_subtiles - 1:
                umma_pipeline.consumer_release(umma_cons_state)
                # advance() rebinds pipeline state; the DSL cannot thread that out
                # of a method called inside a runtime `if`, so the rem>0 dispatch
                # passes False and advances textually after its if/else.
                if advance_umma:
                    umma_cons_state.advance()
            acc_vec = tTR_rAcc.load()
            sw_col = sW[(None, q_stage_local)]
            if cutlass.const_expr(self.fused_weight_scale):
                sqs_col = sQScale[(None, q_stage_local)]
            # Reg-path: weights from registers
            reg_h_end = min(subtile_n, max(0, NUM_W_IN_REG - i * subtile_n))
            for h in range(0, reg_h_end, 4):
                n0 = h
                h_g = i * subtile_n + h
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    pa01 = max_f16x2(pack_f16x2(Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])), pz)
                    pa23 = max_f16x2(
                        pack_f16x2(Float16(acc_vec[n0 + 2]), Float16(acc_vec[n0 + 3])), pz
                    )
                    r0 = t * NUM_W_IN_REG + h_g
                    ps01 = fma_f16x2(pa01, pack_f16x2(w_cache[r0], w_cache[r0 + 1]), ps01)
                    ps23 = fma_f16x2(pa23, pack_f16x2(w_cache[r0 + 2], w_cache[r0 + 3]), ps23)
                else:
                    x0 = acc_vec[n0]
                    x1 = acc_vec[n0 + 1]
                    x2 = acc_vec[n0 + 2]
                    x3 = acc_vec[n0 + 3]
                    if cutlass.const_expr(self.use_relu_trick):
                        a0, a1 = cute.arch.add_packed_f32x2(
                            (x0, x1), (abs(x0), abs(x1)), rnd=_RND_RN
                        )
                        a2, a3 = cute.arch.add_packed_f32x2(
                            (x2, x3), (abs(x2), abs(x3)), rnd=_RND_RN
                        )
                    else:
                        a0 = cute.arch.fmax(x0, cutlass.Float32(0.0), nan=True)
                        a1 = cute.arch.fmax(x1, cutlass.Float32(0.0), nan=True)
                        a2 = cute.arch.fmax(x2, cutlass.Float32(0.0), nan=True)
                        a3 = cute.arch.fmax(x3, cutlass.Float32(0.0), nan=True)
                    r0 = t * NUM_W_IN_REG + h_g
                    w0 = w_cache[r0]
                    w1 = w_cache[r0 + 1]
                    w2 = w_cache[r0 + 2]
                    w3 = w_cache[r0 + 3]
                    s0x, s0y = cute.arch.fma_packed_f32x2(
                        (a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN
                    )
                    s1x, s1y = cute.arch.fma_packed_f32x2(
                        (a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN
                    )
            # SMEM-path: weights from shared mem
            smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
            for h in range(smem_h_start, subtile_n, 4):
                n0 = h
                h_g = i * subtile_n + h
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    pa01 = max_f16x2(pack_f16x2(Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])), pz)
                    pa23 = max_f16x2(
                        pack_f16x2(Float16(acc_vec[n0 + 2]), Float16(acc_vec[n0 + 3])), pz
                    )
                    w_i = t * num_heads + h_g
                    ps01 = fma_f16x2(pa01, pack_f16x2(sw_col[w_i], sw_col[w_i + 1]), ps01)
                    ps23 = fma_f16x2(pa23, pack_f16x2(sw_col[w_i + 2], sw_col[w_i + 3]), ps23)
                else:
                    x0 = acc_vec[n0]
                    x1 = acc_vec[n0 + 1]
                    x2 = acc_vec[n0 + 2]
                    x3 = acc_vec[n0 + 3]
                    if cutlass.const_expr(self.use_relu_trick):
                        a0, a1 = cute.arch.add_packed_f32x2(
                            (x0, x1), (abs(x0), abs(x1)), rnd=_RND_RN
                        )
                        a2, a3 = cute.arch.add_packed_f32x2(
                            (x2, x3), (abs(x2), abs(x3)), rnd=_RND_RN
                        )
                    else:
                        a0 = cute.arch.fmax(x0, cutlass.Float32(0.0), nan=True)
                        a1 = cute.arch.fmax(x1, cutlass.Float32(0.0), nan=True)
                        a2 = cute.arch.fmax(x2, cutlass.Float32(0.0), nan=True)
                        a3 = cute.arch.fmax(x3, cutlass.Float32(0.0), nan=True)
                    w_i = t * num_heads + h_g
                    w0 = sw_col[w_i]
                    w1 = sw_col[w_i + 1]
                    w2 = sw_col[w_i + 2]
                    w3 = sw_col[w_i + 3]
                    if cutlass.const_expr(self.fused_weight_scale):
                        w0 = (w0 * sqs_col[w_i]) * weight_scale_factor
                        w1 = (w1 * sqs_col[w_i + 1]) * weight_scale_factor
                        w2 = (w2 * sqs_col[w_i + 2]) * weight_scale_factor
                        w3 = (w3 * sqs_col[w_i + 3]) * weight_scale_factor
                    s0x, s0y = cute.arch.fma_packed_f32x2(
                        (a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN
                    )
                    s1x, s1y = cute.arch.fma_packed_f32x2(
                        (a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN
                    )
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            # Spelled out, not reduce(): the (S0+S2)+(S1+S3) pairing
            # lets both adds land in one packed HADD2, then a second
            # folds the halves -- 2 instructions, and bit-exact.
            s0, s1 = unpack_f16x2(add_f16x2(ps01, ps23))
            result_t = s0 + s1
        else:
            # Spelled out, not reduce(): pins the original
            # left-to-right order.
            result_t = s0x + s0y + s1x + s1y
        if cutlass.const_expr(self.use_flat_logits_view):
            out_off_t = out_row_off + t * logits_stride0 + kv_pos
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                mLogits_flat[out_off_t] = self.output_dtype(result_t * Float16(scale_val))
            else:
                # The relu trick's missing /2 folds in here
                # instead of precomputing scale_half
                # (scale_val is reassigned across several
                # branches upstream); with the trick off,
                # result_t is already relu'd, no /2 needed.
                if cutlass.const_expr(self.use_relu_trick):
                    scale_mul = scale_val * 0.5
                else:
                    scale_mul = scale_val
                mLogits_flat[out_off_t] = self.output_dtype(result_t * scale_mul)
        else:
            out_row = self._atom_out_row_base(q_idx) + t
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                mLogits_2d[(out_row, kv_pos)] = self.output_dtype(result_t * Float16(scale_val))
            else:
                if cutlass.const_expr(self.use_relu_trick):
                    scale_mul = scale_val * 0.5
                else:
                    scale_mul = scale_val
                mLogits_2d[(out_row, kv_pos)] = self.output_dtype(result_t * scale_mul)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 32

        # Full-K: tile K = head_dim (128), 1 TMA per block
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k  # 4
        full_k = mma_inst_shape_k * mma_inst_tile_k  # 128
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            full_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # KV SMEM: 3 stages per group, each stage holds full [128, 128]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        # Q SMEM: 1 stage, holds full [N, 128]
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.memory.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
        )

        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] FP8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size]
        weights: cute.Tensor,  # [N, batch_size] (transposed for TMA)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Accepted ABI: pre-scaled weights.  ``fused_weight_scale`` must be off."""
        if cutlass.const_expr(self.fused_weight_scale):
            raise ValueError(
                "fused_weight_scale=True must be launched through "
                "FP8MQALogitsFusedWeightScaleKernel (raw weights + Q scale ABI)"
            )
        # ``weights`` doubles as the (never read) Q-scale placeholder so the
        # shared launch body has one static signature.
        self._launch_body(
            kv_fused,
            b,
            weights,
            weights,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            num_phys_blocks,
            batch_size,
            cutlass.Float32(1.0),
            stream,
        )

    @cute.jit
    def _launch_body(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] FP8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size]
        weights: cute.Tensor,  # [N, batch_size] (transposed for TMA); raw if fused
        q_scales: cute.Tensor,  # [N, batch_size] FP32 Q scale (fused only)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        weight_scale_factor: cutlass.Float32,  # fused only
        stream: cuda.CUstream,
    ):
        # Derive KV and Scale views from fused buffer using CuTE ops.
        # Fused layout per physical block: [KV data (phys_block_kv*head_dim)] [Scales (phys_block_kv*4)]
        phys_block_kv = self.phys_block_kv
        scale_offset_elems = phys_block_kv * self.head_dim  # in FP8 elements

        # Recast fused buffer to FP8 (same 1-byte elements, needed for MMA type inference)
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)

        # Q (b) was passed as uint8 to work around DLPack's lack of float8 support;
        # recast back to FP8 so MMA type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)

        # Read the real per-block stride (bytes = FP8 elements) from the input.
        # When KV is the indexer K-cache pool view, the pool is laid out as
        # [num_blocks, num_layers, kvFactor, blockSize], so dim-0 stride =
        # num_layers * kvFactor * phys_block_bytes (not phys_block_bytes).
        # Using the input stride keeps both contiguous test path and the
        # strided prod path correct. FP8 = 1 byte per element, so the byte
        # stride is the same as the element stride after recast.
        kv_block_stride = kv_fused.layout.stride[0]

        # KV view: [phys_block_kv, head_dim, num_phys_blocks] FP8
        # Each TMA loads one physical block; multiple TMAs fill a compute tile.
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, kv_block_stride),
        )
        a = cute.make_tensor(kv_fp8.iterator, kv_layout)

        # Scale view: offset pointer to scale region, recast FP8 → Float32
        # [phys_block_kv, num_phys_blocks] float32 (after recast)
        scale_fp8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks),
            stride=(1, kv_block_stride),
        )
        scale_fp8 = cute.make_tensor(kv_fp8.iterator + scale_offset_elems, scale_fp8_layout)
        scales = cute.recast_tensor(scale_fp8, cutlass.Float32)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = cutlass.tensor_utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = cutlass.tensor_utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA for KV (A) — fmha_decode_paged pattern.
        # Build a TMA SMEM layout via tiled_divide on the full compute-tile
        # layout, then select to drop trivial K dim. Atom uses mode [0] as
        # single-tile SMEM layout and (phys, head) as cta_tiler.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        # ((tile_M, tile_K), rest_M, rest_K, stages) → drop trivial rest_K
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        # ((tile_M, tile_K), rest_M=num_sub_blocks, stages)
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            a,
            self.a_tma_view_layout[0],  # atom SMEM = single-tile (mode 0)
            (self.phys_block_kv, self.head_dim),
        )

        # TMA for Q (B) — full K=128, L dim = batch_size (unchanged)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA for Weights — [N, batch_size], tile [N], L=batch_size
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            self.w_smem_layout_staged.shape[:1],
        )

        # Fusion B: TMA for the FP32 Q scale — same [N, batch_size] shape and
        # per-stage tiling as the weights, on the same Q-stage barrier.  When
        # the fusion is off the weight atom stands in and is never used.
        self.qscale_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.qscale_stage_stride),
        )
        if cutlass.const_expr(self.fused_weight_scale):
            qscale_smem_per_stage = cute.select(self.qscale_smem_layout_staged, mode=[0])
            tma_atom_qscale, tma_tensor_qscale = cpasync.make_tiled_tma_atom(
                tma_load_op,
                q_scales,
                qscale_smem_per_stage,
                self.qscale_smem_layout_staged.shape[:1],
            )
        else:
            tma_atom_qscale, tma_tensor_qscale = tma_atom_w, tma_tensor_w
        if cutlass.const_expr(self.uneven):
            # Uneven split: overlay the per-token windows on the TMA COORDINATE tensors
            # (the descriptors stay plain flat tensors). Window t starts at token t:
            # Q row t*H, weights element t*H; the box is N = atom*H rows / elements.
            H = self.num_heads
            T = cute.size(b, mode=[0]) // H
            tma_tensor_b = cute.make_tensor(
                tma_tensor_b.iterator,
                cute.make_layout(
                    (self.N, self.head_dim, T),
                    stride=(
                        cute.ScaledBasis(1, 1),
                        cute.ScaledBasis(1, 0),
                        cute.ScaledBasis(H, 1),
                    ),
                ),
            )
            tma_tensor_w = cute.make_tensor(
                tma_tensor_w.iterator,
                cute.make_layout(
                    (self.N, T), stride=(cute.ScaledBasis(1, 0), cute.ScaledBasis(H, 0))
                ),
            )
            if cutlass.const_expr(self.fused_weight_scale):
                tma_tensor_qscale = cute.make_tensor(
                    tma_tensor_qscale.iterator,
                    cute.make_layout(
                        (self.N, T),
                        stride=(cute.ScaledBasis(1, 0), cute.ScaledBasis(H, 0)),
                    ),
                )
            else:
                tma_tensor_qscale = tma_tensor_w

        # TMA for Scales — [phys_block_kv, num_phys_blocks], tile [phys_block_kv]
        # SMEM holds compute_block_kv scales per stage; filled by
        # num_blocks_per_mma sub-block TMAs at consecutive offsets.
        self.s_smem_layout_staged = cute.make_layout((self.block_kv, self.num_kv_stages))
        s_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op,
            scales,
            s_smem_per_subblock,
            (phys_block_kv,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per sub-block: phys_block_kv * head_dim (KV) + phys_block_kv * 4 (scales)
        kv_tma_bytes_per_subblock = phys_block_kv * self.head_dim
        scale_tma_bytes_per_subblock = phys_block_kv * 4
        # Total per compute tile = num_blocks_per_mma sub-blocks
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        # Q + Weights share barrier (like DeepGEMM)
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size
        if cutlass.const_expr(self.fused_weight_scale):
            # Fusion B: the Q-scale copy completes on the same barrier.
            self.num_q_tma_bytes += self.N * self.qscale_bytes

        num_ctas = self.num_sms

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_qscale,
            tma_tensor_qscale,
            tma_atom_s,
            tma_tensor_s,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            batch_size,
            weight_scale_factor,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.qscale_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.pdl,
            # Pins the entry register count so ptxas keeps the setmaxnreg pair
            # (otherwise diagnostic C7508 and the math warps compile at 168 regs;
            # seen at 32/48/96 heads).
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV pool
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = batch_size; batch*next_n tokens if rem>0)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, batch_size]
        tma_atom_qscale: cute.CopyAtom,  # Fusion B; aliases W atom when off
        mQScale_tma: cute.Tensor,  # Q-scale TMA coord tensor [N, batch_size]
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,  # Scales TMA coord tensor [block_kv, num_phys_blocks]
        mLogits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        mBlockTable: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        mContextLens: cute.Tensor,  # [batch_size]
        mScheduleMeta: cute.Tensor,  # [num_sms+1, 2] int32
        batch_size: cutlass.Int32,
        weight_scale_factor: cutlass.Float32,  # Fusion B only
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        qscale_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Warp roles (matches DeepGEMM SM100)
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11

        # Early schedule metadata load: issue global loads ASAP so their
        # ~200-cycle L2 latency overlaps with subsequent prologue setup
        # (SMEM alloc, TMA partition, MMA fragment creation, etc.)
        NUM_MATH_WG = 2  # kNumMathWarpGroups
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        # In-kernel atom-split addressing lives in the self._atom_* methods
        # (identity when num_next_n_atoms == 1). qa is a q_atom_idx in
        # [0, batch*num_atoms): seq = qa//num_atoms is the native block_table /
        # context_lens row, atom = qa%num_atoms (0 = oldest, shortest context).
        sm_idx = bidz
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]
        if cutlass.const_expr(self.early_prologue):
            # Warp-resident context_lens: lane l holds ctx[min(l, B-1)], loaded
            # in parallel with the schedule metadata so every later num_kv is a
            # shuffle instead of a dependent global round trip.  Only valid when
            # the batch fits one warp; larger batches keep the load.
            ctx_lane = tidx % 32
            ctx_lane_clamped = min(ctx_lane, batch_size - 1)
            lane_ctx = mContextLens[ctx_lane_clamped]
            ctx_fits_warp = batch_size <= 32

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)
            if cutlass.const_expr(self.fused_weight_scale):
                cpasync.prefetch_descriptor(tma_atom_qscale)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        # Positions processed per kernel task = one atom (next_n_atom). The
        # logical next_n only appears in the output row stride (_atom_out_row_base).
        next_n = self.next_n_atom
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer → Math consumer (8 math warps)
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps × 1 lane-0 = 8 arrives.
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        # Both Math WGs share the same pipeline state (advance in lockstep)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # UMMA warps observe Q pipeline (wait only, no release)
        # to ensure Q is in SMEM before GEMM. Critical for UMMA warp 1
        # since TMA warp 1 only loads KV1 (not Q).
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # Merged KV+Scale pipelines (per-group, 3 stages each)
        # Like DeepGEMM: KV data and scales share one barrier.
        # TMA loads both under one barrier. Math is consumer (releases).
        # UMMA also waits on this barrier (for KV GEMM) but does NOT release.
        math_warps_per_group = self.num_math_warps // 2  # 4 warps
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, math_warps_per_group)
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )

        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        # UMMA consumer states (wait only, no release)
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        # Math consumer states (wait + release)
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        # TMEM — only Math warps (8×32=256) + UMMA warps (2×32=64) = 320 threads
        # TMA warps do NOT participate, so they can start TMA loads earlier.
        # Math warp 0 is the allocator (like fp16_gemm_3's epilogue warp 0),
        # because math warps are the last TMEM consumers (epilogue reads).
        tmem_alloc_num_threads = 320  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=tmem_alloc_num_threads)
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
        )

        if cutlass.const_expr(self.early_prologue):
            # Each TMA warp initializes exactly the barriers it produces on
            # (defer_init above skipped the built-in warp-0 init), so it can
            # issue on them before the CTA-wide sync below; the fence that
            # follows publishes the inits.  Consumers only touch a barrier
            # after pipeline_init_wait, so their ordering is unchanged.
            init_lane0 = (tidx % 32) == 0
            if is_tma_warp_0:
                if init_lane0:
                    for pipe in (
                        q_pipeline,
                        kv_pipeline_0,
                        umma_pipeline_0,
                        umma_pipeline_1,
                    ):
                        for so in (pipe.sync_object_full, pipe.sync_object_empty):
                            for st in range(so.num_stages):
                                cute.arch.mbarrier_init(so.get_barrier(st), so.arrive_count)
            if is_tma_warp_1:
                if init_lane0:
                    for so in (
                        kv_pipeline_1.sync_object_full,
                        kv_pipeline_1.sync_object_empty,
                    ):
                        for st in range(so.num_stages):
                            cute.arch.mbarrier_init(so.get_barrier(st), so.arrive_count)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB)
        # to avoid sub-bank conflicts with UMMA reading sKV from
        # sub-partition 0.
        if cutlass.const_expr(self.smem_pad_bytes > 0):
            _ = smem.allocate(self.smem_pad_bytes)
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # Fusion B: FP32 Q scale SMEM: [N, num_q_stages], shared Q barrier
        if cutlass.const_expr(self.fused_weight_scale):
            sQScale = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=qscale_smem_layout_staged,
                byte_alignment=128,
            )
        else:
            sQScale = None  # keeps the _epi_t_body argument bound on every path
        # Scales SMEM: [block_kv, num_kv_stages] float32, per group
        sScales_0 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )
        sScales_1 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): fmha_decode_paged pattern.
        # SMEM view is ((tile), num_sub_blocks, stages) — built in __call__.
        # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
        # GMEM: local_tile by (phys, head), then group first 2 modes into tile.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(
            mA_mkl,
            (self.phys_block_kv, self.head_dim),
            coord=(None, None, None),
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_for_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_for_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): shared SMEM, L dim = batch_size
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, batch_size] → [N] per stage
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )
        if cutlass.const_expr(self.fused_weight_scale):
            tQScS, tQSgQS = cpasync.tma_partition(
                tma_atom_qscale,
                0,
                w_cta_layout,
                cute.group_modes(sQScale, 0, 1),
                cute.group_modes(mQScale_tma, 0, 1),
            )

        # Partition Scales: explicit sub_blocks + stages dims.
        # Layout (phys_block_kv, num_sub, stages) K-major with custom strides.
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
        sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        # GMEM: local_tile by phys to match atom's tile size
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_0_for_tma,
            gS,
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_1_for_tma,
            gS,
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE) — dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = cutlass.tensor_utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], num_heads // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # ===== SCHEDULER: derive values from early-loaded schedule metadata =====
        # Consumers of the early schedule-metadata loads, placed at their
        # real use site. Known codegen limitation (not worked around):
        # the compiler does not keep these global loads and their
        # consumers apart — regardless of source
        # placement, the clamp+add+shift chain is hoisted right after its
        # load, exposing the two L2 round-trips instead of hiding them
        # under the independent mbarrier-init work. ptxas trades exposed
        # load latency for shorter register lifetimes and re-solves to that
        # optimum no matter the IR order; getting the sink from source needs
        # a scheduling barrier (cfence) or a latency-aware cost model fix.
        # Clamp to avoid OOB when start_q == batch_size (zero-work CTA
        # sentinel). Zero-work CTAs get a stale current_num_kv, never used
        # because has_work will be False.
        start_q_clamped = min(start_q, batch_size - 1)
        if cutlass.const_expr(self.early_prologue):
            start_ctx = cutlass.Int32(0)
            if ctx_fits_warp:
                start_ctx = cute.arch.shuffle_sync(lane_ctx, start_q_clamped)
            else:
                start_ctx = self._atom_ctx_len(start_q_clamped, mContextLens)
            current_num_kv = (start_ctx + self.block_kv - 1) // self.block_kv
            # Block-table entries of this CTA's first task, loaded by the TMA
            # warps now so the round trip overlaps the pipeline init instead of
            # serializing behind the Q TMA in the task loop (lane l holds the
            # entries of the l-th tile of its group, like the loop's cache).
            pre_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            if is_tma_warp:
                pre_lane = tidx % 32
                pre_group = cutlass.Int32(0)
                if is_tma_warp_1:
                    pre_group = cutlass.Int32(1)
                pre_kv = start_kv_half * NUM_MATH_WG + pre_group + pre_lane * NUM_MATH_WG
                if pre_kv < current_num_kv:
                    pre_base = pre_kv * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        pre_blks[i] = mBlockTable[(self._atom_seq(start_q_clamped), pre_base + i)]
        else:
            current_num_kv = (
                self._atom_ctx_len(start_q_clamped, mContextLens) + self.block_kv - 1
            ) // self.block_kv

        # ===== SCHEDULER STATE: sunk into each warp-role branch =====
        # Each warp role derives its own scheduler-state copy (end_kv_idx,
        # next_q_idx/next_kv_idx/next_num_kv, q_idx sentinel, has_work) from
        # the early-loaded schedule metadata AFTER its warpgroup_reg_alloc/
        # dealloc call. Deriving it here (before the branches) placed the
        # loop-carried values under the shared prologue register budget,
        # where the register pressure before the setmaxnreg
        # reallocation forced ptxas to spill the kv cursor + bound pair
        # to local memory; deriving per role eliminates
        # the spill. The block is pure arithmetic on already-loaded values,
        # so per-branch duplication is semantics-preserving.

        if cutlass.const_expr(self.early_prologue):
            # First copies BEFORE the CTA-wide init sync: the barriers were
            # initialized above by the same warps, and nothing in this CTA
            # consumes the stages before pipeline_init_wait.  This is what
            # moves the first DRAM round trip off the prologue's critical
            # path (first KV TMA at ~0.9 us after entry instead of ~1.3 us).
            # Never valid with pdl: these copies read predecessor data.
            early_has_work = (start_q != end_q_idx) | (
                (start_kv_half * NUM_MATH_WG) != (end_kv_half * NUM_MATH_WG)
            )
            # The first Q-stage barrier (pure pointer arithmetic on the fresh
            # producer state), computed outside the warp-0 branch so that TMA
            # warp 0 can seed its loop-carried `q_bar` from it below: the DSL
            # makes every name assigned in either arm of a const_expr if/else
            # visible to the region analysis of the following loop, so the
            # loop's conditional `q_bar = ...` must see a pointer on this path
            # too (as it does on the pdl path from the in-branch prefetch).
            early_q_bar = q_pipeline.producer_get_barrier(q_prod_state)
            if is_tma_warp_0:
                q_pipeline.producer_acquire(q_prod_state)
                q_l = self._q_tma_coord(start_q)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, q_l)],
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=early_q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, q_l)],
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=early_q_bar,
                )
                if cutlass.const_expr(self.fused_weight_scale):
                    cute.copy(
                        tma_atom_qscale,
                        tQSgQS[(None, q_l)],
                        tQScS[(None, q_prod_state.index)],
                        tma_bar_ptr=early_q_bar,
                    )
                q_prod_state.advance()
                # First KV tile of group 0 (the loop skips this issue once).
                if early_has_work:
                    early_blks_0 = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        early_blks_0[i] = cute.arch.shuffle_sync(pre_blks[i], 0)
                    kv_pipeline_0.producer_acquire(kv_prod_state_0)
                    early_bar_0 = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                    early_stage_0 = kv_prod_state_0.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_0[(None, 0, 0, early_blks_0[i])],
                            tAsA_0[(None, i, early_stage_0)],
                            tma_bar_ptr=early_bar_0,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_0[(None, 0, early_blks_0[i])],
                            tSsS_0[(None, i, early_stage_0)],
                            tma_bar_ptr=early_bar_0,
                        )
                    kv_prod_state_0.advance()
            if is_tma_warp_1:
                # First KV tile of group 1 (the loop skips this issue once).
                if early_has_work:
                    early_blks_1 = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        early_blks_1[i] = cute.arch.shuffle_sync(pre_blks[i], 0)
                    kv_pipeline_1.producer_acquire(kv_prod_state_1)
                    early_bar_1 = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                    early_stage_1 = kv_prod_state_1.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_1[(None, 0, 0, early_blks_1[i])],
                            tAsA_1[(None, i, early_stage_1)],
                            tma_bar_ptr=early_bar_1,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_1[(None, 0, early_blks_1[i])],
                            tSsS_1[(None, i, early_stage_1)],
                            tma_bar_ptr=early_bar_1,
                        )
                    kv_prod_state_1.advance()
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if cutlass.const_expr(self.pdl):
            # PDL: the two TMA warps are the only roles that read predecessor
            # data (Q / weights / Q scale from the quantize and weight-scale
            # nodes, K cache from the cache update, block table).  Everything
            # above this point is CTA-local setup plus the
            # schedule-metadata and context-length loads, which are produced
            # before the predecessor kernel starts.  Math/UMMA/epilogue roles
            # consume SMEM only after a barrier armed by these TMA copies, and
            # the epilogue writes logits that no in-flight predecessor reads.
            if is_tma_warp:
                cute.arch.griddepcontrol_wait()

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q (prefetch) + KV for group 0
            # Matches DeepGEMM's TMA warp with kv_group_idx == 0
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            lane_idx = tidx % 32

            # Block table prefetch: 32 lanes cache block indices,
            # distributed via shuffle. Each lane holds num_blocks_per_mma
            # physical block indices per compute tile. (DeepGEMM L233-244)
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use
            if cutlass.const_expr(self.early_prologue):
                # The first Q / W / Q-scale copies and this group's first KV
                # tile were issued before the CTA-wide sync; the first loop
                # iteration reuses the preloaded block-table entries and skips
                # its KV issue once (the shuffle bookkeeping still runs).
                # `q_bar` is seeded so the loop's conditional re-assignment
                # joins a pointer, exactly as after the pdl path's prefetch
                # (see the early copies above); the seed itself is never read.
                blk_preloaded = cutlass.Boolean(True)
                kv_first_skip_0 = cutlass.Boolean(True)
                q_bar = early_q_bar
            else:
                # Prefetch first Q before loop (like DeepGEMM line 203-204)
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                q_l = self._q_tma_coord(next_q_idx)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, q_l)],
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, q_l)],
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
                if cutlass.const_expr(self.fused_weight_scale):
                    cute.copy(
                        tma_atom_qscale,
                        tQSgQS[(None, q_l)],
                        tQScS[(None, q_prod_state.index)],
                        tma_bar_ptr=q_bar,
                    )
                q_prod_state.advance()

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Q prefetch: when batch changes, load Q for NEXT batch
                if q_idx != q_idx_old:
                    kv_blk_ptr = cutlass.Int32(32)  # force re-prefetch
                    prefetch_next = q_idx + 1
                    if prefetch_next < end_q_idx:
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        q_l = self._q_tma_coord(prefetch_next)
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, q_l)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, q_l)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        if cutlass.const_expr(self.fused_weight_scale):
                            cute.copy(
                                tma_atom_qscale,
                                tQSgQS[(None, q_l)],
                                tQScS[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                            )
                        q_prod_state.advance()
                    elif prefetch_next == end_q_idx:
                        if end_kv_idx > 0:
                            q_pipeline.producer_acquire(q_prod_state)
                            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                            q_l = self._q_tma_coord(prefetch_next)
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, 0, q_l)],
                                tBsB[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                                mcast_mask=b_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_w,
                                tWgW[(None, q_l)],
                                tWsW[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                            )
                            if cutlass.const_expr(self.fused_weight_scale):
                                cute.copy(
                                    tma_atom_qscale,
                                    tQSgQS[(None, q_l)],
                                    tQScS[(None, q_prod_state.index)],
                                    tma_bar_ptr=q_bar,
                                )
                            q_prod_state.advance()

                # Block table prefetch for group 0 (like DeepGEMM L233-241).
                # Each lane loads num_blocks_per_mma physical block indices
                # for one compute tile (kv_idx counts compute tiles).
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + lane_idx * NUM_MATH_WG
                    if cutlass.const_expr(self.early_prologue):
                        if blk_preloaded:
                            # First task: entries were loaded during the init.
                            blk_preloaded = cutlass.Boolean(False)
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = pre_blks[i]
                        elif prefetch_kv < num_kv:
                            base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = mBlockTable[(self._atom_seq(q_idx), base_phys + i)]
                        else:
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = cutlass.Int32(0)
                    else:
                        if prefetch_kv < num_kv:
                            base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = mBlockTable[(self._atom_seq(q_idx), base_phys + i)]
                        else:
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = cutlass.Int32(0)

                # Get block indices via shuffle before barrier (like DeepGEMM L244)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                # Load KV + Scale for group 0: num_blocks_per_mma TMAs per tile.
                if cutlass.const_expr(self.early_prologue):
                    if kv_first_skip_0:
                        # Issued before the CTA-wide sync (early prologue).
                        kv_first_skip_0 = cutlass.Boolean(False)
                    else:
                        kv_pipeline_0.producer_acquire(kv_prod_state_0)
                        bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                        stage = kv_prod_state_0.index
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cute.copy(
                                tma_atom_a,
                                tAgA_0[(None, 0, 0, phys_blks[i])],
                                tAsA_0[(None, i, stage)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_s,
                                tSgS_0[(None, 0, phys_blks[i])],
                                tSsS_0[(None, i, stage)],
                                tma_bar_ptr=bar,
                            )
                        kv_prod_state_0.advance()
                else:
                    kv_pipeline_0.producer_acquire(kv_prod_state_0)
                    bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                    stage = kv_prod_state_0.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_0[(None, 0, 0, phys_blks[i])],
                            tAsA_0[(None, i, stage)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_0[(None, 0, phys_blks[i])],
                            tSsS_0[(None, i, stage)],
                            tma_bar_ptr=bar,
                        )
                    kv_prod_state_0.advance()

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        if cutlass.const_expr(self.early_prologue):
                            next_ctx = cutlass.Int32(0)
                            if ctx_fits_warp:
                                next_ctx = cute.arch.shuffle_sync(lane_ctx, next_q_idx)
                            else:
                                next_ctx = self._atom_ctx_len(next_q_idx, mContextLens)
                            next_num_kv = (next_ctx + block_kv_val - 1) // block_kv_val
                        else:
                            next_num_kv = (
                                self._atom_ctx_len(next_q_idx, mContextLens) + block_kv_val - 1
                            ) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_tma_warp_1:
            # TMA warp 1: loads KV + Scale for group 1 only
            # Matches DeepGEMM's TMA warp with kv_group_idx == 1
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            lane_idx = tidx % 32

            # Block table prefetch for group 1
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use
            if cutlass.const_expr(self.early_prologue):
                # See TMA warp 0: first entries preloaded, first KV issue done.
                blk_preloaded = cutlass.Boolean(True)
                kv_first_skip_1 = cutlass.Boolean(True)

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # New q_idx → force block table re-prefetch
                if q_idx != q_idx_old:
                    kv_blk_ptr = cutlass.Int32(32)

                # Block table prefetch for group 1 (like DeepGEMM L233-241).
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + 1 + lane_idx * NUM_MATH_WG
                    if cutlass.const_expr(self.early_prologue):
                        if blk_preloaded:
                            # First task: entries were loaded during the init.
                            blk_preloaded = cutlass.Boolean(False)
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = pre_blks[i]
                        elif prefetch_kv < num_kv:
                            base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = mBlockTable[(self._atom_seq(q_idx), base_phys + i)]
                        else:
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = cutlass.Int32(0)
                    else:
                        if prefetch_kv < num_kv:
                            base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = mBlockTable[(self._atom_seq(q_idx), base_phys + i)]
                        else:
                            for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cached_blks[i] = cutlass.Int32(0)

                # Get block indices via shuffle before barrier (like DeepGEMM L244)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                # Load KV + Scale for group 1: num_blocks_per_mma TMAs per tile.
                if cutlass.const_expr(self.early_prologue):
                    if kv_first_skip_1:
                        # Issued before the CTA-wide sync (early prologue).
                        kv_first_skip_1 = cutlass.Boolean(False)
                    else:
                        kv_pipeline_1.producer_acquire(kv_prod_state_1)
                        bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                        stage = kv_prod_state_1.index
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cute.copy(
                                tma_atom_a,
                                tAgA_1[(None, 0, 0, phys_blks[i])],
                                tAsA_1[(None, i, stage)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_s,
                                tSgS_1[(None, 0, phys_blks[i])],
                                tSsS_1[(None, i, stage)],
                                tma_bar_ptr=bar,
                            )
                        kv_prod_state_1.advance()
                else:
                    kv_pipeline_1.producer_acquire(kv_prod_state_1)
                    bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                    stage = kv_prod_state_1.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_1[(None, 0, 0, phys_blks[i])],
                            tAsA_1[(None, i, stage)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_1[(None, 0, phys_blks[i])],
                            tSsS_1[(None, i, stage)],
                            tma_bar_ptr=bar,
                        )
                    kv_prod_state_1.advance()

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        if cutlass.const_expr(self.early_prologue):
                            next_ctx = cutlass.Int32(0)
                            if ctx_fits_warp:
                                next_ctx = cute.arch.shuffle_sync(lane_ctx, next_q_idx)
                            else:
                                next_ctx = self._atom_ctx_len(next_q_idx, mContextLens)
                            next_num_kv = (next_ctx + block_kv_val - 1) // block_kv_val
                        else:
                            next_num_kv = (
                                self._atom_ctx_len(next_q_idx, mContextLens) + block_kv_val - 1
                            ) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp_0:
            # UMMA warp for group 0
            # Must wait on Q pipeline: TMA operations with different
            # barriers are NOT visibility-ordered even within the same
            # warp. KV0 barrier arriving does not guarantee Q SMEM
            # writes are visible.
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            # TMEM: wait for math warp 0's allocation, retrieve pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)

            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Wait for Q pipeline when batch changes
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index

                    # Process KV block for group 0 (kv_idx + 0)
                    # Unconditional UMMA (like DeepGEMM): OOB iterations
                    # compute on garbage data; results written to aligned
                    # padding region in logits buffer.
                    # Wait KV first, then TMEM empty (like DeepGEMM)
                    kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                    umma_pipeline_0.producer_acquire(umma_prod_state_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma_0.index
                    tCtAcc_0 = tCtAcc_base_0[(None, None, None, umma_prod_state_0.index)]
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_0,
                            tCrA_0[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_0,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # No consumer_release here — Math WG 0 releases
                    kv_cons_state_umma_0.advance()

                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    # Advance: inline fetch_next_task
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            if cutlass.const_expr(self.early_prologue):
                                next_ctx = cutlass.Int32(0)
                                if ctx_fits_warp:
                                    next_ctx = cute.arch.shuffle_sync(lane_ctx, next_q_idx)
                                else:
                                    next_ctx = self._atom_ctx_len(next_q_idx, mContextLens)
                                next_num_kv = (next_ctx + block_kv_val - 1) // block_kv_val
                            else:
                                next_num_kv = (
                                    self._atom_ctx_len(next_q_idx, mContextLens) + block_kv_val - 1
                                ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp_1:
            # UMMA warp for group 1
            # Explicitly waits on Q pipeline — critical because TMA warp 1
            # only loads KV1, not Q. Without this wait, UMMA warp 1 can
            # start GEMM before TMA warp 0 finishes loading Q into SMEM.
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            # TMEM: wait for umma_warp_0's allocation, retrieve pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)

            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Wait for Q pipeline when batch changes
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional UMMA (like DeepGEMM)
                    # Wait KV first, then TMEM empty (like DeepGEMM)
                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index
                    tCtAcc_1 = tCtAcc_base_1[(None, None, None, umma_prod_state_1.index)]
                    for k_block in cutlass.range_constexpr(num_k_blocks_1):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_1],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # No consumer_release here — Math WG 1 releases
                    kv_cons_state_umma_1.advance()

                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                    # Advance: inline fetch_next_task
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            if cutlass.const_expr(self.early_prologue):
                                next_ctx = cutlass.Int32(0)
                                if ctx_fits_warp:
                                    next_ctx = cute.arch.shuffle_sync(lane_ctx, next_q_idx)
                                else:
                                    next_ctx = self._atom_ctx_len(next_q_idx, mContextLens)
                                next_num_kv = (next_ctx + block_kv_val - 1) // block_kv_val
                            else:
                                next_num_kv = (
                                    self._atom_ctx_len(next_q_idx, mContextLens) + block_kv_val - 1
                                ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(240)

            # Per-role scheduler state (see comment above the branch chain).
            # Derived AFTER warpgroup_reg_alloc(240) so the loop-carried
            # scheduler words live under the 240-register math budget.
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
            tmem.allocate(num_tmem_alloc_cols_total)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)

            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)

            if warpgroup_idx == 0:
                # Math WG 0: process group 0
                # Reference setup (stage 0) for m_coord
                # flat_divide by sub-tile to get sub-tile partitions
                tAcc_0_ref = tCtAcc_base_0[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_0_ref_epi = cute.flat_divide(tAcc_0_ref, epi_sub_mn)
                tiled_copy_ref_0 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_0_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_0 = tiled_copy_ref_0.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_0.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # Weight register cache: only the first NUM_W_IN_REG heads
                # per next_n slot live in registers (like DeepGEMM
                # min(52, kNumHeads)); the rest are read from SMEM in the
                # epilogue. Budget rationale: see _max_num_w_in_reg.
                NUM_W_IN_REG = min(_max_num_w_in_reg(self.epi_dtype, next_n, num_heads), num_heads)
                w_cache = cute.make_rmem_tensor((NUM_W_IN_REG * next_n,), self.epi_dtype)
                q_stage_local = cutlass.Int32(0)

                # Flat logits view + carried row offset: the store address
                # is base + out_row*stride0 + kv_pos, but out_row only
                # changes on q-change. Recomputing the row term there (cold
                # path) leaves the hot path a single add — the wide-multiply
                # address chain otherwise re-materializes every task because
                # the while-loop control flow hides the invariance.
                # Gated off on Rubin: there the backend schedules the
                # UMMA consumer-release late in the math loop, throttling the
                # MMA pipe per KV tile — regresses, growing with ctx.
                # The 2D alias below keeps `mLogits` itself out of the gated
                # store code: referencing a defined tensor from the untraced
                # arm changes the loop-region captures (and the SASS schedule
                # with it), while an undefined name is dropped harmlessly.
                if cutlass.const_expr(self.use_flat_logits_view):
                    logits_stride0 = cutlass.Int64(mLogits.layout.stride[0])
                    # Sized by the pitched span (rows * stride0), not the
                    # visible width: stores also cover the aligned padding
                    # past max_context_len (host allocates the full pitch).
                    mLogits_flat = cute.make_tensor(
                        mLogits.iterator,
                        cute.make_layout(cute.size(mLogits, mode=[0]) * logits_stride0),
                    )
                    out_row_off = cutlass.Int64(0)
                    mLogits_2d = None  # Nones keep _epi_t_body's arguments bound
                else:
                    mLogits_2d = mLogits
                    mLogits_flat = out_row_off = logits_stride0 = None

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Q pipeline consumer: wait for Q+Weights SMEM
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        if cutlass.const_expr(self.use_flat_logits_view):
                            out_row_off = (
                                cutlass.Int64(self._atom_out_row_base(q_idx)) * logits_stride0
                            )
                        # Preload first NUM_W_IN_REG weights per slot
                        for t_i in cutlass.range_constexpr(next_n):
                            for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                                if cutlass.const_expr(self.fused_weight_scale):
                                    # Fusion B: I8's exact left-associated FP32
                                    # chain, once per q change per thread.
                                    w_i = t_i * num_heads + w_j
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = (
                                        sW[(w_i, q_stage_local)] * sQScale[(w_i, q_stage_local)]
                                    ) * weight_scale_factor
                                else:
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                        (t_i * num_heads + w_j, q_stage_local)
                                    ]

                    # ---- inner: hot kv loop within this q ----
                    # Two-level restructure: the two cold blocks (q-change weight
                    # preload above, q-rollover ctx reload below) execute rarely
                    # but sat inline in the single flat loop, costing taken-branch
                    # fetch bubbles on every hot iteration. The inner loop below
                    # replays exactly the original task sequence:
                    #   original: process (q, kv); next_kv = kv + W;
                    #             roll over to (q+1, 0) if next_kv >= num_kv;
                    #             stop when (next_q, next_kv) == (end_q, end_kv).
                    # Truth table for kv_go after processing (q, kv) and advancing
                    # kv += W (W = NUM_MATH_WG):
                    #   kv <  num_kv, (q, kv) != end -> True  (more kv in this q)
                    #   kv <  num_kv, (q, kv) == end -> False (global end mid-q)
                    #   kv >= num_kv, any            -> False (q exhausted; outer
                    #                                          rollover runs next)
                    # First iteration is unconditional, matching the original: the
                    # outer while is only entered when the committed (q, kv) != end.
                    # Gate (use_two_level_task_loop=False, Rubin nn=4 auto):
                    # forcing kv_go False after the body degenerates the inner
                    # loop to a single trip, restoring the original flat loop
                    # (per-task q-change check) without duplicating the skeleton.
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        # Process KV block for group 0 (kv_idx + 0)
                        # Unconditional Math: OOB results
                        # written to aligned padding region in logits buffer.
                        kv_pos = kv_idx * block_kv_val + m_coord

                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            # Skip KV wait, rely on UMMA barrier's
                            # transitive visibility.
                            umma_pipeline_0.consumer_wait(umma_cons_state_0)
                        elif cutlass.const_expr(self.use_paired_pipeline_polls):
                            # Issue both barrier polls back-to-back so their
                            # mbarrier round-trips overlap instead of running
                            # serially; each blocking wait below short-circuits
                            # when its poll already saw the barrier flipped.
                            kv_ready_0 = kv_pipeline_0.consumer_try_wait(kv_cons_state_math_0)
                            umma_ready_0 = umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                            kv_pipeline_0.consumer_wait(kv_cons_state_math_0, kv_ready_0)
                            sc_stage_0 = kv_cons_state_math_0.index
                            scale_val = sScales_0[(m_coord, sc_stage_0)]
                            umma_pipeline_0.consumer_wait(umma_cons_state_0, umma_ready_0)
                        else:
                            # Gated (Rubin nn=4): plain waits, KV first to
                            # overlap the scale SMEM load with UMMA computation.
                            kv_pipeline_0.consumer_wait(kv_cons_state_math_0)
                            sc_stage_0 = kv_cons_state_math_0.index
                            scale_val = sScales_0[(m_coord, sc_stage_0)]
                            umma_pipeline_0.consumer_wait(umma_cons_state_0)

                        # --- TMEM sub-tile setup ---
                        # flat_divide accumulator by sub-tile shape;
                        # partition once, then loop over sub-tiles.
                        tCtAcc_c0 = tCtAcc_base_0[(None, None, None, umma_cons_state_0.index)]
                        tAcc_c0 = tCtAcc_c0[((None, None), 0, 0)]
                        tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                        # Every UMMA stage shares the staged-fragment layout and
                        # differs only in TMEM base address, so the pre-loop
                        # tiled copy / thread slice partition every stage;
                        # rebuilding them per task re-derives loop-invariant
                        # copy descriptors inside the hot loop.
                        tTR_0 = thr_copy_ref_0.partition_S(tAcc_c0_epi)

                        # --- First sub-tile LDTM + KV release ---
                        if cutlass.const_expr(self.early_tmem_copy):
                            # Issue first sub-tile LDTM early
                            cute.copy(
                                tiled_copy_ref_0,
                                tTR_0[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            # Scale LDS + KV release fill latency
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_0 = kv_cons_state_math_0.index
                                scale_val = sScales_0[(m_coord, sc_stage_0)]
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                            cute.arch.fence_view_async_tmem_load()
                        else:
                            # Default: scale LDS + KV release first
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_0 = kv_cons_state_math_0.index
                                scale_val = sScales_0[(m_coord, sc_stage_0)]
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                            cute.copy(
                                tiled_copy_ref_0,
                                tTR_0[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            cute.arch.fence_view_async_tmem_load()

                        # --- Sub-tile compute loop ---
                        # Each sub-tile: LDTM.xN → fence → load →
                        # ReLU+FMA. Breaks FMA chain (16→4 per chunk)
                        # and interleaves LDTM with FP32 compute to
                        # reduce ShadowPipeThrottle.
                        # Sub-tiles are within each t-slot (num_heads
                        # // num_epi_subtiles wide). flat_divide yields
                        # next_n * num_epi_subtiles sub-tiles total;
                        # global index = t * num_epi_subtiles + i.
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            packed_zero = cute.full((4,), 0.0, dtype=cutlass.Float16)
                        else:
                            packed_zero = None

                        ops = (
                            tTR_0,
                            tiled_copy_ref_0,
                            tTR_rAcc,
                            w_cache,
                            sW,
                            sQScale,
                            q_stage_local,
                            NUM_W_IN_REG,
                            umma_pipeline_0,
                            umma_cons_state_0,
                            scale_val,
                            kv_pos,
                            out_row_off,
                            logits_stride0,
                            mLogits_flat,
                            mLogits_2d,
                            q_idx,
                            weight_scale_factor,
                            packed_zero,
                        )
                        if cutlass.const_expr(not self.uneven):
                            self._epi_tokens(next_n, True, *ops)
                        elif cutlass.const_expr(self.atom_rem == 0):
                            # Tile wider than the atom (no remainder): every
                            # atom holds atom_base valid tokens, a compile-time
                            # count, so the UMMA state advances inside like the
                            # exact path (the rem>0 path advances after its
                            # runtime branch instead).
                            self._epi_tokens(self.atom_base, True, *ops)
                        else:
                            # Uneven split: dispatch this atom's valid-token count
                            # (base+1 for the first atom_rem atoms of a sequence,
                            # else base) to a compile-time loop bound; one
                            # well-predicted branch per KV tile. UMMA state
                            # advances once here (see _epi_t_body).
                            if q_idx % self.num_next_n_atoms < self.atom_rem:
                                self._epi_tokens(self.atom_base + 1, False, *ops)
                            else:
                                self._epi_tokens(self.atom_base, False, *ops)
                            umma_cons_state_0.advance()
                        # Advance within this q
                        kv_idx = kv_idx + NUM_MATH_WG
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                        if cutlass.const_expr(not self.use_two_level_task_loop):
                            kv_go = cutlass.Boolean(False)

                    # ---- outer: rollover ----
                    if kv_idx >= num_kv:
                        # Inner loop exhausted this q's kv range: advance to the
                        # next q (same as the original rollover path).
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            if cutlass.const_expr(self.early_prologue):
                                next_ctx = cutlass.Int32(0)
                                if ctx_fits_warp:
                                    next_ctx = cute.arch.shuffle_sync(lane_ctx, next_q_idx)
                                else:
                                    next_ctx = self._atom_ctx_len(next_q_idx, mContextLens)
                                next_num_kv = (next_ctx + block_kv_val - 1) // block_kv_val
                            else:
                                next_num_kv = (
                                    self._atom_ctx_len(next_q_idx, mContextLens) + block_kv_val - 1
                                ) // block_kv_val
                    else:
                        # Global end hit mid-q: next == (q_idx, kv_idx) == end,
                        # so has_work goes False below. kv_idx was already
                        # advanced past the last processed task in the inner
                        # loop, so this matches the original's next_kv_idx.
                        next_q_idx = q_idx
                        next_kv_idx = kv_idx
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

                # Release last Q stage (WG 0)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            else:
                # Math WG 1: process group 1
                tAcc_1_ref = tCtAcc_base_1[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_1_ref_epi = cute.flat_divide(tAcc_1_ref, epi_sub_mn)
                tiled_copy_ref_1 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_1_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_1 = tiled_copy_ref_1.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_1.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # Weight register cache, same budget as the WG0 site above
                # (see _max_num_w_in_reg).
                NUM_W_IN_REG = min(_max_num_w_in_reg(self.epi_dtype, next_n, num_heads), num_heads)
                w_cache = cute.make_rmem_tensor((NUM_W_IN_REG * next_n,), self.epi_dtype)
                q_stage_local = cutlass.Int32(0)

                # Flat logits view + carried row offset (see Math WG 0).
                # Gated off on Rubin: there the backend schedules the
                # UMMA consumer-release late in the math loop, throttling the
                # MMA pipe per KV tile — regresses, growing with ctx.
                # The 2D alias below keeps `mLogits` itself out of the gated
                # store code: referencing a defined tensor from the untraced
                # arm changes the loop-region captures (and the SASS schedule
                # with it), while an undefined name is dropped harmlessly.
                if cutlass.const_expr(self.use_flat_logits_view):
                    logits_stride0 = cutlass.Int64(mLogits.layout.stride[0])
                    # Sized by the pitched span (rows * stride0), not the
                    # visible width: stores also cover the aligned padding
                    # past max_context_len (host allocates the full pitch).
                    mLogits_flat = cute.make_tensor(
                        mLogits.iterator,
                        cute.make_layout(cute.size(mLogits, mode=[0]) * logits_stride0),
                    )
                    out_row_off = cutlass.Int64(0)
                    mLogits_2d = None  # Nones keep _epi_t_body's arguments bound
                else:
                    mLogits_2d = mLogits
                    mLogits_flat = out_row_off = logits_stride0 = None

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Q pipeline consumer: wait for Q+Weights SMEM
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        if cutlass.const_expr(self.use_flat_logits_view):
                            out_row_off = (
                                cutlass.Int64(self._atom_out_row_base(q_idx)) * logits_stride0
                            )
                        # Preload first NUM_W_IN_REG weights per slot
                        for t_i in cutlass.range_constexpr(next_n):
                            for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                                if cutlass.const_expr(self.fused_weight_scale):
                                    # Fusion B: I8's exact left-associated FP32
                                    # chain, once per q change per thread.
                                    w_i = t_i * num_heads + w_j
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = (
                                        sW[(w_i, q_stage_local)] * sQScale[(w_i, q_stage_local)]
                                    ) * weight_scale_factor
                                else:
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                        (t_i * num_heads + w_j, q_stage_local)
                                    ]

                    # ---- inner: hot kv loop within this q ----
                    # Two-level restructure: the two cold blocks (q-change weight
                    # preload above, q-rollover ctx reload below) execute rarely
                    # but sat inline in the single flat loop, costing taken-branch
                    # fetch bubbles on every hot iteration. The inner loop below
                    # replays exactly the original task sequence:
                    #   original: process (q, kv); next_kv = kv + W;
                    #             roll over to (q+1, 0) if next_kv >= num_kv;
                    #             stop when (next_q, next_kv) == (end_q, end_kv).
                    # Truth table for kv_go after processing (q, kv) and advancing
                    # kv += W (W = NUM_MATH_WG):
                    #   kv <  num_kv, (q, kv) != end -> True  (more kv in this q)
                    #   kv <  num_kv, (q, kv) == end -> False (global end mid-q)
                    #   kv >= num_kv, any            -> False (q exhausted; outer
                    #                                          rollover runs next)
                    # First iteration is unconditional, matching the original: the
                    # outer while is only entered when the committed (q, kv) != end.
                    # Gate (use_two_level_task_loop=False, Rubin nn=4 auto):
                    # forcing kv_go False after the body degenerates the inner
                    # loop to a single trip, restoring the original flat loop
                    # (per-task q-change check) without duplicating the skeleton.
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        # Process KV block for group 1 (kv_idx + 1)
                        # Unconditional Math
                        kv_idx_1 = kv_idx + 1

                        kv_pos = kv_idx_1 * block_kv_val + m_coord

                        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                            umma_pipeline_1.consumer_wait(umma_cons_state_1)
                        elif cutlass.const_expr(self.use_paired_pipeline_polls):
                            # Overlapped polls (see Math WG 0 for rationale).
                            kv_ready_1 = kv_pipeline_1.consumer_try_wait(kv_cons_state_math_1)
                            umma_ready_1 = umma_pipeline_1.consumer_try_wait(umma_cons_state_1)
                            kv_pipeline_1.consumer_wait(kv_cons_state_math_1, kv_ready_1)
                            sc_stage_1 = kv_cons_state_math_1.index
                            scale_val = sScales_1[(m_coord, sc_stage_1)]
                            umma_pipeline_1.consumer_wait(umma_cons_state_1, umma_ready_1)
                        else:
                            # Gated (Rubin nn=4): plain waits, KV first to
                            # overlap the scale SMEM load with UMMA computation.
                            kv_pipeline_1.consumer_wait(kv_cons_state_math_1)
                            sc_stage_1 = kv_cons_state_math_1.index
                            scale_val = sScales_1[(m_coord, sc_stage_1)]
                            umma_pipeline_1.consumer_wait(umma_cons_state_1)

                        # --- TMEM sub-tile setup (WG1) ---
                        tCtAcc_c1 = tCtAcc_base_1[(None, None, None, umma_cons_state_1.index)]
                        tAcc_c1 = tCtAcc_c1[((None, None), 0, 0)]
                        tAcc_c1_epi = cute.flat_divide(tAcc_c1, epi_sub_mn)
                        # Loop-invariant copy descriptors (see Math WG 0).
                        tTR_1 = thr_copy_ref_1.partition_S(tAcc_c1_epi)

                        # --- First sub-tile LDTM + KV release (WG1) ---
                        if cutlass.const_expr(self.early_tmem_copy):
                            cute.copy(
                                tiled_copy_ref_1,
                                tTR_1[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_1 = kv_cons_state_math_1.index
                                scale_val = sScales_1[(m_coord, sc_stage_1)]
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                            cute.arch.fence_view_async_tmem_load()
                        else:
                            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                                sc_stage_1 = kv_cons_state_math_1.index
                                scale_val = sScales_1[(m_coord, sc_stage_1)]
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                            cute.copy(
                                tiled_copy_ref_1,
                                tTR_1[(None, None, None, 0, 0)],
                                tTR_rAcc,
                            )
                            cute.arch.fence_view_async_tmem_load()

                        # --- Sub-tile compute loop (WG1) ---
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            packed_zero = cute.full((4,), 0.0, dtype=cutlass.Float16)
                        else:
                            packed_zero = None

                        ops = (
                            tTR_1,
                            tiled_copy_ref_1,
                            tTR_rAcc,
                            w_cache,
                            sW,
                            sQScale,
                            q_stage_local,
                            NUM_W_IN_REG,
                            umma_pipeline_1,
                            umma_cons_state_1,
                            scale_val,
                            kv_pos,
                            out_row_off,
                            logits_stride0,
                            mLogits_flat,
                            mLogits_2d,
                            q_idx,
                            weight_scale_factor,
                            packed_zero,
                        )
                        if cutlass.const_expr(not self.uneven):
                            self._epi_tokens(next_n, True, *ops)
                        elif cutlass.const_expr(self.atom_rem == 0):
                            # Tile wider than the atom (no remainder): every
                            # atom holds atom_base valid tokens, a compile-time
                            # count, so the UMMA state advances inside like the
                            # exact path (the rem>0 path advances after its
                            # runtime branch instead).
                            self._epi_tokens(self.atom_base, True, *ops)
                        else:
                            # Uneven split: dispatch this atom's valid-token count
                            # (base+1 for the first atom_rem atoms of a sequence,
                            # else base) to a compile-time loop bound; one
                            # well-predicted branch per KV tile. UMMA state
                            # advances once here (see _epi_t_body).
                            if q_idx % self.num_next_n_atoms < self.atom_rem:
                                self._epi_tokens(self.atom_base + 1, False, *ops)
                            else:
                                self._epi_tokens(self.atom_base, False, *ops)
                            umma_cons_state_1.advance()
                        # Advance within this q
                        kv_idx = kv_idx + NUM_MATH_WG
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                        if cutlass.const_expr(not self.use_two_level_task_loop):
                            kv_go = cutlass.Boolean(False)

                    # ---- outer: rollover ----
                    if kv_idx >= num_kv:
                        # Inner loop exhausted this q's kv range: advance to the
                        # next q (same as the original rollover path).
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            if cutlass.const_expr(self.early_prologue):
                                next_ctx = cutlass.Int32(0)
                                if ctx_fits_warp:
                                    next_ctx = cute.arch.shuffle_sync(lane_ctx, next_q_idx)
                                else:
                                    next_ctx = self._atom_ctx_len(next_q_idx, mContextLens)
                                next_num_kv = (next_ctx + block_kv_val - 1) // block_kv_val
                            else:
                                next_num_kv = (
                                    self._atom_ctx_len(next_q_idx, mContextLens) + block_kv_val - 1
                                ) // block_kv_val
                    else:
                        # Global end hit mid-q: next == (q_idx, kv_idx) == end,
                        # so has_work goes False below. kv_idx was already
                        # advanced past the last processed task in the inner
                        # loop, so this matches the original's next_kv_idx.
                        next_q_idx = q_idx
                        next_kv_idx = kv_idx
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

                # Release last Q stage (WG 1)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            # TMEM dealloc: math warps are allocator + last consumer
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

            if cutlass.const_expr(self.pdl):
                # PDL: the math warps own every logits store; they reach this
                # point only after their last store and the TMEM release.  The
                # TMA/UMMA warps exit instead of signalling, so the CTA-level
                # trigger fires from here.  A dependent grid still has to
                # execute griddepcontrol.wait before reading logits.
                cute.arch.griddepcontrol_launch_dependents()

        else:
            cute.arch.warpgroup_reg_dealloc(24)


class FP8MQALogitsFusedWeightScaleKernel(FP8MQALogitsKernel):
    """Fusion B ABI: raw FP32 weights + FP32 Q scale + runtime scale factor.

    Only the launch signature differs from :class:`FP8MQALogitsKernel`; the
    kernel body is shared and selected by ``fused_weight_scale=True``.
    """

    def __init__(self, *args, **kwargs):
        kwargs["fused_weight_scale"] = True
        super().__init__(*args, **kwargs)

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] FP8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size]
        raw_weights: cute.Tensor,  # FP32 [N, batch_size] (transposed for TMA)
        q_scales: cute.Tensor,  # FP32 [N, batch_size] (transposed for TMA)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        weight_scale_factor: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._launch_body(
            kv_fused,
            b,
            raw_weights,
            q_scales,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            num_phys_blocks,
            batch_size,
            weight_scale_factor,
            stream,
        )


# ---- Standalone correctness driver -----------------------------------------
# ---- Constants --------------------------------------------------------------


# Element-wise tolerance for each dtype that can affect compute or storage.
