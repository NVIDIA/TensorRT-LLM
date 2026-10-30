# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Small-M MoE router GEMM for the GLM pre-MoE hop (the router contract).

``router_gemm(hidden, weight)`` returns ``hidden @ weight.T`` as FP32 logits
``[R, E]`` for BF16 ``hidden [R, K]`` and BF16 ``weight [E, K]``.

Every logit is computed with exactly the floating-point operations of the
native ``trtllm::dsv3_router_gemm_op`` small-M kernel, in the same order, so
the result is bitwise identical to it:

* the 128 threads that own one expert split K the native way: thread ``t``
  owns elements ``k = i * 1024 + t * 8 + j`` (``i < K / 1024``, ``j < 8``) and
  accumulates their BF16 x BF16 products into an FP32 register in ``(i, j)``
  order, one FMA each (the products are exact in FP32);
* the per-thread partials are combined by the XOR butterfly 16, 8, 4, 2, 1
  inside each warp;
* the four warp sums are added in warp order onto 0.0.

What differs is the data movement and the schedule:

* Operands are staged in shared memory with ``cp.async``. The expert-weight
  slice does not depend on the preceding kernel, so its copies are issued
  before the programmatic grid dependency wait; under PDL the weight fetch
  overlaps the tail of the all-reduce/RMSNorm producer. (Register loads issued
  before ``griddepcontrol.wait`` are not kept there by ptxas, which schedules
  the wait first; asynchronous copies have no register results to sink.)
* With one expert per CTA (the default) every thread stages exactly the
  activation vectors it consumes, one ``cp.async`` group per K step, and
  computes step ``i`` while later steps are still landing. With several
  experts per CTA the CTA stages the activation block once for all of them.
* Dependents are released as soon as the activation copies are issued; they
  still wait for this grid to complete before reading the logits.

The CuTe DSL is imported lazily so that this module can be imported (and its
eligibility checks used) where the DSL is absent.
"""

import os
import threading
from typing import Dict, Optional, Tuple

import torch

# Geometry of the native kernel whose arithmetic is reproduced.
THREADS_PER_EXPERT = 128
ELEMS_PER_VECTOR = 8  # one 16-byte vector of BF16
K_PER_ITERATION = THREADS_PER_EXPERT * ELEMS_PER_VECTOR  # 1024
WARPS_PER_EXPERT = THREADS_PER_EXPERT // 32

NUM_EXPERTS = 256
SUPPORTED_HIDDEN_SIZES = (4096, 6144, 7168)
MAX_ROWS = 16

# Dynamic shared memory available to one CTA on SM100-family GPUs.
_MAX_SMEM_BYTES = 227 * 1024
_SMEM_RESERVE_BYTES = 4 * 1024  # warp-sum scratch and alignment slack

# Launch shape of this target: one expert per CTA, single-element FMAs.
DEFAULT_EXPERTS_PER_CTA = 1
DEFAULT_PAIRED_FMA = False


def pdl_enabled() -> bool:
    """Whether kernels launch with programmatic dependent launch."""
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def router_gemm_supported(hidden: torch.Tensor, weight: torch.Tensor) -> Tuple[bool, str]:
    """Return ``(eligible, reason)`` for ``router_gemm(hidden, weight)``.

    Checks only tensor metadata, never device data, so it is safe inside CUDA
    graph capture and launches nothing.
    """
    if not (hidden.is_cuda and weight.is_cuda):
        return False, "tensors not on CUDA"
    if hidden.device != weight.device:
        return False, "device mismatch"
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False, f"dtype {hidden.dtype}/{weight.dtype} is not bf16/bf16"
    if hidden.dim() != 2 or weight.dim() != 2:
        return False, "expects 2-D hidden and weight"
    rows, hidden_size = hidden.shape
    if not 1 <= rows <= MAX_ROWS:
        return False, f"rows={rows} outside [1, {MAX_ROWS}]"
    if tuple(weight.shape) != (NUM_EXPERTS, hidden_size):
        return False, (f"weight shape {tuple(weight.shape)} is not [{NUM_EXPERTS}, {hidden_size}]")
    if hidden_size not in SUPPORTED_HIDDEN_SIZES:
        return False, f"hidden size {hidden_size} unsupported"
    if not (hidden.is_contiguous() and weight.is_contiguous()):
        return False, "non-contiguous operand"
    if hidden.data_ptr() % 16 or weight.data_ptr() % 16:
        return False, "operand not 16-byte aligned"
    if _smem_bytes(rows, hidden_size, 1) > _MAX_SMEM_BYTES:
        return False, f"[{rows}, {hidden_size}] activation exceeds shared memory"
    major, minor = torch.cuda.get_device_capability(hidden.device)
    if not 100 <= major * 10 + minor < 110:
        return False, f"sm{major}{minor} outside the validated SM100 family"
    return True, ""


def _smem_bytes(num_rows: int, hidden_size: int, experts_per_cta: int) -> int:
    """Shared memory of one CTA: activation block, weight slices, scratch."""
    return (num_rows + experts_per_cta) * hidden_size * 2 + _SMEM_RESERVE_BYTES


def fit_experts_per_cta(num_rows: int, hidden_size: int, requested: int) -> int:
    """Largest power-of-two experts-per-CTA <= requested whose staging fits."""
    epc = requested
    while epc > 1 and _smem_bytes(num_rows, hidden_size, epc) > _MAX_SMEM_BYTES:
        epc //= 2
    return epc


_compile_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _build_kernel_class():
    """Define the CuTe DSL kernel; imported only on first compile."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    def _asm(res, operands, text, constraints, side_effects, loc, ip):
        return llvm.inline_asm(
            res,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def griddep_wait(*, loc=None, ip=None) -> None:
        _asm(None, [], "griddepcontrol.wait;", "~{memory}", True, loc, ip)

    @dsl_user_op
    def griddep_launch_dependents(*, loc=None, ip=None) -> None:
        _asm(None, [], "griddepcontrol.launch_dependents;", "", True, loc, ip)

    @dsl_user_op
    def cp_async_16(smem_addr, gmem_addr, *, loc=None, ip=None) -> None:
        # .cg: through L2 only; every vector is staged once per CTA.
        _asm(
            None,
            [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
            "cp.async.cg.shared.global [$0], [$1], 16;",
            "r,l,~{memory}",
            True,
            loc,
            ip,
        )

    @dsl_user_op
    def cp_async_commit(*, loc=None, ip=None) -> None:
        _asm(None, [], "cp.async.commit_group;", "~{memory}", True, loc, ip)

    @dsl_user_op
    def cp_async_wait_all(*, loc=None, ip=None) -> None:
        _asm(None, [], "cp.async.wait_group 0;", "~{memory}", True, loc, ip)

    @dsl_user_op
    def lds_v4(smem_addr, *, loc=None, ip=None):
        ret = _asm(
            llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,r,~{memory}",
            True,
            loc,
            ip,
        )
        return tuple(Int32(llvm.extractvalue(T.i32(), ret, [i], loc=loc, ip=ip)) for i in range(4))

    @dsl_user_op
    def bf16x2_to_f32x2(word, *, loc=None, ip=None):
        # Exact widening of the low and high BF16 halves of one 32-bit word.
        ret = _asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, -65536;",
            "=f,=f,r",
            False,
            loc,
            ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), ret, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), ret, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def fma_rn(a, b, c, *, loc=None, ip=None):
        return Float32(
            _asm(
                T.f32(),
                [
                    Float32(a).ir_value(loc=loc, ip=ip),
                    Float32(b).ir_value(loc=loc, ip=ip),
                    Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def add_rn(a, b, *, loc=None, ip=None):
        return Float32(
            _asm(
                T.f32(),
                [
                    Float32(a).ir_value(loc=loc, ip=ip),
                    Float32(b).ir_value(loc=loc, ip=ip),
                ],
                "add.rn.f32 $0, $1, $2;",
                "=f,f,f",
                False,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def fma_rn_x2(a0, a1, b, c0, c1, *, loc=None, ip=None):
        # Two independent IEEE FMAs (one per lane): (a0*b+c0, a1*b+c1).
        ret = _asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [
                Float32(a0).ir_value(loc=loc, ip=ip),
                Float32(a1).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c0).ir_value(loc=loc, ip=ip),
                Float32(c1).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b64 fa, fb, fc;\n\tmov.b64 fa, {$2, $3};\n\t"
            "mov.b64 fb, {$4, $4};\n\tmov.b64 fc, {$5, $6};\n\t"
            "fma.rn.f32x2 fc, fa, fb, fc;\n\tmov.b64 {$0, $1}, fc;\n\t}",
            "=f,=f,f,f,f,f,f",
            False,
            loc,
            ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), ret, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), ret, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def cp_async_wait(pending: int, *, loc=None, ip=None) -> None:
        _asm(None, [], f"cp.async.wait_group {pending};", "~{memory}", True, loc, ip)

    # Plain Python helpers, inlined while the kernel is traced.
    def widen(words):
        vals = []
        for w in range(4):
            lo, hi = bf16x2_to_f32x2(words[w])
            vals.append(lo)
            vals.append(hi)
        return vals

    def accumulate(acc, a_words, b_words, paired):
        """acc[m] += a[m, j] * b[j] for j = 0..7 in order, one FMA each."""
        num_rows = len(acc)
        b_vals = widen(b_words)
        a_vals = [widen(words) for words in a_words]
        if paired:
            for m in range(0, num_rows - 1, 2):
                for j in range(ELEMS_PER_VECTOR):
                    acc[m], acc[m + 1] = fma_rn_x2(
                        a_vals[m][j], a_vals[m + 1][j], b_vals[j], acc[m], acc[m + 1]
                    )
            if num_rows % 2:
                for j in range(ELEMS_PER_VECTOR):
                    acc[-1] = fma_rn(a_vals[-1][j], b_vals[j], acc[-1])
        else:
            for m in range(num_rows):
                for j in range(ELEMS_PER_VECTOR):
                    acc[m] = fma_rn(a_vals[m][j], b_vals[j], acc[m])
        return acc

    class RouterGemmKernel:
        """``out[m, e] = sum_k a[m, k] * b[e, k]`` with native arithmetic.

        ``experts_per_cta == 1``: every thread stages exactly the activation
        and weight vectors it consumes, one ``cp.async`` group per K step, so
        K step ``i`` is computed while later steps are still landing and no
        CTA barrier is needed before the final reduction. ``> 1``: the CTA
        stages the activation block once for all its experts.
        """

        def __init__(
            self,
            num_rows: int,
            hidden_size: int,
            num_experts: int,
            experts_per_cta: int,
            use_pdl: bool,
            prefetch_weights: bool,
            paired_fma: bool,
            attribution: str,
        ):
            assert hidden_size % K_PER_ITERATION == 0
            assert num_experts % experts_per_cta == 0
            assert attribution in ("", "no_activation", "no_compute")
            self.num_rows = num_rows
            self.hidden_size = hidden_size
            self.num_experts = num_experts
            self.experts_per_cta = experts_per_cta
            self.use_pdl = use_pdl
            self.prefetch_weights = prefetch_weights
            self.paired_fma = paired_fma
            self.attribution = attribution
            self.k_iterations = hidden_size // K_PER_ITERATION
            self.threads = THREADS_PER_EXPERT * experts_per_cta
            self.a_vectors = num_rows * hidden_size // ELEMS_PER_VECTOR

        @cute.jit
        def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: CUstream):
            self.kernel(mA, mB, mC).launch(
                grid=(self.num_experts // self.experts_per_cta, 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.kernel
        def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            M = self.num_rows
            K = self.hidden_size
            KI = self.k_iterations
            EPC = self.experts_per_cta
            NT = self.threads
            row_bytes = K * 2
            iter_bytes = K_PER_ITERATION * 2
            private = EPC == 1
            stage_a = self.attribution != "no_activation"
            compute = self.attribution != "no_compute"

            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            slot = tidx // THREADS_PER_EXPERT
            t = tidx % THREADS_PER_EXPERT
            warp = t // 32
            lane = tidx % 32
            expert = bidx * EPC + slot

            smem = SmemAllocator()
            s_a = smem.allocate_tensor(BFloat16, cute.make_layout(M * K), byte_alignment=128)
            s_b = smem.allocate_tensor(BFloat16, cute.make_layout(EPC * K), byte_alignment=128)
            warp_sums = smem.allocate_tensor(Float32, cute.make_layout((WARPS_PER_EXPERT, M, EPC)))

            s_a_base = Int32(s_a.iterator.toint())
            s_b_base = Int32(s_b.iterator.toint())
            g_a_base = Int64(mA.iterator.toint())
            # This thread's weight vectors: expert `expert`, k = i*1024+t*8.
            b_off = slot * row_bytes + t * 16
            g_b_mine = (
                Int64(mB.iterator.toint()) + Int64(bidx * EPC) * Int64(row_bytes) + Int64(b_off)
            )
            s_b_mine = s_b_base + b_off
            g_a_mine = g_a_base + Int64(t * 16)
            # Shared-memory address of activation vector (i, m) read by `t`.
            # Private staging packs a thread's vectors K-step-major; shared
            # staging keeps the row-major [M, K] image of the activation.
            if cutlass.const_expr(private):
                s_a_mine = s_a_base + t * 16
                s_a_step = M * THREADS_PER_EXPERT * 16
                s_a_row = THREADS_PER_EXPERT * 16
            else:
                s_a_mine = s_a_base + t * 16
                s_a_step = iter_bytes
                s_a_row = row_bytes

            if cutlass.const_expr(self.prefetch_weights):
                for i in cutlass.range_constexpr(KI):
                    cp_async_16(s_b_mine + i * iter_bytes, g_b_mine + Int64(i * iter_bytes))
                cp_async_commit()

            if cutlass.const_expr(self.use_pdl):
                griddep_wait()

            if cutlass.const_expr(not self.prefetch_weights):
                for i in cutlass.range_constexpr(KI):
                    cp_async_16(s_b_mine + i * iter_bytes, g_b_mine + Int64(i * iter_bytes))
                cp_async_commit()

            if cutlass.const_expr(private):
                for i in cutlass.range_constexpr(KI):
                    if cutlass.const_expr(stage_a):
                        for m in cutlass.range_constexpr(M):
                            cp_async_16(
                                s_a_mine + i * s_a_step + m * s_a_row,
                                g_a_mine + Int64(m * row_bytes + i * iter_bytes),
                            )
                    cp_async_commit()
            else:
                # The activation block [M, K] is one contiguous range.
                if cutlass.const_expr(stage_a):
                    for v in cutlass.range_constexpr((self.a_vectors + NT - 1) // NT):
                        idx = tidx + v * NT
                        if cutlass.const_expr(self.a_vectors % NT == 0):
                            cp_async_16(s_a_base + idx * 16, g_a_base + Int64(idx) * Int64(16))
                        else:
                            if idx < self.a_vectors:
                                cp_async_16(s_a_base + idx * 16, g_a_base + Int64(idx) * Int64(16))
                cp_async_commit()
                cp_async_wait(0)
                cute.arch.sync_threads()

            if cutlass.const_expr(self.use_pdl):
                griddep_launch_dependents()

            acc = [Float32(0.0)] * M
            for i in cutlass.range_constexpr(KI):
                if cutlass.const_expr(private):
                    # Groups complete in order: weights, then K steps 0..i.
                    cp_async_wait(KI - 1 - i)
                if cutlass.const_expr(compute):
                    b_words = lds_v4(s_b_mine + i * iter_bytes)
                    a_words = []
                    for m in cutlass.range_constexpr(M):
                        a_words.append(lds_v4(s_a_mine + i * s_a_step + m * s_a_row))
                    acc = accumulate(acc, a_words, b_words, self.paired_fma)

            for m in cutlass.range_constexpr(M):
                s = acc[m]
                for r in cutlass.range_constexpr(5):
                    s = add_rn(s, cute.arch.shuffle_sync_bfly(s, offset=16 >> r))
                if lane == 0:
                    warp_sums[warp, m, slot] = s

            cute.arch.sync_threads()

            if t < M:
                total = Float32(0.0)
                for w in cutlass.range_constexpr(WARPS_PER_EXPERT):
                    total = add_rn(total, warp_sums[w, t, slot])
                mC[t, expert] = total

    return cutlass, cute, RouterGemmKernel


_kernel_class = None


def _get_compiled(
    num_rows: int,
    hidden_size: int,
    num_experts: int,
    experts_per_cta: int,
    use_pdl: bool,
    prefetch_weights: bool,
    paired_fma: bool,
    attribution: str,
):
    key = (
        num_rows,
        hidden_size,
        num_experts,
        experts_per_cta,
        use_pdl,
        prefetch_weights,
        paired_fma,
        attribution,
    )
    fn = _compiled.get(key)
    if fn is not None:
        return fn
    global _kernel_class
    with _compile_lock:
        fn = _compiled.get(key)
        if fn is not None:
            return fn
        if _kernel_class is None:
            _kernel_class = _build_kernel_class()
        cutlass, cute, kernel_cls = _kernel_class
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (num_rows, hidden_size), stride_order=(1, 0), assumed_align=16
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (num_experts, hidden_size), stride_order=(1, 0), assumed_align=16
        )
        c_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (num_rows, num_experts), stride_order=(1, 0), assumed_align=16
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = kernel_cls(
            num_rows,
            hidden_size,
            num_experts,
            experts_per_cta,
            use_pdl,
            prefetch_weights,
            paired_fma,
            attribution,
        )
        fn = cute.compile(kernel, a_fake, b_fake, c_fake, stream, options="--enable-tvm-ffi")
        _compiled[key] = fn
        return fn


def router_gemm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    experts_per_cta: Optional[int] = None,
    use_pdl: Optional[bool] = None,
    prefetch_weights: bool = True,
    paired_fma: bool = DEFAULT_PAIRED_FMA,
    _attribution: str = "",
) -> torch.Tensor:
    """FP32 router logits ``hidden @ weight.T``, bitwise equal to native.

    Args:
        hidden: ``[R, K]`` BF16, contiguous, ``1 <= R <= 16``.
        weight: ``[256, K]`` BF16, contiguous.
        out: optional ``[R, 256]`` FP32 contiguous output.

    The caller must have checked ``router_gemm_supported``. The first call
    for a geometry compiles the kernel, so it must happen outside CUDA graph
    capture (the executor's eager warmup covers every captured shape).
    ``_attribution`` ("no_activation" / "no_compute") drops one phase for
    timing attribution only; its output is meaningless.
    """
    rows, hidden_size = hidden.shape
    num_experts = weight.shape[0]
    if experts_per_cta is None:
        experts_per_cta = DEFAULT_EXPERTS_PER_CTA
    experts_per_cta = fit_experts_per_cta(rows, hidden_size, experts_per_cta)
    if use_pdl is None:
        use_pdl = pdl_enabled()
    if out is None:
        out = torch.empty((rows, num_experts), dtype=torch.float32, device=hidden.device)
    fn = _get_compiled(
        rows,
        hidden_size,
        num_experts,
        experts_per_cta,
        use_pdl,
        prefetch_weights,
        paired_fma,
        _attribution,
    )
    fn(hidden, weight, out)
    return out
