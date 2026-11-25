# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-boundary coefficients of up to 8 rows in one CUDA-core launch: BF16 x TF32 products are
exact in FP32, so only the FP32 summation order differs from DeepGEMM's TF32 GEMM + bigfuse."""

import contextlib
import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import cuda as cuda_dialect
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait

Int32, Int64, Uint32, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Float32

MULT = 4
MIXES = 24  # pre (4) + post (4) + comb (16)
PART = MIXES + 1  # floats per token and CTA partial: the mixes, then the sum of squares
COLUMNS = 256  # residual columns per CTA
WARPS = 8
MAX_ROWS = 8
NEG_LOG2E = "0fBFB8AA3B"
LOG2E = "0f3FB8AA3B"

_ATT = llvm.AsmDialect.AD_ATT


def _f32_asm(text, *values, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
            text,
            ",".join(["=f"] + ["f"] * len(values)),
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _void_asm(text, operands, constraints, loc=None, ip=None):
    llvm.inline_asm(
        None,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return _f32_asm("fma.rn.ftz.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return _f32_asm("add.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _sub(a, b, *, loc=None, ip=None):
    return _f32_asm("sub.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    return _f32_asm("mul.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _max(a, b, *, loc=None, ip=None):
    return _f32_asm("max.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _rcp(a, *, loc=None, ip=None):
    return _f32_asm("rcp.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def _div(a, b, *, loc=None, ip=None):
    return _f32_asm("div.approx.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _rsqrt(a, *, loc=None, ip=None):
    return _f32_asm("rsqrt.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def _exp(x, *, negate: bool = False, loc=None, ip=None):
    """``expf(x)`` (or ``expf(-x)``) of a fast-math build: ``ex2.approx.ftz(x * (-)log2(e))``."""
    k = NEG_LOG2E if negate else LOG2E
    return _f32_asm(
        f"{{\n.reg .f32 t;\nmul.ftz.f32 t, $1, {k};\nex2.approx.ftz.f32 $0, t;\n}}",
        x,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sigmoid_of_exp(e, *, loc=None, ip=None):
    return _f32_asm(
        "{\n.reg .f32 t;\nadd.ftz.f32 t, $1, 0f3F800000;\nrcp.approx.ftz.f32 $0, t;\n}",
        e,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _zero(*, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [],
            "mov.b32 $0, 0f00000000;",
            "=f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_ATT,
        )
    )


@dsl_user_op
def _shfl_xor(value, mask, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip), Int32(mask).ir_value(loc=loc, ip=ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_weight(addr, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None):
    """A BF16 residual value as FP32 (exact); ordered after the dependency wait."""
    word = Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b16 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return Float32(llvm.bitcast(T.f32(), (word << 16).ir_value(loc=loc, ip=ip)))


@dsl_user_op
def _ld_partial(addr, *, loc=None, ip=None):
    """A float other CTAs of this launch stored: from L2."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_f32(addr, value, *, loc=None, ip=None):
    # No memory clobber: the kernel never reads these back; the barrier and the ticket order them.
    _void_asm(
        "st.global.f32 [$0], $1;",
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_s32(addr, value, *, loc=None, ip=None):
    _void_asm(
        "st.global.b32 [$0], $1;",
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ticket(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(1).ir_value(loc=loc, ip=ip)],
            "atom.add.acq_rel.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_ATT,
            loc=loc,
            ip=ip,
        )
    )


class MhcCoefficientFmaKernel:
    """``(pre, post, comb)`` of ``rows`` tokens from the residual ``[rows, 4 * hidden]``: a CTA per
    256 columns; the last CTA to arrive sums the partials in CTA order and runs bigfuse's tail.
    With ``release`` set, dependents may launch once every CTA has passed the dependency wait."""

    def __init__(self, rows: int, hidden: int, sinkhorn_iters: int, use_pdl: bool):
        assert 1 <= rows <= MAX_ROWS and self.supports(hidden) and sinkhorn_iters >= 1
        self.rows = rows
        self.k = MULT * hidden
        self.ctas = self.k // COLUMNS
        self.sums = rows * PART  # (token, value) sums of the last CTA, one thread each
        self.sinkhorn_iters = sinkhorn_iters
        self.use_pdl = use_pdl

    @staticmethod
    def supports(hidden: int) -> bool:
        return (MULT * hidden) % COLUMNS == 0

    @cute.jit
    def __call__(
        self,
        residual,
        weight,
        scale,
        base,
        partials,
        counter,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        release: Int32,
        stream,
    ):
        self.kernel(
            residual,
            weight,
            scale,
            base,
            partials,
            counter,
            pre_out,
            post_out,
            comb_out,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            release,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[32 * WARPS, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            # 58%: the direct decode's carveout, so its CTAs share the SMs this kernel holds early.
            preferred_smem_carveout=58,
        )

    @cute.kernel
    def kernel(
        self,
        residual,
        weight,
        scale,
        base,
        partials,
        counter,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        release: Int32,
    ):
        M = self.rows
        K = self.k
        NS = self.sums
        tid = cute.arch.thread_idx()[0]
        cta = cute.arch.block_idx()[0]
        warp = tid // 32
        lane = tid % 32
        k0 = cta * COLUMNS + lane
        res_addr = residual.iterator.toint()
        part_addr = partials.iterator.toint()
        counter_addr = counter.iterator.toint()

        smem = SmemAllocator()
        s_hc = smem.allocate_tensor(Float32, cute.make_layout(MIXES + 3), 16)
        s_sum = smem.allocate_tensor(Float32, cute.make_layout(MAX_ROWS * PART), 16)
        s_ticket = smem.allocate_tensor(Int32, cute.make_layout(1), 16)

        w = cute.make_rmem_tensor(cute.make_layout((3, 8), stride=(8, 1)), Float32)
        x = cute.make_rmem_tensor(cute.make_layout((M, 8), stride=(8, 1)), Float32)
        v = cute.make_rmem_tensor(cute.make_layout(self.ctas), Float32)
        # Weights are read after the wait: read before it, they slow the producer they overlap.
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            if release != Int32(0):
                # The dependents' griddepcontrol.wait covers every prerequisite grid, this one too.
                griddepcontrol_launch_dependents()
        w_addr = weight.iterator.toint()
        for j in cutlass.range_constexpr(3):
            for i in cutlass.range_constexpr(8):
                w[j, i] = _ld_weight(w_addr + Int64(4) * (Int64(3 * warp + j) * K + k0 + 32 * i))
        if tid < MIXES:
            s_hc[tid] = _ld_weight(base.iterator.toint() + Int64(4) * Int64(tid))
        if (tid >= MIXES) & (tid < MIXES + 3):
            s_hc[tid] = _ld_weight(scale.iterator.toint() + Int64(4) * Int64(tid - MIXES))

        for r in cutlass.range_constexpr(M):
            for i in cutlass.range_constexpr(8):
                x[r, i] = _ld_bf16(res_addr + Int64(2) * (Int64(r) * K + k0 + 32 * i))
        for r in cutlass.range_constexpr(M):
            zero = _zero()
            a0 = zero
            a1 = zero
            a2 = zero
            sq = zero
            for i in cutlass.range_constexpr(8):
                xi = x[r, i]
                a0 = _fma(xi, w[0, i], a0)
                a1 = _fma(xi, w[1, i], a1)
                a2 = _fma(xi, w[2, i], a2)
                sq = _fma(xi, xi, sq)
            for o in cutlass.range_constexpr(5):
                a0 = _add(a0, _shfl_xor(a0, 16 >> o))
                a1 = _add(a1, _shfl_xor(a1, 16 >> o))
                a2 = _add(a2, _shfl_xor(a2, 16 >> o))
                sq = _add(sq, _shfl_xor(sq, 16 >> o))
            if lane == 0:
                p = part_addr + Int64(4) * ((Int64(cta) * M + r) * PART + 3 * warp)
                _st_f32(p, a0)
                _st_f32(p + Int64(4), a1)
                _st_f32(p + Int64(8), a2)
                if warp == 0:
                    _st_f32(p + Int64(4 * MIXES), sq)

        cute.arch.sync_threads()
        if tid == 0:
            s_ticket[0] = _ticket(counter_addr)
        cute.arch.sync_threads()
        if s_ticket[0] == Int32(self.ctas - 1):
            if tid < NS:
                for c in cutlass.range_constexpr(self.ctas):
                    v[c] = _ld_partial(part_addr + Int64(4) * (Int64(c) * NS + tid))
                acc = _add(v[0], _zero())
                for c in cutlass.range_constexpr(1, self.ctas):
                    acc = _add(acc, v[c])
                s_sum[tid] = acc
            cute.arch.sync_threads()
            if warp == 0:
                self.token_tail(
                    lane,
                    s_sum,
                    s_hc,
                    pre_out,
                    post_out,
                    comb_out,
                    rms_eps,
                    hc_eps,
                    sinkhorn_eps,
                    post_mult,
                )
            if tid == 0:
                _st_s32(counter_addr, Int32(0))  # the next launch's ticket

    @cute.jit
    def token_tail(
        self,
        lane: Int32,
        s_sum,
        s_hc,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
    ):
        """Bigfuse's coefficients of token ``lane // 4``, ``comb`` row ``lane % 4``; xor-1, xor-2
        shuffles sum a column as bigfuse's ``(c0 + c1) + (c2 + c3)``. Extra lanes store nothing."""
        M = self.rows
        token = lane // 4
        r = lane % 4
        valid = token < M
        tc = token % M
        b = tc * PART
        y_pre = s_sum[b + r]
        y_post = s_sum[b + MULT + r]
        y_comb = cute.make_rmem_tensor(cute.make_layout(MULT), Float32)
        base_comb = cute.make_rmem_tensor(cute.make_layout(MULT), Float32)
        for k in cutlass.range_constexpr(MULT):
            y_comb[k] = s_sum[b + 2 * MULT + MULT * r + k]
            base_comb[k] = s_hc[2 * MULT + MULT * r + k]
        rstd = _rsqrt(_add(rms_eps, _div(s_sum[b + MIXES], Float32(float(self.k)))))
        pre = _add(
            hc_eps,
            _sigmoid_of_exp(_exp(_fma(s_hc[MIXES], _mul(rstd, y_pre), s_hc[r]), negate=True)),
        )
        post = _mul(
            post_mult,
            _sigmoid_of_exp(
                _exp(_fma(s_hc[MIXES + 1], _mul(rstd, y_post), s_hc[MULT + r]), negate=True)
            ),
        )
        out = Int64(4) * (tc * MULT + r)
        if valid:
            _st_f32(pre_out.iterator.toint() + out, pre)
            _st_f32(post_out.iterator.toint() + out, post)
        c = cute.make_rmem_tensor(cute.make_layout(MULT), Float32)
        e = cute.make_rmem_tensor(cute.make_layout(MULT), Float32)
        for k in cutlass.range_constexpr(MULT):
            c[k] = _fma(s_hc[MIXES + 2], _mul(rstd, y_comb[k]), base_comb[k])
        row_max = _max(_max(c[0], c[1]), _max(c[2], c[3]))
        for k in cutlass.range_constexpr(MULT):
            e[k] = _exp(_sub(c[k], row_max))
        inv = _rcp(_add(_add(_add(e[0], e[1]), e[2]), e[3]))
        for k in cutlass.range_constexpr(MULT):
            c[k] = _fma(inv, e[k], sinkhorn_eps)
        self.normalize_columns(c, sinkhorn_eps)
        for _ in cutlass.range_constexpr(self.sinkhorn_iters - 1):
            inv = _rcp(_add(sinkhorn_eps, _add(_add(_add(c[0], c[1]), c[2]), c[3])))
            for k in cutlass.range_constexpr(MULT):
                c[k] = _mul(inv, c[k])
            self.normalize_columns(c, sinkhorn_eps)
        if valid:
            comb = comb_out.iterator.toint() + Int64(4) * ((tc * MULT + r) * MULT)
            for k in cutlass.range_constexpr(MULT):
                _st_f32(comb + Int64(4 * k), c[k])

    @cute.jit
    def normalize_columns(self, c, sinkhorn_eps: Float32):
        for k in cutlass.range_constexpr(MULT):
            t1 = _add(c[k], _shfl_xor(c[k], 1))
            t2 = _add(t1, _shfl_xor(t1, 2))
            c[k] = _mul(_rcp(_add(sinkhorn_eps, t2)), c[k])


def prepare_fma_weight(fn: torch.Tensor) -> torch.Tensor:
    """``fn`` rounded to TF32 (nearest even), as DeepGEMM's TF32 ``hc_prenorm`` GEMM rounds it."""
    bits = fn.contiguous().view(torch.int32)
    low = bits & 0x1FFF
    up = (low > 0x1000) | ((low == 0x1000) & ((bits & 0x2000) != 0))
    rounded = (bits & -0x2000) + up.to(torch.int32) * 0x2000
    special = (bits & 0x7F800000) == 0x7F800000
    return torch.where(special, bits, rounded).view(torch.float32)


def fma_counter(device) -> torch.Tensor:
    """A zeroed arrival ticket for ``mhc_coefficients_fma``; the kernel leaves it zeroed. Launches
    sharing it must not overlap."""
    return torch.zeros(1, dtype=torch.int32, device=device)


_COMPILED = set()


def _mat(dtype, *shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16
    )


@contextlib.contextmanager
def _carveout_at_load():
    """Scoped: a launch's ``preferred_smem_carveout`` goes into the kernel's load-time attributes,
    which the CUDA-dialect executors of CuTe DSL 4.8 apply (they never read ``kernel_info``)."""
    import cutlass.cutlass_dsl.cutlass as dsl

    base = dsl.CutlassBaseDSL
    generate = base._generate_kernel_attrs

    def with_carveout(self, config):
        attrs = generate(self, config)
        if config.preferred_smem_carveout is not None:
            i32 = ir.IntegerType.get_signless(32)
            names = cuda_dialect.CUFunctionAttribute
            # Replaces the kernel op's own entries, so it repeats them.
            attrs["cu_attrs"] = ir.DictAttr.get(
                {
                    str(names.non_portable_cluster_size_allowed): ir.IntegerAttr.get(i32, 1),
                    str(
                        names.max_dynamic_shared_size_bytes
                    ): cuda_dialect.DevMaxSharedMemoryOptinAttr.get(),
                    str(names.preferred_shared_memory_carveout): ir.IntegerAttr.get(
                        i32, config.preferred_smem_carveout
                    ),
                }
            )
        return attrs

    base._generate_kernel_attrs = with_carveout
    try:
        yield
    finally:
        base._generate_kernel_attrs = generate


@functools.cache
def _compile(rows: int, hidden: int, sinkhorn_iters: int, use_pdl: bool):
    kernel = MhcCoefficientFmaKernel(rows, hidden, sinkhorn_iters, use_pdl)
    with _carveout_at_load():
        fn = cute.compile(
            kernel,
            _mat(cutlass.BFloat16, rows, MULT * hidden),
            _mat(Float32, MIXES, MULT * hidden),
            _mat(Float32, 3),
            _mat(Float32, MIXES),
            _mat(Float32, kernel.ctas, rows, PART),
            _mat(Int32, 1),
            _mat(Float32, rows, MULT),
            _mat(Float32, rows, MULT),
            _mat(Float32, rows, MULT * MULT),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _COMPILED.add((rows, hidden, sinkhorn_iters, use_pdl))
    return fn


def compile_all(hidden: int, sinkhorn_iters: int) -> None:
    for rows in range(1, MAX_ROWS + 1):
        _compile(rows, hidden, sinkhorn_iters, TRTLLM_ENABLE_PDL)


def compiled(rows: int, hidden: int, sinkhorn_iters: int) -> bool:
    """Whether ``mhc_coefficients_fma`` is compiled for this variant (not allowed in capture)."""
    return (rows, hidden, sinkhorn_iters, TRTLLM_ENABLE_PDL) in _COMPILED


def mhc_coefficients_fma(
    residual,
    weight,
    scale,
    base,
    rms_eps,
    hc_eps,
    sinkhorn_eps,
    post_mult,
    sinkhorn_iters,
    counter,
    release_after_wait: bool = False,
):
    """``(pre, post, comb)`` of a boundary from its BF16 residual ``[M, 4 * hidden]`` (``M <= 8``);
    ``weight`` is ``prepare_fma_weight``'s and ``counter`` ``fma_counter``'s. ``release_after_wait``
    lets dependents launch from the dependency wait on, for consumers that read the coefficients
    after their own wait."""
    rows, k = residual.shape
    dev = residual.device
    kernel = _compile(rows, k // MULT, sinkhorn_iters, TRTLLM_ENABLE_PDL)
    pre, post = torch.empty(2, rows, MULT, dtype=torch.float32, device=dev)
    comb = torch.empty(rows, MULT * MULT, dtype=torch.float32, device=dev)
    partials = torch.empty(k // COLUMNS, rows, PART, dtype=torch.float32, device=dev)
    kernel(
        residual,
        weight,
        scale,
        base,
        partials,
        counter,
        pre,
        post,
        comb,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        post_mult,
        int(release_after_wait),
    )
    return pre, post, comb
