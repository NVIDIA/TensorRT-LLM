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
"""CuTe DSL kernel of the post_moe_hop (the post_moe_hop contract), cluster variant.

Same protocol and arithmetic as the Triton kernel in ``kernels.py`` and the
same workspace, but one thread block cluster per token row: ``CLUSTER``
CTAs of ``THREADS`` threads, every thread owning 8 consecutive hidden
columns (one 16-byte access per tensor). Per thread:

1. fp32 FMA finalize of the raw tuple in ``k`` order, bf16 rounding, bf16
   add of the shared-expert partial, ``-0.0`` sanitize;
2. 16-byte store through the multicast address into slot ``[row][rank]``;
3. share of the clear of the previous call's Lamport rows (and partial
   slice, which only the Triton kernel uses but both keep clean);
4. per-thread Lamport poll of the 8 rank slots (one inline-PTX spin loop,
   no block-wide reduction per iteration, ``%globaltimer`` deadline checked
   every 1,024 polls), rank-ordered fp32 sum, bf16;
5. bf16 residual add and store;
6. RMSNorm: per-thread sum of bf16-rounded squares, warp butterfly, the
   per-warp sums are written into every CTA of the cluster through
   distributed shared memory, one cluster barrier, then every thread sums
   the same ``CLUSTER * WARPS`` values in the same order.

All global accesses go through inline PTX on raw addresses so the memory
operations (volatile polls, multicast stores, acquire/release atomics) are
explicit. Every tensor is therefore accessed with 16-byte vectors: the caller
(``hop.py``) guarantees 16-byte aligned bases and row pitches.

Error handling is the same as in the Triton kernel: an expired deadline sets
error bit 1 and the largest poll count, and NaN replaces this thread's
residual (hence the row's normalized output); a permutation entry at or beyond
``fc2_rows`` is skipped and sets error bit 4. Error bits go into the error
word of the call's Lamport buffer; a call reads the previous buffer's word
(complete at the kernel boundary) and, when it is nonzero, publishes NaN and
carries the bits into its own word, so every later call of this rank
publishes NaN until the collective workspace reset (see ``kernels.py``).
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from .kernels import MAX_CHUNKS

_SENTINEL_WORD = -2147450880  # 0x80008000: two bf16 -0.0
_FP32_SENTINEL = -2147483648  # 0x80000000
# Polls between two deadline checks; the clock starts at the first check.
_POLLS_PER_CLOCK_CHECK = 1024


def _asm(res, operands, asm, constraints, loc, ip):
    return llvm.inline_asm(
        res,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _struct(n, ty):
    return llvm.StructType.get_literal([ty] * n)


def _unpack(res, n, cls, ty, loc, ip):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    res = _asm(
        _struct(4, T.i32()),
        [_v(addr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, Int32, T.i32(), loc, ip)


@dsl_user_op
def _st_v4(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, loc, ip)] + [_v(x, loc, ip) for x in (a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _poll8(base: Int64, stride: Int64, timeout_ns: Int64, *, loc=None, ip=None):
    """Spin until 8 rank slots (16 bytes each, ``stride`` apart) hold no sentinel.

    The ``%globaltimer`` deadline is checked every ``_POLLS_PER_CLOCK_CHECK``
    polls and its clock starts at the first check, so a poll that finds every
    slot within that many iterations never reads the timer. Returns the 32
    words, a nonzero flag when the deadline expired, and the poll count.
    """
    loads = []
    checks = []
    for r in range(8):
        o = 4 * r
        loads.append(
            f"ld.volatile.global.v4.b32 {{${o}, ${o + 1}, ${o + 2}, ${o + 3}}}, [a];\n"
            "add.s64 a, a, $35;\n"
        )
        for w in range(4):
            checks.append(
                f"and.b32 t, ${o + w}, 0xFFFF0000;\n"
                "setp.eq.u32 p, t, 0x80000000;\n"
                "selp.b32 t, 1, 0, p;\n"
                "or.b32 d, d, t;\n"
            )
    period = _POLLS_PER_CLOCK_CHECK
    asm = (
        "{\n.reg .pred p;\n.reg .b32 t, d, n;\n.reg .b64 a, t0, now;\n"
        "mov.b32 n, 0;\nmov.b64 t0, 0;\n"
        "POST_MOE_POLL:\nmov.b64 a, $34;\nmov.b32 d, 0;\n"
        + "".join(loads)
        + "".join(checks)
        + "add.u32 n, n, 1;\nsetp.eq.u32 p, d, 0;\n@p bra POST_MOE_POLL_DONE;\n"
        f"and.b32 t, n, {period - 1};\nsetp.ne.u32 p, t, 0;\n@p bra POST_MOE_POLL;\n"
        "mov.u64 now, %globaltimer;\n"
        f"setp.eq.u32 p, n, {period};\n@p mov.b64 t0, now;\n"
        "sub.s64 now, now, t0;\nsetp.lt.s64 p, now, $36;\n@p bra POST_MOE_POLL;\n"
        "POST_MOE_POLL_DONE:\nmov.b32 $32, d;\nmov.b32 $33, n;\n}\n"
    )
    constraints = ",".join(["=r"] * 34 + ["l", "l", "l"])
    res = _asm(
        _struct(34, T.i32()),
        [_v(base, loc, ip), _v(stride, loc, ip), _v(timeout_ns, loc, ip)],
        asm,
        constraints,
        loc,
        ip,
    )
    return _unpack(res, 34, Int32, T.i32(), loc, ip)


@dsl_user_op
def _unpack_bf16x2(word: Int32, *, loc=None, ip=None):
    res = _asm(
        _struct(2, T.f32()),
        [_v(word, loc, ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\n"
        "and.b32 t, $2, 0xFFFF0000;\nmov.b32 $1, t;\n}\n",
        "=f,=f,r",
        loc,
        ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(lo, loc, ip), _v(hi, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _round_bf16(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(x, loc, ip)],
            "{\n.reg .b32 t;\ncvt.rn.bf16x2.f32 t, $1, $1;\n"
            "and.b32 t, t, 0xFFFF0000;\nmov.b32 $0, t;\n}\n",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sanitize(word: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(word, loc, ip)],
            "{\n.reg .b32 t;\n.reg .pred p, q;\n"
            "and.b32 t, $1, 0xFFFF;\nsetp.eq.u32 p, t, 0x8000;\n"
            "and.b32 t, $1, 0xFFFF0000;\nsetp.eq.u32 q, t, 0x80000000;\n"
            "mov.b32 $0, $1;\n@p and.b32 $0, $0, 0xFFFF0000;\n"
            "@q and.b32 $0, $0, 0xFFFF;\n}\n",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_v(a, loc, ip), _v(b, loc, ip)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def _mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_v(a, loc, ip), _v(b, loc, ip)], "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def _select(pred: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(pred, loc, ip), _v(a, loc, ip), _v(b, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}\n",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _perm_check(index: Int32, rows: Int32, *, loc=None, ip=None):
    """Return ``(valid, safe_index, out_of_range)`` of one permutation entry.

    ``valid`` is ``0 <= index < rows``; ``safe_index`` is ``index`` when valid
    and 0 otherwise; ``out_of_range`` flags ``index >= rows`` (negative
    entries are the native "no local row" marker, not an error).
    """
    res = _asm(
        _struct(3, T.i32()),
        [_v(index, loc, ip), _v(rows, loc, ip)],
        "{\n.reg .pred p, q;\nsetp.lt.u32 p, $3, $4;\nselp.b32 $0, 1, 0, p;\n"
        "selp.b32 $1, $3, 0, p;\nsetp.ge.s32 q, $3, $4;\nselp.b32 $2, 1, 0, q;\n}\n",
        "=r,=r,=r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 3, Int32, T.i32(), loc, ip)


@dsl_user_op
def _poison(word: Int32, flag: Int32, *, loc=None, ip=None) -> Int32:
    """Two bf16 NaNs instead of ``word`` when ``flag`` is nonzero."""
    return Int32(
        _asm(
            T.i32(),
            [_v(word, loc, ip), _v(flag, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\nselp.b32 $0, 0x7FC07FC0, $1, p;\n}\n",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bits_f32(word: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(word, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip))


@dsl_user_op
def _inv_rms(square_sum: Float32, hidden: Float32, eps: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(square_sum, loc, ip), _v(hidden, loc, ip), _v(eps, loc, ip)],
            "{\n.reg .f32 t;\ndiv.rn.f32 t, $1, $2;\nadd.rn.f32 t, t, $3;\n"
            "rsqrt.approx.f32 $0, t;\n}\n",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _shfl_xor(x: Float32, mask: int, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(x, loc, ip)],
            f"shfl.sync.bfly.b32 $0, $1, {mask}, 0x1f, 0xffffffff;",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_cluster_f32(
    local_addr: Int32, target_cta: Int32, value: Float32, enable: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [_v(local_addr, loc, ip), _v(target_cta, loc, ip), _v(value, loc, ip), _v(enable, loc, ip)],
        "{\n.reg .pred p;\n.reg .b32 r;\nsetp.ne.s32 p, $3, 0;\n"
        "@p mapa.shared::cluster.u32 r, $0, $1;\n@p st.shared::cluster.f32 [r], $2;\n}\n",
        "r,r,f,r",
        loc,
        ip,
    )


@dsl_user_op
def _ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(addr, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip))


@dsl_user_op
def _error_word(words: list, index: Int32, *, loc=None, ip=None) -> Int32:
    """Select error word ``index`` (0..2) of the three loaded state words."""
    return Int32(
        _asm(
            T.i32(),
            [_v(index, loc, ip)] + [_v(w, loc, ip) for w in words],
            "{\n.reg .pred p;\nsetp.eq.u32 p, $1, 1;\nselp.b32 $0, $3, $2, p;\n"
            "setp.eq.u32 p, $1, 2;\nselp.b32 $0, $4, $0, p;\n}\n",
            "=r,r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _flag_error(word: Int64, flag: Int32, bits: Int32, *, loc=None, ip=None) -> None:
    """OR ``bits`` into the error word at ``word`` when ``flag`` is nonzero."""
    _asm(
        None,
        [_v(word, loc, ip), _v(flag, loc, ip), _v(bits, loc, ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\n@p red.relaxed.gpu.global.or.b32 [$0], $2;\n}\n",
        "l,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _flag_timeout(
    word: Int64, state: Int64, flag: Int32, polls: Int32, *, loc=None, ip=None
) -> None:
    """Error bit 1 into the error word and the largest poll count at expiry (word 5)."""
    _asm(
        None,
        [_v(word, loc, ip), _v(state, loc, ip), _v(flag, loc, ip), _v(polls, loc, ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\n"
        "@p red.relaxed.gpu.global.or.b32 [$0], 1;\n"
        "@p red.relaxed.gpu.global.max.u32 [$1+20], $3;\n}\n",
        "l,l,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _fault_delay(
    a: Int32, b: Int32, c: Int32, max_ns: int, site: int, *, loc=None, ip=None
) -> None:
    """Test-only fault injection: about one thread in four spins before ``site``.

    The delay (below ``max_ns``) is a hash of ``a``, ``b`` and ``c``, so the
    individual 16-byte stores of one call land in a scrambled order.
    """
    asm = (
        "{\n.reg .pred p;\n.reg .b32 h, t;\n.reg .b64 t0, now, d;\n"
        "mul.lo.u32 h, $0, 0x9E3779B1;\nmul.lo.u32 t, $1, 0x85EBCA77;\nxor.b32 h, h, t;\n"
        "mul.lo.u32 t, $2, 0xC2B2AE3D;\nxor.b32 h, h, t;\nshr.u32 t, h, 15;\n"
        "xor.b32 h, h, t;\nmul.lo.u32 h, h, 0x2C1B3C6D;\nshr.u32 t, h, 13;\n"
        "xor.b32 h, h, t;\nand.b32 t, h, 3;\nsetp.ne.u32 p, t, 0;\n"
        f"@p bra POST_MOE_FAULT_END_{site};\nshr.u32 t, h, 2;\nrem.u32 t, t, {max_ns};\n"
        "cvt.u64.u32 d, t;\nmov.u64 t0, %globaltimer;\n"
        f"POST_MOE_FAULT_LOOP_{site}:\nmov.u64 now, %globaltimer;\nsub.s64 now, now, t0;\n"
        f"setp.lt.s64 p, now, d;\n@p bra POST_MOE_FAULT_LOOP_{site};\n"
        f"POST_MOE_FAULT_END_{site}:\n}}\n"
    )
    _asm(None, [_v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip)], asm, "r,r,r", loc, ip)


@dsl_user_op
def _arrive(
    state: Int64,
    is_leader: Int32,
    num_ctas: Int32,
    num_rows: Int32,
    next_epoch: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Leader thread of each CTA counts in; the last CTA publishes the next epoch."""
    _asm(
        None,
        [
            _v(state, loc, ip),
            _v(is_leader, loc, ip),
            _v(num_ctas, loc, ip),
            _v(num_rows, loc, ip),
            _v(next_epoch, loc, ip),
        ],
        "{\n.reg .pred p, q;\n.reg .b32 old, c;\n"
        "setp.ne.s32 p, $1, 0;\nmov.b32 old, 0;\n"
        "@p atom.acq_rel.gpu.global.add.u32 old, [$0+8], 1;\n"
        "add.u32 c, old, 1;\nsetp.eq.u32 q, c, $2;\nand.pred q, q, p;\n"
        "@q st.global.u32 [$0+4], $3;\n@q st.global.u32 [$0+8], 0;\n"
        "@q red.relaxed.gpu.global.add.u32 [$0+16], 1;\n"
        "@q st.global.u32 [$0], $4;\n}\n",
        "l,r,r,r,r",
        loc,
        ip,
    )


class PostMoeHopClusterKernel:
    """Compile-time configuration of the cluster kernel for one hop shape.

    Test-only knobs (the hop never sets them):

    * ``race_probe``: fault injection for the compute-sanitizer positive
      control; the reduction buffer is read between the cluster arrive and
      wait, so the distributed shared memory stores of the other CTAs race
      with the reads.
    * ``fault_delay_ns``: delays about one thread in four by a hashed amount
      below this bound before its multicast publish, before its share of the
      clear and before the arrival, so the individual stores of a call land
      in a scrambled order (protocol fault injection).

    routed_moe_into_hop entry (``routed_input``, reachable only through
    ``PostMoeHop.enable_routed_input``; the default ``False`` is the post_moe_hop
    on the raw tuple): only how the rank-local routed partial is formed
    changes, everything after it is the same code.

    * ``"row"`` (or ``True``): ``fc2`` is the rank's finalized routed output
      ``[R, H]`` bf16 (no scales, no permutation);
    * ``"identity"``: raw FC2 rows in the identity layout (row ``t * 8 + k``)
      with bf16 scales ``[R, 8]``; the native finalize without the
      permutation round trip;
    * ``"fp32"``: ``fc2`` is an FP32 ``[R, H]`` accumulator of the top-k
      contributions, rounded to bf16 here and cleared after the read.
    """

    def __init__(
        self,
        hidden: int,
        top_k: int,
        world: int,
        max_rows: int,
        cluster: int,
        has_shared: bool,
        scale_fp32: bool,
        use_pdl: bool,
        race_probe: bool = False,
        fault_delay_ns: int = 0,
        routed_input=False,
    ):
        assert top_k == 8 and world == 8, "the cluster kernel is specialized to top-8 and TP8"
        assert hidden % (8 * 32 * cluster) == 0
        self.hidden = hidden
        self.top_k = top_k
        self.world = world
        self.max_rows = max_rows
        self.cluster = cluster
        self.threads = hidden // (8 * cluster)
        self.warps = self.threads // 32
        self.has_shared = has_shared
        self.scale_fp32 = scale_fp32
        self.use_pdl = use_pdl
        self.race_probe = race_probe
        self.fault_delay_ns = fault_delay_ns
        self.routed_input = "row" if routed_input is True else routed_input

    @cute.jit
    def __call__(
        self,
        fc2: Int64,
        fc2_row_bytes: Int64,
        fc2_rows: Int32,
        scales: Int64,
        perm: Int64,
        shared: Int64,
        residual: Int64,
        gamma: Int64,
        out: Int64,
        residual_out: Int64,
        mc: Int64,
        uc: Int64,
        partials: Int64,
        state: cute.Tensor,
        rank: Int32,
        eps: Float32,
        timeout_ns: Int64,
        num_rows: Int32,
        stream: CUstream,
    ):
        # ``state`` is passed as a GPU tensor so the TVM-FFI launcher can bind
        # the caller's current stream; every other buffer is a raw address.
        self.kernel(
            fc2,
            fc2_row_bytes,
            fc2_rows,
            scales,
            perm,
            shared,
            residual,
            gamma,
            out,
            residual_out,
            mc,
            uc,
            partials,
            state,
            rank,
            eps,
            timeout_ns,
            num_rows,
        ).launch(
            grid=[self.cluster, num_rows, 1],
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            smem=4 * self.cluster * self.warps + 64,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        fc2: Int64,
        fc2_row_bytes: Int64,
        fc2_rows: Int32,
        scales: Int64,
        perm: Int64,
        shared: Int64,
        residual: Int64,
        gamma: Int64,
        out: Int64,
        residual_out: Int64,
        mc: Int64,
        uc: Int64,
        partials: Int64,
        state_tensor: cute.Tensor,
        rank: Int32,
        eps: Float32,
        timeout_ns: Int64,
        num_rows: Int32,
    ):
        hidden = self.hidden
        world = self.world
        threads = self.threads
        warps = self.warps
        cluster = self.cluster
        buffer_bytes = self.max_rows * world * hidden * 2

        tidx, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        state = Int64(state_tensor.iterator.toint())

        smem = SmemAllocator()
        reduce_buf = smem.allocate_array(Float32, num_elems=cluster * warps)
        reduce_addr = Int32(reduce_buf.toint())

        cute.arch.cluster_arrive_relaxed()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # Words 0-7: epoch, prev_rows, arrivals, error[0], calls, timeout
        # polls, error[1], error[2]; error[b] collects the bits of the calls
        # that use Lamport buffer b.
        epoch, prev_rows, _, err0 = _ld_v4(state)
        _, _, err1, err2 = _ld_v4(state + Int64(16))
        cur = epoch % Int32(3)
        dirty = (epoch + Int32(2)) % Int32(3)
        # Bits of the previous call (complete at the kernel boundary): when set,
        # this call publishes NaN and carries them into its own error word.
        sticky = _error_word([err0, err1, err2], dirty)
        err_cur = state + Int64(_error_word([Int32(12), Int32(24), Int32(28)], cur))

        col = (cta * Int32(threads) + tidx) * Int32(8)
        row_off = Int64(row) * Int64(hidden * 2) + Int64(col) * Int64(2)
        res_w = _ld_v4(residual + row_off)
        gamma_w = _ld_v4(gamma + Int64(col) * Int64(2))
        if cutlass.const_expr(self.has_shared):
            sh_w = _ld_v4(shared + row_off)

        # 1. Finalize (fp32 FMA chain in k order), bf16, + shared, sanitize.
        if cutlass.const_expr(self.routed_input == "row"):
            # routed_moe_into_hop: the routed kernel already finalized this rank's output
            r_w = _ld_v4(fc2 + Int64(row) * fc2_row_bytes + Int64(col) * Int64(2))
            local = [r_w[j] for j in range(4)]
            out_of_range = Int32(0)
        elif cutlass.const_expr(self.routed_input == "fp32"):
            # routed_moe_into_hop: FP32 accumulator of the top-k contributions; round, then
            # clear it for the routed kernel of the next call
            a_addr = fc2 + Int64(row) * fc2_row_bytes + Int64(col) * Int64(4)
            a_w = _ld_v4(a_addr) + _ld_v4(a_addr + Int64(16))
            a_f = [_bits_f32(w) for w in a_w]
            local = [_pack_bf16x2(a_f[2 * j], a_f[2 * j + 1]) for j in range(4)]
            zero = Int32(0)
            _st_v4(a_addr, zero, zero, zero, zero)
            _st_v4(a_addr + Int64(16), zero, zero, zero, zero)
            out_of_range = Int32(0)
        elif cutlass.const_expr(self.routed_input == "identity"):
            # routed_moe_into_hop: raw rows t * 8 + k, every slot valid; the native finalize
            # without the permutation round trip
            s_w = _ld_v4(scales + Int64(row) * Int64(16))
            weights = []
            for w in s_w:
                lo, hi = _unpack_bf16x2(w)
                weights += [lo, hi]
            acc = [Float32(0.0)] * 8
            for k in cutlass.range_constexpr(8):
                x_w = _ld_v4(
                    fc2 + (Int64(row) * Int64(8) + Int64(k)) * fc2_row_bytes + Int64(col) * Int64(2)
                )
                for j in cutlass.range_constexpr(4):
                    x_lo, x_hi = _unpack_bf16x2(x_w[j])
                    acc[2 * j] = _fma(weights[k], x_lo, acc[2 * j])
                    acc[2 * j + 1] = _fma(weights[k], x_hi, acc[2 * j + 1])
            local = [_pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]
            out_of_range = Int32(0)
        else:
            perm_lo = _ld_v4(perm + Int64(row) * Int64(32))
            perm_hi = _ld_v4(perm + Int64(row) * Int64(32) + Int64(16))
            perm_k = perm_lo + perm_hi
            if cutlass.const_expr(self.scale_fp32):
                s_lo = _ld_v4(scales + Int64(row) * Int64(32))
                s_hi = _ld_v4(scales + Int64(row) * Int64(32) + Int64(16))
                weights = [_bits_f32(w) for w in s_lo + s_hi]
            else:
                s_w = _ld_v4(scales + Int64(row) * Int64(16))
                weights = []
                for w in s_w:
                    lo, hi = _unpack_bf16x2(w)
                    weights += [lo, hi]
            acc = [Float32(0.0)] * 8
            out_of_range = Int32(0)
            for k in cutlass.range_constexpr(8):
                valid, safe, oob = _perm_check(perm_k[k], fc2_rows)
                out_of_range = out_of_range + oob
                x_w = _ld_v4(fc2 + Int64(safe) * fc2_row_bytes + Int64(col) * Int64(2))
                for j in cutlass.range_constexpr(4):
                    x_lo, x_hi = _unpack_bf16x2(x_w[j])
                    acc[2 * j] = _select(valid, _fma(weights[k], x_lo, acc[2 * j]), acc[2 * j])
                    acc[2 * j + 1] = _select(
                        valid, _fma(weights[k], x_hi, acc[2 * j + 1]), acc[2 * j + 1]
                    )
            local = [_pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]
        if cutlass.const_expr(self.has_shared):
            summed = []
            for j in cutlass.range_constexpr(4):
                r_lo, r_hi = _unpack_bf16x2(local[j])
                s_lo_f, s_hi_f = _unpack_bf16x2(sh_w[j])
                summed.append(_pack_bf16x2(_add(s_lo_f, r_lo), _add(s_hi_f, r_hi)))
            local = summed
        # After an earlier error on this rank, publish NaN until the reset.
        local = [_poison(_sanitize(w), sticky) for w in local]

        # 2. Publish through multicast into slot [row][rank].
        if cutlass.const_expr(self.fault_delay_ns > 0):
            _fault_delay(
                tidx + cta * Int32(1024),
                row + rank * Int32(64),
                epoch * Int32(4),
                self.fault_delay_ns,
                0,
            )
        slot = (
            Int64(cur) * Int64(buffer_bytes)
            + Int64(row) * Int64(world * hidden * 2)
            + Int64(rank) * Int64(hidden * 2)
            + Int64(col) * Int64(2)
        )
        _st_v4(mc + slot, local[0], local[1], local[2], local[3])
        _flag_error(err_cur, out_of_range, Int32(4))
        _flag_error(err_cur, sticky, sticky)

        # 3. Clear the previous call's Lamport rows and partial slice.
        if cutlass.const_expr(self.fault_delay_ns > 0):
            _fault_delay(
                tidx + cta * Int32(1024),
                row + rank * Int32(64),
                epoch * Int32(4) + Int32(1),
                self.fault_delay_ns,
                1,
            )
        total_threads = num_rows * Int32(cluster * threads)
        gtid = (row * Int32(cluster) + cta) * Int32(threads) + tidx
        clear_vecs = prev_rows * Int32(world * hidden // 8)
        sentinel = Int32(_SENTINEL_WORD)
        dirty_base = uc + Int64(dirty) * Int64(buffer_bytes)
        for v in cutlass.range(gtid, clear_vecs, total_threads):
            _st_v4(dirty_base + Int64(v) * Int64(16), sentinel, sentinel, sentinel, sentinel)
        part_vecs = prev_rows * Int32(MAX_CHUNKS // 4)
        part_base = partials + Int64(dirty) * Int64(self.max_rows * MAX_CHUNKS * 4)
        fp32_sentinel = Int32(_FP32_SENTINEL)
        for v in cutlass.range(gtid, part_vecs, total_threads):
            _st_v4(
                part_base + Int64(v) * Int64(16),
                fp32_sentinel,
                fp32_sentinel,
                fp32_sentinel,
                fp32_sentinel,
            )

        # 4. Per-thread Lamport poll, rank-ordered fp32 sum.
        poll_base = (
            uc
            + Int64(cur) * Int64(buffer_bytes)
            + Int64(row) * Int64(world * hidden * 2)
            + Int64(col) * Int64(2)
        )
        polled = _poll8(poll_base, Int64(hidden * 2), timeout_ns)
        timed_out = polled[32]
        _flag_timeout(err_cur, state, timed_out, polled[33])
        reduced = [Float32(0.0)] * 8
        for r in cutlass.range_constexpr(8):
            for j in cutlass.range_constexpr(4):
                v_lo, v_hi = _unpack_bf16x2(polled[4 * r + j])
                reduced[2 * j] = _add(reduced[2 * j], v_lo)
                reduced[2 * j + 1] = _add(reduced[2 * j + 1], v_hi)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # 5. Residual added once, after the reduction; NaN if this poll timed out.
        new_res = []
        x = []
        for j in cutlass.range_constexpr(4):
            rb = _pack_bf16x2(reduced[2 * j], reduced[2 * j + 1])
            a_lo, a_hi = _unpack_bf16x2(rb)
            r_lo, r_hi = _unpack_bf16x2(res_w[j])
            word = _poison(_pack_bf16x2(_add(a_lo, r_lo), _add(a_hi, r_hi)), timed_out)
            new_res.append(word)
            n_lo, n_hi = _unpack_bf16x2(word)
            x += [n_lo, n_hi]
        _st_v4(residual_out + row_off, new_res[0], new_res[1], new_res[2], new_res[3])

        # 6. RMSNorm with a cluster-wide square sum.
        thread_sum = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            thread_sum = _add(thread_sum, _round_bf16(_mul(x[i], x[i])))
        for mask in (16, 8, 4, 2, 1):
            thread_sum = _add(thread_sum, _shfl_xor(thread_sum, mask))
        cute.arch.cluster_wait()
        my_slot = reduce_addr + (cta * Int32(warps) + warp) * Int32(4)
        _st_cluster_f32(my_slot, lane, thread_sum, Int32(lane < Int32(cluster)))
        cute.arch.cluster_arrive()
        if cutlass.const_expr(not self.race_probe):
            cute.arch.cluster_wait()
        square_sum = Float32(0.0)
        for i in cutlass.range_constexpr(cluster * warps):
            square_sum = _add(square_sum, _ld_shared_f32(reduce_addr + Int32(4 * i)))
        if cutlass.const_expr(self.race_probe):
            cute.arch.cluster_wait()
        inv_rms = _inv_rms(square_sum, Float32(float(hidden)), eps)
        out_w = []
        for j in cutlass.range_constexpr(4):
            g_lo, g_hi = _unpack_bf16x2(gamma_w[j])
            out_w.append(
                _pack_bf16x2(
                    _mul(_mul(x[2 * j], inv_rms), g_lo), _mul(_mul(x[2 * j + 1], inv_rms), g_hi)
                )
            )
        _st_v4(out + row_off, out_w[0], out_w[1], out_w[2], out_w[3])

        # 7. Arrival; the last CTA publishes the next epoch.
        if cutlass.const_expr(self.fault_delay_ns > 0):
            _fault_delay(
                tidx + cta * Int32(1024),
                row + rank * Int32(64),
                epoch * Int32(4) + Int32(2),
                self.fault_delay_ns,
                2,
            )
        cute.arch.sync_threads()
        _arrive(
            state,
            Int32(tidx == Int32(0)),
            num_rows * Int32(cluster),
            num_rows,
            (epoch + Int32(1)) % Int32(3),
        )


_COMPILED = {}


def compiled_cluster_hop(
    hidden: int,
    top_k: int,
    world: int,
    max_rows: int,
    cluster: int,
    has_shared: bool,
    scale_fp32: bool,
    use_pdl: bool,
    state_words: int,
    race_probe: bool = False,
    fault_delay_ns: int = 0,
    routed_input=False,
):
    """Return the compiled kernel for one shape, compiling it on first use (no GPU work).

    ``routed_input`` selects a routed_moe_into_hop variant (see ``PostMoeHopClusterKernel``); the
    default variants keep their key.
    """
    mode = "row" if routed_input is True else routed_input
    key = (
        hidden,
        top_k,
        world,
        max_rows,
        cluster,
        has_shared,
        scale_fp32,
        use_pdl,
        state_words,
        race_probe,
        fault_delay_ns,
    ) + ((mode,) if mode else ())
    compiled = _COMPILED.get(key)
    if compiled is None:
        from cutlass.cute import runtime as cute_runtime

        kernel = PostMoeHopClusterKernel(
            hidden,
            top_k,
            world,
            max_rows,
            cluster,
            has_shared,
            scale_fp32,
            use_pdl,
            race_probe,
            fault_delay_ns,
            mode,
        )
        zero64, zero32 = Int64(0), Int32(0)
        state_fake = cute_runtime.make_fake_compact_tensor(
            Int32, (state_words,), stride_order=(0,), assumed_align=16
        )
        compiled = cute.compile(
            kernel,
            zero64,
            zero64,
            zero32,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            state_fake,
            zero32,
            Float32(0.0),
            zero64,
            zero32,
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


def launch_cluster_hop(
    *,
    fc2_output,
    expert_scale_factor,
    expanded_idx_to_permuted_idx,
    shared_output,
    residual,
    norm_weight,
    out,
    residual_out,
    mc_buffer,
    uc_buffer,
    partials,
    state,
    rank: int,
    world: int,
    max_rows: int,
    eps: float,
    cluster: int,
    use_pdl: bool,
    timeout_ns: int,
    race_probe: bool = False,
    fault_delay_ns: int = 0,
    fc2_rows: Optional[int] = None,
) -> None:
    """Launch the cluster kernel on the current stream (compiled once per shape).

    Every tensor must have a 16-byte aligned base, ``fc2_output`` a 16-byte
    row pitch, and the other tensors must be contiguous (``hop.py`` enforces
    this). ``race_probe`` and ``fault_delay_ns`` select test-only
    fault-injected variants (see ``PostMoeHopClusterKernel``); ``fc2_rows`` overrides
    the permutation bound for the memcheck positive control only. The hop
    always launches the regular kernel.
    """
    num_rows, hidden = residual.shape
    top_k = expanded_idx_to_permuted_idx.shape[1]
    has_shared = shared_output is not None
    scale_fp32 = expert_scale_factor.dtype.itemsize == 4
    compiled = compiled_cluster_hop(
        hidden,
        top_k,
        world,
        max_rows,
        cluster,
        has_shared,
        scale_fp32,
        use_pdl,
        state.numel(),
        race_probe,
        fault_delay_ns,
    )
    compiled(
        fc2_output.data_ptr(),
        fc2_output.stride(0) * 2,
        fc2_output.shape[0] if fc2_rows is None else fc2_rows,
        expert_scale_factor.data_ptr(),
        expanded_idx_to_permuted_idx.data_ptr(),
        shared_output.data_ptr() if has_shared else 0,
        residual.data_ptr(),
        norm_weight.data_ptr(),
        out.data_ptr(),
        residual_out.data_ptr(),
        mc_buffer.data_ptr(),
        uc_buffer.data_ptr(),
        partials.data_ptr(),
        state,
        rank,
        eps,
        timeout_ns,
        num_rows,
    )


def launch_cluster_hop_routed(
    *,
    routed,
    shared_output,
    residual,
    norm_weight,
    out,
    residual_out,
    mc_buffer,
    uc_buffer,
    partials,
    state,
    rank: int,
    world: int,
    max_rows: int,
    eps: float,
    cluster: int,
    use_pdl: bool,
    timeout_ns: int,
    top_k: int = 8,
    mode: str = "row",
    scales=None,
    race_probe: bool = False,
    fault_delay_ns: int = 0,
) -> None:
    """routed_moe_into_hop entry: the cluster kernel on a routed input prepared by routed_moe.

    Same workspace protocol and post-finalize arithmetic as the regular hop
    (``bf16(routed + shared)``, rank-ordered FP32 sum, residual, RMSNorm);
    only the finalize gather of the raw tuple changes with ``mode`` (see
    ``PostMoeHopClusterKernel``): "row" loads the finalized bf16 row, "identity"
    gathers the identity-layout raw rows with ``scales``, "fp32" rounds the
    FP32 accumulator and clears it. The buffers are hop-owned (16-byte
    aligned, contiguous); ``race_probe`` and ``fault_delay_ns`` select the
    test-only fault-injected builds, as for ``launch_cluster_hop``.
    """
    num_rows, hidden = residual.shape
    has_shared = shared_output is not None
    compiled = compiled_cluster_hop(
        hidden,
        top_k,
        world,
        max_rows,
        cluster,
        has_shared,
        False,
        use_pdl,
        state.numel(),
        race_probe,
        fault_delay_ns,
        mode,
    )
    compiled(
        routed.data_ptr(),
        routed.stride(0) * routed.element_size(),
        routed.shape[0],
        scales.data_ptr() if scales is not None else 0,
        0,
        shared_output.data_ptr() if has_shared else 0,
        residual.data_ptr(),
        norm_weight.data_ptr(),
        out.data_ptr(),
        residual_out.data_ptr(),
        mc_buffer.data_ptr(),
        uc_buffer.data_ptr(),
        partials.data_ptr(),
        state,
        rank,
        eps,
        timeout_ns,
        num_rows,
    )


def cluster_kernel_supported(hidden: int, top_k: int, world: int, cluster: int) -> Optional[str]:
    """Return why the cluster kernel cannot serve a shape, or ``None``."""
    if top_k != 8 or world != 8:
        return "cluster kernel specialized to top-8 / TP8"
    if cluster < 1 or cluster > 8 or hidden % (8 * 32 * cluster):
        return f"hidden {hidden} does not split into {cluster} CTAs of whole warps"
    if hidden // (8 * cluster) > 1024:
        return "too many threads per CTA"
    return None
