# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DFlash 2 drafter hop in one launch: all-reduce, block convolution, residual add, RMSNorm.

Each drafter layer has two hops: after the row-parallel ``o_proj`` and after
the row-parallel ``down_proj``. The built-in step runs three kernels at each:
the MNNVL one-shot all-reduce of the projection's BF16 partial
(``mnnvl::oneshotAllreduceFusionKernel``), the DFlash 2 block convolution of
its output (``finish`` side, ``ops/spec/dflash2_block_conv.py``), then
flashinfer's fused residual add + RMSNorm of the next norm
(``FusedAddRMSNormKernel``, in place on both operands). This entry is the three
in one kernel, bit for bit:

* all-reduce: every rank's partial summed in FP32 from ``+0`` in ascending
  rank order, rounded once to BF16 (the one-shot's arithmetic; its ``-0``
  sanitization does not change a sum that starts at ``+0``);
* convolution: ``dflash2_block_conv``'s statements -- each coefficient, tap
  product, mask product and running sum rounded to BF16 where it rounds,
  unfused FP32 add / multiply in between;
* add + RMSNorm: flashinfer's CuTe-DSL ``FusedAddRMSNormKernel`` (what the
  built-in ``RMSNorm`` reaches on this GPU), statement for statement as its
  PTX has them for a 8192-wide BF16 row: ``h = x + r``; per thread one FMA
  chain ``s = fma(h, h, s)`` from ``+0`` over its 64 columns
  (``8 * tid + 1024 * b + e``, ``b`` then ``e`` ascending); a butterfly over
  each warp (lane offsets 1, 2, 4, 8, 16); the same butterfly over the 4 warp
  sums padded with zeros; ``rstd = rsqrt.approx.ftz(fma(s, 2**-13, eps))``;
  ``y = (w + 0) * (h * rstd)``. All IEEE round-to-nearest, no flush to zero
  except the ``rsqrt``.

Every floating-point operation is an inline PTX instruction, so the compiler
can neither contract, reorder nor change the flush mode of any of them.

Thread map. A row is one cluster of 8 CTAs of 128 threads; CTA ``c`` owns
flashinfer threads ``f = 16 c .. 16 c + 15`` whole: its thread ``j`` takes
block ``b = j mod 8`` of flashinfer thread ``f = 16 c + j / 8``, the 8 columns
at ``8 f + 1024 b``. The all-reduce, the convolution and ``h`` are computed
there; the 8 threads of one ``f`` leave their ``h`` in CTA shared memory, 16
threads run the 16 FMA chains, and each CTA sends its 16 chain sums to every
CTA of the cluster through distributed shared memory (64 bytes per CTA pair).
Every CTA then runs both butterflies on the 128 sums -- lane ``l`` of warp
``w`` holds flashinfer thread ``32 w + l``'s -- and reaches the same ``rstd``
on flashinfer's tree.

Transport (one-shot Lamport over the NVLink fabric). The convolution reads the
reduced rows ``t`` and ``t - 1``, so a row's slot has two readers (the CTAs of
rows ``t`` and ``t + 1``) and cannot be reset by its reader as it is read. The
workspace therefore holds one receive area per hop site (``sites``; the drafter
calls the sites in the same order on every rank and step), each ``8 source
ranks x 8 rows x 8192`` BF16, every 32-bit word initialized to the sentinel
``0xFFFEFFFE`` (two BF16 NaNs with a payload no arithmetic produces). A slot row
holds its 8192 values thread-major (thread ``j`` of CTA ``c`` at ``8 (128 c +
j)``), so each warp's accesses to it are 512 contiguous bytes; on a ``direct``
workspace it holds them in column order, as the GEMM stores its output. A call at
site ``k``:

1. publishes this rank's partial row into slot ``rank`` of area ``k`` on every
   rank with one multicast store (or, with ``direct``, finds it there already:
   the projection GEMM stored its output through the multicast address);
2. resets every row of every peer slot of area ``k - 1`` (mod ``sites``) to the
   sentinel;
3. polls the peer slots of rows ``t`` and ``t - 1`` of area ``k`` until no word
   holds the sentinel -- stale-only: after one read of every slot only the
   slots still holding a sentinel word are read again, all in flight together.

Why the reset is safe: area ``k - 1`` was last read by this rank's previous
call, which completed before this call's dependency wait returned; a peer
writes it next in its call ``sites`` calls later, which it can reach only after
this rank published every call in between, the first of which is this one. So
every area is clean before it is written and is written only after it is read,
for ``sites >= 3``. Calls of different row counts interleave (every row of an
area is reset). There is no device-side buffer index: graph replays and eager
calls interleave freely, and a GEMM captured into a graph can store into a
fixed area.

Wide workspace (``max_rows`` above 8, fast path ``wide_drafter_hops``): the same kernel on
drafter blocks of up to ``WIDE_MAX_ROWS`` rows (several requests' blocks; the convolution's
tap-1 mask is per block, ``row % block``), with one slab per rank of the group and ``areas``
receive areas used in site order modulo ``areas`` -- a divisor of ``sites`` of at least 3, so
every window of ``areas`` consecutive calls (across forwards too) names distinct areas and
the reset argument above holds with ``areas`` in place of ``sites``. A call resets the rows of
the previous area that the previous call can have written: all ``max_rows`` at site 0 (the
previous forward may have had any row count), at every other site the larger of its own row
count and the previous call's (a forward's calls share one row count when served).

Progress: a wide call launches ``CLUSTER * rows`` CTAs of 128 threads. A cluster publishes its
row before it polls, and polls only the slots of rows ``t`` and ``t - 1`` of its column slice,
which only the clusters of rows ``t`` and ``t - 1`` of the peers write. No other compute
stream of the step is live during the drafter forward (the ``context_kv_fork`` side stream
joins the main stream before the drafter's first hop), so no kernel of this rank holds an SM
the hop's clusters wait for. Up to 32 rows (32 clusters = 256 CTAs) every cluster of the call
is resident together on a GB300 (at most two 128-thread CTAs per SM), so progress needs no
dispatch order. Above that (sm103's served 56 rows are 448 CTAs on 152 SMs, sm107's served 64
rows are 512 CTAs on 212 SMs, and the qualification tests' 64..256 rows) not every cluster
need be resident at once, and progress
also relies on CTAs dispatched in grid order on every rank (the assumption the built-in MNNVL
all-reduce kernels and ``post_attn_hop_wide`` make, which CUDA does not guarantee): the
lowest-indexed unfinished cluster is then resident on every rank once every lower one has
finished, so it publishes and finishes.

Ordering contract: calls on one workspace must be issued in site order
``0, 1, ..., sites - 1, 0, ...`` on every rank, serialized on one stream (graph
replay included), and every kernel launched with programmatic dependent launch
between two calls must wait on its predecessor before it completes -- the
contract the built-in MNNVL all-reduce relies on as well. A caller that knows
its forwards marks each start (``begin_forward``), which lets a forward whose
graph capture failed part way give the site order back.

Memory order around the dependency wait: only the norm weight and the
convolution's base kernel (model parameters, written at load time only) are
read before ``griddepcontrol.wait``; the partial, the residual, the
convolution coefficients and every transport access come after it. So
``base`` and ``weight`` must be the parameters themselves, never a tensor a
kernel just wrote. With the early trigger the hop's dependents may start
before its output and residual stores: a kernel launched after it with
programmatic dependent launch must read those only after its own wait.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: The kernels launch with programmatic dependent launch: ``TRTLLM_ENABLE_PDL`` on is a
#: precondition of every composition that builds them.
_LAUNCH_PDL = True

#: Hidden width the thread map is written for.
HIDDEN = 8192
#: Rows of a drafter block the transport is sized for (K + 1 at K = 7).
MAX_ROWS = 8
#: Rows of a wide workspace: the drafter blocks of up to 32 requests at K = 7.
WIDE_MAX_ROWS = 256
#: Rows of the served wide workspace (``wide_drafter_hops``) per compute capability
#: (``wide_served_rows``). sm103: the drafter blocks of 2..7 requests at K = 7, where the
#: one-shot hop is faster than the built-in all-reduce, convolution and add-RMSNorm on
#: GB300 (at 8 requests the two tie; above, the built-in two-shot all-reduce wins). sm107:
#: the blocks of 2..8 requests; on Rubin the one-shot hop is still the faster one at 64 rows
#: and the slower one from 128 rows on.
WIDE_SERVED_ROWS_BY_SM: dict[tuple[int, int], int] = {(10, 3): 56, (10, 7): 64}


def wide_served_rows(capability: Optional[Tuple[int, int]] = None) -> int:
    """The served wide workspace's rows on ``capability`` (this process's device when None).

    Raises:
        ValueError: on a device ``WIDE_SERVED_ROWS_BY_SM`` does not list.
    """
    key = tuple(torch.cuda.get_device_capability() if capability is None else capability)
    rows = WIDE_SERVED_ROWS_BY_SM.get(key)
    if rows is None:
        raise ValueError(
            f"drafter_hop: no served wide rows for SM {key} "
            f"(supported: {sorted(WIDE_SERVED_ROWS_BY_SM)})"
        )
    return rows


#: Largest tensor-parallel group; slots are laid out for it whatever the group size.
MAX_RANKS = 8
#: BF16 values per thread.
VECTOR = 8
#: CTAs per row and threads per CTA: 8 x 128 threads x 8 columns = 8192.
CLUSTER = 8
THREADS = 128
WARPS = THREADS // 32
assert CLUSTER * THREADS * VECTOR == HIDDEN
#: flashinfer threads per CTA: a CTA owns all 8 blocks of each of them.
OWNERS = THREADS // CLUSTER
#: FP32 values of one flashinfer thread's ``h`` in shared memory, padded so that the 16
#: chain readers hit distinct bank quads (2-way conflicts at most).
OWNER_STRIDE = CLUSTER * VECTOR + 4
#: Shared memory: the CTA's ``h`` (16 padded rows of 64), its 16 chain sums, all 128 chain
#: sums of the row (8 CTAs x 16, through distributed shared memory), the 4 warp sums, the
#: mbarrier, alignment slack.
H_BYTES = OWNERS * OWNER_STRIDE * 4
SUMS_BYTES = THREADS * 4
SMEM_BYTES = H_BYTES + OWNERS * 4 + SUMS_BYTES + WARPS * 4 + 8 + 64
#: The DFlash 2 convolution geometry this kernel takes.
TAPS = 2
GROUP_SIZE = 16
#: A 32-bit word no BF16 pair produced by arithmetic equals (two non-canonical NaNs).
SENTINEL = 0xFFFEFFFE
ROW_BYTES = HIDDEN * 2
SLAB_BYTES = MAX_ROWS * ROW_BYTES
AREA_BYTES = MAX_RANKS * SLAB_BYTES
#: Fewest sites a workspace may have (the reset argument needs 3).
MIN_SITES = 3


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    HAS_CUTE_DSL = False
else:
    HAS_CUTE_DSL = True


if HAS_CUTE_DSL:
    _FOUR_WORDS = "!llvm.struct<(i32,i32,i32,i32)>"

    def _asm(result_type, operands, text, constraints, loc, ip, side_effects=True):
        return llvm.inline_asm(
            result_type,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    def _u32(value, loc, ip):
        return Uint32(value).ir_value(loc=loc, ip=ip)

    def _i64(value, loc, ip):
        return Int64(value).ir_value(loc=loc, ip=ip)

    def _f32(value, loc, ip):
        return Float32(value).ir_value(loc=loc, ip=ip)

    def _split_words(result, loc, ip):
        return tuple(
            Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
        )

    # ---- global memory (every access is a side-effecting asm statement, so none
    # ---- can move above griddepcontrol.wait or merge across polling rounds) ----

    @dsl_user_op
    def _load_x4(address, *, loc=None, ip=None):
        """Coherent 16-byte global load."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip)],
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _load_x4_if(address, enabled, *, loc=None, ip=None):
        """Coherent 16-byte global load when ``enabled``; else four zero words, no access."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip), _u32(enabled, loc, ip)],
            "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\n"
            "mov.b32 a, 0;\nmov.b32 b, 0;\nmov.b32 c, 0;\nmov.b32 d, 0;\n"
            "setp.ne.u32 p, $5, 0;\n"
            "@p ld.global.v4.b32 {a, b, c, d}, [$4];\n"
            "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}",
            "=r,=r,=r,=r,l,r",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _load_u16(address, *, loc=None, ip=None):
        """One BF16 value (zero-extended into the low half of a word)."""
        result = _asm(
            T.i32(),
            [_i64(address, loc, ip)],
            "ld.global.u16 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
        return Uint32(result)

    @dsl_user_op
    def _poll_x4(address, enabled, o0, o1, o2, o3, *, loc=None, ip=None):
        """Relaxed system-scope 16-byte load when ``enabled``; else the given words, no access."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip), _u32(enabled, loc, ip)]
            + [_u32(w, loc, ip) for w in (o0, o1, o2, o3)],
            "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\n"
            "mov.b32 a, $6;\nmov.b32 b, $7;\nmov.b32 c, $8;\nmov.b32 d, $9;\n"
            "setp.ne.u32 p, $5, 0;\n"
            "@p ld.relaxed.sys.global.v4.b32 {a, b, c, d}, [$4];\n"
            "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}",
            "=r,=r,=r,=r,l,r,r,r,r,r",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _reset_x4(address, enabled, *, loc=None, ip=None):
        """Write the sentinel into one 16-byte slot when ``enabled``."""
        _asm(
            None,
            [_i64(address, loc, ip), _u32(enabled, loc, ip)],
            "{\n.reg .pred p;\n.reg .b32 s;\nmov.b32 s, 0xFFFEFFFE;\n"
            "setp.ne.u32 p, $1, 0;\n"
            "@p st.global.v4.b32 [$0], {s, s, s, s};\n}",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _store_x4(address, w0, w1, w2, w3, *, loc=None, ip=None):
        """Plain 16-byte global store."""
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _stale_bit(w0, w1, w2, w3, bit, *, loc=None, ip=None):
        """``bit`` while any of the four words is the sentinel, else 0, without a branch."""
        result = _asm(
            T.i32(),
            [_u32(w, loc, ip) for w in (w0, w1, w2, w3, bit)],
            "{\n.reg .pred p0, p1, p2, p3;\n"
            "setp.eq.u32 p0, $1, 0xFFFEFFFE;\nsetp.eq.u32 p1, $2, 0xFFFEFFFE;\n"
            "setp.eq.u32 p2, $3, 0xFFFEFFFE;\nsetp.eq.u32 p3, $4, 0xFFFEFFFE;\n"
            "or.pred p0, p0, p1;\nor.pred p2, p2, p3;\nor.pred p0, p0, p2;\n"
            "selp.b32 $0, $5, 0, p0;\n}",
            "=r,r,r,r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None):
        """The global nanosecond timer (bounded-poll variant only)."""
        result = _asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip)
        return Int64(result)

    @dsl_user_op
    def _trap_if(expired, *, loc=None, ip=None):
        """Abort this rank's context when ``expired`` is non-zero (bounded-poll variant only)."""
        _asm(
            None,
            [_u32(expired, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p trap;\n}",
            "r",
            loc,
            ip,
        )

    # ---- arithmetic: one PTX instruction each, as the replaced kernels run them ----

    def _f32_asm(text, operands, loc, ip):
        result = _asm(
            T.f32(),
            [_f32(x, loc, ip) for x in operands],
            text,
            "=f" + ",f" * len(operands),
            loc,
            ip,
            side_effects=False,
        )
        return Float32(result)

    @dsl_user_op
    def _add_rn(a, b, *, loc=None, ip=None):
        """IEEE FP32 add (the one-shot all-reduce's accumulation, the convolution's adds)."""
        return _f32_asm("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)

    @dsl_user_op
    def _mul_rn(a, b, *, loc=None, ip=None):
        """IEEE FP32 multiply (the convolution's products)."""
        return _f32_asm("mul.rn.f32 $0, $1, $2;", (a, b), loc, ip)

    @dsl_user_op
    def _fma_rn(a, b, c, *, loc=None, ip=None):
        """FP32 ``a * b + c``, one rounding (the norm's sum of squares and mean)."""
        return _f32_asm("fma.rn.f32 $0, $1, $2, $3;", (a, b, c), loc, ip)

    @dsl_user_op
    def _rsqrt_ftz(a, *, loc=None, ip=None):
        """``cute.math.rsqrt(..., fastmath=True)`` as compiled: ``rsqrt.approx.ftz.f32``."""
        return _f32_asm("rsqrt.approx.ftz.f32 $0, $1;", (a,), loc, ip)

    @dsl_user_op
    def _round_bf16(a, *, loc=None, ip=None):
        """``a`` rounded to BF16 (nearest even) and widened back, exactly."""
        return _f32_asm(
            "{\n.reg .b16 r;\ncvt.rn.bf16.f32 r, $1;\ncvt.f32.bf16 $0, r;\n}", (a,), loc, ip
        )

    @dsl_user_op
    def _keep(position, *, loc=None, ip=None):
        """1.0 when ``position >= 1``, else 0.0 (the convolution's tap-1 mask)."""
        result = _asm(
            T.f32(),
            [Int32(position).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ge.s32 p, $1, 1;\nselp.f32 $0, 0f3F800000, 0f00000000, p;\n}",
            "=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return Float32(result)

    @dsl_user_op
    def _shfl_bfly(value, lane_mask, *, loc=None, ip=None):
        """``shfl.sync.bfly.b32`` over the full warp (flashinfer's ``shfl_xor_sync``)."""
        result = _asm(
            T.f32(),
            [_f32(value, loc, ip), _u32(lane_mask, loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
        )
        return Float32(result)

    @dsl_user_op
    def _float_bits(value, *, loc=None, ip=None):
        """The int32 with the same bit pattern as an FP32 value."""
        bits = _asm(
            T.i32(),
            [_f32(value, loc, ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            loc,
            ip,
            side_effects=False,
        )
        return Int32(bits)

    @dsl_user_op
    def _unpack_bf16x2(word, *, loc=None, ip=None):
        """FP32 values of the low and high BF16 halves of ``word`` (exact)."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(f32,f32)>"),
            [_u32(word, loc, ip)],
            "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
            "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
            "=f,=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def _bf16_bits_to_f32(half, *, loc=None, ip=None):
        """FP32 value of a BF16 held in the low 16 bits of ``half`` (exact)."""
        result = _asm(
            T.f32(),
            [_u32(half, loc, ip)],
            "shl.b32 $0, $1, 16;",
            "=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return Float32(result)

    @dsl_user_op
    def _pack_bf16x2(low, high, *, loc=None, ip=None):
        """Two FP32 values rounded to BF16 (nearest even), ``low`` in the low half."""
        result = _asm(
            T.i32(),
            [_f32(high, loc, ip), _f32(low, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    class _DrafterHopKernel:
        """All-reduce + DFlash 2 finish convolution + flashinfer add + RMSNorm, one row per cluster."""

        def __init__(
            self,
            ranks: int,
            eps: float,
            pdl: bool,
            early_trigger: bool,
            direct: bool,
            max_rows: int = MAX_ROWS,
            timeout_ns: int = 0,
        ):
            self.ranks = ranks
            self.eps = float(eps)
            self.pdl = pdl
            self.early_trigger = early_trigger
            self.direct = direct
            # Rows of one slab (one source rank's rows of an area) and its bytes.
            self.max_rows = int(max_rows)
            self.slab_bytes = self.max_rows * ROW_BYTES
            # 0: the polls spin without a bound (served); else a poll still stale after this
            # many nanoseconds executes ``trap`` (qualification builds).
            self.timeout_ns = int(timeout_ns)

        @cute.jit
        def __call__(
            self,
            partial: Int64,
            residual: Int64,
            delta: Int64,
            base: Int64,
            weight: Int64,
            output: Int64,
            receive: Int64,
            previous: Int64,
            mcast: Int64,
            rows: Int32,
            rank: Int32,
            block: Int32,
            delta_row: Int32,
            delta_tap: Int32,
            dry: Int32,
            reset_rows: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial,
                residual,
                delta,
                base,
                weight,
                output,
                receive,
                previous,
                mcast,
                rows,
                rank,
                block,
                delta_row,
                delta_tap,
                dry,
                reset_rows,
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
            partial: Int64,
            residual: Int64,
            delta: Int64,
            base: Int64,
            weight: Int64,
            output: Int64,
            receive: Int64,
            previous: Int64,
            mcast: Int64,
            rows: Int32,
            rank: Int32,
            block: Int32,
            delta_row: Int32,
            delta_tap: Int32,
            dry: Int32,
            reset_rows: Int32,
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
            # A dry launch only makes the driver load the module.
            if dry == 0:
                if tid == 0:
                    cute.arch.mbarrier_init(bars.iterator, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator, SUMS_BYTES)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

                # Model parameters (written at load time only) before the wait.
                column_bytes = Int64(column) * 2
                w = _load_x4(weight + column_bytes)
                b0 = _load_x4(base + column_bytes)
                b1 = _load_x4(base + Int64(ROW_BYTES) + column_bytes)
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                has_prev = Uint32(row != 0)
                row_offset = Int64(row) * ROW_BYTES + column_bytes
                # Where this thread's 8 values sit in a slot row: thread-major, so that a
                # warp's publish, resets and polls cover 512 contiguous bytes; in column
                # order on a direct workspace, where the GEMM stores the row as it is.
                slot_bytes = column_bytes
                if cutlass.const_expr(not self.direct):
                    slot_bytes = Int64((crank * THREADS + tid) * VECTOR) * 2
                slot_offset = Int64(row) * ROW_BYTES + slot_bytes
                own = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                own_prev = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                if cutlass.const_expr(not self.direct):
                    own = _load_x4(partial + row_offset)
                    own_prev = _load_x4_if(partial + row_offset - ROW_BYTES, has_prev)
                    # Publish this rank's partial row into its slot on every rank
                    # (one store to the multicast address reaches every member).
                    _store_x4(
                        mcast + Int64(rank) * self.slab_bytes + slot_offset,
                        own[0],
                        own[1],
                        own[2],
                        own[3],
                    )
                # Retire the previous site's area: every row, every slot this
                # rank polls (rows t, t + rows, ... of this CTA's columns).
                if cutlass.const_expr(self.max_rows == MAX_ROWS):
                    for i in cutlass.range_constexpr(MAX_ROWS):
                        reset_row = row + i * rows
                        in_area = Uint32(reset_row < MAX_ROWS)
                        for p in cutlass.range_constexpr(self.ranks):
                            reset_on = Uint32(1)
                            if cutlass.const_expr(not self.direct):
                                reset_on = Uint32(rank != p)
                            _reset_x4(
                                previous
                                + Int64(p) * self.slab_bytes
                                + Int64(reset_row) * ROW_BYTES
                                + slot_bytes,
                                in_area & reset_on,
                            )
                else:
                    # A wide area: the rows the previous call can have written
                    # (``reset_rows``), rows t, t + rows, ... of this CTA's columns.
                    for reset_row in cutlass.range(row, reset_rows, rows, unroll=1):
                        for p in cutlass.range_constexpr(self.ranks):
                            reset_on = Uint32(1)
                            if cutlass.const_expr(not self.direct):
                                reset_on = Uint32(rank != p)
                            _reset_x4(
                                previous
                                + Int64(p) * self.slab_bytes
                                + Int64(reset_row) * ROW_BYTES
                                + slot_bytes,
                                reset_on,
                            )
                r = _load_x4(residual + row_offset)
                group = Int64(column // GROUP_SIZE)
                delta_address = delta + (Int64(row) * Int64(delta_row) + group) * 2
                d0 = _bf16_bits_to_f32(_load_u16(delta_address))
                d1 = _bf16_bits_to_f32(_load_u16(delta_address + Int64(delta_tap) * 2))

                # Poll rows t (slots 0..ranks-1) and t - 1 (slots ranks..2 ranks-1)
                # of every source slot this rank does not hold itself.
                receive_row = receive + slot_offset
                values = cute.make_rmem_tensor(cute.make_layout((2 * self.ranks, 4)), Uint32)
                enabled = cute.make_rmem_tensor(cute.make_layout((2 * self.ranks,)), Uint32)
                stale = Uint32(0)
                for half in cutlass.range_constexpr(2):
                    for p in cutlass.range_constexpr(self.ranks):
                        slot = half * self.ranks + p
                        on = Uint32(1)
                        if cutlass.const_expr(not self.direct):
                            on = Uint32(rank != p)
                        if cutlass.const_expr(half == 1):
                            on = on & has_prev
                        mine = own if half == 0 else own_prev
                        v0, v1, v2, v3 = _poll_x4(
                            receive_row + Int64(p) * self.slab_bytes - half * ROW_BYTES,
                            on,
                            mine[0],
                            mine[1],
                            mine[2],
                            mine[3],
                        )
                        values[slot, 0] = v0
                        values[slot, 1] = v1
                        values[slot, 2] = v2
                        values[slot, 3] = v3
                        enabled[slot] = on
                        stale = stale | _stale_bit(v0, v1, v2, v3, on << Uint32(slot))
                start = Int64(0)
                if cutlass.const_expr(self.timeout_ns > 0):
                    start = _globaltimer()
                while stale != Uint32(0):
                    polled = stale
                    for half in cutlass.range_constexpr(2):
                        for p in cutlass.range_constexpr(self.ranks):
                            slot = half * self.ranks + p
                            v0, v1, v2, v3 = _poll_x4(
                                receive_row + Int64(p) * self.slab_bytes - half * ROW_BYTES,
                                (polled >> Uint32(slot)) & Uint32(1),
                                values[slot, 0],
                                values[slot, 1],
                                values[slot, 2],
                                values[slot, 3],
                            )
                            values[slot, 0] = v0
                            values[slot, 1] = v1
                            values[slot, 2] = v2
                            values[slot, 3] = v3
                    stale = Uint32(0)
                    for slot in cutlass.range_constexpr(2 * self.ranks):
                        stale = stale | _stale_bit(
                            values[slot, 0],
                            values[slot, 1],
                            values[slot, 2],
                            values[slot, 3],
                            enabled[slot] << Uint32(slot),
                        )
                    if cutlass.const_expr(self.timeout_ns > 0):
                        elapsed = _globaltimer() - start
                        _trap_if(Uint32((stale != Uint32(0)) & (elapsed > Int64(self.timeout_ns))))
                # Early trigger: past the last transport access.
                if cutlass.const_expr(self.pdl and self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

                keep = _keep(row % block)
                x = cute.make_rmem_tensor(cute.make_layout((VECTOR,)), Float32)
                for e in cutlass.range_constexpr(VECTOR):
                    k = e // 2
                    # All-reduce of rows t and t - 1: FP32 from +0 in rank order, one BF16 round.
                    total = Float32(0.0)
                    total_prev = Float32(0.0)
                    for p in cutlass.range_constexpr(self.ranks):
                        total = _add_rn(total, _unpack_bf16x2(values[p, k])[e % 2])
                        total_prev = _add_rn(
                            total_prev, _unpack_bf16x2(values[self.ranks + p, k])[e % 2]
                        )
                    current = _round_bf16(total)
                    shifted = _round_bf16(total_prev)
                    # The finish-side block convolution, rounded where dflash2_block_conv rounds.
                    coefficient = _round_bf16(_add_rn(_unpack_bf16x2(b0[k])[e % 2], d0))
                    acc = _round_bf16(_mul_rn(coefficient, current))
                    coefficient = _round_bf16(_add_rn(_unpack_bf16x2(b1[k])[e % 2], d1))
                    term = _round_bf16(_mul_rn(coefficient, shifted))
                    term = _round_bf16(_mul_rn(term, keep))
                    acc = _round_bf16(_add_rn(acc, term))
                    # flashinfer's residual add.
                    x[e] = _add_rn(acc, _unpack_bf16x2(r[k])[e % 2])

                # Block ``block_b`` of local flashinfer thread ``owner``: at [owner][block_b].
                for e in cutlass.range_constexpr(VECTOR):
                    h_local[owner * OWNER_STRIDE + block_b * VECTOR + e] = x[e]
                cute.arch.barrier()
                # Local flashinfer thread ``tid``'s FMA chain over blocks 0..7, 8 columns each.
                if tid < OWNERS:
                    chain = Float32(0.0)
                    for k in cutlass.range_constexpr(CLUSTER * VECTOR):
                        value = h_local[tid * OWNER_STRIDE + k]
                        chain = _fma_rn(value, value, chain)
                    chains[tid] = chain
                cute.arch.barrier()
                # The 16 sums of flashinfer threads 16 crank .. 16 crank + 15, to every CTA of
                # the row's cluster (this one included), 4 per store.
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
                    lo = _mul_rn(_add_rn(lo_w, Float32(0.0)), _mul_rn(x[2 * k], rms_rcp))
                    hi = _mul_rn(_add_rn(hi_w, Float32(0.0)), _mul_rn(x[2 * k + 1], rms_rcp))
                    normed.append(_pack_bf16x2(lo, hi))
                    updated.append(_pack_bf16x2(x[2 * k], x[2 * k + 1]))
                _store_x4(residual + row_offset, *updated)
                _store_x4(output + row_offset, *normed)
                # Late trigger: after every store (flashinfer's placement).
                if cutlass.const_expr(self.pdl and not self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=64)
    def _compiled_kernel(
        device_index: int,
        ranks: int,
        eps: float,
        pdl: bool,
        early_trigger: bool,
        direct: bool,
        max_rows: int = MAX_ROWS,
        timeout_ns: int = 0,
    ):
        kernel = _DrafterHopKernel(ranks, eps, pdl, early_trigger, direct, max_rows, timeout_ns)
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *([Int64(0)] * 9),
                Int32(1),
                Int32(0),
                Int32(8),
                Int32(0),
                Int32(0),
                Int32(1),
                Int32(max_rows),
                stream,
                options="--enable-tvm-ffi",
            )


class DrafterHopWorkspace:
    """Transport memory of the fused drafter hop for one tensor-parallel group.

    Private to this op (its protocol is not the built-in MNNVL all-reduce's).
    ``sites`` receive areas of ``AREA_BYTES`` each, one per hop site of a
    drafter forward; calls must name the sites in order ``0 .. sites - 1``,
    repeatedly, on every rank, serialized on one stream.

    ``direct`` builds the poll-only variant: the caller's projection GEMM
    stores its output into ``direct_output(site, rows)`` (this rank's slot of
    the site's area through the multicast address) and the hop reads every
    slot, its own included, from its receive area.

    ``early_trigger`` places the programmatic-dependent-launch trigger right
    after the last transport access instead of after the output stores.

    ``pre_wait`` names the tensors the calls will pass as ``base`` and
    ``weight`` (model parameters): the kernel reads those before its
    dependency wait, so they must not be written while the workspace is in
    use; construction checks that each is on this device, contiguous and
    16-byte aligned.

    ``max_rows`` above ``MAX_ROWS`` builds a wide workspace (module docstring): one slab per
    rank of the group, ``areas`` receive areas (default: the smallest divisor of ``sites`` of
    at least 3), multicast publication only.

    ``timeout_ns`` > 0 builds the bounded-poll variant for qualification: a poll still stale
    after that long executes ``trap`` (aborting this rank's context) instead of spinning.

    Construction is collective over the group (the ``pre_wait`` check, fabric
    allocation, sentinel fill, compilation and a dry launch per ``eps``) and
    either succeeds on every rank or raises on every rank.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        sites: int,
        eps: Sequence[float] = (),
        early_trigger: bool = False,
        direct: bool = False,
        pre_wait: Sequence[torch.Tensor] = (),
        max_rows: int = MAX_ROWS,
        areas: Optional[int] = None,
        timeout_ns: int = 0,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.sites = int(sites)
        self.pdl = _LAUNCH_PDL
        self.early_trigger = bool(early_trigger)
        self.direct = bool(direct)
        if self.ranks not in (4, MAX_RANKS):
            raise ValueError(f"the fused drafter hop serves 4 or 8 ranks, got {self.ranks}")
        if self.sites < MIN_SITES:
            raise ValueError(f"the fused drafter hop needs at least {MIN_SITES} sites")
        self.max_rows = int(max_rows)
        self.wide = self.max_rows > MAX_ROWS
        self.timeout_ns = int(timeout_ns)
        #: Rows of the last call issued (the reset bound of the next one inside a forward).
        self._last_rows = self.max_rows
        if self.wide:
            if self.direct:
                raise ValueError(
                    "a wide drafter hop workspace publishes through the multicast only"
                )
            if not MAX_ROWS < self.max_rows <= WIDE_MAX_ROWS:
                raise ValueError(f"a wide workspace takes {MAX_ROWS + 1}..{WIDE_MAX_ROWS} rows")
            if areas is None:
                areas = next(a for a in range(MIN_SITES, self.sites + 1) if self.sites % a == 0)
            if areas < MIN_SITES or self.sites % areas:
                raise ValueError(
                    f"areas must divide the {self.sites} sites and be at least {MIN_SITES}"
                )
            self.areas = int(areas)
            self.area_bytes = self.ranks * self.max_rows * ROW_BYTES
        else:
            self.areas = self.sites
            self.area_bytes = AREA_BYTES
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._kernels = {}
        self._comm = _get_mnnvl_workspace_comm(mapping)
        size = self.areas * self.area_bytes
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("CuTe DSL is unavailable")
            if torch.cuda.get_device_capability(self.device) not in ((10, 3), (10, 7)):
                raise RuntimeError("the fused drafter hop runs on SM103 or SM107 only")
            for tensor in pre_wait:
                if (
                    tensor.device != self.device
                    or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16
                ):
                    raise ValueError(
                        f"an operand read before the dependency wait ({tuple(tensor.shape)} at "
                        f"{tensor.data_ptr():#x} on {tensor.device}) is not a contiguous, "
                        f"16-byte aligned tensor on {self.device}"
                    )
            use_fabric = mapping.is_multi_node()
            self._handle = _make_mnnvl_mcast_buffer(self._comm, size, mapping, use_fabric)
            words = size // 4
            self._uc = self._handle.get_uc_buffer(self.rank, (words,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            with torch.cuda.device(self.device):
                self._uc.fill_(SENTINEL - (1 << 32))
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                f"drafter hop workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self._uc_base = self._uc.data_ptr()
        self._mc_base = self._mc.data_ptr()
        #: The site the next call must name (host-side check of the ordering contract).
        self.next_site = 0
        #: Whether the forward begun last (``begin_forward``) was being captured.
        self._forward_captured = False
        self.build(eps)

    def begin_forward(self) -> None:
        """A forward starts: its calls name the sites ``0 .. sites - 1``.

        A forward that raised part way leaves ``next_site`` past 0. If it was
        being captured into a CUDA graph (whose capture then failed, so the
        graph is never replayed), none of its calls ran: the order resumes at
        site 0. If it ran eagerly, the area of the last site it called holds
        peers' rows that only the next site's reset would have retired;
        starting over at site 0 would read them as fresh, so this raises.
        """
        if self.next_site != 0:
            if not self._forward_captured:
                raise RuntimeError(
                    f"the previous drafter forward stopped before site {self.next_site} outside "
                    "graph capture: its hop areas are not retired, so the fused hops cannot "
                    "continue on this workspace"
                )
            self.next_site = 0
        self._forward_captured = torch.cuda.is_current_stream_capturing()

    def build(self, eps: Sequence[float]) -> None:
        """Compile and load the kernel of every ``eps`` not built yet. Collective."""
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        error: Optional[BaseException] = None
        try:
            for value in eps:
                self._kernel(float(value))
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "drafter hop kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _kernel(self, eps: float):
        compiled = self._kernels.get(eps)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"drafter hop variant eps={eps} was not built before graph capture")
        compiled = _compiled_kernel(
            self.device.index,
            self.ranks,
            eps,
            self.pdl,
            self.early_trigger,
            self.direct,
            self.max_rows,
            self.timeout_ns,
        )
        dummy = self._uc_base
        compiled(*([dummy] * 9), 1, self.rank, 8, 0, 0, 1, self.max_rows, self._stream())
        self._kernels[eps] = compiled
        return compiled

    def _stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def _area(self, base: int, site: int) -> int:
        return base + (site % self.areas) * self.area_bytes

    def nbytes(self) -> int:
        """Fabric memory this rank holds for the transport."""
        return self.areas * self.area_bytes

    def direct_output(self, site: int, rows: int) -> torch.Tensor:
        """This rank's slot of ``site``'s area through the multicast address, ``[rows, HIDDEN]`` BF16.

        For a ``direct`` workspace: the projection GEMM of that site stores its
        output here (every element once, its final value), and the hop then
        reads it from every rank's receive area.
        """
        if not self.direct or self.wide:
            raise RuntimeError("direct_output is only for a direct (poll-only) workspace")
        start = (site % self.sites) * AREA_BYTES + self.rank * SLAB_BYTES
        flat = self._mc.view(torch.bfloat16)
        return flat[start // 2 : start // 2 + rows * HIDDEN].view(rows, HIDDEN)

    def launch(
        self,
        site: int,
        partial: Optional[torch.Tensor],
        residual: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
    ) -> torch.Tensor:
        rows = residual.shape[0]
        if residual.device != self.device:
            raise ValueError(
                f"drafter hop inputs are on {residual.device}, its workspace on {self.device}"
            )
        if site % self.sites != self.next_site:
            raise RuntimeError(
                f"drafter hop called at site {site} where site {self.next_site} is next: the "
                "sites must be called in order on every rank"
            )
        compiled = self._kernel(float(eps))
        output = torch.empty_like(residual)
        # The rows of the previous area its call can have written: any row count before the
        # first site of a forward (the last forward that ran may not be the last one issued
        # here: a capture that failed, a graph replayed after eager forwards), else the
        # previous call's (wide areas only).
        reset_rows = self.max_rows if site % self.sites == 0 else max(rows, self._last_rows)
        compiled(
            0 if partial is None else partial.data_ptr(),
            residual.data_ptr(),
            delta.data_ptr(),
            base.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            self._area(self._uc_base, site),
            self._area(self._uc_base, site - 1 + self.sites),
            self._area(self._mc_base, site),
            rows,
            self.rank,
            block_size,
            delta.stride(0),
            delta.stride(1),
            0,
            reset_rows,
            self._stream(),
        )
        self.next_site = (site + 1) % self.sites
        self._last_rows = rows
        return output


def hop_takes(
    rows: int,
    residual: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    max_rows: int = MAX_ROWS,
) -> bool:
    """Whether the hop computes this call: rows, dtypes, shapes and devices only.

    Every rank of the group gives the same answer for the same drafter call
    (``rows`` is the drafter block's row count), which is what lets a caller
    choose between this collective and the built-in all-reduce.
    """
    if not 1 <= rows <= max_rows or block_size < 1:
        return False
    if any(t.dtype != torch.bfloat16 for t in (residual, delta, base, weight)):
        return False
    if tuple(residual.shape) != (rows, HIDDEN) or tuple(weight.shape) != (HIDDEN,):
        return False
    if tuple(delta.shape) != (rows, TAPS, HIDDEN // GROUP_SIZE):
        return False
    if tuple(base.shape) != (TAPS, HIDDEN):
        return False
    return all(t.is_cuda and t.device == residual.device for t in (delta, base, weight))


def hop_layout_ok(
    partial: Optional[torch.Tensor],
    residual: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether the kernel's 16-byte vector map and scalar coefficient loads read these as they are."""
    vectors = [residual, base, weight] + ([] if partial is None else [partial])
    return (
        all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in vectors) and delta.stride(-1) == 1
    )


class DrafterHop(OpWrapper):
    """All-reduce a drafter projection's partial, convolve, add the residual, RMSNorm.

    Returns the BF16 normed rows; ``residual`` is updated in place to
    ``bf16(conv(allreduce(partial)) + residual)``, as the built-in fused add +
    RMSNorm updates it. ``partial`` is this rank's BF16 ``[rows, 8192]``
    projection output (None on a ``direct`` workspace, whose GEMM already
    stored it into the site's area); ``delta`` the ``[rows, 2, 512]`` finish
    coefficients (``coefficients[:, 1]`` of the convolution's ``prepare``);
    ``base`` the ``[2, 8192]`` finish base kernel; ``weight`` / ``eps`` the
    next RMSNorm's.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: none. Call sites: the DFlash 2 drafter's two hops per
    layer (after o_proj and after down_proj) in `draft.windowed_block_decode`,
    on drafter blocks of at most 8 rows, in place of the built-in MNNVL
    one-shot all-reduce, the finish-side block convolution and flashinfer's
    CuTe-DSL fused add + RMSNorm of the next norm (`input_layernorm`, then the next
    layer's `post_attention_layernorm`, `model.norm` after the last layer).

    Bit-identical to that chain, gated by its own multi-GPU test (every cell,
    both transports, graph replay with changing inputs, a delayed rank, a long
    burst over the sites). Ranks 4 (one node) and 8 (two nodes). The
    cancelling cell makes the all-reduce's summation order observable.

    Loud preconditions: `hop_takes` (rank-uniform) and `hop_layout_ok`;
    the sites called in order on every rank (the workspace raises otherwise).
    Not certified: rows above 8, widths other than 8192, convolutions other
    than two taps over groups of 16.
    """

    def __call__(
        self,
        site: int,
        partial: Optional[torch.Tensor],
        residual: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
        workspace: DrafterHopWorkspace,
    ) -> torch.Tensor:
        rows = residual.shape[0] if residual.dim() == 2 else 0
        if not hop_takes(
            rows, residual, delta, base, weight, block_size, workspace.max_rows
        ) or not hop_layout_ok(partial, residual, delta, base, weight):
            raise ValueError(
                f"drafter_hop takes BF16 [1..{workspace.max_rows}, 8192] partial / residual, [rows, 2, 512] "
                "coefficients with unit inner stride, a contiguous [2, 8192] base kernel and an "
                "8192 weight, 16-byte aligned"
            )
        if (partial is None) != workspace.direct or (
            partial is not None
            and (tuple(partial.shape) != tuple(residual.shape) or partial.dtype != torch.bfloat16)
        ):
            raise ValueError("drafter_hop: partial must be given unless the workspace is direct")
        return workspace.launch(site, partial, residual, delta, base, weight, eps, block_size)

    def reference(
        self,
        per_rank_partials: Sequence[torch.Tensor],
        residual: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The op's rounding points, float64 in between; returns ``(normed, updated_residual)``.

        The documented collective exception: takes the group's partials in
        ascending rank order. The all-reduce is an FP32 sum from ``+0`` in rank
        order rounded once to BF16; the convolution rounds each coefficient,
        product, mask product and sum to BF16; the residual sum is rounded once
        to BF16 for ``updated``; the norm is float64, rounded once to BF16.
        Does not modify ``residual``.
        """
        summed = torch.zeros_like(per_rank_partials[0], dtype=torch.float32)
        for part in per_rank_partials:
            summed = summed + part.float()
        x = summed.to(torch.bfloat16).float()
        rows = x.shape[0]
        coef = (base.float().unsqueeze(0) + delta.float().repeat_interleave(GROUP_SIZE, -1)).to(
            torch.bfloat16
        )
        position = torch.arange(rows, device=x.device) % block_size
        acc = (coef[:, 0].float() * x).to(torch.bfloat16).float()
        shifted = torch.zeros_like(x)
        shifted[1:] = x[:-1]
        term = (coef[:, 1].float() * shifted).to(torch.bfloat16).float()
        term = term * (position >= 1).float().unsqueeze(-1)
        acc = (acc + term).to(torch.bfloat16).double()
        s = acc + residual.double()
        normed = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + eps) * weight.double()
        return normed.to(torch.bfloat16), s.to(torch.bfloat16)


drafter_hop = DrafterHop()


def log_once_enabled(sites: int, direct: bool, early_trigger: bool) -> None:
    logger.info(
        f"[musespark_open] drafter_hops: fused drafter hop (one-shot AR + DFlash 2 finish conv + add + RMSNorm) at "
        f"{sites} sites per drafter forward (blocks of at most {MAX_ROWS} rows, "
        f"{'direct GEMM publication' if direct else 'multicast publication'}, "
        f"{'early' if early_trigger else 'late'} dependent-launch trigger)"
    )
