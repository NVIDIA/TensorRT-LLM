# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM (Apache-2.0), with TensorRT-LLM changes: the one-group path of WaveMoeTail in
# https://github.com/vllm-project/vllm/blob/dc2137cd398462c0b1bc5367aeb8d0c56c7ef007/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/tail.py
# and the inline-PTX helpers it uses from
# https://github.com/vllm-project/vllm/blob/dc2137cd398462c0b1bc5367aeb8d0c56c7ef007/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/common/primitives.py
"""One-kernel Kimi K3 latent-MoE tail for up to 8 tokens at TP8 (the "wave" tail, after vLLM's
``WaveMoeTail``).

One launch of 112 CTAs (seven 16-CTA clusters, one CTA per SM, 256 threads) replaces the
collective and the up-projection of ``latent_moe_tail`` and their flags protocol:

* at entry every thread loads its fragment of the rank's up-projection shard ``[896, 3584]``
  into registers (cluster ``c`` owns output rows ``128 c ..``, CTA ``q`` the latent columns
  ``224 q ..``), which overlaps the routed experts;
* after ``griddepcontrol.wait`` (FC2 done) it releases its dependents (the reader of the output
  mailbox: the next layer's mix or the copy kernel), cluster 0 does the top-16 weighted sum of
  this rank's FC2 rows and multicasts the latent fragment, and the CTAs behind cluster 0 (3.5
  per token) store the shared-expert partials to their owner ranks (reduce-scatter);
* every cluster polls the eight ranks' latent fragments (the data is the signal: an unpublished
  16-byte chunk holds 0x80000000 words), sums them in rank order in FP32, rounds to BF16, shares
  the sum of squares over the cluster, normalizes and runs the up-projection as BF16 ``mma.sync``
  chains against the weights it already holds;
* each 64-row output line has one owner CTA, which adds the shared shard and multicasts the
  line into the output mailbox (the all-gather).

Unlike vLLM's kernel, which reads latent rows the routed experts publish from their epilogue,
this one does the top-16 sum itself after the experts finish (TRTLLM-Gen's FC2 is precompiled)
and releases its dependents only after that wait, as the other tail kernels do. It also keeps
the rotation of its three buffer slots on the device: ``state`` holds the slot and the token
count of the previous launch, read after the wait and advanced by one thread once every CTA has
read it, so the host passes nothing that depends on the call sequence and CUDA graphs replay
it. Each launch re-arms the slot of the previous launch (complete by then) after its own output
is out.

More than 8 tokens (more than one 8-token MMA group) keep the collective and the up-projection:
neither vLLM's way of handling them (an all-reduce into a local buffer that every cluster then
reads) nor polling the sources directly measured faster on GB300.

Numerics are those of the collective and the up-projection except for the summation order of
the sum of squares and of the up-projection: rank-order FP32 sum of the routed fragments, one
BF16 rounding, BF16-rounded squares, up-projection rounded to BF16 before the BF16 shared shard
is added in FP32.
"""

from collections.abc import Iterable
from functools import partial, wraps

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .primitives import (
    NUM_LAMPORT_BUFFERS,
    finalize_top16_bf16,
    fragment_is_dirty,
    load_global_u32x4,
    load_shared_f32x4,
    load_volatile_u32,
    red_async_release_gpu_add_u32,
    sanitize_negative_zero,
    sanitize_negative_zero_u32,
    stmc_bf16x8,
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)

TP = 8
LATENT, HIDDEN = 3584, 7168
SHARD = HIDDEN // TP
#: Tokens per slot of the two regions (one 8-token MMA group).
CAPACITY = 8
shuffle = cute.arch.shuffle_sync


# --------------------------------------------------------------------------- inline PTX
def ptx(assembly, constraints, operands=(), result=None, *, effects, loc=None, ip=None):
    """One verbatim asm block; a tuple ``result`` (even a 1-tuple) returns a tuple."""
    if isinstance(result, tuple) and len(result) == 1:
        args = (assembly, constraints, operands, result[0])
        return (ptx(*args, effects=effects, loc=loc, ip=ip),)
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([dtype.mlir_type for dtype in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        assembly,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            dtype(llvm.extractvalue(dtype.mlir_type, value, [i], loc=loc, ip=ip))
            for i, dtype in enumerate(result)
        )
    return result(value)


def asm(result=None, *, pure=False):
    """``f(*args) -> (code, constraints, operands[, result, pure])`` as one inline-PTX op.
    Pointer operands pass their address: 32-bit for "r" (shared), 64-bit for "l"."""

    def wrap(fn):
        @dsl_user_op
        @wraps(fn)
        def op(*args, loc=None, ip=None, **kwargs):
            code, constraints, operands, *rest = fn(*args, **kwargs)
            res, fx = (*rest, pure)[:2] if rest else (result, pure)
            kinds = [c for c in constraints.split(",") if c[:1] not in ("=", "~", "")]
            values = []
            for kind, v in zip(kinds, operands):
                if hasattr(v, "toint"):
                    v = v.toint(loc=loc, ip=ip)
                    v = Int32(v) if kind == "r" else v
                values.append(v)
            return ptx(code, constraints, values, res, effects=not fx, loc=loc, ip=ip)

        return op

    return wrap


def _vector(count, first=0):
    """(".vN", "{$first,...}") of an N-word access; ("", "$first") for one word."""
    names = ",".join(f"${first + i}" for i in range(count))
    return (f".v{count}", "{" + names + "}") if count > 1 else ("", names)


@asm(Uint32, pure=True)
def round_pair_square(word):
    return "mul.rn.bf16x2 $0,$1,$1;", "=r,r", (Uint32(word),)


@asm(Float32, pure=True)
def inv_rms(value, eps):
    """rsqrt(value / 3584 + eps): the mean as a reciprocal product refined by its FMA residual."""
    mean = "mul.f32 q,$1,0f39924925; neg.f32 r,q; fma.rn.f32 r,r,0f45600000,$1; "
    mean += "fma.rn.f32 r,r,0f39924925,q;"
    code = f"{{ .reg .f32 q,r; {mean} add.f32 r,r,$2; rsqrt.approx.ftz.f32 $0,r; }}"
    return code, "=f,f,f", (Float32(value), Float32(eps))


def bits_float(bits):
    return Float32(llvm.bitcast(T.f32(), Uint32(bits).ir_value()))


def bf16(word, high=False):
    """FP32 of the low (or high) BF16 half of ``word``."""
    return bits_float(word & 0xFFFF0000 if high else word << 16)


@asm((Float32, Float32), pure=True)
def mul2(a, b):
    """Lanewise a * b of FP32 pairs in one packed instruction (IEEE per lane)."""
    code = "{ .reg .b64 a,b,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += "mul.rn.f32x2 d,a,b; mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f", (*a, *b)


@asm(Uint32, pure=True)
def pack_pair(low, high):
    """BF16x2 of (low, high), rounded to nearest even."""
    return "cvt.rn.bf16x2.f32 $0,$2,$1;", "=r,f,f", (Float32(low), Float32(high))


def round_bf16(value):
    """``value`` rounded to BF16 (nearest even), as FP32."""
    return bf16(pack_pair(value, value))


@asm((Float32,) * 4, pure=True)
def mma(a, b, accumulator):
    """m16n8k16 BF16 mma.sync."""
    code = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    code += "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", (*a, *b, *accumulator)


def thread():
    """(thread, warp, lane, CTA) indices."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    return tid, tid // 32, tid % 32, cta


def view(base, offset, dtype, shape):
    """``dtype`` tensor at byte ``offset`` of pointer ``base``."""
    pointer = cute.recast_ptr(base + offset, dtype=dtype)
    return cute.make_tensor(pointer, cute.make_layout(shape))


def zeros(shape, dtype):
    tensor = cute.make_rmem_tensor(shape, dtype)
    tensor.fill(0)
    return tensor


@asm()
def load_words(pointer, *, count=2, shared=False, evict_first=False, in_place=False):
    """``count`` 32-bit words (a tuple). ``in_place`` makes the load a side-effecting op, which
    the compiler neither hoists nor sinks: for data written during the step, and for the weight
    loads, which must stay in front of the wait. Only static weights use ``evict_first``."""
    space = "shared" if shared else "global.cg"
    policy = hint = tail = ""
    if evict_first:
        assert not shared
        space = "global.nc.L1::no_allocate"
        policy = ".reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
        hint, tail = ".L2::cache_hint", ", pol"
    vector, registers = _vector(count)
    address = f"[${count}]{tail}"
    code = f"{{ {policy}ld.{space}{hint}{vector}.b32 {registers}, {address}; }}"
    constraints = ["=r"] * count + ["r" if shared else "l"]
    # Shared memory is written by other threads between reads: never pure.
    pure = not (shared or in_place)
    return code, ",".join(constraints), (pointer,), (Uint32,) * count, pure


load_shared_words = partial(load_words, count=4, shared=True)
load_weight_words = partial(load_words, count=4, evict_first=True, in_place=True)


@asm()
def store_words(pointer, words, *, shared=False, scope=None):
    """``scope`` is a relaxed global store at that scope."""
    vector, registers = _vector(len(words), 1)
    if shared:
        # Other threads read it: a memory side effect, like the global stores.
        space, constraints = f"shared{vector}.b32", "r" + ",r" * len(words) + ",~{memory}"
    else:
        space = f"relaxed.{scope}.global{vector}.u32" if scope else f"global{vector}.b32"
        constraints = "l" + ",r" * len(words) + ",~{memory}"
        words = tuple(Uint32(v) for v in words)
    return f"st.{space} [$0], {registers};", constraints, (pointer, *words)


store_shared_words = partial(store_words, shared=True)
store_relaxed = partial(store_words, scope="sys")


@asm()
def push_floats(pointer, values, barrier, peer):
    """DSMEM store of an FP32 tuple into CTA ``peer`` of the cluster, completing on ``barrier``."""
    values = tuple(map(Float32, values))
    vector, registers = _vector(len(values), 1)
    b, p = (f"${len(values) + i}" for i in (1, 2))
    code = f"{{ .reg .b32 dst,bar; mapa.shared::cluster.u32 dst,$0,{p}; "
    code += f"mapa.shared::cluster.u32 bar,{b},{p}; "
    code += f"st.async.shared::cluster.mbarrier::complete_tx::bytes{vector}.b32 "
    code += f"[dst],{registers},[bar]; }}"
    constraints = "r" + ",f" * len(values) + ",r,r,~{memory}"
    return code, constraints, (pointer, *values, barrier, Int32(peer))


@asm(Int32)
def try_wait(barrier):
    """Acquire probe (phase 0) of an mbarrier that receives cluster arrivals."""
    code = "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
    code += "p,[$1],0; selp.u32 $0,1,0,p; }"
    return code, "=r,r,~{memory}", (barrier,)


@cute.jit
def wait(barrier):
    done = try_wait(barrier)
    while done == 0:
        done = try_wait(barrier)


def init_barriers(barriers, count):
    for i in range(count):
        cute.arch.mbarrier_init(barriers + i, 1)
    cute.arch.mbarrier_init_fence()


@asm()
def sleep_ns(ns):
    return f"nanosleep.u32 {int(ns)};", "", ()


@cute.jit
def scatter_reduce(values, lane, count: cutlass.Constexpr[int]):
    """Warp-reduce ``count`` per-lane values: every pair of lanes ends up with the warp sum of
    one of them."""
    for stage in cutlass.range_constexpr(5):
        offset = 16 >> stage
        remaining = max(count >> stage, 1)
        if cutlass.const_expr(remaining > 1):
            half = remaining // 2
            upper = (lane & offset) != 0
            for i in cutlass.range_constexpr(half):
                send = values[i] if upper else values[i + half]
                keep = values[i + half] if upper else values[i]
                values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset)
        else:
            values[0] += cute.arch.shuffle_sync_bfly(values[0], offset)
    return values[0]


# --------------------------------------------------------------------------- the kernel
# Shared memory (bytes): x [8 tokens, 224] BF16; MMA partials [8 warps, 8 tokens, 16 rows] FP32;
# gathered squares [8 tokens, 16 slices, 28 lanes] FP32; the owned line's partials [16 slices, 64]
# FP32 and its shared-shard rows [64] FP32; two barriers.
_X, _PG, _SQ, _CELLS, _RS, _BAR = 0, 3584, 7680, 22016, 26112, 26368


class WaveTailKernel:
    """The wave tail for exactly ``m`` tokens (1..8) on TP rank ``rank``."""

    threads, grid, cluster = 256, 112, 16
    #: One CTA per SM: the kernel uses about 26 KB, the request makes every CTA own an SM.
    smem = 116 * 1024
    poll_ns = 64

    def __init__(self, m: int, rank: int):
        if not 1 <= m <= CAPACITY:
            raise ValueError(f"wave tail supports 1..{CAPACITY} tokens, got {m}")
        self.m, self.rank = m, rank

    @cute.jit
    def __call__(
        self,
        gemm2: cute.Tensor,
        route_weights: cute.Tensor,
        route_index: cute.Tensor,
        shared: cute.Tensor,
        weight: cute.Tensor,
        gamma: cute.Tensor,
        latent: cute.Tensor,
        rs: cute.Tensor,
        peers: cute.Tensor,
        state: cute.Tensor,
        latent_mc: Int64,
        output_mc: Int64,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            gemm2,
            route_weights,
            route_index,
            shared,
            weight,
            gamma,
            latent,
            rs,
            peers,
            state,
            latent_mc,
            output_mc,
            eps,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.cluster, 1, 1),
            smem=self.smem,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gemm2: cute.Tensor,
        route_weights: cute.Tensor,
        route_index: cute.Tensor,
        shared: cute.Tensor,
        weight: cute.Tensor,
        gamma: cute.Tensor,
        latent: cute.Tensor,
        rs: cute.Tensor,
        peers: cute.Tensor,
        state: cute.Tensor,
        latent_mc: Int64,
        output_mc: Int64,
        eps: Float32,
    ):
        tid, warp, lane, block = thread()
        q = cute.arch.block_idx_in_cluster()
        m, rank = self.m, self.rank
        row0 = block // self.cluster * 128
        gq, tq = lane // 4, lane % 4
        owner = q < 2 * m
        own_token, own_line = q // 2, q % 2
        rs_j, lat = 4 * warp + lane - 28, lane < 28
        rs_lane = (lane >= 28) & owner & (rs_j < 8)
        sm = (
            cutlass.utils.SmemAllocator()
            .allocate_tensor(Uint8, cute.make_layout(self.smem), 128)
            .iterator
        )
        xs = view(sm, _X, BFloat16, 1792).iterator
        pg = view(sm, _PG, Float32, 1024)
        squares = view(sm, _SQ, Float32, 3584)
        cells = view(sm, _CELLS, Float32, 1024)
        rs_sum = view(sm, _RS, Float32, 64)
        bar = view(sm, _BAR, Uint64, 2).iterator
        if tid == 0:
            init_barriers(bar, 2)
            # Every CTA of the cluster sends 28 lane sums per token.
            cute.arch.mbarrier_arrive_and_expect_tx(bar, self.cluster * m * 28 * 4)
            size = 4096 if owner else 0
            cute.arch.mbarrier_arrive_and_expect_tx(bar + 1, size)
        cute.arch.cluster_arrive_relaxed()
        # The MMA group always has 8 token columns: the ones past m are zero.
        for i in cutlass.range(tid, (8 - m) * 28, self.threads, unroll=1):
            store_shared_words(xs + m * 224 + 8 * i, (Uint32(0),) * 4)
        gx = cute.make_rmem_tensor(4, Uint32)
        if lat & (warp < m):
            for k, v in enumerate(load_words(gamma.iterator + q * 224 + 8 * lane, count=4)):
                gx[k] = v
        # Rows 16 warp + gq (+ 8) of the cluster's tile, columns 224 q + 32 k + 8 tq.
        base = weight.iterator + (row0 + 16 * warp + gq) * LATENT + 224 * q + 8 * tq
        a = [
            [load_weight_words(base + h * 8 * LATENT + 32 * k) for h in range(2)] for k in range(7)
        ]

        # FC2 and everything before it are complete from here on. The dependents (the reader of
        # the output mailbox) are released after this wait, as the other tail kernels do.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        prev = load_words(state.iterator, count=2, in_place=True)
        prev_slot, prev_m = Int32(prev[0]), Int32(prev[1])
        slot = (prev_slot + 1) % NUM_LAMPORT_BUFFERS
        # This rank's first token row in the slot, and the slot's first row.
        mine, first = (slot * TP + rank) * CAPACITY, slot * TP * CAPACITY

        # Cluster 0: this rank's latent fragment (top-16 weighted sum), multicast to every rank.
        if (block < self.cluster) & lat & (warp < m):
            col = 224 * q + 8 * lane
            # Side-effecting: the FC2 rows are written during the step.
            packed = finalize_top16_bf16(
                gemm2.iterator + col,
                route_index.iterator + warp * 16,
                route_weights.iterator + warp * 16,
                side_effects=True,
            )
            offset = (mine + warp) * LATENT + col
            stmc_bf16x8(latent_mc + Int64(offset) * 2, sanitize_negative_zero(packed))
        # The CTAs behind cluster 0, one thread per 16-byte chunk (3.5 CTAs per token): the
        # shared-expert partials to their owner ranks (reduce-scatter).
        task = (block - self.cluster) * self.threads + tid
        if (block >= self.cluster) & (task < m * (HIDDEN // 8)):
            token, column = task // (HIDDEN // 8), 8 * (task % (HIDDEN // 8))
            word = load_words(shared.iterator + token * HIDDEN + column, count=4, in_place=True)
            clean = tuple(sanitize_negative_zero_u32(w) for w in word)
            offset = (mine + token) * SHARD + column % SHARD
            store_relaxed(peers[column // SHARD] + Int64(offset) * 2, clean)

        # Lanes 0..27 of warp t poll token t's latent slice; lanes 28..31 of an owner's warps 0
        # and 1 poll the eight 16-byte chunks of its output line's shared shard.
        xwords = cute.make_rmem_tensor(4, Uint32)
        sq = Float32(0)
        if lat & (warp < m) | rs_lane:
            rs_at = (first + own_token) * SHARD + row0 + 64 * own_line + 8 * (rs_j & 7)
            ptr = latent.iterator + (first + warp) * LATENT + 224 * q + 8 * lane
            ptr = ptr if lat else rs.iterator + rs_at
            stride = CAPACITY * (LATENT if lat else SHARD)
            sums = self.sum_sources(self.poll(ptr, stride))
            if lat:
                sq = self.latent_words(sums, xwords)
            else:
                for j in cutlass.range_constexpr(8):
                    rs_sum[8 * rs_j + j] = round_bf16(sums[j])
        cute.arch.sync_warp()
        cute.arch.cluster_wait()
        self.push_squares(squares, sq, bar, (q, warp, lane))
        wait(bar)
        cute.arch.sync_warp()
        if warp < m:
            total = self.square_total(squares, warp, lane)
            if lat:
                self.scale_words(xwords, gx, inv_rms(total, eps), xs, warp, lane)
        cute.arch.sync_warp()
        cute.arch.sync_threads()
        # Every thread of this CTA has read ``state`` by now.
        if tid == 0:
            red_async_release_gpu_add_u32(state.iterator + 2, Uint32(1))
        self.mma_group(a, xs, pg, cells, bar, (q, warp, lane, gq, tq))
        if owner & (warp == 0):
            self.write_line(cells, rs_sum, bar, output_mc, (lane, row0, own_token, own_line))
        # After the output is out: re-arm the previous launch's slot, then advance the rotation
        # once all CTAs have read it.
        self.rearm(latent, rs, prev_slot, prev_m, block, tid)
        if (block == self.grid - 1) & (tid == self.threads - 1):
            seen = load_volatile_u32(state.iterator + 2)
            while seen < self.grid:
                sleep_ns(self.poll_ns)
                seen = load_volatile_u32(state.iterator + 2)
            store_words(state.iterator, (Uint32(slot), Uint32(m), Uint32(0), Uint32(0)))

    @cute.jit
    def poll(self, base, stride):
        """The eight ranks' 16-byte chunks; re-polls the ones that still hold the sentinel."""
        values = cute.make_rmem_tensor((TP, 4), Uint32)
        pending = Uint32(0)
        for source in cutlass.range_constexpr(TP):
            pending |= self.poll_source(values, source, base + source * stride)
        while pending != 0:
            sleep_ns(self.poll_ns)
            again = Uint32(0)
            for source in cutlass.range_constexpr(TP):
                if (pending & (Uint32(1) << source)) != 0:
                    again |= self.poll_source(values, source, base + source * stride)
            pending = again
        return values

    @cute.jit
    def poll_source(self, values, source: cutlass.Constexpr[int], ptr):
        word = load_global_u32x4(ptr, volatile=True)
        for k in cutlass.range_constexpr(4):
            values[source, k] = word[k]
        return Uint32(fragment_is_dirty(word)) << source

    @cute.jit
    def sum_sources(self, values):
        acc = zeros(8, Float32)
        for source in cutlass.range_constexpr(TP):
            for k in cutlass.range_constexpr(8):
                acc[k] = acc[k] + bf16(values[source, k // 2], k % 2)
        return acc

    @cute.jit
    def latent_words(self, sums, words):
        """BF16 latent words of a lane's eight columns; returns their square sum."""
        s2 = Float32(0)
        for j in cutlass.range_constexpr(4):
            words[j] = pack_pair(sums[2 * j], sums[2 * j + 1])
            square = round_pair_square(words[j])
            s2 += bf16(square)
            s2 += bf16(square, True)
        return s2

    @cute.jit
    def push_squares(self, squares, s2, bar, ids):
        """The warp's token's 28 lane sums of this slice to every CTA of the cluster."""
        q, warp, lane = ids
        if warp < self.m:
            chunk, first = lane % 7, lane // 7
            vector = tuple(shuffle(s2, 4 * chunk + i) for i in range(4))
            if first < 4:
                dst = squares.iterator + warp * 448 + 28 * q + 4 * chunk
                for peer in cutlass.range(first, self.cluster, 4, unroll=1):
                    push_floats(dst, vector, bar, peer)

    @cute.jit
    def square_total(self, squares, warp, lane):
        groups = cute.make_rmem_tensor(16, Float32)
        for c in cutlass.range_constexpr(16):
            groups[c] = Float32(0)
            if lane < (16 if c % 4 == 3 else 32):
                groups[c] = squares[warp * 448 + c // 4 * 112 + c % 4 * 32 + lane]
        group_sum = scatter_reduce(groups, lane, 16)
        return sum((shuffle(group_sum, 2 * c) for c in range(16)), Float32(0))

    def scale_words(self, words, gx, inverse, xs, warp, lane):
        """(x * inverse) * gamma per column, as packed FP32x2 multiplies."""
        packed = []
        for k in range(4):
            scaled = mul2((bf16(words[k]), bf16(words[k], True)), (inverse, inverse))
            scaled = mul2(scaled, (bf16(gx[k]), bf16(gx[k], True)))
            packed.append(pack_pair(*scaled))
        store_shared_words(xs + warp * 224 + 8 * lane, packed)

    @cute.jit
    def mma_group(self, a, xs, pg, cells, bar, ids):
        """Two seven-step BF16 MMA chains; each 4-row partial goes to its 64-row line's owner."""
        q, warp, lane, gq, tq = ids
        b = [load_shared_words(xs + gq * 224 + 8 * tq + 32 * k) for k in range(7)]
        d = e = (Float32(0),) * 4
        for k in cutlass.range_constexpr(7):
            (w0, w1), x = a[k], b[k]
            d = mma((w0[0], w1[0], w0[1], w1[1]), x[0:2], d)
            e = mma((w0[2], w1[2], w0[3], w1[3]), x[2:4], e)
        for i in cutlass.range_constexpr(4):  # [warp, token, row] partial products
            pg[warp * 128 + (2 * tq + i % 2) * 16 + gq + 8 * (i // 2)] = d[i] + e[i]
        cute.arch.sync_warp()
        token, row = lane // 4, 16 * warp + 4 * (lane % 4)
        if token < self.m:
            values = load_shared_f32x4(pg.iterator + warp * 128 + 4 * lane)
            # Line 2 token + row // 64 is owned by the CTA of that rank in the cluster.
            push_floats(cells.iterator + q * 64 + row % 64, values, bar + 1, 2 * token + row // 64)

    @cute.jit
    def write_line(self, cells, rs_sum, bar, output_mc, ids):
        """Sum the 16 slices' partials, add the shared shard and multicast the 64 output rows."""
        lane, row0, own_token, own_line = ids
        wait(bar + 1)
        cute.arch.sync_warp()
        out = []
        for h in cutlass.range_constexpr(2):
            parts = (cells[64 * p + 2 * lane + h] for p in range(16))
            value = round_bf16(sum(parts, Float32(0)))
            out.append(value + rs_sum[2 * lane + h])
        # No half may be the mailbox's negative-zero marker.
        pair = sanitize_negative_zero_u32(pack_pair(*out))
        words = tuple(shuffle(pair, (4 * lane + k) & 31) for k in range(4))
        if lane < 8:
            offset = own_token * HIDDEN + self.rank * SHARD + row0 + 64 * own_line
            stmc_bf16x8(output_mc + Int64(offset + 8 * lane) * 2, words)

    @cute.jit
    def rearm(self, latent, rs, slot, m, block, tid):
        """Sentinels back into the first ``m`` token rows of every source of ``slot`` (latent
        and shared shard): the launch that used them is complete."""
        nl, nr = TP * m * (LATENT // 8), TP * m * (SHARD // 8)
        start, step = block * self.threads + tid, self.grid * self.threads
        for i in cutlass.range(start, nl + nr, step, unroll=1):
            if i < nl:
                row, frag = i // (LATENT // 8), i % (LATENT // 8)
                source, token = row // m, row % m
                dst = latent.iterator + ((slot * TP + source) * CAPACITY + token) * LATENT
                store_lamport_sentinel_128(dst + 8 * frag)
            else:
                j = i - nl
                row, frag = j // (SHARD // 8), j % (SHARD // 8)
                source, token = row // m, row % m
                dst = rs.iterator + ((slot * TP + source) * CAPACITY + token) * SHARD
                store_lamport_sentinel_128(dst + 8 * frag)


# --------------------------------------------------------------------------- host side
def region_bytes() -> dict[str, int]:
    """Bytes of the two symmetric regions: latent ``[3, TP, CAPACITY, 3584]`` and shared shard
    ``[3, TP, CAPACITY, 896]``, BF16."""
    rows = NUM_LAMPORT_BUFFERS * TP * CAPACITY
    return {"latent": rows * LATENT * 2, "rs": rows * SHARD * 2}


class WaveTail:
    """The compiled wave tails of one rank over caller-owned symmetric regions.

    ``latent`` / ``rs`` are this rank's views of the two regions (sentinel-filled by the caller
    before any rank launches), ``latent_mc`` the latent region's multicast address, ``rs_peers``
    the peers' addresses of the shard region and ``output_mc`` the multicast address of the
    tail's output mailbox ``[1, tokens, 7168]``."""

    def __init__(
        self,
        *,
        rank: int,
        token_counts: Iterable[int],
        latent: torch.Tensor,
        rs: torch.Tensor,
        latent_mc: int,
        rs_peers: torch.Tensor,
        output_mc: int,
        rms_eps: float,
    ):
        device = latent.device
        self.rms_eps = float(rms_eps)
        self._latent, self._rs, self._peers = latent, rs, rs_peers
        self._latent_mc, self._output_mc = int(latent_mc), int(output_mc)
        # (slot, token count) of the previous launch and the CTAs that have read them.
        self._state = torch.tensor(
            [NUM_LAMPORT_BUFFERS - 1, 0, 0, 0], dtype=torch.int32, device=device
        )
        self._compiled = {}
        for m in sorted(set(token_counts)):
            gemm2 = torch.empty((m * 16, LATENT), dtype=torch.bfloat16, device=device)
            weights = torch.empty((m, 16), dtype=torch.bfloat16, device=device)
            index = torch.empty((m, 16), dtype=torch.int32, device=device)
            shared = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
            up = torch.empty((SHARD, LATENT), dtype=torch.bfloat16, device=device)
            gamma = torch.empty((LATENT,), dtype=torch.bfloat16, device=device)
            self._compiled[m] = cute.compile(
                WaveTailKernel(m, rank),
                *self._args(gemm2, weights, index, shared, up, gamma),
            )

    def supports(self, m: int) -> bool:
        return m in self._compiled

    def _args(self, gemm2, weights, index, shared, up_shard, gamma):
        return (
            to_cute_dynamic_m(gemm2, mode=0, assumed_align=16),
            to_cute_dynamic_m(weights, mode=0, assumed_align=16),
            to_cute_dynamic_m(index, mode=0, assumed_align=16),
            to_cute_dynamic_m(shared, mode=0, assumed_align=16),
            to_cute(up_shard, 16),
            to_cute(gamma, 16),
            to_cute(self._latent, 16),
            to_cute(self._rs, 16),
            to_cute(self._peers, 16),
            to_cute(self._state, 16),
            Int64(self._latent_mc),
            Int64(self._output_mc),
            Float32(self.rms_eps),
            cuda.CUstream(torch.cuda.current_stream(gamma.device).cuda_stream),
        )

    def __call__(self, routed, shared, gamma, up_shard) -> None:
        """Launch for ``routed`` (an ``UnfinalizedRoutedOutput``) and this rank's shared-expert
        partial ``[M, 7168]``; the output lands in the mailbox rows ``[0, M)`` on every rank."""
        m = routed.expanded_idx_to_permuted_idx.shape[0]
        self._compiled[m](
            *self._args(
                routed.gemm2_permuted,
                routed.expert_weights,
                routed.expanded_idx_to_permuted_idx,
                shared,
                up_shard,
                gamma,
            )
        )


__all__ = ["WaveTail", "region_bytes"]
