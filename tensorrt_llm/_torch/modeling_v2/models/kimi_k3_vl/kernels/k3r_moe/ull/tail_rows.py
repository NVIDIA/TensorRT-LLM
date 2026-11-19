# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""MoE tail above the tuned 64 rows (``moe.prepared.choose`` wide shapes, TP16).

Inputs are plain tensors: this rank's routed partial latent [M, 3584] (the experts'
``local`` publish, MoEScratch.latent_partial) and its shared-expert partial [M, 7168]
(MoEScratch.shared_partial). Every exchange is a Lamport mailbox sized to the
deployment's largest graph (``TailRowsBox``):

- ``PushShared`` (side stream, after the shared expert): each row's 448-column slices
  to their owners' RS slots [3, TP, cap, 448] (-0 made +0).
- ``LatentRows``: the latent partial's 224-column slices to their owners' two-shot
  slots; each owner sums its slice of every row (``common.twoshot.reduce``: the TP
  slots in rank order in FP32, one rounding, multicast to every rank's gather, slots
  re-armed). CTA j handles row items j, j + G, ... (``latent_chunk`` rows each):
  push, then reduce, so the items' exchanges overlap.
- ``NormRows`` (one CTA per row): the gathered latent row (re-armed) RMS-normalized,
  (x * rsqrt(mean(x^2) + eps)) * gamma rounded once to BF16 into ``xnorm``; this
  rank's 448 shared columns summed over the TP slots in rank order (FP32) into
  ``shared``.
- the up projection [M, 3584] x [448, 3584]^T in FP32 (cuBLAS) into ``up``;
- ``OutRows``: BF16(up + shared) (one rounding, -0 made +0) multicast into every
  rank's output ring row [cap, 7168] at this rank's 448 columns; the next AttnRes
  polls the ring (one slot) and re-arms it.

Every reader re-arms what it read in the same call, and every kernel reads its inputs
only after its PDL wait and releases its dependents after it. No re-arm is deferred, so
there is no wrap and any parity sequence is safe (unlike the tuned tail's deferred
re-arm, ``tail_runtime.tail_parity``): a peer's next push into a slot follows its own
previous MoE layer, which needs this rank's output rows, written after its re-arms.
The parities (MoE ordinal mod 3) only keep consecutive layers' slots apart.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ....ops.common.k3_ull_compat import UnfinalizedMoEOutput

from ....ops.common.k3_ull_cute_host import (
    dynamic_fake,
    dynamic_view,
    fake,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ...k3r_common.ull.mailbox import poll, rearm
from ...k3r_common.ull.policy import batch_table
from ...k3r_common.ull.primitives import (
    add_rn,
    at,
    bf16,
    fma,
    inv_rms,
    launch,
    load_words,
    multicast_words,
    named_barrier,
    no_neg_zero,
    pack_pair,
    store_relaxed,
    store_words,
    thread,
    zeros,
)
from ...k3r_comm.ull_twoshot import TwoShot, push, reduce
from ...k3r_oproj.ull.prepared import OprojMailbox

H, LATENT = 7168, 3584


def latent_chunk(rows, grid=112):
    """Rows per LatentRows work item (CTA j takes items j, j + grid, ...): small items
    spread small M over more CTAs, as long as the items fit one pass of the grid;
    large ones cut the per-item push / reduce overhead (4 ranks, us per tail: chunk 2
    / 4 / 8 at M128 15.9 / 16.4 / 18.1, M256 22.2 / 19.3 / 21.1, M512 33.3 / 30.1 /
    29.1; chunk 1 / 2 at M72 11.9 / 12.2, M96 12.0 / 12.9, M128 14.8 / 12.4, and with
    the unfinalized top-k sum M72 12.8 / 14.5, M96 13.0 / 15.0, M128 19.1 / 15.6)."""
    return 1 if rows <= grid else 2 if rows <= 128 else 4 if rows <= 256 else 8


# [axl wpush] Batches whose wide routed experts (``experts_runtime.WideExperts``)
# push the latent columns into the owners' two-shot slots from their finalize
# (publish "twoshot"), so LatentRows only reduces: the reduce-scatter's stores leave
# with the experts' last rows instead of after the experts' grid. Same words, same
# slots, same reduction order: bitwise. 129-256 rows (axhb, 16 ranks, 93 layers, 32k,
# interpolated unions, us/step vs the same overlay without them): B17 (136 rows) -85,
# B24 -195, B28 -133, B32 -173. 257-512 rows (axhb2 sx1, DL3 mix, mns 192, 16 ranks, 93
# layers, same-node A, us/step): B96 (384 rows) -371, B128 (512) -467, B64 bitwise.
EXPERTS_PUSH = {16: "65-768:1"}


# The up projection [M, 3584] x [448, 3584]^T in FP32: "cublas" (torch.mm) or a
# split-K tactic of ``gemm.projection.PreparedProjection`` (M tile, N tile, split,
# stages), which chains to its neighbours by PDL.
UP_GEMM = "cublas"
# Rows (per TP) whose up projection runs fused with the output (``up_tmem.UpTmem64``:
# the weights' K slices in tensor memory before the PDL wait, 64-token tiles, the FP32
# partials reduced over DSMEM in rank order, + the shared rows, one rounding, multicast
# into the output ring) instead of cuBLAS (nvjet + splitKreduce, no PDL) + OutRows. An
# FP32 accumulation reorder over K of the same GEMM (comm3 upnum: >= 99.9 % of outputs
# bitwise, the rest 1-7 ulp, the same relative error vs FP64). 16 ranks (axt mb-sw1,
# 93 layers, 32k), us/step: B9 (72 rows) 9615 -> 9396, B10 (80) 9992 -> 9827, B8 +7.
# At 96-256 rows it was neutral (comm3 4o, ds25). 129-256 rows on ds35 (axhb, 16 ranks,
# 93 layers, 32k, us/step vs the cuBLAS up): B17 -66, B24 -32, B28 -98, B32 -35.
UP_ROWS = {16: "65-80:1,129-256:1"}

# Rows whose NormRows skips the PDL dependency wait (``NormRows(polled=True)``): its
# inputs are Lamport-polled (the gather rows, the RS slots) or static (gamma), and
# LatentRows releases its dependents only after its own wait, so the previous layer's
# readers of xnorm / shared have completed. 4 ranks + 12 emulated, side push, us per
# tail call (max over ranks, two passes, outputs bitwise): wait -> polled M96 12.37 ->
# 12.66, M128 12.52 -> 13.27, M160 13.60 -> 14.02, M192 14.67 -> 14.64, M224 15.34 ->
# 14.87, M256 16.45 -> 15.59 (the early CTAs, one per row, contend with LatentRows'
# owner reduce up to ~192 rows).
NORM_POLLED = {16: "193-768:1"}


def norm_polled(tp, rows):
    return bool(batch_table(NORM_POLLED.get(tp, "")).get(rows))


class TailRowsBox:
    """The wide tail's mailboxes and scratch at ``cap`` rows: latent two-shot (3
    parities, width 3584), shared RS slots [3, TP, cap, H / TP] with the owners'
    bases, the output ring [1, cap, H] (multicast), and the local xnorm [cap, 3584],
    up / shared [cap, H / TP] FP32 rows."""

    def __init__(self, group, device, tp, rank, cap):
        self.tp, self.rank, self.cap = tp, rank, cap
        self.latent = TwoShot(group, device, tp, rank, LATENT, cap, parities=3)
        self.rs = OprojMailbox(group, device, tp, rank, (3, tp, cap, H // tp))
        self.rs_peers = torch.tensor(
            self.rs.handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        self.out = OprojMailbox(group, device, tp, rank, (1, cap, H))
        f32, bf = (
            dict(dtype=torch.float32, device=device),
            dict(dtype=torch.bfloat16, device=device),
        )
        self.xnorm = torch.empty((cap, LATENT), **bf)
        self.up = torch.empty((cap, H // tp), **f32)
        self.shared = torch.empty((cap, H // tp), **f32)


class PushShared:
    """Each row's [H] shared partial, 448-column slices to their owners' RS slots."""

    def __init__(self, tp):
        self.tp, self.shard = tp, H // tp

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, args[0].shape[0], 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        shared, peers, parity, rank, cap = args
        tid, _, _, token = thread()
        for i in cutlass.range_constexpr(H // 8 // 128):
            column = 8 * (tid + 128 * i)
            words = load_words(at(shared, token, column), count=4, mutable=True)
            offset = ((parity * self.tp + rank) * cap + token) * self.shard
            offset += column % self.shard
            packed = tuple(no_neg_zero(w) for w in words)
            store_relaxed(peers[column // self.shard] + Int64(offset) * 2, packed)


class LatentRows:
    """The latent partial's two-shot all-reduce over ``chunk``-row items (module
    doc). ``topk`` (the experts' deferred top-k finalize, M <= 128 in the stock runner):
    the partial arrives unfinalized (``UnfinalizedMoEOutput``: permuted unweighted rows
    [P, 3584], routing weights [M, topk], permute map [M, topk], -1 = not local) and
    each pushed chunk is its top-k sum: FP32 fused multiply-adds in k order, rounded
    once to BF16 (a stated FP32 reorder vs the experts' own finalize)."""

    def __init__(self, tp, grid=112, chunk=2, topk=0, pushed=False):
        self.tp, self.grid, self.width, self.chunk = tp, grid, LATENT // tp, chunk
        self.topk = topk
        # ``pushed``: the experts pushed every row's chunks already (EXPERTS_PUSH):
        # reduce only (the partial argument is not read).
        self.pushed = pushed
        assert not (pushed and topk)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 256)

    @cute.jit
    def finalize(self, rows, weights, index, token, column):
        """Token's 8 latent columns: sum over k of weight x permuted row, FP32. All
        k rows load before the sum (no branch: a non-local expert reads row 0 with
        weight 0, which adds +0 to the FP32 sum)."""
        K: cutlass.Constexpr = self.topk
        raw = cute.make_rmem_tensor((K, 4), Uint32)
        scale = cute.make_rmem_tensor(K, Float32)
        for k in cutlass.range_constexpr(K):
            permuted, row = index[token, k], Int32(0)
            scale[k] = Float32(0)
            if permuted >= 0:
                scale[k] = Float32(weights[token, k])
                row = permuted
            words = load_words(at(rows, row, column), count=4, mutable=True)
            for v in cutlass.range_constexpr(4):
                raw[k, v] = words[v]
        acc = zeros(8, Float32)
        for k in cutlass.range_constexpr(K):
            for j in cutlass.range_constexpr(8):
                acc[j] = fma(bf16(raw[k, j // 2], j % 2), scale[k], acc[j])
        return tuple(pack_pair(acc[2 * j], acc[2 * j + 1]) for j in range(4))

    @cute.kernel
    def kernel(self, args):
        if cutlass.const_expr(self.topk > 0):
            rows, weights, index = args[:3]
            slots, peers, slot, recv, gather, gather_mc, rank = args[3:]
            partial, M = rows, weights.shape[0]
        else:
            partial, slots, peers, slot, recv, gather, gather_mc, rank = args
            M = partial.shape[0]
        tid, _, _, cta = thread()
        per: cutlass.Constexpr = self.width // 8  # 8-column chunks per owner row
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        chunk: cutlass.Constexpr = self.chunk
        chunks = (M + chunk - 1) // chunk
        for c in cutlass.range(cta, chunks, self.grid, unroll=1):
            row0 = c * chunk
            pushes = 0 if self.pushed else chunk * (LATENT // 8)
            for v in cutlass.range(tid, pushes, 256, unroll=1):
                token, column = row0 + v // (LATENT // 8), v % (LATENT // 8) * 8
                if token < M:
                    if cutlass.const_expr(self.topk > 0):
                        words = self.finalize(rows, weights, index, token, column)
                    else:
                        words = load_words(
                            at(partial, token, column), count=4, mutable=True
                        )
                    words = tuple(no_neg_zero(w) for w in words)
                    push(slots, peers, slot, token, column, words)
            if tid < chunk * per:
                token = row0 + tid // per
                if token < M:
                    column = rank * self.width + tid % per * 8
                    reduce(recv, gather, gather_mc, rank, token, column, True)


class NormRows:
    """One CTA (448 threads) per row: threads < 56 first sum this rank's shared RS
    slots into ``shared``, then the CTA RMS-normalizes the gathered latent row into
    ``xnorm``."""

    def __init__(self, tp, polled=False):
        self.tp, self.shard, self.polled = tp, H // tp, polled

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, args[0].shape[0], 448)

    @cute.kernel
    def kernel(self, args):
        xnorm, gather, gamma, rs, shared, eps = args
        tid, warp, lane, token = thread()
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(Float32, cute.make_layout(16), 16)
        gw = load_words(gamma.iterator + tid * 8, count=4)
        if cutlass.const_expr(not self.polled):  # NORM_POLLED
            cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # The shared rows first: their RS slots fill while the latent all-reduce runs
        # (the push precedes it or runs beside it), so only the gathered row's
        # normalization follows its arrival.
        if tid < self.shard // 8:
            base = rs.iterator + token * self.shard + tid * 8
            stride = rs.shape[1] * self.shard  # slot stride
            slots = poll(base, stride, self.tp)
            acc = zeros(8, Float32)
            for s in cutlass.range_constexpr(self.tp):
                rearm(base + s * stride)
                for k in cutlass.range_constexpr(8):
                    acc[k] = acc[k] + bf16(slots[s, k // 2], k % 2)
            for k in cutlass.range_constexpr(8):
                shared[token, tid * 8 + k] = acc[k]
        cell = at(gather, 0, token, tid * 8)
        got = poll(cell, 0, 1)
        rearm(cell)
        x = cute.make_rmem_tensor(8, Float32)
        for k in cutlass.range_constexpr(8):
            x[k] = bf16(got[0, k // 2], k % 2)
        square = x[0] * x[0]
        for k in cutlass.range_constexpr(1, 8):
            square = square + x[k] * x[k]
        for shift in cutlass.range_constexpr(4, -1, -1):
            square = add_rn(square, cute.arch.shuffle_sync_bfly(square, 1 << shift))
        if lane == 0:
            partials[warp] = square
        named_barrier(1, 448)
        total = partials[0]
        for w in cutlass.range_constexpr(1, 14):
            total = add_rn(total, partials[w])
        inverse = inv_rms(total, eps, m8=True)
        words = tuple(
            pack_pair(
                (x[2 * j] * inverse) * bf16(gw[j]),
                (x[2 * j + 1] * inverse) * bf16(gw[j], True),
            )
            for j in range(4)
        )
        store_words(at(xnorm, token, tid * 8), words)


class OutRows:
    """BF16(up + shared) of this rank's 448 columns, multicast into every rank's
    output ring row (one rounding; -0 made +0). ``owners`` (row-sharded AttnRes,
    ``attn_res.sharded``): stored into the row owner's ring [1, per_cap, H] instead
    (args: up, shared, the owners' ring bases, this rank)."""

    def __init__(self, tp, owners=False):
        self.tp, self.shard, self.owners = tp, H // tp, owners

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, args[0].shape[0], self.shard // 8)

    @cute.kernel
    def kernel(self, args):
        up, shared, out_mc, rank = args
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        words = []
        for j in cutlass.range_constexpr(4):
            lo = up[token, tid * 8 + 2 * j] + shared[token, tid * 8 + 2 * j]
            hi = up[token, tid * 8 + 2 * j + 1] + shared[token, tid * 8 + 2 * j + 1]
            words.append(pack_pair(lo, hi, policy="latent"))
        if cutlass.const_expr(self.owners):
            per: cutlass.Constexpr = (up.shape[0] + self.tp - 1) // self.tp
            owner = token // per
            offset = (token - owner * per) * H + rank * self.shard + tid * 8
            store_relaxed(Int64(out_mc[owner]) + Int64(offset) * 2, tuple(words))
        else:
            offset = token * H + rank * self.shard + tid * 8
            multicast_words(out_mc + Int64(offset) * 2, tuple(words))


# Rubin TP16 rows whose wide-tail kernels (PushShared, LatentRows, NormRows, OutRows)
# request the 100% shared-memory carve-out, as the tuned MoeTail does: an SM they leave
# in a small configuration takes the next AttnRes CTAs at that size, and the next
# layer's in-projection (SplitKTmemGemm64, ~180-200 KB) cannot load beside them until
# they exit. Launch attribute only (same code). 1 GPU, KDA in-projection chained behind
# an experts / LatentRows / NormRows / cuBLAS up / splitKreduce / OutRows / AttnRes
# stand-in chain, us: M72 N80 6.49 -> 5.93, M96 N96 6.73 -> 6.73.
ROWS_CARVEOUT = {16: "65-96:1"}


@cache
def compile_rows(kind, tp, rows, cap, device, chunk=None, deferred=None):
    """The wide tail's kernels at ``rows`` (static) over ``cap``-row boxes."""
    code = _compile_rows(kind, tp, rows, cap, device, chunk, deferred)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms > 152 and batch_table(ROWS_CARVEOUT.get(tp, "")).get(rows):
        return prepare_shared_carveout(code, device, 100)
    return code


def _compile_rows(kind, tp, rows, cap, device, chunk=None, deferred=None):
    width, shard = LATENT // tp, H // tp
    with torch.accelerator.device_index(device):
        s = make_fake_stream()
        if kind == "push":
            args = (fake((rows, H), (H, 1)), fake((tp,), None, torch.int64))
            args += (Int32(0), Int32(0), Int32(cap))
            return cute.compile(PushShared(tp), *args, s)
        if kind in ("latent", "latent_pushed"):
            args = (fake((rows, LATENT), (LATENT, 1)), fake((3 * tp, cap, width)))
            args += (fake((tp,), None, torch.int64), Int32(0), fake((tp, cap, width)))
            args += (fake((1, cap, LATENT)), Int64(0), Int32(0))
            chunk = chunk or latent_chunk(rows)
            topk = 0
            if deferred is not None:  # (top-k, routing weights dtype): unfinalized
                topk, dtype = deferred
                permuted = dynamic_fake(2)  # [P, 3584], P the experts' padded rows
                routing = (fake((rows, topk), (topk, 1), dtype),)
                routing += (fake((rows, topk), (topk, 1), torch.int32),)
                args = (permuted, *routing, *args[1:])
            pushed = kind == "latent_pushed"
            kernel = LatentRows(tp, chunk=chunk, topk=topk, pushed=pushed)
            return cute.compile(kernel, *args, s)
        if kind == "norm":
            args = (fake((rows, LATENT), (LATENT, 1)), fake((1, cap, LATENT)))
            args += (fake((LATENT,), (1,)), fake((tp, cap, shard)))
            args += (fake((rows, shard), (shard, 1), torch.float32), Float32(1e-6))
            return cute.compile(NormRows(tp, norm_polled(tp, rows)), *args, s)
        if kind in ("out", "out_owners"):
            f32, owners = torch.float32, kind == "out_owners"
            args = (
                fake((rows, shard), (shard, 1), f32),
                fake((rows, shard), (shard, 1), f32),
            )
            target = fake((tp,), None, torch.int64) if owners else Int64(0)
            args += (target, Int32(0))
            return cute.compile(OutRows(tp, owners), *args, s)
    raise ValueError(kind)


class PreparedTailRows:
    """One MoE layer's wide tail (``placement`` as ``PreparedTail``: the model's
    ``TailRowsBox``, MoE ordinal, MoE layer count)."""

    def __init__(
        self, tp, rank, up, gamma, eps, batches, box, ordinal, *, chunk=None,
        gemm=UP_GEMM, sharded=None, deferred=None, experts_push=(),
    ):  # fmt: skip
        if max(batches) > box.cap:
            raise ValueError(f"wide MoE tail batches {batches} exceed {box.cap} rows")
        self.tp, self.rank, self.box = tp, rank, box
        self.up_shard = up.narrow(0, rank * (H // tp), H // tp)  # [448, 3584]
        self.projection = self.fused_up = None
        if gemm == "uptmem":  # gemm64's up projection fused with the output
            from ...k3r_gemm.ull.projection import weight_map
            from .up_tmem import compile_up_tmem

            self.w_map = weight_map(self.up_shard, 128)  # kept alive here
            self.fused_up = compile_up_tmem(H // tp, gamma.device.index)
            self.fused_args = (
                Int64(box.out.mc), Int32(rank * (H // tp)), Int64(self.w_map.data_ptr())
            )  # fmt: skip
        elif gemm != "cublas":
            from ...k3r_gemm.ull.projection import PreparedProjection

            weight = self.up_shard.contiguous()
            policies = {b: tuple(gemm) for b in batches}
            self.projection = PreparedProjection(
                weight, policies, output_dtype=torch.float32
            ).prepare()
        rows_up = batch_table(UP_ROWS.get(tp, ""))
        self.fused_rows = {b for b in batches if rows_up.get(b)}
        if self.fused_rows and self.fused_up is None:
            from ...k3r_gemm.ull.projection import weight_map
            from .up_tmem import compile_up_tmem

            self.w_map = weight_map(self.up_shard, 128)  # kept alive here
            self.fused_up = compile_up_tmem(H // tp, gamma.device.index)
            self.fused_args = (
                Int64(box.out.mc), Int32(rank * (H // tp)), Int64(self.w_map.data_ptr())
            )  # fmt: skip
        if gemm == "uptmem":
            self.fused_rows = set(batches)
        self.gamma, self.eps = to_cute(gamma.reshape(-1)), Float32(eps)
        self.parity = parity = ordinal % 3
        device = gamma.device.index
        # ``experts_push``: batches whose routed experts can push (the wide engines),
        # kept where EXPERTS_PUSH lists them; their LatentRows only reduces and the
        # experts take ``experts_box``.
        table = batch_table(EXPERTS_PUSH.get(tp, ""))
        self.experts_push = {b for b in experts_push if table.get(b) and b in batches}
        self.codes = {
            b: tuple(
                compile_rows(
                    "latent_pushed" if k == "latent" and b in self.experts_push else k,
                    tp,
                    b,
                    box.cap,
                    device,
                    chunk,
                )  # fmt: skip
                for k in ("push", "latent", "norm", "out")
            )
            for b in batches
        }
        # Batches whose routed partial arrives unfinalized (``deferred`` = (batches,
        # top-k): the stock runner defers the top-k finalize there): LatentRows sums
        # the top k itself, for either routing-weights dtype.
        self.deferred = {}
        if deferred is not None:
            topk = deferred[1]
            for b in set(batches) & set(deferred[0]):
                self.deferred[b] = {
                    dtype: compile_rows(
                        "latent", tp, b, box.cap, device, chunk, (topk, dtype)
                    )
                    for dtype in (torch.bfloat16, torch.float32)
                }
        recv, peers, gather, gather_mc = box.latent.views(parity)
        slots = box.latent.recv.view(3 * tp, box.cap, -1)
        # the experts' push box (``WideExperts`` pushed): slots, owners' bases, slot
        self.experts_box = (slots, box.latent.peers[0], parity * tp + rank)
        self.latent_args = (
            to_cute(slots), to_cute(box.latent.peers[0], 16), Int32(parity * tp + rank),
            to_cute(recv), to_cute(gather), Int64(gather_mc), Int32(rank),
        )  # fmt: skip
        self.gather = to_cute(gather)
        self.rs = to_cute(box.rs.buffer[parity])
        self.push_args = (
            to_cute(box.rs_peers, 16),
            Int32(parity),
            Int32(rank),
            Int32(box.cap),
        )
        self.out_args = (Int64(box.out.mc), Int32(rank))
        # Row-sharded batches (``sharded`` = (batches, attn_res.sharded.ShardBox)): the
        # output rows go to their row owners' ring, which the owner's AttnRes polls.
        self.owners = {}
        if sharded is not None and sharded[1] is not None:
            peers = to_cute(sharded[1].ring_peers, 16)
            for b in set(batches) & set(sharded[0]):
                code = compile_rows("out_owners", tp, b, box.cap, device, chunk)
                self.owners[b] = code, (peers, Int32(rank))

    def push(self, shared):
        """Side stream, after the shared expert: its rows into the owners' RS slots."""
        b = shared.shape[0]
        self.codes[b][0](to_cute(shared), *self.push_args, stream(shared.device))

    def run(self, partial, b, join=None):
        """The tail of ``b`` rows from this rank's routed partial; the output lands in
        the ring the next AttnRes polls. ``join``: makes the stream wait for the
        shared rows' push (the caller's side stream) after LatentRows, before NormRows,
        whose CTAs spin on the RS slots that push fills: queued beside a push that has
        not been placed yet, the spinning CTAs can take every SM it needs (b at or
        above the SM count: the push never runs, every rank's NormRows waits)."""
        _, latent, norm, out = self.codes[b]
        if b in self.experts_push and isinstance(partial, UnfinalizedMoEOutput):
            raise RuntimeError(f"B{b}: the wide tail expects the experts' push")
        if isinstance(partial, UnfinalizedMoEOutput):
            box, s = self.box, stream(partial.gemm2_permuted.device)
            weights, index = (
                partial.expert_weights,
                partial.expanded_idx_to_permuted_idx,
            )
            code = self.deferred[b][weights.dtype]
            routing = (to_cute(weights[:b]), to_cute(index[:b]))
            code(dynamic_view(partial.gemm2_permuted), *routing, *self.latent_args, s)
        else:
            box, s = self.box, stream(partial.device)
            latent(to_cute(partial[:b]), *self.latent_args, s)
        if join is not None:
            join()
        xnorm, up, shared = box.xnorm[:b], box.up[:b], box.shared[:b]
        norm(
            to_cute(xnorm),
            self.gather,
            self.gamma,
            self.rs,
            to_cute(shared),
            self.eps,
            s,
        )
        if b in self.fused_rows and b not in self.owners:
            from ...k3r_gemm.ull.projection import matrix_view

            weight, x = matrix_view(self.up_shard), matrix_view(xnorm)
            self.fused_up(weight, x, dynamic_view(shared), *self.fused_args, s)
            return
        if self.projection is not None:
            self.projection.run(xnorm, up)
        else:
            torch.mm(xnorm, self.up_shard.t(), out_dtype=torch.float32, out=up)
        if b in self.owners:
            code, args = self.owners[b]
            code(to_cute(up), to_cute(shared), *args, s)
            return
        out(to_cute(up), to_cute(shared), *self.out_args, s)
