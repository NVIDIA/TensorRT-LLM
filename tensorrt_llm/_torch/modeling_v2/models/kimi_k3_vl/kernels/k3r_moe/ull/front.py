# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""MoE front: the weight-stationary [router shard; down shard] GEMM publishes logits
and the latent shard into a multicast mailbox row; the routing kernel polls all 896
logits, ranks the top16 (sigmoid + bias, renormalized) and Lamport-copies the latent."""

import functools

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ....ops.common.k3_ull_cute_host import fake, prepare_shared_carveout, stream, to_cute
from ...k3r_common.ull.mailbox import poll, rearm
from ...k3r_common.ull.policy import batch_table
from ...k3r_common.ull.primitives import (
    alloc,
    asm,
    bits_float,
    div_rn,
    float_bits,
    launch,
    redux,
    sigmoid,
    sigmoids,
    store_words,
    view,
)
from ...k3r_gemm.ull.projection import PreparedProjection, matrix_view, weight_map
from ...k3r_gemm.ull.splitk_situ import PreparedSitu
from ...k3r_gemm.ull.ws_gemm import WsGemm
from .experts import PLAN_BYTES, plan_chunk
from .experts_rubin import ROUTE_WORDS, count_image, image_chunk
from .experts_runtime import polled, rubin
from .shared import compile_down, compile_shared, situ, weight_maps
from .tail_runtime import CAP


@asm(Int32)
def shared_add(address, value):
    code = "atom.shared.add.u32 $0,[$1],$2;"
    return code, "=r,r,r,~{memory}", (address, Int32(value))


@cute.jit
def ordered(score):
    """Monotone uint32 key of a float; NaN maps to 0 (below every number)."""
    bits = Uint32(float_bits(score))
    key = ~bits if (bits & Uint32(0x80000000)) != 0 else bits | Uint32(0x80000000)
    return key if score == score else Uint32(0)


class RouteSitu:
    def __init__(
        self, batch, shared, renorm, early, roles, polled, plan=False, lean=None,
        rcp=False, halves=False, signal=False, sync_copy=False, early_ids=False,
        nv=None,
    ):  # fmt: skip
        # Lean (B>8 by default): few registers, so routing CTAs co-reside with the front
        # and the fused shared expert, polled logits skip the front's completion wait,
        # and 8 threads rank each candidate.
        self.lean = batch > 8 if lean is None else lean
        self.batch, self.shared, self.experts, self.renorm = batch, shared, 896, renorm
        # early: release the persistent experts at entry (safe once down drains first);
        # polled: logits arrive as FP32 words in mailbox columns [3584, 5376) (Lamport);
        # plan: the copy role also writes the latent as MXFP8 into the experts' plan
        # ("image": in the Rubin experts' SMEM image layout).
        self.early, self.polled, self.plan = early, polled, plan
        # rcp (lean, polled): the four exact sigmoids per thread share one fallback
        # branch (``sigmoids``) instead of a div.rn slow-path region each.
        self.rcp = rcp and self.lean and polled
        self.roles = tuple(r for r, on in zip(("route", "copy", "situ"), roles) if on)
        # halves (route + copy, B<=64): two copy CTAs per token, each thread one
        # 8-column chunk (instead of one CTA with 192 threads on two chunks), so the
        # gather and the plan write after the last latent fragment lands are one chunk
        # deep.
        self.halves = halves and self.roles == ("route", "copy") and batch <= 64
        # signal (halves, image plan): the route role also writes the ids into the
        # experts' route mailbox and each copy CTA counts its image part there, so the
        # experts start before this kernel completes (``RubinExpertsKernel.polled``).
        self.signal = signal
        assert not signal or (self.halves and self.lean and plan == "image")
        # sync_copy (one copy CTA per token): every warp's chunks land before any warp
        # stores. The token's row arrives from different ranks at different times; a
        # warp that writes the plan (block maxima and scale words by shuffles) while
        # its sibling warps still spin on volatile loads has its shuffles queued behind
        # those loads (the SM's MIO pipe): ~2.2 -> 0.76 us for the plan writes at B64.
        self.sync_copy = sync_copy and not self.halves
        # early_ids (signal, lean): the thread that ranks a chosen expert also stores
        # its id into the experts' route mailbox (each word is valid on its own), so
        # the experts read the ids before select's last barrier and warp 0's publish.
        self.early_ids = early_ids and signal and self.lean
        # TensorRT-LLM local change (K3 chain, NVFP4 routed experts): ``nv`` (the image
        # and plan writers of the NVFP4 Rubin experts, ``ull/nv_experts.image_chunk_nv`` /
        # ``plan_chunk_nv``): the copy role writes the latent as NVFP4 under the layer's
        # activation global scale (a runtime FP32 argument after ``plan`` / ``signal``)
        # instead of MXFP8 (``image_chunk`` / ``plan_chunk``). None: vLLM MR8 unchanged.
        self.nv = nv
        assert nv is None or (plan and batch <= 64)

    def exits_unwaited(self):
        """Whether no thread would execute griddepcontrol.wait: every role is the
        polled lean route or the (Lamport) copy, and the release is not after the
        front (TensorRT-LLM local; only rows above 8 are gated)."""
        waits = self.early == "front" or "situ" in self.roles
        waits = waits or ("route" in self.roles and not (self.polled and self.lean))
        return not waits and self.batch > 8

    @cute.jit
    def __call__(self, *args):
        ctas = (len(self.roles) + self.halves) * self.batch
        launch(self.kernel, args, ctas, 256)

    @cute.kernel
    def kernel(self, args):
        logits, bias, ids, weights, mailbox, latent, gate_up, hidden = args[0:8]
        scale, beta, linear_beta = args[8:11]
        plan = args[11] if self.plan else None
        signal = args[11 + bool(self.plan)] if self.signal else None
        # TensorRT-LLM local change: ``nv``'s activation global scale
        gsf = args[11 + bool(self.plan) + self.signal] if self.nv is not None else None
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        token, role = cta % self.batch, cta // self.batch
        if c.const_expr(self.halves):  # CTAs [B, 3B): the two copy halves of each token
            role = Int32(cta >= self.batch)
        if c.const_expr(self.early == "front"):
            # Release the experts once this rank's front GEMM has completed: their
            # launch then never overlaps it (they still wait for this kernel's end).
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        elif c.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        for r in c.range_constexpr(len(self.roles)):
            if role == r:
                if c.const_expr(self.roles[r] == "route"):
                    work = (logits, bias, ids, weights, scale, token, tid, mailbox)
                    self.route(*work, signal)
                elif c.const_expr(self.roles[r] == "copy"):
                    # Lamport: no dependency wait; the sentinel poll is the barrier.
                    if c.const_expr(self.halves):
                        if tid < 224:  # whole warps: the image's block shuffles
                            self.copy_half(mailbox, latent, token, tid, plan, gsf)
                        if c.const_expr(self.signal):
                            # The CTA's image part, then its count (release), then
                            # the mailbox re-arm (not ahead of the count's release).
                            cute.arch.sync_threads()
                            if tid == 0:
                                count_image(signal.iterator + 128)
                            if tid < 224:
                                rearm(self.copy_cell(mailbox, token, tid))
                    else:
                        base = mailbox.iterator + token * mailbox.shape[2] + tid * 8
                        if tid < 192:
                            self.copy(base, latent, token, tid, 2, plan, gsf)
                        else:
                            self.copy(base, latent, token, tid, 1, plan, gsf)
                else:
                    cute.arch.griddepcontrol_wait()
                    rounds = (self.shared + 255) // 256  # issue every load first
                    pair = cute.make_rmem_tensor((rounds, 2), c.Float32)
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            pair[i, 0] = gate_up[token, tid + 256 * i]
                            pair[i, 1] = gate_up[token, self.shared + tid + 256 * i]
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            gate = c.Float32(c.BFloat16(pair[i, 0]))
                            up = c.Float32(c.BFloat16(pair[i, 1]))
                            value = situ(gate, up, beta, linear_beta)
                            hidden[token, tid + 256 * i] = c.BFloat16(value)
        # Late release: the persistent experts must not take SMs from the down GEMM.
        if c.const_expr(not self.early):
            cute.arch.griddepcontrol_launch_dependents()
        if c.const_expr(self.exits_unwaited()):
            # TensorRT-LLM local change: no role waited (the lean polled route and the
            # copy roles, 9-64 rows), so one thread waits for the predecessor grids
            # before this grid can complete (after the release: the dependents still
            # launch as before). This kernel's completion then implies the front's,
            # every CTA of it: the shared expert's gate / up tiles (``h`` for
            # SharedDown, ordered behind this kernel's completion) and the
            # transitive-completion chain of ``kda.head.POLLED_OUT``.
            if (cta == 0) & (tid == 0):
                cute.arch.griddepcontrol_wait()

    @cute.jit
    def select(self, key, threshold, buffers, signal=None, token=None):
        """B>8 candidates (keys at or above ``threshold``) append to ``dense`` at a
        per-warp offset from the shared count, then rank themselves: 8 threads per
        candidate (4 past 32 candidates) each count their share of the candidates above
        it (key, then lower expert). Ranks below 16 store the expert and its sigmoid."""
        dense, sig, chosen, picked, found = buffers
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        packed = cute.make_rmem_tensor(4, Uint64)
        ballots = cute.make_rmem_tensor(4, Uint32)
        total = Int32(0)
        for i in c.range_constexpr(4):  # packed 0: not a candidate
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballots[i] = cute.arch.vote_ballot_sync(hit)
            packed[i] = Uint64(0)
            if hit:
                packed[i] = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
            total += cute.arch.popc(ballots[i])
        offset = Int32(0)
        if (lane == 0) & (total > 0):
            offset = shared_add(found.iterator, total)
        offset = cute.arch.shuffle_sync(offset, 0)
        below = (Uint32(1) << lane) - Uint32(1)
        for i in c.range_constexpr(4):
            if packed[i] != Uint64(0):
                dense[offset + cute.arch.popc(ballots[i] & below)] = packed[i]
            offset += cute.arch.popc(ballots[i])
        cute.arch.sync_threads()
        n = found[0]
        if n <= 32:  # usually: 8 threads per candidate, 4 slots each
            part, index = tid % 8, tid // 8
            mine, rank = Uint64(0), Int32(0)
            if index < n:
                mine = Uint64(dense[index])
            for k in c.range_constexpr(4):  # slots past n do not count
                j = part + 8 * k
                rank += Int32((j < n) & (Uint64(dense[j]) > mine))
            for shift in c.range_constexpr(3):
                rank += cute.arch.shuffle_sync_bfly(rank, 1 << shift)
            if (part == 0) & (index < n) & (rank < 16):
                expert = Int32(~Uint32(mine))
                chosen[rank], picked[rank] = expert, sig[expert]
                if c.const_expr(self.early_ids):
                    signal[token * 16 + rank] = expert
        else:
            part = tid % 4
            for first in c.range(0, n, 64, unroll=1):
                index = first + tid // 4
                mine, rank = Uint64(0), Int32(0)
                if index < n:
                    mine = Uint64(dense[index])
                if n <= 64:  # every load issues at once; slots past n do not count
                    for k in c.range_constexpr(16):
                        j = part + 4 * k
                        rank += Int32((j < n) & (Uint64(dense[j]) > mine))
                else:
                    for j in c.range(part, n, 4, unroll=2):
                        rank += Int32(Uint64(dense[j]) > mine)
                rank += cute.arch.shuffle_sync_bfly(rank, 1)
                rank += cute.arch.shuffle_sync_bfly(rank, 2)
                if (part == 0) & (index < n) & (rank < 16):
                    expert = Int32(~Uint32(mine))
                    chosen[rank], picked[rank] = expert, sig[expert]
                    if c.const_expr(self.early_ids):
                        signal[token * 16 + rank] = expert
        cute.arch.sync_threads()

    @cute.jit
    def publish(self, chosen, picked, ids, weights, scale, token, signal=None):
        """Warp 0: the 16 ids by rank and their sigmoids renormalized (lane r holds
        rank r; the butterfly sums in the same order for every lane); ``signal``: the
        ids also into the experts' route mailbox."""
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        if tid < 32:
            value = c.Float32(0)
            if lane < 16:
                ids[token, lane] = chosen[lane]
                if c.const_expr(signal is not None and not self.early_ids):
                    signal[token * 16 + lane] = chosen[lane]
                value = picked[lane]
            total = value
            for shift in c.range_constexpr(4):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane < 16:
                value = value * scale
                if c.const_expr(self.renorm):
                    value = div_rn(value, total + c.Float32(1e-20))
                weights[token, lane] = c.BFloat16(value)

    def expert(self, tid, i):
        return tid * 4 + i if self.polled else tid + i * 256

    @cute.jit
    def copy_cell(self, mailbox, token, tid):
        """The mailbox cell of ``copy_half``'s chunk."""
        cta, _, _ = cute.arch.block_idx()
        chunk = (cta - self.batch) // self.batch * 224 + tid
        return mailbox.iterator + token * mailbox.shape[2] + chunk * 8

    @cute.jit
    def copy_half(self, mailbox, latent, token, tid, plan=None, gsf=None):
        """``copy`` of chunk 224 h + tid (h: this CTA's half of the token's row);
        ``signal``: the caller re-arms the cell after counting the image."""
        cta, _, _ = cute.arch.block_idx()
        chunk = (cta - self.batch) // self.batch * 224 + tid
        base = mailbox.iterator + token * mailbox.shape[2] + chunk * 8
        got = poll(base, 0, 1)
        dst = latent.iterator + token * latent.stride[0] + chunk * 8
        words = [got[0, k] for k in range(4)]
        store_words(dst, words)
        if c.const_expr(not self.signal):
            rearm(base)
        if c.const_expr(plan is not None and self.batch > 64):
            # 64-row plans: token t at row t % 64 of block t // 64 (as ``copy``)
            block = view(plan.iterator, token // 64 * PLAN_BYTES, c.Uint8, PLAN_BYTES)
            plan_chunk(block, token % 64, chunk, words)
        elif c.const_expr(plan is not None and self.nv is not None):  # TensorRT-LLM local
            self.nv[int(self.plan != "image")](plan, token, chunk, words, gsf)
        elif c.const_expr(plan is not None):
            write = image_chunk if self.plan == "image" else plan_chunk
            write(plan, token, chunk, words)

    @cute.jit
    def copy(
        self, base, latent, token, tid, count: c.Constexpr[int], plan=None, gsf=None
    ):  # fmt: skip
        got = poll(base, 2048, count)
        if c.const_expr(self.sync_copy):  # every thread of the CTA: once per path
            cute.arch.sync_threads()
        for i in c.range_constexpr(count):
            dst = latent.iterator + token * latent.stride[0] + (tid + 256 * i) * 8
            words = [got[i, k] for k in range(4)]
            store_words(dst, words)
            rearm(base + i * 2048)
            if c.const_expr(plan is not None and self.batch > 64):
                # 64-row plans: token t at row t % 64 of block t // 64
                block = view(
                    plan.iterator, token // 64 * PLAN_BYTES, c.Uint8, PLAN_BYTES
                )
                plan_chunk(block, token % 64, tid + 256 * i, words)
            elif c.const_expr(plan is not None and self.nv is not None):  # TRT-LLM local
                self.nv[int(self.plan != "image")](plan, token, tid + 256 * i, words, gsf)
            elif c.const_expr(plan is not None):
                write = image_chunk if self.plan == "image" else plan_chunk
                write(plan, token, tid + 256 * i, words)

    @cute.jit
    def route(
        self, logits, bias, ids, weights, scale, token, tid, mailbox, signal=None
    ):
        """Exact top16 by threads [0, 256): candidates at or above the 16th largest
        lane-group maximum are compacted and ranked by an unrolled 64-way compare."""
        warp, lane = tid // 32, tid % 32
        smem = c.utils.SmemAllocator()
        km = alloc(smem, Uint32, (8, 32))
        cand, dense = (alloc(smem, Uint64, shape) for shape in ((8, 128), 64))
        sig = alloc(smem, c.Float32, 1024)
        counts = alloc(smem, Int32, 8)
        cutoff = alloc(smem, Uint32, 1)
        chosen = alloc(smem, Int32, 16)
        if c.const_expr(self.lean):  # candidate count; the chosen experts' sigmoids
            found, picked = alloc(smem, Int32, 1), alloc(smem, c.Float32, 16)
        # Thread t owns experts t + 256 i, or 4 t + i when polling 16-byte chunks.
        offsets = cute.make_rmem_tensor(4, c.Float32)
        for i in c.range_constexpr(4):  # bias is a weight: load before the wait
            offsets[i] = c.Float32(0)
            if self.expert(tid, i) < self.experts:
                offsets[i] = bias[self.expert(tid, i)]
        raw = cute.make_rmem_tensor(4, c.Float32)
        key, top = cute.make_rmem_tensor(4, Uint32), Uint32(0)
        for i in c.range_constexpr(4):
            key[i] = Uint32(0)
        if c.const_expr(self.polled):
            if tid < 224:
                cell = mailbox.iterator + token * mailbox.shape[2] + 3584 + tid * 8
                got = poll(cell, 0, 1)
                rearm(cell)
                for i in c.range_constexpr(4):
                    raw[i] = bits_float(got[0, i])
                if c.const_expr(self.lean):  # all four experts: one block, no branches
                    if c.const_expr(self.rcp):
                        scores = list(sigmoids(raw[0], raw[1], raw[2], raw[3]))
                    else:
                        scores = [sigmoid(raw[i]) for i in range(4)]
                    for i in c.range_constexpr(4):
                        sig[4 * tid + i] = scores[i]
                        key[i] = ordered(scores[i] + offsets[i])
            if c.const_expr(not self.lean):
                cute.arch.griddepcontrol_wait()
        else:
            cute.arch.griddepcontrol_wait()
            for i in c.range_constexpr(4):
                if self.expert(tid, i) < self.experts:
                    raw[i] = logits[token, self.expert(tid, i)]
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            if c.const_expr(not (self.lean and self.polled)):  # noqa: SIM102
                if expert < self.experts:
                    score = sigmoid(raw[i])
                    sig[expert] = score
                    key[i] = ordered(score + offsets[i])
            top = max(top, key[i])
        km[warp, lane] = top
        if c.const_expr(self.lean):  # noqa: SIM102
            if tid == 0:
                found[0] = Int32(0)
        cute.arch.sync_threads()
        if warp == 0:
            group = km[0, lane]
            for w in c.range_constexpr(1, 8):
                group = max(group, km[w, lane])
            km[0, lane] = group
            cute.arch.sync_warp()
            rank = Int32(0)
            for j in c.range_constexpr(32):
                rank += Int32(km[0, j] > group)
            low = redux(group if rank < 16 else Uint32(0xFFFFFFFF))
            if lane == 0:
                cutoff[0] = low
        cute.arch.sync_threads()
        if c.const_expr(self.lean):
            dense = view(cand.iterator, 0, Uint64, (1024,))
            buffers = (dense, sig, chosen, picked, found)
            self.select(key, cutoff[0], buffers, signal, token)
            self.publish(chosen, picked, ids, weights, scale, token, signal)
            return
        threshold, offset = cutoff[0], Int32(0)
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballot = cute.arch.vote_ballot_sync(hit)
            if hit:
                below = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
                packed = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
                cand[warp, offset + below] = packed
            offset += cute.arch.popc(ballot)
        if lane == 0:
            counts[warp] = offset
        cute.arch.sync_threads()
        if warp == 0:
            n = Int32(0)
            for v in c.range_constexpr(8):
                n += counts[v]
            for half in c.range_constexpr(2):  # compact the regions densely
                index = lane + 32 * half
                rest, region = Int32(index), Int32(0)
                for v in c.range_constexpr(8):
                    if (region == v) & (rest >= counts[v]):
                        rest -= counts[v]
                        region += 1
                value = Uint64(0)
                if index < n:
                    value = cand[region, rest]
                dense[index] = value
            cute.arch.sync_warp()
            if n <= 64:
                for half in c.range_constexpr(2):
                    mine, rank = dense[lane + 32 * half], Int32(0)
                    for j in c.range_constexpr(64):
                        rank += Int32(dense[j] > mine)
                    if (lane + 32 * half < n) & (rank < 16):
                        chosen[rank] = Int32(~Uint32(mine))
            else:  # rare: rank every candidate against all regions
                for q in range(lane, n, 32):
                    rest, region = Int32(q), Int32(0)
                    for v in c.range_constexpr(8):
                        if (region == v) & (rest >= counts[v]):
                            rest -= counts[v]
                            region += 1
                    mine, rank = cand[region, rest], Int32(0)
                    for v in c.range_constexpr(8):
                        for j in range(counts[v]):
                            rank += Int32(cand[v, j] > mine)
                    if rank < 16:
                        chosen[rank] = Int32(~Uint32(mine))
            cute.arch.sync_warp()
            expert, value = chosen[lane % 16], c.Float32(0)
            if lane < 16:
                ids[token, lane] = expert
                value = sig[expert]
            total = value
            for shift in c.range_constexpr(4):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane < 16:
                value = value * scale
                if c.const_expr(self.renorm):
                    value = div_rn(value, total + c.Float32(1e-20))
                weights[token, lane] = c.BFloat16(value)


# Front GEMM geometry "ks.kb.tg.rt.ch" per batch, by packed rows (TP8 560, TP16 288),
# from GB300 sweeps. ks: cluster K-split; kb: 32-wide
# K blocks per warp, so a row tile takes 224 / (ks * kb) warps (at most 16); tg: tokens
# per group (0: the single-pass path); rt: 16-row tiles per CTA; ch: accumulator chains
# per 8-token tile (0: 4 / tiles per group).
FRONT = {
    560: "1-4:4.14.0.1.0,5-32:16.7.16.5.2,33-64:16.14.16.5.2",
    288: "1-4:4.14.0.1.0,5-64:16.7.16.3.2",
}
# Devices with more SMs than GB300's 152 (Rubin: 212), TP16 B3-8: an 8-CTA K-split over
# 48 CTAs beside the 7-way split shared gate/up (84 CTAs, ``SHARED_WIDE``); together
# they finish 0.4-0.6 us sooner after the o-projection than 96 + 48 CTAs.
FRONT_WIDE = {288: "1-4:4.14.0.1.0,3-8:8.7.16.3.2,9-64:16.7.16.3.2"}
# Those devices, TP16 B3-8: routing on the lean path (8 threads rank each candidate, no
# front completion wait), 0.55-0.6 us sooner after the logits land; and the split-K
# chain's SiTU releases the shared down GEMM at entry, so the down launches and preloads
# its weights during the SiTU (shared chain 1.2-1.3 us shorter). Their 212-SM grids keep
# the experts waiting for both chains, so only the pair pays: -0.2 to -0.7 us per layer
# in a one-GPU layer proxy. Bitwise-identical outputs.
ROUTE_LEAN_WIDE = {288: "3-8:1"}
SITU_EARLY_WIDE = {288: "3-8:1"}
# Those devices, TP16 B3-8 (gate/up tactic M tile 64): SiTU in the gate/up GEMM's
# epilogue (``SplitKSitu``: 32 gate + 32 up rows per tile, the same K split and
# reduction order), so the shared down depends on the GEMM directly and the SiTU kernel
# and its launch boundary leave the shared chain. Bitwise-identical h.
SITU_FUSED_WIDE = {384: "3-8:1"}
# Those devices, the batches whose experts ``release_early`` lets launch early: routing
# releases them once this rank's front GEMM has completed (``RouteSitu`` early="front")
# instead of at routing entry. With the entry release, the 212-CTA experts grid comes
# up while the front GEMM is still running, and in serving (16 ranks, TP16 B8) 43% of
# front GEMM instances then end ~1.9 us late on every rank alike; routing waits for
# the slowest rank's. The experts still launch and run their prologue ~4 us before
# routing ends. MoE front proxy (routing's PDL primary the front GEMM, other ranks'
# rows 1 us later), us per layer: -0.24 (B8), -0.05..-0.11 (B5). Bitwise identical.
# M64 (B8 verify): the merged front kernel triggers at entry, so an entry release
# brings the 208-CTA grid up beside it; after it: -0.7 us per layer (one-GPU proxy).
RELEASE_AFTER_FRONT_WIDE = {288: "3-8:1,64:1"}
# Those devices, TP16 B3-8 (lean polled routing): each thread's four exact sigmoids
# share one div.rn fallback branch (``primitives.sigmoids``, bitwise equal on all 2^32
# inputs) instead of a slow-path region per division, so their fast paths interleave:
# routing's post-arrival time -0.11 us (one GPU, B8 and B5). Same ids and weights.
# Rows 9-56: with COPY_HALVES_WIDE, below.
ROUTE_RCP_WIDE = {288: "3-56:1"}
# Those devices, TP16 B8: routing's copy role on two CTAs per token, one 8-column chunk
# per thread (``RouteSitu`` halves). Same latent and image bytes. Rows 9-56, with the
# rcp sigmoids (one GPU, front + routing behind a 3 us stand-in predecessor, the other
# ranks' rows landed, us per call): -0.21..-0.47 at M12-32 (front kernel), -0.27..-0.60
# at M36-56 (merged front); same ids, weights, latent and plan bytes.
COPY_HALVES_WIDE = {288: "8-56:1"}
# Those devices, batches whose route role stores each chosen id into the experts' route
# mailbox from the ranking thread (``RouteSitu.early_ids``; with ``signal`` only).
EARLY_IDS_WIDE = {288: "8:1"}
# Those devices, TP16 B64 (one copy CTA per token): a CTA barrier between the copy
# role's latent poll and its stores (``RouteSitu`` sync_copy). Two ranks, the M64 MoE
# layer (front -> routing -> experts): the plan writes 2.2 -> 0.76 us, the layer -1.23
# us. Same latent and plan bytes.
COPY_SYNC_WIDE = {288: "64:1"}
# Rubin verify steps (q > 1), by front rows: batches whose routing kernel runs with the
# whole-SM shared memory carve-out (it uses 14 KB). An SM keeps the carve-out of the
# kernel it runs, so with the default (132 KB) the routed experts CTAs (212 KB),
# launched beside routing by PDL, could not start on routing's SMs until it exited.
# B64 (B8 x q8): routing's 128 CTAs kept the M64 experts (212 KB) off their SMs the
# same way (2.11 us p50 overlap in the ds16 B8 serving capture).
ROUTE_CARVEOUT_WIDE = {288: "8:1,64:1"}
# Those devices, TP16 B3-8 (token-group geometry): the front GEMM issues its register
# weight loads at entry, ahead of the barrier setup. Issued right before the PDL wait,
# ptxas schedules the wait first and every weight load after it (SASS: ACQBULK, then the
# 14 LDG.E.NA.128.CONSTANT), so the 4.1 MB of front weights streamed only after the
# o-projection completed although the CTAs are resident ~4 us earlier. Same loads and
# arithmetic: bitwise-identical logits and latent.
EARLY_WEIGHTS_WIDE = {288: "3-8:1"}
# Those devices, TP16 B8: 16-row tiles per front GEMM cluster, in launch order. Beside
# the o-projection (112 CTAs) and the shared gate/up (84) only 2-3 of the 8-CTA clusters
# fit, so the rest are placed when the o-projection ends and stream their weights then
# (86 KB per CTA at three tiles: ~0.6 us through one SM, ~2 us when it stalls). The
# clusters placed first keep three tiles; the late ones get two or one (57 / 29 KB per
# CTA, 64 CTAs). Same K split and summation order per row: bitwise-identical outputs.
FRONT_TILES_WIDE = {288: "8:3.3.3.2.2.2.2.1"}
WIDTH = 3584 + 2 * 896  # front mailbox row: BF16 latent, then FP32 logit words
# Batches whose shared expert is the fused kernel (``SharedUmma``), not the split-K
# chain; of those, split batches run gate/up + SiTU beside the front and the down rows
# (``SharedDown``) on SPLIT SMs beside the routed experts (0: fused).
FUSED, SPLIT = "9-64:1", "9-12:6,13-24:4,25-64:2"
# Per-TP overrides of SPLIT entries, (q = 1, verify steps): on real prompts (GB300
# SPEED-Bench 8k mixed) SharedDown on 2 SMs (~123 us at M32) outlasts the experts (TP8
# q=1 B32: 3 SMs, -5.4% step). Verify steps (q > 1) share experts among a request's q
# tokens: TP8 DL7 M16/32/64 -5.9/-12.9/-1.1%, TP16 M32 -5.9%.
SPLITS = {8: ("25-32:3", "13-16:6,25-32:4,33-64:3"), 16: ("", "25-32:3")}
# ``SPLITS`` entries overridden on devices with more SMs than GB300's 152 (Rubin: 212),
# (q = 1, verify steps). TP16 verify M64 (B8 x q8): real DSpark verify routes reach
# ~300 distinct experts per layer (8 requests' 8 consecutive tokens share experts), so
# the routed experts take ~25-50 us and SharedDown on 2 SMs (~58 us) set the layer.
# One-GPU MoE layer proxy (real q8 routes of 11 layers, 150-391 distinct), us/layer:
# 2 SMs 72.7-73.1 at every layer; 4 SMs 45.6-59.3 (-13.5..-27.4); 3/5/6/8 SMs within
# +-0.6 of 4 except 3 at the widest layers. TP16 verify 13-63 rows: with the routed
# experts on the grouped 64-row kernel, SharedDown on 4 / 3 / 2 SMs outlasts them at
# every layer (the proxy reads a flat 41 / 43 / 53 / 71-73 us at M16 / 24 / 32 /
# 40-56); on 8 SMs, us/layer over 3-8 real-route layers (79-300 distinct): M16
# -1.2..-8.9, M24 -2.4..-5.6, M32 -5.4..-16.1, M40-56 -17.1..-30.4 (6 SMs within
# 0.5 of 8; M64 stays at 4: 6 / 8 within -1.1..+0.6).
SPLITS_WIDE = {16: ("", "13-63:8,64:4")}
# Split batches whose front GEMM rows join the gate/up kernel as extra M-tiles (one
# kernel instead of two that slow each other ~2x side by side).
MERGED = {8: "41-64:1", 16: "33-64:1"}
# ``MERGED`` entries added on devices with more SMs than GB300's 152 (Rubin), by TP:
# TP16 verify 16-32 rows, the merged front + gate/up kernel instead of the front GEMM
# beside the split gate/up (the routed experts join the gate/up there). 1-GPU MoE layer
# proxy (real verify routes of layers 5 / 20 / 54, SharedDown on 8 SMs, the other
# ranks' rows 1-2.5 us after the o-projection), us/layer: M16 -0.24..-0.85, M24
# -0.68..-1.26, M32 -1.20..-1.43 (q8) / -1.56 (q4); with the rows 4 us late M16-32
# +0.03..-0.31. The front's K split changes (a stated FP32 reorder of the logits and
# the latent rows); top-16 ids identical on the test routes.
MERGED_WIDE = {16: "16-32:1"}
# Devices with more SMs than GB300's 152 (Rubin), by packed front rows: ``MERGED``
# batches whose merged front + gate/up kernel is ``MoeFrontTmem64`` (128-row tiles
# split over K in clusters of 8, the weights staged in tensor memory before the PDL
# wait, one-hop DSMEM reduce-scatter; h for SharedDown as before). Front rows
# published 2.6 us after the PDL wait instead of ~5.9. 1-GPU MoE layer proxy (real q8
# routes, SharedDown on 4 SMs, the other ranks' rows 1-2 us after the o-projection),
# us/layer: -0.64 / -1.08 / -0.32 (layers 20 / 54 / 85), -0.52 (layer 20, 2 us).
# Sums 8 split partials in rank order (an FP32 accumulation reorder of the merged
# kernel's 6); top-16 ids identical on the test routes. On later trees (ds22) the
# one-GPU proxy had M64 on the merged kernel 0.34-0.67 us/layer faster, but 16-rank
# serving (ds23 sweep) lost 3.7% at DL7 B8 and 2.8% at DL3 B16 without this kernel:
# kept at M64. 16-32 rows: 16 ranks (32 layers), us per step DL7 B2 1801.7 -> 1766.1,
# DL7 B4 2333.4 -> 2244.6, DL3 B8 2362.5 -> 2283.6 (the 1-GPU MoE layer proxy had read
# +0.8..+1.5 us/layer, the wrong sign as at M64); logits within 1.2e-6 of FP64.
FRONT_TMEM_WIDE = {288: "16-32:1,64:1"}
# Shared-expert split-K tactics (M tile, N tile, split, stages) by width (TP8 768, TP16
# 384): gate_up, then down.
SHARED_TACTICS = {
    768: (
        "1-8:128.8.4.6,9-16:64.16.4.10,17-32:64.32.4.8,33-64:64.64.4.5",
        "1-8:128.8.1.6,9-16:64.16.1.6,17-32:64.32.1.6,33-64:64.64.1.6",
    ),
    384: (
        "1-16:64.8.4.8,17-32:64.16.4.10,33-64:64.32.4.8",
        "1-8:128.8.1.3,9-16:64.16.1.3,17-32:64.32.1.3,33-64:64.64.1.3",
    ),
}
# ``SHARED_TACTICS`` entries overridden on those devices (gate/up at B<=8: split 7).
SHARED_WIDE = {384: ("1-8:64.8.7.8", "")}


@functools.cache
def front_table(n, wide=False):
    return batch_table((FRONT_WIDE if wide and n in FRONT_WIDE else FRONT).get(n, ""))


def front_ks(batch, n, wide=False):
    return front_table(n, wide).get(batch, (4,))[0]


def front_geometry(batch, n, ks, kb=None, tg=None, rt=None, ch=None, wide=False):
    """(rt, kq, staged, tg, ch) of the front GEMM over ``n`` rows. Unset arguments come
    from the table when ``ks`` is its K-split, else from the single pass; token groups
    whose x slice does not fit in SMEM fall back to the single pass."""
    entry = front_table(n, wide).get(batch, ())
    table = entry[1:] if entry[:1] == (ks,) else (14, 0, 1, 0)
    kb, tg, rt, ch = (t if v is None else v for v, t in zip((kb, tg, rt, ch), table))
    kq = min(16, 224 // (ks * kb))
    kb, nt = 224 // (ks * kq), -(-batch // 8)
    staged = max(kb - 8, 0) if kq == 16 else 0
    used = (
        rt * kq * (nt * 544 + max(staged, 1) * 1024) + -(-nt * rt // ks) * ks * kq * 512
    )
    if nt * 8 * (7168 // ks * 2 + 64) > 220 * 1024 - used:
        tg = 0
    return rt, kq, staged, tg, ch


@functools.cache
def compile_route(
    batch, shared, renorm, roles, device, polled=False, plan=False, lean=None,
    early=None, rcp=False, halves=False, signal=False, sync_copy=False,
    early_ids=False, box_rows=64, nv=None,
):  # fmt: skip
    """``box_rows``: the front mailbox's rows (``FrontRows``: its capacity); above 64
    tokens the plan is 64-row blocks of PLAN_BYTES. ``nv`` (TensorRT-LLM local change):
    ``RouteSitu``'s NVFP4 writers; the kernel then takes the activation global scale."""
    early = batch <= 2 if early is None else early
    kernel = RouteSitu(
        batch, shared, renorm, early, roles, polled, plan, lean, rcp, halves, signal,
        sync_copy, early_ids, nv,
    )  # fmt: skip
    f32 = c.Float32
    args: tuple = (fake((batch, 896), None, f32), fake((896,), None, f32))
    box = fake((1, box_rows, WIDTH))
    args += (fake((batch, 16), None, c.Int32), fake((batch, 16)), box)
    args += (fake((batch, 3584)), fake((batch, 2 * shared), None, f32))
    args += (fake((batch, shared)), *(f32(v) for v in (1, 4, 25)))
    blocks = -(-batch // 64)
    args += (fake((blocks * PLAN_BYTES,), None, c.Uint8),) * bool(plan)
    args += (fake((ROUTE_WORDS,), None, c.Int32),) * signal
    args += (f32(1),) * (nv is not None)  # TensorRT-LLM local: ``nv``'s scale
    args += (make_fake_stream(),)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args)


@functools.cache
def compile_front_tmem(split, valid, shard, hidden, device):
    """``MoeFrontTmem64``: the packed front rows (router ``valid`` padded to
    ``split``, then the latent shard) and ``hidden`` gate/up pairs."""
    from .front_tmem import MoeFrontTmem64

    kernel = MoeFrontTmem64(valid=valid, split_rows=split, shard=shard, hidden=hidden)

    def matrix():
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (cute.sym_int(64), 1, 0)
        return make_fake_tensor(c.BFloat16, shape, stride, assumed_align=16)

    args: tuple = (matrix(), matrix(), fake((64, hidden)), fake((1, 64, WIDTH)))
    args += (c.Int64(0), Int32(0), Int32(0), c.Float32(1), c.Float32(1))
    args += (c.Int64(0), c.Int64(0), make_fake_stream())
    with torch.accelerator.device_index(device):
        arch = "sm_{}{}a".format(*torch.cuda.get_device_capability(device))
        return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


@functools.cache
def compile_publish(n, split, valid, batch, device, ks, geometry=None, wide=False):
    """Front GEMM that only publishes; routing then runs in the polled kernel."""
    rt, kq, staged, tg, ch = geometry or front_geometry(batch, n, ks, wide=wide)
    early = batch_table(EARLY_WEIGHTS_WIDE.get(n, "") if wide else "").get(batch)
    early = bool(early and tg)
    tiles = None
    if wide and tg and geometry is None:
        tiles = batch_table(FRONT_TILES_WIDE.get(n, "")).get(batch)
    kernel = WsGemm(
        n, 7168, batch, rt, kq, staged, split, valid, ks, tg, ch, early, tiles
    )
    args: tuple = (fake((batch, 7168)), fake((n, 7168)), fake((1, 64, WIDTH)))
    args += (c.Int64(0), c.Int32(0), c.Int32(0), fake((batch, 896), None, c.Float32))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


class SharedFront:
    def __init__(
        self, moe, packed, box, storage, batches, beta, linear_beta, plan=None,
        counters=None, q=1, routes=None, nv=None,
    ):  # fmt: skip
        """``plan``: the experts' plan buffer, written by the routing kernel at B>8 and
        at ``experts_runtime.rubin`` batches; ``routes``: the experts' route mailbox
        (``experts_runtime.route_mailbox``), written at ``experts_runtime.polled``
        batches;
        ``counters``: the fused shared expert's model-owned barrier counters; ``q``: the
        deployment's verify length (its SPLITS entries).

        TensorRT-LLM local change (K3 chain, NVFP4 routed experts): ``nv``
        (``k3_vllm.nv_moe.NvFront``): the layer's routed experts are the NVFP4 Rubin
        engines. ``nv.image(b)`` / ``nv.polled(b)`` pick the image and polled batches (in
        place of ``experts_runtime.rubin`` / ``polled``), the routing kernel writes the
        NVFP4 image / plan (``RouteSitu`` ``nv``: ``nv.writers``) under ``nv.gsf``, and
        ``nv.wide_routing`` takes the routing options from the *_WIDE tables on other
        devices too (GB300 forced onto the Rubin experts path). None: vLLM MR8 unchanged."""
        experts = moe.shared_experts
        gate_up, down = experts.gate_up_proj.weight, experts.down_proj.weight
        self.shared = shared = gate_up.shape[0] // 2
        tp = 6144 // shared
        shard, valid = 3584 // tp, 896 // tp
        self.device, self.packed, self.storage = packed.device, to_cute(packed), storage
        sms = torch.cuda.get_device_properties(packed.device).multi_processor_count
        wide = sms > 152  # Rubin-class SM counts: the *_WIDE tables
        tactics = SHARED_TACTICS[shared]
        if wide and shared in SHARED_WIDE:
            tactics = tuple(f"{t},{w}" for t, w in zip(tactics, SHARED_WIDE[shared]))
        gu, dn = map(batch_table, tactics)
        self.gate_up = PreparedProjection(
            gate_up, {b: gu[b] for b in batches}, output_dtype=torch.float32
        ).prepare()
        self.down = PreparedProjection(down, {b: dn[b] for b in batches}).prepare()
        fuse = batch_table(SITU_FUSED_WIDE.get(shared, "") if wide else "")
        fuse = {b: gu[b] for b in batches if fuse.get(b) and gu[b][0] == 64}
        self.gate_situ = (
            PreparedSitu(gate_up, fuse, beta, linear_beta) if fuse else None
        )
        scale, renorm = moe.routed_scaling_factor, moe.moe_renormalize
        self.scalars = tuple(map(c.Float32, (scale, beta, linear_beta)))
        self.box: tuple = (to_cute(box.buffer), c.Int64(box.mc))
        self.box += (c.Int32(box.rank * valid), c.Int32(box.rank * shard))
        bias = to_cute(moe.gate.e_score_correction_bias)
        dev, n = packed.device.index, packed.shape[0]
        self.wide, self.rows = wide, n
        fused = batch_table(FUSED)
        fused = {b for b in batches if b > 8 and counters is not None and fused.get(b)}
        verify = q > 1
        splits = SPLITS[tp][verify]
        if wide and tp in SPLITS_WIDE:
            splits = f"{splits},{SPLITS_WIDE[tp][verify]}"
        split = batch_table(f"{SPLIT},{splits}")
        self.split = {b: split[b] for b in fused if split.get(b)}  # down SMs
        if fused:  # TMA maps and counters of the fused kernel
            self.maps = weight_maps(gate_up, down)
            self.fused = to_cute(self.maps, 64), to_cute(counters)
        merged_spec = MERGED[tp]
        if wide and tp in MERGED_WIDE:
            merged_spec = f"{merged_spec},{MERGED_WIDE[tp]}"
        merged = batch_table(merged_spec)
        self.merged = {b for b in self.split if merged.get(b)}
        tmem = batch_table(FRONT_TMEM_WIDE.get(packed.shape[0], "") if wide else "")
        # MERGED batches on MoeFrontTmem64: code, (packed, gate/up) tensor maps
        self.front_tmem: dict[int, tuple] = {}
        for b in sorted(self.merged):
            if tmem.get(b):
                split_rows, hidden = packed.shape[0] - shard, gate_up.shape[0] // 2
                code = compile_front_tmem(split_rows, valid, shard, hidden, dev)
                if not self.front_tmem:
                    self.tmem_maps = (weight_map(packed, 128), weight_map(gate_up, 64))
                    self.tmem_views = matrix_view(packed)
                maps = tuple(c.Int64(m.data_ptr()) for m in self.tmem_maps)
                self.front_tmem[b] = code, maps
        if self.merged:  # gate/up, down and front maps; unused RS arguments
            self.merged_maps = weight_maps(gate_up, down, front=packed)
            # Kept alive here: a CuTe view does not hold its tensor.
            self.unused = torch.zeros(tp, dtype=torch.int64, device=packed.device)
            self.merged_args = to_cute(self.merged_maps, 64), to_cute(self.unused)
            geometry = (n, n - shard, valid, box.buffer.shape[-1])
        rwide = wide or (nv is not None and nv.wide_routing)  # TensorRT-LLM local
        lean = batch_table(ROUTE_LEAN_WIDE.get(n, "") if rwide else "")
        rcp = batch_table(ROUTE_RCP_WIDE.get(n, "") if rwide else "")
        halves = batch_table(COPY_HALVES_WIDE.get(n, "") if rwide else "")
        sync_copy = batch_table(COPY_SYNC_WIDE.get(n, "") if rwide else "")
        early_ids = batch_table(EARLY_IDS_WIDE.get(n, "") if rwide else "")
        situ_early = batch_table(SITU_EARLY_WIDE.get(n, "") if wide else "")
        carve = ROUTE_CARVEOUT_WIDE.get(n, "") if wide and verify else ""
        self.route_carveout = {b for b, on in batch_table(carve).items() if on}
        self.plans: dict[int, tuple] = {}
        self.route_keys: dict[int, tuple] = {}
        self.route_opts: dict[int, dict] = {}
        for b in batches:
            views = tuple(t[:b] for t in storage)
            logits, gate, ids, weights, h, _, latent = map(to_cute, views)
            ks = front_ks(b, n, wide)
            front = compile_publish(n, n - shard, valid, b, dev, ks, wide=wide)
            planned = plan is not None and b > 8
            image = rubin(tp, b, dev) if nv is None else nv.image(b)  # TRT-LLM local
            if plan is not None and image:
                planned = "image"  # the Rubin experts' SMEM latent image
            roles = (True, True, False)
            self.route_keys[b] = (b, shared, renorm, roles, dev, True, planned)
            self.route_keys[b] += (bool(lean.get(b)) or None,)
            lean_b = bool(lean.get(b)) or b > 8  # RouteSitu's default: lean above 8
            self.route_opts[b] = {"rcp": True} if rcp.get(b) and lean_b else {}
            if halves.get(b):
                self.route_opts[b]["halves"] = True
            if sync_copy.get(b):
                self.route_opts[b]["sync_copy"] = True
            poll = polled(tp, b, dev) if nv is None else nv.polled(b)  # TRT-LLM local
            signal = routes is not None and planned == "image" and poll
            if signal:
                self.route_opts[b]["signal"] = True
                if early_ids.get(b) and lean.get(b):
                    self.route_opts[b]["early_ids"] = True
            if nv is not None:  # TensorRT-LLM local: the NVFP4 image / plan
                assert planned, f"B{b}: the NVFP4 experts need the routing kernel's plan"
                self.route_opts[b]["nv"] = nv.writers
            route = self.carved(
                b, compile_route(*self.route_keys[b], **self.route_opts[b])
            )
            situ = compile_route(
                b, shared, renorm, (False, False, True), dev,
                early=bool(situ_early.get(b)) or None,
            )  # fmt: skip
            args = (logits, bias, ids, weights, self.box[0], latent, gate, h)
            extra = (to_cute(plan),) if planned else ()
            extra += (to_cute(routes),) if signal else ()
            extra += (c.Float32(nv.gsf),) if nv is not None else ()  # TRT-LLM local
            code = None
            if b in self.merged:
                gate = compile_shared(tp, b, dev, True, geometry)
                code = gate, compile_down(tp, b, dev, self.split[b])
            elif b in self.split:
                gate = compile_shared(tp, b, dev, gate=True)
                code = gate, compile_down(tp, b, dev, self.split[b])
            elif b in fused:
                code = compile_shared(tp, b, dev)
            self.plans[b] = views, front, route, situ, args + self.scalars, extra, code

    def route(self, x):
        """Router logits + published down shard, then top16 and the latent copy (and
        at B>8 the experts' MXFP8 plan). Merged batches publish them from the gate/up
        kernel, which also leaves h for ``shared_expert``'s down rows."""
        views, front, route, _, args, extra, code = self.plans[x.shape[0]]
        s = stream(self.device)
        if x.shape[0] in self.front_tmem:
            tmem, maps = self.front_tmem[x.shape[0]]
            work = (self.tmem_views, matrix_view(x), args[7], *self.box)
            tmem(*work, *self.scalars[1:], *maps, s)
        elif x.shape[0] in self.merged:
            maps, unused = self.merged_args
            work = (to_cute(x), maps, args[7], self.fused[1], unused)
            work += (Int32(0), Int32(0), Int32(CAP), *self.scalars[1:], *self.box[1:])
            code[0](*work, s)
        else:
            front(to_cute(x), self.packed, *self.box, args[0], s)
        route(*args, *extra, s)
        return views[2], views[3], views[6]

    def release_early(self, batches):
        """Routing kernels that release the persistent experts at entry, so their launch
        and prologue overlap routing. Only where early experts cannot starve the shared
        expert's down GEMM (``moe.prepared``)."""
        table = RELEASE_AFTER_FRONT_WIDE.get(self.rows, "") if self.wide else ""
        after = batch_table(table)
        for b in set(batches) & self.plans.keys():
            plan = self.plans[b]
            if not after.get(b):
                # The MoE tail's polls reload published rows each pass (``moe.tail``):
                # experts released at routing entry could start the next tail, whose
                # entry re-arm would hit rows a lane of this tail still polls (a hang).
                raise RuntimeError(
                    f"MoE batch {b}: an entry release of the experts is unsafe with "
                    "the tail's reloading polls; release them after the front GEMM"
                )
            early = "front"
            opts = self.route_opts[b]
            route = self.carved(
                b, compile_route(*self.route_keys[b], early=early, **opts)
            )
            self.plans[b] = (*plan[:2], route, *plan[3:])

    def carved(self, batch, route):
        """``ROUTE_CARVEOUT_WIDE`` batches: the routing kernel with carve-out 100."""
        if batch not in self.route_carveout:
            return route
        return prepare_shared_carveout(route, self.device.index, 100)

    def merged_batch(self, batch):
        return batch in self.merged

    def planned(self, batch):
        return bool(self.plans[batch][5])

    def shared_expert(self, x, tail):
        """The shared expert's rows into the tail's RS mailboxes (``tail.push``); the
        fused kernel peer-stores them itself and leaves the returned rows unwritten.
        Returns (rows, the split's down launch to run beside the experts, or None)."""
        views, _, _, situ, args, _, code = self.plans[x.shape[0]]
        s = stream(self.device)
        if code is not None:
            maps, counters = self.fused
            mailbox = (tail.peers, *tail.ring, Int32(CAP))
            front = (to_cute(x), maps, args[7], counters, *mailbox, *self.scalars[1:])
            if x.shape[0] not in self.split:
                code(*front, s)
                return views[5], None
            gate, down = code
            if x.shape[0] not in self.merged:  # merged: h comes from ``route``
                gate(*front, s)
            return views[5], lambda: down(args[7], maps, *mailbox, stream(self.device))
        if self.gate_situ is not None and x.shape[0] in self.gate_situ.kernels:
            self.gate_situ.run(x, views[4])  # h straight from the gate/up GEMM
        else:
            self.gate_up.run(x, views[1])
            situ(*args, s)
        shared = self.down.run(views[4], views[5])
        tail.push(shared)
        return shared, None
