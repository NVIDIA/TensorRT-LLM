# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""The K3.1 flat chain's fused KDA head: MR8's ``KdaHeadVerify`` at FP32 state with bounded waits.

Derived from MR8 ``cutedsl/kda/head.py`` (``KdaHead.poll_fa``, ``KdaHead.rearm``,
``PreparedKdaHead.prepare_caches``) and ``cutedsl/kda/head_verify.py``
(``compile_head_verify``); see ``kernels/SOURCE.json``. Change: the two F_A waits spin with a
time bound (``kernels.spin_trap``: past it the thread writes a record into mapped pinned host
memory and traps): ``poll_fa``'s ``while state == 0`` (site 211: state, target, token) and
``rearm``'s ``while seen < target`` (site 212: seen, target). Every other instruction, and so
every output and checkpoint bit, is MR8's.

F_A / beta come from MR8's ``SignalGemm`` on the step's side stream, in one of two launch
orders (``PreparedKdaHeadFp32.order``, ``features.K31Features.kda_head_order``):

- "event": ``SignalGemm`` is launched with a programmatic event that triggers once every one of
  its CTAs has started (``SignalGemmEvent``), and the main stream waits on that event before
  the head: the head is a programmatic dependent of both the AttnRes before it and
  ``SignalGemm``, so it launches only when every ``SignalGemm`` CTA is resident (its F_A poll
  can no longer hold the SMs that a not-yet-placed ``SignalGemm`` CTA needs) and its PDL wait
  also covers ``SignalGemm``. Its pre-wait weight stream still overlaps ``SignalGemm``. The
  head starts about 3.5 us later than in MR8's order on Rubin.
- "sibling": MR8's ``PreparedKdaHead.run_verify`` order: ``SignalGemm`` (MR8's, no event) on the
  side stream and the head on the main stream, both programmatic dependents of the AttnRes
  only, no edge between them. The head may be resident and spin on F_A while ``SignalGemm`` is
  not yet placed. That cannot wedge while this margin condition holds: after the head's six
  16-CTA clusters (226,304 B each, one CTA per SM) the device still has room for
  ``SignalGemm``'s three 8-CTA clusters of whole SMs (232,448 B), and nothing resident beside
  them waits on the head. On Rubin (212 SMs; 8 head clusters and 22 ``SignalGemm`` clusters fit
  an empty device, a head cluster takes at most two of the latter: 22 - 12 = 10 >= 3) the only
  kernels resident in the head's launch window are the previous layer's experts and MoE tail
  and the AttnRes ring (measured on every KDA layer of all 16 ranks), which are earlier in the
  PDL chain and retire on their own; the head's only waiter (the o-projection) launches after
  its F_A poll. So a late ``SignalGemm`` is a stall bounded by their drain, not a wedge, and the
  bounded F_A spin still traps a real one. On GB300 (152 SMs: 15 - 12 = 3, no margin, and one
  small-carve-out CTA keeps ``SignalGemm`` out) it can stall for as long as such a CTA lives;
  the target keeps "event" there (``FlatKda._prepare_head``). A change that puts a persistent
  kernel or small default-carve-out CTAs (e.g. L2 prefetch) in a KDA layer's head window, or
  grows the head or ``SignalGemm``, must redo this count.

The flat chain runs one request x q8 verify rows at FP32 state (``PreparedKdaHeadFp32`` with
``verify=((8, 8),)`` and no q = 1 batches); the vendored ``head.py`` / ``head_verify.py`` are
unchanged and still provide ``KdaSide``, ``polled_out_batches`` and the kernel body.
"""

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Uint32

from ..vendor.vllm_mr8 import module
from ..common import rubin_ull_spin_trap as spin_trap

_head = module("cutedsl.kda.head")
_verify = module("cutedsl.kda.head_verify")
_prim = module("cutedsl.common.primitives")
_host = module("cutedsl.common.host")
_at, _bits_float, _load_words = _prim.at, _prim.bits_float, _prim.load_words

#: ``spin_trap`` sites.
SITE_POLL_FA = 211
SITE_REARM = 212


class KdaHeadVerifyFp32(_verify.KdaHeadVerify):
    """``KdaHeadVerify`` whose F_A waits trap after ``spin_trap.BOUND_NS``; ``trap_record``:
    ``spin_trap.record_address()`` (set before compiling)."""

    trap_record = 0

    @cute.jit
    def poll_fa(self, fa_beta, counter, target, token, quarter, head):
        """MR8's ``KdaHead.poll_fa`` with a bounded ``while state == 0``: this lane's F_A
        words (8 x 2) and beta, polled until no BF16 half is the 0xFFFF sentinel (the
        counter as the fallback for a genuine 0xFFFF value)."""
        lane = cute.arch.lane_idx()
        words = cute.make_rmem_tensor(17, Uint32)
        beta_word = _at(fa_beta, token, 128 + (head - head % 2))
        state = Int32(2)
        if target > 0:
            state = Int32(0)
        spins, since = spin_trap.start()
        while state == 0:
            spins, since = spin_trap.tick(
                spins, since, SITE_POLL_FA, self.trap_record, state, target, token
            )
            bad = False
            for c in cutlass.range_constexpr(8):
                source = _at(fa_beta, token, 16 * c + 4 * quarter)
                pair = _load_words(source, count=2, word_type="u32", scope="gpu")
                words[2 * c], words[2 * c + 1] = pair
            words[16] = _load_words(beta_word, count=1, word_type="u32", scope="gpu")[0]
            for i in cutlass.range_constexpr(17):
                low, high = words[i] & Uint32(0xFFFF), words[i] >> Uint32(16)
                bad |= (low == Uint32(0xFFFF)) | (high == Uint32(0xFFFF))
            if cute.arch.vote_all_sync(not bad):
                state = Int32(1)
            else:
                done = Int32(0)
                if lane == 0:
                    done = Int32(Int32(_head._acquire_load(counter)) >= target)
                if cute.arch.shuffle_sync(done, 0) != 0:
                    state = Int32(2)
        if state == 2:
            for c in cutlass.range_constexpr(8):
                source = _at(fa_beta, token, 16 * c + 4 * quarter)
                words[2 * c], words[2 * c + 1] = _head.load_activation_bf16x4(source)
            words[16] = _load_words(beta_word, count=1, word_type="u32", mutable=True)[0]
        ar = [(words[2 * c], words[2 * c + 1]) for c in range(8)]
        half = (words[16] >> Uint32(16 * (head % 2))) & Uint32(0xFFFF)
        return ar, BFloat16(_bits_float(half << Uint32(16)))

    @cute.jit
    def rearm(self, counter, target, lane, fa_beta=None):
        """MR8's ``KdaHead.rearm`` with a bounded ``while seen < target``: the last of the
        head CTAs past the F_A counter waits for the counter to complete, re-arms the
        pair and re-fills F_A's rows with the sentinel."""
        last = Int32(0)
        if (target > 0) & (lane == 0):
            readers = Int32(_head.CLUSTER * self.heads)
            if Int32(_head._fetch_add(counter + 4)) == readers - 1:
                last = Int32(1)
                if cutlass.const_expr(self.fa_poll):
                    # readers may have seen F_A's rows before its counter adds landed
                    seen = Int32(_head._acquire_load(counter))
                    spins, since = spin_trap.start()
                    while seen < target:
                        spins, since = spin_trap.tick(
                            spins, since, SITE_REARM, self.trap_record, seen, target
                        )
                        seen = Int32(_head._acquire_load(counter))
                _head._store_zero(counter)
                _head._store_zero(counter + 4)
        if cutlass.const_expr(self.fa_poll):
            self.refill_fa(fa_beta, cute.arch.shuffle_sync(last, 0), lane)


class SignalGemmEvent(_head.SignalGemm):
    """MR8's ``SignalGemm`` (kernel unchanged) launched with ``event`` as its programmatic
    event, triggered at block start: the event fires once every CTA has started."""

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        a_map: cutlass.Int64,
        counter: cutlass.Int64,
        stream: _cuda.CUstream,
        event: _cuda.CUevent,
    ):
        self.kernel(a, b, c, a_map, counter).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=True,
            programmatic_event=event,
            programmatic_event_trigger_at_block_start=1,
        )


@cache
def compile_signal_event(tactic, arch):
    """MR8's ``compile_signal`` for ``SignalGemmEvent`` (the event is a call argument)."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def fake(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = SignalGemmEvent(tactic=_head.SplitKTactic(*tactic))
    args = (fake(1), fake(1), fake(0), cutlass.Int64(0), cutlass.Int64(0))
    return cute.experimental.compile(
        kernel, *args, make_fake_stream(), _cuda.CUevent(0), options=f"--gpu-arch={arch}"
    )


@cache
def compile_head_verify_fp32(heads, has_bias, specs, arch, tile, hidden, polled, record):
    """MR8's ``compile_head_verify`` for ``KdaHeadVerifyFp32`` (FP32 state); ``record``: the
    ``spin_trap`` record's device address."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    tensors = [_host.fake(shape, stride, dtype) for shape, stride, dtype in specs]
    scalars = (Float32(-5), Float32(128**-0.5), Float32(1e-6))
    kernel = KdaHeadVerifyFp32(
        heads,
        stages=_verify.VERIFY_STAGES,
        cta_k=tile[1],
        tmem_tiles=_verify.VERIFY_TMEM_TILES,
        early_loads=_verify.VERIFY_EARLY_LOADS,
        hidden=hidden,
        bf16_state=False,
        has_bias=has_bias,
    )
    kernel.polled_out = polled
    kernel.trap_record = record
    args = (matrix(1), matrix(1), cutlass.Int64(0), *tensors, *scalars)
    args += (cutlass.Int64(0), Int32(0), cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


#: The head / ``SignalGemm`` launch orders (module docstring).
ORDERS = ("event", "sibling")


class PreparedKdaHeadFp32(_head.PreparedKdaHead):
    """MR8's ``PreparedKdaHead`` for one request x q verify rows (``verify`` (q, q) shapes,
    no q = 1 ``batches``) at FP32 state, on ``KdaHeadVerifyFp32``; ``run_verify`` is MR8's with,
    in ``order`` "event", ``SignalGemmEvent`` and the head behind its programmatic event, in
    "sibling" MR8's own order (module docstring: when that is safe)."""

    def __init__(self, *args, order: str = "event", **kwargs):
        if order not in ORDERS:
            raise ValueError(f"PreparedKdaHeadFp32: order {order!r} is not one of {ORDERS}")
        super().__init__(*args, **kwargs)
        self.order = order

    def _signal(self, tactic, arch):
        """The compiled ``SignalGemm`` of ``tactic`` for this head's order."""
        if self.order == "sibling":
            return _head.compile_signal(tactic, arch)
        return compile_signal_event(tactic, arch)

    def prepare_caches(self, caches):
        if self.batches or any(b != q for b, q in self.verify):
            raise ValueError("PreparedKdaHeadFp32 runs one request x q verify rows only")
        if caches[1].dtype != torch.float32:
            raise TypeError("PreparedKdaHeadFp32 needs the FP32 KDA state")
        if self.tile is None:
            raise ValueError("PreparedKdaHeadFp32 needs KdaHeadWide's tile")
        arch = _host.architecture(self.device)
        record = spin_trap.record_address()
        meta = dict(device="meta")
        with torch.accelerator.device_index(self.device.index):
            self.fa_code = self._signal(_head.FA_TACTIC, arch)
            self.fa_event = None
            if self.order == "event":
                # SignalGemmEvent's programmatic event (timing off, created by a first record)
                self.fa_event = torch.cuda.Event(enable_timing=False)
                self.fa_event.record(torch.cuda.current_stream(self.device))
            for b, q in self.verify:
                if b in _head.FA_TACTIC_ROWS:
                    self._signal(_head.FA_TACTIC_ROWS[b], arch)
                slots = torch.empty((1, q), dtype=torch.int32, **meta)
                out = torch.empty(b, self.core.dim, dtype=torch.bfloat16, **meta)
                accepted = torch.empty((1,), dtype=torch.int32, **meta)
                tensors = (*self._tensors(caches, slots, out), accepted)
                specs = tuple(map(_host.layout, tensors))
                self.verify_kernels[b, q] = compile_head_verify_fp32(
                    self.core.heads,
                    self.core.has_bias,
                    specs,
                    arch,
                    self.tile,
                    self.weight.shape[1],
                    b in self.polled,
                    record,
                )
        return self

    def run_verify(self, x, out, *, caches, slots, accepted, rearm=None):
        """MR8's ``PreparedKdaHead.run_verify`` (one request x q verify rows); in ``order``
        "event" with ``SignalGemmEvent`` on the side stream and the head behind its programmatic
        event, in "sibling" in MR8's order."""
        b, q = x.shape[0], slots.shape[1]
        if b != q:
            raise ValueError("PreparedKdaHeadFp32 runs one request x q verify rows only")
        fa = self.fa_out[:b]
        counter = cutlass.Int64(self.counter.data_ptr())
        fa_tactic = _head.FA_TACTIC_ROWS.get(b, _head.FA_TACTIC)
        fa_code = self.fa_code
        if fa_tactic != _head.FA_TACTIC:
            fa_code = self._signal(fa_tactic, _host.architecture(self.device))
        operands = (
            _head.matrix_view(self.fa_weight),
            _head.matrix_view(x),
            _head.matrix_view(fa, output=True),
            cutlass.Int64(self.fa_map.data_ptr()),
            counter,
        )
        with torch.cuda.stream(self.side.fork()):
            if self.order == "sibling":
                fa_code(*operands, _host.stream(self.device))
            else:
                event = _cuda.CUevent(int(self.fa_event.cuda_event))
                fa_code(*operands, _host.stream(self.device), event)
        if self.order == "event":
            torch.cuda.current_stream(self.device).wait_event(self.fa_event)
        if slots.stride() != (slots.shape[1], 1):
            slots = slots.contiguous()
        tensors = (*self._tensors(caches, slots[:1], out), accepted[:1])
        views = [_host.to_cute(t, 16) for t in tensors]
        if (rearm is None) != (b not in self.polled):
            raise ValueError("KdaHeadVerify: a polled batch needs the buffer to re-arm")
        address = rearm.data_ptr() if rearm is not None else 0
        target = self.fa_tiles * -(-b // fa_tactic[1])  # SignalGemm's tile owners
        self.verify_kernels[b, q](
            _head.matrix_view(self.weight),
            _head.matrix_view(x),
            cutlass.Int64(self.map.data_ptr()),
            *views,
            *self.core.scalars,
            counter,
            Int32(target),
            cutlass.Int64(address),
            _host.stream(self.device),
        )
        return out
