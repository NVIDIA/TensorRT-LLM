# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""The K3.1 flat chain's fused KDA head for two requests x q8 verify rows (16 rows): MR8's
``KdaHeadMulti`` at FP32 state with bounded waits.

Derived from MR8 ``cutedsl/kda/head_multi.py`` (``KdaHeadMulti``,
``compile_head_multi``) and ``cutedsl/kda/head.py`` (``PreparedKdaHead.prepare_caches`` /
``run_verify``, their several-requests branch); see ``kernels/SOURCE.json``. Changes:

- the two F_A waits are ``rubin_ull_kda_head_fp32.KdaHeadVerifyFp32``'s bounded ones
  (``spin_trap`` sites 211 / 212): ``KdaHeadMultiFp32`` takes ``poll_fa`` / ``rearm`` from it;
  every other instruction is MR8's ``KdaHeadMulti``;
- FP32 state only (``bf16_state=False``), output never polled (``POLLED_OUT`` is an 8-row policy);
- ``PreparedKdaHeadMultiFp32`` runs only the 2 x q8 shape, with ``SignalGemm`` in the layer's
  head order (``PreparedKdaHeadFp32.order``: "event" or MR8's "sibling").

Numerics against the 16-row split pair (``SplitKTmemGemm64`` in-projection -> ``KdaSplit``):
the Q / K / V / gate rows are summed over K in four split-K partials (rank order) instead of
six, and F_A / beta come from ``SignalGemm``'s split-K 8 (``FA_TACTIC_ROWS[16]``) instead of the
in-projection's split 6: a stated FP32 accumulation reorder of the projected inputs, then one
BF16 rounding each, as before. Given the same projected inputs the recurrence, outputs,
checkpoints and conv windows are ``KdaSplit``'s B1 verify schedule per request (MR8
``head_multi.py``'s docstring).

Residency in the "sibling" order (``rubin_ull_kda_head_fp32``'s module docstring): the head is
the same six 16-CTA clusters (one CTA per SM; ``MULTI_TILE``'s 7-stage ring keeps its shared
memory within ``BESIDE_SMEM``, as the 8-row head's) and ``SignalGemm`` the same 24 CTAs
(``FA_TACTIC_ROWS[16]``: one N tile for the 16 rows), so the margin count is unchanged. At 16 rows the next-attention L2 prefetch is not
skipped before a KDA layer (``BEFORE_HEAD`` lists 8 rows only); its CTAs do not wait on the head
and retire on their own, so a late ``SignalGemm`` is again a stall bounded by their drain, not a
wedge, and the bounded F_A spin still traps a real one. GB300 keeps "event".
"""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from ..common import rubin_ull_spin_trap as spin_trap
from ..vendor.vllm_mr8 import module
from . import rubin_ull_kda_head_fp32 as _fp32

_head, _host = _fp32._head, _fp32._host
_multi = module("cutedsl.kda.head_multi")

#: The one shape this head runs: two requests x q8 (16 verify rows).
REQUESTS, TOKENS = 2, 8
ROWS = REQUESTS * TOKENS


class KdaHeadMultiFp32(_multi.KdaHeadMulti, _fp32.KdaHeadVerifyFp32):
    """MR8's ``KdaHeadMulti`` whose F_A waits trap after ``spin_trap.BOUND_NS``: ``poll_fa`` /
    ``rearm`` resolve to ``KdaHeadVerifyFp32``'s (``KdaHeadMulti`` defines neither); every
    other method is ``KdaHeadMulti``'s. ``trap_record``: ``spin_trap.record_address()``."""


@cache
def compile_head_multi_fp32(heads, has_bias, specs, arch, hidden, record):
    """MR8's ``compile_head_multi`` for ``KdaHeadMultiFp32`` at ``REQUESTS`` x ``TOKENS``, FP32
    state, unpolled output; ``record``: the ``spin_trap`` record's device address."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    tensors = [_host.fake(shape, stride, dtype) for shape, stride, dtype in specs]
    scalars = (Float32(-5), Float32(128**-0.5), Float32(1e-6))
    stages, tmem_tiles, early = _multi.MULTI_TILE[REQUESTS, TOKENS]
    kernel = KdaHeadMultiFp32(
        heads,
        requests=REQUESTS,
        tokens=TOKENS,
        stages=stages,
        tmem_tiles=tmem_tiles,
        early_loads=early,
        hidden=hidden,
        bf16_state=False,
        has_bias=has_bias,
    )
    kernel.polled_out = False
    kernel.trap_record = record
    args = (matrix(1), matrix(1), cutlass.Int64(0), *tensors, *scalars)
    args += (cutlass.Int64(0), Int32(0), cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class PreparedKdaHeadMultiFp32(_fp32.PreparedKdaHeadFp32):
    """MR8's ``PreparedKdaHead`` for ``REQUESTS`` x ``TOKENS`` verify rows (``verify`` =
    ((``ROWS``, ``TOKENS``),), no q = 1 ``batches``, no polled output) at FP32 state on
    ``KdaHeadMultiFp32``. Its F_A / beta buffer ([``ROWS``, 144]) and [ready, done] counter are
    its own (not the layer's 8-row head's), re-armed by their last reader, so it runs at most once
    per step. ``order`` as ``PreparedKdaHeadFp32``."""

    def prepare_caches(self, caches):
        if self.batches or self.verify != ((ROWS, TOKENS),):
            raise ValueError(f"PreparedKdaHeadMultiFp32 runs {REQUESTS} requests x q{TOKENS} only")
        if self.polled:
            raise ValueError("PreparedKdaHeadMultiFp32 has no polled output")
        if caches[1].dtype != torch.float32:
            raise TypeError("PreparedKdaHeadMultiFp32 needs the FP32 KDA state")
        if self.tile is None:
            raise ValueError("PreparedKdaHeadMultiFp32 needs KdaHeadWide's tile (F_A sentinel)")
        if self.fa_out.shape[0] < ROWS:
            raise ValueError(f"PreparedKdaHeadMultiFp32 needs a [>= {ROWS}, F_A] buffer")
        arch = _host.architecture(self.device)
        record = spin_trap.record_address()
        meta = dict(device="meta")
        with torch.accelerator.device_index(self.device.index):
            self.fa_code = self._signal(_head.FA_TACTIC_ROWS[ROWS], arch)
            self.fa_event = None
            if self.order == "event":
                # SignalGemmEvent's programmatic event (timing off, created by a first record)
                self.fa_event = torch.cuda.Event(enable_timing=False)
                self.fa_event.record(torch.cuda.current_stream(self.device))
            slots = torch.empty((REQUESTS, TOKENS), dtype=torch.int32, **meta)
            out = torch.empty(ROWS, self.core.dim, dtype=torch.bfloat16, **meta)
            accepted = torch.empty((REQUESTS,), dtype=torch.int32, **meta)
            tensors = (*self._tensors(caches, slots, out), accepted)
            specs = tuple(map(_host.layout, tensors))
            self.verify_kernels[ROWS, TOKENS] = compile_head_multi_fp32(
                self.core.heads, self.core.has_bias, specs, arch, self.weight.shape[1], record
            )
        return self

    def run_verify(self, x, out, *, caches, slots, accepted, rearm=None):
        """MR8's ``PreparedKdaHead.run_verify`` for ``REQUESTS`` x ``TOKENS`` rows (its
        ``KdaHeadMulti`` branch), ``SignalGemm`` in ``order`` as ``PreparedKdaHeadFp32``."""
        b, q = x.shape[0], slots.shape[1]
        if (b, q) != (ROWS, TOKENS):
            raise ValueError(f"PreparedKdaHeadMultiFp32 runs {REQUESTS} requests x q{TOKENS} only")
        if rearm is not None:
            raise ValueError("PreparedKdaHeadMultiFp32 has no polled output")
        fa = self.fa_out[:b]
        counter = cutlass.Int64(self.counter.data_ptr())
        fa_tactic = _head.FA_TACTIC_ROWS[b]
        operands = (
            _head.matrix_view(self.fa_weight),
            _head.matrix_view(x),
            _head.matrix_view(fa, output=True),
            cutlass.Int64(self.fa_map.data_ptr()),
            counter,
        )
        with torch.cuda.stream(self.side.fork()):
            if self.order == "sibling":
                self.fa_code(*operands, _host.stream(self.device))
            else:
                event = _fp32._cuda.CUevent(int(self.fa_event.cuda_event))
                self.fa_code(*operands, _host.stream(self.device), event)
        if self.order == "event":
            torch.cuda.current_stream(self.device).wait_event(self.fa_event)
        if slots.stride() != (slots.shape[1], 1):
            slots = slots.contiguous()
        tensors = (*self._tensors(caches, slots[:REQUESTS], out), accepted[:REQUESTS])
        views = [_host.to_cute(t, 16) for t in tensors]
        target = self.fa_tiles * -(-b // fa_tactic[1])  # SignalGemm's tile owners
        self.verify_kernels[b, q](
            _head.matrix_view(self.weight),
            _head.matrix_view(x),
            cutlass.Int64(self.map.data_ptr()),
            *views,
            *self.core.scalars,
            counter,
            Int32(target),
            cutlass.Int64(0),
            _host.stream(self.device),
        )
        return out
