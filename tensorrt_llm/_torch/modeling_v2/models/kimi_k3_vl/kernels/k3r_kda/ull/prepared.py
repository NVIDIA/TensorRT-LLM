# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""Prepared KDA core. One packed projection supplies Q/K/V, the output gate, F_A and
beta; per-batch policies select the gate schedule and the projection row pitch. Verify
batches the caller lists in ``prepared`` also get ``KdaVerifyPrepared``, which reads
the operands an in-projection epilogue prepared (``run(..., prepared=...)``) and polls
the projection's F_A rows: the caller arms them (every BF16 element ``FA_SENT``) before
the first run and gives that projection buffer to this path only; each run re-arms
them."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ....ops.common.k3_ull_cute_host import fake, layout, prepare_shared_carveout, stream, to_cute
from ...k3r_common.ull.policy import batch_table
from .kernel import KdaSplit
from .verify import KdaVerifyCols, KdaVerifyPrepared, KdaVerifyRows

# KdaSplit schedule "early.lanes.occupancy.cluster.evict" per batch by heads (TP8 12,
# TP16 6), from GB300 sweeps: early predecessor-independent loads (MMA gate modes only;
# 2 also stages the state rows in shared memory, 3 also the F_B operands), lanes per
# state row, minimum CTAs per SM, CTAs per head (0: defaults) and L2 evict-first state
# traffic.
SCHEDULE = {12: "16-32:3.16.3.2.1,64:3.16.4.2.1", 6: "16:1.0.0,32-64:3.16.3.2"}
# Per compute capability: overrides on top of SCHEDULE. Rubin (10.7, 212 SMs) TP16
# B1-8, from Rubin sweeps of the in-projection -> KdaSplit chain: staged state rows
# (evict-first), lanes 16, three CTAs per SM, eight-CTA clusters up to B4, then four.
DEVICE_SCHEDULE = {(10, 7): {6: "1-4:2.16.3.8.1,5-8:2.16.3.4.1"}}
# BF16 recurrent state (--mamba-ssm-cache-dtype bfloat16): overrides on top of those,
# from the same Rubin chain sweeps: four-CTA clusters win at B2-4. B5-8 keep the FP32
# geometry: behind an AttnRes-like predecessor and before the o-projection (the serving
# chain), two CTAs per head with staged F_B operands (3.16.3.2.1, fastest without the
# o-projection) are 0.48 us per layer slower at B8 and 0.67 at B7.
BF16_DEVICE_SCHEDULE = {(10, 7): {6: "2-4:2.16.3.4.1"}}
# Verify (q > 1) schedules by request count B, from GB300
# sweeps over q 4-8: no staging; eight-CTA clusters, then fewer as B grows.
SPEC_SCHEDULE = {
    12: "1:1.8.2.8,2-4:1.8.3.8,5-12:1.8.3.4,13-16:1.8.3.2",
    6: "1:1.8.2.8,2:1.8.0.8,3-12:1.8.3.8,13-16:1.8.3.4",
}
# Per compute capability: verify schedules on top of SPEC_SCHEDULE. Rubin TP16 above 16
# requests (the GB300 table stops at 16, leaving the default geometry: 16 lanes, four
# CTAs per head, no occupancy bound), from a 1-GPU sweep of KdaSplit verify alone (69
# layers, cold state slots, BF16 state), us per layer at B32 / B64: q8 34.4 / 64.4 ->
# 18.1 / 27.9, q4 15.6 / 25.9 -> 12.1 / 18.6 (lanes 8, three CTAs per SM, four per
# head); same arithmetic (8 or 16 lanes per row reduce in the same tree). B4: two CTAs
# per SM instead of three (GB300's 1.8.3.8 packs the 192 CTAs onto ~72 SMs and the
# token loop takes ~3 us); 1-GPU in-projection -> KdaSplit -> o-projection-footprint
# chain, the o-projection stand-in's PDL wait returned 11.75 -> 10.50 us after the
# AttnRes stand-in (us/call over the chain 14.34 -> 12.52). B4 also marks its
# checkpoint stores L2 evict-first (the same values; that chain, BF16 state: 10.46 ->
# 9.98 us, 13.00 -> 12.52 us/call; B2 gains nothing from it).
DEVICE_SPEC_SCHEDULE = {(10, 7): {6: "4:1.16.2.8.1,17-64:1.8.3.4"}}
# Compute capabilities whose verify steps activate only the gates a CTA uses and each
# beta once per CTA (``KdaSplit(lean=True)``, bitwise identical). Rubin TP16 B8 x q8,
# 1-GPU in-projection -> KdaSplit -> o-projection-footprint chain: 14.76 -> 14.66 us.
LEAN_VERIFY = {(10, 7)}
# Per compute capability: verify request counts whose KdaSplit normalizes q / k at
# operand load (``KdaSplit(load_norm=True)``, bitwise identical) instead of in a pass
# and a CTA barrier of its own. Rubin TP16 q8, the same chain, the o-projection
# stand-in's PDL wait returned (us after the AttnRes stand-in): B2 9.82 -> 9.62, B4
# 10.46 -> 10.35.
VERIFY_LOAD_NORM: dict[tuple[int, int], str] = {(10, 7): "2-4:1"}
# Per compute capability: state rows per recurrence thread of the prepared verify
# (``KdaVerifyRows``, bitwise) by request count B, "rows[.spare warps]"; 1
# (absent) keeps one row per thread. Rubin TP16 B8 x q8, in-projection -> verify ->
# o-projection-footprint chain: two rows per thread plus a spare warp (three CTAs per
# SM, as with one row; 168 registers would let four pack) 14.25 -> 13.93 us, token loop
# 7640 -> 6340 cycles. Four rows (247 registers) pack four per SM and lose; capping by
# an explicit shared-memory size delays the CTAs' residency by ~4 us.
VERIFY_ROWS: dict[tuple[int, int], str] = {(10, 7): "8:2.1"}
# Per compute capability: the prepared verify's recurrence transposed
# (``KdaVerifyCols``, bitwise) by request count B, the value its spare warps (86
# registers: one caps CTAs at three per SM); takes precedence over VERIFY_ROWS. Rubin
# TP16 B8 x q8, the same chain: 13.50 -> 12.49 us per call vs VERIFY_ROWS (one LDS.128
# per operand per lane-token).
VERIFY_COLS: dict[tuple[int, int], str] = {(10, 7): "8:1"}
# Per verify length overrides for the split-K schedule.
DEVICE_SPEC_SCHEDULE_Q: dict = {(10, 7): {4: {6: "6:1.8.2.8,14:1.8.2.4"}}}


@cache
def executable(specs, heads, gate_mode, has_bias, device, tokens=1, prepared=False):
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    batch = specs[0][0][0]
    table = SCHEDULE if tokens == 1 else SPEC_SCHEDULE
    spec, bf16_state = table.get(heads, ""), specs[9][2] == torch.bfloat16
    capability = torch.cuda.get_device_capability(device)
    if tokens == 1:
        tables = (DEVICE_SCHEDULE, BF16_DEVICE_SCHEDULE)[: 1 + bf16_state]
        overrides = (t.get(capability, {}).get(heads, "") for t in tables)
        spec = ",".join(filter(None, (spec, *overrides)))
    else:
        override = DEVICE_SPEC_SCHEDULE.get(capability, {}).get(heads, "")
        by_q = DEVICE_SPEC_SCHEDULE_Q.get(capability, {}).get(tokens, {})
        spec = ",".join(filter(None, (spec, override, by_q.get(heads, ""))))
    schedule = batch_table(spec)
    schedule = schedule.get(batch // tokens, (0, 0, 0)) if gate_mode != 2 else (0,) * 3
    if schedule[0] >= 2 and specs[9][1][-2:] != (128, 1):  # staging needs whole rows
        schedule = (1, *schedule[1:4])  # no evict hint without staging
    requests = batch // tokens
    options = (heads, requests, gate_mode, has_bias, schedule, tokens, bf16_state)
    kernel: KdaSplit
    rows = batch_table(VERIFY_ROWS.get(capability, "")).get(requests, 1)
    rows = rows if isinstance(rows, tuple) else (rows,)
    cols = batch_table(VERIFY_COLS.get(capability, "")).get(requests)
    if prepared and cols is not None:  # the transposed recurrence
        kernel = KdaVerifyCols(*options, cols)
    elif (
        prepared and rows[0] > 1
    ):  # several state rows per thread, operands loaded once
        kernel = KdaVerifyRows(*options, *rows)
    elif prepared:  # the gates and betas arrive activated (the in-projection epilogue)
        kernel = KdaVerifyPrepared(*options)
    else:
        lean = tokens > 1 and capability in LEAN_VERIFY
        norm = batch_table(VERIFY_LOAD_NORM.get(capability, "")).get(requests)
        kernel = KdaSplit(*options, lean, bool(norm))
    scalars = (cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        compiled = cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )
    carveout = getattr(kernel, "shared_carveout", None)  # several staged CTAs per SM
    if carveout is not None:
        compiled = prepare_shared_carveout(compiled, device, carveout)
    return compiled


def decode_conv_weight(weight):
    """The [3, width, D] (Q, K, V) decode layout of the [3 * D, 1, width] conv
    weight."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


class PreparedKda:
    def __init__(self, attention, policies, conv_dim_first, prepared=()):
        """``prepared``: verify policy keys (requests, tokens) that also run on
        prepared operands."""
        bias, dt = attention.conv1d.bias, attention.dt_bias
        self.heads = attention.A_log.numel()
        self.dim, self.has_bias = self.heads * 128, bias is not None
        conv_w = decode_conv_weight(attention.conv1d.weight)
        bias, fb = dt if bias is None else bias, attention.f_b_proj.weight
        norm = attention.o_norm.weight.detach().float()  # BF16 slows the fused kernel
        self.weights = (conv_w, bias, fb, attention.A_log, dt, norm)
        self.device, self.policies = dt.device, policies
        scalars = (attention.gate_lower_bound, 128**-0.5, attention.o_norm.eps)
        self.scalars = tuple(map(cutlass.Float32, scalars))
        self.conv_dim_first, self.kernels = conv_dim_first, {}
        self.prepared, self.prepared_kernels = frozenset(prepared), {}
        # KdaVerifyPrepared's F_A reader counts per request (its last reader re-arms)
        requests = max((r for r, _ in self.prepared), default=0)
        self.readers = torch.zeros(requests, dtype=torch.int32, device=self.device)

    def _operands(self, caches, projected, slots, out, accepted=None):
        """Kernel tensors in launch order: 13 before the scalars, then F_B (and the
        verify step's accepted counts)."""
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        ops = (p(0, 3 * d), w, bias, conv, p(4 * d, 128), p(4 * d + 128, h), a_log, dt)
        tail = (fb,) if accepted is None else (fb, accepted)
        return ops + (slots, state, out, p(3 * d, d), norm) + tail

    def prepare_caches(self, caches):
        """Policy keys: a batch, or (requests, tokens) for verify steps; prepared
        operands: [requests * tokens, 4 * dim + 8] FP32."""
        for key, (pitch, mode) in self.policies.items():
            requests, tokens = key if isinstance(key, tuple) else (key, 1)
            batch, meta = requests * tokens, dict(device="meta")
            projected = torch.empty((batch, pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty(batch, self.dim, dtype=torch.bfloat16, **meta)
            accepted = torch.empty(requests, dtype=torch.int32, **meta)
            shape = (requests, tokens) if tokens > 1 else (batch,)
            slots = torch.empty(shape, dtype=torch.int32, **meta)
            accepted = accepted if tokens > 1 else None
            operands = self._operands(caches, projected, slots, out, accepted)
            specs = tuple(map(layout, operands))
            options = (self.heads, mode, self.has_bias, self.device.index, tokens)
            self.kernels[key] = executable(specs, *options)
            if key in self.prepared:
                width = 4 * self.dim + 8
                prepared = torch.empty((batch, width), dtype=torch.float32, **meta)
                specs = (layout(prepared), *specs[1:], layout(self.readers[:requests]))
                self.prepared_kernels[key] = executable(specs, *options, True)
        return self

    def run(self, projected, out, *, caches, slots, accepted=None, prepared=None):
        """``accepted`` [>= B] (verify steps): ``slots`` is [>= B, q] contiguous;
        ``prepared``: the verify operands of ``KdaVerifyPrepared`` (its key must be in
        ``self.prepared``; ``projected``'s F_A rows armed, see the module docstring)."""
        batch = projected.shape[0]
        key, rows = batch, batch
        if accepted is not None:
            rows = batch // slots.shape[1]
            key, accepted = (rows, slots.shape[1]), accepted[:rows]
            if slots.stride() != (slots.shape[1], 1):  # e.g. a block-table slice
                slots = slots.contiguous()
        tensors = self._operands(caches, projected, slots[:rows], out, accepted)
        kernel = self.kernels[key]
        if prepared is not None:
            tensors = (prepared, *tensors[1:], self.readers[:rows])
            kernel = self.prepared_kernels[key]
        views = tuple(to_cute(t, 16) for t in tensors)
        kernel(*views[:13], *self.scalars, *views[13:], stream(self.device))
        return out
