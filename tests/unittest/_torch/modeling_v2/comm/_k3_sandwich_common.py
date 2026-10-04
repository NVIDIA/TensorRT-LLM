# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the Kimi K3 sandwich matrices (``_k3_sandwich_oproj_op_matrix.py``,
``_k3_sandwich_tail_op_matrix.py``, ``_k3_sandwich_plain_op_matrix.py``): one class per op holding one call's
arguments on every rank and its native-torch reference, a driver for call sequences (chained the way the model chains
its residual streams; eager with a random rank late, or captured and replayed), and the state checks the three
matrices run the same way on a ``K3SandwichWorkspace``.

Started by file path like ``_lockstep`` (this tree is not a package). Importing it pulls in torch only; ``bind``
(called by the rank body) imports the catalog wrappers.

Payloads and references. Every rank draws every rank's inputs from one seed, so each rank holds the whole reference.
The projections' inputs are small multiples of powers of two (``_lockstep.exact_bf16``), so every partial product
sum is exact in fp32 in whatever order the kernel accumulates: the reference is the fp64 product rounded once to bf16
per rank (the kernel rounds its fp32 accumulator once), the ranks' partials added in fp32 and rounded to bf16, then
the prefix sum (or residual) added in fp32 and rounded -- the kernel's own steps, so ``updated`` of
``k3_sandwich_oproj`` and ``k3_sandwich_plain`` is compared bit for bit. ``k3_sandwich_tail`` scales its latent
accumulator by the latent row's RMS, an fp32 rsqrt that is not torch's, so a partial element can round to the
neighbouring bf16: its ``updated`` is compared within ``TOL_TAIL`` of an fp64 reference of the kernel's arithmetic.
``normed`` and the tail's tap (softmax, rsqrt) are compared within ``TOL`` of an fp32 reference. Every output is also
compared bitwise across the ranks.
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import _lockstep as ls
import torch

H = 7168
K_O = 768  # core / o_weight columns at TP16: 96 heads x 128 / 16
LATENT = 3584  # the reduced latent row
LAT_SLICE = 224  # a rank's latent columns at TP16
LAT_PAD = 256  # the latent slice zero-padded to whole k-tiles inside tail_weight
SLICES = LATENT // LAT_SLICE  # 16
ACT = 384  # the shared-expert activation at TP16
PLAIN_K = 384  # the drafter's o_proj slice at TP16
DOWN_K = 896  # the drafter's down projection slice at TP16; its gate_up output is [M, 2 x 896]
NUM_CTAS = 56  # the kernel's CTAs: one call counter each, in 64 flag words
FLAG_WORDS = 64
RMS_EPS = 1e-6
OUT_EPS = 1e-6
LAT_EPS = 1e-6
EPS = 1e-6
TOL = 2e-2  # normed and the tap: max |err| / max |ref|
TOL_TAIL = (
    8e-3  # the tail's updated: max |err| / max |ref|, about one bf16 ulp of its largest elements
)
GATES = (0.0, 32.0, 64.0)  # swiglu gate values on which silu is exact in fp32 (see PlainCall)
WEIGHT_SETS = 3
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8)
INTERLEAVE = "AABABBAAAB" * 2
REPLAYS = 8
WEIGHT_SEEDS = {"oproj": 900, "tail": 910, "plain": 920, "down": 930}

CAPTURE_REFUSED = "outside CUDA-graph capture"  # create()'s reason on a rank that is capturing
PEER_REFUSED = "another rank cannot"  # its reason on the other ranks

R = None  # this rank (a _lockstep.Rank), set by bind()
OPS: Dict[str, Callable] = {}
WEIGHTS: Dict[str, List[List[torch.Tensor]]] = {}
STATS = {"normed": 0.0, "tail_updated": 0.0, "tap": 0.0}
# The multicast allocations create() has reached on this rank, counted from bind() on.
ALLOCATIONS = [0]


def bind(rank: ls.Rank, weight_sets: Dict[str, int]) -> None:
    """Bind this rank and the three catalog wrappers, draw ``weight_sets[kind]`` weight sets of each kind ("oproj",
    "tail", "plain", "down"), every rank's slice, from fixed seeds, and count from here on the multicast allocations
    ``create`` reaches (``ALLOCATIONS``), so that a refused ``create`` can be shown to allocate nothing."""
    global R
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        k3_sandwich_oproj,
        k3_sandwich_plain,
        k3_sandwich_tail,
    )

    R = rank
    OPS["oproj"] = k3_sandwich_oproj.k3_sandwich_oproj
    OPS["tail"] = k3_sandwich_tail.k3_sandwich_tail
    OPS["plain"] = k3_sandwich_plain.k3_sandwich_plain
    for kind, sets in weight_sets.items():
        WEIGHTS[kind] = []
        for s in range(sets):
            g = _gen(WEIGHT_SEEDS[kind] + s)
            WEIGHTS[kind].append([_weight(kind, g) for _ in range(R.world)])
    _count_allocations()


def _count_allocations() -> None:
    """Wrap the multicast allocation ``K3SandwichWorkspace.create`` makes (``_make_mnnvl_mcast_buffer``, which it looks
    up at every call) with a counter; the allocation itself is unchanged."""
    from tensorrt_llm._torch.distributed import ops

    allocate = ops._make_mnnvl_mcast_buffer

    def counted(*args, **kwargs):
        ALLOCATIONS[0] += 1
        return allocate(*args, **kwargs)

    ops._make_mnnvl_mcast_buffer = counted


def report() -> None:
    if R.rank == 0:
        print(f"[rank 0] world {R.world}; max rel err: normed {STATS['normed']:.3e}, tail updated "
              f"{STATS['tail_updated']:.3e}, tap {STATS['tap']:.3e}", flush=True)  # fmt: skip


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _weight(kind: str, g: torch.Generator) -> torch.Tensor:
    """One rank's weight slice, entries in {-2, ..., 2} / 16; tail_weight = [latent up (224) | zeros (32) | shared
    down (384)]."""
    if kind == "tail":
        lat = ls.exact_bf16(g, (H, LAT_SLICE), -2, 3, 1 / 16)
        pad = torch.zeros(H, LAT_PAD - LAT_SLICE, dtype=torch.bfloat16, device="cuda")
        act = ls.exact_bf16(g, (H, ACT), -2, 3, 1 / 16)
        return torch.cat([lat, pad, act], dim=1).contiguous()
    k = {"oproj": K_O, "plain": PLAIN_K, "down": DOWN_K}[kind]
    return ls.exact_bf16(g, (H, k), -2, 3, 1 / 16)


def _running_sum(g: torch.Generator, tokens: int) -> torch.Tensor:
    """A drawn prefix sum or residual: {-32, ..., 32} / 16."""
    return ls.exact_bf16(g, (tokens, H), -32, 33, 1 / 16)


def _attn_res_inputs(g: torch.Generator, snapshots: int, tokens: int):
    """``block_residual`` [S, M, 7168] and the epilogue's three [7168] weights."""
    block = torch.randn(snapshots, tokens, H, generator=g, device="cuda").bfloat16()
    res_w = (torch.randn(H, generator=g, device="cuda") * 0.05).bfloat16()
    rms_w = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()
    out_w = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()
    return block, res_w, rms_w, out_w


def _gates(g: torch.Generator, shape) -> torch.Tensor:
    idx = torch.randint(0, len(GATES), tuple(shape), generator=g, device="cuda")
    return torch.tensor(GATES, device="cuda")[idx].bfloat16()


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def bf16_partial(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """bf16(x @ w^T) from the exact fp64 product: the kernel's one rounding of an fp32 accumulator that is exact for
    these payloads."""
    return (x.double() @ w.double().t()).float().bfloat16()


def reduce_ref(partials: Sequence[torch.Tensor], carry: Optional[torch.Tensor]) -> torch.Tensor:
    """bf16(carry + bf16(sum of the partials)): the partials added in fp32 in rank order (the kernel's order up to 8
    ranks; exact sums make any order the same), then the carry -- the prefix sum or the residual -- added in fp32;
    the rounded sum alone without a carry."""
    total = partials[0].float()
    for p in partials[1:]:
        total = total + p.float()
    updated = total.bfloat16()
    if carry is not None:
        updated = (carry.float() + updated.float()).bfloat16()
    return updated


def attn_res_mixture(updated, block, res_w, rms_w) -> torch.Tensor:
    """The attention-residual mixture of [block..., updated] before the output RMSNorm, in fp32 and rounded to bf16:
    the tensor ``_lockstep.residual_update_ref`` normalizes, and the tail's tap."""
    v = torch.cat([block, updated.unsqueeze(0)], dim=0).float()
    rs = (v.square().mean(dim=-1) + RMS_EPS).rsqrt()
    logits = (v * rs[..., None] * (rms_w.float() * res_w.float())).sum(dim=-1)
    probs = torch.softmax(logits, dim=0)
    return (probs[..., None] * v).sum(dim=0).bfloat16()


def _within(
    got: torch.Tensor, want: torch.Tensor, tol: float, key: str, where: str, what: str
) -> None:
    err = ls.rel_err(got, want)
    STATS[key] = max(STATS[key], err)
    assert err <= tol, f"{where}: {what} rel err {err:.3e} > {tol}"


class Call:
    """One call's arguments on every rank and its reference. ``carry`` is the running sum the call adds to: the prefix
    sum of oproj / tail (None: the sum alone) or the residual of plain. Subclasses define ``ref``, ``run`` and
    ``refill``."""

    tokens: int
    carry: Optional[torch.Tensor]

    def ref(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def run(self, ws) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def refill(self, fresh: "Call", carry: bool) -> None:
        """Copy ``fresh``'s inputs into this call's tensors in place (a captured call reads them there), the carry too
        when ``carry`` (a chain's first call; later calls read the previous call's output). Weights and scalar
        arguments stay: the graph holds them."""
        raise NotImplementedError

    def verify(self, got, where: str) -> None:
        normed, updated = got
        want_normed, want_updated = self.ref()
        assert torch.equal(updated, want_updated), (
            f"{where}: updated differs from the exact reference"
        )
        _within(normed, want_normed, TOL, "normed", where, "normed")
        assert R.same_on_ranks(normed, updated), f"{where}: ranks disagree"

    def wrong_fraction(self, got) -> float:
        """The fraction of ``updated``'s elements that differ from the reference."""
        return (got[1] != self.ref()[1]).float().mean().item()

    def wrong_margin(self, got) -> float:
        """``updated``'s largest error over the comparison's bound: any difference fails an exact comparison."""
        return float("inf") if self.wrong_fraction(got) > 0 else 0.0


class OprojCall(Call):
    """k3_sandwich_oproj: core {-2, ..., 2} / 8 [M, 768] per rank, o_weight from weight set ``weights``. ``prefix``:
    True (drawn), None, or a tensor."""

    def __init__(self, seed, tokens, snapshots, prefix=True, weights=0):
        g = _gen(seed)
        self.tokens = tokens
        self.cores = [ls.exact_bf16(g, (tokens, K_O), -2, 3, 1 / 8) for _ in range(R.world)]
        self.carry = _running_sum(g, tokens) if prefix is True else prefix
        self.weights = WEIGHTS["oproj"][weights]
        self.block, self.res_w, self.rms_w, self.out_w = _attn_res_inputs(g, snapshots, tokens)

    def ref(self):
        updated = reduce_ref(
            [bf16_partial(c, w) for c, w in zip(self.cores, self.weights)], self.carry
        )
        normed = ls.residual_update_ref(
            updated, self.block, self.res_w, self.rms_w, RMS_EPS, self.out_w, OUT_EPS
        )
        return normed, updated

    def run(self, ws):
        r = R.rank
        return OPS["oproj"](self.cores[r], self.weights[r], self.carry, self.block, self.res_w, self.rms_w,
                            self.out_w, RMS_EPS, OUT_EPS, ws)  # fmt: skip

    def refill(self, fresh, carry):
        for mine, new in zip(self.cores, fresh.cores):
            mine.copy_(new)
        self.block.copy_(fresh.block)
        if carry:
            self.carry.copy_(fresh.carry)


class TailCall(Call):
    """k3_sandwich_tail: the reduced latent {-16, ..., 16} / 8 [M, 3584] (the same on every rank), act
    {-2, ..., 2} / 8 [M, 384] per rank, rank r's latent slice at lo = ((5 r + shift) % 16) x 224 (``shift``: the seed
    unless given), tail_weight from weight set ``weights``. ``tap``: None, "mix" (the pre-norm mixture into columns
    [2 H, 3 H) of a NaN-filled [M, 5 H] capture buffer) or "updated" (``updated`` into its columns [4 H, 5 H));
    ``updated_out``: store ``updated`` into row 1 of a NaN-filled [3, M, H] bank."""

    def __init__(
        self,
        seed,
        tokens,
        snapshots,
        prefix=True,
        weights=0,
        shift=None,
        tap=None,
        updated_out=False,
    ):
        g = _gen(seed)
        self.tokens = tokens
        self.latent = ls.exact_bf16(g, (tokens, LATENT), -16, 17, 1 / 8)
        self.acts = [ls.exact_bf16(g, (tokens, ACT), -2, 3, 1 / 8) for _ in range(R.world)]
        shift = seed if shift is None else shift
        self.los = [((5 * r + shift) % SLICES) * LAT_SLICE for r in range(R.world)]
        self.carry = _running_sum(g, tokens) if prefix is True else prefix
        self.weights = WEIGHTS["tail"][weights]
        self.block, self.res_w, self.rms_w, self.out_w = _attn_res_inputs(g, snapshots, tokens)
        self._set_options(tap, updated_out)

    def _set_options(self, tap, updated_out) -> None:
        self.tap_kind, self.cap, self.tap = tap, None, None
        if tap is not None:
            self.cap = torch.full(
                (self.tokens, 5 * H), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            self.tap = self.cap[:, self._tap_col() * H : (self._tap_col() + 1) * H]
        self.bank, self.updated_out = None, None
        if updated_out:
            self.bank = torch.full(
                (3, self.tokens, H), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            self.updated_out = self.bank[1]

    def _tap_col(self) -> int:
        return 2 if self.tap_kind == "mix" else 4

    def with_options(self, tap=None, updated_out=False) -> "TailCall":
        """The same call -- the same input tensors -- with other output options."""
        other = copy.copy(self)
        other._set_options(tap, updated_out)
        return other

    def partials(self) -> List[torch.Tensor]:
        """Every rank's partial in fp64: scale_t (acc_lat) + acc_act, one rounding to bf16; the padding columns of
        tail_weight are zero, so the latent slice is the 224 columns from lo."""
        lat = self.latent.double()
        scale = (lat.square().mean(dim=1, keepdim=True) + LAT_EPS).rsqrt()
        out = []
        for act, w, lo in zip(self.acts, self.weights, self.los):
            w64 = w.double()
            acc_lat = lat[:, lo : lo + LAT_SLICE] @ w64[:, :LAT_SLICE].t()
            acc_act = act.double() @ w64[:, LAT_PAD:].t()
            out.append((acc_lat * scale + acc_act).float().bfloat16())
        return out

    def ref(self):
        updated = reduce_ref(self.partials(), self.carry)
        normed = ls.residual_update_ref(
            updated, self.block, self.res_w, self.rms_w, RMS_EPS, self.out_w, OUT_EPS
        )
        return normed, updated

    def run(self, ws):
        r = R.rank
        return OPS["tail"](self.latent, self.acts[r], self.weights[r], self.los[r], LAT_EPS, self.carry, self.block,
                           self.res_w, self.rms_w, self.out_w, RMS_EPS, OUT_EPS, ws, tap=self.tap,
                           tap_updated=self.tap_kind == "updated", updated_out=self.updated_out)  # fmt: skip

    def refill(self, fresh, carry):
        self.latent.copy_(fresh.latent)
        for mine, new in zip(self.acts, fresh.acts):
            mine.copy_(new)
        self.block.copy_(fresh.block)
        if carry:
            self.carry.copy_(fresh.carry)

    def verify(self, got, where):
        normed, updated = got
        want_normed, want_updated = self.ref()
        _within(updated, want_updated, TOL_TAIL, "tail_updated", where, "updated")
        _within(normed, want_normed, TOL, "normed", where, "normed")
        outputs = [normed, updated]
        if self.updated_out is not None:
            assert updated.data_ptr() == self.updated_out.data_ptr(), (
                f"{where}: updated is not updated_out"
            )
            assert bool(torch.isnan(self.bank[0::2].float()).all()), (
                f"{where}: another bank row was written"
            )
        if self.tap is not None:
            col = self._tap_col()
            rest = torch.cat([self.cap[:, : col * H], self.cap[:, (col + 1) * H :]], dim=1)
            assert bool(torch.isnan(rest.float()).all()), (
                f"{where}: the capture buffer was written outside the tap"
            )
            if self.tap_kind == "mix":
                mix = attn_res_mixture(want_updated, self.block, self.res_w, self.rms_w)
                _within(self.tap, mix, TOL, "tap", where, "tapped mixture")
            else:
                assert torch.equal(bits(self.tap), bits(updated)), (
                    f"{where}: the tap is not updated"
                )
            outputs.append(self.tap)
        assert R.same_on_ranks(*outputs), f"{where}: ranks disagree"

    def wrong_fraction(self, got) -> float:
        """The fraction of ``updated``'s elements off the reference by more than the tolerance."""
        want = self.ref()[1].float()
        return ((got[1].float() - want).abs() > TOL_TAIL * want.abs().max()).float().mean().item()

    def wrong_margin(self, got) -> float:
        """``updated``'s largest error in units of the tolerance."""
        want = self.ref()[1].float()
        return ((got[1].float() - want).abs().max() / (TOL_TAIL * want.abs().max())).item()


class PlainCall(Call):
    """k3_sandwich_plain: x {-2, ..., 2} / 8 [M, 384] per rank and a [7168, 384] slice; with ``swiglu`` the gate_up
    output [M, 1792] per rank (gate columns first: gates from GATES, up {-2, ..., 2} / 64) and a [7168, 896] slice.
    silu is exact in fp32 on these gates -- silu(0) = 0, and for g >= 32 the sum 1 + exp(-g) rounds to 1, so the
    sigmoid is 1 -- hence silu_and_mul(x) = gate * up exactly, in the kernel (bf16((g * sigmoid(g)) * up)) as in the
    reference. ``residual``: True (drawn) or a tensor."""

    def __init__(self, seed, tokens, swiglu=False, residual=True, weights=0):
        g = _gen(seed)
        self.tokens = tokens
        self.swiglu = swiglu
        if swiglu:
            self.xs = [
                torch.cat(
                    [
                        _gates(g, (tokens, DOWN_K)),
                        ls.exact_bf16(g, (tokens, DOWN_K), -2, 3, 1 / 64),
                    ],
                    dim=1,
                )
                for _ in range(R.world)
            ]
            self.weights = WEIGHTS["down"][weights]
        else:
            self.xs = [ls.exact_bf16(g, (tokens, PLAIN_K), -2, 3, 1 / 8) for _ in range(R.world)]
            self.weights = WEIGHTS["plain"][weights]
        self.carry = _running_sum(g, tokens) if residual is True else residual
        self.norm_w = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()

    def operand(self, x: torch.Tensor) -> torch.Tensor:
        """The projection's input: x, or silu_and_mul(x) = gate * up (exact on these payloads)."""
        if not self.swiglu:
            return x
        return (x[:, :DOWN_K].float() * x[:, DOWN_K:].float()).bfloat16()

    def ref(self):
        partials = [bf16_partial(self.operand(x), w) for x, w in zip(self.xs, self.weights)]
        updated = reduce_ref(partials, self.carry)
        u = updated.float()
        normed = (
            u * (u.square().mean(dim=-1, keepdim=True) + EPS).rsqrt() * self.norm_w.float()
        ).bfloat16()
        return normed, updated

    def run(self, ws):
        r = R.rank
        return OPS["plain"](
            self.xs[r], self.weights[r], self.carry, self.norm_w, EPS, ws, swiglu=self.swiglu
        )

    def refill(self, fresh, carry):
        for mine, new in zip(self.xs, fresh.xs):
            mine.copy_(new)
        if carry:
            self.carry.copy_(fresh.carry)


def _label(i: int, call: Call) -> str:
    return f"call {i} ({type(call).__name__} M {call.tokens})"


def run_sequence(
    seq, ws, verify: bool = True, late: Optional[random.Random] = None, where: str = "sequence"
):
    """Run ``seq`` -- (chain, call) pairs -- in order on ``ws``. A call's carry is the ``updated`` of the previous call
    of its chain (a chain's first call keeps its own), as the model chains its residual streams. With ``late`` (a
    random.Random seeded alike on every rank) every call starts after a barrier, a random rank 5 ms late; with
    ``verify`` every call is checked against its reference as it returns. Returns the outputs."""
    last: Dict[str, torch.Tensor] = {}
    outs = []
    for i, (chain, call) in enumerate(seq):
        if chain in last:
            call.carry = last[chain]
        if late is not None:
            R.barrier()
            R.late(late.randrange(R.world))
        got = call.run(ws)
        if verify:
            call.verify(got, f"{where} {_label(i, call)}")
        last[chain] = got[1]
        outs.append(got)
    return outs


def _chain_heads(seq) -> set:
    seen, heads = set(), set()
    for i, (chain, _) in enumerate(seq):
        if chain not in seen:
            seen.add(chain)
            heads.add(i)
    return heads


def armed_and_sized(ws) -> None:
    """``create`` returned this rank's view of a buffer sized for the group (two halves of [8 tokens][W ranks][7168]
    bf16 as int32 words), every word empty, every counter zero."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import k3_sandwich_kernel as kernel

    assert ws.world_size == R.world and ws.rank == R.rank
    assert ws.uc.dtype == ws.mc.dtype == ws.flags.dtype == torch.int32
    words = 2 * 8 * R.world * H // 2
    assert ws.uc.numel() == ws.mc.numel() == words == kernel.buffer_words(R.world)
    assert ws.flags.numel() == FLAG_WORDS
    assert bool((ws.uc == kernel.EMPTY_WORD).all()), "every word empty"
    assert int(ws.flags.abs().sum()) == 0, "every counter 0"


def _create(workspace_type, capture: bool) -> str:
    """This rank's ``create``, under CUDA-graph capture on a side stream or eagerly; returns the message of the
    RuntimeError it raised ("" if it returned), once neither stream is left capturing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    message = ""
    try:
        if capture:
            with torch.cuda.graph(graph, stream=stream):
                workspace_type.create(R.mapping, fabric_handle=R.fabric)
        else:
            workspace_type.create(R.mapping, fabric_handle=R.fabric)
    except RuntimeError as exc:
        message = str(exc)
    with torch.cuda.stream(stream):
        side_capturing = torch.cuda.is_current_stream_capturing()
    assert not (torch.cuda.is_current_stream_capturing() or side_capturing), (
        "a stream was left capturing"
    )
    return message


def create_refuses_capture(workspace_type, ws, next_call: Call) -> None:
    """``create`` is collective and the ranks agree before it allocates, so a rank under CUDA-graph capture makes every
    rank raise RuntimeError. (a) Every rank captures: every rank raises, naming the capture. (b) The last rank captures
    while its peers call ``create`` eagerly at the same point: every rank raises, the capturing rank naming the
    capture, its peers saying another rank cannot. No rank reaches the allocation in either case (the counted
    multicast allocation, which the workspaces in use went through), each rank frees the communicator made for it, no
    stream is left capturing, and the workspace in use is untouched: the next call is correct."""
    from tensorrt_llm._torch.distributed import ops

    assert ALLOCATIONS[0] > 0, "the counter did not see the workspaces in use being created"
    before = ALLOCATIONS[0]
    make = ops._get_mnnvl_tp_group_comm
    comms = []

    def recording_make(mapping):
        comms.append(make(mapping))
        return comms[-1]

    capturing = R.world - 1
    ops._get_mnnvl_tp_group_comm = recording_make
    try:
        every = _create(workspace_type, capture=True)
        R.barrier()
        one = _create(workspace_type, capture=R.rank == capturing)
    finally:
        ops._get_mnnvl_tp_group_comm = make
    every_ok = CAPTURE_REFUSED in every
    one_ok = (CAPTURE_REFUSED if R.rank == capturing else PEER_REFUSED) in one
    allocated = ALLOCATIONS[0] - before
    freed = len(comms) == 2 and all(c == R.MPI.COMM_NULL for c in comms)
    assert R.all_true(every_ok and one_ok and allocated == 0 and freed), (
        f"every rank capturing: {every!r}; rank {capturing} capturing: {one!r}; allocations reached {allocated}; "
        f"communicators freed {freed}"
    )
    next_call.verify(next_call.run(ws), "after the refused creates")


def counters_advance_once(ws, call: Call) -> None:
    """One call advances every one of the 56 CTAs' counters by exactly one (the parity, the half the next call uses,
    is shared by all of them) and leaves the spare flag words alone."""
    before = ws.flags.clone()
    call.verify(call.run(ws), "counter call")
    torch.cuda.synchronize()
    delta = (ws.flags - before)[:NUM_CTAS]
    assert bool((delta == 1).all()), f"counters advanced by {delta.unique().tolist()}"
    assert int(ws.flags[NUM_CTAS:].abs().sum()) == 0, "a spare flag word was written"


def unsupported_raises(ws, bad_calls, next_call: Call) -> None:
    """Each of ``bad_calls`` ((label, thunk) pairs) raises ValueError on every rank before any launch: every counter
    is as it was. The next call is correct."""
    before = ws.flags.clone()
    raised = {}
    for label, thunk in bad_calls:
        try:
            thunk()
            raised[label] = False
        except ValueError:
            raised[label] = True
    torch.cuda.synchronize()
    kept = torch.equal(ws.flags, before)
    assert R.all_true(all(raised.values()) and kept), f"raised {raised}, counters kept {kept}"
    next_call.verify(next_call.run(ws), "after the rejected calls")


def dip_and_regrow(ws, make_step: Callable[[int, int], list], where: str = "dip") -> None:
    """Decode steps (``make_step(seed, M)``) at M = 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, a random rank 5 ms late at every
    call, every call against its reference: a call after a smaller one must not see words an older, larger call
    left."""
    late = random.Random(7)
    for i, t in enumerate(DIP_STEPS):
        run_sequence(make_step(3000 + 100 * i, t), ws, late=late, where=f"{where} step {i} M {t}")


def interleaved(ws_a, ws_b, make_call: Callable[[int], Call], where: str = "interleaved") -> None:
    """Calls alternate between two workspaces in an irregular pattern (A A B A B B ...), so that the two objects'
    counters differ, every call against its reference. The pattern is the same on every rank: calls on one stream
    run in order and each waits for its peers, so ranks issuing calls on two workspaces in different orders deadlock
    (measured for the MNNVL entry; not exercised)."""
    for i, which in enumerate(INTERLEAVE):
        call = make_call(i)
        call.verify(call.run(ws_a if which == "A" else ws_b), f"{where} {which} {i}")


def capture_and_replay(ws, make_seq: Callable[[int], list], eager_between: Callable[[int], List[Call]], where: str,
                       replays: int = REPLAYS) -> None:  # fmt: skip
    """Capture the call sequence ``make_seq(seed)`` on ``ws`` -- after one eager, verified run of it: the kernels
    compile on their first call -- and replay it ``replays`` times with every input rewritten in place from
    ``make_seq(another seed)``, every replayed call against its reference, with the eager calls ``eager_between(rep)``
    on the same workspace between replays, each against its reference. Replays and eager calls advance the same
    counters, in the same order on every rank."""
    seq = make_seq(5000)
    heads = _chain_heads(seq)
    run_sequence(seq, ws, where=f"{where} eager run")
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = run_sequence(seq, ws, verify=False)
    R.barrier()
    for rep in range(replays):
        fresh = make_seq(6000 + 100 * rep)
        for i, ((_, call), (_, new)) in enumerate(zip(seq, fresh)):
            call.refill(new, carry=i in heads)
        R.barrier()
        graph.replay()
        for i, ((_, call), got) in enumerate(zip(seq, outs)):
            call.verify(got, f"{where} replay {rep} {_label(i, call)}")
        for j, call in enumerate(eager_between(rep)):
            call.verify(call.run(ws), f"{where} eager {_label(j, call)} after replay {rep}")
    del graph


def swapped_pair_is_wrong(ws, c1: Call, c2: Call, c3: Call) -> None:
    """Negative control: rank 0 makes two same-shaped calls on one workspace in swapped order. Every call returns and
    nothing raises or hangs (the counters agree), but every rank's two results are wrong: a call pairs with the
    peers' call at the same position. Wrong means more than half of ``updated``'s elements fail the comparison and
    the largest error is over 10 times its bound. Then a plain call is correct again."""
    R.barrier()
    if R.rank == 0:
        got2, got1 = c2.run(ws), c1.run(ws)
    else:
        got1, got2 = c1.run(ws), c2.run(ws)
    torch.cuda.synchronize()
    wrong = [c.wrong_fraction(got) for c, got in ((c1, got1), (c2, got2))]
    margin = [c.wrong_margin(got) for c, got in ((c1, got1), (c2, got2))]
    assert R.all_true(min(wrong) > 0.5 and min(margin) > 10), (
        f"the swap went unnoticed: wrong fractions {wrong}, largest errors over the bound {margin}"
    )
    c3.verify(c3.run(ws), "after the swapped pair")
