# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/mnnvl_allgather_split`` catalog entry on its ``MnnvlWorkspace``.

The op takes one turn of the workspace's Lamport rotation, which every MNNVL op of the TP group advances (this op,
``comm/mnnvl_fusion_allreduce`` on either path and ``comm/mnnvl_allreduce_attn_res``). So beyond single calls (every
certified split at every certified token count against a bit-exact reference) this drives call *sequences*: decode
steps whose token count dips and grows back with a random rank late at every call; two workspaces interleaved; the
three MNNVL ops interleaved on one workspace; CUDA-graph capture and replay mixed with eager calls; and a negative
control in which one rank swaps two calls and every rank gets a wrong answer without an error. After every eager call
the workspace's ``buffer_flags`` are compared with this file's model of the rotation (``Rotation``): one turn per call
whatever the op, one stage, the bytes the call wrote.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_allgather_split_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _mnnvl_allgather_split_op_matrix.py --launcher srun \
        --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
rotations). The collected entry point is ``test_modeling_v2_mnnvl_allgather_split_op_matrix.py``.

Every rank draws every rank's rows from one seed, so each rank holds the whole reference: the bf16 columns rounded by
torch (round to nearest even), the fp32 columns copied, -0.0 made +0.0, compared bit for bit. The rows exercise the
rounding (normal values over exponents 2^-16..2^16, inexact in bf16, and exact midpoints between two bf16 values),
the -0.0 rule in both parts, and special fp32 words. Every output is also compared bitwise across the ranks. The
all-reduce and attention-residual calls of the shared-workspace, capture and sequence checks are checked as in their
own matrices: sums bit for bit (small multiples of 1/16), normed outputs within a tolerance.
"""

import copy
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "mnnvl_allgather_split requires CUDA devices"

DEADLINE_S = 900
# Kimi K3's MoE head, sharded over the TP group: the routed latent (the bf16 columns) and the router logits of the
# routed experts (the fp32 columns).
LATENT = 3584
EXPERTS = 896
H_MODEL = 7168  # Kimi K3's hidden size (the all-reduces this op shares the workspace with)
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
# (bf16 columns, fp32 columns) per rank: Kimi K3's MoE head at W = 2, 4, 8, 16, then the smallest split.
SPLITS = tuple((LATENT // w, EXPERTS // w) for w in (2, 4, 8, 16)) + ((8, 4),)
DECODE_MAX_TOKENS = 8
# Kimi K3's one-shot ceilings: DECODE_AR_ONE_SHOT_MAX_BYTES on decode steps (at most 8 tokens),
# WIDE_AR_ONE_SHOT_MAX_BYTES (main's default) on wide decode steps.
DECODE_ONE_SHOT_MAX_BYTES = 4 << 20
WIDE_ONE_SHOT_MAX_BYTES = 1 << 20
EPS = 1e-5
TOL = 1e-2  # the fused all-reduce's normed, as its own matrix bounds it
ATTN_RES_TOL = 2e-2  # mnnvl_allreduce_attn_res's normed, as its own matrix bounds it
ATTN_RES_MAX_TOKENS = 16  # Kimi K3 sends the attention-residual all-reduce up to 16 tokens
INT16_MIN = torch.iinfo(torch.int16).min  # the bf16 -0.0 word
INT32_MIN = torch.iinfo(torch.int32).min  # the fp32 -0.0 word, the Lamport buffers' empty word
NEG_ZERO_STRIDE = 37  # every rank's all-reduce input holds -0.0 in these columns
# fp32 words the all-gather carries unchanged: +inf, -inf, NaN, +-the smallest denormal, the largest float.
SPECIAL_WORDS = (0x7F800000, -0x00800000, 0x7FC00000, 1, -0x7FFFFFFF, 0x7F7FFFFF)
LAYERS = 6
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 64, 3, 32, 8, 16, 1, 64, 8)
SHARED_STEPS = (8, 2, 16, 64, 1, 7, 32, 8, 3, 16)
INTERLEAVED_TOKENS = (3, 8, 1, 64, 5, 16, 2)

R = None
allgather_split = None
required_buffer_bytes = None
fusion_allreduce = None
MnnvlWorkspace = None
BUFFER_BYTES = None
WS_A = None
WS_B = None
ROT = {}
STATS = {"normed_err": 0.0, "attn_res_err": 0.0}


def buffer_bytes(world: int) -> int:
    """One Lamport buffer holds the largest call this matrix makes: the all-gather at the widest certified split and
    64 tokens, a wide step's [64, 7168] all-reduce sent two-shot, or the attention-residual all-reduce at 16
    tokens."""
    widest = max(2 * b + 4 * f for b, f in SPLITS)
    return max(
        max(TOKENS) * world * widest,
        2 * -(-max(TOKENS) // world) * world * H_MODEL * 2,
        ATTN_RES_MAX_TOKENS * H_MODEL * world * 2,
    )


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def positive_zero(t: torch.Tensor) -> torch.Tensor:
    """``t`` with every -0.0 made +0.0, by bit pattern: the Lamport buffers' empty word never travels."""
    words = bits(t)
    sign = torch.iinfo(words.dtype).min
    return torch.where(words == sign, torch.zeros_like(words), words).view(t.dtype)


def one_shot_bytes(tokens: int, hidden: int) -> int:
    return tokens * hidden * R.world * 2


def ceiling(tokens: int, hidden: int, path: str) -> int:
    """An all-reduce's one_shot_max_bytes: at the boundary for "one" (the call's one-shot footprint itself, so
    one-shot) and "two" (one byte less, so two-shot), or Kimi K3's ceiling for a step of ``tokens`` ("k3")."""
    if path == "one":
        return one_shot_bytes(tokens, hidden)
    if path == "two":
        return one_shot_bytes(tokens, hidden) - 1
    return DECODE_ONE_SHOT_MAX_BYTES if tokens <= DECODE_MAX_TOKENS else WIDE_ONE_SHOT_MAX_BYTES


def k3_split():
    """Kimi K3's MoE head shard at this world size: (latent columns, router-logit columns) per rank."""
    return LATENT // R.world, EXPERTS // R.world


class Rotation:
    """This test's model of a workspace's ``buffer_flags`` (``mnnvl_workspace.FLAG_WORDS``): the buffer the next call
    takes, the one the last call used, the bytes per buffer, the last call's stage count and the bytes it wrote per
    stage (what the next call clears), and the arrival counter (0 between calls)."""

    def __init__(self, ws):
        self.ws = ws
        self.flags = [0, 2, ws.buffer_bytes, 0, 0, 0, 0, 0, 0]  # as create() arms them

    def read(self):
        torch.cuda.synchronize()
        return self.ws.buffer_flags.view(torch.int32).tolist()

    def unchanged(self, where: str) -> None:
        got = self.read()
        assert got == self.flags, f"{where}: buffer_flags {got} moved from {self.flags}"

    def advance(self, record, where: str, calls: int = 1) -> None:
        """``calls`` more calls ran on the workspace, the last of which recorded ``record`` = (stages, bytes)."""
        stages, written = record
        current = (self.flags[0] + calls) % 3
        self.flags = [current, (current - 1) % 3, self.ws.buffer_bytes, stages, *written, 0]
        got = self.read()
        assert got == self.flags, f"{where}: buffer_flags {got}, expected {self.flags}"


def rot(ws) -> Rotation:
    return ROT[id(ws)]


def gather_payload(g, tokens, bf16_columns, fp32_columns):
    """One rank's fp32 rows for the all-gather: normal values over exponents 2^-16..2^16 in the bf16 columns
    (inexact in bf16), exact midpoints between two bf16 values in every fourth of them (ties go to even), -0.0 in
    every fourth column of each part, and special words (infinities, NaN, denormals, the largest float) in the first
    row of the fp32 part."""
    b = bf16_columns
    x = torch.randn(tokens, b + fp32_columns, generator=g, device="cuda")
    x[:, :b] *= torch.exp2(((torch.arange(b, device="cuda") % 5) - 2).float() * 8)
    nearest = x[:, 1:b:4].bfloat16().float()
    x[:, 1:b:4] = (nearest.view(torch.int32) | 0x8000).view(torch.float32)
    words = x.view(torch.int32)
    words[:, 3:b:4] = INT32_MIN
    words[:, b + 2 :: 4] = INT32_MIN
    special = SPECIAL_WORDS[: min(len(SPECIAL_WORDS), fp32_columns)]
    words[0, b : b + len(special)] = torch.tensor(special, dtype=torch.int32, device="cuda")
    return x


class AG:
    """One mnnvl_allgather_split call's arguments on every rank and its reference."""

    def __init__(self, seed, tokens, bf16_columns, fp32_columns):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.tokens, self.bf16_columns, self.fp32_columns = tokens, bf16_columns, fp32_columns
        self.inputs = [
            gather_payload(g, tokens, bf16_columns, fp32_columns) for _ in range(R.world)
        ]

    def fresh(self, seed):
        return AG(seed, self.tokens, self.bf16_columns, self.fp32_columns)

    def run(self, ws, x=None):
        return allgather_split(self.inputs[R.rank] if x is None else x, self.bf16_columns, ws)

    def ref(self):
        """Every rank's leading columns rounded to bf16 by torch (round to nearest even) and its other columns as
        they are, in rank order, -0.0 made +0.0."""
        b = self.bf16_columns
        bf16_out = torch.cat([x[:, :b] for x in self.inputs], dim=1).bfloat16()
        fp32_out = torch.cat([x[:, b:] for x in self.inputs], dim=1)
        return positive_zero(bf16_out), positive_zero(fp32_out)

    def verify(self, got, where: str) -> None:
        want = self.ref()
        assert all(bits_equal(a, b) for a, b in zip(got, want)), (
            f"{where}: the gather differs from the reference"
        )
        assert R.same_on_ranks(*got), f"{where}: ranks disagree"

    def record(self):
        """What the call leaves in buffer_flags: one stage, the bytes it wrote."""
        written = self.tokens * R.world * (2 * self.bf16_columns + 4 * self.fp32_columns)
        return 1, (written, 0, 0, 0)

    def static(self):
        return {"x": self.inputs[R.rank].clone()}

    def refill(self, fresh, static) -> None:
        """Take ``fresh``'s rows (same shape) into this call and its static buffer."""
        self.inputs = fresh.inputs
        static["x"].copy_(fresh.inputs[R.rank])


class AR:
    """One comm/mnnvl_fusion_allreduce call's arguments on every rank and its reference. ``residual``: a tensor
    (chained), True (drawn) or None (the plain sum); ``path``: see ``ceiling``."""

    def __init__(self, seed, tokens, hidden, residual=None, path="k3"):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.tokens, self.hidden, self.path = tokens, hidden, path
        self.one_shot_max_bytes = ceiling(tokens, hidden, path)
        self.inputs = []
        for _ in range(R.world):
            x = ls.exact_bf16(g, (tokens, hidden), -4, 5, 1 / 16)
            x.view(torch.int16)[:, ::NEG_ZERO_STRIDE] = INT16_MIN
            self.inputs.append(x)
        if residual is True:
            residual = ls.exact_bf16(g, (tokens, hidden), -32, 33, 1 / 16)
        self.residual = residual
        self.gamma = None
        if residual is not None:
            self.gamma = (1.0 + 0.1 * torch.randn(hidden, generator=g, device="cuda")).bfloat16()

    @property
    def fused(self) -> bool:
        return self.residual is not None

    @property
    def one_shot(self) -> bool:
        return one_shot_bytes(self.tokens, self.hidden) <= self.one_shot_max_bytes

    def fresh(self, seed):
        return AR(
            seed, self.tokens, self.hidden, residual=True if self.fused else None, path=self.path
        )

    def run(self, ws, x=None, residual=None):
        x = self.inputs[R.rank] if x is None else x
        if not self.fused:
            return fusion_allreduce(x, ws, self.one_shot_max_bytes)
        residual = self.residual if residual is None else residual
        return fusion_allreduce(x, ws, self.one_shot_max_bytes, residual, self.gamma, EPS)

    def ref(self):
        total = positive_zero(torch.stack([x.float() for x in self.inputs]).sum(dim=0))
        if not self.fused:
            return total.bfloat16()
        updated = (total + self.residual.float()).bfloat16()
        x = updated.float()
        normed = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS) * self.gamma.float()
        return normed.bfloat16(), updated

    def verify(self, got, where: str) -> None:
        want = self.ref()
        if not self.fused:
            assert bits_equal(got, want), f"{where}: the sum differs from the exact reference"
            assert R.same_on_ranks(got), f"{where}: ranks disagree"
            return
        (normed, updated), (want_normed, want_updated) = got, want
        assert bits_equal(updated, want_updated), (
            f"{where}: updated differs from the exact reference"
        )
        err = ls.rel_err(normed, want_normed)
        STATS["normed_err"] = max(STATS["normed_err"], err)
        assert err <= TOL, f"{where}: normed rel err {err:.3e} > {TOL}"
        assert R.same_on_ranks(normed, updated), f"{where}: ranks disagree"

    def record(self):
        if self.one_shot:
            return 1, (one_shot_bytes(self.tokens, self.hidden), 0, 0, 0)
        rows = -(-self.tokens // R.world) * R.world
        return 2, (rows * self.hidden * 2, self.tokens * self.hidden * 2, 0, 0)

    def static(self):
        static = {"x": self.inputs[R.rank].clone()}
        if self.fused:
            static["residual"] = self.residual.clone()
        return static

    def refill(self, fresh, static) -> None:
        self.inputs = fresh.inputs
        static["x"].copy_(fresh.inputs[R.rank])
        if self.fused:
            self.residual = fresh.residual
            static["residual"].copy_(fresh.residual)


class AttnRes:
    """One comm/mnnvl_allreduce_attn_res call (G1's entry, called through its op on the workspace's comm_buffer and
    buffer_flags, as that entry's wrapper does) and its reference. ``prefix``: a tensor (chained) or True (drawn)."""

    def __init__(self, seed, tokens, snapshots, prefix=True):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.tokens, self.snapshots = tokens, snapshots
        self.inputs = [ls.exact_bf16(g, (tokens, H_MODEL), -4, 5, 1 / 16) for _ in range(R.world)]
        if prefix is True:
            prefix = ls.exact_bf16(g, (tokens, H_MODEL), -32, 33, 1 / 16)
        self.prefix = prefix
        self.block = torch.randn(snapshots, tokens, H_MODEL, generator=g, device="cuda").bfloat16()
        self.res_w = (torch.randn(H_MODEL, generator=g, device="cuda") * 0.05).bfloat16()
        self.rms_w = (1.0 + 0.1 * torch.randn(H_MODEL, generator=g, device="cuda")).bfloat16()
        self.out_w = (1.0 + 0.1 * torch.randn(H_MODEL, generator=g, device="cuda")).bfloat16()

    def run(self, ws):
        normed, updated = torch.ops.trtllm.mnnvl_allreduce_attn_res(
            self.inputs[R.rank],
            self.prefix,
            self.block,
            self.res_w,
            self.rms_w,
            self.out_w,
            EPS,
            EPS,
            ws.comm_buffer(torch.bfloat16),
            ws.buffer_flags,
        )
        return normed, updated

    def ref(self):
        updated = (sum(x.float() for x in self.inputs) + self.prefix.float()).bfloat16()
        normed = ls.residual_update_ref(
            updated, self.block, self.res_w, self.rms_w, EPS, self.out_w, EPS
        )
        return normed, updated

    def verify(self, got, where: str) -> None:
        (normed, updated), (want_normed, want_updated) = got, self.ref()
        assert bits_equal(updated, want_updated), (
            f"{where}: attn_res updated differs from the exact sum"
        )
        err = ls.rel_err(normed, want_normed)
        STATS["attn_res_err"] = max(STATS["attn_res_err"], err)
        assert err <= ATTN_RES_TOL, f"{where}: attn_res normed rel err {err:.3e} > {ATTN_RES_TOL}"
        assert R.same_on_ranks(normed, updated), f"{where}: ranks disagree"

    def record(self):
        return 1, (self.tokens * H_MODEL * R.world * 2, 0, 0, 0)


def call_and_check(call, ws, where: str, late=None):
    """Run ``call`` eagerly on ``ws`` (rank ``late``, if given, 5 ms late), check its result and what it left in the
    workspace's flags; return the result."""
    if late is not None:
        R.barrier()
        R.late(late)
    got = call.run(ws)
    call.verify(got, where)
    rot(ws).advance(call.record(), where)
    return got


def call_late(call, ws, where: str, late_rng: random.Random):
    """``call_and_check`` with a rank drawn from ``late_rng`` (the same draw on every rank) 5 ms late."""
    return call_and_check(call, ws, where, late=late_rng.randrange(R.world))


def expect_refusal(fn, error, where: str) -> None:
    """``fn`` raises ``error`` on every rank and WS_A's flags do not move."""
    try:
        fn()
        raised = False
    except error:
        raised = True
    assert R.all_true(raised), f"{where}: not refused with {error.__name__} on every rank"
    rot(WS_A).unchanged(where)


def with_inputs(call, inputs):
    """``call`` as if the ranks had sent ``inputs`` (one tensor per rank)."""
    mixed = copy.copy(call)
    mixed.inputs = inputs
    return mixed


def check_workspaces_are_armed_and_sized() -> None:
    """create() armed both workspaces (every Lamport word -0.0; flags at buffer 0, buffer 2 dirty with nothing to
    clear), and one buffer holds the largest certified all-gather and every other call of this matrix."""
    for ws in (WS_A, WS_B):
        assert ws.world_size == R.world and ws.rank == R.rank
        assert ws.buffer_bytes == BUFFER_BYTES and BUFFER_BYTES % 32 == 0
        assert ws.comm_buffer(torch.bfloat16).shape == (3, BUFFER_BYTES // 2)
        armed = ws.lamport.view(torch.int32)
        assert bool((armed == torch.tensor(INT32_MIN, dtype=torch.int32, device="cuda")).all()), (
            "every word -0.0"
        )
        rot(ws).unchanged("armed")
    for b, f in SPLITS:
        assert required_buffer_bytes(max(TOKENS), b, f, R.world) <= BUFFER_BYTES


def check_single_calls() -> None:
    """Every certified split at every certified token count, one call after the other on one workspace, bit for bit
    against the reference (the rounding, ties to even, -0.0, special fp32 words), the flags after each recording one
    stage and the bytes the call wrote, which equal required_buffer_bytes."""
    for b, f in SPLITS:
        for t in TOKENS:
            where = f"T {t} split {b} + {f}"
            call = AG(1000 + 37 * t + b, t, b, f)
            call_and_check(call, WS_A, where)
            need = required_buffer_bytes(t, b, f, R.world)
            assert need == call.record()[1][0], f"{where}: required_buffer_bytes {need}"


def check_unsupported_calls_raise_on_every_rank() -> None:
    """Refused on every rank before the workspace is touched (its flags do not move), and the next call is correct:
    more rows than one Lamport buffer holds (the wrapper's ValueError); bf16 columns not a multiple of 8, remaining
    columns not a multiple of 4, a bf16 input, no rows (the op's RuntimeError)."""
    b, f = k3_split()
    t = BUFFER_BYTES // (R.world * (2 * b + 4 * f)) + 1
    over = AG(2000, t, b, f)
    expect_refusal(lambda: over.run(WS_A), ValueError, f"T {t}: more rows than one buffer holds")
    rows = torch.randn(4, 16, device="cuda")
    narrow = rows[:, :10].contiguous()
    expect_refusal(lambda: allgather_split(rows, 12, WS_A), RuntimeError, "12 bf16 columns")
    expect_refusal(lambda: allgather_split(narrow, 8, WS_A), RuntimeError, "2 fp32 columns")
    expect_refusal(lambda: allgather_split(rows.bfloat16(), 8, WS_A), RuntimeError, "bf16 rows")
    expect_refusal(lambda: allgather_split(rows[:0], 8, WS_A), RuntimeError, "no rows")
    call_and_check(AG(2001, 8, b, f), WS_A, "after the refused calls")


def check_dip_and_regrow_sequence() -> None:
    """The k3_spec_accept failure mode: a call after a smaller one must not read what an older, larger call left.
    16 steps of 6 layers (one MoE head all-gather per layer, Kimi K3's split) at T 8, 8, 8, 2, 7, 8, 1, 1, 64, 3, 32,
    8, 16, 1, 64, 8, a random rank late at every call."""
    late = random.Random(7)
    b, f = k3_split()
    for i, t in enumerate(DIP_STEPS):
        for layer in range(LAYERS):
            call = AG(3000 + 100 * i + layer, t, b, f)
            call_late(call, WS_A, f"step {i} T {t} layer {layer}", late)


def check_two_workspaces_interleaved() -> None:
    """Two workspaces are two rotations: all-gathers of mixed token counts and splits alternate between them in an
    irregular pattern (A A B A B B ...), every call is correct and each workspace's flags move with its own calls
    only. The pattern is the same on every rank: calls on one stream are serialized and each waits for its peers, so
    ranks issuing calls on two workspaces in different orders deadlock (measured for mnnvl_allreduce_attn_res,
    runs/drafter/u4-mnnvl-srun-2)."""
    pattern = "AABABBAAAB" * 2
    for i, which in enumerate(pattern):
        t = INTERLEAVED_TOKENS[i % len(INTERLEAVED_TOKENS)]
        call = AG(4700 + i, t, *SPLITS[i % len(SPLITS)])
        call_and_check(call, WS_A if which == "A" else WS_B, f"interleaved {which} {i}")


def check_one_workspace_three_ops() -> None:
    """The three MNNVL entries on one workspace in Kimi K3's order, a random rank late at every call. A step of at
    most 16 tokens: per layer the pre-attention all-reduce with the attention-residual epilogue
    (comm/mnnvl_allreduce_attn_res, prefix chained), the MoE head all-gather (this entry, K3's split), the
    routed-latent all-reduce and the fused all-reduce (comm/mnnvl_fusion_allreduce, chained). A wide step (32, 64
    tokens): the wide all-reduce [T, 7168], the all-gather and the routed-latent all-reduce, at the 1 MiB ceiling.
    Every call takes one turn of the rotation whatever the op (the flags after each), and every result is right."""
    late = random.Random(11)
    b, f = k3_split()
    for i, t in enumerate(SHARED_STEPS):
        seed = 5000 + 100 * i
        g = torch.Generator(device="cuda").manual_seed(seed)
        prefix = ls.exact_bf16(g, (t, H_MODEL), -32, 33, 1 / 16)
        residual = ls.exact_bf16(g, (t, H_MODEL), -32, 33, 1 / 16)
        for layer in range(2):
            s = seed + 10 * layer + 1
            where = f"shared step {i} T {t} layer {layer}"
            if t <= ATTN_RES_MAX_TOKENS:
                attn = AttnRes(s, t, (0, 2, 5)[(i + layer) % 3], prefix=prefix)
                prefix = call_late(attn, WS_A, f"{where} attn_res", late)[1]
            else:
                call_late(AR(s, t, H_MODEL), WS_A, f"{where} wide", late)
            call_late(AG(s + 1, t, b, f), WS_A, f"{where} all-gather", late)
            call_late(AR(s + 2, t, LATENT), WS_A, f"{where} latent", late)
            if t <= ATTN_RES_MAX_TOKENS:
                fused = AR(s + 3, t, H_MODEL, residual=residual)
                residual = call_late(fused, WS_A, f"{where} fused", late)[1]


def check_graph_capture_and_replay() -> None:
    """A captured step of five calls on WS_B, the MoE layers of a decode step: the head all-gather (T 8), the
    routed-latent all-reduce one-shot, the head all-gather again, a [32, 3584] all-reduce sent two-shot and the head
    all-gather at T 32. Replayed 8 times with rewritten rows, an eager all-gather of another token count or split, or
    an all-reduce, on the same workspace between replays: replays and eager calls take turns of one rotation (the
    flags after each), in the same order on every rank, and every replayed and eager result is right."""
    b, f = k3_split()
    calls = [
        AG(6000, 8, b, f),
        AR(6001, 8, LATENT),
        AG(6002, 8, b, f),
        AR(6003, 32, LATENT, path="two"),
        AG(6004, 32, b, f),
    ]
    statics = [c.static() for c in calls]

    def step():
        return [c.run(WS_B, **s) for c, s in zip(calls, statics)]

    outs = step()  # every call once eagerly, outside capture
    for i, (c, got) in enumerate(zip(calls, outs)):
        c.verify(got, f"eager step call {i}")
    rot(WS_B).advance(calls[-1].record(), "eager step", calls=len(calls))
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = step()
    R.barrier()
    rot(WS_B).unchanged("captured, not replayed")
    eager = (
        lambda seed: AG(seed, 1, b, f),
        lambda seed: AG(seed, 64, b, f),
        lambda seed: AR(seed, 3, LATENT, path="one"),
        lambda seed: AG(seed, 16, *SPLITS[-1]),
    )
    for rep in range(8):
        for i, (c, s) in enumerate(zip(calls, statics)):
            c.refill(c.fresh(7000 + 100 * rep + i), s)
        R.barrier()
        graph.replay()
        for i, (c, got) in enumerate(zip(calls, outs)):
            c.verify(got, f"replay {rep} call {i}")
        rot(WS_B).advance(calls[-1].record(), f"replay {rep}", calls=len(calls))
        call_and_check(eager[rep % len(eager)](7500 + rep), WS_B, f"eager after replay {rep}")
    del graph


def check_create_under_capture_raises() -> None:
    """MnnvlWorkspace.create allocates and exchanges handles, so it refuses CUDA-graph capture: with every rank
    capturing, it raises RuntimeError on every rank (before any communication), and the next call on a workspace in
    use is correct."""
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    raised = False
    with torch.cuda.graph(graph, stream=stream):
        try:
            MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        except RuntimeError as exc:
            raised = "before capture" in str(exc)
    del graph
    assert R.all_true(raised), "create() under capture did not raise on every rank"
    call_and_check(AG(8500, 8, *k3_split()), WS_A, "after the refused create")


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 issues two same-shaped all-gathers on one workspace in swapped order. Nothing raises
    and nothing hangs (the two calls write the same slots of the same buffers), but every rank's two results are
    wrong: rank 0's columns hold rank 0's rows of the other call, every other rank's columns are right -- the k-th
    call on every rank gathers what every rank sent at position k. A plain call right after is correct again: a
    swapped pair realigns the positions."""
    b, f = k3_split()
    c1, c2 = AG(9000, 8, b, f), AG(9001, 8, b, f)
    R.barrier()
    if R.rank == 0:
        got2, got1 = c2.run(WS_A), c1.run(WS_A)
    else:
        got1, got2 = c1.run(WS_A), c2.run(WS_A)
    torch.cuda.synchronize()
    # Position 1 gathered rank 0's c2 rows with the others' c1 rows, position 2 the other way round.
    first = with_inputs(c1, [c2.inputs[0]] + c1.inputs[1:]).ref()
    second = with_inputs(c2, [c1.inputs[0]] + c2.inputs[1:]).ref()
    at1, at2 = (got2, got1) if R.rank == 0 else (got1, got2)
    paired = all(bits_equal(a, w) for a, w in zip(at1 + at2, first + second))
    assert paired, "not the position-paired gathers"
    right = [
        all(bits_equal(a, w) for a, w in zip(got, c.ref())) for c, got in ((c1, got1), (c2, got2))
    ]
    assert R.all_true(not any(right)), f"the swap went unnoticed: results right {right}"
    rot(WS_A).advance(c2.record(), "swapped pair", calls=2)
    call_and_check(AG(9002, 8, b, f), WS_A, "after the swapped pair")


CHECKS = [
    check_workspaces_are_armed_and_sized,
    check_single_calls,
    check_unsupported_calls_raise_on_every_rank,
    check_dip_and_regrow_sequence,
    check_two_workspaces_interleaved,
    check_one_workspace_three_ops,
    check_graph_capture_and_replay,
    check_create_under_capture_raises,
    # Stays last: it deliberately disagrees on call order.
    check_wrong_call_order_is_detected,
]


def _run_one_rank(args) -> int:
    global R, allgather_split, required_buffer_bytes, fusion_allreduce, MnnvlWorkspace
    global BUFFER_BYTES, WS_A, WS_B
    R = ls.Rank(args)
    assert R.world in (2, 4, 8, 16), f"K3's shapes shard over 2, 4, 8 or 16 ranks, not {R.world}"
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_allgather_split as module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_fusion_allreduce as reduce_module,
    )

    allgather_split = module.mnnvl_allgather_split
    required_buffer_bytes = module.required_buffer_bytes
    fusion_allreduce = reduce_module.mnnvl_fusion_allreduce
    MnnvlWorkspace = module.MnnvlWorkspace
    BUFFER_BYTES = buffer_bytes(R.world)
    with torch.inference_mode():
        WS_A = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        WS_B = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        for ws in (WS_A, WS_B):
            ROT[id(ws)] = Rotation(ws)
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(
            f"[rank 0] world {R.world}; buffer {BUFFER_BYTES} B; "
            f"max all-reduce normed rel err {STATS['normed_err']:.3e}; "
            f"max attn_res normed rel err {STATS['attn_res_err']:.3e}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
