# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/mnnvl_fusion_allreduce`` catalog entry on its ``MnnvlWorkspace``.

The op's correctness depends on state that outlives a call: the workspace's Lamport rotation, which every MNNVL op of
the TP group advances (this op on either path, ``comm/mnnvl_allgather_split`` and ``comm/mnnvl_allreduce_attn_res``).
So beyond single calls (every certified shape sent one-shot and two-shot back to back, the path chosen per call by
``one_shot_max_bytes`` at the exact boundary) this drives call *sequences*: decode steps whose token count dips and
grows back with Kimi K3's one-shot ceilings flipping the path inside the sequence, a random rank late at every call;
two workspaces interleaved; the three MNNVL ops interleaved on one workspace; the same mix queued with no host
synchronization between calls while ranks run ahead of one another; CUDA-graph capture and replay mixed with eager
calls; ``MnnvlWorkspace.create``'s failure model; and a negative control in which one rank swaps two calls and every
rank gets a wrong answer without an error. After every eager call the workspace's ``buffer_flags`` are compared with
this file's model of the rotation (``Rotation``): one turn per call whatever the op, the path the call took, the
bytes it wrote.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_fusion_allreduce_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _mnnvl_fusion_allreduce_op_matrix.py --launcher srun \
        --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
rotations). The collected entry point is ``test_modeling_v2_mnnvl_fusion_allreduce_op_matrix.py``.

Every rank draws every rank's inputs from one seed, so each rank holds the whole reference. The inputs of the sums
are small multiples of 1/16 (``_lockstep.exact_bf16``), so every sum over the ranks is exact in fp32 and in bf16
whatever the summation order, and the residual add is one bf16 rounding of an exact fp32 value in the op and in the
reference alike: the sum and ``updated`` are compared bit for bit, ``normed`` (an RMSNorm) against the fp32 reference
within ``TOL``. Every output is also compared bitwise across the ranks.
"""

import copy
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "mnnvl_fusion_allreduce requires CUDA devices"

DEADLINE_S = 900
H_LATENT = 3584  # Kimi K3's routed latent: the MoE all-reduce
H_MODEL = 7168  # Kimi K3's hidden size: the drafter's residual + RMSNorm all-reduces, a wide step's attention ones
HIDDENS = (H_LATENT, H_MODEL)
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
EXPERTS = 896  # the router logits the MoE head all-gather carries beside the latent
DECODE_MAX_TOKENS = 8
# Kimi K3's one-shot ceilings: DECODE_AR_ONE_SHOT_MAX_BYTES on decode steps (at most 8 tokens),
# WIDE_AR_ONE_SHOT_MAX_BYTES (main's default) on wide decode steps.
DECODE_ONE_SHOT_MAX_BYTES = 4 << 20
WIDE_ONE_SHOT_MAX_BYTES = 1 << 20
EPS = 1e-5
# normed: max |err| / max |ref|. The kernels round each square to bf16 before summing (at most 2^-9 on the rsqrt)
# and round the output to bf16 (2^-8); the fp32 reference does neither.
TOL = 1e-2
ATTN_RES_TOL = 2e-2  # mnnvl_allreduce_attn_res's normed, as its own matrix bounds it
ATTN_RES_MAX_TOKENS = 16  # Kimi K3 sends the attention-residual all-reduce up to 16 tokens
INT16_MIN = torch.iinfo(torch.int16).min  # the bf16 -0.0 word
INT32_MIN = torch.iinfo(torch.int32).min  # the fp32 -0.0 word, the Lamport buffers' empty word
NEG_ZERO_STRIDE = 37  # every rank's all-reduce input holds -0.0 in these columns
# fp32 words the all-gather carries unchanged: +inf, -inf, NaN, +-the smallest denormal, the largest float.
SPECIAL_WORDS = (0x7F800000, -0x00800000, 0x7FC00000, 1, -0x7FFFFFFF, 0x7F7FFFFF)
LAYERS = 8
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 64, 3, 32, 8, 16, 1, 64, 8)
SHARED_STEPS = (8, 2, 16, 64, 1, 7, 32, 8, 3, 16)
QUEUED_STEPS = (8, 2, 64, 16, 1, 32, 7)  # one round of the queued check: 52 calls
QUEUED_ROUNDS = 3
QUEUE_LATE_S = 0.02
INTERLEAVED = (  # (T, H, fused, path)
    (3, H_LATENT, False, "one"),
    (8, H_MODEL, True, "two"),
    (1, H_MODEL, False, "two"),
    (16, H_LATENT, True, "one"),
    (64, H_LATENT, False, "two"),
    (5, H_MODEL, True, "k3"),
    (32, H_MODEL, False, "k3"),
)

R = None
fusion_allreduce = None
required_buffer_bytes = None
allgather_split = None
allreduce_attn_res = None
MnnvlWorkspace = None
BUFFER_BYTES = None
WS_A = None
WS_B = None
WS_C = None  # created by the failure-model check after the refused creates
ROT = {}
STATS = {"normed_err": 0.0, "normed_paths": 0.0, "attn_res_err": 0.0}


def buffer_bytes(world: int) -> int:
    """One Lamport buffer holds the largest certified call sent one-shot, [64, 7168]; every other call is smaller."""
    return max(TOKENS) * H_MODEL * world * 2


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
    """The call's one_shot_max_bytes: at the boundary for "one" (the call's one-shot footprint itself, so one-shot)
    and "two" (one byte less, so two-shot), or Kimi K3's ceiling for a step of ``tokens`` ("k3")."""
    if path == "one":
        return one_shot_bytes(tokens, hidden)
    if path == "two":
        return one_shot_bytes(tokens, hidden) - 1
    return DECODE_ONE_SHOT_MAX_BYTES if tokens <= DECODE_MAX_TOKENS else WIDE_ONE_SHOT_MAX_BYTES


def k3_split():
    """Kimi K3's MoE head shard at this world size: (latent columns, router-logit columns) per rank."""
    return H_LATENT // R.world, EXPERTS // R.world


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


class AR:
    """One mnnvl_fusion_allreduce call's arguments on every rank and its reference. ``residual``: a tensor (chained
    from an earlier call), True (drawn) or None (the plain sum). ``path``: see ``ceiling``. Every rank's input holds
    -0.0 in every NEG_ZERO_STRIDE-th column."""

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
        """A call of the same shape, fusion and path with new inputs."""
        return AR(
            seed, self.tokens, self.hidden, residual=True if self.fused else None, path=self.path
        )

    def chain_from(self, updated: torch.Tensor) -> None:
        """Take an earlier fused call's ``updated`` as this fused call's residual (as a layer's residual stream)."""
        assert self.fused
        self.residual = updated

    def run(self, ws, x=None, residual=None):
        x = self.inputs[R.rank] if x is None else x
        if not self.fused:
            return fusion_allreduce(x, ws, self.one_shot_max_bytes)
        residual = self.residual if residual is None else residual
        return fusion_allreduce(x, ws, self.one_shot_max_bytes, residual, self.gamma, EPS)

    def ref(self):
        """The exact sum (+0.0 where every rank sent -0.0); with a residual ``(normed, updated)``: ``updated`` exact,
        ``normed`` the fp32 RMSNorm of it."""
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
        """What the call leaves in buffer_flags: (stages, bytes it wrote into each)."""
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
        """Take ``fresh``'s inputs (same shape) into this call and its static buffers."""
        self.inputs = fresh.inputs
        static["x"].copy_(fresh.inputs[R.rank])
        if self.fused:
            self.residual = fresh.residual
            static["residual"].copy_(fresh.residual)


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
    """One comm/mnnvl_allgather_split call's arguments on every rank and its reference."""

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
        written = self.tokens * R.world * (2 * self.bf16_columns + 4 * self.fp32_columns)
        return 1, (written, 0, 0, 0)

    def static(self):
        return {"x": self.inputs[R.rank].clone()}

    def refill(self, fresh, static) -> None:
        self.inputs = fresh.inputs
        static["x"].copy_(fresh.inputs[R.rank])


class AttnRes:
    """One comm/mnnvl_allreduce_attn_res call (the entry's wrapper, on the same workspace) and its reference.
    ``prefix``: a tensor (chained) or True (drawn)."""

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

    def fresh(self, seed):
        return AttnRes(seed, self.tokens, self.snapshots)

    def chain_from(self, updated: torch.Tensor) -> None:
        """Take an earlier call's ``updated`` as this call's prefix sum (as the next layer does)."""
        self.prefix = updated

    def run(self, ws, x=None, prefix=None, block=None):
        return allreduce_attn_res(
            self.inputs[R.rank] if x is None else x,
            self.prefix if prefix is None else prefix,
            self.block if block is None else block,
            self.res_w,
            self.rms_w,
            self.out_w,
            EPS,
            EPS,
            ws,
        )

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

    def static(self):
        return {
            "x": self.inputs[R.rank].clone(),
            "prefix": self.prefix.clone(),
            "block": self.block.clone(),
        }

    def refill(self, fresh, static) -> None:
        self.inputs, self.prefix, self.block = fresh.inputs, fresh.prefix, fresh.block
        static["x"].copy_(fresh.inputs[R.rank])
        static["prefix"].copy_(fresh.prefix)
        static["block"].copy_(fresh.block)


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
    clear), and one buffer holds the largest certified call sent one-shot and every other call of this matrix."""
    for ws in (WS_A, WS_B):
        assert ws.world_size == R.world and ws.rank == R.rank
        assert ws.buffer_bytes == BUFFER_BYTES and BUFFER_BYTES % 32 == 0
        assert ws.comm_buffer(torch.bfloat16).shape == (3, BUFFER_BYTES // 2)
        armed = ws.lamport.view(torch.int32)
        assert bool((armed == torch.tensor(INT32_MIN, dtype=torch.int32, device="cuda")).all()), (
            "every word -0.0"
        )
        rot(ws).unchanged("armed")
    one_shot = required_buffer_bytes(max(TOKENS), H_MODEL, R.world, torch.bfloat16, 1 << 62)
    assert one_shot == BUFFER_BYTES, f"the largest one-shot call needs {one_shot} bytes"
    b, f = k3_split()
    assert max(TOKENS) * R.world * (2 * b + 4 * f) <= BUFFER_BYTES
    assert ATTN_RES_MAX_TOKENS * H_MODEL * R.world * 2 <= BUFFER_BYTES


def check_single_calls_both_paths() -> None:
    """Every certified shape, plain and fused, sent one-shot with one_shot_max_bytes = its one-shot footprint (the
    boundary is inclusive) and right after two-shot with one byte less, on one workspace, the same inputs both
    times: each against the reference, the flags after each recording the path and the bytes the formula says, and
    required_buffer_bytes equal to the space the call's stages take."""
    for hidden in HIDDENS:
        for t in TOKENS:
            for fused in (False, True):
                outs = []
                for path in ("one", "two"):
                    call = AR(
                        1000 + 37 * t + 3 * hidden + int(fused),
                        t,
                        hidden,
                        residual=fused or None,
                        path=path,
                    )
                    where = f"T {t} H {hidden} fused {fused} {path}-shot"
                    assert call.one_shot == (path == "one"), where
                    outs.append(call_and_check(call, WS_A, where))
                    stages, written = call.record()
                    need = required_buffer_bytes(
                        t, hidden, R.world, torch.bfloat16, call.one_shot_max_bytes
                    )
                    assert need == (written[0] if stages == 1 else 2 * written[0]), (
                        f"{where}: needs {need}"
                    )
                if fused:
                    STATS["normed_paths"] = max(
                        STATS["normed_paths"], ls.rel_err(outs[0][0], outs[1][0])
                    )


def check_unsupported_calls_raise_on_every_rank() -> None:
    """Refused on every rank before the workspace is touched (its flags do not move), and the next call is correct:
    more tokens than one Lamport buffer holds one-shot (the wrapper's ValueError; the same rows sent two-shot fit,
    above 2 ranks), a residual without norm_weight and eps (the wrapper's ValueError), a hidden size not a multiple
    of 8 (the op's RuntimeError)."""
    t = max(TOKENS) + 1
    over = AR(2000, t, H_MODEL, path="one")
    expect_refusal(lambda: over.run(WS_A), ValueError, f"T {t} one-shot over one buffer")
    if R.world > 2:
        # Two-shot takes about 2 / W of the one-shot space.
        call_and_check(AR(2000, t, H_MODEL, path="two"), WS_A, f"T {t} sent two-shot")
    pair = AR(2001, 4, H_LATENT, residual=True)
    expect_refusal(
        lambda: fusion_allreduce(pair.inputs[R.rank], WS_A, pair.one_shot_max_bytes, pair.residual),
        ValueError,
        "a residual without norm_weight and eps",
    )
    odd = AR(2002, 2, H_LATENT - 4)
    expect_refusal(lambda: odd.run(WS_A), RuntimeError, f"hidden {H_LATENT - 4}")
    call_and_check(AR(2003, 8, H_MODEL, residual=True), WS_A, "after the refused calls")


def run_step(ws, seed, tokens, late):
    """One decode step of LAYERS layers, a random rank late at every call: per layer the routed-latent all-reduce
    [T, 3584] and the fused all-reduce [T, 7168] chained through ``updated``, both at Kimi K3's ceiling."""
    residual = ls.exact_bf16(
        torch.Generator(device="cuda").manual_seed(seed), (tokens, H_MODEL), -32, 33, 1 / 16
    )
    for layer in range(LAYERS):
        where = f"step seed {seed} T {tokens} layer {layer}"
        call_late(AR(seed + 2 * layer + 1, tokens, H_LATENT), ws, f"{where} latent", late)
        fused = AR(seed + 2 * layer + 2, tokens, H_MODEL, residual=residual)
        residual = call_late(fused, ws, f"{where} fused", late)[1]


def check_dip_and_regrow_sequence() -> None:
    """The k3_spec_accept failure mode: a call after a smaller one must not read what an older, larger call left.
    16 steps of 8 layers at T 8, 8, 8, 2, 7, 8, 1, 1, 64, 3, 32, 8, 16, 1, 64, 8 at Kimi K3's ceilings, so the path
    flips inside the sequence (at W = 4 [64, 3584] and [32 or 64, 7168] go two-shot, at W = 16 every step above 8
    tokens), a random rank late at every call."""
    late = random.Random(7)
    for i, t in enumerate(DIP_STEPS):
        run_step(WS_A, 3000 + 100 * i, t, late)


def check_two_workspaces_interleaved() -> None:
    """Two workspaces are two rotations: calls of mixed shapes and paths alternate between them in an irregular
    pattern (A A B A B B ...), every call is correct and each workspace's flags move with its own calls only. The
    pattern is the same on every rank: calls on one stream are serialized and each waits for its peers, so ranks
    issuing calls on two workspaces in different orders deadlock (seen with mnnvl_allreduce_attn_res)."""
    pattern = "AABABBAAAB" * 2
    for i, which in enumerate(pattern):
        t, hidden, fused, path = INTERLEAVED[i % len(INTERLEAVED)]
        call = AR(4700 + i, t, hidden, residual=fused or None, path=path)
        call_and_check(call, WS_A if which == "A" else WS_B, f"interleaved {which} {i}")


def check_one_workspace_three_ops() -> None:
    """The three MNNVL entries on one workspace in Kimi K3's order, a random rank late at every call. A step of at
    most 16 tokens: per layer the pre-attention all-reduce with the attention-residual epilogue
    (comm/mnnvl_allreduce_attn_res, prefix chained), the MoE head all-gather (comm/mnnvl_allgather_split, K3's
    split), the routed-latent all-reduce and the fused all-reduce (chained). A wide step (32, 64 tokens): the wide
    all-reduce [T, 7168], the all-gather and the routed-latent all-reduce, at the 1 MiB ceiling. Every call takes one
    turn of the rotation whatever the op (the flags after each), and every result is right."""
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
            call_late(AR(s + 2, t, H_LATENT), WS_A, f"{where} latent", late)
            if t <= ATTN_RES_MAX_TOKENS:
                fused = AR(s + 3, t, H_MODEL, residual=residual)
                residual = call_late(fused, WS_A, f"{where} fused", late)[1]


def queued_plan(seed: int):
    """One round of the queued check, in issue order: per step of QUEUED_STEPS, two layers of Kimi K3's calls -- a
    step of at most 16 tokens: the attention-residual all-reduce, the head all-gather, the latent all-reduce and the
    fused all-reduce, the two all-reduces on opposite paths, alternating from layer to layer; a wide step: the wide
    all-reduce [T, 7168], the all-gather and the latent all-reduce at Kimi K3's ceiling. The second layer's prefix sum
    and residual are the first layer's outputs. Returns ``[(call, index of the call it chains from, or None)]``."""
    b, f = k3_split()
    plan = []
    for i, t in enumerate(QUEUED_STEPS):
        decode = t <= ATTN_RES_MAX_TOKENS
        prev_attn = prev_fused = None
        for layer in range(2):
            s = seed + 100 * i + 10 * layer
            one, two = ("one", "two") if (i + layer) % 2 == 0 else ("two", "one")
            if decode:
                plan.append((AttnRes(s, t, (0, 2, 5)[(i + layer) % 3]), prev_attn))
                prev_attn = len(plan) - 1
            else:
                plan.append((AR(s, t, H_MODEL), None))
            plan.append((AG(s + 1, t, b, f), None))
            plan.append((AR(s + 2, t, H_LATENT, path=one if decode else "k3"), None))
            if decode:
                plan.append((AR(s + 3, t, H_MODEL, residual=True, path=two), prev_fused))
                prev_fused = len(plan) - 1
    return plan


def check_queued_calls_without_host_sync() -> None:
    """Ranks running ahead of one another: in each round every rank enqueues the same 52 calls on one workspace (the
    three ops, both all-reduce paths, Kimi K3's step sizes dipping and growing back, a layer's prefix sum and residual
    taken from the previous layer's outputs on the device) with no host synchronization between them -- no barrier,
    no .item(), no synchronize -- a random rank sleeping 20 ms before it starts enqueueing and another pausing 20 ms
    halfway, so the other ranks' queues run deep while it is idle. Then every result is checked, and the flags. This
    cannot deadlock: a call waits only for its peers' words of the same call, every rank enqueues every call, and a
    stream runs its calls in order, so the ranks' earliest pending call always completes."""
    late = random.Random(13)
    for rnd in range(QUEUED_ROUNDS):
        plan = queued_plan(60_000 + 1000 * rnd)
        sleeper, pauser = late.randrange(R.world), late.randrange(R.world)
        R.barrier()
        R.late(sleeper, QUEUE_LATE_S)
        outs = []
        for k, (call, src) in enumerate(plan):
            if k == len(plan) // 2:
                R.late(pauser, QUEUE_LATE_S)
            if src is not None:
                call.chain_from(outs[src][1])
            outs.append(call.run(WS_A))
        torch.cuda.synchronize()
        for k, ((call, _), got) in enumerate(zip(plan, outs)):
            call.verify(got, f"queued round {rnd} call {k}")
        rot(WS_A).advance(plan[-1][0].record(), f"queued round {rnd}", calls=len(plan))


def check_graph_capture_and_replay() -> None:
    """A captured step of six calls on WS_B: the attention-residual all-reduce, the plain and fused all-reduces
    one-shot (T 8, Kimi K3's ceiling), the head all-gather, a plain [32, 7168] and a fused [16, 7168] sent two-shot
    (the fused two-shot call is two kernels, the exchange and the residual + RMSNorm one; both are in the graph).
    Replayed 8 times with rewritten inputs, an eager call of another shape or op on the same workspace between
    replays: replays and eager calls take turns of one rotation (the flags after each), in the same order on every
    rank, and every replayed and eager result is right."""
    b, f = k3_split()
    calls = [
        AttnRes(6000, 8, 2),
        AR(6001, 8, H_LATENT),
        AG(6002, 8, b, f),
        AR(6003, 8, H_MODEL, residual=True),
        AR(6004, 32, H_MODEL, path="two"),
        AR(6005, 16, H_MODEL, residual=True, path="two"),
    ]
    static_bufs = [c.static() for c in calls]

    def step():
        return [c.run(WS_B, **s) for c, s in zip(calls, static_bufs)]

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
        lambda seed: AR(seed, 3, H_LATENT, path="one"),
        lambda seed: AR(seed, 64, H_MODEL, residual=True, path="two"),
        lambda seed: AG(seed, 5, b, f),
        lambda seed: AR(seed, 64, H_LATENT, path="one"),
    )
    for rep in range(8):
        for i, (c, s) in enumerate(zip(calls, static_bufs)):
            c.refill(c.fresh(7000 + 100 * rep + i), s)
        R.barrier()
        graph.replay()
        for i, (c, got) in enumerate(zip(calls, outs)):
            c.verify(got, f"replay {rep} call {i}")
        rot(WS_B).advance(calls[-1].record(), f"replay {rep}", calls=len(calls))
        call_and_check(eager[rep % len(eager)](7500 + rep), WS_B, f"eager after replay {rep}")
    del graph


def try_create(capturing: bool):
    """Every rank calls MnnvlWorkspace.create; this one inside a CUDA-graph capture if ``capturing``. Returns the
    RuntimeError's message, or None if it returned a workspace."""
    try:
        if not capturing:
            MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
            return None
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        try:
            with torch.cuda.graph(graph, stream=stream):
                MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        finally:
            del graph
        return None
    except RuntimeError as exc:
        return str(exc)


def check_create_refuses_on_every_rank() -> None:
    """MnnvlWorkspace.create's failure model: before allocating, the ranks agree that each of them can. With one
    random rank inside a CUDA-graph capture and the others not, and then with every rank capturing, every rank raises
    RuntimeError and none allocates; the capturing ranks' message names the capture. A create right after, eager on
    every rank, returns an armed workspace whose first call is correct, and the workspaces in use are untouched."""
    global WS_C
    capturer = random.Random(17).randrange(R.world)
    for every in (False, True):
        capturing = every or R.rank == capturer
        message = try_create(capturing)
        refused = message is not None and "not every rank can allocate" in message
        named = not capturing or (refused and "before capture" in message)
        assert R.all_true(refused and named), (
            f"{'every rank' if every else f'rank {capturer}'} capturing: not refused on every rank ({message})"
        )
        rot(WS_A).unchanged("after the refused create")
    WS_C = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
    ROT[id(WS_C)] = Rotation(WS_C)
    armed = WS_C.lamport.view(torch.int32)
    assert bool((armed == torch.tensor(INT32_MIN, dtype=torch.int32, device="cuda")).all()), (
        "the workspace created after the refusals: every word -0.0"
    )
    rot(WS_C).unchanged("created after the refusals")
    call_and_check(AR(8500, 8, H_MODEL, residual=True), WS_C, "first call on the new workspace")
    call_and_check(AR(8501, 8, H_MODEL, residual=True, path="two"), WS_A, "after the refusals")


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 issues two same-shaped calls on one workspace in swapped order, once sent one-shot
    and once two-shot. Nothing raises and nothing hangs (the two calls write the same slots of the same buffers), but
    every rank's two results are wrong, and exactly the position-paired sums: the k-th call on every rank adds what
    every rank sent at position k. A plain call right after is correct again: a swapped pair realigns the
    positions."""
    for path in ("one", "two"):
        c1 = AR(9000, 8, H_LATENT, path=path)
        c2 = AR(9001, 8, H_LATENT, path=path)
        R.barrier()
        if R.rank == 0:
            got2, got1 = c2.run(WS_A), c1.run(WS_A)
        else:
            got1, got2 = c1.run(WS_A), c2.run(WS_A)
        torch.cuda.synchronize()
        # Position 1 summed rank 0's c2 with the others' c1, position 2 rank 0's c1 with the others' c2.
        first = with_inputs(c1, [c2.inputs[0]] + c1.inputs[1:]).ref()
        second = with_inputs(c2, [c1.inputs[0]] + c2.inputs[1:]).ref()
        at1, at2 = (got2, got1) if R.rank == 0 else (got1, got2)
        assert bits_equal(at1, first) and bits_equal(at2, second), (
            f"{path}-shot: not the position-paired sums"
        )
        wrong = [
            (bits(got) != bits(c.ref())).float().mean().item()
            for c, got in ((c1, got1), (c2, got2))
        ]
        assert R.all_true(min(wrong) > 0.5), (
            f"{path}-shot: the swap went unnoticed: wrong fractions {wrong}"
        )
        rot(WS_A).advance(c2.record(), f"{path}-shot swapped pair", calls=2)
        call_and_check(
            AR(9002, 8, H_LATENT, path=path), WS_A, f"{path}-shot after the swapped pair"
        )


CHECKS = [
    check_workspaces_are_armed_and_sized,
    check_single_calls_both_paths,
    check_unsupported_calls_raise_on_every_rank,
    check_dip_and_regrow_sequence,
    check_two_workspaces_interleaved,
    check_one_workspace_three_ops,
    check_queued_calls_without_host_sync,
    check_graph_capture_and_replay,
    check_create_refuses_on_every_rank,
    # Stays last: it deliberately disagrees on call order.
    check_wrong_call_order_is_detected,
]


def _run_one_rank(args) -> int:
    global R, fusion_allreduce, required_buffer_bytes, allgather_split, allreduce_attn_res
    global MnnvlWorkspace, BUFFER_BYTES, WS_A, WS_B
    R = ls.Rank(args)
    assert R.world in (2, 4, 8, 16), f"K3's shapes shard over 2, 4, 8 or 16 ranks, not {R.world}"
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_allgather_split as gather_module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_allreduce_attn_res as attn_res_module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_fusion_allreduce as module,
    )

    fusion_allreduce = module.mnnvl_fusion_allreduce
    required_buffer_bytes = module.required_buffer_bytes
    allgather_split = gather_module.mnnvl_allgather_split
    allreduce_attn_res = attn_res_module.mnnvl_allreduce_attn_res
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
            f"[rank 0] world {R.world}; buffer {BUFFER_BYTES} B; max normed rel err {STATS['normed_err']:.3e}; "
            f"max normed one-shot vs two-shot {STATS['normed_paths']:.3e}; "
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
