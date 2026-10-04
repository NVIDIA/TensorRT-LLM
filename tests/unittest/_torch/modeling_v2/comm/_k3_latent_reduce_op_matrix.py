# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/k3_latent_reduce`` catalog entry and its ``K3LatentExchange``.

The op is the consumer half of Kimi K3's latent all-reduce at decode size: every rank's producer pushes its routed
partial into the exchange, and the op, on every rank, sums the ranks' rows. Its correctness depends on state that
outlives a call (the call count in ``flags[0]``, whose parity picks the half every push and reduce use, and the words
a reduce empties for the push two calls later), so beyond single calls this drives call *sequences*: layers x steps
with the token count dipping and growing back and a random rank late, ranks a whole step apart, two exchanges
interleaved, CUDA-graph capture and replay mixed with eager calls, the count across its int32 wrap, and two negative
controls in which ranks break the call order and get a wrong answer without an error.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_latent_reduce_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _k3_latent_reduce_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the exchanges and their counts).
The collected entry point is ``test_modeling_v2_k3_latent_reduce_op_matrix.py``.

The producers (the push-only routed-expert kernels) are not part of this entry, so every push is emulated as they
store it: this rank's partial rows, -0.0 as +0.0, as int32 words (bf16 pairs) copied through the multicast mapping
into slot [rank] of the call's half of every rank's buffer. The emulation takes the half from the host's count of the
exchange's calls, which ``assert_clean`` checks against ``flags[0]``; a producer reads ``flags[0]`` on the device.
The copies launch without programmatic dependent launch, so the op's PDL condition (every kernel between a reduce
and the next push ends only after its predecessor has ended) holds throughout.

Every rank draws every rank's partial from one seed, so each holds the whole reference: the sum in the MNNVL
one-shot's order (fp32 over chunks of 8 ranks in rank order, each from +0, the chunks added in order, then bf16),
computed with torch. Most checks use multiples of 1/16, whose sum is exact in any order. The bit-identity check adds
partials whose sum depends on the order (per element a +B / -B pair that absorbs the small values summed beside it;
another order is shown to change at least a quarter of the elements) and compares the op bit for bit with that
reference and with the MNNVL one-shot all-reduce itself (``comm/mnnvl_fusion_allreduce`` over an ``MnnvlWorkspace``).
Every output is also compared bitwise across the ranks.
"""

import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_latent_reduce requires CUDA devices"

DEADLINE_S = 900
H = 3584  # the routed latent: the all-reduce's row width
ROW_WORDS = H // 2  # int32 words (bf16 pairs) of one row
MAX_TOKENS = 8
RANK_CHUNK = 8  # the one-shot sums the ranks in chunks of 8
EMPTY_WORD = -(2**31)  # 0x80000000: a word no producer has written
NEG_ZERO = -(2**15)  # bf16 -0.0 as int16, which the producers store as +0.0
WORLDS = (4, 8, 16)  # the op's TP sizes
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
LAYERS = 12  # even: a captured step keeps the halves' parity (check_graph_capture_and_replay)
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8)
# One step of LAYERS calls whose M dips and grows back (check_ranks_run_a_step_apart).
RUN_AHEAD_TOKENS = (8, 3, 8, 1, 5, 8, 2, 8, 7, 4, 8, 6)
GRAPH_TOKENS = (8, 3)  # one captured step per batch size, as an engine keeps one graph per size
EAGER_BETWEEN = (5, 1)  # an even number of eager calls between two replays
REPLAYS = 4
MNNVL_BUFFER_BYTES = 1 << 20  # one Lamport buffer: 8 rows of 3584 bf16 from 16 ranks go one-shot
# The least share of the elements that another summation order must change in the "cancel" partials.
ORDER_GUARD = 0.25
WRAP_PRESET = 2**31 - 3

R = None
entry = None
create_exchange = None
mnnvl = None
EX_A = None
EX_B = None
MNNVL_WS = None
STATS = {"order_guard": 1.0}


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def differing(a: torch.Tensor, b: torch.Tensor) -> float:
    """The share of the elements whose bits differ (same shapes)."""
    return (bits(a) != bits(b)).float().mean().item()


def int32(n: int) -> int:
    """``n`` as the int32 count holds it (two's complement wrap)."""
    return (n + 2**31) % 2**32 - 2**31


def ordered_sum(parts: Sequence[torch.Tensor], chunk: int) -> torch.Tensor:
    """fp32 sums of ``chunk`` consecutive partials each, in the given order and from +0; those sums added in order to
    +0; then bf16 (round to nearest even). With ``chunk`` = 8 and the ranks in order, the MNNVL one-shot's order."""
    total = torch.zeros(parts[0].shape, dtype=torch.float32, device=parts[0].device)
    for base in range(0, len(parts), chunk):
        acc = torch.zeros_like(total)
        for part in parts[base : base + chunk]:
            acc = acc + part.float()
        total = total + acc
    return total.bfloat16()


def exact_partials(g: torch.Generator, tokens: int) -> List[torch.Tensor]:
    """Multiples of 1/16 in [-1/4, 1/4]: a sum of up to 16 of them is exact in fp32 and bf16, whatever the order."""
    return [ls.exact_bf16(g, (tokens, H), -4, 5, 1 / 16) for _ in range(R.world)]


def randn_partials(g: torch.Generator, tokens: int) -> List[torch.Tensor]:
    return [
        (torch.randn(tokens, H, generator=g, device="cuda") * 0.5).bfloat16()
        for _ in range(R.world)
    ]


def cancel_partials(g: torch.Generator, tokens: int) -> List[torch.Tensor]:
    """Small values (multiples of 1/8, at most 31/8 in magnitude) on every rank, then per element +B on one rank and
    -B on another, B = 2^30 (1 + k/128) with k in 1..127. Half an fp32 ulp at B is 64, more than any sum of the small
    values (at most 14 of them), so a running fp32 sum that holds B absorbs every small value added to it, and the
    pair cancels exactly: the result depends on where the pair falls in the order of summation, and another order
    gives another result in most elements."""
    shape = (tokens, H)
    parts = torch.stack(
        [
            (torch.randint(-31, 32, shape, generator=g, device="cuda") / 8).bfloat16()
            for _ in range(R.world)
        ]
    )
    big = (128 + torch.randint(1, 128, shape, generator=g, device="cuda")).float() * 2.0**23
    sign = torch.randint(0, 2, shape, generator=g, device="cuda").float() * 2 - 1
    first = torch.randint(0, R.world, shape, generator=g, device="cuda")
    second = (first + torch.randint(1, R.world, shape, generator=g, device="cuda")) % R.world
    parts.scatter_(0, first.unsqueeze(0), (sign * big).bfloat16().unsqueeze(0))
    parts.scatter_(0, second.unsqueeze(0), (-sign * big).bfloat16().unsqueeze(0))
    return list(parts.unbind(0))


PARTIALS = {"exact": exact_partials, "randn": randn_partials, "cancel": cancel_partials}


class Exchange:
    """A ``K3LatentExchange`` with the host's count of its calls, from which the emulated pushes take their half."""

    def __init__(self, state):
        self.state = state
        self.calls = 0

    def push(self, partial: torch.Tensor, half: Optional[int] = None) -> None:
        """What a push-only producer stores: this rank's rows, -0.0 as +0.0, as int32 words into slot [rank] of
        ``half`` (default: this call's, the count's parity) of every rank's buffer, through the multicast mapping."""
        rows = partial.shape[0]
        words = partial.contiguous().view(torch.int16)
        words = words.masked_fill(words == NEG_ZERO, 0).view(torch.int32)
        dest = self.state.mc.view(2, MAX_TOKENS, R.world, ROW_WORDS)
        dest[self.calls & 1 if half is None else half, :rows, R.rank].copy_(words)

    def reduce(self, tokens: int) -> torch.Tensor:
        out = entry(tokens, self.state)
        self.calls += 1
        return out


class Call:
    """One push + reduce of ``tokens`` rows: every rank's partial (each rank draws them all from one seed, so each
    holds the whole reference) and the reference. Every partial has -0.0 entries (pushed as +0.0), and in some columns
    every rank's is -0.0, whose sum is +0.0."""

    def __init__(self, seed: int, tokens: int, kind: str = "exact"):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.tokens = tokens
        self.parts = PARTIALS[kind](g, tokens)
        for r, part in enumerate(self.parts):
            part[:, r::97] = -0.0
            part[:, 5::101] = -0.0

    def ref(self) -> torch.Tensor:
        """The MNNVL one-shot's order (``reduceOneshotLamport``), which the kernel states it follows."""
        return ordered_sum(self.parts, RANK_CHUNK)

    def run(self, ex: Exchange) -> torch.Tensor:
        ex.push(self.parts[R.rank])
        return ex.reduce(self.tokens)


def verify(call: Call, got: torch.Tensor, where: str) -> None:
    want = call.ref()
    assert got.dtype == torch.bfloat16 and got.is_contiguous(), f"{where}: {got.dtype} output"
    assert got.shape == want.shape, (
        f"{where}: shape {tuple(got.shape)}, expected {tuple(want.shape)}"
    )
    assert same_bits(got, want), f"{where}: {differing(got, want):.2%} of the elements differ"
    assert R.same_on_ranks(got), f"{where}: ranks disagree"


def assert_clean(ex: Exchange, where: str) -> None:
    """Every rank's last reduce on ``ex`` has ended and no rank has pushed the next call: both halves empty and flags
    [count, 0, 0, 0] (the count, the arrivals word back to 0, two unused words) on every rank."""
    R.barrier()
    flags = ex.state.flags.tolist()
    left = int((ex.state.uc != EMPTY_WORD).sum())
    expected = [int32(ex.calls), 0, 0, 0]
    # The allgather is also the barrier that keeps every rank from pushing until every rank has looked.
    assert R.all_true(left == 0 and flags == expected), (
        f"{where}: rank {R.rank} flags {flags} (expected {expected}), {left} words not empty"
    )


def raised_under_capture(fn) -> str:
    """Run ``fn`` under CUDA-graph capture; the message of the RuntimeError it raised, or '' if it raised none."""
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    resting = torch.cuda.current_stream()
    stream.wait_stream(resting)
    message = ""
    try:
        with torch.cuda.graph(graph, stream=stream):
            try:
                fn()
            except RuntimeError as exc:
                message = str(exc)
    finally:
        # A capture that fails when it ends leaves its own stream current; put the resting one back.
        torch.cuda.set_stream(resting)
    del graph
    return message


def check_exchange_is_armed_and_sized() -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import k3_latent_reduce as kernel

    words = 2 * MAX_TOKENS * R.world * ROW_WORDS
    assert kernel.buffer_words(R.world) == words
    for ex in (EX_A, EX_B):
        s = ex.state
        assert (s.rank, s.world_size) == (R.rank, R.world)
        assert s.uc.dtype == s.mc.dtype == s.flags.dtype == torch.int32
        assert s.uc.numel() == s.mc.numel() == words, f"{s.uc.numel()} words, expected {words}"
        assert s.uc.device.index == torch.cuda.current_device() and s.flags.device == s.uc.device
        assert bool((s.uc == EMPTY_WORD).all()), "every word empty (0x80000000)"
        assert s.flags.tolist() == [0, 0, 0, 0], f"flags {s.flags.tolist()}"
        uc, mc, flags, rank = s.push_args()
        assert uc is s.uc and mc is s.mc and flags is s.flags and rank == R.rank
    assert EX_A.state.uc.data_ptr() != EX_B.state.uc.data_ptr(), "two exchanges, two buffers"
    assert EX_A.state.flags.data_ptr() != EX_B.state.flags.data_ptr(), "two exchanges, two counts"


def check_create_needs_only_the_tp_group() -> None:
    """Under MPI a create() makes its communicator from its TP group's ranks alone (the helper every Kimi K3 state's
    create() uses). With the job split into two TP groups of W / 2 (pipeline parallel 2), the first group's ranks make
    theirs while the second group's ranks do not call at all; each communicator holds exactly its group, in TP-rank
    order. The exchanges in use also hold their TP group's communicator."""
    from tensorrt_llm._torch.distributed import ops
    from tensorrt_llm.mapping import Mapping

    for ex in (EX_A, EX_B):
        assert ex.state.comm.Get_size() == R.world and ex.state.comm.Get_rank() == R.rank
    half = Mapping(
        world_size=R.world,
        rank=R.rank,
        gpus_per_node=R.mapping.gpus_per_node,
        tp_size=R.world // 2,
        pp_size=2,
    )
    R.barrier()
    ok = True
    if half.pp_rank == 0:
        comm = ops._get_mnnvl_tp_group_comm(half)
        ok = (
            comm.Get_size() == half.tp_size
            and comm.Get_rank() == half.tp_rank
            and comm.allgather(R.rank) == list(half.tp_group)
        )
        comm.Free()
    R.comm.Barrier()
    assert R.all_true(ok), (
        f"rank {R.rank}: the first TP group's communicator is not exactly that group"
    )


def check_capture_refusals() -> None:
    """``K3LatentExchange.create`` is collective: every rank joins the TP group's communicator, and before allocating
    the ranks agree that each of them can. With every rank capturing a CUDA graph, and with one rank capturing while its
    peers call it eagerly at the same point, every rank raises RuntimeError, and a capturing rank's message names the
    capture. Every rank raises at that agreement ("not every rank can allocate"; a failure after allocating reads
    "allocation failed"), so nothing is allocated, and each frees the communicator made for it. The op's first call,
    which would compile the kernel, is refused per rank: under capture it raises on every rank before it launches
    anything. The exchange is untouched and the next call is correct. Runs before every eager call of the op: the
    compile cache must still be cold."""
    # Imported by the op's first call; imported here so that nothing is imported inside the capture.
    import cutlass.cute.runtime  # noqa: F401

    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import k3_latent_reduce  # noqa: F401

    def create() -> None:
        create_exchange(R.mapping, fabric_handle=R.fabric)

    def refusal(capturing: bool) -> str:
        """The message of the RuntimeError ``create`` raised on this rank ('' if it raised none)."""
        if capturing:
            return raised_under_capture(create)
        try:
            create()
        except RuntimeError as exc:
            return str(exc)
        return ""

    from tensorrt_llm._torch.distributed import ops

    make = ops._get_mnnvl_tp_group_comm
    comms = []

    def recording_make(mapping):
        comms.append(make(mapping))
        return comms[-1]

    ops._get_mnnvl_tp_group_comm = recording_make
    try:
        for case, capturing in (("every rank", True), ("one rank", R.rank == R.world - 1)):
            R.barrier()
            message = refusal(capturing)
            refused = "not every rank can allocate" in message
            named = "outside CUDA-graph capture" in message or not capturing
            assert R.all_true(refused and named), (
                f"{case} capturing: rank {R.rank} (capturing {capturing}) got {message!r}"
            )
    finally:
        ops._get_mnnvl_tp_group_comm = make
    freed = len(comms) == 2 and all(c == R.MPI.COMM_NULL for c in comms)
    assert R.all_true(freed), "a refused create kept the communicator made for it"
    R.barrier()
    first = raised_under_capture(lambda: entry(MAX_TOKENS, EX_A.state))
    assert R.all_true("outside CUDA-graph capture first" in first), f"first call: {first!r}"
    assert_clean(EX_A, "after the refused calls")
    call = Call(900, MAX_TOKENS)
    verify(call, call.run(EX_A), "after the refused calls")


def check_single_calls() -> None:
    """One push + reduce at every token count 1-8, exact partials: the reference bit for bit on every rank; after each
    call both halves are empty again, the count is up by one and the arrivals word is 0."""
    for t in TOKENS:
        call = Call(1000 + t, t)
        verify(call, call.run(EX_A), f"M {t}")
        assert_clean(EX_A, f"after M {t}")


def check_bit_identical_to_mnnvl_oneshot() -> None:
    """The op's statement: its rows are the MNNVL one-shot all-reduce of the partials, bit for bit. At every token
    count, with partials whose sum depends on the order (summing the ranks in reverse, or at W > 8 without chunks,
    changes at least ORDER_GUARD of the elements) and with normal-distributed ones, the op, ``mnnvl_fusion_allreduce``
    sent one-shot and the torch reference in the one-shot's order agree bit for bit."""
    for t in TOKENS:
        assert t * H * R.world * 2 <= MNNVL_BUFFER_BYTES, "the reference call must go one-shot"
        for k, kind in enumerate(("cancel", "randn")):
            call = Call(2000 + 10 * t + k, t, kind)
            want = call.ref()
            if kind == "cancel":
                others = [ordered_sum(call.parts[::-1], R.world)]
                if R.world > RANK_CHUNK:
                    others.append(ordered_sum(call.parts, R.world))
                share = min(differing(other, want) for other in others)
                STATS["order_guard"] = min(STATS["order_guard"], share)
                assert share >= ORDER_GUARD, f"M {t}: another order changes only {share:.2%}"
            oneshot = mnnvl(call.parts[R.rank], MNNVL_WS, MNNVL_BUFFER_BYTES)
            got = call.run(EX_A)
            assert same_bits(got, oneshot), f"M {t} {kind}: differs from the MNNVL one-shot"
            assert same_bits(oneshot, want), (
                f"M {t} {kind}: the MNNVL one-shot differs from the reference"
            )
            verify(call, got, f"M {t} {kind}")
    assert_clean(EX_A, "after the bit-identity calls")


def check_unsupported_token_counts_raise_on_every_rank() -> None:
    """Token counts 0 and 9 raise ValueError on every rank before the op touches the exchange (no producer pushes
    them: a half holds 8 rows); the count and the words are unchanged and the next call is correct."""
    for t in (0, MAX_TOKENS + 1):
        try:
            entry(t, EX_A.state)
            raised = False
        except ValueError:
            raised = True
        assert R.all_true(raised), f"M {t} did not raise ValueError on every rank"
    assert_clean(EX_A, "after the rejected calls")
    call = Call(3000, MAX_TOKENS)
    verify(call, call.run(EX_A), "after the rejected calls")


def run_step(ex: Exchange, seed: int, tokens: int, late: Optional[random.Random] = None) -> None:
    """One decode step: LAYERS push + reduce pairs queued back to back (no host synchronization between them, as in a
    step), a random rank late before each when ``late`` is given; every result checked after the step."""
    calls = [Call(seed + layer, tokens) for layer in range(LAYERS)]
    R.barrier()
    outs = []
    for call in calls:
        if late is not None:
            R.late(late.randrange(R.world))
        outs.append(call.run(ex))
    for layer, (call, got) in enumerate(zip(calls, outs)):
        verify(call, got, f"step seed {seed} M {tokens} layer {layer}")


def check_dip_and_regrow_sequence() -> None:
    """Decode steps of LAYERS layers at M 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, a random rank 5 ms late before every call
    (its push lands while the others' reduces poll): a call after a smaller one must not read words an older, larger
    call left. After every step both halves are empty and the count is right on every rank."""
    late = random.Random(7)
    for i, t in enumerate(DIP_STEPS):
        run_step(EX_A, 10_000 + 100 * i, t, late)
        assert_clean(EX_A, f"after step {i} (M {t})")


def check_ranks_run_a_step_apart() -> None:
    """Calls queued across ranks. Rank 0 enqueues a whole step (LAYERS push + reduce pairs, M dipping and growing
    back) and waits until its first push has completed, while its peers wait at a host barrier; only then do they
    start. Before issuing anything, each peer finds rank 0's first push in its buffer and nothing of its second (the
    other half is empty): a push is issued after its rank's previous reduce, which waits for every rank's push of its
    call, so however far a rank's host runs ahead, its GPU runs at most one call ahead. Then the same with every rank
    but the last a step ahead of the last. Every result is correct and the exchange ends clean."""
    ex = EX_A
    sizes = RUN_AHEAD_TOKENS
    for phase, ahead in enumerate((range(1), range(R.world - 1))):
        calls = [Call(90_000 + 100 * phase + layer, t) for layer, t in enumerate(sizes)]
        half = ex.calls & 1
        R.barrier()
        outs = []
        seen = (True, True, True)
        if R.rank in ahead:
            pushed = torch.cuda.Event()
            for layer, call in enumerate(calls):
                ex.push(call.parts[R.rank])
                if layer == 0:
                    pushed.record()
                outs.append(ex.reduce(call.tokens))
            # The event, not the stream: this rank's first reduce waits for the peers' pushes.
            pushed.synchronize()
            R.comm.Barrier()
        else:
            R.comm.Barrier()
            rows = ex.state.uc.view(2, MAX_TOKENS, R.world, ROW_WORDS)
            seen = (
                all(bool((rows[half, : sizes[0], a] != EMPTY_WORD).all()) for a in ahead),
                bool((rows[half, :, R.rank] == EMPTY_WORD).all()),
                bool((rows[half ^ 1] == EMPTY_WORD).all()),
            )
        assert R.all_true(all(seen)), (
            f"phase {phase}: rank {R.rank} saw (first pushes landed, own slot empty, other half empty) = {seen}"
        )
        if R.rank not in ahead:
            outs = [call.run(ex) for call in calls]
        for layer, (call, got) in enumerate(zip(calls, outs)):
            verify(call, got, f"phase {phase} (ranks {list(ahead)} ahead), layer {layer}")
        assert_clean(ex, f"after phase {phase}")


def check_two_exchanges_interleaved() -> None:
    """Two exchanges are two counts and two buffers: calls alternate between them in an irregular pattern (A A B A B B
    ...), so the halves they use differ from call to call, a random rank late before each; every call is correct and
    each exchange ends clean at its own count. The pattern is the same on every rank: a reduce waits for its peers'
    pushes of the same call and a stream runs its kernels in order, so ranks issuing calls on two exchanges in
    different orders would deadlock (not run)."""
    pattern = "AABABBAAAB" * 2
    late = random.Random(9)
    for i, which in enumerate(pattern):
        ex = EX_A if which == "A" else EX_B
        call = Call(20_000 + i, (3, 8, 1, 8, 5)[i % 5])
        R.late(late.randrange(R.world))
        verify(call, call.run(ex), f"interleaved {which} {i}")
    assert_clean(EX_A, "exchange A after the interleaving")
    assert_clean(EX_B, "exchange B after the interleaving")


def check_graph_capture_and_replay() -> None:
    """Two captured steps on one exchange, as an engine keeps one graph per batch size: LAYERS push + reduce pairs at
    M 8 in one graph and at M 3 in another, replayed alternately REPLAYS times each with rewritten partials and a
    random rank late, two eager calls of other sizes between replays. Replays and eager calls advance one count.

    The emulated pushes are copies whose half is chosen on the host when they are captured, so each graph holds an
    even number of pairs and an even number of eager calls runs between replays: every replay starts on the parity
    its capture assumed. The reduce reads the count on the device at replay time, as a producer does."""
    ex = EX_B
    for t in GRAPH_TOKENS:
        run_step(ex, 30_000 + 100 * t, t)  # each size eagerly first, as an engine's warm-up does
    R.barrier()
    base = ex.calls
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {}
    for t in GRAPH_TOKENS:
        bufs = [torch.zeros(t, H, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = []
            for layer, buf in enumerate(bufs):
                ex.push(buf, half=(base + layer) & 1)
                outs.append(entry(t, ex.state))
        assert len(outs) == LAYERS
        graphs[t] = (graph, bufs, outs)
    R.barrier()
    late = random.Random(11)
    for rep in range(REPLAYS):
        for t in GRAPH_TOKENS:
            graph, bufs, outs = graphs[t]
            calls = [Call(40_000 + 1000 * rep + 100 * t + layer, t) for layer in range(LAYERS)]
            for buf, call in zip(bufs, calls):
                buf.copy_(call.parts[R.rank])
            assert (ex.calls - base) % 2 == 0, "a replay must start on the parity of its capture"
            R.barrier()
            R.late(late.randrange(R.world))
            graph.replay()
            ex.calls += LAYERS
            for layer, (call, got) in enumerate(zip(calls, outs)):
                verify(call, got, f"replay {rep} of M {t}, layer {layer}")
            for j, m in enumerate(EAGER_BETWEEN):
                call = Call(50_000 + 100 * rep + 10 * t + j, m)
                verify(call, call.run(ex), f"eager M {m} after replay {rep} of M {t}")
    assert_clean(ex, "after the replays")
    del graphs


def check_count_parity_across_the_int32_wrap() -> None:
    """The count is int32 and only its parity is read: preset to 2^31 - 3 on every rank while the exchange is clean
    and idle, it crosses 2^31 - 1 -> -2^31 in the next calls, which stay correct, and then reads -2^31 + 1."""
    ex = EX_A
    R.barrier()
    ex.state.flags[0] = WRAP_PRESET
    ex.calls = WRAP_PRESET
    R.barrier()
    for i, t in enumerate((8, 2, 5, 8)):
        call = Call(60_000 + i, t)
        verify(call, call.run(ex), f"call {i} across the wrap")
    assert int32(ex.calls) == -(2**31) + 1
    assert_clean(ex, "after the wrap")


def check_swapped_calls_are_wrong() -> None:
    """Negative control: rank 0 makes two same-shaped calls in swapped order (it pushes its partial of the second call
    first). Every push is still followed by one reduce of its token count, so the counts agree and nothing raises or
    hangs, but every rank's two results are wrong: each reduce sums rank 0's partial of the other call (exactly the
    position-paired sums, bit for bit), and more than half of each result differs from the intended call's. The
    exchange is clean afterwards and a plain call is correct."""
    first, second = Call(70_000, MAX_TOKENS), Call(70_001, MAX_TOKENS)
    order = (second, first) if R.rank == 0 else (first, second)
    R.barrier()
    got = [call.run(EX_A) for call in order]
    # The k-th reduce sums every rank's k-th push: rank 0's partial of one call with the others' of the other.
    paired = [
        ordered_sum([second.parts[0]] + first.parts[1:], RANK_CHUNK),
        ordered_sum([first.parts[0]] + second.parts[1:], RANK_CHUNK),
    ]
    exact = all(same_bits(g, p) for g, p in zip(got, paired))
    wrong = [differing(g, call.ref()) for g, call in zip(got, order)]
    assert R.all_true(exact and min(wrong) > 0.5), (
        f"rank {R.rank}: position-paired sums {exact}, shares differing from the intended calls {wrong}"
    )
    assert_clean(EX_A, "after the swapped pair")
    call = Call(70_002, MAX_TOKENS)
    verify(call, call.run(EX_A), "after the swapped pair")


def check_token_count_mismatch_returns_stale_rows() -> None:
    """Negative control: rank 0's reduces disagree with the pushed token count. Every rank pushes 8 rows and rank 0
    reduces 4: its 4 rows are right, but rows 4-7 of that half stay full in its buffer (a reduce empties only the rows
    it sums). Two calls later, on the same half, every rank pushes 4 rows and rank 0 reduces 8: it does not wait for
    rows 4-7, which are already full, and returns the older call's sums for them. Nothing raises or hangs; that reduce
    empties all 8 rows, so the exchange is clean again and the next call is correct.

    Not run, because they wait forever: a reduce of rows nobody pushed (with nothing stale there), and a push into the
    other half."""
    ex = EX_A
    old, between, short = Call(80_000, 8), Call(80_001, 8), Call(80_002, 4)
    mine = 4 if R.rank == 0 else 8
    R.barrier()
    half = ex.calls & 1
    ex.push(old.parts[R.rank])
    got = ex.reduce(mine)
    R.barrier()
    rows = ex.state.uc.view(2, MAX_TOKENS, R.world, ROW_WORDS)
    left = int((ex.state.uc != EMPTY_WORD).sum())
    stuck = 4 * R.world * ROW_WORDS if R.rank == 0 else 0
    full = R.rank != 0 or bool((rows[half, 4:] != EMPTY_WORD).all())
    ok = same_bits(got, old.ref()[:mine]) and left == stuck and full
    assert R.all_true(ok), (
        f"rank {R.rank}: {left} words not empty (expected {stuck}), rows 4-7 full {full}"
    )
    verify(between, between.run(ex), "the call on the other half")
    assert ex.calls & 1 == half
    ex.push(short.parts[R.rank])
    got = ex.reduce(8 if R.rank == 0 else 4)
    torch.cuda.synchronize()
    if R.rank == 0:
        ok = same_bits(got[:4], short.ref()) and same_bits(got[4:], old.ref()[4:])
    else:
        ok = same_bits(got, short.ref())
    assert R.all_true(ok), f"rank {R.rank}: the reduce of 8 rows did not return the 4 stale ones"
    assert_clean(ex, "after the mismatched calls")
    call = Call(80_003, MAX_TOKENS)
    verify(call, call.run(ex), "after the mismatched calls")


CHECKS = [
    check_exchange_is_armed_and_sized,
    check_create_needs_only_the_tp_group,
    # Before every eager call of the op: it needs a cold compile cache.
    check_capture_refusals,
    check_single_calls,
    check_bit_identical_to_mnnvl_oneshot,
    check_unsupported_token_counts_raise_on_every_rank,
    check_dip_and_regrow_sequence,
    check_ranks_run_a_step_apart,
    check_two_exchanges_interleaved,
    check_graph_capture_and_replay,
    check_count_parity_across_the_int32_wrap,
    # Stay last: they deliberately break the call-order invariant.
    check_swapped_calls_are_wrong,
    check_token_count_mismatch_returns_stale_rows,
]


def _run_one_rank(args) -> int:
    global R, entry, create_exchange, mnnvl, EX_A, EX_B, MNNVL_WS
    R = ls.Rank(args)
    assert R.world in WORLDS, f"k3_latent_reduce runs on TP groups of {WORLDS} ranks, not {R.world}"
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        k3_latent_reduce as module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_fusion_allreduce as mnnvl_module,
    )

    entry = module.k3_latent_reduce
    create_exchange = module.K3LatentExchange.create
    mnnvl = mnnvl_module.mnnvl_fusion_allreduce
    with torch.inference_mode():
        EX_A = Exchange(create_exchange(R.mapping, fabric_handle=R.fabric))
        EX_B = Exchange(create_exchange(R.mapping, fabric_handle=R.fabric))
        MNNVL_WS = mnnvl_module.MnnvlWorkspace.create(
            R.mapping, MNNVL_BUFFER_BYTES, fabric_handle=R.fabric
        )
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(
            f"[rank 0] world {R.world}; calls on A {EX_A.calls} (count {int32(EX_A.calls)}), on B {EX_B.calls}; "
            f"least share of elements another summation order changes {STATS['order_guard']:.2%}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
