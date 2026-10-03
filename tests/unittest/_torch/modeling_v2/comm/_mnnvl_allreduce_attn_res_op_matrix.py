# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/mnnvl_allreduce_attn_res`` catalog entry and its ``MnnvlWorkspace``.

The op's correctness depends on state that outlives a call (the workspace's Lamport rotation), so beyond single
calls this drives call *sequences*: layers x steps with the token count dipping and growing back and a random rank
late, two workspaces interleaved, CUDA-graph capture and replay mixed with eager calls, and a negative control in
which one rank swaps two calls and every rank gets a wrong answer without an error -- the failure the sequence tests
exist to catch.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_allreduce_attn_res_op_matrix.py [--world-size 4]
    srun -n 16 --mpi=pmix python _mnnvl_allreduce_attn_res_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
rotation). The collected entry point is ``test_modeling_v2_mnnvl_allreduce_attn_res_op_matrix.py``.

Every rank draws every rank's inputs from one seed, so each rank holds the whole reference. The inputs of the sum
are small multiples of 1/16, so ``updated`` is exact in fp32 and bf16 whatever the summation order and is compared
bit for bit; ``normed`` (softmax, rsqrt) against the fp32 reference within ``TOL``. Every output is also compared
bitwise across the ranks.
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "mnnvl_allreduce_attn_res requires CUDA devices"

DEADLINE_S = 900
H = 7168
BUFFER_BYTES = 4 << 20  # the Lamport buffer size Kimi K3 decodes with
RMS_EPS = 1e-6
OUT_EPS = 1e-6
TOL = 2e-2  # normed: max |err| / max |ref|
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 16)
SNAPSHOTS = (0, 1, 5, 8, 11)  # candidates 1..12
LAYERS = 12
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 8, 16, 3, 8)

R = None
entry = None
WS_A = None
WS_B = None
STATS = {"normed_err": 0.0}


class Call:
    """One call's arguments on every rank and its reference. ``prefix``: a tensor (chained from the previous call's
    ``updated``), True (drawn), or None."""

    def __init__(self, seed, tokens, snapshots, prefix=True):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.inputs = [ls.exact_bf16(g, (tokens, H), -4, 5, 1 / 16) for _ in range(R.world)]
        if prefix is True:
            prefix = ls.exact_bf16(g, (tokens, H), -32, 33, 1 / 16)
        self.prefix = prefix
        self.block = torch.randn(snapshots, tokens, H, generator=g, device="cuda").bfloat16()
        self.res_w = (torch.randn(H, generator=g, device="cuda") * 0.05).bfloat16()
        self.rms_w = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()
        self.out_w = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()

    def ref(self):
        total = sum(x.float() for x in self.inputs)
        if self.prefix is not None:
            total = total + self.prefix.float()
        updated = total.bfloat16()
        normed = ls.residual_update_ref(
            updated, self.block, self.res_w, self.rms_w, RMS_EPS, self.out_w, OUT_EPS
        )
        return normed, updated

    def run(self, ws):
        return entry(self.inputs[R.rank], self.prefix, self.block, self.res_w, self.rms_w, self.out_w, RMS_EPS,
                     OUT_EPS, ws)  # fmt: skip


def verify(call: Call, got, where: str) -> None:
    normed, updated = got
    want_normed, want_updated = call.ref()
    assert torch.equal(updated, want_updated), f"{where}: updated differs from the exact sum"
    err = ls.rel_err(normed, want_normed)
    STATS["normed_err"] = max(STATS["normed_err"], err)
    assert err <= TOL, f"{where}: normed rel err {err:.3e} > {TOL}"
    assert R.same_on_ranks(normed, updated), f"{where}: ranks disagree"


def check_workspace_is_armed_and_sized() -> None:
    assert WS_A.world_size == R.world and WS_A.rank == R.rank
    assert WS_A.comm_buffer(torch.bfloat16).shape == (3, BUFFER_BYTES // 2)
    assert WS_A.max_one_shot_tokens(H) == BUFFER_BYTES // (H * R.world * 2)
    armed = WS_A.lamport.view(torch.int32)
    assert bool((armed == torch.tensor(-(2**31), dtype=torch.int32, device="cuda")).all()), (
        "every word -0.0"
    )
    assert WS_A.buffer_flags.view(torch.int32).tolist()[:3] == [0, 2, BUFFER_BYTES]


def check_create_refuses_on_every_rank() -> None:
    """One rank asks for three buffers past its device's free memory: every rank raises before any allocates, and the
    workspaces in use stay correct."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
        MnnvlWorkspace,
    )

    free_bytes, _ = torch.cuda.mem_get_info()
    too_big = (free_bytes // 3 // 16 + (64 << 20) // 16) * 16
    try:
        MnnvlWorkspace.create(
            R.mapping, too_big if R.rank == 0 else BUFFER_BYTES, fabric_handle=R.fabric
        )
        raised = False
    except RuntimeError as exc:
        raised = "not every rank can allocate" in str(exc)
    assert R.all_true(raised), "a rank short of memory did not make every rank raise"
    call = Call(2100, 8, 3)
    verify(call, call.run(WS_A), "after the refused create")


def check_single_calls() -> None:
    for t in TOKENS:
        for s in SNAPSHOTS:
            for with_prefix in (True, False):
                call = Call(1000 + 37 * t + s, t, s, prefix=with_prefix or None)
                verify(call, call.run(WS_A), f"T {t} snapshots {s} prefix {with_prefix}")


def check_a_call_over_one_buffer_raises_on_every_rank() -> None:
    t = WS_A.max_one_shot_tokens(H) + 1
    call = Call(2000, t, 1)
    try:
        call.run(WS_A)
        raised = False
    except RuntimeError as exc:
        raised = "exceeds one Lamport buffer" in str(exc)
    assert R.all_true(raised), f"T {t} over one Lamport buffer did not raise on every rank"
    # The rotation did not move: the next call is still correct.
    call = Call(2001, 8, 2)
    verify(call, call.run(WS_A), "after the rejected call")


def run_step(ws, seed, tokens, layers=LAYERS, late_rng=None):
    """One decode step: ``layers`` calls, each layer's prefix the previous layer's ``updated``; verified."""
    prefix = ls.exact_bf16(
        torch.Generator(device="cuda").manual_seed(seed), (tokens, H), -32, 33, 1 / 16
    )
    for layer in range(layers):
        call = Call(seed + 1 + layer, tokens, (layer * 5) % 12, prefix=prefix)
        R.barrier()
        R.late(late_rng.randrange(R.world) if late_rng is not None else None)
        got = call.run(ws)
        verify(call, got, f"step seed {seed} T {tokens} layer {layer}")
        prefix = got[1]


def check_dip_and_regrow_sequence() -> None:
    """A call after a smaller one must not read what an older, larger call left in the buffer (the failure of a
    re-arm sized by the current call). Steps of 12 layers at T 8, 8, 8, 2, 7, 8, 1, 1, 8, 16, 3, 8, a random rank
    late at every call."""
    late = random.Random(7)
    for i, t in enumerate(DIP_STEPS):
        run_step(WS_A, 3000 + 100 * i, t, late_rng=late)


def check_two_workspaces_interleaved() -> None:
    """Two workspaces are two rotations: calls alternate between them in an irregular pattern (A A B A B B ...), so
    the two objects' positions differ, and every call is correct. The pattern is the same on every rank: calls on one
    stream are serialized and each waits for its peers, so two ranks issuing calls on two workspaces in different
    orders deadlock (measured; not exercised here)."""
    pattern = "AABABBAAAB" * 2
    for i, which in enumerate(pattern):
        ws = WS_A if which == "A" else WS_B
        call = Call(4000 + i, (3, 8, 1, 8, 5)[i % 5], i % 12)
        verify(call, call.run(ws), f"interleaved {which} {i}")


def check_graph_capture_and_replay() -> None:
    """A captured step of 12 chained calls (T 8) replayed with rewritten inputs, eager calls of other shapes on the
    same workspace between replays: replays and eager calls share one rotation, in the same order on every rank."""
    t = 8
    calls = [Call(5000 + layer, t, (layer * 5) % 12, prefix=True) for layer in range(LAYERS)]
    bufs = [(c.inputs[R.rank].clone(), c.block.clone()) for c in calls]
    prefix0 = calls[0].prefix.clone()

    def step():
        outs, prefix = [], prefix0
        for c, (x, blk) in zip(calls, bufs):
            outs.append(entry(x, prefix, blk, c.res_w, c.rms_w, c.out_w, RMS_EPS, OUT_EPS, WS_B))
            prefix = outs[-1][1]
        return outs

    step()  # the first call of every shape eagerly
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = step()
    R.barrier()
    for rep in range(8):
        fresh = [
            Call(6000 + 100 * rep + layer, t, (layer * 5) % 12, prefix=True)
            for layer in range(LAYERS)
        ]
        prefix0.copy_(fresh[0].prefix)
        for (x, blk), f, c in zip(bufs, fresh, calls):
            x.copy_(f.inputs[R.rank])
            blk.copy_(f.block)
            c.inputs, c.block = f.inputs, f.block
        R.barrier()
        graph.replay()
        prefix = prefix0
        for layer, (c, got) in enumerate(zip(calls, outs)):
            c.prefix = prefix
            verify(c, got, f"replay {rep} layer {layer}")
            prefix = got[1]
        eager = Call(7000 + rep, (3, 1, 16, 5)[rep % 4], rep % 12)
        verify(eager, eager.run(WS_B), f"eager after replay {rep}")
    del graph


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 swaps two same-shaped calls on one workspace. Every call still returns and nothing
    raises or hangs (the rotation positions agree), but every rank's two results are wrong: a call pairs with the
    peers' call at the same position. Then a plain call is correct again: a swapped pair realigns the positions."""
    c1, c2 = Call(8000, 8, 3), Call(8001, 8, 3)
    R.barrier()
    if R.rank == 0:
        got2, got1 = c2.run(WS_A), c1.run(WS_A)
    else:
        got1, got2 = c1.run(WS_A), c2.run(WS_A)
    torch.cuda.synchronize()
    wrong = [(got[1] != c.ref()[1]).float().mean().item() for c, got in ((c1, got1), (c2, got2))]
    assert R.all_true(min(wrong) > 0.5), f"the swap went unnoticed: wrong fractions {wrong}"
    c3 = Call(8002, 8, 3)
    verify(c3, c3.run(WS_A), "after the swapped pair")


CHECKS = [
    check_workspace_is_armed_and_sized,
    check_create_refuses_on_every_rank,
    check_single_calls,
    check_a_call_over_one_buffer_raises_on_every_rank,
    check_dip_and_regrow_sequence,
    check_two_workspaces_interleaved,
    check_graph_capture_and_replay,
    # Stays last: it deliberately disagrees on call order.
    check_wrong_call_order_is_detected,
]


def _run_one_rank(args) -> int:
    global R, entry, WS_A, WS_B
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        mnnvl_allreduce_attn_res as module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
        MnnvlWorkspace,
    )

    entry = module.mnnvl_allreduce_attn_res
    with torch.inference_mode():
        WS_A = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        WS_B = MnnvlWorkspace.create(R.mapping, BUFFER_BYTES, fabric_handle=R.fabric)
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(f"[rank 0] world {R.world}; max normed rel err {STATS['normed_err']:.3e}", flush=True)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
