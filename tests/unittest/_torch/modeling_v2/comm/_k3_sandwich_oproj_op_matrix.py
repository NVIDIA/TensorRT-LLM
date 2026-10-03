# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/k3_sandwich_oproj`` catalog entry and its ``K3SandwichWorkspace``.

The kernel's correctness depends on state that outlives a call (each CTA's call count, whose parity picks the
buffer half every rank pushes into), so beyond single calls this drives call *sequences*: layers x steps with the
token count dipping and growing back and a random rank late, two workspaces interleaved, CUDA-graph capture and
replay mixed with eager calls, and a negative control in which one rank swaps two calls and every rank gets a wrong
answer without an error. The workspace is also shared the way the model shares it: one decode step of target layers
(this op, then ``k3_sandwich_tail``) with the drafter's ``k3_sandwich_plain`` calls interleaved, call by call, then
that step captured and replayed between eager calls of the three ops. And ``create`` must refuse CUDA-graph capture.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_sandwich_oproj_op_matrix.py [--world-size 4]
    srun -n 16 --mpi=pmix python _k3_sandwich_oproj_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
counters). The collected entry point is ``test_modeling_v2_k3_sandwich_oproj_op_matrix.py``.

Shapes are TP16's per-rank shapes (core [M, 768], o_weight [7168, 768]; the shared step's tail and plain calls at
theirs) whatever W; the kernel sums the ranks in chunks of 8, so W <= 8 exercises one chunk. Every rank draws every
rank's core and weight slice from one seed, so each rank holds the whole reference. core and o_weight are small
multiples of 1/8 and 1/16: every partial product sum is exact in fp32, so ``updated`` = bf16(prefix + bf16(sum_r
bf16(core_r @ o_weight_r^T))) is compared bit for bit; ``normed`` against the fp32 reference within 2e-2; every
output bitwise across the ranks (``_k3_sandwich_common``).
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _k3_sandwich_common as cm  # noqa: E402
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_sandwich_oproj requires CUDA devices"

DEADLINE_S = 900
LAYERS = 12
SNAPSHOTS = (0, 1, 4, 8)  # candidates 1..9
TARGET_LAYERS = 3  # the shared step's target layers (one o_weight set each)
SHARED_STEPS = ((8, 3), (2, 8), (7, 7))  # (target M, drafter M) of the eager shared steps
SHARED_CAPTURED = (8, 4)
SHARED_REPLAYS = 6

R = None
WORKSPACE = None  # the K3SandwichWorkspace type, as this entry's wrapper exports it
WS_A = None
WS_B = None


def check_workspace_is_armed_and_sized() -> None:
    cm.armed_and_sized(WS_A)
    cm.armed_and_sized(WS_B)


def check_create_refuses_capture() -> None:
    """``create`` under CUDA-graph capture raises on every rank at once and on one rank alone (before any
    collective); the workspace in use is untouched."""
    cm.create_refuses_capture(WORKSPACE, WS_A, cm.OprojCall(600, 8, 2))


def check_single_calls() -> None:
    for t in cm.TOKENS:
        for s in SNAPSHOTS:
            for with_prefix in (True, False):
                call = cm.OprojCall(
                    1000 + 37 * t + s, t, s, prefix=with_prefix or None, weights=t % cm.WEIGHT_SETS
                )
                call.verify(call.run(WS_A), f"M {t} snapshots {s} prefix {with_prefix}")


def check_counters_advance_once_per_call() -> None:
    cm.counters_advance_once(WS_A, cm.OprojCall(1500, 4, 2))


def check_unsupported_shape_raises_on_every_rank() -> None:
    """M 9 raises ValueError on every rank before any launch; the next call is correct."""
    bad = cm.OprojCall(2000, 9, 1)
    cm.unsupported_raises(WS_A, [("M 9", lambda: bad.run(WS_A))], cm.OprojCall(2001, 8, 2))


def _step(seed, tokens):
    """One decode step: ``LAYERS`` chained calls, each layer's prefix the previous layer's ``updated``."""
    return [
        (
            "target",
            cm.OprojCall(
                seed + 1 + layer,
                tokens,
                layer % 9,
                prefix=True if layer == 0 else None,
                weights=layer % cm.WEIGHT_SETS,
            ),
        )  # fmt: skip
        for layer in range(LAYERS)
    ]


def check_dip_and_regrow_sequence() -> None:
    """Steps of 12 layers at M 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, a random rank late at every call: a call after a
    smaller one must not see words an older, larger call left."""
    cm.dip_and_regrow(WS_A, _step)


def check_two_workspaces_interleaved() -> None:
    """Two workspaces are two sets of counters: 20 calls alternate between them in an irregular pattern."""
    cm.interleaved(
        WS_A,
        WS_B,
        lambda i: cm.OprojCall(4000 + i, (3, 8, 1, 8, 5)[i % 5], i % 9, weights=i % cm.WEIGHT_SETS),
    )


def check_graph_capture_and_replay() -> None:
    """A captured step of 12 chained calls (M 8) replayed 8 times with rewritten inputs, an eager call of another M on
    the same workspace between replays."""
    cm.capture_and_replay(
        WS_B,
        lambda seed: _step(seed, 8),
        lambda rep: [
            cm.OprojCall(7000 + rep, (3, 1, 6, 5)[rep % 4], rep % 9, weights=rep % cm.WEIGHT_SETS)
        ],
        "captured step",
    )


def _shared_step(seed, t_tokens, d_tokens):
    """One decode step as the model runs it on one workspace. Target layer l: k3_sandwich_oproj (post-attention), then
    k3_sandwich_tail (the MoE tail and the next pre-attention step; layer 1's stores ``updated`` into a bank row),
    chained through the target's prefix sum. The drafter's layers -- k3_sandwich_plain (o_proj, K 384), then its
    SwiGLU form (down, K 896), chained through the drafter's residual -- run after target layers 0 and 2 at the
    drafter's own token count."""
    seq = []
    for layer in range(TARGET_LAYERS):
        s = seed + 10 * layer
        seq.append(("target", cm.OprojCall(s + 1, t_tokens, (4 * layer) % 9, prefix=True if layer == 0 else None,
                                           weights=layer)))  # fmt: skip
        seq.append(
            (
                "target",
                cm.TailCall(
                    s + 2, t_tokens, (4 * layer + 1) % 9, prefix=None, updated_out=layer == 1
                ),
            )
        )
        if layer != 1:
            seq.append(
                ("drafter", cm.PlainCall(s + 3, d_tokens, residual=True if layer == 0 else None))
            )
            seq.append(("drafter", cm.PlainCall(s + 4, d_tokens, swiglu=True, residual=None)))
    return seq


def check_shared_sequence() -> None:
    """The three sandwich ops on one workspace, as the model runs them (``_shared_step``, 10 calls): steps at (target
    M, drafter M) = (8, 3), (2, 8), (7, 7), a random rank late at every call, every call against its reference.
    The three wrappers export one workspace type."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        k3_sandwich_plain,
        k3_sandwich_tail,
    )

    assert (
        k3_sandwich_tail.K3SandwichWorkspace is WORKSPACE is k3_sandwich_plain.K3SandwichWorkspace
    )
    late = random.Random(11)
    for i, (t, d) in enumerate(SHARED_STEPS):
        cm.run_sequence(
            _shared_step(9000 + 100 * i, t, d),
            WS_A,
            late=late,
            where=f"shared step {i} M {t} / {d}",
        )


def check_shared_sequence_captured() -> None:
    """That step captured at (target M, drafter M) = (8, 4) and replayed 6 times with rewritten inputs, eager calls of
    the three ops at other token counts on the same workspace between replays (one or two, so the replays start at
    either counter parity)."""
    eager = (
        lambda rep: [cm.TailCall(9700 + rep, 3, 2)],
        lambda rep: [cm.PlainCall(9710 + rep, 5), cm.OprojCall(9720 + rep, 1, 4)],
        lambda rep: [cm.PlainCall(9730 + rep, 8, swiglu=True)],
    )
    cm.capture_and_replay(
        WS_A,
        lambda seed: _shared_step(seed, *SHARED_CAPTURED),
        lambda rep: eager[rep % len(eager)](rep),
        "captured shared step",
        replays=SHARED_REPLAYS,
    )


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 swaps two same-shaped calls on one workspace. Every call returns and nothing raises
    or hangs, but every rank's two results are wrong (more than half the ``updated`` elements differ); then a plain
    call is correct again."""
    cm.swapped_pair_is_wrong(
        WS_A,
        cm.OprojCall(8000, 8, 3),
        cm.OprojCall(8001, 8, 3, weights=1),
        cm.OprojCall(8002, 8, 3),
    )


CHECKS = [
    check_workspace_is_armed_and_sized,
    check_create_refuses_capture,
    check_single_calls,
    check_counters_advance_once_per_call,
    check_unsupported_shape_raises_on_every_rank,
    check_dip_and_regrow_sequence,
    check_two_workspaces_interleaved,
    check_graph_capture_and_replay,
    check_shared_sequence,
    check_shared_sequence_captured,
    # Stays last: it deliberately disagrees on call order.
    check_wrong_call_order_is_detected,
]


def _run_one_rank(args) -> int:
    global R, WORKSPACE, WS_A, WS_B
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_oproj import (
        K3SandwichWorkspace,
    )

    WORKSPACE = K3SandwichWorkspace
    with torch.inference_mode():
        cm.bind(R, {"oproj": cm.WEIGHT_SETS, "tail": 1, "plain": 1, "down": 1})
        WS_A = K3SandwichWorkspace.create(R.mapping, fabric_handle=R.fabric)
        WS_B = K3SandwichWorkspace.create(R.mapping, fabric_handle=R.fabric)
        code = ls.run_checks(R, CHECKS)
    cm.report()
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
