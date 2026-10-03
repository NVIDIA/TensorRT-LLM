# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/k3_sandwich_plain`` catalog entry on a ``K3SandwichWorkspace``.

The drafter's sandwich: a row-parallel projection (the o_proj slice, K 384; or with ``swiglu`` SiLU-and-mul of the
gate_up output and the down projection slice, K 896), the TP all-reduce, the residual add and an RMSNorm. The
kernel's correctness depends on state that outlives a call (each CTA's call count, whose parity picks the buffer half
every rank pushes into), so beyond single calls this drives call *sequences*: drafter layers x steps with the token
count dipping and growing back and a random rank late, two workspaces interleaved, CUDA-graph capture and replay mixed
with eager calls, and a negative control in which one rank swaps two calls and every rank gets a wrong answer without
an error. The workspace shared with ``k3_sandwich_oproj`` and ``k3_sandwich_tail`` is certified by
``_k3_sandwich_oproj_op_matrix.py``.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_sandwich_plain_op_matrix.py [--world-size 4]
    srun -n 16 --mpi=pmix python _k3_sandwich_plain_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
counters). The collected entry point is ``test_modeling_v2_k3_sandwich_plain_op_matrix.py``.

Shapes are TP16's per-rank shapes (x [M, 384] and weight [7168, 384]; with swiglu x [M, 1792] and weight [7168, 896])
whatever W; the kernel sums the ranks in chunks of 8, so W <= 8 exercises one chunk. Every rank draws every rank's
inputs from one seed, so each rank holds the whole reference. x and the weights are small multiples of powers of two,
and the swiglu gates are 0, 32 or 64, on which silu is exact in fp32: every partial sum is exact, so ``updated`` =
bf16(residual + bf16(sum_r bf16(x_r @ weight_r^T))) is compared bit for bit in both forms; ``normed`` (the one-shot's
RMSNorm sums bf16-rounded squares in its own tree) against the fp32 RMSNorm within 2e-2; every output bitwise across
the ranks (``_k3_sandwich_common``).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _k3_sandwich_common as cm  # noqa: E402
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_sandwich_plain requires CUDA devices"

DEADLINE_S = 900
DRAFTER_LAYERS = 6  # two calls each: 12 per step

R = None
WORKSPACE = None  # the K3SandwichWorkspace type, as this entry's wrapper exports it
WS_A = None
WS_B = None


def check_workspace_is_armed_and_sized() -> None:
    cm.armed_and_sized(WS_A)
    cm.armed_and_sized(WS_B)


def check_create_refuses_capture() -> None:
    """``create`` under CUDA-graph capture: every rank capturing, every rank raises; one rank capturing while its
    peers call it eagerly, every rank raises too. No rank allocates, no stream is left capturing, and the workspace
    in use is untouched."""
    cm.create_refuses_capture(WORKSPACE, WS_A, cm.PlainCall(600, 8))


def check_single_calls() -> None:
    """M 1-8 x both forms x two draws."""
    for t in cm.TOKENS:
        for swiglu in (False, True):
            for draw in (0, 1):
                call = cm.PlainCall(1000 + 37 * t + 7 * draw + int(swiglu), t, swiglu=swiglu,
                                    weights=(t + draw) % cm.WEIGHT_SETS)  # fmt: skip
                call.verify(call.run(WS_A), f"M {t} swiglu {swiglu} draw {draw}")


def check_counters_advance_once_per_call() -> None:
    cm.counters_advance_once(WS_A, cm.PlainCall(1500, 4))
    cm.counters_advance_once(WS_A, cm.PlainCall(1501, 4, swiglu=True))


def check_unsupported_calls_raise_on_every_rank() -> None:
    """M 9, the SwiGLU form on a K 384 slice (it takes K 896 only) and an x whose K is not the weight's each raise
    ValueError on every rank before any launch; the next call is correct."""
    big = cm.PlainCall(2000, 9)
    narrow = cm.PlainCall(2001, 4)
    narrow.swiglu, narrow.xs = (
        True,
        [torch.cat([x, x], dim=1) for x in narrow.xs],
    )  # [4, 768] on a K 384 slice
    wide = cm.PlainCall(2002, 4)
    wide.xs = [torch.zeros(4, 512, dtype=torch.bfloat16, device="cuda") for _ in wide.xs]
    cm.unsupported_raises(
        WS_A,
        [
            ("M 9", lambda: big.run(WS_A)),
            ("swiglu on K 384", lambda: narrow.run(WS_A)),
            ("x K 512, weight K 384", lambda: wide.run(WS_A)),
        ],
        cm.PlainCall(2003, 8),
    )


def _step(seed, tokens):
    """One drafter step: ``DRAFTER_LAYERS`` layers of k3_sandwich_plain (o_proj, K 384) then its SwiGLU form (down,
    K 896), chained through the residual (each call's residual the previous call's ``updated``)."""
    seq = []
    for layer in range(DRAFTER_LAYERS):
        s = seed + 2 * layer
        w = layer % cm.WEIGHT_SETS
        seq.append(
            (
                "drafter",
                cm.PlainCall(s + 1, tokens, residual=True if layer == 0 else None, weights=w),
            )
        )
        seq.append(("drafter", cm.PlainCall(s + 2, tokens, swiglu=True, residual=None, weights=w)))
    return seq


def check_dip_and_regrow_sequence() -> None:
    """Steps of 6 drafter layers (12 calls) at M 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, a random rank late at every call: a
    call after a smaller one must not see words an older, larger call left."""
    cm.dip_and_regrow(WS_A, _step)


def check_two_workspaces_interleaved() -> None:
    """Two workspaces are two sets of counters: 20 calls of both forms alternate between them in an irregular
    pattern."""
    cm.interleaved(
        WS_A,
        WS_B,
        lambda i: cm.PlainCall(
            4000 + i, (3, 8, 1, 8, 5)[i % 5], swiglu=i % 3 == 1, weights=i % cm.WEIGHT_SETS
        ),
    )


def check_graph_capture_and_replay() -> None:
    """A captured drafter step (6 layers, 12 chained calls, M 8) replayed 8 times with rewritten inputs, an eager call
    of another M on the same workspace between replays."""
    cm.capture_and_replay(
        WS_B,
        lambda seed: _step(seed, 8),
        lambda rep: [
            cm.PlainCall(
                7000 + rep, (3, 1, 6, 5)[rep % 4], swiglu=rep % 2 == 1, weights=rep % cm.WEIGHT_SETS
            )
        ],  # fmt: skip
        "captured step",
    )


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 swaps two same-shaped calls on one workspace. Every call returns and nothing raises
    or hangs, but every rank's two results are wrong (more than half the ``updated`` elements differ); then a plain
    call is correct again."""
    cm.swapped_pair_is_wrong(
        WS_A, cm.PlainCall(8000, 8), cm.PlainCall(8001, 8, weights=1), cm.PlainCall(8002, 8)
    )


CHECKS = [
    check_workspace_is_armed_and_sized,
    check_create_refuses_capture,
    check_single_calls,
    check_counters_advance_once_per_call,
    check_unsupported_calls_raise_on_every_rank,
    check_dip_and_regrow_sequence,
    check_two_workspaces_interleaved,
    check_graph_capture_and_replay,
    # Stays last: it deliberately disagrees on call order.
    check_wrong_call_order_is_detected,
]


def _run_one_rank(args) -> int:
    global R, WORKSPACE, WS_A, WS_B
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_plain import (
        K3SandwichWorkspace,
    )

    WORKSPACE = K3SandwichWorkspace
    with torch.inference_mode():
        cm.bind(R, {"plain": cm.WEIGHT_SETS, "down": cm.WEIGHT_SETS})
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
