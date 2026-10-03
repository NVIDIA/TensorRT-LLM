# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``comm/k3_sandwich_tail`` catalog entry on a ``K3SandwichWorkspace``.

The pre-attention sandwich: this rank's row-parallel MoE tail ``[rmsnorm(latent)[:, lo:lo+224] | act] @
tail_weight^T`` (the latent RMS taken over the whole reduced latent row and applied to the fp32 accumulator), the TP
all-reduce and the next layer's residual update, with the optional tap (the pre-norm mixture, or ``updated``) and
``updated_out``. The kernel's correctness depends on state that outlives a call (each CTA's call count, whose parity
picks the buffer half every rank pushes into), so beyond single calls this drives call *sequences*: layers x steps
with the token count dipping and growing back and a random rank late, two workspaces interleaved, CUDA-graph capture
and replay (with a tapping layer and a bank-row layer inside) mixed with eager calls, and a negative control in which
one rank swaps two calls and every rank gets a wrong answer without an error. The workspace shared with
``k3_sandwich_oproj`` and ``k3_sandwich_plain`` is certified by ``_k3_sandwich_oproj_op_matrix.py``.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_sandwich_tail_op_matrix.py [--world-size 4]
    srun -n 16 --mpi=pmix python _k3_sandwich_tail_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces and their
counters). The collected entry point is ``test_modeling_v2_k3_sandwich_tail_op_matrix.py``.

Shapes are TP16's per-rank shapes (latent [M, 3584], act [M, 384], tail_weight [7168, 256 + 384]) whatever W; rank
r's latent slice ``lo`` moves from call to call over the 16 slices. The kernel sums the ranks in chunks of 8, so
W <= 8 exercises one chunk. Every rank draws every rank's inputs from one seed, so each rank holds the whole
reference: the fp64 tail partial rounded once to bf16 per rank, the ranks' sum and the prefix in fp32 as the kernel
adds them. The kernel's latent rsqrt is not torch's, so a partial element may round to the neighbouring bf16:
``updated`` is compared within 8e-3 of its largest magnitude (the negative control's wrong pairing is checked against
the same bound), ``normed`` and the tapped mixture within 2e-2, the tapped ``updated`` bit for bit, every output
bitwise across the ranks (``_k3_sandwich_common``).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _k3_sandwich_common as cm  # noqa: E402
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_sandwich_tail requires CUDA devices"

DEADLINE_S = 900
LAYERS = 12
SNAPSHOTS = (0, 1, 4, 8)  # candidates 1..9
CAPTURED_OPTIONS = {3: {"tap": "mix"}, 7: {"updated_out": True}, 10: {"tap": "updated"}}

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
    cm.create_refuses_capture(WORKSPACE, WS_A, cm.TailCall(600, 8, 2))


def check_single_calls() -> None:
    """M 1-8 x snapshots 0, 1, 4, 8 x prefix or none. Call i puts rank r's slice at ((5 r + i) % 16) x 224, so every
    rank takes every slice, lo = 0 to 3360 (the last one's padding columns run past the latent row)."""
    i = 0
    for t in cm.TOKENS:
        for s in SNAPSHOTS:
            for with_prefix in (True, False):
                call = cm.TailCall(1000 + 37 * t + s, t, s, prefix=with_prefix or None, weights=t % cm.WEIGHT_SETS,
                                   shift=i)  # fmt: skip
                call.verify(
                    call.run(WS_A),
                    f"M {t} snapshots {s} prefix {with_prefix} lo {call.los[R.rank]}",
                )
                i += 1


def check_tap_and_updated_out() -> None:
    """At every M, one call's inputs four times: no option, the mixture tapped into a column slice of a capture
    buffer, ``updated`` tapped there, ``updated`` stored into a bank row (then returned as ``updated``). Every call
    against the reference, nothing written outside the tap and the bank row, and the four calls' ``normed`` and
    ``updated`` bit-identical."""
    for t in cm.TOKENS:
        base = cm.TailCall(1300 + t, t, 3)
        got0 = base.run(WS_A)
        base.verify(got0, f"M {t} no options")
        for options in ({"tap": "mix"}, {"tap": "updated"}, {"updated_out": True}):
            call = base.with_options(**options)
            got = call.run(WS_A)
            call.verify(got, f"M {t} {options}")
            same = all(torch.equal(cm.bits(a), cm.bits(b)) for a, b in zip(got, got0))
            assert same, f"M {t} {options}: outputs differ from the call without options"


def check_counters_advance_once_per_call() -> None:
    cm.counters_advance_once(WS_A, cm.TailCall(1500, 4, 2))


def check_unsupported_calls_raise_on_every_rank() -> None:
    """M 9, a latent 2 bytes off 16-byte alignment (its rows are bulk-copied) and a tap whose rows are 7172 elements
    apart (not a multiple of 8) each raise ValueError on every rank before any launch; the next call is correct."""
    big = cm.TailCall(2000, 9, 1)
    misaligned = cm.TailCall(2001, 4, 1)
    store = torch.zeros(4 * cm.LATENT + 8, dtype=torch.bfloat16, device="cuda")
    shifted = store[1 : 1 + 4 * cm.LATENT].view(
        4, cm.LATENT
    )  # contiguous, 2 bytes past an aligned address
    shifted.copy_(misaligned.latent)
    misaligned.latent = shifted
    strided = cm.TailCall(2002, 4, 1)
    rows = torch.zeros(4, cm.H + 4, dtype=torch.bfloat16, device="cuda")
    strided.tap_kind, strided.tap = "mix", rows[:, : cm.H]  # only run, never verified
    cm.unsupported_raises(
        WS_A,
        [
            ("M 9", lambda: big.run(WS_A)),
            ("misaligned latent", lambda: misaligned.run(WS_A)),
            ("tap row stride 7172", lambda: strided.run(WS_A)),
        ],
        cm.TailCall(2003, 8, 2),
    )


def _step(seed, tokens, options=None):
    """One decode step: ``LAYERS`` chained calls, each layer's prefix the previous layer's ``updated``; ``options``
    maps a layer to its tap / updated_out."""
    options = options or {}
    return [
        (
            "target",
            cm.TailCall(
                seed + 1 + layer,
                tokens,
                layer % 9,
                prefix=True if layer == 0 else None,
                weights=layer % cm.WEIGHT_SETS,
                **options.get(layer, {}),
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
        lambda i: cm.TailCall(4000 + i, (3, 8, 1, 8, 5)[i % 5], i % 9, weights=i % cm.WEIGHT_SETS),
    )


def check_graph_capture_and_replay() -> None:
    """A captured step of 12 chained calls (M 8; layer 3 taps the mixture, layer 7 stores ``updated`` into a bank row,
    layer 10 taps ``updated``) replayed 8 times with rewritten inputs, an eager call of another M on the same
    workspace between replays."""
    cm.capture_and_replay(
        WS_B,
        lambda seed: _step(seed, 8, CAPTURED_OPTIONS),
        lambda rep: [
            cm.TailCall(7000 + rep, (3, 1, 6, 5)[rep % 4], rep % 9, weights=rep % cm.WEIGHT_SETS)
        ],
        "captured step",
    )


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 swaps two same-shaped calls (no prefix) on one workspace. Every call returns and
    nothing raises or hangs, but every rank's two results are wrong, far outside the 8e-3 tolerance: more than half
    the ``updated`` elements are off by more than it, the largest by over 10 times it. Then a plain call is correct
    again."""
    cm.swapped_pair_is_wrong(
        WS_A,
        cm.TailCall(8000, 8, 3, prefix=None),
        cm.TailCall(8001, 8, 3, prefix=None, weights=1),
        cm.TailCall(8002, 8, 3),
    )


CHECKS = [
    check_workspace_is_armed_and_sized,
    check_create_refuses_capture,
    check_single_calls,
    check_tap_and_updated_out,
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
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_tail import (
        K3SandwichWorkspace,
    )

    WORKSPACE = K3SandwichWorkspace
    with torch.inference_mode():
        cm.bind(R, {"tail": cm.WEIGHT_SETS})
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
