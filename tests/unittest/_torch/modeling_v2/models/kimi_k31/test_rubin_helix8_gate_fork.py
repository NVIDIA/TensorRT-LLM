# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's output gate projection on CUDA graph steps: ``g_proj(x)`` follows the RoPE kernel on the
fork stream, beside the indexer, instead of running between the combine and the gate.

The attention joins the event recorded right after the RoPE kernel, so it never waits for g; the
gate joins the event recorded after g. Same kernel and the same input ``x`` as before, so the
same bits; only the stream and the place in the step change.

CPU (``cpu_only``), through the recorder of ``test_rubin_helix8_fork_q.py``: helix8's MLA block
runs from its modeling.py over recording stand-ins, on a graph step of each indexer path (the
dense Top-K and the key-sharded one) with CUDA graph padding rows:

* ``g_proj(x)`` runs once, on the fork stream, after the RoPE kernel in that stream's order, over
  the block's input and the gate weight, and the once-per-process INFO line is logged;
* the attention's last wait on the fork stream covers the RoPE kernel and not g; the gate reads
  g after a wait that covers it; every kernel touches memory after its earlier users;
* the same block without the gate's wait is flagged;
* an eager step projects g on the current stream right before the gate, and logs no MLA line.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_helix8_fork_q_recorder", Path(__file__).with_name("test_rubin_helix8_fork_q.py")
)
fq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fq)

pytestmark = pytest.mark.cpu_only

_LINE_PREFIX = "Kimi K3.1 Rubin MLA:"


class _ClockPlan(fq._Plan):
    """The fork_q plan, which also keeps what each kernel's stream had waited for when it ran."""

    def __init__(self):
        super().__init__()
        self.waited: list = []  # per kernel, its stream's clock at that kernel

    def kernel(self, name: str, payload) -> None:
        super().kernel(name, payload)
        self.waited.append(dict(self.clock[self.stream]))


def _run(path: str, tree: ast.Module | None = None, cuda_graph: bool = True):
    """One ``mla_generate`` call of ``tree`` (modeling.py by default): (plan, layer, x, log)."""
    plan = _ClockPlan()
    streams = {s: fq._Stream(plan, s) for s in fq._Plan.STREAMS}
    log: list = []
    logger = SimpleNamespace(info_once=lambda *msg, key: log.append((" ".join(msg), key)))
    namespace = {"torch": fq._Torch(plan, streams), "logger": logger, **fq._entries(plan)}
    block = fq._block(tree or fq._tree(), namespace)
    st, rt = fq._cpu_step(path, plan, streams, cuda_graph)
    layer, x = fq._cpu_layer(), torch.empty(fq._ROWS, fq._HIDDEN, dtype=torch.bfloat16)
    with fq._Record(plan):
        out = block["mla_generate"](layer, fq._LAYER, x, st, rt)
    assert tuple(out.shape) == (fq._ROWS, fq._HIDDEN)
    return plan, layer, x, log


def _index(plan: _ClockPlan, name: str, weight=None) -> list:
    """The positions in ``plan.kernels`` of the kernels called ``name`` (with ``weight`` second)."""
    found = []
    for k, (_, _, kname, (args, _, _)) in enumerate(plan.kernels):
        if kname == name and (weight is None or (len(args) > 1 and args[1] is weight)):
            found.append(k)
    return found


def _without_the_gate_wait(tree: ast.Module) -> ast.Module:
    """``tree`` with ``_mla_output`` reading the fork stream's g without waiting for its event."""
    tree = copy.deepcopy(tree)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mla_output"]
    removed = []

    class _Drop(ast.NodeTransformer):
        def visit_Expr(self, node):
            if ast.unparse(node) == "ready.wait()":
                removed.append(node)
                return None
            return node

    _Drop().visit(fn)
    assert len(removed) == 1, "_mla_output has no ready.wait()"
    return ast.fix_missing_locations(tree)


@pytest.mark.parametrize("path", ["dense", "key_shard"])
def test_a_graph_step_projects_g_on_the_fork_stream_after_the_rope_kernel(path):
    plan, layer, x, log = _run(path)
    assert not plan.problems, "\n".join(plan.problems)
    (g,) = _index(plan, "rubin_linear", layer.g)
    (rope,) = _index(plan, "mla_rope_generation")
    at_g, stream_g, _, (args, _, _) = plan.kernels[g]
    at_rope, stream_rope, _, _ = plan.kernels[rope]
    assert stream_g == "fork" and stream_rope == "fork", (stream_g, stream_rope)
    assert at_rope < at_g, "g is not after the RoPE kernel on the fork stream"
    assert args[0] is x and args[2] is False, "g is not g_proj(x) in BF16"
    fork_kernels = [name for _, s, name, _ in plan.kernels if s == "fork"]
    assert fork_kernels[-1] == "rubin_linear", fork_kernels
    # The MLA line only: the indexer logs lines of its own.
    mla = [m for m in log if m[0].startswith(_LINE_PREFIX)]
    assert len(mla) == 1 and mla[0][0] == mla[0][1], log


@pytest.mark.parametrize("path", ["dense", "key_shard"])
def test_the_attention_does_not_wait_for_g_and_the_gate_does(path):
    plan, layer, _, _ = _run(path)
    (g,) = _index(plan, "rubin_linear", layer.g)
    (rope,) = _index(plan, "mla_rope_generation")
    (attention,) = _index(plan, "thop_attention")
    (gate,) = _index(plan, "k31_sigmoid_gate_mul")
    at_g, at_rope = plan.kernels[g][0], plan.kernels[rope][0]
    assert plan.kernels[attention][1] == "main" and plan.kernels[gate][1] == "main"
    seen = plan.waited[attention].get("fork", 0)
    assert at_rope < seen < at_g, f"the attention waited for the fork stream up to {seen}"
    assert plan.waited[gate].get("fork", 0) > at_g, "the gate reads g before g's event"
    gate_args, g_out = plan.kernels[gate][3][0], plan.kernels[g][3][2]
    assert gate_args[1] is g_out, "the gate does not read the fork stream's g"
    assert plan.clock["main"]["fork"] > at_g, "the fork stream is not joined"


def test_the_check_flags_a_gate_without_its_wait():
    plan, _, _, _ = _run("key_shard", _without_the_gate_wait(fq._tree()))
    flagged = [p for p in plan.problems if p.startswith("k31_sigmoid_gate_mul on the main stream")]
    assert flagged, plan.problems


@pytest.mark.parametrize("path", ["dense", "key_shard"])
def test_an_eager_step_projects_g_right_before_the_gate(path):
    plan, layer, x, log = _run(path, cuda_graph=False)
    assert {s for _, s, _, _ in plan.kernels} == {"main"} and not plan.problems
    (g,) = _index(plan, "rubin_linear", layer.g)
    (gate,) = _index(plan, "k31_sigmoid_gate_mul")
    assert gate == g + 1, [name for _, _, name, _ in plan.kernels[g:]]
    assert plan.kernels[g][3][0][0] is x
    assert not [m for m in log if m[0].startswith(_LINE_PREFIX)], log
