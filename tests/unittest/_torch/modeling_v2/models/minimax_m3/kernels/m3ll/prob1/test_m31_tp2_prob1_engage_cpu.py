# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_PROB1 engagement on the TP2 target: its kill switch, the collective decline and the engagement lines,
with CPU doubles.

- The switch is the TP2 target's registered ``DRAFT_PROB1`` item, read once at import through the target's strict
  registry: set / unset words toggle it, an unknown word fails the import, an unregistered id is ignored.
- A target-level decline still takes part in the rank agreement and allocates nothing.
- Engagement and decline lines are logged once per key.
"""

import ast
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[9]
BASE = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
TP2 = BASE / "minimax_m3_1_flash_nvfp4__sm_107__tp2"
OFF = "off (TRTLLM_M31_DISABLE_DRAFT_PROB1 set)"


def _exec_nodes(path, keep, namespace):
    tree = ast.parse(path.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)]
    body += [n for n in tree.body if keep(n)]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _not_the_logger_import(node):
    return not (isinstance(node, ast.ImportFrom) and node.module == "tensorrt_llm.logger")


def _prob1_off_reason(env):
    """Imports the TP2 switch registry and the worker's switch read under ``env`` (by path, no package import)."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_M31_DISABLE_")}
    clean.update(env)
    warnings = []
    logger = types.SimpleNamespace(warning=warnings.append, info=warnings.append)
    with patch.dict(os.environ, clean, clear=True):
        switches = _exec_nodes(TP2 / "m31_switches.py", _not_the_logger_import, {"logger": logger})

        def keep(node):
            if isinstance(node, ast.FunctionDef):
                return node.name == "prob1_off_reason"
            return isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_PROB1_KILLED" for t in node.targets
            )

        ns = _exec_nodes(
            TP2 / "dspark_markov_worker.py",
            keep,
            {"m31_disabled": switches["m31_disabled"], "Optional": object},
        )
    return ns["prob1_off_reason"](), warnings


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, None),
        ({"TRTLLM_M31_DISABLE_DRAFT_PROB1": "0"}, None),
        ({"TRTLLM_M31_DISABLE_DRAFT_PROB1": "false"}, None),
        ({"TRTLLM_M31_DISABLE_DRAFT_PROB1": ""}, None),
        ({"TRTLLM_M31_DISABLE_DRAFT_PROB1": "on"}, OFF),
        ({"TRTLLM_M31_DISABLE_DRAFT_PROB1": " 1 "}, OFF),
        ({"TRTLLM_M31_DISABLE_DRAFT_PROB1": "YES"}, OFF),
    ],
)
def test_tp2_kill_switch_read_at_import(env, expected):
    assert _prob1_off_reason(env)[0] == expected


def test_an_unknown_word_fails_the_import():
    with pytest.raises(ValueError, match="TRTLLM_M31_DISABLE_DRAFT_PROB1='2'"):
        _prob1_off_reason({"TRTLLM_M31_DISABLE_DRAFT_PROB1": "2"})


def test_an_unregistered_id_is_ignored_with_a_warning():
    reason, warnings = _prob1_off_reason({"TRTLLM_M31_DISABLE_PROB1_ENGAGE": "1"})
    assert reason is None
    assert any("TRTLLM_M31_DISABLE_PROB1_ENGAGE is not a switch" in w for w in warnings)


def _worker_namespace(records):
    logger = types.SimpleNamespace(
        info=lambda line: records.append(line), warning=lambda line: records.append(line)
    )
    prob1 = types.SimpleNamespace(TAG="[P1]", supported=lambda *args: None)
    eng = _exec_nodes(
        BASE / "kernels/m3ll/prob1/draft_prob1.py",
        lambda n: isinstance(n, ast.FunctionDef) and n.name == "engages",
        {"MAX_ROWS": 8},
    )
    prob1.engages = eng["engages"]
    ns = {"prob1": prob1, "logger": logger, "_RNG_SLOT_SPAN": 32, "torch": None}
    return _exec_nodes(
        BASE / "ops/prob1_worker.py", lambda n: isinstance(n, ast.FunctionDef), ns
    ), prob1


def test_target_off_declines_collectively_without_allocation():
    records = []
    ns, prob1 = _worker_namespace(records)
    created = []
    prob1.DraftProb1 = types.SimpleNamespace(
        create=lambda *a: created.append(a) or types.SimpleNamespace(describe=lambda: "ready")
    )
    dspark = types.ModuleType("tensorrt_llm._torch.speculative.dspark")
    dspark._ANCESTRAL_BLOCK_OVERRIDE = None
    dspark.set_ancestral_block_override = lambda h: setattr(dspark, "_ANCESTRAL_BLOCK_OVERRIDE", h)
    spec = types.ModuleType("tensorrt_llm._torch.speculative")
    spec.dspark = dspark
    ops = types.ModuleType("tensorrt_llm._torch.distributed.ops")
    ops._get_mnnvl_workspace_comm = lambda m: "comm"
    mapping = types.SimpleNamespace(tp_size=2, tp_rank=1)
    with patch.dict(sys.modules, {spec.__name__: spec, ops.__name__: ops}):
        for off, expected in [(OFF, False), (None, True)]:
            votes = []
            w = types.SimpleNamespace(
                _s4_logits_on=True, _s4_ranks_agree=lambda v: votes.append(v) or v
            )
            ns["prebuild"](w, mapping, "W1", "W2", 0, 200064, 7, off)
            # The decline still takes part in the rank agreement (no peer is left in a collective alone).
            assert votes == [expected]
            assert (w._p1 is not None) == expected and len(created) == int(expected)
            assert (dspark._ANCESTRAL_BLOCK_OVERRIDE is ns["draw_ancestral_block"]) == expected
            if not expected:
                assert any("TRTLLM_M31_DISABLE_DRAFT_PROB1" in r for r in records)


def test_engagement_lines_once_per_key():
    records = []
    ns, _ = _worker_namespace(records)
    w = types.SimpleNamespace(_p1=object())
    for _ in range(3):
        ns["_log_once"](w, "taken", "taken line")
        assert ns["_ancestral_decline"](w, "outside the S4-logits step") is None
    assert records.count("taken line") == 1
    assert sum("declined (outside the S4-logits step)" in r for r in records) == 1
    # Without a prepared PROB1 the stock loop is the configured path: no decline line.
    quiet = types.SimpleNamespace(_p1=None)
    before = len(records)
    assert ns["_ancestral_decline"](quiet, "S4 / PROB1 eligibility") is None
    assert len(records) == before
