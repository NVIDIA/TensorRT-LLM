# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the static capacity-constant audit (``capacity_audit.py``).

The synthetic target reproduces the shape of the DeepSeek-V4.1 defect: a
module-level ``_TOPK_MAX_ROWS = 8192`` sizing a constant offsets buffer at
construction, which killed every engine above ``max_num_tokens`` 8192 in
warm-up. Under the gates' hull (8192) it passes, which is how it shipped;
under the declared envelope (16384) it must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import capacity_audit as audit
from agent_flow.workflows.trtllm_modeling_v2 import envelope
from agent_flow.workflows.trtllm_modeling_v2.state import load_state

from .test_delivery import TARGET_REL
from .test_workflow import Script, _install, _workflow, _write_vocabulary

_TOPK = '''\
"""An indexer with a fixed-size offsets buffer."""

import torch

_TOPK_MAX_ROWS = 8192{binding}
_HEAD_DIM = 128
_BLOCK = 64


class TopK:
    def __init__(self, model_config):
        self._offsets = torch.arange(_TOPK_MAX_ROWS + 1, dtype=torch.int32)
        self._rows = torch.empty(model_config.max_num_tokens, _HEAD_DIM)

    def forward(self, scores, n):
        assert n <= _TOPK_MAX_ROWS
        return scores
'''

DECLARED = envelope.ServingEnvelope(
    max_batch_size=256, max_num_tokens=16384, max_seq_len=1048576, members=()
)
GATE_HULL = envelope.ServingEnvelope(
    max_batch_size=32, max_num_tokens=8192, max_seq_len=16384, declared=False
)


def _target(tmp_path: Path, body: str, name: str = "topk.py") -> Path:
    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    (target / name).write_text(body, encoding="utf-8")
    return target


def _by_name(result: audit.CapacityAudit) -> dict[str, audit.CapacityFinding]:
    return {f.name: f for f in result.findings}


def test_an_unbound_capacity_constant_fails_whatever_the_envelope(tmp_path):
    target = _target(tmp_path, _TOPK.format(binding=""))
    result = audit.audit_target(target, DECLARED)

    finding = _by_name(result)["_TOPK_MAX_ROWS"]
    assert not finding.passed
    assert "unbound capacity" in finding.detail
    assert "sizes a buffer" in finding.detail and "used as a ceiling" in finding.detail
    # A small dimension is not a capacity; an engine-sized buffer is tied.
    assert "_HEAD_DIM" not in _by_name(result) and "_BLOCK" not in _by_name(result)
    assert len(result.tied) == 0  # the engine-sized buffer also reads _HEAD_DIM
    assert not result.passed


def test_the_bound_constant_passes_the_gate_hull_and_fails_the_envelope(tmp_path):
    """Exactly how it shipped: inside what the gates ran, outside what is served."""
    target = _target(tmp_path, _TOPK.format(binding="  # capacity: max_num_tokens"))

    hull = audit.audit_target(target, GATE_HULL)
    assert _by_name(hull)["_TOPK_MAX_ROWS"].passed
    assert "DERIVED envelope" in hull.render()

    declared = audit.audit_target(target, DECLARED)
    finding = _by_name(declared)["_TOPK_MAX_ROWS"]
    assert not finding.passed
    assert "8192 < max_num_tokens = 16384" in finding.detail
    checks = declared.checks()
    assert [c.name for c in checks] == ["capacity:_TOPK_MAX_ROWS"]
    assert "topk.py:5" in checks[0].detail


def test_a_capacity_sized_from_the_engine_needs_no_binding(tmp_path):
    body = """\
import torch


class TopK:
    def __init__(self, model_config):
        self._offsets = torch.arange(model_config.max_num_tokens + 1)
        self._slots = torch.zeros(self.max_batch_size, 4)
"""
    result = audit.audit_target(_target(tmp_path, body), DECLARED)
    assert result.findings == () and len(result.tied) == 2
    assert result.passed
    assert [c.name for c in result.checks()] == ["capacity:audit"]


def test_bindings_in_the_bounds_mapping_and_their_factors(tmp_path):
    body = """\
import torch

CAPACITY_BOUNDS = {
    "_GATHER_ROWS": "max_num_tokens * world_size",
    "Pool.SLOTS": "max_batch_size",
    "_GONE": "max_seq_len",
}
_GATHER_ROWS = 65536
_WINDOW_LEN = 128  # capacity: fixed: the checkpoint's sliding window


class Pool:
    SLOTS = 128
"""
    dep = envelope.ServingEnvelope(
        max_batch_size=256,
        max_num_tokens=16384,
        max_seq_len=131072,
        members=(envelope.parse_envelope_member("dep4"),),
    )
    findings = _by_name(audit.audit_target(_target(tmp_path, body), dep))

    assert findings["_GATHER_ROWS"].passed
    assert "65536 >= max_num_tokens * world_size = 65536" in findings["_GATHER_ROWS"].detail
    assert not findings["Pool.SLOTS"].passed  # 128 < 256
    assert findings["_WINDOW_LEN"].passed and "sliding window" in findings["_WINDOW_LEN"].detail
    assert "does not define" in findings["_GONE"].detail

    wider = envelope.ServingEnvelope(
        max_batch_size=256,
        max_num_tokens=16384,
        max_seq_len=131072,
        members=(envelope.parse_envelope_member("dep8"),),
    )
    assert not _by_name(audit.audit_target(_target(tmp_path, body), wider))["_GATHER_ROWS"].passed


def test_a_large_literal_buffer_needs_a_binding(tmp_path):
    body = """\
import torch

_WORKSPACE = torch.empty(1 << 20, dtype=torch.uint8)
_SCRATCH = torch.empty(2048)  # capacity: fixed: one page of host scratch
_SMALL = torch.empty(16)
"""
    result = audit.audit_target(_target(tmp_path, body), DECLARED)
    buffers = [f for f in result.findings if f.kind == "buffer"]
    assert [(f.value, f.passed) for f in buffers] == [(1 << 20, False), (2048, True)]
    assert "literal size at import" in buffers[0].detail


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("fixed", "needs a reason"),
        ("max_tokens", "not one of"),
        ("max_num_tokens * two", "is not an integer"),
    ],
)
def test_malformed_bindings_fail_with_the_reason(text, error):
    binding = audit.parse_binding(text)
    assert error in binding.error


def test_an_envelope_without_the_axis_cannot_prove_the_bound(tmp_path):
    target = _target(tmp_path, "_MAX_LEN = 4096  # capacity: max_seq_len\n")
    hull = envelope.ServingEnvelope(max_batch_size=8, max_num_tokens=2048, max_seq_len=None)
    finding = _by_name(audit.audit_target(target, hull))["_MAX_LEN"]
    assert not finding.passed and "cannot be proven" in finding.detail


def test_unparsable_files_and_bounds_are_reported(tmp_path):
    target = _target(tmp_path, "CAPACITY_BOUNDS = dict(a='b')\n")
    (target / "broken.py").write_text("def (:\n", encoding="utf-8")
    result = audit.audit_target(target, DECLARED)
    assert not result.passed
    assert any("cannot parse" in e for e in result.errors)
    assert any("literal mapping" in e for e in result.errors)


def test_the_targets_own_ops_are_left_to_their_cells(tmp_path):
    """``<target>/ops/`` entries are certified per cell, like catalog entries."""
    target = _target(tmp_path, "import torch\n")
    op_dir = target / audit.TARGET_OPS_DIR / "moe"
    op_dir.mkdir(parents=True)
    (op_dir / "runner.py").write_text(_TOPK.format(binding=""), encoding="utf-8")
    nested = target / "attention" / "ops.py"
    nested.parent.mkdir()
    nested.write_text(_TOPK.format(binding=""), encoding="utf-8")

    files = [p.relative_to(target).as_posix() for p in audit.target_files(target)]
    assert files == ["attention/ops.py", "topk.py"]
    result = audit.audit_target(target, DECLARED)
    assert [f.file for f in result.findings if f.name == "_TOPK_MAX_ROWS"] == ["attention/ops.py"]


# ----------------------------------------------------------------- workflow


def test_the_reviewer_reads_the_audit_and_delivery_fails_on_it(
    tmp_path, task_file, repo, monkeypatch
):
    (repo / TARGET_REL / "topk.py").write_text(_TOPK.format(binding=""), encoding="utf-8")
    spec = yaml.safe_load(task_file.read_text())
    spec["runtime"] = {"extra_flags": ["--max_seq_len=4096", "--max_batch_size=4"]}
    spec["accuracy"].update(max_seq_len=4096, max_batch_size=4)
    spec["serving_envelope"] = {
        "max_batch_size": 64,
        "max_num_tokens": 16384,
        "max_seq_len": 32768,
        "cuda_graph": False,
        "members": ["tp1"],
    }
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    reviewer = next(prompt for role, prompt in calls if role == "reviewer")
    assert "## Capacity constants in the target" in reviewer
    assert "| FAIL | constant | `_TOPK_MAX_ROWS` | 8192 |" in reviewer
    assert "## Serving envelope (declared)" in reviewer
    state = load_state(tmp_path / "ws" / ".trtllm_modeling_v2_state.json")
    assert state.verdict == "delivery-failed"
    assert "capacity:_TOPK_MAX_ROWS" in (tmp_path / "ws" / "journal.md").read_text()


def test_the_reviewer_and_assembler_are_told_about_capacity_constants():
    from agent_flow.workflows.trtllm_modeling_v2.prompts import DEFAULT_PROMPTS

    assert "Capacity constants against the serving envelope" in DEFAULT_PROMPTS.reviewer
    assert "# capacity: <axis>" in DEFAULT_PROMPTS.assembler
