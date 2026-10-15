# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the smith brief (``smith_brief.py``) and the kernel-verify skill's probes.

The fixture is the DeepSeek-V4.1 delivery audit that came back UNCOVERED on
seven entries for one reason: capture lists held ``G`` and ``6·G`` while the
target also reached those ops at ``3·G`` (ratio-2 compressed rows) and ``5·G``
(rope's q form), ``dep4`` saw ``4·G`` / ``24·G`` after the gather, and one call
site (the decode-replay tail) only ever ran on context rows. The capture
batches are that run's: boot ``1..8`` unpadded plus the accuracy gate's padded
list to 32.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_flow.workflows.staircase_onboard import envelope, smith_brief
from agent_flow.workflows.staircase_onboard.vocabulary import QueueItem

from .test_envelope import _history_spec, _load
from .test_workflow import Script, _install, _workflow, _write_vocabulary

STEPS = [
    {
        "step": "compressed latent, ratio 2",
        "state": "CERT",
        "entry": "catalog/attention/compressor.py",
        "call_sites": [
            {"site": "Compressor.forward, verify step", "row_multiple": 3, "row_width": 512},
            {"site": "_TailSlots, decode-replay tail", "eager_only": True},
        ],
    },
    {
        "step": "rope, q form",
        "state": "OK",
        "entry": "catalog/rope/rope_q.py",
        "call_sites": [{"site": "drafter q", "row_multiple": 5}],
    },
    {
        "step": "routed experts",
        "state": "CERT",
        "entry": "catalog/moe/nvfp4_runner.py",
        "call_sites": [
            {"site": "MoE after the DP gather, decode", "row_multiple": 1, "gathered": True},
            {"site": "MoE after the DP gather, verify", "row_multiple": 6, "gathered": True},
        ],
    },
]


def _plan(tmp_path: Path, **envelope_block) -> envelope.CertificationPlan:
    spec = _history_spec(tmp_path)
    if envelope_block:
        spec["serving_envelope"] = envelope_block
    return envelope.derive_certification(_load(tmp_path, spec), world_size=4)


def _item(target: str, n: int = 75, mode: str = "verify") -> QueueItem:
    return QueueItem(n=n, mode=mode, target=target)


def test_each_call_site_states_its_row_multiple_over_every_captured_batch(tmp_path):
    brief = smith_brief.render_smith_brief(
        _item("attention/compressor"), steps=STEPS, plan=_plan(tmp_path)
    )
    # The legacy run's capture batches: 1..8 and 16, 24, 32.
    assert "captured batch sizes G: 1..8,16,24,32" in brief
    assert "| Compressor.forward, verify step | 3·G | 3,6,9,12,15,18,21,24,48,72,96 | - |" in brief
    assert "| _TailSlots, decode-replay tail | eager-only | never captured | - |" in brief
    assert "every eager row count 1..8192" in brief


def test_a_gathered_site_is_multiplied_by_every_attention_dp_width(tmp_path):
    brief = smith_brief.render_smith_brief(
        _item("moe/nvfp4_runner"), steps=STEPS, plan=_plan(tmp_path)
    )
    assert "dep4: 4·G = 4,8,12,16,20,24,28,32,64,96,128" in brief
    assert "dep4: 24·G = 24,48,72,96,120,144,168,192,384,576,768" in brief
    # Eager rows after the gather reach the gathered token budget.
    assert "every eager row count 1..32768" in brief


def test_the_declared_envelope_widens_the_rows_and_the_probes_flag_reachable_ones(tmp_path):
    plan = _plan(
        tmp_path,
        max_batch_size=256,
        max_num_tokens=16384,
        max_seq_len=1048576,
        max_draft_len=5,
        cuda_graph={"padding": True},
        members=["tep4", "dep4", "tp4", "dep8"],
    )
    brief = smith_brief.render_smith_brief(_item("moe/nvfp4_runner"), steps=STEPS, plan=plan)
    assert "dep8: 48·G" in brief and "12288" in brief  # 48 x 256
    assert "every eager row count 1..131072" in brief  # 16384 x 8 after the dep8 gather

    compressor = smith_brief.render_smith_brief(_item("compressor"), steps=STEPS, plan=plan)
    assert (
        "2^31 / 512 = 4194304 rows, probe at 4194305 (the envelope reaches 16384 rows)"
        in compressor
    )
    assert "2^24 + 1" in compressor and "2^31" in compressor


def test_without_declared_call_sites_the_brief_assumes_decode_and_verify(tmp_path):
    plan = _plan(
        tmp_path,
        max_batch_size=32,
        max_num_tokens=8192,
        max_seq_len=16384,
        max_draft_len=5,
        cuda_graph={"padding": True},
        members=["tep4", "dep4", "tp4"],
    )
    brief = smith_brief.render_smith_brief(_item("norm/rmsnorm"), steps=STEPS, plan=plan)
    assert "ASSUMED" in brief
    assert "| decode step | G |" in brief and "| speculative verify step | 6·G |" in brief
    assert "read them from the entry's __call__" in brief


def test_malformed_call_sites_are_named(tmp_path):
    steps = [
        {
            "step": "x",
            "entry": "catalog/a/b.py",
            "call_sites": [
                {"site": "s", "row_multiple": 0},
                {"row_multiple": 2},
                {"site": "t", "rows": 3},
            ],
        }
    ]
    sites, errors = smith_brief.parse_call_sites(steps[0])
    assert [s.site for s in sites] == ["t"]
    assert len(errors) == 3
    brief = smith_brief.render_smith_brief(_item("a/b"), steps=steps, plan=_plan(tmp_path))
    assert "Malformed `call_sites`" in brief


def test_entry_keys_match_every_spelling():
    for spelled in (
        "catalog/attention/compressor.py",
        "attention/compressor",
        "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/compressor.py",
    ):
        assert smith_brief.entry_key(spelled) == "attention/compressor"
    assert len(smith_brief.steps_for(STEPS, "compressor")) == 1
    assert smith_brief.steps_for(STEPS, "other/compressor") == []
    # A target's own entry keeps its ops/ prefix: it is not the shared entry.
    for spelled in (
        "ops/moe/router.py",
        "tensorrt_llm/_torch/_experimental/modeling_v2/models/fam/ckpt__sm_103__tp1/ops/moe/router",
    ):
        assert smith_brief.entry_key(spelled) == "ops/moe/router"


def test_the_brief_carries_the_queue_row(tmp_path):
    row = {"n": 75, "tier": 1, "unit": "dspark", "surface": "rows 3G", "note": "R20 union"}
    brief = smith_brief.render_smith_brief(
        _item("attention/compressor"), steps=STEPS, plan=None, queue_row=row
    )
    for line in ("- tier: 1", "- unit: dspark", "- surface: rows 3G", "- note: R20 union"):
        assert line in brief
    # No plan: no envelope section, and the probes are still owed.
    assert "### Envelope" not in brief and "Integer-width boundary probes" in brief


def test_the_command_prints_the_brief_for_a_written_dispatch(tmp_path, capsys):
    workspace = tmp_path / "ws"
    (workspace / "dispatch").mkdir(parents=True)
    (workspace / "dispatch" / "075-smith.prompt.md").write_text(
        "Dispatch 75: **verify** the catalog entry `attention/compressor`.\n", encoding="utf-8"
    )
    (workspace / "vocabulary.yaml").write_text(
        yaml.safe_dump({"steps": STEPS, "queue": [{"n": 75, "tier": 1, "unit": "core"}]}),
        encoding="utf-8",
    )
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(_history_spec(tmp_path)), encoding="utf-8")
    (workspace / "task.yaml").write_text(task.read_text(), encoding="utf-8")

    assert smith_brief._main(["--workspace", str(workspace), "075"]) == 0
    out = capsys.readouterr().out
    assert "dispatch 75, verify `attention/compressor`" in out
    assert "- unit: core" in out and "3·G" in out
    assert smith_brief._main(["--workspace", str(workspace), "999"]) == 2


def test_every_smith_prompt_carries_the_brief(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    queue = [{"n": 1, "mode": "verify", "target": "flashinfer_rmsnorm", "surface": "T=3"}]
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", queue)
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    smith = next(prompt for role, prompt in calls if role == "smith")
    assert "## Smith brief: dispatch 1, verify `flashinfer_rmsnorm`" in smith
    assert "### Envelope (DERIVED from the gate tuples (legacy))" in smith
    assert "Integer-width boundary probes (mandatory)" in smith
    # An `ops/<category>/<name>` dispatch lives in this target's own ops/.
    assert "\nTarget: tensorrt_llm/_torch/_experimental/modeling_v2/models/" in smith


def test_the_verify_skill_and_the_prompts_carry_the_probes():
    from agent_flow.workflows.staircase_onboard.prompts import DEFAULT_PROMPTS

    skills = Path(smith_brief.__file__).parent / "skills"
    verify = (skills / "staircase-kernel-verify" / "SKILL.md").read_text()
    assert "## Integer-width boundary probes (mandatory)" in verify
    assert "2^31 / row width rows" in verify and "2^24 + 1" in verify
    probes = (
        skills / "staircase-kernel-verify" / "references" / "integer-width-probes.md"
    ).read_text()
    assert "## Hard limits" in probes
    entry_spec = (skills / "staircase-kernel-onboard" / "references" / "entry-spec.md").read_text()
    assert "## Hard limits" in entry_spec
    assert "## The smith brief" in DEFAULT_PROMPTS.smith
    assert "call_sites" in DEFAULT_PROMPTS.plan_drafter
    assert "Captured rows come per call site" in DEFAULT_PROMPTS.auditor
