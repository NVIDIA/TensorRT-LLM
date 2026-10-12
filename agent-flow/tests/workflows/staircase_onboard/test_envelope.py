# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the serving envelope and the feature matrix (``envelope.py``).

The fixture spec is the shape of the DeepSeek-V4.1 onboarding's gates: boot at
``max_num_tokens`` 2048 / ``max_batch_size`` 8 / ``max_seq_len`` 16384 with
graphs and no padding, accuracy at 8192 / 32 / 16384 with padded graphs to 32,
a kv_quant witness at batch 1. That run took "the tuples the gates launch" as
its certified set, and three crashes lived just above those tuples. The tests
pin both readings: the legacy hull reproduces that set exactly, and a declared
envelope probes above it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import envelope, preflight, task_schema
from agent_flow.workflows.staircase_onboard.gates import ShellResult, resolve_runtime
from agent_flow.workflows.staircase_onboard.state import load_state

from .test_workflow import Script, _install, _workflow, _write_vocabulary, passing_shell


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _history_spec(tmp_path: Path, **overrides) -> dict:
    """The DeepSeek-V4.1 run's gate configuration, as a unified spec."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {"max_position_embeddings": 1048576},
            }
        ),
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    options = tmp_path / "scripts"
    options.mkdir(exist_ok=True)
    graph = {"enable_padding": True, "max_batch_size": 32}
    _write(
        options / "core_llm_api.yaml",
        {"max_num_tokens": 8192, "cuda_graph_config": graph, "enable_chunked_prefill": True},
    )
    _write(
        options / "dspark_llm_api.yaml",
        {
            "max_num_tokens": 8192,
            "cuda_graph_config": graph,
            "speculative_config": {"decoding_type": "DSpark", "max_draft_len": 5},
        },
    )
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "unified",
        "members": {"default": ["tep4", "dep4", "tp4"]},
        "features": ["dspark", "kv_quant"],
        "trtllm_repo_path": str(repo),
        "runtime": {
            "default": {
                "cuda_graph": True,
                "chunked_prefill": True,
                "extra_flags": [
                    "--max_seq_len=16384",
                    "--max_num_tokens=2048",
                    "--max_batch_size=8",
                    "--kv_cache_fraction=0.5",
                ],
            }
        },
        "environment": {
            "kind": "local",
            "python": "/usr/bin/python3",
            "gpus_per_node": 4,
            "env": {"TMPDIR": "/scratch/tmp", "CLAUDE_CODE_TMPDIR": "/scratch/tmp"},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "llm_api_options": {
                "default": str(options / "dspark_llm_api.yaml"),
                "core": str(options / "core_llm_api.yaml"),
            },
            "max_seq_len": 16384,
            "max_batch_size": 32,
            "witness": {
                "kv_quant": {
                    "contrast_env": {"KV_STORAGE": "bf16"},
                    "expected_bytes_per_token": 890,
                    "max_batch_size": 1,
                }
            },
        },
    }
    spec.update(overrides)
    return spec


#: What the DeepSeek-V4.1 target is deployed at, which no gate reached.
DECLARED = {
    "max_batch_size": 256,
    "max_num_tokens": 16384,
    "max_seq_len": 1048576,
    "max_draft_len": 5,
    "cuda_graph": {"padding": True},
    "members": ["tep4", "dep4", "tp4", "dep8", "tep4pp2"],
}


def _load(tmp_path: Path, spec: dict) -> dict:
    return task_schema.load_and_validate_task_yaml(_write(tmp_path / "task.yaml", spec))


def _kinds(unit: str) -> set[str]:
    kinds = {"boot", "accuracy"}
    if unit == "kv_quant":
        kinds.add("witness")
    if unit == "dspark":
        kinds.add("acceptance")
    return kinds


def _plan(tmp_path: Path, spec: dict) -> envelope.CertificationPlan:
    data = _load(tmp_path, spec)
    config = json.loads((Path(data["checkpoint_path"]) / "config.json").read_text())
    return envelope.derive_certification(
        data, world_size=4, gpus_per_node=4, checkpoint_config=config, unit_kinds=_kinds
    )


# --------------------------------------------------------------- primitives


def test_the_graph_generator_mirrors_the_engine():
    assert envelope.generate_graph_batch_sizes(8, False) == list(range(1, 9))
    assert envelope.generate_graph_batch_sizes(32, True) == [1, 2, 4, 8, 16, 24, 32]
    padded = envelope.generate_graph_batch_sizes(256, True)
    assert padded[-3:] == [128, 192, 256]
    assert envelope.generate_graph_batch_sizes(300, False)[-2:] == [256, 300]


def test_envelope_members_carry_pipeline_and_node_count():
    member = envelope.parse_envelope_member("tep4pp2")
    assert (member.kind, member.size, member.pp, member.world_size) == ("tep", 4, 2, 8)
    assert member.nodes(4) == 2
    assert "PP2" in member.describe(4) and "2 nodes" in member.describe(4)
    assert envelope.parse_envelope_member("dep8").attention_dp
    for bad in ("unified", "tp", "pp2", "tep4p2", "ep4"):
        with pytest.raises(ValueError):
            envelope.parse_envelope_member(bad)


# ------------------------------------------------------------------- legacy


def test_without_an_envelope_the_certified_set_is_the_gate_hull(tmp_path):
    """The legacy reading, stated: exactly R30's set, and nothing above it."""
    plan = _plan(tmp_path, _history_spec(tmp_path))

    assert not plan.declared
    env = plan.envelope
    assert (env.max_batch_size, env.max_num_tokens, env.max_seq_len) == (32, 8192, 16384)
    # Boot captures 1..8 without padding; accuracy captures the padded list to 32.
    assert plan.certified_capture_batches == (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32)
    assert plan.probes == ()
    assert plan.external_cells == []
    assert plan.errors == ()
    note = plan.notes()[0]
    assert "no serving_envelope declared" in note and "migrate" in note
    assert "DERIVED" in plan.render()


def test_gate_tuples_are_read_from_the_runtime_block_options_and_witness(tmp_path):
    plan = _plan(tmp_path, _history_spec(tmp_path))
    by_gate = {(t.gate, t.unit, t.member): t for t in plan.tuples}

    boot = by_gate[("boot", "core", "tep4")]
    assert (boot.max_batch_size, boot.max_num_tokens, boot.max_seq_len) == (8, 2048, 16384)
    assert boot.graph_batch_sizes == tuple(range(1, 9))

    accuracy = by_gate[("accuracy", "dspark", "dep4")]
    assert (accuracy.max_batch_size, accuracy.max_num_tokens) == (32, 8192)
    assert accuracy.graph_batch_sizes == (1, 2, 4, 8, 16, 24, 32)

    # The witness runs once, at the cheapest member, at its own batch limit.
    witnesses = [t for t in plan.tuples if t.gate == "witness"]
    assert [(t.unit, t.member, t.max_batch_size) for t in witnesses] == [("kv_quant", "tep4", 1)]
    assert ("acceptance", "dspark", "tp4") in by_gate


# ----------------------------------------------------------------- declared


def test_a_declared_envelope_derives_a_ceiling_probe_per_unit(tmp_path):
    plan = _plan(tmp_path, _history_spec(tmp_path, serving_envelope=DECLARED))

    assert plan.declared and plan.errors == ()
    assert [p.unit for p in plan.probes] == ["core", "dspark", "kv_quant"]
    probe = plan.probes[0]
    # The cheapest member, at every limit of the envelope.
    assert probe.member == "tep4"
    assert (probe.max_batch_size, probe.max_num_tokens, probe.max_seq_len) == (
        256,
        16384,
        1048576,
    )
    assert probe.graph_batch_sizes[-2:] == (192, 256)
    flags = envelope.probe_flags(probe)
    assert "--max_num_tokens=16384" in flags
    assert "--cuda_graph_padding_enabled" in flags  # the envelope's padding policy
    assert any(flag.startswith("--cuda_graph_batch_sizes 1 2 4 8") for flag in flags)


def test_members_no_unit_gates_owe_external_evidence(tmp_path):
    """Multi-node and PP members are claims this allocation cannot check."""
    plan = _plan(tmp_path, _history_spec(tmp_path, serving_envelope=DECLARED))

    external = {cell.member: cell for cell in plan.external_cells}
    assert set(external) == {"dep8", "tep4pp2"}
    assert external["dep8"].features == ("dspark", "kv_quant")
    assert external["dep8"].evidence == "boot" and external["dep8"].at == "ceiling"
    assert any("tep4pp2" in note and "evidence" in note for note in plan.notes())
    assert any("spans nodes" in w for w in plan.warnings)


def test_a_gate_launched_outside_the_envelope_is_an_error(tmp_path):
    narrow = {**DECLARED, "max_num_tokens": 4096}
    plan = _plan(tmp_path, _history_spec(tmp_path, serving_envelope=narrow))
    assert any(
        "accuracy core@tep4" in e
        and "max_num_tokens=8192 > serving_envelope.max_num_tokens=4096" in e
        for e in plan.errors
    )


def test_a_gated_member_the_envelope_does_not_claim_is_an_error(tmp_path):
    spec = _history_spec(tmp_path, serving_envelope={**DECLARED, "members": ["tep4", "dep4"]})
    plan = _plan(tmp_path, spec)
    assert any("member tp4 is gated by this run but is not in" in e for e in plan.errors)


def test_an_unset_max_seq_len_reads_as_the_checkpoint_maximum(tmp_path):
    """Leaving it to the engine sizes for 1M tokens, which the envelope may not claim."""
    spec = _history_spec(tmp_path, serving_envelope={**DECLARED, "max_seq_len": 131072})
    spec["runtime"]["default"]["extra_flags"] = ["--max_num_tokens=2048", "--max_batch_size=8"]
    plan = _plan(tmp_path, spec)
    errors = [e for e in plan.errors if "boot core" in e]
    assert errors and "max_seq_len=1048576 > serving_envelope.max_seq_len=131072" in errors[0]
    assert "pass --max_seq_len" in errors[0]


def test_an_envelope_beyond_the_checkpoint_is_an_error(tmp_path):
    spec = _history_spec(tmp_path, serving_envelope={**DECLARED, "max_seq_len": 2097152})
    plan = _plan(tmp_path, spec)
    assert any("exceeds the checkpoint's max_position_embeddings" in e for e in plan.errors)


def test_a_batch_limit_the_token_budget_cannot_fill_is_a_warning(tmp_path):
    spec = _history_spec(tmp_path, serving_envelope={**DECLARED, "max_num_tokens": 1024})
    spec["runtime"]["default"]["extra_flags"] = [
        "--max_seq_len=16384",
        "--max_num_tokens=1024",
        "--max_batch_size=8",
    ]
    for name in ("core_llm_api.yaml", "dspark_llm_api.yaml"):
        path = tmp_path / "scripts" / name
        data = yaml.safe_load(path.read_text())
        data["max_num_tokens"] = 1024
        _write(path, data)
    plan = _plan(tmp_path, spec)
    assert any("256 x 6 = 1536 tokens" in w for w in plan.warnings)


# ------------------------------------------------------------------- matrix


def test_matrix_cells_are_classified_by_how_their_evidence_is_obtained(tmp_path):
    matrix = [
        {"name": "spec-acc", "features": ["dspark"], "members": ["dep4"], "evidence": "accuracy"},
        {
            "name": "spec-long",
            "features": ["dspark"],
            "members": ["tep4"],
            "evidence": ["boot", "accuracy"],
            "at": {"max_seq_len": 131072},
            "prompt_tokens": 12000,
        },
        {
            "name": "kvq-witness",
            "features": ["kv_quant"],
            "members": ["tp4"],
            "evidence": "witness",
        },
        {"name": "mn", "features": [], "members": ["dep8"], "evidence": "accuracy"},
    ]
    spec = _history_spec(tmp_path, serving_envelope=DECLARED, feature_matrix=matrix)
    plan = _plan(tmp_path, spec)
    cells = {
        (c.cell, c.member, c.evidence): c
        for c in plan.cells
        if c.cell in {m["name"] for m in matrix}
    }

    assert cells[("spec-acc", "dep4", "accuracy")].status == envelope.CELL_GATED
    long_boot = cells[("spec-long", "tep4", "boot")]
    assert long_boot.status == envelope.CELL_PROBE and long_boot.unit == "dspark"
    long_accuracy = cells[("spec-long", "tep4", "accuracy")]
    assert long_accuracy.status == envelope.CELL_EXTERNAL
    # Measured against the accuracy gate's own tuple, with only the named axis moved.
    assert "runs below B=32 T=8192 L=131072" in long_accuracy.reason
    assert "prompt>=" not in long_accuracy.reason
    assert "witness runs at tep4 only" in cells[("kvq-witness", "tp4", "witness")].reason
    assert cells[("mn", "dep8", "accuracy")].status == envelope.CELL_EXTERNAL

    point = [p for p in plan.probes if p.unit == "dspark" and p.max_seq_len == 131072]
    assert len(point) == 1
    # Only the named axis moves; the rest stay at the unit's boot tuple.
    assert (point[0].max_num_tokens, point[0].max_batch_size) == (2048, 8)
    assert point[0].prompt_tokens == 12000


def test_a_declared_cell_an_existing_gate_covers_is_not_owed_twice(tmp_path):
    matrix = [{"features": [], "members": ["tep4"], "evidence": ["boot", "accuracy"]}]
    plan = _plan(
        tmp_path, _history_spec(tmp_path, serving_envelope=DECLARED, feature_matrix=matrix)
    )
    core = [c for c in plan.cells if c.features == () and c.member == "tep4" and c.at == "gate"]
    assert {c.evidence for c in core} == {"boot", "accuracy"}
    assert all(c.status == envelope.CELL_GATED for c in core)


# ------------------------------------------------------------------- probes


def test_a_probe_is_the_boot_gate_with_only_the_limits_changed(tmp_path):
    data = _load(tmp_path, _history_spec(tmp_path, serving_envelope=DECLARED))
    plan = envelope.derive_certification(data, world_size=4, unit_kinds=_kinds)
    probe = plan.probes_for("core")[0]

    probed = envelope.probe_task_data(data, "core", probe)
    runtime = resolve_runtime(probed, "dspark")
    flags = runtime["extra_flags"]
    assert "--kv_cache_fraction=0.5" in flags
    assert not any(f.startswith("--max_num_tokens=2048") for f in flags)
    assert runtime["chunked_prefill"] is True and runtime["cuda_graph"] is True
    # Round trip: reading the probe's runtime block back gives the probe's point.
    back = envelope.runtime_tuple(runtime, gate="boot", unit="core", member="tep4")
    assert (
        back.point()
        == envelope.EngineTuple(
            gate="boot",
            unit="core",
            member="tep4",
            max_batch_size=probe.max_batch_size,
            max_num_tokens=probe.max_num_tokens,
            max_seq_len=probe.max_seq_len,
            graph_batch_sizes=probe.graph_batch_sizes,
        ).point()
    )
    # The spec itself is untouched.
    assert "--max_num_tokens=2048" in data["runtime"]["default"]["extra_flags"]


def test_a_probe_with_a_prompt_length_adds_one_long_prompt():
    probe = envelope.EngineTuple(
        gate="envelope",
        unit="core",
        member="tp1",
        max_batch_size=1,
        max_num_tokens=1,
        max_seq_len=1,
        prompt_tokens=300,
    )
    prompts = envelope.probe_prompts(("a", "b"), probe)
    assert prompts[:2] == ("a", "b") and len(prompts[2].split()) == 300


# ----------------------------------------------------------------- evidence


def test_external_cells_need_a_passing_evidence_file(tmp_path):
    plan = _plan(tmp_path, _history_spec(tmp_path, serving_envelope=DECLARED))
    workspace = tmp_path / "ws"
    checks = envelope.check_external_evidence(plan, workspace)
    assert len(checks) == 2 and not any(c.passed for c in checks)

    cell = plan.external_cells[0]
    root = workspace / envelope.EVIDENCE_DIRNAME
    root.mkdir(parents=True)
    _write(root / f"{cell.slug}.yaml", {"verdict": "fail", "tree": "abc", "log": "x.log"})
    assert not envelope.check_external_evidence(plan, workspace)[0].passed
    _write(root / f"{cell.slug}.yaml", {"verdict": "pass", "tree": "abc", "log": "x.log"})
    checks = {c.name: c for c in envelope.check_external_evidence(plan, workspace)}
    assert checks[f"matrix:{cell.slug}"].passed


# ------------------------------------------------------------------- schema


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cuda_graph": None}, "'serving_envelope.cuda_graph' is required"),
        ({"cuda_graph": {"batch_sizes": [1, 512]}}, "captures 512, over"),
        ({"cuda_graph": {}}, "needs 'batch_sizes'"),
        ({"members": ["tep4", "tep4"]}, "lists 'tep4' twice"),
        ({"members": ["unified"]}, "is not an envelope member"),
        ({"max_num_tokens": 0}, "'serving_envelope.max_num_tokens' must be an integer >= 1"),
        ({"probe_prompt_tokens": 99999}, "caps at 128 KiB"),
        ({"max_num_token": 1}, "did you mean 'serving_envelope.max_num_tokens'"),
    ],
)
def test_the_envelope_block_is_validated(tmp_path, change, message):
    block = {**DECLARED, **change}
    block = {k: v for k, v in block.items() if v is not None}
    with pytest.raises(
        task_schema.TaskSchemaError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        _load(tmp_path, _history_spec(tmp_path, serving_envelope=block))


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        ({"features": ["mtp"]}, "which this run does not build"),
        ({"members": ["tep16"]}, "does not claim"),
        ({"evidence": "perf"}, "evidence' must be one of"),
        ({"at": {"max_seq_len": 2097152}}, "is outside serving_envelope.max_seq_len"),
        ({"at": "somewhere"}, "must be 'gate', 'ceiling' or a mapping"),
        ({"atx": "gate"}, "unknown key"),
        ({"name": "a b"}, "file-name-safe"),
    ],
)
def test_matrix_cells_are_validated(tmp_path, cell, message):
    spec = _history_spec(tmp_path, serving_envelope=DECLARED, feature_matrix=[cell])
    with pytest.raises(task_schema.TaskSchemaError, match=message):
        _load(tmp_path, spec)


def test_cell_names_are_unique_because_they_name_evidence_files(tmp_path):
    cells = [{"name": "x", "features": []}, {"name": "x", "features": ["dspark"]}]
    spec = _history_spec(tmp_path, serving_envelope=DECLARED, feature_matrix=cells)
    with pytest.raises(task_schema.TaskSchemaError, match="is used by another cell"):
        _load(tmp_path, spec)


def test_a_matrix_needs_an_envelope(tmp_path):
    spec = _history_spec(tmp_path, feature_matrix=[{"features": []}])
    with pytest.raises(task_schema.TaskSchemaError, match="needs 'serving_envelope'"):
        _load(tmp_path, spec)


# ---------------------------------------------------------------- migration


def test_the_migration_block_restates_the_legacy_set_as_a_declared_one(tmp_path):
    data = _load(tmp_path, _history_spec(tmp_path))
    block = envelope.migration_block(data, world_size=4)
    assert "WIDEN IT" in block

    path = tmp_path / "task.yaml"
    path.write_text(path.read_text() + "\n" + block, encoding="utf-8")
    migrated = task_schema.load_and_validate_task_yaml(path)
    plan = envelope.derive_certification(migrated, world_size=4, unit_kinds=_kinds)
    assert plan.declared and plan.errors == ()
    # Adopted unchanged, it owes nothing the gates did not already produce.
    assert plan.external_cells == []
    env = plan.envelope
    assert (env.max_batch_size, env.max_num_tokens, env.max_seq_len) == (32, 8192, 16384)
    assert plan.certified_capture_batches == (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32)


def test_the_migrate_command_appends_once(tmp_path):
    _write(tmp_path / "task.yaml", _history_spec(tmp_path))
    path = str(tmp_path / "task.yaml")
    assert envelope._main(["migrate", path, "--write"]) == 0
    assert "serving_envelope:" in Path(path).read_text()
    assert envelope._main(["migrate", path]) == 1
    assert envelope._main(["plan", path]) == 0


# ---------------------------------------------------------------- preflight


def _smi(argv, env=None, timeout=0):
    return preflight.CommandResult(returncode=0, stdout="ok\n")


def test_preflight_refuses_an_envelope_the_gates_contradict(tmp_path):
    spec = _history_spec(tmp_path, serving_envelope={**DECLARED, "max_num_tokens": 4096})
    data = _load(tmp_path, spec)
    with pytest.raises(
        preflight.SpecInconsistencyError, match="serving envelope and the gates disagree"
    ):
        preflight.run_preflight(data, runner=_smi, skip_device_claim=True)


def test_preflight_carries_the_plan_and_prints_the_bill(tmp_path):
    data = _load(tmp_path, _history_spec(tmp_path, serving_envelope=DECLARED))
    result = preflight.run_preflight(data, runner=_smi, skip_device_claim=True)
    assert result.certification is not None and result.certification.declared
    joined = "\n".join(result.notes)
    assert "serving envelope: B<=256 T<=16384 L<=1048576" in joined
    assert "envelope probe(s)" in joined


# ----------------------------------------------------------------- workflow


@pytest.fixture
def enveloped_task(tmp_path, task_file):
    spec = yaml.safe_load(task_file.read_text())
    spec["runtime"] = {"extra_flags": ["--max_seq_len=4096", "--max_batch_size=4"]}
    spec["accuracy"].update(max_seq_len=4096, max_batch_size=4)
    spec["serving_envelope"] = {
        "max_batch_size": 64,
        "max_num_tokens": 16384,
        "max_seq_len": 32768,
        "cuda_graph": {"batch_sizes": [1, 2, 4, 8, 16, 32, 64]},
        "members": ["tp1", "tp2"],
    }
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_file


def _recording_shell(commands, *, fail_probe: int = 0):
    inner = passing_shell()
    failures = {"left": fail_probe}

    def runner(command, env=None, timeout=0):
        commands.append(command)
        if "--max_num_tokens=16384" in command and failures["left"]:
            failures["left"] -= 1
            return ShellResult(returncode=1, stdout="", stderr="index out of bounds in warm-up")
        return inner(command, env, timeout)

    return runner


def _with_vocabulary(monkeypatch, workflow, tmp_path):
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)


def test_the_workflow_probes_the_ceiling_and_owes_the_unclaimed_member(
    tmp_path, enveloped_task, monkeypatch
):
    commands: list[str] = []
    workflow = _workflow(tmp_path, shell_runner=_recording_shell(commands))
    calls = _install(monkeypatch, workflow, Script(None))
    _with_vocabulary(monkeypatch, workflow, tmp_path)
    with workflow:
        workflow.run(enveloped_task)

    probes = [c for c in commands if "--max_num_tokens=16384" in c]
    assert len(probes) == 1
    assert "--max_seq_len=32768" in probes[0] and "--max_batch_size=64" in probes[0]
    assert "--cuda_graph_batch_sizes 1 2 4 8 16 32 64" in probes[0]
    assert "--use_cuda_graph" in probes[0]

    workspace = tmp_path / "ws"
    assert "tp2" in (workspace / "certification.md").read_text()
    drafter = next(prompt for role, prompt in calls if role == "plan_drafter")
    assert "## Serving envelope (declared)" in drafter
    # tp2 is claimed and gated by nothing, so delivery stops on it.
    state = load_state(workspace / ".staircase_onboard_state.json")
    assert state.verdict == "delivery-failed"
    assert "matrix:envelope-member__tp2__boot__ceiling" in (workspace / "journal.md").read_text()


def test_a_failed_probe_sends_the_unit_back_to_the_assembler(tmp_path, enveloped_task, monkeypatch):
    commands: list[str] = []
    workflow = _workflow(tmp_path, shell_runner=_recording_shell(commands, fail_probe=1))
    calls = _install(monkeypatch, workflow, Script(None))
    _with_vocabulary(monkeypatch, workflow, tmp_path)
    evidence = tmp_path / "ws" / envelope.EVIDENCE_DIRNAME
    evidence.mkdir(parents=True)
    _write(
        evidence / "envelope-member__tp2__boot__ceiling.yaml",
        {"verdict": "pass", "tree": "c0ffee1", "log": "/logs/tp2.log"},
    )
    with workflow:
        workflow.run(enveloped_task)

    assert [role for role, _ in calls].count("assembler") == 2
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "[FAIL] envelope@tp1: at B=64 T=16384 L=32768" in journal
    assert load_state(tmp_path / "ws" / ".staircase_onboard_state.json").verdict == "delivered"


def test_the_plan_roles_judge_coverage_against_the_envelope():
    from agent_flow.workflows.staircase_onboard.prompts import DEFAULT_PROMPTS

    assert "serving envelope" in DEFAULT_PROMPTS.plan_drafter
    assert "is `CERT`, not `OK`" in DEFAULT_PROMPTS.plan_drafter
    assert "serving envelope" in DEFAULT_PROMPTS.plan_reviewer
