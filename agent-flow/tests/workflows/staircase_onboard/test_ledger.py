# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the results ledger: what a verdict records, and how it is stored.

The ledger exists because a verdict without the code it was measured on can
be attributed to any code. The DeepSeek-V4.1 run's records cited evidence
from more than ten trees and none of them said which verdict belonged to
which, so a final report could call a head "16/16 PASS" that no gate had run
on. Every test here pins one thing a verdict must carry for that to be
decidable later.
"""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import threading
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import ledger as ledger_mod
from agent_flow.workflows.staircase_onboard.delivery import CATALOG_DIR
from agent_flow.workflows.staircase_onboard.gates import GateLog, GateResult
from agent_flow.workflows.staircase_onboard.ledger import (
    OUTCOME_FAIL,
    OUTCOME_FINDING,
    OUTCOME_PASS,
    Ledger,
    VerdictContext,
    blob_hash,
    command_env,
    dependency_paths,
    harvest_switch_defaults,
    native_build,
    record_gate_result,
    spec_fingerprint,
    tree_identity,
    verdict_from_result,
)

from .test_delivery import TARGET_REL, _build_repo


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def _git_repo(root: Path) -> Path:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.st")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "seed")
    return root


def _result(**fields) -> GateResult:
    fields.setdefault("name", "accuracy[gsm8k]")
    fields.setdefault("passed", True)
    fields.setdefault("detail", "ok")
    return GateResult(**fields)


# ------------------------------------------------------------------- storage


def test_a_verdict_round_trips_and_the_file_is_jsonl(tmp_path):
    ledger = Ledger.in_workspace(tmp_path)
    verdict = record_gate_result(
        ledger,
        _result(member="tep4", measured=92.5, reference=92.3, tolerance=5.0),
        VerdictContext(unit="core"),
    )
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == verdict.id
    (back,) = ledger.verdicts()
    assert back.key == ("accuracy[gsm8k]", "core", "tep4")
    assert back.value == 92.5
    assert back.threshold == pytest.approx(87.3)
    assert back.outcome == OUTCOME_PASS


def test_the_ledger_is_append_only_and_the_newest_verdict_per_key_wins(tmp_path):
    """Nothing is rewritten: a re-run adds a line, and the old one stays as history."""
    ledger = Ledger.in_workspace(tmp_path)
    context = VerdictContext(unit="core")
    record_gate_result(ledger, _result(member="tp4", passed=False, detail="bad"), context)
    record_gate_result(ledger, _result(member="tp4", passed=True), context)
    record_gate_result(ledger, _result(name="boot", member="tp4"), context)

    assert len(ledger.verdicts()) == 3
    latest = ledger.latest()
    assert [k for k in latest] == [("accuracy[gsm8k]", "core", "tp4"), ("boot", "core", "tp4")]
    assert latest[("accuracy[gsm8k]", "core", "tp4")].outcome == OUTCOME_PASS


def test_a_torn_last_line_is_skipped_and_counted(tmp_path):
    """A process killed mid-append must not make the whole ledger unreadable."""
    ledger = Ledger.in_workspace(tmp_path)
    record_gate_result(ledger, _result(), VerdictContext(unit="core"))
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "verdict", "gate": "bo')
    assert len(ledger.verdicts()) == 1
    assert ledger.skipped == 1


def test_concurrent_appends_from_threads_never_interleave(tmp_path):
    ledger = Ledger.in_workspace(tmp_path)

    def write(index: int) -> None:
        for n in range(20):
            record_gate_result(
                ledger, _result(member=f"m{index}", detail="x" * 5000), VerdictContext(unit=str(n))
            )

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(ledger.verdicts()) == 160
    assert ledger.skipped == 0


def _append_from_process(path: str, index: int) -> None:
    ledger = Ledger(path)
    for n in range(20):
        record_gate_result(
            ledger, _result(member=f"p{index}", detail="y" * 5000), VerdictContext(unit=str(n))
        )


def test_concurrent_appends_from_processes_never_interleave(tmp_path):
    """Two orchestrators, or a harness beside the run, share one ledger file."""
    path = str(tmp_path / "ledger.jsonl")
    context = multiprocessing.get_context("fork")
    procs = [context.Process(target=_append_from_process, args=(path, i)) for i in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    ledger = Ledger(path)
    assert len(ledger.verdicts()) == 80
    assert ledger.skipped == 0


# --------------------------------------------------------- outcome and values


def test_a_finding_is_its_own_outcome_and_still_passes(tmp_path):
    """The run that motivated this reported "no finding" over a FINDING verdict."""

    class Flagged:
        name = "acceptance"
        passed = True
        detail = "below the paired reference"
        finding = True

    verdict = verdict_from_result(Flagged(), VerdictContext(unit="dspark"))
    assert verdict.outcome == OUTCOME_FINDING
    assert verdict.passed


def test_an_explicit_outcome_wins_and_a_failed_gate_is_fail():
    class Explicit:
        name = "perf"
        passed = True
        detail = ""
        outcome = "finding"

    assert verdict_from_result(Explicit(), VerdictContext()).outcome == OUTCOME_FINDING
    assert verdict_from_result(_result(passed=False), VerdictContext()).outcome == OUTCOME_FAIL


def test_an_explicit_threshold_is_kept_as_measured():
    verdict = verdict_from_result(
        _result(name="perf", measured=1.03, reference=1.0, tolerance=0.1),
        VerdictContext(),
    )
    assert verdict.threshold == pytest.approx(0.9)

    class WithThreshold:
        name = "perf"
        passed = True
        detail = ""
        measured = 1.03
        threshold = 1.05

    assert verdict_from_result(WithThreshold(), VerdictContext()).threshold == 1.05


# ------------------------------------------------------------- dependencies


def test_dependencies_are_the_target_its_family_and_its_imported_catalog_entries(tmp_path):
    root = _build_repo(tmp_path)
    family = (root / TARGET_REL).parent
    (family / "shared_rope.py").write_text("X = 1\n", encoding="utf-8")
    (root / TARGET_REL / "__pycache__").mkdir()
    (root / TARGET_REL / "__pycache__" / "modeling.cpython-312.pyc").write_bytes(b"\0")

    paths = dependency_paths(root, TARGET_REL)

    assert f"{TARGET_REL}/modeling.py" in paths
    assert f"{TARGET_REL}/weights.py" in paths
    assert f"{family.relative_to(root).as_posix()}/shared_rope.py" in paths
    assert f"{CATALOG_DIR}/norm/flashinfer_rmsnorm.py" in paths
    assert f"{CATALOG_DIR}/gemm/cublas_mm.py" in paths
    assert f"{CATALOG_DIR}/activation/silu_and_mul.py" in paths
    # a contract is a certification, not code the engine runs
    assert not [p for p in paths if p.endswith(".md")]
    assert not [p for p in paths if "__pycache__" in p]


def test_a_target_that_does_not_parse_still_records_its_files(tmp_path):
    """A half-written target is exactly what a failing gate measures."""
    root = _build_repo(tmp_path, modeling="def broken(:\n")
    paths = dependency_paths(root, TARGET_REL)
    assert f"{TARGET_REL}/modeling.py" in paths
    assert not [p for p in paths if p.startswith(CATALOG_DIR)]


def test_blob_hash_is_the_id_git_gives_the_file(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("print('hi')\n", encoding="utf-8")
    expected = subprocess.run(
        ["git", "hash-object", str(path)], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert blob_hash(path) == expected
    assert blob_hash(tmp_path / "missing.py") is None


# ------------------------------------------------------------- tree identity


def test_tree_identity_names_the_commit_and_its_tree(tmp_path):
    root = _git_repo(_build_repo(tmp_path))
    identity = tree_identity(root)
    assert identity.commit == _git(root, "rev-parse", "HEAD").strip()
    assert identity.tree == _git(root, "rev-parse", "HEAD^{tree}").strip()
    assert identity.key == identity.tree
    assert not identity.dirty


def test_a_re_signed_commit_with_the_same_content_is_the_same_tree(tmp_path):
    """A DCO re-sign rewrote 225 commits; evidence about their content still holds."""
    root = _git_repo(_build_repo(tmp_path))
    before = tree_identity(root)
    _git(root, "commit", "-q", "--amend", "-s", "-m", "seed, signed")
    after = tree_identity(root)
    assert before.commit != after.commit
    assert before.key == after.key


def test_an_uncommitted_dependency_is_part_of_the_identity(tmp_path):
    root = _git_repo(_build_repo(tmp_path))
    paths = dependency_paths(root, TARGET_REL)
    clean = tree_identity(root, dependencies=ledger_mod.hash_paths(root, paths))

    (root / TARGET_REL / "modeling.py").write_text("# edited\n", encoding="utf-8")
    first = tree_identity(root, dependencies=ledger_mod.hash_paths(root, paths))
    (root / TARGET_REL / "modeling.py").write_text("# edited again\n", encoding="utf-8")
    second = tree_identity(root, dependencies=ledger_mod.hash_paths(root, paths))

    assert first.dirty == (f"{TARGET_REL}/modeling.py",)
    assert len({clean.key, first.key, second.key}) == 3


def test_outside_git_the_identity_is_unknown_rather_than_an_error(tmp_path):
    identity = tree_identity(tmp_path)
    assert not identity.known
    assert identity.short() == "unknown"


# --------------------------------------------------------------- native build


def _lib(root: Path, name: str, payload: bytes) -> Path:
    path = root / "tensorrt_llm" / "libs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_the_native_fingerprint_follows_content_not_mtime(tmp_path):
    lib = _lib(tmp_path, "libtensorrt_llm.so", b"\x7fELF" + b"a" * 1000)
    first = native_build(tmp_path)
    assert first.known and len(first.libraries) == 1

    lib.touch()  # a reinstall of the same bytes
    assert native_build(tmp_path).fingerprint == first.fingerprint

    lib.write_bytes(b"\x7fELF" + b"b" * 1000)
    assert native_build(tmp_path).fingerprint != first.fingerprint


def test_a_large_library_is_sampled_and_a_change_at_either_end_is_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_mod, "_FULL_HASH_LIMIT", 1024)
    monkeypatch.setattr(ledger_mod, "_SAMPLE_EDGE", 64)
    monkeypatch.setattr(ledger_mod, "_SAMPLE_CHUNK", 16)
    payload = bytearray(b"z" * 100_000)
    lib = _lib(tmp_path, "libth_common.so", bytes(payload))
    base = native_build(tmp_path).fingerprint
    payload[-1:] = b"q"
    lib.write_bytes(bytes(payload))
    assert native_build(tmp_path).fingerprint != base


def test_no_native_libraries_is_an_empty_fingerprint(tmp_path):
    assert not native_build(tmp_path).known


# ------------------------------------------------------------------ switches


def test_switch_defaults_are_read_from_the_code():
    source = """
import os
A = os.environ.get("TRTLLM_A", "1")
B = os.getenv("TRTLLM_B")
C = os.environ["TRTLLM_C"]
D = os.environ.get("HOME", "/root")
E = int(os.environ.get("TRTLLM_E", 0))
F = os.environ.get("TRTLLM_F", DEFAULT_F)
G = os.environ.setdefault("TRTLLM_G", "on")
"""
    assert harvest_switch_defaults(source) == {
        "TRTLLM_A": "1",
        "TRTLLM_B": None,
        "TRTLLM_C": "<required>",
        "TRTLLM_E": "0",
        "TRTLLM_F": "<expr DEFAULT_F>",
        "TRTLLM_G": "on",
    }


def test_a_switch_read_with_two_defaults_records_both():
    source = 'import os\nx = os.environ.get("TRTLLM_X", "1")\ny = os.environ.get("TRTLLM_X", "0")\n'
    assert harvest_switch_defaults(source) == {"TRTLLM_X": "0 | 1"}


def test_a_file_that_does_not_parse_has_no_switches():
    assert harvest_switch_defaults("def broken(:\n") == {}


def test_command_env_reads_every_assignment():
    command = "TRTLLM_MODELING_V2=require FOO='a b' python run.py --flag=1"
    assert command_env(command) == {"TRTLLM_MODELING_V2": "require", "FOO": "a b"}


# ---------------------------------------------------------------------- spec


def _spec(**overrides):
    spec = {
        "checkpoint_path": "/ckpt",
        "features": ["mtp", "dspark"],
        "accuracy": {
            "benchmark": {"default": "gsm8k", "dspark": "gsm8k_long"},
            "witness": {"dspark": {"contrast_env": {"X": "0"}}},
            "tolerance": 5.0,
        },
        "environment": {"kind": "local", "python": "/py", "env": {"TRTLLM_A": "1"}},
        "remediation": {"max_rounds": 2},
    }
    spec.update(overrides)
    return spec


def test_another_units_settings_do_not_change_a_units_spec():
    base = _spec()
    changed = _spec(
        accuracy={
            "benchmark": {"default": "gsm8k", "dspark": "other"},
            "witness": {"dspark": {"contrast_env": {"X": "1"}}},
            "tolerance": 5.0,
        }
    )
    assert spec_fingerprint(base, "core") == spec_fingerprint(changed, "core")
    assert spec_fingerprint(base, "mtp") == spec_fingerprint(changed, "mtp")
    assert spec_fingerprint(base, "dspark") != spec_fingerprint(changed, "dspark")


def test_adding_a_later_feature_does_not_change_an_earlier_units_spec():
    assert spec_fingerprint(_spec(), "mtp") == spec_fingerprint(
        _spec(features=["mtp", "dspark", "kv_quant"]), "mtp"
    )


def test_substrate_and_bookkeeping_do_not_change_the_spec_but_the_env_does():
    base = _spec()
    moved = _spec(
        environment={"kind": "slurm", "python": "/py", "env": {"TRTLLM_A": "1"}},
        remediation={"max_rounds": 9},
    )
    assert spec_fingerprint(base, "core") == spec_fingerprint(moved, "core")
    switched = _spec(environment={"kind": "local", "python": "/py", "env": {"TRTLLM_A": "0"}})
    assert spec_fingerprint(base, "core") != spec_fingerprint(switched, "core")


# -------------------------------------------------------- a complete verdict


def test_a_verdict_carries_everything_freshness_needs(tmp_path):
    root = _git_repo(_build_repo(tmp_path))
    (root / TARGET_REL / "modeling.py").write_text(
        (root / TARGET_REL / "modeling.py").read_text()
        + '\nimport os\nREPLAY = os.environ.get("TRTLLM_REPLAY", "1")\n',
        encoding="utf-8",
    )
    _git(root, "commit", "-qam", "replay switch")
    _lib(root, "libtensorrt_llm.so", b"\x7fELF")

    verdict = verdict_from_result(
        _result(
            member="tep4",
            measured=92.5,
            reference=92.3,
            tolerance=5.0,
            command="TRTLLM_MODELING_V2=require trtllm-eval --model /ckpt gsm8k",
            tree_state="abc1234",
        ),
        VerdictContext(
            repo_root=str(root),
            target_rel=TARGET_REL,
            unit="core",
            env={"TMPDIR": "/scratch/t", "TRTLLM_A": "1"},
            task_data=_spec(),
        ),
    )

    assert verdict.tree.commit == _git(root, "rev-parse", "HEAD").strip()
    assert verdict.native.known
    assert verdict.dependencies[f"{TARGET_REL}/modeling.py"] == blob_hash(
        root / TARGET_REL / "modeling.py"
    )
    assert verdict.env == {"TRTLLM_A": "1", "TRTLLM_MODELING_V2": "require"}
    assert verdict.switch_defaults["TRTLLM_REPLAY"] == "1"
    assert verdict.config_fingerprint.startswith("sha256:")
    assert verdict.spec_fingerprint == spec_fingerprint(_spec(), "core")
    assert verdict.tree_state == "abc1234"
    # Only switch values are stored in the clear; the rest are fingerprinted.
    assert "TMPDIR" not in json.dumps(verdict.to_record())


# -------------------------------------------------------------- the one hook


def test_the_gate_log_sink_sees_every_result_once():
    seen: list[str] = []
    log = GateLog(sink=lambda result: seen.append(result.name))
    log.record(_result(name="boot"))
    log.record(_result(name="accuracy[gsm8k]"))
    assert seen == ["boot", "accuracy[gsm8k]"]
    assert [r.name for r in log.results] == seen


def test_a_gate_log_without_a_sink_behaves_as_before():
    log = GateLog()
    assert log.record(_result()).name == "accuracy[gsm8k]"


def test_a_workflow_run_puts_every_gate_into_the_ledger_with_its_unit(
    tmp_path, task_file, monkeypatch
):
    from .test_workflow import Script, _install, _plan_then_vocabulary, _workflow

    spec = yaml.safe_load(task_file.read_text())
    spec["features"] = ["mtp"]
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, [])
    with workflow:
        workflow.run(task_file)

    # Engine gates carry their unit; the early and delivery checks carry none.
    verdicts = [v for v in Ledger.in_workspace(tmp_path / "ws").verdicts() if v.unit]
    keys = [(v.gate.split("[")[0], v.unit) for v in verdicts]
    assert ("boot", "core") in keys and ("accuracy", "core") in keys
    assert ("boot", "mtp") in keys and ("acceptance", "mtp") in keys
    assert len(verdicts) == len(workflow.gate_log.results)
    core_boot = next(v for v in verdicts if v.key[:2] == ("boot", "core"))
    assert f"{TARGET_REL}/modeling.py" in core_boot.dependencies
    assert core_boot.config["features"] == []


# ----------------------------------------------------------------------- CLI


def test_a_harness_records_a_verdict_through_the_cli(tmp_path, capsys):
    root = _build_repo(tmp_path)
    workspace = tmp_path / "ws"
    code = ledger_mod.main(
        [
            "record",
            str(workspace),
            "--repo",
            str(root),
            "--target-rel",
            TARGET_REL,
            "--gate",
            "accuracy[gsm8k]",
            "--unit",
            "core",
            "--member",
            "tp4",
            "--outcome",
            "PASS",
            "--value",
            "92.1",
            "--threshold",
            "87.3",
            "--env",
            "TRTLLM_X=1",
        ]
    )
    assert code == 0
    (verdict,) = Ledger.in_workspace(workspace).verdicts()
    assert verdict.key == ("accuracy[gsm8k]", "core", "tp4")
    assert verdict.threshold == 87.3
    assert verdict.env == {"TRTLLM_X": "1"}
    assert f"{TARGET_REL}/modeling.py" in verdict.dependencies
    assert "recorded" in capsys.readouterr().out
