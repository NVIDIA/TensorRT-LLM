# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The target marker: a fingerprint of what the gates measure, over a real checkout."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.committer import CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.infra.vcs import git_output
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import workflow as workflow_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_workflow import _git

_FILES = {
    "tensorrt_llm/models/target.py": "T = 1\n",
    "tensorrt_llm/README.md": "the package\n",
    "cpp/kernel.cu": "// k\n",
    "tests/integration/test_lists/test-db/l0_b300.yml": "l0_b300: []\n",
    "tests/integration/defs/accuracy/references/gsm8k.yaml": "m: [{accuracy: 50}]\n",
    "tests/integration/defs/accuracy/references/mmlu.yaml": "m: [{accuracy: 70}]\n",
    "docs/source/index.md": "# docs\n",
    "README.md": "# readme\n",
    ".github/CODEOWNERS": "* @someone\n",
    "examples/llm-api/quickstart_advanced.py": "print('q')\n",
    "examples/llm-api/other.py": "print('o')\n",
    ".gitignore": "*.so\n",
}


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    for name, text in _FILES.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _workflow(repo: Path, **accuracy) -> TrtllmModelingV2Workflow:
    workflow = TrtllmModelingV2Workflow(workspace=repo / "ws")
    workflow.task_data = {
        "trtllm_repo_path": str(repo),
        "accuracy": {
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            **accuracy,
        },
    }
    return workflow


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")


@pytest.mark.parametrize(
    "name",
    [
        "tests/integration/test_lists/test-db/l0_b300.yml",
        "tests/unittest/test_new.py",
        "docs/source/index.md",
        "README.md",
        "tensorrt_llm/README.md",
        ".github/CODEOWNERS",
        "examples/llm-api/other.py",
        # Another benchmark's anchors, which this spec does not name.
        "tests/integration/defs/accuracy/references/mmlu.yaml",
    ],
)
def test_a_commit_no_gate_can_see_keeps_the_marker(checkout, name):
    workflow = _workflow(checkout)
    before = workflow._target_marker()
    assert before.startswith("tree:")
    _commit(checkout, name, "edited\n")
    assert workflow._target_marker() == before


@pytest.mark.parametrize(
    "name",
    [
        "tensorrt_llm/models/target.py",
        "tensorrt_llm/_torch/new_op.py",
        "cpp/kernel.cu",
        "setup.py",
        # What a gate runs or reads under the ungated paths.
        "examples/llm-api/quickstart_advanced.py",
        "tests/integration/defs/accuracy/references/gsm8k.yaml",
        "tests/integration/test_input_files/merlion.png",
    ],
)
def test_a_commit_a_gate_can_see_changes_the_marker(checkout, name):
    workflow = _workflow(checkout)
    before = workflow._target_marker()
    _commit(checkout, name, "edited\n")
    assert workflow._target_marker() != before


def test_only_a_commit_moves_the_marker(checkout):
    """Checks measure snapshots of commits: an uncommitted edit is in none of them."""
    workflow = _workflow(checkout)
    clean = workflow._target_marker()
    (checkout / "tensorrt_llm/models/target.py").write_text("T = 2\n")
    (checkout / "tensorrt_llm" / "untracked.py").write_text("U = 1\n")
    assert workflow._target_marker() == clean
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "the edit")
    assert workflow._target_marker() not in ("", clean)


def test_a_native_rebuild_changes_the_marker_and_a_copied_inode_does_not(checkout):
    library = checkout / "tensorrt_llm" / "libs" / "libth_common.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF one")
    workflow = _workflow(checkout)
    before = workflow._target_marker()

    # A fresh inode with the same bytes and times, as the rebuild helper makes.
    stat = library.stat()
    staging = library.with_name(".staging")
    staging.write_bytes(library.read_bytes())
    os.utime(staging, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(staging, library)
    assert workflow._target_marker() == before

    library.write_bytes(b"\x7fELF two")
    os.utime(library, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert workflow._target_marker() != before


def test_a_file_the_spec_names_is_a_gate_input_wherever_it_lives(checkout):
    workflow = _workflow(
        checkout,
        boot={"prompts": [{"prompt": "What is this?", "images": ["examples/llm-api/cat.png"]}]},
    )
    (checkout / "examples/llm-api/cat.png").write_bytes(b"cat")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "an image")
    before = workflow._target_marker()
    _commit(checkout, "examples/llm-api/cat.png", "another cat")
    assert workflow._target_marker() != before
    assert "examples/llm-api/cat.png" in workflow_mod.gate_inputs(workflow.task_data, checkout)

    # A script a command line runs, quoted or as a flag's value.
    _commit(checkout, "examples/disaggregated/launch.sh", "echo leg B\n")
    workflow.task_data["accuracy"]["witness"] = {
        "disagg": {"contrast_command": "bash 'examples/disaggregated/launch.sh' --n=2"}
    }
    inputs = workflow_mod.gate_inputs(workflow.task_data, checkout)
    assert "examples/disaggregated/launch.sh" in inputs
    assert "examples/llm-api/other.py" not in inputs


def test_the_marker_is_not_established_without_a_checkout(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    workflow = _workflow(plain)
    assert workflow._target_marker() == ""
    workflow.task_data = {}
    assert workflow._target_marker() == ""


def test_every_question_is_a_read_through_the_injected_runner(checkout):
    asked: list[tuple[str, ...]] = []

    def recording(repo, *args):
        asked.append(args)
        return git_output(repo, *args)

    workflow = TrtllmModelingV2Workflow(workspace=checkout / "ws", git_runner=recording)
    workflow.task_data = {"trtllm_repo_path": str(checkout)}
    assert workflow._target_marker()
    assert asked and {args[0] for args in asked} <= {"rev-parse", "ls-tree", "status"}
    # The head's tree is listed once, not on every marker.
    workflow._target_marker()
    assert sum(1 for args in asked if args[0] == "ls-tree") == 1


def test_status_entries_name_a_renames_source_too():
    entries = list(workflow_mod.porcelain_paths("R  new.py\0old.py\0?? a b.py\0 M c.py\0"))
    assert entries == [("new.py", "R "), ("old.py", "R "), ("a b.py", "??"), ("c.py", " M")]


def test_gate_relevance_keeps_what_a_gate_reads_and_denies_the_workspace():
    relevant = workflow_mod.gate_relevant
    assert relevant("tensorrt_llm/x.py")
    assert not relevant("tests/x.py")
    assert not relevant("tensorrt_llm/_torch/GUIDE.md")
    assert relevant("tests/ref.yaml", keep={"tests/ref.yaml"})
    assert not relevant("ws/journal.md", denied=("ws/",))
    assert not relevant("ws/x.py", keep={"ws/x.py"}, denied=("ws/",))


def _result(name, member, passed=True):
    return SimpleNamespace(name=name, member=member, passed=passed, detail="d", tree_state="t")


def test_a_verdict_under_the_head_marker_stands_where_the_tree_is_the_same(checkout):
    head = _git(checkout, "rev-parse", "HEAD").strip()
    clean = workflow_mod._CLEAN_STATUS
    status = contract_mod.ContractStatus.load(checkout / "ws")
    status.record("core", _result("boot", "tp1"), marker=f"{head}:{clean}")
    status.record("core", _result("accuracy[gsm8k]", "tp1"), marker=f"{head}:0123456789ab")
    contract = contract_mod.Contract(
        checks=(
            contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER),
            contract_mod.Check("core", "accuracy[gsm8k]", "tp1", contract_mod.TIER_INNER),
        )
    )
    _commit(checkout, "tests/integration/test_lists/test-db/l0_b300.yml", "l0_b300: [y]\n")

    workflow = _workflow(checkout)
    evaluation = contract_mod.evaluate(
        contract, workflow._contract_status(), marker=workflow._target_marker()
    )
    # Clean at a head whose tree the gates see as unchanged: it stands. Recorded
    # over uncommitted changes nobody can reconstruct: it is stale.
    assert [c.key for c in evaluation.passed] == ["core:boot@tp1"]
    assert [c.key for c in evaluation.missing] == ["core:accuracy[gsm8k]@tp1"]
    # The file keeps what was recorded.
    assert f"{head}:{clean}" in (checkout / "ws" / contract_mod.STATUS_FILENAME).read_text()

    _commit(checkout, "tensorrt_llm/models/target.py", "T = 9\n")
    later = _workflow(checkout)
    evaluation = contract_mod.evaluate(
        contract, later._contract_status(), marker=later._target_marker()
    )
    assert not evaluation.passed


def test_a_verdict_under_the_head_marker_is_stale_once_a_library_is_rebuilt(checkout):
    head = _git(checkout, "rev-parse", "HEAD").strip()
    status = contract_mod.ContractStatus.load(checkout / "ws")
    status.record("core", _result("boot", "tp1"), marker=f"{head}:{workflow_mod._CLEAN_STATUS}")
    library = checkout / "tensorrt_llm" / "libs" / "libth_common.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF")
    later = time.time() + 3600
    os.utime(library, (later, later))

    workflow = _workflow(checkout)
    contract = contract_mod.Contract(
        checks=(contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER),)
    )
    evaluation = contract_mod.evaluate(
        contract, workflow._contract_status(), marker=workflow._target_marker()
    )
    assert not evaluation.passed


# ------------------------------------------------------------- anchor tables, by entry

_TABLE = "tests/integration/defs/accuracy/references/gsm8k.yaml"
_OTHER_TABLE = "tests/integration/defs/accuracy/references/mmlu.yaml"
_ROWS = """\
org/target:
  - accuracy: 56.8
  - spec_dec_algo: MTP
    accuracy: 56.8
org/other:
  - accuracy: 70.1
"""


def _anchored(repo: Path, **accuracy) -> TrtllmModelingV2Workflow:
    """A run anchored to ``org/target``'s entry of a table every model shares; the first commits it."""
    if "org/target" not in (repo / _OTHER_TABLE).read_text():
        _commit(repo, _TABLE, _ROWS)
        _commit(repo, _OTHER_TABLE, _ROWS)
    return _workflow(repo, model_key="org/target", **accuracy)


def test_an_anchor_table_is_read_by_the_entry_of_the_runs_model(checkout):
    workflow = _anchored(checkout)
    assert workflow_mod.anchor_tables(workflow.task_data, checkout) == {
        _TABLE: frozenset({"org/target"})
    }
    assert _TABLE not in workflow_mod.gate_inputs(workflow.task_data, checkout)


@pytest.mark.parametrize(
    "rows",
    [
        _ROWS.replace("70.1", "71.0"),
        _ROWS + "org/new:\n  - accuracy: 12.0\n",
        _ROWS.replace("org/other:\n  - accuracy: 70.1\n", ""),
        "# A comment the parser drops.\n" + _ROWS,
    ],
)
def test_an_edit_to_another_models_anchor_keeps_the_marker(checkout, rows):
    workflow = _anchored(checkout)
    before = workflow._target_marker()
    _commit(checkout, _TABLE, rows)
    assert workflow._target_marker() == before
    (checkout / _TABLE).write_text(rows + "org/uncommitted:\n  - accuracy: 1.0\n")
    assert workflow._target_marker() == before


@pytest.mark.parametrize(
    "rows",
    [
        _ROWS.replace("  - accuracy: 56.8\n  - spec", "  - accuracy: 57.0\n  - spec"),
        _ROWS.replace("    accuracy: 56.8\n", "    accuracy: 50.0\n"),
        _ROWS.replace("org/other:", "  - quant_algo: FP8\n    accuracy: 55.0\norg/other:"),
        _ROWS.replace("org/target:", "org/renamed:"),
        "org/target: [",
    ],
)
def test_an_edit_to_this_models_anchor_changes_the_marker(checkout, rows):
    workflow = _anchored(checkout)
    before = workflow._target_marker()
    _commit(checkout, _TABLE, rows)
    edited = workflow._target_marker()
    assert edited not in ("", before)
    _git(checkout, "rm", "-q", _TABLE)
    _git(checkout, "commit", "-q", "-m", "drop the table")
    assert workflow._target_marker() not in (before, edited)


def test_a_table_anything_else_may_read_whole_stays_whole(checkout):
    # Named by a command line as well: whatever it runs may read every entry.
    workflow = _anchored(
        checkout, witness={"disagg": {"contrast_command": f"python leg_b.py --anchors={_TABLE}"}}
    )
    assert _TABLE in workflow_mod.gate_inputs(workflow.task_data, checkout)
    assert workflow_mod.anchor_tables(workflow.task_data, checkout) == {}
    before = workflow._target_marker()
    _commit(checkout, _TABLE, _ROWS.replace("70.1", "71.0"))
    assert workflow._target_marker() != before

    # No model key, and no checkpoint to derive one from: which entry is read is unknown.
    unkeyed = _workflow(checkout)
    assert _TABLE in workflow_mod.gate_inputs(unkeyed.task_data, checkout)
    before = unkeyed._target_marker()
    _commit(checkout, _TABLE, _ROWS.replace("71.0", "72.0"))
    assert unkeyed._target_marker() != before


def test_a_units_and_a_variants_own_tables_are_read_by_their_entries(checkout):
    workflow = _anchored(checkout)
    workflow.task_data["features"] = ["multimodal"]
    workflow.task_data["accuracy"]["reference_yaml"] = {"default": _TABLE}
    workflow.task_data["accuracy"]["model_key"] = {"default": "org/target"}
    workflow.task_data["variants"] = {
        "kv_fp8": {
            "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
            "accuracy": {"reference_yaml": _OTHER_TABLE, "model_key": "org/target-fp8"},
        }
    }
    assert workflow_mod.anchor_tables(workflow.task_data, checkout) == {
        _TABLE: frozenset({"org/target"}),
        _OTHER_TABLE: frozenset({"org/target-fp8"}),
    }
    before = workflow._target_marker()
    _commit(checkout, _OTHER_TABLE, _ROWS.replace("70.1", "71.0"))
    assert workflow._target_marker() == before
    _commit(checkout, _OTHER_TABLE, _ROWS + "org/target-fp8:\n  - accuracy: 55.0\n")
    assert workflow._target_marker() != before

    # The multimodal unit anchored on the other table: its model's entry there counts too.
    workflow.task_data["accuracy"]["reference_yaml"]["multimodal"] = _OTHER_TABLE
    tables = workflow_mod.anchor_tables(workflow.task_data, checkout)
    assert tables[_OTHER_TABLE] == frozenset({"org/target", "org/target-fp8"})


def _whole_table_marker(workflow: TrtllmModelingV2Workflow, repo: Path) -> str:
    """The committed tree's marker as it was before anchor tables were read by entry."""
    keep = workflow_mod.gate_inputs(workflow.task_data, repo) | {_TABLE}
    denied = CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace).denied_paths
    head = _git(repo, "rev-parse", "HEAD").strip()
    blobs = workflow_mod.tree_blobs(_git(repo, "ls-tree", "-r", "-z", "--full-tree", head))
    kept = {
        p: b for p, b in blobs.items() if workflow_mod.gate_relevant(p, keep=keep, denied=denied)
    }
    return workflow_mod.fingerprint(kept, workflow_mod.native_libraries(repo))


def test_a_verdict_under_the_whole_table_marker_stands_on_the_same_tree(checkout):
    workflow = _anchored(checkout)
    earlier = _whole_table_marker(workflow, checkout)
    _commit(checkout, _TABLE, _ROWS.replace("70.1", "71.0"))
    now = _whole_table_marker(workflow, checkout)
    assert workflow._tree_markers() == (workflow._target_marker(), now)

    status = contract_mod.ContractStatus.load(checkout / "ws")
    status.record("core", _result("boot", "tp1"), marker=now)
    status.record("core", _result("accuracy[gsm8k]", "tp1"), marker=earlier)
    contract = contract_mod.Contract(
        checks=(
            contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER),
            contract_mod.Check("core", "accuracy[gsm8k]", "tp1", contract_mod.TIER_INNER),
        )
    )
    resumed = _anchored(checkout)
    evaluation = contract_mod.evaluate(
        contract, resumed._contract_status(), marker=resumed._target_marker()
    )
    # Recorded on this tree: it stands. Recorded on a tree that differed only in
    # another model's entry: that tree cannot be rebuilt from its marker, so it is
    # stale this once; recorded under the marker now, the same edit leaves it standing.
    assert [c.key for c in evaluation.passed] == ["core:boot@tp1"]
    assert [c.key for c in evaluation.missing] == ["core:accuracy[gsm8k]@tp1"]
    assert now in (checkout / "ws" / contract_mod.STATUS_FILENAME).read_text()


def test_a_verdict_under_the_head_marker_reads_the_anchor_entries_at_that_head(checkout):
    _anchored(checkout)
    head = _git(checkout, "rev-parse", "HEAD").strip()
    status = contract_mod.ContractStatus.load(checkout / "ws")
    status.record("core", _result("boot", "tp1"), marker=f"{head}:{workflow_mod._CLEAN_STATUS}")
    contract = contract_mod.Contract(
        checks=(contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER),)
    )

    # Another model's entry moved since: the head's tree is this one, as the gates see it.
    _commit(checkout, _TABLE, _ROWS.replace("70.1", "71.0"))
    resumed = _anchored(checkout)
    evaluation = contract_mod.evaluate(
        contract, resumed._contract_status(), marker=resumed._target_marker()
    )
    assert [c.key for c in evaluation.passed] == ["core:boot@tp1"]

    # This model's entry moved since: the verdict read another anchor.
    _commit(checkout, _TABLE, _ROWS.replace("56.8", "60.0"))
    later = _anchored(checkout)
    evaluation = contract_mod.evaluate(
        contract, later._contract_status(), marker=later._target_marker()
    )
    assert not evaluation.passed
