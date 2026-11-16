# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 preflight behavior and regression checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import (
    PROBE_SOURCE,
    RECEIPT_PASSED,
    format_surface,
    parse_inventory,
    probe_args,
)
from agent_flow.workflows.trtllm_modeling_v2.run.startup import datasets, preflight


def _smi_rows(*rows: str) -> str:
    return "\n".join(rows) + "\n"


def _healthy_row(index: int, cap: str = "10.3") -> str:
    return f"{index}, 180000, 183359, 0, {cap}"


def make_runner(*, smi: str, probe_ok=lambda index: True):
    """Build a runner that answers nvidia-smi, the lone probe, and torchrun."""

    def runner(argv, env=None, timeout=0):
        joined = " ".join(argv)
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=smi)
        if argv and argv[0] == "torchrun":
            return CommandResult(returncode=0, stdout="[preflight] OK\n")
        if "torch.cuda.set_device" in joined:
            index = int((env or {}).get("CUDA_VISIBLE_DEVICES", "0"))
            if probe_ok(index):
                return CommandResult(returncode=0, stdout="ok\n")
            return CommandResult(returncode=1, stderr="cudaErrorDevicesUnavailable")
        raise AssertionError(f"unexpected command: {argv}")

    return runner


_SCRATCH = "/scratch/mv2flow-run-tmp"


def _task(tmp_path, **overrides):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["Qwen3MoeForCausalLM"]}))
    task = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "trtllm_repo_path": str(tmp_path),
        "deployments": [{"name": "tp1", "options": {}}],
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH},
        },
    }
    task.update(overrides)
    return task


def test_run_preflight_resolves_identity(tmp_path):
    dep4 = {"tensor_parallel_size": 4, "enable_attention_dp": True}
    task = _task(
        tmp_path, deployments=[{"name": "dep4", "options": dep4}], target={"parallel": "dep4"}
    )
    (tmp_path / "ckpt").rename(tmp_path / "Qwen3-30B-A3B")
    task["checkpoint_path"] = str(tmp_path / "Qwen3-30B-A3B")
    runner = make_runner(smi=_smi_rows(*[_healthy_row(i) for i in range(4)]))
    result = preflight.run_preflight(task, runner=runner)
    assert result.slug == "qwen3-30b-a3b-sm_103-dep4"
    assert result.branch == "trtllm-modeling-v2/qwen3-30b-a3b-sm_103-dep4"
    assert result.devices == (0, 1, 2, 3)
    assert result.world_size == 4
    assert result.checkpoint_config["architectures"] == ["Qwen3MoeForCausalLM"]
    # The target's directory is named by `target.parallel`; the devices by the widest session.
    assert result.target_rel.endswith("qwen3_30b_a3b__sm_103__dep4")
    assert "dep4 (dep4); the widest session holds 4 device(s)" in " ".join(result.notes)


def test_a_routed_architecture_takes_the_directory_of_the_family_that_routes_it(tmp_path):
    package = preflight.MODELING_V2_PACKAGE
    family = tmp_path / package / "models" / "kimi_k3_vl"
    family.mkdir(parents=True)
    (family / "routing.py").write_text(
        '@routes("KimiK3ForConditionalGeneration")\ndef route(config, llm_args, cc):\n    pass\n'
    )
    task = _task(tmp_path)
    routed, target = preflight.resolve_target_identity(
        task, {"architectures": ["KimiK3ForConditionalGeneration"]}
    )
    assert routed == "kimi_k3_vl"
    assert target == f"{package}/models/kimi_k3_vl/ckpt__sm_103__unified"
    # An architecture no family routes yet names a new family after itself.
    fresh, _ = preflight.resolve_target_identity(
        task, {"architectures": ["FooBarForConditionalGeneration"]}
    )
    assert fresh == "foo_bar"


def test_what_a_deployment_is_held_to_and_where_it_fits_are_notes_not_stops(tmp_path):
    moves = {"cache_transceiver_config": {"backend": "DEFAULT"}}
    wide = {"tensor_parallel_size": 4, **moves}
    task = _task(
        tmp_path,
        deployments=[
            {"name": "tp4", "options": {"tensor_parallel_size": 4}},
            {
                "name": "kv_fp8",
                "options": {"tensor_parallel_size": 4, "kv_cache_config": {"dtype": "fp8"}},
                "relative_to": "tp4",
            },
            {"name": "pd", "ctx": {"options": wide}, "gen": {"options": wide}},
        ],
        accuracy={"benchmark": "gsm8k", "model_key": "org/toy"},
    )
    task["environment"]["gpus_per_node"] = 4
    notes = preflight.check_accuracy_anchors(task)
    # No reference row anywhere: nothing here stops the run; each benchmark's own check is
    # blocked before its load, and a deployment relative to another is held to that score.
    assert "accuracy anchor [tp4]: gsm8k has no declared anchor and no reference row" in notes[0]
    assert "its check will be blocked before the load, and everything else goes on" in notes[0]
    assert notes[1] == (
        "accuracy anchor [kv_fp8]: gsm8k is held to the score of `tp4`, measured as it is"
    )
    assert preflight.check_accuracy_anchors(task, ["kv_fp8"]) == [notes[1]]
    placed = preflight.check_gate_budgets(task)
    assert placed[0].startswith("session [tp4]: 4 device(s), ")
    # Eight devices on a four-device node: said here, refused at the session's seam.
    assert "session [pd]: `pd` holds 8 devices and one node has 4" in placed[2]


def test_a_session_longer_than_one_holder_is_refused_before_it_waits_for_ever():
    task = {
        "environment": {"spool": "/g", "slurm": {"time": "02:00:00"}},
        "accuracy": {"benchmark": "gsm8k"},
        "deployments": [{"name": "tp1", "options": {}}],
    }
    # A boot's half hour, a benchmark's four hours and the teardown: longer than two hours.
    with pytest.raises(preflight.PreflightError, match="the session of `tp1` is 16800s"):
        preflight.check_gate_budgets(task)
    # A holder too short for any session is told the one thing that can be followed: a
    # stranger was advised "at most 0s", which the spec's own load refuses.
    task["environment"]["slurm"]["time"] = "00:10:00"
    with pytest.raises(preflight.PreflightError) as refused:
        preflight.check_gate_budgets(task)
    advice = str(refused.value).split("so it would wait for ever. ")[1]
    assert advice == (
        "No `timeout` fits a holder that short: give the holders a longer time "
        "(17220s admits it as stated)."
    )
    task["environment"]["slurm"]["time"] = "02:00:00"
    task["deployments"][0]["timeout"] = 5400
    said, fits = preflight.check_gate_budgets(task)
    assert said == "session [tp1]: 1 device(s), 6000s"
    assert "tp1 6000s" in fits and "7200s" in fits
    assert preflight.gate_limits(task) == {"tp1": 6000}
    # One host has no holder that ends: nothing to fit in.
    local = {"environment": {"kind": "local"}, "deployments": task["deployments"]}
    assert preflight.check_gate_budgets(local) == [said]
    # A Slurm holder lasts its `time`; a wall that leaves no fit margin never starts either.
    slurm = {"environment": {"slurm": {"time": "01:40:30"}}, "deployments": task["deployments"]}
    with pytest.raises(preflight.PreflightError, match="longer than one holder \\(6030s\\)"):
        preflight.check_gate_budgets(slurm)
    slurm["environment"]["slurm"]["time"] = "02:00:00"
    assert preflight.check_gate_budgets(slurm)


def test_multirank_probe_launches_the_run_interpreter_on_claimed_devices():
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

    seen = []

    def runner(argv, env, timeout):
        seen.append((argv, env))
        return CommandResult(0, "[preflight] OK")

    preflight.check_multirank_substrate(
        2, RunEnvironment("local", "/venv/bin/python", devices=(3, 5)), runner=runner
    )
    argv, env = seen[0]
    assert argv[3:6] == ["--no-python", "/venv/bin/python", "-c"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3,5"


def _files(path: Path) -> Path:
    (path / "main").mkdir(parents=True, exist_ok=True)
    (path / "main" / "test-00000-of-00001.parquet").write_text("rows")
    return path


def test_a_missing_dataset_is_downloaded_into_the_workspace(tmp_path):
    asked = []

    def run(argv, **kwargs):
        asked.append(argv[-2:])
        _files(Path(argv[-1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    entry = datasets.stage({"accuracy": {}}, tmp_path / "ws", "gsm8k", python="/py", run=run)
    assert entry["state"] == "ready" and entry["source"] == "downloaded openai/gsm8k"
    assert asked == [["openai/gsm8k", str(tmp_path / "ws/datasets/openai/gsm8k")]]


def test_the_datasets_staged_are_those_of_the_benchmarks_the_deployments_score():
    spec = {
        "accuracy": {"benchmark": "mmlu"},
        "deployments": [
            {"name": "tp1", "options": {}},
            {"name": "tp2", "options": {}, "benchmarks": [{"name": "gsm8k"}]},
        ],
    }
    # `mmlu` is no dataset this stages; `gsm8k` is listed by one deployment alone.
    assert datasets.benchmarks(spec) == ["gsm8k"]
    assert datasets.benchmarks({"deployments": [{"name": "tp1", "options": {}}]}) == []


def test_the_tree_tests_read_a_models_root_that_holds_the_staged_copy(tmp_path):
    root = tmp_path / "models"
    (root / "stepfun").mkdir(parents=True)
    (root / "datasets" / "mmlu").mkdir(parents=True)
    staged = _files(tmp_path / "ws" / "datasets" / "openai" / "gsm8k")
    overlay = Path(datasets.models_root(tmp_path / "ws", str(root), "gsm8k", str(staged)))
    assert overlay != root
    assert (overlay / "stepfun").resolve() == (root / "stepfun").resolve()
    assert (overlay / "datasets" / "mmlu").resolve() == (root / "datasets" / "mmlu").resolve()
    assert (overlay / "datasets/openai/gsm8k/main").is_dir()
    # A root that holds it already is used as it is.
    assert datasets.models_root(tmp_path / "ws", str(overlay), "gsm8k", str(staged)) == str(overlay)


def _entry_source(impl, archs):
    arch_set = ", ".join(f"Arch.{a}" for a in archs)
    cell_list = "".join(f"Cell(why='c{i}', spec={{}}), " for i in range(2))
    return (
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper\n\n\n"
        "class Entry(OpWrapper):\n"
        '    """Mentions torch.ops.trtllm.neighbour in prose, which is not the op."""\n\n'
        f"    ARCHS = frozenset({{{arch_set}}})\n"
        f"    CELLS = ({cell_list})\n\n"
        "    def __call__(self, x):\n"
        f"        return {impl}(x)\n\n"
        "    def reference(self, x):\n"
        "        return x\n"
    )


def _fake_repo(tmp_path, entries):
    """A repo tree with catalog entries, and nothing else: ``(category/name.py, impl, archs)`` each."""
    catalog = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
    catalog.mkdir(parents=True)
    (catalog / "_op.py").write_text("class OpWrapper:\n    pass\n", encoding="utf-8")
    for path, impl, archs in entries:
        py = catalog / path
        py.parent.mkdir(parents=True, exist_ok=True)
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
    return tmp_path


def _run_probe(repo, arch="sm_103", prelude=""):
    proc = subprocess.run(
        [sys.executable, "-c", prelude + PROBE_SOURCE, *probe_args(repo, arch)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch=arch)


_INLINE_PASS = ("SM_103",)


def test_the_survey_is_given_what_the_shared_catalog_certifies_on_this_arch(tmp_path):
    """The list a survey's rows need, and nothing a reader was meant to browse."""
    repo = _fake_repo(
        tmp_path,
        [
            ("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS),
            ("norm/y.py", "torch.ops.trtllm.y", ("SM_100",)),
        ],
    )
    # Another target's op is that target's: it is not listed.
    other = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/models/fam/t/ops/moe/r.py"
    other.parent.mkdir(parents=True)
    other.write_text(_entry_source("torch.ops.trtllm.r", _INLINE_PASS), encoding="utf-8")
    inventory = _run_probe(repo)
    assert [e.name for e in inventory.spendable] == ["norm/x"]
    assert [e.name for e in inventory.uncertified] == ["norm/y"]
    assert inventory.spendable[0].receipt == RECEIPT_PASSED and len(inventory.catalog) == 2
    said = format_surface(inventory)
    assert "**Certified on this arch and still wired** (1):\n\n- `norm/x`" in said
    assert "un-run on this arch: driving their cells here is what is missing** (1):" in said
    assert "moe/r" not in said and "surface.md" not in said


def test_every_anchor_is_found_by_what_identifies_it_in_this_checkout():
    """Inside a TensorRT-LLM checkout every anchor's definition finds a file: the drift test.

    An anchor holds no place, so nothing but its definition can find it. One that finds
    nothing here means the engine renamed what it defines: its marker is what to change.
    """
    root = Path(__file__).resolve().parents[4]
    if not (root / "tensorrt_llm").is_dir():
        pytest.skip("not inside a TensorRT-LLM checkout")
    found = tree_anchors.resolve_all(root)
    assert set(found) == set(tree_anchors.ANCHORS)
    unfindable = [key for key, one in found.items() if not one.found]
    assert not unfindable, f"anchors no file of this checkout defines: {unfindable}"
    # The workflow's own copy in the checkout quotes these definitions and is passed over.
    own = Path(__file__).resolve().parents[3].name
    assert not [p for one in found.values() for p in one.paths if p.startswith(f"{own}/")]


def test_a_dry_run_freezes_no_anchor_and_the_start_after_it_does(tmp_path):
    """An operator dry-ran with a placeholder anchor, filled it in, and started a run held to the placeholder."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import anchors
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    said: list[str] = []

    def spec(anchor):
        scored = {"name": "gsm8k", "anchor": anchor, "source": "a placeholder"}
        return {"deployments": [{"name": "tp1", "options": {}, "benchmarks": [scored]}]}

    workflow = SimpleNamespace(
        workspace=tmp_path,
        task_data=spec(10),
        dry_run=True,
        journal=SimpleNamespace(section=lambda title, text: said.append(title)),
    )
    bootstrap._freeze_anchors(workflow)
    assert said == ["anchors: none frozen"] and not (tmp_path / anchors.ANCHORS_FILENAME).exists()
    # The anchor is filled in, and the run starts: it is held to what the spec says now.
    workflow.task_data, workflow.dry_run = spec(41.77), False
    bootstrap._freeze_anchors(workflow)
    assert said[-1] == "anchors: 1 frozen"
    assert anchors.frozen(tmp_path, "tp1", "gsm8k").value == 41.77


def test_a_dry_run_takes_no_measurement_that_only_a_holder_could_run(tmp_path, monkeypatch, capsys):
    """A dry run on Slurm waited ten minutes for a probe it had sent to a spool no holder served."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import Inventory
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    said: list[tuple[str, str]] = []
    workflow = SimpleNamespace(
        task_data={"arch": "sm_100", "trtllm_repo_path": str(tmp_path)},
        workspace=tmp_path,
        dry_run=True,
        _shell_runner_override=None,
        journal=SimpleNamespace(section=lambda title, text: said.append((title, text))),
    )
    environment = RunEnvironment(kind="slurm", python="python3", spool=str(tmp_path / "spool"))
    result = SimpleNamespace(environment=environment, notes=[])
    asked: list[str] = []
    taken = Inventory(arch="sm_100", error="the import failed")
    monkeypatch.setattr(
        bootstrap, "run_inventory_probe", lambda *a, **k: asked.append("inventory") or taken
    )
    monkeypatch.setattr(
        bootstrap, "check_processor_dependencies", lambda *a, **k: asked.append("processor") or []
    )
    assert bootstrap._check_processor(workflow, result, None) is True
    bootstrap._take_survey(workflow, result, None)
    # Neither was sent to the spool, and the dry run says which measurement it left out.
    assert asked == []
    out = " ".join(capsys.readouterr().out.split())
    assert "survey not taken: a dry run starts no holder" in out and "probe failed" not in out
    assert said[0][0] == "surface survey" and said[0][1].startswith("not taken: a dry run")
    # A start that is no dry run has a holder to ask, and asks it.
    workflow.dry_run = False
    assert bootstrap._check_processor(workflow, result, None) is True
    bootstrap._take_survey(workflow, result, None)
    assert asked == ["processor", "inventory"]
    assert "inventory probe failed: the import failed" in " ".join(capsys.readouterr().out.split())
