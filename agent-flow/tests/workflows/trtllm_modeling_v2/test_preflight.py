# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 preflight behavior and regression checks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import datasets
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import (
    PROBE_SOURCE,
    parse_inventory,
)
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight


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
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path),
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH},
        },
    }
    task.update(overrides)
    return task


def test_run_preflight_resolves_identity(tmp_path):
    task = _task(tmp_path, parallel="dep4")
    (tmp_path / "ckpt").rename(tmp_path / "Qwen3-30B-A3B")
    task["checkpoint_path"] = str(tmp_path / "Qwen3-30B-A3B")
    runner = make_runner(smi=_smi_rows(*[_healthy_row(i) for i in range(4)]))
    result = preflight.run_preflight(task, runner=runner)
    assert result.slug == "qwen3-30b-a3b-sm_103-dep4"
    assert result.branch == "trtllm-modeling-v2/qwen3-30b-a3b-sm_103-dep4"
    assert result.devices == (0, 1, 2, 3)
    assert result.world_size == 4
    assert result.checkpoint_config["architectures"] == ["Qwen3MoeForCausalLM"]


def test_preflight_claims_the_devices_the_widest_deployment_on_this_host_takes(tmp_path):
    """A disaggregated deployment starts two engines; one behind a launch prefix places its own."""

    def claimed(**spec):
        result = preflight.run_preflight(_task(tmp_path, **spec), claim=False)
        return result.devices, result.world_size, result.notes

    assert claimed(parallel="unified", members=["tp1", "tep2"])[:2] == ((0, 1), 2)
    devices, world, notes = claimed(
        parallel="unified",
        members={"default": ["tp1", "tep2"], "disagg": ["tep2"]},
        features=["disagg"],
    )
    assert (devices, world) == ((0, 1, 2, 3), 2)
    assert any(
        "4 claimed, not 2: a disaggregated deployment starts two engines" in n for n in notes
    )
    stated = [{"name": "dep16", "options": {"tensor_parallel_size": 16}, "launch": "srun -N4"}]
    devices, world, notes = claimed(parallel=None, deployments=stated)
    assert (devices, world) == ((), 1)
    assert any("every server starts behind its launch prefix" in n for n in notes)
    # One side behind a prefix: the other is this host's, and takes its ranks here.
    halves = {
        "name": "pd",
        "ctx": {"options": {"tensor_parallel_size": 4}, "launch": "srun -N1"},
        "gen": {"options": {"tensor_parallel_size": 2}},
    }
    assert claimed(parallel=None, deployments=[halves])[:2] == ((0, 1), 2)


def test_a_spec_the_checkpoint_cannot_meet_is_refused_with_every_reason(tmp_path):
    task = _task(tmp_path, parallel="unified", members=["tp1", "tp4", "tep4", "dep4"])
    task["features"] = ["mtp", "multimodal"]
    config = {"architectures": ["X"], "num_attention_heads": 6, "num_experts": 6}
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps(config))
    with pytest.raises(preflight.SpecInconsistencyError) as refused:
        preflight.run_preflight(task, claim=False)
    said = str(refused.value)
    assert "tp4: 6 attention heads do not split 4 ways" in said
    assert "tep4: 6 experts do not split 4 ways" in said
    # Attention DP replicates attention: only its experts are split.
    assert "dep4: 6 experts do not split 4 ways" in said and "dep4: 6 attention heads" not in said
    assert "mtp: it drafts with MTP, and config.json states no draft layers" in said
    assert "multimodal: it takes images, and the checkpoint has no tower" in said


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


def test_a_dataset_is_found_where_it_is_on_this_host_and_never_downloaded(tmp_path, monkeypatch):
    # Nothing of the host this runs on: its models root and its caches are out of the way.
    monkeypatch.delenv("LLM_MODELS_ROOT")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    # Nowhere: a host problem, said before a load is spent, with where to put it.
    path, said = datasets.find("gsm8k")
    assert path == "" and "is not on this host, and accuracy runs offline" in said
    assert "$LLM_MODELS_ROOT/datasets/openai/gsm8k" in said
    # A benchmark this module does not know is the evaluator's to look up.
    assert datasets.find("mmlu") == ("", "")
    # What an earlier evaluation prepared is the evaluator's own to find: nothing is passed.
    prepared = tmp_path / "hf/datasets/openai___gsm8k/main/0.0.0"
    prepared.mkdir(parents=True)
    (prepared / "gsm8k-test.arrow").write_text("rows")
    assert datasets.find("gsm8k") == ("", "")
    # The HuggingFace cache, then the models root over it, then a stated path over both.
    cached = _files(tmp_path / "hf/hub/datasets--openai--gsm8k/snapshots/abc123")
    assert datasets.find("gsm8k") == (str(cached), "the HuggingFace cache")
    rooted = _files(tmp_path / "models/datasets/openai/gsm8k")
    env = {"LLM_MODELS_ROOT": str(tmp_path / "models")}
    assert datasets.find("gsm8k", env=env) == (str(rooted), "the models root")
    stated = _files(tmp_path / "mine")
    assert datasets.find("gsm8k", str(stated), env)[0] == str(stated)
    assert "does not exist on this host" in datasets.find("gsm8k", str(tmp_path / "gone"), env)[1]


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


def _fake_repo(tmp_path, entries, model_ops=()):
    """A repo tree with catalog entries, and nothing else.

    ``entries`` are ``(category/name.py, impl, archs)`` written under
    ``catalog/``; ``model_ops`` are ``(family/target, category/name.py, impl,
    archs)`` kept in a target's ``ops/``.
    """
    catalog = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
    catalog.mkdir(parents=True)
    (catalog / "_op.py").write_text("class OpWrapper:\n    pass\n", encoding="utf-8")
    for path, impl, archs in entries:
        py = catalog / path
        py.parent.mkdir(parents=True, exist_ok=True)
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
    models = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
    for scope, path, impl, archs in model_ops:
        py = models / scope / "ops" / path
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def test_the_inventory_lists_what_the_tree_holds_as_the_surveyor_reads_it(tmp_path):
    repo = _fake_repo(
        tmp_path,
        [
            ("norm/x.py", "torch.ops.trtllm.x", ("SM_103",)),
            ("norm/y.py", "torch.ops.trtllm.y", ("SM_100",)),
        ],
        model_ops=[
            ("fam/ckpt__sm_103__tp1", "moe/router.py", "torch.ops.trtllm.router", ("SM_103",)),
            ("fam", "norm/shared.py", "torch.ops.trtllm.shared", ("SM_103",)),
        ],
    )
    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SOURCE, str(repo), "sm_103", "1"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    inventory = parse_inventory(proc.stdout + proc.stderr)
    lists = {
        title.split(":")[0]: names.split()
        for title, names in re.findall(
            r"\*\*(.+?)\*\* \(\d+\):\n\n((?:- .*\n)+)", inventory.surface
        )
    }
    # Certified here and on another arch only; another target's and a family's own ops.
    assert lists["Spendable"] == ["-", "`norm/x`"]
    assert lists["Present but un-run on this arch"] == ["-", "`norm/y`"]
    assert lists["Family and target ops"] == [
        "-", "`fam/ckpt__sm_103__tp1/ops/moe/router`", "-", "`fam/ops/norm/shared`",
    ]  # fmt: skip
    # The dispatch carries the head of each list and says where the rest is.
    assert "- ... 1 more, see `surface.md`" in inventory.block
    assert "`fam/ops/norm/shared`" not in inventory.block and "## Versions" in inventory.surface
    assert "1/4 entries certified on sm_103" in inventory.headline or (
        inventory.headline.startswith("inventory probe failed: ")
    )
    # A probe that printed nothing is said to the surveyor, never read as an empty catalog.
    silent = parse_inventory("Traceback\nImportError: no torch\n")
    assert "printed no payload" in silent.block and "ImportError: no torch" in silent.block
    assert silent.headline == "inventory probe failed: it printed no payload"
