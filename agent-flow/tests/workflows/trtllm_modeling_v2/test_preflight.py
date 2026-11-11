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

from ._helpers import healthy_smi

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


def _weights(tmp_path, gib: int) -> None:
    index = {"metadata": {"total_size": gib << 30}}
    (tmp_path / "ckpt" / "model.safetensors.index.json").write_text(json.dumps(index))


def test_preflight_resolves_identity_and_claims_the_first_deployments_devices(tmp_path):
    task = _task(tmp_path, parallel="unified", members=["dep4", "tp1"])
    (tmp_path / "ckpt").rename(tmp_path / "Qwen3-30B-A3B")
    task["checkpoint_path"] = str(tmp_path / "Qwen3-30B-A3B")
    runner = healthy_smi(6)
    result = preflight.run_preflight(task, runner=runner)
    assert result.slug == "qwen3-30b-a3b-sm_103-unified"
    assert result.branch == "trtllm-modeling-v2/qwen3-30b-a3b-sm_103-unified"
    assert (result.devices, result.world_size) == ((0, 1, 2, 3), 4)
    assert result.checkpoint_config["architectures"] == ["Qwen3MoeForCausalLM"]
    assert result.family == "qwen3_moe" and result.target_rel.endswith("__sm_103__unified")
    # One probe, in the tree, under the run's interpreter, with no prefix: on this host.
    [command] = runner.seen
    assert command.startswith(f"cd {tmp_path} && ") and "/opt/venv/bin/python -c " in command
    assert "srun" not in command and "TRTLLM_MODELING_V2=require" in command
    # A dry run probes nothing and takes the first devices.
    dry = preflight.run_preflight(task, runner=None, claim=False)
    assert dry.devices == (0, 1, 2, 3) and any(n.startswith("dry run:") for n in dry.notes)
    # The devices claimed are the widest attempted deployment's: a later one on this host
    # runs its checks on them, and a disaggregated one starts two engines.
    wider = _task(tmp_path, parallel="unified", members=["tp1", "tep2"])
    wider["checkpoint_path"] = task["checkpoint_path"]
    result = preflight.run_preflight(wider, runner=healthy_smi(6, free=170000))
    assert (result.devices, result.world_size) == ((0, 1), 2)
    wider.update(members={"default": ["tp1", "tep2"], "disagg": ["tep2"]}, features=["disagg"])
    result = preflight.run_preflight(wider, runner=healthy_smi(6))
    assert (result.devices, result.world_size) == ((0, 1, 2, 3), 2)
    # One side behind a prefix: the other is this host's, and takes its ranks here.
    halves = {
        "name": "pd",
        "ctx": {"options": {"tensor_parallel_size": 4}, "launch": "srun -N1"},
        "gen": {"options": {"tensor_parallel_size": 2}},
    }
    split = _task(tmp_path, parallel=None, deployments=[halves])
    split["checkpoint_path"] = task["checkpoint_path"]
    assert preflight.run_preflight(split, runner=None, claim=False).devices == (0, 1)


def test_the_probe_runs_where_the_first_deployments_engine_will(tmp_path):
    stated = [
        {"name": "dep16", "options": {"tensor_parallel_size": 16}, "launch": "srun -N4 -n16"},
        {"name": "tp1", "options": {}},
    ]
    task = _task(tmp_path, parallel=None, deployments=stated)
    runner = healthy_smi(4)
    result = preflight.run_preflight(task, runner=runner)
    # Behind the prefix, which places the ranks: no device of this host is claimed for it.
    assert "; srun -N4 -n16 /opt/venv/bin/python -c " in runner.seen[0]
    assert result.devices == (0,)  # the later deployment's, which starts here
    assert any(n.startswith("devices: the first 1 of this host are taken") for n in result.notes)
    alone = _task(tmp_path, parallel=None, deployments=stated[:1])
    result = preflight.run_preflight(alone, runner=healthy_smi(4))
    assert result.devices == () and "(none claimed here)" in result.report()
    # With no prefix of its own, the spec's prefix for one-GPU commands is where it runs.
    task = _task(tmp_path)
    task["environment"]["launch"] = "srun -N1 --gpus=1"
    runner = healthy_smi(1)
    preflight.run_preflight(task, runner=runner)
    assert "; srun -N1 --gpus=1 /opt/venv/bin/python -c " in runner.seen[0]


@pytest.mark.parametrize(
    "seen,said",
    [
        (dict(tree_error="ImportError: no tensorrt_llm"), "the tree does not import where `tp1`"),
        (dict(tmp_error="PermissionError: /scratch"), "the temporary directory is not writable"),
        (dict(cap="10.0"), "`tp1` needs sm_103 devices; where `tp1` runs (on this host) the"),
        (dict(free=1000), "`tp1` needs 1 free sm_103 device(s) on this host; 0 of 1 are free"),
        (
            dict(count=0, devices_error="nvidia-smi: not found"),
            "the probe saw nvidia-smi: not found",
        ),
    ],
)
def test_only_a_hard_blocker_of_the_first_deployment_stops_the_run(tmp_path, seen, said):
    with pytest.raises(preflight.PreflightError) as stopped:
        preflight.run_preflight(_task(tmp_path), runner=healthy_smi(**seen))
    assert said in str(stopped.value)


def test_the_other_hard_blockers_are_read_from_files(tmp_path, monkeypatch):
    task = _task(tmp_path)
    silent = lambda command, env=None, timeout=0: CommandResult(3, "", "bash: srun: not found")  # noqa: E731
    with pytest.raises(preflight.PreflightError, match="the probe did not answer where `tp1` runs"):
        preflight.run_preflight(task, runner=silent)
    # The first deployment's weights do not fit its devices.
    _weights(tmp_path, 300)
    with pytest.raises(preflight.PreflightError, match="`tp1` cannot hold the weights: 300 GiB"):
        preflight.run_preflight(task, runner=healthy_smi(1))
    _weights(tmp_path, 30)
    # A benchmark it is scored on has no dataset on this host.
    monkeypatch.setenv("LLM_MODELS_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    scored = _task(tmp_path, accuracy={"model_key": "org/toy"})
    with pytest.raises(preflight.PreflightError, match="`tp1` is scored on `gsm8k`: "):
        preflight.run_preflight(scored, runner=healthy_smi(1))
    (tmp_path / "ckpt" / "config.json").write_text("[not, a, config")
    with pytest.raises(preflight.PreflightError, match="the checkpoint is not readable"):
        preflight.run_preflight(task, runner=healthy_smi(1))


def test_a_later_deployment_is_not_attempted_and_a_config_that_does_not_load_is_recorded(
    tmp_path,
):
    stated = [
        {"name": "tp4", "options": {"tensor_parallel_size": 4}, "benchmarks": []},
        {"name": "tp1", "options": {}},
        {"name": "tp8", "options": {"tensor_parallel_size": 8}},
        {
            "name": "scored",
            "options": {"tensor_parallel_size": 4},
            "benchmarks": [{"name": "mmmu"}, {"name": "gsm8k", "dataset_path": "/no/such/dir"}],
        },
    ]
    task = _task(tmp_path, parallel=None, deployments=stated)
    _weights(tmp_path, 300)
    runner = healthy_smi(4, config_error="KeyError: 'kimi_k3'")
    result = preflight.run_preflight(task, runner=runner, workspace=tmp_path / "ws")
    # The first deployment passed; the others cost the run nothing but a record.
    assert result.devices == (0, 1, 2, 3)
    assert [(r["deployment"], r["check"]) for r in result.not_attempted] == [
        ("tp1", ""),
        ("tp8", ""),
        ("scored", "accuracy[mmmu]"),
        ("scored", "accuracy[gsm8k]"),
    ]
    reasons = [r["reason"] for r in result.not_attempted]
    assert reasons[0].startswith("`tp1` cannot hold the weights: 300 GiB on 1 device(s)")
    assert reasons[1] == "`tp8` needs 8 free devices on this host; it has 4"
    assert reasons[2].startswith("no declared anchor and no reference row")
    assert reasons[3] == "`dataset_path` /no/such/dir does not exist on this host"
    # The tree cannot load the checkpoint's config yet: recorded, and not a stop.
    assert result.config_error == "KeyError: 'kimi_k3' (on node-1)"
    assert "config:      the tree cannot load it yet: KeyError" in result.report()
    # The anchors were frozen by preflight, once for the workspace.
    assert (tmp_path / "ws" / "anchors.yaml").is_file() and "scored:mmmu" in result.anchors


def test_what_the_checkpoint_cannot_meet_is_a_note_and_stops_nothing(tmp_path):
    task = _task(tmp_path, parallel="unified", members=["tp1", "tp4", "tep4", "dep4"])
    task["features"] = ["mtp", "multimodal"]
    reuse = {"kv_cache_config": {"enable_block_reuse": True}}
    task["variants"] = {"reuse": {"llm_api_options": reuse}}
    config = {"architectures": ["X"], "num_attention_heads": 6, "num_experts": 6}
    config["layer_types"] = ["linear_attention", "full_attention"]
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps(config))
    result = preflight.run_preflight(task, runner=None, claim=False)
    said = [note for note in result.notes if note.startswith("checkpoint: ")]
    assert "checkpoint: tp4: 6 attention heads do not split 4 ways" in said
    assert "checkpoint: tep4: 6 experts do not split 4 ways" in said
    # Attention DP replicates attention: only its experts are split.
    assert "checkpoint: dep4: 6 experts do not split 4 ways" in said
    assert not any("dep4: 6 attention heads" in note for note in said)
    assert "checkpoint: mtp: it drafts with MTP, and config.json states no draft layers" in said
    assert "checkpoint: multimodal: it takes images, and the checkpoint has no tower" in said
    assert any(n.startswith("checkpoint: reuse: block reuse on a hybrid checkpoint") for n in said)
    assert len(said) == 9  # with the KV heads of tp4 and tep4, and the heads of tep4


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
