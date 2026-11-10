# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 preflight behavior and regression checks."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import datasets
from agent_flow.workflows.trtllm_modeling_v2.checks.verdict import GateResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import (
    PROBE_SOURCE,
    RECEIPT_PASSED,
    format_surface,
    parse_inventory,
)
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.startup.preflight import PreflightError


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


def test_a_relative_variant_needs_its_units_benchmark_and_disagg_one_node(tmp_path):
    fp8 = {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}, "members": ["dep4"]}
    task = _task(
        tmp_path,
        parallel="unified",
        members=["tp4", "dep4"],
        accuracy={"benchmark": "gsm8k", "reference_yaml": "x.yaml", "benchmark_members": "tp4"},
        variants={"kv_fp8": {**fp8, "gates": ["accuracy"]}},
    )
    with pytest.raises(PreflightError, match="`core`'s benchmark at none of dep4"):
        preflight.check_variant_anchors(task, 4)
    del task["accuracy"]["benchmark_members"]
    assert "relative, core's own score at dep4" in preflight.check_variant_anchors(task, 4)[0]
    task["features"] = ["disagg"]
    task["environment"]["gpus_per_node"] = 4
    task["accuracy"]["disagg"] = {"test": "tests/test_d.py::TestM::test_auto_dtype", "devices": 8}
    with pytest.raises(PreflightError, match="more than the 4 devices one node holds"):
        preflight.check_disagg_placement(task)


def test_preflight_claims_the_devices_the_widest_check_takes(tmp_path):
    """The disaggregated test starts two engines: more devices than the widest member."""
    task = _task(tmp_path, parallel="unified", members=["tp1", "tep2"])
    assert preflight.devices_needed(task, 2) == 2
    task["features"] = ["disagg"]
    task["accuracy"] = {"disagg": {"test": "tests/test_d.py::TestM::test_auto_dtype"}}
    assert preflight.devices_needed(task, 2) == 4  # two engines at tep2
    task["accuracy"]["disagg"]["devices"] = 3
    assert preflight.devices_needed(task, 2) == 3  # as declared
    task["environment"]["gpus_per_node"] = 2
    del task["accuracy"]["disagg"]["devices"]
    assert preflight.devices_needed(task, 2) == 2  # never more than the host holds


def _fake_boot(fails: dict[str, str], booted: list):
    """A boot gate that fails the members named in ``fails`` with their detail."""

    def run_boot_gate(environment, *, member=None, **_):
        booted.append(member)
        detail = fails.get(member)
        return GateResult(
            name="boot[off]", member=member, passed=detail is None, detail=detail or "generated"
        )

    return run_boot_gate


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


def _run_probe(repo, arch="sm_103", prelude=""):
    proc = subprocess.run(
        [sys.executable, "-c", prelude + PROBE_SOURCE, str(repo), arch],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch=arch)


_INLINE_PASS = ("SM_103",)


def test_another_targets_ops_are_listed_not_spendable(tmp_path):
    repo = _fake_repo(
        tmp_path,
        [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)],
        model_ops=[
            ("fam/ckpt__sm_103__tp1", "moe/router.py", "torch.ops.trtllm.router", _INLINE_PASS)
        ],
    )
    inventory = _run_probe(repo)
    (op,) = inventory.other_targets
    assert (op.scope, op.path, op.impl) == (
        "fam/ckpt__sm_103__tp1",
        "moe/router.py",
        "torch.ops.trtllm.router",
    )
    assert op.receipt == RECEIPT_PASSED
    assert op not in inventory.spendable and op not in inventory.uncertified
    assert "router" not in inventory.unwrapped_ops
    assert "`fam/ckpt__sm_103__tp1/ops/moe/router`" in format_surface(inventory)


def test_family_shared_ops_are_discovered_alongside_target_ops(tmp_path):
    repo = _fake_repo(
        tmp_path,
        [],
        model_ops=[
            ("fam", "norm/shared.py", "torch.ops.trtllm.shared", _INLINE_PASS),
        ],
    )
    inventory = _run_probe(repo)
    assert any(
        entry.scope == "fam" and entry.path == "norm/shared.py" for entry in inventory.other_targets
    )


ANCHORS = tree_anchors.ANCHORS


def test_every_anchor_is_found_by_what_identifies_it_in_this_checkout():
    """Inside a TensorRT-LLM checkout, a search that ignores each hint finds what the hint points at.

    The search is what a run relies on once the tree moves, so it is what this
    holds. A hint this tree lacks (an older or newer layout) only warns.
    """
    root = Path(__file__).resolve().parents[4]
    if not (root / "tensorrt_llm").is_dir():
        pytest.skip("not inside a TensorRT-LLM checkout")
    unfindable = []
    for key, anchor in ANCHORS.items():
        here = tree_anchors.resolve(root, anchor)
        if here.via != "hint":
            where = f"found at {list(here.paths[:3])}" if here.found else "missing"
            warnings.warn(f"anchor {key}: no {anchor.hint} in this tree, {where}", stacklevel=1)
            continue
        moved = dataclasses.replace(anchor, hint=f"{anchor.hint}.moved")
        found = tree_anchors.resolve(root, moved).paths
        hint = anchor.hint
        if not any(
            p == hint or p.startswith(hint + "/") or hint.startswith(p + "/") for p in found
        ):
            unfindable.append(f"{key}: a search found {list(found[:3])}")
    assert not unfindable, f"anchors a search cannot find by what identifies them: {unfindable}"


@pytest.mark.parametrize(
    "limit,host_cache_size,warned",
    [
        (64 << 30, None, True),
        (64 << 30, 8 << 30, False),
        (None, None, False),
        (900 << 30, None, False),
    ],
)
def test_a_memory_limited_host_is_told_to_bound_the_kv_host_tier(
    tmp_path, task_file, monkeypatch, limit, host_cache_size, warned
):
    spec = yaml.safe_load(task_file.read_text())
    kv = {"enable_block_reuse": False}
    if host_cache_size is not None:
        kv["host_cache_size"] = host_cache_size
    options = tmp_path / "core.yaml"
    options.write_text(yaml.safe_dump({"kv_cache_config": kv}), encoding="utf-8")
    spec["accuracy"]["llm_api_options"] = {"core": str(options)}
    monkeypatch.setattr(preflight, "memory_limit_and_free", lambda: (limit, 690 << 30))
    notes = preflight.check_kv_cache_configs(spec, {})
    said = [n for n in notes if n.startswith("host memory:")]
    assert bool(said) is warned
    if warned:
        assert "690 GiB" in said[0] and "64 GiB" in said[0] and "host_cache_size" in said[0]


def test_the_memory_limit_is_read_from_the_process_own_cgroup(tmp_path):
    root = tmp_path / "cgroup"
    step = root / "system.slice" / "slurmstepd.scope" / "job_7" / "step_0"
    step.mkdir(parents=True)
    (root / "system.slice" / "memory.max").write_text("max\n")
    (step / "memory.max").write_text(str(280 << 30) + "\n")
    (step.parent / "memory.max").write_text(str(900 << 30) + "\n")
    v2 = "0::/system.slice/slurmstepd.scope/job_7/step_0\n"
    assert preflight.cgroup_memory_limit(v2, root) == 280 << 30
    # v1: the memory controller's own hierarchy; a huge number means no limit.
    mem = root / "memory" / "slurm" / "uid_1" / "job_7"
    mem.mkdir(parents=True)
    (mem / "memory.limit_in_bytes").write_text(str(64 << 30))
    (root / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712")
    v1 = "12:cpuset:/slurm/uid_1/job_7\n9:memory:/slurm/uid_1/job_7\n"
    assert preflight.cgroup_memory_limit(v1, root) == 64 << 30
    assert preflight.cgroup_memory_limit("0::/\n", tmp_path / "none") is None
