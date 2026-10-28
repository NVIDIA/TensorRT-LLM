# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Develop in the target, then promote: kernel tasks on one shared entry run side by side."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab
from agent_flow.workflows.trtllm_modeling_v2.operators import promotion as promo
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution import kernel_task as worker
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage

from .test_workflow import _workflow

TARGET = f"{catalog_paths.MODELING_V2_PACKAGE}/models/newmodel/base__sm_103__tp1"
OTHER = f"{catalog_paths.MODELING_V2_PACKAGE}/models/oldmodel/base__sm_100__tp1"
TESTS = catalog_paths.model_ops_tests_dir(TARGET)
ENTRY_HEAD = "from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper\n"


def _write(root: Path, rel: str, text: str = "") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def _tree(root: Path, *, variants=("attention/thop__r3", "attention/thop__r4"), imports=None):
    """A checkout holding the shared ``attention/thop``, local variants of it and a target."""
    _write(root, f"{catalog_paths.CATALOG_DIR}/attention/thop.py", ENTRY_HEAD)
    _write(root, f"{catalog_paths.CATALOG_TESTS_DIR}/attention/test_modeling_v2_thop.py")
    _write(root, f"{TARGET}/ops/moe/permute.py", ENTRY_HEAD)  # the target's own entry
    for variant in variants:
        _write(root, f"{TARGET}/ops/{variant}.py", ENTRY_HEAD)
        category, name = variant.split("/")
        _write(root, f"{TESTS}/{category}/test_modeling_v2_{name}.py")
    if imports is None:
        imports = [
            f"from .ops.{v.replace('/', '.')} import thop as _{i}" for i, v in enumerate(variants)
        ]
    _write(
        root,
        f"{TARGET}/modeling.py",
        "\n".join(["from .ops.moe.permute import permute", *imports]) + "\n",
    )


def _task(tid, kind="kernel", status="pending", touches=(), deps=(), ref=""):
    return g.Task(
        id=tid, kind=kind, goal=tid, status=status, touches=list(touches), depends_on=list(deps),
        ref=ref,
    )  # fmt: skip


def _variant_task(tid, site, status="running"):
    paths = vocab.entry_paths(
        vocab.QueueItem(n=1, mode="verify", target="attention/thop", site=site), TARGET
    )
    return _task(tid, status=status, touches=paths, ref="1")


# ---------------------------------------------------------------- the rows


def test_one_row_per_shared_entry_lands_in_a_variant_named_for_its_site():
    rows = vocab.parse_vocabulary(
        {
            "verdict": "assemblable_after_additions",
            "queue": [
                {"n": 3, "mode": "verify", "target": "attention/thop", "surface": "mqa"},
                {"n": 4, "mode": "verify", "target": "gemm/cublas_mm", "surface": "fp8",
                 "site": "mtp_verify"},
                {"n": 5, "mode": "onboard", "target": "ops/moe/permute", "tier": 4},
            ],
        }
    )  # fmt: skip
    three, four, five = rows
    assert three.variant == "ops/attention/thop__r3"
    assert four.variant == "ops/gemm/cublas_mm__mtp_verify"
    assert five.variant is None  # the target's own entry lands in place
    task = g.Task(id="3", kind="kernel", goal="", touches=list(vocab.entry_paths(three, TARGET)))
    # It changes no code another model runs, so no regression review follows it.
    assert not g.shared_touches(task, g.own_paths(TARGET))
    with pytest.raises(vocab.VocabularyError, match="second verify row on 'attention/thop'"):
        vocab.parse_vocabulary(
            {
                "verdict": "assemblable_after_additions",
                "queue": [
                    {"n": 3, "mode": "verify", "target": "attention/thop", "surface": "mqa"},
                    {"n": 4, "mode": "verify", "target": "attention/thop", "surface": "fp8",
                     "site": "mtp_verify"},
                ],
            }
        )  # fmt: skip


# ------------------------------------------------------------- the delivery


def test_delivery_refuses_a_target_that_still_holds_a_variant(tmp_path):
    _tree(tmp_path)
    (check,) = promo.check_promoted(tmp_path, TARGET)
    assert not check.passed and check.name == "ops:promoted"
    assert (
        f"{TARGET}/ops/attention/thop__r3.py is a local variant of attention/thop" in check.detail
    )
    assert f"{TARGET}/modeling.py imports ops/attention/thop__r4" in check.detail
    assert "target.keep_local" in check.detail

    # The variants removed but an import left: still refused.
    for name in ("thop__r3.py", "thop__r4.py"):
        (tmp_path / TARGET / "ops" / "attention" / name).unlink()
    (check,) = promo.check_promoted(tmp_path, TARGET)
    assert not check.passed and "is a local variant" not in check.detail


# ------------------------------------------------------------ when it is due


def test_an_entry_waits_while_a_variant_of_it_is_still_to_land(tmp_path):
    _tree(tmp_path)
    assert promo.plan([], tmp_path / "empty", TARGET) == promo.Promotion()

    running = promo.plan([_variant_task("k3-thop", "r3")], tmp_path, TARGET)
    assert running.task is None and running.entries == ("attention/thop",)
    assert running.waiting == "attention/thop (a variant of it is still to land)"
    pending = promo.plan([_variant_task("k5-thop", "r5", status="pending")], tmp_path, TARGET)
    assert pending.task is None

    # Feature work does not hold back an entry the target already imports: the promote
    # declares the files it writes.
    ready = [_task("core", "assemble", "done"), _task("mtp", "feature", deps=["core"])]
    assert promo.plan(ready, tmp_path, TARGET).task is not None


def test_once_its_variants_landed_an_entrys_promote_is_added(tmp_path):
    _tree(tmp_path)
    _write(tmp_path, f"{OTHER}/ops/attention/thop.py", ENTRY_HEAD)  # another target's own copy
    tasks = [
        _variant_task("k3-thop", "r3", "done"),
        _variant_task("k4-thop", "r4", "done"),
        _task("core", "assemble", "done", touches=[f"{TARGET}/modeling.py"]),
    ]
    promotion = promo.plan(tasks, tmp_path, TARGET)
    task = g.Task.from_dict(promotion.task)
    assert (task.id, task.kind, task.ref, task.priority) == (
        "promote", "kernel", "promote:attention/thop", "required",
    )  # fmt: skip
    assert task.touches == [
        f"{catalog_paths.CATALOG_DIR}/attention/thop.py",
        f"{catalog_paths.CATALOG_TESTS_DIR}/attention/test_modeling_v2_thop.py",
        f"{TARGET}/ops/attention/thop__r3.py",
        f"{TESTS}/attention/test_modeling_v2_thop__r3.py",
        f"{TARGET}/ops/attention/thop__r4.py",
        f"{TESTS}/attention/test_modeling_v2_thop__r4.py",
        f"{OTHER}/ops/attention/thop.py",
        f"{catalog_paths.model_ops_tests_dir(OTHER)}/attention/test_modeling_v2_thop.py",
        f"{TARGET}/modeling.py",
    ]
    # It declares the files it rewrites, so a task that declares one never runs beside it.
    assert g.touches_conflict(task, _task("mtp", "feature", touches=[f"{TARGET}/modeling.py"]))
    assert task.gpus == 1 and task.est_minutes == 45
    assert promo.plan(tasks, tmp_path, TARGET, keep=("attention/thop",)) == promo.Promotion()


def test_one_promote_is_open_at_a_time_and_a_failed_one_goes_to_the_planner(tmp_path):
    _tree(tmp_path)
    ref = promo.promote_ref(["attention/thop"])
    open_ = promo.plan([_task("promote", ref=ref)], tmp_path, TARGET)
    assert open_.task is None and open_.waiting == "attention/thop (task promote promotes it)"

    failed = promo.plan([_task("promote", status="failed", ref=ref)], tmp_path, TARGET)
    assert failed.task is None and "the planner retries or reshapes it" in failed.waiting

    # Variants landed after a promote that is done get one of their own.
    again = promo.plan([_task("promote", status="done", ref=ref)], tmp_path, TARGET)
    assert again.task["id"] == "promote-2"


def test_a_due_promote_holds_the_checks_that_have_not_started(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import GraphOrchestrator, Services
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

    pending = _task("check-core", kind=g.CHECK_KIND)
    running = _task("check-mtp", kind=g.CHECK_KIND, status="running")
    spec = {"id": "promote", "kind": "kernel", "goal": "merge", "touches": [], "ref": "promote:x"}
    services = Services(
        plan=lambda turn: None,
        work=lambda task: "LANDED",
        check=lambda chk: None,
        free_gpus=lambda: 8,
        promotion=lambda graph: SimpleNamespace(entries=["x"], task=spec, waiting=""),
    )
    orchestrator = GraphOrchestrator(
        services, c.Contract(checks=()), c.ContractStatus.load(tmp_path), tmp_path
    )
    orchestrator.graph = g.Graph(tasks=[pending, running])
    assert orchestrator._promote() == "promote promotes them"
    assert pending.depends_on == ["promote"]
    assert running.depends_on == []


# ------------------------------------------------------------- the dispatch


def _kernel_writer(tmp_path, repo="/repo"):
    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": str(repo), "arch": "sm_103"})
    workflow._preflight = SimpleNamespace(target_rel=TARGET)
    return workflow


def test_a_verify_of_a_shared_entry_is_told_to_land_its_variant(tmp_path):
    workflow = _kernel_writer(tmp_path)
    item = vocab.QueueItem(n=3, mode="verify", target="attention/thop", surface="mqa")
    prompt = worker._kernel_writer_prompt(ExecutionStage(workflow), item)
    assert "## Where it lands" in prompt
    assert f"- `{TARGET}/ops/attention/thop__r3.py`" in prompt
    assert f"- `{TESTS}/attention/test_modeling_v2_thop__r3.py`" in prompt
    assert f"Leave `{catalog_paths.CATALOG_DIR}/attention/thop.py` and" in prompt
    assert "keeping its class and its instance name (`thop`)" in prompt
    assert "(`from .ops.attention.thop__r3 import thop`)" in prompt
    assert "one `promote` task merges every variant of `attention/thop`" in prompt

    onboard = vocab.QueueItem(n=5, mode="onboard", target="norm/new_norm")
    assert "Write the entry there." in worker._kernel_writer_prompt(
        ExecutionStage(workflow), onboard
    )


def test_a_promote_is_told_what_it_merges(tmp_path):
    repo = tmp_path / "checkout"
    _tree(repo)
    _write(repo, f"{OTHER}/ops/attention/thop.py", ENTRY_HEAD)
    workflow = _kernel_writer(tmp_path, repo)
    item = vocab.queue_item_from_ref("promote:attention/thop,ssm/fresh", "promote", "merge them")
    prompt = worker._kernel_writer_prompt(ExecutionStage(workflow), item)
    assert "**promote** this target's local variants of `attention/thop`, `ssm/fresh`" in prompt
    assert "## Tag `kernel`" in prompt and "`trtllm-modeling-v2-entry` skill" in prompt
    assert f"local variant `ops/attention/thop__r3`: `{TARGET}/ops/attention/thop__r3.py`" in prompt
    assert (
        f"`{TARGET}/modeling.py` imports `ops/attention/thop__r3`, `ops/attention/thop__r4`"
        in prompt
    )
    assert f"another target's own copy: `{OTHER}/ops/attention/thop.py`" in prompt
    assert "(not in the catalog yet: you create it)" in prompt  # ssm/fresh
    assert "no local variant on the checkout" in prompt
    assert "Sourcing:" not in prompt
