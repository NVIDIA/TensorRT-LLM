# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Develop in the target, then promote: kernel tasks on one shared entry run side by side.

Every certification of a shared entry wrote the one catalog file, and the
scheduler never runs two tasks that declare one file, so seven certifications
of one attention entry ran one after another for hours. A kernel task on a
shared entry now lands a local variant in the target's own ``ops/``; one
``promote`` task merges the variants into the catalog before delivery, and
delivery refuses a target that still holds one the spec does not keep local.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.onboard import delivery
from agent_flow.workflows.trtllm_modeling_v2.onboard import promotion as promo
from agent_flow.workflows.trtllm_modeling_v2.onboard import vocabulary as vocab
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.integration import Integration
from agent_flow.workflows.trtllm_modeling_v2.run.stages.smith import is_entry_file

from .test_integration import _ENV, _git, _repo
from .test_workflow import Script, _install, _workflow

TARGET = f"{delivery.MODELING_V2_PACKAGE}/models/newmodel/base__sm_103__tp1"
OTHER = f"{delivery.MODELING_V2_PACKAGE}/models/oldmodel/base__sm_100__tp1"
TESTS = delivery.model_ops_tests_dir(TARGET)
ENTRY_HEAD = "from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper\n"


def _write(root: Path, rel: str, text: str = "") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def _tree(root: Path, *, variants=("attention/thop__r3", "attention/thop__r4"), imports=None):
    """A checkout holding the shared ``attention/thop``, local variants of it and a target."""
    _write(root, f"{delivery.CATALOG_DIR}/attention/thop.py", ENTRY_HEAD)
    _write(root, f"{delivery.CATALOG_TESTS_DIR}/attention/test_modeling_v2_thop.py")
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
    """Every call site's cells in one row: one smith, one variant, nothing merged later."""
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
    ).queue  # fmt: skip
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


@pytest.mark.parametrize(
    "row, said",
    [
        ({"target": "attention/thop", "site": "MTP"}, "lowercase words"),
        ({"target": "attention/thop", "site": "a__b"}, "lowercase words"),
        ({"target": "ops/moe/permute", "site": "decode"}, "only a shared entry has"),
        ({"target": "attention/thop__r9"}, "marks a local variant"),
    ],
)
def test_a_row_that_would_land_ambiguously_is_refused(row, said):
    doc = {"verdict": "assemblable_after_additions",
           "queue": [{"n": 1, "mode": "verify", "surface": "s", **row}]}  # fmt: skip
    with pytest.raises(vocab.VocabularyError, match=said):
        vocab.parse_vocabulary(doc)


def test_two_rows_on_one_variant_are_refused():
    doc = {"verdict": "assemblable_after_additions", "queue": [
        {"n": 1, "mode": "verify", "target": "attention/thop", "surface": "a", "site": "decode"},
        {"n": 2, "mode": "verify", "target": "attention/thop", "surface": "b", "site": "decode"},
    ]}  # fmt: skip
    with pytest.raises(vocab.VocabularyError, match="its own `site`"):
        vocab.parse_vocabulary(doc)


def test_a_multi_rank_entry_and_its_variant_are_tested_by_an_op_matrix(tmp_path):
    tests = f"{delivery.CATALOG_TESTS_DIR}/comm"
    _write(tmp_path, f"{tests}/test_modeling_v2_allgather_op_matrix.py")
    _write(tmp_path, f"{tests}/_allgather_op_matrix.py")
    row = vocab.QueueItem(n=2, mode="verify", target="comm/allgather")
    assert vocab.entry_paths(row, repo_root=tmp_path) == (
        f"{delivery.CATALOG_DIR}/comm/allgather.py",
        f"{tests}/test_modeling_v2_allgather_op_matrix.py",
        f"{tests}/_allgather_op_matrix.py",
    )
    assert vocab.entry_paths(row, TARGET, tmp_path) == (
        f"{TARGET}/ops/comm/allgather__r2.py",
        f"{TESTS}/comm/test_modeling_v2_allgather__r2_op_matrix.py",
        f"{TESTS}/comm/_allgather__r2_op_matrix.py",
    )
    for path in vocab.entry_paths(row, TARGET, tmp_path):
        assert is_entry_file(path, "ops/comm/allgather__r2"), path
    # A new entry, or no checkout to read: the conventional name.
    assert vocab.entry_paths(row)[1] == f"{tests}/test_modeling_v2_allgather.py"


def test_a_task_ref_names_its_dispatch_and_numbers_it_as_before(tmp_path):
    item = vocab.queue_item_from_ref("verify:attention/thop", "k-thop-MTP.2", "re-certify")
    assert item.mode == "verify" and item.note == "re-certify"
    assert item.variant == "ops/attention/thop__k_thop_mtp_2"
    assert item.n == 10_000 + zlib.crc32(b"k-thop-MTP.2") % 90_000
    assert vocab.queue_item_from_ref("verify:attention/thop@decode", "t").variant == (
        "ops/attention/thop__decode"
    )
    assert vocab.queue_item_from_ref("onboard:ops/moe/x", "t").variant is None
    promote = vocab.queue_item_from_ref("promote: ssm/b , attention/thop", "promote")
    assert promote.mode == vocab.MODE_PROMOTE and promote.entries == ("ssm/b", "attention/thop")
    assert promote.skill == "trtllm-modeling-v2-kernel-verify"
    assert vocab.entry_paths(promote, TARGET) is None
    for ref in ("promote:ops/moe/x", "promote:", "verify:attention/thop@Bad", "audit:a/b",
                "verify:", "onboard:ops/moe/x@site"):  # fmt: skip
        assert vocab.queue_item_from_ref(ref, "t") is None, ref

    # The number a planner-added task's decision file is kept under does not move.
    workflow = _workflow(tmp_path)
    task = g.Task(id="k-grouped", kind="kernel", goal="g", ref="verify:ops/gemm/grouped_mm")
    assert workflow._queue_item(task).n == vocab.queue_item_from_ref(task.ref, task.id).n


# ---------------------------------------------------------- what is on disk


def test_variants_and_the_target_files_that_import_them_are_found(tmp_path):
    _tree(tmp_path, imports=[
        "from .ops.attention.thop__r3 import thop",
        "from .ops.attention import thop__r4",
        "from tensorrt_llm._torch._experimental.modeling_v2.models.newmodel.base__sm_103__tp1"
        ".ops.attention.thop__r5 import thop as t5",
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop import thop as t",
    ])  # fmt: skip
    _write(tmp_path, f"{TARGET}/ops/attention/__init__.py")
    found = promo.local_variants(tmp_path, TARGET)
    assert [v.target for v in found] == ["ops/attention/thop__r3", "ops/attention/thop__r4"]
    assert found[0].entry == "attention/thop" and found[0].files == (
        f"{TARGET}/ops/attention/thop__r3.py",
        f"{TESTS}/attention/test_modeling_v2_thop__r3.py",
    )
    assert promo.variant_imports(tmp_path, TARGET) == {
        f"{TARGET}/modeling.py": (
            "ops/attention/thop__r3", "ops/attention/thop__r4", "ops/attention/thop__r5",
        )
    }  # fmt: skip
    assert promo.local_variants(tmp_path, "") == ()


def test_the_spec_keeps_entries_local_by_name():
    assert promo.keep_local({}) == ()
    assert promo.keep_local({"target": {"keep_local": "attention/thop/"}}) == ("attention/thop",)
    assert promo.keep_local({"target": {"keep_local": ["b/x", "a/y", "b/x"]}}) == ("a/y", "b/x")
    assert promo.keep_local({"target": {"keep_local": 3}}) == ()


# ------------------------------------------------------------- the delivery


def test_delivery_refuses_a_target_that_still_holds_a_variant(tmp_path):
    """The check fails on exactly what it guards: a variant left behind, or imported."""
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


def test_a_promoted_or_kept_target_passes(tmp_path):
    _tree(tmp_path)
    (kept,) = promo.check_promoted(tmp_path, TARGET, keep=("attention/thop",))
    assert kept.passed and "kept local by target.keep_local: attention/thop" in kept.detail

    promoted = tmp_path / "promoted"
    _tree(promoted, variants=(), imports=[
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop import thop",
    ])  # fmt: skip
    (check,) = promo.check_promoted(promoted, TARGET)
    assert check.passed and check.detail == "no local variant waits for promotion"


def test_a_target_test_that_imports_a_variant_is_rewritten_by_its_promote(tmp_path):
    """The target's own tests import its variants by module path, as its modeling.py does.

    A promote removed two variants a model-level test of the target imported:
    neither its touches nor its dispatch named the file, and delivery's check
    read only the package, so the test failed at import for five hours until a
    later promote re-pointed it. The tests are read beside the package; a
    variant's own test belongs to the variant, so it is not an importer.
    """
    absolute = f"from {TARGET.replace('/', '.')}.ops.attention"
    model_test = f"{TESTS}/test_modeling_v2_vision.py"
    own_test = f"{TESTS}/attention/test_modeling_v2_thop__r3.py"
    repo = tmp_path / "checkout"
    _tree(repo)
    _write(repo, model_test, f"{absolute}.thop__r4 import thop as thop_vision\n")
    _write(repo, own_test, f"{absolute}.thop__r3 import thop\n")
    found = promo.variant_imports(repo, TARGET)
    assert found[model_test] == ("ops/attention/thop__r4",)
    assert own_test not in found
    assert model_test in promo.promote_touches(repo, TARGET, ["attention/thop"])
    item = vocab.queue_item_from_ref("promote:attention/thop", "promote", "merge them")
    prompt = _smith(tmp_path, repo)._smith_prompt(item)
    assert f"`{model_test}` imports `ops/attention/thop__r4`" in prompt
    # The code that calls a variant is the package's: a test alone does not make it due.
    assert promo.variant_imports(repo, TARGET, tests=False) == {
        f"{TARGET}/modeling.py": ("ops/attention/thop__r3", "ops/attention/thop__r4")
    }

    # Promoted, with the model test still importing the removed variant: refused.
    promoted = tmp_path / "promoted"
    _tree(promoted, variants=(), imports=[
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop import thop",
    ])  # fmt: skip
    _write(promoted, model_test, f"{absolute}.thop__r4 import thop as thop_vision\n")
    (check,) = promo.check_promoted(promoted, TARGET)
    assert not check.passed
    assert f"{model_test} imports ops/attention/thop__r4" in check.detail


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


def test_an_entry_the_target_does_not_import_yet_waits_for_the_code_that_calls_it(tmp_path):
    _tree(tmp_path, imports=[])
    core = _task("core", "assemble")
    waiting = promo.plan([core], tmp_path, TARGET)
    assert waiting.task is None
    assert waiting.waiting == "attention/thop (the target does not import every variant of it yet)"
    # With nothing left to change the target, whatever waits is due.
    core.status = "done"
    assert promo.plan([core], tmp_path, TARGET).task is not None


def test_each_entry_is_promoted_once_its_own_variants_landed(tmp_path):
    _tree(tmp_path, variants=("attention/thop__r3", "norm/rms__r5"))
    _write(tmp_path, f"{delivery.CATALOG_DIR}/norm/rms.py", ENTRY_HEAD)
    still = _task("k6-rms", status="running", ref="6",
                  touches=list(vocab.entry_paths(vocab.QueueItem(n=6, mode="verify",
                                                                 target="norm/rms", site="r6"),
                                                 TARGET)))  # fmt: skip
    promotion = promo.plan([still], tmp_path, TARGET)
    assert promotion.entries == ("attention/thop", "norm/rms")
    assert promotion.task["ref"] == "promote:attention/thop"
    assert promotion.waiting == "norm/rms (a variant of it is still to land)"
    assert f"{TARGET}/ops/norm/rms__r5.py" not in promotion.task["touches"]


def test_a_promote_that_moves_no_import_runs_beside_one_that_does(tmp_path):
    """Verify variants can hold cells only, while the target calls the catalog entry.

    Their promote merges cells into the catalog and rewrites no file of the
    target, so it does not declare the target's ``modeling.py`` and runs
    beside a promote that rewrites the target's imports there. Declaring it
    queued one promote behind another on a file neither shared.
    """
    catalog = (
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop import thop"
    )
    _tree(tmp_path, imports=[catalog])
    tasks = [_variant_task("k3-thop", "r3", "done"), _variant_task("k4-thop", "r4", "done")]
    task = g.Task.from_dict(promo.plan(tasks, tmp_path, TARGET).task)
    assert f"{TARGET}/ops/attention/thop__r3.py" in task.touches
    assert f"{TARGET}/modeling.py" not in task.touches
    other = _task("promote-2", status="running", touches=[f"{TARGET}/modeling.py"])
    assert not g.touches_conflict(task, other)


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
        f"{delivery.CATALOG_DIR}/attention/thop.py",
        f"{delivery.CATALOG_TESTS_DIR}/attention/test_modeling_v2_thop.py",
        f"{TARGET}/ops/attention/thop__r3.py",
        f"{TESTS}/attention/test_modeling_v2_thop__r3.py",
        f"{TARGET}/ops/attention/thop__r4.py",
        f"{TESTS}/attention/test_modeling_v2_thop__r4.py",
        f"{OTHER}/ops/attention/thop.py",
        f"{delivery.model_ops_tests_dir(OTHER)}/attention/test_modeling_v2_thop.py",
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


# ------------------------------------------------------------- the dispatch


def _smith(tmp_path, repo="/repo"):
    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": str(repo), "arch": "sm_103"})
    workflow._preflight = SimpleNamespace(target_rel=TARGET)
    return workflow


def test_a_verify_of_a_shared_entry_is_told_to_land_its_variant(tmp_path):
    workflow = _smith(tmp_path)
    item = vocab.QueueItem(n=3, mode="verify", target="attention/thop", surface="mqa")
    prompt = workflow._smith_prompt(item)
    assert "## Where it lands" in prompt
    assert f"- `{TARGET}/ops/attention/thop__r3.py`" in prompt
    assert f"- `{TESTS}/attention/test_modeling_v2_thop__r3.py`" in prompt
    assert f"Leave `{delivery.CATALOG_DIR}/attention/thop.py` and" in prompt
    assert "keeping its class and its instance name (`thop`)" in prompt
    assert "(`from .ops.attention.thop__r3 import thop`)" in prompt
    assert "one `promote` task merges every variant of `attention/thop`" in prompt

    onboard = vocab.QueueItem(n=5, mode="onboard", target="norm/new_norm")
    assert "Write the entry there." in workflow._smith_prompt(onboard)


def test_the_task_declares_where_its_variant_is_or_that_it_lands_in_place(tmp_path):
    workflow = _smith(tmp_path)
    item = vocab.queue_item_from_ref("verify:attention/thop", "k-recert", "re-certify")
    named = _variant_task("k-recert", "mtp_verify", "pending")
    assert workflow._landing(item, named)[0] == "ops/attention/thop__mtp_verify"

    in_place = g.Task(id="k", kind="kernel", goal="", ref="verify:attention/thop",
                      touches=list(vocab.shared_entry_paths("attention/thop")))  # fmt: skip
    assert workflow._landing(item, in_place) == (
        "attention/thop",
        vocab.shared_entry_paths("attention/thop"),
    )
    assert "## Where it lands" not in workflow._smith_prompt(item, task=in_place)

    own = vocab.QueueItem(n=2, mode="onboard", target="ops/moe/permute")
    assert "## Where it lands" not in workflow._smith_prompt(own)


def test_a_promote_is_told_what_it_merges(tmp_path):
    repo = tmp_path / "checkout"
    _tree(repo)
    _write(repo, f"{OTHER}/ops/attention/thop.py", ENTRY_HEAD)
    workflow = _smith(tmp_path, repo)
    item = vocab.queue_item_from_ref("promote:attention/thop,ssm/fresh", "promote", "merge them")
    prompt = workflow._smith_prompt(item)
    assert "**promote** this target's local variants of `attention/thop`, `ssm/fresh`" in prompt
    assert (
        "`trtllm-modeling-v2-kernel-verify` skill" in prompt and "references/promote.md" in prompt
    )
    assert f"local variant `ops/attention/thop__r3`: `{TARGET}/ops/attention/thop__r3.py`" in prompt
    assert (
        f"`{TARGET}/modeling.py` imports `ops/attention/thop__r3`, `ops/attention/thop__r4`"
        in prompt
    )
    assert f"another target's own copy: `{OTHER}/ops/attention/thop.py`" in prompt
    assert "(not in the catalog yet: you create it)" in prompt  # ssm/fresh
    assert "no local variant on the checkout" in prompt
    assert "Sourcing:" not in prompt


def test_a_promote_is_pointed_at_what_the_smiths_of_its_variants_reported(tmp_path):
    """The smiths that land the variants write what the merge must keep in their reports.

    A promote the orchestrator adds carries a bare goal and may start before a
    planner revision copies those notes into it, so the dispatch itself names
    the reports of the returned smiths on each entry.
    """
    repo = tmp_path / "checkout"
    _tree(repo)
    workflow = _smith(tmp_path, repo)
    journal = workflow.journal
    for mode, entry in (
        ("verify", "attention/thop"),
        ("onboard", "attention/thop"),
        ("verify", "attention/thop"),  # still running: no result yet
        ("verify", "ssm/fresh"),
    ):
        report = journal.record_dispatch("smith", f"{mode} {entry}", "the dispatch")
        if report.name.startswith("03-"):
            continue
        journal.record_result(report, "## For the orchestrator\n- the promote must carry X")
    item = vocab.queue_item_from_ref("promote:attention/thop", "promote", "merge them")
    prompt = workflow._smith_prompt(item)
    reports = journal.reports_dir
    assert (
        f"- what the smiths that landed it said the merge must keep (each report's result, "
        f"'For the orchestrator'), in `{reports}`: `01-smith-verify-attention-thop.md`, "
        f"`02-smith-onboard-attention-thop.md`" in prompt
    )
    assert "03-smith-verify-attention-thop.md" not in prompt  # no result: nothing to read
    assert "04-smith-verify-ssm-fresh.md" not in prompt  # another entry's


def test_a_failed_receipt_of_its_variants_files_is_integrated(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    workflow = _workflow(tmp_path)
    workflow._preflight = SimpleNamespace(target_rel=TARGET)
    workflow._integration = Integration(repo, workflow.workspace, "mv2/run", _ENV)
    tree = workflow._integration.open("k3-thop")
    head = _git(repo, "rev-parse", "HEAD")
    files = (
        f"{TARGET}/ops/attention/thop__r3.py",
        f"{TESTS}/attention/test_modeling_v2_thop__r3.py",
    )

    def smith(turn, prompt):
        for name in files:
            _write(tree, name, "x = 1\n")
        _git(tree, "add", *files)
        _git(tree, "commit", "-q", "-m", "thop: fp8 kv refused\n\nSigned-off-by: T <t@example.com>")
        return {"decision": "FAILED_RECEIPT", "summary": "fp8 kv fails", "entry": files[0]}

    _install(monkeypatch, workflow, Script(None, smith=smith))
    monkeypatch.setattr(type(workflow), "_smith_prompt", lambda self, item, note="", task=None: "D")
    item = vocab.QueueItem(n=3, mode="verify", target="attention/thop")
    workflow._run_one_smith(item, task=_variant_task("k3-thop", "r3"))
    assert _git(repo, "rev-parse", "HEAD~1") == head
    assert "the entry's own files only, are integrated" in workflow._smith_decision_for(item)


# ---------------------------------------------------------------- the rules


def _flat(*parts: str) -> str:
    import agent_flow.workflows.trtllm_modeling_v2 as pkg

    path = Path(pkg.__file__).parent.joinpath("agents", "skills", *parts)
    return " ".join(path.read_text(encoding="utf-8").split())


def test_the_skills_and_the_role_say_where_a_shared_entry_lands_and_how_it_is_promoted():
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS

    smith = " ".join(DEFAULT_PROMPTS.smith.split())
    spec = _flat("trtllm-modeling-v2-kernel-onboard", "references", "entry-spec.md")
    verify = _flat("trtllm-modeling-v2-kernel-verify", "SKILL.md")
    promote = _flat("trtllm-modeling-v2-kernel-verify", "references", "promote.md")
    runtime = _flat("trtllm-modeling-v2-core", "references", "runtime-interface.md")
    for text in (smith, spec, verify, runtime):
        assert "ops/<category>/<name>__<site>" in text
    assert "`promote`" in smith and "references/promote.md" in verify
    for rule in ("the union of", "one test file", "never widen", "imports the catalog",
                 "the claims gates", "once"):  # fmt: skip
        assert rule in promote, rule


def test_the_verify_skill_says_pre_merge_deselects_large_cells_and_who_registers_the_marker():
    verify = _flat("trtllm-modeling-v2-kernel-verify", "SKILL.md")
    assert 'pre-merge runs the suite with `-m "not large"`' in verify
    assert "registered in `tests/unittest/pytest.ini`" in verify
    assert "an engine task" in verify


def test_cells_only_variants_beside_a_catalog_call_are_due_while_writers_run(tmp_path):
    """The target calls the shared entry; its variants hold verified cells and are imported by none.

    Their merge rewrites nothing a writer could still be changing, so it is due
    at once rather than after the last writer of the run.
    """
    catalog = (
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop import thop"
    )
    _tree(tmp_path, imports=[catalog])
    assert promo.catalog_callers(tmp_path, TARGET) == {"attention/thop"}
    tasks = [
        _variant_task("k3-thop", "r3", "done"),
        _variant_task("k4-thop", "r4", "done"),
        _task("mtp", kind="feature", status="running"),
    ]
    assert promo.plan(tasks, tmp_path, TARGET).task is not None


def test_a_due_promote_holds_the_checks_that_have_not_started(tmp_path):
    """They would measure the variants it replaces, and be stale once it lands."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import GraphOrchestrator, Services

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
