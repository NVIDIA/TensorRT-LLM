# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One CPU case for each fix the weak-model GPU runs paid for.

Eight runs of two weak agent models (Kimi K3.1 Pro, DeepSeek V4.1 Flash) delivered
only after these defects were fixed. Each case is named after what it protects and
says which defect it answers. A change that alters one of these behaviours on
purpose changes its case here, in the same commit, and says so; none is dropped.

What each run lost, and the case that holds its fix:

- a fresh CLI configuration blocked the agent probe, which said nothing useful:
  ``test_a_blocked_agent_probe_says_how_to_allow_it``
- a worker read the checkout's own ``agent-flow/`` to learn the gate:
  ``test_every_dispatch_names_the_running_package``
- kernel tasks side by side conflicted on the target's package files:
  ``test_tasks_run_one_at_a_time``
- the serve gate's default frontends need a proxy a one-rank engine lacks:
  ``test_the_serve_gate_runs_one_frontend``
- a finished task started over for a decision file without ``decision``:
  ``test_a_decision_file_without_its_decision_gets_one_turn_to_write_it``
- the decision file's example put a long summary before the decision:
  ``test_the_decision_is_the_first_line_of_the_decision_file``
- a missing anchor row only said "add the entry":
  ``test_a_missing_anchor_row_says_how_to_measure_and_write_one``
- engines pinned a KV host tier past the step's memory limit:
  ``test_a_memory_limited_host_is_told_to_bound_the_kv_host_tier`` and
  ``test_the_memory_limit_is_read_from_the_process_own_cgroup``
- a defect in shared code became a fix task the commit policy refused:
  ``test_a_kernel_writer_guards_a_cell_the_target_never_serves``
- an entry without ``CELLS`` landed and failed the final round:
  ``test_an_entry_without_cells_or_this_arch_is_refused_when_its_task_lands`` and
  ``test_landing_and_delivery_apply_one_entry_rule``
- nothing read what the workers left uncertified:
  ``test_the_report_lists_what_the_workers_left_uncertified``
- a measured small-model anchor was called a placeholder:
  ``test_a_measured_small_anchor_is_not_called_a_placeholder``
- a large tool result ended a turn at the SDK's 1 MiB message limit:
  ``test_the_claude_backend_reads_stream_messages_past_the_sdk_default``
- a static delivery problem cost a whole certification round:
  ``test_a_landed_unit_is_read_with_the_static_delivery_checks``
- a valid entry under another file name was refused and renamed:
  ``test_the_landing_check_reads_the_entry_the_decision_names``
- delivered tests' rank processes imported another checkout:
  ``test_delivered_tests_run_with_the_tree_first_on_pythonpath``
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.backends import claude_code
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import _common
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.worker import ENDING_EXTRAS
from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import delivery
from agent_flow.workflows.trtllm_modeling_v2.checks import serve as serve_gate
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import certify
from agent_flow.workflows.trtllm_modeling_v2.run import workflow as workflow_mod
from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import Pipeline, Task, static_repair_block
from agent_flow.workflows.trtllm_modeling_v2.run.startup import agent_probe, preflight
from agent_flow.workflows.trtllm_modeling_v2.spec import lint

from ._helpers import Script, _run, _state, _task_id, _tasks, _write_vocabulary

# ----------------------------------------------------------------- the probe, the dispatch


def test_a_blocked_agent_probe_says_how_to_allow_it():
    """Defect 1: a fresh Claude Code configuration refuses the probe's command."""
    said = (
        "Command was blocked by the Bash tool before it could run: "
        "Error: Contains brace with quote character (expansion obfuscation)"
    )
    hint = agent_probe.hint_for(said, environ={})
    assert "bypassPermissions" in hint and "IS_SANDBOX=1" in hint
    assert "does not change that policy" in hint


def test_every_dispatch_names_the_running_package(tmp_path, task_file, monkeypatch):
    """Defect 3: a worktree's own ``agent-flow/`` is another copy of the flow, maybe an older one."""
    package = Path(workflow_mod.__file__).resolve().parents[1]
    assert package.name == "trtllm_modeling_v2"
    calls = _run(tmp_path, task_file, monkeypatch, Script())
    assert calls
    for role, prompt in calls:
        assert f"is the package at `{package}`" in prompt, role
        assert "read the running one" in prompt, role


def test_tasks_run_one_at_a_time(tmp_path, task_file, monkeypatch):
    """Defect 2: every task starts from the commits of the ones before it."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_vocabulary(
        workspace,
        [
            {"n": 1, "mode": "onboard", "target": "ops/norm/rms"},
            {"n": 2, "mode": "onboard", "target": "ops/norm/scale"},
            {"n": 3, "mode": "onboard", "target": "ops/norm/last", "depends_on": [1]},
        ],
    )
    lock = threading.Lock()
    running: list[str] = []
    seen: list[tuple[str, int]] = []

    def kernel_writer(turn, prompt):
        task = _task_id(prompt)
        with lock:
            running.append(task)
            seen.append((task, len(running)))
        with lock:
            running.remove(task)
        return {"decision": "LANDED", "summary": task}

    _run(tmp_path, task_file, monkeypatch, Script(kernel_writer=kernel_writer))
    assert _state(tmp_path).verdict == "delivered"
    assert seen == [("k1-rms", 1), ("k2-scale", 1), ("k3-last", 1)]


# ------------------------------------------------------------------------------ the checks


def test_the_serve_gate_runs_one_frontend(tmp_path):
    """Defect 4: several frontends need the IPC proxy a one-rank single-process engine lacks."""
    options = tmp_path / "core.llm_api.yaml"
    task = {"accuracy": {"max_seq_len": 9216}}
    args = serve_gate.serve_args(task, None, member="tp1", options=options)
    assert args[:3] == ["--config", str(options), "--num_serve_frontends=1"]


def test_a_missing_anchor_row_says_how_to_measure_and_write_one(tmp_path):
    """Defect 8: the run never invents an anchor, so it says where one comes from."""
    references = tmp_path / "gsm8k.yaml"
    references.write_text("org/other:\n  - accuracy: 50.0\n", encoding="utf-8")
    with pytest.raises(KeyError) as raised:
        accuracy_checks.load_reference_score(references, "org/model")
    message = str(raised.value)
    assert "never invented" in message
    assert "`org/model:` then `  - accuracy: <score>`" in message
    assert "trtllm-eval --model <checkpoint> <benchmark>" in message


def test_a_measured_small_anchor_is_not_called_a_placeholder(tmp_path):
    """Review of the runs: Qwen3-0.6B's measured 41.77 is an anchor, not a guess to replace."""
    references = tmp_path / "gsm8k.yaml"
    references.write_text("Qwen/Qwen3-0.6B:\n  - accuracy: 41.77\n", encoding="utf-8")
    task = {
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": str(references),
            "model_key": "Qwen/Qwen3-0.6B",
        }
    }
    [finding] = lint.accuracy_rules(lint.LintContext(task=task, world=1))
    assert finding.severity == lint.WARN
    assert "is a low placeholder" not in finding.message
    assert "if it is a placeholder" in finding.message
    assert "a measured score of a small model is right as it is" in finding.fix


def test_delivered_tests_run_with_the_tree_first_on_pythonpath(tmp_path):
    """Defect 15: rank processes a test starts inherit the tree, as the gates' commands do."""
    line = certify.test_job_line("python -m pytest t.py", tmp_path)
    assert line.startswith(f"cd {tmp_path.resolve()} && export PYTHONPATH={tmp_path.resolve()}")
    assert line.endswith("&& python -m pytest t.py")


# ----------------------------------------------------------------------------- host memory


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
    """Defect 9: engines pin a KV host tier sized from the node, past a cgroup limit."""
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
    """Defect 9, as reviewed: a Slurm step's limit is on its own cgroup, not the mount's root."""
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


# ------------------------------------------------------------------------ the decision file


def test_a_decision_file_without_its_decision_gets_one_turn_to_write_it(
    tmp_path, task_file, monkeypatch
):
    """Defect 6: the work was done and committed; only the file's key was missing."""
    prompts: list[str] = []

    def assembler(turn, prompt):
        prompts.append(prompt)
        if turn == 1:
            return {"summary": "UNIT_READY: assembled, boot passed"}  # the key forgotten
        return {"decision": "UNIT_READY", "summary": "assembled, boot passed"}

    _run(tmp_path, task_file, monkeypatch, Script(assembler=assembler))
    assert _state(tmp_path).verdict == "delivered"
    assert len(prompts) == 2
    assert "## Your decision file is not valid" in prompts[1]
    assert "has no `decision` key" in prompts[1] and "Do not do the task again" in prompts[1]
    [core] = [t for t in _tasks(tmp_path) if t["id"] == "core"]
    assert (core["status"], core["attempts"]) == ("done", 1)


def test_the_decision_is_the_first_line_of_the_decision_file():
    """Defect 7: the example starts with ``decision:``, a role's extra keys top-level beside it."""
    plain = _common.decision_file_block("surveyor")
    extra = _common.decision_file_block("assembler", ("blocked_reason", "uncertified"))
    assert "```yaml\ndecision: <exactly one of the decisions listed above>\nsummary: |" in plain
    assert "\nblocked_reason: ...\nuncertified: ...\n```" in extra


def test_a_kernel_writer_guards_a_cell_the_target_never_serves():
    """Defect 10: a queued cell that fails on shared code and is never served does not stop a landing."""
    ending = ENDING_EXTRAS["kernel_writer"]
    assert "A queued cell the target never serves" in ending
    assert "reject it in `is_valid`" in ending and "out of `CELLS`" in ending
    assert "list it under `uncertified`" in ending and "and land." in ending


# ------------------------------------------------------------------------------- a landing

_TARGET = "tensorrt_llm/_torch/_experimental/modeling_v2/models/fam/tgt"
_ENTRY_HEAD = (
    "class Rms(OpWrapper):\n    def __call__(self, x): ...\n    def reference(self, x): ...\n"
)
_ENTRY_CELLS = "    CELLS = (Cell(why='decode', spec={}),)\n"
_ENTRY_ARCHS = "    ARCHS = frozenset({Arch.SM_103})\n"
_KERNEL_TASK = Task(id="k1-rms", kind="kernel", unit="core", ref="1")


def _landing() -> SimpleNamespace:
    """What ``Pipeline._entry_problems`` reads of a pipeline: the queue row, the target, the arch."""
    return SimpleNamespace(
        _queue_item=lambda task: QueueItem(1, "onboard", "ops/norm/rms"),
        w=SimpleNamespace(
            _preflight=SimpleNamespace(target_rel=_TARGET),
            task_data={"arch": "sm_103", "trtllm_repo_path": ""},
        ),
    )


def test_an_entry_without_cells_or_this_arch_is_refused_when_its_task_lands(tmp_path):
    """Defect 11: the entry is read when its kernel task lands, with one short turn to fix it."""
    entry = tmp_path / _TARGET / "ops/norm/rms.py"

    def problems():
        return Pipeline._entry_problems(_landing(), _KERNEL_TASK, tmp_path)

    assert problems() == [f"`{_TARGET}/ops/norm/rms.py` is not in your worktree"]
    entry.parent.mkdir(parents=True)
    entry.write_text(_ENTRY_HEAD + _ENTRY_ARCHS)
    assert problems() == ["Rms declares no CELLS: nothing is certified"]
    entry.write_text(_ENTRY_HEAD + _ENTRY_CELLS + "    ARCHS = frozenset({Arch.SM_100})\n")
    assert problems() == ["ARCHS names sm_100, not sm_103"]
    entry.write_text(_ENTRY_HEAD + _ENTRY_CELLS + _ENTRY_ARCHS)
    assert problems() == []
    block = static_repair_block(["Rms declares no CELLS: nothing is certified"], entry=True)
    assert "Do not start the task again" in block and "- Rms declares no CELLS" in block
    assert "`CELLS` name the cells" in block


def test_landing_and_delivery_apply_one_entry_rule(tmp_path):
    """Review of the runs: a landing and the delivery check hold an entry to the same rule."""
    entry = tmp_path / _TARGET / "ops/norm/rms.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(_ENTRY_HEAD + _ENTRY_CELLS + "    ARCHS = frozenset({Arch.SM_100})\n")
    landing = Pipeline._entry_problems(_landing(), _KERNEL_TASK, tmp_path)
    delivered, _ = delivery.entry_problems(entry, "sm_103")
    assert landing == delivered == ["ARCHS names sm_100, not sm_103"]


def test_the_landing_check_reads_the_entry_the_decision_names(tmp_path):
    """Defect 14: the module the decision names counts, whatever its name (the target imports it)."""
    named = f"{_TARGET}/ops/norm/rms__r1.py"
    entry = tmp_path / named
    entry.parent.mkdir(parents=True)
    entry.write_text(_ENTRY_HEAD + _ENTRY_CELLS + _ENTRY_ARCHS)
    landing = _landing()
    assert Pipeline._entry_problems(landing, _KERNEL_TASK, tmp_path, {"entry": named}) == []
    assert Pipeline._entry_problems(landing, _KERNEL_TASK, tmp_path, {"entry": "nope.py"}) == [
        f"neither `nope.py` (your decision's `entry`) nor `{_TARGET}/ops/norm/rms.py` is in your worktree"
    ]


def test_a_landed_unit_is_read_with_the_static_delivery_checks(tmp_path, task_file):
    """Defect 13: the checks that read files run at a landing, not only at the end."""
    spec = yaml.safe_load(task_file.read_text())
    target = (
        "tensorrt_llm/_torch/_experimental/modeling_v2/models/gpt_oss/gpt_oss_120b__sm_103__tp1"
    )
    workflow = SimpleNamespace(
        task_data=spec, _preflight=SimpleNamespace(target_rel=target, family="gpt_oss")
    )
    tree = tmp_path / "tree"
    tree.mkdir()
    failures = certify.static_delivery_failures(workflow, tree)
    names = {f.split(":", 2)[0] + ":" + f.split(":", 2)[1] for f in failures}
    # An empty tree misses what the final certification reads: the target and its accuracy row.
    assert "target:modeling.py" in names
    assert "accuracy:reference" in names
    # Smoke and bringup runs owe no CI files: the same waiver as the certification's.
    spec["profile"] = "bringup"
    waived = certify.static_delivery_failures(workflow, tree)
    assert not any(f.startswith(("accuracy:reference", "tests:test-db")) for f in waived)
    assert any(f.startswith("target:modeling.py") for f in waived)
    unit = static_repair_block(["tests:test-db: not listed"], entry=False)
    assert "Put the tree right" in unit and "- tests:test-db: not listed" in unit


def test_the_report_lists_what_the_workers_left_uncertified(tmp_path):
    """Review of the runs: each task's last ``uncertified`` list reaches the report."""
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "k1-rms-1.yaml").write_text("decision: LANDED\nuncertified: [old attempt]\n")
    (decisions / "k1-rms-2.yaml").write_text(
        "decision: LANDED\nuncertified:\n  - cached-prefix context (31+33 tokens) fails in shared code\n"
    )
    (decisions / "core-1.yaml").write_text(
        "decision: UNIT_READY\nuncertified: |\n  lm_head at M=8192\n"
    )
    (decisions / "planner-3.yaml").write_text("decision: CONTINUE\nuncertified: [not a task]\n")
    assert certify.uncertified_lines(tmp_path) == [
        "- `core`: lm_head at M=8192",
        "- `k1-rms`: cached-prefix context (31+33 tokens) fails in shared code",
    ]


# ----------------------------------------------------------------------------- the backend


def test_the_claude_backend_reads_stream_messages_past_the_sdk_default():
    """Defect 12: a tool result is one stream-json message, and the SDK's default is 1 MiB.

    That the size reaches the SDK's options is pinned beside the backend's other option
    cases; this holds the size and that case's presence.
    """
    from tests import test_backends

    assert claude_code.CLI_MESSAGE_BUFFER_BYTES >= 64 << 20
    assert callable(
        test_backends.TestClaudeBackendCreateClient.test_create_client_reads_large_cli_messages
    )
