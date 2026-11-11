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
- a server's default frontends need a proxy a one-rank engine lacks:
  ``test_a_server_a_check_starts_runs_one_frontend``
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
- an entry without ``CELLS`` landed and failed the final checks, a static delivery problem
  cost a whole pass of them, and a valid entry under another file name was refused
  and renamed: ``test_a_landing_is_read_for_the_conventions_and_no_finding_holds_it``
- nothing read what the workers left not done:
  ``test_the_report_lists_what_the_workers_left_not_done``
- a measured small-model anchor was called a placeholder:
  ``test_a_measured_small_anchor_is_not_called_a_placeholder``
- a large tool result ended a turn at the SDK's 1 MiB message limit:
  ``test_the_claude_backend_reads_stream_messages_past_the_sdk_default``
- delivered tests' rank processes imported another checkout:
  ``test_the_delivered_tests_run_once_at_the_commit_and_hold_on_their_passing``
  (``test_pytest_runner.py``)
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.backends import claude_code
from agent_flow.workflows.trtllm_modeling_v2.agents import prompts as prompt_texts
from agent_flow.workflows.trtllm_modeling_v2.agents.decisions import (
    decision_problem,
    read_decision,
    shape_problem,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import anchors, conventions, servers
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import report as report_mod
from agent_flow.workflows.trtllm_modeling_v2.run import workflow as workflow_mod
from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import Pipeline, Task
from agent_flow.workflows.trtllm_modeling_v2.run.startup import agent_probe
from agent_flow.workflows.trtllm_modeling_v2.spec import advice
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

from ._helpers import Script, _run, _state, _task_id, _tasks, _write_vocabulary, facts

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
    # Two turns inside at once meet at the barrier; a turn alone waits it out. A run that
    # started independent rows together would always meet here, whatever the scheduler does.
    together = threading.Barrier(2)
    met: list[str] = []
    earlier: dict[str, str] = {}

    def kernel_writer(turn, prompt):
        task = _task_id(prompt)
        earlier[task] = prompt
        try:
            together.wait(timeout=0.3)
            met.append(task)
        except threading.BrokenBarrierError:
            together.reset()
        return {"decision": "ready", "summary": task}

    _run(tmp_path, task_file, monkeypatch, Script(kernel_writer=kernel_writer))
    assert _state(tmp_path).onboarded
    assert met == [] and list(earlier) == ["k1-rms", "k2-scale", "k3-last"]
    # Each dispatch says the tasks before it are done: it starts from their commits.
    assert "`k1-rms` (kernel): done ready" in earlier["k2-scale"]
    assert "`k2-scale` (kernel): done ready" in earlier["k3-last"]


# ------------------------------------------------------------------------------ the checks


def test_a_server_a_check_starts_runs_one_frontend(tmp_path):
    """Defect 4: several frontends need the IPC proxy a one-rank single-process engine lacks."""
    for role in (None, "ctx", "gen"):
        argv = servers.serve_argv("/ckpt", tmp_path / "tp1.yaml", tmp_path / "address", role=role)
        assert "--num_serve_frontends=1" in argv


def test_a_missing_anchor_row_says_how_to_measure_and_write_one(tmp_path):
    """Defect 8: the run never invents an anchor, so it says where one comes from."""
    references = tmp_path / "gsm8k.yaml"
    references.write_text("org/other:\n  - accuracy: 50.0\n", encoding="utf-8")
    with pytest.raises(KeyError) as raised:
        anchors.load_reference_score(references, "org/model")
    message = str(raised.value)
    assert "never invented" in message
    assert "`org/model:` then `  - accuracy: <score>`" in message
    assert "trtllm-eval --model <checkpoint> <benchmark>" in message


def test_a_measured_small_anchor_is_not_called_a_placeholder(task_file, repo):
    """Review of the runs: Qwen3-0.6B's measured 41.77 is an anchor, not a guess to replace.

    The spec's warnings say when the first deployment has no anchor; an anchor that is
    there is never questioned for being low.
    """
    spec = yaml.safe_load(task_file.read_text())
    spec["accuracy"]["model_key"] = "Qwen/Qwen3-0.6B"
    task_file.write_text(yaml.safe_dump(spec))
    task = load_and_validate_task_yaml(task_file)
    [missing] = advice.spec_warnings(task)
    assert missing.startswith("no accuracy anchor: `tp1` on `gsm8k` has no declared anchor")
    assert "has no entry for 'Qwen/Qwen3-0.6B'" in missing
    references = repo / spec["accuracy"]["reference_yaml"]
    references.write_text("Qwen/Qwen3-0.6B:\n  - accuracy: 41.77\n", encoding="utf-8")
    assert advice.spec_warnings(task) == []


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
    monkeypatch, limit, host_cache_size, warned
):
    """Defect 9: engines pin a KV host tier sized from the node, past a cgroup limit."""
    kv = {} if host_cache_size is None else {"host_cache_size": host_cache_size}
    spec = {"deployments": [{"name": "tp1", "options": {"kv_cache_config": kv}}]}
    # The node's free memory is stated too: the note must not depend on the host the test runs on.
    node = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": (690 << 30) // 4096}
    monkeypatch.setattr(advice.os, "sysconf", node.__getitem__)
    monkeypatch.setattr(advice, "cgroup_memory_limit", lambda *a, **k: limit)
    said = [n for n in advice.spec_warnings(spec) if n.startswith("host memory:")]
    assert bool(said) is warned
    if warned:
        assert said[0].startswith("host memory: tp1: set no `kv_cache_config.host_cache_size`")
        assert (
            "690 GiB" in said[0] and "this process may use 64 GiB (its cgroup's limit)" in said[0]
        )


def test_the_memory_limit_is_read_from_the_process_own_cgroup(tmp_path):
    """Defect 9, as reviewed: a Slurm step's limit is on its own cgroup, not the mount's root."""
    root = tmp_path / "cgroup"
    step = root / "system.slice" / "slurmstepd.scope" / "job_7" / "step_0"
    step.mkdir(parents=True)
    (root / "system.slice" / "memory.max").write_text("max\n")
    (step / "memory.max").write_text(str(280 << 30) + "\n")
    (step.parent / "memory.max").write_text(str(900 << 30) + "\n")
    v2 = "0::/system.slice/slurmstepd.scope/job_7/step_0\n"
    assert advice.cgroup_memory_limit(v2, root) == 280 << 30
    # v1: the memory controller's own hierarchy; a huge number means no limit.
    mem = root / "memory" / "slurm" / "uid_1" / "job_7"
    mem.mkdir(parents=True)
    (mem / "memory.limit_in_bytes").write_text(str(64 << 30))
    (root / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712")
    v1 = "12:cpuset:/slurm/uid_1/job_7\n9:memory:/slurm/uid_1/job_7\n"
    assert advice.cgroup_memory_limit(v1, root) == 64 << 30
    assert advice.cgroup_memory_limit("0::/\n", tmp_path / "none") is None


# ------------------------------------------------------------------------ the decision file


def test_a_decision_file_without_its_decision_gets_one_turn_to_write_it(
    tmp_path, task_file, monkeypatch
):
    """Defect 6: the work was done and committed; only the file's key was missing."""
    prompts: list[str] = []

    def assembler(turn, prompt):
        prompts.append(prompt)
        if turn == 1:
            return {"summary": "ready: assembled, boot passed"}  # the key forgotten
        return {"decision": "ready", "summary": "assembled, boot passed"}

    _run(tmp_path, task_file, monkeypatch, Script(assembler=assembler))
    assert _state(tmp_path).onboarded
    assert len(prompts) == 2
    assert "## Your receipt is not valid" in prompts[1]
    assert "has no `decision` key" in prompts[1] and "Do not do the task again" in prompts[1]
    [core] = [t for t in _tasks(tmp_path) if t["id"] == "core"]
    assert (core["status"], core["attempts"]) == ("done", 1)


def test_a_receipt_key_of_the_wrong_type_gets_one_turn_and_never_costs_the_decision(
    tmp_path, task_file, monkeypatch
):
    """The work was accepted; a number where a list belongs must not end the run at its report."""
    prompts: list[str] = []

    def assembler(turn, prompt):
        prompts.append(prompt)
        return {"decision": "ready", "summary": ["boot", "passed"], "not_done": 7, "checks": "all"}

    _run(tmp_path, task_file, monkeypatch, Script(assembler=assembler))
    # One short turn to write it again, told which key and what it is; then the decision stands.
    assert len(prompts) == 2 and "## Your receipt is not valid" in prompts[1]
    assert (
        "`summary` in `" in prompts[1] and "must be text; it is ['boot', 'passed']." in prompts[1]
    )
    [core] = [t for t in _tasks(tmp_path) if t["id"] == "core"]
    assert (core["status"], core["attempts"]) == ("done", 1) and _state(tmp_path).onboarded
    # The report prints whatever it finds as text.
    report = (tmp_path / "ws" / "delivery_report.md").read_text()
    assert "\n- `core`: 7\n" in report and "| ready | ['boot', 'passed'] |" in report
    path = tmp_path / "receipt.yaml"
    listed = "must be a list with one line of text per item (`[]` when nothing); it is"
    forms = {
        'summary: s\nnot_done: []\nchecks:\n  - "gate @ 1a2b: [PASS]"': "",
        "not_done:\n  - what: why\n  - plain\n  - 3": "",
        "summary:\nnot_done:\nchecks:": "",
        "": "",
        "not_done: 7": f"`not_done` in `{path}` {listed} 7",
        "not_done: |\n  one\n  two": f"`not_done` in `{path}` {listed} 'one\\ntwo\\n'",
        "checks: not-a-list": f"`checks` in `{path}` {listed} 'not-a-list'",
        "checks: {boot: pass}": f"`checks` in `{path}` {listed} {{'boot': 'pass'}}",
        "not_done:\n  - [a, b]": f"`not_done` in `{path}` {listed} [['a', 'b']]",
        "not_done:\n  - what: [a, b]": f"`not_done` in `{path}` {listed} [{{'what': ['a', 'b']}}]",
        "summary: 7": f"`summary` in `{path}` must be text; it is 7",
        "summary: {a: b}": f"`summary` in `{path}` must be text; it is {{'a': 'b'}}",
    }
    for form, said in forms.items():
        path.write_text(f"decision: ready\n{form}\n")
        assert shape_problem(path) == said, form
        assert decision_problem(path) == "" and read_decision(path)["decision"] == "ready"
    # What is no receipt at all is the other question's.
    path.write_text("summary: [a]\n")
    assert shape_problem(path) and "has no `decision` key" in decision_problem(path)


def test_the_decision_is_the_first_line_of_the_decision_file():
    """Defect 7: the example starts with ``decision:``, every other key top-level after it."""
    for kind in ("survey", "kernel", "core", "deployment", "fix", "config"):
        said = prompt_texts.dispatch(kind, facts())
        # What the text says is what the parser holds: the key, wherever a worker put it.
        assert (
            "Begin with `decision: ready` or `decision: blocked`: without that key the file "
            "is no receipt. Every key starts at column 0." in said
        )
        assert "```yaml\ndecision: ready\nsummary: |" in said and "\nnot_done:\n" in said


def test_a_kernel_writer_guards_a_cell_the_target_never_serves():
    """Defect 10: a queued cell that fails on shared code and is never served does not stop a landing."""
    said = prompt_texts.dispatch("kernel", facts())
    assert "The options never serve that cell (a cached prefix with block reuse off)" in said
    assert "reject it in `is_valid`, leave it out of `CELLS`, list it under `not_done`" in said
    assert "and land." in said


def test_what_a_weak_run_slipped_without_is_in_the_text_its_worker_reads():
    """Checklist L2: each sentence was paid for by a recorded run, and stays."""
    kernel = prompt_texts.dispatch("kernel", facts())
    # An entry's CELLS and ARCHS, and the name it lands under.
    assert "An entry without `CELLS`, or whose `ARCHS` lacks `sm_103`, is a finding" in kernel
    assert "Use these exact paths: the report looks for the entry there." in kernel
    for kind in ("kernel", "core", "deployment", "fix", "config"):
        said = prompt_texts.dispatch(kind, facts())
        assert "is the package at `/flow/trtllm_modeling_v2`" in said
        assert "One engine at a time: wait until one has exited before you start the next" in said
        assert "While a job you need is running, keep polling" in said
        assert "Only commits are integrated: uncommitted work, or a source file left" in said
        assert "Read `/ws/operator_notes.md` first" in said
        assert "Read it again before you write your receipt." in said


# ------------------------------------------------------------------------------- a landing

_TARGET = "tensorrt_llm/_torch/_experimental/modeling_v2/models/fam/tgt"
_ENTRY_HEAD = (
    "class Rms(OpWrapper):\n    def __call__(self, x): ...\n    def reference(self, x): ...\n"
)
_ENTRY_CELLS = "    CELLS = (Cell(why='decode', spec={}),)\n"
_ENTRY_ARCHS = "    ARCHS = frozenset({Arch.SM_103})\n"
_KERNEL_TASK = Task(id="k1-rms", kind="kernel", unit="core", ref="1")


def _landing() -> SimpleNamespace:
    """What a landing reads of a pipeline: the queue row, the target, the arch."""
    return SimpleNamespace(
        _queue_item=lambda task: QueueItem(1, "onboard", "ops/norm/rms"),
        w=SimpleNamespace(
            _preflight=SimpleNamespace(target_rel=_TARGET),
            task_data={"arch": "sm_103", "trtllm_repo_path": ""},
            vocabulary_path=Path("/nonexistent/vocabulary.yaml"),
        ),
    )


def _land(tmp_path, *decisions: dict) -> tuple[object, list[str]]:
    """One kernel task's turn through the pipeline's own landing path; its result and dispatches.

    The worktree is ``tmp_path``; each dispatch writes the next of ``decisions``.
    """
    answers, prompts = list(decisions), []

    def dispatch(task, role, facts, path):
        prompts.append(facts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(answers.pop(0)), encoding="utf-8")

    pipeline = _landing()
    pipeline.integration = SimpleNamespace(open=lambda task_id: tmp_path, repo=tmp_path)
    pipeline.workspace = tmp_path / "ws"
    pipeline.w.journal = SimpleNamespace(event=lambda *said: None)
    pipeline._dispatch = dispatch
    pipeline._land = lambda task, role, tree, facts: ""
    pipeline._facts = lambda task, tree: {}
    pipeline._save = lambda: None
    pipeline.w.task_data["trtllm_repo_path"] = str(tmp_path)
    pipeline._scan = lambda *a: Pipeline._scan(pipeline, *a)
    task = Task(id="k1-rms", kind="kernel", unit="core", ref="1", attempts=1)
    return Pipeline._work(pipeline, task, False), prompts


def test_a_landing_is_read_for_the_conventions_and_no_finding_holds_it(tmp_path):
    """Defects 11, 13 and 14 as they are answered now: found at the landing, never a gate.

    An entry without ``CELLS``, driven on another architecture, with no test, or landed
    under another name than its row's is found when its task lands. The finding is kept
    for the report and the final fix task; it costs the task no turn and does not fail it.
    """
    named = f"{_TARGET}/ops/norm/rms__r1.py"
    landed = {"decision": "ready", "summary": "landed", "entry": named}
    entry = tmp_path / named
    entry.parent.mkdir(parents=True)
    entry.write_text(_ENTRY_HEAD + "    ARCHS = frozenset({Arch.SM_100})\n")
    result, prompts = _land(tmp_path, landed)
    assert (result.decision, result.error, len(prompts)) == ("ready", "", 1)
    tests = "tests/unittest/_torch/modeling_v2/models/fam/tgt/norm"
    assert conventions.load(tmp_path / "ws") == {
        f"`{_TARGET}/ops/norm/rms.py` is not in the tree: the entry is not where its row "
        f"lands it, it is at `{named}`": "k1-rms",
        f"{named}: Rms declares no CELLS: nothing is certified": "k1-rms",
        f"{named}: Rms's ARCHS names sm_100, not sm_103": "k1-rms",
        f"{named}: no test at {tests}/test_modeling_v2_rms__r1.py": "k1-rms",
    }
    # Put right, the next landing adds nothing; what was found stays on the record.
    entry.rename(entry.with_name("rms.py"))
    (tmp_path / _TARGET / "ops/norm/rms.py").write_text(_ENTRY_HEAD + _ENTRY_CELLS + _ENTRY_ARCHS)
    test = tmp_path / tests / "test_modeling_v2_rms.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_cells():\n    assert Rms.CELLS\n")
    result, _ = _land(tmp_path, {"decision": "ready", "summary": "x"})
    assert (result.decision, result.error) == ("ready", "")
    assert len(conventions.load(tmp_path / "ws")) == 4


def test_the_report_lists_what_the_workers_left_not_done(tmp_path, task_file, monkeypatch):
    """Review of the runs: each task's last ``not_done`` list reaches the report."""
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "k1-rms-1.yaml").write_text("decision: ready\nnot_done: [old attempt]\n")
    (decisions / "k1-rms-2.yaml").write_text(
        "decision: ready\nnot_done:\n  - cached-prefix context (31+33 tokens) fails in shared code\n"
    )
    # A short turn after the first of an attempt is the later word of that attempt.
    (decisions / "core-1.yaml").write_text("decision: ready\nnot_done: [first turn]\n")
    (decisions / "core-1.2.yaml").write_text("decision: ready\nnot_done: |\n  lm_head at M=8192\n")
    assert report_mod._not_done(tmp_path) == [
        "- `core`: lm_head at M=8192",
        "- `k1-rms`: cached-prefix context (31+33 tokens) fails in shared code",
    ]
    # The receipt's own example is a mapping to YAML; no form a worker writes stops the report.
    forms = {
        "\n  - CI list: no attempt left\n  - plain": ["CI list: no attempt left", "plain"],
        "\n  tests: no time\n  docs: later": ["tests: no time", "docs: later"],
        "7": ["7"],
        "true": ["True"],
        "none yet": ["none yet"],
        "[]": [],
        "0": [],
        "": [],
    }
    for form, lines in forms.items():
        (decisions / "core-1.2.yaml").write_text(f"decision: ready\nnot_done: {form}\n")
        said = [line for line in report_mod._not_done(tmp_path) if "`core`" in line]
        assert said == [f"- `core`: {line}" for line in lines], form
    # A sentence several tasks wrote is one line that names them, before each task's own:
    # the first whole run's report had 24 such lines, most of them one sentence five times.
    same = "The new test is not registered in a CI test list"
    for name, own in (("k1-rms-2", "2^31 probes not run"), ("k2-mm-1", "one cell at M=1")):
        text = f"decision: ready\nnot_done:\n  - {same}\n  - {own}\n"
        (decisions / f"{name}.yaml").write_text(text)
    (decisions / "core-1.2.yaml").write_text(f"decision: ready\nnot_done: [{same}, lm_head]\n")
    assert report_mod._not_done(tmp_path) == [
        f"- `core`, `k1-rms`, `k2-mm`: {same}",
        "- `core`: lm_head",
        "- `k1-rms`: 2^31 probes not run",
        "- `k2-mm`: one cell at M=1",
    ]
    # In the report a reader opens, under a heading of its own: what the flow itself did
    # not check is not drowned by what the receipts say.
    said = {"decision": "ready", "summary": "x", "not_done": ["lm_head at M=8192"]}
    _run(tmp_path, task_file, monkeypatch, Script(assembler=lambda turn, prompt: said))
    report = (tmp_path / "ws" / "delivery_report.md").read_text()
    left = report.split("## What the workers left")[1].split("##")[0]
    assert left.strip() == "- `core`: lm_head at M=8192"
    unchecked = report.split("## What the run did not check")[1].split("##")[0]
    assert "lm_head" not in unchecked and unchecked.strip().startswith(
        "- whether the target's tests are registered with the repository's CI"
    )
    assert report.index("## What the run did not check") < report.index("## What the workers left")


# ----------------------------------------------------------------------------- the backend


async def test_the_claude_backend_reads_stream_messages_past_the_sdk_default(monkeypatch):
    """Defect 12: a tool result is one stream-json message, and the SDK's default is 1 MiB.

    Held where it acts: the options the backend hands the SDK when it creates a client.
    """
    from tests import test_backends

    options = await test_backends.TestClaudeBackendCreateClient()._capture_options(monkeypatch)
    assert options.max_buffer_size == claude_code.CLI_MESSAGE_BUFFER_BYTES >= 64 << 20
