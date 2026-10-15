# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a command through an agent without letting the agent decide anything.

The whole design rests on one property, and most of this file is about it:
the agent is a **trigger, not a witness**. It gets a job submitted; the job
writes its own log and exit code; the orchestrator reads those. An agent
that lies, hallucinates a result, or says nothing at all must not be able to
move the verdict.

The rest covers the quoting hazard that motivated submitting a script rather
than a command line, and the ordering rule that makes a receipt trustworthy.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from agent_flow.runtime.commands import TIMEOUT_RETURNCODE, CommandSpec, DelegatedCommandExecutor
from agent_flow.runtime.turns import TurnExecutor, TurnOutcome, TurnRequest


class FakeAgent(TurnExecutor):
    """A driver that runs the script it is handed, like a real one would.

    ``reply`` is what it claims afterwards, which is the interesting knob:
    nothing downstream should depend on it.
    """

    name = "fake-agent"

    def __init__(self, *, reply: str = "submitted", submits: bool = True, delay: float = 0.01):
        self.reply = reply
        self.submits = submits
        self.delay = delay
        self.requests: list[TurnRequest] = []

    def run_script(self, prompt: str) -> None:
        # The prompt carries the resolved submission call; pull the script
        # path out of it the way a real driver would read its argument.
        import re
        import subprocess

        match = re.search(r"bash (\S+\.sh)", prompt)
        if not match:
            return
        script = match.group(1)
        time.sleep(self.delay)
        subprocess.run(["bash", script], check=False, capture_output=True)

    def run(self, request: TurnRequest) -> TurnOutcome:
        self.requests.append(request)
        if self.submits:
            threading.Thread(target=self.run_script, args=(request.prompt,), daemon=True).start()
        return TurnOutcome(role=request.role, text=self.reply, transport=self.name)


def _executor(tmp_path: Path, agent: TurnExecutor, **kwargs) -> DelegatedCommandExecutor:
    return DelegatedCommandExecutor(
        agent,
        root=tmp_path / "broker",
        render_submission=lambda script: f'submit(command=["bash", "-lc", "bash {script}"])',
        poll_s=0.01,
        **kwargs,
    )


# --------------------------------------------------- the verdict is not the agent's


def test_the_exit_code_comes_from_the_job_not_from_the_agent():
    """An agent claiming success cannot make a failing command pass."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        agent = FakeAgent(reply="Everything passed, accuracy was 99.9!")
        record = _executor(Path(tmp), agent).execute(CommandSpec(argv="exit 7", timeout_s=30))
    assert record.returncode == 7
    assert not record.ok


def test_the_output_comes_from_the_log_not_from_the_agent():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        agent = FakeAgent(reply="I saw accuracy: 99.9")
        record = _executor(Path(tmp), agent).execute(
            CommandSpec(argv="echo 'lm-eval gsm8k average accuracy: 42.0'", timeout_s=30)
        )
    assert "42.0" in record.stdout
    assert "99.9" not in record.stdout


def test_an_agent_that_says_nothing_useful_changes_nothing():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        record = _executor(Path(tmp), FakeAgent(reply="")).execute(
            CommandSpec(argv="echo fine", timeout_s=30)
        )
    assert record.ok and record.stdout.strip() == "fine"


def test_a_job_nobody_submits_times_out_rather_than_passing():
    """The failure mode if the agent silently does not submit."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        agent = FakeAgent(reply="done!", submits=False)
        record = _executor(Path(tmp), agent).execute(CommandSpec(argv="echo never", timeout_s=0))
    assert record.timed_out
    assert record.returncode == TIMEOUT_RETURNCODE
    assert not record.ok


# ------------------------------------------------------------------- the script


def test_the_command_is_submitted_as_a_path_not_as_a_quoted_string(tmp_path):
    """The reason this executor writes a script at all.

    Submission templates embed their payload in a quoted string, and a real
    gate command carries `--model_dir="/p"` and
    `--chat_template_kwargs={"enable_thinking": false}`. Interpolating one
    produces a call that is mis-parsed or truncated.
    """
    agent = FakeAgent(submits=False)
    hostile = '/venv/bin/python run.py --model_dir="/models/x" --kw={"enable_thinking": false}'
    _executor(tmp_path, agent).execute(CommandSpec(argv=hostile, timeout_s=0))

    prompt = agent.requests[0].prompt
    assert '--model_dir="' not in prompt.split("```python")[1].split("```")[0]
    script = sorted((tmp_path / "broker").glob("*.sh"))[0]
    assert hostile in script.read_text(encoding="utf-8")


def test_the_script_publishes_its_own_exit_code(tmp_path):
    agent = FakeAgent(submits=False)
    _executor(tmp_path, agent).execute(CommandSpec(argv="true", timeout_s=0))
    body = sorted((tmp_path / "broker").glob("*.sh"))[0].read_text(encoding="utf-8")
    assert "rc=$?" in body
    # The receipt is renamed into place, so seeing it means the log is done.
    assert ".tmp\n" in body and "mv " in body


def test_the_script_does_not_set_e_because_the_exit_code_is_the_measurement(tmp_path):
    """`set -e` would kill the script before it could record a failure."""
    agent = FakeAgent(submits=False)
    _executor(tmp_path, agent).execute(CommandSpec(argv="false", timeout_s=0))
    body = sorted((tmp_path / "broker").glob("*.sh"))[0].read_text(encoding="utf-8")
    assert "set -u" in body
    assert "set -e" not in body


def test_the_environment_travels_as_exports(tmp_path):
    agent = FakeAgent(submits=False)
    _executor(tmp_path, agent).execute(
        CommandSpec(argv="env", env={"TRTLLM_MODELING_V2": "require"}, timeout_s=0)
    )
    body = sorted((tmp_path / "broker").glob("*.sh"))[0].read_text(encoding="utf-8")
    assert "export TRTLLM_MODELING_V2=require" in body


def test_the_environment_reaches_the_command(tmp_path):
    record = _executor(tmp_path, FakeAgent()).execute(
        CommandSpec(argv="echo $MV2FLOW_DELEGATED", env={"MV2FLOW_DELEGATED": "yes"}, timeout_s=30)
    )
    assert record.stdout.strip() == "yes"


# ------------------------------------------------------------------- the brief


def test_the_agent_is_handed_a_resolved_call_and_told_not_to_change_it(tmp_path):
    """No image to pick, no partition to discover, no flags to invent."""
    agent = FakeAgent(submits=False)
    _executor(tmp_path, agent).execute(CommandSpec(argv="true", timeout_s=0))
    prompt = agent.requests[0].prompt
    assert "change nothing in it" in prompt
    assert "blocking" in prompt.lower()
    assert "nothing reads your answer" in prompt


def test_the_turn_is_dispatched_to_the_runner_role_by_default(tmp_path):
    agent = FakeAgent(submits=False)
    _executor(tmp_path, agent).execute(CommandSpec(argv="true", timeout_s=0))
    assert agent.requests[0].role == "runner"


def test_the_record_names_its_artifacts_for_the_audit_trail(tmp_path):
    record = _executor(tmp_path, FakeAgent()).execute(
        CommandSpec(argv="echo hi", timeout_s=30, label="boot")
    )
    assert record.executor == "delegated"
    assert record.label == "boot"
    assert record.log_path and record.log_path.endswith(".log")
    assert len(record.artifacts) == 3


def test_it_declares_itself_distributed_and_gpu_capable():
    caps = DelegatedCommandExecutor.capabilities
    assert caps.gpu is True and caps.distributed is True
