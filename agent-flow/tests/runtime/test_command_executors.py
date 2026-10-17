# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command execution, with no GPU, no scheduler and no model in sight.

The local executor is exercised for real — ``echo``, ``sh -c 'exit 3'``, a
command that outlives its deadline — because that is cheap and the thing
being checked is exactly the process boundary. The spool executor is
exercised against a fake watcher, since what it is responsible for is the
file protocol and not what runs on the far side of it.

The Slurm executor gets contract tests only. It renders nothing and decides
nothing; asserting more would require a controller.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from agent_flow.runtime.commands import (
    NOT_FOUND_RETURNCODE,
    TIMEOUT_RETURNCODE,
    CommandSpec,
    ExecutionRecord,
    LocalCommandExecutor,
    SlurmStepExecutor,
    SpoolCommandExecutor,
)

# --------------------------------------------------------------- CommandSpec


def test_a_string_is_a_shell_line_and_a_sequence_is_not():
    """The distinction decides whether a shell gets to split the words."""
    assert CommandSpec(argv="a b 'c d'").is_shell_line
    assert not CommandSpec(argv=("a", "b", "c d")).is_shell_line


def test_a_sequence_renders_quoted_so_a_log_can_be_pasted_back():
    assert CommandSpec(argv=("echo", "a b")).rendered == "echo 'a b'"


def test_a_shell_line_renders_verbatim():
    """Re-quoting an `srun` line would mean reimplementing the shell."""
    line = "FOO=1 srun --ntasks=8 bash -c 'exec thing'"
    assert CommandSpec(argv=line).rendered == line


# ------------------------------------------------------------ ExecutionRecord


def test_a_record_is_ok_only_when_it_exited_zero_and_finished():
    assert ExecutionRecord(command="x", returncode=0).ok
    assert not ExecutionRecord(command="x", returncode=1).ok
    # A timeout has said nothing about the thing it was running, so it is
    # never a pass -- even though the code it carries is conventional.
    assert not ExecutionRecord(command="x", returncode=0, timed_out=True).ok


def test_the_tail_is_the_end_of_stdout_and_stderr_together():
    record = ExecutionRecord(command="x", returncode=0, stdout="a\nb\nc\n", stderr="err\n")
    assert record.tail(2) == "c\nerr"


# ------------------------------------------------------- LocalCommandExecutor


def test_a_shell_line_runs_through_a_shell():
    record = LocalCommandExecutor().execute(CommandSpec(argv="echo one; echo two", timeout_s=30))
    assert record.ok
    assert record.stdout.split() == ["one", "two"]
    assert record.executor == "local"


def test_an_argv_sequence_is_not_word_split():
    """A probe's argument must survive spaces without quoting ceremony."""
    record = LocalCommandExecutor().execute(CommandSpec(argv=("echo", "a b c"), timeout_s=30))
    assert record.stdout == "a b c\n"


def test_env_is_layered_over_the_parent_rather_than_replacing_it():
    """Everything here runs inside an environment somebody else set up.

    A venv's PATH, a container's library paths, TMPDIR. Replacing the
    environment outright is the stricter choice and the wrong one: it turns
    a working host into a broken one.
    """
    record = LocalCommandExecutor().execute(
        CommandSpec(argv="echo $MV2FLOW_PROBE:$PATH", env={"MV2FLOW_PROBE": "set"}, timeout_s=30)
    )
    assert record.stdout.startswith("set:")
    assert len(record.stdout.strip()) > len("set:")


def test_a_non_zero_exit_is_reported_not_raised():
    record = LocalCommandExecutor().execute(CommandSpec(argv="exit 3", timeout_s=30))
    assert record.returncode == 3
    assert not record.ok
    assert not record.timed_out


def test_stderr_is_kept_separate_from_stdout():
    record = LocalCommandExecutor().execute(
        CommandSpec(argv="echo out; echo err 1>&2", timeout_s=30)
    )
    assert record.stdout.strip() == "out"
    assert record.stderr.strip() == "err"


def test_a_command_that_outlives_its_deadline_is_timed_out_not_merely_failed():
    """The two need different handling.

    A command that timed out has said nothing about what it was running; one
    that exited non-zero has. Collapsing them loses that.
    """
    started = time.monotonic()
    record = LocalCommandExecutor().execute(CommandSpec(argv="sleep 30", timeout_s=1))
    assert record.timed_out
    assert record.returncode == TIMEOUT_RETURNCODE
    assert not record.ok
    assert time.monotonic() - started < 20


def test_a_missing_executable_is_distinguishable_from_a_disagreement():
    """127 says the probe could not run; non-zero says it ran and disagreed."""
    record = LocalCommandExecutor().execute(
        CommandSpec(argv=("mv2flow-no-such-binary-xyz",), timeout_s=30)
    )
    assert record.returncode == NOT_FOUND_RETURNCODE
    assert "mv2flow-no-such-binary-xyz" in record.stderr


def test_an_interrupt_is_not_turned_into_a_failed_record(monkeypatch):
    """Ctrl-C has to reach the orchestrator, not become a gate that failed.

    The orchestrator catches ``KeyboardInterrupt`` around the whole run to
    checkpoint and tell the operator how to resume. An executor that caught
    it here and returned a non-zero record would turn "the operator stopped
    this" into "the target failed its gate", and the run would close with a
    verdict it never earned.
    """
    import subprocess as subprocess_mod

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess_mod, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        LocalCommandExecutor().execute(CommandSpec(argv="echo hi", timeout_s=30))


def test_the_record_carries_the_provenance_a_verdict_rests_on():
    record = LocalCommandExecutor().execute(CommandSpec(argv="echo hi", timeout_s=30, label="boot"))
    assert record.command == "echo hi"
    assert record.executor == "local"
    assert record.label == "boot"
    assert record.duration_s >= 0.0
    assert record.started_at > 0.0


def test_cwd_is_honoured_when_asked_for(tmp_path):
    (tmp_path / "marker").write_text("x", encoding="utf-8")
    record = LocalCommandExecutor().execute(
        CommandSpec(argv=("ls",), cwd=str(tmp_path), timeout_s=30)
    )
    assert record.stdout.split() == ["marker"]


def test_the_local_executor_declares_what_it_can_do():
    caps = LocalCommandExecutor.capabilities
    assert caps.gpu is True
    assert caps.distributed is False
    assert caps.persistent_allocation is False


# ------------------------------------------------------- SpoolCommandExecutor


def _watch(root, *, returncode=0, output="done\n", delay=0.02):
    """A fake session watcher: drain `pending`, write the log, then the rc."""

    def run():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            scripts = sorted((root / "pending").glob("*.sh")) if (root / "pending").is_dir() else []
            if scripts:
                name = scripts[0].stem
                time.sleep(delay)
                (root / "logs").mkdir(parents=True, exist_ok=True)
                (root / "logs" / f"{name}.log").write_text(output, encoding="utf-8")
                (root / "done").mkdir(parents=True, exist_ok=True)
                # `.rc` last, same ordering rule as the dispatch mailbox.
                (root / "done" / f"{name}.rc").write_text(f"{returncode}\n", encoding="utf-8")
                return
            time.sleep(0.005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_spool_command_round_trips_through_the_filesystem(tmp_path):
    _watch(tmp_path, output="lm-eval gsm8k average accuracy: 90.5\n")
    record = SpoolCommandExecutor(tmp_path, poll_s=0.01).execute(
        CommandSpec(argv="trtllm-eval --model x gsm8k", timeout_s=30, label="accuracy")
    )
    assert record.ok
    assert "90.5" in record.stdout
    assert record.executor == "spool"
    assert record.label == "accuracy"


def test_a_receipt_is_read_only_once_it_holds_an_exit_code(tmp_path):
    """A receipt created empty and filled a moment later must not read as exit code 1.

    The fake watcher above writes its `.rc` without renaming it into place, so
    the executor could see the file between its creation and its first byte;
    that intermittently turned a passing command into a failure.
    """

    def slow_writer():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            scripts = (
                sorted((tmp_path / "pending").glob("*.sh"))
                if (tmp_path / "pending").is_dir()
                else []
            )
            if scripts:
                name = scripts[0].stem
                (tmp_path / "done").mkdir(parents=True, exist_ok=True)
                receipt = tmp_path / "done" / f"{name}.rc"
                receipt.touch()  # exists, empty
                time.sleep(0.3)
                receipt.write_text("0\n", encoding="utf-8")
                return
            time.sleep(0.005)

    threading.Thread(target=slow_writer, daemon=True).start()
    record = SpoolCommandExecutor(tmp_path, poll_s=0.01).execute(
        CommandSpec(argv="true", timeout_s=30)
    )
    assert record.returncode == 0 and record.ok


def test_the_receipt_is_the_exit_code(tmp_path):
    _watch(tmp_path, returncode=7, output="it failed\n")
    record = SpoolCommandExecutor(tmp_path, poll_s=0.01).execute(
        CommandSpec(argv="false", timeout_s=30)
    )
    assert record.returncode == 7
    assert not record.ok


def test_the_script_carries_the_environment_as_exports(tmp_path):
    """The far side is a fresh shell; an env dict has to become shell text."""
    executor = SpoolCommandExecutor(tmp_path, poll_s=0.01)
    _watch(tmp_path)
    executor.execute(
        CommandSpec(argv="echo hi", env={"TRTLLM_MODELING_V2": "require"}, timeout_s=30)
    )
    # The watcher consumed it, so read what it was handed: re-submit with no
    # watcher and inspect the pending script instead.
    lonely = SpoolCommandExecutor(tmp_path, poll_s=0.01)
    lonely.execute(CommandSpec(argv="echo hi", env={"A": "b c"}, timeout_s=0))
    script = sorted((tmp_path / "pending").glob("*.sh"))[-1].read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash\nset -u\n")
    assert "export A='b c'\n" in script
    assert script.rstrip().endswith("echo hi")


def test_the_script_is_renamed_into_pending_never_written_in_place(tmp_path):
    """A watcher must not be able to claim a half-written script."""
    SpoolCommandExecutor(tmp_path, poll_s=0.01).execute(CommandSpec(argv="echo hi", timeout_s=0))
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list((tmp_path / "pending").glob("*.sh"))) == 1


def test_a_spool_command_nobody_drains_times_out_with_whatever_log_exists(tmp_path):
    record = SpoolCommandExecutor(tmp_path, poll_s=0.01).execute(
        CommandSpec(argv="sleep 100", timeout_s=0)
    )
    assert record.timed_out
    assert record.returncode == TIMEOUT_RETURNCODE
    assert record.log_path and record.log_path.endswith(".log")


def test_the_spool_declares_that_its_allocation_is_persistent():
    """The reason to prefer it: one queue wait per run, not one per command."""
    caps = SpoolCommandExecutor.capabilities
    assert caps.persistent_allocation is True
    assert caps.distributed is True
    assert caps.gpu is True


# ---------------------------------------------------------- SlurmStepExecutor


def test_the_slurm_executor_runs_the_rendered_line_here_and_says_so():
    """An `srun` line is a shell line; the step is still a local subprocess.

    What the separate class buys is the two facts nobody can read off the
    command string: that this executor is distributed, and that the record
    came from Slurm.
    """
    record = SlurmStepExecutor().execute(CommandSpec(argv="echo pretend-srun", timeout_s=30))
    assert record.ok
    assert record.stdout.strip() == "pretend-srun"
    assert record.executor == "slurm"


def test_the_slurm_executor_declares_itself_distributed():
    assert SlurmStepExecutor.capabilities.distributed is True
    assert SlurmStepExecutor(max_world_size=8).capabilities.max_world_size == 8


def test_bounding_the_world_size_does_not_leak_into_the_class():
    """Two runs of different sizes in one process must not see each other's."""
    sized = SlurmStepExecutor(max_world_size=8)
    assert sized.capabilities.max_world_size == 8
    assert SlurmStepExecutor.capabilities.max_world_size is None


def test_every_executor_reports_the_command_it_actually_ran():
    """The record has to be defensible without the caller's memory of it."""
    for executor in (LocalCommandExecutor(), SlurmStepExecutor()):
        record = executor.execute(CommandSpec(argv=("echo", "x y"), timeout_s=30))
        assert record.command == "echo 'x y'"


def test_the_executors_are_importable_without_a_gpu_or_a_scheduler():
    """Every test in this file is the proof; this states the intent."""
    assert sys.modules["agent_flow.runtime.commands.local"]
