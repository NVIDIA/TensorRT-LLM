# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a command somewhere only an agent can reach.

THE SITUATION THIS EXISTS FOR

Some clusters put the scheduler behind a tool surface rather than a CLI:
``sinfo`` and ``sbatch`` cannot contact the controller from the process that
wants them, and the only path through is a broker the *agent* holds and the
orchestrator does not. The orchestrator is a plain Python process; it is not
an MCP client and cannot become one just because a gate needs to run.

THE OBVIOUS ANSWER IS THE WRONG ONE

"Let the agent run the gate and tell us the number." That hands the verdict
to the thing being audited. Every guarantee in this workflow rests on the
orchestrator reading evidence itself, and an agent that reports `94.7` is
not evidence -- it is a claim about evidence, produced by a process with an
interest in the answer.

WHAT THIS DOES INSTEAD

Split the job in two, and only delegate the half that has no verdict in it:

    orchestrator                         agent (holds the broker tool)
    ------------                         -----------------------------
    write <dir>/<name>.sh
    ask for ONE submission          -->  make exactly the call it was given
                                         (no image to pick, no partition to
                                          discover, no flags to invent)
    wait for <dir>/<name>.rc
    read the log and the exit code
    apply its own thresholds

The submitted script writes its own log and its own exit code to the shared
filesystem, and renames the receipt into place **last** -- so a reader that
sees it can trust the log beside it is complete. That is the same ordering
rule the spool and the dispatch mailbox use, for the same reason.

The consequence worth stating: **the agent is a trigger, not a witness.** If
it lies about what happened, nothing downstream changes, because nothing
downstream reads what it said. If it never answers, the run times out on the
receipt rather than on the turn.

WHY THE COMMAND IS A FILE

The submission templates these brokers take embed the command inside a
quoted string. Real gate commands carry ``--model_dir="/path"``,
``--chat_template_kwargs={"enable_thinking": false}`` and worse, so
interpolating one into a quoted template produces a call that is either
mis-parsed or silently truncated. A path has no quoting hazard.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, ClassVar

from ..turns import TurnExecutor, TurnRequest
from .base import CommandExecutor, CommandSpec, ExecutionCapabilities, ExecutionRecord
from .local import TIMEOUT_RETURNCODE

#: Renders the instruction that makes the far side run ``script_path``.
#: Supplied by whatever knows the cluster's submission surface; this module
#: deliberately does not.
SubmissionRenderer = Callable[[str], str]


class DelegatedCommandExecutor(CommandExecutor):
    """Ask an agent to launch a command; read the result off the filesystem."""

    name: ClassVar[str] = "delegated"
    capabilities = ExecutionCapabilities(gpu=True, distributed=True)

    def __init__(
        self,
        turns: TurnExecutor,
        *,
        root: Path | str,
        render_submission: SubmissionRenderer,
        role: str = "runner",
        poll_s: float = 5.0,
        name_prefix: str = "gate",
    ) -> None:
        self.turns = turns
        self.root = Path(root)
        self.render_submission = render_submission
        self.role = role
        self.poll_s = poll_s
        self.name_prefix = name_prefix

    def _paths(self, name: str) -> tuple[Path, Path, Path]:
        return (
            self.root / f"{name}.sh",
            self.root / f"{name}.log",
            self.root / f"{name}.rc",
        )

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        staging = path.with_suffix(path.suffix + ".tmp")
        staging.write_text(text, encoding="utf-8")
        staging.rename(path)

    def _script(self, spec: CommandSpec, log: Path, receipt: Path) -> str:
        """The script the far side runs. It publishes its own verdict.

        Two details are load-bearing, both for the same reason -- the script
        has to outlive whatever the command does, or there is no receipt and
        the run waits out its full timeout on a job that already finished:

        * ``set -u`` but deliberately **not** ``set -e``. The exit code is
          the measurement; aborting on it destroys the thing being measured.
        * the command runs in a **subshell**. A payload that calls ``exit``
          -- directly, or through a wrapper that does -- would otherwise
          terminate this script before it could record anything.
        """
        exports = "".join(
            f"export {key}={_shell_quote(str(value))}\n"
            for key, value in sorted((spec.env or {}).items())
        )
        return (
            "#!/bin/bash\n"
            "set -u\n"
            f"{exports}"
            f"( {spec.rendered} ) > {log} 2>&1\n"
            "rc=$?\n"
            # The receipt is written last and renamed into place, so its
            # existence means the log beside it is complete.
            f"echo $rc > {receipt}.tmp\n"
            f"mv {receipt}.tmp {receipt}\n"
        )

    def execute(self, spec: CommandSpec) -> ExecutionRecord:
        name = f"{self.name_prefix}-{int(time.time() * 1000)}"
        script, log, receipt = self._paths(name)
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_atomic(script, self._script(spec, log, receipt))

        started_at = time.time()
        started = time.monotonic()

        prompt = (
            f"Run one command on the cluster. **Make exactly this call and change "
            f"nothing in it** — the environment was resolved before you started and "
            f"is not yours to rediscover:\n\n"
            f"```python\n{self.render_submission(str(script))}\n```\n\n"
            f"Then poll the job's status until it leaves PENDING/RUNNING, and end "
            f"your turn. Do **not** use a blocking submit call: the tool layer aborts "
            f"after a few minutes of silence and every job here is longer than that, "
            f"so blocking reports a failure for a job that is running fine.\n\n"
            f"You do not need to report the result and nothing reads your answer. "
            f"The job writes its own log to `{log}` and its own exit code to "
            f"`{receipt}`, and the orchestrator reads both itself. Your only job is "
            f"to get it submitted.\n\n"
            f"If the submission itself fails, say so plainly — that is a real "
            f"failure and it is the one thing this turn can tell us."
        )
        self.turns.run(TurnRequest(role=self.role, what=f"submit {name}", prompt=prompt))

        deadline = started + spec.timeout_s
        while not receipt.exists():
            if time.monotonic() > deadline:
                tail = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                return ExecutionRecord(
                    command=spec.rendered,
                    returncode=TIMEOUT_RETURNCODE,
                    stdout=tail,
                    timed_out=True,
                    duration_s=time.monotonic() - started,
                    executor=self.name,
                    label=spec.label,
                    log_path=str(log),
                    started_at=started_at,
                )
            time.sleep(self.poll_s)

        try:
            returncode = int((receipt.read_text(encoding="utf-8").strip() or "1"))
        except ValueError:
            returncode = 1
        output = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        return ExecutionRecord(
            command=spec.rendered,
            returncode=returncode,
            stdout=output,
            duration_s=time.monotonic() - started,
            executor=self.name,
            label=spec.label,
            log_path=str(log),
            started_at=started_at,
            artifacts=(str(script), str(log), str(receipt)),
        )


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


__all__ = ["DelegatedCommandExecutor", "SubmissionRenderer"]
