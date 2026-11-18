# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A check the spec defines (``checks``): its command, run on the target, passes when it exits 0.

The spec says what the check is for (``purpose``) and how to run it (``run``);
the command is the judge, as a test is. It runs as a gate does: on a session,
from the tree being checked, at its member's devices.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from ..infra.compute.commands import ended_early
from .check_runtime import ShellRunner
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

#: How long a check of the spec's runs when it does not say.
DEFAULT_TIMEOUT_S = 3600


def run_spec_check(
    gate: str,
    entry: Mapping[str, Any],
    *,
    task_data: Mapping[str, Any],
    runner: ShellRunner,
    member: str | None = None,
) -> GateResult:
    """Run the command ``entry`` names and judge it by its exit status."""
    _verdict = verdict_factory(inspect_tree(task_data))
    command = str(entry["run"])
    purpose = " ".join(str(entry["purpose"]).split())
    timeout = int(entry.get("timeout") or DEFAULT_TIMEOUT_S)
    started = time.monotonic()
    result = runner(command, None, timeout)
    combined = result.stdout + result.stderr
    ran = dict(
        name=gate,
        member=member,
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
    )
    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out:
        return _verdict(passed=False, detail=f"{ended_early(result, timeout)}: {purpose}", **ran)
    passed = result.returncode == 0
    return _verdict(passed=passed, detail=f"exited {result.returncode}: {purpose}", **ran)


__all__ = ["DEFAULT_TIMEOUT_S", "run_spec_check"]
