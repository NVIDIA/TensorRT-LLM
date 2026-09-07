# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote-execution prompt contract for perf-optimize agents.

The static policy explains *how* agents cross the local/remote boundary.
``RemoteExecutionContext`` supplies the concrete locations for one turn.
Keeping those concerns separate lets persistent agent sessions retain a
stable system prompt while round/item/attempt paths change between turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import yaml

REMOTE_EXECUTION_POLICY = """\
## Remote execution policy

This run uses remote execution. The agent process and its cwd remain on
the local control host. The control workspace and execution workspace
are separate namespaces, not workspace mirrors.

The per-turn `REMOTE_EXECUTION_CONTEXT` is authoritative for the remote
host and all paths. It overrides any earlier unqualified reference to
"the workspace", "the attempt directory", or another directory.

### Control side

- Read roadmap, progress, prior-agent reports, and other coordination
  files from the named control paths.
- Write Markdown reports and workflow-owned compact YAML/JSON directly
  to the exact control output paths.
- Local file tools such as `Read`, `Write`, and `Edit` may access only
  control paths.
- `control_task_path` is the canonical campaign specification. Read its
  non-path settings locally. Never use a path-valued field from it as a
  local path or as the location of an execution operation.

### Execution side

- Source and live-config inspection or editing, Git, builds, tests,
  serving, benchmarking, profiling, GPU work, Slurm, and raw artifacts
  belong on the execution host.
- Access execution paths only with `Bash` invoking SSH through
  `remote_host`. Never pass an execution path to a local file tool or
  test it as though it were local.
- `execution_task_path` is the execution-side task projection. Read it
  through SSH when an execution command needs a path-valued task field.
- Do not copy files between control and execution. The workflow owns all
  cross-side synchronization.

The local cwd is only a control-side communication directory. It does
not make code, config, Git, performance, GPU, or Slurm operations local.

### Minimal examples

Substitute the exact values from `REMOTE_EXECUTION_CONTEXT`:

```text
# Inspect code
Bash("ssh <remote_host> 'cd <execution_command_cwd> && rg \"symbol\" src/'")

# Apply a source patch; the edit still happens in the remote worktree
Bash("ssh <remote_host> 'cd <execution_command_cwd> && git apply -' <<'PATCH'\n<unified diff>\nPATCH")

# Run a test or benchmark
Bash("ssh <remote_host> 'cd <execution_command_cwd> && <command>'")

# Submit and inspect Slurm work
Bash("ssh <remote_host> 'cd <execution_command_cwd> && sbatch <job-script>'")
Bash("ssh <remote_host> 'squeue -j <job-id>'")
```

Never run `sbatch`, `srun`, `squeue`, `sacct`, or `scancel` directly on
the local control host. Even when another prompt section or a skill shows
a bare Slurm command, run it through SSH to `remote_host`.
"""


@dataclass(frozen=True)
class RemoteExecutionContext:
    """Concrete control/execution locations for one remote agent turn.

    Common fields have a fixed order. Role-specific locations stay flat
    and must identify their side in the field name, which keeps rendered
    contexts concise without losing the namespace boundary.
    """

    remote_host: str
    control_workspace: str
    control_cwd: str
    control_task_path: str
    execution_workspace: str
    execution_task_path: str
    execution_campaign_repo: str
    execution_command_cwd: str
    locations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.locations.items():
            if not name.startswith(("control_", "execution_")):
                raise ValueError(
                    "remote execution context location names must start with "
                    f"'control_' or 'execution_': {name!r}"
                )
            if not value:
                raise ValueError(f"remote execution context location {name!r} is empty")

    def render(self) -> str:
        """Render an authoritative, copy-pasteable YAML context block."""
        values = {
            "mode": "remote",
            "remote_host": self.remote_host,
            "control_workspace": self.control_workspace,
            "control_cwd": self.control_cwd,
            "control_task_path": self.control_task_path,
            "execution_workspace": self.execution_workspace,
            "execution_task_path": self.execution_task_path,
            "execution_campaign_repo": self.execution_campaign_repo,
            "execution_command_cwd": self.execution_command_cwd,
            **self.locations,
        }
        rendered = yaml.safe_dump(
            {"REMOTE_EXECUTION_CONTEXT": values},
            sort_keys=False,
            allow_unicode=True,
        ).rstrip()
        return (
            "The following remote execution context is authoritative for "
            "this turn:\n\n"
            f"```yaml\n{rendered}\n```"
        )


def append_remote_execution_context(
    instruction: str,
    context: RemoteExecutionContext,
) -> str:
    """Append a turn's dynamic context after all role instructions."""
    return instruction.rstrip() + "\n\n" + context.render()


__all__ = [
    "REMOTE_EXECUTION_POLICY",
    "RemoteExecutionContext",
    "append_remote_execution_context",
]
