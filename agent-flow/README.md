# agent-flow

`agent-flow` is a torch-like framework for composing agent-backed layers.
Each layer implements `forward`, and each `AgentLayer.forward` performs one
Claude Code or Codex SDK execution. Modules can be composed directly or used
to build small orchestration harnesses.

## Installation

Since `agent-flow` is not a published wheel package, use the following way to install:

```bash
pip install -e /path/to/agent-flow
```

This will automatically install two SDK runtimes as dependencies: [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) and [Codex App Server Python SDK](https://github.com/openai/codex/tree/main/sdk/python).

## Quick start

```python
from agent_flow import (
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    CLAUDE_CODE_DEFAULT_MODEL,
    SessionConfig,
)

agent = AgentLayer(
    AgentLayerConfig(
        name="Agent",
        backend=BackendConfig(kind="claude-code",
                              model=CLAUDE_CODE_DEFAULT_MODEL),
        session=SessionConfig(mode="stateless"),
    )
)
reply = agent("introduce the project")
```

`AgentLayer.forward` takes a prompt string and returns the agent's final
response as a string. Compose layers with `Sequential` (or any custom
`Module`) to pipe one agent's response into the next.

The examples use `CLAUDE_CODE_DEFAULT_MODEL` and `CODEX_DEFAULT_MODEL` as
shared defaults. Override them with same-named environment variables before
running an example:

```bash
export CLAUDE_CODE_DEFAULT_MODEL=claude-opus-4-7[1m]
export CODEX_DEFAULT_MODEL=gpt-5.4
```

Run it directly:

```bash
python examples/quick_start.py
```

## Workflows

Three ready-to-run multi-agent workflows ship as subpackages of
`agent_flow.workflows`:

- [`agent_flow.workflows.agent_team`](agent_flow/workflows/agent_team) —
  generic plan ↔ build harness (PlanDrafter ↔ PlanReviewer [↔ Human] →
  Coder ↔ Reviewer ↔ QA). Launch via `agent-team`.
- [`agent_flow.workflows.modeling_bringup`](agent_flow/workflows/modeling_bringup) —
  domain specialization of `agent_team` for TensorRT-LLM model bring-up.
  Reuses the agent_team CLI, orchestrator, MCP tools, and checkpoint logic
  verbatim, and only swaps the prompt bundle (source boundary, validation
  policy, attention/MoE/full-model scope, accuracy-gate framework).
  Launch via `modeling-bringup`.

```
PlanDrafter ⇄ PlanReviewer [⇄ Human]  →  Coder ⇄ Reviewer  →  QA  ✔
```

![Agent-team workflow](./agent_flow/workflows/agent_team/agent-team.svg)

### Run the modeling-bringup workflow

For an end-to-end walkthrough that brings up ChatGLM3-6B from scratch,
follow [`agent_flow/workflows/modeling_bringup/quick_start.md`](agent_flow/workflows/modeling_bringup/quick_start.md).
It walks through fetching the reference HuggingFace repo, writing the
bring-up brief, launching the workflow, and watching QA close out.

To bring up your own model, write your bring-up brief into a `task.yaml`.
A ready-to-edit template lives at
[`agent_flow/workflows/modeling_bringup/task.example.yaml`](agent_flow/workflows/modeling_bringup/task.example.yaml) —
copy it, fill in the reference-code path, checkpoint path, TensorRT-LLM repo
path, target stage (attention vs. full-model), and any accuracy gate. If
launching from a Slurm login node, also add `slurm-environment` with
`slurm_partition` and `docker_image`; only then are Slurm/container prompt
instructions enabled. Then run:

```bash
modeling-bringup \
    --task task.yaml \
    --workspace workspace/modeling-bringup
```

The workspace directory is created on demand and holds all shared state for
the run (`plan.md`, `acceptance-criteria.md`, `progress.yaml`, `status.md`,
and the resume checkpoint). If the run is interrupted, rerun the same
command — it auto-resumes from the checkpoint. Pass `--clean` to wipe the
checkpoint and start over.

### Steering a live run

- **Mid-run feedback:** stop the workflow (Ctrl-C), re-run with
  `--feedback "your note"` (or a file path), and the next coder/reviewer/qa
  turn will pick it up via `read_human_feedback`. Entries are appended, never
  cleared, so each iteration sees the full guidance history.
- **Pre-supplied plan:** pass `--plan path/to/plan.md` and
  `--acceptance-criteria path/to/criteria.md` to skip the plan phase
  entirely and go straight to the build loop.
- **Enable the plan-stage human checkpoint:** pass `--plan-human-review`
  to have the PlanDrafter ask the human for sign-off via `ask_human`
  after PlanReviewer `APPROVE`s. Off by default — PlanReviewer
  `APPROVE` flows straight into the build phase.
- **Enable the Coder's `ask_human` escape hatch:** pass
  `--build-human-review` to let the Coder pause mid-iteration and ask
  the human a question when it hits user-only information (credentials,
  environment facts, etc.). Off by default — the build phase is
  expected to be unattended once the plan is approved.

All flags from `agent-team` are accepted by `modeling-bringup`
as-is. See [`agent_flow/workflows/agent_team/README.md`](agent_flow/workflows/agent_team/README.md)
for the underlying workflow contract (stages, shared files, MCP tools, full
flag table), and
[`agent_flow/workflows/modeling_bringup/README.md`](agent_flow/workflows/modeling_bringup/README.md)
for the prompt-extension layout.

To embed a workflow programmatically:

```python
from pathlib import Path

from agent_flow.workflows.agent_team import AgentTeamWorkflow
from agent_flow.workflows.modeling_bringup import MODELING_BRINGUP_PROMPTS

with AgentTeamWorkflow(workspace=Path("workspace/run"),
                       prompts=MODELING_BRINGUP_PROMPTS) as workflow:
    workflow.run("...")
```
