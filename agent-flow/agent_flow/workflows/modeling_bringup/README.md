# Modeling-bringup workflow

## Introduction

`modeling_bringup` is an application built on the
[`agent-flow`](../../../README.md) framework. It orchestrates a team of coding
agents to implement a new model in TensorRT-LLM from reference model code,
checkpoints, and task-specific bring-up requirements.

The workflow specializes the generic [`agent_team`](../agent_team/README.md)
plan-build-verify loop for TensorRT-LLM model development. Its goal is to turn a
model bring-up request into an implementation that uses TensorRT-LLM
infrastructure and is validated against accuracy criteria.

![Agent-team workflow](../agent_team/docs/agent-team-workflow.svg)

## Capabilities

### What it can do

- Implement text-only LLMs in TensorRT-LLM using its high-performance backends
  and modules.
- Produce implementations compatible with core TensorRT-LLM features including
  CUDA graphs, CPU overlap, and chunked prefill.
- Support various model-parallel strategies used by TensorRT-LLM.
- Validate the implemented model's accuracy on datasets.

### What it cannot do yet

- Multimodal models are not well exercised yet.
- Complex serving features such as disaggregated serving and speculative
  decoding are not validated by the workflow today.
- Accuracy debugging can still fall short of human expert tuning, and human
  effort may still be needed to fine-tune accuracy in complex cases. For
  example, a human expert may tune GSM8K accuracy to 96%, while
  `modeling_bringup` may only reach 90% in some cases.

## Usage

### Installation

```bash
pip install -e .[devel]
```

Note: this automatically installs Claude-Agent-SDK and Codex-SDK. Users still
need to complete Claude Code and Codex authentication and activation themselves.

Start with [`quick_start.md`](quick_start.md), an end-to-end walkthrough that
brings up ChatGLM3-6B. It covers fetching the reference Hugging Face repo,
writing the task brief, launching the workflow, and watching QA complete final
verification.

To bring up your own model, copy [`task.example.yaml`](task.example.yaml) to
`task.yaml`, fill in the source paths, completion criteria, and implementation
tips, then run:

```bash
modeling-bringup \
    --task task.yaml \
    --workspace workspace/modeling-bringup
```

This is equivalent to:

```bash
python -m agent_flow.workflows.modeling_bringup.cli \
    --task task.yaml \
    --workspace workspace/modeling-bringup
```

![Modeling-bringup workflow](docs/workflow.svg)

## Human Feedback

You can run a task incrementally by passing feedback with `--feedback`:

```bash
modeling-bringup \
    --task task.yaml \
    --workspace workspace/modeling-bringup \
    --feedback "your feedback for the next iteration"
```

`agent-flow` restores the previous run from the workspace state files, then
continues the task with the new feedback. This lets you steer the workflow one
iteration at a time without losing the plan, progress, or checkpoint state.

To rerun a task from scratch, add `--clean`:

```bash
modeling-bringup \
    --task task.yaml \
    --workspace workspace/modeling-bringup \
    --clean
```

## Task file

The three required fields are `reference_code_path`, `checkpoint_path`, and
`trtllm_repo_path`. They are validated up front: a missing field or a path that
does not exist aborts the run before any agent is constructed.

`completion_criteria` and `implements_tips` are optional lists of strings. The
`implements_tips` name is part of the task schema and is used for implementation
guidance. Extra YAML keys are preserved on disk for the agents to read.

To run from a Slurm login node, add an optional `slurm-environment` section
with `slurm_partition` and `docker_image`. The presence of this section is
what injects the Slurm/container prompt block into the agents. See
[`task.slurm.example.yaml`](task.slurm.example.yaml) for a full example.

All flags from the generic `agent-team` CLI are accepted as-is. See the
[`agent_team` README](../agent_team/README.md#useful-flags) for their semantics.
The workflow source lives in `agent_flow/workflows/modeling_bringup/`; import
the bundle programmatically with
`from agent_flow.workflows.modeling_bringup import MODELING_BRINGUP_PROMPTS`.

## Environment

`modeling-bringup` agents support two execution modes for different runtime
environments.

### Container mode

In container mode, agents edit and run code inside the TensorRT-LLM container.
This mode is suitable when the target model can be developed and validated in a
single container environment.

![Container mode](docs/workflow-container.svg)

### Slurm mode

Slurm mode is for models that cannot run on a single node. In this mode, the
workflow runs directly on the login node of a Slurm cluster and uses the cluster
environment to execute larger model bring-up and validation tasks.

![Slurm mode](docs/workflow-slurm-mode.svg)

To run in Slurm mode, add a `slurm-environment` section to the task YAML, as
shown in `task.slurm.example.yaml`, and provide the required
`slurm_partition` and `docker_image` values.

In Slurm mode, the build-phase agents share an additional workspace file
`test_command.md` that caches verified `srun`/`sbatch` wrappers around
`trtllm-bench`, `trtllm-eval`, and pytest. Coder, Reviewer, and QA append to
the file, delete entries that no longer pass, and overwrite entries when a
command is corrected. On local tasks without `slurm-environment`, the Slurm
prompt block is not injected and this file is not created. Bring-up-specific
drafting rules live in
`agent_flow/workflows/modeling_bringup/prompts/_common.py:TEST_COMMAND_CACHE`.

## Implementation

`modeling-bringup` inherits the implementation of
[`agent_team`](../agent_team/README.md). It reuses the same orchestration loop,
workspace state files, checkpoint/resume behavior, MCP tools, and CLI flags, and
specializes the agents with TensorRT-LLM modeling bring-up prompts.

The `agent_team` workflow starts with a planning phase: `PlanDrafter` writes the
implementation plan and acceptance criteria, then `PlanReviewer` reviews them
and either approves or asks for a revision. After the plan is approved, the
workflow enters the build phase, where `Coder` implements the change,
`Reviewer` checks the implementation, and `QA` runs the final verification. If
review or QA rejects the result, the workflow loops back to the coder for the
next iteration.

For the full implementation contract, shared workspace files, flags, and MCP
tool details, see the [`agent_team` README](../agent_team/README.md).

On top of the inherited `agent_team` implementation, `modeling-bringup` extends
the agent prompts with domain-specific principles for TensorRT-LLM model
bring-up. These prompts guide the agents to respect TensorRT-LLM implementation
boundaries, prefer existing high-performance modules, validate model accuracy,
and use TensorRT-LLM-related skills during planning, coding, review, and QA.

![Modeling agent workflow](docs/modeling-agent-workflow.svg)
