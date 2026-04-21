# Custom Claude Code Skills & Agents for TensorRT-LLM

## Background: Skills & agents in Claude Code

Claude Code supports two extensibility mechanisms — **skills** and **agents** —
that let teams encode domain expertise into reusable, version-controlled
components.

**Skills** are markdown playbooks that Claude follows step-by-step when
triggered. They are invoked via `/slash-commands` (e.g. `/perf-analysis`) or
matched automatically from natural-language requests. Each skill lives in its
own directory under `.claude/skills/` and can bundle reference materials that
Claude reads during execution. See
[Custom slash commands](https://code.claude.com/docs/en/skills)
for details.

**Agents** (sub-agents) are specialist workers that Claude spawns in a separate
context to handle focused tasks. Each agent has its own system prompt, tool
access, and domain knowledge. Claude delegates to them when it determines a task
fits a specialist's scope, while you can also invoke agents directly. Agent
definitions live under `.claude/agents/`. See
[Custom sub-agents](https://code.claude.com/docs/en/sub-agents)
for details.

## How skills and agents are loaded

For users who are working with Claude Code under TensorRT-LLM project directory,
skills and agents are automatically discovered by Claude Code at startup — no
manual registration needed. Files placed in `.claude/skills/` and
`.claude/agents/` are picked up by convention.

To verify what's loaded, launch Claude Code under TensorRT-LLM project directory
and type `/skills` or `/agents` in the Claude Code prompt to see available
skills and sub-agents.

## How to use skills and agents

There are two ways to trigger skills and agents:

1. **Automatic dispatch** — just describe what you need in plain language
   (e.g. "profile this workload", "compile TensorRT-LLM"). Claude Code will
   match your request to the appropriate skill or delegate to the right
   sub-agent automatically.

2. **Manual invoke** — type `/<skill-name>` (e.g. `/perf-analysis`,
   `/serve-config-guide`) to explicitly run a skill. For sub-agents, type
   `@"<agent-name>" (agent)` (e.g. `@"exec-compile-specialist (agent)"`) to
   delegate a task directly. This is useful when you know exactly which workflow you want.

In most cases, automatic dispatch is sufficient — you don't need to memorize
skill or agent names. Manual invoke is there for when you want precise control.

References:
* [Extend Claude with skills](https://code.claude.com/docs/en/skills)
* [Work with subagents](https://code.claude.com/docs/en/sub-agents#work-with-subagents)

## Naming convention

Every skill and agent name uses the format `<prefix>-<descriptive-name>`.
The prefix identifies the primary work area; the descriptive part should be
short and not repeat it.

| Prefix | Domain | Definition |
|---|---|---|
| `ad-` | AutoDeploy | Model onboarding, pipeline debugging, and execution for the AutoDeploy backend |
| `ci-` | CI/CD | CI failure retrieval, test diagnostics, and pipeline workflows |
| `exec-` | Execution infra | Environment setup and job execution (compile, run, container) |
| `kernel-` | Kernel development | Kernel writing, generation, and kernel-specific transforms |
| `perf-` | Performance work | Profiling, analysis, and tuning above the kernel layer (kernel modifications belong under `kernel-`) |
| `serve-` | Serving | Serving configuration, deployment, and runtime workflows |
| `trtllm-` | TRT-LLM dev workflows | Workflows for reading, modifying, and contributing to the codebase (static subsystem knowledge belongs in repo docs) |

Guidelines:

* If a skill doesn't fit any prefix, propose a new one and agree on its
  boundary before using it.
* Use the prefix of the skill's **primary** domain, even if it orchestrates
  across multiple domains.
* Agents follow the same convention.
* Good: `exec-local-compile`, `kernel-cuda-writing`, `perf-host-analysis`
* Bad: `exec-trtllm-compile`, `kernel-cuda-kernel-writing`,
  `perf-trtllm-host-analysis`
