# Quick Start

End-to-end walkthrough of bringing up ChatGLM3-6B in TensorRT-LLM via the
modeling-bringup workflow. The agents read a reference HuggingFace
implementation plus checkpoint, translate the model into TRT-LLM, and
iterate until QA verifies the accuracy gate (GSM8K via `trtllm-eval`)
passes.

Install `agent-flow` first — see the [root README](../../README.md#preparation)
for the Claude Code / Codex SDK prerequisites and `pip install -e .`.

Fetch the reference HuggingFace repo (the agents treat its
`modeling_*.py` as the source of truth for layer math, and the weights
as the validation oracle) and the TensorRT-LLM checkout the agents will
write into:

```bash
git clone https://huggingface.co/zai-org/chatglm3-6b /tmp/chatglm3-6b
git clone git@github.com:NVIDIA/TensorRT-LLM.git /tmp/TensorRT-LLM
```

Write the bring-up brief into the workspace as YAML. The three
`*_path` fields are required and must point to existing paths — the
workflow refuses to start otherwise. `completion_criteria` is the
contract QA ultimately tests against (keep each bullet bound to *what
passes* — an eval score, a feature being on — not *how to get there*).
`implements_tips` is free-form guidance for the PlanDrafter and Coder
that QA does not enforce:

```bash
mkdir -p workspace/chatglm3-6b-bringup
cat > workspace/chatglm3-6b-bringup/task.yaml <<'EOF'
reference_code_path: /tmp/chatglm3-6b/modeling_chatglm.py
checkpoint_path:     /tmp/chatglm3-6b/
trtllm_repo_path:    /tmp/TensorRT-LLM

completion_criteria:
  - "GSM8K score measured via `trtllm-eval` is greater than 95"

implements_tips:
  - "Use KVCacheManagerV2"
  - "Support common features, such as CUDA graph and scheduler overlap"
EOF
```

Kick off the workflow. The workspace directory is created on demand and
holds all shared state:

```bash
modeling-bringup \
    --task workspace/chatglm3-6b-bringup/task.yaml \
    --workspace workspace/chatglm3-6b-bringup
```

The run starts in the plan phase (PlanDrafter ⇄ PlanReviewer), then
enters the build loop (Coder ⇄ Reviewer ⇄ QA), and exits when QA
`APPROVE`s a build whose `weighted_score` clears `--min-score`. Tail
`progress.yaml` to watch each agent's decisions and rationale land in
real time.

Steering the run:

- **Interrupted?** Rerun the same command — the orchestrator
  auto-resumes from the checkpoint.
- **Start over:** pass `--clean` to wipe the workspace's
  workflow-managed files and restart from the plan phase.

See the [modeling-bringup README](README.md) for the prompt-extension
overview, and the [agent-team README](../agent_team/README.md) for the
full flag table, workflow contract, shared-file schema, and MCP tools.
