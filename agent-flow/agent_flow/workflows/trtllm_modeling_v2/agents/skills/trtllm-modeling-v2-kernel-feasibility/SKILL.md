---
name: trtllm-modeling-v2-kernel-feasibility
description: Load when deciding whether a checkpoint can be assembled from the current kernel catalog, and what has to be added first. The reference material the PlanDrafter maps a checkpoint with — the derivation checklist, the OK / CERT / GAP triage rules, and the generated op inventory. The procedure and the vocabulary.yaml schema are in the PlanDrafter's prompt.
---

# trtllm-modeling-v2-kernel-feasibility

The question — **can this checkpoint be assembled from the catalog as it
stands, and if not, what has to be added first?** — is the PlanDrafter's, and
its procedure and the `vocabulary.yaml` schema are in that role's prompt
(`agent_flow/workflows/trtllm_modeling_v2/agents/prompts/plan_drafter.py`). The
schema the orchestrator enforces is
`agent_flow/workflows/trtllm_modeling_v2/vocabulary.py`.

What this skill carries is the material the mapping is judged with:

| reference | read it for |
|---|---|
| `references/derivation-checklist.md` | every axis to record per computation step, so the step list carries this checkpoint's value on each |
| `references/mapping-and-triage.md` | how to read an entry's certified column, the OK / CERT / GAP outcomes with their dispatch, the native-torch exception, and what a CERT row must quote |
| `references/trtllm-ops.md` | the generated `torch.ops.*` inventory: tier-1 candidates for a GAP. It cannot see FlashInfer's API (the survey's `surface.md` lists that) or pybind bindings (`attention/thop_attention` wraps one), so absence here is never a ceiling on its own |
