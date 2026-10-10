# Regression Cookbook — Kernel fusion

This module is the fused-op layer: the in-tree fused linear / QK-norm / RoPE /
gate kernels and the eligibility logic that decides whether the fused variant or
the unfused reference runs. The kernels themselves are rarely the defect. What
breaks is the *predicate* in front of them — an arch gate, a `torch.compile`
interaction, or a wrapper choice that routes a layer to a generic path such as
`F.linear` with an extra bias copy. Because the unfused path is functionally
identical, nothing errors and the config is unchanged; the layer is simply doing
more memory traffic than it used to. First thing to check: for the specific
layers named in the report, confirm the fused op actually ran on this arch and
under this compile mode.

## Recurring patterns in this module

- **Fast path silently fell back** — a fused op self-disables on a capability
  gate or because it is not visible to the compiler it is running under, and the
  generic implementation runs correctly and slower. Both cases here are that
  shape, from opposite directions: an SM100+ gate pushing GPT-OSS linears onto
  `F.linear`'s extra bias copy, and Qwen3.5's fused ops disabling *themselves*
  because raw Triton was Dynamo-visible under `torch.compile`. Test the fallback
  on the arch and compile mode the workload actually uses — a fused op verified
  in eager mode proves nothing about the piecewise-graph path.
  _(Instances: both cases below.)_

_Note, carried from the old kernel-and-fusion index: **fusion pattern drift** —
a fusion that stops matching after upstream graph/node changes (dep bumps,
refactors) — has **no confirmed case** in the corpus. Sightings live in
`data/pending.yaml` (nvbugs 6160248, 5940460, 5973199); see
`pattern-fusion-pattern-drift` in `data/patterns.yaml`. It is not listed as a
pattern above because no case in this module carries it._

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [GPT-OSS linears hit `F.linear`'s extra bias memory copy on SM100+](flinear-bias-copy-sm100.md) | `Request Generation Tokens Per Second` 403.43 → 375.14 (−7.0%) on B200, TP1/BS1; affects `qkv_proj` and `o_proj` | fast-path-fallback |
| [Qwen3.5 fused QK-norm/RoPE/gate ops self-disabled under torch.compile](qwen35-fused-ops-disabled-under-torch-compile.md) | ITL and TTFT up 5–20% with piecewise graphs on Qwen3.5-4B-FP8 | fast-path-fallback |
