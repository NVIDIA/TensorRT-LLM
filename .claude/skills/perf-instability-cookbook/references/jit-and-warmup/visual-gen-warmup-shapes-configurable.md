---
id: case-visual-gen-warmup-shapes-configurable
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [warmup-gap]
signals: [first-iter-spike, midrun-stall]
subsystems: [visual-gen-pipeline]
introduced_via: [incomplete-coverage]
phase: [warmup]
patterns: [pattern-jit-on-hot-path]
nvbugs: []
commits: ["18d02df53a15"]
success_prs: [12107]
failed_prs: []
---

# Visual-gen warmup shapes hardcoded per pipeline — un-warmed shape triggers torch.compile

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `18d02df53a15` · PR #12107 — Configurable warmup
  shapes for VisualGen. Merged to `main` 2026-03-13. No NVBug: tracked as JIRA
  **TRTLLM-11288**, and filed as `[feat]` rather than `[fix]` — the weakest
  provenance in this family. Read it as the *capability gap* behind the other
  visual-gen cases (a user had no way to declare the shapes they serve) rather
  than as a defect with a measured perf delta: the PR quantifies nothing, its
  30 new tests are CPU-only config/plan tests, and "seconds of delay" is its own
  phrasing, not a measurement. Kept here because the mechanism it removes is the
  same `pattern-jit-on-hot-path` the sibling cases hit, but do not cite it as
  evidence of a magnitude.
- **Symptom (variance signature):** any visual-gen request served at a
  (resolution, num_frames) tuple not in the pipeline's hardcoded warmup
  list triggers `torch.compile` recompilation with seconds of first-request
  latency — the metric therefore varies by tens of seconds depending on
  whether the client happens to request a warmed shape.
- **Root cause:** each pipeline class (e.g. WAN) hardcoded its warmup
  shapes as a list (`[(480, 832, 33), (480, 832, 81), (720, 1280, 81)]`),
  and users could not extend that list from config. Any request outside
  the hardcoded set re-entered torch.compile.
- **How introduced:** pipeline classes were authored with hardcoded warmup
  shapes intended as the "common case" — the assumption that other shapes
  would be rare was falsified in practice.
- **Fix mechanism:** add a `CompilationConfig` sub-config with `resolutions`
  and `num_frames` fields (combined via Cartesian product at warmup time),
  plus `BasePipeline.resolve_warmup_plan()` (user > model-default > empty)
  and `BasePipeline.validate_resolution()` for constraint checking, so
  users can declare the shapes they will serve.
- **Detection signal:** a torch.compile recompile span in the served
  timeline at a specific resolution / num_frames tuple; check the
  configured warmup plan vs the served shapes —
  `grep -n 'CompilationConfig\|resolve_warmup_plan\|default_warmup_resolutions' tensorrt_llm/`
  and inspect the effective warmup set in logs.
- **Prevention/guard:** a warmup-coverage warning at serve-start when the
  configured warmup does not include shapes the client's request set
  suggests; per-pipeline validation of the resolution/frame combinations.
  Bound the worry before widening a warmup plan, though: the PR reports testing
  128 shapes and finding that "dynamo automatic dynamic shapes merges all shapes
  after ~3 recompilations, cache limit is not a practical concern" — so the cost
  of an un-warmed shape is a handful of recompiles at the head of a long-lived
  process, not one per distinct shape forever. Frame counts are rounded to the
  nearest valid value with a warning (diffusers convention) rather than
  rejected, so a "warmed" plan can silently cover a neighbouring frame count.
- **Generalizes to:** `pattern-jit-on-hot-path`; carries to any config-
  driven warmup where users need to declare their shape distribution,
  torch.compile-wrapped pipelines with a shape-conditional graph, and
  future visual / audio pipelines added with hardcoded warmup lists.
