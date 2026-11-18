---
id: case-mamba2-flashinfer-head-group-ratio-gate
type: regression-case
family: kernel-and-fusion
module: model-definition
maturity: full
regression_class: [fast-path-fallback]
signals: [throughput-drop, itl-increase, perf-ci-bar-failure, slower-kernel-in-trace]
subsystems: [model-definition]
introduced_via: [new-feature, incomplete-coverage]
phase: [decode]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6405747", "6419042"]
commits: ["1bb55a3e0c47"]
success_prs: [16031]
failed_prs: []
---

# Mamba2 selective-state-update drops to the native kernel on a too-narrow head_group_ratio allowlist

> Part of the [Model definition regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `6405747` / `6419042` · commit `1bb55a3e0c47` ·
  PR #16031 — "[TRTLLM-13977][fix] Fix NT3 NVFP4 perf regression in
  Blackwell".
- **Symptom:** an NT3 NVFP4 perf regression on Blackwell, per
  PR #16031's title and description; the decode-side state update is the
  gated path, so expect the loss in Output-Token-Time / ITL. No percentage is
  stated in the PR. A separate defect can mask this one on the same model:
  per PR #15961's description, the `nemotron_3_ultra_550b_nvfp4-serve`
  `/health` endpoint never bound ("did not become ready within 3600s") on
  *both* baseline and candidate wheels on 2026-07-05, an IPC-HMAC-fd hang
  fixed separately — a serve-path hang that blocks measurement is not the
  mean shift being measured. That confounder is now a case in its own right:
  [IPC HMAC key by fd deadlocks the benchmark launcher](../measurement-and-test/ipc-hmac-key-via-fd-breaks-bench.md)
  — check it first when a window's probes come back null on *both* sides.
- **Root cause:** `Mamba2Mixer.__init__` in
  `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` gates the flashinfer
  selective-state-update kernel on a hard-coded allowlist. The
  `supported_head_group_ratios = [1, 8, 16]` list under-reported what
  upstream flashinfer actually supports (`[1, 2, 4, 8, 16, 32, 64]`, per the
  `kernel_selective_state_update_stp.cuh` header the fix cites), so the NT
  model family's ratio fell outside it, `_use_flashinfer` evaluated False,
  and `self.selective_state_update_func` was bound to the in-tree
  `selective_state_update_native` (the Triton implementation) instead of
  flashinfer's. Only the decode-side state update is gated this way —
  prefill goes through `mamba_chunk_scan_combined` regardless — which is why
  the regression reads as an Output-Token-Time / ITL loss. The ratio is
  computed **post-shard** (`head_group_ratio = self.tp_nheads // self.tp_ngroups`),
  so eligibility depends on the run's TP size, not on the checkpoint alone.
- **How introduced:** PR #13476 ("[TRTLLM-12242][feat] Add Marlin NVFP4
  backend for MoE and Linear on Hopper", merged 2026-06-25). Its `mamba2_mixer.py` hunk (+16/−14) widened the gate
  from `self._use_flashinfer = head_dim in supported_head_dims` to a
  three-way conjunction that also requires `head_group_ratio` and `d_state`
  membership. PR #16031's description names #13476 explicitly: "That PR adds
  constraints about `head_group_ratio` to enable `flashinfer`." A
  Hopper-targeted feature PR tightened a gate shared with Blackwell.
- **Fix mechanism:** a 3-line edit (+3/−1) widening
  `supported_head_group_ratios` to `[1, 2, 4, 8, 16, 32, 64]` to match
  flashinfer v0.6.14, with the upstream header URL left in a comment as the
  source of truth. `supported_head_dims` and `supported_d_states` are
  untouched — only the over-narrow condition is relaxed, so the guard still
  fails closed on genuinely unsupported head dims (e.g. Nemotron-v2-Nano's
  `mamba_head_dim=80`).
- **Detection signal:** the fallback announces itself once, at info level:
  `grep -n "for selective state update" <serve-or-bench log>` returns
  `Using native for selective state update` on a bad build and
  `Using flashinfer for selective state update` on a good one (both are
  `logger.info_once(..., key="selective_state_update")`, so there is exactly
  one line per run — a `tail` of the log will miss it). Then audit the gate
  against the run's actual parallelism with
  `grep -n "supported_head_group_ratios\|supported_d_states\|head_group_ratio =" tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`
  and evaluate `tp_nheads // tp_ngroups` at the deployed TP size, not from
  the HF config. In an nsys decode trace the flashinfer SSM kernel is
  replaced by the Triton `selective_state_update` kernel. **Mis-triage
  warning:** on a MoE + NVFP4 workload the MoE GEMMs are the obvious
  suspect and the gated SSM layer is not — check backend-selection log lines
  across the range before bisecting kernel commits.
- **Prevention/guard:** none added — the PR is a 3-line list edit whose only
  guard is the upstream-header URL now pinned in the comment. Two concrete
  gaps: (1) the allowlist duplicates a table that lives in flashinfer's
  headers, so it silently drifts at every flashinfer bump — it should be
  derived from, or unit-tested against, the installed flashinfer rather than
  transcribed; (2) the fallback branch logs `info_once`, not
  `warning_once` — the adjacent stochastic-rounding fallback in the same
  constructor *does* use `logger.warning_once`, so the louder idiom was
  already in the file. A unit test asserting `_use_flashinfer` is True for
  each shipped Nemotron config at its production TP size would have failed
  in #13476's own CI.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — an eligibility
  allowlist excludes a supported configuration and a correct-but-slower
  in-tree kernel runs. Carries to: hard-coded allowlists transcribed from an
  external library's supported-shape table (flashinfer / cutlass / cuDNN
  dispatch lists) that go stale on the next dep bump; eligibility predicates
  computed on **post-sharding** values, where the same checkpoint qualifies
  at one TP/EP size and not another; a feature PR for one architecture
  tightening a gate shared with another (Marlin-on-Hopper narrowing a
  Blackwell mamba path); and any gate narrowed from a single condition to a
  conjunction where only one conjunct was measured.
