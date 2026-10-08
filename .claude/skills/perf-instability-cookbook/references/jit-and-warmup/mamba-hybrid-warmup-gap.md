---
id: case-mamba-hybrid-warmup-gap
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [warmup-gap]
signals: [first-iter-spike, midrun-stall, rep-to-rep-variance, perf-ci-bar-flake]
subsystems: [attention-kernel, model-engine]
introduced_via: [incomplete-coverage]
phase: [prefill]
patterns: [pattern-jit-on-hot-path]
nvbugs: []
commits: ["002f099e59b9"]
success_prs: [16177]
failed_prs: [15876]
---

# Mamba hybrid models skip general warmup — iter-3 30 s SSD kernel JIT

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `002f099e59b9` · PR #16177 — Close Mamba hybrid
  warmup gap in autotuner warmup. Fixes 5 unstable perf cases across
  4 clusters (GB200 / GB300 / B300) on Nemotron 3 Super 120B and Nemotron
  Nano 12B v2, per the PR summary. No NVBug: the PR is titled `[None]` and
  names none — the five cases came from perf-CI, so the provenance here is
  PR-only by construction.
- **Failed attempts:** PR #15876 — "Pre-JIT Mamba SSD HAS_INITSTATES=True
  kernels during warmup" (CLOSED unmerged 2026-07-12). It hooked
  `PyTorchModelEngine._warmup` with a per-module, `hasattr`-guarded
  `warmup_ssd_initstates_kernels` opt-in that pre-JIT'd `mamba_chunk_scan_combined`
  + `_state_passing_fwd` only. It *worked* on its one case — B300
  `nemotron_nano_12b_v2` 3-rep median 9472 → 9788 t/s with **CV 13.9 % → 1.25 %**
  — but was closed by its own author as "Superseded by #16177, which fixes both
  this case … and the nemotron_3_super_120b run-1 warmup deficit across
  GB200/GB300/B300 with a single autotuner-warmup path". The lesson for a
  future agent: a per-module pre-JIT hook is the narrower shape of this fix and
  covers one model family; extending `_run_autotuner_warmup` covers both. Do not
  re-propose the per-module hook.
- **Symptom (variance signature):** Nemotron 3 Super 120B (and other
  `MambaHybridCacheManager` models) trail on their first serve/bench
  iterations because the warmup path leaves 12 Mamba SSD Triton kernel
  variants uncompiled. The first mixed chunked-prefill iter with cached
  tokens (typically iter 3) then stalls ~30 s JIT-compiling them mid-
  inference, producing a large run-1 throughput deficit and a P99 spike;
  the perf number therefore varies rep-to-rep depending on whether the
  compile landed in the measurement window.
  **Launch mode decides whether this reads as variance or as a flat loss, and
  the PR's own tables show both:** in *serve* mode one `trtllm-serve` process
  spans all reps, so the stall is paid once and run 1 sits far below the warmed
  steady state (Case 1 GB200: r2 is +31 % over r1 before the fix, +0.05 % after;
  Case 5 B300: r1 is −33 % against the warmed r2/r3). In *bench* mode the worker
  is respawned per rep, so **every** rep pays the same stall (Cases 2/3/4:
  "r1 ≈ r2 both slow") and the defect presents as a uniform deficit with no
  rep-to-rep spread at all. It is filed here because the reported cases were
  perf-CI *instability* and the serve-mode signature is a genuine within-job
  gap — but an agent reading only a bench-mode 3-rep CV will see a stable,
  uniformly-slow workload and must not conclude "no warmup gap".
- **Root cause:** the root-cause chain (all three must hold, per the PR):
  (1) `_general_warmup` is skipped entirely for Mamba hybrid models —
  `can_run_general_warmup` is `False` when
  `isinstance(kv_cache_manager, MambaHybridCacheManager)`; (2)
  `_run_autotuner_warmup` issues a single `least_requests=True` prefill of one
  sequence of `curr_max_num_tokens`, so
  `cu_seqlens_to_chunk_indices_offsets_triton` takes its `num_seqs == 1` fast
  path and `_cu_seqlens_triton_kernel` never launches; (3) dummy warmup requests
  have `num_cached_tokens_per_seq = 0`, so `use_initial_states` is `False` and
  every `HAS_INITSTATES=True` variant stays uncompiled. The 12 missing kernels
  per the PR: `_chunk_state_varlen_kernel` × 5 configs,
  `_state_passing_fwd_kernel` × 4 (`HAS_INITSTATES` × `IS_CONT_BATCHED`),
  `_chunk_scan_fwd_kernel` × 2 (`HAS_INITSTATES`), `_cu_seqlens_triton_kernel` × 1.
- **How introduced:** the `MambaHybridCacheManager` path bypassed
  `_general_warmup` because early Mamba integrations didn't need it, and
  the autotuner warmup wasn't extended to the multi-seq HAS_INITSTATES=True
  case that mixed chunked-prefill later required.
- **Fix mechanism:** extend `_run_autotuner_warmup`
  (`tensorrt_llm/_torch/pyexecutor/model_engine.py`) with a small shape sweep for
  all models (pure-ctx-max plus mixed ctx+gen) gated by `TLLM_AUTOTUNE_SWEEP`
  (default on); for Mamba hybrid specifically add two further passes that build
  multi-sequence prefill batches (`least_requests=False`) and force
  `HAS_INITSTATES=True` through a new `TLLM_MAMBA_WARMUP_FORCE_INITSTATES` hook
  in `Mamba2Metadata.prepare()`
  (`tensorrt_llm/_torch/modules/mamba/mamba2_metadata.py`), gated by
  `TLLM_MAMBA_MULTISEQ_WARMUP` (default on). The force hook lives in warmup only
  and does not change real-inference behavior.
  Measured per PR (5 cases, before → after): serve-mode run-1 output tok/s
  6344.72 → 9725.66 (**+53 %**, GB200 nemotron_3_super_120b) and 1854 → 3278
  (**+77 %**, B300, same model) with P99 E2EL 53811 → 17745 ms and
  49476 → 11480 ms respectively; bench-mode 2-rep averages 6469 → 7441 t/s
  (**+15 %**, GB300 nemotron_nano_12b_v2), 7318 → 8346 t/s (**+14 %**, GB200),
  and mean TTFT 3798 → 1362 ms (**−64 %**, B300) — the PR calls TTFT "where the
  mamba warmup gap most directly manifests", since a 130 s / 512-request bench
  run amortizes the stall down to +3 % on aggregate throughput.
- **Detection signal:** first-iter throughput deficit + P99 spike on
  Mamba hybrid models; nsys `triton.jit` spans appearing near iter 3;
  `grep -n 'can_run_general_warmup\|HAS_INITSTATES\|TLLM_MAMBA_MULTISEQ_WARMUP\|TLLM_MAMBA_WARMUP_FORCE_INITSTATES' tensorrt_llm/_torch/`
  to check whether the warmup fix is present. Because bench mode hides the
  signal in rep-to-rep spread (see Symptom), the discriminating measurement is
  **run 1 vs run 2 inside one serve process**, not a 3-rep CV; on bench-mode
  cases use mean TTFT rather than aggregate throughput, which the run length
  dilutes.
- **Prevention/guard:** any custom cache-manager path that opts out of
  general warmup must document what it replaces and provide equivalent
  coverage; a per-model warmup coverage assertion would catch this class.
- **Generalizes to:** `pattern-jit-on-hot-path`; carries to any custom
  cache manager that inherits a subset of warmup (Mamba, hybrid, SSM
  variants), Triton kernels compiled lazily with more than one variant
  key, and models added after a warmup was authored without a matching
  warmup extension.
