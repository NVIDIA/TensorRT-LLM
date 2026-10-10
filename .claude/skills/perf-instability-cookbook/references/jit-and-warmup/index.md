# Instability Cookbook — JIT & warmup

This module is every place TRT-LLM decides *what to compile before serving*:
the model engine's warmup and autotuner-warmup passes
(`PyTorchModelEngine._warmup`, `_run_autotuner_warmup`), the CUDA-graph capture
passes that double as warmup for whatever they touch
(`_run_capture_pass(force_non_greedy=True)`, `cuda_graph_batch_sizes`), the
per-pipeline visual-gen warmup plans, and the kernel-side JIT caches those
passes are supposed to fill — DeepGEMM, flashinfer sampling, trtllm-gen FMHA /
NVRTC, Mamba SSD Triton. The one cold-start case here (the KV-aware ADP router)
belongs for the same reason: it is warmup of a *cache*, not of a kernel. What
recurs is a warmup grid that does not cover a mode, shape, or path the live
request can legitimately ask for, so a compile lands inside a served or measured
iteration.

First thing to check: dump the set of shapes / modes actually warmed and diff it
against what live traffic requested; then look for compile spans
(`ninja: Entering`, torch.compile / inductor, an NVRTC JIT miss) *inside* served
iterations rather than at startup. Ask also whether the cost is paid once per
process (that is a regression) or at unpredictable iterations (that is this
cookbook).

## Recurring patterns in this module

Match on these transferable failure modes, not on a case title.

- **JIT on the hot path** — warmup entirely missing for a mode / kernel / path,
  so a served iteration compiles. Worst sub-shape: a warmup that exists only as
  a *side effect* of another feature's warmup, so a config flag that disables
  that feature silently removes the coverage — with `cuda_graph_config: null`
  the advanced-sampling capture pass never runs, and nothing builds the
  flashinfer sampling kernels until a request is served. Check that each warmup
  is requested explicitly, not inherited.
  _(Instances: [FMHA cubin drop + kernel-selection alignment](fmha-cubin-drop-align-kernel-selection.md),
  [Mamba hybrid warmup gap](mamba-hybrid-warmup-gap.md),
  [DeepGEMM paged_mqa_logits prewarm](deepgemm-paged-mqa-logits-prewarm.md),
  [Visual-gen I2V warmup](visual-gen-i2v-warmup.md),
  [Visual-gen configurable warmup shapes](visual-gen-warmup-shapes-configurable.md),
  [flashinfer sampling JIT on served path](flashinfer-sampling-jit-on-served-path.md).)_
- **Warmup coverage hole** — warmup exists but its grid enumerates only the
  obvious shapes, and a live request lands on an uncovered bucket. Check the
  warmed bucket set against the buckets traffic can produce, and densify only
  those. _(Instance: [DeepGEMM paged_mqa_logits prewarm](deepgemm-paged-mqa-logits-prewarm.md)
  — CUDA-graph warmup touches only `cuda_graph_batch_sizes`, leaving the other
  32-aligned buckets to compile on live traffic.
  A general memory-pool warmup sibling (PR #10340) was removed on 2026-08-12:
  its nvbug 5820734 is a functional bug, an L0 post-merge accuracy failure in
  `accuracy/test_llm_api_pytorch.py`. Per `data/removed.yaml`, the removing pass
  first cited "6108808" for it — an unrelated, non-TensorRT-LLM bug; the
  conclusion held on 5820734, the id did not.)_
- **Warmup path mismatch** — warmup ran, but the served path (`image=None` vs
  `image=…`, T2V vs I2V, cached-kv vs no-cache) is a different sub-graph or
  shape, so the compiled artifact is invalidated. Check every served forward
  signature that changes the compiled graph, not just every shape.
  _(Instance: [Visual-gen I2V warmup](visual-gen-i2v-warmup.md).)_
- **Cold-start worker imbalance** — an affinity-scoring router pins every
  request carrying a shared prefix to the first rank that cached it; cold ranks
  stay cold and pay full re-prefill when a spike finally reaches them. Check the
  router's cold-start knob, not just the code: the mitigation is opt-in and
  **off by default**.
  _(Instance: [KV-aware ADP cold start](kv-aware-adp-cold-start.md).)_

One further pattern of this module, `pattern-warmup-sizing-tradeoff`, has **no
instance here since 2026-08-12** and so is not listed above: the cap-side case's
bug 5805494 is a functional bug (an int32 overflow / IMA at the 16384-token
warmup shape, i.e. a crash). The *other* end of the tradeoff is a live case in
the regression cookbook —
`perf-regression-cookbook/references/kernel-and-fusion/warmup-token-cap-revert.md`,
nvbug 6185713, a performance bug — and `data/removed.yaml` records that PR
#15887 has since removed the overflow at source. Read both before
re-introducing any global warmup cap; prefer per-op clamps.

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [FMHA cubin drop + align kernel selection](fmha-cubin-drop-align-kernel-selection.md) | ~6 s FMHA JIT recompile on an eager generation iteration after CUDA-graph warmup, at points set by live `max_past_kv_length` | warmup-gap |
| [Mamba hybrid models skip general warmup](mamba-hybrid-warmup-gap.md) | 12 Mamba SSD Triton variants uncompiled → ~30 s JIT at the first mixed chunked-prefill iter (typically iter 3); 5 unstable perf cases on 4 clusters | warmup-gap |
| [DeepGEMM paged_mqa_logits prewarm](deepgemm-paged-mqa-logits-prewarm.md) | DSA first-iter 2.31× variance; ~3 s per unwarmed 32-aligned bucket, `_prepare_inputs` 3,092 / 3,144 / 3,158 ms at iters 140 / 144 / 149 vs `_forward_step` ≈ 400 ms | warmup-gap |
| [Visual-gen warmup shapes hardcoded per pipeline](visual-gen-warmup-shapes-configurable.md) | `torch.compile` recompile on any (resolution, num_frames) tuple outside the pipeline's hardcoded list — the PR quantifies no magnitude | warmup-gap |
| [flashinfer sampling JIT on the served path](flashinfer-sampling-jit-on-served-path.md) | 98.2 s and 58.1 s mid-serving stalls → `Hang detected` → `MPI_Abort` (died at iteration 9) when `cuda_graph_config: null` | warmup-gap |
| [LTX-2 I2V requests trigger torch.compile recompilation](visual-gen-i2v-warmup.md) | recompile on the first I2V inference step — `gen_step1` 3.25 s vs 0.78 s once I2V is warmed (1-GPU) | warmup-path-mismatch |
| [KV-aware ADP router pins the shared prefix to one rank](kv-aware-adp-cold-start.md) | with `match_rate_threshold=0.1`, cold ranks pay the full re-prefill of the shared prompt when a traffic spike arrives | cold-start-imbalance |
