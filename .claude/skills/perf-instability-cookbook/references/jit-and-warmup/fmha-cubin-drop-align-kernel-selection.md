---
id: case-fmha-cubin-drop-align-kernel-selection
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [warmup-gap]
signals: [midrun-stall, first-iter-spike]
subsystems: [fmha-kernel, mla-attention, cuda-graph]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-jit-on-hot-path]
nvbugs: []
commits: ["ebf19a496e1d"]
success_prs: [13505]
failed_prs: [13312]
---

# ~6 s FMHA JIT recompile in eager generation because kernel selection didn't match CUDA-graph warmup

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `ebf19a496e1d` · PR #13505 — Drop cubin and eliminate
  ~6 s FMHA JIT recompile in eager generation by aligning kernel selection
  with CUDA graph warmup. No NVBug: the PR is titled `[None]` and its body
  names none, so this is PR-only provenance. Merged to `main` 2026-05-06.
  Code change is only two files (`cpp/tensorrt_llm/common/attentionOp.cpp`,
  `cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h`); the
  rest of the diff is the regenerated FP4-KV cubin set.
- **Failed attempts:** PR #13312 — same fix minus the cubin half, titled
  "Eliminate ~6s FMHA JIT recompile in eager generation by aligning kernel
  selection with CUDA graph warmup" (CLOSED unmerged; #13505's body opens
  "from: …/pull/13312"). It was abandoned after repeated `L0_MergeRequest_PR`
  failures and a `/bot kill` rather than on a review objection, then re-filed as
  #13505 with the cubin-export restriction folded in. Treat it as the
  superseded predecessor, not as a rejected approach — the mechanism it
  proposed is the one that landed.
- **Symptom (variance signature):** an approximately 6 s stall on the first
  eager generation iteration after CUDA-graph warmup; the graph-captured
  path was fine, but a subsequent eager iteration selected a *different*
  FMHA kernel variant than the one warmup had compiled, forcing a JIT
  recompile. Latency mode / variance signature — a large P99 or first-iter
  outlier that steady-state throughput hides.
- **Root cause:** in `AttentionOp::mlaGeneration` the trtllm-gen FMHA runner was
  handed `tllmRunnerParams.mMaxSeqLenKv = generation_params.max_past_kv_length`
  — the *actual current* KV length — whereas CUDA-graph warmup runs at the full
  cache capacity. `mMaxSeqLenKv` participates in kernel selection, so the two
  modes resolved to different kernels for the same layer and the eager path took
  an NVRTC JIT miss inline. Because `max_past_kv_length` grows with the
  conversation, *which* kernel the eager path asks for is a function of live
  traffic, not of the config — hence a stall that recurs at unpredictable points
  rather than once at startup. (CodeRabbit's release note for the PR reads
  "Fixed attention window parameter handling during multi-turn generation in MLA
  scenarios.")
- **How introduced:** `mMaxSeqLenKv` was set from the live sequence length
  because that is the honest value for the kernel's own bounds, without noticing
  it also keys kernel selection and therefore had to match what warmup compiled.
  The un-restricted cubin export compounded it: non-FP4-KV variants were left to
  the NVRTC JIT path, so a selection miss meant a compile rather than a
  precompiled-cubin hit.
- **Fix mechanism:** override `mMaxSeqLenKv` with
  `generation_params.max_attention_window_size` (the max cache capacity) so FMHA
  picks the same kernel as CUDA-graph warmup — safe on this path because, per
  the in-diff comment, "the strides do not depend on mMaxSeqLenKv, and extra KV
  CTAs exit early through seqLensKvPtr". Alongside it, hoist the
  sliding-window determination into a precomputed `XQAParams::is_sliding_window`
  set once in `convertMMHAParamsToXQAParams` from the declared mask type / RoPE
  config, "so kernel-selection code does not need to re-derive it from
  window/rope heuristics" — i.e. remove the duplicated heuristic that let the
  two modes disagree. The diff carries a `TODO` to mirror the `is_swa` + W+1
  logic once MLA gains SWA support, so this alignment is not yet total.
- **Detection signal:** an FMHA JIT compile span visible in the first eager
  iter's nsys trace, sized ~seconds, with no matching span in the warmup
  region; grep for FMHA kernel-name divergence between graph-mode and
  eager-mode profiles: `nsys stats --format=csv --report=cudaapisum <rep>.nsys-rep`
  and compare kernel identifiers across the two modes. To check whether the fix
  is present:
  `grep -n 'mMaxSeqLenKv\|is_sliding_window' cpp/tensorrt_llm/common/attentionOp.cpp`
  — the fixed tree assigns `max_attention_window_size`, the broken one
  `max_past_kv_length`.
- **Prevention/guard:** a graph-vs-eager parity test that asserts identical
  kernel selection for shared shapes; treat the selection heuristic as an
  interface between the two modes. Concretely: any field that feeds kernel
  selection must be derived from *capacity*, not from the current request's
  live extent, or warmup cannot cover it.
- **Generalizes to:** `pattern-jit-on-hot-path`; carries to autotuner /
  cutlass-JIT / DeepGEMM / Triton kernels where graph and eager paths use
  different code, MoE routing that differs across capture-vs-run, and
  attention backends with separate prefill / decode kernel-selection paths.
