---
id: case-deepgemm-paged-mqa-logits-prewarm
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [warmup-gap]
signals: [first-iter-spike, rep-to-rep-variance, midrun-stall]
subsystems: [dsa-attention, cuda-graph]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-jit-on-hot-path, pattern-warmup-coverage-hole]
nvbugs: ["6388787"]
commits: ["8dba04bebfb2"]
success_prs: [16178]
failed_prs: []
---

# DeepGEMM paged_mqa_logits_metadata JIT buckets not prewarmed — DSA first-iter 2.31× variance

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `8dba04bebfb2` · PR #16178 — Prewarm DeepGEMM
  paged_mqa_logits_metadata JIT buckets; nvbug 6388787
  (a performance bug, closed as verified, B200,
  `deepseek_v3.2_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-input_output_len:128,128-ep:8-gpus:8`).
  The NVBug carries the link in both directions: it traces the reported
  instability to a JIT-compile stall in DeepGEMM's `paged_mqa_logits_metadata`
  scheduler kernel, reproduces the bucket arithmetic below, and names PR #16178
  as the fix, since merged. Note PR #16178 itself is titled `[None]`,
  so a PR→bug lookup finds nothing — the association exists only on the bug side.
  **Not** part of this case: PR #15961 (commit `0d97e9c76fd3`) is *also*
  titled against nvbug 6388787, but it reverts an unrelated
  IPC-HMAC-key-over-fd change (#15654) that silently deadlocked the MPI
  launch path on the same benchmark — a different defect with no file overlap
  with #16178, so it is neither a fix nor a failed attempt for this warmup gap.
  6388787 is therefore a two-root-cause bug; do not fold the two mechanisms.
  That other mechanism now has its own case —
  `perf-regression-cookbook/references/measurement-and-test/ipc-hmac-key-via-fd-breaks-bench.md`
  — so the two halves of 6388787 are recorded once each, in the cookbook
  matching their nature: the launch deadlock is deterministic (a regression),
  the bucket hole varies run to run (this case).
- **Symptom (variance signature):** DSA models exhibit first-iter throughput
  variance of 2.31× because DeepGEMM's `paged_mqa_logits_metadata` JIT-
  compiles a fresh cubin (spawning `nvcc` → `cicc` → `ptxas`, ~3 s per
  bucket on Blackwell) the first time each 32-aligned batch bucket is
  requested. Because CUDA-graph warmup only touches
  `cuda_graph_batch_sizes` buckets, the *other* 32-aligned buckets are
  unwarmed and compile on live traffic — the perf-CI number therefore
  varies depending on which bucket lands in the measurement window.
  **Why this is instability and not a cold-start regression:** the stall is not
  paid once per process at a fixed point — it fires whenever a *new* uncovered
  bucket is first requested, at whatever iteration traffic happens to produce
  that `num_generations`. The bug's nsys evidence is exactly that shape: iters
  **140 / 144 / 149** with `_prepare_inputs` at **3,092 / 3,144 / 3,158 ms**
  against `_forward_step ≈ 400 ms` — three mid-run spikes, hundreds of iters in,
  at iteration indices no config predicts. deep_gemm's in-memory `LruCache` is
  torn down with the process, so the set of stalls reshuffles on every fresh
  container / rerun. The bug itself was filed as an unstable regression, and a
  three-rerun check of intra-commit CV on one commit found the perf unstable.
- **Root cause:** DSA's `Indexer.prepare_scheduler_metadata`
  (`tensorrt_llm/_torch/attention_backend/sparse/dsa.py`) calls
  `deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)`
  on every iteration; the underlying kernel
  `deep_gemm::sched::smxx_paged_mqa_logits_metadata` is templated on
  `kAlignedBatchSize = ceil(num_generations, 32)`, and deep_gemm's Python-
  side JIT (`deep_gemm_cpp_tllm.so`) compiles a fresh cubin per bucket on
  first request. For `max_batch_size=512`, cuda-graph warmup hit only
  `{32, 64, 96, 128, 192, 256, 320, 384, 448, 512}` — leaving
  `{160, 224, 288, 352, 416, 480}` unwarmed.
- **How introduced:** the DSA indexer + DeepGEMM integration inherited
  cuda-graph warmup's batch-bucket set, but the JIT bucket granularity is
  strictly finer (every 32-aligned batch) — a coverage gap by construction.
- **Fix mechanism:** prewarm every 32-aligned batch bucket up to
  `max_batch_size` during CUDA-graph warmup, so no bucket compiles on the
  live path. Same failure class as
  [case-mamba-hybrid-warmup-gap](mamba-hybrid-warmup-gap.md).
- **Detection signal:** `nvcc` / `cicc` / `ptxas` child processes visible
  in the first few served iters (or intermittently mid-run), each ~3 s;
  first-iter throughput deficit that varies by ~2× rep-to-rep on DSA
  models; `grep -n 'get_paged_mqa_logits_metadata\|kAlignedBatchSize' tensorrt_llm/_torch/attention_backend/sparse/dsa.py`.
  The sharpest single signal is a per-iter breakdown where `_prepare_inputs`
  alone jumps to seconds while `_forward_step` is unchanged — attributing the
  spike to `_prepare_inputs` rather than to the model forward is what separates
  this from a genuine kernel regression.
- **Prevention/guard:** any per-iter JIT whose bucket key is finer than
  the cuda-graph warmup set must have its own prewarm loop; assert that
  every bucket the kernel can be templated on has a warmup entry.
- **Generalizes to:** `pattern-warmup-coverage-hole`; carries to any
  DeepGEMM / cutlass-JIT / TorchInductor kernel keyed on a bucket set
  broader than cuda-graph batch bucket set, MoE variants keyed by
  aligned-batch, and any per-iter JIT whose bucket granularity was not
  cross-checked against warmup coverage.
