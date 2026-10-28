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
  paged_mqa_logits_metadata JIT buckets; nvbug 6388787. Note PR #16178
  itself is titled `[None]` and its body names no bug, so a PR→bug lookup
  finds nothing.
  **Not** part of this case: PR #15961 (commit `0d97e9c76fd3`) is titled
  against nvbug 6388787, but it reverts an unrelated IPC-HMAC-key-over-fd
  change (#15654) that, per its PR description, silently deadlocked the MPI
  launch path of `trtllm-bench` on a `deepseek_v3.2_fp4-bench-pytorch-float4`
  ep:8 benchmark — a different defect with no file overlap with #16178, so it
  is neither a fix nor a failed attempt for this warmup gap. Do not fold the
  two mechanisms. That other mechanism has its own case —
  `perf-regression-cookbook/references/measurement-and-test/ipc-hmac-key-via-fd-breaks-bench.md`
  — recorded in the cookbook matching its nature: the launch deadlock is
  deterministic (a regression), the bucket hole varies run to run (this case).
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
  that `num_generations`. PR #16178's nsys evidence is exactly that shape:
  iters **140 / 144 / 149** with `_prepare_inputs` at **3,092 / 3,144 /
  3,158 ms** against `_forward_step ≈ 400 ms` — three mid-run spikes, hundreds
  of iters in, at iteration indices no config predicts (`num_generations` 139 /
  198 / 268 → buckets 160 / 224 / 288). deep_gemm's in-memory `LruCache` is
  torn down with the process (and `$DG_JIT_CACHE_DIR` defaults to a
  container-ephemeral path), so the set of stalls reshuffles on every fresh
  container. The PR's own before/after: rep 1 in a fresh container ran at
  4,233 vs ~9,750 tok/s for reps 2–3 (2.31×, CV ~40 %) on B300 `ep:8`, and
  0.99× / CV 0.24 % after the fix.
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
