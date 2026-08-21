---
id: case-triton-to-cpp-op
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [host-overhead, launch]
signals: [many-small-kernels, recompilation-churn, host-prep-on-critical-path]
architectures: [any-sm]
model_scope: [sparse-attention, deepseek-v32, model-agnostic]
phase: [decode]
patterns: [pattern-triton-to-precompiled-cpp-op]
accuracy_risk: lossless
apply_via_kind: [kernel-change]
knobs: []
specialists: [kernel-cuda-specialist, perf-host-analysis]
commits: []
interactions:
  - {case: case-relax-tl-constexpr, relation: alternative-to, note: the cheaper Triton-side fix when you keep Triton}
measured: []
---

# Replace a Triton hot-path kernel with a precompiled C++ custom op to cut host dispatch overhead

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Applies when:** a Triton kernel sits on the **host-bound** hot path and is
  small/cheap on the GPU, so its host-side cost dominates — Triton's per-launch
  Python dispatch, autotune-cache lookup, argument marshalling, and JIT /
  recompilation overhead. Symptoms: host time in the Triton launch wrapper, GPU
  idle around the op, recompiles across layers/shapes. Classify host-bound first
  with `perf-host-analysis` / `perf-nsight-systems` — a *compute-bound* Triton
  kernel (near SOL) will **not** benefit from this. Instance: the DSA indexer's
  `triton_gather_k_cache` and `triton_convert_req_index_to_global_index`.
- **Mechanism:** Triton pays a fixed Python-side cost per launch (and recompiles
  per new `constexpr`/shape). Re-implementing the kernel in C++/CUDA and exposing
  it as a Torch library op (`torch.ops.trtllm.*`) makes it precompiled with a
  thin C++ launch path, removing the JIT and most of the per-launch host
  overhead. A `register_fake` (meta) implementation keeps it traceable under
  `torch.compile` / fake tensors.
- **Generalizes to:** the pattern "**when a Triton kernel is host-bound (tiny GPU
  work, heavy launch/JIT overhead), move it to a precompiled C++/CUDA custom op
  with a `register_fake`.**" Carries to: other small index/gather/scatter/copy
  kernels on the decode hot path; any Triton op recompiling across layers or
  shapes; ops where a CUDA implementation already exists or is cheap to write.
  Adapt by: writing the `.cu` kernel + thop binding, registering the op schema +
  `register_fake`, and swapping the call site to `torch.ops.trtllm.*`; keep the
  Triton version as reference/fallback and port its tests. Only worth it when
  **host** overhead, not GPU SOL, is the bottleneck.
- **Apply via:** **not a server config knob** — a kernel/op change. Delegate to
  **kernel-cuda-specialist** (raw CUDA C++ + thop/
  pybind). Touch points (prior art, TRT-LLM repo): kernels
  `cpp/tensorrt_llm/kernels/{indexerKCacheGather.cu, convertReqIndexToGlobal.cu}`
  (+ `.h`), Torch op bindings
  `cpp/tensorrt_llm/thop/{IndexerKCacheGatherOp.cpp, convertReqIndexToGlobalOp.cpp}`
  (+ `CMakeLists.txt`), the `register_fake` meta impls in
  `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`, and the call sites in
  `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  (`torch.ops.trtllm.indexer_k_cache_gather_op`,
  `torch.ops.trtllm.convert_req_index_to_global`).
- **Expected effect:** lower host launch overhead and fewer recompiles → smaller
  GPU-idle gaps and higher throughput when host-bound; no change to GPU compute.
  Largest at decode / small token counts where launch overhead dominates. Record
  the local Δ (per-iteration host time, launch overhead, throughput) — PR #12581
  ships it without a public number; do **not** cite one not measured locally.
- **Accuracy risk:** lossless **by intent**, but it is a kernel rewrite, so
  **verify vs the Triton/PyTorch reference within tolerance** — a C++ kernel bug
  changes outputs. Watch dtype/stride and edge cases (empty gather, non-contiguous
  `k_cache`).
- **Verify:** correctness — the new op tests
  `tests/unittest/_torch/attention/sparse/test_cpp_custom_ops.py` compare both
  ops to a Python reference incl. edge cases; DSA e2e still passes. Perf —
  `perf-host-analysis`: host launch time + GPU-idle around the op, throughput
  before vs after; confirm `register_fake` lets `torch.compile` trace the op.
- **Rollback:** swap the call sites back to the Triton wrappers (kept in
  `kernel.py`); trigger: any accuracy mismatch outside tolerance or >5% perf
  regression.
- **Prior art:** PR #12581; paths above; new op test
  `test_cpp_custom_ops.py` (replaces the removed `test_triton_gather_k_cache.py`).
  Owning specialist: **kernel-cuda-specialist**; detection: **perf-host-analysis**.
  Related: the [relax-`tl.constexpr` case](relax-tl-constexpr.md) (the cheaper Triton-side fix when
  you keep Triton).
