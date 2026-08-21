---
id: case-move-bookkeeping-into-cpp-op
type: case
family: runtime-execution
maturity: full
bottleneck: [host-overhead, launch, sync]
signals: [gpu-idle-between-steps, host-prep-on-critical-path]
architectures: [sm90, sm100]
model_scope: [model-agnostic, sparse-attention, mla]
phase: [decode]
patterns: [pattern-fold-bookkeeping-into-consuming-op]
accuracy_risk: lossless
apply_via_kind: [code-change, kernel-change]
knobs: []
specialists: [kernel-cuda-specialist, perf-torch-sync-free, perf-host-analysis]
commits: ['ae84aaddb6f1', '7081f254cf']
interactions:
  - {case: case-cache-step-invariant-per-layer, relation: composes-with, note: same DSA host-overhead theme}
  - {case: case-hoist-torch-compile-closures, relation: composes-with, note: same DSA host-overhead theme}
measured: []
---

# Move per-step tensor bookkeeping off the Python hot path into a C++/fused op

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `ae84aaddb6f1` [perf] Reduce host overhead in DSA MLA attention (#12631); related: `7081f254cf` [perf] Add custom indexer k cache scatter op (#8960, the op's origin — replaced a PyTorch advanced-indexing scatter with the custom CUDA kernel this case then widens).
- **Applies when:** host-bound per step with a short GPU step — the Python attention/indexer path does per-step tensor reshaping/viewing/striding (byte reinterprets, `.view(uint8)`, `as_strided`, slot-map slicing) and/or `.item()` syncs to derive counts; nsys shows GPU idle, time in Python view/stride ops, a `cudaStreamSynchronize` from `.item()` on the critical path.
- **Mechanism:** (a) fuse bookkeeping into the consuming C++ op: `indexer_k_cache_scatter_op` now takes raw `k_fp8`/`k_scale` + full slot-mapping buffers + a `num_tokens` count and does the FP8/float32 byte-reinterpret and `[:num_tokens]` slice internally — deleting the Python view/as_strided/slice. (b) Remove a host-device sync: counts previously derived via `host_context_lengths.sum().item()` and a Python `_parse_request_types` (`host_request_types.sum().item()`) are deleted; counts are passed in precomputed.
- **Generalizes to:** "push per-step Python tensor wrangling and count-deriving syncs into the kernel/op that already runs"; carries to other backends' cache scatter/gather, RoPE/quant pre-steps, any op fed by Python reshapes; adapt by widening the op signature to accept raw tensors + a length and threading known counts through instead of `.item()`. Upstream sibling: **replace a PyTorch advanced-indexing scatter/gather** (`arange`+broadcast+`_unravel_indices`+masked assign) **with a stride-aware custom CUDA scatter** — one block per token, vectorized 4-byte stores — especially for writes into a *non-contiguous paged pool* (#8960 introduced exactly this op before #12631 pushed the remaining bookkeeping into it).
- **Apply via:** C++ custom-op signature change (`torch.ops.trtllm.indexer_k_cache_scatter_op(..., num_tokens)`); delegate the op to **kernel-cuda-specialist** and the `.item()` removal to **perf-torch-sync-free**.
- **Expected effect:** lower per-step host time / fewer Python ops + one fewer host-device sync → less GPU idle in DSA attention; direction only — measured Δ (host prep, GPU idle, step latency) to be recorded from run.
- **Accuracy risk:** lossless — same scatter result and counts; only where/when the reinterpret/counting happens changed. (Op `TORCH_CHECK`s element sizes, failing loudly on dtype mismatch.)
- **Verify:** nsys shows reduced Python view/stride time and removal of the `.item()` sync; step latency down; KV-scatter parity / DSA accuracy unchanged.
- **Rollback:** restore the Python byte-view/slice path and old op signature; re-add `_parse_request_types`. Trigger: cache-scatter mismatch or DSA accuracy drop.
- **Prior art:** PR #12631. Files: `cpp/.../thop/IndexerKCacheScatterOp.cpp`, `thop/attentionOp.cpp`, `_torch/attention/backends/sparse/dsa.py`, `attention/backends/trtllm_gen.py`. Detection: **perf-host-analysis**. Related: the [DSA pool-view-cache](cache-step-invariant-per-layer.md) and [torch.compile-closure](hoist-torch-compile-closures.md) cases (same DSA host-overhead theme).
