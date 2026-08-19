---
id: case-deepep
type: case
family: communication
maturity: full
bottleneck: [communication]
signals: [alltoall-dominates, slow-path-fallback]
architectures: [sm90, sm100]
model_scope: [moe, deepseek-v3r1, qwen3-moe]
phase: [any-phase]
patterns: [pattern-swap-generic-collective-for-specialized-comm]
accuracy_risk: lossless
apply_via_kind: [env-var]
knobs: [TRTLLM_CAN_USE_DEEP_EP]
specialists: [perf-sweep-workflow, trtllm-moe-develop]
commits: ['0b60da2c45ac']
eligibility:
  - "deep_ep_installed and dtype==bfloat16 (path auto-resolve gate, fused_moe_cutlass.py)"
interactions:
  - {feature: cuda-graphs, relation: incompatible-with, note: "the DeepEP method only; auto-resolves to DeepEPLowLatency under CUDA Graphs"}
measured: []
---

# Use a dedicated expert-parallel all-to-all comm backend (DeepEP) for MoE dispatch/combine

> Part of the [Communication casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `0b60da2c45ac` feat: large-scale EP (part 7: DeepEP integration) (#4792).
- **Applies when:** communication-bound + signals: MoE model with expert parallelism (large-scale EP); token routing dispatch/combine all-to-all is a large share of step time; intranode NVLink and/or internode RDMA available; default `allgather`/`reducescatter` MoE path in use. Decode/low-latency with CUDA Graphs needs the LowLatency variant.
- **Mechanism:** replaces generic allgather/reducescatter token exchange with DeepEP's purpose-built EP all-to-all kernels (patched NVSHMEM, `NVSHMEM_IBGDA_SUPPORT=1`) routing only each rank's selected tokens to their expert owners — cutting bytes and overlapping NVL/RDMA transfer with layout compute. Two methods via `AlltoallMethodType` (`fused_moe_cutlass.py`): `DeepEP` (intranode/internode, no CUDA-Graph, IBGDA for internode; currently faster) and `DeepEPLowLatency` (CUDA-Graph-compatible, IBGDA) — backed by `VariableLengthBuffer`/`VariableLengthLowLatencyBuffer` (`deep_ep_utils.py`).
- **Generalizes to:** "swap a generic collective for a workload-specialized comm library"; carries to MoE EP combine, attention-DP all-to-all, PP send/recv; adapt by matching the backend's buffer-reservation + layout-adapter contract and picking the graph-safe variant for decode.
- **Apply via:** env `TRTLLM_CAN_USE_DEEP_EP=1`; auto-resolves to `DeepEPLowLatency` under CUDA Graphs else `DeepEP` (only when `deep_ep_installed and dtype==bfloat16`); module `enable_alltoall` gates the path; requires the DeepEP/NVSHMEM build (`docker/common/install_deep_ep.sh`). Delegate config exploration to **perf-sweep-workflow**.
- **Expected effect:** lower MoE dispatch/combine latency + higher EP throughput at scale; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless (bf16 token exchange; path asserts `dtype==bfloat16` — changes transport, not math).
- **Verify:** end-to-end throughput/latency vs allgather/reducescatter baseline; correctness with `test_fused_moe.py`; sanity accuracy on a small eval.
- **Rollback:** unset `TRTLLM_CAN_USE_DEEP_EP` (falls back to allgather/reducescatter). Trigger: DeepEP build/IBGDA unavailable, regression vs baseline, or CUDA-Graph capture failure on the non-LL method.
- **Prior art:** PR #4792. Files: `_torch/modules/fused_moe/deep_ep_utils.py`, `fused_moe_cutlass.py` (`AlltoallMethodType`), `modeling_deepseekv3.py`, `docker/common/install_deep_ep.sh`. Owning skill: **trtllm-moe-develop**; sweep with **perf-sweep-workflow**.
