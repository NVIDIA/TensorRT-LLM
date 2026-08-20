---
id: case-overlap-online-eplb
type: case
family: runtime-execution
maturity: full
bottleneck: [communication, launch]
signals: [expert-load-imbalance, gpu-idle-between-steps]
architectures: [sm90, sm100]
model_scope: [model-agnostic, moe, deepseek-v3r1, qwen3-moe]
phase: [decode]
patterns: [pattern-overlap-independent-work]
accuracy_risk: lossless
apply_via_kind: [config-knob]
knobs: [num_slots, layer_updates_per_iter]
specialists: [trtllm-moe-develop, perf-sweep-workflow]
commits: ['ead89a0e408e', '9241ccaf2770']
eligibility:
  - "EP>1 with attention-DP (use_dp); online EPLB active (layer_updates_per_iter > 0)"
measured: []
---

# Overlap online-EPLB rebalance with MoE compute, and extend EPLB to more backends

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `ead89a0e408e` [perf] Improve the performance of online EPLB on Hopper by better overlapping (#6624); related: `9241ccaf2770` Enable EPLB for trtllm-gen and cutlass backend (#8886).
- **Applies when:** communication/sync-exposed MoE with expert load imbalance across EP ranks (hot experts overloading some ranks while others idle), online/dynamic EPLB enabled (`layer_updates_per_iter > 0`). Signals: MoE-heavy model, EP>1 + `use_dp`, per-rank expert-time skew, a per-iteration bubble where statistics gather / expert-weight reshuffle stalls compute.
- **Mechanism:** #6624 routes the load-balancer's per-iteration work — local statistic collection (`update_local_statistic`, `update_statistic_with_local_ids`/`...with_gathered_statistic`) and the GPU "wait stage" that swaps in rebalanced expert weights — onto a dedicated `aux_stream` (`AuxStreamType.MoeBalancer`) ordered against compute via `EventType.Main`/`EventType.MoeBalancer`, replacing the old `cudagraph_stream`+`statistic_stream` pair, so rebalance overhead hides behind MoE GEMMs. It also splits the statistic path (`start_wait_gpu_stage`/`done_wait_gpu_stage`) to overlap allgather of statistics with routing. #8886 lifts EPLB out of `fused_moe_wide_ep` into the shared `MoEInterface` base, so the **cutlass** and **trtllm-gen** MoE backends (and GPT-OSS) gain EPLB, not just wide-EP.
- **Generalizes to:** "move periodic background bookkeeping/weight-movement off the critical stream and overlap it with steady-state compute"; carries to KV-cache defrag/transfer, weight prefetch/streaming, any allgather of side metadata; adapt by isolating the background work on an aux stream with event ordering and only doing it when the dynamic path is active.
- **Apply via:** `MoeLoadBalancerConfig(num_slots=<divisible by ep_size>, layer_updates_per_iter=<N>)` (`llmapi/llm_args.py`); `layer_updates_per_iter=0` = static routing (no overlap needed), `>0` = online EPLB (this overlap engages). Requires `use_dp` (ADP) and EP>1. Delegate wiring to **trtllm-moe-develop**; sweep `num_slots`/`layer_updates_per_iter` with **perf-sweep-workflow**.
- **Expected effect:** higher MoE/decode throughput by hiding online-EPLB rebalance cost; for #8886, EPLB becomes available on cutlass/trtllm-gen backends; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless (EPLB relocates expert replicas/slots across ranks; routing math and outputs unchanged — slots map back to the same experts).
- **Verify:** decode throughput + per-rank MoE-time balance (straggler spread) with online EPLB on vs off; confirm MoeBalancer stream concurrent with MoE GEMMs in nsys. Sanity-check accuracy unchanged.
- **Rollback:** `layer_updates_per_iter=0` (static routing) or omit the `moe_load_balancer` config. Trigger: no measured imbalance to recover, overlap not materializing, or instability during weight swap.
- **Prior art:** PRs #6624, #8886. Files: `_torch/modules/fused_moe/moe_load_balancer.py` (`start_wait_gpu_stage`, `update_local_statistic`, `is_static_routing`), `fused_moe/interface.py` (`_init_load_balancer`), `_torch/utils.py` (`AuxStreamType.MoeBalancer`), `llmapi/llm_args.py` (`MoeLoadBalancerConfig`).
