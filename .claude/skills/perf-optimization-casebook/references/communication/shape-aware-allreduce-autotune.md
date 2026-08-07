---
id: case-shape-aware-allreduce-autotune
type: case
family: communication
maturity: full
bottleneck: [communication]
signals: [allreduce-dominates, high-concurrency]
architectures: [sm80, sm90, sm100]
model_scope: [model-agnostic, dense, moe, deepseek-v3r1]
phase: [any-phase]
patterns: [pattern-autotune-collective-per-shape]
accuracy_risk: lossless
apply_via_kind: [default-on, config-knob, env-var]
knobs: [allreduce_strategy, TLLM_DISABLE_ALLREDUCE_AUTOTUNE, OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY]
specialists: [perf-sweep-workflow]
commits: ['22257457824a', 'd272f1a9bcbc']
measured: []
---

# Shape-aware strategy selection / autotuning of collectives (pick AllReduce algo by message size & concurrency)

> Part of the [Communication casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `22257457824a` [TRTLLM-8129][feat] Allreduce tuning and benchmark script revising (#7870); related: `d272f1a9bcbc` [TRTLLM-8821][feat] Apply AutoTuner to AllReduce Op for strategy tuning (#8531).
- **Applies when:** communication-bound + signals: TP AllReduce where no single algorithm wins across shapes — a one-shot kernel regresses vs NCCL at some sizes (A100/H100) while one/two-shot win elsewhere; throughput varies strongly with message size (seq_len×hidden) and concurrency; `allreduce_strategy` currently pinned.
- **Mechanism:** make the algorithm a function of shape. #7870 adds two `AUTO` heuristics in `customAllReduceUtils.h`: an LP threshold (`SelectStrategyLP`: per-`(SM_major, TP_size)` `(NCCL_num_token_threshold, TWO_SHOT_numel_threshold)` → TWOSHOT if `message_size ≥ two_shot_numel_threshold`, else ONESHOT, else NCCL) and a benchmark-derived LUT (`selectStrategyLookUpTable`, indexed `[sm][tp][fusionOp][log2(hidden)-7][log2(tokens)]`) generated offline by `allreduce_heuristic_code_gen.py`. Also fixes that ONESHOT/TWOSHOT couldn't be overridden, fixes a TWOSHOT perf bug, cleans dispatch. #8531 wires it into the runtime AutoTuner via `AllReduceRunner(TunableRunner)` + `trtllm::tunable_allreduce`, with `AutoTuner.choose_one` picking per-profile.
- **Generalizes to:** "autotune/heuristically pick the collective implementation per shape & concurrency"; carries to allgather/reduce-scatter, GEMM tactic selection, attention backend choice; adapt by benchmarking candidates offline into a LUT or registering a `TunableRunner` for the AutoTuner.
- **Apply via:** `allreduce_strategy=AUTO` (default) engages heuristic/LUT; AutoTuner runtime tuning runs under `AUTO` unless `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1`; `OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY` forces a strategy for benchmarking. Regenerate the LUT with `allreduce_heuristic_code_gen.py`. Delegate sweeping to **perf-sweep-workflow**.
- **Expected effect:** recovers per-shape best AllReduce time, removes the one-shot-vs-NCCL regression. **PR #7870 reports ~3–4% end-to-end gain for DeepSeek-R1 at concurrency 256 and 512** (fixes a known one-shot regression). Re-measure on your shapes.
- **Accuracy risk:** lossless (selects among bit-equivalent reduction implementations; precision unchanged).
- **Verify:** AllReduce time across the shape/concurrency grid (`all_reduce.py` microbench + `allreduce_perf_viz.py`); end-to-end throughput at concurrency 256/512; confirm no regression vs NCCL at small sizes.
- **Rollback:** pin `allreduce_strategy` to a fixed value (e.g. `NCCL`), or `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1` to keep AUTO heuristics without runtime autotuning. Trigger: tuner-induced regression or instability.
- **Prior art:** PRs #7870, #8531. Files: `cpp/.../common/customAllReduceUtils.h` (`SelectStrategyLP`, `selectStrategyLookUpTable`), `thop/allreduceOp.cpp`, `_torch/custom_ops/torch_custom_ops.py` (`AllReduceRunner`), `tests/scripts/allreduce_perf/allreduce_heuristic_code_gen.py`, `tests/microbenchmarks/all_reduce.py`. Sweep with **perf-sweep-workflow**.
