# Source-Backed Tuning Notes

Read an exact or nearby checked-in config and the model's deployment guide **before** using these notes. These are not universal thresholds.

## Commonly Tuned Fields

| Field | Guidance |
|---|---|
| `max_batch_size` | Affects throughput. Reachable batch size depends on total sequence length and GPU memory. MoE models generally cap lower than dense. |
| `max_num_tokens` | Affects throughput and memory. Must exceed ISL plus chat template overhead; sweet spot is ISL to 2× ISL. Tune together with `max_batch_size`. |
| `enable_attention_dp` | Scenario-dependent; primarily benefits high-throughput / high-concurrency traffic. At low concurrency the memory overhead can be prohibitive with little throughput gain. Follow the exact model guide/config. |
| `kv_cache_config.free_gpu_memory_fraction` | OOM lever. Safe range depends on attention architecture (MLA vs GQA) and whether ADP is enabled. Guides often adjust `max_batch_size` or `max_seq_len` first. |
| `moe_expert_parallel_size` | MoE only. Copy from checked-in source; do not assume it equals TP. |
| `moe_config.backend` | Backend choice depends on quantization, GPU generation, and concurrency level. Set only when model guide or checked-in config specifies it. |
| `stream_interval` | Model-specific variation; no single global value. |
| `num_postprocess_workers` | Present in some streaming/higher-concurrency configs. |
| `attention_dp_config.*` | Preserve when present in source configs. |
| `cuda_graph_config.max_batch_size` / `batch_sizes` | Align with source config. Not necessarily equal to server `max_batch_size`. |
| MTP fields (`num_nextn_predict_layers`, `use_relaxed_acceptance_for_thinking`, `relaxed_topk`, `relaxed_delta`) | DeepSeek-R1 MTP only. Copy verbatim from the selected checked-in config; never interpolate. |

## Generally Observed Patterns

These patterns are generally observed across benchmark data. A checked-in config or deployment guide always takes precedence.

- **`enable_attention_dp`** — A high-throughput knob. Generally turned on for throughput-oriented workloads at higher concurrency. MoE+GQA models tend to benefit at lower concurrency thresholds than MoE+MLA or Dense+GQA. Memory overhead is architecture-dependent: small for MLA (compressed attention), substantial for GQA (full attention replication). Can trigger OOM when combined with aggressive KV cache fraction.

- **`moe_config.backend`** — Highest-impact MoE knob. Backend choice depends on quantization and GPU generation: CUTLASS and DEEPGEMM tend to dominate at higher concurrency; TRTLLM is generally safer at low concurrency. FP8 on Hopper uses CUTLASS (DEEPGEMM unavailable on SM90). Wrong backend at high concurrency can leave significant throughput on the table.

- **`kv_cache_config.free_gpu_memory_fraction`** — Safe range depends on attention architecture. MLA models tolerate higher fractions (compressed KV footprint). GQA models need more headroom (larger per-token KV). Lower the fraction when ADP is enabled to account for replicated attention weight overhead. Large MoE models with ADP may need notably conservative fractions to avoid MoE GEMM autotuner workspace failures.

- **`max_num_tokens`** — Must exceed ISL plus chat template overhead or requests get rejected. Sweet spot is generally ISL to 2× ISL. Setting it far above ISL wastes activation buffer memory and can counterintuitively shrink KV cache capacity. Higher values help scheduler packing at high concurrency but with diminishing returns.

- **`max_batch_size`** — MoE models generally cap lower than dense models due to per-expert memory scaling. Reachable batch size is bounded by KV cache capacity, which depends on sequence length, GPU memory, and ADP state.

## Bench-Derived Hints (fallback only)

These come from `tensorrt_llm/bench/` and are benchmark heuristics, not serve rules. Mark any bench-derived value as unverified.

- `benchmark/throughput.py` and `build/build.py` seed target ISL/OSL from `DatasetMetadata.avg_isl`/`avg_osl` — use as a fallback when no exact scenario match exists.
- `build/tuning.py` derives `max_batch_size` and `max_num_tokens` together from ISL/OSL and estimated KV-cache capacity. Rounds upward into coarse buckets; enforces min `max_num_tokens` of 2048. Outputs are "slightly optimistic" upper bounds for benchmark engine builds.
- `benchmark/utils/general.py` raises `max_num_tokens` to at least `max_isl + max_batch_size` on the non-chunked-prefill path. Benchmark safeguard, not a serve invariant.
- `dataclasses/configuration.py` fills `cuda_graph_config.max_batch_size` from runtime `max_batch_size` when both `batch_sizes` and `max_batch_size` are unset. Skip if the source config already sets graph batch sizes.
- If a bench-derived value diverges from nearby checked-in serve configs, prefer the serve config.

## OOM Triage

Follow the exact model guide/config first. Common levers: lower `max_batch_size`, `max_num_tokens`, `max_seq_len`, or `free_gpu_memory_fraction`. If the deployment guide provides a backend support matrix, use it for `moe_config.backend`.
