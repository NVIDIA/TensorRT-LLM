---
name: basic-agg-singlenode-config-guide
description: Lightweight guide for manually tuning basic aggregate (inflight batching,
  single node, PyTorch backend, non-speculative) TRT-LLM serving configs. Covers the
  4 critical knobs (moe_backend, attention_dp, kv_fraction, max_num_tokens),
  architecture-class heuristics, and pointers to existing config databases. Use when
  a human is manually tuning a TRT-LLM config for standard throughput-optimized
  serving via trtllm-serve. NOT for multi-node, disaggregated serving, speculative
  decoding, or TensorRT engine configs.
---

# Basic Aggregate Single-Node Config Guide

**Scope — basic aggregate serving only:**
- Aggregate / inflight-batched (IFB) — prefill and decode colocated on same GPUs in same scheduler batch (not disaggregated prefill/decode)
- Single node (not multi-node tensor parallelism)
- PyTorch backend (not TensorRT engine build)
- Non-speculative decoding (no MTP, Eagle, Medusa, etc.)

Input: model name, GPU type, ISL, OSL, target concurrency, TP.
Output: a tuned `extra_llm_api_options` YAML config for `trtllm-serve`.

## Step 1: Find a Starting Config

Check existing configs before tuning from scratch:

1. **Database** (`examples/configs/database/lookup.yaml`) — 169 benchmark-optimal configs for DeepSeek-R1 (FP4/FP8), gpt-oss-120b on B200/H200. Search for exact (model, GPU, ISL, OSL, concurrency, TP) match. Use `database.py`'s `RecipeList.from_yaml()` for programmatic lookup.
2. **Curated** (`examples/configs/curated/`) — 10 production configs for 6+ model families. Good starting points even if scenario doesn't match exactly. Note: some curated configs include speculative decoding — strip `speculative_config` to stay in basic-agg scope.
3. **No match** — classify the model (read `references/architecture.md`), then pick the matching template from `references/templates.md`.

## Step 2: Read the Closest Database Config

If a database config exists for a nearby scenario (same model, different concurrency), read it. Only 4 knobs change across scenarios for a given model — everything else is constant.

## Step 3: Tune the 4 Critical Knobs

These are the only parameters that materially affect basic aggregate single-node throughput. Read `references/knob-heuristics.md` for decision tables and warning signs for each knob.

| Knob | Applies to | One-line summary |
|---|---|---|
| `moe_config.backend` | MoE only | TRTLLM at low concurrency, CUTLASS/DEEPGEMM at high (GPU+quant dependent) |
| `enable_attention_dp` | All | Trades memory for attention throughput; arch-dependent thresholds |
| `kv_cache_config.free_gpu_memory_fraction` | All | OOM knob; lower when ADP is on or GQA model |
| `max_num_tokens` | All | Must be >= ISL+64; sweet spot ISL to 2x ISL |

Everything else is a constant — see the constants table in `references/architecture.md`.

## Validation Checklist

Before deploying, verify:

- [ ] `max_seq_len >= ISL + OSL` (add +68 safety margin)
- [ ] `max_num_tokens >= ISL + 64` (if chunked prefill is disabled)
- [ ] `moe_expert_parallel_size == tensor_parallel_size` for MoE models
- [ ] ADP on + `free_gpu_memory_fraction >= 0.85` on 80 GiB GPU → likely OOM
- [ ] ADP on + GQA model → verify 15-20 GiB headroom per GPU
- [ ] `moe_config.backend: DEEPGEMM` only on Blackwell (not Hopper)
- [ ] `cuda_graph_config.max_batch_size` not >> concurrency
- [ ] No `speculative_config` present (out of basic-agg scope)

## Repo Resources

| Resource | Path |
|---|---|
| Benchmark database (169 configs) | `examples/configs/database/` |
| Scenario lookup | `examples/configs/database/lookup.yaml` |
| Programmatic API | `examples/configs/database/database.py` |
| Curated configs (10 configs) | `examples/configs/curated/` |
| Config README | `examples/configs/README.md` |
