---
name: serve-config-guide
description: Generate a source-backed starting `trtllm-serve --config` YAML for
  basic aggregate single-node PyTorch serving, aligned with checked-in TensorRT-LLM
  configs and deployment docs. Preserves explicit latency / balanced / throughput
  objectives. Excludes disaggregated, multi-node, and non-MTP speculative configs.
---

# Serve Config Guide

**Scope:** aggregate/IFB (in-flight batching) colocated prefill+decode, single node, PyTorch backend, non-speculative by default; DeepSeek-R1 MTP is the standard mode (all checked-in configs include it).

**Input:** model, GPU, ISL (input sequence length), OSL (output sequence length), concurrency, TP, performance objective (`Min Latency` | `Balanced` | `Max Throughput` | unspecified).
**Output:** repo-grounded starting YAML for `trtllm-serve --config`.

If the request is adjacent but out of scope, provide a best-effort answer using the nearest in-scope config as a starting point, clearly label inferred vs. verified fields, and point to the relevant feature doc in `docs/source/features/` (e.g., speculative-decoding, disagg-serving, parallel-strategy) or `examples/llm-api/`.

## Constraints

1. **Speculative exclusion:** Exclude configs containing `speculative_config` by default. Exception: exact checked-in DeepSeek-R1 MTP configs (models with `decoding_type: MTP` in `examples/configs/`). When including MTP, copy the full `speculative_config` block verbatim — never interpolate speculative fields.

2. **Objective preservation:** Preserve the user's stated objective through config selection. Use `database.py` profile labels (`Min Latency`, `Balanced`, `Max Throughput`; plus `Low Latency`/`High Throughput` in smaller sets) as selection aids. If a config is unlabeled, treat it as a default starting point — do not claim it matches a specific objective. If the only match conflicts with the stated objective, call out the mismatch.

3. **Source preference:** Prefer checked-in configs over interpolation. When docs and configs disagree, prefer the config for the exact scenario and note the mismatch. Mark any interpolation as unverified.

## Response Format

For **exact matches**: `Config` → `Source` → `Launch command`

For **interpolated configs**: `Config` → `Source used as starting point` → `What to benchmark` (single list of knobs worth sweeping, not per-field unverified tags)

## Step 0: Lock Objective and Decode Mode

Identify the user's objective (`Min Latency` | `Balanced` | `Max Throughput` | unspecified) and decode mode (non-speculative or DeepSeek-R1 MTP per **Constraint 1**). Preserve both through the remaining steps.

## Step 1: Exact Database Match

Search `examples/configs/database/lookup.yaml` for an exact `(model, gpu, isl, osl, concurrency, num_gpus)` match. Use `database.py` as a loader/helper.

- Apply **speculative exclusion**.
- When multiple recipes exist at different concurrency points, use profile labels to match the user's objective per **objective preservation**.
- Prefer an exact match that also matches the stated objective over manual tuning.

## Step 2: Nearest Checked-In Config

If no exact match, widen the search to also include `examples/configs/curated/lookup.yaml`.

Apply the same constraints as Step 1. Additionally:
- A partial match from `database/` is preferred over a partial match from `curated/` for the same model (database configs are benchmark-tuned).
- Exclude disaggregated-only or prefill-only entries (e.g., `qwen3-disagg-prefill.yaml`).
- For curated configs, only treat intent as explicit when the repo labels it (e.g., `*-latency.yaml`, `*-throughput.yaml`, or guide text).
- If no in-scope config matches the stated objective, pick the nearest same-model starting point and call out the mismatch.

## Step 3: Read Model Docs

Search `docs/source/deployment-guide/` and `examples/models/core/` for the model's deployment guide and README. Read both before adjusting knobs.

**Excluded sources:** Do NOT use `docs/source/legacy/` tuning values or benchmark numbers — those were measured on the TensorRT engine-building backend and do not transfer to PyTorch backend serving.

**DeepSeek-V3 caveat:** For DeepSeek-V3/V3.2-Exp, use `examples/models/core/deepseek_v3/README.md`, not the R1 deployment guide.

## Step 4: Adjust Source-Backed Fields

Commonly scenario-dependent fields (adjust only these, guided by the checked-in source):

`max_batch_size`, `max_num_tokens`, `max_seq_len`, `enable_attention_dp`, `attention_dp_config.*`, `kv_cache_config.free_gpu_memory_fraction`, `moe_expert_parallel_size` (MoE), `moe_config.backend` (when guide specifies), `stream_interval`, `num_postprocess_workers`, `cuda_graph_config.max_batch_size`/`batch_sizes`, and MTP-specific fields when using DeepSeek-R1 MTP configs.

Do not assume other fields are constant across models/GPUs. For tuning notes, read `references/knob-heuristics.md`.

## Validation Checklist

- [ ] `trust_remote_code: true` called out as trust boundary when present
- [ ] `max_num_tokens` >= ISL + chat template overhead (requests rejected if violated)
- [ ] If interpolated: single "What to benchmark" section listing knobs to sweep, not per-field unverified tags
