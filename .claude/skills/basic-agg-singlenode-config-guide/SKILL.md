---
name: basic-agg-singlenode-config-guide
description: Use when a human wants a source-backed starting `trtllm-serve --config`
  for basic aggregate single-node PyTorch serving and wants to stay aligned with
  current checked-in TensorRT-LLM configs and deployment docs while preserving
  any explicit latency / balanced / throughput objective. Avoid disaggregated,
  multi-node, architecture-guessing, or non-MTP speculative extrapolation;
  exact-source-backed MTP recipes are allowed.
---

# Basic Aggregate Single-Node Config Guide

**Scope - basic aggregate serving only:**
- Aggregate / inflight-batched (IFB) with colocated prefill + decode
- Single node only
- PyTorch backend only
- Non-speculative decoding by default; exact-source-backed MTP allowed
- Use this skill conservatively: prefer exact checked-in configs and model-specific docs over interpolation

Input: model name, GPU type, ISL, OSL, target concurrency, TP, and performance objective (`Min Latency`, `Balanced`, `Max Throughput`, or unspecified).
Output: a repo-grounded starting YAML for `trtllm-serve --config`.

## Step 0: Lock the Performance Objective and Decode Mode

Before choosing a config, identify whether the user wants `Min Latency`, `Balanced`, `Max Throughput`, or has left the objective unspecified. Also identify whether the requested or checked-in starting point is non-speculative, `MTP`, or unspecified.

- Preserve that objective through the rest of the search when current checked-in sources label it explicitly.
- Do not silently substitute a throughput-oriented config for a latency-oriented request, or vice versa.
- Preserve decode mode when current checked-in sources make it explicit.
- Default to non-speculative scope unless the exact checked-in serve config you are following is explicitly `speculative_config.decoding_type: MTP`.
- Do not broaden an `MTP` recommendation into generic speculative decoding support.
- `examples/configs/database/database.py` contains helper logic that labels database recipes by relative profile (`Min Latency`, `Balanced`, `Max Throughput`, plus `Low Latency` / `High Throughput` in some smaller sets). Use that as a database-selection aid, not as a universal SLA classifier.
- If the current exact-model source only provides one unlabeled config, treat it as a recommended default / working starting point rather than claiming it is latency- or throughput-optimized.

## Step 1: Prefer an Exact Database Match

Search `examples/configs/database/lookup.yaml` for an exact **in-scope** `(model, gpu, isl, osl, concurrency, num_gpus)` match.

- Use `examples/configs/database/database.py` as a loader/helper for `lookup.yaml` if programmatic inspection is easier.
- If an exact database match exists, inspect the YAML before using it.
- When the current database contains multiple recipes for the same `(model, gpu, isl, osl, num_gpus)` at different concurrency points, use the helper profile labels in `database.py` to understand whether the candidate sits nearer `Min Latency`, `Balanced`, or `Max Throughput`.
- Exclude database configs containing `speculative_config` by default.
- Exception: exact checked-in serve configs for `deepseek-ai/DeepSeek-R1-0528` or `nvidia/DeepSeek-R1-0528-FP4-v2` that explicitly set `speculative_config.decoding_type: MTP` are in scope for this skill.
- If you follow that exception, copy the `speculative_config` block from the checked-in YAML you selected instead of synthesizing speculative fields.
- Do not admit other speculative modes, and do not infer MTP eligibility from model family similarity alone.
- Treat `cache_transceiver_config` as requiring exact-model verification. It often signals disaggregated serving, but if an exact-model deployment guide explicitly points you to a config that contains it, do not exclude that config solely for that reason; call out the exception instead of silently treating it as a generic basic-agg default.
- If the only exact database match conflicts with the user's stated objective, call out the mismatch explicitly instead of presenting it as a clean recommendation.
- Prefer an exact in-scope database match that also matches the user's stated objective over manual tuning.

## Step 2: Use the Closest Checked-In Config

If no exact match exists, look for the nearest checked-in config in:

- `examples/configs/database/lookup.yaml`
- `examples/configs/curated/lookup.yaml`

Stay in scope explicitly for both database and curated configs:

- Preserve performance objective before minimizing distance on other axes.
- Preserve decode mode before minimizing distance on other axes.
- Exclude entries that are disaggregated or prefill-only.
- Exclude configs containing `speculative_config` by default.
- Exception: checked-in same-model DeepSeek-R1 serve configs for `deepseek-ai/DeepSeek-R1-0528` or `nvidia/DeepSeek-R1-0528-FP4-v2` whose YAML explicitly sets `speculative_config.decoding_type: MTP` remain in scope.
- Treat `cache_transceiver_config` as requiring exact-model verification, not as an automatic exclusion. If the exact-model guide explicitly uses it, keep the config and call out the exception.
- For curated configs, only treat intent as explicit when the current repo labels it explicitly, for example `*-latency.yaml`, `*-throughput.yaml`, or guide text that names the use case.
- If an exact-model source provides only one unlabeled config, use it as a default starting point only and say that objective fit is unverified.
- If the nearby checked-in config is MTP-backed, keep the decode-mode match explicit and do not swap it for a non-speculative config unless the current exact-model sources give you a better in-scope non-spec alternative.
- If no in-scope checked-in config matches the user's stated objective, choose the nearest same-model starting point and call out the profile mismatch or uncertainty explicitly.
- Treat curated configs as starting points, not universal templates.

## Step 3: Read the Model's Current Docs

Before changing knobs, read the model-specific deployment guide and model README when available.

- DeepSeek-R1: deployment guide plus nearby curated/database configs. Current checked-in exact-model serve sources include MTP-backed configs; the deployment guide explicitly points to `examples/configs/curated/deepseek-r1-throughput.yaml` and `examples/configs/curated/deepseek-r1-deepgemm.yaml`. Treat `examples/configs/curated/deepseek-r1-latency.yaml` as source-backed but not deployment-guide-explicit.
- DeepSeek-V3 / DeepSeek-V3.2-Exp: `examples/models/core/deepseek_v3/README.md` plus nearby checked-in configs
- GPT-OSS: deployment guide plus nearby curated/database configs
- Qwen3: exact-model deployment guide plus curated configs
- Qwen3-Next: exact-model deployment guide plus curated configs
- Llama4 Scout: exact-model deployment guide plus nearby curated config
- Kimi-K2: exact-model deployment guide plus curated config; note that the current checked-in Kimi config is an explicit `cache_transceiver_config` exception, not a generic basic-agg template. The current curated lookup maps Kimi-K2 to `DeepseekV3ForCausalLM`, so nearby DeepSeek-V3 / R1 configs are only secondary comparators for unverified interpolation, not substitutes for exact Kimi sources.

If docs and configs disagree, prefer the current checked-in config for the exact scenario you are following and call out the mismatch.

## Step 4: Adjust Only Source-Backed Fields

Current checked-in docs/configs show these fields are commonly scenario-dependent:

- `max_batch_size`
- `max_num_tokens`
- `enable_attention_dp`
- `kv_cache_config.free_gpu_memory_fraction`
- `moe_expert_parallel_size` for MoE models
- `moe_config.backend` when the model-specific guide or support matrix calls it out
- `stream_interval` and `num_postprocess_workers` in some streaming / higher-concurrency cases
- `cuda_graph_config.max_batch_size` or `batch_sizes`
- `speculative_config.num_nextn_predict_layers` when following the exact-source-backed DeepSeek-R1 MTP exception
- `speculative_config.use_relaxed_acceptance_for_thinking`, `relaxed_topk`, and `relaxed_delta` when present in the exact checked-in config you are following

Do not assume the rest are constants across models or GPUs.

Do not infer hard thresholds unless the current checked-in guide for that model states them.

If you are following this skill's exact-source-backed MTP exception, keep `speculative_config.decoding_type: MTP` and any sibling MTP fields copied from the checked-in config you selected. Do not invent or interpolate speculative fields.

For conservative tuning notes, including a small `trtllm-bench`-derived token-budget and batching hint layer, read `references/knob-heuristics.md`.


## Validation Checklist

Before deploying, verify:

- [ ] The config is still in the basic-agg, single-node scope, with non-speculative decoding by default or an exact-source-backed `MTP` exception called out explicitly.
- [ ] The selected config's labeled intent matches the user's stated objective when current repo sources label that intent, or the mismatch / uncertainty has been called out explicitly.
- [ ] If `speculative_config` is present, it is only because the selected checked-in config explicitly uses `speculative_config.decoding_type: MTP` for `deepseek-ai/DeepSeek-R1-0528` or `nvidia/DeepSeek-R1-0528-FP4-v2`; otherwise `speculative_config` is absent.
- [ ] Any `speculative_config` block was copied from the exact checked-in config you selected rather than interpolated from nearby configs.
- [ ] Disaggregated-only sections/settings are absent for this use case, or any exact-model-guide exception (for example a guide-recommended config that contains `cache_transceiver_config`) has been called out explicitly.
- [ ] For models with multiple curated or database variants, latency / balanced / throughput alternatives were checked before choosing a nearby config.
- [ ] `cuda_graph_config.max_batch_size` has been checked against the exact checked-in config/docs for this model and scenario; do not assume it always equals the server `max_batch_size`.
- [ ] For MoE models, `moe_expert_parallel_size` still matches the checked-in model docs/config you are following.
- [ ] If OOM occurs, follow the exact model guide/config first. Common levers in current docs/configs include lowering `max_batch_size`, `max_num_tokens`, `max_seq_len`, or `kv_cache_config.free_gpu_memory_fraction`.
- [ ] Any `moe_config.backend` choice is justified by a current model-specific guide or an existing checked-in config.
- [ ] Any remaining interpolation is called out as unverified.

## Repo Resources

| Resource | Path |
|---|---|
| Exact scenario database | `examples/configs/database/lookup.yaml` |
| `lookup.yaml` loader/helper | `examples/configs/database/database.py` |
| Curated starting points | `examples/configs/curated/lookup.yaml` |
| Config directory overview | `examples/configs/README.md` |
| Model deployment guides | `docs/source/deployment-guide/` |
| Model READMEs | `examples/models/core/` |
