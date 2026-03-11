---
name: basic-agg-singlenode-config-guide
disable-model-invocation: true
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
Output: a repo-grounded starting YAML for `trtllm-serve --config` when the request is in scope.

If the request is adjacent but outside this skill's scope, do **not** stop at a hard refusal. Instead:
- state the boundary briefly,
- provide the best grounded config guess or config sketch you can assemble from nearby checked-in sources,
- avoid presenting any out-of-scope config as a fully verified recommendation,
- and split the answer cleanly into verified source-backed pieces versus inferred or assembled pieces.

## Response Format

For **in-scope** requests, structure the answer as:
- `Recommended config`
- `Why this source`
- `What is verified`
- `What is unverified`

For **adjacent but out-of-scope** requests, structure the answer as:
- `Scope note`
- `Best grounded config guess`
- `Verified source-backed pieces`
- `Inferred / assembled pieces`
- `Main risks / assumptions`
- `Supporting checked-in sources`

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
- Exclude database configs containing `speculative_config` by default **when selecting a config for this skill**.
- Exception: exact checked-in serve configs for `deepseek-ai/DeepSeek-R1-0528` or `nvidia/DeepSeek-R1-0528-FP4-v2` that explicitly set `speculative_config.decoding_type: MTP` are in scope for this skill.
- If you follow that exception, copy the `speculative_config` block from the checked-in YAML you selected instead of synthesizing speculative fields.
- Do not admit other speculative modes when selecting a config for this skill, and do not infer MTP eligibility from model family similarity alone.
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
- Exclude entries that are disaggregated or prefill-only when selecting a config for this skill.
- Exclude configs containing `speculative_config` by default when selecting a config for this skill.
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
- Kimi-K2: exact-model deployment guide plus curated config; note that the current checked-in Kimi config is an explicit `cache_transceiver_config` exception, not a generic basic-agg template. It also enables `trust_remote_code: true`; call that out explicitly as an exact-model requirement and a user trust boundary, not just as incidental YAML content. The current curated lookup maps Kimi-K2 to `DeepseekV3ForCausalLM`, so nearby DeepSeek-V3 / R1 configs are only secondary comparators for unverified interpolation, not substitutes for exact Kimi sources.

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

## Out-of-Scope Handling

When the request is outside this skill's strict scope, do **not** dead-end. If there are relevant checked-in sources nearby, assemble the best grounded config guess or config sketch you can. Do **not** present it as fully verified config guidance; instead, make the verified versus inferred boundary explicit.

Use this policy:
- State the specific boundary in one or two sentences.
- If the request is adjacent, identify the nearest checked-in baseline config, algorithm-specific config fragment, and topology guidance that can anchor a best guess.
- Start from the most relevant same-model checked-in serving baseline when possible, then layer in the nearest checked-in fragment for the adjacent feature.
- Keep the separation explicit between:
  - what this skill can verify directly from checked-in configs or docs,
  - what has been inferred or assembled from multiple sources,
  - and what risks or assumptions remain.
- Prefer a concrete best guess over a redirect-only answer, but keep the answer honest about what is assembled.

Use these nearby sources conservatively:
- Non-DeepSeek speculative decoding: use `docs/source/features/speculative-decoding.md` for supported algorithms, YAML structure, and backend caveats. For generic non-DeepSeek speculative assembly on the PyTorch backend, explicitly say that the current checked-in doc notes serve-time support only for `Eagle3`. Do not let that generic caveat override this skill's separate exact-source-backed DeepSeek-R1 `MTP` exception.
- Multi-node or broader topology questions: use `docs/source/features/parallel-strategy.md` for TP, PP, DP, EP, CP, and related topology fields.
- Disaggregated serving: use `docs/source/features/disagg-serving.md` and `examples/disaggregated/README.md` only when the assembled answer actually crosses into disaggregated serving.
- Multi-node launch skeletons: use `examples/llm-api/llm_mgmn_trtllm_serve.sh` only as an operational template after you have already anchored the config guess in model-specific and topology-specific sources. Call out that it is a launch-oriented Slurm example with hard-coded values, not a verified starting config recommendation.

For assembled answers, use this construction order:
1. Exact checked-in same-model serving config or deployment-guide baseline
2. Same-model nearby checked-in config for the closest supported mode
3. Feature-specific fragment from the relevant checked-in doc or example
4. Topology fields from the nearest checked-in topology guide or example
5. Explicit labeling of every assembled or inferred field

Do **not** turn the adjacent-help path into a generic repo-assembly bot. The skill should still avoid:
- presenting an out-of-scope config as fully verified,
- inventing unsupported YAML from unrelated docs,
- copying values across unrelated model families without saying so,
- or blurring the line between exact checked-in config content and your own assembled guess.

For the common failure mode that prompted this change:
- If the user asks for speculative decoding on a non-DeepSeek model, especially on multiple nodes, say that this is outside this skill's config-selection scope.
- Then give the best grounded config guess you can by:
  - starting from the exact model's checked-in serving baseline,
  - adding the speculative block shape from `docs/source/features/speculative-decoding.md`,
  - explicitly surfacing the current generic non-DeepSeek PyTorch caveat that the checked-in doc supports only `Eagle3` for serve-time speculative assembly,
  - and using `docs/source/features/parallel-strategy.md` for the topology side.
- For PyTorch `trtllm-serve`, only assemble speculative best-guess configs for `Eagle3`. Do **not** assemble `NGram`, `DraftTarget`, `PARD`, or `SA` serve YAML under this skill, because the current checked-in speculative-decoding doc says PyTorch serve-time support is `Eagle3` only.
- If the assembled answer crosses into disaggregated serving, say so explicitly and use `docs/source/features/disagg-serving.md` as the source for those fields.
- Make it explicit that you are providing a best grounded guess, not certifying a fully verified starting config under this skill.


## Validation Checklist

For **in-scope** answers, verify:

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
- [ ] If the selected exact-model config enables `trust_remote_code: true`, the answer called that out explicitly as a trust boundary rather than silently inheriting it.
- [ ] Any remaining interpolation is called out as unverified.

For **adjacent out-of-scope assembled answers**, verify:

- [ ] If the request was adjacent but out of scope, the answer still provided a concrete best grounded config guess or config sketch rather than stopping at a redirect-only answer.
- [ ] The answer separated verified source-backed pieces from inferred or assembled pieces clearly.
- [ ] If `docs/source/features/speculative-decoding.md` informed a non-DeepSeek speculative assembled answer, the answer surfaced the current generic PyTorch `Eagle3` caveat rather than implying broader serve-time speculative support.
- [ ] That generic `Eagle3` caveat did not override or contradict this skill's exact-source-backed DeepSeek-R1 `MTP` exception.
- [ ] If the request asked for speculative decoding on the PyTorch backend, any assembled serve-time speculative config used only `Eagle3`; unsupported PyTorch speculative modes were called out rather than sketched.
- [ ] If `examples/llm-api/llm_mgmn_trtllm_serve.sh` informed the answer, the answer labeled it as an operational launch template with hard-coded values, not as a verified starting config.
- [ ] The answer did not include a `How to restate the request` section or equivalent gatekeeping redirect.

## Repo Resources

| Resource | Path |
|---|---|
| Exact scenario database | `examples/configs/database/lookup.yaml` |
| `lookup.yaml` loader/helper | `examples/configs/database/database.py` |
| Curated starting points | `examples/configs/curated/lookup.yaml` |
| Config directory overview | `examples/configs/README.md` |
| Model deployment guides | `docs/source/deployment-guide/` |
| Model READMEs | `examples/models/core/` |
| Speculative decoding guide | `docs/source/features/speculative-decoding.md` |
| Parallelism strategy guide | `docs/source/features/parallel-strategy.md` |
| Disaggregated serving guide | `docs/source/features/disagg-serving.md` |
| Disaggregated examples | `examples/disaggregated/README.md` |
| Multi-node LLM API serve skeleton (operational template, not verified config) | `examples/llm-api/llm_mgmn_trtllm_serve.sh` |
