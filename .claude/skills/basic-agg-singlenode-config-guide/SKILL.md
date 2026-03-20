---
name: basic-agg-singlenode-config-guide
disable-model-invocation: true
description: Generate a source-backed starting `trtllm-serve --config` YAML for
  basic aggregate single-node PyTorch serving, aligned with checked-in TensorRT-LLM
  configs and deployment docs. Preserves explicit latency / balanced / throughput
  objectives. Excludes disaggregated, multi-node, and non-MTP speculative configs;
  exact-source-backed MTP recipes are allowed.
---

# Basic Aggregate Single-Node Config Guide

**Scope:** aggregate/IFB colocated prefill+decode, single node, PyTorch backend, non-speculative by default.

**Input:** model, GPU, ISL, OSL, concurrency, TP, performance objective (`Min Latency` | `Balanced` | `Max Throughput` | unspecified).
**Output:** repo-grounded starting YAML for `trtllm-serve --config`.

If the request is adjacent but out of scope, do **not** hard-refuse — see [Out-of-Scope Handling](#out-of-scope-handling).

## Constraints

Canonical definitions — steps and checklist reference these by **bold label**.

1. **MTP carveout:** Exact checked-in serve configs for `deepseek-ai/DeepSeek-R1-0528` or `nvidia/DeepSeek-R1-0528-FP4-v2` that explicitly set `speculative_config.decoding_type: MTP` are in scope. Copy the full `speculative_config` block verbatim — never interpolate speculative fields. Do not generalize to other models or speculative modes, and do not infer MTP eligibility from model-family similarity.

2. **Speculative exclusion:** Exclude configs containing `speculative_config` by default. The only exception is the **MTP carveout**.

3. **`cache_transceiver_config` rule:** Requires exact-model verification. Often signals disaggregated serving, but if the exact-model deployment guide explicitly points to a config containing it, keep it and call out the exception.

4. **Objective preservation:** Preserve the user's stated objective through config selection. Use `database.py` profile labels (`Min Latency`, `Balanced`, `Max Throughput`; plus `Low Latency`/`High Throughput` in smaller sets) as selection aids. If a config is unlabeled, treat it as a default starting point — do not claim it matches a specific objective. If the only match conflicts with the stated objective, call out the mismatch.

5. **Source preference:** Prefer checked-in configs over interpolation. When docs and configs disagree, prefer the config for the exact scenario and note the mismatch. Mark any interpolation as unverified.

## Response Format

**In-scope:** `Recommended config` → `Why this source` → `What is verified` → `What is unverified`

**Adjacent out-of-scope:** `Scope note` → `Best grounded config guess` → `Verified pieces` → `Inferred pieces` → `Risks / assumptions` → `Supporting sources`

## Step 0: Lock Objective and Decode Mode

Identify the user's objective and whether the starting point is non-speculative or MTP. Preserve both through the remaining steps per **objective preservation** and **speculative exclusion**.

`examples/configs/database/database.py` labels recipes by profile — use as a selection aid, not a universal SLA classifier.

## Step 1: Exact Database Match

Search `examples/configs/database/lookup.yaml` for an exact `(model, gpu, isl, osl, concurrency, num_gpus)` match. Use `database.py` as a loader/helper.

- Apply **speculative exclusion** + **MTP carveout**.
- Apply **`cache_transceiver_config` rule**.
- When multiple recipes exist at different concurrency points, use profile labels to match the user's objective per **objective preservation**.
- Prefer an exact match that also matches the stated objective over manual tuning.

## Step 2: Nearest Checked-In Config

If no exact match, search both lookup files:
- `examples/configs/database/lookup.yaml`
- `examples/configs/curated/lookup.yaml`

Apply the same constraints as Step 1. Additionally:
- Exclude disaggregated-only or prefill-only entries (e.g., `qwen3-disagg-prefill.yaml`).
- For curated configs, only treat intent as explicit when the repo labels it (e.g., `*-latency.yaml`, `*-throughput.yaml`, or guide text).
- If no in-scope config matches the stated objective, pick the nearest same-model starting point and call out the mismatch.

## Step 3: Read Model Docs

Consult `references/architecture.md` for the model-to-source mapping table, then read the model-specific deployment guide and README before adjusting knobs.

Key model-specific notes:
- **DeepSeek-R1:** Deployment guide points to `deepseek-r1-throughput.yaml` and `deepseek-r1-deepgemm.yaml`. Treat `deepseek-r1-latency.yaml` as source-backed but not deployment-guide-explicit. Checked-in configs include MTP per the **MTP carveout**.
- **Kimi-K2:** Current config is a **`cache_transceiver_config` rule** exception, not a generic basic-agg template. Enables `trust_remote_code: true` — call that out as a trust boundary. Curated lookup maps Kimi-K2 to `DeepseekV3ForCausalLM`; nearby DeepSeek configs are secondary comparators only.

## Step 4: Adjust Source-Backed Fields

Commonly scenario-dependent fields (adjust only these, guided by the checked-in source):

`max_batch_size`, `max_num_tokens`, `enable_attention_dp`, `kv_cache_config.free_gpu_memory_fraction`, `moe_expert_parallel_size` (MoE), `moe_config.backend` (when guide specifies), `stream_interval`, `num_postprocess_workers`, `cuda_graph_config.max_batch_size`/`batch_sizes`, and MTP-specific fields under the **MTP carveout**.

Do not assume other fields are constant across models/GPUs. For tuning notes and bench-derived hints, read `references/knob-heuristics.md`.

## Out-of-Scope Handling

When the request falls outside scope, provide a best-effort assembled answer with explicit verified/inferred separation rather than a hard refusal.

**Construction order:**
1. Nearest same-model checked-in serving baseline
2. Same-model nearby config for the closest supported mode
3. Feature-specific fragment from the relevant checked-in doc
4. Topology fields from the nearest checked-in topology guide
5. Explicit labeling of every assembled or inferred field

**Nearby sources for assembly:**
- Non-DeepSeek speculative: `docs/source/features/speculative-decoding.md`. PyTorch `trtllm-serve` supports only **Eagle3** for non-DeepSeek speculative assembly — do not assemble `NGram`/`DraftTarget`/`PARD`/`SA` serve YAML. This Eagle3-only caveat does not override the **MTP carveout**.
- Multi-node topology: `docs/source/features/parallel-strategy.md`
- Disaggregated serving: `docs/source/features/disagg-serving.md`, `examples/disaggregated/README.md`
- Multi-node launch skeleton: `examples/llm-api/llm_mgmn_trtllm_serve.sh` (operational Slurm template with hard-coded values, not a verified config)

Do not present assembled configs as fully verified or copy values across unrelated model families without saying so.

## Validation Checklist

**In-scope:**
- [ ] Config is basic-agg, single-node, non-speculative (or **MTP carveout** called out)
- [ ] **Objective preservation** satisfied (match or mismatch called out)
- [ ] **Speculative exclusion** satisfied (`speculative_config` absent unless **MTP carveout**; block copied verbatim)
- [ ] No disaggregated-only settings (or **`cache_transceiver_config` rule** exception called out)
- [ ] `cuda_graph_config.max_batch_size` checked against source (not assumed equal to server `max_batch_size`)
- [ ] `moe_expert_parallel_size` and `moe_config.backend` match checked-in source
- [ ] `trust_remote_code: true` called out as trust boundary when present
- [ ] OOM advice follows model guide first (levers: `max_batch_size`, `max_num_tokens`, `max_seq_len`, `free_gpu_memory_fraction`)
- [ ] Remaining interpolation labeled as unverified

**Adjacent out-of-scope:**
- [ ] Concrete best guess provided (not redirect-only)
- [ ] Verified vs. inferred pieces separated clearly
- [ ] Non-DeepSeek PyTorch speculative used Eagle3 only; other modes called out as unsupported
- [ ] Eagle3 caveat did not override **MTP carveout**
- [ ] No gatekeeping redirect section

## Repo Resources

| Resource | Path |
|---|---|
| Scenario database | `examples/configs/database/lookup.yaml` |
| Database loader/helper | `examples/configs/database/database.py` |
| Curated configs | `examples/configs/curated/lookup.yaml` |
| Deployment guides | `docs/source/deployment-guide/` |
| Model READMEs | `examples/models/core/` |
| Speculative decoding | `docs/source/features/speculative-decoding.md` |
| Parallelism strategy | `docs/source/features/parallel-strategy.md` |
| Disaggregated serving | `docs/source/features/disagg-serving.md` |
| Multi-node launch skeleton | `examples/llm-api/llm_mgmn_trtllm_serve.sh` |
