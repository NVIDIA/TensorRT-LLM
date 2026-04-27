---
name: ad-conf-check
description: >
  Check whether AutoDeploy YAML configs were actually applied by analyzing server logs
  and optionally graph dumps (AD_DUMP_GRAPHS_DIR). Use when the user wants to verify
  config application, debug config issues, or check if AutoDeploy transforms (piecewise
  CUDA graph, multi-stream, sharding, fusion, etc.) were applied or fell back. Triggers
  on: "check config", "verify config", "ad-conf-check", "were my configs applied",
  "config not working", "check if piecewise is enabled", "check log for config", or any
  request to compare AD YAML settings against runtime behavior.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy Config Checker

Verify that AutoDeploy YAML configs were applied at runtime by cross-referencing with server logs and optionally graph dumps.

## Input

- **TensorRT-LLM source directory** (required) — path to the TensorRT-LLM repo root. Used to read the latest `default.yaml` and source code for up-to-date log patterns (the bundled reference doc may be stale).
- **YAML config file path(s)** (required) — one or more AutoDeploy YAML configs. When multiple files are provided, they are deep-merged left-to-right (later files override earlier ones for overlapping keys).
- **Server log file path** (required) — log output from the AutoDeploy server run.
- **Graph dump directory** (optional) — `AD_DUMP_GRAPHS_DIR` output directory containing per-transform graph snapshots (`NNN_stage_transform.txt`). Provides additional evidence for resolving UNKNOWN results.
- **Nsys trace file** (optional) — Nsight Systems profile (`.nsys-rep` or `.sqlite`) from the server run. Useful for verifying executor-level configs that produce no log output (e.g., `enable_chunked_prefill`, multi-stream concurrency, CUDA graph capture/replay).
- **Table output file path** (optional) — path to write human-friendly table results.
- **JSON output file path** (optional) — path to write machine-friendly JSON results.

## Output

### Human-friendly table (always presented to user)
- **Verification table** — one row per config key with columns: **Config** (key=value), **Result** (APPLIED / FAILED / SKIPPED / DISABLED / UNKNOWN), **Evidence** (log line or graph analysis proving the result).
- **Summary line** — total counts per status (e.g., `Total configs checked: 29 | APPLIED: 23 | UNKNOWN: 4 | ...`).
- **FAILED/WARNING details** — expanded information for any configs that failed or had warnings.

### Machine-friendly JSON (when JSON output path is given)
JSON file with two top-level keys:
- **`results`** — array of objects, each with `config`, `value`, `status`, `evidence`.
- **`summary`** — object with `total` (int) and `counts` (object mapping status to count, only non-zero statuses included).

## Workflow

1. **[Collect Inputs]** Ask the user for the following inputs:
   - **TensorRT-LLM source directory** (required) — path to the TensorRT-LLM repo root. Used to cross-check `default.yaml` and source code for the latest log patterns.
   - **YAML config file path(s)** (required) — one or more AutoDeploy configs used for the run. When multiple YAMLs are provided, they are deep-merged left-to-right: **later files override earlier ones** for overlapping keys. Tell the user: *"If you have multiple configs (e.g., a default config and a user override), list them in priority order — lowest priority first, highest priority last."*
   - **Server log file path** (required) — the log output from the server
   - **Graph dump directory** (optional but recommended) — the `AD_DUMP_GRAPHS_DIR` output directory containing per-transform graph snapshots. Files are named `NNN_stage_transform.txt` and show the graph AFTER each transform. When provided, graph analysis provides additional evidence (e.g., verifying sharded weights, collective ops, fused ops). This is especially useful for resolving UNKNOWN results.
   - **Nsys trace file** (optional) — Nsight Systems profile (`.nsys-rep` or `.sqlite`) from the server run. Useful for verifying executor-level configs that produce no log output (e.g., `enable_chunked_prefill`, multi-stream concurrency, CUDA graph capture/replay).
   - TensorRT-LLM source reference paths:
     - Example configs: `<trtllm_src>/examples/auto_deploy/model_registry/configs/*.yaml`
     - Default transform config (all available transforms and their defaults): `<trtllm_src>/tensorrt_llm/_torch/auto_deploy/config/default.yaml`

2. **[Update Reference Doc]** Before checking configs, ensure the bundled reference doc is up-to-date with the TensorRT-LLM source.

   Launch the `ad-conf-check-update` agent with:
   - `<trtllm_src>` — the TensorRT-LLM source directory from step 1
   - `<skill_dir>` — the directory containing this SKILL.md file

   The agent compares `<trtllm_src>/tensorrt_llm/_torch/auto_deploy/config/default.yaml` and the AutoDeploy source code against `<skill_dir>/references/config_log_patterns.md`. If any configs were added, removed, renamed, or if log patterns have changed, the agent updates the reference doc in-place and reports what changed.

   **After the agent completes:**
   - If the reference doc was updated, inform the user: *"Updated references/config_log_patterns.md to match the latest TensorRT-LLM source — see the agent's change summary below."* Then show the agent's summary.
   - If no changes were needed, briefly note: *"Reference doc is up-to-date with the TensorRT-LLM source."*

3. **[Parse Configs]** Run the parser script to flatten the YAML configs (`<skill_dir>` is the directory containing this SKILL.md file):

   **Input:** The TensorRT-LLM `default.yaml` as the base, followed by the user's YAML config path(s) from step 1. Always include `default.yaml` first so that user configs override the defaults.

   ```bash
   python3 <skill_dir>/scripts/parse_config.py <trtllm_src>/tensorrt_llm/_torch/auto_deploy/config/default.yaml <yaml_path1> [<yaml_path2> ...]
   ```
   This deep-merges the YAML files left-to-right (later files override earlier ones) and flattens nested keys into dotted notation (e.g., `kv_cache_config.enable_block_reuse`). By including `default.yaml` first, every known config key appears in the output even if the user only overrode a subset.

   **Output:** Flat JSON with all config `{key, value}` pairs. Example:
   ```json
   {
     "yaml_files": ["default.yaml", "user_override.yaml"],
     "total_configs": 15,
     "configs": [
       {"key": "compile_backend", "value": "torch-cudagraph"},
       {"key": "kv_cache_config.free_gpu_memory_fraction", "value": "0.85"},
       {"key": "transforms.compile_model.piecewise_enabled", "value": "True"}
     ]
   }
   ```

4. **[Quick Scan]** Check each config against the server log using parallel agents.

   **Input:** Config list from step 3, server log path from step 1, and [references/config_log_patterns.md](references/config_log_patterns.md).

   Split the configs from step 3 into 3 groups by section and launch 3 agents **in parallel**, each checking its group:

   | Agent | Config group | Keys starting with | Reference section |
   |-------|-------------|-------------------|-------------------|
   | Agent 1 | Top-level configs | `runtime`, `compile_backend`, `attn_backend`, `max_seq_len`, `max_num_tokens`, `max_batch_size`, `cuda_graph_batch_sizes`, `enable_chunked_prefill`, `model_factory`, `dtype`, etc. | "Top-Level Config Parameters" |
   | Agent 2 | KV cache configs | `kv_cache_config.*` | "kv_cache_config Parameters" |
   | Agent 3 | Transform configs | `transforms.*` (or any key matching a transform name like `compile_model`, `detect_sharding`, `multi_stream_*`, `fuse_*`, `gather_logits_*`, etc.) | "Transform Parameters" |

   Each agent receives:
   - Its subset of `{key, value}` pairs
   - The server log file path
   - The reference doc [references/config_log_patterns.md](references/config_log_patterns.md) (including verification source tags: `[log]`, `[graph]`, `[nsys]`)
   - The nsys trace file path (if provided)

   Each agent, for every config in its group:
   1. Reads the reference doc to find the relevant keywords and patterns for this config key.
   2. Greps the server log for those patterns. Key search strategies:
      - For transform configs: grep for `[stage=..., transform=<name>]` and check the `[SUMMARY]` line (`matches=N` → APPLIED if N>0, SKIPPED if N=0).
      - For configs with success/failure indicators: grep for those specific strings.
      - For configs with no known log pattern: grep for `key=value` or the key name near the value.
      - For configs with `enabled: false`: mark as DISABLED without log search.
   3. Assigns a status based on what was found:
      - **APPLIED** — log confirms the config took effect
      - **FAILED** — log shows the config was attempted but fell back or errored
      - **SKIPPED** — transform ran but found nothing to do (0 matches)
      - **DISABLED** — config explicitly set `enabled: false`
      - **UNKNOWN** — no log evidence found (config may still be active but unlogged)
   4. Records the evidence (the matching log line or lack thereof).

   **Output:** Each agent returns a list of `{config, value, status, evidence}` entries for its group. Merge all 3 lists into the combined result.

5. **[Double Check]** For any UNKNOWN entries from step 4, investigate further **before** presenting results to the user (FAILED entries already have concrete log evidence and do not need double-checking):

   **Input:** List of UNKNOWN config entries from step 4 output, the server log file, and [references/config_log_patterns.md](references/config_log_patterns.md).

   - Re-read [references/config_log_patterns.md](references/config_log_patterns.md) for alternative patterns
   - Grep the log more broadly for the transform name: `[stage=..., transform=<name>]`
   - Look for `[APPLY]` prefixed lines and `[SUMMARY]` lines for that transform
   - Check for `"Falling back"`, `"Skipping"`, or `"failed"` near the transform logs
   - If graph dump directory was provided:
     - Graph files are named `NNN_stage_transform.txt` — each contains the FX graph AFTER that transform. Compare before/after by reading consecutive files.
     - Graph evidence can upgrade UNKNOWN to APPLIED (e.g., collective ops after lm_head confirm sharding, fused custom ops confirm fusion transforms).
     - Graph analysis verifies: sharding (collective ops, weight shape changes), attention backend (op types), MoE fusion (fused op presence), GEMM fusion (linear op count changes), RMSNorm/SwiGLU/RoPE pattern matching (custom op presence).
     - See [references/graph_verification_patterns.md](references/graph_verification_patterns.md) for the full list of graph-based checks.
   - If nsys trace was provided, check for executor-level configs tagged `[nsys]` in the reference doc (e.g., `enable_chunked_prefill`, `enable_block_reuse`, multi-stream concurrency, CUDA graph capture/replay)

   **Output:** For each investigated UNKNOWN entry, either additional evidence found (with status upgrade) or confirmation that the config is genuinely unlogged.

6. **[Report]** Present the final results to the user.

   **ALWAYS show the full detailed table.** Do NOT summarize or condense. Present one row per config with columns:
   - **Config** — the config key and its value (e.g., `compile_backend = torch-cudagraph`)
   - **Result** — one of: APPLIED, FAILED, SKIPPED, DISABLED, UNKNOWN
   - **Evidence** — the log line or pattern that proves the result

   After the table, show the summary line (e.g., `Total configs checked: 29 | APPLIED: 23 | ...`) and any FAILED/WARNING details. Include any additional findings from the Double Check step (step 5).

   If the user requested output files, write:
   - **Table output** — the human-friendly table as plain text
   - **JSON output** — machine-friendly JSON with `results` array and `summary` object

## Key Patterns to Know

- Every transform logs: `[stage=<stage>, transform=<name>] [SUMMARY] matches=N | time: ...`
- Piecewise success chain: `dual-mode enabled` -> `prepared with N submodules` -> `captured graphs`
- Piecewise failure: `"model is not a GraphModule...Falling back to eager execution"`
- Sharding: `"Using allreduce strategy: SYMM_MEM"`, `"Applied N TP shards from config"`

## Gotchas

- **Every YAML key must appear in the output.** Check all configs from the YAML, not just ones with known patterns. If a config key has no entry in the reference doc, grep the log for the key name and value. New/unknown configs should still be reported — never silently skip them.
- **UNKNOWN does not mean the config was ignored.** Some configs (e.g., `enable_chunked_prefill`, `enable_block_reuse`) are consumed at executor/runtime level and produce no log output. UNKNOWN means "no log evidence found", not "config was not applied".
- **Deprecated config names may cause FAILED.** For example, `torch_dtype` is deprecated in favor of `dtype`, and `cuda_graph_batch_sizes` (top-level) is replaced by `cuda_graph_config.batch_sizes`. Look for deprecation warning messages in the log. Old keys may be silently ignored.
- **Runtime may adjust configured values.** For example, `max_seq_len` may be configured as 262144 but adjusted down to 16384 at runtime due to memory constraints. Report this as APPLIED with a WARNING annotation.
- **ANSI color codes in logs.** AutoDeploy uses colored log output. Strip or ignore ANSI escape sequences when matching patterns.
- **Reference doc is auto-updated.** Step 2 runs the `ad-conf-check-update` agent to sync [references/config_log_patterns.md](references/config_log_patterns.md) with the latest TensorRT-LLM source before any config checking begins. If the agent reports changes, review its summary to understand what shifted.
