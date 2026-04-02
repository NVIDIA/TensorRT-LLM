---
name: ad-conf-check
description: >
  Check whether AutoDeploy YAML configs were actually applied by analyzing server logs.
  Use when the user wants to verify config application, debug config issues, or check if
  AutoDeploy transforms (piecewise CUDA graph, multi-stream, sharding, fusion, etc.) were
  applied or fell back. Triggers on: "check config", "verify config", "ad-conf-check",
  "were my configs applied", "config not working", "check if piecewise is enabled",
  "check log for config", or any request to compare AD YAML settings against runtime behavior.
---

# AutoDeploy Config Checker

Verify that AutoDeploy YAML configs were applied at runtime by cross-referencing with server logs.

## Workflow

1. Ask the user for two inputs:
   - **YAML config file path(s)** — one or more AutoDeploy configs used for the run. When multiple YAMLs are provided, they are deep-merged left-to-right: **later files override earlier ones** for overlapping keys. Tell the user: *"If you have multiple configs (e.g., a default config and a user override), list them in priority order — lowest priority first, highest priority last."*
   - **Server log file path** — the log output from the server
   - Example configs: `examples/auto_deploy/model_registry/configs/*.yaml`

2. Run the checker script:
   ```bash
   python3 <skill_dir>/scripts/check_config.py <yaml_path1> [<yaml_path2> ...] --log <log_path>
   ```
   To save results to file (if requested):
   ```bash
   python3 <skill_dir>/scripts/check_config.py <yaml_path1> [<yaml_path2> ...] --log <log_path> --output <output_path>
   ```

3. **ALWAYS show the full detailed table to the user.** Do NOT summarize or condense the script output. Present the complete table as-is with all rows. The table has 3 columns:
   - **Config** — the config key and its value (e.g., `compile_backend = torch-cudagraph`)
   - **Result** — one of: APPLIED, FAILED, SKIPPED, DISABLED, UNKNOWN
   - **Evidence** — the log line or pattern that proves the result

   Status meanings:
   - **APPLIED** — log confirms the config took effect
   - **FAILED** — log shows the config was attempted but fell back or errored
   - **SKIPPED** — transform ran but found nothing to do (0 matches)
   - **DISABLED** — config explicitly set `enabled: false`
   - **UNKNOWN** — no log evidence found (config may still be active but unlogged)

4. After the table, show the summary line (e.g., `Total configs checked: 29 | APPLIED: 23 | ...`) and any FAILED/WARNING details.

5. For any FAILED or UNKNOWN entries, investigate further:
   - Read [references/config_log_patterns.md](references/config_log_patterns.md) for the full pattern catalog
   - Grep the log for the specific transform name: `[stage=..., transform=<name>]`
   - Look for `[APPLY]` prefixed lines and `[SUMMARY]` lines for that transform
   - Check for `"Falling back"`, `"Skipping"`, or `"failed"` near the transform logs

## Key Patterns to Know

- Every transform logs: `[stage=<stage>, transform=<name>] [SUMMARY] matches=N | time: ...`
- Piecewise success chain: `dual-mode enabled` -> `prepared with N submodules` -> `captured graphs`
- Piecewise failure: `"model is not a GraphModule...Falling back to eager execution"`
- Sharding: `"Using allreduce strategy: SYMM_MEM"`, `"Applied N TP shards from config"`

## Gotchas

- **Every YAML key must appear in the output.** The script checks all configs from the YAML, not just the ones with hardcoded checkers. If a config key appears in the YAML but has no specific checker, the script uses a generic fallback that searches the log for the key name and value. This means new/unknown configs are still reported — they will not be silently skipped.
- **UNKNOWN does not mean the config was ignored.** Some configs (e.g., `enable_chunked_prefill`, `enable_block_reuse`) are consumed at executor/runtime level and produce no log output. UNKNOWN means "no log evidence found", not "config was not applied".
- **Deprecated config names may cause FAILED.** For example, `torch_dtype` is deprecated in favor of `dtype`. The script detects fallback warnings and marks these as FAILED with the deprecation note.
- **Runtime may adjust configured values.** For example, `max_seq_len` may be configured as 262144 but adjusted down to 16384 at runtime due to memory constraints. The script reports this as APPLIED with a WARNING annotation.
- The script handles ANSI color codes in logs (AutoDeploy uses colored log output).
- For configs where the generic fallback marks UNKNOWN, manually grep the log using patterns from [references/config_log_patterns.md](references/config_log_patterns.md).
