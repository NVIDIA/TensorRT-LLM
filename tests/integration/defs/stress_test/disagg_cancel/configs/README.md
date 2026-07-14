# Marathon configs

Per-marathon YAML files consumed by ``test_disagg_cancel_stress.py``.

| File | Model | KV cache | Transceiver |
|------|-------|----------|-------------|
| `marathon_cpp_v1_deepseek.yaml` | `DeepSeek-V3-Lite/bf16` (MLA) | V1 | C++ |
| `marathon_python_v2_qwen.yaml` | `Qwen2.5-7B-Instruct` (GQA) | V2 | Python |

Only configurations listed in `_MARATHON_CONFIGS` in
`test_disagg_cancel_stress.py` are actually parametrized; the other
YAMLs live in this directory as ready-to-wire templates.

## Schema

The YAML extends the existing
`tests/integration/defs/disaggregated/test_configs/disagg_config_cancel_stress_test*.yaml`
schema. The top-level `hostname / model / backend / context_servers /
generation_servers` keys pass through to
`tests/integration/defs/disaggregated/disagg_test_utils.py::setup_disagg_cluster`
unchanged.

The new `stress_config:` top-level block is consumed by
`harness.py::StressConfig`. Field-level documentation lives in
`StressConfig` itself (dataclass field docstrings) and the
example values in `marathon_cpp_v1_deepseek.yaml`.

## Harness modes

`stress_config.mode` is the switch between the regular guardrail and
the full cancellation/poison marathon:

- `log_only`: registered CI mode. The harness launches the real
  disaggregated cluster, sends normal non-cancel probes, and scans
  worker/server logs for UAF, broken-promise, and segmentation-fault
  signatures. This mode is safe before in-flight cancellation and
  poison-buffer hardening are available.
- `full_cancel_poison`: opt-in mode for the completed runtime. The
  harness enables the cancellation load, SIGSTOP/SIGKILL injections,
  token-equivalent canaries, metrics scraping, and KV-growth checks.

When switching from `log_only` to `full_cancel_poison`, update
`duration_min`, canary references, poison-buffer log expectations, and
the test-list timeout together. The top-level README has the exact
checklist.

## Backend-knob axis: KV-cache manager × transceiver runtime

Two knobs select which (KV cache manager × transceiver runtime)
backend the marathon exercises:

- `stress_config.kv_cache_manager: v1 | v2` controls
  `kv_cache_config.use_kv_cache_manager_v2: false | true` on each
  worker. V1 is the C++ KV cache manager (`KVCacheManager`); V2 is
  the newer pure-Python manager (`KVCacheManagerV2`).
- `stress_config.transceiver: cpp | python` controls
  `cache_transceiver_config.transceiver_runtime: "CPP" | "PYTHON"`
  on each worker. `cpp` selects the C++-backed transceiver
  (`BindKvCacheTransceiver`); `python` selects the pure-Python
  transceiver (`KvCacheTransceiverV2`).

The `(v2, cpp)` combination is unsupported (the C++ transceiver
requires the V1 KV cache manager) and rejected by
`StressConfig.validate()`.

## Adding a new YAML

Additional marathon scenarios land as new YAMLs here, with **no
Python changes** required beyond extending the parametrize list. To
add a new config:

1. Copy `marathon_cpp_v1_deepseek.yaml` as a template.
2. Choose `stress_config.mode`, then adjust `model`,
   `kv_cache_manager`, `transceiver`, and any
   load-shape knobs (`base_concurrency`, `client_cancel_rate`,
   `output_length`, `injections:`, `pass_criteria:`).
3. Add the new filename to `_MARATHON_CONFIGS` in
   `test_disagg_cancel_stress.py`.
4. (If applicable) generate canary references for the new model and
   add the reference JSON next to this directory.
5. Register the new test ID in
   `tests/integration/test_lists/qa/llm_function_stress.txt`.
