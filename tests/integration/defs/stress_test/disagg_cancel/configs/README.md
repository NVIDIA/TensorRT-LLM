# Marathon configs

Per-marathon YAML files consumed by ``test_disagg_cancel_stress.py``.

| File | Model | KV cache | Transceiver |
|------|-------|----------|-------------|
| `marathon_cpp_v1_deepseek.yaml` | `DeepSeek-V3-Lite/bf16` (MLA) | V1 | C++ |
| `marathon_python_v2_musespark_open.yaml` | `Muse-Spark-Open-NVFP4` (1P1D TP4, single-node GPU-shared) | V2 | Python |
| `marathon_python_v2_musespark_open_attach.yaml` | `musespark_open` (served name; attaches to an externally-launched 1P1D cluster) | V2 | Python |

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

## Attach mode: `stress_config.external_cluster`

By default the harness launches the disaggregated cluster locally via
`setup_disagg_cluster` (one node, `Popen` workers). Models whose
per-instance GPU footprint exceeds one node (e.g. TP4 instances on
4-GPU nodes, where 1P1D needs 8 GPUs) cannot be launched that way; for
those, boot the cluster externally (one node per instance) and let the
marathon **attach** to it:

```yaml
stress_config:
  mode: log_only            # full_cancel_poison is rejected (see below)
  external_cluster:
    server_url: ${TRTLLM_TEST_EXTERNAL_CLUSTER_URL}
    server_log: ${TRTLLM_TEST_EXTERNAL_CLUSTER_LOG_DIR}/orch/server.log   # optional
    worker_logs:            # required, >=1 entry; feeds the log scanner
      - role: ctx           # ctx | gen; per-role index = YAML order
        path: ${TRTLLM_TEST_EXTERNAL_CLUSTER_LOG_DIR}/ctx/server.log
      - role: gen
        path: ${TRTLLM_TEST_EXTERNAL_CLUSTER_LOG_DIR}/gen1/server.log
    worker_health_urls: []  # optional; each is GET-checked once at setup()
    model_name: musespark_open # optional; falls back to top-level `model`
```

When the block is present, `harness.setup()` skips the local launcher
entirely: it binds the orchestrator endpoint for the probe thread,
verifies `server_url` + `worker_health_urls` respond with HTTP 200,
and registers the log paths for the log-scanner thread. The attach
test itself needs **no GPUs** — only HTTP reachability to the
orchestrator and filesystem access to the worker logs (a shared
filesystem when the cluster spans nodes).

**Endpoint indirection is via environment variables**, not YAML
templating: string fields may contain `${VAR}` references, expanded by
the harness at `setup()` time. An unset variable is a hard error at
setup, but parse/validation is env-independent, so checked-in attach
YAMLs pass `test_all_marathon_yamls_parse_and_validate` (and pytest
collection) with nothing exported; the parametrized marathon test
**skips** when a referenced variable is unset. The conventional
variables used by the shipped attach YAML are
`TRTLLM_TEST_EXTERNAL_CLUSTER_URL` (orchestrator base URL) and
`TRTLLM_TEST_EXTERNAL_CLUSTER_LOG_DIR` (serve directory containing the
worker/orchestrator logs).

Attach mode supports `mode: log_only` only. `full_cancel_poison` is
rejected by `StressConfig.validate()` because fault injection signals
locally-launched worker subprocesses by PID and respawns them through
the shared launcher, neither of which exists for an external cluster.
Teardown of the external cluster is likewise the launcher's job, not
the harness's — `harness.stop()` only stops its own threads.

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
