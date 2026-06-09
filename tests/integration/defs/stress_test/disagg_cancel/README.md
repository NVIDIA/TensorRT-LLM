# Disaggregated Cancellation Stress-Test Suite

Marathon-style stress tests that gate regressions of the bug class
fixed by <https://github.com/NVIDIA/TensorRT-LLM/pull/13713>
(cleanup / lifetime / quiescence invariants in the disagg KV
transceiver under heavy mid-flight cancellation).

| | |
|---|---|
| **Tracked by** | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648), [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Bug it gates** | NVBug 6104831 (disaggregated permanent wedge) |
| **Fix it gates** | <https://github.com/NVIDIA/TensorRT-LLM/pull/13713> |

## Status

The harness class structure and lifecycle are in place. Thread bodies
land incrementally:

| Thread | Status |
|--------|--------|
| `log_scanner_thread` | Implemented — hard-zero log fail-fast |
| `metrics_thread` | Implemented — `trtllm_kv_cache_utilization` scraper |
| `injector_thread` | Implemented — SIGSTOP/SIGCONT/SIGKILL + respawn |
| `canary_thread` | Implemented — greedy canaries + token-equivalence |
| `load_thread` | Implemented — duration-bounded steady/burst cancellation load |

Component-level coverage: `test_log_scanner.py`, `test_metrics_thread.py`,
`test_injector.py`, `test_canary.py`, `test_load_thread.py`. The
parametrized C++/V1 DeepSeek marathon is registered in the QA stress
test list; it still runs a lifecycle smoke until `setup()` launches a
real cluster.

## File layout

```
tests/integration/defs/stress_test/disagg_cancel/
├── README.md                       (this file)
├── __init__.py
├── harness.py                      (DisaggCancellationStressHarness)
├── test_disagg_cancel_stress.py    (pytest entry point)
├── test_log_scanner.py             (log_scanner unit tests)
├── test_metrics_thread.py          (metrics_thread unit tests)
├── test_injector.py                (injector unit tests)
├── test_canary.py                  (canary_thread unit tests)
├── test_load_thread.py             (load_thread unit tests)
└── configs/
    ├── README.md                   (YAML schema + how to add a config)
    ├── marathon_cpp_v1_deepseek.yaml
    └── marathon_python_v2_qwen.yaml   (placeholder; not yet parametrized)
```

Future additions:
- `tools/generate_canary_references.py` — one-shot reference generator
  that records greedy-decode token IDs for the canary prompts.
- `configs/stress_canary_prompts.json` — canary prompts + recorded
  reference token IDs (consumed by the canary thread).
- Per-scenario YAMLs covering additional axes: 1P1D, 4P2D,
  V1+Python, UCX, block-reuse-off, overlap-off, aggressive-timeout,
  multi-node (all Python-only test-side configuration).

## How to run

### Automatic QA stress run

The C++/V1 DeepSeek marathon is registered in
`tests/integration/test_lists/qa/llm_function_stress.txt` for scheduled
QA stress runs. The registered entry is:

```text
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_cpp_v1_deepseek.yaml] TIMEOUT (150)
```

The integration test-list parser interprets `TIMEOUT (150)` in
minutes. CI should run the list from `tests/integration/defs` with:

```bash
pytest --test-list=../test_lists/qa/llm_function_stress.txt \
  --output-dir=<ci-output-dir> \
  -s -v
```

The automatic runner must use the normal TRT-LLM integration container
or virtual environment. Once `setup()` launches the real disaggregated
cluster, it also needs `LLM_MODELS_ROOT` set so
`DeepSeek-V3-Lite/bf16` resolves to local model weights. At this stage
the entry still completes as a lifecycle smoke because `setup()` has
not launched a real disaggregated cluster yet; once cluster setup
lands, the same test-list entry runs the configured 120-minute
marathon.

### Unit tests (no GPU, no cluster)

Component tests for individual harness threads run in isolation. They
do **not** need `LLM_MODELS_ROOT`, GPUs, or a TRT-LLM venv with
`transformers` — use `--confcutdir` so pytest skips the parent
`tests/integration/defs/conftest.py`.

From the repository root:

```bash
cd /path/to/TensorRT-LLM

export PYTHONPATH=tests/integration/defs:tests/integration/defs/disaggregated

# Steps 1-2 — log scanner + metrics (optional sanity)
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/test_log_scanner.py \
  tests/integration/defs/stress_test/disagg_cancel/test_metrics_thread.py -v

# Step 3 — injector thread (SIGSTOP / SIGCONT / SIGKILL + respawn)
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/test_injector.py -v

# Step 4 — canary thread (greedy canaries + token-equivalence)
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/test_canary.py -v

# Step 5 — load thread (steady/burst wrapper around cancel stress load)
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/test_load_thread.py -v

# Marathon YAML parse/validate (includes stress_config.injections schedule)
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_all_marathon_yamls_parse_and_validate -v
```

All component tests together:

```bash
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/ -q
```

In a full TRT-LLM dev container/venv (with `transformers` installed),
the same tests also run under the normal integration pytest path:

```bash
pytest -sv tests/integration/defs/stress_test/disagg_cancel/test_injector.py
```

### Direct lifecycle smoke (no GPU, no cluster)

```bash
PYTHONPATH=tests/integration/defs:tests/integration/defs/disaggregated \
python3 -m pytest -c /dev/null -o addopts= \
  -p no:cacheprovider \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon -sv
```

`setup()` is still a stub, so this only checks harness lifecycle
(`setup` → `start` → `wait` → `stop`). The injector thread exits
immediately because no workers are registered via
`bind_tracked_workers()`.

### Manual QA stress-list selection

From a full TRT-LLM integration environment:

```bash
cd /path/to/TensorRT-LLM/tests/integration/defs
export LLM_MODELS_ROOT=/path/to/model/root  # required once real setup() lands

pytest stress_test/disagg_cancel/test_disagg_cancel_stress.py \
  --test-list=../test_lists/qa/llm_function_stress.txt \
  --output-dir=/tmp/trtllm-disagg-cancel-stress \
  -s -v
```

To collect without running:

```bash
pytest stress_test/disagg_cancel/test_disagg_cancel_stress.py \
  --test-list=../test_lists/qa/llm_function_stress.txt \
  --output-dir=/tmp/trtllm-disagg-cancel-stress \
  -s --co -q
```

### Manual CI trigger

On a GitHub pull request, ask the CI bot which stress stages are
available, then trigger the QA stress stage that consumes
`tests/integration/test_lists/qa/llm_function_stress.txt`:

```text
/bot help
/bot run --extra-stage "<QA stress stage that runs llm_function_stress.txt>"
```

The bot stage name is owned by CI/Jenkins configuration and is not
declared in this directory.

### Local marathon (after `setup()` lands)

Once `setup()` launches a real 3P3D cluster and registers workers,
the full 2-hour marathon runs via the same pytest entry point. For
development, set `duration_min: 10` and trim `injections:` in the
YAML.

## Pass criteria

A marathon run is "clean" iff all of the following hold:

- No hard-zero log patterns (e.g. `Cannot cancel request`, `Broken
  promise`, `unquiesced`, double-free / UAF traces) appear in any
  worker's stderr.
- All workers (context + generation) are alive at the end of the
  marathon. The injector-induced respawns must succeed; subsequent
  SIGKILLs would otherwise be observable as never-recovered ranks.
- Final 5/5 canary requests pass within 30 s of marathon end, with
  100 % token-equivalence against the recorded references.
- Canary error rate < 1 % overall and < 10 % within any 1-min
  injection window.
- Recovery time < 30 s after each SIGCONT / SIGKILL-respawn event
  (measured against the canary stream).
- KV-cache utilization growth ≤ 10 percentage points end-to-end
  (leak guard).

Concrete thresholds for each metric are declared in the marathon
YAML's `pass_criteria:` block.

## How to debug a failure

(Stub — the full debug guide lands together with the thread
implementations.)

For now, when the skeleton test fails:

1. Confirm the YAML parses:
   ```bash
   python -c "from harness import StressConfig; StressConfig.from_yaml_path('configs/marathon_cpp_v1_deepseek.yaml')"
   ```
2. Check the `failure_reason` field in `collect_results()` output.
3. Look at the pytest stdout for harness `logger` lines (each thread
   logs its identity on entry / exit).

## Cross-references

- <https://github.com/NVIDIA/TensorRT-LLM/pull/13713> —
  the bug fix this suite gates regressions against.
- [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648),
  [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) —
  the cancellation / poison hardening initiative this suite is part
  of.
- `tests/integration/defs/disaggregated/test_disaggregated.py` —
  `run_cancel_stress_test`, `setup_disagg_cluster` (the
  building blocks the harness composes).
