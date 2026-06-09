# Disaggregated Cancellation Stress-Test Suite

Disaggregated stress tests that gate regressions of the bug class
fixed by <https://github.com/NVIDIA/TensorRT-LLM/pull/13713>
(cleanup / lifetime / quiescence invariants in the disagg KV
transceiver under heavy mid-flight cancellation).

| | |
|---|---|
| **Tracked by** | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648), [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Bug it gates** | NVBug 6104831 (disaggregated permanent wedge) |
| **Fix it gates** | <https://github.com/NVIDIA/TensorRT-LLM/pull/13713> |

## Status

The registered QA stress entry now launches a real C++/V1 DeepSeek
disaggregated cluster in `log_only` mode. That mode sends normal
non-cancel completion probes through the front-end and scans saved
worker/server logs for UAF, broken-promise, and segmentation-fault
signatures. It is intentionally narrow so it can run regularly before
in-flight cancellation and poison-buffer hardening are available.

The full cancellation/poison marathon is implemented as an explicit
mode switch, but it is not the registered default yet.

| Mode | CI status | Threads | Coverage |
|------|-----------|---------|----------|
| `log_only` | Registered in `qa/llm_function_stress.txt` | log-only probe + log scanner | startup/data-path crash guard: UAF, broken promise, segfault-class logs |
| `full_cancel_poison` | Opt-in only | load, canary, injector, log scanner, metrics | cancellation load, failure injection, poison canaries, KV-growth guard |

Thread bodies:

| Thread | Status |
|--------|--------|
| `log_scanner_thread` | Implemented — hard-zero log fail-fast |
| `metrics_thread` | Implemented — `trtllm_kv_cache_utilization` scraper |
| `injector_thread` | Implemented — SIGSTOP/SIGCONT/SIGKILL + respawn |
| `canary_thread` | Implemented — greedy canaries + token-equivalence |
| `load_thread` | Implemented — duration-bounded steady/burst cancellation load |

Component-level coverage: `test_log_scanner.py`, `test_metrics_thread.py`,
`test_injector.py`, `test_canary.py`, `test_load_thread.py`. The
parametrized C++/V1 DeepSeek run is registered in the QA stress test
list as a real `log_only` guardrail.

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
  reference token IDs for `full_cancel_poison`.
- Per-scenario YAMLs covering additional axes: 1P1D, 4P2D,
  V1+Python, UCX, block-reuse-off, overlap-off, aggressive-timeout,
  multi-node (all Python-only test-side configuration).

## Mode Switch

The active mode is controlled by
`configs/marathon_cpp_v1_deepseek.yaml`:

```yaml
stress_config:
  mode: log_only
  duration_min: 10
```

Use `log_only` for regular CI until both runtime features are in
place:

- in-flight request cancellation support for the disaggregated path.
- poison-buffer hardening that makes poisoned cache transfers
  expected and recoverable.

To switch to the full cancellation/poison marathon after those
features are ready:

1. Set `stress_config.mode: full_cancel_poison`.
2. Set `stress_config.duration_min: 120` for the two-hour marathon.
3. Keep or tune `base_concurrency`, `bursts`, and `injections`.
4. Add `configs/stress_canary_prompts.json` with token references and
   keep `canary.check_token_equivalent: true`.
5. Add the poison-buffer hard-zero/expected-recovery patterns that
   match the finalized runtime behavior.
6. Raise the test-list timeout back to a full-marathon budget, e.g.
   `TIMEOUT (150)`.

## How to run

### Automatic QA stress run

The C++/V1 DeepSeek marathon is registered in
`tests/integration/test_lists/qa/llm_function_stress.txt` for scheduled
QA stress runs. The registered entry is:

```text
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_cpp_v1_deepseek.yaml] TIMEOUT (45)
```

The integration test-list parser interprets `TIMEOUT (45)` in
minutes. CI should run the list from `tests/integration/defs` with:

```bash
pytest --test-list=../test_lists/qa/llm_function_stress.txt \
  --output-dir=<ci-output-dir> \
  -s -v
```

The automatic runner must use the normal TRT-LLM integration container
or virtual environment with GPU access, `trtllm-serve` on `PATH`, and
`LLM_MODELS_ROOT` set so `DeepSeek-V3-Lite/bf16` resolves to local
model weights. The current registered run is `log_only`: setup can
take up to 20 minutes, then the harness probes for 10 minutes and
tails worker/server logs.

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

All component tests together, excluding the real cluster entry:

```bash
python3 -m pytest -c /dev/null -o addopts= \
  --confcutdir=tests/integration/defs/stress_test \
  tests/integration/defs/stress_test/disagg_cancel/ \
  -k "not test_disagg_cancellation_marathon" -q
```

In a full TRT-LLM dev container/venv (with `transformers` installed),
the same tests also run under the normal integration pytest path:

```bash
pytest -sv tests/integration/defs/stress_test/disagg_cancel/test_injector.py
```

### Manual regular guardrail run

From a full TRT-LLM integration environment:

```bash
cd /path/to/TensorRT-LLM/tests/integration/defs
export LLM_MODELS_ROOT=/path/to/model/root

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

### Manual full cancellation/poison run

After the runtime support is in place, switch the YAML to
`mode: full_cancel_poison`, set the intended duration and canary
references, then run the same pytest entry point. For development,
use a shorter `duration_min` and trim `injections:` locally before
restoring the checked-in values.

## Pass criteria

`log_only` is clean iff all of the following hold:

- The 3P3D disaggregated cluster starts and reaches readiness.
- At least one normal completion probe succeeds through the
  disaggregated front-end.
- No hard-zero log patterns for UAF, broken promise, or
  segmentation-fault-class failures appear in any saved worker or
  disagg-server log.

`full_cancel_poison` is clean iff all of the following hold:

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

Concrete thresholds for each metric are declared in the marathon YAML.

## How to debug a failure

When the regular guardrail fails:

1. Confirm the YAML parses:
   ```bash
   python -c "from harness import StressConfig; StressConfig.from_yaml_path('configs/marathon_cpp_v1_deepseek.yaml')"
   ```
2. Check the `failure_reason` field in `collect_results()` output.
3. Inspect the log tails printed by `disagg_test_utils.terminate()`
   during teardown; saved worker logs and `disagg_server.log` are
   tailed before cleanup.
4. If setup times out, confirm `LLM_MODELS_ROOT`, GPU count, and
   `trtllm-serve` availability in the integration environment.

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
