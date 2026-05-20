# Disaggregated Cancellation Stress-Test Suite

Marathon-style stress tests that gate regressions of the bug class
fixed by [PR #13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713)
(cleanup / lifetime / quiescence invariants in the disagg KV
transceiver under heavy mid-flight cancellation).

| | |
|---|---|
| **Tracked by** | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648), [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Bug it gates** | NVBug 6104831 (disaggregated permanent wedge) |
| **Fix it gates** | [PR #13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) |

## Status

This is the **skeleton** stage. The harness class structure is in
place; thread bodies are intentionally stubs that exit immediately.
The pytest test exercises the lifecycle (`setup → start →
wait_until_done → stop`) only.

Thread bodies are implemented incrementally in subsequent commits,
in roughly this order (read-only first, side-effecting later):

1. `log_scanner_thread` (read-only — easiest)
2. `metrics_thread` (read-only — almost as easy)
3. `injector_thread` (subprocess control)
4. `canary_thread` (HTTP client + token-equivalence)
5. `load_thread` (wraps existing `run_cancel_stress_test`)

## File layout

```
tests/integration/defs/stress_test/disagg_cancel/
├── README.md                       (this file)
├── __init__.py
├── harness.py                      (DisaggCancellationStressHarness)
├── test_disagg_cancel_stress.py    (pytest entry point)
└── configs/
    ├── README.md                   (YAML schema + how to add a config)
    ├── marathon_a_v1_cpp_deepseek.yaml
    └── marathon_b_v2_py_qwen.yaml   (placeholder; not yet parametrized)
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

The marathons are **not** registered in pre-merge CI. They are run
nightly / weekly via
`tests/integration/test_lists/qa/llm_function_stress.txt` (wiring
lands together with the load-thread implementation).

### Local smoke (skeleton stage)

```bash
cd /path/to/TensorRT-LLM
LLM_MODELS_ROOT=/path/to/models \
  pytest -sv tests/integration/defs/stress_test/disagg_cancel/
```

In the skeleton stage this should complete in seconds because all
threads are no-ops; it only verifies the harness lifecycle compiles
and the YAMLs parse.

### Local marathon (once thread bodies are wired)

Once thread bodies are implemented, the same command will run the
full 2-hour marathon against Marathon A. To run a shorter smoke
during development, set `duration_min: 10` and trim
`injections:` in the YAML.

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
   python -c "from harness import StressConfig; StressConfig.from_yaml_path('configs/marathon_a_v1_cpp_deepseek.yaml').validate()"
   ```
2. Check the `failure_reason` field in `collect_results()` output.
3. Look at the pytest stdout for harness `logger` lines (each thread
   logs its identity on entry / exit).

## Cross-references

- [PR #13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) —
  the bug fix this suite gates regressions against.
- [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648),
  [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) —
  the cancellation / poison hardening initiative this suite is part
  of.
- `tests/integration/defs/disaggregated/test_disaggregated.py` —
  `run_cancel_stress_test`, `setup_disagg_cluster` (the
  building blocks the harness composes).
