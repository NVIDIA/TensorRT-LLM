# Performance tests

## CI performance sanity

CI performance sanity tests use `test_perf_sanity.py` and the per-platform
`*perf_sanity*` test lists. See the [regression system guide](README_perf_regression_system.md)
for metric collection and regression checks, and the
[VisualGen performance guide](README_test_visual_gen_perf_sanity.md) for VisualGen coverage.

## QA benchmarks

`test_perf.py` provides the dynamically generated benchmark cases used by QA
test lists under `tests/integration/test_lists/qa/`. Run these cases with both
`--perf` and `--test-list=<rendered-test-list.txt>`; see the
[runnable example](../../README.md#run-perf-tests).
`--perf-log-formats csv` and `--perf-log-formats yaml` select the result formats
and require `--output-dir=<results-directory>`.

The [CPU harness tests](../../../unittest/others/test_perf_qa_functional.py)
exercise dynamic test generation, bench/serve commands, metric output, and
failure handling with simulated GPU processes. They run through the
`unittest/others` entry in `l0_cpu.yml`; real GPU execution remains QA coverage.

## Warmup pass

Before measuring, `test_perf.py` warms up with one un-measured wave of the benchmark's own traffic (its concurrency, or `min(reqs, maxbs)` without one), so JIT kernels for the real prefill shapes compile outside the measured window: `bench` cases raise `trtllm-bench --warmup` to that wave, and `serve` cases send it with other prompts before each client. `TRTLLM_PERF_WARMUP_PASS=0` turns it off.
