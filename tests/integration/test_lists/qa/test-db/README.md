# QA functional test-db

`qa_functional.yml` defines which QA functional tests run on which hardware. It replaces
the flat `llm_function_core.txt`, which ran the same 997 lines on every machine and let
pytest markers skip whatever did not apply.

Blocks are ordered by capability tier, so placing a test is: which hardware, then how many
GPUs. `system_gpu_count` is a condition only on `8xB200`, `GB200`, `8xB300`, `GB300` — the
machines whose SLURM allocation size the pipeline can vary. Elsewhere the machine is
scheduled whole and the requirement is carried as a `# N-GPU` comment inside the block.

## Block index

| # | tests | condition | GPUs | machines |
|---|---:|---|---|---|
| | | **no compute-capability constraint** | | |
| 1 | 117 | `cc any` | 1 GPU | 8xA100 8xL40S 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 2 | 25 | `cc any` | 2 GPU | 8xA100 8xL40S 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 3 | 28 | `cc any` | 4 GPU | 8xA100 8xL40S 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 4 | 10 | `cc any` | 8 GPU | 8xA100 8xL40S 8xH20 H100 8xB200 8xB300 |
| 5 | 15 | `cc any` | 1 GPU | 8xA100 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 6 | 4 | `cc any` | 2 GPU | 8xA100 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 7 | 1 | `cc any` | 4 GPU | 8xA100 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| | | **compute capability >= 8.0** | | |
| 8 | 1 | `cc <= 10.0` | 1 GPU | 8xA100 8xH20 H100 8xB200 GB200 |
| | | **compute capability >= 8.9** | | |
| 9 | 12 | `cc >= 8.9` | 1 GPU | 8xL40S 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 10 | 19 | `cc >= 8.9` | 4 GPU | 8xL40S 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 11 | 2 | `cc 8.9-9.0` | whole | 8xL40S 8xH20 H100 |
| | | **compute capability >= 9.0** | | |
| 12 | 148 | `cc >= 9.0` | 1 GPU | 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 13 | 40 | `cc >= 9.0` | 2 GPU | 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 14 | 136 | `cc >= 9.0` | 4 GPU | 8xH20 H100 8xB200 GB200 8xB300 GB300 |
| 15 | 12 | `cc >= 9.0` | 8 GPU | 8xH20 H100 8xB200 8xB300 |
| 16 | 1 | `cc >= 9.0` | 1 GPU | 8xH20 8xB200 GB200 8xB300 GB300 |
| 17 | 1 | `cc >= 9.0` | 4 GPU | 8xH20 8xB200 GB200 8xB300 GB300 |
| 18 | 2 | `cc 9.0-10.0` | 1 GPU | 8xH20 H100 8xB200 GB200 |
| 19 | 62 | `cc == 9.0` | whole | 8xH20 H100 |
| | | **compute capability >= 10.0** | | |
| 20 | 72 | `cc >= 10.0` | 1 GPU | 8xB200 GB200 8xB300 GB300 |
| 21 | 6 | `cc >= 10.0` | 2 GPU | 8xB200 GB200 8xB300 GB300 |
| 22 | 149 | `cc >= 10.0` | 4 GPU | 8xB200 GB200 8xB300 GB300 |
| 23 | 83 | `cc >= 10.0` | 8 GPU | 8xB200 8xB300 |
| 24 | 1 | `cc == 10.0` | 1 GPU | 8xB200 GB200 |
| 25 | 3 | `cc == 10.0` | 4 GPU | 8xB200 GB200 |
| 26 | 13 | `cc == 10.0` | 4 GPU | GB200 |
| | | **compute capability >= 10.3** | | |
| 27 | 2 | `cc >= 10.3` | 4 GPU | 8xB300 GB300 |

## Placing a new test

1. **Which hardware does it need?** Find the capability tier above.
2. **How many GPUs?** Pick the `N GPU` block within that tier. Blocks marked `whole` are
   not split by GPU count — add the test under the matching `# N-GPU` comment group
   inside the block.
3. **If nothing matches, add a block.** A condition MUST NOT be looser than the test's own
   skip markers: selecting a test on a machine where its markers would skip it schedules
   work that cannot run and reports coverage that never happened. Stricter than the
   markers is fine — that is QA scheduling policy.

## Preview what a machine runs

```
trt-test-db --interface v0 -d tests/integration/test_lists/qa/test-db \
  --context qa_functional --test-names \
  --match '{"compute_capability": 10.0, "gpu_memory": 183359, "system_gpu_count": 8, "cpu": "x86_64"}' \
  -o /tmp/my_list.txt
```

Fact keys are what `tests/integration/defs/sysinfo/get_sysinfo.py` emits. Pass
`--interface v0` explicitly: releases from 1.8.6 default to a CLI where `--context` and
`--match` do not exist.
