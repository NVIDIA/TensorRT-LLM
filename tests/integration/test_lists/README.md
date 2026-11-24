<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Test Lists Reference

Detailed reference for `waives.txt` matching and `unittest/` batch
consolidation. For routine edits, start with the short rules in
[`AGENTS.md`](AGENTS.md).

## Waive list reference

### How PR CI merges waives

In PR CI, Jenkins normally merges the PR's `waives.txt` with the latest
target-branch list (tip of tree, or TOT), then applies exact-line removals from
the PR diff. Pytest receives this merged file via `--waives-file`, so waives on
the PR's target branch can still skip tests even when absent from the PR
checkout (regardless of the target branch name).

### Syntax and quoting

Copy the test path and command-line options, omitting `ISOLATION`,
`TIMEOUT (N)`, and YAML comments — that metadata is not part of the collected
node ID and prevents matching if left in:

```yaml
- accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_fp8_block_scales[latency] TIMEOUT (90) ISOLATION
- unittest/executor/test_base_worker.py ISOLATION
```

```text
accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_fp8_block_scales[latency] SKIP (Known issue in this test)
unittest/executor/test_base_worker.py SKIP (Known issue in this file)
```

- One waive per line; blank lines are ignored.
- `#` starts a comment **anywhere** on the line, including inside quotes or
  reasons (`-k "case#tag"` gets truncated, and a `#fragment` in a tracking URL
  can strip the closing `)` and break parsing of the whole file).
- Replace illustrative names and reasons above with actual tests and bugs.

### Matching: outer entries vs. inner unit tests

Ordinary integration tests match exact node IDs. A `unittest/` test-db entry
runs in two layers — an outer integration wrapper, then an inner pytest
subprocess — and waives apply during `pytest_collection_modifyitems` at each
layer before test bodies run.

**Outer entry:** `unittest/<entry>` becomes
`test_unittests.py::test_unittests_v2[unittest/<entry>]`. Copy options such as
`-k`, `-m`, and `--ignore` verbatim (order, spaces, quotes — shell arguments are
not normalized, so changing options requires rechecking the waive). An exact
match skips the wrapper and its entire inner run; inner waives cannot restore
it.

For `unittest/_torch/modeling SKIP (Known modeling issue)`:

| Separate test-db entry | Entire entry skipped? |
| --- | --- |
| `unittest/_torch/modeling` | Yes: exact entry. |
| `unittest/_torch/modeling -k "modeling_llama"` | No: the options make it a different entry. |
| `unittest/_torch/modeling/test_modeling_llama.py` | No: the file path is a different entry. |

This is not a directory-prefix match. To skip several filtered entries, list
each complete entry on its own line.

**Inner unittest:** if the wrapper runs, the inner matcher checks these names in
order (first match wins):

| Priority | Waive name (append `SKIP (reason)`) | Scope |
| --- | --- | --- |
| 1 | `unittest/test_example.py::TestExample::test_forward[fp16]` | One parameter instance. |
| 2 | `unittest/test_example.py::TestExample::test_forward` | All parameter instances of the method. |
| 3 | `unittest/test_example.py::TestExample` | All collected tests in the class. |
| 4 | `unittest/test_example.py` | All collected tests in the file. |

It strips the `[...]` parameter suffix, then strips trailing `::` components
until the file level. Matching stops at the file (no directory walking); a
directory invocation can only be waived as an exact outer entry. Neither layer
supports globs, regexes, or partial parameter names, and ordinary integration
tests do not inherit inner file/class/function matching.

A file waive skips that file even when collected via another outer entry (e.g.,
one with `-k`), leaving other files unaffected. Use collected `unittest/` paths,
not repository-root `tests/unittest/` paths.

### Platform and SM scope

Platform scope applies to runners that pass `--waives-file` and `--test-prefix`.
CI passes `--test-prefix=<stage-name>` to outer pytest and
`<stage-name>/unittest` to inner pytest. `<test>` can be an outer/integration
entry or a supported inner unittest name.

| Waive form (append `SKIP (reason)`) | Scope |
| --- | --- |
| `<test>` | All platforms receiving the waive file. Unprefixed integration waives also work locally without `--test-prefix`. |
| `full:DGX_H100/<test>` | `DGX_H100` stages, e.g., `DGX_H100-PyTorch-1`. |
| `full:GH200/<test>` | `GH200` stages, e.g., `GH200-PyTorch-Post-Merge-1`. |
| `full:A100X/<test>` | `A100X` stages, e.g., `A100X-PyTorch-1`. |
| `full:CPU/<test>` | Standard `CPU-Generic-x86-*` and `CPU-Generic-arm-*` stages (`l0_cpu.yml`, zero GPUs). |
| `full:sm90/<test>` | Exact SM90 detected by the collecting pytest process (e.g., `DGX_H100` and `GH200`). See limits below. |

- **Prefix matching:** `full:` supplies the prefix instead of prepending the
  current one. When the part before the first `-` in `--test-prefix` matches,
  the parser expands it to the full `--test-prefix`. Names are case-sensitive
  (`H100` is not `DGX_H100`; `A100` is not `A100X`). Copy the exact prefix from
  the invocation or [stage definitions](../../../jenkins/L0_Test.groovy). A bare
  `DGX_H100/<test>` without `full:` gets the current prefix prepended again.
- **Multiple platforms:** use one line per platform (no comma lists or `*`).
  Remove any unprefixed `SKIP` when narrowing scope, or it will keep skipping
  everywhere.
- **Substring replacement caveat:** expansion currently replaces **every
  occurrence** of the platform/SM token in the line (for example,
  `full:CPU/test_example.py::test_device[CPU]` also rewrites `[CPU]` and can
  stop matching; repeated `smXX` strings behave the same way).
- **SM detection limits:** `full:sm<N>/` matches only exact `get_sm_version()`
  equality (not a minimum or range), queried from
  `torch.cuda.get_device_properties(0)` in the collecting pytest process (device
  0 only; inner pytest needs its own CUDA visibility; stage names and YAML
  conditions do not determine it). When CUDA is unavailable or `get_device_properties(0)`
  raises `RuntimeError`, it returns `0`; other import/detection errors are caught
  by the parser. On mismatch or failure, `sm90/` stays literal and silently does
  not match (no fallback or error). Use `full:CPU/` for CPU stages.

### Verification and implementation pointers

Check the effective `--waives-file` (including target-branch waives). Use
`--collect-only` to discover exact node IDs and verify the skip status/reason in
the actual run.

- [Jenkins `mergeWaiveList`](../../../jenkins/L0_MergeRequest.groovy) and
  [`mergeWaiveList.py`](../../../jenkins/scripts/mergeWaiveList.py): PR/TOT
  merging.
- [`parse_test_list_lines` / `apply_waives`](../defs/test_list_parser.py):
  parsing, prefixes, and exact outer matching.
- [`test_unittests_v2`](../defs/test_unittests.py) /
  [`apply_waives_ut`](../../unittest/conftest.py): inner pytest invocation,
  waive forwarding, and inner matching.
- [`get_sm_version`](../defs/conftest.py): SM detection.

When changing waive merging, matching, or prefix handling, update this reference
and [`AGENTS.md`](AGENTS.md) in the same change (and the
[CI overview](../../../docs/source/developer-guide/ci-overview.md#waiving-tests)
if its summary or entry point changes).

## Batching and re-grouping reference

Use this checklist when consolidating, renaming, or optimizing `unittest/`
entries. Routine single-test additions only need the rules in
[`AGENTS.md`](AGENTS.md).

### Collection, isolation, and worker policy

- **Compare real collections** under the stage's marker, exclusions, and
  waivers: check exact node IDs, multiplicity, and skip/xfail markers, not just
  total counts. Call out any deliberate case removal separately.
- **Check execution compatibility:** run the proposed batch to catch shared
  fixture, event-loop, or environment-variable interactions. Matching
  collections do not prove runtime isolation.
- **Recheck effective parallelism:** the
  [unittest bridge](../defs/test_unittests.py) looks up `(gpu_name, normalized_entry)`
  in `agg_unit_mem_df.csv` (capped at 8 workers, defaulting unmatched entries to
  serial) and honors `LLM_TEST_PARALLEL_OVERRIDE`. Do not assume a regrouped
  entry inherits an old worker count or is safe at a smaller entry's factor.
- **Check CBTS and downstream keys:** inspect the generated filtered test-db
  (stages, retained tests, split counts, and any domain rules tied to explicit
  registrations). Renaming entries can also change report, waiver, and duration
  keys — recheck references, coverage-history fallback, and measured shard
  balance (summing old per-function startup times does not predict the merged
  batch).

### GPU batch validation

- Run the proposed batch on representative target GPU types/counts with the
  intended worker and rank counts. Check peak device memory and state left
  between cases (CUDA errors, live allocations, process groups). Existing
  cleanup fixtures do not prove isolation:
  [`torch.cuda.empty_cache()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html)
  releases unused allocator cache, not live tensors or arbitrary process state.
- Preserve distributed launchers, rank-to-device mapping, and collective
  cleanup. A plain `unittest/` entry does not replace a multi-rank wrapper. Keep
  process boundaries where GPU state or memory makes sharing unsafe.

### Measuring batching changes fairly

- Keep selected cases, inputs, allocated resources, and worker counts fixed when
  measuring batching alone, and note cold vs. warm cache conditions. Report
  worker-policy changes and intentional coverage pruning separately from
  batching gains.
- Include setup/teardown cost and report first-attempt failures separately from
  retry outcomes (a parallel failure followed by a serial retry pass does not
  validate the parallel batch).
- Report batch wall time, allocated GPU-seconds, and slowest shard alongside the
  [unit-test measurement rules](../../unittest/AGENTS.md#coverage-and-execution-cost);
  fewer entries alone does not prove lower GPU time.
