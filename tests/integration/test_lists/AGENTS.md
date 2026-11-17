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

# AGENTS.md — test lists

Scope: this directory and its descendants, including `waives.txt` and `test-db/`.
Supplements the repository-root and `tests/AGENTS.md` guidance.

## Waive list

For failing tests that should be retained and fixed, add a line to
`tests/integration/test_lists/waives.txt` and keep their YAML registrations.
Use `<test> SKIP (reason)`, with an explicit `SKIP` and a tracking bug or brief
explanation. A matching waive skips the test body. If review instead establishes
that a test is low-value or redundant and should be retired, remove the test,
its registrations and obsolete waives together, explaining the retained or
deliberately relinquished coverage in the PR.

In PR CI, Jenkins normally merges the PR list with the latest target-branch
list (tip of tree, or TOT), then applies exact-line removals from the PR diff.
Pytest receives this effective list via `--waives-file`: waives from the PR's
target branch can therefore affect the run even when absent from the PR
checkout, regardless of the target branch's name.

### Writing a waive

Copy the test path and command-line options, omitting `ISOLATION`, `TIMEOUT (N)`,
and YAML comments. This metadata is not part of the collected node ID; leaving
it in a waive name prevents a match. This applies to integration and `unittest/`
entries alike:

```yaml
- accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_fp8_block_scales[latency] TIMEOUT (90) ISOLATION
- unittest/executor/test_base_worker.py ISOLATION
```

```text
accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_fp8_block_scales[latency] SKIP (Known issue in this test)
unittest/executor/test_base_worker.py SKIP (Known issue in this file)
```

Use one waive per line; blank lines are ignored. `#` starts a comment anywhere,
including inside quotes or reasons. `-k "modeling_llama"` is valid, but
`-k "case#tag"` is truncated. Use tracking URLs without fragments: a `#` can
remove the reason's closing parenthesis and break parsing of the whole file.
Replace illustrative names and reasons with actual tests and bugs.

### Matching: outer entries and inner unit tests

Ordinary integration tests match exact node IDs. A `unittest/` test-db entry
has two layers: an integration wrapper, then a separate pytest run for its
selected unit tests. Waives apply during `pytest_collection_modifyitems` at
each layer, before test bodies run.

**Outer entry:** `unittest/<entry>` becomes
`test_unittests.py::test_unittests_v2[unittest/<entry>]`. The full entry includes
options such as `-k`, `-m`, and `--ignore`: copy their order, internal spaces,
and quotes exactly. Equivalent shell arguments are not normalized. Changing
them requires rechecking the waive. An exact match skips the wrapper and its
entire inner run; inner waives cannot restore it.

For `unittest/_torch/modeling SKIP (Known modeling issue)`:

| Separate test-db entry | Entire entry skipped? |
| --- | --- |
| `unittest/_torch/modeling` | Yes: exact entry. |
| `unittest/_torch/modeling -k "modeling_llama"` | No: the options make it a different entry. |
| `unittest/_torch/modeling/test_modeling_llama.py` | No: the file path is a different entry. |

This is not a directory-prefix match. To skip several filtered entries, list
each complete entry with its own `SKIP (reason)`.

**Inner unittest:** if the wrapper runs, the matcher checks these names in order:

| Priority | Waive name (append `SKIP (reason)`) | Scope |
| --- | --- | --- |
| 1 | `unittest/test_example.py::TestExample::test_forward[fp16]` | One parameter instance. |
| 2 | `unittest/test_example.py::TestExample::test_forward` | All parameter instances of the method. |
| 3 | `unittest/test_example.py::TestExample` | All collected tests in the class. |
| 4 | `unittest/test_example.py` | All collected tests in the file. |

The first matching name wins: check the full node ID, remove the parameter
suffix starting at `[`, then remove the last `::` component repeatedly. A
standalone function has no class step. Matching stops at the file; the current
inner matcher does not walk directories. A directory invocation can only be
waived as an exact outer entry. Neither layer supports globs, regular
expressions, or partial parameter names; ordinary integration tests do not
inherit the inner file/class/function matching.

A file waive also skips its tests when collected through another outer entry
(for example, one with `-k`), leaving other selected files unaffected. Use a
method or case waive to keep the rest of that file running. Use collected node
IDs and `unittest/` paths, not repository-root `tests/unittest/` paths.

### Platform scope

Scope applies to runners receiving `--waives-file`, not automatically to every
Jenkins stage or independent runner. CI passes `--test-prefix=<stage-name>` to
outer pytest and `<stage-name>/unittest` to inner pytest. The forms below assume
that setup. `<test>` can be an exact integration/outer entry or a supported
inner unittest name; platform scope does not change its matching rules.

| Waive form (append `SKIP (reason)`) | Scope |
| --- | --- |
| `<test>` | All platforms where the test matches and the waive file is supplied. The current prefix is added if set; unprefixed integration waives also work locally without `--test-prefix`. |
| `full:DGX_H100/<test>` | `DGX_H100` stages, e.g. `DGX_H100-PyTorch-1`. |
| `full:GH200/<test>` | `GH200` stages, e.g. `GH200-PyTorch-Post-Merge-1`. |
| `full:A100X/<test>` | `A100X` stages, e.g. `A100X-PyTorch-1`. |
| `full:CPU/<test>` | Standard `CPU-Generic-x86-*` and `CPU-Generic-arm-*` stages (`l0_cpu.yml`, zero GPUs), without GPU detection. |
| `full:sm90/<test>` | Exact SM90 detected by the collecting pytest process; can cover both `DGX_H100` and `GH200`. See detection limits below. |

`full:` supplies a prefix instead of having one prepended. For a matching
platform, the parser expands the part before the first `-` in `--test-prefix`
to that full prefix. Names are case-sensitive: `H100` is not an alias for
`DGX_H100`, nor is `A100` an alias for `A100X`. Copy the name from
the actual invocation or [stage definitions](../../../jenkins/L0_Test.groovy),
not the GPU model or test-db filename. A bare `DGX_H100/<test>` without `full:`
gets the current prefix prepended again.

For several platforms, use one line each, such as `full:A100X/<test>` and
`full:DGX_H100/<test>`; these exclude `GH200`. Comma lists and `*` are not
expanded. Remove any unprefixed `SKIP` for that test when narrowing its scope,
or it will still skip on other platforms.

Expansion currently replaces **every occurrence** of the platform/SM string.
For example, `full:CPU/test_example.py::test_device[CPU]` also rewrites the
parameter ID and can stop matching; repeated `smXX` strings have the same issue.

**SM detection:** `full:sm<version>/` expands to the current test prefix only
when `get_sm_version()` equals that version, not a minimum version or range. It queries
`torch.cuda.get_device_properties(0)` in the collecting pytest process and
returns `major * 10 + minor`. Device 0 is that process's first visible CUDA
device, not every GPU in a multi-GPU stage. The inner pytest process needs its
own visibility. Stage names, YAML conditions, `cpu_only` markers, and GPUs used
later by model workers or remote servers do not establish the detected SM.

- CUDA unavailable (including CPU-only processes), or `get_device_properties(0)`
  raising `RuntimeError`: the helper returns `0`, not a real GPU SM version.
- Other import/detection exceptions: the waive parser catches them.
- On mismatch or detection failure, `sm90/` stays literal under a normal platform
  prefix, so the waive does not match. There is no all-platform fallback or
  unmatched-waive error. SM detection is not guaranteed in every stage;
  use `full:CPU/` for standard CPU stages.

### Choosing and checking a waive

- Self-review every added or broadened waive before submitting the PR. Identify
  the observed failure, tracking bug or reason, intended cases and platforms,
  and the coverage being disabled. For a batch of waives, check each entry;
  do not copy a broad list just to make CI pass.
- Choose scope based on the failure and the cost of maintaining the waive.
  Case/function or platform-specific waives suit isolated failures. Widespread
  issues can justify a temporary whole-entry, file/class, multi-platform or
  unprefixed all-platform waive rather than many narrow entries. Use the matching
  rules above to choose outer versus inner scope; broader waives also disable
  unaffected coverage, so explain that tradeoff. Track the bugs, prioritize
  fixes, and narrow or remove waives as fixes are verified.
- Verify the intended entry/case and platform against the run's effective
  `--waives-file`, including target-branch waives. `--collect-only` helps identify
  names but does not prove skipping: check the run's skip status and reason.

Implementation references:

- [Jenkins `mergeWaiveList`](../../../jenkins/L0_MergeRequest.groovy) and
  [`mergeWaiveList.py`](../../../jenkins/scripts/mergeWaiveList.py): PR/TOT merging.
- [`parse_test_list_lines` / `apply_waives`](../defs/test_list_parser.py):
  parsing, prefixes, and exact outer matching.
- [`test_unittests_v2`](../defs/test_unittests.py) /
  [`apply_waives_ut`](../../unittest/conftest.py): inner pytest invocation,
  waive-file forwarding, and inner matching.
- [`get_sm_version`](../defs/conftest.py): SM detection.

When changing waive merging, matching, or prefix handling, update this section
in the same change. Update the [CI overview](../../../docs/source/developer-guide/ci-overview.md#waiving-tests)
when its summary or entry point changes.

## CPU test registration and batching

- Reuse a file or parent-directory entry when its actual marker/options already
  select the intended case. Adding a function entry alongside that parent repeats
  execution. Prefer one file entry for compatible functions from the same file
  when all of its CPU cases are intended to run.
- Keep intentional partial selectors and required `ISOLATION` boundaries. Before
  consolidating entries, compare real pytest collections under the stage's marker,
  exclusions and waivers: check node IDs, multiplicity and skip/xfail markers, not
  just total counts. Explain any deliberate case removal separately.
- Run the proposed batch to check fixture and process-state interactions, including
  event loops and environment variables. Retain separate processes where the tests
  require them; a matching collection does not prove execution compatibility.
- Check CBTS through the generated filtered test-db, including file/function
  changes and any domain rules that depend on explicit registrations. Confirm the
  intended stages, retained tests and split counts. Document any coarser selection
  granularity and its cost; a successful path lookup alone is not sufficient.
- Renaming entries or regrouping parameters can change report, waiver and duration
  keys. Recheck references and safe coverage-history fallback. Use measured batch
  times when evaluating shard balance; summing old per-function startup costs does
  not estimate the new batch correctly. Preserve the existing waiver rules above.

## Test coverage

When adding or reducing cases, follow the shared
[test-value and placement rules](../../AGENTS.md#test-value-and-execution-placement).
For unit-test-specific details, follow the
[unit-test coverage guidance](../../unittest/AGENTS.md#coverage-and-execution-cost).
Preserve useful coverage and efficient execution without fixing a maximum test
count or preventing justified new regression tests.
