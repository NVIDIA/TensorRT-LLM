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

# Continuous Integration Overview

This page explains how TensorRT‑LLM's CI is organized and how individual tests map to Jenkins stages. Most stages execute integration tests defined in YAML files, while unit tests run as part of a merge‑request pipeline. The sections below describe how to locate a test and trigger the stage that runs it.

## Table of Contents
1. [CI pipelines](#ci-pipelines)
2. [Test definitions](#test-definitions)
3. [Unit tests](#unit-tests)
4. [Jenkins stage names](#jenkins-stage-names)
5. [Finding the stage for a test](#finding-the-stage-for-a-test)
6. [Waiving tests](#waiving-tests)
7. [Multi-GPU Tests](#multi-gpu-tests)
8. [Triggering CI Best Practices](#triggering-ci-best-practices)

## CI pipelines

Pull requests do not start testing by themselves. Developers trigger the CI by commenting `/bot run` (optionally with arguments) on the pull request (see [Pull Request Template](source:.github/pull_request_template.md) for more details). That kicks off the **merge-request pipeline** (defined in `jenkins/L0_MergeRequest.groovy`), which runs unit tests and integration tests whose YAML entries specify `stage: pre_merge`. Once a pull request is merged, the **post-merge pipeline** runs the same `jenkins/L0_MergeRequest.groovy` as a separate Jenkins job. It runs the pre-merge stages plus the multi-GPU stages and every stage with `Post-Merge` in its name. In both pipelines, the test stages run in child jobs defined by `jenkins/L0_Test.groovy`, which maps each stage name to a test-db YAML file.

`stage` tags live in the YAML files under `tests/integration/test_lists/test-db/`. Searching those files for `stage: pre_merge` shows exactly which tests the merge-request pipeline covers.

## Test definitions

Integration tests are listed under `tests/integration/test_lists/test-db/`. Most YAML files are named after the GPU or configuration they run on (for example `l0_a100.yml`). Some files, like `l0_sanity_check.yml`, use wildcards and can run on multiple hardware types. Entries contain conditions and a list of tests. Two important terms in each entry are:

- `stage`: either `pre_merge` or `post_merge`.
- `backend`: for example `pytorch`, `cpp`, or `fmha`. Grep the YAML files for `backend:` to see the values currently in use.

Example from `l0_a30.yml`:

```yaml
      terms:
        stage: pre_merge
        backend: pytorch
  tests:
  - unittest/_torch/sampler/test_beam_search.py
```

## Unit tests

Unit tests live under `tests/unittest/`. They are listed in the same test-db YAML files as integration tests, as entries starting with `unittest/` (see the `l0_a30.yml` example above), and run in the `jenkins/L0_Test.groovy` stages mapped to those files, like any other test.

## Jenkins stage names

`jenkins/L0_Test.groovy` maps stage names to these YAML files.  For A100 the mapping includes:

```groovy
    "A100X-PyTorch-1": ["a100x", "l0_a100", 1, 1],
    "A100X-PyTorch-Post-Merge-1": ["a100x", "l0_a100", 1, 1],
```

The array elements are: GPU type, YAML file (without extension), shard index, and total number of shards. Only tests with `stage: post_merge` from that YAML file are selected when a `Post-Merge` stage runs.

Stage names are unbracketed, with `Post-Merge` embedded directly in the name. Copy the name exactly as it appears in the Groovy map. Jenkins `stage_list` matches names without a `*` wildcard exactly, so a stale spelling such as `A100X-Triton-[Post-Merge]-1` silently selects nothing. The mapping script's `--stages` option is stricter: it reports unknown names and near-match suggestions on stderr.

## Finding the stage for a test

1. Locate the test in the appropriate YAML file under `tests/integration/test_lists/test-db/` and note its `stage` and `backend` values.
2. Search `jenkins/L0_Test.groovy` for a stage whose YAML file matches (for example `l0_a100`) and whose name contains `Post-Merge` if the YAML entry uses `stage: post_merge`.
3. The resulting stage name(s) are what you pass to Jenkins via the `stage_list` parameter when triggering a job.

### Using `test_to_stage_mapping.py`

Manually searching YAML and Groovy files can be tedious.  The helper script
`scripts/test_to_stage_mapping.py` automates the lookup:

```bash
python scripts/test_to_stage_mapping.py --tests "unittest/_torch/sampler/test_beam_search.py"
python scripts/test_to_stage_mapping.py --tests test_beam_search
python scripts/test_to_stage_mapping.py --stages A100X-PyTorch-Post-Merge-1
python scripts/test_to_stage_mapping.py --test-list my_tests.txt
python scripts/test_to_stage_mapping.py --test-list my_tests.yml
```

The first two commands print the Jenkins stages that run the specified tests or
patterns. Patterns are matched by substring, so partial test names are
supported out of the box. The third lists every test executed in the given stage. When
providing tests on the command line, quote each test string so the shell does
not interpret the `[` and `]` characters as globs. Alternatively, store the
tests in a newline‑separated text file or a YAML list and supply it with
`--test-list`.


To run the same tests on your pull request, comment:

```bash
/bot run --stage-list "A100X-PyTorch-Post-Merge-1"
```

This executes the same tests that run post-merge for this hardware/backend.


## Waiving tests

For known failures, add a line to `tests/integration/test_lists/waives.txt` rather
than removing the test from its YAML list. Use `<test> SKIP (reason)`, with an
explicit `SKIP` and a tracking bug or brief explanation. A matching waive skips
the test body.

In PR CI, Jenkins normally merges the PR list with the latest target-branch
list (tip of tree, or TOT), then applies exact-line removals from the PR diff.
Pytest receives this effective list via `--waives-file`: main TOT waives can
therefore affect PRs targeting main even when absent from the PR checkout.

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
Replace the illustrative names and reasons below with actual tests and bugs.

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
| `full:H100_PCIe/<test>` | `H100_PCIe` stages, e.g. `H100_PCIe-CPP-1`. |
| `full:A100X/<test>` | `A100X` stages, e.g. `A100X-PyTorch-1`. |
| `full:CPU/<test>` | Standard `CPU-Generic-x86-*` and `CPU-Generic-arm-*` stages (`l0_cpu.yml`, zero GPUs), without GPU detection. |
| `full:sm90/<test>` | Exact SM90 detected by the collecting pytest process; can cover both `DGX_H100` and `H100_PCIe`. See detection limits below. |

`full:` supplies a prefix instead of having one prepended. For a matching
platform, the parser expands the part before the first `-` in `--test-prefix`
to that full prefix. Names are case-sensitive: `H100` is not an alias for
`DGX_H100` or `H100_PCIe`, nor is `A100` an alias for `A100X`. Copy the name from
the actual invocation or [stage definitions](source:jenkins/L0_Test.groovy),
not the GPU model or test-db filename. A bare `DGX_H100/<test>` without `full:`
gets the current prefix prepended again.

For several platforms, use one line each, such as `full:A100X/<test>` and
`full:DGX_H100/<test>`; these exclude `H100_PCIe`. Comma lists and `*` are not
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

- Use a case/function waive for isolated failures. For widespread unit-test
  failures, a temporary whole-entry waive can restore CI more efficiently,
  but also skips unaffected tests in that entry. Track the bugs, prioritize
  fixes, and remove the waive after verifying them to restore coverage.
- Verify the intended entry/case and platform against the run's effective
  `--waives-file`, including target-branch waives. `--collect-only` helps identify
  names but does not prove skipping: check the run's skip status and reason.

Implementation references:

- [Jenkins `mergeWaiveList`](source:jenkins/L0_MergeRequest.groovy) and
  [`mergeWaiveList.py`](source:jenkins/scripts/mergeWaiveList.py): PR/TOT merging.
- [`parse_test_list_lines` / `apply_waives`](source:tests/integration/defs/test_list_parser.py):
  parsing, prefixes, and exact outer matching.
- [`test_unittests_v2`](source:tests/integration/defs/test_unittests.py) /
  [`apply_waives_ut`](source:tests/unittest/conftest.py): inner pytest invocation,
  waive-file forwarding, and inner matching.
- [`get_sm_version`](source:tests/integration/defs/conftest.py): SM detection.

When changing waive merging, matching, or prefix handling, update this section
and [`tests/integration/test_lists/AGENTS.md`](source:tests/integration/test_lists/AGENTS.md)
in the same change.

## Multi-GPU Tests

Running multi-GPU tests (`--add-multi-gpu-test`, `--only-multi-gpu-test`) requires the
`ci: full pre-merge approved` label on the PR. Without this label the
`[Test-x86_64-Multi-GPU] Remote Run` and `[Test-SBSA-Multi-GPU] Remote Run` stages will
fail with an explanatory error.

To obtain the label, ask a member of `NVIDIA/trt-llm-ci-approvers` to apply it. Only
members of that team can add the label; unauthorized additions are automatically removed
by the `Guard Full Pre-Merge Approval Label` GitHub Actions workflow.

Once the label is present, re-trigger CI with the same command you used
originally. For example:

```bash
# Run the normal pipeline plus multi-GPU stages
/bot run --add-multi-gpu-test

# Run only multi-GPU stages
/bot run --only-multi-gpu-test
```

Post-merge pipelines and GitLab MR builds are exempt from this label check.

## Triggering CI Best Practices

### Triggering Post-merge tests

Full `/bot run --post-merge` runs require the `ci: post-merge approved` PR
label because they can consume substantial shared GPU resources. The label is
intended to be applied by an active member of the
`NVIDIA/trt-llm-ci-approvers` GitHub team. A GitHub workflow validates the label
actor and normally removes invalid approvals. This is a best-effort resource
governance guard, not a strict authorization boundary. The label remains in
place when new commits are pushed and can be removed manually when the approval
no longer applies.

When you only need to verify a handful of post-merge tests, specify exactly
which stages to run:

```bash
/bot run --stage-list "stage-A,stage-B"
```

This runs **only** the stages listed. You can also add stages on top of the
default pre-merge set:

```bash
/bot run --extra-stage "stage-A,stage-B"
```

Both options accept stage names and wildcard patterns defined in
`jenkins/L0_Test.groovy`. Explicit Post-merge stage test doesn't need
`ci: post-merge approved` PR label to run. However, the `"*"`,
`"*Post-Merge*"`, and `"*PerfSanity*"` selectors require the same approval
label, including when they appear in a comma-separated list. Equivalent escaped
or repeated-star forms are treated the same. Other stage selectors, including
explicit stage names and other limited wildcard patterns, retain their existing
behavior.

Being selective keeps CI turnaround fast and conserves hardware resources.

### Avoiding unnecessary `--disable-fail-fast` usage

Avoid habitually using `--disable-fail-fast` as it wastes scarce hardware resources. The CI system automatically reuses successful test stages when commits remain unchanged, and subsequent `/bot run` commands only retry failed stages. Overusing `--disable-fail-fast` keeps failed pipelines consuming resources (like DGX-H100s), increasing queue backlogs and reducing team efficiency.
