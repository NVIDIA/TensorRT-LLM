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

Sometimes a test is known to fail due to a bug or unsupported feature. Instead
of removing it from the YAML test lists, add the test name to
`tests/integration/test_lists/waives.txt`. CI passes this file to pytest via
`--waives-file`. A matching entry with a `SKIP` marker prevents the test body
from running. Always include `SKIP (reason)` explicitly.

Use one waive per line, with a bug link or a brief explanation in parentheses:

```text
accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype[False] SKIP (https://nvbugs/6120535)
```

Blank lines are ignored. `#` starts a comment anywhere on a line, even inside
quotes or a reason. Use tracking URLs without `#fragment` and avoid `#` in test
names or `-k` expressions. A URL fragment can truncate a reason before its closing
parenthesis and make the whole waive file fail to parse.

The examples below use illustrative test names and reasons; replace them with
the actual test and tracking reference.
These rules describe `SKIP` waives.

### Test names and test-list metadata

For both ordinary integration tests and `unittest/` entries, copy the test path
and command-line options into the waive name, but omit test-list metadata such
as `ISOLATION` and `TIMEOUT (N)`, along with any YAML comment. Jenkins removes
`ISOLATION` before collection, while the test-list parser handles `TIMEOUT (N)`
as timeout metadata. Neither belongs in the collected node ID.

Do not append `SKIP` to a copied entry that still contains `TIMEOUT (N)`: the
waive parser then retains the timeout text in the name, so it will not match.
It also does not remove `ISOLATION` from waive names. For example, these YAML
entries:

```yaml
- accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_fp8_block_scales[latency] TIMEOUT (90) ISOLATION
- unittest/executor/test_base_worker.py ISOLATION
```

need these waives, without the test-list metadata:

```text
accuracy/test_llm_api_pytorch.py::TestQwen3_8B::test_fp8_block_scales[latency] SKIP (Known issue in this test)
unittest/executor/test_base_worker.py SKIP (Known issue in this file)
```

### Two matching layers for unit tests

A test-db entry beginning with `unittest/` launches an integration wrapper,
`test_unittests.py::test_unittests_v2`, which starts a separate pytest run for the
selected unit tests. The waive file is applied at both layers:

1. **Outer integration test:** an exact match skips the entire wrapper entry,
   before it launches the inner pytest run.
2. **Inner unit tests:** if the wrapper runs, waives can skip individual collected
   tests, methods/functions, classes, or files. Directory matching is not supported.

The inner matching rules apply only to unit tests. Ordinary integration test
waives use exact node ID matching; they do not inherit the inner unittest
file/class/function matching described below.

### Outer layer: match the complete test-db entry

The parser converts a `unittest/` entry into a wrapper node ID, preserving the
test path and command-line options such as `-k`, `-m`, and `--ignore` inside the
brackets:

```text
unittest/_torch/modeling
  -> test_unittests.py::test_unittests_v2[unittest/_torch/modeling]

unittest/_torch/modeling -k "modeling_llama"
  -> test_unittests.py::test_unittests_v2[unittest/_torch/modeling -k "modeling_llama"]
```

The outer waive lookup compares the complete node IDs. It does not interpret
the waive name as a directory prefix, glob, or regular expression, and it does
not normalize equivalent shell arguments. Copy the entry exactly, including
option order, internal spaces, and quotes, after omitting test-list metadata as
described above.

For this waive:

```text
unittest/_torch/modeling SKIP (Known modeling issue)
```

| Separate test-db entry | Entire entry skipped by this waive? | Reason |
| --- | --- | --- |
| `unittest/_torch/modeling` | Yes | The complete entry matches. |
| `unittest/_torch/modeling -k "modeling_llama"` | No | The `-k` option is part of a different entry. |
| `unittest/_torch/modeling -k "modeling_gemma3"` | No | The `-k` option is part of a different entry. |
| `unittest/_torch/modeling/test_modeling_llama.py` | No | Its wrapper parameter is the file path, which does not equal the waived directory path. |

Each row is a separate wrapper task. This waive skips the first task as a whole;
the file and `-k` tasks do not match it, even though they select tests from the
same directory.
To skip several filtered entries as whole tasks, list each complete entry:

```text
unittest/_torch/modeling -k "modeling_llama" SKIP (Known modeling issue)
unittest/_torch/modeling -k "modeling_gemma3" SKIP (Known modeling issue)
unittest/_torch/modeling -k "modeling_gpt_oss" SKIP (Known modeling issue)
unittest/_torch/modeling -k "modeling_gemma4" SKIP (Known modeling issue)
unittest/_torch/modeling -k "modeling_whisper" SKIP (Known modeling issue)
```

### Inner layer: match a case, function, class, or file

For an illustrative collected unittest node ID:

```text
unittest/_torch/modeling/test_example.py::TestExample::test_forward[fp16]
```

the inner matcher tries the following names in order. Append `SKIP (reason)` to
the desired name in `waives.txt`:

| Priority | Waive name | Scope |
| --- | --- | --- |
| 1 | `unittest/_torch/modeling/test_example.py::TestExample::test_forward[fp16]` | This parameter instance only. |
| 2 | `unittest/_torch/modeling/test_example.py::TestExample::test_forward` | All parameter instances of this method. |
| 3 | `unittest/_torch/modeling/test_example.py::TestExample` | All collected tests in this class. |
| 4 | `unittest/_torch/modeling/test_example.py` | All collected tests in this file. |
| No inner match | `unittest/_torch/modeling` | A directory path cannot match the collected case shown above. |

The first matching name wins. The matcher first checks the complete node ID,
then removes the parameter suffix beginning with `[`, then repeatedly removes
the last `::` component. It stops at the file path; it never walks up directory
components. A standalone function follows the same rule without the class step.
Globs, regular expressions, and partial parameter names are not supported.
To skip an entire directory invocation, use the outer entry rule instead.

A file-level waive applies even when that file is reached through a different
outer entry. For example:

```text
unittest/_torch/modeling/test_example.py SKIP (Known issue in this file)
```

If an outer entry such as `unittest/_torch/modeling -k "example"` collects tests
from that file, those inner tests are skipped. Other selected files are unaffected
by this waive. Use a method or parameter-instance waive to keep the rest of the
file running. Conversely, an inner waive cannot restore tests whose entire outer
entry was already skipped.

### Platform scope: all platforms, one platform, or one SM version

These rules apply to the shared integration/unittest runner that receives
`--waives-file`, not automatically to every Jenkins stage or independent test
runner. Choose the platform scope separately from the test scope. In the table below,
`<test>` is a complete integration node ID, a complete `unittest/` entry, or an
inner unittest name at one of the supported levels above.

CI supplies `--test-prefix=<stage-name>` to the outer pytest run and forwards it
as `<stage-name>/unittest` to the inner unittest run. The table below assumes
this CI setup. Unprefixed integration waives also work in a local run without
`--test-prefix`.

| Waive form | Platform scope |
| --- | --- |
| `<test> SKIP (reason)` | All platforms where the test matches and the waive file is supplied. |
| `full:DGX_H100/<test> SKIP (reason)` | The `DGX_H100` test-prefix namespace. |
| `full:H100_PCIe/<test> SKIP (reason)` | The `H100_PCIe` test-prefix namespace. |
| `full:CPU/<test> SKIP (reason)` | The `CPU` test-prefix namespace, without querying a GPU. |
| `full:sm90/<test> SKIP (reason)` | Runs where the current pytest process can detect SM90 on its visible CUDA device 0. |

**All platforms:** omit `full:` and the platform name. The parser prepends the
current `--test-prefix`, if set, so the same waive follows the test across CI
platforms. For example, this skips the complete filtered entry wherever it runs:

```text
unittest/_torch/modeling -k "modeling_llama" SKIP (Known issue on all platforms)
```

**One platform:** add `full:<platform>/` using the actual CI test-prefix name:

```text
full:DGX_H100/unittest/_torch/modeling -k "modeling_llama" SKIP (Known issue on this platform)
```

`full:` means the name already supplies a prefix. For a matching platform, the
parser expands the platform name to the current test prefix, including its
stage suffix. The platform component is the part before the first `-` in
`--test-prefix`: use `DGX_H100` from `DGX_H100-PyTorch-1`, not the whole
stage-qualified prefix. With `--test-prefix=DGX_H100-PyTorch-1`, both the
unprefixed waive and the `full:DGX_H100/` waive above produce:

```text
DGX_H100-PyTorch-1/test_unittests.py::test_unittests_v2[unittest/_torch/modeling -k "modeling_llama"]
```

Expansion currently replaces every occurrence of the platform string in the name.
If it also appears in a path or parameter ID (for example,
`full:CPU/test_example.py::test_device[CPU]`), that text is rewritten too and can
prevent a match. SM-qualified waives have the same limitation for repeated
`smXX` strings.

With `--test-prefix=A100X-PyTorch-1`, the unprefixed waive follows that prefix,
while `full:DGX_H100/` does not match. Platform names are case-sensitive strings
from the test prefix, not GPU-name aliases: `full:H100/` does not match
`DGX_H100-PyTorch-1` or `H100_PCIe-CPP-1`, and `full:A100/` does not match
`A100X-PyTorch-1`. Copy the platform component from the actual invocation or the
[stage definitions](source:jenkins/L0_Test.groovy), not the GPU model name or
test-db YAML filename.

**Several selected platforms:** add a separate prefixed line for each platform:

```text
full:A100X/unittest/_torch/modeling -k "modeling_llama" SKIP (Known issue on these platforms)
full:DGX_H100/unittest/_torch/modeling -k "modeling_llama" SKIP (Known issue on these platforms)
```

These lines do not include `H100_PCIe`. Comma-separated platform lists and `*`
wildcards are not expanded. Do not also retain an unprefixed `SKIP` for the same
test when the intended scope is limited to these platforms: that line would
still skip the matching test on other platforms.

**One SM version across platforms:** use `full:sm<version>/`:

```text
full:sm90/unittest/_torch/modeling -k "modeling_llama" SKIP (Known issue on SM90)
```

When SM detection succeeds and returns 90, the parser replaces `sm90` with the
current test prefix. This can cover both `DGX_H100` and `H100_PCIe` runs without
listing each name, provided their pytest processes can query the device. It is
an exact SM-version match, not a minimum version or a wildcard.

SM detection uses `get_sm_version()` in the integration conftest. It queries
`torch.cuda.get_device_properties(0)` in the current pytest process and returns
`major * 10 + minor`. It does not infer SM from the stage name, a YAML platform
condition, or a remote inference server. Device 0 means the first CUDA device
visible to that process; this is not a check of every GPU in a multi-GPU stage.
Waives are applied by `pytest_collection_modifyitems`, before test bodies run.
The relevant GPU visibility is therefore that of the pytest process collecting
the tests, including the inner pytest process for unittest waives. A GPU used
later by a model worker or remote server does not establish that visibility.

| Runtime condition | Detection result | Effect of `full:sm90/` with a normal platform prefix |
| --- | --- | --- |
| Visible device 0 reports compute capability 9.0 | `90` | Expands to the current prefix and can match the test. |
| `torch.cuda.is_available()` is false, including a CPU-only environment | `0` | Does not match. |
| `get_device_properties(0)` raises `RuntimeError` | `0` | Does not match. |
| Importing or calling the helper raises another exception | The waive parser catches the exception. | Does not match. |

When the SM does not match or detection raises, the parser leaves the literal
`sm90/` prefix unchanged. It does not fall back to an all-platform waive or raise
an error for the unmatched waive. Therefore, SM waives are conditional on runtime
GPU visibility; they are not a guarantee that every stage can determine its SM.

**CPU stages:** the standard `CPU-Generic-x86-*` and `CPU-Generic-arm-*` stages
use `l0_cpu.yml` (zero GPUs) and pass their stage names as `--test-prefix`.
In a CPU-only process, `get_sm_version()` returns `0`, not a real GPU SM version.
To waive a test on these CPU stages, use a platform prefix:

```text
full:CPU/unittest/_torch/modeling SKIP (Known CPU-only issue)
```

This matches the whole `unittest/_torch/modeling` entry in `l0_cpu.yml`; it does
not recursively cover other entries or variants with `-k` options. `full:CPU/`
also works with the supported inner unittest names. A `cpu_only` test marker
alone does not determine SM; the helper queries the process's actual CUDA
visibility.

The same platform syntax applies to individual unit tests. For example, this
skips one illustrative parameter instance only in the `DGX_H100` namespace:

```text
full:DGX_H100/unittest/_torch/modeling/test_example.py::TestExample::test_forward[fp16] SKIP (Known issue on this platform)
```

Use `full:DGX_H100/`, not a bare `DGX_H100/`: without `full:`, the parser treats the
entire name as unprefixed and prepends the current prefix again. Platform scoping
does not change the outer exact-entry or inner unittest matching rules.

### Checking a waive

- Decide whether to skip a whole test-db entry or only a file/class/function/case
  inside it. Prefer the narrowest scope that covers the known failure.
- For a whole entry, copy the test path and all command-line options from the
  YAML value after `- `, omitting metadata such as `ISOLATION`, `TIMEOUT (N)`,
  and YAML comments before appending `SKIP (reason)`.
  Changing the command-line options can stop an existing waive from matching.
- For an inner unittest, use the actual collected node ID, including parameter
  IDs when needed. Use `unittest/` paths as in the test lists, not repository-root
  `tests/unittest/` paths. Check any `full:` hardware restriction as well.
- Verify that the invocation receives `--waives-file`. Collection output helps
  identify names, but `--collect-only` does not prove that a test body will be
  skipped. Check the run's skip status and reason for the intended entry or case.
- A successful stage lookup with `test_to_stage_mapping.py` does not prove a
  waive match: that helper supports substring search, while the outer waive
  matcher requires an exact node ID. Other waives and pytest skip conditions can
  also affect the final result.

Implementation references:

- [`parse_test_list_lines` and `apply_waives`](source:tests/integration/defs/test_list_parser.py)
  parse entries and apply exact outer matches.
- [`test_unittests_v2`](source:tests/integration/defs/test_unittests.py)
  launches the inner pytest run and forwards the waive file.
- [`apply_waives_ut`](source:tests/unittest/conftest.py)
  applies inner unittest matches in the order above.
- [`get_sm_version`](source:tests/integration/defs/conftest.py)
  queries visible device 0 and returns `0` when CUDA is unavailable or its device
  property query raises `RuntimeError`.

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
