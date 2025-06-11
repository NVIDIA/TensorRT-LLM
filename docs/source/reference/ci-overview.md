# Continuous Integration Overview

This page explains how TensorRT‑LLM's CI is organized and how individual tests map to Jenkins stages. Most stages execute integration tests defined in YAML files, while unit tests run as part of a merge‑request pipeline. The sections below describe how to locate a test and trigger the stage that runs it.

## Table of Contents
1. [CI pipelines](#ci-pipelines)
2. [Test definitions](#test-definitions)
3. [Unit tests](#unit-tests)
4. [Jenkins stage names](#jenkins-stage-names)
5. [Finding the stage for a test](#finding-the-stage-for-a-test)
6. [Triggering CI politely](#triggering-ci-politely)

## CI pipelines

Pull requests do not start testing by themselves. Developers trigger the CI by commenting `/bot run` (optionally with arguments) on the pull request. That kicks off the **merge-request pipeline**, which runs unit tests and integration tests whose YAML entries specify `stage: pre_merge`. Once a pull request is merged, a separate **post-merge pipeline** runs every test marked `post_merge` across all supported GPU configurations.

`stage` tags live in the YAML files under `tests/integration/test_lists/test-db/`. Searching those files for `stage: pre_merge` shows exactly which tests the merge-request pipeline covers.

## Test definitions

Integration tests are listed under `tests/integration/test_lists/test-db/`. Most YAML files are named after the GPU or configuration they run on (for example `l0_a100.yml`). Some files, like `l0_sanity_check.yml`, use wildcards and can run on multiple hardware types. Entries contain conditions and a list of tests. Two important terms in each entry are:

- `stage`: either `pre_merge` or `post_merge`.
- `backend`: `pytorch`, `tensorrt` or `triton`.

Example from `l0_a100.yml`:

```yaml
      terms:
        stage: post_merge
        backend: triton
  tests:
  - triton_server/test_triton.py::test_gpt_ib_ptuning[gpt-ib-ptuning]
```

## Unit tests

Unit tests live under `tests/unittest/` and run during the merge-request pipeline. They are invoked from `jenkins/L0_MergeRequest.groovy` and do not require mapping to specific hardware stages.

## Jenkins stage names

`jenkins/L0_Test.groovy` maps stage names to these YAML files.  For A100 the mapping includes:

```groovy
    "A100X-Triton-Python-[Post-Merge]-1": ["a100x", "l0_a100", 1, 2],
    "A100X-Triton-Python-[Post-Merge]-2": ["a100x", "l0_a100", 2, 2],
```

The array elements are: GPU type, YAML file (without extension), shard index, and total number of shards. Only tests with `stage: post_merge` from that YAML file are selected when a `Post-Merge` stage runs.

## Finding the stage for a test

1. Locate the test in the appropriate YAML file under `tests/integration/test_lists/test-db/` and note its `stage` and `backend` values.
2. Search `jenkins/L0_Test.groovy` for a stage whose YAML file matches (for example `l0_a100`) and whose name contains `[Post-Merge]` if the YAML entry uses `stage: post_merge`.
3. The resulting stage name(s) are what you pass to Jenkins via the `stage_list` parameter when triggering a job.

### Example

`triton_server/test_triton.py::test_gpt_ib_ptuning[gpt-ib-ptuning]` appears in `l0_a100.yml` under `stage: post_merge` and `backend: triton`.  The corresponding Jenkins stages are `A100X-Triton-Python-[Post-Merge]-1` and `A100X-Triton-Python-[Post-Merge]-2` (two shards).

To run the same tests on your pull request, comment:

```bash
/bot run --stage-list "A100X-Triton-Python-[Post-Merge]-1,A100X-Triton-Python-[Post-Merge]-2"
```

This executes the same tests that run post-merge for this hardware/backend.

## Triggering CI politely

When you only need to verify a handful of post-merge tests, avoid the heavy
`/bot run --post-merge` command. Instead, specify exactly which stages to run:

```bash
/bot run --stage-list "stage-A,stage-B"
```

This runs **only** the stages listed. You can also add stages on top of the
default pre-merge set:

```bash
/bot run --extra-stage "stage-A,stage-B"
```

Both options accept any stage name defined in `jenkins/L0_Test.groovy`. Being
selective keeps CI turnaround fast and conserves hardware resources.
