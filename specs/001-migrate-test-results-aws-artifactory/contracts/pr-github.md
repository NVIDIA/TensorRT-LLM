<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contract: PR_Github's Change

PR_Github's pipeline code and its scan step live outside this repository. This contract states what
their change must do (spec FR-012; research R10). It lands any time before urm is retired, and the
switch does not wait for it. Where PR_Github's redacted copies go is PR_Github's concern; this work
checks only that none lands on the new instance before the merge (FR-019).

## Until the change lands

- PR_Github works as it does today: it reads a failed run's `test-results/` on urm with its urm
  credential, publishes its redacted copies where it does today, fetches `rerun_report.html` from urm,
  and links the report there. Dual publishing keeps urm's `test-results/` complete for every run
  (FR-002).
- Before the switch merges, the CI maintainers confirm that those redacted copies are not on the new
  instance (research R13, item 2): otherwise they would overwrite the replicas of every failed run
  after cutover. FR-019 makes that a blocker, which the path change below would resolve.
- The switch pull request's own failed and passing runs show PR_Github working unchanged from urm's
  copies (FR-019).

## Reading a failed run's test results

- Lists `llm-artifacts/<job>/<build>/test-results/` on the new location, with
  `aws-artifactory-credentials`, and on urm (`sw-tensorrt-generic`), as FR-006 requires of CI's own
  steps, and merges the entries by name: during dual publishing nearly every name is on both, and a
  run split by a rollback, or one whose replica upload failed, has some files on urm only.
- Reads each name once by the read rule: the new location first, urm only on a 404, no other
  location after a 401 or 403, the SHA-1 check, and one retry without the redirect when the
  content-delivery network answers 403.
- Implements this in PR_Github's own tooling. It does not run the pull request's copy of
  `fetch_build_artifact.sh`, because PR_Github is the gate in front of the pull request's code and
  holds the publishing credential.
- When neither location has the run's test results, it behaves as it does today.

## Publishing redacted copies

- Path: a top-level directory of their own, never `llm-artifacts/`, in the repository and instance
  that PR_Github writes to today. Proposed:
  - `redacted/llm-artifacts/<job>/<build>/test-results/results-<stage>.tar.gz` for each stage;
  - `redacted/llm-artifacts/<job>/<build>/test-results.tar.gz` for the combined archive.
- A redacted copy's path never equals an original's on either instance, including after urm's
  `sw-tensorrt-llm-generic-local` repository moves to the new instance.
- The release bundle's query, the path on the edge site, and the `test_results_url` handed to
  `.github/workflows/l0-test.yml` follow the new path. `l0-test.yml` and the repository secret it
  downloads with do not change.
- Copies published before the change stay where they are.

## Passing runs

- Fetches `rerun_report.html` by the read rule with the publishing credential. A report absent from
  both locations is known at once, without today's two minutes of retries.
- The PR comment links the report in the native browser on the new instance
  (`https://artifactory.nvidia.com/ui/native/sw-tensorrt-llm-generic-local/<layout path>`), or
  directly on urm when only urm has it.

## Credentials

- `aws-artifactory-credentials`, made available to PR_Github's job by the owners of the credential
  store before the change lands. It never appears on a command line or in a log.

## Evidence when the change lands (FR-012)

- A failed run that published to the new location: PR_Github's log shows its reads served by the new
  location, the redacted copies under the new path, and `blossom-ci` set with the test summary; the
  run's originals on both locations keep the SHA-1s they had before PR_Github ran.
- A passing run that reran tests: the comment's link opens the report in the native browser after
  the company sign-in.
- A failed and a passing run that published to urm only, such as a release branch's or a pull
  request not rebased past the switch: PR_Github reads both through the fallback, and `blossom-ci` and
  the comment are as before (spec User Story 3).
