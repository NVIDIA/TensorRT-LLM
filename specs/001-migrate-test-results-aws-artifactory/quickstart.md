<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Quickstart: Validating the Test-Result Migration

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

Each step names the requirements it shows. Record dates, outcomes, and links to run logs in the
sign-off evidence, never a token, a password, or netrc content (spec FR-023).

## Prerequisites

1. The program has confirmed that the publishing identity's grant covers deploy and delete/overwrite
   under `test-results/`, `rerun_reports/`, and `cbts-coverage/` (FR-009), and has been asked for the
   wider read grant for people and told the volume (FR-016, FR-021; research R9).
2. The owners of the credential store have made `aws-artifactory-credentials` available to every job
   that writes, lists, or reads test results. PR_Github's job needs it only for PR_Github's own
   change (step 11).
3. This branch is rebased onto TRTLLM-17014's current pull-request head (research R17).
4. The test pods' `jnlp` images have `curl`, which pod stages use to delete their checkpoints: the
   first draft run showed it for amd64, and the arm64 image's history installs it (research R4).
5. Before the merge only: TRTLLM-17014 (PR #20494) has merged and passed its validation after
   cutover, and this branch is rebased onto main (FR-017).

## 1. Unit tests

Modules, all under directories that `l0_cpu.yml` (and, for `unittest/scripts`, `l0_a10.yml`)
already run:

- `tests/unittest/scripts/test_fetch_build_artifact.py`: list mode
  ([contracts/read-helper-list.md](contracts/read-helper-list.md)) and the exit-code names, now with
  `jenkins/L0_MergeRequest.groovy` and the retry read as readers.
- `tests/unittest/scripts/test_progress_snapshot.py` (new): both checkpoint scripts
  ([contracts/checkpoint-scripts.md](contracts/checkpoint-scripts.md)).
- `tests/unittest/scripts/cbts/test_coverage_artifact.py`: coverage archives through the helper,
  including the real helper falling back to urm, a read that fails in transfer skipping that build,
  and a rejection by the location runs currently publish to declining selection.
- `tests/unittest/others/test_perf_regression_artifact_url.py` (new): the perf record's link for each
  location and job type.

Locally, `tests/unittest/conftest.py` imports TensorRT-LLM, so run these with
`pytest --confcutdir=tests/unittest/scripts` (or `tests/unittest/others`), from a directory outside
the checkout with absolute paths and `--rootdir=<checkout>`: a `specs/` folder at the checkout root
breaks an unrelated gcc-based test there (17014 R19). Also run `shellcheck` on the changed scripts and
a parse-only check of the changed Groovy files.

**Expected**: all pass.

## 2. Access check (FR-009, FR-019)

The access-check script, back in the draft only, runs with `aws-artifactory-credentials` on a probe
under `test-results/`, `rerun_reports/`, and `cbts-coverage/` of a probe build: create, overwrite,
list, read, and delete. Run it:

- in a test pod, from its `trt-llm` container and from its `jnlp` container;
- in an `alpine` pod like Collect Test Result's;
- on a Slurm dispatcher pod;
- on a compute node of each remote cluster where test stages run, through `srun` from the login node,
  with the netrc copied in and removed afterwards.

Once, it also asks the new instance for the combined redacted archive of a recent failed run,
`llm-artifacts/<job>/<build>/test-results.tar.gz` in `sw-tensorrt-llm-generic-local`.

**Expected**: every operation succeeds; an anonymous listing and an anonymous read of the probe's
`test-results/` folder are rejected; no token appears in the output; the redacted archive is absent
(404) from the new instance. A 200 is a blocker (research R10). A 403 leaves the question to
PR_Github's scan-step configuration, which must name urm.

## 3. The switch pull request's default pre-merge run (FR-002 to FR-008, FR-010, FR-019)

Start it as TRTLLM-17014's rules say: `/bot run --disable-fail-fast` for the first run on each newly
pushed commit, plain `/bot run` for re-runs.

**Expected**:

- Every file under the run's `test-results/` is on both locations with the same SHA-1. The draft-only
  comparison in Collect Test Result logs a match for every name (research R13, item 3).
- The run's `rerun_reports/` files, if any, are on the new location only.
- Each stage shows its test branch and its checkpoint branch. The checkpoint branch's log shows each
  snapshot uploaded to urm and then to aws every ten minutes while results change, one final
  snapshot, and a delete on each location after the final archive. After the run, no `-progress`
  file is left on either location for a stage that reached its end.
- Collect Test Result logs its `[test-results] listed ...` line with every archive that the stages
  published, none on only one location, and all of them read; the run's JUnit report covers every
  stage.
- Coverage-based test selection logs `served-by=urm` for coverage archives (only post-merge runs
  publish them), and selects as before.
- Links in the console use the native-browser form on the new instance.

## 4. Targeted runs for every writing environment (FR-019)

From the stage configurations, list every combination of cluster, container runtime, and launch mode
(Slurm agent or batch job) that test stages use. For each one that no default pre-merge stage
covers, add one post-merge stage by exact name to a `--stage-list` run.

**Expected**: each such stage uploads its final archive and checkpoints to both locations and deletes
its checkpoint on both.

## 5. Draft-only changes for rarely reached paths (FR-007, FR-010, FR-019)

Add these to the draft's commit, run them, and remove them before the pull request is marked ready:

- a test that fails once and passes on rerun, in one pod stage and one Slurm stage. **Expected**: the
  per-stage rerun reports are on the new location only, the build-level `rerun_report.html` is on
  both locations, and the printed links open in the native browser;
- a switch that ends one stage's first attempt as an infrastructure failure after its final upload.
  **Expected**: the retry logs `served-by=aws` for the earlier attempt's archive and reuses its passed
  tests;
- a test that fails when it finds an Artifactory credential while it runs: an environment variable
  named like those the pipelines bind (`ART_USER`, `ART_PASS`, `ARTIFACTORY_USER`,
  `ARTIFACTORY_PASSWORD`), or a readable netrc file with an entry for artifactory.nvidia.com or
  urm.nvidia.com under the workspace, its temporary folder, `$HOME`, or `/tmp`. Run it in a pod stage,
  a Docker-runtime agent stage, and an ENROOT-runtime agent stage, long enough to span a checkpoint
  upload. **Expected**: it passes;
- a change that points one pod stage's and one Slurm stage's replica uploads at a repository that the
  identity cannot deploy to. **Expected**: the stages' urm uploads succeed; the failed checkpoint
  uploads are only logged; each failed final archive is logged with the credential message and the
  urm link and marks its stage unstable; the sub-jobs' build results are what they would otherwise
  have been; Collect Test Result reads those archives from urm and counts them as on urm only.

Removing them is a significant code change, so a default pre-merge run passes again afterwards.

## 6. PR_Github unchanged (FR-012, FR-019, User Story 3)

**Expected**, for runs of the switch pull request (for a failed run, any failure will do, or a
draft-only test that always fails in one stage):

- a failed run: PR_Github's log shows it reading the run's test results from urm, as today; it
  publishes its redacted copies as today; `blossom-ci` shows the test summary that the run's JUnit
  report implies;
- a passing run that reran tests: the comment links `rerun_report.html` on urm, and the link opens.

## 7. People (FR-016)

A developer outside the infrastructure team, with only the company sign-in, opens from the switch
pull request's runs one link of each kind: a Slurm stage's full-output archive, a per-stage rerun
report, the build-level rerun report as the console prints it, the docs preview, a perf record's
link, and a probe file that the access check leaves under a probe build's `cbts-coverage/` for this
check and removes afterwards (no pre-merge run writes coverage archives).

**Expected**: each opens or downloads. Record whether HTML reports render or download; if links to
files do not open, switch them to their folders (research R8). The switch does not merge before this
passes.

## 8. Rollback and branch isolation (FR-017, FR-018)

- With `LLM_ARTIFACTS_PRIMARY` set to `urm` for a rehearsal, a run of the switch pull request writes
  its test results, rerun reports, and coverage archives to urm only, with no replica; deletes its
  checkpoints on `sw-tensorrt-generic-local` only; and prints direct urm links. Collect Test Result
  lists both locations and reads everything from urm.
- A run without the switch, such as a release branch or a pull request not rebased past it, still
  writes to urm only, and PR_Github still finds it.

## 9. Before marking the pull request ready and merging (FR-015, FR-016, FR-019, FR-024)

- Steps 2 to 8 have passed; the draft-only files (the design documents, the access-check script, the
  copy comparison, and the step 5 changes) are gone from the commit.
- The owners of each known outside reader have been told what changes, that urm keeps their test
  results until its retirement date, and what access the new location needs; whether each internal
  tool runs in production is recorded; the TRT-LLM developers have been told what changes for them.
- The pull request's description states what FR-024 lists, including dual publishing until urm is
  retired, PR_Github's later change, the Blue Ocean change from the checkpoint branch, and the
  residual risk of research R3.

## 10. After cutover (FR-020, SC-001 to SC-014)

- Compare the first merge-request run and the first post-merge run with those of the commit before
  the merge: the same test-result archives by name and number, the same test counts in each JUnit
  report, and the same stage outcomes.
- On the first post-merge run: the coverage step and the coverage task publish their archives to the
  new location only; the next pre-merge run's coverage-based selection reads them there.
- Over the first two weeks: track the success criteria from Collect Test Result's
  `[test-results] listed ...` lines (names on one location only, for SC-001; names read, for SC-002),
  stages marked unstable by a replica failure, leftover checkpoints on both locations (SC-006), and
  the pipeline's and the changed steps' median durations (SC-007).
- Submit the evidence through the program's support channel for sign-off (FR-023), recording what
  remains before urm's retirement: PR_Github's change, outside readers not yet moved, and the
  follow-up change.

## 11. Before urm's retirement date (FR-002, FR-012, FR-015, SC-015)

- PR_Github's change has landed, and its evidence is recorded
  ([contracts/pr-github.md](contracts/pr-github.md), "Evidence when the change lands").
- Every outside reader in production reads the new location, according to its owner.
- The follow-up change that ends dual publishing (research R18) has merged, and the first
  merge-request and post-merge runs after it publish their test results to the new location only and
  read them there.
