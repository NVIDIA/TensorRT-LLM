<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implementation Plan: Migrate Test Results to AWS Artifactory

**Branch**: `TRTLLMINF-472-migrate-test-results-to-aws-artifactory` | **Date**: 2026-10-28 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-migrate-test-results-aws-artifactory/spec.md`

## Summary

Move CI's test results to `sw-tensorrt-llm-generic-local/llm-artifacts` on artifactory.nvidia.com
while urm's `sw-tensorrt-generic/llm-artifacts` keeps every file under `test-results/` until urm is
retired. In scope are each stage attempt's final archive and checkpoint, the build-level and
per-stage rerun reports, the docs preview, and the post-merge coverage archives, together with every
CI step that writes, lists, reads, deletes, or links them. The switch goes into main only, after
TRTLLM-17014 (PR #20494) has merged and passed its validation, and it reuses what that ticket
settled.

This revision follows the spec's clarifications of 2026-10-28. Everything a run writes under
`test-results/` still goes to urm as today and is replicated to the same paths on the new location
(dual publishing); `rerun_reports/` and `cbts-coverage/` go to the new location only. urm's upload
stays the one that fails a stage. PR_Github keeps reading urm's copies and moves before urm is
retired, so the switch no longer waits for it. A follow-up change ends dual publishing before urm's
retirement date. Three answers recorded later that day are in as well: SC-013 is withdrawn, so where
PR_Github puts its redacted copies is PR_Github's concern and only a check before the merge remains;
failed checkpoint uploads are only logged; and coverage-based test selection skips a post-merge run
whose coverage archive cannot be read, as TRTLLM-17014 does for build records.

The approach, from [research.md](research.md):

- **Targets**: a stage attempt resolves the primary once. When it is aws, a `test-results/` output
  goes to urm, the required copy, and then to aws, the replica; `rerun_reports/` and `cbts-coverage/`
  go to the primary alone; during a rollback everything goes to urm. One function per pipeline file,
  `testResultTargets`, decides this (R2).
- **Writes** keep the shared library's `uploadArtifacts` with the destination argument that
  TRTLLM-17014 added; urm's call stays as it is, and the replica follows it (R2).
- **Failures**: a failed replica upload of a final archive, the docs preview, or the build-level
  rerun report is logged with the credential message and the urm link, and marks only its stage
  unstable. A sub-job whose build result is UNSTABLE counts as infra-incomplete to the parent
  pipeline, which can block a pre-merge run's multi-GPU tests, so the build result never changes.
  Readers use the same mark for incomplete reads. A failed checkpoint upload or delete, on either
  location, is only logged (R7).
- **Checkpoints** are uploaded from a `parallel` checkpoint branch, through the Jenkins agent's own
  upload, after credential-free scripts prepare them. Each target receives a snapshot only when it
  differs from what that target last received, so neither location gets the same content twice. No
  Artifactory credential for either location is bound where tests run, on any runtime (R3).
- **Checkpoint deletes** run on every target where no test runs: the pod's `jnlp` container, which
  has `curl`, or the Slurm dispatcher pod, with a netrc private to that place (R4).
- **Listing**: the read helper gains `--list`, which asks both locations' storage APIs. Readers merge
  the entries by name and log how many each location holds (R5, R6).
- **Reads**: Collect Test Result, the coverage step, the retried stage's read, and coverage-based
  test selection all read through the helper: the new location first, urm when the replica is
  missing. Coverage-based test selection skips a post-merge run whose coverage archive fails to
  read, and declines only when the credential is rejected (R6).
- **Links** for people point to the new instance's native browser, which signs them in through the
  company sign-in, also for files that urm holds too; during a rollback they point to urm (R8).
- **The program** confirms the publishing identity's grant on the new paths, grants read to everyone
  who can open urm links today, and is told the volume (R9).
- **PR_Github** is unchanged at the merge. Before the merge, the access check confirms that its
  redacted copies are not on the new instance (FR-019). Its change (reads on the new location first,
  redacted copies under their own path, the comment's link in the native browser) lands before urm
  is retired; where its redacted copies go is PR_Github's concern (R10).
- **Rollback** shares `LLM_ARTIFACTS_PRIMARY` with build artifacts (R12).
- **Validation**: unit tests; an access check in every writing environment; the switch pull
  request's CI runs, with a draft-only comparison of both copies' SHA-1s; targeted runs for each
  cluster, runtime, and launch mode; draft-only changes for reruns, a retry, a credential probe, and
  a failed replica; PR_Github working unchanged on a failed and a passing run; and a developer's
  check of the links. The coverage-archive writers are validated on the first post-merge run after
  cutover (R13).
- **End of dual publishing**: a follow-up change before urm's retirement date, once PR_Github and the
  outside readers have moved (R18).

## Technical Context

**Language/Version**: Jenkins scripted-pipeline Groovy, Bash, Python 3.10+ (standard library only)

**Primary Dependencies**:

- Jenkins with the JFrog Artifactory plugin, used through `trtllm_utils.uploadArtifacts` in
  `trtllm-jenkins-shared-lib` (destination `aws` or `urm`, already merged; it throws on a failed
  upload), and the Kubernetes plugin (`container`, pod templates).
- The Artifactory REST API on both instances: the storage API (folder info, file info, checksums),
  downloads, and `DELETE`.
- `jenkins/scripts/fetch_build_artifact.sh`, TRTLLM-17014's read helper, extended with `--list`.
- curl. No new third-party dependencies.

**Storage**: Artifactory generic repositories: `sw-tensorrt-llm-generic-local` on artifactory.nvidia.com
(new); `sw-tensorrt-generic` (virtual, reads and writes) and `sw-tensorrt-generic-local` (deletes) on
urm.nvidia.com (old). During dual publishing, both hold each run's `test-results/`.

**Testing**:

- pytest in `tests/unittest/scripts/` (run by `l0_cpu.yml` and `l0_a10.yml`) and
  `tests/unittest/others/` (run by `l0_cpu.yml`): the helper's list mode and exit-code names, the
  checkpoint scripts, coverage-based selection's coverage reads, and the perf record's link.
- The switch pull request's CI runs under TRTLLM-17014's rules: the default pre-merge run first, then
  targeted `--stage-list` runs with post-merge stages added by exact name.
- Draft-only changes: a comparison of both copies' SHA-1s, reruns, a retry, a credential probe, and a
  failed replica upload.
- An access check in every writing environment, and a developer's check of the links.
- The first post-merge run after cutover, for the coverage-archive writers.

**Target Platform**: TRT-LLM CI. Kubernetes pods (a `trt-llm` container beside the `jnlp` agent
container; `alpine` pods for the parent pipeline's collection steps), Slurm dispatcher pods, and
Jenkins agents inside Slurm jobs on remote clusters (4 Docker-runtime and 13 ENROOT-runtime clusters
in the shared library's configuration), plus PR_Github on the top-level controller.

**Project Type**: CI/CD infrastructure change in a library repository, plus a later change outside
it to PR_Github and a follow-up change here that ends dual publishing.

**Performance Goals**: the median merge-request pipeline duration within 10% of its value before
cutover; the median durations of Collect Test Result, each coverage step, and a stage's result
submission within 10%, or one minute (SC-007). Collect Test Result reads a pre-merge run's 75 files in
about 8 seconds today. A stage's result submission gains one replica upload of its final archive
(about 1 MiB on average) and one more delete (R14).

**Constraints**:

- No Artifactory credential, for either location, where tests run (FR-010); nothing in logs, on
  command lines, or in published archives, including `debug_env.txt`.
- No anonymous requests to the new location; main only.
- Dual publishing of `test-results/` until urm is retired, with no setting that ends it earlier
  (FR-002); `rerun_reports/` and `cbts-coverage/` to the primary only.
- urm's upload keeps failing a stage as today; the replica is best effort (FR-007), and "unstable"
  marks the stage, never the build (R7). Checkpoint uploads and deletes, on either location, are
  only logged (FR-004).
- One set of targets per stage attempt; no new CI identity; no shared-library change.
- People reach outputs on the new location through the native browser and the program's read grant
  (FR-016).
- Repository files name no internal tools, hosts beyond those already tracked, people, lists,
  portals, or channels (R15).

**Scale/Scope**:

- In `jenkins/L0_Test.groovy` and `jenkins/L0_MergeRequest.groovy`: 10 write sites, 7 of them
  writing `test-results/` to both locations; 3 delete call sites; 5 link sites; and 3 read sites.
  Also coverage-based selection's reads in `artifact.py`, the perf record's link in
  `perf_regression_utils.py`, the read helper, and the two checkpoint scripts (R1).
- Per run, on each location that receives it: pre-merge, about 75 files and 52 MiB plus about 400
  checkpoint uploads; post-merge, 157 to 186 files and 196 to 217 MiB in `test-results/`, 59 to 69 MiB
  in `cbts-coverage/`, and at most 0.2 MiB in `rerun_reports/` (R14).

No item is left as NEEDS CLARIFICATION. Where the redacted copies live is settled by evidence (R10)
and confirmed by the access check before the merge (R13).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against constitution v1.0.0. **Result: PASS**, with two justified deviations from Principle
VI (see Complexity Tracking).

| Item | Status | How the plan complies, or why it does not apply |
|---|---|---|
| I. Guideline-Conformant Code | Compliant | Python follows `CODING_GUIDELINES.md`, with narrow exception handling and the helper's exit codes by name; Groovy catches `InterruptedException` before `Exception`, as the pipelines do. New scripts and test modules carry the NVIDIA SPDX header, and modified files keep their 2026 year. `pre-commit` and `shellcheck` run on the changed files. Comments describe the code; rationale and history go to the PR description. The design documents under `specs/` carry the SPDX header too, in an HTML comment. |
| II. Stable Public API Contracts | Not applicable | No LLM API, `LlmArgs`, or API-stability reference changes. `LLM_ARTIFACTS_PRIMARY` is a CI setting, not a library knob. |
| III. Validated Configuration & Generated Schemas | Not applicable | No serving YAML or configuration class changes. |
| IV. Telemetry Privacy Boundaries | Not applicable | No LLM args or architecture identifiers. The perf record's link changes its value, not what telemetry collects. |
| V. Architectural Conformance | Compliant | Extends the established mechanisms instead of adding parallel ones: the shared upload helper's destination argument, called once per target; the read helper (a list mode rather than a second script); TRTLLM-17014's netrc and request patterns from `Build.groovy`; and the location helpers that each pipeline file keeps, with `testResultTargets` beside them. No subsystem guide applies. |
| VI. Testing Discipline | Compliant, two deviations | Changed behavior in scripts and Python gets tests at the lowest layer: the helper, the checkpoint scripts, `artifact.py`, and the perf link. Every new module is under `tests/unittest/scripts/` or `tests/unittest/others/`, which `l0_cpu.yml` (and `l0_a10.yml`) already run. The Groovy changes, including the targets and the replica's failure handling, have no unit-test harness, and the coverage-archive writers run only after merge; both are in Complexity Tracking. |
| VII. Traceable, Single-Concern Contributions | Compliant | `git commit -s`, no co-author trailers, title `[TRTLLMINF-472][infra] ...`. One concern: moving test results, with dual publishing until urm is retired. Ending dual publishing is a separate follow-up pull request with its own ticket, and PR_Github's change is its own change set in its own repository. No `AGENTS.md` change: no repository document names test-result locations (searched 2026-10-28), and how people open links is in internal documentation (FR-016). |
| Platform constraints | Compliant | Python 3.10+ standard library; no new third-party dependencies; no vendored sources; the PyTorch backend and the library code are untouched. |
| Gate 1: intake | Compliant | TRTLLMINF-472 leads the PR title. |
| Gate 2: commit | Compliant | `pre-commit` on every commit. |
| Gate 3: generated artifacts | Not applicable | No golden manifest, schema, or API reference changes. |
| Gate 4: local tests | Compliant | [quickstart.md](quickstart.md) step 1. |
| Gate 5: review | Compliant | A TRT-LLM engineer plus the code owners of `/jenkins` and `/scripts` (infrastructure developers), `/jenkins/scripts/cbts` (for `artifact.py`), and `/tests/integration/defs/perf` (for the perf link). Findings follow TRTLLM-17014 FR-026 (spec FR-025): blocking problems are fixed in the PR, the rest go to a linked follow-up. |
| Gate 6: CI | Compliant | `/bot run` under TRTLLM-17014's rules: `--disable-fail-fast` only on each newly pushed commit's first run, targeted `--stage-list` runs afterwards, and the PR passes `blossom-ci` through its test stages. |

**Post-design re-check (after Phase 1): PASS, unchanged.**

- The contracts add one helper mode ([contracts/read-helper-list.md](contracts/read-helper-list.md)),
  two credential-free scripts that replace two existing ones
  ([contracts/checkpoint-scripts.md](contracts/checkpoint-scripts.md)), and the targets rule
  ([contracts/test-result-locations.md](contracts/test-result-locations.md)), which reuses the
  existing upload helper: Principle V.
- Every new script and Python change has a test module listed in quickstart step 1; the Groovy
  failure paths are exercised by draft-only changes (quickstart step 5): Principle VI.
- PR_Github's contract ([contracts/pr-github.md](contracts/pr-github.md)) and the follow-up change
  live outside this pull request: Principle VII and Gate 1.
- No new dependency, no new identity, and no gate changes status.

## Project Structure

### Documentation (this feature)

```text
specs/001-migrate-test-results-aws-artifactory/
├── spec.md
├── plan.md                 # this file
├── research.md             # Phase 0
├── data-model.md           # Phase 1
├── quickstart.md           # Phase 1
├── contracts/              # Phase 1
│   ├── test-result-locations.md
│   ├── read-helper-list.md
│   ├── checkpoint-scripts.md
│   └── pr-github.md
├── checklists/
│   └── requirements.md
└── tasks.md                # created by /speckit-tasks
```

### Source Code (repository root)

```text
jenkins/
├── L0_Test.groovy                # location helpers, testResultTargets, and browse links; targets
│                                 # resolved once per stage attempt; final archive and docs preview
│                                 # to the targets, per-stage report to the primary; replica failures
│                                 # marked on the stage only; checkpoint branch; retry read through
│                                 # the helper with its status; deletes on every target from jnlp or
│                                 # the dispatcher pod
├── L0_MergeRequest.groovy        # testResultTargets and browse links; Collect Test Result and the
│                                 # coverage step: list both locations, merge by name, read through
│                                 # the helper, counts, unstable on gaps; rerun report to the
│                                 # targets; coverage archives to the primary
└── scripts/
    ├── fetch_build_artifact.sh   # --list (R5)
    ├── progress_snapshot.sh      # new; replaces progress_upload_snapshot.sh (removed)
    ├── progress_collect_slurm.sh # new; replaces progress_upload_watcher.sh (removed)
    └── cbts/cbts/command/coverage/selection/artifact.py   # coverage archives through the helper
tests/
├── integration/defs/perf/perf_regression_utils.py         # artifact_url (R8)
└── unittest/
    ├── scripts/test_fetch_build_artifact.py               # list mode; new readers of exit codes
    ├── scripts/test_progress_snapshot.py                  # new
    ├── scripts/cbts/test_coverage_artifact.py             # coverage archives
    └── others/test_perf_regression_artifact_url.py        # new
```

Outside this repository: PR_Github's pipeline and its scan step (R10), and the program's grants (R9).
`trtllm-jenkins-shared-lib` does not change.

**Structure Decision**: no new directories. The scripts sit in `jenkins/scripts/` beside the ones
they replace, and the tests in `tests/unittest/scripts/` and `tests/unittest/others/`, which CI
already runs.

## Delivery Sequence

| Step | Work | Gate to the next step |
|---|---|---|
| 0 | Send the requests to the program (R9): grant scope, wider read grant, volume, and retention. Identify the internal tools' owners and find out whether the tools run in production (R11). | Requests sent |
| 1 | Rebase this branch onto TRTLLM-17014's current pull-request head (R17). Open the draft pull request, with its single commit carrying the design documents, the access-check script, and the copy comparison. Check `curl` in the `jnlp` container on the first draft run (R4). Run the access check in every environment, with the probe for the redacted copies (quickstart step 2). | Access check passes; delete mechanism settled; redacted copies confirmed off the new instance |
| 2 | Implement in that commit; run the unit tests, the default pre-merge run, the targeted runs, the draft-only changes, PR_Github's unchanged runs, the developer's link check, and the rollback rehearsal (quickstart steps 1 and 3 to 8). | Quickstart steps 1 to 8 pass |
| 3 | When TRTLLM-17014 has merged and passed its validation, rebase onto main; drop the draft-only files; pass the default pre-merge run again; send the notices (FR-015, FR-016); write the PR description (FR-024); mark the pull request ready; handle review findings under TRTLLM-17014 FR-026. | FR-016's check passed; FR-019 complete; no blocking finding open |
| 4 | Merge, which is the cutover and starts dual publishing. Validate after cutover (quickstart step 10), with the coverage-archive writers on the first post-merge run and the rollback available. Request sign-off, recording what remains before urm's retirement. | Sign-off through the program's support channel |
| 5 | PR_Github's change, any time before urm is retired ([contracts/pr-github.md](contracts/pr-github.md)); the owners of outside readers move them (R11). | PR_Github's evidence recorded; SC-015 |
| 6 | Before urm's retirement date, the follow-up change ends dual publishing (R18; quickstart step 11). Hand off to TRTLLMINF-295 (wheels) and the remaining llm-artifacts outputs; urm's `sw-tensorrt-generic` is retired only when everything has moved. | None |

## Notes for the Spec

Found while planning. The plan follows the reading given here until the spec says otherwise.

1. **FR-013's example of the PR comment's link**: PR_Github posts that link, and it keeps pointing to
   urm until PR_Github's change lands (FR-012). The plan applies FR-013 to the links that CI prints
   from the switch on, and to the comment's link from PR_Github's change on.
2. **Edge case "Credentials on Slurm batch jobs"**: TRTLLM-17014's pull request now deletes that
   netrc before the tests start (R17). Nothing in this plan depends on it either way.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Principle VI: the Groovy pipeline changes ship without unit tests, including the targets and the handling of a failed replica | The repository has no Groovy unit-test harness, and the changed Groovy runs only inside Jenkins. | Adding a harness is a separate project. The testable logic lives in scripts and Python with pytest coverage (the helper's list mode, the checkpoint scripts, `artifact.py`, the perf link). The pipeline paths run in the switch pull request's CI runs and targeted runs, and the failure paths in draft-only changes, including a failed replica, which must leave the build result unchanged (quickstart steps 3 to 5). |
| Principle VI: the coverage-archive writers (the coverage step and Collect Test Result's coverage task) are not exercised before cutover | They run only when the job is the post-merge pipeline (`L0_MergeRequest.groovy:583`), and the coverage stages that produce their input run only then too. | Turning them on in a pull request's run would need the coverage stages as well, which a draft-only change cannot reach. Instead, their identity and listing are checked from an `alpine` pod by the access check; they reuse the listing, read, and upload code that Collect Test Result runs in every pull request's run; and the rollback covers a failure on the first post-merge run after cutover. |
