<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Data Model: Migrate Test Results to AWS Artifactory

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Research**: [research.md](research.md)

The "data" here are files in Artifactory and the facts that pipelines keep about them while a run is
in progress. Paths below are relative to a location's repository.

## Location

| Field | `aws` (new) | `urm` (old) |
|---|---|---|
| Root | `https://artifactory.nvidia.com/artifactory` | `https://urm.nvidia.com/artifactory` |
| Repository for reads, lists, and writes | `sw-tensorrt-llm-generic-local` | `sw-tensorrt-generic` (virtual) |
| Repository for deletes | `sw-tensorrt-llm-generic-local` | `sw-tensorrt-generic-local` (the virtual repository refuses deletes) |
| Write and delete credential | `aws-artifactory-credentials` | `urm-artifactory-creds` |
| Read access in CI | The same credential | Anonymous |
| Link form for people | `https://artifactory.nvidia.com/ui/native/<repository>/<path>`, through the company sign-in | `https://urm.nvidia.com/artifactory/<repository>/<path>` |

- **Primary**: `LLM_ARTIFACTS_PRIMARY`, `aws` when unset, passed by a parent pipeline to the jobs it
  starts. **Secondary**: the other location.
- Mapping from the legacy paths in `UPLOAD_PATH` to layout paths: TRTLLM-17014's, unchanged.

## Targets

Where an output goes, decided by the primary (research R2):

| Primary | `test-results/` outputs, in upload order | `rerun_reports/` and `cbts-coverage/` |
|---|---|---|
| `aws` (dual publishing) | `urm` (required), then `aws` (replica) | `aws` |
| `urm` (rollback) | `urm` (required) | `urm` |

- **Required**: its upload fails a step where today's upload fails it (spec FR-007).
- **Replica**: best effort; a failed upload is logged with the urm link and marks the stage unstable,
  never the build (research R7). A checkpoint's failed upload, to either location, is only logged
  (spec FR-004).
- When urm is retired, the follow-up change makes `aws` the only target and the required one
  (research R18).

## Stage attempt

One run of one test stage, sub-job, and attempt.

| Field | Meaning |
|---|---|
| `stageName` | The stage, for example `A10-PyTorch-1` |
| `postTag` | Empty, or the sub-job and attempt suffix, for example `-attempt-2` or `-SubJob-RunTest-attempt-2` |
| `primary` | `LLM_ARTIFACTS_PRIMARY` when the attempt starts |
| `targets` | The attempt's `test-results/` targets, from `primary` |
| Execution context | A pod stage, a Slurm agent stage (Docker or ENROOT runtime), or a Slurm batch stage driven from its dispatcher pod |

**Rules**: an attempt writes its checkpoints and its final archive to each of its `targets` in
order, writes its per-stage rerun report to its `primary`, and deletes its checkpoint on each of its
`targets` (spec FR-002, FR-004). A retried attempt reads earlier attempts' final archives by the read
rule (FR-005, FR-008).

## Outputs

| Output | Path | Writer | Locations | Count | Notes |
|---|---|---|---|---|---|
| Final archive | `llm-artifacts/<job>/<build>/test-results/results-<stage><postTag>.tar.gz` | The attempt, when it ends | Targets | At most one per attempt | JUnit files (renamed with `postTag`, or `superseded-` when a retry is planned), logs, rerun data, coverage data files for coverage stages, `debug_env.txt` |
| Checkpoint | `.../test-results/results-<stage><postTag>-progress.tar.gz` | The attempt's checkpoint branch | Targets | At most one per location at a time | See its lifecycle below |
| Per-stage rerun report | `.../rerun_reports/<stage>_rerun_results.html` | The attempt, after reruns | Primary | At most one per stage | The data is also inside the final archive |
| Build-level rerun report | `.../test-results/rerun_report.html` | Collect Test Result | Targets, resolved when it uploads | At most one per run | Built from every archive's `rerun_results.xml` |
| Docs preview | `.../test-results/doc-html-preview.tar.gz` | The documentation build stage | Targets | At most one per run | Not a test result; no CI step reads it |
| Coverage archive per architecture | `.../cbts-coverage/cbts_pystart_report_<arch>.tar.gz` | The coverage step, post-merge | Primary | One per architecture | Read by coverage-based test selection in later pre-merge runs |
| Combined coverage archive | `.../cbts-coverage/cbts_pystart_report.tar.gz` | Collect Test Result's coverage task, post-merge | Primary | At most one per run | For people |

`<job>/<build>` and every file name are unchanged (FR-003). During dual publishing, a run's
`test-results/` on urm holds every file that its replica holds (research R2).

## Checkpoint lifecycle

On each location among the attempt's targets:

```text
none --first change--> uploaded --change, at most every 10 min--> uploaded (replaced)
uploaded --tests end--> final snapshot uploaded --final archive uploaded--> deleted
uploaded / final snapshot --stage stops before its final upload--> left behind (the attempt's only result)
final snapshot --delete fails--> left behind beside the final archive
```

- "Change" is a new content hash of the stage's results directory, judged for each location against
  the content that location last received. The hash of the content that every target last received
  is kept in `<PROGRESS_TAR>.last_hash`; a new one waits in `<PROGRESS_TAR>.last_hash.pending` and
  replaces it only when the snapshot reached every target. The checkpoint branch keeps the hash each
  location last received, so a location that missed an upload gets the next snapshot even when the
  content has not changed, and a location that already has it is not sent it again (spec FR-004).
- A failed checkpoint upload or delete, on either location, is logged and marks nothing (spec
  FR-004).
- A replica that missed its last upload keeps the previous snapshot, which readers read first. That
  matters only for an attempt that also stopped before its final upload; the failed upload is in the
  stage's log (research R3).
- Leftovers stay until retention removes the run's outputs; Collect Test Result and the coverage step
  read them as they do today (clarified 2026-10-27).
- The deletes run on every target, in a place where no test runs (research R4).

## Listing

| Field | Meaning |
|---|---|
| Entry | `<location> <file name>`, one per file directly in the listed folder |
| Order | The primary's entries, then the secondary's |
| Completeness | The helper's exit status: complete (0), folder absent from both (`FETCH_ABSENT`), rejected credential (`FETCH_REJECTED`), failed listing (`FETCH_TRANSFER_FAILED`) |

A reader keeps the names it needs and merges the two locations' entries by name; during dual
publishing nearly every name appears on both. It reads each name once by the read rule, which serves
the primary's copy and the secondary's only when the primary lacks the file, and logs the entries on
each location, the names, the names that only one location has, and the names read (FR-006,
FR-007).

## Redacted copy (outside this repository)

| Field | Value |
|---|---|
| Until PR_Github's change | `llm-artifacts/<job>/<build>/test-results/results-<stage>.tar.gz` and `llm-artifacts/<job>/<build>/test-results.tar.gz` in `sw-tensorrt-llm-generic-local`, almost certainly on urm (research R10) |
| After PR_Github's change | `redacted/llm-artifacts/<job>/<build>/test-results/results-<stage>.tar.gz` and `redacted/llm-artifacts/<job>/<build>/test-results.tar.gz` (proposed), in the same repository and instance |
| Marker | Property `scanned=true`; a release bundle carries the files to urm's edge site |
| Rule for this work | Never on the new instance, where it would replace originals; the check before the merge confirms it (FR-019) |
| Otherwise | Where the copies go, now and later, is PR_Github's concern; its change gives them a path of their own (FR-012) |

## Perf record link

`artifact_url` in a perf record written by a run after cutover: the run's `test-results/` folder on
the location in `globalVars["llm_artifacts_primary"]`, in that location's link form (FR-013,
FR-014). During dual publishing that is the new location, although urm holds the folder too.
Records written earlier keep their urm build-directory links.

## Who touches what

| Actor | Where it runs | Writes | Lists | Reads | Deletes | Credential handling |
|---|---|---|---|---|---|---|
| Pod test stage | `trt-llm` container; agent in `jnlp` | Final archive and checkpoints to its targets, per-stage report to its primary (agent uploads) | | Earlier attempts (before tests) | Checkpoint on each target, from `jnlp` | Agent holds both upload credentials; delete netrc with an entry per host, private to `jnlp`; read netrc only during the read |
| Slurm agent stage | Compute node, Docker or ENROOT | As above (agent uploads) | | As above | Checkpoint on each target, from the dispatcher pod | As above; delete netrc on the dispatcher pod |
| Slurm batch stage | Dispatcher pod; tests on the cluster | Final archive and checkpoints to its targets (agent uploads) | | Earlier attempts | Checkpoint on each target, on the dispatcher pod | No test runs on the pod |
| Documentation build | Pod | Docs preview to the targets (agent uploads) | | | | Agent holds the credentials |
| Collect Test Result | `alpine` pod of the parent | Build-level report to the targets; combined coverage archive to the primary | Run's `test-results/`, both locations | `results-*.tar.gz`, once per name | | Read netrc during listing and downloads only |
| Coverage step | `alpine` pod of the parent | Coverage archive per architecture to the primary | Run's `test-results/`, both locations | One architecture's `results-*.tar.gz` | | As above |
| Coverage-based selection | Pod of the parent, pre-merge | | | Coverage archives of recent post-merge runs | | TRTLLM-17014's netrc for the helper |
| PR_Github, until its change | Its own pod | Redacted copies, as today | urm's `test-results/` | Failed run's archives and `rerun_report.html`, on urm | | urm credential, as today |
| PR_Github, after its change | Its own pod | Redacted copies at their own path | Run's `test-results/`, both locations | The same, new location first | | Publishing credential in its own tooling |
| Internal tools | Their own hosts | | urm's `test-results/`, through its HTML index | Its archives | | Anonymous; their owners move them before urm is retired |
| People | Browser | | Native browser | Native browser after the company sign-in; urm links as today | | The program's read grant |
