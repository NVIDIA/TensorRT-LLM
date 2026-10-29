<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contract: Test-Result Locations

Applies to every pipeline step, script, and tool that writes, lists, reads, deletes, or links an
in-scope output (spec FR-001). TRTLLM-17014's location contract (locations, path mapping, read rule,
primary and secondary) applies unchanged; this contract adds what test results need. Terms are those
of [data-model.md](../data-model.md).

## Outputs

| Output | Path below the repository | Locations | Written by | Read by |
|---|---|---|---|---|
| Final archive | `llm-artifacts/<job>/<build>/test-results/results-<stage><postTag>.tar.gz` | Targets | The stage attempt | Collect Test Result, the coverage step, the retry read, PR_Github, people, internal tools |
| Checkpoint | `.../test-results/results-<stage><postTag>-progress.tar.gz` | Targets | The stage attempt's checkpoint branch | Collect Test Result and the coverage step, as today; PR_Github; people; internal tools |
| Build-level rerun report | `.../test-results/rerun_report.html` | Targets | Collect Test Result | PR_Github, people |
| Docs preview | `.../test-results/doc-html-preview.tar.gz` | Targets | The documentation build stage | People, internal tools |
| Per-stage rerun report | `.../rerun_reports/<stage>_rerun_results.html` | Primary | The stage attempt | People |
| Coverage archive per architecture | `.../cbts-coverage/cbts_pystart_report_<arch>.tar.gz` | Primary | The coverage step (post-merge) | Coverage-based test selection, people |
| Combined coverage archive | `.../cbts-coverage/cbts_pystart_report.tar.gz` | Primary | Collect Test Result's coverage task (post-merge) | People |

Paths and names are those of today (spec FR-003).

## Targets

`testResultTargets(primary)`, in `jenkins/L0_Test.groovy` and `jenkins/L0_MergeRequest.groovy`:

| `primary` | Targets, in upload order |
|---|---|
| `aws` | `urm` (required), `aws` (replica) |
| `urm` | `urm` (required) |

- A stage attempt resolves `llmArtifactsPrimary()` once, when it starts, and with it its targets;
  every write and delete of its outputs uses them. A parent pipeline's step resolves them when it
  uploads.
- No setting turns dual publishing off. It ends with the follow-up change ("End of dual publishing"
  below).

## Writing

- Every upload goes through `trtllm_utils.uploadArtifacts(<file pattern>, <target directory>, <location>)`:
  - `aws`: target `sw-tensorrt-llm-generic-local/<layout path>/<directory>/`;
  - `urm`: target `<UPLOAD_PATH>/<directory>/`, as today.
- A `test-results/` output is uploaded to its targets in order: urm first, as today, then the
  replica.
- urm's upload fails or not as it does today: a final archive's and the docs preview's failures fail
  the step; the checkpoints' and the build-level rerun report's failures are logged.
- A replica upload's failure never fails the step. After the shared helper's message, which names
  the credential (FR-011), the step logs
  `[test-results] replica upload of <file> to artifactory.nvidia.com failed; urm copy: <direct urm link>`
  and marks its stage unstable with `catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE')`,
  never by changing the build result (research R7). An `InterruptedException` is rethrown. A
  checkpoint's failed replica is the exception: it is only logged, like every failed checkpoint
  upload ([checkpoint-scripts.md](checkpoint-scripts.md); spec FR-004).
- Outputs under `rerun_reports/` and `cbts-coverage/` are uploaded to the primary only. A failed
  per-stage rerun report upload fails the stage, and a failed coverage archive upload is logged, as
  today.
- The upload runs in the Jenkins agent's process, which holds the credential.
- No step that writes, replaces, or deletes an output binds an Artifactory credential, for either
  location, where tests run: not in their environment, not in a file they can read, not on a command
  line (FR-010).

## Deleting (checkpoints only)

| Stage | Where the deletes run |
|---|---|
| Pod stage | The pod's `jnlp` container, which has `curl` (research R4) |
| Slurm agent stage, Docker or ENROOT | The stage's dispatcher pod, after the agent's block returns |
| Slurm batch stage | The stage's dispatcher pod |

- One request per target: `DELETE <root>/<delete repository>/<layout path>` (see
  [data-model.md](../data-model.md), "Location"), sent by `curl` with `--netrc-file`. The netrc holds
  one entry per host it deletes from, sits in a temporary directory private to that container,
  outside the shared workspace volume, and is removed in a `finally` block.
- Outcomes, per location: 200 or 204, deleted; 404, absent; 401 or 403, the credential message
  (FR-011); anything else, logged. The stage is never failed or marked because of a delete.

## Listing

- `fetch_build_artifact.sh --list <directory path>` lists the folder on both locations
  ([read-helper-list.md](read-helper-list.md)).
- A reader keeps the names it needs (`results-*.tar.gz`, final archives and checkpoints alike, for
  test-result archives), merges the two locations' entries by name, and reads each name once with
  the read rule.
- A reader that needs every archive of a run logs one line,
  `[test-results] listed <p> on <primary> and <s> on <secondary>: <n> names, <x> only on <primary>, <y> only on <secondary>; read <r> of <n>`,
  and marks its stage unstable when the listing's status is not 0 or `FETCH_ABSENT`, or when it read
  fewer than `<n>` (FR-006, FR-007). It never fails the run. It stops reading after three reads in a
  row end with `FETCH_REJECTED` or `FETCH_TRANSFER_FAILED`, since each further read would wait out
  the helper's retries against a location that is down or refuses the credential.

## Reading

TRTLLM-17014's read rule, through `fetch_build_artifact.sh` (download, `--exists`, `--properties`),
for every read by name: the retry read, Collect Test Result's and the coverage step's downloads, and
coverage-based test selection's coverage archives (FR-005). During dual publishing the rule serves
the new location's copy, and urm's when the replica is missing.

## Links shown to people

| Location | Link form |
|---|---|
| `aws` | `https://artifactory.nvidia.com/ui/native/sw-tensorrt-llm-generic-local/<layout path>` |
| `urm` | `https://urm.nvidia.com/artifactory/sw-tensorrt-generic/<layout path>` |

- Links point to the primary: to the new location while dual publishing is on, also for files that
  urm holds too (FR-013), and to urm during a rollback.
- `llmArtifactsBrowseUrl(location, legacyPath)` builds them in `jenkins/L0_Test.groovy` and
  `jenkins/L0_MergeRequest.groovy`.
- Sites: the Slurm job log's pointer to the full test output, the per-stage and build-level rerun
  reports, the coverage archives, PR_Github's comment once its change lands
  ([pr-github.md](pr-github.md)), and the perf record's `artifact_url`, which names the run's
  `test-results/` folder.
- A direct `aws` URL answers a browser with a password prompt, so links for people never use it.

## Compatibility

- Branches without the switch (release branches, pull requests not rebased past it) keep writing to,
  reading from, and linking to urm, unchanged.
- urm's `test-results/` keeps every file it holds today, so readers that stay on urm (PR_Github until
  its change, the internal tools, people with urm links) see no change until urm is retired.
  `rerun_reports/` and `cbts-coverage/` of runs after cutover are on the new location only.
- Results published before cutover stay on urm: CI readers reach them through the fallback, and
  people through their old links.
- TRTLLM-17014's artifacts, uploads, and links are unchanged.

## End of dual publishing

Before urm's retirement date, and only after PR_Github and every outside reader in production read
the new location, a follow-up change makes `testResultTargets` return `aws` alone, as the required
target, and removes the replica's handling, urm's checkpoint deletes, and urm's side of listing and
reading test results (research R18).
