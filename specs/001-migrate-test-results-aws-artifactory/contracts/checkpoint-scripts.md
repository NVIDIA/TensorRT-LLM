<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contract: Checkpoint Scripts, the Checkpoint Branch, and the Delete

The checkpoint scripts prepare files; the pipeline uploads and deletes them (research R3, R4).
Neither script makes an Artifactory request or reads an Artifactory credential.

## `jenkins/scripts/progress_snapshot.sh` (replaces `progress_upload_snapshot.sh`)

Prepares one checkpoint tarball of a stage's results folder.

| Variable | Meaning | Required |
|---|---|---|
| `WORKSPACE` | The stage's workspace | Yes |
| `STAGE_NAME` | The results folder's name and the log prefix | Yes |
| `PROGRESS_TAR` | The tarball's name, relative to `WORKSPACE` | Yes |
| `LABEL` | A short description for the log, for example `checkpoint` or `run1 final snapshot` | Yes |
| `POST_TAG` | The attempt's suffix; renames `results*.xml` inside the tarball, not on disk | No |
| `FINAL_SNAPSHOT` | Non-empty for the snapshot taken when the tests end | No |
| `TIMEOUT_XML_SCRIPT` | With `FINAL_SNAPSHOT`, regenerates `results-timeout.xml` from `unfinished_test.txt` | No |
| `PROGRESS_HASH_FILE` | The hash of the content that every target last received; default `${WORKSPACE}/${PROGRESS_TAR}.last_hash`; empty turns deduplication off | No |

Steps, as today up to the tarball: the final snapshot's timeout file, the testsuite rename from
`pytest` to the stage name, the content hash, and the tarball with the `POST_TAG` transform. It then
writes the new hash to `${PROGRESS_HASH_FILE}.pending` instead of uploading.

| Exit | Meaning |
|---|---|
| 0 | The tarball is ready, and the pending hash is written |
| 3 | The content has not changed since the last upload; no tarball is written |
| 1 | The tarball could not be written |

Log lines keep today's `[PROGRESS-UPLOAD] <stage>: <label> ...` form.

## `jenkins/scripts/progress_collect_slurm.sh` (replaces the loop of `progress_upload_watcher.sh`)

One pass of today's Slurm collection, for Slurm batch stages, on the dispatcher pod.

| Variable | Meaning | Required |
|---|---|---|
| `WORKSPACE`, `STAGE_NAME` | As above | Yes |
| `SLURM_SSH_STAT_CMD` | Prints the remote `results.xml` mtime, run on the compute node | Yes |
| `SLURM_SSH_REFRESH_CACHE_CMD` | Primes the login node's file-system cache before copying | Yes |
| `SLURM_SCP_XML_CMD` | Copies the remote `results*.xml` | Yes |
| `SLURM_SCP_UNFINISHED_CMD` | Copies `unfinished_test.txt` | No |
| `SLURM_SSH_LIST_PERF_CMD`, `SLURM_SCP_PERF_TEMPLATE` | List and copy the perf result folders | No |
| `PROGRESS_MTIME_FILE` | Holds the last mtime seen | Yes |

When the mtime advanced, it primes the cache and copies the XML files, `unfinished_test.txt`, and the
perf folders, each up to three times, as today, then records the mtime.

| Exit | Meaning |
|---|---|
| 0 | New results were copied |
| 3 | Nothing new |
| 1 | The XML copy failed after its retries |

## The checkpoint branch (pipeline)

- Runs in `parallel` with the test step, in the same node and container: the `trt-llm` container of a
  pod stage, the Slurm agent of an agent stage, or the dispatcher pod of a batch stage. The test
  branch sets a done flag in its `finally` block.
- Loop: one shell step waits until the test branch's done file exists or `PROGRESS_UPLOAD_INTERVAL_SEC`
  has passed. A failed wait is logged and tried again after a pause; the third failure in a row ends
  the loop. On the interval: for a batch stage, `progress_collect_slurm.sh`, and no checkpoint in
  that pass unless it exits 0 or 3 (a failed copy can leave partial results); then
  `progress_snapshot.sh`; on exit 0, read the new hash from `${PROGRESS_HASH_FILE}.pending` and, for
  each of the attempt's targets in order (urm, then the replica on aws during dual publishing) whose
  last received hash differs from it,
  `trtllm_utils.uploadArtifacts("${WORKSPACE}/${PROGRESS_TAR}", <test-results target>, <location>)`.
  The branch keeps, for each location, the hash of the content that location last received, updated
  when its upload succeeds. The pending hash replaces the hash file once every target has the new
  content; until then the script keeps preparing snapshots, and only the locations that lack the
  content are sent them, so no location's checkpoint is replaced unless its content changed (spec
  FR-004).
- After the tests end: `progress_snapshot.sh` with `FINAL_SNAPSHOT=1` and `TIMEOUT_XML_SCRIPT`, and the
  uploads when it exits 0. When either branch was interrupted (an abort, a timeout, or a fail-fast
  stop), there is no final snapshot, and the interruption is rethrown even when the other branch
  failed first.
- Every failure, on either location, is logged with `[PROGRESS-UPLOAD]` and never fails or marks the
  stage (spec FR-004). Reruns use the same branch, with their own done flag.
- No Artifactory credential is bound in the test step or in the branch's shell steps.

## The delete (pipeline)

`deleteProgressArtifact(stageName, postTag, targets)` deletes
`<layout path>/test-results/results-<stage><postTag>-progress.tar.gz` on each of the attempt's
targets after the final archive is uploaded, one request per location. Where it runs, the requests,
and their outcomes are in [test-result-locations.md](test-result-locations.md), "Deleting".

## Tests

`tests/unittest/scripts/test_progress_snapshot.py` runs both scripts on temporary folders, with stub
`ssh` and `scp` commands for the Slurm script: deduplication by content, the pending hash, the
`POST_TAG` transform inside the tarball and not on disk, the final snapshot's timeout file, the
testsuite rename, the exit codes, and a check that neither script reads `ART_USER`, `ART_PASS`, or
`ARTIFACTORY_NETRC` or starts `curl`.
