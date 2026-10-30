<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Research: Migrate Test Results to AWS Artifactory

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-27, revised 2026-10-28

Findings come from this branch at `48dfa50464`, which carries TRTLLM-17014's change (PR #20494, not on
main yet); from the shared CI libraries as of 2026-10-20 (`trtllm-jenkins-shared-lib`) and
2026-10-23 (`bloom-jenkins-shared-lib`); from PR_Github's build logs; from the CI monitor's source;
and from anonymous requests to urm and artifactory.nvidia.com, on 2026-10-27 and 2026-10-28. Line
numbers refer to `48dfa50464`; R17 lists what TRTLLM-17014's pull request has changed since.
"17014 R<n>" cites TRTLLM-17014's research (`specs/001-migrate-builds-aws-artifactory/research.md`,
kept with that ticket's work).

The revision of 2026-10-28 follows the spec's clarifications of that day: everything under
`test-results/` is still published to urm and is replicated to the new location until urm is retired
(dual publishing); `rerun_reports/` and `cbts-coverage/` go to the new location only; urm's upload
stays the one that fails a stage; and PR_Github's change no longer has to land before the switch.
Three later answers that day are in too: SC-013 is withdrawn, since where PR_Github puts its
redacted copies is PR_Github's concern, and only the check before the merge stays (FR-019); failed
checkpoint uploads are only logged (FR-004); and coverage-based test selection skips a post-merge
run whose coverage archive cannot be read, as TRTLLM-17014 does for build records (FR-007).

## R1. Inventory: what writes, reads, deletes, and links test results

**Writers**

| Output | Where | Today | After |
|---|---|---|---|
| Final archive, pod and Slurm-agent stages | `jenkins/L0_Test.groovy:3895-3900` | Shared upload helper, urm | Shared helper: urm, then the replica on aws (R2) |
| Final archive, Slurm batch stages | `L0_Test.groovy:430-437` (`uploadResults`) | Shared helper, urm | Same as above |
| Checkpoint, pod and Slurm-agent stages | `L0_Test.groovy:5604-5653` | Shell watcher in the test step, `curl -u` with `urm-artifactory-creds` | Checkpoint branch, agent-side uploads to urm and aws (R3) |
| Checkpoint, reruns | `L0_Test.groovy:4857-4891` | Same | Same as above |
| Checkpoint, Slurm batch stages | `L0_Test.groovy:2796-2888` | Watcher beside the job monitor on the dispatcher pod, `curl -u` | Checkpoint branch on the dispatcher pod (R3) |
| Checkpoint delete | `L0_Test.groovy:5208-5236`, called at `:438`, `:2968`, `:3901` | `curl -X DELETE -u` in the test container, on the Slurm agent, or on the dispatcher pod | On urm and on aws, where no test runs (R4) |
| Per-stage rerun report | `L0_Test.groovy:5039-5045` | Shared helper, urm, with a urm link | Shared helper, the primary, with a browse link (R8) |
| Docs preview | `L0_Test.groovy:4360-4365` | Shared helper, urm | Shared helper: urm, then the replica on aws |
| Build-level rerun report | `jenkins/L0_MergeRequest.groovy:2163-2166` | Shared helper, urm, with a urm link | Shared helper: urm, then the replica on aws, with a browse link |
| Coverage archive per architecture | `L0_MergeRequest.groovy:2082-2084` | Shared helper, urm, with a urm link | Shared helper, the primary, with a browse link |
| Combined coverage archive | `L0_MergeRequest.groovy:2209-2211` | Shared helper, urm, with a urm link | Same as above |

**Readers**

| Reader | Where | Today | After |
|---|---|---|---|
| Collect Test Result | `L0_MergeRequest.groovy:2106-2131` | `wget` of urm's HTML index and of every `.tar.gz` it lists, anonymously; failures allowed | Listing of both locations, merged by name, and helper downloads of `results-*.tar.gz` (R5, R6) |
| Coverage step | `L0_MergeRequest.groovy:2048-2104` | Same, keeping one architecture | Same as above |
| Retry read | `L0_Test.groovy:5118-5152` | Anonymous probe, then `wget` | Helper download with status (R6) |
| Coverage-based test selection | `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py:134-164`, `:534-600`, `:649-668` | Anonymous `urllib` on urm | Helper existence checks and downloads (R6) |
| PR_Github | Outside this repository | urm credential (failed runs), anonymous (passing runs) | Unchanged on urm's copies until its change, which lands before urm is retired (R10) |
| Internal tools | Outside this repository | urm's HTML index, anonymously | Unchanged on urm's copies; their owners move them before urm is retired (R11) |

**Links for people**: the Slurm job log's pointer to the full test output (`L0_Test.groovy:2768`),
the per-stage rerun report (`:5044`), the build-level rerun report (`L0_MergeRequest.groovy:2166`),
the coverage archives (`:2084`, `:2211`), PR_Github's PR comment, and the perf record's link
(`tests/integration/defs/perf/perf_regression_utils.py:36`, `:98-100`).

**Unchanged**: the coverage-selection pins (`artifact.py` `PIN_BASE` and the pin upload in
`L0_MergeRequest.groovy`) stay on urm (spec FR-001); `jenkins/scripts/cbts/cbts/command/dryrun.py:556`
names no host; `jenkins/scripts/get_image_key_to_tag.py` is out of scope, and
`scripts/package_trt_llm.py` is on TRTLLM-17014's allowlist. No document, skill, or workflow in the
repository names a test-results address (searched again on 2026-10-28).

## R2. Writing: targets, order, and the replica

**Decision**:

- Every in-scope upload except the checkpoints keeps going through `trtllm_utils.uploadArtifacts`
  with the destination argument that TRTLLM-17014 added, as TRTLLM-17014's two calls in
  `L0_MergeRequest.groovy` do (`:679-683`, `:790-793`): destination `aws` with target
  `sw-tensorrt-llm-generic-local/<layout path>/<directory>/`, or destination `urm` with target
  `<UPLOAD_PATH>/<directory>/`. The helper throws when an upload fails, and for aws it first prints
  a message that names the credential.
- `L0_Test.groovy` gains the location helpers that `Build.groovy` defines (`:39-92`): the locations
  map with repository, backing local repository, and credential; `llmArtifactsSecondary`,
  `llmArtifactsLayoutPath`, `llmArtifactsRepoPath`, `llmArtifactsUrl`, `llmArtifactsLocalRepoPath`,
  and `llmArtifactsHost`. It and `L0_MergeRequest.groovy` also gain `llmArtifactsBrowseUrl` (R8) and
  `testResultTargets`, which returns where a `test-results/` output goes, in upload order: `urm`,
  then `aws`, when the primary is aws, the urm copy being the required one and the aws copy the
  replica; `urm` alone when the primary is urm (a rollback, FR-018). Outputs under `rerun_reports/`
  and `cbts-coverage/` go to the primary alone. Each pipeline file keeps its own copy, as
  TRTLLM-17014 left them; pipeline files share no code outside the libraries.
- A stage attempt resolves the primary, and with it its targets, once, when it starts, and uses them
  for its checkpoints, its final archive, its per-stage rerun report, and its checkpoint deletes. An
  attempt's outputs therefore stay together when a rollback lands while it runs (spec FR-004,
  FR-008). The parent pipeline's steps resolve them when they upload.
- urm's upload comes first and is the same call as today, so it fails a step exactly when it does
  today (FR-007). The replica follows. A failed replica upload is caught (an `InterruptedException`
  is rethrown): the step keeps the helper's message, which names the credential (FR-011), adds the
  file's direct urm link, and marks its stage unstable without changing the build result (R7). A
  checkpoint's failed replica is only logged, like any failed checkpoint upload (spec FR-004; R3).
- Because urm goes first and a failed urm upload of a final archive or of the docs preview ends the
  step, urm's `test-results/` holds every file that the replica holds. A missing replica is the only
  difference that can arise between the two copies, and this repository's readers cover it through
  the fallback (FR-005).
- The Artifactory plugin uploads from the Jenkins agent's own process: it reads the file from the
  workspace and sends the credential it holds, so no shell or test process sees the credential, for
  either location. That already holds for these uploads today, and it meets FR-010.

**Rationale**: the helper, its destination argument, and the rule that urm's failure counts are
settled (spec, "Settled by TRTLLM-17014", and the clarifications of 2026-10-28). Keeping urm's call
as it is and adding the replica after it leaves today's failures untouched, and the follow-up
change that ends dual publishing (R18) edits `testResultTargets` and the replica's handling instead
of every call site.

**Alternatives considered**:

- `curl` uploads with a netrc, as the packaged builds use. Rejected: a credential in a shell of the
  stage (FR-010), and a second upload path to test.
- The replica first. No reader gains, and a urm failure would then leave a file that only the new
  location holds, so urm would stop being the complete copy that outside readers rely on.
- A setting that turns dual publishing off before urm is retired. Declined by clarification
  (FR-002).

## R3. Checkpoints: uploading from a process the tests cannot read

**Findings**:

- Today the watcher (`jenkins/scripts/progress_upload_watcher.sh`, which runs
  `progress_upload_snapshot.sh`) starts in the background of the test's own `sh` step, inside
  `withCredentials(urm-artifactory-creds)`, and passes `-u "$ART_USER:$ART_PASS"` to `curl`. Every
  test process inherits `ART_USER` and `ART_PASS`. For Slurm batch stages it runs beside the job
  monitor on the dispatcher pod.
- Test pods have a `trt-llm` container, where the stage runs, and a `jnlp` container, where the
  Jenkins agent runs (`createKubernetesPodConfig`, `L0_Test.groovy:3920-4290`;
  `trtllm_utils.launchKubernetesPod` runs the stage in `container(<name>)`). They share the workspace
  volume but not their file systems or process namespaces; no pod sets `shareProcessNamespace`.
- Slurm agent stages: the shared library's Slurm configuration lists 4 clusters with the Docker
  runtime and 13 with ENROOT. With Docker, the agent runs on the host and the stage's steps run in
  `docker.image(...).inside` (`runInDockerOnNodeMultiStage`, `L0_Test.groovy:6315-6345`). With
  ENROOT, the agent runs inside the job's container (17014 R5) and the stage runs on it directly
  (`runInEnrootOnNode`, `:6355-6367`), so the tests share the agent's container and user.
- A pipeline branch runs one step at a time, so uploading while the test `sh` step runs needs a
  `parallel` branch. A process left in the background of a step shares that step's environment.
- `uploadArtifacts(..., 'aws')` and `uploadArtifacts(..., 'urm')` upload from the agent's process.
  The Artifactory plugin has no delete step.

**Decision**:

1. The test step runs in `parallel` with a second branch, the checkpoint branch, in every path: pod
   stages, Slurm agent stages of both runtimes, reruns, and Slurm batch stages on their dispatcher
   pod. The test branch sets a done flag in its `finally` block.
2. While the tests run, the checkpoint branch runs a credential-free script every
   `PROGRESS_UPLOAD_INTERVAL_SEC` (600 seconds). For a Slurm batch stage it first copies the latest
   results back from the cluster, as the watcher does today. The script prepares the checkpoint: the
   same tarball, testsuite renaming, post-tag transform, and content-hash deduplication as today.
   When the content changed, the branch uploads the tarball with
   `trtllm_utils.uploadArtifacts(<tarball>, <test-results target>, <location>)` to each of the
   attempt's targets, urm first, whose last received content differs from it. The branch keeps, for
   each location, the hash of the content that location last received, and the script's hash file
   moves on only once every target has the new content. A location that missed an update therefore
   gets the next snapshot even when the content has not changed again, and a location that already
   has the content is not sent it again, so neither location's checkpoint is replaced unless its
   results changed (spec FR-004). When the tests end, one final snapshot, with the timeout file
   generated from the unfinished tests, is prepared and uploaded the same way.
3. The scripts lose their network code and their credential variables.
   `progress_upload_snapshot.sh` becomes `progress_snapshot.sh`, which only prepares; the loop of
   `progress_upload_watcher.sh` moves into the pipeline, and its Slurm collection becomes the
   one-shot `progress_collect_slurm.sh` ([contracts/checkpoint-scripts.md](contracts/checkpoint-scripts.md)).
4. The branch waits with `waitUntil` on the done flag and the next due time, instead of sleeping in
   short steps, so it adds few steps to the build. Its failures, on either location, are logged and
   never fail or mark the stage (spec FR-004, R7).
5. No step that writes, replaces, or deletes test results binds an Artifactory credential where the
   tests run, so their environment carries none, for either location (FR-010). This also ends
   today's exposure of `urm-artifactory-creds`, which dual publishing would otherwise have kept.

**Rationale**: uploading from the agent's own process keeps the token out of every test process's
environment, readable files, and command lines on every runtime, including ENROOT, where the tests
share the agent's container. The agent holds the credentials only in its memory, which is not one of
the channels FR-010 names. On a host that lets processes of the same user attach a debugger to each
other, a determined test could still read that memory. That residual risk stays within
TRTLLM-17014's accepted risk for hosts that run a pull request's code, and the pull request's
description states it (FR-024). Dual publishing changes only the number of uploads per snapshot.

**Known limit**: a replica that missed its last update keeps the previous snapshot on the new
location, and readers read the new location first. That matters only for an attempt that also stops
before its final upload, so that the checkpoint is its only result, and the failed upload is in the
stage's log. Today a failed checkpoint upload leaves the previous snapshot on urm in the same way.

**Costs**: the stage shows two parallel branches in Blue Ocean instead of one box, which the
current code avoids on purpose; each snapshot is uploaded twice (R14); and each upload adds a few
lines to the stage's log.

**Alternatives considered**:

- Keep the shell watcher in the test step, with a netrc file instead of `-u`. Rejected: the tests can
  read the file (FR-010, clarified 2026-10-27).
- Keep today's watcher for urm and add only the replica from the agent. Rejected: the watcher binds
  `urm-artifactory-creds` where the tests run (FR-010).
- Run the shell watcher in the pod's `jnlp` container, with a netrc in that container's own
  temporary directory. It keeps the script, but it covers pod stages only, needs `curl` in the
  `jnlp` image (present, R4), and still needs `parallel`; Slurm agents would need another
  mechanism anyway.
- A sidecar container with `curl` in the test pod: the same limits, and one more image.
- Uploading a Slurm agent stage's checkpoints from its dispatcher pod: the pod cannot read the
  agent's workspace.

## R4. Deleting checkpoints outside the tests' reach

**Decision**: the attempt deletes its checkpoint on each of its targets, where no test runs, with one
netrc that holds an entry for each host it deletes from, in a temporary directory private to that
place. It follows `Build.groovy`'s `artifactoryRequestStatus` (`:423-447`): `curl` with
`--netrc-file` and `--request DELETE`, and no secret on a command line.

- Pod stages: in the pod's `jnlp` container, with the netrc outside the shared workspace volume. The
  workspace and its `@tmp` sibling are on the volume that the `trt-llm` container mounts too.
- Slurm agent stages, both runtimes: on the dispatcher pod, after the agent's `node` block returns
  (the `finally` after `executeLLMTestOnSlurm`, `L0_Test.groovy:1532`), instead of inside it
  (`:3901`).
- Slurm batch stages: on the dispatcher pod, in `uploadResults` (`:438`), as today; the second call
  (`:2968`) duplicates it and goes.
- Per location: on aws, `sw-tensorrt-llm-generic-local` with `aws-artifactory-credentials`; on urm,
  the backing local repository `sw-tensorrt-generic-local` with `urm-artifactory-creds`, because the
  virtual repository refuses deletes (`:5209-5212`). urm's deletes keep its credential during dual
  publishing and while CI is rolled back (FR-009).
- Outcomes, per location, as today: 200 or 204 deleted, 404 absent, anything else logged. A 401 or
  403 logs the credential message (FR-011). A delete never fails the stage and marks nothing
  (FR-004).

**Settled on the first draft run**: that run checked `command -v curl` in the `jnlp` container of
an A10 pod stage and found `/usr/bin/curl` (`curl 7.88.1`) in the amd64 image. The arm64 image,
which the SBSA pod stages use, installs `curl` (7.64.0) according to its image history. So pod
stages delete from `jnlp`, and the test pod template gains no sidecar. The tokens stay out of the
`trt-llm` container's file system and processes.

**Alternatives considered**:

- `httpRequest`, from the HTTP Request plugin: no pipeline here or in the shared libraries uses it,
  so whether the controllers have it is unknown.
- A delete function in the shared library that runs on the controller: a third change set and a
  library review for one call.
- Deleting from the `trt-llm` container after the tests end: a process that the tests leave behind
  could still read the netrc or the environment of the step (FR-010).
- Deleting at the end of the job instead of the stage: FR-004 has the stage delete its checkpoint
  after its final upload.

## R5. Listing a test-results directory

**Findings**: urm's storage API answers a folder request on the virtual repository anonymously:
`api/storage/sw-tensorrt-generic/llm-artifacts/<job>/<build>/test-results` returns 200 with
`children` (each with `uri` and `folder`), and 404 for a folder that does not exist. It answered in
about 50 ms on 2026-10-27 and in about 0.25 s on 2026-10-28. Its recursive file list
(`?list&deep=1`) answers 403 to anonymous requests. artifactory.nvidia.com answered 403 to an
anonymous folder request on 2026-10-28, and the read helper already calls its storage API with the
publishing identity for checksums (17014 R18).

**Decision**: the read helper gains `--list <directory path>`
([contracts/read-helper-list.md](contracts/read-helper-list.md)). It asks the storage API of the
primary and then of the secondary for the folder's direct children and prints one line,
`<location> <file name>`, for each file. It exits 0 when both locations answered, a missing folder
counting as empty; `FETCH_ABSENT` when neither has the folder; `FETCH_REJECTED` when the primary
rejected the credential; and `FETCH_TRANSFER_FAILED` when a listing failed after retries or the
secondary rejected the request, as file reads already map a secondary's rejection. It prints what it
listed in every case, so that a caller can carry on with it (FR-007). The path mapping, the netrc for
aws only, the retries, and the xtrace rule are those of the read mode. During dual publishing nearly
every name is listed once per location; the helper prints both and leaves merging to its callers
(R6), so the counts they log show what each location holds.

**Rationale**: one implementation of the location rules, tested once, already present wherever the
readers run.

**Alternatives considered**: a separate listing script, which would duplicate those rules; AQL
searches, a second API that returns more than a folder's children; urm's HTML index, which is
anonymous-only and fragile; listing urm alone because it holds every file during dual publishing,
which would miss a run's archives on the new location after a failed urm upload and is not what
FR-006 asks.

## R6. Reading test results in CI

**Decision**:

- **Collect Test Result** (`L0_MergeRequest.groovy:2106-2131`): its `alpine` pod installs `bash`,
  which the helper needs (`curl` is already installed, and busybox provides `sha1sum`). It takes the
  helper from the run's commit with `trtllm_utils.checkoutFile`, as `L0_Test` does (`:7693`), and
  writes a netrc as TRTLLM-17014's coverage-based selection step does (`:1530-1544`), removing it
  before any of the pull request's scripts run. It lists `${UPLOAD_PATH}/test-results` and keeps
  `results-*.tar.gz`: final archives and checkpoints. It no longer downloads the docs preview or the
  rerun report. It merges the two locations' entries by name and downloads each name once with the
  helper, which serves it from the new location, or from urm when the replica is missing. It logs
  how many entries it listed on each location, how many names that makes, how many names only one
  location has, and how many it read (FR-006, FR-007). It marks its stage unstable (R7) when the
  listing's status is neither 0 nor `FETCH_ABSENT`, or when it read fewer names than it listed.
- **Coverage step** (`uploadArchCoverage`, `:2048-2104`): the same, keeping one architecture's
  archives.
- **Retry read** (`reusePassedTestResults`, `L0_Test.groovy:5096-5191`): `fetchBuildArtifact` gains a
  download that returns the helper's status, built on TRTLLM-17014's current version, which already
  separates final statuses from transfer failures (R17). `FETCH_ABSENT` skips the attempt, as a 404
  does today. Any other failure logs the helper's message, marks the stage unstable, and skips only
  that attempt, keeping the reuse found in the test history and in other attempts; today one failure
  drops all reuse (`:5193-5197`). The read runs before the tests start, under TRTLLM-17014's rules
  for reads.
- **Coverage-based test selection** (`artifact.py`): `_exists` and `download` for coverage archives
  run the helper (`--exists`, then a download), which reaches archives published before cutover on
  urm through the fallback. The pins keep their urm reads and writes, and `ARTIFACT_BASE` stays a
  legacy path that the helper maps. Coverage archives are read as `build_commit` reads
  `build_info.txt` (R17, spec FR-007): an absent archive or a read that fails in transfer, which
  includes the helper's report of a rejection by the other location, skips that build, and the probe
  moves to an older one; a rejection by the location runs currently publish to raises, and selection
  declines at once. Probing up to 50 candidate builds costs up to two storage-API requests per archive
  instead of one ranged GET. While the new instance is down, each candidate whose archive has to be
  read there waits out the helper's three attempts before it is skipped, as `build_commit`'s reads
  do, and a pre-merge run's Setup Environment waits for the probe.
- **Exit-code names**: the consistency test (`tests/unittest/scripts/test_fetch_build_artifact.py:1160-1196`)
  gains `L0_MergeRequest.groovy` and the retry-read block of `L0_Test.groovy` as readers, and checks
  `artifact.py`'s new functions.

**Rationale**: one read rule for every read (FR-005); listing both locations covers a run split by
a rollback and a final archive whose replica failed (FR-006).

**Performance**: Collect Test Result downloads a pre-merge run's 75 files in about 8 seconds today.
Through the helper each file costs a storage-API request, a download, and a `sha1sum`, about 15 to
20 seconds in sequence; the listing costs two storage-API requests instead of one HTML page; and the
step stops downloading the 49 MiB docs preview that it never reads. If a step exceeds SC-007's
allowance, it downloads a few files at a time.

## R7. Best effort, failures, and what "unstable" marks

**Findings**: an UNSTABLE build result means "infra-incomplete" in these pipelines. A test sub-job
that saw only infra aborts resolves to UNSTABLE (`L0_Test.groovy:213-228`). The parent's `launchJob`
treats an UNSTABLE sub-job as incomplete coverage rather than a failure
(`L0_MergeRequest.groovy:2493-2504`), and in a pre-merge run that requires multi-GPU tests, an
UNSTABLE single-GPU sub-job blocks the multi-GPU stage (`:2720-2780`).

**Decision**: a step "marks itself unstable" (FR-007) with
`catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') { error(<reason>) }`. That marks the
stage and leaves the build result alone, as Collect Test Result already does
(`L0_MergeRequest.groovy:857`, `:2173-2176`). Jenkins' `unstable` and `warnError` steps also set the
build result, so a failed replica upload could block a pre-merge run's multi-GPU tests; they are not
used for these.

| Step | When a write or read fails |
|---|---|
| Final archive, urm upload | Fails the stage, as today |
| Final archive, replica upload | Logged with the urm link; the stage is marked unstable |
| Checkpoint upload or delete, on either location | Logged; the stage carries on and is not marked, as today (spec FR-004) |
| Docs preview, urm upload | Fails its stage, as today |
| Docs preview, replica upload | Logged with the urm link; the stage is marked unstable |
| Build-level rerun report, urm upload | Logged; the step carries on, as today |
| Build-level rerun report, replica upload | Logged with the urm link; the stage is marked unstable |
| Per-stage rerun report, to the primary | Fails the stage, as a failed upload does today: nothing catches it where the report is uploaded (`L0_Test.groovy:5039-5044`, called from `:5751-5755`) |
| Coverage archive uploads, to the primary | Logged; the coverage step and task stay non-fatal, as today |
| Collect Test Result listing or downloads | Logged with counts; the stage is marked unstable; the run goes on |
| Coverage step listing or downloads | Logged; marked unstable; non-fatal |
| Retry read of an earlier attempt | Logged; marked unstable; that attempt is skipped |
| Coverage-based selection's archive reads | Logged; a transfer failure skips that build, and a rejection by the location runs currently publish to declines selection |

A rejected credential never leads to a read of the other location (TRTLLM-17014 FR-014). On the new
location it also stops the replicas, while urm's copies go on (spec, "Token expired or revoked
mid-run"). The first version of this research counted the per-stage rerun report's upload as best
effort; the code shows otherwise.

## R8. Links shown to people

**Findings** (2026-10-27): a direct file URL on artifactory.nvidia.com answers a browser that sends
no credentials with `401` and `WWW-Authenticate: Basic realm="Artifactory Realm"`: a password
prompt, not the company sign-in page. The instance's native browser,
`https://artifactory.nvidia.com/ui/native/<repository>/<path>`, answers with its web application,
which signs people in through the company sign-in. urm serves direct URLs to anyone today.

**Decision**:

- Links for people to an in-scope output point to the primary: to the new location, in the native
  browser, while dual publishing is on, also for files that urm holds too (FR-013); directly to urm,
  as today, during a rollback and for runs before cutover. `llmArtifactsBrowseUrl(location, path)`
  builds them in `L0_Test.groovy` and `L0_MergeRequest.groovy`.
- A failed replica upload's message gives the file's direct urm link, so a person who followed a
  new-location link that leads nowhere can find the copy.
- The sites are those of R1. TRTLLM-17014's own links (build info, waive list) do not change.
- **Perf records**: `perf_regression_utils.py` builds `artifact_url` from the location in the
  `llm_artifacts_primary` entry of `globalVars`, which `L0_MergeRequest` passes to every job it
  starts (`:2447`) and which the test processes already read. It points to the run's `test-results/`
  directory: in the native browser on aws, directly on urm. Today's link names the build directory,
  which now spans both instances, and one field cannot hold one link per instance (FR-013), so it
  names the directory that holds the test results. A test goes to `tests/unittest/others/`, which
  `l0_cpu.yml` already runs.

**To verify** in FR-016's check: that a native-browser link to a file opens or downloads it after
the company sign-in, and whether an HTML report renders or downloads. If links to files do not open,
links point to the file's folder in the native browser.

## R9. Requests to the program (FR-009, FR-016, FR-021)

1. **Grant scope**: confirm that the permission target of `aws-artifactory-credentials` on
   `sw-tensorrt-llm-generic-local` covers deploy and delete/overwrite under
   `llm-artifacts/**/test-results/**`, `llm-artifacts/**/rerun_reports/**`, and
   `llm-artifacts/**/cbts-coverage/**`, not only the paths TRTLLM-17014 writes. The access check
   proves it (R13).
2. **Wider read grant**: read, and nothing else, on those three patterns for everyone who can open
   urm links today; the program chooses the group. The switch merges only after a developer outside
   the infrastructure team has opened one link of each kind with the company sign-in alone (FR-016).
   A refusal is a blocker.
3. **Volume and retention** (R14), including about 400 checkpoint uploads per pre-merge run on each
   location.

The requests and answers go through the program's support channel (FR-023) and into the sign-off
evidence.

## R10. PR_Github, outside this repository

**Findings** (PR_Github's build logs of 2026-10-27, and on 2026-10-28 the PR_Github build for a
failed pre-merge run):

- After a failed run, its scan step reads `sw-tensorrt-generic/llm-artifacts/<job>/<build>` on urm
  with only a urm credential bound, downloading the run's results archives with their SHA-1s. It
  extracts and redacts them: it strips the test logs from flagged JUnit files and leaves out files it
  cannot redact, such as an environment dump that holds internal identifiers. It repacks each failed
  stage's archive under the original's name (a leftover checkpoint included) and uploads them with a
  combined `<build>/test-results.tar.gz`; the log does not print the instance it uploads to. It sets
  the `scanned` property, creates a signed release bundle from a query on repository
  `sw-tensorrt-llm-generic-local`, path `llm-artifacts/<job>/<build>*`, and `@scanned=true`,
  distributes it to urm's edge site, and dispatches `.github/workflows/l0-test.yml` with the
  combined archive's address on that site.
- When the step finds nothing to download, it raises an error and the PR_Github build fails.
- After a passing run, PR_Github fetches `rerun_report.html` from urm anonymously, retrying three
  times about a minute apart when the report is absent, which is the usual case.
- A repository named `sw-tensorrt-llm-generic-local` exists on urm as well as on the new instance,
  and both refuse anonymous requests for it. Only a urm credential is bound and the bundle goes to
  urm's edge site, so the redacted copies are almost certainly on urm, at the paths and names that
  the originals' replicas take on the new instance.

**Decision** ([contracts/pr-github.md](contracts/pr-github.md)):

- **Until its change lands**: nothing changes for PR_Github. Dual publishing keeps urm's
  `test-results/` as complete as today, so its reads, its redacted copies, `blossom-ci`, and its
  comment work as they do (FR-002, FR-012). The switch does not wait for it.
- **Before the switch merges**: confirm that the redacted copies are not on the new instance, with
  the access check (R13) or from the scan step's configuration, which its maintainers, the CI
  maintainers, can read. If they are there, PR_Github would overwrite the replicas of every failed
  run after cutover, so FR-019 makes that a blocker (FR-023); PR_Github's path change would resolve
  it.
- **Where the redacted copies go** otherwise, now and after urm's `sw-tensorrt-llm-generic-local`
  moves, is PR_Github's concern: SC-013 was withdrawn on 2026-10-28, and this work measures only
  the check above.
- **Its change**, any time before urm is retired (FR-012):
  - Reads: it lists and reads a run's test results on the new location first, and on urm when they
    are absent there, by the read rule, in its own tooling. It does not run the pull request's copy
    of the read helper: PR_Github is the gate in front of the pull request's code, and it would hold
    the publishing credential.
  - Identity: `aws-artifactory-credentials`, made available to PR_Github's job by the owners of the
    credential store. That is a prerequisite of its change, not of the switch.
  - Redacted copies: a top-level directory of their own in the same repository and instance,
    proposed `redacted/llm-artifacts/<job>/<build>/...`; any path outside `llm-artifacts/` works. The
    release bundle's query, the edge path, and the URL handed to `l0-test.yml` follow; `l0-test.yml`
    does not change, and older copies stay.
  - Rerun report: read by the read rule with the publishing identity. An absent report is known at
    once, without two minutes of retries, and the comment links the report in the native browser
    (R8) on the instance that holds it.
  - Evidence: when it lands, a failed and a passing run show its new reads, the redacted copies under
    their own path, and the originals on both locations unchanged (FR-012).

**Alternatives considered**: running the pull request's read helper inside PR_Github, rejected
above; making PR_Github's change a prerequisite of the switch, as the first version of this plan
did, which dual publishing makes unnecessary (clarified 2026-10-28).

## R11. Other outside readers (FR-015)

- Two internal tools list `test-results/` through urm's HTML index by fixed address. The CI monitor
  is in active use (spec, checked 2026-10-28). Its source, read on 2026-10-28, reads only
  `test-results/` on urm, so it keeps working on urm's copies until urm is retired, and the move of
  `rerun_reports/` and `cbts-coverage/` off urm does not touch it. It also walks release-1.3 jobs,
  which stay on urm (FR-017). The other tool recovers a build's commit from its archives; its source
  was not available here, and whether it runs in production is unknown.
- Before the merge, their owners learn what changes, when (dual publishing until urm's retirement
  date), and what access the new location needs; whether each tool runs in production is determined;
  a tool that has not moved is recorded with the sign-off request. The owners of dashboards that show
  perf records' links are told too.

## R12. Rollback (FR-018)

Test results share `LLM_ARTIFACTS_PRIMARY` with build artifacts (spec default), which parent
pipelines pass to the jobs they start (17014 R7). After a rollback, new attempts publish
`test-results/` to urm alone, with no replica (FR-002), and `rerun_reports/` and `cbts-coverage/`
to urm; they delete on `sw-tensorrt-generic-local` only; and they print direct urm links. Attempts
that started before the rollback keep their targets (R2). The readers list both locations, and
PR_Github, once changed, reads the new location first and falls back, so results on either side stay
readable and nothing is lost when dual publishing resumes.

While the new instance is down and before the rollback lands, replica uploads fail, those of final
archives marking their stages unstable, while urm receives every file under `test-results/`. This
repository's readers, which ask the new location first, cannot read it: a failure is not an
absence, so they do not fall back, and they mark themselves unstable (spec, "New instance outage").
After the rollback they read urm first, but listing still asks the new location as the secondary,
so Collect Test Result stays marked unstable during the outage even when it reads every archive
from urm. Coverage-based test selection skips each candidate whose archive it cannot read only after
the helper's retries, so it delays pre-merge runs for as long as the outage lasts (R6).

## R13. Validation before cutover (FR-019)

1. **Unit tests**: the helper's list mode, the checkpoint scripts, `artifact.py`'s coverage reads,
   the perf link, and the exit-code names ([quickstart.md](quickstart.md) step 1).
2. **Access check**: TRTLLM-17014's access-check script, which no commit holds and which was kept
   with the internal documentation, comes back in the draft only. It gains a listing, an
   overwrite, and a delete of a probe under a `test-results/` directory and the other two
   directories. It runs with the publishing identity on a pod (also from its `jnlp` container), on a
   Slurm dispatcher pod, and on a compute node of each remote cluster where test stages run (through
   `srun` from the login node, with a netrc copied in and removed, as 17014 R5 does), and it checks
   that an anonymous listing and read of a `test-results/` directory are rejected. Once, it also asks
   the new instance for the combined redacted archive of a recent failed run,
   `llm-artifacts/<job>/<build>/test-results.tar.gz` in `sw-tensorrt-llm-generic-local`, a path no
   original uses: a 404 confirms that PR_Github does not write to the new instance (R10, spec
   FR-019), a 200 is a blocker, and a 403 leaves the question to the scan step's configuration.
3. **CI runs of the switch pull request**, under TRTLLM-17014's rules: the default pre-merge run
   first, which reaches Collect Test Result, coverage-based selection through the fallback, and every
   pre-merge stage's checkpoints, final archives, and deletes on both locations. A draft-only step in
   Collect Test Result compares the two copies: for each name the listing shows, the helper's
   `--exists`, run once with each location as the primary, returns that location's SHA-1, and the two
   must match (FR-019). Then targeted runs that add post-merge stages by exact name, one for each
   combination of cluster, runtime, and launch mode (agent or batch) that no pre-merge stage uses;
   the inventory comes from the stage configurations at implementation time.
4. **Rarely reached paths**: draft-only changes, removed before the pull request is marked ready. (a)
   A temporary test that fails once and passes on rerun, in one pod stage and one Slurm stage,
   produces the per-stage and build-level rerun reports. (b) A temporary switch ends one stage's first
   attempt as an infrastructure failure after its final upload, so that its retry reads the earlier
   attempt's archive. (c) A temporary test fails when it finds an Artifactory credential while it
   runs. (d) A temporary change points one pod stage's and one Slurm stage's replica uploads at a
   repository that the identity cannot deploy to: the stage's urm uploads succeed; the failed
   checkpoint uploads are only logged; the failed final archive is logged with the credential message
   and the urm link and marks the stage unstable; the sub-job's build result is what it would
   otherwise have been; and Collect Test Result reads that archive from urm and counts it as on urm
   only. Removing these changes is a significant code change, so the
   default pre-merge run passes again afterwards, as TRTLLM-17014's rules require.
5. **PR_Github**: the switch pull request's own failed and passing runs show it working unchanged from
   urm's copies: the `blossom-ci` status and test summary that the run's JUnit report implies, and
   the comment with its rerun report link on urm (FR-019). Its own change is proven when it lands
   (R10).
6. **People**: the program's read grant and the developer's check (FR-016).
7. **Coverage archives**: the access check covers the identity from an `alpine` pod; the steps that
   build them are validated on the first post-merge run after cutover, with the rollback available.

**Considered**: an infrastructure dry run (`InfraDryRun`), started by hand from a development-folder
copy of the merge-request job, runs one tiny test in every stage on every platform. It would cover
every writing environment quickly, but it runs outside `/bot` and starts child jobs by hand; it is
kept as a fallback for a cluster that targeted runs cannot reach.

## R14. Volume, retention, and performance (FR-021, SC-007)

Measured on urm on 2026-10-27 with the anonymous storage API, for post-merge builds 3031 to 3037:

| Directory | Files per run | Size per run |
|---|---|---|
| `test-results/` | 157 to 186 | 196 to 217 MiB, of which the docs preview is about 49 MiB |
| `rerun_reports/` | 1 to 5 | 0.2 MiB at most |
| `cbts-coverage/` | 3 | 59 to 69 MiB |

Three of the six runs sampled had leftover checkpoints, one to three each. The description's
pre-merge figures (about 75 files and 52 MiB, plus about 400 checkpoint uploads) stand; a re-run on
the same commit writes far fewer files.

With dual publishing, the new location receives all of this, as a move would have sent it, and urm
keeps receiving `test-results/` as today while `rerun_reports/` and `cbts-coverage/` leave it.
Checkpoint uploads go to both, about 400 to each per pre-merge run. The program is asked to keep test
results at least as long as urm does.

Each `test-results/` write adds a replica upload: in a stage's result submission, the final archive
(about 1 MiB on average, from the figures above) and a second checkpoint delete; after the tests, the
final checkpoint snapshot. By estimate that stays within SC-007's one-minute allowance; the first
draft run measures it. The documentation stage uploads the 49 MiB docs preview twice; the other
checkpoint uploads run beside the tests.

## R15. Confidentiality

As 17014 R11: repository files name only hosts already present in tracked files
(artifactory.nvidia.com and urm.nvidia.com; the native browser is on the same host). They and the
public pull request name no internal tools, edge hosts, internal repositories, people, lists,
portals, or channels. The design documents in the draft use roles, and urm's edge site stays unnamed.

## R16. Change sets and order

1. The requests to the program and the access check (R9, R13).
2. One pull request in this repository, from this branch: the helper's list mode, the checkpoint
   scripts, the pipeline changes, `artifact.py`, the perf link, and their tests. While it is a draft,
   its single commit also carries the design documents, the access-check script, the copy
   comparison, and the temporary changes of R13, all removed before it is marked ready.
3. TRTLLM-17014 (PR #20494) merges and passes its validation after cutover; this branch is rebased
   onto main.
4. The merge, after FR-016's check and FR-019, starts dual publishing; then the comparison after
   cutover (FR-020), and the sign-off request (FR-023), which records what remains before urm's
   retirement.
5. PR_Github's change, in its own repository, any time before urm is retired (R10).
6. The owners of outside readers move them (R11).
7. The follow-up change that ends dual publishing (R18), before urm's retirement date.

No shared-library change is needed: the destination argument is enough.

## R17. TRTLLM-17014's pull request since this branch's base

PR #20494 was still open on 2026-10-28, with its head amended that day to `d118ddc3a5`. Compared with
`48dfa50464`, it changes files that this work changes too:

- `fetchBuildArtifact` (`L0_Test.groovy`) now fails at once on a final helper status, such as
  `FETCH_ABSENT` or a rejection, and repeats only `FETCH_TRANSFER_FAILED`. The retry read's
  status-returning download (R6) builds on that.
- `artifact.py`'s `build_commit` skips a build whose `build_info.txt` read fails in transfer, and
  still raises on a rejection by the primary. Coverage-archive reads follow the same rule (spec
  FR-007, clarified 2026-10-28; R6), so while the new instance is down the probe waits out each
  candidate's failed read before it skips it, as `build_commit`'s reads do.
- `slurm_install.sh` deletes a Slurm batch job's netrc once the package is downloaded, before the
  tests start. FR-010 leaves that netrc to TRTLLM-17014, so nothing here depends on it; the spec's
  edge case that has it present while tests run describes the older commit.
- `test_fetch_build_artifact.py` and `test_coverage_artifact.py` change as well.

This branch is rebased onto the pull request's head before implementation starts, and onto main
after the pull request merges; line numbers in this research refer to `48dfa50464`.

## R18. Ending dual publishing

- **When**: before urm's retirement date, once PR_Github's change has landed and been proven and
  every outside reader in production reads the new location (SC-015). Nothing in the pipelines turns
  dual publishing off earlier (FR-002).
- **What**: a follow-up pull request with its own ticket (the pattern of FR-025).
  `testResultTargets` returns the new location alone, and its upload then fails a step where urm's
  does today (FR-007). The replica's handling, urm's checkpoint deletes, and urm's side of listing
  and reading test results go. The rollback to urm goes with them, coordinated with the matching
  change for build artifacts, since both use `LLM_ARTIFACTS_PRIMARY` and the read helper.
- **Consequence for this work**: one function per pipeline file chooses the targets, and urm-specific
  details (the delete repository, urm's entry in the delete's netrc) live in the location helpers,
  so the follow-up stays small.
