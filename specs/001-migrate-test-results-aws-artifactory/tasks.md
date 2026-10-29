---

description: "Task list for TRTLLMINF-472: migrate test results to AWS Artifactory, with dual publishing until urm is retired"
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tasks: Migrate Test Results to AWS Artifactory

**Input**: Design documents from `specs/001-migrate-test-results-aws-artifactory/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: Included. Constitution Principle VI asks for tests of changed behavior at the lowest
layer, the spec's reuse of TRTLLM-17014 FR-005 requires the exit-code consistency test, and the plan
lists the test modules. Groovy pipeline code has no unit-test harness (plan, Complexity Tracking):
the switch pull request's CI runs validate it, draft-only changes exercise its failure paths, and the
first post-merge run after cutover validates the coverage-archive writers.

**Organization**: Tasks are grouped by user story. Line numbers refer to `d118ddc3a5`, TRTLLM-17014's
pull-request head on 2026-10-28, onto which T008 moved this branch (research R17). research.md cites
`48dfa50464`, whose lines are the same up to `jenkins/L0_Test.groovy:1830` and
`jenkins/L0_MergeRequest.groovy:1524`, and 20 and 9 lower past `L0_Test.groovy:2160` and
`L0_MergeRequest.groovy:1541`. When that pull request changes again, or merges and this branch is
rebased (T066), find each place by its content.

**Progress on 2026-10-28**: T008 is done (the branch is on `d118ddc3a5`), and so is the first half
of T019: the draft pull request, #21223, carries the design documents with their headers.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: The user story the task serves (US1–US5)
- Each description starts with its change set:
  - **PR**: the one pull request in this repository, #21223, from
    `TRTLLMINF-472-migrate-test-results-to-aws-artifactory`. Cutover, and with it dual publishing,
    starts when it merges.
  - **PRG**: PR_Github's change, in PR_Github's own repository, outside this one
    ([contracts/pr-github.md](contracts/pr-github.md)). It lands before urm is retired and is not a
    prerequisite of the merge, unless T049 finds the redacted copies on the new instance.
  - **FUP**: the follow-up pull request in this repository that ends dual publishing before urm's
    retirement date (research R18).
  - **OPS**: work outside the code: requests, CI runs, checks, notices, and evidence.

## Path Conventions

- Paths are relative to the repository root: `jenkins/`, `tests/`.
- The pull request holds exactly one commit of this work, on top of TRTLLM-17014's commit until that
  merges. While it is a draft, that commit also carries this spec store, the access-check script and
  its tests, and the draft-only changes (T038, T041, T042, T047, and T050's failing test if used);
  T068 removes them before the pull request is marked ready (TRTLLM-17014's rule). Every change to
  the pushed commit is an amend, pushed with `--force-with-lease=<branch>:<sha just fetched>`.
- Every new file in the commit, draft-only ones included, carries the NVIDIA SPDX header
  (constitution Principle I); Markdown files carry it in an HTML comment.
- `specs/` and `.specify/` are excluded from Git through `info/exclude`; the draft's commit adds the
  spec store with `git add -f`.
- Evidence for sign-off goes in `specs/001-migrate-test-results-aws-artifactory/signoff-evidence.md`
  (T002): dates, outcomes, and links to run logs, never a token, a password, or netrc content.
- Repository files and the public pull request name no internal tools, internal repositories, edge
  hosts, people, lists, portals, or channels (research R15).

---

## Phase 1: Setup (Prerequisites With Lead Time)

**Purpose**: The requests and findings that gate the merge, and the branch's base (plan, Delivery
Sequence steps 0 and 1). T003–T007 change no code and can start at once.

- [X] T001 OPS: Check that `specs/`, `.specify/`, and `.claude/skills/speckit-*` are listed in
  `$(git rev-parse --git-common-dir)/info/exclude`, and add any that is missing, so that only
  `git add -f` can put the spec store into the pull request's commit.
- [X] T002 OPS: Create `specs/001-migrate-test-results-aws-artifactory/signoff-evidence.md` (NVIDIA
  SPDX header in an HTML comment) with one section per item, each "pending" until filled: grant scope
  (T003); wider read grant (T004); volume and retention (T005); credential availability (T006,
  T051); outside readers (T007); base commit (T008, T066); delete container (T021); default
  pre-merge run and copy comparison (T039); targeted runs (T040); rerun reports and retry read
  (T041); failed replica (T042); rollback rehearsal and branch isolation (T043); access check per
  environment (T046); credential probe and log scan (T048); redacted copies' instance (T049);
  PR_Github unchanged (T050); people check (T061); notices (T062); internal documentation (T063);
  reference inventory and allowlist (T064); final pre-merge run (T069); review (T071); duration
  baseline (T072); merge (T073); after cutover (T074–T077); sign-off (T065); PR_Github's change
  (T055); end of dual publishing (T078–T080); hand-off (T081). State at the top: dates, outcomes, and
  links only, never a token, a password, or netrc content.
- [ ] T003 [P] OPS: Through the program's support channel, ask the program to confirm that the
  permission target of `aws-artifactory-credentials` on `sw-tensorrt-llm-generic-local` covers
  deploy and delete/overwrite under `llm-artifacts/**/test-results/**`,
  `llm-artifacts/**/rerun_reports/**`, and `llm-artifacts/**/cbts-coverage/**`, not only the paths
  TRTLLM-17014 writes (spec FR-009, research R9). Record the request and the answer in
  `signoff-evidence.md`.
- [ ] T004 [P] OPS: Through the program's support channel, request read, and nothing else, on those
  three path patterns for everyone who can open urm links today, the group being the program's
  choice (FR-016, research R9). Say that the switch does not merge until the grant is in place and a
  developer outside the infrastructure team has opened a link of each kind with only the company
  sign-in. Record it in `signoff-evidence.md`; a refusal is a blocker (FR-023).
- [ ] T005 [P] OPS: Tell the program the expected volume on the new location and ask it to keep test
  results at least as long as urm does (FR-021). With dual publishing, the new location receives
  every in-scope output and urm keeps receiving `test-results/` as today. Per pre-merge run: about 75
  files and 52 MiB, almost all of it the docs preview, plus about 400 checkpoint uploads to each
  location. Per post-merge run on the new location: 157 to 186 files and 196 to 217 MiB in
  `test-results/`, 59 to 69 MiB in `cbts-coverage/`, at most 0.2 MiB in `rerun_reports/` (research
  R14). Record any difference in retention in `signoff-evidence.md`; a quota or rate limit that would
  refuse this volume is a blocker (FR-021).
- [ ] T006 [P] OPS: Confirm with the owners of the CI credential store that `aws-artifactory-credentials`
  and `urm-artifactory-creds` are usable by the test jobs on every controller, including from their
  pods' `jnlp` containers and from Slurm dispatcher pods, and by the merge-request job's pods. They
  already publish build artifacts with the first and test results with the second. PR_Github's job
  needs the publishing credential only for its own change (T051). Record in `signoff-evidence.md`.
- [ ] T007 [P] OPS: Identify the owners of the two internal tools that list `test-results/` through
  urm's HTML index, the CI monitor (in production, spec "Current state") and the tool that recovers a
  build's commit from its archives, and of the dashboards that show perf records' links. Find out
  whether the second tool runs in production, and whether either tool reads `rerun_reports/` or
  `cbts-coverage/` on urm, which leave urm at the merge (FR-015, research R11). Record owners as roles
  and the answers in `signoff-evidence.md`.
- [X] T008 PR: Keep this branch on TRTLLM-17014's current pull-request head (research R17). Done on
  2026-10-28 onto `d118ddc3a5`. Whenever PR #20494's head moves before it merges: fetch the remote
  that holds its branch, list what changed with `git diff d118ddc3a5 <head> -- jenkins/
  tests/unittest/scripts/`, and run `git -c user.email=<noreply address> rebase --onto <head> <old
  base>`. Record each base in `signoff-evidence.md`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The helper's list mode, the checkpoint scripts, and the location, target, and upload
helpers that every story's pipeline changes use.

**⚠️ CRITICAL**: No user story's pipeline work can begin until this phase is complete.

### Read helper (PR)

- [X] T009 [P] PR: Extend `tests/unittest/scripts/test_fetch_build_artifact.py` with list-mode tests
  against [contracts/read-helper-list.md](contracts/read-helper-list.md), using its existing
  `FakeArtifactory` servers (`:186`) and `Helper` runner (`:269`). Serve storage-API folder answers
  (`{"repo": ..., "path": ..., "children": [{"uri": "/<name>", "folder": <bool>}, ...]}`) and assert:
  - a folder on the primary only, the secondary only, both, and neither: the printed lines are
    `<location> <name>`, the primary's first ("Order: The primary's entries, then the secondary's",
    data-model.md), folders skipped, and the exit is 0, 0, 0, and 2;
  - a folder on both with the same names prints each name once per location, unmerged;
  - a primary 401 or 403: exit 3, the credential message, and no request to the secondary;
  - a secondary 401 or 403, and a secondary 500 after retries: exit 4, with the primary's entries
    printed;
  - a primary 500 that then succeeds is retried; one that keeps failing exits 4, and the secondary is
    still listed and printed;
  - `LLM_ARTIFACTS_PRIMARY=urm` reverses the order; aws requests carry the netrc and urm requests
    none; a missing `ARTIFACTORY_NETRC` when aws must be asked exits 64; `--list` with a
    `<dest-file>` exits 64; an unmappable path exits 64; a trailing `/` is accepted;
  - stderr has `fetch_build_artifact: listed <n> on <primary>, <m> on <secondary> path=<layout path>`;
  - the netrc password never appears in stdout or stderr.
  Run them and see them fail before T010.
- [X] T010 PR: Implement `--list <directory-path>` in `jenkins/scripts/fetch_build_artifact.sh` per
  [contracts/read-helper-list.md](contracts/read-helper-list.md), reusing `map_path` (`:101`),
  `use_location` (`:127`), `storage_url` (`:145`), `http_get` (`:165`), `with_retries` (`:297`), and
  `report_rejection` (`:341`). Add a `list_step` that requests `storage_url` of the folder, returns
  `FETCH_ABSENT` on 404, `FETCH_REJECTED` on 401 or 403, 1 (transient) on other failures or an
  unparsable body, and on 200 extracts the children whose `"folder"` is false with text tools only
  (no Python, no `jq`). Ask the secondary unless the primary rejected the credential; map a
  secondary rejection to `FETCH_TRANSFER_FAILED`, as file reads do. Print `<location> <name>` lines,
  primary first, without merging. Update `usage()` (`:44-78`) and its exit-code help for list mode.
  Make T009 pass, and run `shellcheck` on the file.

### Checkpoint scripts (PR)

- [X] T011 [P] PR: Write `tests/unittest/scripts/test_progress_snapshot.py` (new, NVIDIA SPDX header)
  against [contracts/checkpoint-scripts.md](contracts/checkpoint-scripts.md), running the scripts
  with `bash` on temporary folders. For `progress_snapshot.sh` assert: exit 0 writes
  `${WORKSPACE}/${PROGRESS_TAR}` and `${PROGRESS_HASH_FILE}.pending`; a second run with the pending
  hash moved over the hash file and no change exits 3 and writes no tarball; a second run without
  that move exits 0 again; `POST_TAG` renames `results*.xml` inside the tarball and not on disk;
  `FINAL_SNAPSHOT` with `TIMEOUT_XML_SCRIPT` creates `results-timeout.xml` from `unfinished_test.txt`,
  and without `FINAL_SNAPSHOT` it does not; `testsuite name="pytest"` becomes the stage name; an
  unwritable tarball exits 1. For `progress_collect_slurm.sh`, with stub commands in the `SLURM_*`
  variables: a newer mtime copies the XML and records the mtime (exit 0), the same mtime exits 3, a
  copy that fails three times exits 1, and the optional copies are skipped when unset. For both:
  neither starts `curl` (a failing `curl` stub on `PATH`), and neither reads `ART_USER`, `ART_PASS`,
  or `ARTIFACTORY_NETRC` (set them to sentinel values and assert they appear nowhere in the output or
  the tarball).
- [X] T012 PR: Create `jenkins/scripts/progress_snapshot.sh` (NVIDIA SPDX header) from
  `jenkins/scripts/progress_upload_snapshot.sh` per the contract: keep the timeout-file generation,
  the testsuite rename, the content hash (`PROGRESS_HASH_FILE`, "The hash of the content that every
  target last received"), and the `POST_TAG` transform; remove the `curl` upload (`:75-79`), the
  `.upload_ok` marker, `PROGRESS_URL`, and every mention of `ART_USER` and `ART_PASS`; write the new
  hash to `${PROGRESS_HASH_FILE}.pending`; exit 0, 3, or 1 as the contract says. Delete
  `jenkins/scripts/progress_upload_snapshot.sh`. Make the `progress_snapshot.sh` part of T011 pass.
- [X] T013 PR: Create `jenkins/scripts/progress_collect_slurm.sh` (NVIDIA SPDX header) from the
  Slurm branch of the loop in `jenkins/scripts/progress_upload_watcher.sh` (the `SLURM_SSH_STAT_CMD`
  block, `:57` onwards): one pass, the mtime kept in `PROGRESS_MTIME_FILE`, exit 0, 3, or 1 per the
  contract. Delete `jenkins/scripts/progress_upload_watcher.sh`. Make the rest of T011 pass, and run
  `shellcheck` on both new scripts.

### Locations, targets, and uploads in `jenkins/L0_Test.groovy` (PR)

- [X] T014 [P] PR: In `jenkins/L0_Test.groovy`, after `llmArtifactsPrimary()` (`:60-67`), add:
  - `LLM_ARTIFACTS_LOCATIONS`, copied with its comment from `jenkins/Build.groovy:36-44` (`aws`:
    repository and local repository `sw-tensorrt-llm-generic-local`, credential
    `aws-artifactory-credentials`; `urm`: repository `sw-tensorrt-generic`, local repository
    `sw-tensorrt-generic-local`, credential `urm-artifactory-creds`), and the helpers
    `llmArtifactsSecondary`, `llmArtifactsLayoutPath`, `llmArtifactsRepoPath`, `llmArtifactsUrl`,
    `llmArtifactsLocalRepoPath`, `llmArtifactsHost` (`Build.groovy:59-92`), and
    `artifactoryFailureMessage` (`Build.groovy:451-471`);
  - `llmArtifactsBrowseUrl(String location, String legacyPath)`, returning
    `https://artifactory.nvidia.com/ui/native/${llmArtifactsRepoPath('aws', legacyPath)}` for `aws` and
    `llmArtifactsUrl('urm', legacyPath)` for `urm` ([contracts/test-result-locations.md](contracts/test-result-locations.md),
    "Links shown to people");
  - `testResultTargets(String primary)`, returning `[[location: "urm", replica: false], [location:
    "aws", replica: true]]` for `aws` and `[[location: "urm", replica: false]]` for `urm` ("`urm`
    (required), then `aws` (replica)", data-model.md "Targets"), with a comment that this one
    function decides dual publishing;
  - `testResultsDirTarget(String location, String directory)`, returning
    `"${llmArtifactsRepoPath('aws', UPLOAD_PATH)}/${directory}/"` for `aws` and
    `"${UPLOAD_PATH}/${directory}/"` for `urm`;
  - `ATTEMPT_PRIMARIES = [:]` and `attemptPrimary(String stageName, String postTag)`, which stores
    `llmArtifactsPrimary()` under `stageName + postTag` the first time and returns the stored value
    afterwards ("`LLM_ARTIFACTS_PRIMARY` when the attempt starts", data-model.md), and
    `attemptTargets(String stageName, String postTag)`, which returns
    `testResultTargets(attemptPrimary(stageName, postTag))`;
  - `CHECKPOINT_UPLOADS = [:]`, which maps `stageName + postTag` to a map from each location that the
    checkpoint branch tried to upload to, to the hash of the content that location last received
    (null until an upload there succeeds); the delete uses its locations (T027), and the checkpoint
    branch its hashes (T025);
  - `markStageUnstable(String reason)`: `catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
    error(reason) }`, with a comment that the build result must stay as it is because an UNSTABLE
    sub-job counts as infra-incomplete to the parent pipeline (research R7).
- [X] T015 PR: Add `uploadTestResult(String filePattern, String directory, List targets, boolean
  requiredFailureFails = true)` to `jenkins/L0_Test.groovy` per
  [contracts/test-result-locations.md](contracts/test-result-locations.md), "Writing", for final
  archives, the docs preview, and the build-level rerun report; checkpoints have their own uploads
  (T025). For each target in order it calls `trtllm_utils.uploadArtifacts(filePattern,
  testResultsDirTarget(t.location, directory), t.location)`:
  - the required target's upload is the call made today: with `requiredFailureFails`, its exception
    propagates and no replica is tried; without it, the exception is caught (an
    `InterruptedException` is rethrown), logged as `[test-results] upload of <name> to urm failed
    (non-fatal): <exception>`, and counted, and the replica is still tried;
  - a replica's exception is always caught (an `InterruptedException` is rethrown), logged after the
    shared helper's own message as `[test-results] replica upload of <name> to artifactory.nvidia.com
    failed; urm copy: <llmArtifactsUrl('urm', "${UPLOAD_PATH}/${directory}/<name>")>`, and the stage is
    marked with `markStageUnstable` ("best effort; a failed upload is logged with the urm link and
    marks the stage unstable, never the build", data-model.md).
  `<name>` is the last path component of `filePattern`. It returns true when every upload succeeded.
- [X] T016 PR: In `jenkins/L0_Test.groovy`:
  - change `writeArtifactoryNetrc(pipeline, netrcPath)` (`:1796-1810`) to
    `writeArtifactoryNetrc(pipeline, String netrcPath, String location = "aws")`, writing a
    `machine <llmArtifactsHost(location)>` entry with `LLM_ARTIFACTS_LOCATIONS[location].credential`
    and appending (`>>`) as `Build.groovy:409-421` does, so that one file can hold an entry per host.
    Every existing caller writes to a fresh path, so its file does not change;
  - give `fetchBuildArtifact` (`:1821-1883`) a mode `"download-status"` that downloads like
    `"download"` but returns the helper's final status instead of failing the step: the status the
    script records (0, `FETCH_ABSENT`, or another final status), or `FETCH_TRANSFER_FAILED` when
    `llmExecStepWithRetry`'s retries are spent (catch its failure, rethrowing `InterruptedException`).
    Update the comment above it, and keep every existing mode's behavior.

### Targets, uploads, and the reader in `jenkins/L0_MergeRequest.groovy` (PR)

- [X] T017 [P] PR: In `jenkins/L0_MergeRequest.groovy`, after `llmArtifactsUrl` (`:86-89`), add
  `llmArtifactsBrowseUrl`, `testResultTargets`, `testResultsDirTarget`, `uploadTestResult`, and
  `markStageUnstable` with the behavior of T014 and T015 (each pipeline file keeps its own copy,
  research R2), and `FETCH_ABSENT = 2` with the comment of `L0_Test.groovy:73-75`.
- [X] T018 PR: In `jenkins/L0_MergeRequest.groovy`, add:
  - `withTestResultsReader(Closure body)`, which checks the read helper out of the run's commit with
    `trtllm_utils.checkoutFile(LLM_REPO, env.gitlabCommit, "jenkins/scripts/fetch_build_artifact.sh",
    <dir>)`, as `L0_Test.groovy:7713` does; writes a netrc for `aws-artifactory-credentials` as
    `_cbtsCoverageAudit` does (`:1539-1551`); calls `body(<helper path>)` inside
    `withEnv(["LLM_ARTIFACTS_PRIMARY=${llmArtifactsPrimary()}", "ARTIFACTORY_NETRC=<path>",
    "ARTIFACTORY_CREDENTIAL_ID=aws-artifactory-credentials"])`; and removes the netrc in `finally`;
  - `readRunTestResults(String helper, Closure keep)` per
    [contracts/test-result-locations.md](contracts/test-result-locations.md), "Listing": run
    `bash <helper> --list "${UPLOAD_PATH}/test-results"` with `returnStatus` and its stdout in a file;
    keep the entries whose name `keep` accepts; merge them by name; download each name once with
    `bash <helper> "${UPLOAD_PATH}/test-results/<name>" "<name>"`; log `[test-results] listed <p> on
    <primary> and <s> on <secondary>: <n> names, <x> only on <primary>, <y> only on <secondary>; read
    <r> of <n>`; and call `markStageUnstable` when the listing's status is neither 0 nor
    `FETCH_ABSENT`, or when `r < n` (FR-006, FR-007). Compare the status with `FETCH_ABSENT` by name,
    never with its number (T023). Return the names it read.

### Delete container (OPS)

- [X] T019 OPS: The draft pull request exists: #21223, opened on 2026-10-28 from this branch with the
  design documents, and stacked on TRTLLM-17014 (#20494). Put a temporary step in this work's commit:
  in `runLLMTestlistOnPlatformImpl` in `jenkins/L0_Test.groovy` (`:5340`),
  `container('jnlp') { sh 'command -v curl && curl --version | head -1 || echo "no curl in jnlp"' }`.
  Amend the commit and push it with an explicit lease, after fetching the branch.
- [X] T020 OPS: Start one targeted run of a single pod stage (`/bot run --stage-list "<one A10
  stage>"`) and read the probe's output in its log. Ran on 2026-10-29 with `A10-PyTorch-1`: the
  probe printed `/usr/bin/curl` and `curl 7.88.1`.
- [X] T021 OPS: Record the result in `signoff-evidence.md` ("delete container"), and remove the
  temporary step from the commit (amend, push with a lease). If `jnlp` has `curl`, pod stages delete
  from `jnlp` (T027); otherwise T028 adds the sidecar and pod stages delete from it. Both `jnlp`
  images have `curl` (amd64 by this probe, arm64 by its image history; research R4), so pod stages
  delete from `jnlp` and T028 is not needed.

**Checkpoint**: the helper lists folders, the checkpoint scripts prepare without network or
credentials, and both pipeline files can name locations, targets, and links, upload to targets, and
read a run's test results.

---

## Phase 3: User Story 1 - Test results stay complete in CI (Priority: P1) 🎯 MVP

**Goal**: every test stage publishes its test results to the new location, and to urm as today for
everything under `test-results/`; every CI step that reads them finds all of them (spec FR-001 to
FR-008).

**Independent Test**: run CI on the switch pull request; for every stage, the archives it published
and those Collect Test Result found and read match one for one; every file under the run's
`test-results/` is on both locations with the same SHA-1, and its `rerun_reports/` and
`cbts-coverage/` files are on the new location only ([quickstart.md](quickstart.md) step 3).

### Tests for User Story 1

- [X] T022 [P] [US1] PR: In `tests/unittest/scripts/cbts/test_coverage_artifact.py`, replace the
  `_exists` and URL mocks for coverage archives with tests of T036's helper-based reads, using the
  helper stub and fixtures of TRTLLM-17014's probe tests
  (`test_probe_reads_each_build_once_and_declines_while_every_transfer_fails`, `:599`, and the tests
  after it):
  - an archive on aws and one only on urm are both found, and the selection's `urls` are the URLs that
    `--exists` printed;
  - exit 2 for one archive of a pair makes that build incomplete, and the probe moves to an older
    build;
  - exit 4 from `--exists` for one build's archive skips that build with `coverage archive
    unreadable; skipped` and selects an older complete build, as
    `test_probe_selects_an_older_build_when_one_transfer_fails` (`:624`) does for `build_info.txt`;
    exit 4 for every build declines, reporting how many builds were skipped;
  - exit 3 raises `ArtifactAuthError` after one call, as
    `test_probe_stops_when_the_primary_location_rejects_a_read` (`:644`) does, and `main` exits 1;
  - `prepare` downloads each archive with the helper by its legacy path; a download that exits 4
    moves to the next older complete build; for a pinned build it declines, as
    `test_prepare_of_the_pinned_build_exits_1_when_its_transfer_fails` (`:684`) does;
  - with the real helper, an archive only on urm is found through the fallback, like
    `test_build_commit_with_the_real_helper_falls_back_to_urm` (`:748`).
  Keep the pin tests reading urm. See them fail before T036.
- [X] T023 [P] [US1] PR: In `tests/unittest/scripts/test_fetch_build_artifact.py`, extend
  `EXIT_CODE_READERS` (`:1169`): add `jenkins/L0_MergeRequest.groovy` with `_GROOVY_DEFINITION`
  (`:1165`) and a `checks` that picks `readRunTestResults` with `_braced_block` (`:1116`); add a second
  `jenkins/L0_Test.groovy` reader whose `checks` picks `reusePassedTestResults`; add the new coverage
  functions of `artifact.py` (T036) to its `_python_definitions` (`:1135`) set. See them fail until
  T017, T018, T030, and T036 define and use the names.

### Implementation for User Story 1: writing in `jenkins/L0_Test.groovy`

The tasks on `jenkins/L0_Test.groovy` run in order: T024 → T025 → T026 → T027 → T028 (only if
needed) → T029 → T030 → T031 → T032.

- [X] T024 [US1] PR: In `jenkins/L0_Test.groovy`, upload the final archive with
  `uploadTestResult("results-${stageName}${postTag}.tar.gz", "test-results", attemptTargets(stageName,
  postTag))` in `cacheErrorAndUploadResult`'s `finally` (`:3917-3920`) and in `uploadResults`
  (`:434-437`). A failed urm upload still fails the stage, and a failed replica marks it unstable
  (research R7).
- [X] T025 [US1] PR: Add `runWithCheckpoints(pipeline, Map checkpoint, Closure tests)` to
  `jenkins/L0_Test.groovy` per [contracts/checkpoint-scripts.md](contracts/checkpoint-scripts.md),
  "The checkpoint branch":
  - `checkpoint` holds `stageName`, `postTag`, `workspace`, `progressTar`, `label`, the scripts'
    folder, `timeoutXmlScript`, and for batch stages the `SLURM_*` commands, an `mtimeFile`, and a
    `finalCollect` closure that runs before the final snapshot;
  - it runs `parallel("Run tests": { try { tests() } finally { testsDone = true } }, "Upload
    checkpoints": { ... })`;
  - the checkpoint branch loops with `waitUntil(quiet: true)` on `testsDone` or the next due time
    (`PROGRESS_UPLOAD_INTERVAL_SEC`, `:262`, measured with `System.currentTimeMillis()`); when due, it
    runs `progress_collect_slurm.sh` for batch stages, then `progress_snapshot.sh` with `returnStatus:
    true`. On 0 it reads the new hash from `${PROGRESS_HASH_FILE}.pending`, and for each target of
    `attemptTargets(stageName, postTag)` in order whose hash in `CHECKPOINT_UPLOADS[stageName +
    postTag]` differs from it, it records the location there and calls
    `trtllm_utils.uploadArtifacts("${workspace}/${progressTar}", testResultsDirTarget(location,
    "test-results"), location)`, storing the new hash for that location when the upload succeeds.
    Once every target holds the new hash, it moves the pending hash over the hash file ("judged for
    each location against the content that location last received", data-model.md; spec FR-004);
  - after the `parallel`, in `finally`, it prepares the final snapshot (`FINAL_SNAPSHOT=1`,
    `TIMEOUT_XML_SCRIPT`) and uploads it the same way;
  - every failure in the branch, on either location, is caught, logged with `[PROGRESS-UPLOAD]
    ${stageName}:` after the shared helper's own message, and never fails or marks the stage ("A
    failed checkpoint upload or delete, on either location, is logged and marks nothing",
    data-model.md); `InterruptedException` is rethrown; no `withCredentials` for Artifactory anywhere
    in it.
- [X] T026 [US1] PR: Use `runWithCheckpoints` for pod and Slurm-agent stages in
  `runLLMTestlistOnPlatformImpl` in `jenkins/L0_Test.groovy` (`:5626-5674`): remove `progressUrl`,
  the `withCredentials(urm-artifactory-creds)` block, the background watcher, the `touch`/`wait`, and
  the inline final snapshot. The `tests` closure runs `rm -rf '${stageName}/'`, `cd`, and the pytest
  command in one `sh` step and fails it on a non-zero exit as today, so that the existing `catch`
  (`:5689` onwards, the rerun logic) still runs. Do the same for each rerun pass in `rerunFailedTests`
  (`:4877-4914`), with label `rerun${times} checkpoint`, keeping its `catch` (`:4915-4929`).
- [X] T027 [US1] PR: In `jenkins/L0_Test.groovy`, rewrite `deleteProgressArtifact` (`:5219-5256`) as
  `deleteProgressArtifact(pipeline, String stageName, String postTag)` per
  [contracts/test-result-locations.md](contracts/test-result-locations.md), "Deleting":
  - it deletes on each location in `CHECKPOINT_UPLOADS[stageName + postTag]`, returns when there is
    none, and removes the entry afterwards, so that a second call for the attempt does nothing;
  - one request per location, `DELETE <root>/${llmArtifactsLocalRepoPath(location,
    "${UPLOAD_PATH}/test-results/results-${stageName}${postTag}-progress.tar.gz")}`, sent as
    `artifactoryRequestStatus` sends one (`Build.groovy:426-449`), with a netrc that holds an entry
    for each location (`writeArtifactoryNetrc(pipeline, <path>, location)` per location) in a fresh
    `mktemp -d` directory of the current container, outside the workspace and its `@tmp`, which a
    pod's `trt-llm` container also mounts;
  - outcomes per location: 200 or 204 deleted, 404 absent, 401 or 403
    `artifactoryFailureMessage(location, <path>, status, true, "delete")`, anything else logged; the
    directory removed in `finally`; the stage never failed or marked (FR-004).
  Then place the calls:
  - `cacheErrorAndUploadResult` (`:3826`, call at `:3921`) gains a trailing `boolean
    deferCheckpointDelete = false`; when false it calls the delete inside `container('jnlp')`, and when
    true it leaves it;
  - `executeLLMTestOnSlurm` (`:1585`) passes true, and `runLLMTestlistWithAgent`'s `finally` (`:1550`)
    calls the delete on the dispatcher pod;
  - `uploadResults` (`:438`) keeps its call, on the dispatcher pod; remove the duplicate call at
    `:2988`.
- [X] T028 [US1] PR (only if T021 found no `curl` in `jnlp`): Not needed: T021 found `curl` in
  `jnlp`, so no sidecar was added. As planned: in `createKubernetesPodConfig` in
  `jenkins/L0_Test.groovy` (`:3940` onwards), add to every test pod type a container
  `artifactory-client`, from the same mirror as the `alpine` agent image, with `curl` (installed by its
  command or present in the image), command `cat`, `tty: true`, and no volume besides the default
  workspace; make T027's pod-stage delete run in `container('artifactory-client')`.
- [X] T029 [US1] PR: Use `runWithCheckpoints` for Slurm batch stages in `runLLMTestlistWithSbatch` in
  `jenkins/L0_Test.groovy` (`:2816-2908`): remove `progressUrl` (`:2818`), the
  `withCredentials(urm-artifactory-creds)` block (`:2841`), and the background watcher
  (`:2852-2863`). The `tests` closure runs the foreground track loop (`:2865-2874`) in its own `sh`
  step; the checkpoint map carries the `SLURM_*` commands already built above it (`sshStatCmd`
  `:2834`, `scpXmlCmd` `:2835`, and the rest) and an `mtimeFile` in the workspace; its `finalCollect`
  closure runs the immediate copies (`:2879-2903`: the XML files, `unfinished_test.txt`, and the perf
  folders, with today's retries) unchanged in a credential-free `sh` step, before the final snapshot.
  The verdict handling after the block (`:2915` onwards) is unchanged.
- [X] T030 [US1] PR: Make the retry read in `reusePassedTestResults` in `jenkins/L0_Test.groovy`
  (`:5116-5218`) read through the helper. Give it a leading `pipeline` parameter and pass it from both
  callers (`:2107`, `:5540`). For each prior tag, call `fetchBuildArtifact(pipeline,
  "${UPLOAD_PATH}/test-results/${tarName}", "${priorDir}/${tarName}", "download-status")`.
  `FETCH_ABSENT` skips the attempt, as the 404 does today; any other non-zero status logs `Reading the
  archive of attempt ${priorTag} failed (fetch_build_artifact.sh exited ${status}, see its message
  above); its passed tests are not reused`, calls `markStageUnstable`, and continues with the other
  attempts and the test history. Remove the anonymous `curl` probe and the `wget` (`:5145-5163`).
  Compare the status with `FETCH_ABSENT` by name (`:76`), never with its number (T023).
- [X] T031 [US1] PR: In `generateRerunReport` in `jenkins/L0_Test.groovy` (`:4936`), add a `postTag`
  parameter and pass it from the caller (`:5773`). Upload `rerun_results.html` (`:5060-5063`) to the
  attempt's primary only: `def primary = attemptPrimary(stageName, postTag)`, then
  `trtllm_utils.uploadArtifacts("${WORKSPACE}/${stageName}/rerun_results.html",
  "${testResultsDirTarget(primary, 'rerun_reports')}${stageName}_rerun_results.html", primary)`. A
  failed upload still fails the stage, as today (research R7).
- [X] T032 [US1] PR: In `runLLMDocBuild` in `jenkins/L0_Test.groovy` (`:4382-4385`), upload
  `doc-html-preview.tar.gz` with `uploadTestResult("doc-html-preview.tar.gz", "test-results",
  testResultTargets(llmArtifactsPrimary()))`; a failed urm upload still fails the stage.

### Implementation for User Story 1: reading and writing in `jenkins/L0_MergeRequest.groovy`

The tasks on `jenkins/L0_MergeRequest.groovy` run in order: T033 → T034 → T035.

- [X] T033 [US1] PR: Rewrite the download part of the "Collect Test Result" stage in
  `collectTestResults` in `jenkins/L0_MergeRequest.groovy` (`:2123-2134`): `apk add --no-cache curl
  bash`; `withTestResultsReader { helper -> readRunTestResults(helper, { it ==~ /results-.*\.tar\.gz/
  }) }`; remove `testResultLink`, the `wget` of the HTML index and of each file, and the old counts.
  The netrc is gone before `trtllm_utils.checkoutSource` (`:2137`) and the pull request's scripts run.
  Extraction, checkout, `junit`, and the parallel tasks are unchanged.
- [X] T034 [US1] PR: In `uploadArchCoverage` in `jenkins/L0_MergeRequest.groovy` (`:2057-2113`),
  replace `testResultLink` and the downloads (`:2070`, `:2074-2077`) with `withTestResultsReader` and
  `readRunTestResults`, keeping the `results-*.tar.gz` names that `archGrepCmd`'s prefix pattern
  selects for the architecture (SBSA names match `^results-(GH200|GB10|GB200|GB300|CPU-Generic-arm)-`,
  x86_64 names do not); add `bash` to its `apk add`. Upload `cbts_pystart_report_${arch}.tar.gz`
  (`:2092`) to the primary alone, as the waive-list upload does (`:679-683`): `def primary =
  llmArtifactsPrimary()`, target `testResultsDirTarget(primary, "cbts-coverage")`, destination
  `primary`. The step stays non-fatal.
- [X] T035 [US1] PR: In `collectTestResults` in `jenkins/L0_MergeRequest.groovy`, upload
  `rerun/rerun_report.html` (`:2174`) with `uploadTestResult("rerun/rerun_report.html",
  "test-results", testResultTargets(llmArtifactsPrimary()), false)` inside the existing `try`
  (`:2173-2178`), so that a failed urm upload stays logged-only, and upload
  `cov/cbts_pystart_report.tar.gz` (`:2219`) to the primary alone, as in T034.

### Implementation for User Story 1: coverage-based test selection

- [X] T036 [US1] PR: In `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`, read
  coverage archives through the helper, as `build_commit` (`:208`) reads `build_info.txt` (spec
  FR-007):
  - add `_coverage_archive_location(build, name, artifact_base)`, which runs `bash
    _FETCH_BUILD_ARTIFACT --exists <artifact_base>/<build>/cbts-coverage/<name>` and returns the URL
    of the `<url> <sha1>` line on 0, None on `_FETCH_ABSENT`, and raises `ArtifactTransferError` on
    `_FETCH_TRANSFER_FAILED` and `ArtifactAuthError` on `_FETCH_REJECTED`, mapping statuses as
    `_fetch_build_artifact` (`:174`) does;
  - add `_download_coverage_archive(build, name, dest_dir, artifact_base)`, which runs the helper with
    a `<dest-file>` in `dest_dir` and maps its statuses the same way;
  - in `select_tarball` (`:568-569`), skip a build whose archive is absent or whose read raises
    `ArtifactTransferError`, logging `[artifact] build <b>: coverage archive unreadable; skipped` for
    the latter and reporting the count with the final message, as the `commit unknown` skips are
    reported; let `ArtifactAuthError` propagate, so `main` exits 1 and selection declines; give
    `select_tarball` a `start_build` argument, the build to probe first (default: the latest);
  - in `select_build` (`:619-620`), the pinned build has no older candidate: a transfer failure
    declines (returns None), and `ArtifactAuthError` propagates;
  - keep in the selection's `url` and `urls` the URLs that `--exists` returned, and add `paths`, the
    archives' legacy paths;
  - in `prepare` (`:752-760`), download each of `sel["paths"]` with `_download_coverage_archive`
    instead of `download(url, temp)`. When a download raises `ArtifactTransferError`, skip that build
    too: for a probed selection, continue with `select_tarball(pr_base_commit, start_build=<build -
    1>)` within the same `_MAX_PROBE` window, and repeat the compatibility check, the meta file, and
    the downloads for the new selection; for a pinned build, decline;
  - remove `_exists` (`:134`), `tarball_url`/`tarball_urls` (`:159-164`), and `download` (`:674`) if
    nothing else uses them; keep `_URM` for the pins; update the module comment (`:84-87`).
  Make T022 pass.

### Validation for User Story 1

- [ ] T037 [US1] OPS: Run the unit tests of [quickstart.md](quickstart.md) step 1 locally
  (`tests/unittest/scripts/test_fetch_build_artifact.py`, `test_progress_snapshot.py`, and
  `cbts/test_coverage_artifact.py`) and the shell and Groovy checks; fix failures before pushing.
- [X] T038 [US1] PR (draft only): In `readRunTestResults` in `jenkins/L0_MergeRequest.groovy`, compare
  the two copies (research R13 item 3): for each name listed on both locations, run `bash <helper>
  --exists "${UPLOAD_PATH}/test-results/<name>"` once with `LLM_ARTIFACTS_PRIMARY=aws` and once with
  `LLM_ARTIFACTS_PRIMARY=urm`, compare the SHA-1s they print, log `[test-results] copies: <m> of <k>
  names match on both locations` and each mismatch, and call `markStageUnstable` on a mismatch.
- [ ] T039 [US1] OPS: On the draft, start the default pre-merge run (`/bot run --disable-fail-fast` on
  the first run of a newly pushed commit, plain `/bot run` for re-runs) and check
  [quickstart.md](quickstart.md) step 3: every `test-results/` file on both locations and T038's
  comparison matching for every name; `rerun_reports/` files on the new location only; the checkpoint
  branch sending each snapshot, urm first, only to the locations that lack it, and deleting on both;
  Collect Test Result's `[test-results] listed ...` line with no name on only one location and every
  name read; coverage-based selection reading coverage archives with `served-by=urm`; native-browser
  links. Record in `signoff-evidence.md`.
- [ ] T040 [US1] OPS: List each combination of cluster, container runtime, and launch mode that the
  stage configurations in `jenkins/L0_Test.groovy` use; for each one that T039 did not reach, add one
  post-merge stage by exact name to a `--stage-list` run (quickstart step 4). Expect uploads to both
  locations and deletes on both. Record per combination.
- [ ] T041 [US1] OPS: Add the draft-only changes of quickstart step 5 for reruns and retries: a test
  in `tests/integration/defs/test_rerun_once_probe.py` (new, NVIDIA SPDX header) that fails when a
  marker file under `/tmp` is absent (creating it) and passes when it exists, listed in one pod
  stage's and one Slurm stage's test-db entry; and, in `runLLMTestlistOnPlatformImpl`
  (`jenkins/L0_Test.groovy:5340`), `throw new InfraFailure("draft-only forced retry", null,
  InfraFailure.TRANSIENT, InfraFailure.K8S, "<typed:draft-forced-retry>")` after the tests and the
  final upload of one chosen pod stage's first attempt. Run them; check the per-stage rerun reports on
  the new location only, the build-level `rerun_report.html` on both, and the retry's `served-by=aws`
  and reuse. Record; keep the changes for T068 to remove.
- [ ] T042 [US1] OPS: Check a failed replica (quickstart step 5): in `testResultsDirTarget` in
  `jenkins/L0_Test.groovy`, temporarily return a target in `sw-tensorrt-llm-draft-missing-local`, a
  repository that does not exist, for `aws`. Push it, run one pod stage and one Slurm stage with
  `--stage-list`, then remove the change and push again before any other run, since it fails every
  replica while it is in the commit. Check that the stages' urm uploads succeeded; the failed
  checkpoint uploads are only logged; each failed final archive is logged with the credential or
  upload message and the urm link and marks its stage unstable; the sub-jobs' build results are what
  they would otherwise have been, and the parent does not report them as infra-incomplete; Collect
  Test Result reads those archives from urm and counts them as only on urm. Record.
- [ ] T043 [US1] OPS: Rehearse the rollback and check branch isolation (quickstart step 8) with the CI
  administrators. With `LLM_ARTIFACTS_PRIMARY=urm` on the controllers, a run of the draft writes test
  results, rerun reports, and coverage archives to urm only, with no replica; deletes on
  `sw-tensorrt-generic-local` only; prints direct urm links; and Collect Test Result lists both
  locations and reads everything from urm. Record how long after the change the first stage attempt
  published to urm only, against SC-011's one hour. Set it back to `aws` while a second run of the
  draft is in progress: the jobs that run started before the change keep publishing to urm, the jobs
  it starts afterwards and its own steps follow the new setting, and Collect Test Result reads all of
  them (FR-008). A run without the switch still writes to urm only. Record in `signoff-evidence.md`.

**Checkpoint**: User Story 1 works on the draft: test results are written to both locations as the
targets say, and listed and read by CI's own steps, with a failed replica leaving the run complete.

---

## Phase 4: User Story 2 - Access proven without exposing the token (Priority: P1)

**Goal**: every environment that writes, replaces, deletes, lists, or reads test results does so with
the publishing identity, proven before cutover, with no Artifactory credential for either location
where tests run (FR-009 to FR-011, FR-019).

**Independent Test**: the access check succeeds in every writing environment, and anonymous requests
are rejected; the credential probe finds nothing; no token appears in logs, on command lines, or in
archives ([quickstart.md](quickstart.md) steps 2 and 5).

### Tests for User Story 2

- [ ] T044 [P] [US2] PR (draft only): Write `tests/unittest/scripts/test_check_artifactory_access.py`
  (new, NVIDIA SPDX header) against the access check of T045, with the local servers of
  `test_fetch_build_artifact.py`: each operation (create, overwrite, list, read, delete) is reported;
  a 401 or 403 names the credential; the anonymous checks pass when the server rejects them;
  `--probe-redacted` reports present (200), absent (404), and undetermined (403) without downloading;
  and no secret reaches the output.

### Implementation for User Story 2

- [ ] T045 [US2] PR (draft only): Add `jenkins/scripts/check_artifactory_access.sh` (NVIDIA SPDX
  header) from the copy that TRTLLM-17014 kept with the internal documentation, extended with:
  - a probe under `llm-artifacts/<probe job>/<probe build>/test-results/`, `/rerun_reports/`, and
    `/cbts-coverage/`: create, overwrite, list (storage API), read, and delete each, then check that
    an anonymous listing and read of the probe's `test-results/` are rejected;
  - `--keep-coverage-probe`, which leaves the `cbts-coverage/` probe for T061;
  - `--probe-redacted <path in sw-tensorrt-llm-generic-local>`, which asks the new instance's storage
    API for that path with the publishing identity and reports present, absent, or undetermined.
  Make T044 pass; run `shellcheck` on it.
- [ ] T046 [US2] OPS: Run the access check (quickstart step 2) in a test pod's `trt-llm` and `jnlp`
  containers, in an `alpine` pod, on a Slurm dispatcher pod, and on a compute node of each remote
  cluster where test stages run (through `srun` from the login node, the netrc copied in and removed).
  Record each environment's result, and T003's answer, in `signoff-evidence.md`.
- [ ] T047 [US2] PR (draft only): Add the credential probe of quickstart step 5:
  `tests/integration/defs/test_no_artifactory_credentials.py` (new, NVIDIA SPDX header), which for
  about three minutes checks every 15 seconds that no environment variable named `ART_USER`,
  `ART_PASS`, `ARTIFACTORY_USER`, or `ARTIFACTORY_PASSWORD` is set, and that no readable file under the
  workspace, its `@tmp`, `$HOME`, or `/tmp` holds a netrc entry for artifactory.nvidia.com or
  urm.nvidia.com. List it in one pod stage's, one Docker-runtime agent stage's, and one ENROOT-runtime
  agent stage's test-db entry, and set `PROGRESS_UPLOAD_INTERVAL_SEC = 60`
  (`jenkins/L0_Test.groovy:262`) in the draft so that the probe spans checkpoint uploads to both
  locations.
- [ ] T048 [US2] OPS: Run the three stages of T047 with `--stage-list`; confirm that the probe passed
  and that each stage uploaded checkpoints to both locations meanwhile. Scan the run's console logs and
  a sample of its archives' `debug_env.txt` for `ART_PASS`, `ARTIFACTORY_PASSWORD`, `--netrc`, `-u `,
  `machine artifactory.nvidia.com`, `machine urm.nvidia.com`, and `Authorization`, expecting only the
  masked or path forms. Record in `signoff-evidence.md`; keep the changes for T068 to remove.

**Checkpoint**: access proven in every environment, and no credential reachable by tests.

---

## Phase 5: User Story 3 - The required check and PR comments keep working (Priority: P2)

**Goal**: `blossom-ci` reports each run's test outcome and the PR comment links the rerun report:
from urm's copies until PR_Github's change, and from either location after it (FR-012).

**Independent Test**: before PR_Github's change, a failed and a passing run of the switch get the
status and comment they got before; after it, so do runs on either location
([quickstart.md](quickstart.md) step 6, [contracts/pr-github.md](contracts/pr-github.md)).

### Before the merge

- [ ] T049 [US3] OPS: Confirm that PR_Github's redacted copies are not on the new instance ("Never on
  the new instance, where it would replace originals; the check before the merge confirms it
  (FR-019)", data-model.md). Take a recent failed `L0_MergeRequest_PR` run whose PR_Github build
  published redacted copies, and run `check_artifactory_access.sh --probe-redacted
  llm-artifacts/LLM/main/L0_MergeRequest_PR/<build>/test-results.tar.gz` from a pod with the
  publishing identity: that combined archive is a path no original uses. Absent: confirmed. Present:
  a blocker under FR-019; raise it through the program's support channel and with the spec's author.
  PR_Github's path change (T053) resolves it and then becomes a prerequisite of the merge.
  Undetermined: read the Artifactory server that PR_Github's scan step uploads to from its
  configuration. Record in `signoff-evidence.md`.
- [ ] T050 [US3] OPS: From the switch pull request's runs, collect quickstart step 6's evidence that
  PR_Github works unchanged from urm's copies (FR-019): a failed run, where PR_Github's log shows its
  reads from urm and `blossom-ci` shows the test summary that the run's JUnit report implies; and a
  passing run with reruns (T041), whose comment links `rerun_report.html` on urm, and the link opens.
  If no run of the draft fails on its own, add a draft-only
  `tests/integration/defs/test_draft_always_fails.py` (new, NVIDIA SPDX header) to one stage's test-db
  entry for one run, and keep it for T068 to remove. Record in `signoff-evidence.md`.

### PR_Github's change (any time before urm is retired)

- [ ] T051 [US3] OPS: Ask the owners of the CI credential store to make `aws-artifactory-credentials`
  usable by PR_Github's job; record in `signoff-evidence.md`.
- [ ] T052 [US3] PRG: In PR_Github's scan step, read a failed run's test results per
  [contracts/pr-github.md](contracts/pr-github.md), "Reading a failed run's test results": list
  `llm-artifacts/<job>/<build>/test-results/` on the new location with `aws-artifactory-credentials`
  and on urm, merge the entries by name, and read each name once new-first by the read rule with its
  SHA-1 check and the content-delivery network rule, in PR_Github's own tooling, never with the pull
  request's `fetch_build_artifact.sh`.
- [ ] T053 [US3] PRG: In the scan step, publish redacted copies under
  `redacted/llm-artifacts/<job>/<build>/test-results/results-<stage>.tar.gz` and
  `redacted/llm-artifacts/<job>/<build>/test-results.tar.gz` (or another top-level directory outside
  `llm-artifacts/`; "its change gives them a path of their own (FR-012)", data-model.md); change the
  release bundle's query to that path, and the `test_results_url` handed to
  `.github/workflows/l0-test.yml` to the edge path that follows. Leave older copies in place.
- [ ] T054 [US3] PRG: In PR_Github's pipeline, per [contracts/pr-github.md](contracts/pr-github.md),
  "Passing runs": fetch `rerun_report.html` after a passing run by the read rule with the publishing
  credential, report an absent report without retrying, and link the report in the PR comment in the
  native-browser form on the new instance, or directly on urm when only urm has it.
- [ ] T055 [US3] OPS: When T052–T054 have landed, collect the evidence of
  [contracts/pr-github.md](contracts/pr-github.md), "Evidence when the change lands", from runs that
  published to the new location (the draft's runs, or any run after cutover): a failed run (reads
  served by the new location, redacted copies at the new path, `blossom-ci` with the test summary,
  the originals' SHA-1s on both locations unchanged) and a passing run with reruns (the comment's link
  opens). Also collect a failed and a passing run that published to urm only, such as release-branch
  pull requests or pull requests not rebased past the switch: PR_Github reads both through the
  fallback, and `blossom-ci` and the comment are as before (spec User Story 3). Record in
  `signoff-evidence.md`.

**Checkpoint**: the required check works from urm's copies at the merge, and from either location
once PR_Github's change lands.

---

## Phase 6: User Story 4 - People and tools can still find test results (Priority: P3)

**Goal**: links lead to test results wherever they are and open with the company sign-in; outside
readers' owners know what changes and until when urm keeps their test results (FR-013 to FR-016).

**Independent Test**: a developer outside the infrastructure team opens each kind of link with only
the company sign-in; a perf record written after cutover links the run's test results
([quickstart.md](quickstart.md) step 7).

### Tests for User Story 4

- [X] T056 [P] [US4] PR: Write `tests/unittest/others/test_perf_regression_artifact_url.py` (new,
  NVIDIA SPDX header) for `get_job_info()` in `tests/integration/defs/perf/perf_regression_utils.py`:
  with `globalVars` carrying `llm_artifacts_primary` `aws`, `urm`, or nothing, and a pre-merge or
  post-merge parent, `artifact_url` is
  `https://artifactory.nvidia.com/ui/native/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/<job>/<id>/test-results/`
  for `aws` and for nothing, and
  `https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/<job>/<id>/test-results/`
  for `urm`; an empty job id gives an empty link.

### Implementation for User Story 4

- [X] T057 [US4] PR: In `tests/integration/defs/perf/perf_regression_utils.py`, replace `_URM_BASE`
  (`:36`) with a base per location and build `artifact_url` (`:98-100`) from
  `global_vars.get("llm_artifacts_primary") or "aws"`, pointing to the run's `test-results/` folder in
  the forms of T056. Make T056 pass.
- [X] T058 [US4] PR: In `jenkins/L0_Test.groovy`, print links with `llmArtifactsBrowseUrl` and the
  attempt's primary: the Slurm track script's full-output pointer (`:2787-2788`) with
  `attemptPrimary(stageName, postTag)`, and the per-stage rerun report (`:5064`) with the primary of
  T031. Links point to the new location while dual publishing is on, also for files that urm holds
  too (FR-013).
- [X] T059 [P] [US4] PR: In `jenkins/L0_MergeRequest.groovy`, print the coverage-archive links
  (`:2093`, `:2220`) and the build-level rerun report link (`:2175`) with `llmArtifactsBrowseUrl` and
  `llmArtifactsPrimary()`.
- [ ] T060 [US4] OPS: Once T004's grant is in place, keep a coverage probe with
  `check_artifactory_access.sh --keep-coverage-probe` for the check below.
- [ ] T061 [US4] OPS: Have a developer outside the infrastructure team, with only the company
  sign-in, open one link of each kind from the switch pull request's runs (quickstart step 7): a Slurm
  stage's full-output archive, a per-stage rerun report, the build-level rerun report as the console
  prints it, the docs preview, a perf record's link, and T060's probe. Record whether each opened, and
  whether HTML reports render or download; if file links do not open, change the link helpers to
  point at folders (research R8) and repeat. Remove the probe. Record in `signoff-evidence.md`. The
  switch does not merge before this passes.
- [ ] T062 [US4] OPS: Send the notices (FR-015, FR-016) at least a week before the planned merge: to
  the owners found in T007, saying what changes, that urm keeps every file under `test-results/` until
  its retirement date while `rerun_reports/` and `cbts-coverage/` leave it at the merge, and what
  access the new location needs; to the TRT-LLM developers, saying that test-result links move to the
  new instance and open with the company sign-in, and that command-line downloads need an Artifactory
  access token, with a pointer to the internal documentation. Record the dates in
  `signoff-evidence.md`.
- [ ] T063 [US4] OPS: Update the internal documentation that TRTLLM-17014 created for access (outside
  this repository): add test results, dual publishing until urm's retirement, the native-browser link
  form, and the command-line token steps. Record the date in `signoff-evidence.md`.

**Checkpoint**: people and outside tools know where test results are, and the links open.

---

## Phase 7: User Story 5 - No in-scope reference to the old location remains (Priority: P3)

**Goal**: every reference to the old location for in-scope test results is repointed, needed by dual
publishing, the fallback, or the rollback, or allowlisted, and the program signs off (FR-022,
FR-023).

**Independent Test**: a search of main after the merge finds only references that dual publishing,
the fallback, or the rollback needs, and allowlisted ones.

- [ ] T064 [US5] OPS: On the final commit, run `git grep -n -E
  'urm\.nvidia\.com/artifactory/\$\{UPLOAD_PATH\}|urm-artifactory-creds|test-results|rerun_report|cbts-coverage|progress\.tar\.gz|progress_upload'`
  over `jenkins/`, `tests/`, `scripts/`, and `docs/`. Classify each match: repointed; needed by dual
  publishing (`testResultTargets`, urm's entry in `LLM_ARTIFACTS_LOCATIONS`, urm's checkpoint
  deletes); needed by the fallback or the rollback (the helper's urm base); out of scope with a reason
  (the coverage-selection pins, the image-tag map, other urm uploads); or allowlisted with a reason.
  Record the list in `signoff-evidence.md`.
- [ ] T065 [US5] OPS: After T074 and T075, submit the evidence in `signoff-evidence.md` through the
  program's support channel for sign-off, recording what remains before urm's retirement: PR_Github's
  change if T055 is still pending, outside readers not yet moved, and the follow-up change (T078).
  Record the outcome (SC-010).

**Checkpoint**: references are accounted for, and sign-off is requested.

---

## Phase 8: Polish & Cross-Cutting Concerns (Cutover, Validation, End of Dual Publishing)

- [ ] T066 OPS: Wait until TRTLLM-17014 (#20494) has merged and passed its validation after cutover
  (FR-017); then fetch the fork and `upstream`, compare the fork's branch with the local one
  (`git range-diff`), and rebase this branch onto `upstream/main` with the noreply address as
  committer, so that the pull request holds only this work's commit. Record the new base in
  `signoff-evidence.md`.
- [ ] T067 PR: After the rebase, re-run the unit tests of [quickstart.md](quickstart.md) step 1 and
  the shell and Groovy checks on `jenkins/L0_Test.groovy` and `jenkins/L0_MergeRequest.groovy`; fix
  any conflict fallout, finding moved code by its content.
- [ ] T068 PR: Remove the draft-only files and changes from the commit: the spec store,
  `check_artifactory_access.sh` and its tests (T044, T045; the copy stays with the internal
  documentation), the copy comparison (T038), the rerun probe and forced retry (T041), the credential
  probe and the interval change (T047), and T050's failing test if it was added. T042's change is
  already gone.
- [ ] T069 OPS: Pass the default pre-merge run on the final commit under TRTLLM-17014's rules: a
  significant code change needs a new run (removing the changes of T068 counts as one), other
  amendments carry the pass forward with `/bot reuse-pipeline`, and the run counts once `blossom-ci`
  passes through its test stages. Record the passing run in `signoff-evidence.md`.
- [ ] T070 PR: Write the pull request's description per FR-024, drawing on
  `specs/001-migrate-test-results-aws-artifactory/plan.md` and `signoff-evidence.md`, whose internal
  names stay out of it: the new location; that people open test-result links there with the company
  sign-in through the program's read grant, while command-line downloads need an Artifactory access
  token; that everything under `test-results/` is still published to urm, as today, and replicated to
  the new location until urm is retired, with urm's upload still the one that fails a stage, a failed
  replica marking only its stage unstable, and failed checkpoint uploads only logged; how the
  checkpoint branch keeps credentials from the tests, the Blue Ocean change, and the residual risk of
  research R3; that PR_Github keeps working from urm until its own change, which lands before then;
  coverage-based selection now reading coverage archives from the new location first and skipping a
  run whose archive cannot be read; the outside readers whose owners were told; the rollback switch;
  the test list; the follow-up change; and TRTLLMINF-295. Name no internal tools, hosts beyond those
  already in the repository, people, lists, portals, or channels. Title: `[TRTLLMINF-472][infra]
  Migrate test results to artifactory.nvidia.com`. Update #21223's description with `gh api -X PATCH`.
- [ ] T071 OPS: Mark the pull request ready once T039–T043, T046, T048, T049, T050, and T061 are
  recorded and FR-019 is complete; send it for review to the TRT-LLM infrastructure developers and the
  code owners of `/jenkins/scripts/cbts` and `/tests/integration/defs/perf`. Fix blocking findings in
  the commit and move the rest to a follow-up pull request tracked in a linked ticket (TRTLLM-17014
  FR-026). Record the review outcome in `signoff-evidence.md`.
- [ ] T072 OPS: In the week before the merge, record SC-007's baseline in `signoff-evidence.md` from
  Jenkins step timings (`wfapi`) of that week's `L0_MergeRequest_PR` runs: the median duration of a
  run, of Collect Test Result, of each `Upload Single-GPU Coverage` stage of post-merge runs, and of
  the steps that upload a stage's results.
- [ ] T073 OPS: Merge when the review passes and T061 has passed: the cutover, which starts dual
  publishing. Record the merge commit and date in `signoff-evidence.md`.
- [ ] T074 OPS: Compare the first merge-request run and the first post-merge run after cutover with
  those of the commit before (quickstart step 10, FR-020): the same archives by name and number, the
  same test counts, the same stage outcomes; on the first post-merge run, the coverage step and the
  coverage task publish to the new location only, and the next pre-merge run's selection reads them
  there. Roll back with `LLM_ARTIFACTS_PRIMARY` if the new location makes a path fail (FR-018). Record
  in `signoff-evidence.md`.
- [ ] T075 OPS: For the first two weeks after cutover, track the success criteria from Collect Test
  Result's `[test-results] listed ...` lines (names on one location only, for SC-001; names read, for
  SC-002), the stages marked unstable by a failed replica, leftover checkpoints on both locations
  (SC-006), coverage-based selection's declines (SC-012), and the median durations against T072's
  baseline (SC-007). Record in `signoff-evidence.md`.
- [ ] T076 OPS: For the first week after cutover, fetch the console log (`consoleText`) of every run
  of `L0_MergeRequest_PR`, `L0_MergeRequest`, and `L0_PostMerge` and of the test jobs they start, and
  the `debug_env.txt` of every test-result archive of the runs compared in T074, with a script kept
  outside the repository, and scan them for the patterns of T048. Record the number of exposed tokens
  or passwords found, expected 0, in `signoff-evidence.md` (SC-008).
- [ ] T077 OPS: In the first week after cutover, have a developer outside the infrastructure team
  repeat T061's check with only the company sign-in on links from runs after cutover, one of each
  kind, including a post-merge run's coverage archive. Record in `signoff-evidence.md` (SC-014).
- [ ] T078 OPS: Open the ticket for ending dual publishing (research R18), linked from the pull
  request, with its preconditions: PR_Github's change landed and proven (T055), every outside reader
  in production on the new location (SC-015), and urm's retirement date from the program. Record it
  in `signoff-evidence.md`.
- [ ] T079 FUP: When T078's preconditions hold, and before urm's retirement date, change
  `testResultTargets` in `jenkins/L0_Test.groovy` and `jenkins/L0_MergeRequest.groovy` to return the
  new location alone, as the required target, so that its failure fails a step where urm's does today
  (FR-007); remove the replica's handling in `uploadTestResult`, urm's checkpoint uploads and deletes,
  urm's side of listing and reading test results, and the rollback to urm for them, coordinated with
  the matching change for build artifacts, since both use `LLM_ARTIFACTS_PRIMARY` and the read helper;
  update the affected tests. Open it as its own pull request under T078's ticket.
- [ ] T080 OPS: Before urm's retirement date, check quickstart step 11: T055 recorded, every outside
  reader in production on the new location according to its owner (SC-015), and T079 merged, with the
  first merge-request and post-merge runs after it publishing test results to the new location only
  and reading them there. Record in `signoff-evidence.md`.
- [ ] T081 OPS: Hand off to TRTLLMINF-295 (wheels) and the remaining llm-artifacts outputs, noting
  that urm's `sw-tensorrt-generic` is retired only when everything has moved; record the hand-off in
  `signoff-evidence.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 and T002 first; T003–T007 at once; T008 is done and repeats only when
  PR #20494's head moves. T004 has the longest lead time and gates the merge (T061, T073).
- **Foundational (Phase 2)**: the helper (T009 → T010), the scripts (T011 → T012 → T013), the
  `L0_Test.groovy` helpers (T014 → T015 → T016), and the `L0_MergeRequest.groovy` helpers (T017 →
  T018) are independent of each other. T019 → T020 → T021 settle the delete container and can run
  while the rest of Phase 2 is written.
- **US1 (Phase 3)**: needs all of Phase 2. Its runs (T039–T043) need the implementation tasks of
  their paths; T042's change is pushed and removed around its own run only.
- **US2 (Phase 4)**: T044 → T045 → T046 can start after T010; T047 → T048 need US1's checkpoint
  branch (T025, T026, T029).
- **US3 (Phase 5)**: T049 needs T045 and gates the merge; T050 needs US1's runs (T039, T041). The PRG
  tasks (T052–T054) can start at any time after T051 and do not gate the merge, unless T049 finds the
  redacted copies on the new instance, which makes T053 a merge prerequisite; T055 follows them.
- **US4 (Phase 6)**: T056 → T057 any time; T058 after T031 (same file); T059 after T035; T060 → T061
  need T004 and US1's runs; T062 needs T007.
- **US5 (Phase 7)**: T064 on the final commit (after T068); T065 after T074 and T075.
- **Polish (Phase 8)**: T066 → T067 → T068 → T069 → T070 → T071 → T073, with T072 in the week before
  T073; T074–T077 and T078 after T073; T079 when T078's preconditions hold; T080 after T079; T081
  last.

### User Story Dependencies

- **US1 (P1)**: independent after Foundational; it does not wait for PR_Github, which dual publishing
  keeps working from urm's copies.
- **US2 (P1)**: independent after Foundational; the credential probe needs US1's checkpoint branch.
- **US3 (P2)**: before the merge, T049 and T050 check that PR_Github is unaffected; PR_Github's own
  change follows on its own timeline before urm is retired.
- **US4 (P3)**: link tasks share files with US1 and follow it; the people check needs the program's
  grant.
- **US5 (P3)**: reviews the final commit and closes the work.

### Within Each User Story

- Tests come first and fail before the implementation: T009 → T010, T011 → T012/T013, T022 → T036,
  T023 → T018/T030/T036, T044 → T045, T056 → T057.
- Tasks on the same file run in order:
  - `jenkins/L0_Test.groovy`: T014 → T015 → T016 → T024 → T025 → T026 → T027 → T028 (if needed) →
    T029 → T030 → T031 → T032 → T058 (draft-only: T041, T042, T047 after their targets)
  - `jenkins/L0_MergeRequest.groovy`: T017 → T018 → T033 → T034 → T035 → T059 (draft-only: T038
    after T018)
  - `tests/unittest/scripts/test_fetch_build_artifact.py`: T009 → T023
- OPS validation tasks come last in each story.

### Parallel Opportunities

- Setup: T003–T007 at once.
- Foundational: T009, T011, T014, and T017 at once (different files); T019–T021 beside them.
- US1: T022 and T023 at once; then T036 (after T022) beside the Groovy sequences, and the
  `L0_MergeRequest.groovy` sequence (T033–T035) beside the `L0_Test.groovy` one (T024–T032).
- US2: T044 beside US1's implementation.
- US3: the PRG tasks beside everything in this repository.
- US4: T056, then T057, beside US1; T059 beside T058.
- Polish: T074–T077 side by side in the first weeks after cutover.

---

## Parallel Example: Foundational

```bash
# Tests and helpers in different files, together:
Task: "T009 List-mode tests in tests/unittest/scripts/test_fetch_build_artifact.py"
Task: "T011 Checkpoint script tests in tests/unittest/scripts/test_progress_snapshot.py"
Task: "T014 Locations, targets, and attempt primaries in jenkins/L0_Test.groovy"
Task: "T017 Targets, uploads, and browse links in jenkins/L0_MergeRequest.groovy"
```

## Parallel Example: User Story 1

```bash
# Tests, together:
Task: "T022 Coverage-archive reads in tests/unittest/scripts/cbts/test_coverage_artifact.py"
Task: "T023 New exit-code readers in tests/unittest/scripts/test_fetch_build_artifact.py"
# Then the two pipeline files and artifact.py, side by side:
Task: "T024 Final archives to the targets in jenkins/L0_Test.groovy"
Task: "T033 Collect Test Result through the helper in jenkins/L0_MergeRequest.groovy"
Task: "T036 Coverage archives through the helper in artifact.py"
```

## Parallel Example: User Story 4

```bash
# The perf link's test and the parent pipeline's links, together:
Task: "T056 Perf link tests in tests/unittest/others/test_perf_regression_artifact_url.py"
Task: "T059 Links in jenkins/L0_MergeRequest.groovy"
# Then T057 (after T056) beside T058 (jenkins/L0_Test.groovy).
```

---

## Implementation Strategy

### MVP First (User Stories 1 and 2)

1. Start Phase 1's requests at once: they have the longest lead time.
2. Complete Phase 2, including the delete-container probe (T019–T021); the branch is already on
   TRTLLM-17014's current head (T008).
3. Complete US1 and US2 on the draft: CI's own writes to both locations and reads from the new one
   first, with no credential where tests run, proven in every environment.
4. **Stop and validate**: quickstart steps 1 to 5 pass on the draft.

### Incremental Delivery

1. US1 + US2 → the draft writes test results to both locations and reads them safely.
2. US3 → PR_Github shown unaffected, and the redacted copies confirmed off the new instance.
3. US4 → links open for people; owners and developers told.
4. US5 and Polish → references accounted for, the commit cleaned, merged after TRTLLM-17014, validated
   after cutover, signed off.
5. After the merge → PR_Github's change (US3), the outside readers' moves, and the follow-up change
   that ends dual publishing before urm's retirement date.

The switch ships in one pull request; the stories are increments of the draft, not separate merges.
PR_Github's change and the follow-up change are separate change sets.

### Parallel Team Strategy

- One person on the pipeline changes (`jenkins/L0_Test.groovy`, then `jenkins/L0_MergeRequest.groovy`).
- One on the scripts and Python (helper, checkpoint scripts, `artifact.py`, perf link) and their tests.
- One on the OPS requests, notices, and checks, and later on PR_Github's change.

---

## Notes

- [P] tasks touch different files and depend on no incomplete task.
- Commit with `git commit -s` and the noreply address; never add co-author trailers or name AI tools;
  the pull request keeps one commit of this work, amended and pushed with an explicit
  `--force-with-lease=<branch>:<sha>`.
- Run `pytest` for these modules from outside the checkout with absolute paths and
  `--rootdir=<checkout>` and `--confcutdir` set to the module's folder (quickstart step 1).
- A failed replica marks its stage, never the build: an UNSTABLE build means infra-incomplete to the
  parent pipeline (research R7). Failed checkpoint uploads and deletes mark nothing.
- Never record a token, password, or netrc content anywhere.
