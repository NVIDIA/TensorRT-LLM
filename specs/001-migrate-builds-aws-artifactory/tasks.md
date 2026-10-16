---

description: "Task list for TRTLLM-17014: migrate build artifacts to AWS Artifactory"
---

# Tasks: Migrate Build Artifacts to AWS Artifactory

**Input**: Design documents from `specs/001-migrate-builds-aws-artifactory/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: Included. Constitution Principle V requires tests for every new or changed code path,
and its Governance section requires task lists to include them even when the spec does not ask.
This repository has no unit-test harness for Groovy pipeline code or the image-build files (plan,
Complexity Tracking). CI runs of the switch PR validate them where a pull request reaches them, and
the first post-merge run after cutover validates the rest (research R16).

**Organization**: Tasks are grouped by user story. Line numbers refer to commit `91404de7bc` of this
repository, commit `9e56d133` of `bloom-jenkins-shared-lib`, and commit `9b67bcf` of
`trtllm-jenkins-shared-lib`'s merge request branch. The PR is written on `upstream/main` at
`dac84b01a6`, where some of this code has moved: find each place by its content, using
`git show 91404de7bc:<file>` for the line cited. Coverage-based test selection moved in that
range, from `jenkins/scripts/cbts/coverage_selection/artifact.py` to
`jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`, now a click command run as
`python3 -m cbts.command coverage selection artifact`, with its tests in
`tests/unittest/scripts/cbts/test_coverage_artifact.py` (research R19).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: The user story the task serves (US1–US4)
- Each description starts with its change set:
  - **LIB**: the merge request to `trtllm-jenkins-shared-lib` from
    `feature/trtllm-17014-aws-upload-destination` (!111), implemented and open; it merges before
    the PR merges (FR-025)
  - **PR**: the one pull request in this repository, from
    `TRTLLM-17014-migrate-builds-to-aws-artifactory`: the helper scripts and their tests, and the
    switch. Cutover happens when it merges.
  - **OPS**: work outside the pull request and the merge request (branch housekeeping, requests,
    verification runs, sign-off)

## Path Conventions

- Paths in this repository are relative to its root: `jenkins/`, `scripts/`, `docker/`,
  `tests/unittest/scripts/`.
- Paths prefixed `trtllm-jenkins-shared-lib/` are in that separate repository, cloned in the `dev/`
  folder beside `TensorRT-LLM/`. Push to it over SSH.
- Branch, set up by T001: the PR comes from `TRTLLM-17014-migrate-builds-to-aws-artifactory`, based
  on `upstream/main`. It always holds exactly one commit. While the PR is a draft, that commit
  also carries the spec store, with internal names replaced by role descriptions (clarified
  2026-10-16); T053 removes it from the commit before the PR is marked ready.
- All work happens in this worktree, which has that branch checked out. No other worktree is
  created for this feature.
- After T001, the spec store (`specs/001-migrate-builds-aws-artifactory/`) stays excluded from Git
  (`info/exclude`), except that the draft PR's commit carries it, added with `git add -f`. Evidence
  for the sign-off request (FR-023) goes in its `signoff-evidence.md`, which T002 creates.

---

## Phase 1: Setup (Branches and Prerequisites With Lead Time)

**Purpose**: Branch housekeeping, then the requests and findings that gate cutover (plan, Delivery
Sequence steps 0 and 2). None of them changes pipeline code, so T003–T010 can start at once.

- [X] T001 OPS: Take the spec store off the pull-request branches before any of them is pushed for
  review. Today `TRTLLM-17014-migrate-builds-to-aws-artifactory` carries it as the commit "spec
  store (to be amended and removed)", which is on `origin`, the local-only commit "Revert guardword
  scrub in spec store", and uncommitted edits. The spec store is not backed up anywhere: it stays in
  this worktree only. Steps:
  - Add `specs/`, `.specify/`, and `.claude/skills/speckit-*` to
    `$(git rev-parse --git-common-dir)/info/exclude`, then restore `.gitignore` with
    `git checkout -- .gitignore`, so that neither the spec store nor a `.gitignore` change can reach
    a pull request.
  - On `TRTLLM-17014-migrate-builds-to-aws-artifactory`, run `git reset --soft` to the commit before
    the spec store (`91404de7bc` today), then `git rm -r --cached specs/`. The files stay in the
    worktree, untracked. Push with
    `git push --force-with-lease origin TRTLLM-17014-migrate-builds-to-aws-artifactory`, to the fork
    only, never to `upstream`. That removes the spec store commit from the fork's branch.
  - Confirm that `git log upstream/main..TRTLLM-17014-migrate-builds-to-aws-artifactory
    --name-only` lists nothing under `specs/`, `.specify/`, or `.claude/`, and that
    `git status --short` shows no `specs/` entry.
- [X] T002 OPS: Create `specs/001-migrate-builds-aws-artifactory/signoff-evidence.md` with one
  section per item the sign-off request needs: permission grants, identity readiness, access-check
  results and timings per environment (including both roles on the BoltProfileGen cluster
  frontend), the build-info finding, the source release package's producer, the owner's decisions,
  the internal access documentation, the library pin and its removal, the rollback rehearsal, the
  reference allowlist, the post-cutover comparisons, and the first post-merge run's checks of the
  post-merge-only paths. Record dates and outcomes only, never secrets.
- [ ] T003 [P] OPS: Through the program's support channel (FR-023), request a permission target on
  `sw-tensorrt-llm-generic-local` for the Jenkins credential `aws-artifactory-credentials`
  (publisher). It must hold exactly read, deploy, annotate, and delete/overwrite, and nothing
  administrative (FR-010, research R5). If the program will not extend that identity, onboard a
  dedicated one through its self-service process and use its credential ID wherever this list says
  `aws-artifactory-credentials`. Record the grant in `signoff-evidence.md`.
- [ ] T004 [P] OPS: Request read permission on `sw-tensorrt-llm-generic-local` for
  `trtllm-artifactory-credentials` (the reader, `jenkins/L0_Test.groovy:72`), and ask the program
  to confirm that it holds no deploy or delete permission there (FR-011). If it can write, onboard a
  dedicated read-only identity before cutover (research R5, Risk). Record the outcome in
  `signoff-evidence.md`.
- [ ] T005 [P] OPS: Record each identity's token expiry date and renewal owner in internal
  documentation outside this repository (FR-013), and note in `signoff-evidence.md` that they are
  recorded.
- [ ] T006 [P] OPS: Run the build-info check (FR-017, research R8, quickstart prerequisite 4): list
  `https://urm.nvidia.com/artifactory/api/build` with a netrc for `urm-artifactory-creds`, and note
  any build names that belong to TRT-LLM pipelines. Send them to the Artifactory team before
  cutover, or record "none" in `signoff-evidence.md` (US3 acceptance scenario 5).
- [ ] T007 [P] OPS: Ask the program for urm access records for `sw-tensorrt-generic/llm-artifacts/`
  (research R10). A client outside TRT-LLM CI in them would disprove the assumption that nothing
  reads in-scope artifacts by a fixed address. Contact any such client directly before cutover, and
  record the result in `signoff-evidence.md`.
- [ ] T008 [P] OPS: Verify the code version that jobs run (research R12). From the Jenkins
  configuration of `/LLM/helpers/Build-*`, the `L0_Test-*` jobs, and `/LLM/helpers/BoltProfileGen`,
  confirm that each loads its pipeline code at the `gitlabCommit` its parent passes
  (`jenkins/L0_MergeRequest.groovy:2190-2197`). Also confirm that the merge-request job loads
  `jenkins/L0_MergeRequest.groovy` from the PR's commit, which the library pin in T027 relies on
  (research R17). Record the result under R12 in
  `specs/001-migrate-builds-aws-artifactory/research.md`. If either does not hold, stop: FR-002
  cannot hold, and the design needs revisiting before the PR.
- [ ] T009 [P] OPS: Identify the producer of `tensorrt-llm[-sbsa]-release-src-<commit>.tar.gz`
  (research R13). Its only reader here, `jenkins/L0_Test.groovy:5988-5990`, sits in
  `runPackageSanityCheck`, whose only call (`:7062`) is commented out, so only the producer matters
  (FR-001). If the producer is in this repository, add a the PR task to publish it to the primary
  with the other packaged builds; otherwise raise it with its owner and list it in
  `signoff-evidence.md`. Correct the spec's "Current state" assumption that its readers here use the
  fallback, in `specs/001-migrate-builds-aws-artifactory/spec.md`.
- [ ] T010 [P] OPS: Get the owner's decisions on release branches (FR-018), on not copying
  pre-cutover builds (FR-014), and on the packaged-build list (FR-001), as research R15 lists them.
  The packaged-build list is needed before the PR merges, because it decides whether the
  optimized-package publish (T028) moves with this change. Record the decisions in
  `specs/001-migrate-builds-aws-artifactory/spec.md`, replacing the FR-018 marker, and in `plan.md`
  under "Open decisions".

**Checkpoint**: T001 is done before any branch of this repository is pushed for
review (T018, T041). The permissions from T003 and T004 are granted before the US1 verification
runs (T019–T021). T005–T010 are done before cutover (T058).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The library opt-in, the read helper, and the per-pipeline location helpers that every
story builds on.

**⚠️ CRITICAL**: The PR cannot merge before the library change has merged (T056). It can be tested
earlier, against the library change's branch (T027, research R17).

### Library opt-in (LIB)

- [X] T011 [P] LIB: Confirm that the change already on `feature/trtllm-17014-aws-upload-destination`
  (commit `9b67bcf`), in `trtllm-jenkins-shared-lib/vars/trtllm_utils.groovy` (`uploadArtifacts`)
  and `trtllm-jenkins-shared-lib/src/test/groovy/UploadArtifactsTest.groovy`, matches
  [contracts/shared-library-upload.md](contracts/shared-library-upload.md):
  - `uploadArtifacts(patternAbsPath, target, destination = 'urm')`, where `null` also means `urm`,
    and the `urm` path runs today's code unchanged;
  - `aws` defines server id `ArtifactoryAws` with `https://artifactory.nvidia.com/artifactory` and
    `aws-artifactory-credentials`, with the same `bypassProxy`, `timeout`, and `llmRetry(2, ...)`,
    and never defines `Artifactory`;
  - any other value fails through `error()` before a server is defined;
  - a failed `aws` upload logs one note that names `aws-artifactory-credentials` and says what to
    check if the log shows 401 or 403, then rethrows.

  Run `run-tests.sh` and confirm that `UploadArtifactsTest` passes. Amend the branch only to fix a
  mismatch, then run the tests again.
- [ ] T012 LIB: The merge request from `feature/trtllm-17014-aws-upload-destination` against the
  library's `main` is open (!111). Make sure it has gone for review to the library owners and all
  TRT-LLM infrastructure developers, stating the new location and that an Artifactory access token
  is needed (FR-024, quickstart step 11). The branch name carries no username, so the PR's public
  diff can name it (research R17, FR-008). Do not wait for it to merge: the PR tests against this
  branch, and T056 merges it.

### Read helper (PR)

- [X] T013 [P] PR: Write `tests/unittest/scripts/test_fetch_build_artifact.py` (new, NVIDIA SPDX
  header) against [contracts/helper-scripts.md](contracts/helper-scripts.md). Run three local
  `http.server` instances on different host names, for example `127.0.0.1`, `127.0.0.2`, and
  `127.0.0.3`, because netrc matches by host name: the two locations, set through
  `LLM_ARTIFACTS_AWS_URL` and `LLM_ARTIFACTS_URM_URL`, and a stand-in for the CDN. Use no network
  access, and put a no-op `sleep` on `PATH` so that retries do not wait. Assert that:
  - legacy paths (`sw-tensorrt-generic/...`, `sw-tensorrt-generic-local/...`) and layout paths
    (`llm-artifacts/...`) map to the same URL, and anything else exits 64;
  - primary 200: exit 0, with `fetch_build_artifact: served-by=aws path=<layout path>` on stderr;
  - primary 404 and secondary 200: exit 0 and `served-by=urm`; both 404: exit 2;
  - primary 401 or 403 from the location itself: exit 3, the secondary is never requested, and the
    message names `ARTIFACTORY_CREDENTIAL_ID`, says its token needs renewal, and contains neither
    `403 Forbidden` nor `URL returned error: 403`;
  - a download that the primary redirects to the CDN stand-in, which answers 403, is retried once
    with `X-JFrog-Download-Redirect-To: None` and then succeeds; if the retry fails too, exit 4 with
    no credential message, and the secondary is never requested;
  - the file is checked against `checksums.sha1` from `api/storage/<repository>/<layout path>`: a
    match renames `<dest>.part` to `<dest>`; a mismatch removes `<dest>.part`, is retried, and
    exits 4 without requesting the secondary;
  - primary 500: 3 attempts in total, then exit 4, and the secondary is never requested;
  - `--exists` writes no file and prints `<url> <sha1>` on stdout;
  - `--properties` requests `api/storage/<repository>/<layout path>?properties` under the same rule;
  - `LLM_ARTIFACTS_PRIMARY=urm` reverses the order;
  - requests to the aws server carry the netrc credentials, requests to the urm server and the CDN
    stand-in carry none, and a missing `ARTIFACTORY_NETRC` when aws must be contacted exits 64;
  - the netrc password never appears in stdout or stderr.
- [X] T014 PR: Write `jenkins/scripts/fetch_build_artifact.sh` (new, executable, NVIDIA SPDX
  header) so that T013 passes, implementing the
  [read rule](contracts/artifact-locations.md#read-rule) and the
  [interface](contracts/helper-scripts.md#jenkinsscriptsfetch_build_artifactsh-new):
  - Use `curl --netrc-file "$ARTIFACTORY_NETRC"` for aws only, and anonymous requests for urm.
  - Follow redirects, and read the status and the final URL with
    `-w '%{http_code} %{url_effective}'` instead of `-f`, so that curl's own
    `URL returned error: 403` text never reaches the log.
  - A 403 whose final URL is on another host than the location is a CDN rejection: retry it once
    with `-H 'X-JFrog-Download-Redirect-To: None'` (research R18).
  - Before a download, read `checksums.sha1` from the storage API on the location that holds the
    file. On urm, query the repository that the existing property lookup at
    `jenkins/L0_Test.groovy:1756` queries. After the download, compare it with `sha1sum` of
    `<dest>.part`.
  - Retry connection errors, 5xx, and checksum mismatches 3 times in total, waiting 2 s and then
    4 s.
  - `--exists` uses the storage API, which reports both existence and the SHA-1.
  - No `set -x`, and no secret on any command line.

### Pipeline location helpers (the PR)

- [X] T015 [P] PR: Add the location helpers from
  [contracts/artifact-locations.md](contracts/artifact-locations.md) to each pipeline that publishes
  or reads in-scope artifacts: `jenkins/Build.groovy`, `jenkins/L0_MergeRequest.groovy`,
  `jenkins/L0_Test.groovy`, `jenkins/BoltProfileGen.groovy`, and `jenkins/BuildDockerImage.groovy`.
  Each file gets only what it uses:
  - the primary, from `env.LLM_ARTIFACTS_PRIMARY` (`aws` or `urm`; unset means `aws`; anything else
    is an error), resolved once per step, and the secondary;
  - the layout path of a legacy path;
  - the repository path on a location (`sw-tensorrt-llm-generic-local/<layout path>` or
    `sw-tensorrt-generic/<layout path>`), and its URL.

  `ARTIFACT_PATH` and `UPLOAD_PATH` stay as they are (research R3). The repository has no shared
  Groovy file, so these are small functions beside each file's existing path constants.

**Checkpoint**: The library change is under review (T012), the read helper passes its tests, and the
location helpers are in place. User-story work can start.

---

## Phase 3: User Story 1 - Verified, authenticated access to the new repository (Priority: P1)

**Goal**: Every CI environment that writes or reads build artifacts holds an identity for
artifactory.nvidia.com that is proven to work before cutover.

**Independent Test**: Quickstart step 2. `check_artifactory_access.sh` passes for the publisher
role from a build pod and on the BoltProfileGen cluster frontend, and for the reader role from a
Kubernetes test pod, a Docker-build pod, the BoltProfileGen cluster frontend, and a node on each
Slurm cluster. Anonymous reads are rejected, and no token appears in logs or in files left on the
hosts.

### Tests for User Story 1

- [X] T016 [P] [US1] PR: Write `tests/unittest/scripts/test_check_artifactory_access.py` (new,
  NVIDIA SPDX header). Run it against a local server that implements PUT with `;key=value`
  properties, GET, HEAD, `api/storage/<path>` with `checksums.sha1`, `api/storage/<path>?properties`,
  directory listing through `api/storage/<dir>`, `api/copy`, and DELETE, and that enforces
  credentials per role. Assert that:
  - the publisher role prints `PASS <check>` for each publisher check in
    [the contract](contracts/helper-scripts.md#jenkinsscriptscheck_artifactory_accesssh-new), in
    order, ending with the delete of `probe-copy.txt`, exits 0, and leaves `probe.txt` in place;
  - the reader role reads `probe.txt` and matches its SHA-1 to `checksums.sha1`, reads its
    `access.check` property, expects 401 for its anonymous read and 403 for its write attempt, and
    exits 0;
  - `--role publisher --cleanup` deletes `probe.txt` and runs no other check;
  - a SHA-1 mismatch or an unexpected status prints `FAIL <check>: <reason>` and exits 1;
  - `--time` prints seconds and MB/s for both locations;
  - the netrc password never appears in the output.

### Implementation for User Story 1

- [X] T017 [US1] PR: Write `jenkins/scripts/check_artifactory_access.sh` (new, executable,
  NVIDIA SPDX header) so that T016 passes: `--role publisher|reader --probe-dir <layout-dir>
  [--time <artifact-path>]` and `--role publisher --cleanup --probe-dir <layout-dir>`, with
  `ARTIFACTORY_NETRC`, the base-URL overrides, and the same path mapping as T014. The publisher role
  deletes `probe-copy.txt`, which shows that it holds delete/overwrite (FR-010), and leaves
  `probe.txt` for the reader checks.
- [X] T018 [US1] PR: Commit T013, T014, T016, and T017 as the first commit of the PR branch
  (`TRTLLM-17014-migrate-builds-to-aws-artifactory`, T001), once `pre-commit` and the local unit
  tests pass on those files. Confirm that the commit touches only those four files, and search it
  for internal distribution lists, portals, chat channels, people, and internal URLs (FR-008,
  research R11). Push the branch to the fork (`origin`) only, so that the access-check runs
  (T019–T021) can take `jenkins/scripts/check_artifactory_access.sh` from that commit: it reaches
  `main` only with the PR.
- [ ] T019 [US1] OPS: After T003 and T018, run the publisher role of
  `jenkins/scripts/check_artifactory_access.sh` from a build pod with `aws-artifactory-credentials`
  (quickstart step 2). Use a sandbox Jenkins job that takes the script from T018's commit on the
  fork, writes the netrc with `umask 077`, and deletes it in `finally`. Its `probe.txt` stays in
  place for T021.
- [ ] T020 [US1] OPS: After T003 and T018, run the publisher role of
  `jenkins/scripts/check_artifactory_access.sh` on the BoltProfileGen cluster frontend, where the
  optimized package is published (`publishBoltedCanonical`, `jenkins/BoltProfileGen.groovy:801-840`),
  with `aws-artifactory-credentials` and its own probe directory, then at once with `--cleanup`
  there. Feed the netrc over stdin and remove it with a trap, as `publishBoltedCanonical` does
  (quickstart step 2, spec FR-020, research R16).
- [ ] T021 [US1] OPS: After T004 and T019, run the reader role of
  `jenkins/scripts/check_artifactory_access.sh` with `trtllm-artifactory-credentials` against T019's
  probe directory, with `--time` on a real packaged build. Run it from a Kubernetes test pod, a
  Docker-build pod (where the image job's wait step and image build run), the BoltProfileGen cluster
  frontend (where its input read runs; netrc over stdin, removed by a trap), and one node on each
  Slurm cluster (both ENROOT and the other runtime), delivering the netrc to each as research R5
  describes. Then run `--role publisher --cleanup` on T019's probe directory.
- [ ] T022 [US1] OPS: Inspect the logs of the T019–T021 jobs, and the hosts they ran on, for tokens
  or passwords, including netrc files left in Slurm workspaces or on the BoltProfileGen frontend
  after the jobs end (US1 acceptance scenario 3, FR-012, SC-006). Record the finding in
  `signoff-evidence.md`.
- [ ] T023 [US1] OPS: Record each environment's results and timings in `signoff-evidence.md`, and
  mark both identities `Verified` (data-model, Identity readiness). If any environment is more than
  10% slower than urm, raise it with the program before cutover (research R14).

**Checkpoint**: Both identities work in every environment, including those of the paths that only
the post-merge pipeline runs. The PR can be marked ready (T053); its draft CI runs (T041) do not
wait for this.

---

## Phase 4: User Story 2 - Builds and tests run end to end on the new repository (Priority: P1) 🎯 MVP

**Goal**: Every run whose code contains the change publishes `build_info.txt`, the packaged builds,
and `waive_list/` only to the new location, and its own stages, including the image build, read them
from there.

**Independent Test**: Quickstart step 3 on the PR. Every in-scope file returns 200 from the new
location and 404 from the old one, stage logs show `served-by=aws`, the build-info and waive-list
links name artifactory.nvidia.com, test-result links still name urm, and stage outcomes match the
pre-change baseline. The image build, BoltProfileGen's input read, and the `bolted-<tar>` property
run only in the post-merge pipeline, so the first post-merge run after cutover checks them (T060).

### Tests for User Story 2

- [X] T024 [P] [US2] PR: Update `tests/unittest/scripts/test_slurm_install.py`:
  - Replace `test_l0_artifact_download_retries_overwrite_archive` (`:105-113`). It expects exactly
    four `wget -nv -O ... llmTarfile` lines in `jenkins/L0_Test.groovy`. The replacement checks that
    every packaged-build download there goes through `fetch_build_artifact.sh` inside a retry
    wrapper, and that no `wget` of a packaged build remains.
  - Add tests that run the download step of `jenkins/scripts/slurm_install.sh` with a stub `curl` on
    `PATH`. Factor that step into a function the test can call, as
    `test_retry_command_returns_terminal_status` calls `retry_command`. Assert that:
    - `--netrc-file "$ARTIFACTORY_NETRC"` is passed when the variable is set and omitted otherwise;
    - the file is written to the temporary path and moved only on success;
    - with `llmTarfileSha1` set, a file whose SHA-1 differs fails the attempt and is not moved into
      place, and a matching file is moved; with it unset, no check is made;
    - a 403 reported with a final URL on another host is retried once with
      `X-JFrog-Download-Redirect-To: None`, and is not reported as a credential failure;
    - on HTTP 401 or 403 from Artifactory itself, the message names `ARTIFACTORY_CREDENTIAL_ID` and
      contains neither `403 Forbidden` nor `URL returned error: 403`;
    - no secret appears on a command line.
- [X] T025 [P] [US2] PR: Write `tests/unittest/scripts/test_get_wheel_from_package.py` (new,
  NVIDIA SPDX header), with `subprocess.run` and `time.sleep` patched. Assert that
  `get_wheel_from_package`:
  - calls `<repository root>/jenkins/scripts/fetch_build_artifact.sh <artifact_path>/<tarball>
    <tarball>`, with `TensorRT-LLM.tar.gz` for `x86_64` and `TensorRT-LLM-GH200.tar.gz` otherwise;
  - passes the environment through;
  - keeps polling on exit codes 2 and 4 for up to `timeout` attempts;
  - stops at once on exit code 3 with an authentication error.

### Implementation for User Story 2: publishing

- [X] T026 [P] [US2] PR: In `jenkins/Build.groovy`, publish the packaged builds to the primary.
  Define an `rtServer` with id `ArtifactoryAws` (`https://artifactory.nvidia.com/artifactory`,
  `aws-artifactory-credentials`, the same `bypassProxy` and `timeout`) beside both existing
  definitions (`:436-444`, `:829-837`). When the primary is `aws`, call this file's own
  `uploadArtifacts` (defined at `:391`, called at `:446`; not `trtllm_utils.uploadArtifacts`) with
  the prefix `sw-tensorrt-llm-generic-local/<layout path of UPLOAD_PATH>` and
  `serverId: 'ArtifactoryAws'`. When it is `urm`, keep today's server and prefix. The `artifacts`
  map does not change, so `bolted-<tar>` keeps its `props` (`:694`) and `unbolted-<tar>` (`:707`) is
  published as before (FR-004); the `bolted-<tar>` upload runs only in the post-merge pipeline, so
  T060 checks it. The plugin reports a 401 or 403 only in its log, so when an `aws` upload fails,
  log one note that names `aws-artifactory-credentials` and says what to check if the log shows 401
  or 403, as the shared helper does (research R2), then fail the step as today
  ([publish rule](contracts/artifact-locations.md#publish-rule)).
- [X] T027 [P] [US2] PR: In `jenkins/L0_MergeRequest.groovy`:
  - pass the primary as the destination of `trtllm_utils.uploadArtifacts` at the two in-scope
    uploads, `waives*.txt` (`:616`) and `build_info.txt` (`:668`), and map each target to the
    primary's repository;
  - change the links at `:617` and `:669` to the URL on the location the run published to (FR-007);
  - keep both uploads best effort, as they are today (FR-013). When an `aws` upload fails, the
    shared helper already logs which credential to check before it rethrows (research R2);
  - at `:17`, load `trtllm-jenkins-shared-lib@feature/trtllm-17014-aws-upload-destination` instead
    of `@main` while the library merge request is open (research R17). T057 restores `@main` before
    the PR merges.
- [X] T028 [P] [US2] PR: In `jenkins/BoltProfileGen.groovy`, publish the BOLTed canonical package
  and `unbolted-<tar>` to the primary (`:545-551`, and `publishBoltedCanonical` at `:801-840`):
  - take the base URL from the location helpers instead of `URM_ARTIFACTORY_BASE` (`:50`), and
    derive the netrc host from it;
  - use `aws-artifactory-credentials` when the primary is `aws`, or `urm-artifactory-creds` when it
    is `urm`;
  - keep the netrc fed over stdin with trap cleanup, the existence check before uploading
    `unbolted-`, and the PUT upload. That upload overwrites the canonical file, so it needs
    delete/overwrite (FR-010);
  - on a 401 or 403, fail with the credential message from the publish rule.

  The profile store (`PROFILE_PROMOTE_DIR`, `:113`, `:749-800`) stays on urm. This path runs only
  with `boltPublishCanonical` on, and every pipeline sets it to `"false"`
  (`jenkins/L0_MergeRequest.groovy:2625`), so T066 validates it on its first real run (spec FR-020).

### Implementation for User Story 2: reading in L0_Test

- [X] T029 [P] [US2] PR: In `jenkins/L0_Test.groovy`, make the read helper available to every
  stage (see [Getting the helper](contracts/helper-scripts.md#getting-the-helper)). Test stages
  take their source tree from the package they are about to download (`:5060-5115`,
  `:1819-1858`), so they need the helper from the run's commit:
  - In "Setup Environment" (`:7480`), fetch `jenkins/scripts/fetch_build_artifact.sh` once with
    `trtllm_utils.checkoutFile(LLM_REPO, env.gitlabCommit,
    'jenkins/scripts/fetch_build_artifact.sh', <dir>)`, in a container that has git, and `stash`
    it. `jenkins/L0_MergeRequest.groovy:581` fetches `waives.txt` the same way.
  - Add one function, for example `fetchBuildArtifact(pipeline, artifactPath, dest, mode)`, for
    every in-scope read in this file. It unstashes the helper when the workspace lacks it, writes a
    netrc for `ARTIFACTORY_CREDENTIALS_ID` (`:72`) with `umask 077` inside `withCredentials`, and
    deletes it in `finally`. It sets `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, and
    `ARTIFACTORY_CREDENTIAL_ID`, and runs the helper inside `trtllm_utils.llmExecStepWithRetry`. It
    supports downloads, `--properties`, and `--exists`, returning stdout (`<url> <sha1>`) for
    `--exists`.
- [X] T030 [US2] PR: In `jenkins/L0_Test.groovy:1746-1773`, make `boltedTarUrl` return an
  artifact path (`${ARTIFACT_PATH}/<tar>` or `${ARTIFACT_PATH}/bolted-<tar>`) instead of a urm URL,
  and rename it to `boltedTarArtifactPath`. Read the `bolt.ref` property through T029's function
  with `--properties` instead of the anonymous `curl` at `:1756-1762`. Keep the pin check and its
  messages. The property read runs only in post-merge runs (`BOLT_TEST_BOLTED_TARBALL`, `:6358`),
  so T060 validates it after cutover (research R16).
- [X] T031 [US2] PR: In `jenkins/L0_Test.groovy:5108-5115` (Kubernetes test pods), download the
  packaged build through T029's function into `${llmPath}/${tarName}`, after the workspace reset at
  `:5062-5068`, and keep the 30-minute timeout.
- [X] T032 [US2] PR: In the Slurm launcher in `jenkins/L0_Test.groovy` (`:1819-1858`,
  `:1942-1968`, `:2198-2222`):
  - On the agent, resolve the packaged build with `--exists`, and download the build for local
    extraction through T029's function.
  - In the job script, export `llmTarfile=<url>` and `llmTarfileSha1=<sha1>` from the `--exists`
    result, `ARTIFACTORY_NETRC=<netrc path on the node>`, and `ARTIFACTORY_CREDENTIAL_ID`.
  - On ENROOT clusters, the node netrc is `${enrootConfigDirNode}/.credentials`, which `:1942-1968`
    already writes for the same host. On other runtimes, write a per-job netrc in `${jobWorkspace}`
    the same way. `cleanUpSlurmResources` (`:819`) removes both with the job workspace.

  The node still receives a URL, so a reused package's older `slurm_install.sh` keeps working.
- [X] T033 [P] [US2] PR: In `jenkins/scripts/slurm_install.sh:52-67`, download `$llmTarfile`
  with `curl` instead of `wget`, adding `--netrc-file "$ARTIFACTORY_NETRC"` when it is set. Keep
  `retry_command --timeout 1800` and the temporary-file-then-move pattern. Read the status and final
  URL with `-w`; a 403 after a redirect to another host is retried once with
  `-H 'X-JFrog-Download-Redirect-To: None'` and is not reported as a credential failure. When
  `llmTarfileSha1` is set, compare it with `sha1sum` of the temporary file before moving it, and
  fail the attempt on a mismatch. On HTTP 401 or 403 from Artifactory itself, print the exit-3
  wording, naming `ARTIFACTORY_CREDENTIAL_ID`
  ([contract](contracts/helper-scripts.md#jenkinsscriptsslurm_installsh-changed)). T024 passes.
- [X] T034 [US2] PR: In `jenkins/L0_Test.groovy`, route the remaining in-scope reads through
  T029's function:
  - the documentation build (`:4046-4049`);
  - the test-list check (`:4149-4153`);
  - `mergeWaivesTxt` (`:4790-4804`), for `${ARTIFACT_PATH}/waive_list/waives.txt`. It stays best
    effort: after any failure it logs the helper's message, which names the credential on exit code
    3, and uses the default waive list, as today (spec FR-013).

  Leave the source release package read (`:5988-5990`) unchanged: its only call (`:7062`) is
  commented out, and T050 allowlists it (research R13). Keep `URM_ARTIFACTORY_BASE` (`:50`), which
  `:4402` still uses for an out-of-scope file.

### Implementation for User Story 2: reading in other pipelines and the image build

- [X] T035 [US2] PR: In `jenkins/BoltProfileGen.groovy`, read the input package under the read
  rule while keeping the parallel ranged download (`:296-385`), which a single stream would slow
  down:
  - Outside `container('alpine')` (`:962`), whose image has no git, bash, or curl, fetch the helper
    with `trtllm_utils.checkoutFile` at the run's commit, and copy it to the cluster workspace with
    `Utils.copyFileToRemoteHost`.
  - In the cluster script (`:386-400`), export `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, and
    `ARTIFACTORY_CREDENTIAL_ID` explicitly, because Jenkins environment variables do not reach the
    cluster ([contract](contracts/artifact-locations.md#primary-and-secondary)).
  - Resolve `unbolted-<tar>` first and then `<tar>` with `fetch_build_artifact.sh --exists`, and
    pass the URL and SHA-1 to `fetch_verified` (`:369`). Add `--netrc-file` to its curl calls for
    aws URLs only. Its single-stream fallback sends `X-JFrog-Download-Redirect-To: None`. After the
    existing archive check, compare the file's SHA-1, and fail on a mismatch
    ([contract](contracts/helper-scripts.md#boltprofilegen-input-download-changed)). The llvm-bolt
    download from GitHub stays anonymous.
  - Feed the reader netrc over stdin and remove it with a trap, as `publishBoltedCanonical` does.

  This replaces the input URLs at `:274` and `:390`. It runs after T028, which edits the same file.
  Only the post-merge pipeline runs this path, so T060 validates it after cutover (research R16).
- [X] T036 [P] [US2] PR: In `jenkins/BuildDockerImage.groovy:1044-1100` ("Wait For Build Job
  Complete"), poll with the helper. This stage polls urm anonymously for `TensorRT-LLM.tar.gz` and
  `TensorRT-LLM-GH200.tar.gz` (research R1).
  - Install `git` and `curl` beside the existing `wget` install (`:1055`).
  - Fetch the helper with `trtllm_utils.checkoutFile(LLM_REPO, LLM_COMMIT_OR_BRANCH, ...)`, and
    write a reader netrc with `umask 077`, deleted in `finally`.
  - Check each file with `fetch_build_artifact.sh --exists ${UPLOAD_PATH}/<file>`. Exit 0 means
    ready, exit 2 or 4 means keep polling, and exit 3 fails the stage at once with the credential
    message.

  Only the post-merge pipeline runs this stage, so T060 validates it after cutover (research R16).
- [X] T037 [P] [US2] PR: In `docker/Makefile`:
  - add `ARTIFACTORY_NETRC ?=` and `LLM_ARTIFACTS_PRIMARY ?=` beside `GITHUB_CLONE_TOKEN`
    (`:57-59`);
  - define
    `ARTIFACTORY_SECRET_ARG := $(if $(ARTIFACTORY_NETRC),--secret id=artifactory_netrc$(COMMA)src=$(ARTIFACTORY_NETRC))`,
    and add it after `$(GITHUB_SECRET_ARG)` in `%_build` (`:95-97`);
  - pass `--build-arg LLM_ARTIFACTS_PRIMARY=...` only when that variable is set.

  Never pass the netrc as a build argument, because `docker history` shows those.
- [X] T038 [P] [US2] PR: In the `wheel` stage of `docker/Dockerfile.multi` (`:146-172`), add
  `ARG LLM_ARTIFACTS_PRIMARY=""`. To its `RUN`, add `--mount=type=secret,id=artifactory_netrc` and
  `--mount=type=bind,source=jenkins/scripts/fetch_build_artifact.sh,target=/src/tensorrt_llm/jenkins/scripts/fetch_build_artifact.sh`.
  The stage copies `scripts/` but not `jenkins/` (`:148-154`), and `:120` already bind-mounts a
  single script the same way. Export `ARTIFACTORY_NETRC=/run/secrets/artifactory_netrc` only when
  that file exists, and export `LLM_ARTIFACTS_PRIMARY` from the argument. The default
  `scripts/build_wheel.py` path is unaffected.
- [X] T039 [US2] PR: In `jenkins/BuildDockerImage.groovy`, when `prepareWheelFromBuildStage`
  (`:290-377`) returns arguments, write a netrc for `trtllm-artifactory-credentials` with
  `umask 077`, outside the build context (`${LLM_ROOT}`). Pass `ARTIFACTORY_NETRC=<path>` and
  `LLM_ARTIFACTS_PRIMARY=<primary>` to the `make` calls that carry `${buildWheelArgs}`
  (`:683-689`, `:705-711`, `:722-728`), and delete the file in `finally`. It runs after T036 (same
  file), T037, and T038. Only the post-merge pipeline builds the SBSA release image with the
  build-stage wheel, so T060 validates this path after cutover (research R16).
- [X] T040 [P] [US2] PR: In `scripts/get_wheel_from_package.py:121-146`, download through
  `<repository root>/jenkins/scripts/fetch_build_artifact.sh <artifact_path>/<tarball> <tarball>`
  instead of `wget` on a urm URL. Keep polling on exit codes 2 and 4, and stop at once on exit
  code 3 with an error that names the credential. Leave the command line unchanged, update the
  copyright year to 2022-2026, and add no lint violations (the file is in the legacy lint group).
  T025 passes.

### Validation for User Story 2

- [ ] T041 [US2] PR: Start right away (clarified 2026-10-16): the PR is opened as a draft and
  iterates on its CI runs; until the grants (T003, T004) are in place, runs that fail at the first
  upload to the new location are expected. The grants, the access check (T023), and the
  code-version check (T008) gate marking the PR ready (T053, T058), not these runs. Rebase
  `TRTLLM-17014-migrate-builds-to-aws-artifactory` on `upstream/main`, confirm that it is one
  commit whose `specs/` part names no internal lists, portals, channels, people, or internal hosts
  (T052), and open the PR as a draft titled
  `[TRTLLM-17014][infra] migrate build artifacts to artifactory.nvidia.com`. Then run quickstart
  step 3, in this order (spec FR-020):
  0. Ask a member of `NVIDIA/trt-llm-ci-approvers` to apply `ci: full pre-merge approved` to the
     PR; step 1 does not wait for it. The PR changes `jenkins/L0_Test.groovy`, which is in
     `getMultiGpuFileChanged`'s list in `jenkins/L0_MergeRequest.groovy`, so CI adds the
     multi-GPU stages to every default run. CI checks for the label when each architecture's
     single-GPU tests finish (`requireMultiGpuApprovalLabel`): if it is there, the same run goes
     on to the multi-GPU stages; if not, they are reported as Blocked, the run ends UNSTABLE, and
     a later `/bot run` on the same commit reuses the build and the stages that passed. A new
     commit ends that reuse, so push nothing until the multi-GPU stages have run.
  1. The default `/bot run`, with no options. It runs the pre-merge stages that apply and
     coverage-based test selection, and it sets the PR's CI status. Note its packaged-build
     download times next to a recent main pre-merge run (research R14).
  2. Only after it ends, `/bot run --stage-list` runs for what it did not reach, with stage names
     from `scripts/test_to_stage_mapping.py`: one Slurm test stage per runtime (ENROOT and the
     other), and, where no pre-merge stage reaches a cluster or runtime that the change affects, a
     post-merge test stage there by its exact name. Exact names need no label (research R16), but
     a stage in the multi-GPU sub-pipeline needs `ci: full pre-merge approved`.

  Each `/bot run` kills the PR's running jobs, so start one only after the previous run ends. If
  the first run fails while loading the shared library, Jenkins refuses the version override
  (research R17): drop the pin from T027 and wait for T056 before testing. Confirm:
  - 200 from the new location and 404 from the old one for `build_info.txt`,
    `waive_list/waives.txt`, and each packaged build;
  - `served-by=aws` in stage logs;
  - build-info and waive-list links on artifactory.nvidia.com, and
    `Download merged waives.txt successfully` in test stages, since those steps log a failure and
    carry on; test-result links still on urm;
  - if the default run's coverage-based test selection ran its coverage audit
    (`CBTS coverage pilot:` in the log), `artifact.py` logged `served-by=` and no credential message;
  - after the jobs end, no netrc file in the Slurm job workspaces (FR-012).

  Record the result in `signoff-evidence.md`.

**Checkpoint**: US2 works on the PR's own runs for every path that a pull request reaches. This is
the minimum for a working cutover.

---

## Phase 5: User Story 3 - History-reading tools keep working across the cutover (Priority: P2)

**Goal**: From the change's first merge-request run onward, tools that read earlier builds keep
finding pre-cutover artifacts through the fallback.

**Independent Test**: Quickstart steps 4, 5, and 7, plus step 6 during the rollback rehearsal
(T054). Pre-cutover `build_info.txt`, packaged builds, and waive lists come back from urm with
`served-by=urm` and identical bytes. A rejected token exits 3 without contacting urm. Coverage-based
test selection resolves the same build metadata as before the change. Reusing a build that sits on
the other location rebuilds and logs why.

### Tests for User Story 3

- [X] T042 [P] [US3] PR: Extend `tests/unittest/scripts/cbts/test_coverage_artifact.py` with
  `build_commit` cases, patching `urllib.request.urlopen` and supplying a netrc file:
  - aws 200 returns the commit without requesting urm;
  - aws 404 and urm 200 returns the commit and logs `served-by=urm`;
  - aws 401 or 403 raises `ArtifactAuthError` without requesting urm;
  - content whose SHA-1 differs from the storage API's `checksums.sha1` raises
    `ArtifactTransferError` without requesting urm;
  - both 404 return `None`;
  - the aws request carries netrc credentials and `X-JFrog-Download-Redirect-To: None`, and the urm
    request carries neither credentials nor that header;
  - `LLM_ARTIFACTS_PRIMARY=urm` reverses the order;
  - the click command `main` turns either error into a message on stderr and exit code 1.

  The existing `_get` mocks for pins and coverage archives keep passing.

### Implementation for User Story 3

- [X] T043 [US3] PR: In `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`, read
  `build_info.txt` under the read rule, in-process
  ([contract](contracts/helper-scripts.md#jenkinsscriptscbtscbtscommandcoverageselectionartifactpy-changed)):
  - `build_info_url` and `build_commit` try the primary first, with credentials from the standard
    `netrc` module and `ARTIFACTORY_NETRC` for aws, and fall back to the secondary only on 404;
  - aws requests send `X-JFrog-Download-Redirect-To: None`, since the file is small;
  - the content must match `checksums.sha1` from the storage API on the serving location;
  - add `ArtifactAuthError` for 401 and 403, with the exit-3 wording, and `ArtifactTransferError`
    for a checksum mismatch, and log which location served the file;
  - the click command `main` catches both, prints the message to stderr, and exits 1, as it does
    after its other failures;
  - coverage archives and pins keep `_URM`.

  Catch only `urllib.error.HTTPError` and `OSError`, as `_get` does, and annotate new functions.
  T042 passes.
- [X] T044 [P] [US3] PR: In `_cbtsCoverageAudit` in `jenkins/L0_MergeRequest.groovy`, run both
  `python3 -m cbts.command coverage selection artifact` invocations (`--resolve-pin` and
  `--prepare`; `:1355` and `:1396` at `91404de7bc`) with a reader netrc
  (`trtllm-artifactory-credentials`, `umask 077`, deleted in `finally`), and with
  `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, and `ARTIFACTORY_CREDENTIAL_ID` in the
  environment. Keep their `|| true`: when `artifact.py` exits 1 on an `ArtifactAuthError` or
  `ArtifactTransferError`, coverage-based test selection declines through `_cbtsCoverageDecline`,
  as after any failure today, and the log keeps `artifact.py`'s message naming the credential (spec
  FR-013). It runs after T027, which edits the same file.
- [X] T045 [P] [US3] PR: In `copyCachedArtifacts` in `jenkins/Build.groovy:341-389`, apply the
  reuse rule (research R6, and the
  [contract](contracts/artifact-locations.md#reuse-of-an-earlier-build)):
  - map `reuseArtifactPath` and `UPLOAD_PATH` to the primary's local repository:
    `sw-tensorrt-llm-generic-local/...` on aws, or `sw-tensorrt-generic-local/...` on urm, as today;
  - check each packaged build with a HEAD request on the primary;
  - if they all exist, copy them with the copy API on that instance;
  - if one is missing there but present on the secondary, log
    `reuse source is on <secondary>; rebuilding` and return false;
  - otherwise keep today's message and rebuild;
  - a 401 or 403 from the copy API fails with the credential message from the publish rule.

  Authenticate with a netrc file (`aws-artifactory-credentials` for aws, `urm-artifactory-creds`
  for urm) instead of `curl --user` (`:372`). Whether artifacts are placed into the current run's
  directory does not change (FR-016). It runs after T026, which edits the same file. T054 validates
  it.

### Validation for User Story 3

- [ ] T046 [US3] OPS: Run quickstart step 4 with `jenkins/scripts/fetch_build_artifact.sh` from
  the PR's branch, with a personal reader netrc, for the 20 most recent main post-merge builds before
  cutover and 10 older retained ones. `build_info.txt`, a packaged build, and
  `waive_list/waives.txt` each return exit 0, `served-by=urm`, and bytes identical to a direct urm
  download (SC-004). Record the results in `signoff-evidence.md`.
- [ ] T047 [US3] OPS: Run quickstart step 5: `jenkins/scripts/fetch_build_artifact.sh` with an
  invalid token. Expect exit code 3, a message that names the credential and says its token needs
  renewal without naming internal systems, and no `served-by=` line (FR-013, FR-014). Record the
  result in `signoff-evidence.md`.
- [ ] T048 [US3] OPS: Run quickstart step 7 (FR-015, US3 acceptance scenario 2). From a checkout of
  the PR, with a personal reader netrc in `ARTIFACTORY_NETRC`, run
  `PYTHONPATH=jenkins/scripts/cbts python3 -m cbts.command coverage selection artifact --resolve-build --build <a recent main post-merge build> --pr-head <a main commit>`,
  then the same command from a checkout of the PR's base commit. Both print the same build metadata,
  and the PR run logs that urm served `build_info.txt`. Record the result in
  `signoff-evidence.md`.

**Checkpoint**: History tools keep finding pre-cutover builds, and failures surface instead of
falling back.

---

## Phase 6: User Story 4 - No in-scope reference to the old location remains (Priority: P2)

**Goal**: The only in-scope references to the old location left are those the fallback and the
rollback need, plus allowlisted ones, each with a reason. Developers can download build outputs
with a personal token.

**Independent Test**: The reference search in quickstart step 10 finds only out-of-scope matches,
kept references, and allowlisted entries, each with a reason. A developer follows the internal
documentation with a personal token and downloads a build output from the new location.

### Implementation for User Story 4

- [X] T049 [P] [US4] PR: In `scripts/package_trt_llm.py:171-176`, confirm that `--addr` is never
  used to download; only `:118` and `:171` mention it. If so, leave its default, which names a
  pre-cutover build that stays on urm, and allowlist it with that reason. If the tool does download
  from it, its help text says that URLs on artifactory.nvidia.com need an Artifactory access token
  ([contract](contracts/helper-scripts.md#scriptspackage_trt_llmpy-developer-tool)).
- [X] T050 [US4] PR: On the PR branch, run
  `git grep -nE "urm\.nvidia\.com|sw-tensorrt-generic" -- jenkins scripts tests docs docker`, and
  classify every match by role, scope, and disposition (data-model, Artifact reference). Allowlist,
  with reasons, the unreachable source release package read (`jenkins/L0_Test.groovy:5988-5990`,
  research R13) and `scripts/package_trt_llm.py`'s `--addr` default (T049). Record the references
  kept for the fallback and the rollback (FR-019) and the allowlist in `signoff-evidence.md` and the
  the PR description. Repoint any in-scope match that is neither.
- [ ] T051 [P] [US4] OPS: Before the PR merges, write the internal developer documentation, outside
  this repository, for getting access and a personal token, and for downloading a build output with
  it (FR-008). Follow it with a personal token to download a packaged build from the new location
  (US4 acceptance scenario 2, quickstart prerequisite 3), and record the result in
  `signoff-evidence.md`.
- [ ] T052 [US4] PR: Search the PR's diff (the spec store included, while the draft carries it) and
  description for internal distribution lists, portals,
  chat channels, people, and internal URLs (FR-008, research R11), including the library branch
  name in the `@Library` pin (`feature/trtllm-17014-aws-upload-destination`, T027), and confirm that
  the confidentiality scan in a run of the PR reports no hits (`jenkins/L0_MergeRequest.groovy:710-724`;
  the scan only warns).

**Checkpoint**: The reference inventory and allowlist are ready for the sign-off request.

---

## Phase 7: Polish & Cross-Cutting Concerns (Cutover, Validation, Sign-off)

**Purpose**: Gates and follow-up that span every story (plan, Delivery Sequence steps 3–6).

- [ ] T053 PR: When marking the draft ready (clarified 2026-10-16: only after T008 and T023 hold;
  the draft starts with a short description), amend the PR's one commit to remove `specs/`
  (`git rm -r --cached specs/`, then `git commit --amend`), and complete the PR description. It
  covers:
  - what changed and why, and why it is one PR: the switch cannot be split (cutover at merge, with
    no dual publishing, FR-002), and the helper scripts exist only for it;
  - the Test Coverage list: `tests/unittest/scripts/test_fetch_build_artifact.py`,
    `test_check_artifactory_access.py`, `test_slurm_install.py`, `test_get_wheel_from_package.py`,
    `cbts/test_coverage_artifact.py`, and the library's `UploadArtifactsTest.groovy`;
  - the performance impact (constitution Principle VI): download times per environment against urm
    (T023), the default run's download times against a recent main pre-merge run (T041), and one
    extra storage-API request and a `sha1sum` per download;
  - the CI-wide risk and the `LLM_ARTIFACTS_PRIMARY` rollback switch;
  - that the image build's reads, BoltProfileGen's input read, and the `bolted-<tar>` property run
    only in the post-merge pipeline, so the first post-merge run after cutover validates them, and
    that the optimized-package publish is validated when it is first turned on (T066, research
    R16);
  - that pre-change builds stay readable on urm until it is retired, and that an Artifactory access
    token for artifactory.nvidia.com is needed;
  - that `publishBoltedCanonical` now uploads `unbolted-<tar>` only after its existence check
    answers 404, and fails the publish on any other failed check (T084): a failed check can no
    longer overwrite an existing snapshot, and curl has already retried transient failures five
    times by then;
  - that `jenkins/L0_MergeRequest.groovy:17` points at the library merge request's branch only
    while under test, and that T057 removes the pin before merge;
  - the library merge request, named by its branch with no internal URL (FR-008), and
    TRTLLMINF-472 and TRTLLMINF-295.

  Send the PR for review to all TRT-LLM infrastructure developers, and to the code owners of
  `/jenkins`, `/scripts`, `/docker`, and `/jenkins/scripts/cbts` (FR-024, quickstart step 11).
- [ ] T054 OPS: Before merging, rehearse a rollback and the reuse case (quickstart steps 8 and 6,
  FR-022, FR-016, SC-009). With a Jenkins administrator, set `LLM_ARTIFACTS_PRIMARY=urm`; before
  cutover, only the PR's runs read it. Then:
  1. Run the PR's build stages and one test stage. Confirm that in-scope artifacts publish to urm,
     including `build_info.txt` and `waive_list/` through the shared helper, and that reads try urm
     first and fall back to the new location.
  2. On the BoltProfileGen cluster frontend, run `fetch_build_artifact.sh --exists` through T035's
     export lines, and confirm `served-by=urm` for a file published in step 1. That shows scripts on
     other hosts follow the switch.
  3. Unset the variable, and run the same commit again. The build stage logs
     `reuse source is on urm; rebuilding` and rebuilds on the new location (T045).
  4. Run the same commit once more. The server-side copy on the new location succeeds without a
     rebuild.

  Record when the switch took effect (under an hour) and each result in `signoff-evidence.md`.
- [X] T055 PR: Run `pre-commit` on every file the PR changes. Confirm that new files
  carry the NVIDIA SPDX header, new scripts are executable, and every place that writes a netrc
  (T028, T029, T032, T035, T036, T039, T044, T045) uses `umask 077` and removes it in a `finally`
  block or a trap (FR-012). Run the unit tests in
  `specs/001-migrate-builds-aws-artifactory/quickstart.md` step 1 locally, with
  `pytest -c /dev/null --noconftest` and `PYTHONPATH=jenkins/scripts/cbts` (research R19), and the
  Groovy syntax check on every changed pipeline file.
- [ ] T056 LIB: After review, merge the library merge request (T012) into
  `trtllm-jenkins-shared-lib`'s `main`. Then check quickstart prerequisite 5: a `main` run without
  the PR still publishes `build_info.txt` and `waive_list/` to urm, with unchanged log links. Record
  the result in `signoff-evidence.md`.
- [ ] T057 PR: After T056, change `jenkins/L0_MergeRequest.groovy:17` back to
  `trtllm-jenkins-shared-lib@main`, push, and run the default `/bot run` on that commit; it must
  pass (FR-025, quickstart step 11), including its multi-GPU stages, which need the
  `ci: full pre-merge approved` label from T041 step 0 still on the PR. A `--stage-list` run
  cannot stand in for it, because it does not update the PR's CI status. Confirm that
  `git diff upstream/main...HEAD -- jenkins/L0_MergeRequest.groovy` no longer touches the
  `@Library` line.
- [ ] T058 OPS: Merge the PR (cutover) only when:
  - both identities are verified (T023);
  - T022, T041, T046–T048, T050, and T052–T057 have passed, including the default `/bot run` on the
    final commit (T057);
  - the Phase 1 findings are recorded: token expiry dates and renewal owners (T005), the build-info
    check and any build names sent (T006, FR-017), the urm access records (T007), the code-version
    check (T008), the source release package's producer (T009), and the owner's answer on the
    packaged-build list (T010, FR-001);
  - the internal access documentation exists (T051, FR-008).

  Record the merge commit and time in `signoff-evidence.md`.
- [ ] T059 OPS: After cutover, run quickstart step 10 (FR-021, SC-001, SC-002, SC-003, SC-006):
  - compare the first merge-request run and the first post-merge run that contain the change with
    the runs of the commit just before the merge;
  - for the first 20 merge-request runs and 20 post-merge runs, confirm that `build_info.txt`,
    `waive_list/waives.txt`, and the packaged builds return 200 on the new location and 404 on the
    old one (SC-001), because a failed best-effort upload does not fail its run;
  - for two weeks, search run logs for `fetch_build_artifact` exits 2, 3, and 4 and for the upload
    note naming `aws-artifactory-credentials`, and record each run failure caused by an in-scope
    artifact (SC-002);
  - compare the median merge-request duration in the week after cutover with the week before
    (SC-003);
  - scan the first week's logs for exposed secrets: `Authorization: Basic`, `password` next to
    `artifactory.nvidia.com`, and Artifactory token prefixes (`eyJ` for access tokens, `cmVmdGtu`
    for reference tokens) (SC-006).

  Record the results in `signoff-evidence.md`.
- [ ] T060 OPS: In the first post-merge run after cutover, check the paths that only the post-merge
  pipeline runs (quickstart step 10, spec FR-020, research R16):
  - the image job's "Wait For Build Job Complete" stage finds both packaged builds, with
    `served-by=aws` (T036);
  - the SBSA release image's build log shows the wheel download with `served-by=aws` (T037–T040);
  - BoltProfileGen's input read logs `served-by=aws` (T035), and no netrc file is left on its
    cluster frontend; its log shows `Build artifacts: primary location aws, passed by the parent
    pipeline` (T083, research R7);
  - `fetch_build_artifact.sh --properties` on that run's `bolted-<tar>` returns the same keys as on
    a pre-change post-merge build (FR-004, T026), and the SBSA test stages' `bolt.ref` check reads
    it with `served-by=aws` (T030);
  - quickstart step 7's command, run with that build's number, logs that the new location served
    its `build_info.txt` (T043).

  If the new location makes one of these fail, have a Jenkins administrator set
  `LLM_ARTIFACTS_PRIMARY=urm` while it is fixed (FR-022); a fault in the code needs a fix instead.
  Record the results in `signoff-evidence.md`.
- [ ] T061 [P] OPS: Run quickstart step 9 on each release branch that has not adopted the change.
  Its in-scope artifacts appear only on urm, and it passes as before (FR-018). Record the result in
  `signoff-evidence.md`.
- [ ] T062 [P] OPS: Repeat T050's reference search on every in-scope branch, confirm SC-007, and
  record the result in `signoff-evidence.md`.
- [ ] T063 OPS: Submit the migration for sign-off through the program's support channel, with the
  contents of `signoff-evidence.md` (FR-023, SC-008), and resolve any blockers it raises.
- [ ] T064 OPS: If the owner's FR-018 decision (T010) includes release branches, file an NVBug for
  each one, since its CI breaks when urm retires. Cherry-pick the PR to it as a separate PR that
  names the NVBug in its title (constitution Gate 2), and repeat quickstart steps 3 and 9 there. Otherwise, record in `signoff-evidence.md` that no backport is needed.
- [ ] T065 [P] OPS: Hand off to TRTLLMINF-472 (test results) and TRTLLMINF-295 (wheels). They can
  pass `aws` at the other 14 calls of the shared helper, and reuse both identities and
  `jenkins/scripts/check_artifactory_access.sh`.
- [ ] T066 OPS: When `boltPublishCanonical` is first turned on after cutover, confirm on that run
  that the BOLTed canonical package lands on the new location through `publishBoltedCanonical`
  (`jenkins/BoltProfileGen.groovy:801-840`) with the publisher credential, and that readers then get
  the optimized package (spec FR-020, edge case "Package replaced after the build"). Record it in
  `signoff-evidence.md`.
- [ ] T067 OPS: Before urm is retired, ask the program for urm access records for
  `sw-tensorrt-generic/llm-artifacts/` since cutover, and confirm that TRT-LLM CI reaches it only for
  pre-cutover builds through the fallback (spec edge case "A missed reader of the old location").
  Contact any other client directly, and record the result in `signoff-evidence.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 comes first and must finish before the PR branch is pushed (T018,
  T041). T003 and T004 have the longest lead time, so start them at once.
  T005–T010 must be done before cutover (T058).
- **Foundational (Phase 2)**: The library tasks (T011 → T012), the read helper (T013 → T014), and
  the location helpers (T015) are independent of each other and of Phase 1.
- **US1 (Phase 3)**: T016 → T017 → T018, which needs T014. The verification runs need the grants:
  T019 and T020 after T003 and T018, T021 after T004 and T019, then T022 and T023.
- **US2 (Phase 4)**: Needs T015 for every Groovy task and T014 for every reader. T027's pin uses the
  branch from T011. T041 starts on the draft PR right away; the grants and the access check
  (T003, T004, T023), and the code-version check (T008), gate marking the PR ready (T053).
- **US3 (Phase 5)**: Needs T015 and T014. Its validation runs after T041 (T046, T047) or from a
  checkout of the PR once T043 is done (T048). T054 validates T045.
- **US4 (Phase 6)**: T050 and T052 need the PR's final code from US2 and US3.
- **Polish (Phase 7)**: T053–T055 come before T056; then T056 → T057 → T058 (cutover). T059 and
  T060 use the first runs after cutover. T061–T065 come after cutover. T066 and T067 happen when
  their trigger comes.

### User Story Dependencies

- **US1 (P1)**: Independent after Foundational. It gates marking the PR ready (T053) and the
  cutover (T058), not the PR's code or its draft CI runs (T041).
- **US2 (P1)**: Independent after Foundational. It shares `jenkins/Build.groovy` and
  `jenkins/L0_MergeRequest.groovy` with US3, so T045 follows T026 and T044 follows T027.
- **US3 (P2)**: Independent after Foundational, apart from those shared files. It ships in the PR
  with US2, because FR-014 and FR-016 apply from the first run that contains the change.
- **US4 (P2)**: Reviews the finished PR. T049 and T051 can start at any time.

### Within Each User Story

- Tests come first and fail before the implementation: T013 → T014, T016 → T017, T024 → T033,
  T025 → T040, and T042 → T043. The library's tests already exist (T011).
- Tasks on the same file run in order:
  - `jenkins/L0_Test.groovy`: T029 → T030 → T031 → T032 → T034
  - `jenkins/BoltProfileGen.groovy`: T028 → T035
  - `jenkins/BuildDockerImage.groovy`: T036 → T039, with T037 and T038 also before T039
  - `jenkins/Build.groovy`: T026 → T045
  - `jenkins/L0_MergeRequest.groovy`: T027 → T044 → T057
- Validation tasks come last in each story.

### Parallel Opportunities

- Setup: T003–T010 at once, after T001 and T002.
- Foundational: T011, T013, and T015, which are in different repositories and files.
- US1: T016 alongside T013, since both are helper tests in different files.
- US2: T024 and T025 together, then T026, T027, T028, T029, T033, T036, T037, T038, and T040 at
  once, since each is in a different file.
- US3: T042 alongside T044 and T045, once their US2 predecessors are done.
- US4: T049 and T051.
- After cutover: T061, T062, and T065.
- Different developers can take the library, the helper scripts, the Groovy work, and the other
  script and Docker work at the same time.

---

## Parallel Example: User Story 1

```bash
# Helper tests, together (T013 is Foundational but ships in the same commit):
Task: "T013 Write tests/unittest/scripts/test_fetch_build_artifact.py"
Task: "T016 Write tests/unittest/scripts/test_check_artifactory_access.py"
```

## Parallel Example: User Story 2

```bash
# Tests first, together:
Task: "T024 Update tests/unittest/scripts/test_slurm_install.py"
Task: "T025 Write tests/unittest/scripts/test_get_wheel_from_package.py"

# Then one task per file, together:
Task: "T026 Publish packaged builds to the primary in jenkins/Build.groovy"
Task: "T027 Pass the destination and pin the library in jenkins/L0_MergeRequest.groovy"
Task: "T028 Publish the BOLTed canonical to the primary in jenkins/BoltProfileGen.groovy"
Task: "T029 Fetch, stash, and wrap the read helper in jenkins/L0_Test.groovy"
Task: "T033 Download with curl, a netrc, and a checksum in jenkins/scripts/slurm_install.sh"
Task: "T036 Poll with the helper in jenkins/BuildDockerImage.groovy"
Task: "T037 Pass the netrc as a BuildKit secret in docker/Makefile"
Task: "T038 Mount the helper and the secret in docker/Dockerfile.multi"
Task: "T040 Download through the helper in scripts/get_wheel_from_package.py"
```

## Parallel Example: User Story 3

```bash
Task: "T042 Extend tests/unittest/scripts/cbts/test_coverage_artifact.py"
Task: "T044 Run artifact.py with the reader netrc in jenkins/L0_MergeRequest.groovy"
Task: "T045 Apply the reuse rule in jenkins/Build.groovy"
```

---

## Implementation Strategy

### MVP First (User Stories 1 and 2)

1. Phase 1: separate the spec store (T001), then start the permission requests (T003, T004) at
   once, since they have the longest lead time.
2. Phase 2: confirm the library change and send it for review (T011, T012) without waiting for it
   to merge, and write the read helper.
3. Phase 3 (US1): commit the helpers, push them to the fork, and prove access in every
   environment, including the BoltProfileGen cluster frontend.
4. Phase 4 (US2): build the PR against the library merge request's branch, and validate it with the
   default pre-merge run, then targeted runs (T041).
5. **Stop and validate**: US1 and US2 together are a working cutover. History tools would degrade,
   but CI would pass, which is why US3 is P2.

### Incremental Delivery

1. The helpers, then access verified (US1), with the library merge request under review in
   parallel.
2. The PR grows by story (US2, then US3, then US4), and each story is validated on the PR's runs before
   the next. Before the PR merges, the rollback and reuse are rehearsed (T054), the library merge
   request merges (T056), the pin is removed (T057), and the Phase 1 findings are in (T058). The PR
   merges once, because the cutover happens at merge (FR-002).
3. After cutover: the first runs (T059), the post-merge-only paths on the first post-merge run
   (T060), sign-off, release-branch backports if the owner decides on them, and the hand-off. T066
   and T067 follow when their triggers come.

### Parallel Team Strategy

- Developer A: the library change (T011, T012, T056).
- Developer B: the helper scripts (T013, T014, T016–T018), then the Python work in US3 (T042, T043).
- Developer C: the Groovy work (T015, T026–T032, T034–T036, T039, T044, T045, T057).
- Developer D: the other scripts and the Docker work (T024, T025, T033, T037, T038, T040, T049).
- CI maintainer: Phase 1 and the OPS verification tasks.

---

## Notes

- [P] tasks touch different files and do not depend on incomplete tasks.
- Commit with `git commit -s`. Never name an AI tool in the sign-off line, and add no co-author
  trailers unless the user asks (AGENTS.md). If `pre-commit` hooks modify files, re-stage them and
  commit again.
- The PR comes from the fork's `TRTLLM-17014-migrate-builds-to-aws-artifactory` and targets
  `NVIDIA/TensorRT-LLM` `main`, titled `[TRTLLM-17014][infra] ...`. It is always one commit; the
  scrubbed spec store rides in it only while the PR is a draft.
- Repository files, commit messages, PR descriptions, and branch names that appear in them name no
  internal distribution lists, portals, chat channels, people, or internal URLs (FR-008). Tool
  messages say only that an Artifactory access token for artifactory.nvidia.com is needed.
- Secrets never appear on command lines, in URLs, or under `set -x`. Netrc files are written with
  `umask 077` and removed in `finally` or a trap (FR-012).
- Every download of an in-scope artifact is checked against the SHA-1 that the serving location
  reports, and a 403 after a redirect to the CDN is retried once without the redirect (FR-014,
  research R18).
- Steps that are best effort today stay best effort after a rejected credential: they log the
  credential message and carry on, and never read from or publish to the other location instead
  (FR-013).
- Comments explain the code as it stands; rationale and history go in PR descriptions.
- CI runs of the PR validate the Groovy changes and the image-build files where a pull request
  reaches them (T041, T046–T048, T054, T057); the first post-merge run after cutover validates the
  rest (T060). The scripts have unit tests.
- The shared library is reached over SSH only.
- The spec store is committed only into the draft PR's single commit, with internal names replaced
  by role descriptions, and removed from it before the PR is marked ready (T053). `.specify/` and
  the speckit skills are never committed.

---

## Phase 8: Convergence

**Purpose**: Remaining work that `/speckit-converge` found on 2026-10-16, after the clarification
that every read goes through one fetch script (spec FR-005). The PR's commit (`950dc538ab`) still
has a copy of the read rule in `slurm_install.sh` and another in `artifact.py`. T068–T073 replace
them and amend the commit; the rest are smaller gaps. Tests come first: T068 → T069 → T070, and
T071 → T072. Tasks on `jenkins/L0_Test.groovy` run in order: T070 → T074 → T080 → T081.

- [X] T068 PR: Rewrite `tests/unittest/scripts/test_slurm_install.py` for a Slurm node that runs the
  read helper, per FR-005 (contradicts). Today it tests `slurm_install.sh`'s own download and the
  launcher's checksum and aws-only credential exports
  ([contract](contracts/helper-scripts.md#jenkinsscriptsslurm_installsh-changed)):
  - Remove the tests of `download_llm_tarfile` and their curl stub: `test_download_*`,
    `test_failed_download_*`, `test_cdn_*`, `test_persistent_cdn_*`, `test_artifactory_rejection_*`,
    `test_anonymous_rejection_*`, and `test_unreadable_netrc_*`. The read rule, the CDN retry, and
    the checksum stay tested once, in `test_fetch_build_artifact.py`.
  - In `test_slurm_install_requires_artifact_inputs`, require `llmTarArtifactPath` instead of
    `llmTarfile`.
  - Rewrite `test_slurm_install_setup_downloads_package_through_retries` and
    `test_slurm_install_setup_fails_after_retries` to run `slurm_install_setup` from a
    `<prefix>slurm_install.sh` copy with a stub `<prefix>fetch_build_artifact.sh` beside it, as the
    job workspace names them. Assert that the stub gets `$llmTarArtifactPath` and the job-specific
    temporary path and sees `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`, and
    `ARTIFACTORY_CREDENTIAL_ID`; that `retry_command` retries a failed attempt; that the archive is
    moved into place only after a successful one; and that a netrc password in the environment
    never reaches the output, although the script runs with xtrace.
  - Replace `test_l0_slurm_node_gets_checksum_and_aws_only_credentials` with a check of
    `runLLMTestlistWithSbatch` in `jenkins/L0_Test.groovy`: it copies the helper to
    `${jobWorkspace}/${jobUID}-fetch_build_artifact.sh`; writes and copies the reader netrc for every
    job; exports `llmTarArtifactPath`, `LLM_ARTIFACTS_PRIMARY`, `ARTIFACTORY_NETRC`,
    `ARTIFACTORY_CREDENTIAL_ID`, and `llmTarfile` without a condition; has no `llmTarfileSha1`; and
    lists the helper and the netrc in `filesToKeepWhenRetry`. Keep the netrc writer checks (one
    `machine artifactory.nvidia.com` entry, no `default`).
  - Keep `test_retry_command_returns_terminal_status` and the `test_l0_*` checks that every
    packaged-build read goes through `fetchBuildArtifact`.
- [X] T069 PR: In `jenkins/scripts/slurm_install.sh`, download the packaged build by running the read
  helper on the node, per FR-005 (contradicts):
  - Delete `download_llm_tarfile` (`:25-121`).
  - Find the helper beside the script from the script's own file name, as `bashUtilsPath` finds
    `bash_utils.sh` (`:22`): `<prefix>slurm_install.sh` → `<prefix>fetch_build_artifact.sh`. That
    holds on every launch path: `slurm_run.sh` sources the script, and the multi-node perf scripts
    run it by its path.
  - Require `llmTarArtifactPath` instead of `llmTarfile` (`:125`).
  - Download as `dac84b01a6` does, with the helper in place of `wget` (`:158-162`):
    `retry_command --timeout 1800 bash -c 'bash "$1" "$2" "$3" && mv -f "$3" "$4"' _ "$fetchHelperPath" "$llmTarArtifactPath" "$archive_tmp" "$archive_path"`,
    and name `$llmTarArtifactPath` in the failure message.
  - Leave the rest as it is at `dac84b01a6`. The LLM image already runs the helper in Kubernetes
    test pods, so it has `curl` and `sha1sum`.

  T068 passes.
- [X] T070 PR: In `runLLMTestlistWithSbatch` in `jenkins/L0_Test.groovy`, send the read helper and
  the reader netrc to every Slurm job, per FR-005 (contradicts):
  - Copy the helper that `fetchBuildArtifact` unstashes into
    `${pipeline.pwd(tmp: true)}/fetch-build-artifact/` (factor that unstash into a function that
    returns the path) to `${jobWorkspace}/${jobUID}-fetch_build_artifact.sh` with
    `Utils.copyFileToRemoteHost`, beside `slurm_install.sh` and `bash_utils.sh` (`:1989-2003`).
  - Write and copy the reader netrc for every job, not only when `llmTarfileOnAws`
    (`:2092-2111`): the helper on the node contacts aws whenever aws is the primary.
  - In the launch prefix (`:2344-2350`), which both launch paths use, export `llmTarArtifactPath`,
    `LLM_ARTIFACTS_PRIMARY=${llmArtifactsPrimary()}`, `ARTIFACTORY_NETRC`, and
    `ARTIFACTORY_CREDENTIAL_ID` without a condition. Keep `export llmTarfile=<url>` from the
    `--exists` result, which only the older `slurm_install.sh` in a package built before this change
    reads, and drop `llmTarfileSha1`.
  - List the helper and the netrc in `filesToKeepWhenRetry` without a condition (`:2450-2454`).
  - Remove `llmTarfileOnAws` (`:1978`), and `LLM_ARTIFACTS_LOCATIONS` (`:56-59`) if nothing else
    reads it (T015: each file gets only what it uses).

  T068 passes.
- [X] T071 PR: Rewrite the `build_commit` tests in
  `tests/unittest/scripts/cbts/test_coverage_artifact.py` for an `artifact.py` that runs the read
  helper, per FR-005 (contradicts). Today they patch `urllib.request.urlopen` and test the module's
  own copy of the read rule
  ([contract](contracts/helper-scripts.md#jenkinsscriptscbtscbtscommandcoverageselectionartifactpy-changed)):
  - Point the module at a stub helper that records its arguments and environment, writes the file,
    and exits with a chosen status. Exit 0 returns the commit, and the stub got
    `<artifact_base>/<build>/build_info.txt` and a destination file, with the environment passed
    through. Exit 2 returns `None`. Exit 3 raises `ArtifactAuthError`, 4 `ArtifactTransferError`,
    and 64 `ArtifactError`. Nothing the helper prints on stdout reaches the command's stdout, which
    carries JSON. `main` turns each error into a message on stderr and exit code 1.
  - Add one test that runs the real `jenkins/scripts/fetch_build_artifact.sh` against two local
    servers, through `LLM_ARTIFACTS_AWS_URL` and `LLM_ARTIFACTS_URM_URL`, with aws answering 404 and
    urm 200, and gets the commit.
  - Remove the tests of what T072 deletes: `build_info_url`, the location overrides, netrc parsing,
    and the module's rejection messages. The tests that predate this work keep passing.
- [X] T072 PR: In `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`, read
  `build_info.txt` by running the read helper, per FR-005 (contradicts):
  - Delete the module's copy of the read rule: the location and environment-name constants,
    `_Location`, `_primary_location`, `_location`, `_layout_path`, `_credential_id`,
    `_no_credentials`, `_authorization`, `_rejection`, `_location_get`, `_reported_sha1`,
    `_read_location`, `_read_build_artifact`, and `build_info_url`, with the imports nothing else
    uses.
  - Run `jenkins/scripts/fetch_build_artifact.sh`, found relative to the module
    (`Path(__file__).resolve().parents[5]`), as
    `bash <helper> <artifact path> <file in a temporary directory>` with `subprocess.run`: no shell,
    the environment passed through, stdout discarded, and stderr left to the job log. Map its exit
    status: 0 the file's content, 2 `None`, 3 `ArtifactAuthError`, 4 `ArtifactTransferError`,
    anything else `ArtifactError`. An `OSError` from starting it is an `ArtifactError`.
  - `build_commit` reads `f"{artifact_base}/{build}/{BUILD_INFO_NAME}"` that way and keeps its
    parsing. Keep the three exception classes and `main`'s handler.
  - Say in the module's comments that `build_info.txt` is read through the helper.

  T071 passes. T044's environment wiring stays as it is.
- [X] T073 PR: When T068–T072, and whichever of T074–T084 change code, are done, re-run the PR's
  checks and amend its one commit, per plan: Delivery Sequence step 3 (partial). The pushed commit
  `950dc538ab` carries the replaced read design and older design documents, and T050 and T055 ran
  against it:
  - Run T055's checks on the changed files: `pre-commit`; the five test modules in quickstart step
    1, with `pytest -c /dev/null --noconftest` and `PYTHONPATH=jenkins/scripts/cbts`; `shellcheck`
    on the changed scripts; and the Groovy syntax check on the changed pipeline files.
  - Re-run T050's reference search, and update the reference table in `signoff-evidence.md` if the
    test fixtures it lists changed.
  - Run T052's search for internal names, then amend the PR's single commit with the code and the
    current `specs/` (added with `git add -f` while the PR is a draft), and push it to the fork's
    branch with `--force-with-lease`, never to `upstream`.
- [X] T074 PR: In `runLLMTestlistWithSbatch` (`jenkins/L0_Test.groovy:2097-2099`), record this
  attempt's job workspace for the off-pod cleanup before the reader netrc is copied into it, even
  when an earlier attempt's `<stage>/slurm` ledger entry is still live, per FR-012 (partial). In
  that case the early registration is skipped today, so if the pod dies between the copy and the job
  submission (`:2616`), the netrc stays in the workspace on the login host. After T070 every job gets
  the netrc. For example, give each attempt its own ledger entry, keyed by `jobUID`.
- [X] T075 PR: Find out whether the workspace of a Jenkins agent on a Slurm node
  (`customWorkspace = "/tmp/${nodeName}"`, `jenkins/L0_Test.groovy:1099`) outlives the Slurm job,
  per FR-012 (partial). `fetchBuildArtifact` writes the reader netrc under its `@tmp` and removes it
  in `finally`, but `cleanUpNodeResources` (`:859-899`) does not touch that directory, so an agent
  killed mid-fetch could leave the netrc there. If the workspace outlives the job, remove leftover
  `artifactory-netrc-*` files after the job, or write the netrc where the existing cleanup removes
  it; otherwise record in research R5 why none can remain.
- [X] T076 PR: Make `map_path` in `jenkins/scripts/fetch_build_artifact.sh` (`:91-103`) reject, with
  exit 64, what `layout_path` in `jenkins/scripts/check_artifactory_access.sh` (`:74-86`) rejects (an
  empty, `.`, or `..` segment, whitespace, `;`, `?`, and `#`), and make the base-URL overrides
  refuse embedded credentials as the check does (`:97-99`), per T017 (contradicts). T017 calls for
  the same path mapping in both scripts. Add the cases to `test_unmappable_path_is_a_usage_error` in
  `tests/unittest/scripts/test_fetch_build_artifact.py`.
- [X] T077 PR: In `jenkins/scripts/fetch_build_artifact.sh`, say in the header comment (`:17-19`)
  that the helper reads any file under `llm-artifacts/`, not only the three in-scope kinds, and
  parametrize one success test in `tests/unittest/scripts/test_fetch_build_artifact.py` over
  `build_info.txt`, `waive_list/waives.txt`, and a file outside those kinds, per FR-005 (partial).
  The follow-up migrations (TRTLLMINF-472, TRTLLMINF-295) rely on that (research R4).
- [X] T078 PR: Turn xtrace off at the start of `jenkins/scripts/fetch_build_artifact.sh`, and add a
  test that runs it with xtrace inherited (`SHELLOPTS=xtrace` in its environment) through a CDN
  redirect and checks that neither the netrc password nor the CDN URL's signature reaches the
  output, per FR-012 (partial). After T069 it runs under `slurm_install.sh`, which uses `set -x`.
- [X] T079 PR: Bring `specs/001-migrate-builds-aws-artifactory/contracts/helper-scripts.md` up to
  date with what the scripts already do, per plan: contracts (unrequested):
  - `check_artifactory_access.sh` prints a `TIME <location> <seconds> s <MB/s> MB/s <url>` line per
    location for `--time`, and exits 64 on a usage error;
  - `fetch_build_artifact.sh` exits 3 without naming a credential when urm is the primary and
    rejects the anonymous request;
  - the image build passes `ARTIFACTORY_CREDENTIAL_ID` as a build argument (`docker/Makefile`,
    `docker/Dockerfile.multi`, `jenkins/BuildDockerImage.groovy`), so that
    `get_wheel_from_package.py` can name the credential.
- [X] T080 PR: In `boltedTarArtifactPath` (`jenkins/L0_Test.groovy:1863-1882`), log a property
  lookup that kept failing (helper exit 4) as a failed lookup rather than as
  `carries no bolt.ref (artifact predates the label)`, and keep going on the pin as today, per T030
  (partial).
- [X] T081 PR: Restore the `pwd` and `ls -alh` that the test-list check ran around its
  packaged-build download (`jenkins/L0_Test.groovy:4368`; at `dac84b01a6`,
  `pwd && wget -nv -O '${tarName}' ... && ls -alh`), which the change dropped, per T034
  (unrequested).
- [X] T082 PR: In the image job's "Wait For Build Job Complete" stage
  (`jenkins/BuildDockerImage.groovy:1166-1171`), make the `error()` text for helper exit 3 also say
  that the credential's token needs renewal, as the helper's own message does, per FR-013
  (partial).
- [X] T083 PR: Make BoltProfileGen's use of its parent's primary visible, per research R7
  (partial). Its declarative `parameters` block replaces the job's parameter list on every run
  (`jenkins/BoltProfileGen.groovy:991-996`), so the new `globalVars` parameter (`:1088-1092`) is
  dropped on the first run of this Jenkinsfile, and again after any run from a branch without the
  change; the job then uses its own controller's `LLM_ARTIFACTS_PRIMARY`. Log
  `Build artifacts: primary location <primary>, passed by the parent pipeline` when the parent's
  value arrives, as `jenkins/Build.groovy:979` and `jenkins/BuildDockerImage.groovy:1090` do; record
  the caveat in research R7; and in T060, check for that line in BoltProfileGen's log.
- [X] T084 PR: Review the existence check before uploading `unbolted-<tar>` in
  `publishBoltedCanonical` (`jenkins/BoltProfileGen.groovy:951-966`), per T028 (unrequested). It
  now uploads only after a 404 and fails the publish on any other status, while T028 asked only to
  fail on 401 or 403. Keep it and say why in the PR description (a failed check can no longer
  overwrite an existing snapshot), or restore today's rule apart from the credential failure.
- [ ] T085 OPS: In T054, also run one Slurm test stage, and confirm that its launch script, printed
  in the stage log, exports `LLM_ARTIFACTS_PRIMARY=urm`, per plan: quickstart step 8 (partial).
  After T070 the node chooses the location itself.
- [ ] T086 PR: When completing the PR description (T053), state that every reader runs
  `jenkins/scripts/fetch_build_artifact.sh`, and that BoltProfileGen keeps its parallel ranged
  download, its only user, and takes the location and checksum from the helper, per plan: Gate 5
  (partial).

## Phase 9: Convergence

**Purpose**: Remaining work that `/speckit-converge` found on 2026-10-16, after the clarification
that the read helper and its readers refer to each exit code by the same name (spec FR-005). The
helper and every reader still use bare numbers (for example `propsStatus in [0, 2, 4]`), and no test
compares a reader's values with the helper's. Names: `FETCH_ABSENT` (2), `FETCH_REJECTED` (3),
`FETCH_TRANSFER_FAILED` (4), `FETCH_USAGE` (64); success stays `0`. The test comes first:
T087 → T088 → T089–T092 → T093 → T094.

- [X] T087 PR: Add a test to `tests/unittest/scripts/test_fetch_build_artifact.py` that reads the
  helper's `readonly FETCH_<NAME>=<n>` lines and fails when a reader defines one of those names with
  another value, or defines none, per FR-005 (missing). Readers: `jenkins/L0_Test.groovy`,
  `jenkins/BuildDockerImage.groovy`, the cluster script in `jenkins/BoltProfileGen.groovy`,
  `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`, and
  `scripts/get_wheel_from_package.py`. Also fail when one of the checks listed in T089–T092 still
  compares the helper's status with a bare 2, 3, 4, or 64. The CPU stage runs this file pre-merge.
- [X] T088 PR: In `jenkins/scripts/fetch_build_artifact.sh`, define `readonly FETCH_ABSENT=2`,
  `FETCH_REJECTED=3`, `FETCH_TRANSFER_FAILED=4`, and `FETCH_USAGE=64` beside the other constants,
  and use them in every `exit`, `fail`, and `usage_error` that ends the script, per FR-005
  (missing). The step functions' own return values (0 done, 1 transient, 2 absent, 3 rejected) may
  use the same names where they mean the same outcome. The usage text keeps the numbers. The
  existing tests in the file must pass unchanged.
- [X] T089 PR: In `jenkins/L0_Test.groovy`, define `FETCH_ABSENT`, `FETCH_REJECTED`,
  `FETCH_TRANSFER_FAILED`, and `FETCH_USAGE` as top-level constants next to
  `FETCH_BUILD_ARTIFACT_STASH`, and use them in `boltedTarArtifactPath`'s status checks and log
  message and in `fetchBuildArtifact`'s comment, per FR-005 (partial).
- [X] T090 PR: In `jenkins/BuildDockerImage.groovy`, define the same top-level constants and use
  them in the wait stage's checks of `exitCode`, per FR-005 (partial).
- [X] T091 PR: In `jenkins/BoltProfileGen.groovy`, give the cluster script that resolves the input
  package (`tarStage`) `FETCH_ABSENT`, and compare `rc` with it instead of `2`, per FR-005
  (partial).
- [X] T092 PR: In `artifact.py`, define `_FETCH_ABSENT`, `_FETCH_REJECTED`, and
  `_FETCH_TRANSFER_FAILED`, and key `_HELPER_FAILURES` and the `status in (0, 2)` check by them; in
  `scripts/get_wheel_from_package.py`, rename `_FETCH_NOT_FOUND` to `_FETCH_ABSENT`, per FR-005
  (partial). Their tests in `tests/unittest/scripts/` must pass unchanged.
- [X] T093 Docs: Add each code's name to the exit-code table in
  [contracts/helper-scripts.md](contracts/helper-scripts.md), and state that readers refer to the
  codes by those names and that T087's test keeps them equal, per FR-005 (partial).
- [X] T094 PR: Run the quickstart's unit tests, pre-commit, and the Groovy parse check, then amend
  the PR's one commit, which also carries the spec change, and push it to the fork, as T073 did, per
  FR-024 (missing).
