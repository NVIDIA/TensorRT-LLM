<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Feature Specification: Migrate Test Results to AWS Artifactory

**Feature Branch**: `TRTLLMINF-472-migrate-test-results-to-aws-artifactory`

**Created**: 2026-10-27

**Status**: Draft

**Input**: User description: "TRTLLMINF-472 — Migrate test results to AWS Artifactory. This follows
TRTLLM-17014, which moves CI's build artifacts (build_info.txt, packaged builds, waive_list/) from
urm.nvidia.com/.../sw-tensorrt-generic/llm-artifacts to
artifactory.nvidia.com/.../sw-tensorrt-llm-generic-local/llm-artifacts and left test results on urm
for this ticket. It lands after TRTLLM-17014 is merged and validated, on main only, and reuses what
that ticket settled instead of reopening it (its spec, plan and research are in
specs/001-migrate-builds-aws-artifactory): the publishing identity aws-artifactory-credentials
(read, deploy, annotate, delete/overwrite), which CI also reads with; the read helper
jenkins/scripts/fetch_build_artifact.sh (new location first, urm only on a 404, SHA-1 check, CDN
handling, exit codes by name); the LLM_ARTIFACTS_PRIMARY rollback switch and the legacy-path
mapping; the shared library's uploadArtifacts destination argument, already merged; no copying of
old builds, which stay readable through the fallback; release branches staying on urm; and its rules
for CI validation runs, review findings, confidentiality and sign-off evidence. Blockers and sign-off
go through the program's support channel.

In scope are the test results CI writes under `llm-artifacts/<job>/<build>/test-results/`: each test
stage's final `results-<stage><postTag>.tar.gz` (one per attempt), the
`results-<stage><postTag>-progress.tar.gz` checkpoint that each stage re-uploads every ten minutes
and deletes when it ends, and the build-level rerun_report.html, along with every CI step that
writes, reads, deletes or links them. Decide whether three neighbours move with them: the docs
preview doc-html-preview.tar.gz, which sits in test-results/ but is not a test result; the per-stage
`rerun_reports/<stage>_rerun_results.html` beside it; and the cbts-coverage/ archives built from the
results tarballs. Wheels (TRTLLMINF-295) and the other llm-artifacts outputs stay out of scope; the
old repository cannot be retired until everything has moved.

The new location rejects anonymous reads, and every reader of test results reads urm anonymously
today, so access is the core of this work. In CI, three readers lose data silently when a read
fails: the parent pipeline's Collect Test Result step, which lists test-results/ through urm's HTML
index and downloads every tarball to build the run's JUnit report (which also feeds the test history
behind test-level reuse); the post-merge per-architecture coverage step, which reads them the same
way; and a retried stage reading its earlier attempts' tarballs to reuse tests that passed. The read
helper cannot list a directory. Outside this repository: for failed runs, PR_Github reads
test-results/ with a urm credential and republishes a scanned copy that .github/workflows/l0-test.yml
downloads to set the required blossom-ci status; for passing runs it fetches rerun_report.html
anonymously and links it in a PR comment; perf records in OpenSearch carry a urm link; at least two
internal tools read test-results/ by fixed urm address (whether they run in production is
unconfirmed); and people open test-result links from console logs, PR comments and CI reports, which
would require each of them to hold a token from the program's self-service portal and membership in
its distribution list.

Open questions: how a test-results directory is listed on the new location; the order of changes and
any transition for outside readers, since blossom-ci depends on one of them, and whether review by
the infrastructure developers is still enough notice; what people who follow links need; whether
test results share LLM_ARTIFACTS_PRIMARY or get their own switch; which steps stay best effort, and
whether the silent readers should fail on a rejected credential; how the progress uploads and
deletes, which today pass urm-artifactory-creds to curl on the command line (the uploads in the same
step that runs the tests), get the publishing identity without exposing it; whether uploads and
deletes work from Jenkins agents inside Slurm jobs on remote clusters, where TRTLLM-17014 proved only
reads; retention and volume (a pre-merge run writes about 75 files, 52 MiB, almost all of it the docs
preview, plus about 400 progress uploads; a post-merge run writes about 180 files, 316 MiB); per-step
performance next to the pipeline-duration criterion; and how to validate paths that a pull request's
CI rarely reaches (the retry read and rerun reports need failing tests, and the coverage step runs
only after merge). The spec names roles, not people, and names the program's portal, distribution
list and support channel only generically."

## Clarifications

### Session 2026-10-27

- Q: Which of the three neighbours move with the test results: the docs preview
  (`doc-html-preview.tar.gz`), the per-stage rerun reports (`rerun_reports/`), and the coverage
  archives (`cbts-coverage/`)? → A: All three (the author). The docs preview and the per-stage rerun
  reports move with the test results, and so do the coverage archives. Coverage-based test selection
  then reads coverage archives through the read helper, which falls back to the old location for runs
  published before cutover, as its read of build_info.txt already does; its pins stay on the old
  location (FR-001, FR-005).
- Q: Checkpoints are uploaded every ten minutes from the step that runs the tests. While tests run,
  where may the publishing token be? → A: Out of the tests' reach (the author). The steps that write,
  replace, or delete test results make no Artifactory credential, for either location, reachable by
  the processes that run tests: not in their environment, not in a file they can read, and not on a
  command line. Checkpoint uploads and deletes run from a process that the tests cannot read. This
  also ends today's exposure of the old location's credential to test processes. The netrc that
  TRTLLM-17014 places in a Slurm batch job's workspace for its packaged-build read is not changed
  (FR-010).
- Q: `blossom-ci` depends on PR_Github reading test results. In what order do PR_Github's change and
  the switch land? → A: PR_Github first (the author). PR_Github reads the new location first, with
  the publishing identity, falls back to the old location when a run's test results are absent there,
  and links the rerun report where it is. Its change lands before the switch pull request's CI runs
  that count as validation, and changes nothing until runs publish to the new location; the switch
  pull request's own failed and passing runs then exercise it. Cutover happens at the merge, with no
  period of publishing to both (FR-002, FR-012). Updated later in this session: besides its reads,
  the change also moves PR_Github's redacted copies to a path of their own (see below). Updated on
  2026-10-28: everything under `test-results/` is published to both locations until urm is retired,
  and PR_Github's change lands before then instead of before the merge (Session 2026-10-28).
- Q: Where should PR_Github's redacted copies go so that they never share a path with the originals
  on the new instance? → A: To a path of their own, set in PR_Github's change, which lands first (the
  author). The originals keep TRTLLM-17014's layout. The release bundle that carries the redacted
  copies to the edge site, and the URL handed to `.github/workflows/l0-test.yml`, follow the new
  path; redacted copies published before the change stay where they are (FR-012). Updated on
  2026-10-28: the change lands before urm is retired, not before the merge.
- Q: How should the internal tools that read `test-results/` by fixed urm address move over? → A:
  Their owners move them, and the merge does not wait (the author). Before the merge, whether each
  tool runs in production is determined, and its owner is told what changes, when, and what access
  the new location needs; a tool that has not moved is recorded with the sign-off request (FR-015).
- Q: What must be in place for people who open test-result links once results move to the new
  instance? → A: The same as on urm today, through a wider grant (the author). People open
  test-result links in a browser with their normal company sign-in, without a personal token or
  joining the program's distribution list. This work asks the program to grant read on the in-scope
  outputs to everyone who can open urm links today, and the switch merges only once that grant is in
  place and a developer outside the infrastructure team has opened a test-result link on the new
  instance that way (FR-016).
- Q: Should FR-010 keep requiring that no Artifactory credential is reachable by the processes that
  run tests, or only that checkpoint uploads stay within TRTLLM-17014's accepted risk for hosts that
  run a pull request's code? → A: Keep it (the author). The rule decided earlier in this session
  stands; it is stricter than TRTLLM-17014's accepted risk, so it stays within it, and it holds on
  every runtime, including Slurm agents whose tests run inside the agent's own container (FR-010).
- Q: How should CI treat a checkpoint that is still there after its stage attempt's final archive was
  uploaded? → A: As today (the author). Collect Test Result and the coverage step read every archive
  they list, checkpoints included, and leftover checkpoints stay until retention removes them. A
  failed delete can still bring a superseded attempt's results into the run's report, as it can today
  (FR-004, FR-006).

### Session 2026-10-28

- Q: During the transition, which test-result outputs should CI also publish to urm, beside the new
  location? → A: Everything under `test-results/` (the author, at a reviewer's request to publish
  test results to both instances until the repositories that read them have moved). CI keeps
  publishing urm's `test-results/` as it does today, the final archives, the checkpoints with their
  uploads and deletes, the build-level rerun report, and the docs preview, and replicates it to the
  same paths on the new location; `rerun_reports/` and `cbts-coverage/` go to the new location only.
  This repository's readers move to the new location in the switch pull request, and readers outside
  it move afterwards. This replaces "no period of publishing to both" for test results; build
  artifacts keep TRTLLM-17014's rule (FR-002, FR-004).
- Q: What should end dual publishing of test results to urm, and how should it be turned off? → A:
  urm's retirement (the author). Test results are published to both locations until the program
  retires urm, and nothing ends dual publishing earlier, so readers outside this repository can move
  at their own pace until then. Before the retirement date, a follow-up change removes the
  old-location copies together with the fallback reads (FR-002, FR-015).
- Q: With dual publishing until urm retires, does PR_Github still have to change before the switch
  merges? → A: No, it moves before urm retires (the author). PR_Github keeps reading urm's copies, so
  `blossom-ci` and the PR comment work unchanged, and the switch merges without waiting for it. Its
  change (reads on the new location first, redacted copies at a path of their own, the comment's
  link in the native browser) lands any time before urm is retired, and is proven on a failed and a
  passing run when it lands. The switch pull request's CI runs show PR_Github still working from the
  old location's copies (FR-012, FR-019).
- Q: During dual publishing, when a stage's upload to one of the two locations fails, which failure
  should fail the stage? → A: urm's, as today; the replica is best effort (the author). A failed
  upload of a final archive to urm still fails its stage. A failed replica upload to the new location
  is logged, with the credential message when the credential is rejected, and marks the stage
  unstable; this repository's readers then find the urm copy through the fallback. When urm retires,
  the new location's upload takes over today's rule (FR-007). Refined later in this session: a
  failed checkpoint upload marks nothing (see below).
- Q: Should SC-013 count only PR_Github's redacted copies that land on the new instance, so that the
  switch does not wait for PR_Github's own path change? → A: Drop SC-013 (the author). Where
  PR_Github publishes its redacted copies is PR_Github's concern, and this specification focuses on
  this repository's change. It keeps one check: before the switch pull request is marked ready,
  PR_Github is shown not to publish redacted copies to the new instance, where they would replace
  originals (FR-019).
- Q: When a checkpoint fails to upload to the new location but succeeds on urm, should the stage be
  marked unstable, or should the failure only be logged? → A: Logged only, as today for checkpoints
  (the author). A failed checkpoint upload or delete, on either location, marks nothing; a failed
  replica upload of a final archive, the docs preview, or the build-level rerun report still marks
  the stage unstable (FR-004, FR-007).
- Q: When coverage-based test selection cannot read one candidate run's coverage archive after the
  read helper's retries, should it give up on coverage selection for the run, or skip that candidate
  and try older post-merge runs? → A: Skip it and try older runs, following TRTLLM-17014's logic for
  `build_info.txt` (the author). A read that fails in transfer, including a rejection by the other
  location, skips that candidate as an absent archive does; a rejection by the location runs
  currently publish to still declines at once, since every other read would be rejected too
  (FR-007).

## User Scenarios & Testing *(mandatory)*

People affected:

- **CI maintainers**, the TRT-LLM infrastructure developers, own the pipelines and PR_Github, and
  carry out the migration.
- **PR authors** need CI's required check, `blossom-ci`, to reflect their run's test results, and
  open test-result links from console logs, PR comments, and CI reports to debug failures.
- **Owners of readers outside this repository**: PR_Github, which feeds the required check and posts
  PR comments; at least two internal tools that read `test-results/` by fixed address; and the
  dashboards that show perf records from OpenSearch with their links.
- **Owners of features built on test results**: test-level reuse, whose test history comes from each
  run's JUnit report, and coverage-based test selection, which reads the coverage archives built from
  the results archives.
- **The Artifactory migration program** requires every reference to be repointed and signs off on
  the migration.

Terms used below: "test results" and "in-scope" mean the outputs that FR-001 lists. "The new
location" is `llm-artifacts/` in `sw-tensorrt-llm-generic-local` on artifactory.nvidia.com, and "the
old location" is `llm-artifacts/` in `sw-tensorrt-generic` on urm.nvidia.com, as in TRTLLM-17014.
"The switch" is the change in this repository that moves test results, and a "run after cutover" is
a run whose pipeline code contains it. A "Slurm dispatcher pod" is the Kubernetes pod that submits a
test stage's Slurm job and monitors it. A "redacted copy" is what PR_Github publishes for GitHub
after a failed run (see Key Entities). "Dual publishing" is publishing everything under
`test-results/` to both locations, to urm as today and to its replica on the new location, until urm
is retired (FR-002). "TRTLLM-17014 FR-nnn" is a requirement of that ticket's specification.

Priorities follow the order the author set: CI's own writes and reads first, then PR_Github's test
summary, then people and outside tools.

### User Story 1 - Test results stay complete in CI (Priority: P1)

As a PR author, I want every test stage of my run to publish its results to the new location, and
every step of the run that reads them to find all of them there, so that my run's test report, the
reuse of tests that already passed, and the rerun report stay as complete as they are today.

**Why this priority**: This is the core of the ticket. Three readers in CI lose data without failing
when a read goes wrong: Collect Test Result, which builds the run's JUnit report and, through it, the
test history behind test-level reuse; the post-merge per-architecture coverage step; and a retried
stage that reuses the tests its earlier attempts passed. A silent loss looks like a clean run, with
fewer results, fewer reused tests, and no error.

**Independent Test**: Run CI on the switch pull request. For every stage, compare the test-result
archives it published with those that Collect Test Result found and read: they must match one for
one, every file under the run's `test-results/` must be on both locations with the same SHA-1, and
the run's `rerun_reports/` and `cbts-coverage/` files must be on the new location only.

**Acceptance Scenarios**:

1. **Given** a run after cutover, **When** a test stage attempt ends, **Then** its final archive,
   `results-<stage><postTag>.tar.gz`, is in the run's `test-results/` directory on the old location
   as today and on the new location, at the same relative path.
2. **Given** a test stage that is running, **When** its results change, **Then** its checkpoint
   archive on both locations is replaced, at most once per ten-minute interval as today, and once
   more when its tests end; **when** its final archive has been uploaded, **then** the checkpoint is
   deleted from both locations.
3. **Given** the run's test stages have ended, **When** Collect Test Result runs, **Then** it finds
   every test-result archive the run published, reads each one, and logs how many it found and how
   many it read, and the run's JUnit report contains the results of every stage that published them.
4. **Given** a stage attempt retried after an infrastructure failure, **When** the retry starts,
   **Then** it reads the earlier attempts' final archives from where they were published and reuses
   the tests they passed; an earlier attempt that published no archive is skipped, as today.
5. **Given** tests that failed and then passed on rerun, **When** the run ends, **Then** the rerun
   reports are on the new location, and the links the run prints open them there.
6. **Given** a post-merge run after cutover, **When** an architecture's single-GPU stages finish,
   **Then** the coverage step reads all of that architecture's results archives from the new
   location and publishes its coverage archive there.
7. **Given** a pre-merge run after cutover that uses coverage-based test selection, **When** it reads
   the coverage archives of recent post-merge runs, **Then** it reads those published after cutover
   from the new location and those published before from the old one, and selects tests as it would
   have before cutover.

---

### User Story 2 - Access proven without exposing the token (Priority: P1)

As a CI maintainer, I need every CI environment that writes, replaces, deletes, lists, or reads test
results to do so with the publishing identity, proven before cutover, and without the identity's
token appearing where it must not, so that the switch does not fail on day one and does not leak a
token that can also overwrite packaged builds and push CI's container images.

**Why this priority**: Stories 1 and 3 depend on it. The new location rejects anonymous reads, and
every reader of test results reads the old location anonymously today. TRTLLM-17014 proved only
reads from the remote GPU test clusters, while test results are also uploaded, replaced, and deleted
there, by Jenkins agents inside Slurm jobs. The checkpoint uploads and deletes pass the old
location's credential on curl's command line, and on Kubernetes test pods and agents inside Slurm
jobs that credential sits in the environment of every test process.

**Independent Test**: From each environment that writes or deletes test results, a verification run
uploads, replaces, lists, reads, and deletes a probe file in a `test-results/` directory on the new
location with the publishing identity, and confirms that an unauthenticated listing and read are
rejected. A CI run of the switch then shows no token in its logs, on its process command lines, or in
its published archives.

**Acceptance Scenarios**:

1. **Given** a Kubernetes test pod, a Slurm dispatcher pod, a Jenkins agent inside a
   Slurm job on each remote GPU test cluster where test stages run, and the pods of the parent
   pipeline's collection and coverage steps, **When** each performs the operations it needs on a
   probe file with the publishing identity, **Then** every operation succeeds.
2. **Given** a request without credentials, **When** it lists or reads a `test-results/` directory on
   the new location, **Then** the request is rejected.
3. **Given** a completed CI run of the switch, **When** its logs, its published archives (including
   the environment dump inside each stage's archive), and the files left on its hosts are inspected,
   **Then** no token or password appears, and no step passed a credential on a command line.
4. **Given** a test stage running on a Kubernetes test pod or on a Jenkins agent inside a Slurm job,
   **When** its test processes inspect their environment, the files they can read, and the command
   lines they can see, **Then** they find no Artifactory credential for either location (FR-010).

---

### User Story 3 - The required check and PR comments keep working (Priority: P2)

As a PR author, I need `blossom-ci` to report my run's test outcome, and the PR comment to link the
rerun report, whichever location my run published its test results to, so that the move never leaves
the required check pending or wrong.

**Why this priority**: `blossom-ci` is required to merge every pull request. For a failed run it
depends on PR_Github reading the run's `test-results/` and publishing a redacted copy that
`.github/workflows/l0-test.yml` downloads; for a passing run, PR_Github links `rerun_report.html` in a
PR comment. PR_Github's code lives outside this repository, so it does not move with the switch; dual
publishing keeps it working from the old location until its change lands, before urm is retired
(FR-002, FR-012). It ranks below Stories 1 and 2 because CI's own writes and reads come first.

**Independent Test**: Before PR_Github's change, a failed and a passing run of the switch get the
`blossom-ci` status and comment they got before the move, from the old location's copies. After it,
a failed and a passing run that published to the new location, and one of each that published only
to the old location (a pull request not rebased past the merge, or a run during a rollback), do too,
and the comment's rerun report link opens the report.

**Acceptance Scenarios**:

1. **Given** a failed run after cutover and before PR_Github's change, **When** PR_Github handles it,
   **Then** it reads the run's test results from the old location's copies, as today, and
   `blossom-ci` shows the failed tests.
2. **Given** a failed run after PR_Github's change, **When** PR_Github handles it, **Then** it reads
   the run's test results from the new location, publishes the redacted copy under its own path, from
   which `.github/workflows/l0-test.yml` downloads it as today, and `blossom-ci` shows the failed
   tests, while the originals on both locations stay as CI published them.
3. **Given** a passing run that reran tests, **When** PR_Github posts its comment, **Then** the rerun
   report link opens the report: on the old location before PR_Github's change, and in the native
   browser on the new location after it, with a developer's normal company sign-in.
4. **Given** a run that published its test results only to the old location, **When** PR_Github
   handles it, **Then** it finds them there, and `blossom-ci` and the comment are as they are today.

---

### User Story 4 - People and tools can still find test results (Priority: P3)

As a PR author or the owner of a tool that reads test results, I need the links and addresses I use
to lead to the test results wherever they are, and to know what access I need, so that I can still
debug failures after the move.

**Why this priority**: People open test-result links from console logs, PR comments, CI reports, and
perf records, and a Slurm stage's log points them to its archive for the full test output. urm
serves these links to them without any sign-in, while the new location rejects anonymous requests.
Two internal tools read `test-results/` by fixed address on the old location. CI keeps running when
these break, so this story ranks below Stories 1 to 3.

**Independent Test**: As a developer outside the infrastructure team, with only a normal company
sign-in, follow each kind of test-result link that a run after cutover prints, posts, or records;
check a perf record written after cutover; and confirm that the owners of every known outside reader
and the TRT-LLM developers were told before the merge.

**Acceptance Scenarios**:

1. **Given** a test-result link that a run after cutover printed, posted, or recorded, **When** a
   person who can open urm links today opens it in a browser, **Then** it opens the file on the new
   location after their normal company sign-in, without a personal token or membership in the
   program's distribution list.
2. **Given** a perf record written to OpenSearch by a run after cutover, **When** a user follows its
   link, **Then** it reaches the run's test results; a record written before cutover keeps its link,
   which works until the old repository is retired.
3. **Given** the switch is ready to merge, **When** the notices are checked, **Then** the owners of
   each known outside reader, and the TRT-LLM developers, were told what changes, when, and what
   access they need.

---

### User Story 5 - No in-scope reference to the old location remains (Priority: P3)

As the Artifactory program, and as the CI maintainer who requests sign-off, I need every reference to
the old location for in-scope test results repointed, so that test results no longer depend on the
old repository and the program can sign off.

**Why this priority**: Sign-off depends on it, and a missed reference becomes an outage when the old
repository stops serving. It finishes the migration rather than enabling it.

**Independent Test**: Search main after the switch merges for references to the old location in the
code that writes, reads, deletes, or links test results.

**Acceptance Scenarios**:

1. **Given** main after the switch merges, **When** it is searched, **Then** every remaining
   reference to the old location for in-scope test results is one that dual publishing, the
   fallback, or the rollback needs, or is on the allowlist with a reason.
2. **Given** all stories are complete, **When** the migration is submitted through the program's
   support channel, **Then** it is signed off with no open blockers.

### Edge Cases

- **Runs and pull requests in flight at the merge**: a run publishes where its pipeline code says
  (TRTLLM-17014 FR-002). A run whose code predates the switch, and a pull request not rebased past
  the merge, keep publishing test results to the old location, and every reader finds them there
  (FR-006, FR-012).
- **Run split by a rollback**: a run in progress when CI rolls back, or returns to the new location,
  can leave some stages' test results in each location. Collect Test Result and the coverage step
  find them in both, and a retried stage finds an earlier attempt's archive through the fallback
  (FR-005, FR-006, FR-008).
- **Earlier attempt that published nothing**: an attempt that ended before its final upload left no
  archive in either location, and the retry skips it, as it skips a missing archive today.
- **Redacted copies with the originals' names**: PR_Github's redacted per-stage archives carry the
  originals' file names. Under their own path (FR-012), a redacted copy never replaces an original on
  either instance, and no reader of originals lists or reads one, including when urm's
  `sw-tensorrt-llm-generic-local` repository moves to the new instance later.
- **Coverage archives from before cutover**: coverage-based test selection reads the coverage
  archives of recent post-merge runs, some published before cutover. It reaches those on the old
  location through the fallback, and its pins, which stay on the old location, name runs whose
  archives may be on either location (FR-005).
- **Credentials on Slurm batch jobs**: the netrc that TRTLLM-17014 places in a Slurm batch job's
  workspace for its packaged-build read is there while the job's tests run. The checkpoints of those
  jobs are uploaded from the job's Slurm dispatcher pod, so this work makes no credential reachable on
  the cluster (FR-010).
- **Tests inside the Jenkins agent's own container**: on some remote clusters, a stage's tests run in
  the same container as its Jenkins agent. The checkpoint uploads and deletes still run from a process
  those tests cannot read, so FR-010 holds there too.
- **Checkpoint left behind**: a stage that stops before its final upload leaves its last checkpoint
  as its only result, and the run's report uses it. A checkpoint whose delete failed stays beside its
  final archive, and Collect Test Result reads both, as today; for an attempt superseded by a retry,
  that checkpoint still carries result files that the final archive renamed, so they reach the run's
  report, as they can today. Either kind stays until retention removes it (FR-004, FR-006).
- **Deleting a checkpoint that was never uploaded**: the location reports it absent, and the stage
  carries on, as today.
- **No test results yet**: a listing of a run's `test-results/` directory that does not exist on a
  location counts as empty, not as a failure (FR-006).
- **Rejected credential**: the new location answers 401 or 403. The step reports the credential
  message (FR-011) and never reads the old location instead. A replica upload logs the message and,
  unless it is a checkpoint's, marks the stage unstable, while the stage's urm uploads go on as
  today; a read that is best effort
  today logs the message, marks itself unstable, and carries on (FR-007).
- **Content-delivery network rejects a download**: the read helper's rule applies (TRTLLM-17014
  FR-014): one retry without the redirect, then a transfer failure, never a fallback.
- **Remote GPU test cluster that can read but not write**: agents inside Slurm jobs upload final
  archives and checkpoints, and delete checkpoints, from the cluster's compute nodes. A cluster that
  cannot write to the new location would fail the result submission of every stage it runs, so
  writes and deletes are proven from each cluster before cutover (FR-019).
- **Checkpoint volume**: a pre-merge run makes about 400 checkpoint uploads to each location, each
  replacing the last. If the new instance throttles or refuses them, the failed checkpoint uploads
  are logged without marking the stage, as today (FR-004), and the final archives still upload; the
  program is told the volume beforehand (FR-021).
- **Token expired or revoked mid-run**: publishing and reading of build artifacts on the new location
  stop, and so do the test-result replicas and this repository's reads of test results, because CI
  uses one identity there (TRTLLM-17014 FR-011). Test results still reach urm as today. Steps that
  are best effort log the credential message and carry on; every other step fails, and the
  packaged-build reads already fail the run.
- **People who open links**: in a browser, the new location asks for the normal company sign-in, and
  the program's wider read grant then lets everyone who can open urm links today see the file. Someone
  outside that grant is refused, and a command-line download needs a personal token (FR-016).
- **Program refuses the wider grant**: people's access could not stay as it is today, so the refusal
  is raised as a blocker and the switch does not merge (FR-016, FR-023).
- **Links written before cutover**: links in PR comments, console logs, and perf records point to the
  old location, where those test results stay until the old repository is retired; nothing is copied
  (TRTLLM-17014 FR-014).
- **Outside reader not yet moved**: an internal tool that reads by fixed address keeps finding runs'
  test results on the old location through dual publishing (FR-002), until urm is retired; its owner
  must move it before then. Its owner is told before the merge, and the merge does not wait for it
  (FR-015).
- **Rollback while dual publishing is on**: test results go to the old location only (FR-002); the
  readers list both locations, so nothing is lost when dual publishing resumes.
- **New instance outage**: during dual publishing, the replica uploads fail, and those of final
  archives mark their stages unstable, while urm keeps every file under `test-results/`; this
  repository's readers cannot read the new location, which is not an absence, so they mark
  themselves unstable until CI rolls back within one hour (FR-018) and reads urm first. Outside
  readers on urm are unaffected. `rerun_reports/` and `cbts-coverage/`, which exist only on the new
  location, stay unreadable until it recovers. Coverage-based test selection waits out the read
  helper's retries for each candidate post-merge run whose coverage archive has to be read there
  before skipping it, as TRTLLM-17014's reads of build records do, so the start of pre-merge runs is
  delayed while the outage lasts (FR-007).
- **Old repository retired**: test results published before cutover become unavailable, and readers
  treat them like results deleted by retention.

## Requirements *(mandatory)*

### Functional Requirements

**Scope and publishing**

- **FR-001**: The migration MUST cover these outputs of each run under `llm-artifacts/<job>/<build>/`,
  and every CI step that writes, replaces, deletes, lists, reads, or links them:
  - `test-results/results-<stage><postTag>.tar.gz`, the final archive, one per stage, sub-job, and
    attempt, where `<postTag>` is empty or names the sub-job and attempt;
  - `test-results/results-<stage><postTag>-progress.tar.gz`, the checkpoint that each stage replaces
    while its tests run and deletes after its final archive is uploaded;
  - `test-results/rerun_report.html`, the build-level rerun report;
  - `test-results/doc-html-preview.tar.gz`, the docs preview, which is not a test result but sits in
    the same directory;
  - `rerun_reports/<stage>_rerun_results.html`, each stage's rerun report;
  - `cbts-coverage/cbts_pystart_report_<arch>.tar.gz` and `cbts-coverage/cbts_pystart_report.tar.gz`,
    the coverage archives that post-merge runs build from the final archives.

  In this specification, "test results" and "in-scope" mean these outputs. Wheels uploaded on their
  own (TRTLLMINF-295), the coverage-selection pins, which are not outputs of one run, and every other
  output under llm-artifacts stay where they are; TRTLLM-17014's artifacts have already moved.
- **FR-002**: A run's publish location MUST follow its pipeline code, as TRTLLM-17014 FR-002 decides
  for build artifacts. Every run after cutover MUST keep publishing everything it writes under
  `test-results/` (the final archives, the checkpoints, the build-level rerun report, and the docs
  preview) to the old location as it does today, and MUST replicate it to the same paths on the new
  location (dual publishing); `rerun_reports/` and `cbts-coverage/` go to the new location only. The
  readers in this repository read the new location first (FR-005, FR-006), and readers outside it
  move to the new location before urm is retired (FR-012, FR-015). Dual publishing lasts until the
  program retires the old repository, and nothing turns it off earlier; before the retirement date, a
  follow-up change removes the old-location writes together with the fallback reads. While CI is
  rolled back (FR-018), test results go to the old location only.
- **FR-003**: Each test result's path below `llm-artifacts/` (job, build, directory, and file name)
  MUST stay as it is today, so that lookups by job and build keep working.
- **FR-004**: The checkpoint's lifecycle MUST stay as it is today: while the stage's tests run, it is
  replaced at most once per ten-minute interval and only when the results changed, and it is
  uploaded once more when the tests end; after the final archive is uploaded, it is deleted, also
  after a failed test run; and its uploads and delete stay best effort: a failed upload or delete, on
  either location, is logged and marks nothing. A stage deletes its
  checkpoint on every location it published it to, both of them while dual publishing is on
  (FR-002). A checkpoint left behind, because its stage stopped
  before the final upload or because its delete failed, stays until retention removes it; this work
  adds no cleanup.

**Reading in CI**

- **FR-005**: Every CI step that reads a test result by name MUST read it through the shared read
  helper, `jenkins/scripts/fetch_build_artifact.sh`, which applies TRTLLM-17014's read rule: the
  location runs currently publish to first, the other location only when the file is absent there,
  never after a rejected credential, with the checksum check, the content-delivery network rule, and
  exit codes checked by name (TRTLLM-17014 FR-005 and FR-014). This covers the retried stage's read of
  earlier attempts' archives, the archives that Collect Test Result and the coverage step read, and
  coverage-based test selection's existence checks and reads of earlier post-merge runs' coverage
  archives, which reach those published before cutover on the old location through the fallback.
- **FR-006**: A step that needs every test-result archive of a run (Collect Test Result, and the
  coverage step for one architecture's archives) MUST obtain the complete list of archives the run
  published, from the location runs currently publish to and from the other location, and then read
  each one under FR-005. As today, the list includes every checkpoint still there, whether or not
  its attempt also left a final archive. On the new location the listing MUST be authenticated; no
  step may rely on an anonymous request or on an HTML index page there. A directory that does not
  exist on a location counts as empty. When the same file name is listed in both locations, the read
  rule decides which copy is read. The step MUST log how many archives it found in each location and
  how many it read.
- **FR-007**: The steps that read test results without failing today (Collect Test Result, the
  coverage step, and the retried stage's read of earlier attempts) MUST stay best effort: they do not
  fail the run, and they carry on with what they could read. They MUST NOT lose data silently: when a
  listing or read is rejected or fails, or fewer archives are read than were listed, the step MUST
  log the reason, with the credential message for a rejection (FR-011), and MUST mark itself
  unstable. Coverage-based test selection stays best effort and reads coverage archives as
  TRTLLM-17014 reads build records: when a coverage archive's read fails in transfer, including a
  rejection by the other location, it logs the reason and skips that post-merge run, as it skips one
  whose archive is absent, and tries older ones; when the location runs currently publish to rejects
  the credential, it logs the reason and declines at once, without reading the other location. A step
  that fails today when an upload fails, such as
  the upload of a final archive to urm, MUST still fail. During dual publishing (FR-002), a failed
  replica upload to the new location MUST NOT fail the stage: it is logged, with the credential
  message for a rejection (FR-011), and marks the stage unstable, except for a checkpoint, whose
  failed uploads are only logged (FR-004); this repository's readers find the urm copy through the
  fallback. When urm is retired, the new location's upload takes over
  today's rule.
- **FR-008**: A run MUST read its own test results from the location it published them to, including
  runs in progress at cutover and runs in progress when CI rolls back or returns to the new location.

**Access and secrets**

- **FR-009**: Every automated write, replacement, deletion, listing, and read of test results on the
  new location MUST authenticate with the publishing identity, `aws-artifactory-credentials`, which
  TRTLLM-17014 settled for publishing and reading (TRTLLM-17014 FR-010 and FR-011). This work adds no
  identity and needs no permission beyond the four that identity holds: read also covers listing, and
  delete/overwrite covers replacing and deleting checkpoints. Before cutover, the grant MUST be
  confirmed to cover overwrites and deletes under the `test-results/`, `rerun_reports/`, and
  `cbts-coverage/` directories of runs, not only the paths that TRTLLM-17014 writes. Reads of the old
  location stay anonymous, and writes and deletes there, during dual publishing and while CI is rolled
  back, keep `urm-artifactory-creds`.
- **FR-010**: Tokens and passwords MUST NOT appear in logs, in published archives (including
  `debug_env.txt`, the environment dump inside each stage's archive), in printed links, on a
  process's command line, or in files left on shared hosts after a job ends (TRTLLM-17014 FR-012).
  The steps that write, replace, or delete test results MUST NOT make any Artifactory credential, for
  either location, reachable by the processes that run tests: not in their environment, not in a
  file they can read, and not on a command line. Checkpoint uploads and deletes therefore run from a
  process that the tests cannot read, on Kubernetes test pods and on Jenkins agents inside Slurm jobs
  alike, including agents whose tests run inside the agent's own container. This rule is stricter
  than TRTLLM-17014's accepted risk for hosts that run a pull request's code, and it also ends
  today's exposure of the old location's credential to test processes. The netrc that TRTLLM-17014
  places in a Slurm batch job's workspace for its packaged-build read is not changed by this work.
- **FR-011**: When the new location rejects a credential, the message MUST name the credential (never
  its secret) and say that its token needs renewal or that it lacks the permission the request needs
  (read, deploy, or delete) on the repository, for a 401 and a 403 alike, and MUST name no internal
  systems (TRTLLM-17014 FR-008 and FR-013).

**Readers outside this repository, links, and people**

- **FR-012**: PR_Github MUST find a run's test results wherever the run published them. For a failed
  run, it reads them as FR-006 requires of CI's own steps; for a passing run, it fetches
  `rerun_report.html` under the read rule (FR-005). Both read the new location first, with the
  publishing identity, and the old location only when the files are absent there. Its PR comment
  MUST link the rerun report on the instance that holds it. PR_Github's change MUST also publish its
  redacted copies (each failed stage's redacted archive and the combined archive that
  `.github/workflows/l0-test.yml` downloads) under a path of their own, which no original uses on
  either instance, instead of `<build>/test-results/` and `<build>/test-results.tar.gz` in
  `sw-tensorrt-llm-generic-local/llm-artifacts/<job>/`. The release bundle that carries them to the
  edge site and the URL handed to `l0-test.yml` follow that path; `l0-test.yml`, which takes the URL
  as an input, and the repository secret it downloads with do not change, and redacted copies
  published before the change stay where they are. Until the change lands, PR_Github keeps reading
  the old location's copies (FR-002), so `blossom-ci` and the comment work as today, and the switch
  does not wait for it. The change MUST land before urm is retired, and when it lands, a failed and a
  passing run that published to the new location MUST show its new reads, the redacted copies under
  their own path, and the originals unchanged.
- **FR-013**: Every link that CI prints or posts to a test result (for example, the final archive
  named in a Slurm job's log, the rerun reports, the coverage archives, and the PR comment's link)
  MUST point to the instance that holds the file; for a file that dual publishing keeps on both, it
  points to the new location. A link to a directory that spans both instances MUST become one link
  per instance (TRTLLM-17014 FR-007).
- **FR-014**: A perf record that a run after cutover writes to OpenSearch MUST carry a link that
  reaches the run's test results on the instance that holds them. Records written before cutover keep
  their links.
- **FR-015**: Before the switch merges, the owners of each known outside reader of test results
  (PR_Github's maintainers, the owners of the internal tools that read `test-results/` by fixed
  address, and the owners of the dashboards that show perf records' links) MUST be told what changes,
  when, and what access they need, and whether each internal tool runs in production MUST be
  determined. Moving those tools is their owners' work and does not hold the merge: dual publishing
  keeps them working until urm is retired, and they MUST read the new location before then (FR-002).
  A tool that has not moved, or whose production status stays unknown, MUST be recorded with the
  sign-off request.
- **FR-016**: Everyone who can open test-result links on urm today MUST be able to open them on the
  new location in a browser with their normal company sign-in, without a personal token or
  membership in the program's distribution list. This work MUST ask the program to grant read on the
  in-scope outputs to that group, and the switch MUST NOT merge until the grant is in place and a
  developer outside the infrastructure team has opened a test-result link on the new location that
  way; a refusal is a blocker (FR-023). Command-line downloads still need an individual Artifactory
  access token: how to get one stays in internal documentation, and repository files say only that an
  Artifactory access token for artifactory.nvidia.com is needed (TRTLLM-17014 FR-008). Before the
  switch merges, the TRT-LLM developers MUST be told that test-result links move and what the move
  means for them. This work adds no other way to view test results.

**Rollout, verification, and sign-off**

- **FR-017**: The switch MUST merge into main only, after TRTLLM-17014 (PR #20494) has merged and its
  validation after cutover has passed. Release branches keep publishing and reading test results on
  the old location (TRTLLM-17014 FR-018).
- **FR-018**: Test results MUST follow the rollback setting that build artifacts follow,
  `LLM_ARTIFACTS_PRIMARY` (TRTLLM-17014 FR-022): changing that one CI-wide setting returns the
  publishing and reading of both to the old location within one hour of the decision, with no code
  change, and returns them to the new location the same way.
- **FR-019**: Before the switch pull request is marked ready to merge:
  - a verification run MUST perform every operation that FR-009 lists with the publishing identity,
    from each environment that writes or deletes test results: Kubernetes pods, Slurm dispatcher
    pods, and a Jenkins agent inside a Slurm job on each remote GPU test cluster where test stages
    run. It MUST confirm the grant that FR-009 requires and that an unauthenticated listing and read
    of a `test-results/` directory on the new location are rejected;
  - every changed path that a pull request's CI can reach MUST run at least once against the new
    location in the switch pull request's CI runs, under TRTLLM-17014's rules for validation runs
    (TRTLLM-17014 FR-020), and those runs MUST show every file under `test-results/` on both
    locations with the same SHA-1 (FR-002);
  - the switch pull request's own failed and passing runs MUST show PR_Github working unchanged from
    the old location's copies: the same `blossom-ci` status, test summary, and comment as before the
    move (FR-002, FR-012);
  - PR_Github MUST be shown not to publish its redacted copies to the new instance, where they would
    replace originals from the first failed run after cutover; finding them there is a blocker
    (FR-023);
  - coverage-based test selection's reads MUST run in the switch pull request's default pre-merge
    run. Only post-merge runs publish coverage archives, so before cutover they find every archive on
    the old location through the fallback; their read from the new location is validated after
    cutover (FR-020);
  - the paths that a pull request's CI reaches only when tests fail, the rerun reports and the
    retried stage's read of earlier attempts, MUST run at least once against the new location. When
    no CI run of the pull request reaches them on its own, temporary changes in the draft make chosen
    tests fail in a controlled way, and are removed before the pull request is marked ready, as the
    design documents are;
  - the steps that build coverage archives, which only the post-merge pipeline runs, MUST have their
    identity and their listing checked from their environment by the verification run. They are
    validated on the first post-merge run after cutover, with the rollback (FR-018) available.
- **FR-020**: After cutover, the first merge-request run and the first post-merge run MUST be
  compared with the corresponding runs of the commit just before the merge, as TRTLLM-17014 FR-021
  compares them, and MUST show the same test-result archives by name and number, the same number of
  tests in each JUnit report, and the same stage outcomes, apart from where test results are
  published. A stage that failed only because of a download this work does not change is left out
  and listed as not compared. Once the first post-merge run after cutover has published its coverage
  archives, the first pre-merge run whose coverage-based test selection picks that run MUST read them
  from the new location.
- **FR-021**: Before cutover, the program MUST be told the expected volume on the new location (a
  pre-merge run writes about 75 files and 52 MiB, almost all of it the docs preview, plus about 400
  checkpoint uploads; a post-merge run writes about 180 files and 316 MiB; any in-scope output that
  these figures leave out is measured during planning), and MUST confirm that the
  new repository keeps test results at least as long as the old one does today; otherwise the
  difference MUST be recorded with the sign-off request. A quota or rate limit that would refuse this
  volume is a blocker (FR-023).
- **FR-022**: Every reference to the old location for in-scope test results MUST be repointed, except
  those that dual publishing, the fallback, and the rollback need and the entries on a recorded
  allowlist; each kept
  reference and allowlisted entry MUST state why it remains (TRTLLM-17014 FR-019).
- **FR-023**: Blockers MUST be raised, and the completed migration submitted for sign-off, through the
  program's support channel. The sign-off evidence MUST be recorded as TRTLLM-17014 recorded it:
  dates, outcomes, and links to run logs, never a token, password, or netrc content.
- **FR-024**: Before the switch merges, its pull request MUST be sent for review to all TRT-LLM
  infrastructure developers, and its description MUST state the new location; that people open
  test-result links there with their normal company sign-in through the program's wider read grant,
  while command-line downloads need an Artifactory access token (FR-016); how the publishing token is
  kept from the processes that run tests (FR-010); that everything under `test-results/` is still
  published to urm, as today, and replicated to the new location until urm is retired (FR-002); that
  PR_Github keeps working from urm until its own change, which lands before then (FR-012); that
  coverage-based test selection now reads coverage archives from the new location first (FR-005);
  and which outside readers' owners were told (FR-015). TRTLLM-17014's confidentiality rules apply:
  repository files and the public pull request name no internal lists, portals, channels, tools, or
  people, and while the pull request is a draft, its single commit carries the design documents with
  internal names replaced by roles, until they are removed before it is marked ready (TRTLLM-17014
  FR-008 and FR-024).
- **FR-025**: A problem that review or the author's own checks find in the switch pull request MUST
  be handled as TRTLLM-17014 FR-026 decides: fixed in the pull request when it is blocking, and
  otherwise moved to a follow-up pull request tracked in a ticket that the pull request links to.

### Key Entities

- **Final archive**: the results of one stage, sub-job, and attempt,
  `results-<stage><postTag>.tar.gz`, written once when the attempt ends. It carries the stage's JUnit
  files and logs, the data of its rerun report, coverage data files for coverage stages, and
  `debug_env.txt`, a dump of the step's environment.
- **Checkpoint archive**: `results-<stage><postTag>-progress.tar.gz`, replaced while the stage's tests
  run, uploaded once more when they end, and deleted after the final archive is uploaded. A stage that
  stops before its final upload leaves it as its only result; a failed delete leaves it beside the
  final archive.
- **Rerun reports**: the build-level `rerun_report.html` in `test-results/`, which Collect Test Result
  builds from every stage's final archive, and each stage's `rerun_reports/<stage>_rerun_results.html`.
- **Docs preview**: `doc-html-preview.tar.gz`, the built documentation, stored in `test-results/` but
  not a test result.
- **Coverage archives**: `cbts-coverage/cbts_pystart_report_<arch>.tar.gz` and
  `cbts-coverage/cbts_pystart_report.tar.gz`, built in post-merge runs from the final archives and
  read by coverage-based test selection in later pre-merge runs, which records the post-merge run it
  chose in a coverage-selection pin.
- **Test-results directory**: `llm-artifacts/<job>/<build>/test-results/` of one run, on one location
  except for a run split by a rollback. While dual publishing is on, the old location holds the same
  files.
- **Run's JUnit report and test history**: the report that Collect Test Result builds from every final
  archive, which feeds the test history that test-level reuse queries.
- **Redacted copy**: what PR_Github publishes for `.github/workflows/l0-test.yml` after a failed
  run: each failed stage's archive with what its scanners flag removed, plus one combined archive,
  marked as scanned and carried by a release bundle to an edge site that GitHub's runners read. Its
  per-stage files have the same names as the originals, so it lives under a path of its own
  (FR-012). What it removes, and how it reaches the edge site, are not in scope.
- **Test-result reader**: a CI step, outside tool, or person that lists or reads test results,
  characterized by where it runs, how it authenticates, and whether it reads by name or lists a
  directory.
- **Publishing identity**: `aws-artifactory-credentials`, CI's one identity on the new location, with
  read, deploy, annotate, and delete/overwrite (TRTLLM-17014 FR-010).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of runs whose pipeline code contains the switch, and that do not overlap a
  rollback, publish all of their test results to the new location, and 100% of their files under
  `test-results/` also appear at the same paths on the old location (dual publishing), while none of
  their files under `rerun_reports/` or `cbts-coverage/` do.
- **SC-002**: In the first two weeks after cutover, for 100% of runs, Collect Test Result reads every
  results archive that the run left in its `test-results/` directory (every final archive, and every
  checkpoint still there), and the run's JUnit report includes every stage that published results.
- **SC-003**: In the first two weeks after cutover, 0 CI runs fail, and 0 pull requests get a missing
  or wrong `blossom-ci` status, because test results could not be published, replaced, deleted,
  listed, or read, excluding outages of the new instance itself.
- **SC-004**: For the first 20 failed and the first 20 passing pull-request runs after cutover, 100%
  get the `blossom-ci` status and the PR comment they would have got before cutover, and every rerun
  report link in those comments opens with a developer's normal company sign-in.
- **SC-005**: For runs that publish test results to the old location after cutover (pull requests not
  rebased past the merge, release branches, and runs during a rollback), 100% of PR_Github's reads
  find them, and their `blossom-ci` status is set as before.
- **SC-006**: In the runs compared under FR-020, 0 checkpoint archives remain on the new location, or
  on the old location while dual publishing is on, for stages that reached their end.
- **SC-007**: The median duration of a merge-request pipeline run in the first week after cutover is
  within 10% of the median for the week before cutover, and the median durations of Collect Test
  Result, of each per-architecture coverage step, and of a stage's result submission are each within
  10%, or one minute if that is larger, of their medians for the week before cutover.
- **SC-008**: 100% of unauthenticated listings and reads of a `test-results/` directory on the new
  location are rejected during verification, and a scan of all pipeline logs from the first week after
  cutover, and of every test-result archive of the runs compared under FR-020, finds 0 exposed tokens
  or passwords.
- **SC-009**: A search of main finds 0 functional references to the old location for in-scope test
  results other than those that dual publishing, the fallback, and the rollback need; every other
  remaining match is on the allowlist with a reason.
- **SC-010**: The Artifactory program signs off on the migration of test results, with 0 open
  blockers, before the old repository's retirement date.
- **SC-011**: If a rollback is needed, test results return to the old location together with build
  artifacts within 1 hour of the decision.
- **SC-012**: In the first two weeks after cutover, 0 pre-merge runs lose coverage-based test
  selection because a coverage archive could not be read from either location, excluding outages of
  the new instance itself.
- **SC-013**: Withdrawn on 2026-10-28: where PR_Github publishes its redacted copies is PR_Github's
  concern (FR-012); FR-019 keeps the check, before the merge, that none is published to the new
  instance.
- **SC-014**: Before the merge, and again in the first week after cutover, 100% of the test-result
  links checked (at least one of each kind that FR-013 lists) open in a browser for a developer
  outside the infrastructure team with only their normal company sign-in.
- **SC-015**: Before urm's retirement date, PR_Github and every outside reader in production read
  test results from the new location, so that 0 of them depend on dual publishing when it ends.

## Assumptions

### Current state (observed on 2026-10-27)

Observed at the commit this work starts from, which carries TRTLLM-17014's change (PR #20494); that
change had not merged into main yet.

- Every writer of test results except the checkpoint uploads goes through the shared CI library's
  upload helper: the final archives, the build-level and per-stage rerun reports, the docs preview,
  and the coverage archives. None of these calls passes the helper's destination argument, so all of
  them go to the old location with `urm-artifactory-creds`. A failed upload of a final archive fails
  its stage; the checkpoint uploads and deletes, the build-level rerun report's upload, and the
  coverage step are best effort.
- A background watcher, started in the same shell step as the tests, uploads the checkpoints on
  Kubernetes test pods and on Jenkins agents inside Slurm jobs. For Slurm batch jobs, it runs beside
  the job monitor on the job's Slurm dispatcher pod, after copying the results back from the
  cluster. It runs curl with `urm-artifactory-creds` on the command line, and on Kubernetes test pods
  and agents inside Slurm jobs that credential is in the environment of every test process for the
  length of the test run.
- Checkpoint deletes run curl with the same credential on its command line, against the old
  repository's backing local repository, because its virtual repository refuses deletes.
- Collect Test Result runs in a pod of the parent pipeline and builds the run's only aggregated JUnit
  report. It lists `test-results/` by reading urm's HTML index page, downloads every `.tar.gz` file it
  lists (including the docs preview and any checkpoint left behind; a pre-merge run's 75 files take
  about 8 seconds), and extracts every `results-*.tar.gz`. The listing and the downloads may fail
  without failing the step, and the JUnit step accepts an empty result, so a failed read yields an
  incomplete report and no error.
- The coverage step runs only in post-merge runs, after each architecture's single-GPU stages. It
  lists and downloads the same way, keeps that architecture's archives, and is non-fatal.
- A retried stage attempt probes each earlier attempt's final archive anonymously, skips it on a 404,
  and downloads it otherwise. Any other answer raises an error, which the enclosing step catches: it
  logs one line and carries on without reusing any test, including those from the test history.
- Each stage's final archive also carries a dump of the step's environment, with credential-like
  variable names redacted by pattern.
- Coverage-based test selection, in pre-merge runs, checks for and downloads the coverage archives of
  recent post-merge runs from urm anonymously, by a fixed address, and reads and writes its pins on
  urm. A developer dry-run tool tells developers to download a post-merge coverage archive by hand,
  without naming a host.
- Links: a Slurm job's log names the stage's final archive on urm; the per-stage and build-level
  rerun reports and the coverage archives are announced with urm links; PR_Github links the
  build-level rerun report in a PR comment; and perf records in OpenSearch link the run's build
  directory on urm through a fixed address that ignores the rollback setting.
- PR_Github, as its build logs showed on 2026-10-27: after a failed run, its scan step downloads the
  run's results archives from urm with a urm credential, checking each file's SHA-1. It removes what
  its scanners flag, stripping the test logs from flagged JUnit files and leaving out files it cannot
  redact, such as an environment dump that holds internal identifiers. It then publishes each failed
  stage's redacted archive at `sw-tensorrt-llm-generic-local/llm-artifacts/<job>/<build>/test-results/`
  under the original's file name, plus a combined `<build>/test-results.tar.gz`, marks them as
  scanned, and has a release bundle carry them to an edge site, from which
  `.github/workflows/l0-test.yml` downloads the combined archive with a repository secret to set
  `blossom-ci`. When the scan step finds nothing to download, the PR_Github build fails. After a
  passing run, PR_Github fetches `rerun_report.html` from urm anonymously, retrying three times about
  a minute apart when it is absent, and links it in a PR comment.
- A repository named `sw-tensorrt-llm-generic-local` exists on urm as well as on the new instance.
  The scan step binds only a urm credential, and the edge site belongs to urm, so the redacted copies
  are almost certainly on urm today, at the repository, paths, and file names that the originals take
  on the new instance (FR-012). The check before the merge confirms that they are not on the new
  instance (FR-019).
- At least two internal tools read `test-results/` from urm anonymously, by fixed address, through its
  HTML index. One is a CI monitor in active use (checked on 2026-10-28): on a schedule it walks the
  finished runs of `L0_PostMerge`, `L0_MergeRequest`, and `L0_MergeRequest_PR` in the main and
  release-1.3 job folders, mirrors each run's whole `test-results/` folder, extracts every archive,
  maps tests to stages from each final archive's `results.csv` (it finds the archive names, including
  `-attempt-N` ones, in the stages' logs), posts the test records to an internal data feed, and alerts
  on post-merge failures. When it cannot read the folder, the records lose their stages and it only
  logs a warning. The other tool recovers a build's commit from its archives; whether it runs in
  production is unconfirmed.
- The read helper reads one file by its path and cannot list a directory. The new location rejects
  anonymous requests, including listings.
- Volume, from the description: a pre-merge run writes about 75 files and 52 MiB, almost all of it the
  docs preview, plus about 400 checkpoint uploads; a post-merge run writes about 180 files and
  316 MiB.

### Settled by TRTLLM-17014 and reused here

These decisions are taken as they stand and are not reopened:

- The new and old locations, the legacy-path mapping from `sw-tensorrt-generic/` paths to layout
  paths below `llm-artifacts/`, and a publish location decided by a run's pipeline code rather than
  by the merge time (TRTLLM-17014 FR-002 and FR-003).
- The publishing identity, `aws-artifactory-credentials`, with read, deploy, annotate, and
  delete/overwrite on the new location, which CI also reads with in every environment, with no
  separate reading identity (TRTLLM-17014 FR-010 and FR-011).
- The read helper and its rule: the new location first, the old one only on a 404, no fallback after
  a rejection, the SHA-1 check, the content-delivery network rule, and exit codes by name
  (TRTLLM-17014 FR-005 and FR-014).
- The rollback switch, `LLM_ARTIFACTS_PRIMARY`, which parent pipelines pass to the jobs they start
  (TRTLLM-17014 FR-022).
- The shared library's `uploadArtifacts` destination argument, already merged, whose default stays
  the old location (TRTLLM-17014 FR-025).
- No copy of earlier outputs: they stay on the old location, readable through the fallback, until it
  is retired (TRTLLM-17014 FR-014).
- Main only, with release branches staying on the old location (TRTLLM-17014 FR-018).
- The rules for the switch pull request's CI validation runs, review findings, confidentiality, and
  sign-off evidence (respectively TRTLLM-17014 FR-020; FR-026; FR-008 and FR-024; and FR-023).
- Token handling and credential messages (TRTLLM-17014 FR-012 and FR-013).

### Defaults chosen where the description was silent or left the question open

- **Rollback switch**: test results share `LLM_ARTIFACTS_PRIMARY` with build artifacts (FR-018). One
  setting returns everything to the old location, which is what an outage of the new instance needs.
  Reads fall back in both directions, so moving build artifacts back together with test results
  strands nothing, and a second switch would double the combinations to verify and the places it has
  to be passed through.
- **Listing a directory on the new location**: FR-006 sets what a listing must do; how a directory is
  listed is left to the plan.
- **Best effort and failures**: each step keeps failing or carrying on as it does today, and the
  silent readers become visible by logging why and marking themselves unstable, without failing the
  run (FR-007). A rejected credential already fails the run through the packaged-build reads, so
  failing the readers too would add nothing.
- **Notice**: review by the infrastructure developers is no longer enough on its own, because readers
  outside this repository read test results by fixed address and many people open the links. Their
  owners and the TRT-LLM developers are also told directly (FR-015, FR-016, FR-024).
- **PR_Github's identity**: PR_Github reads the new location with the publishing identity, as every CI
  reader does (FR-012); `.github/workflows/l0-test.yml` stays as it is.
- **Rarely reached paths**: the rerun reports and the retried stage's read are exercised before the
  pull request is marked ready, through temporary changes in the draft if needed; the coverage step is
  validated on the first post-merge run after cutover (FR-019).
- **Retention**: the new repository keeps test results at least as long as the old one does, to be
  confirmed with the program (FR-021).
- **Step performance**: each changed step stays within 10%, or one minute, of its median before
  cutover (SC-007).

### Out of scope

- Wheels uploaded on their own (TRTLLMINF-295), and every llm-artifacts output that FR-001 does not
  list, such as the guardwords scan log, image-tag maps, the optimization-profile store, and the
  coverage-selection pins.
- Copying test results of runs before cutover to the new location.
- Changing the content, names, or format of test results, the checkpoint interval, or the retention
  policy.
- What PR_Github's redaction removes, how its redacted copies reach the edge site, and
  `.github/workflows/l0-test.yml`; only the redacted copies' path changes (FR-012).
- Changing the internal tools that read test results by fixed address; their owners move them.
- Release branches cut before the switch merges.
- Container-image pulls, and downloads through remote mirrors, as in TRTLLM-17014.
- Fixing problems that exist today in the flows this work changes, beyond what this specification
  requires.

### Dependencies

- TRTLLM-17014, merged into main and validated after its cutover (FR-017).
- The Artifactory program: the wider read grant for people (FR-016), the volume and retention
  confirmation (FR-021), answers to blockers, and
  sign-off (FR-023).
- PR_Github's maintainers, for its change, including its scan step's new path for redacted copies,
  which lands before urm is retired (FR-012).
- The owners of the CI credential store, to make the publishing identity's credential available to
  every job that writes, deletes, lists, or reads test results, including PR_Github and the parent
  pipeline's collection and coverage steps.
- The owners of the internal tools and of the dashboards that show perf records' links (FR-015).
- The owners of coverage-based test selection, to review the change to its reads of coverage
  archives (FR-005), and the owners of test-level reuse and of coverage-based test selection, to
  check their features after cutover.
- The old repository cannot be retired until every llm-artifacts output has moved, including those
  that this ticket leaves out.
