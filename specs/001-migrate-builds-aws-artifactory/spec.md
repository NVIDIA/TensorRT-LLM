# Feature Specification: Migrate Build Artifacts to AWS Artifactory

**Feature Branch**: `TRTLLM-17014-migrate-builds-to-aws-artifactory`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "TRTLLM-17014 — Migrate Builds to AWS Artifactory. Build artifacts
currently go to urm.nvidia.com/.../sw-tensorrt-generic/llm-artifacts and must instead go to
artifactory.nvidia.com/.../sw-tensorrt-llm-generic-local/llm-artifacts; for each build,
build_info.txt, all *.tar.gz outputs, and the waive_list/ directory are to be migrated. The
enterprise Artifactory-migration program requires every CI/CD reference to be repointed from
urm.nvidia.com to artifactory.nvidia.com. Anonymous access is disabled on the AWS instance, so
every consumer needs a token from the program's self-service portal and membership in the
program's distribution list. Build-info metadata is not carried over by a repository migration and
must be migrated separately via API by build name. Blockers and sign-off go through
the program's support channel. TRTLLM-15275 is the closest precedent, and
its token and service-account work was non-trivial, so check whether the upload pipeline already
has an authenticated service account."

## Clarifications

### Session 2026-10-14

- Q: Which scope applies to this ticket: (a) only build_info.txt, *.tar.gz outputs, and
  waive_list/, plus everything that reads them; (b) everything stored under llm-artifacts; or
  (c) option (b) plus every other urm.nvidia.com reference in the repository? → A: Option (a),
  for this ticket. Other llm-artifacts outputs and other urm.nvidia.com references are left to
  follow-up work.
- Q: Should "all *.tar.gz outputs" include the test-result and coverage archives that test
  stages publish, or only the packaged builds that the build stages produce? → A: Only the
  packaged builds (build artifacts). Test results move under TRTLLMINF-472 and wheels under
  TRTLLMINF-295, both chained after this migration is merged and validated.
- Q: At cutover, should new builds stop publishing these artifacts to the old location
  immediately, or keep publishing to both locations for a short transition period? → A: Stop
  immediately (option A). The switch takes effect when the change merges, and affected parties
  are notified beforehand.
- Q: Do builds published before the switch also need their build artifacts available from the
  new location, or only builds published after it? → A: Option C, no copy: readers look in the
  new location first and fall back to the old location when the artifact is not there.
  Confirmed by the owner on 2026-10-16.
- Q: Should the shared CI library's upload helper get an opt-in setting for the new location,
  with its default left on the old location, so that only this repository's build_info.txt and
  waive_list/ uploads use it? → A: Yes (option A). The helper gains an opt-in and keeps its
  default. In this ticket, only the build_info.txt and waive_list/ uploads use it. The library
  change is a separate merge request in the library's repository and merges before the switch.
- Q: Should the packaged builds in scope include every TRT-LLM package archive in a build's
  directory, whichever job publishes it, rather than only the ones the build stages produce?
  → A: Yes (option A): the build stages' packages and their bolted-/unbolted- copies, the
  optimized package that optimization-profile generation publishes under the original name, and
  the source release package. To be confirmed with the owner.
- Q: While CI is rolled back to the old location, should readers look there first and then fall
  back to the new location, so that artifacts published before, during or after the rollback all
  stay readable? → A: Yes (option A). CI administrators roll back by changing one CI-wide
  setting, with no code change. While it is set, runs publish to the old location and read it
  first, falling back to the new location when a file is not there.
- Q: Should the publishing identity have exactly four permissions on the new repository (read,
  deploy, annotate, and delete/overwrite) and nothing else? → A: Yes (option A). Artifactory
  grants overwrite only together with delete, and the pipelines replace existing files. The
  identity holds no administrative permissions, and the access check tests each permission.
- Q: Should the access steps (joining the distribution list, getting a token from the
  self-service portal, renewing it) be written only in internal documentation, with this
  repository's tool messages and docs saying just that an Artifactory access token for
  artifactory.nvidia.com is needed? → A: Yes (option A). Internal documentation holds the steps.
  CI messages name the CI credential involved, and repository docs say that an Artifactory access
  token for artifactory.nvidia.com is needed; neither names internal lists, portals, channels, or
  people.
- Q: Should a run's publish location be decided by whether the pipeline code it runs contains this
  change, rather than by the merge time? → A: Yes (option A). Runs whose pipeline code contains
  the change, including its own pre-merge runs and pull requests based on it, use the new location;
  runs without it, including pull requests not rebased past the merge, stay on the old location.
  Child jobs follow their parent's commit.
- Q: When a run reuses an earlier build, how should that build's artifacts reach the current run?
  → A: Exactly as today; only the read and publish locations change. The reuse process itself,
  including whether artifacts are placed into the current run's directory, stays the same.
- Q: How are the people affected by the switch told about it before it merges? → A: Through
  review: the change is sent to all TRT-LLM infrastructure developers before it merges. No consumer
  has to switch anything by hand; consumers in this repository move with the code (assumed: no
  consumer outside it reads in-scope artifacts by a fixed address).
- Q: What shows that the publishing identity meets FR-010? → A: CI keeps using identities it
  already holds and uses today; their permissions on the new repository mirror what CI uses on the
  old one. Evidence is that existing, onboarded credential plus a passing access check (FR-020).
- Q: What exactly is "the destination": the whole new repository or its llm-artifacts/ path? →
  A: The `llm-artifacts/` path in `sw-tensorrt-llm-generic-local` on artifactory.nvidia.com, which
  already exists.
- Q: From which run must readers look in both locations: only after the change merges, or earlier?
  → A: From the first run whose pipeline code contains the change, including the change's own
  merge-request runs before it merges.
- Q: Before the switch merges, which changed upload and download paths must have run at least
  once against the new location? → A: Every path that runs with default job options, plus the
  image-build steps that only post-merge runs perform, exercised in one post-merge-mode run of the
  switch pull request (this needs CI's post-merge approval label). Publishing the optimized
  package, which every pipeline launches with publishing off, gets only its publishing identity
  checked before cutover, from the environment that publishes it, and is validated on its first
  real run after cutover. Superseded on 2026-10-16: a pull request's post-merge mode does not
  reach the image-build steps.
- Q: Should the image-build changes also get an automated check that fails if the Artifactory
  credential ever ends up in the image as a build argument? → A: No extra check. Code review covers
  it, and the image-build changes are exercised only by the post-merge-mode run (FR-020). Updated
  on 2026-10-16: they first run on the first post-merge run after cutover.

### Session 2026-10-15

- Q: When the switch pull request merges, must it load the shared library from the library's main
  line again, so the library merge request has to merge first? → A: Yes. While under test, the
  switch pull request may load the library from the library merge request's branch, so testing
  starts before the library change merges. Before the switch merges, the library change merges
  and the switch loads the library from its main line again.
- Q: Should the new-then-old reads keep this repository's own read helper and copy bloom's
  download safeguards into it, instead of calling bloom's download code? → A: Yes. Bloom's code
  falls back after any error, sends one set of credentials to both servers, and runs only on
  Jenkins agents with Python. This repository keeps its own read helper and borrows two
  safeguards: retry a download that the content-delivery network rejects without the redirect,
  and verify each download against its stored checksum. Bloom is not changed.

### Session 2026-10-16

- Q: Before cutover, should only the changed paths that a pull request's CI runs can reach have to
  run against the new location, with the paths that only the post-merge pipeline runs checked on
  its first run after cutover? → A: Yes (option A). Pull-request runs cover every changed path they
  reach, and post-merge test stages can be added to them by name as sanity checks. Four paths run
  only in the post-merge pipeline: the image build's wait for the packaged builds and its download
  of one; optimization-profile generation's read of its input package; publishing a package's
  bolted- copy with its metadata, and reading that metadata; and publishing the optimized package.
  They are checked on the first post-merge run after cutover, and the identities they use are
  checked beforehand from the environments they run in.
- Q: Should the "identical results before and after cutover" measure apply only to coverage-based
  test selection? → A: Drop it (option C). SC-004 already measures reads of pre-cutover builds
  through the fallback, and FR-016 covers build reuse, which rebuilds a pre-cutover build by design.
- Q: When a step that is best effort today gets a rejected credential, should it log which
  credential needs renewal and carry on as it does today, rather than fail? → A: Yes (option A).
  The build_info.txt and waive-list uploads, the merged waive-list download, and coverage-based
  test selection's read of build records log the credential message and carry on. None of them
  falls back to the other location.
- Q: In what order should the switch pull request's CI runs go? → A: The default pre-merge run,
  started with no options (`/bot run`), comes first. Runs limited to chosen stages
  (`--stage-list`), including post-merge test stages added by name, come only after it.
- Q: Does this ticket keep the packaged builds (`TensorRT-LLM.tar.gz` and its variants, which
  contain the wheel), with TRTLLMINF-295 covering only the wheel files published on their own?
  → A: Yes (option A). The packaged builds stay in this ticket, and the wheel packed inside each
  one moves with it. TRTLLMINF-295 covers only the wheel files uploaded on their own, and the
  links to them.
- Q: How is this work split into change sets? → A: Two. One pull request in this repository
  carries every change here, including the new helper scripts and their tests, and the shared CI
  library's change is one merge request, already open (!111).
- Q: The switch changes `jenkins/L0_Test.groovy`, which makes CI add the multi-GPU stages to the
  default run. Who obtains the multi-GPU approval label, and when? → A: The switch pull request's
  author asks the CI approvers for `ci: full pre-merge approved`. Without it the multi-GPU stages
  are blocked and the run cannot pass. The first default run no longer waits for it (see the last
  question of this session).
- Q: Should the packaged builds also be uploaded through the shared CI library's upload helper,
  instead of this repository's own upload code? → A: No (option A), to keep disruption to the flows
  that depend on the old location to a minimum. build_info.txt and waive_list/ use the helper's
  new-location opt-in; packaged builds keep this repository's own upload code, which attaches their
  metadata, pointed at the new location; the library change stays as it is.
- Q: Can the switch pull request be opened as a draft and run its CI pipelines right away, before
  the permission grants and the per-environment access checks are done, with those checks required
  only before it is marked ready to merge? → A: Yes (option A). The draft is opened first and
  iterates on its CI runs. The permission grants, the access check in every environment (including
  those its CI runs never reach), and the confirmation that child jobs run the pull request's own
  pipeline code must be complete before it is marked ready to merge. Its full description is
  written when it is marked ready.
- Q: How should this feature's design documents be shared with the infrastructure developers on the
  draft pull request, given that the fork and the pull request are public? → A: Option A, inside
  the pull request's one commit: the pull request always has exactly one commit, and while it is a
  draft that commit also carries the design documents, with internal names (people, the
  distribution list, the self-service portal, the support channel, the internal GitLab host)
  replaced by role descriptions. They are removed from the commit before it is marked ready.
- Q: Should this work also make the remote GPU test clusters' download of a packaged build resume a
  transfer that drops partway, instead of starting it again? → A: No. This work only moves the
  artifacts, and problems that exist today stay as they are handled today. The download retries
  the whole transfer within its existing retry limits. Unlike today's download, it does not
  continue a dropped transfer within an attempt; that change is accepted.
- Q: Does the owner accept that builds published before cutover are not copied to the new
  location, and that readers reach them through the fallback? → A: Yes (the owner, 2026-10-16).
  Coverage-based test selection and build reuse could even do without those builds during the
  transition; optimization-profile generation needs its input downloads, which the fallback
  covers. When the optimization profiles move in later work, a one-time copy of the current
  profiles to the new location is acceptable in place of a fallback.
- Q: Can the switch pull request's first default run start before the multi-GPU approval label is
  applied, to find out early whether the change works? → A: Yes (the author). Without the label
  the run still publishes build_info.txt, the waive list and the packaged builds, and runs its
  single-GPU stages, each of which reads a packaged build. CI checks for the label when each
  architecture's single-GPU tests finish: a label applied by then lets the same run go on to the
  multi-GPU stages; otherwise they are blocked, and a later `/bot run` on the same commit reuses
  the build and the stages that passed.
- Q: Should every read of an in-scope artifact go through the one fetch script, including
  coverage-based test selection, the remote GPU test clusters' download, optimization-profile
  generation's input download, and the build-reuse check? → A: Option B (the author, after the
  owner asked for one fetch script used by all reads). Coverage-based test selection and the remote
  GPU test clusters run the fetch script itself. Optimization-profile generation keeps its parallel
  ranged download, which predates this work, has no other user, and also fetches a tool from
  GitHub; it takes the file's location and expected checksum from the fetch script. The build-reuse
  check only decides between a server-side copy and a rebuild, is not a read, and stays as it is.
  The parallel download moves into the fetch script only if a second reader needs it. The fetch
  script reads any path under llm-artifacts/, so the test-result and wheel migrations can reuse it.
- Q: How should the read helper's exit codes be shared between the helper and the pipelines and
  tools that check them? → A: Option A (the author): by name, checked by a test. The helper and
  every reader that acts on a specific exit code refer to it by the same name (for example
  `FETCH_ABSENT` for "absent from both locations"), and a unit test that the default pre-merge run
  executes fails when a reader's value differs from the helper's. A single definition that every
  file reads was not chosen: the pipelines and tools cannot read a shell variable, so they would
  have to parse the helper while a job runs, or a separate file of codes would have to travel with
  the helper to every host it runs on.

## User Scenarios & Testing *(mandatory)*

People affected:

- **CI maintainers** own the pipelines and carry out the migration.
- **PR authors** depend on CI passing, open artifact links from their pipeline runs, and download
  build outputs to debug locally.
- **Owners of history-reading tools** (coverage-based test selection, build optimization
  profiles, build reuse) depend on in-scope artifacts from earlier builds.
- **The Artifactory migration program** requires repository owners to repoint every reference and
  signs off on the migration.

### User Story 1 - Verified, authenticated access to the new repository (Priority: P1)

As a CI maintainer, I need every CI environment that writes or reads build artifacts to hold a
managed, authenticated identity for the AWS instance, proven to work before cutover, so that
switching the destination does not fail on day one with access-denied errors.

**Why this priority**: Every other story depends on it, and it has the longest lead time.
Anonymous access is disabled on the AWS instance, and onboarding an identity (distribution-list
membership, a token from the self-service portal, storage in the CI credential store, permission
grants on the destination repository) was the largest hidden cost in the closest precedent. Today,
CI holds credentials for the AWS instance but uses them only for container images; build
artifacts are published with credentials for the old instance, and test stages and helper tools
read artifacts with no credentials at all.

**Independent Test**: From each class of CI execution environment that touches build artifacts
(build agents, containerized test agents, and each remote GPU test cluster), a verification run
publishes a probe file to the destination with the publishing identity, reads it back with the
reading identity, and confirms that an unauthenticated read is rejected.

**Acceptance Scenarios**:

1. **Given** the publishing identity has been onboarded, **When** a verification run performs,
   on a probe file under the destination `llm-artifacts/` path, every operation that FR-010
   lists (create, overwrite, server-side copy, and attaching metadata), **Then** every operation
   succeeds.
2. **Given** a reading environment, including a remote GPU test cluster, **When** it reads the
   probe file with the reading identity, **Then** the read succeeds; **when** it reads without
   credentials, **then** the read is rejected.
3. **Given** a completed verification run, **When** its logs and any files left on the execution
   hosts are inspected, **Then** no token or password appears in them.

---

### User Story 2 - Builds and tests run end to end on the new repository (Priority: P1)

As a PR author, I want my pipeline to publish build outputs to, and fetch them from, the new
repository with no change to how I trigger CI or read its results, so that the migration is
invisible to me.

**Why this priority**: This is the core of the ticket. Every build's build_info.txt, packaged
builds, and waive_list/ directory land in the new location, and the test stages of the same run
consume them from there. CI gates every change to the project, so a broken cutover blocks all
contributors.

**Independent Test**: After the change, run a merge-request pipeline and a post-merge pipeline
that contain it, and compare them with the runs of the commit just before the merge. Confirm that
every stage passes as it did before, that
the in-scope artifacts exist in the new location with the same relative layout as before, that
test stages fetched their inputs from the new location, and that nothing was published to the old
location.

**Acceptance Scenarios**:

1. **Given** a pipeline run after cutover, **When** its build stages finish, **Then**
   build_info.txt, every packaged build, and the waive_list/ directory exist under
   `sw-tensorrt-llm-generic-local/llm-artifacts/` on artifactory.nvidia.com, at the same relative
   path (pipeline name, build number, subdirectory, file name) used today.
2. **Given** those artifacts, **When** the run's test stages start, **Then** they fetch the build
   outputs and the merged waive list from the new location, and their outcomes match the
   corresponding run of the commit just before the merge.
3. **Given** a completed run after cutover, **When** the old location is checked, **Then** none
   of that run's in-scope artifacts were published there.
4. **Given** a link that the run printed or posted to an in-scope artifact (build info, waive
   list, or a packaged build), **When** a user with access opens it, **Then** it resolves on the
   new instance.

---

### User Story 3 - History-reading tools keep working across the cutover (Priority: P2)

As the owner of a tool that reads earlier builds, I need my tool to keep finding the in-scope
artifacts of builds published before the cutover, which stay on the old location, so my tool keeps
its history instead of silently degrading until new history accumulates. This applies from the
first run whose pipeline code contains the change, including the change's own merge-request runs
before it merges (FR-002).

**Why this priority**: Coverage-based test selection, build optimization profile generation, and
build reuse all read the build records or packaged builds of earlier builds. If they looked only
in the new location, they would find nothing from before the cutover, so they would select tests
poorly, find nothing to profile, or rebuild from scratch. CI keeps running in that state, so this
ranks below Story 2.

**Independent Test**: In a run that contains the change (before or after it merges), run each
history-reading tool against a mix of builds published before and after the cutover. Confirm that
it finds every requested artifact, reaching pre-cutover builds through the fallback (SC-004), and
that build reuse rebuilds a pre-cutover build and logs why (FR-016).

**Acceptance Scenarios**:

1. **Given** a build published before cutover and still retained on the old location, **When** a
   reader in a run that contains the change, including the change's first merge-request run,
   requests its build_info.txt, a packaged build, or its waive_list/ files, **Then** the
   reader finds nothing in the new location, falls back to the old location, receives the original
   file, and records that the old location served it.
2. **Given** a tool that reads both in-scope artifacts and data that stays on the old location
   (for example, coverage archives and coverage-selection pins), **When** it runs after cutover,
   **Then** it reads each from the instance that holds it and produces the same result it
   produced before cutover.
3. **Given** a request to reuse the artifacts of a build published before cutover, **When** the
   pipeline handles it, **Then** it rebuilds and logs why, rather than failing (FR-016).
4. **Given** the new location rejects a reader's credentials, **When** the reader requests an
   artifact, **Then** it fails with an authentication error instead of falling back to the old
   location.
5. **Given** the build-info question, **When** the inventory is complete, **Then** either the
   exact build names have been sent to the Artifactory team for migration, or it is recorded that
   no pipeline publishes build-info records.

---

### User Story 4 - No in-scope reference to the old location remains (Priority: P2)

As the Artifactory program, and as the CI maintainer who requests sign-off, I need every in-scope
reference to the old location repointed in pipelines, scripts, tools, tests, configuration, and
documentation, so that the in-scope artifacts no longer depend on the old repository.

**Why this priority**: Program sign-off for this migration depends on it, and any missed reference
becomes an outage once the old repository stops serving these artifacts. It follows Stories 2 and
3 because it finishes the migration rather than enabling it.

**Independent Test**: Search every in-scope branch for in-scope references to the old location.
Only allowlisted, non-functional references may remain, each with a recorded reason. Then follow
the developer documentation for downloading a build output, using a personal token.

**Acceptance Scenarios**:

1. **Given** the in-scope branches after the change, **When** they are searched for in-scope
   references to the old location, **Then** every remaining match is on the allowlist with a
   reason.
2. **Given** a developer tool or document that downloads build outputs, **When** a developer
   follows it with a personal token, **Then** the download succeeds from the new location.
3. **Given** all stories are complete, **When** the migration is submitted through the program's
   support channel, **Then** it is signed off with no open blockers.

---

### Edge Cases

- **Run in progress at cutover**: a run whose pipeline code predates the change, including jobs it
  triggers after the merge, publishes and reads its artifacts on the old location from start to
  finish (FR-002, FR-006).
- **Pull requests in flight at the merge**: the change's own pre-merge runs, and pull requests
  based on it, already publish to the new location; pull requests not rebased past the merge keep
  publishing to the old location until they pick up the change (FR-002).
- **Run in progress at a rollback**: a run that is in progress when CI rolls back, or returns to
  the new location, may end up with its artifacts split across both locations; its readers still
  find each one through the fallback (FR-006, FR-014).
- **One build's outputs on two instances**: until follow-up work moves the rest, in-scope
  artifacts live on the new instance and every other output on the old one, sometimes in the same
  directory. Each link and lookup must point to the instance that holds the specific file (FR-007,
  FR-015).
- **Shared CI components loaded by every branch**: release branches load the same shared CI
  library as main, so changing where it publishes by default would also move their uploads while
  their own steps still download from the old location. The library's change is therefore
  opt-in (FR-025), and a branch that has not adopted the change must see no difference (FR-018).
- **Reuse of a pre-cutover build**: the pipeline reuses that build's artifacts from the old
  location or rebuilds, and says which, instead of failing (FR-016).
- **Package replaced after the build**: when optimization-profile generation replaces a build's
  package with an optimized copy under the same name, the copy is published like any other
  in-scope artifact (FR-002, FR-022), so readers get the optimized package, not the original
  (FR-001).
- **Token expired or revoked mid-run**: the step fails quickly with a message naming the credential
  (never its secret) and saying that its token needs renewal, instead of retrying indefinitely.
  A step that is best effort today logs the same message and carries on (FR-013).
- **Remote GPU test clusters**: these run outside the main CI environment. They receive read
  credentials for the duration of a job, and the credentials do not remain on shared hosts
  afterwards (FR-012).
- **Location derived from the old repository name**: tools that find artifacts by matching or
  stripping the old repository name must still produce correct locations under the new name.
  Verification covers each such tool (FR-020).
- **Developer without access opens a CI link**: they are asked to authenticate, and internal
  developer documentation explains how to get access (FR-008).
- **Earlier build already deleted by retention**: tools report it as unavailable, as they do today.
- **New location rejects a reader's credentials**: the reader reports an authentication error and
  does not fall back, so a broken token cannot hide behind older data on the old location
  (FR-014).
- **Content-delivery network rejects a download**: the new instance redirects downloads to a
  content-delivery network, which can reject a request on its own. The reader retries once without
  the redirect; if that also fails, it reports a transfer failure, not an expired token, and does
  not fall back (FR-014).
- **Old repository retired**: builds published before cutover become unavailable, and readers
  treat them like builds deleted by retention.
- **New instance outage during the transition window**: publishing and reading return to the old
  location within one hour of the decision to roll back (FR-022). Artifacts that exist only on
  the new location stay unreadable until it recovers.
- **A missed reader of the old location**: before the old repository is retired, its access
  records, if the program can supply them, show no remaining requests from TRT-LLM CI.
- **Large and concurrent transfers**: multi-gigabyte packaged builds, uploaded once and downloaded
  by dozens of parallel test stages, transfer within existing pipeline time limits. A transfer
  that drops partway starts again from the beginning within the step's existing retry limits.

## Requirements *(mandatory)*

### Functional Requirements

**Scope and publishing**

- **FR-001**: The migration MUST cover exactly three artifact types per build: build_info.txt,
  the packaged builds, and the waive_list/ directory, together with everything that publishes or
  reads them. Packaged builds are all TRT-LLM package archives (*.tar.gz) in a build's directory,
  whichever job publishes them: the build stages' packages and their bolted-/unbolted- copies,
  the optimized package that optimization-profile generation publishes under the original name,
  and the source release package. In this specification, "in-scope" artifacts and references are
  these. A wheel packed inside a packaged build moves with it. Every other output under
  llm-artifacts (including test-result and coverage archives, and wheels uploaded on their own) and
  every other urm.nvidia.com reference stays where it is until follow-up work moves it.
- **FR-002**: A run's publish location is decided by the pipeline code it runs, not by the time it
  runs. Cutover for a branch is the merge of this change into it; the term "run after cutover" means
  a run whose pipeline code contains the change, and jobs a run triggers follow that run's commit.
  Every such pipeline run MUST publish its build_info.txt, its packaged builds, and its
  waive_list/ directory to artifactory.nvidia.com, repository `sw-tensorrt-llm-generic-local`,
  under `llm-artifacts/`, and MUST NOT publish them to urm.nvidia.com, repository
  `sw-tensorrt-generic`, except while CI is rolled back (FR-022). There is no period of
  publishing to both.
- **FR-003**: Each in-scope artifact's path below `llm-artifacts/` (pipeline name, build number,
  subdirectory, and file name) MUST stay the same as today, so that lookups by pipeline and build
  number keep working.
- **FR-004**: Metadata that pipelines attach to artifacts at publish time today MUST also be
  attached on the new location.

**Consumption**

- **FR-005**: Every pipeline stage, tool, and script that reads in-scope artifacts MUST look for
  them first in the location that runs currently publish to: the new location, or the old
  location while CI is rolled back (FR-022). FR-014 covers the fallback to the other location.
  Every reader MUST apply this order, the fallback, and the download checks of FR-014 through one
  shared read helper, which runs unchanged on CI agents, in containers, inside the image build, and
  on the remote GPU test clusters. A reader that moves the bytes itself, to download in parallel,
  MUST take the file's location and expected checksum from that helper. Deciding whether an earlier
  build can be reused (FR-016) is not a read. The helper reports each outcome other than success
  (absent from both locations, rejected by the location it checks first, a failed transfer, a usage
  error) with its own exit code. The helper and every reader that acts on a specific outcome MUST
  refer to its code by the same name, never by a bare number, and a unit test in the default
  pre-merge run MUST fail when a reader's value for one of those names differs from the helper's.
- **FR-006**: A pipeline run MUST read its own artifacts from the location it published them to,
  including runs in progress at cutover and runs in progress when CI rolls back or returns to
  the new location.
- **FR-007**: Links that pipelines print or post for people MUST point to the instance that holds
  the linked file: the new location for in-scope artifacts, and the old location for everything
  else. Where a linked directory now spans both instances, people MUST be able to reach both
  parts, for example through one link per instance.
- **FR-008**: Developer-facing download tools and documentation MUST work with an individual's own
  token. The steps to get access (membership in the program's distribution list and a token from the
  self-service portal) MUST live in internal documentation. Repository files, including tool
  messages and docs, MUST say only that an Artifactory access token for artifactory.nvidia.com is
  needed, and MUST NOT name internal lists, portals, channels, or people.

**Access and secrets**

- **FR-009**: Every automated read and write of the new location MUST be authenticated; no
  consumer may rely on anonymous access to it. Fallback reads of the old location keep today's
  access method.
- **FR-010**: Publishing MUST use a managed service identity that belongs to
  the program's distribution list, holds a token issued through the self-service portal, and is kept in
  the CI system's managed credential store. On the destination, that identity MUST hold
  exactly four permissions: read, deploy, annotate (attach metadata), and delete/overwrite, which
  Artifactory grants as one permission and which the pipelines need to replace existing files.
  It MUST NOT hold administrative permissions. The four cover every operation the pipelines use
  on in-scope artifacts: create, overwrite, server-side copy within the repository, and attaching
  metadata. These mirror the permissions CI uses to publish to the old repository today. The
  evidence for this requirement is that the identity is one CI already holds in its credential
  store and uses with the AWS instance, together with a passing access check (FR-020).
- **FR-011**: Identities used only to read MUST NOT hold write or delete permission on the
  destination.
- **FR-012**: Tokens and passwords MUST NOT appear in logs, published artifacts, printed links, or
  files that remain on shared execution hosts after a job ends.
- **FR-013**: Each identity's token expiry date and renewal owner MUST be recorded in internal
  documentation. An expired or revoked token MUST make the affected step fail within its normal
  retry budget, with a message that names the credential (never its secret) and says that its
  token needs renewal, without naming internal systems (FR-008). Steps that are best effort today
  (the build_info.txt and waive-list uploads, the merged waive-list download, and coverage-based
  test selection's read of build records) instead log that message and carry on as they do today,
  without falling back to the other location (FR-014).

**Earlier builds and data on both instances**

- **FR-014**: This work does not copy the in-scope artifacts of builds published before cutover;
  they stay on the old location until it is retired. A reader that does not find an in-scope
  artifact in the location it checks first (FR-005) MUST fall back to the other location and
  record which location served it. This applies to every run whose pipeline code contains the
  change (FR-002), including the change's own merge-request runs before it merges. Fallback MUST
  happen only when the first location reports the artifact as absent; any other failure, such as
  rejected credentials, MUST surface as an error. A download rejected by the content-delivery
  network that the new instance redirects downloads to MUST be retried once without that redirect,
  and MUST NOT be reported as rejected credentials (FR-013). Every download MUST match the checksum
  the serving location stores for it; a mismatch is a failed transfer, not an absent artifact, so it
  never triggers the fallback.
- **FR-015**: A tool that reads both in-scope artifacts and data that stays on the old location
  MUST read each from the instance that holds it, using the access each instance requires, until
  follow-up work moves the rest.
- **FR-016**: Reusing the artifacts of an earlier build MUST work as it does today, including
  whether they are placed into the current run's directory; only the locations it reads from and
  publishes to change. When today's process cannot be applied because the earlier build's
  artifacts are on the other location, the run MUST rebuild with a log message saying so. It MUST
  NOT fail because the earlier build was published before cutover.
- **FR-017**: The team MUST determine whether any pipeline publishes Artifactory build-info
  records. If any do, their exact build names MUST be sent to the Artifactory team for migration
  before cutover; if none do, that finding MUST be recorded with the sign-off request.

**Rollout, verification, and sign-off**

- **FR-018**: The change MUST be applied to [NEEDS CLARIFICATION: main only, or also the release
  branches that still run CI (as of 2026-10-14, at least release/1.3)? Awaiting confirmation from
  the owner]. These are the "in-scope branches". Whatever the answer, a branch that has not adopted
  the change MUST keep publishing and reading in-scope artifacts at the old location, even though
  every branch loads the same shared CI components (see FR-025).
- **FR-019**: Every in-scope reference to the old location MUST be repointed, except the
  references that the fallback (FR-014) and the rollback (FR-022) need, and entries on a recorded
  allowlist of non-functional references. Each kept reference and each allowlisted entry MUST
  state why it remains.
- **FR-020**: Before cutover, a verification run MUST exercise every operation listed in FR-010
  with the publishing identity, and both an authenticated read and a rejected anonymous read with
  the reading identity, from each class of execution environment: build agents, containerized test
  agents, and each remote GPU test cluster. Verification MUST also cover every tool that derives
  artifact locations from the repository name, and the fallback rule in FR-014. Every changed path
  that publishes or reads in-scope artifacts and that a pull request's CI runs can reach MUST also
  run at least once against the new location before cutover, in CI runs of the switch pull request.
  The first of those runs is the default pre-merge run, started with no options; runs limited to
  chosen stages follow it, and post-merge test stages MAY be added to them by name as sanity
  checks. Paths that only the post-merge pipeline runs are instead validated on the first
  post-merge run after cutover (FR-021), with the rollback (FR-022) available if the new location
  makes one fail. They are the image build's wait for the packaged builds and its download of one;
  optimization-profile generation's read of its input package; publishing a package's bolted- copy
  with its metadata, and reading that metadata; and publishing the optimized package (FR-001),
  which every pipeline currently launches with publishing off. Before cutover, the identities these
  paths use are checked from the environments they run in. The switch pull request MAY be opened
  as a draft, and its CI runs MAY start, before the permission grants and the verification runs
  above are complete; a run that fails because a grant is missing is expected while it is a draft.
  The permission grants, the verification runs, and the confirmation that child jobs run the pull
  request's own pipeline code (FR-002) MUST be complete before it is marked ready to merge.
- **FR-021**: After cutover, the first merge-request run and the first post-merge run that contain
  the change MUST be compared with the corresponding runs of the commit just before the merge, and
  MUST produce the same artifact set and the same stage outcomes, apart from where the in-scope
  artifacts are published.
- **FR-022**: Until the old repository is retired, it MUST be possible to return publishing and
  reading to the old location within one hour of deciding to roll back. CI administrators roll
  back by changing one CI-wide setting, with no code change, and return to the new location the
  same way. While CI is rolled back, runs publish in-scope artifacts to the old location and read
  it first (FR-005, FR-014), so artifacts published before, during, and after a rollback stay
  readable, even when one run's artifacts end up split across both locations.
- **FR-023**: Blockers MUST be raised through the program's support channel, and the completed
  migration MUST be submitted through the same channel for sign-off.
- **FR-024**: Before the change merges, the pull request and the shared library merge request
  MUST be sent for review to all TRT-LLM infrastructure developers; that review is the notice. It
  MUST state the new location and that an Artifactory access token is needed; the access steps
  stay in internal documentation (FR-008). No consumer has to switch anything by hand: consumers
  in this repository move with the code (FR-002). While the pull request is a draft, its single
  commit also carries this feature's design documents for the infrastructure developers to
  review, with internal names replaced by role descriptions (FR-008); they are removed from that
  commit before the pull request is marked ready.
- **FR-025**: The upload helper in the shared CI library, which every branch loads from the
  library's main line, MUST gain an opt-in setting that selects the new location. Its default
  MUST remain the old location, so that branches that have not adopted the change, and every
  out-of-scope upload that uses the helper, keep their current destination. In this ticket, only
  the build_info.txt and waive_list/ uploads use the opt-in. Packaged builds MUST NOT be moved onto
  the helper: they keep this repository's own upload code, which attaches their metadata, pointed at
  the new location, so that neither the helper's nor this repository's path to the old location
  changes. The library change MUST merge before
  the change that switches this repository, and MUST NOT by itself change where any upload goes.
  Testing of the switch may start before then: while under test, the switch pull request may load
  the library from the library change's branch. It MUST load the library from the library's main
  line again, and pass the default pre-merge run that way, before it merges.

### Key Entities

- **Build artifact set**: everything one pipeline run publishes, identified by pipeline name and
  build number. The in-scope part is the build record (build_info.txt, with the source revision
  and branch), the packaged builds (the TRT-LLM package archives that FR-001 lists), and the
  merged test waive list (waive_list/). The rest, including test-result and coverage archives,
  wheels uploaded on their own, HTML reports, and scan logs, is out of scope for this ticket.
- **Shared cross-build data**: data not tied to one run that tools read repeatedly, such as the
  latest optimization profiles per branch and CPU architecture, coverage-selection pins, and the
  latest image-tag map. It lives outside build directories and is out of scope for this ticket.
- **Artifact location**: an instance, repository, and path prefix. Old: urm.nvidia.com,
  `sw-tensorrt-generic`, `llm-artifacts/`. New: artifactory.nvidia.com,
  `sw-tensorrt-llm-generic-local`, `llm-artifacts/`, which already exists. "Destination" and "new
  location" mean the new `llm-artifacts/` path; "new repository" means
  `sw-tensorrt-llm-generic-local`; "new instance" means artifactory.nvidia.com. The old terms
  follow the same pattern. Permissions in FR-010 and FR-011 apply to the destination.
- **Access identity**: a service or personal identity with distribution-list membership, a token,
  permissions on the destination, a token expiry date, and a renewal owner. CI uses two roles:
  publisher (read-write) and reader (read-only).
- **Artifact consumer**: a pipeline stage, tool, or person that reads artifacts, characterized by
  where it runs and how it authenticates.
- **Build-info record**: Artifactory's own build metadata, keyed by build name. These pipelines
  may not publish any.
- **Reference allowlist**: references to the old location that are kept on purpose, each with a
  reason.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of pipeline runs whose pipeline code contains the change (FR-002) and that do
  not overlap a rollback
  publish all of their in-scope artifacts to the new location, and 0 of those artifacts appear in
  the old location.
- **SC-002**: In the first two weeks after cutover, 0 CI runs fail because an in-scope artifact
  could not be published or retrieved, excluding outages of the new instance itself.
- **SC-003**: The median duration of a merge-request pipeline run in the first week after cutover
  is within 10% of the median for the week before cutover.
- **SC-004**: For at least the 20 most recent main-branch post-merge builds and 10 older retained
  builds published before cutover, 100% of requests for their build_info.txt, packaged builds, and
  waive_list/ files succeed through the fallback and return the original files.
- **SC-005**: Removed on 2026-10-16. SC-004 covers reads of pre-cutover builds, and FR-016 covers
  build reuse.
- **SC-006**: 100% of unauthenticated read attempts against the destination are rejected during
  verification, and a scan of all pipeline logs from the first week after cutover finds 0 exposed
  tokens or passwords.
- **SC-007**: A search of every in-scope branch finds 0 functional in-scope references to the old
  location other than those the fallback and the rollback need (FR-019); every other remaining
  match is on the allowlist with a reason.
- **SC-008**: The Artifactory program signs off on the migration of the in-scope artifacts, with 0
  open blockers, before the old repository's retirement date.
- **SC-009**: If a rollback is needed during the transition window, it completes within 1 hour of
  the decision.

## Assumptions

### Current state (observed in the repository on 2026-10-14)

- CI already holds credentials for the AWS instance, but uses them only for container images: one
  read-write identity publishes images, and another pulls images, including on remote GPU test
  clusters. Neither identity has been confirmed to hold any permission on
  `sw-tensorrt-llm-generic-local`.
- Build artifacts are published with credentials for the old instance.
- Test stages and several helper tools read artifacts from the old instance without credentials.
  Each one needs authenticated access after the change, which is likely to cost more than the
  publishing change itself.
- Most artifact publishing, including build_info.txt and waive_list/, goes through an upload
  helper in a shared CI library. The library is maintained outside this repository, every branch
  loads it from the library's main line, and the helper chooses the destination instance itself:
  it always uploads to urm.nvidia.com with the old instance's credential, and callers cannot
  choose another.
  Packaged builds are the exception: this repository's own pipeline code publishes the build
  stages' packages and their optimized replacements.
- The publisher of the source release package was not found in this repository and is still to
  be identified. Its only reader here sits in a package sanity check whose only call is commented
  out, so it never runs; it stays unchanged, on the reference allowlist (FR-019).
- Neither the pipeline code in this repository nor the shared CI library publishes Artifactory
  build-info records.
- A pull request's CI runs, including its post-merge mode, cannot exercise the image build's reads
  of packaged builds, optimization-profile generation, or publishing a package's bolted- copy with
  its metadata. The pipeline enables these only in the post-merge pipeline itself; post-merge mode
  only selects the post-merge test stages. A run limited to chosen stages skips coverage-based test
  selection and does not update the pull request's CI status.
- The repository is public, and every merge-request run scans it for confidential terms; a hit
  marks the stage unstable but does not block the merge. Tracked files name no internal
  distribution lists, portals, or channels today.
- No GitLab CI or GitHub Actions workflow in this repository references the old location. The
  references are in CI pipeline definitions, helper scripts, test utilities, and documentation.
- The precedent TRTLLM-15275 does not appear in this repository's history and was not reviewed
  directly; its lessons come from the feature description.

### Defaults chosen where the description was silent

- No consumer outside this repository reads in-scope artifacts by a fixed address, so review by
  the infrastructure developers is enough notice (FR-024).
- Builds published before cutover are not copied to the new location. Readers reach them through
  the fallback in FR-014 until the old repository is retired (confirmed by the owner on
  2026-10-16).
- Until follow-up work moves the remaining outputs, each build's outputs are split across the two
  instances, so pipelines and tools that need both kinds hold credentials for both.
- Readers use a read-only identity that is separate from the publishing identity. The existing
  image-pull identity is the first candidate, subject to a permission check.
- The old repository keeps accepting reads and writes through a transition window long enough to
  migrate, verify, and roll back if needed. The program sets its retirement date.
- Retention on the new repository matches the old repository's current policy.
- The new instance can handle today's peak load (multi-gigabyte packages and dozens of parallel
  test stages) within existing pipeline time limits.
- Developers trigger and read CI exactly as they do today.

### Out of scope

- Moving TRT-LLM's own container images, which are already on the AWS instance.
- Test-result archives, which move under TRTLLMINF-472 (Migrate test results to AWS Artifactory).
- Wheel files uploaded on their own, and the links to them, which move under TRTLLMINF-295
  (Migrate wheels to artifactory.nvidia.com). The wheel packed inside each packaged build moves
  with the packaged build in this ticket.
- Every other output under llm-artifacts, including coverage archives, HTML reports, scan logs,
  image-tag maps, and the shared stores outside build directories (optimization profiles,
  coverage-selection pins). No follow-up ticket is named for these yet.
- Every other urm.nvidia.com reference in the repository: container-image proxies, the package
  index, the source mirror, and third-party downloads.
- Changing artifact contents, formats, or retention policy.
- Fixing problems that exist today in the flows this work changes, beyond what this specification
  requires; they stay as they are handled today. In particular, a download is not required to
  resume a transfer that drops partway.
- Provenance records that name an old-instance container image as the source of a golden reference
  file. They describe past runs and stay as they are.

### Dependencies

- The Artifactory program: the destination repository and its permissions, distribution-list
  approval, build-info migration if needed, transition dates, and sign-off.
- The owners of the shared CI library, to review and merge the opt-in (FR-025) before the
  switch, and the owners of the CI credential store.
- The owners of the remote GPU test clusters, to deliver read credentials to test jobs.
- The owners of history-reading tools (coverage-based test selection, optimization profiles,
  build reuse), to verify their tools after cutover.
- Follow-up tickets, to start after this migration is merged and validated: TRTLLMINF-472 (test
  results) and TRTLLMINF-295 (wheels). They can reuse the access identities, the shared read helper
  (FR-005), and the verification approach established here. The old repository cannot be retired
  until every remaining llm-artifacts output has moved.
