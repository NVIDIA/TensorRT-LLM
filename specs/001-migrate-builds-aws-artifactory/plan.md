# Implementation Plan: Migrate Build Artifacts to AWS Artifactory

**Branch**: `TRTLLM-17014-migrate-builds-to-aws-artifactory` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-migrate-builds-aws-artifactory/spec.md`

## Summary

Move three per-build artifact types from urm `sw-tensorrt-generic/llm-artifacts` to AWS Artifactory
`sw-tensorrt-llm-generic-local/llm-artifacts`: `build_info.txt`, the packaged builds (every TRT-LLM
package archive in a build's directory), and `waive_list/`. The switch happens when the change
merges, with no dual publishing. The work ships as one pull request in this repository, from the
current worktree's branch rebased on `upstream/main`, plus the shared library's merge request
!111, already open (research R19).

The approach, from [research.md](research.md):

- **Publishing**: packaged builds keep their upload code in this repository, pointed at the new
  location. `build_info.txt` and `waive_list/` keep going through `trtllm_utils.uploadArtifacts`
  in `trtllm-jenkins-shared-lib`, which gains an opt-in destination whose default stays urm. That
  library change is a separate merge request that lands before the switch merges and moves nothing
  by itself. The switch PR is tested against the merge request's branch before then (research R17).
- **Reading**: every reader runs one helper, `jenkins/scripts/fetch_build_artifact.sh`, which tries
  the primary location and falls back to the other only on 404. It verifies each download's SHA-1,
  and retries once without the CDN redirect when the CDN rejects a download: two safeguards taken
  from bloom's download code, which is not reused itself (research R18). The Slurm nodes and
  coverage-based test selection run it too. BoltProfileGen keeps its parallel download, its only
  user, and takes the file's location and checksum from the helper (spec FR-005, research R4).
- **Identities**: reuse the two Artifactory identities CI already holds. The reader's netrc reaches
  each Slurm job as a file in its workspace, the way the job's enroot credentials already do
  (research R5). The publisher gets exactly read, deploy, annotate, and delete/overwrite.
- **Rollback**: a Jenkins-level switch, `LLM_ARTIFACTS_PRIMARY`, decides where everything in scope
  publishes and reads first, including the two uploads through the shared helper. A rollback takes
  minutes, not a revert merge. A parent pipeline passes its value to the jobs it starts, which can
  run on other Jenkins controllers.
- **Verification**: every changed path that a pull request's CI can reach runs against the new
  location before cutover, in the switch PR's default pre-merge run and the targeted runs after it,
  with post-merge test stages added by name when a sanity check needs them. Four paths run only in the post-merge pipeline: the image
  build's reads, BoltProfileGen's input read, the `bolted-<tar>` property, and the optimized-package
  publish. Their identities are checked beforehand in the environments they run in, and the paths
  are validated on the first post-merge run after cutover, with the rollback switch for failures
  that the new location causes (research R16).

## Technical Context

**Language/Version**: Jenkins scripted-pipeline Groovy (this repository and
`trtllm-jenkins-shared-lib`), Bash, Python 3.10+ (standard library only)

**Primary Dependencies**:

- Jenkins with the JFrog Artifactory plugin (`rtServer`, `rtUpload`) and credentials binding.
- Artifactory REST API: download, storage properties, copy.
- curl.
- `trtllm-jenkins-shared-lib`, loaded at `@main` by every branch: `uploadArtifacts` gains an
  opt-in destination (spec FR-025), implemented on the merge request branch
  `feature/trtllm-17014-aws-upload-destination`, open as merge request !111 (research R2, R19). While the PR is under test,
  `L0_MergeRequest` loads that branch instead (research R17).
- `bloom-jenkins-shared-lib` is not changed, and its `ArtifactManager` download is not reused
  (research R18).

**Storage**: Artifactory generic repositories. urm `sw-tensorrt-generic` (virtual, backed by
`sw-tensorrt-generic-local`) moves to AWS `sw-tensorrt-llm-generic-local` (local).

**Testing**:

- pytest modules in `tests/unittest/scripts/`, already scheduled by `l0_cpu.yml` and `l0_a10.yml`.
- The default pre-merge run (`/bot run` with no options) first: it is the only run that exercises
  coverage-based test selection and sets the PR's CI status. Then targeted runs with
  `/bot run --stage-list`. Post-merge test stages can be added by exact name as sanity checks; only
  patterns such as `*Post-Merge*` need the `ci: post-merge approved` label (research R16).
- A final default pre-merge run after the PR loads the shared library from `@main` again (research
  R17).
- An access-check script run in each environment, including both roles on the BoltProfileGen
  cluster frontend.
- The first post-merge run after cutover, for the four paths that only the post-merge pipeline
  runs (research R16).
- The shared library change, tested in its own merge request by that repository's Groovy harness
  (`UploadArtifactsTest`).

**Target Platform**: TRT-LLM CI: Kubernetes build and test pods, Docker-build pods, and remote Slurm
GPU clusters (ENROOT and other runtimes).

**Project Type**: CI/CD infrastructure change inside a library repository, plus one function in a
shared Jenkins library.

**Performance Goals**:

- Median merge-request pipeline duration within 10% of pre-cutover (SC-003).
- Rollback within 1 hour (SC-009).

**Constraints**:

- No anonymous access on the AWS instance.
- No secrets in logs, command lines, or files left behind.
- Branches without the change are unaffected, even though all branches load the shared library at
  `@main`: the helper's default destination stays urm (FR-025).
- The shared library change merges before the switch merges and moves no upload by itself. The
  switch loads the library from `@main` again, and passes the default pre-merge run that way,
  before it merges (FR-025).
- Cutover at merge, with no dual publishing.
- One read helper for every in-scope read (FR-005), so the read rule, the CDN retry, and the
  checksum check exist once. The only transfer outside it is BoltProfileGen's parallel download,
  which takes the location and checksum from the helper. The reuse check is not a read (FR-016).
- One CI run per pull request at a time: each `/bot run` kills the PR's running jobs (research
  R16).
- Out-of-scope artifacts keep their urm paths, including the 14 other uploads through the shared
  helper.
- Internal names stay out of repository files: the repository is public (FR-008), and every
  merge-request run scans for confidential terms.

**Scale/Scope**:

- 14 consumer sites (one unreachable and left as it is, research R13) and 5 producer sites, across
  5 pipeline files, 4 scripts, and the image build (`docker/Dockerfile.multi`, `docker/Makefile`)
  (research R1), plus one function in `trtllm-jenkins-shared-lib`.
- Multi-gigabyte packaged builds, downloaded by dozens of parallel test stages per run.

**Shared helper, as read** (research R2): `uploadArtifacts(patternAbsPath, target)` always uploads
to urm with `urm-artifactory-creds`. It gains a trailing `destination = 'urm'` parameter
([contracts/shared-library-upload.md](contracts/shared-library-upload.md)).

**Open decisions that do not block this design** (research R15):

- Release-branch rollout (spec FR-018), awaiting the owner.
- No copy of pre-cutover builds (spec FR-014): confirmed by the owner on 2026-10-16.
- The packaged-build list (spec FR-001), awaiting the owner's confirmation. It is needed before the PR
  merges, because it decides whether the optimized-package publish moves with this change.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against constitution v1.0.0. **Result: PASS**, with two justified deviations from
Principle V (see Complexity Tracking).

| Item | Status | How the plan complies, or why it does not apply |
|---|---|---|
| I. Coding guidelines and tooling | Compliant | Python follows `CODING_GUIDELINES.md`: annotated functions and narrow exception handling, with the read helper's exit codes mapped to named outcomes. `get_wheel_from_package.py` stays in the legacy lint group with no new violations. New files carry the NVIDIA SPDX header, and modified files get their copyright year updated. Comments describe the code; rationale goes in PR descriptions. |
| II. Stable public APIs and configuration | Not applicable | No LLM API, `LlmArgs`, or API-stability reference change. `LLM_ARTIFACTS_PRIMARY` is a CI pipeline setting, not an LLM configuration knob. |
| III. Telemetry and privacy | Not applicable | No LLM args fields and no architecture identifiers. |
| IV. Extend the established architecture | Compliant | Extends the shared upload helper with an opt-in instead of bypassing it. Reuses `rtServer`/`rtUpload`, `withCredentials`, the way Slurm jobs already receive their enroot netrc, and its cleanup, `llmExecStepWithRetry` and `retry_command`, and existing credential IDs. One read helper serves every reader, including the Slurm nodes and coverage-based test selection; BoltProfileGen's parallel download, its only user, takes the location and checksum from it (spec FR-005, research R4). Bloom's `ArtifactManager` download was evaluated and not reused, because it falls back after any error and needs Python with an internal package; its CDN-bypass and checksum safeguards are adopted (research R18). No subsystem guide applies (no attention, MoE, or VisualGen code). Contributor workflows and library architecture are unchanged, so neither `AGENTS.md` nor the architecture diagram changes. |
| V. Tests ship with the change | Compliant, two deviations | New and changed scripts get pytest coverage in `tests/unittest/scripts/`. Groovy pipeline changes and the image-build files have no unit tests. Where a pull request's CI reaches them, the PR's runs validate them; the four post-merge-only paths are validated on the first post-merge run after cutover (spec FR-020, research R16). The shared library change is outside this repository and has a unit test in that repository's Groovy harness (`UploadArtifactsTest`). |
| VI. Measured performance, preserved accuracy | Compliant | No library performance claim and no lossy change. Transfer time is measured per environment (research R14) and in the PR's default run, and pipeline duration is tracked (SC-003). The PR description states the performance impact. |
| Platform: languages, backend, builds | Compliant | Python 3.10+ standard library. No backend change. Docker-based builds unchanged. |
| Platform: dependencies | Compliant | No new third-party dependencies (curl and the Python standard library only). The pod that runs coverage-based test selection already has the bash, curl, and `sha1sum` that the read helper needs. |
| Platform: ownership | Compliant | Changed paths in this repository are already owned: `/jenkins`, `/scripts`, and `/docker` by trt-llm-infra-devs, `/jenkins/scripts/cbts` by trt-llm-devs. No CODEOWNERS change. The shared library's owners review its merge request. |
| Platform: documentation | Compliant | Developer access steps live in internal documentation (FR-008). Tool messages say an Artifactory access token for artifactory.nvidia.com is needed and name the credential involved. |
| Gate 1: scope | Compliant | Two change sets, one concern each: the shared library merge request (opt-in, no behavior change; open as !111), and one PR in this repository with the helper scripts, their tests, and the switch. The helpers exist only for the switch, so the PR description explains why they are not split out (Gate 5). Backports are separate PRs. |
| Gate 2: branches | Compliant | From a fork to `NVIDIA/TensorRT-LLM` `main`. The library merge request targets that repository's `main`. The PR loads the library from the merge request's branch only while under test (research R17). Backports happen only if FR-018 calls for them; each then gets an NVBug for its release branch, whose CI breaks when urm retires, so it targets that branch as the gate allows. |
| Gate 3: commits | Compliant | `git commit -s`, no AI attribution in the sign-off, no co-author trailers unless asked. |
| Gate 4: PR title | Compliant | `[TRTLLM-17014][infra] ...` |
| Gate 5: PR description | Compliant | What and why, the test list, the performance impact (download times, and one storage-API request and checksum per download), the CI-wide risk, the rollback switch, why the helpers ship with the switch, that every reader runs the one read helper and why BoltProfileGen keeps its own transfer, the shared library merge request named by its branch with no internal URL (FR-008), TRTLLMINF-472, and TRTLLMINF-295. |
| Gate 6: CI | Compliant | The default pre-merge run first, because the change touches every stage's downloads and only that run sets the PR's CI status. The PR changes `jenkins/L0_Test.groovy`, which makes CI add the multi-GPU stages to the default run, so the author asks the CI approvers for `ci: full pre-merge approved`; the first run does not wait for it, since CI checks for the label only when each architecture's single-GPU tests finish (tasks T041, T057). Then targeted `--stage-list` runs, with post-merge test stages named exactly when a sanity check needs them (research R16). No full post-merge run, and no habitual `--disable-fail-fast`. |
| Gate 7: review | Compliant | A TRT-LLM engineer plus the `/jenkins` and `/scripts` code owners, and the shared library's owners for its merge request. No API or telemetry owners needed. |

**Post-design re-check (after Phase 1): PASS, unchanged.**

- The contracts add one script interface, `fetch_build_artifact.sh`, shared by every reader, and
  one optional argument on the existing shared upload helper (Principle IV).
- Every new script has a test module listed in [quickstart.md](quickstart.md) step 1 (Principle V).
- Every changed path without a unit test that a pull request's CI can reach runs in a CI run of the
  switch PR before cutover (quickstart steps 3–8). The four post-merge-only paths first run on the
  first post-merge run after cutover (quickstart step 10), which Complexity Tracking justifies.
- Re-checked after the 2026-10-15 clarifications (research R17, R18): no new dependency and no new
  deviation.
- Re-checked after the 2026-10-16 clarifications (research R4, R16): the second deviation now
  covers four post-merge-only paths instead of one, and no gate changes status.
- Re-checked after the later 2026-10-16 clarifications (research R19): one PR instead of two, the
  default pre-merge run first, the wheel inside packaged builds in scope, and the base moved to
  `upstream/main`. Gate 1 now counts two change sets; no principle or gate changes status.
- Re-checked after the last 2026-10-16 clarification, one read helper for every read (spec FR-005,
  research R4): the Slurm nodes and coverage-based test selection run the helper, so
  `slurm_install.sh` and `artifact.py` drop their own copies of the read rule, and the rule is
  tested once. BoltProfileGen and the reuse check are unchanged. No new dependency, no new
  deviation, and no gate changes status. The PR's commit still has both copies; replacing them is
  implementation work in that one commit.

## Project Structure

### Documentation (this feature)

```text
specs/001-migrate-builds-aws-artifactory/
├── spec.md
├── plan.md                 # this file
├── research.md             # Phase 0
├── data-model.md           # Phase 1
├── quickstart.md           # Phase 1
├── contracts/              # Phase 1
│   ├── artifact-locations.md
│   ├── helper-scripts.md
│   └── shared-library-upload.md
├── checklists/
│   ├── requirements.md
│   └── migration.md
└── tasks.md                # created by /speckit-tasks
```

### Source Code (repository root)

```text
jenkins/
├── Build.groovy                  # packaged builds to the primary (AWS rtServer); reuse rule (R6)
├── L0_MergeRequest.groovy        # pass the primary to the shared helper for build_info.txt and
│                                 # waive_list/ (R2); log links; library branch while under test (R17)
├── L0_Test.groovy                # helper fetched once and stashed (R4); packaged-build, waive-list,
│                                 # and package reads via the helper; BOLT property lookup; Slurm
│                                 # launcher sends the helper and reader netrc to every job
├── BoltProfileGen.groovy         # input location resolved via the helper, parallel download kept;
│                                 # BOLTed canonical publish
├── BuildDockerImage.groovy       # wait-stage poll via the helper; reader netrc and primary passed
│                                 # to the image build
└── scripts/
    ├── fetch_build_artifact.sh                  # new: read rule, checksum, CDN retry (R18)
    ├── check_artifactory_access.sh              # new: per-environment verification
    ├── slurm_install.sh                         # runs the read helper on the node
    └── cbts/cbts/command/coverage/selection/artifact.py  # build_info.txt through the read helper
docker/
├── Dockerfile.multi              # wheel stage: bind-mount the helper, BuildKit secret for the netrc
└── Makefile                      # pass ARTIFACTORY_NETRC as a BuildKit secret, never a build arg
scripts/
├── get_wheel_from_package.py     # download via the helper
└── package_trt_llm.py            # allowlisted default; help text if it downloads (R1)
tests/unittest/scripts/
├── test_fetch_build_artifact.py      # new
├── test_check_artifactory_access.py  # new
├── test_get_wheel_from_package.py    # new
├── test_slurm_install.py             # helper call on the node; launcher wiring
└── cbts/test_coverage_artifact.py    # build_info.txt per helper exit code
```

```text
trtllm-jenkins-shared-lib/        # separate repository, loaded at @main by every branch
├── vars/trtllm_utils.groovy      # uploadArtifacts: trailing destination, default urm (FR-025)
└── src/test/groovy/              # unit test for the new parameter
```

**Structure Decision**: No new directories in this repository. The helpers sit in
`jenkins/scripts/` beside the other CI scripts. Stages that read before they have a source tree take
the read helper from the run's commit, the Slurm launcher copies it to each job, and the image build
bind-mounts it (research R4). The tests
sit in `tests/unittest/scripts/`, which CI already schedules. One function changes in
`trtllm-jenkins-shared-lib` (`vars/trtllm_utils.groovy`).

## Delivery Sequence

| Step | Work | Gate to the next step |
|---|---|---|
| 0 | Request permission targets for both identities (publisher: read, deploy, annotate, delete/overwrite). Ask the program for urm access records (R10). Run the build-info check (R8). | Permissions granted |
| 1 | In parallel: the shared library merge request (opt-in, default unchanged, on `feature/trtllm-17014-aws-upload-destination`, open as !111), sent for review to all TRT-LLM infrastructure developers before it merges (FR-024), and the helper scripts with their tests, committed first on the PR branch and pushed to the fork | The helpers are on the fork, so the access check can take its script from there, and the library merge request's branch is ready to test against. The access check passes in every environment, including both roles on the BoltProfileGen cluster frontend (quickstart step 2). |
| 2 | Verify the child-job code version (R12) | R12 holds; otherwise revisit the design before the PR |
| 3 | The PR: switch every in-scope producer and consumer, the shared helper's destination at its two in-scope calls, the log links, the reuse rule, and `LLM_ARTIFACTS_PRIMARY`. While under test, `L0_MergeRequest` loads the library merge request's branch (research R17). Validate it with the default pre-merge run, then targeted runs (quickstart step 3, research R16), and steps 4–8. Send it for review to all TRT-LLM infrastructure developers (FR-024). Before merging, the library merge request merges (quickstart prerequisite 5), the PR loads the library from `@main` again, and the default pre-merge run passes on that commit. Cutover happens at merge. | Quickstart steps 3–8 pass on the PR, step 11 is done, and the final default pre-merge run on `@main` passes. The build-info check is recorded (FR-017), the packaged-build list is confirmed (FR-001), and the internal access documentation exists (FR-008). |
| 4 | Post-cutover validation (quickstart steps 9 and 10), including the four post-merge-only paths on the first post-merge run, and program sign-off (FR-023). If the new location makes one of those paths fail, roll back with the switch while it is fixed (FR-022). | Sign-off |
| 5 | Release-branch backports, if FR-018 calls for them | the owner's decision |
| 6 | Hand off to TRTLLMINF-472 (test results) and TRTLLMINF-295 (wheels), which can pass the same destination at the other 14 helper calls | Step 4 complete |

Clarified 2026-10-16 (spec FR-020): step 3's PR opens as a draft and runs CI from the start, in
parallel with steps 0–2. Their outcomes (grants, access check in every environment, code-version
check) gate marking the PR ready, not its first CI runs.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Principle V: the Groovy pipeline changes and the image-build files (`docker/Makefile`, `docker/Dockerfile.multi`) ship without unit tests | The repository has no Groovy unit-test harness, the changed Groovy runs only inside Jenkins, and the image-build files act only inside an image build. | Adding a harness (for example, JenkinsPipelineUnit) is a separate infrastructure project. The testable logic (the read rule, path mapping, access checks, and `get_wheel_from_package.py`) lives in scripts with pytest coverage. Every changed path a pull request's CI can reach runs in the CI runs of quickstart steps 3–8; the next row covers the rest. A static check that the credential stays out of the image was declined by clarification; review covers it. |
| Principle V: four paths are not exercised before cutover: the image build's reads (its wait step and the SBSA release image's wheel download), BoltProfileGen's input read, the `bolted-<tar>` property publish and read, and the optimized-package publish (`publishBoltedCanonical`) | The pipeline runs them only when the job itself is the post-merge pipeline (`L0_MergeRequest.groovy:338`, `:2604`, `:2790`, `:2960`); a pull request's `--post-merge` mode only selects post-merge test stages. Every pipeline also launches the optimized-package publish with publishing off (`:2625`). See research R16. | Launching the image job (with `action=build`) and BoltProfileGen by hand on the PR's build would reach them outside `/bot`; clarification declined that (spec FR-020, 2026-10-16). Instead, their identities are checked in the environments they run in before cutover (quickstart step 2), and the paths are validated on the first post-merge run after cutover (quickstart step 10), with the rollback switch for failures that the new location causes. |
