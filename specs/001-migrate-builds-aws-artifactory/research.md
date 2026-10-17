# Research: Migrate Build Artifacts to AWS Artifactory

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-14

All findings come from the repository at commit `91404de7bc` (branch
`TRTLLM-17014-migrate-builds-to-aws-artifactory`). Line numbers refer to that commit. The
implementation is written on `upstream/main` at `dac84b01a6` (R19), where some of this code has
moved; find it by content.

## R1. Inventory of in-scope producers and consumers

**Decision**: The change touches exactly the call sites below, plus one function outside this
repository: `trtllm_utils.uploadArtifacts` in `trtllm-jenkins-shared-lib` gains an opt-in
destination (R2). Everything else that mentions the old host stays as it is (see "Out of scope,
same files").

### Producers (publish build_info.txt, packaged builds, waive_list/)

| Location | What it publishes | How, today |
|---|---|---|
| `jenkins/Build.groovy:391-414` | Packaged builds: `TensorRT-LLM.tar.gz`, `single-device-TensorRT-LLM.tar.gz`, `llvm-TensorRT-LLM.tar.gz`, `TensorRT-LLM-GH200.tar.gz`, `llvm-TensorRT-LLM-GH200.tar.gz` (map at `:127-148`, `:461`), plus `bolted-<tar>` with properties (`:694`) and `unbolted-<tar>` (`:707`) | `rtUpload` through server `Artifactory`, configured at `:436-444` and `:829-837` as `urm.nvidia.com` with credential `urm-artifactory-creds` |
| `jenkins/Build.groovy:341-389` | Reused packaged builds (copied from an earlier build) | Artifactory copy API on urm, `curl` with `urm-artifactory-creds` (`:357`) |
| `jenkins/L0_MergeRequest.groovy:616-617` | `waives*.txt` into `waive_list/`, plus a log link | `trtllm_utils.uploadArtifacts` (shared library) |
| `jenkins/L0_MergeRequest.groovy:667-669` | `build_info.txt`, plus a log link | `trtllm_utils.uploadArtifacts` (shared library) |
| `jenkins/BoltProfileGen.groovy:550-555`, `:801-840` | BOLTed canonical packaged build and `unbolted-<tar>` (off by default, `PUBLISH_BOLTED_CANONICAL`) | `curl -T` with a netrc built from `urm-artifactory-creds` (`:837`) |

### Consumers (read build_info.txt, packaged builds, waive_list/)

| Location | What it reads | Where it runs | How, today |
|---|---|---|---|
| `jenkins/L0_Test.groovy:1746-1773` | Packaged build URL, and `bolted-<tar>` properties via `api/storage/...?properties` (`:1756`) | Jenkins agent | Anonymous `curl` |
| `jenkins/L0_Test.groovy:1819`, `:1856` | Packaged build | Kubernetes test pod | Anonymous `wget` |
| `jenkins/L0_Test.groovy:2204` → `jenkins/scripts/slurm_install.sh:56-61` | Packaged build | Slurm node | Anonymous `wget` |
| `jenkins/L0_Test.groovy:4047-4049` | Packaged build (documentation build) | Kubernetes pod | Anonymous `wget` |
| `jenkins/L0_Test.groovy:4150-4153` | Packaged build (test-list check) | Kubernetes pod | Anonymous `wget` |
| `jenkins/L0_Test.groovy:4791-4793` | `waive_list/waives.txt` | Test pod | Anonymous `wget` |
| `jenkins/L0_Test.groovy:5110-5115` | Packaged build (second test-stage path) | Test pod | Anonymous `wget` |
| `jenkins/L0_Test.groovy:5988-5990` | Source release package `tensorrt-llm-release-src-<commit>.tar.gz`, in `runPackageSanityCheck`, whose only call (`:7062`) is commented out | Nowhere: unreachable (R13) | Anonymous `wget` |
| `jenkins/BoltProfileGen.groovy:274`, `:390-396` | Packaged build, or its `unbolted-` copy | Slurm node | Anonymous `curl` |
| `jenkins/BuildDockerImage.groovy:345-350` → `scripts/get_wheel_from_package.py:130-140` | Packaged build (to extract the wheel) | Inside the image build: the `wheel` stage of `docker/Dockerfile.multi` (`:146-172`) runs it as `BUILD_WHEEL_SCRIPT` | Anonymous `wget` |
| `jenkins/BuildDockerImage.groovy:1044-1100` | `TensorRT-LLM.tar.gz` and `TensorRT-LLM-GH200.tar.gz`, polled for up to 60 minutes ("Wait For Build Job Complete", when `runSanityCheck` is set) | Image job's agent pod, with no source checkout | Anonymous `wget --spider` |
| `jenkins/scripts/cbts/coverage_selection/artifact.py:146` (now `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`) | `build_info.txt` of main post-merge builds | Jenkins agent | Anonymous `urllib` |
| `jenkins/L0_MergeRequest.groovy:2232` → `jenkins/Build.groovy:341` | Packaged builds of an earlier build (reuse) | Build pod | Copy API |
| `scripts/package_trt_llm.py:171-176` | `--addr` default points at an old main post-merge build (`L0_PostMerge/1379`) | Developer machine | Developer tool |

### Out of scope, same files (unchanged by this work)

- Test results, progress archives, rerun reports, and their deletion: `L0_Test.groovy:397-401`,
  `:2560`, `:2590`, `:2614`, `:3620-3625`, `:4087`, `:4600-4605`, `:4777-4781`, `:4863`,
  `:4946-4958`, `:5345-5356`; `L0_MergeRequest.groovy:1836`, `:1886`, `:1935-1936`. These move under
  TRTLLMINF-472.
- Wheels and attribution files: `L0_Test.groovy:5611`, `:5633`, `:5870`, `:5898`, `:5938`, `:5949`.
  These move under TRTLLMINF-295.
- Coverage archives, the test database, and coverage pins: `L0_MergeRequest.groovy:1272`, `:1855-1856`,
  `:1979-1980`; `artifact.py:138`, `:198`.
- Guardwords scan log (`L0_MergeRequest.groovy:726-727`), image-tag map
  (`BuildDockerImage.groovy:1040`; `jenkins/scripts/get_image_key_to_tag.py`), BOLT profile store
  (`BoltProfileGen.groovy:113`, `:749-800`; `scripts/bolt/internal/artifactory.sh`), and the
  perf-regression record link (`tests/integration/defs/perf/perf_regression_utils.py:36`, which
  points readers at test results).
- `UPLOAD_PATH.replaceFirst(...)` at `L0_Test.groovy:5289`: it derives a stage identifier from the
  old-style path, and keeps working because that path is not changed (R3).
- Other urm repositories: container-image proxies, `sw-tensorrt-pypi`, `github-go-remote`, Triton's
  boost download, and the provenance records in VisualGen golden files.

**Rationale**: Clarifications narrowed the scope to three artifact types (spec FR-001). Each row above
is a place where one of those types is published, read, or linked.

**Alternatives considered**: Repointing whole path variables such as `UPLOAD_PATH`. Rejected because
out-of-scope uploads (test results, wheels, reports) share those variables and must stay on urm.

## R2. How in-scope uploads reach the new instance

**Decision**:

- **Packaged builds** keep their upload code in this repository, pointed at the primary location.
  `Build.groovy`'s own `uploadArtifacts` (`rtServer` + `rtUpload`) gets a server definition for
  `https://artifactory.nvidia.com/artifactory` with the publishing credential, and
  `BoltProfileGen.groovy` keeps its `curl -T` upload with the same credential.
- **build_info.txt and waive_list/** keep going through `trtllm_utils.uploadArtifacts` in
  `trtllm-jenkins-shared-lib` (spec FR-025). The helper gains an optional destination:
  - absent or `urm`: today's server, credential, and output, unchanged;
  - `aws`: `https://artifactory.nvidia.com/artifactory` with the publishing credential.
- The two in-scope call sites (`L0_MergeRequest.groovy:616`, `:668`) pass the primary location and
  a target already mapped to its repository (R3). The other 14 call sites pass nothing and keep
  their destination.
- The library change is its own merge request in `trtllm-jenkins-shared-lib`. It merges before the
  switch pull request does and moves no upload by itself. Until then, the switch pull request is
  tested against the merge request's branch (R17).

**Rationale**:

- Clarification (FR-025): the shared helper is the established upload path, so it is extended
  rather than bypassed.
- An opt-in with an unchanged default keeps FR-018. Every branch loads the library at `@main`, and
  only branches whose own code passes the destination move.
- Keeping the path mapping in this repository leaves one mapping rule (R3) and makes the library
  change a choice of server and credential only.
- The follow-up tickets (TRTLLMINF-472 for test results, TRTLLMINF-295 for wheels) can pass the
  same destination at the other 14 call sites.
- Precedent: `bloom-jenkins-shared-lib` already maps each Artifactory host to its credential
  (`Constants.groovy`: `URM_CREDENTIAL_ID`, and `AWS_ARTIFACTORY_CREDENTIAL_ID =
  "aws-artifactory-credentials"`, used in `CloudManager.groovy:1303-1308`). Its habit of putting
  passwords on command lines (`docker login -p` there, `wget --password=` in `Utils.groovy:1432`)
  is not copied (spec FR-012).

**The helper as it is today** (`trtllm-jenkins-shared-lib` on the internal GitLab,
`vars/trtllm_utils.groovy:1163-1185`): `uploadArtifacts(patternAbsPath, target)` defines an
`rtServer` with id `Artifactory`, URL `https://urm.nvidia.com/artifactory`, and
`urm-artifactory-creds`, then runs `rtUpload` inside `llmRetry(2, ...)`. Callers cannot choose the
server, so the library must change. The change, as implemented on the merge request branch
`feature/trtllm-17014-aws-upload-destination` (commit `9b67bcf`):

- A trailing parameter, `destination = 'urm'`, where `null` also means `urm`. Every existing
  two-argument call runs today's code unchanged.
- For `aws`, a server with its own id, `ArtifactoryAws`, URL
  `https://artifactory.nvidia.com/artifactory`, and `aws-artifactory-credentials`, with the same
  proxy, timeout, `llmRetry(2, ...)` bound, and spec text as `urm`. Its own id avoids replacing the
  `Artifactory` server that `Build.groovy` also defines in the same run.
- Any other value fails through `error()` before a server is defined.
- The Artifactory plugin reports an HTTP 401 or 403 only in its log, so the helper cannot tell it
  apart from other failures. When an `aws` upload gives up, the helper logs one note that names the
  credential and says what to check if the log shows 401 or 403, then rethrows.
- `src/test/groovy/UploadArtifactsTest.groovy`, run by `run-tests.sh`, pins today's `urm` calls and
  covers `aws`, the separate server id, invalid values, the failure note, and interrupts.

**Alternatives considered**:

- Bypass the helper and publish the two files with in-repository plugin steps (the previous
  design). Rejected by clarification: the shared library carries the change.
- Change the helper's default destination. Rejected: all 16 uploads would move on every branch at
  once, including release branches whose test stages still read urm anonymously, and test results
  and wheels that belong to the follow-up tickets.
- Move the packaged builds onto the helper too, adding an optional properties argument so that
  `bolt.ref` stays in the same upload request. Declined by clarification (2026-10-16): it would
  change both the helper's urm path and `Build.groovy`'s, and the goal is minimal disruption to
  the flows that depend on urm. `Build.groovy` keeps its own upload function and gains an
  `ArtifactoryAws` server definition instead.

## R3. Location model and path mapping

**Decision**: Keep `ARTIFACT_PATH` and `UPLOAD_PATH` exactly as they are today (old-repository
relative, for example `sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/1234`). Derive
in-scope locations by mapping:

1. Strip a leading `sw-tensorrt-generic/` or `sw-tensorrt-generic-local/` to get the layout path
   `llm-artifacts/<pipeline>/<build>/...`.
2. Prefix the layout path with the location base:
   - new: `https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/`
   - old: `https://urm.nvidia.com/artifactory/sw-tensorrt-generic/`

Callers of the shared helper pass the mapped target, `sw-tensorrt-llm-generic-local/<layout path>`,
when the primary is `aws` (R2). The full rules are in
[contracts/artifact-locations.md](contracts/artifact-locations.md).

**Rationale**: The layout below `llm-artifacts/` stays identical (FR-003), out-of-scope logic keeps
its inputs, and one mapping rule serves Groovy, shell, and Python callers. Both old prefixes occur
today: `get_image_key_to_tag.py` and the delete path in `L0_Test.groovy:4948` use the backing local
repository name.

**Alternatives considered**: Changing the repository prefix inside `UPLOAD_PATH`. Rejected for the
reason given in R1.

## R4. Reading with fallback

**Decision**: Every in-scope read requests the primary location first.

- HTTP 200: use the file.
- HTTP 404: request the secondary location.
- HTTP 401 or 403 from the primary: fail at once. Never fall back.
- Any other failure: retry with the caller's existing retry policy, then fail. Never fall back.

The reader logs which location served the file. Property lookups for `bolted-<tar>` follow the same
rule. Every download is checked against the SHA-1 that the serving location reports, and a download
that the CDN rejects is retried once without the redirect (R18).

The rule is implemented once, in shell, as `jenkins/scripts/fetch_build_artifact.sh`, and every
reader runs it (spec FR-005, clarified 2026-10-16):

- Kubernetes pods and Jenkins agents, through `sh` steps in `L0_Test.groovy` and
  `BuildDockerImage.groovy`.
- The Slurm node. `slurm_install.sh` runs the helper with the packaged build's artifact path, inside
  its existing `retry_command`, and adds no download code of its own. The launcher sends the helper
  and the reader netrc (R5) to every job and exports the primary with them. It still resolves the
  build on the agent with `--exists` and exports the URL as `llmTarfile`, which only the older
  `slurm_install.sh` inside a package built before this change reads. Such a package is on urm, and
  that script downloads it anonymously, as it does today.
- `scripts/get_wheel_from_package.py`, which already shells out to `wget`, and runs inside the image
  build.
- Coverage-based test selection. `artifact.py` reads `build_info.txt` by running the helper from the
  checked-out tree. Its pod (`buildpack-deps:trixie-scm`) already has bash, curl, and `sha1sum`.
- BoltProfileGen's input download, the one reader that moves the bytes with its own code. It
  resolves the location on the cluster frontend with `--exists`, which prints the URL that holds the
  file and the SHA-1 that location reports. It keeps its parallel ranged download
  (`BoltProfileGen.groovy:296-385`), which a single stream would slow down, checks that SHA-1, and
  sends its single-stream fallback without the CDN redirect.

Deciding whether an earlier build can be reused (R6) is not a read. `Build.groovy` checks both
locations with HEAD requests to choose between a server-side copy and a rebuild, and does not run
the helper.

**Getting the helper to its callers**: the helper is in the source tree, but most test stages
download the packaged build before they have one, because their tree comes out of that package
(`L0_Test.groovy:5060-5115`, `:1819-1858`). Callers therefore take the helper from the run's commit:

- `L0_Test` fetches it once per run in "Setup Environment" with `trtllm_utils.checkoutFile` (as
  `L0_MergeRequest.groovy:581` does for `waives.txt`) and stashes it. Each stage that reads
  unstashes it. The Slurm launcher copies it into the job workspace beside `slurm_install.sh` and
  `bash_utils.sh`, and `slurm_install.sh` finds it by its own file name, as it finds
  `bash_utils.sh`.
- `BuildDockerImage`'s wait stage fetches it with `checkoutFile`. The image build bind-mounts it
  from the build context at its repository path, because the `wheel` stage copies `scripts/` but
  not `jenkins/` (`docker/Dockerfile.multi:148-154`).
- `BoltProfileGen` fetches it with `checkoutFile` outside its `alpine` container, which has no git,
  bash, or curl, and sends it to the cluster frontend with the job's other scripts.
- CBTS runs in `L0_MergeRequest` with the source checked out, and runs the helper from that tree.

**Rationale**: This implements spec FR-014 and the clarification "read from new first, if not present
then from old". It also keeps runs safe at the merge instant: a reader with new code still finds
artifacts that an older run published to urm (FR-006). Refusing to fall back on 401 or 403 stops a
broken token from hiding behind stale data. The shell helper can be unit-tested with pytest, as
`tests/unittest/scripts/test_slurm_install.py` already does for `slurm_install.sh`. One
implementation is tested once, and the follow-up migrations (TRTLLMINF-472 for test results,
TRTLLMINF-295 for wheels) can reuse it, because it reads any file under `llm-artifacts/`.

**Alternatives considered**:

- Separate fallback code in every Groovy call site. Rejected: seven copies of the same rule with no
  unit tests.
- A Python helper for every caller. Rejected: Slurm nodes and pods download before any Python
  environment from the package exists.
- Bloom's `ArtifactManager` download. Rejected; see R18.
- A copy of the rule in each reader that does not run the helper: a curl download in
  `slurm_install.sh`, fed a URL and SHA-1 resolved on the agent, and a `urllib` version in
  `artifact.py`. This was the previous design, and the PR's commit implements it. Rejected by
  clarification (2026-10-16, option B): three implementations of the rule, the CDN retry, and the
  checksum check, each with its own tests.
- Moving BoltProfileGen's parallel ranged download into the helper, so that every transfer also
  goes through it (option A). Declined: BoltProfileGen is its only user, the download predates this
  work and also fetches llvm-bolt from GitHub, and changing it adds risk to a path that only the
  post-merge pipeline runs (R16). It moves into the helper if a second reader needs a parallel
  download.

**Retry interaction**: `llmExecStepWithRetry` treats log lines matching `403 Forbidden` or
`URL returned error: 403` as transient (`trtllm_utils.groovy:29`, `:58`) and retries the step. The
helper's exit-3 message therefore must not contain those phrases, so that a rejected token fails
within the normal retry budget instead of being retried as a network fault (FR-013).

Outer retry loops are bounded: `llmExecStepWithRetry` retries a step that fails within 120 seconds
up to `numRetries` times (default 2), and `retry_command` (`jenkins/scripts/bash_utils.sh`) retries
up to 3 times. A rejected token therefore still fails within the step's normal budget. The loops
that wait for a package to appear (`get_wheel_from_package.py`, and the image job's wait stage) run
for up to an hour, so they stop at once on exit code 3.

**Steps that are best effort today** (spec FR-013): the merged waive-list download uses the default
waive list after any failure (`L0_Test.groovy:4790-4804`), and coverage-based test selection
declines when `artifact.py` prints nothing (`L0_MergeRequest.groovy:1354-1406`). Both keep that
behavior after an authentication error: they log the credential message, and neither reads the
other location instead. An expired token still fails the run, because the packaged-build transfers
use the same credentials.

`artifact.py` maps the helper's exit codes. Exit 2 means the build has no `build_info.txt`, as a
missing file does today. Exits 3 and 4 make the command exit 1, so the selection declines at once.
Today a failed read only skips that build and tries an older one. The helper retries, though, so
one failed call can take minutes, and repeating it for each of up to 50 candidate builds could hold
up the pipeline.

## R5. Identities and credential delivery

**Decision**:

- **Publisher**: reuse the Jenkins credential `aws-artifactory-credentials`. It is already onboarded
  for `artifactory.nvidia.com` and used read-write for image pushes (`BuildDockerImage.groovy:607`)
  and nSpect checks (`:1189`), and `bloom-jenkins-shared-lib` names it as its AWS credential.
  Request a permission target on `sw-tensorrt-llm-generic-local` from the Artifactory program
  covering exactly read, deploy, annotate (properties), and delete/overwrite, with nothing
  administrative (spec FR-010). Artifactory grants overwrite only together with delete, and the
  pipelines overwrite: profile generation replaces the canonical package
  (`BoltProfileGen.groovy:832-835`), and a retried upload can land on an existing file. The shared
  helper's `aws` destination (R2) uses the same credential. If the program will not extend that
  identity, onboard a dedicated one through the program's self-service process, as in
  TRTLLM-15275.
- **Reader**: reuse `trtllm-artifactory-credentials` (`L0_Test.groovy:72`). It is already delivered
  to Kubernetes pods (`:6047`) and written as a netrc for Slurm ENROOT clusters (`:1101-1126`).
  Request read permission on the destination repository. Confirm that it lacks deploy and delete
  there (FR-011); the access check attempts a write with it and expects 403.
- **Delivery**:
  - *Kubernetes and Docker-build pods*: `withCredentials`, then a netrc written with `umask 077` in
    the workspace, used with `curl --netrc-file`, and deleted in a `finally` block.
  - *Image builds*: the reader netrc goes to `docker buildx build` as a BuildKit secret and is
    mounted only in the `wheel` stage's `RUN`, as `github_clone_token` already is
    (`docker/Makefile:59`, `docker/Dockerfile.multi:163`). It is never a build argument, because
    `docker history` shows those.
  - *Slurm clusters, every runtime*: the launcher writes the reader netrc on the agent, copies it
    to `${jobWorkspace}/artifactory-netrc` with its mode kept, deletes the agent's copy, and exports
    the path in the job script. Every job gets it, because the helper on the node contacts aws
    whenever aws is the primary, wherever the file is (R4). The job is registered for the off-pod
    cleanup before the copy, and the job cleanup deletes the job workspace
    (`cleanUpSlurmResources`, `:819`). The netrc the job writes for enroot
    (`${jobWorkspace}/enroot-config/.credentials`, `L0_Test.groovy:1942-1968`) cannot be reused:
    the ENROOT srun prologue deletes that directory before `slurm_install.sh` runs. Each attempt
    gets its own entry in the off-pod cleanup ledger, so an earlier attempt's live entry never
    keeps this attempt's workspace unregistered.
  - *Jenkins agents on Slurm nodes* (`runLLMTestlistWithAgent`): `fetchBuildArtifact` writes the
    reader netrc under the agent's `@tmp` and removes it in `finally`. The agent's workspace,
    `/tmp/<node name>`, is gone when the Slurm job ends, also after a cancellation or the time
    limit, so an agent killed mid-fetch leaves no netrc behind (checked 2026-10-16 against the agent
    setup scripts in `bloom-jenkins-shared-lib`):
    - ENROOT: the agent runs inside the job step's container (`srun --container-image`), with only
      the scripts, Slurm-log, and scratch directories mounted (`getMountListForSlurmTest(cluster,
      false)`). `/tmp/<node name>` is in the container's own file system, which pyxis removes when
      the step ends. This assumes the clusters' enroot configuration mounts no host `/tmp` into
      containers, as enroot's defaults do not.
    - DOCKER: the agent runs on the host, and its setup script removes `/tmp/<node name>` from a
      trap on exit, SIGINT, and SIGTERM. Slurm sends SIGTERM before SIGKILL on cancellation and at
      the time limit, and `cleanUpNodeResources` first deletes the Jenkins node, which ends the
      agent.

    Only a node crash, or a SIGKILL before that trap finishes, can leave a netrc there, and no
    cleanup could reach it: the job cleanup runs on the login node, which cannot reach a compute
    node's `/tmp`. No extra cleanup is added. On the DOCKER hosts the same reader credential
    outlives every job anyway: the stage's image pull runs `docker login` with it
    (`runInDockerOnNodeMultiStage`), which stores it in the service account's Docker configuration.
  - *Everywhere*: no secrets in command lines, URLs, or `set -x` output (precedent:
    `L0_Test.groovy:2212`).
- **Old location**: fallback reads stay anonymous, as today (FR-009).

**Rationale**: Both identities already exist and are already distributed. That removes most of the
onboarding cost seen in TRTLLM-15275, leaving permission grants and verification.

**Risk**: if `trtllm-artifactory-credentials` turns out to have write permission on the destination,
a dedicated read-only identity is needed before cutover.

## R6. Reusing an earlier build

**Decision**: When a run reuses an earlier build:

- If the earlier build's packaged builds exist in the primary location, copy them server-side there
  (copy API on the new instance, `sw-tensorrt-llm-generic-local`).
- If they exist only in the secondary location (for example a pre-cutover build on urm), log
  `reuse source is on <secondary>; rebuilding` and rebuild.
- Existence is checked with a HEAD request on each location before copying. Both copies
  authenticate with a netrc file instead of `curl --user`, which today puts the password on the
  command line (`Build.groovy:372`, spec FR-012).

**Rationale**: FR-016 allows a rebuild. A copy across instances would stream multiple gigabytes
through the pipeline, and the case is rare (see below). The new repository is a local repository,
so the copy API works directly. On urm it targets the backing local repository.

**When it happens**: the bot reuses a build only when the same commit runs again
(`.github/workflows/bot-command.yml:51`), and a run's pipeline code comes from its commit (R12). A
run that contains the change therefore never reuses a build made without it. The rebuild case
arises only when `LLM_ARTIFACTS_PRIMARY` changes between two runs of the same commit, which the
rollback rehearsal produces (quickstart step 6).

**Alternatives considered**: Download from urm and upload to the new location. Rejected for
complexity and time, against a one-time saving.

## R7. Cutover and rollback

**Decision**: When this change merges, the code's default primary location becomes the new instance
(FR-002, clarification "stop immediately"). Pipelines also honor a Jenkins global environment
variable, `LLM_ARTIFACTS_PRIMARY`:

- Values: `aws`, or `urm`. Unset means `aws`.
- It selects where in-scope artifacts are published and which location is read first. Reads always
  fall back to the other location on 404.
- The two uploads through the shared helper pass it as their destination (R2), so they follow the
  setting too.
- Jenkins global properties belong to one controller, and `L0_MergeRequest` starts its child jobs
  on a pool of other controllers. So a parent pipeline adds its value to the `globalVars` of every
  job it starts, as `llm_artifacts_primary`, and the child uses it instead of its own controller's
  setting. A job started on its own reads its own controller's setting.
- Caveat for BoltProfileGen: its declarative `parameters` block replaces the job's parameter list
  on every run, and Jenkins drops a parameter that the job does not declare when the run starts.
  The first run of a Jenkinsfile that declares `globalVars` therefore does not receive it, and
  neither does the first run after a run from a branch without this change. Such a run uses its own
  controller's `LLM_ARTIFACTS_PRIMARY`, which differs from the parent's only while a rollback has
  reached some controllers and not others. Its input read then still finds the package through the
  fallback, but a BOLTed build that it publishes lands where the parent's readers look second. When
  the parent's value arrives, the job logs `Build artifacts: primary location <primary>, passed by the
  parent pipeline`, as `Build` and `BuildDockerImage` do, and the first post-merge run after cutover
  checks for that line (quickstart step 10).
- Rollback: a Jenkins administrator sets `urm` in Global properties on every controller that runs
  TRT-LLM pipelines, then checks each one. Runs started after that use urm, with no revert merge.
  Child jobs use the value their parent passes, whichever controller they run on. A run whose own
  controller is updated while it runs can still split its artifacts across both locations, which
  the fallback covers (spec Edge Cases). Builds published to the new location during the AWS
  period stay readable through the fallback.

**Rationale**: FR-022 asks for a rollback within one hour. A revert PR has to pass CI before it
merges, which can take hours. A global setting is the smallest mechanism that meets the target.
Because reads fall back in both directions, flipping it mid-run does not strand artifacts.

**Alternatives considered**:

- Revert PR only. Rejected: it can't meet one hour.
- Publish to both locations for a window. Rejected by clarification.

## R8. Artifactory build-info records

**Decision**: No step in this repository publishes build-info records (`rtPublishBuildInfo` and
equivalents are absent). The shared library does not publish them either (no `rtPublishBuildInfo` or
equivalent in `trtllm-jenkins-shared-lib`). During implementation, list builds through urm's build
API (`/api/build`, with `urm-artifactory-creds`) and record any build names that belong to these
pipelines. If there are any, send the exact names to the Artifactory team before cutover; otherwise
record "none" with the sign-off (FR-017).

## R9. Test strategy

**Decision**:

- **Shell helpers** (`fetch_build_artifact.sh`, `check_artifactory_access.sh`): new pytest modules
  under `tests/unittest/scripts/`. They serve 200, 404, 401, and 500 from a local
  `http.server` and assert fallback, no-fallback, logging, exit codes, and that secrets never reach
  output. No network access.
- **`slurm_install.sh`**: `tests/unittest/scripts/test_slurm_install.py:105-113` asserts
  `wget -nv -O`. Replace that with checks that the script runs the helper found beside it, with the
  artifact path and a job-specific temporary file, inside `retry_command`, and moves the file into
  place only on success; and that the `L0_Test` launcher sends the helper and the netrc to every
  job, with the exports the helper reads. The read rule itself is tested once, in
  `test_fetch_build_artifact.py`.
- **`artifact.py`**: extend `tests/unittest/scripts/cbts/test_coverage_artifact.py` with
  `build_info.txt` cases for each of the helper's exit codes, without network access.
- **`get_wheel_from_package.py`**: add a unit test for primary and fallback URL construction.
- **Groovy pipelines and the image-build files**: the repository has no Groovy unit-test harness,
  and `docker/Makefile` and `docker/Dockerfile.multi` act only inside an image build. The CI runs
  of the switch PR validate every changed path that a pull request reaches
  ([quickstart.md](quickstart.md) step 3). The four post-merge-only paths, including the image
  build's, are validated on the first post-merge run after cutover, as part of the comparison in
  FR-021 (R16, quickstart step 10). By clarification, there is no separate check that the
  credential stays out of the image; review covers it.
- **Shared helper**: tested in its merge request by `UploadArtifactsTest` (R2). Here, the no-change
  check after it merges (quickstart prerequisite 5) and the switch pull request's runs cover it.

`unittest/scripts` is already scheduled by `l0_cpu.yml:218` and `l0_a10.yml:253`, so new modules
there run in CI without test-list changes.

## R10. Notifying consumers (FR-024)

**Decision**: The notice is the review itself. The pull request and the shared library merge
request go to all TRT-LLM infrastructure developers before they merge, and state the new location,
that a token is needed, and that older builds stay readable on urm until it is retired. Nothing in
the public pull request names internal lists or portals (R11). No consumer switches anything by
hand: every consumer in this repository moves with the code. If the program can supply urm access
records, a client outside TRT-LLM CI in them would disprove the spec's assumption of no
fixed-address consumers and would be contacted directly.

## R11. Confidentiality of repository content

**Decision**: Repository files (code, comments, tests, docs) name only hostnames that are already
present in tracked files (`artifactory.nvidia.com`, `urm.nvidia.com`). They do not name internal
distribution lists, portals, chat channels, or people. Developer access instructions (FR-008) live in
internal documentation, and tool error messages point to "an Artifactory access token for
artifactory.nvidia.com".

**Rationale**: The repository is public (spec FR-008). Every merge-request run also applies
`confidentiality-scan` to the tree (`L0_MergeRequest.groovy:710-718`), which only warns: a hit marks
the stage unstable (`:721-724`). None of those internal names appear in tracked files today. The
`specs/` directory names them only by role (the owner, the program's distribution list,
self-service portal, and support channel): while the switch pull request is a draft, its single
commit carries the spec store for the infrastructure developers to review, and the commit drops it
before the pull request is marked ready (clarified 2026-10-16).

## R12. Child-job code version

**Decision**: Treat as an assumption, to be verified at the start of implementation. Child jobs
(`/LLM/helpers/Build-*`, `L0_Test-*`, `/LLM/helpers/BoltProfileGen`) receive `gitlabCommit`
(`L0_MergeRequest.groovy:2190-2197`) and load their pipeline code at that commit. If so, every job
in a run uses the same code version, and a run publishes and reads consistently (FR-006).

**Verification**: Inspect the helper jobs' Jenkins configuration, or observe that a PR editing
`Build.groovy` runs its own edit. If the assumption fails, stop and revisit the design before the PR:
a child running older pipeline code would ignore any location the parent passed and keep publishing
to urm, so FR-002 could not hold. Reads would still work because of R4.

## R13. Source release package producer

**Finding**: `tensorrt-llm-release-src-<commit>.tar.gz` is read at `L0_Test.groovy:5988`, but no
producer with that name exists in this repository. Comments mention `runLLMPackage`, which is not
defined here. The read sits in `runPackageSanityCheck` (`:5947`), whose only call (`:7062`) is
commented out, so nothing in this repository reads the package either.

**Decision**: It is in scope (spec FR-001, pending the owner's confirmation), but its reader is not
changed: unreachable code cannot run before cutover (FR-020), so the reference goes on the
allowlist with that reason (FR-019). Identify the producer during implementation. If it is in this repository, it moves with the other
packaged builds. If it is outside, raise it with its owner and record it with the sign-off
request.

## R14. Transfer performance (SC-003)

**Decision**: The access check also times a download of a real packaged build from each environment
class, compared with the same download from urm. If any environment, for example a remote Slurm site,
is more than 10% slower end to end on a representative merge-request run, raise it with the program
before cutover (for example, caching or a closer endpoint).

## R15. Pending decisions that do not change this design

- **Release branches (FR-018, awaiting the owner)**: the design is per-branch (R2, R12), so release
  branches keep publishing to and reading from urm until a backport lands. Backports are
  cherry-picks of the main-branch pull requests.
- **No copy of earlier builds (FR-014, confirmed by the owner on 2026-10-16)**: readers reach
  earlier builds through the fallback. The owner noted that coverage-based test selection and build
  reuse could do without them during the transition, and that optimization-profile generation needs
  its input downloads, which the fallback covers. When the optimization profiles move in later
  work, a one-time copy of the current profiles is acceptable in place of a fallback.
- **Packaged-build list (FR-001, awaiting the owner's confirmation)**: if the owner narrows it to
  build-stage outputs, profile generation's canonical publish (off by default,
  `PUBLISH_BOLTED_CANONICAL`) must stay off until it moves. Otherwise readers would get the
  unoptimized package from the new location while the optimized one sits on urm.

## R16. Running every changed path before cutover (FR-020)

**Finding**: a pull request's CI cannot reach four of the changed paths. `L0_MergeRequest.groovy`
turns them on only when the job itself is the post-merge pipeline
(`env.JOB_NAME ==~ /.*PostMerge.*/`), and a pull request runs as `L0_MergeRequest_PR`.
`/bot run --post-merge` only sets `IS_POST_MERGE` (`:247`), which selects post-merge test stages.

| Path | Why a pull request cannot reach it |
|---|---|
| The image job's "Wait For Build Job Complete" stage, and the SBSA release image's wheel download through `get_wheel_from_package.py` (the `docker/Makefile` and `docker/Dockerfile.multi` changes) | The image build is added only for post-merge jobs (`:2960-2962`), with `runSanityCheck` (`:2790`), which turns on the wait stage (`BuildDockerImage.groovy:1044-1048`) |
| BoltProfileGen's input read | BoltProfileGen is launched only for post-merge jobs (`:2604`) |
| Publishing `bolted-<tar>` with its `bolt.ref` property (FR-004), and the test stages' read of that property (`L0_Test.groovy:1746-1773`) | Both follow `BOLT_PUBLISH_VARIANT` (`:338`, and `L0_Test.groovy:6358`), which is on only for post-merge jobs |
| Publishing the optimized package (`publishBoltedCanonical`, `BoltProfileGen.groovy:801-840`) | It runs inside BoltProfileGen, which every pipeline also launches with `boltPublishCanonical: "false"` (`:2625`) |

Asking for the image build from a pull request (`--extra-stage "Build-Docker-Images"`) turns off
that run's `x86_64-Linux` and `SBSA-Linux` sub-pipelines (`:2964-2973`), so the image job has no
packaged build to download, and `runSanityCheck` stays off.

**Decision** (spec FR-020, clarified 2026-10-16):

1. Before cutover, the PR's CI runs (quickstart step 3) cover every changed path that a pull request
   reaches: the build stages, the two uploads through the shared helper, coverage-based test
   selection's read when its coverage audit runs, and the test-stage readers on Kubernetes and
   Slurm. The default `/bot run`, with no options, comes first (clarified 2026-10-16): it is the
   only run that exercises coverage-based test selection (`L0_MergeRequest.groovy:1161-1166`,
   `:1508`) and sets the PR's CI status, which a `--stage-list` run does not
   (`bot-command.yml:55`). Targeted `--stage-list` runs follow for what it did not reach.
   Re-running the same commit covers the reuse copy (R6, quickstart step 6). Each `/bot run` kills
   the PR's running jobs (`.github/workflows/bot-command.yml:50`), so the runs go one after
   another.
2. Post-merge test stages can be added to those runs by exact name, as sanity checks, for example
   to reach a cluster or runtime that no pre-merge stage uses. Only patterns such as `*Post-Merge*`
   need the `ci: post-merge approved` label (`bot-command.yml:55`). In a pull request, these stages
   read packaged builds the same way pre-merge stages do.
3. Before cutover, the identities that the four paths use are checked in the environments they run
   in (quickstart step 2): the publisher role from a build pod and on the BoltProfileGen cluster
   frontend, and the reader role from a Docker-build pod and on the BoltProfileGen cluster
   frontend.
4. The four paths are validated on the first post-merge run after cutover (quickstart step 10), as
   part of the comparison in FR-021. If the new location makes one fail, CI rolls back with
   `LLM_ARTIFACTS_PRIMARY` (R7) while it is fixed; a fault in the code itself needs a fix instead.
   The optimized-package publish is validated when it is first turned on.

**Rationale**: the merge gate stays within what `/bot run` can do. The logic that these paths share
with reachable ones (the read helper, `get_wheel_from_package.py`, and `Build.groovy`'s upload
function) is unit-tested or runs in the PR's runs; what remains is wiring that only the post-merge
pipeline turns on.

**Alternatives considered**:

- A post-merge-mode run of the PR (`/bot run --post-merge`), the previous decision. Rejected: it
  reaches none of the four paths (see the finding).
- Launching the image job by hand on the PR's build with `action=build`, which builds without pushing
  (`BuildDockerImage.groovy:947`), plus `runSanityCheck` and `boltOptimizeWheel`; and launching
  BoltProfileGen with `promote` off. These would reach the paths before cutover, outside `/bot`.
  Declined by clarification.
- A local build of the image's `wheel` stage against the PR's packaged build. Declined by
  clarification.
- `--extra-stage "Build-Docker-Images"`. Rejected: it turns off the run's builds (see the finding).

## R17. Testing the switch against the unmerged library change (FR-025)

**Decision**: While the PR is under test, `jenkins/L0_MergeRequest.groovy:17` loads
`trtllm-jenkins-shared-lib@feature/trtllm-17014-aws-upload-destination`, the library merge
request's branch, instead of `@main`. It is the only
pipeline that needs the opt-in, because the two in-scope uploads through the shared helper are in it
(`:616`, `:668`). Child jobs keep loading `@main`: `Build.groovy` publishes packaged builds with its
own `uploadArtifacts`, and BoltProfileGen uploads with `curl`. Before the PR merges:

1. The library merge request merges, and a `main` run without the PR still publishes everything to urm
   (quickstart prerequisite 5).
2. `L0_MergeRequest.groovy:17` goes back to `@main`.
3. The default pre-merge run passes on that commit. That change is a new commit, so `/bot run` does
   not reuse the earlier run's results, and only the default run sets the PR's CI status.

**Rationale**: Testing starts without waiting for the library review, and `main` never depends on an
unmerged branch (spec FR-025, clarification).

**Checks**:

- Jenkins honors a `@Library('<name>@<branch>')` override only if the library's configuration
  allows overriding the default version. No commit in this repository's history pins a version other
  than `main` in `L0_MergeRequest.groovy` or `L0_Test.groovy`, so the PR's first run shows whether it
  is allowed. If it is not, the PR's testing waits for the library merge.
- This relies on the merge-request job loading `L0_MergeRequest.groovy` from the PR's commit, which
  R12's verification also covers.
- The branch name appears in the PR's public diff, so it must not contain a username (spec FR-008).
  `feature/trtllm-17014-aws-upload-destination` contains none.
- While pinned, the PR tests the merge request's current head. A library change after review means
  re-running the affected checks.

**Alternatives considered**: Waiting for the library merge before testing the PR: slower. Merging the PR
while pinned and moving it back later: rejected by clarification, because `main` would depend on a
branch that can be rebased or deleted.

## R18. Reusing bloom's download code

**Finding** (`bloom-jenkins-shared-lib` at `main`, `9e56d133`; its artifact code is unchanged at
`2d06f0da`, checked 2026-10-16): the download code is
`com.nvidia.bloom.artifacts.ArtifactManager` with `ArtifactManagerBase`, which runs
`resources/artifact_download/artifact_download.py` on a Jenkins agent.

- `ArtifactManager` reads and writes only `artifactory.nvidia.com`, with one credential
  (`ArtifactManager.groovy:15-29`). There is no anonymous read and no per-location credential.
- `artifact_download.py:214-232` tries `READ_URL`, then `WRITE_URL`. It moves to the second after
  any exception, including a rejected token, and sends the same credentials to both.
  `ArtifactManager` passes the same artifactory.nvidia.com URL as both
  (`ArtifactManager.groovy:20-25`), so the second pass repeats the first.
- A missing file is retried 6 times with backoff, 190 seconds of waiting, on each pass (`:32-35`,
  `:150-194`), so one run of the script takes about six minutes to report it. Through
  `ArtifactManager.download` with the default `failNoOp`, bloom's step retry and its three outer
  attempts stack that to about 40 minutes (computed from the code). A rejected token takes the same
  path.
- It needs `python3` and a run-time `pip install` of an internal Artifactory package into the
  caller's container, and that package is published only on urm's package index
  (`ArtifactManagerBase.groovy:101-119`, `:310-421`).
- It reads with `aws-artifactory-credentials`, which is TRT-LLM's publishing identity
  (`ArtifactManager.groovy:20-25`); TRT-LLM's readers use the read-only identity (spec FR-011).
- `probe()` treats every status other than 200, including 401 and 403, as "does not exist"
  (`ArtifactManagerBase.groovy:422-442`).
- From reading the code (not run): `download()` builds a metrics packet whose metadata lookup
  (`Utils.findSeedBranch`) throws outside bloom's own pipelines (`ArtifactManagerBase.groovy:322-338`).
  Constructing `ArtifactManagerBase` directly with metrics turned off avoids that.

**Decision**: Keep `fetch_build_artifact.sh` (R4), and adopt two of bloom's safeguards in every
in-scope download: in the helper, and in BoltProfileGen's parallel download, the one reader that
moves the file with its own code:

- **CDN rejection**: AWS Artifactory redirects downloads to a CloudFront CDN. When a request
  redirected to another host (a `*.cloudfront.net` host in production) gets 403, retry it once with
  `X-JFrog-Download-Redirect-To: None`, which makes Artifactory stream the file itself
  (`artifact_download.py:37-39`, `:81-87`). Bloom matches the host name; matching the redirect
  instead means the same thing in production and can be tested with local servers. If that fails too, it is a transfer failure, not a
  credential failure (spec FR-013, FR-014).
- **Checksum**: compare the downloaded file's SHA-1 with the `checksums.sha1` that the storage API
  reports on the location that served it (`artifact_download.py:110-115` does the same with
  `stat().sha1`). A mismatch is a failed transfer: retried, never a fallback.

**Rationale**: Bloom's mirror loop breaks spec FR-014 (fall back only when the file is absent) and
FR-009 (old-location reads stay anonymous), and costs between six and about 40 minutes for each
absent file. It cannot run in the Slurm batch job (which downloads before any Python environment
exists), inside the image build, on the BoltProfileGen cluster frontend, or inside `artifact.py`,
so the repository's helper is needed whatever the agent-side readers use. The two safeguards
cost one storage-API request per download and a `sha1sum`.

**Alternatives considered** (spec clarification): using bloom's code as is for agent-side reads,
which would relax FR-014; changing bloom to support per-location credentials and 404-only fallback,
which adds a third repository and still leaves a second implementation for the other readers.
Moving the agent-side package downloads onto bloom's code after urm is retired, when the fallback
goes away, stays open for a follow-up.

## R19. Change sets, base commit, and working copy (clarified 2026-10-16)

**Decision**:

- Two change sets. One pull request in this repository carries the helper scripts, their tests,
  and the switch, from `TRTLLM-17014-migrate-builds-to-aws-artifactory`. The shared library's
  change is merge request !111, already open from `feature/trtllm-17014-aws-upload-destination`.
- All work happens in the existing worktree that has that branch checked out; no other worktree
  is created.
- The branch is rebased on `upstream/main` (`dac84b01a6`) before implementation. In
  `91404de7bc..dac84b01a6`, coverage-based test selection moved (#18299): `artifact.py` is now
  `jenkins/scripts/cbts/cbts/command/coverage/selection/artifact.py`, a click command run as
  `PYTHONPATH=jenkins/scripts/cbts python3 -m cbts.command coverage selection artifact`, and its
  tests are `tests/unittest/scripts/cbts/test_coverage_artifact.py`. `build_info_url` and
  `build_commit` still read `build_info.txt` from urm anonymously, and `_cbtsCoverageAudit` still
  runs it twice (`--resolve-pin`, `--prepare`). `L0_MergeRequest.groovy`, `L0_Test.groovy`, and
  `Build.groovy` shifted too.
- The packaged builds stay in scope, including the wheel packed inside each; only wheel files
  uploaded on their own wait for TRTLLMINF-295 (spec FR-001).
- Local checks: `tests/unittest/conftest.py` imports torch and MPI, so the script tests run locally
  with `pytest -c /dev/null --noconftest` and `PYTHONPATH=jenkins/scripts/cbts`. CI runs them
  through `unittest/scripts` as usual. Groovy changes get a parse-only Groovy 2.4 syntax check.

**Rationale**: The helpers have no caller other than the switch, so they are reviewed with it, and
the access check takes its script from the branch on the fork until the PR merges. Writing on the
current `main` avoids a large rebase later in heavily edited pipeline files.

**Alternatives considered**: Separate helper and switch pull requests (the previous plan):
dropped by clarification. Implementing on `91404de7bc` and rebasing at the end: more conflicts,
and the CBTS code would be written against a path that no longer exists.
