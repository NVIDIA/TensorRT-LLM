# Sign-off Evidence: Migrate Build Artifacts to AWS Artifactory

Evidence for the sign-off request (spec FR-023, SC-008). Each section names the task that fills it.
Record dates, outcomes, and links to run logs only. Never record a token, password, or netrc
content.

## Permission grants (T003, T004)

| Identity | Credential | Requested | Granted | Permissions confirmed |
|---|---|---|---|---|
| Publisher | `aws-artifactory-credentials` | pending | pending | read, deploy, annotate, delete/overwrite; nothing administrative |
| Reader | `trtllm-artifactory-credentials` | pending | pending | read only; no deploy or delete |

## Token expiry and renewal owners (T005)

- Recorded in internal documentation: pending

## Build-info records (T006, FR-017)

- urm `api/build` listing checked: pending
- TRT-LLM build names found: pending ("none", or the names sent to the Artifactory team, with the date)

## urm access records before cutover (T007)

- Requested: pending
- Clients outside TRT-LLM CI: pending

## Child-job code version (T008, research R12)

- Helper jobs load pipeline code at the parent's `gitlabCommit`: pending
- The merge-request job loads `jenkins/L0_MergeRequest.groovy` from the PR's commit: pending

## Source release package producer (T009, research R13)

- Producer: pending
- Outcome: pending (moved with the PR, or raised with its owner)

## Owner's decisions (T010)

- Release branches (FR-018): pending
- No copy of pre-cutover builds (FR-014): confirmed 2026-10-16. Reading them through the fallback
  is preferred; coverage-based test selection and build reuse could do without them during the
  transition, and optimization-profile generation needs its input downloads, which the fallback
  covers.
- Packaged-build list (FR-001): pending

## Access check per environment (T019–T023)

| Environment | Role | Date | Result | Download time, new vs urm |
|---|---|---|---|---|
| Build pod | publisher | pending | pending | n/a |
| BoltProfileGen cluster frontend | publisher, then cleanup | pending | pending | n/a |
| Kubernetes test pod | reader | pending | pending | pending |
| Docker-build pod | reader | pending | pending | pending |
| BoltProfileGen cluster frontend | reader | pending | pending | pending |
| Slurm, ENROOT | reader | pending | pending | pending |
| Slurm, other runtime | reader | pending | pending | pending |
| Build pod | publisher `--cleanup` | pending | pending | n/a |

- Logs and hosts inspected for secrets, including netrc files left behind (T022): pending
- Both identities marked `Verified` (T023): pending

## Library change (T011, T012, T056)

- `9b67bcf` matches the contract, and `UploadArtifactsTest` passes: pending
- Merge request sent for review: pending
- Merged, and a `main` run without the PR still publishes to urm with unchanged links: pending

## Helper scripts (T018)

- Committed first on the PR branch and pushed to the fork: pending

## Pre-merge validation of the PR (T041, T057)

- Default `/bot run`, with download times next to a recent main pre-merge run: pending
- Targeted `--stage-list` runs (Slurm per runtime, sanity stages): pending
- New location 200 and old location 404 for each in-scope file: pending
- Coverage-based test selection's read, if its coverage audit ran: pending
- No netrc left in Slurm job workspaces: pending
- Library pin removed, and the default `/bot run` on the final commit passed: pending

## Reads of pre-cutover builds (T046, SC-004)

- 20 recent and 10 older main post-merge builds read through the fallback, with identical bytes: pending

## Rejected credentials (T047)

- Exit code 3, credential named, urm not contacted: pending

## Coverage-based test selection (T048)

- Same build metadata before and after the change: pending

## Reference inventory and allowlist (T049, T050, T062)

Searched on the PR branch with
`git grep -nE "urm\.nvidia\.com|sw-tensorrt-generic" -- jenkins scripts tests docs docker`
(T050, 2026-10-16), and again after the convergence changes (T073, 2026-10-16). Every match is in
one of the rows below. No in-scope match was left to repoint.

| Reference | Disposition | Reason |
|---|---|---|
| `LLM_ARTIFACTS_LOCATIONS.urm` in `jenkins/Build.groovy`, `jenkins/L0_MergeRequest.groovy` and `jenkins/BoltProfileGen.groovy`; `URM_BASE_URL` in `jenkins/scripts/fetch_build_artifact.sh`; `DEFAULT_URM_URL` in `jenkins/scripts/check_artifactory_access.sh` | Kept (FR-019) | The old location, which the fallback (FR-014) reads and the rollback (FR-022) publishes to |
| The legacy-path mapping: `llmArtifactsLayoutPath` in the pipelines and the path patterns in both helper scripts | Kept (FR-019) | Turns a legacy path into the layout path on either location (research R3) |
| The `ARTIFACT_PATH` and `UPLOAD_PATH` defaults in the five pipelines, `reuseArtifactPath` in `jenkins/L0_MergeRequest.groovy`, the `artifactPath` parameter description in `jenkins/BoltProfileGen.groovy`, `ARTIFACT_BASE` in `artifact.py`, and the `uploadPath` label in `runLLMTestlistOnPlatformImpl` | Kept | Legacy paths are identifiers that the helpers map to each location (research R3). Out-of-scope outputs still publish under them on urm |
| The `Artifactory` server (`urm.nvidia.com`) defined twice in `jenkins/Build.groovy` | Kept (FR-019) | Publishes packaged builds while CI is rolled back, and the out-of-scope uploads |
| `pkgUrl` in `runPackageSanityCheck`, `jenkins/L0_Test.groovy` | Allowlisted | Reads the source release package. Its only call is commented out, so it never runs (research R13) |
| The `--addr` default in `scripts/package_trt_llm.py` | Allowlisted | Names a build published before cutover, which stays on urm. The tool never downloads from it (T049) |
| Test-result, progress-file and rerun-report links and deletes in `jenkins/L0_Test.groovy` and `jenkins/L0_MergeRequest.groovy`; `_URM_BASE` in `tests/integration/defs/perf/perf_regression_utils.py` | Out of scope | Test results move under TRTLLMINF-472 (research R1) |
| Wheel links in `jenkins/L0_Test.groovy` | Out of scope | Wheels uploaded on their own move under TRTLLMINF-295 |
| CBTS coverage archives (`jenkins/L0_MergeRequest.groovy`, and `_URM` in `artifact.py`) and test database (`URM_ARTIFACTORY_BASE` in `jenkins/L0_Test.groovy`); coverage-selection pins (`PIN_BASE` in `artifact.py`); the guardwords scan log; the BOLT profile store (`URM_ARTIFACTORY_BASE` and `PROFILE_PROMOTE_DIR` in `jenkins/BoltProfileGen.groovy`, `scripts/bolt/internal/artifactory.sh`); image-tag maps (`jenkins/scripts/get_image_key_to_tag.py`) | Out of scope | Other llm-artifacts outputs and shared stores |
| Container images and proxies: `DLFW_IMAGE` and `MODEL_EXPRESS_SERVER_IMAGE` in `jenkins/L0_Test.groovy`, the base-image rewrite in `jenkins/BuildDockerImage.groovy`, `packageImage` in `jenkins/L0_MergeRequest.groovy`, the enroot image names in `jenkins/BoltProfileGen.groovy`, and `jenkins/scripts/perf/` (`README.md`, `local/README.md`, `local/configs/example.conf`, `local/run_disagg.sh`) | Out of scope | Container images |
| The package index: the `trt-test-db` install in `jenkins/L0_Test.groovy`, `scripts/check_test_list.py`, `tests/integration/test_lists/test-db/README.md` | Out of scope | Package index |
| `GITHUB_MIRROR` in `jenkins/Build.groovy`, `jenkins/BuildDockerImage.groovy` and `jenkins/L0_Test.groovy` | Out of scope | Source mirror |
| `BOOST_URL` in `docker/common/install_base.sh` | Out of scope | Third-party download |
| 26 golden files under `tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips/` | Out of scope | Provenance records of past runs |
| urm locations and legacy paths in `tests/unittest/scripts/` (fetch helper, access check, Slurm install, wheel extraction, CBTS artifact tests) | Kept | Test fixtures for the fallback, the rollback, and the legacy-path mapping |

## Internal access documentation (T051)

- Written, and followed with a personal token to download a packaged build: pending

## Rollback and reuse rehearsal (T054)

- Switch set to `urm`, and time until it took effect: pending
- Publishing to urm, reads falling back to the new location: pending
- BoltProfileGen cluster frontend follows the switch: pending
- Reuse across the switch rebuilds, then copies: pending

## Cutover (T058)

- Merge commit and time: pending

## After cutover (T059, T060)

- First merge-request and post-merge runs compared with the commit before the merge: pending
- First 20 merge-request and 20 post-merge runs publish only to the new location (SC-001): pending
- Two weeks without failures caused by in-scope artifacts (SC-002): pending
- Median merge-request duration, week after versus week before (SC-003): pending
- First week's logs scanned for exposed secrets (SC-006): pending
- Post-merge-only paths on the first post-merge run: pending

## Release branches (T061, T064)

- Branches without the change still publish to urm only: pending
- Backports, with an NVBug per branch, or "none needed": pending

## Optimized-package publish (T066)

- First run with publishing on: pending

## urm access after cutover (T067)

- Only pre-cutover reads through the fallback: pending

## Sign-off (T063)

- Submitted: pending
- Signed off: pending
