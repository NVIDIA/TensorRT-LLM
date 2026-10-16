# Quickstart: Validating the Build Artifact Migration

How to prove the migration works, end to end. Behavior is defined in
[contracts/artifact-locations.md](contracts/artifact-locations.md) and the script interfaces in
[contracts/helper-scripts.md](contracts/helper-scripts.md). Placeholders: `<pipeline>` is a Jenkins
`JOB_NAME` such as `LLM/main/L0_PostMerge`, and `<build>` is a build number.

## Prerequisites

1. Both identities are `Permitted` (see [data-model.md](data-model.md#identity-readiness)). The
   Artifactory program has granted the publisher exactly read, deploy, annotate, and
   delete/overwrite on `sw-tensorrt-llm-generic-local`, with nothing administrative, and granted
   the reader read only.
2. Jenkins holds `aws-artifactory-credentials` and `trtllm-artifactory-credentials`.
3. For manual checks, you have a personal Artifactory access token in a netrc file:

   ```bash
   install -m 600 /dev/null ~/.netrc-artifactory
   # Add one line, using your own token:
   #   machine artifactory.nvidia.com login <user> password <token>
   export ARTIFACTORY_NETRC=~/.netrc-artifactory
   ```

4. Build-info check (FR-017). List urm build records with the urm credential, and note any build names
   that belong to TRT-LLM pipelines:

   ```bash
   curl -fsS --netrc-file <urm-netrc> https://urm.nvidia.com/artifactory/api/build
   ```

   Expected: no TRT-LLM pipeline names. Otherwise, send the exact names to the Artifactory team
   before cutover.

5. Before the switch merges, not before testing it (research R17): the shared helper's opt-in has
   merged on the library's `main` (FR-025), and a CI run of `main` without the switch still
   publishes everything to urm:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" \
       https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/<pipeline>/<build>/build_info.txt
   ```

   Expected: `200`, and the run's log links are the same as before the library change.

## 1. Unit tests (helpers and changed scripts)

```bash
pytest tests/unittest/scripts/test_fetch_build_artifact.py \
       tests/unittest/scripts/test_check_artifactory_access.py \
       tests/unittest/scripts/test_slurm_install.py \
       tests/unittest/scripts/cbts/test_coverage_artifact.py \
       tests/unittest/scripts/test_get_wheel_from_package.py
```

Expected: all pass with no network access. These modules also run in CI through the
`unittest/scripts` entries in `l0_cpu.yml` and `l0_a10.yml`. The `fetch_build_artifact.sh` tests
include a CDN 403 retried without the redirect and a checksum mismatch (research R18). The read rule
is tested only there: `test_slurm_install.py` and `test_coverage_artifact.py` check that those
callers run the helper and handle its exit codes (spec FR-005).

## 2. Access check in every environment (User Story 1, FR-020, SC-006)

The script reaches `main` only with the PR, so each run takes it from the PR branch's commit on
the fork. Run the publisher role first, from a build pod with the publisher credential:

```bash
jenkins/scripts/check_artifactory_access.sh --role publisher \
    --probe-dir llm-artifacts/access-check/<date>-<agent>
```

Run the publisher role again on the BoltProfileGen cluster frontend, where the optimized package is
published, with its own probe directory, then at once with `--cleanup` there. Feed the netrc over
stdin and remove it on exit, as `publishBoltedCanonical` does (research R16).

Then run the reader role with the reader credential, against the build pod's probe directory, from
a Kubernetes test pod, a Docker-build pod (where the image job's wait step and image build run), the
BoltProfileGen cluster frontend (where its input read runs), and one node on each Slurm cluster:

```bash
jenkins/scripts/check_artifactory_access.sh --role reader \
    --probe-dir llm-artifacts/access-check/<date>-<agent> \
    --time sw-tensorrt-generic/llm-artifacts/<pipeline>/<build>/TensorRT-LLM.tar.gz
```

When every reader check has run, delete the probe with the publisher credential:

```bash
jenkins/scripts/check_artifactory_access.sh --role publisher --cleanup \
    --probe-dir llm-artifacts/access-check/<date>-<agent>
```

Expected:

- Every line is `PASS`. The reader's anonymous read returns 401, and its write attempt returns 403.
- Timing lines show the new location within 10% of urm for each environment. Otherwise, raise it
  with the program before cutover (research R14).
- No token appears in the job logs, and no netrc file is left on the hosts.

## 3. CI runs of the switch PR (User Story 2, FR-020)

While the library merge request is open, the PR's `jenkins/L0_MergeRequest.groovy:17` loads
`trtllm-jenkins-shared-lib@feature/trtllm-17014-aws-upload-destination` (research R17). If Jenkins
refuses the override, the run fails while loading the library, and testing waits for the library
merge.

Ask a member of `NVIDIA/trt-llm-ci-approvers` to apply `ci: full pre-merge approved`; the first
run need not wait for it. The PR changes `jenkins/L0_Test.groovy`, so CI adds the multi-GPU stages
to the default run. CI checks for the label when each architecture's single-GPU tests finish: if it
is there, the same run goes on to the multi-GPU stages; if not, they are blocked, and a later
`/bot run` on the same commit reuses the build and the stages that passed. Push nothing until the
multi-GPU stages have run, since a new commit ends that reuse.

First comment `/bot run`, with no options. It runs the pre-merge stages that apply and
coverage-based test selection, and it is the run that sets the PR's CI status. Note how long its
stages take to download the packaged build, next to a recent main pre-merge run (research R14).

When it ends, find the stages that cover each changed path it did not reach, such as a Slurm
runtime, and run only those:

```bash
python scripts/test_to_stage_mapping.py --tests "<a test from each stage type>"
```

```text
/bot run --stage-list "<one Slurm test stage per runtime>"
```

Each `/bot run` kills the PR's running jobs, so start the next one only after the previous run
finishes. A stage in the multi-GPU sub-pipeline needs the `ci: full pre-merge approved` label.

Then check each run's build directory (`<pipeline>` and `<build>` of that run):

```bash
NEW=https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/<pipeline>/<build>
OLD=https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/<pipeline>/<build>
for f in build_info.txt waive_list/waives.txt TensorRT-LLM.tar.gz; do
  curl -s -o /dev/null -w "new $f %{http_code}\n" --netrc-file "$ARTIFACTORY_NETRC" "$NEW/$f"
  curl -s -o /dev/null -w "old $f %{http_code}\n" "$OLD/$f"
done
```

Expected:

- `new ... 200` and `old ... 404` for every in-scope file.
- Every stage passes as it did on the pre-change baseline.
- Stage logs show the helper's `served-by=aws`, including each Slurm job's output, where
  `slurm_install.sh` runs the helper on the node.
- The build-info and waive-list links in the log point at the new location, and test stages log
  `Download merged waives.txt successfully`. Those steps log a failure and carry on, so a passing
  run alone does not show that they worked.
- Test-result links still point at urm.
- If coverage-based test selection ran its coverage audit (`CBTS coverage pilot:` in the log), the
  log shows the helper's `served-by=` line for `build_info.txt` and no credential message.
- After the jobs end, no netrc file is left in the Slurm job workspaces (FR-012).

### Optional sanity stages

To reach a cluster or runtime that no pre-merge stage uses, add a post-merge test stage by its exact
name, for example `--stage-list "<a Post-Merge stage name>"`. Exact names need no
label; patterns such as `*Post-Merge*` need the `ci: post-merge approved` label. In a pull request,
these stages read packaged builds the same way pre-merge stages do. Neither they nor `--post-merge`
reach the image build or BoltProfileGen, which step 10 covers (research R16).

## 4. Reading builds published before cutover (User Story 3, FR-014, SC-004)

For the 20 most recent main post-merge builds before cutover, plus 10 older retained ones:

```bash
jenkins/scripts/fetch_build_artifact.sh \
    sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/<build>/build_info.txt /tmp/bi.txt
echo "exit=$?"
curl -fsS https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/<build>/build_info.txt \
    | cmp - /tmp/bi.txt && echo same
```

Repeat for a packaged build and for `waive_list/waives.txt`.

Expected: `exit=0`, `served-by=urm` on stderr, and `same`.

## 5. Rejected credentials do not fall back (FR-014, FR-013)

Point `ARTIFACTORY_NETRC` at a netrc that holds an invalid token, then fetch any artifact.

Expected:

- Exit code 3.
- The message names the credential and says its token needs renewal. It names no internal lists,
  portals, or channels.
- No `served-by=` line, so urm was never contacted.

## 6. Reuse across a location change (FR-016)

The bot reuses a build only when the same commit runs again, and a run's code comes from its commit,
so a run with the change never meets a build made without it (research R6). The rebuild case happens
when the location switch changes between two runs of one commit. Run this during step 8:

1. With `LLM_ARTIFACTS_PRIMARY=urm`, run the PR's build stages. They publish to urm.
2. Unset the variable, and run the same commit again. Expected: the build stage's log says
   `reuse source is on urm; rebuilding`, and the stage rebuilds and publishes to the new location.
3. Run the same commit once more. Expected: the server-side copy on the new location succeeds, and
   no rebuild happens.

## 7. Coverage-based test selection (FR-015, User Story 3 scenario 2)

Coverage-based test selection reads `build_info.txt` of main post-merge builds, which stay on urm
until cutover, and coverage archives, which stay on urm. From a checkout of the PR, with
`ARTIFACTORY_NETRC` set as in prerequisite 3:

```bash
PYTHONPATH=jenkins/scripts/cbts python3 -m cbts.command coverage selection artifact --resolve-build \
    --build <a recent main post-merge build> --pr-head <a main commit>
```

Run the same command from a checkout of the commit before the change.

Expected: both print the same build metadata, and the new run logs that urm served
`build_info.txt`. After cutover, the first main post-merge build that contains the change is served
by the new location (step 10).

## 8. Rollback rehearsal (FR-022, SC-009)

With a Jenkins administrator, set `LLM_ARTIFACTS_PRIMARY=urm` on every controller that runs TRT-LLM
pipelines, and run one build stage, one Kubernetes test stage, and one Slurm test stage on the PR.
Before cutover, only the PR's runs read the variable. Run step 6 during the rehearsal.

Expected:

- The build and test jobs log `Build artifacts: primary location urm, passed by the parent
  pipeline`, whichever controller they run on.
- In-scope artifacts publish to urm, including `build_info.txt` and `waive_list/` through the
  shared helper.
- Reads try urm first and fall back to the new location for files published there.
- The Slurm stage's launch script, printed in its log, exports `LLM_ARTIFACTS_PRIMARY=urm`, so the
  helper on the node reads urm first.
- On the BoltProfileGen cluster frontend, the helper, run with the settings that BoltProfileGen's
  cluster script exports, reports `served-by=urm` for a file published during the rehearsal, so
  scripts on other hosts follow the switch.
- Time from the decision to the first run taking effect is under one hour.
- Unset the variable afterwards.

## 9. Branch isolation (FR-018)

After the switch merges on main, run CI on a release branch that has not adopted the change.
Expected: its in-scope artifacts appear only on urm, and it passes as before.

## 10. After cutover (FR-020, FR-021, SC-001, SC-002, SC-003, SC-007)

- **First runs**: compare the first merge-request run and the first post-merge run that contain the
  change with the corresponding runs of the commit just before the merge. Expected: the same
  artifact set (names and layout) and the same stage outcomes, apart from where in-scope artifacts
  are published.
- **Paths that only the post-merge pipeline runs** (FR-020, research R16), in that first post-merge
  run:
  - The image job's "Wait For Build Job Complete" stage finds both packaged builds, and its log
    shows `served-by=aws`.
  - The SBSA release image's build log shows the wheel download with `served-by=aws`.
  - BoltProfileGen's input read logs `served-by=aws`, and no netrc file is left on its cluster
    frontend. Its log shows `Build artifacts: primary location aws, passed by the parent
    pipeline`; without that line, the job used its own controller's setting (research R7).
  - `bolted-<tar>` carries the same property keys as on a pre-change post-merge build (FR-004):

    ```bash
    jenkins/scripts/fetch_build_artifact.sh --properties \
        sw-tensorrt-generic/llm-artifacts/<pipeline>/<build>/bolted-<tar> /tmp/props.json
    ```

  - The SBSA test stages' `bolt.ref` check reads that property with `served-by=aws`.
  - If the new location makes one of these fail, roll back with `LLM_ARTIFACTS_PRIMARY=urm` (step 8)
    while it is fixed. The optimized-package publish is checked when it is first turned on.
- **First two weeks**: no run fails because an in-scope artifact could not be published or fetched.
- **First week**: the median merge-request pipeline duration is within 10% of the week before.
- **Reference search** (SC-007): every remaining match is either out of scope or on the allowlist with
  a reason:

  ```bash
  git grep -nE "urm\.nvidia\.com|sw-tensorrt-generic" -- jenkins scripts tests docs docker
  ```

## 11. Before merging the switch (FR-024)

The pull request and the shared library merge request have been sent for review to all TRT-LLM
infrastructure developers, stating the new location and that an access token is needed. Neither
names internal lists or portals (research R10, R11).

Then, in order (FR-025, research R17):

1. The library merge request has merged, and prerequisite 5 passes.
2. `jenkins/L0_MergeRequest.groovy:17` loads `trtllm-jenkins-shared-lib@main` again.
3. The default pre-merge run (`/bot run`) on that commit passes.
