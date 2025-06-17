
# PR title

Please write the PR title by following template:

[JIRA ticket link/nvbug link/github issue link][fix/feat/doc/infra/...] \<summary of this PR\>

For example, assume I have a PR hope to support a new feature about cache manager of Jira TRTLLM-1000 ticket, it would be like

[TRTLLM-1000][feat] Support a new feature about cache manager

## Description

Please explain the issue and the solution in short.

## Test Coverage

<!--
Please list clearly what are the relevant test(s) that can safeguard the changes in the PR. This helps us to ensure we have sufficient test coverage for the PR.
-->

## GitHub Bot Help

`/bot [-h] ['run', 'kill', 'skip', 'reuse-pipeline'] ...`

Provide a user friendly way for developers to interact with a Jenkins server.

Run `/bot [-h|--help]` to print this help message.

See details below for each supported subcommand.

<details>

`run  [--disable-fail-fast --skip-test --stage-list "A10-1, xxx" --gpu-type "A30, H100_PCIe" --test-backend "pytorch, cpp" --add-multi-gpu-test --only-multi-gpu-test --disable-multi-gpu-test --post-merge --extra-stage --debug --detailed-log "H100_PCIe-[Post-Merge]-1, xxx"]`

Launch build/test pipelines. All previously running jobs will be killed.

`--disable-fail-fast ` *(OPTIONAL)* : Disable fail fast on build/tests/infra failures.

`--skip-test ` *(OPTIONAL)* : Skip all test stages, but still run build stages, package stages and sanity check stages. Note: Does **NOT** update GitHub check status.

`--stage-list "A10-1, xxx"` *(OPTIONAL)* : Only run the specified test stages. Examples: "A10-1, xxx". Note: Does **NOT** update GitHub check status.

`--gpu-type "A30, H100_PCIe"` *(OPTIONAL)* : Only run the test stages on the specified GPU types. Examples: "A30, H100_PCIe". Note: Does **NOT** update GitHub check status.

`--test-backend "pytorch, cpp"` *(OPTIONAL)* : Skip test stages which don't match the specified backends. Only support [pytorch, cpp, tensorrt, triton]. Examples: "pytorch, cpp" (does not run test stages with tensorrt backend). Note: Does **NOT** update GitHub pipeline status.

`--only-multi-gpu-test ` *(OPTIONAL)* : Only run the multi-GPU tests. Note: Does **NOT** update GitHub check status.

`--disable-multi-gpu-test ` *(OPTIONAL)* : Disable the multi-GPU tests. Note: Does **NOT** update GitHub check status.

`--add-multi-gpu-test ` *(OPTIONAL)* : Force run the multi-GPU tests. Will also run L0 pre-merge pipeline.

`--post-merge ` *(OPTIONAL)* : Run the L0 post-merge pipeline instead of the ordinary L0 pre-merge pipeline.

`--extra-stage "H100_PCIe-[Post-Merge]-1, xxx"` *(OPTIONAL)* : Run the ordinary L0 pre-merge pipeline and specified test stages. Examples: --extra-stage "H100_PCIe-[Post-Merge]-1, xxx".

`--debug ` *(OPTIONAL)* : Enable access to the Blossom container for debugging purpose. Specify exactly one stage in the `stage-list` parameter to access the appropriate container environment. Note: Does **NOT** update GitHub check status.

`--detailed-log ` *(OPTIONAL)* : Enable flushing out all logs to the Jenkins console. This will significantly increase the log volume and may slow down the job.

For guidance on mapping tests to stage names, see `docs/source/reference/ci-overview.md`.

### kill

`kill  `

Kill all running builds associated with pull request.

### skip

`skip --comment COMMENT `

Skip testing for latest commit on pull request. `--comment "Reason for skipping build/test"` is required. IMPORTANT NOTE: This is dangerous since lack of user care and validation can cause top of tree to break.

### reuse-pipeline

`reuse-pipeline `

Reuse a previous pipeline to validate current commit. This action will also kill all currently running builds associated with the pull request. IMPORTANT NOTE: This is dangerous since lack of user care and validation can cause top of tree to break.

</details>
