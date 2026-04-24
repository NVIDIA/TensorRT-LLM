# Repo Ownership Rules

## TensorRT-LLM

Prefer `TensorRT-LLM` when the root cause is in:
- AutoDeploy model code
- AutoDeploy runtime or transforms
- tests, configs, or execution paths owned by TensorRT-LLM
- code paths surfaced by `ad-debug-agent`

For `TensorRT-LLM` PRs:
- use the PR title format: `[JIRA/NVBUG/None][type] description`
- keep the PR focused on one concern
- validate only the smallest relevant tests or commands

## autodeploy-dashboard

Prefer `autodeploy-dashboard` when the root cause is in:
- failure-analysis scripts
- workload generation
- job URL or raw-log resolution
- pipeline orchestration or reporting gaps

## No PR Target

Do not open a PR when the bucket belongs to cluster infrastructure, GitLab service behavior, or another external system not owned by the checked-out repos.
