# Guardrails

## Before Opening a PR

- Verify no existing open PR for the same bucket or failure signature
- Confirm the PR target repo matches the bucket owner
- Ensure the fix is backed by evidence from logs and code
- PR description must explain why one change covers all jobs in the bucket

## Before Opening an Issue

- Verify no existing open issue or PR for the same bucket
- Confirm the issue target repo is the best home for the bucket
- Issue must explain why no PR was created
- Include enough evidence for another engineer to pick it up
- Use `.github/ISSUE_TEMPLATE/06-bug-report.yml` for TensorRT-LLM failure buckets

## Anti-Patterns

- Do not trust a legacy category without reading logs
- Do not depend on `autodeploy-dashboard` code to resolve pipelines or classify failures
- Do not stop at the first failed bridge if real failures are deeper in the trigger chain
- Do not merge failures just because they mention the same model
- Do not create a PR for a bucket that maps to multiple unrelated fixes
- Do not open PRs for infra-only buckets
- Do not hide uncertainty -- if evidence is mixed, split or skip
