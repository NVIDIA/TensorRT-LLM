# create-fixes

## Trigger

After validate-buckets produces validated buckets.

## Behavior

Work one bucket at a time. For each bucket, decide: **actionable** (PR) or **issue-only** (track without PR).

### Actionable Bucket -> PR

1. Choose the smallest code change that plausibly fixes the shared root cause.
2. Prefer a targeted fix over a broad cleanup.
3. Verify with the smallest relevant test or validation step.
4. If validation reveals multiple root causes, split before opening PRs.
5. One branch, one PR per bucket. Never one PR per job.

### Issue-Only Bucket -> Issue

Create an issue when:
- the bucket has a clear shared failure mode with enough log evidence
- a PR is not justified (fix uncertain, risky, external, or infra-related)

Do NOT create an issue when:
- evidence is too weak to explain the failure
- an open issue/PR already covers the same bucket
- the bucket duplicates another

For `TensorRT-LLM` issues, use `.github/ISSUE_TEMPLATE/06-bug-report.yml`. Include: pipeline ID, representative job URL, first causal failure, matching jobs, likely owner, code hypothesis, and why no PR was created.

### Repo Targeting

See `references/repo-ownership.md` for which repo owns which failures.

### PR Body Template

```markdown
## Summary
- Fixes root-cause bucket: `<repo/component/failure-mode>`
- Resolves failures from pipeline `<pipeline_id>`
- One change covers `<N>` matching jobs because `<shared-cause>`

## Evidence
- Representative job: `<job_url>`
- Representative log snippet: `<first causal failure>`
- Matching jobs: `<count>` across `<models/workloads>`
- Bucket rule: `<why these failures belong together>`

## Validation
- `<focused test or verification step>`

## Not Included
- `<skipped infra-only or mixed-evidence buckets>`
```

### Guardrails

See `references/guardrails.md` for PR/issue pre-checks.

## Output

Created PRs and/or issues, or explicit explanation of why none were created.
