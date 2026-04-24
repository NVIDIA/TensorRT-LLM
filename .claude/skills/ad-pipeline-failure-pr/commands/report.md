# report

## Trigger

After create-fixes completes (or determines no fixes needed).

## Behavior

Print a concise final report with:
1. Target pipeline, terminal pipeline, and workload scope
2. All buckets with status (`actionable` or `issue-only`)
3. Representative evidence for each actionable bucket
4. PRs created, issues created, or why none were created
5. Remaining risks or follow-up validation

## Bucketing Checksum

Every report must include:
- `total failed jobs = <N>`
- `sum of bucket sizes = <N>`

These must be equal. If they differ, fix the bucketing before reporting.

## Output Format

Honor the user's selected format:
- `chat`: print directly in chat
- `md`: also write to a Markdown file
- `csv`: also write a per-failure CSV (one row per failed job: job ID, job URL, workload/model, first causal error, bucket, likely owner, outcome)

## No-Action Report

If no PRs or issues were created, explain whether the blocker was:
- duplicate-checks not yet performed
- evidence too weak for a code owner
- no coherent single fix
- external or infra ownership
