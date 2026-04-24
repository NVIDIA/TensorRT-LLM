---
name: ad-pipeline-failure-pr
description: Analyze AutoDeploy pipeline failures, inspect failed job logs, group
  failures into root-cause buckets, and create PRs or issues per bucket. Trigger on
  "pipeline failure", "failed jobs", "CI failures", "GitLab pipeline", "failure buckets",
  "model-coverage failures", or AutoDeploy pipeline IDs/URLs.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Pipeline Failure PR

**Input:** latest AutoDeploy `model-coverage` GitLab pipeline, or a specific pipeline ID/URL.
**Output:** bucketed failure report + at most one PR per actionable bucket + one issue per trackable non-PR bucket.

Before starting, ask the user's preferred output format: `chat` (default), `md`, or `csv`.

## Auth Requirement

`GITLAB_TOKEN` must be set. If missing, stop immediately: `Set GITLAB_TOKEN to a GitLab personal access token and rerun this skill.`

## Core Rule

This skill is standalone. Resolve pipelines, jobs, and logs directly from GitLab APIs. Do NOT depend on `autodeploy-dashboard` code, scripts, CSVs, or legacy categorization. This skill owns bucketing rules, skip rules, repo ownership, and one-PR-per-bucket behavior.

## Workflow

    resolve-scope → gather-evidence → validate-buckets → create-fixes → report

| Stage | Produces | Key action |
|-------|----------|------------|
| resolve-scope | terminal pipeline ID | Follow bridge chain to model-coverage jobs |
| gather-evidence | per-job failure evidence | Read traces, extract first causal error |
| validate-buckets | validated bucket list | Confirm hypothesis against logs + code |
| create-fixes | PRs and/or issues | One PR per actionable bucket, one issue per trackable bucket |
| report | final report | Bucketed summary with checksum |

## Cross-Stage Rules

- Every failed job must end in exactly one bucket -- no catch-all `other`/`misc`/`untriaged`
- When uncertain, split buckets instead of merging
- Read at least one representative raw log before any bucket hypothesis
- Do not start coding until a bucket has both a log snippet and a code-level hypothesis
- Final report must include checksum: `total failed jobs = sum of bucket sizes`

## References

- `references/bucket-rules.md` -- bucketing criteria, evidence priority, skip rules, infra/external patterns
- `references/repo-ownership.md` -- which repo owns which failures, PR conventions
- `references/guardrails.md` -- PR/issue pre-checks + anti-patterns

## Boundaries

- Scope: `model-coverage` only. Does NOT support benchmark pipelines.
- Does NOT depend on `autodeploy-dashboard` for classification.
- Does NOT merge failures just because they mention the same model.
- Does NOT create PRs for infra-only or mixed-evidence buckets.
