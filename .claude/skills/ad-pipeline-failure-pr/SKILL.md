---
name: ad-pipeline-failure-pr
description: >
  Analyze the latest AutoDeploy pipeline or a user-specified pipeline ID, inspect failed job logs,
  group similar failures into actionable root-cause buckets, and create at most one PR per bucket.
  Use when the user mentions pipeline IDs, failed jobs, GitLab logs, failure buckets, or opening
  PRs from CI failures.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Pipeline Failure PR

**Input:** latest AutoDeploy `model-coverage` GitLab pipeline, or a specific upstream/downstream pipeline ID / pipeline URL. **Auth requirement:** the user must export a GitLab token in `GITLAB_TOKEN` before this skill can query pipelines, jobs, or traces. **Output:** first ask the user which output format is preferred. Default to reporting in chat. Alternative outputs are a Markdown report (`md`) and a per-failure CSV (`csv`). The skill still produces a bucketed failure report plus at most one PR per actionable root-cause bucket, and when a PR is not justified but the bucket is still worth tracking, create one issue for that bucket.

## Core Rule

This skill must be standalone. Resolve pipelines, failed jobs, and raw logs directly from GitLab APIs and job traces. Do **not** depend on `autodeploy-dashboard` code, scripts, CSVs, or its legacy categorization logic. This skill owns the bucketing rules, skip rules, repo ownership decision, and one-PR-per-bucket behavior.

Before any GitLab API call, require `GITLAB_TOKEN` to be set in the environment. If it is missing, stop immediately and tell the user: `Set GITLAB_TOKEN to a GitLab personal access token and rerun this skill.`

Before doing the main analysis, ask the user which output is preferred:
- `chat` (default)
- `md`
- `csv`

If the user does not specify, default to `chat`.

## Phase 0 — Resolve Scope

1. Default scope is `model-coverage`. Do not silently switch to benchmark pipelines.
2. If the user explicitly asks to analyze a benchmark pipeline, stop and tell them this skill does not support benchmark pipelines.
3. If the user gives a pipeline ID or GitLab pipeline URL, use it.
4. Treat a user-provided pipeline as potentially either:
   - an upstream AutoDeploy pipeline in `ftp/infra/autodeploy-dashboard`
   - a downstream triggered pipeline in `dl/jet/ci`
5. If the starting pipeline is upstream, follow the failed bridge chain until you reach the first downstream pipeline with terminal `model-coverage` jobs.
6. Otherwise resolve the latest upstream AutoDeploy pipeline that ran `model-coverage`, then follow the same bridge chain to the terminal pipeline.
7. If `GITLAB_TOKEN` is missing, stop immediately and tell the user exactly how to fix it: `Set GITLAB_TOKEN to a GitLab personal access token and rerun this skill.`

## Pipeline Resolution Rules

Use this resolution order:
1. Identify whether the provided pipeline belongs to the upstream dashboard project or the downstream `dl/jet/ci` project.
2. If it is upstream, inspect its bridge jobs and select the failed `model-coverage` trigger path.
3. If the next pipeline contains only bridge jobs, keep following the failed trigger chain.
4. Stop at the first downstream pipeline that contains terminal failed `model-coverage` jobs with traces.
5. Report both:
   - the user-facing starting pipeline
   - the terminal pipeline that contains the actual failing jobs

Do not analyze only the bridge failure if a deeper downstream pipeline contains the real job traces.

All GitLab API and trace-fetching steps in this skill must authenticate with the token from `GITLAB_TOKEN`.

## Phase 1 — Gather Failure Evidence

For each failed job, collect:
- pipeline ID
- job ID and job URL
- raw log URL
- workload name
- model or benchmark configuration
- first causal error snippet from the raw trace

Also collect:
- starting pipeline ID
- terminal pipeline ID
- whether the job came from a bridge-followed downstream path

Before proposing a fix, read at least one representative raw log for every tentative bucket. Do not rely on legacy labels alone.

Trace-reading rules:
- In `model-coverage` terminal pipelines, jobs often come in triplets like `[1 logs_before]`, `[2 <runner/stage>]`, `[3 logs_after]`. The primary failing workload is usually the `[2 ...]` job. Use `[1]` and `[3]` only as supplemental evidence when needed.
- If the trace ends with generic wrapper failures such as `RuntimeError: Executor worker returned error`, `RuntimeError: Executor worker died during initialization`, or `ERROR: Job failed: Process exited with status 1`, keep scanning upward and record the earlier model-, export-, tokenizer-, or environment-specific exception instead.
- Prefer the first specific exception that explains the failure over later fallout from worker teardown, Slurm cleanup, or proxy startup.
- When the workload dumps its config in the trace, capture the resolved `model:` value and relevant `yaml_extra`/runtime hints. They are often useful for explaining why a bucket is multimodal, world-size-specific, or using a special mode.

## Skill-Owned Bucket Rules

Every analyzed failed job must end up in exactly one bucket. Do **not** leave failures in an implicit catch-all like `other`, `misc`, or `untriaged` in the final report.

This includes infra and external cases. They still need explicit buckets, for example:
- `infra/resource/oom`
- `infra/runtime/timeout-or-freeze`
- `infra/runtime/cancelled`
- `infra/filesystem/hf-lock-permission`
- `external/huggingface/access-forbidden`
- `external/huggingface/missing-revision`
- `external/huggingface/invalid-tokenizer-or-processor`
- `external/env/missing-python-package`
- `external/transformers/api-mismatch`

Do **not** assume `oom` or `timeout-or-freeze` are infra-only. In AutoDeploy pipelines they often reflect real `TensorRT-LLM` / AutoDeploy bugs. Classify them as `infra/...` only when the evidence points to cluster noise or a non-code resource problem. Otherwise bucket them under the real owning repo/component.

Group failures together only when all of these are true:
- they point to the same likely code owner and target repo
- they share the same causal failure signature, such as the same failing symbol, op, assertion, stack frame, or config path
- they appear fixable by one coherent code change
- one PR can reasonably explain why the same fix covers every matched job

Split failures into different buckets when any of these are true:
- the first causal error differs even if the legacy category matches
- the same symptom comes from different repos or subsystems
- one failure is infrastructure noise and the other is a code bug
- the likely fixes would touch unrelated files or require different validation
- the evidence is mixed or contradictory

When uncertain, split instead of merge.

If a failed job does not fit any existing bucket, put it in its own one-job bucket.
Do not leave it uncategorized.

That one-job bucket must still be labeled as exactly one of:
- `actionable` — likely fixable with a PR
- `issue-only` — worth tracking, but not ready for a PR

Do not use a `skip PR` label. If a bucket should not produce a PR, mark it `issue-only` when it is still worth tracking.

Buckets such as OOM, timeout/freeze, cancelled, or Hugging Face access failures must still appear explicitly in the report. If the shared failure mode is clear enough to track, prefer `issue-only`.

The final report must account for **all** failed jobs:
- include the total failed job count
- include bucket counts
- ensure the sum of all bucket sizes equals the total failed job count
- make unmatched or low-confidence cases explicit as singleton buckets instead of hiding them

Use this evidence priority order when bucketing:
1. first causal stack frame or assertion
2. explicit failing symbol, op, layer, config key, or script
3. repeated error snippet near the first failure
4. repeated failure wording across matched traces
5. job naming and workload metadata only as a weak tie-breaker

Each bucket must have:
- a short bucket name in the form `repo/component/failure-mode`
- one representative job
- a list of all matching jobs
- one root-cause hypothesis tied to code

## Skip Rules

Do **not** create a PR for a bucket when any of these are true:
- the failures are pure infrastructure noise such as timeout, preemption, cluster cancellation, or log-access failure without code evidence
- the jobs do not share one plausible code fix
- the evidence is too weak to point at a concrete code path
- the issue belongs to external infrastructure or an external dependency outside the checked-out repos
- an open PR already appears to address the same bucket
- the only commonality is a broad status label or superficial wording

If the starting pipeline failed only because a bridge failed, do not treat the bridge as its own actionable bucket unless the downstream terminal pipeline has no failing jobs or no accessible traces.

Infrastructure and external buckets must still be reported as explicit buckets. They should usually be `issue-only` rather than promoted to a PR unless the evidence clearly points to a repo-owned fix.

Common `issue-only` patterns seen in AutoDeploy model-coverage pipelines:
- gated or forbidden Hugging Face repos (`403`)
- missing or renamed Hugging Face revisions/models (`404`)
- missing optional Python packages such as `timm`, `num2words`, `mamba_ssm`, `causal_conv1d`, or similar runtime dependencies
- filesystem permission problems on Hugging Face cache lock files
- only clearly non-code resource failures after log review; do not auto-classify CUDA OOM or timeout/freeze as infra without checking for an AutoDeploy root cause

## Repo Ownership Rules

Prefer `TensorRT-LLM` when the root cause is in:
- AutoDeploy model code
- AutoDeploy runtime or transforms
- tests, configs, or execution paths owned by `TensorRT-LLM`
- code paths surfaced by `ad-debug-agent`

Prefer `autodeploy-dashboard` when the root cause is in:
- failure-analysis scripts
- workload generation
- job URL or raw-log resolution
- pipeline orchestration or reporting gaps in the AutoDeploy pipeline repo

Do not open a PR when the bucket belongs to cluster infrastructure, GitLab service behavior, or another external system that is not owned by the checked-out repos.

## Phase 2 — Validate Each Bucket

For every bucket:
1. Read the representative job log and isolate the first causal failure, not the downstream fallout.
2. Read the relevant code, config, or script that the failure points to.
3. Confirm that the same hypothesis explains the other jobs in the bucket.
4. If deeper AutoDeploy tracing is needed, use the `ad-debug-agent` workflow to inspect the failing code path before editing.
5. If the representative log does not actually support the bucket hypothesis, split or discard the bucket.

Do not start coding until the bucket has both:
- one representative log snippet
- one code-level hypothesis

## Phase 3 — Create At Most One Fix Per Bucket

Work one bucket at a time.

For an actionable bucket:
1. Choose the smallest code change that plausibly fixes the shared root cause.
2. Prefer a targeted fix over a broad cleanup.
3. Verify with the smallest relevant test or validation step.
4. If the validation suggests the bucket actually contains multiple root causes, split it before opening any PRs.
5. Create one branch and one PR for the full bucket.

Never open one PR per failed job when the jobs share the same fix.

## Phase 3b — Create One Issue When No PR Is Available

If a bucket is worth tracking, but you do **not** have enough confidence for a PR, create one issue for that bucket instead of silently stopping.

Create an issue when all of these are true:
- the bucket has a clear shared failure mode
- the representative logs provide enough evidence to explain the bucket
- one issue can clearly describe the shared failure mode
- a PR is not justified yet because the fix is uncertain, risky, mixed, under-validated, external, or infra-related

Do **not** create an issue when any of these are true:
- the evidence is too weak to explain the failure mode at all
- an open issue or PR already appears to cover the same bucket
- the bucket is just a duplicate restatement of another bucket

Issues for infra or external buckets are valid. Examples include:
- `infra/resource/oom`
- `infra/runtime/timeout-or-freeze`
- `infra/runtime/cancelled`
- `external/huggingface/access-forbidden`
- `external/huggingface/missing-revision`
- `external/env/missing-python-package`

For `oom` and `timeout-or-freeze`, prefer a repo-owned bucket instead when the traces suggest a reproducible AutoDeploy issue rather than infrastructure noise.

When creating an issue in `TensorRT-LLM`, use the repository templates in `.github/ISSUE_TEMPLATE/` instead of inventing a custom issue body.
- For failure buckets from this skill, use `.github/ISSUE_TEMPLATE/06-bug-report.yml` by default.
- Only use another template if the bucket is clearly a feature request or another non-bug category.

Fill the selected issue template with the triage evidence from this skill. At minimum, include:
- pipeline ID and workload scope
- representative job URL
- first causal failure snippet
- matching jobs or affected model families
- likely owner or subsystem when known
- code-level hypothesis when applicable
- why a PR was not created yet

Respect the template's required structure and security guidance. Do not paste sensitive tokens, private credentials, or other secrets into the issue body.

Prefer one issue per bucket, not one issue per job.

## PR Guardrails

Before opening a PR:
- verify there is no existing open PR for the same bucket or failure signature
- confirm the PR target repo matches the bucket owner
- ensure the proposed fix is backed by evidence from logs and code
- make sure the PR description explains why one change covers all jobs in the bucket

For `TensorRT-LLM` PRs, follow the repo workflow:
- use the local PR title format: `[JIRA/NVBUG/None][type] description`
- keep the PR focused on one concern
- validate only the smallest relevant tests or commands

## Issue Guardrails

Before opening an issue:
- verify there is no existing open issue or PR for the same bucket or failure signature
- confirm the issue target repo is the best available home for the bucket
- make sure the issue explains why no PR was created
- include enough evidence that another engineer can pick it up without redoing the initial triage
- use the appropriate file from `.github/ISSUE_TEMPLATE/`, usually `06-bug-report.yml` for failure buckets from this skill

## PR Body Template

Use this structure:

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

## Phase 4 — Final Report

Print a concise final report with:
1. target pipeline, terminal pipeline, and workload scope
2. all buckets with status such as `actionable` or `issue-only`
3. representative evidence for each actionable bucket
4. PRs created, issues created, or why no PR was created for an `issue-only` bucket
5. remaining risks or follow-up validation

The final report must also include a bucketization checksum:
- `total failed jobs = <N>`
- `sum of bucket sizes = <N>`

If no PRs or issues were created, say that explicitly and explain whether the blocker was:
- duplicate-checks not yet performed
- evidence too weak for a concrete code owner
- no coherent single fix
- external or infra ownership

Honor the user's selected output format:
- `chat`: print the final report directly in chat
- `md`: also write the final report to a Markdown file
- `csv`: also write a per-failure CSV with one row per failed job, including at least job ID, job URL, workload/model, first causal error, bucket, likely owner, and outcome

## Anti-Patterns

- Do not trust a legacy category without reading logs.
- Do not depend on `autodeploy-dashboard` code to resolve pipelines or classify failures.
- Do not stop at the first failed bridge if the real `model-coverage` failures are deeper in the downstream trigger chain.
- Do not merge failures just because they mention the same model.
- Do not create a PR for a bucket that maps to multiple unrelated fixes.
- Do not open PRs for infra-only buckets.
- Do not hide uncertainty; if evidence is mixed, split or skip.
