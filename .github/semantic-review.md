<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Semantic conflict review

**Disabled:** The workflow has no scheduled or comment triggers, and its discovery
and publication jobs are unconditionally skipped, including on manual dispatch
and reruns using this configuration. It no longer sends CodeRabbit requests as
`trtllm-agent`, processes replies, or tidies discussion comments. CodeRabbit
automatic chat replies are disabled separately in `.coderabbit.yaml`; humans
can still explicitly mention CodeRabbit. Its review/chat instructions also tell
it to ignore automated reviewers and avoid tagging them or learning from them.

The sections below describe the retained implementation for historical reference.

The workflow asks CodeRabbit to inspect the combined behavior of a PR and its
target branch. It publishes a commit status named
**Semantic conflict with target branch / PR #N** on the requested head commit.
Keep this status and its workflow **non-required** in branch protection/rulesets.
PASS, FAIL and INCONCLUSIVE are all best-effort AI judgments that can be incomplete
or incorrect. Authors and reviewers should independently verify the evidence and
relevant behavior. Their advisory results do not, by themselves, block GitHub
merging; required CI, approvals and other merge rules still apply. A confirmed
defect should not be ignored because the status is non-required.

## Analysis scope

The prompt checks behavioral compatibility involving the PR changes, including
when head already contains target. It asks CodeRabbit to distinguish cross-branch
interactions from PR-local compatibility defects and to verify how both sides'
edits combine. An alleged caller/definition mismatch must remain after three-way
merge analysis before it can support FAIL. Unresolved relevant
textual conflicts must not be replaced with an assumed resolution.

FAIL needs a supported trigger, a reachable failure path and evidence that paired
edits, feature gates or recovery logic do not prevent the problem. Unrelated
pre-existing defects, missing tests alone and wording-only improvements do not
support FAIL. Missing material evidence supports INCONCLUSIVE. Each reply must
include the advisory notice and put its fixed-revision citations inside the
`SEMANTIC_REVIEW` section. These are instructions to the AI, not guarantees of
its compliance or semantic accuracy.

## Triggers

Production jobs run only in `NVIDIA/TensorRT-LLM`:

- **Scheduled request:** `23 */2 * * *` (UTC), every two hours at minute 23.
  Select open, non-draft PRs targeting `main` or `release/**` that have
  `ci: full pre-merge approved` or auto-merge enabled. GitHub can delay runs.
- **Manual request:** **Run workflow** with one open main/release `pull_number`.
  Drafts and PRs without an approval label or auto-merge are accepted. Manual
  requests bypass version deduplication but retain revision and quota checks.
  An ordinary workflow rerun follows its original trigger and inputs.
- **Result publication:** a trusted CodeRabbit PR issue comment is created,
  edited or deleted. Validate the latest request and its replies before updating
  the commit status. This event does not request new analysis or wait for the next
  scan.

Label changes, enabling auto-merge, head pushes and target branch updates do not
directly request analysis. The scheduled scan observes those changes.

## Request policy

- Traverse open PRs in a single rotation ordered by descending PR number. With
  cursor `100`, visit numbers below `100` first, then numbers at least `100`.
  Without a retained cursor, start at the newest PR. Newly eligible PRs join
  this order without resetting or jumping ahead of the cursor. New analyses
  still require the scheduled candidate conditions above; a head that already
  contains target still needs analysis.
- Identify a version by its full head SHA, target SHA and target branch. PASS,
  FAIL and INCONCLUSIVE all complete a request; none automatically reruns that
  version. Completion is established by a valid reply, not by the commit status.
- If the latest request still matches the current version, has no valid reply
  and is at least two hours old, it may receive one automatic retry. The retry
  has a new request ID and records the earlier ID in `automaticRetryOf`. One
  recorded automatic retry exhausts that version's automatic allowance. Manual
  requests neither consume nor reset this allowance. After exhaustion, keep
  waiting for a valid reply or use a manual request.
- Select at most 30 new-analysis or retry request slots per scan. Failed
  selection or delivery attempts consume slots; an ambiguous POST failure may
  already have reached CodeRabbit. Workers recheck eligibility, revisions and
  replies under the PR's lock before sending. A selected worker that skips or
  fails is not replaced in the same batch.
- Stop traversal when the 30-slot budget or API reserve is reached, all open
  PRs have been visited once, or the request/repair matrix reaches 256 jobs.
  Record the last PR actually attempted during selection, including duplicate
  versions, ineligible PRs and individual read failures. A local quota check
  that prevents the PR's first API request does not count as a visit. No visits
  means no cursor change. Cursor movement does not depend on worker completion
  or successful AI requests.
- The scan also repairs result publication for the latest request on open PRs,
  including drafts or PRs that no longer have approval/auto-merge. Repairs do
  not consume AI request slots and cannot turn into new requests. The combined
  request/repair matrix fits GitHub's limit of 256 jobs per matrix.
- Post the request comment before publishing its pending status. If the comment
  was accepted but status publication fails, recovery publishes the status for
  that request without asking AI again. Read comments after an ambiguous delivery
  error before deciding whether the request was accepted.
- Discovery and new-request jobs stop when an applicable token's observed REST
  quota remaining is 1,000 or less, or when rate limited. `GITHUB_TOKEN` and the
  service PAT have separate quota checks. Repair-only jobs and comment-triggered
  publication do not require the service PAT or apply this reserve. Concurrent
  API users can spend quota between observations.

A scan first recovers any valid result already received for the latest request,
then considers a new analysis. A reply received before the worker's final
recheck prevents a timeout retry. When revisions change, analyze the current
version instead of retrying the obsolete one. Pure result recovery always uses
the original request's fixed revisions.

Twelve scheduled scans have a combined budget of 360 request slots; this is not
a calendar-day cap on manual requests, reruns or delayed batches. Each scan
continues after the last visited PR, even if that PR has closed. This gives
candidates recurring processing opportunities as scans make progress. Actual
analysis completion still depends on candidate volume, available quota, workflow
execution and AI delivery; rotation does not promise a completion deadline.

## Cursor persistence

Only scheduled scans restore and save the PR-number cursor. Manual PR requests
and result publication do neither. The scheduled workflow concurrency queue
serializes the complete restore/select/save cycle.

The cursor is stored as `{"last_pr":100}` in `cursor.json`, alternating between
Actions artifact names `semantic-review-cursor` and `semantic-review-cursor-next`.
Discovery adds only `actions: read` to its
`GITHUB_TOKEN` permissions; native artifact upload uses the workflow runtime
credential. Cursor storage does not use the service PAT or repository write
permissions.

Select the newest retained artifact from this repository's scheduled
`semantic-review.yml` runs, ordered by creation time and artifact ID. Ignore
artifacts from PR, manual or other workflows. A run need not have succeeded:
selection progress remains valid when later PR jobs fail. Save to the other
artifact name so the latest saved cursor survives even if a same-run rerun
fails while replacing the older artifact.

API, download, expired-artifact and invalid-file errors fail the scan instead of
falling back to an older cursor or treating the error as initialization. Save
after partial selection failures or quota exhaustion if at least one PR was
visited. Upload failure is reported and prevents dispatching that batch's PR
jobs. Cursor maintenance does not change the version deduplication rules.

Artifacts use a 90-day retention request, subject to repository limits. If no
trusted cursor artifact is retained, log initialization and start with the
newest PR. This includes first use and deletion of all retained cursor artifacts;
artifact storage cannot distinguish those cases. Do not delete these artifacts
when preserving the scan position matters. Remaining PRs, including repairs
beyond a batch's stopping point, resume in the next rotation.

## Results

Requests have a unique ID and fixed head/target/merge-base SHAs. The publisher
validates the bot identity, request identity and all three revisions. Each PR
uses the fixed status context `Semantic conflict with target branch / PR #N`,
where `N` is its PR number. This keeps separate PRs that share a head commit from
overwriting each other's status and keeps the result name independent of the
workflow that publishes it.

| Situation | Commit status | Description | Request completed? |
| --- | --- | --- | --- |
| No valid reply yet | `pending` | `Waiting for CodeRabbit response` | No |
| PASS | `success` | `No semantic conflict found (best effort)` | Yes |
| FAIL | `failure` | `Possible semantic conflict` | Yes |
| INCONCLUSIVE | `pending` | `Review completed: inconclusive` | Yes |

Commit statuses have no neutral completion state. INCONCLUSIVE therefore remains
visually pending, while its valid reply completes the request internally and
prevents an automatic retry. Read the description and linked reply to distinguish
it from a missing response. Waiting links to the request comment; a result links
to the CodeRabbit reply. The comments preserve the request ID and fixed revisions.

Wrong identities, request IDs or revisions are ignored. A reply associated with
the request but lacking a valid result format does not complete it; it remains
eligible for the bounded timeout retry. A correctly bound PASS/FAIL without the
required fixed-revision source citations completes as INCONCLUSIVE. This does
not establish the semantic correctness of those citations or the AI's findings.

Publication uses only the latest request and its matching replies. A new request
updates the same PR context on its requested head commit; an old request's late
reply cannot replace the current result. Status history remains available.

After publication, a tidy job maintains one sticky summary comment per PR
(marked `semantic-review-sticky`): the latest state, derived from the same
reviewState logic as the commit status, plus a per-request history table linking
each request and reply. Historical rows are resolved with the same newest-reply
and recorded-source rules as publication, read from each request's own head, so
corrections and revocations that publication honored are never replaced by an
older reply. It then minimizes (classifier `OUTDATED`) the request and every
bound reply of pairs that have a recorded verdict or were superseded by a newer
request; the active request stays visible while waiting, and is restored
(unminimized) if its recorded verdict is later revoked by a reply edit or
deletion. Minimized comments remain
expandable and link-addressable, so status deep links keep working. The sticky
comment never contains a live reviewer mention and is written only from
validated fields (request IDs, revision SHAs, comment IDs, verdicts), never raw
reply text. Both operations run only on reply events for PRs with a semantic
review request, so between a new request and its reply the sticky summary
still reflects the previous run; the commit status remains the verdict of
record, and minimization is only a display change, not a result override. Minimization is
best-effort: a failure logs a warning, never blocks the sticky summary or the
status, and is retried on the next reply event.

If a published reply is edited or deleted and no longer supplies a valid result,
the current request returns to waiting without asking AI again. Publication
retains the reply source so that it cannot fall back to an older PASS.
Reconciliation uses the same parser and publication code as comment events and
skips writes when the recorded state already matches the reply.

Existing request comments remain valid, including records with a Check Run ID.
Recovery cancels incomplete semantic Check Runs on the current head and latest
request's head. Completed Check Runs remain as historical records; they cannot
be deleted or converted into commit statuses through the Checks API. New results
are published only as commit statuses, and Check cancellation is not an AI result.

Reply events publish without waiting for a scan. Request, publication and tidy
jobs for the same PR share a concurrency queue. Each batch runs up to four workers;
the workers finish after their GitHub operations and do not wait for CodeRabbit.
This is not a limit on concurrent AI analyses. Scheduled batches run one at a
time; manual requests and publication do not share that batch lock.

Target updates do not immediately invalidate a recorded snapshot. A later scan
can request the new combination if eligibility, budget and quota permit. A scan
workflow's success means orchestration succeeded; it is not an AI PASS.

A PR that merges between scans may never be requested. An already requested
analysis may finish after merging and publish its recorded snapshot, not an
audit of the final merge tree. Scheduled recovery covers open PRs; a missed
publication on a closed PR requires a subsequent trusted reply event or a
publisher job rerun.

## Correcting a disputed result

Keep the original request and reply as evidence. Record the disputed finding and
fixed-revision counter-evidence in the PR discussion; a human comment does not
change the machine result. Do not edit the bot's verdict to impersonate a new AI
judgment, or delete the reply to clear a failure. Deleting the latest valid reply
returns its request to waiting and can make it eligible for the bounded timeout
retry. Hiding a comment is only a display change, not a result override.

After a prompt or code fix is available, use **Run workflow** with `pull_number`
to request another analysis. It creates a new request ID for the PR's current
head and current target, even if they match the preceding request. Verify the
new reply's revisions and status; a result for a newer target does not prove that
an earlier fixed-input finding was wrong. An older request's reply cannot replace
the latest request's result. Re-running only the publication job reprocesses the
existing reply; it does not ask AI to reconsider the evidence.

Completed legacy Check Runs remain historical records even after a new commit
status is published. Correcting the current status does not automatically clear
those older checks. A prompt update alone neither reruns completed requests nor
retracts their findings.

## Deployment and permissions

Install the workflows and scripts on the repository's default branch. Privileged
jobs always check out that branch and never execute PR-controlled code. The
read-only automation test workflow runs against the proposed changes.

`TRTLLM_AGENT_SHARED_TOKEN` must belong to `trtllm-agent` (user ID `296075020`)
and permit posting issue comments. It sends CodeRabbit commands and reads its
own identity and quota; repository reads and commit status publication use
`GITHUB_TOKEN`. Discovery has `statuses: read` and `checks: read`. Request and
publication jobs have `statuses: write` and retain `checks: write` only to cancel
incomplete semantic Check Runs. The publisher recognizes `coderabbitai[bot]`
(user ID `136622811`). Keep semantic statuses non-required.

The tidy job keeps comment writes separate from the status/check writers: it
has `issues: write` and `pull-requests: write` (comment upsert and
minimization via `GITHUB_TOKEN`) with read-only status/check access, while the
publish job never holds comment write permissions or the service PAT.
Minimizing another user's comment requires repository write access, which the
workflow's `GITHUB_TOKEN` grants per-scope.

## Validation

Run the deterministic policy and result tests:

```sh
node --test .github/scripts/semantic_review*.test.js
```

`semantic_review_cases.js` freezes three real incidents in divergent,
already-integrated and repaired states, plus three paired-edit controls, for
analysis replay. Integrated inputs use the actual defective merge commits to
exercise `merge_base == target`;
they are not newly synthesized rebases. Repair controls compare each historical
fix with its immediate parent to exclude unrelated intervening changes. Build
each request with the same `command()` used by the workflow:

```sh
node -e "const {randomUUID}=require('node:crypto'); const {command}=require('./.github/scripts/semantic_review'); const cases=require('./.github/scripts/semantic_review_cases'); console.log(command({...cases[0],id:randomUUID()}));"
```

The paired-edit controls use the fixed inputs of PRs #19465, #19397 and #18813.
They exercise coordinated executor changes, a retained mock default, and
coordinated warmup changes. The alleged caller/definition, missing-stub and
missing-metadata mismatches must not be reported when three-way integration
preserves the paired edits. #19397 also has textual conflicts, so rejecting that
specific false finding does not require an overall PASS. Fixtures carry no
expected answer into `command()`.

Replay all twelve inputs with the final prompt, retaining request IDs, input SHAs,
raw replies, concrete findings and timings, including missed/inconclusive results.
Use an independent context without giving the incident explanation or a repair.
If the hosting discussion reveals the answer, label the run as a replay rather
than a blind evaluation. Fixed repeats characterize variability; production
uses one initial request and at most one automatic timeout retry per version.
Assess known defect detection and repair false positives separately; a narrow
repair control does not establish general accuracy.

A defective input is successfully flagged when FAIL identifies at least one
source-supported, in-scope incompatibility for developers to investigate.
Finding every known defect or providing a complete fix is not required. Record
additional omissions as observations, not failed alerts. Uncertainty about one
path must not override a confirmed incompatibility on another. Repair inputs
and paired-edit controls also check for unsupported FAIL findings.

The deterministic tests simulate GitHub. A real AI reply and a real commit
status write are separate validation layers and must be reported as such.
Fixtures preserve historical revisions; replay them explicitly without
broadening the production candidate filter.
