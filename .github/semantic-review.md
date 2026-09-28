<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Semantic conflict review

The workflow asks CodeRabbit to inspect the combined behavior of a PR and its
target branch. Its **Semantic conflict with target branch** check is advisory:
keep it **non-required** in branch protection/rulesets. AI can miss defects and
report false positives; reviewers should read the linked evidence.

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
  the Check. This event does not request new analysis or wait for the next scan.

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
  version. Completion is established by a valid reply, not by a Check's color.
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
- Recover abandoned pending Checks as part of publication repair. A confirmed
  delivery failure cancels the undelivered Check; an unknown delivery outcome
  stays pending until comments can be read again. Recovery cancels superseded
  or unrecorded pending Checks on the current PR head and latest request's head,
  without treating cancellation as an AI result.
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

The cursor is stored as `{"last_pr":100}` in `cursor.json`, in an Actions artifact
named `semantic-review-cursor`. Discovery adds only `actions: read` to its
`GITHUB_TOKEN` permissions; native artifact upload uses the workflow runtime
credential. Cursor storage does not use the service PAT or repository write
permissions.

Select the newest retained artifact from this repository's scheduled
`semantic-review.yml` runs, ordered by creation time and artifact ID. Ignore
artifacts from PR, manual or other workflows. A run need not have succeeded:
selection progress remains valid when later PR jobs fail. Same-run reruns
restore the saved position before replacing that run's artifact.

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
validates the bot identity, request identity and all three revisions. The current
request has the following states:

| Situation | Check status | Conclusion |
| --- | --- | --- |
| No valid reply yet | `in_progress` | None |
| PASS | `completed` | `success` |
| FAIL | `completed` | `failure` |
| INCONCLUSIVE | `completed` | `neutral` |

Wrong identities, request IDs or revisions are ignored. A reply associated with
the request but lacking a valid result format does not complete it; it remains
eligible for the bounded timeout retry. A correctly bound PASS/FAIL without the
required fixed-revision source citations completes as INCONCLUSIVE. This does
not establish the semantic correctness of those citations or the AI's findings.

Each new request creates a new Check Run. Completed Checks cannot reliably be
reset to an empty conclusion through the REST update operation. Historical
Checks remain available; pending Checks on the same head are cancelled when
superseded. Publication selects the newest Check matching the latest request ID,
so an old request's late reply cannot replace the current result.

If a published reply is edited or deleted and no longer supplies a valid result,
the current request returns to waiting through a new Check Run for that same
request, without asking AI again. Reply-source markers and links are stored in
the Check output; they prevent falling back to an older PASS. Reconciliation
uses the same parser and publication code as comment events and skips writes
when the recorded state already matches the reply.

Reply events publish without waiting for a scan. Request and publication jobs
for the same PR share a concurrency queue. Each batch runs up to four workers;
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

## Deployment and permissions

Install the workflows and scripts on the repository's default branch. Privileged
jobs always check out that branch and never execute PR-controlled code. The
read-only automation test workflow runs against the proposed changes.

`TRTLLM_AGENT_SHARED_TOKEN` must belong to `trtllm-agent` (user ID `296075020`)
and permit posting issue comments. It sends CodeRabbit commands and reads its
own identity and quota; repository reads and Check publication use
`GITHUB_TOKEN`. The publisher recognizes `coderabbitai[bot]` (user ID `136622811`).
Keep these checks non-required.

## Validation

Run the deterministic policy and result tests:

```sh
node --test .github/scripts/semantic_review*.test.js
```

`semantic_review_cases.js` freezes three real incidents in divergent,
already-integrated and repaired states for analysis replay. Integrated inputs
use the actual defective merge commits to exercise `merge_base == target`;
they are not newly synthesized rebases. Repair controls compare each historical
fix with its immediate parent to exclude unrelated intervening changes. Build
each request with the same `command()` used by the workflow:

```sh
node -e "const {randomUUID}=require('node:crypto'); const {command}=require('./.github/scripts/semantic_review'); const cases=require('./.github/scripts/semantic_review_cases'); console.log(command({...cases[0],id:randomUUID()}));"
```

Replay all nine inputs with the final prompt, retaining request IDs, input SHAs, raw
replies, concrete findings and timings, including missed/inconclusive results.
Use an independent context without giving the incident explanation or a repair.
If the hosting discussion reveals the answer, label the run as a replay rather
than a blind evaluation. Fixed repeats characterize variability; production
uses one initial request and at most one automatic timeout retry per version.
Assess known defect detection and repair false positives separately; a narrow
repair control does not establish general accuracy.

The deterministic tests simulate GitHub. A real AI reply and a real Check write
are separate validation layers and must be reported as such. Historical cases
are already merged; replay them explicitly without broadening the production
candidate filter.
