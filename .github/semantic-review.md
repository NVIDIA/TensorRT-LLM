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

- Read the current head, target and merge-base. Scheduled requests skip
  previously requested head/target/branch combinations, regardless of whether
  the earlier analysis replied or passed. A head that already contains target
  still needs compatibility analysis: rebase or merge can incorporate semantic
  bugs.
- Select candidates by descending PR number, starting with the newest each scan.
  Select at most 30 requests after deduplication. Failed selection or delivery
  attempts consume slots; a failed POST can still have reached CodeRabbit.
  Workers recheck eligibility, revisions and scheduled-request deduplication under
  the PR's lock. If a selected worker skips or fails, its slot is not refilled.
- Stop the affected discovery or request job when a token's observed REST quota
  remaining is 1,000 or less, or when rate limited. `GITHUB_TOKEN` and the service
  PAT have separate quota checks; the service PAT is checked by request workers.
  Concurrent API users can spend quota between observations. This reserve does
  not apply to result publication.
- Skip delivery if eligibility or revisions change during preparation. A later
  scan can select the PR again. Once a request comment exists, a missing reply
  does not trigger an automatic retry; use a manual request to retry that version.

Twelve scheduled scans have a combined budget of 360 request attempts; this is
not a calendar-day cap on manual requests, reruns or delayed batches.
Newest-first selection can defer older PRs indefinitely when target keeps
advancing and there are more than 30 actionable candidates. Monitor actual
throughput and backlog before changing this policy.

## Results

Requests have a unique ID and fixed head/target/merge-base SHAs. CodeRabbit
replies carry those fields. The result job validates the bot identity, request,
revisions, and presence of source citations before publishing:

| Result | PR check |
| --- | --- |
| PASS | Success: no conflict found for the recorded revisions |
| FAIL | Failure: possible conflict; inspect linked evidence |
| Missing or inconclusive | Neutral: no verified verdict |

Reply events process results without waiting for the next scheduled scan.
Request and publication jobs for the same PR share a concurrency queue, so
switching the current request cannot race an older reply. Different PRs can run
independently; each scan has up to four concurrent request workers. Scheduled
batches run one at a time; manual requests and publication do not share that batch lock. Jobs do
not wait for AI analysis while holding a queue. A scan's success only means its
requests were processed, not that AI approved those PRs.

When the target advances, a reply still describes its requested snapshot; target
updates do not immediately clear the Check. A later scan can request the newer
combination if the PR is eligible, selected within the budget and quota allows.
Switching requests clears the earlier verdict; an old reply cannot update the
new request's check. With a new head, the check is attached to that head.
Editing or deleting the published source reply revokes a conclusion that is no
longer valid when that event is processed.
The scheduled scan does not reconcile missed publication events or failed Check
writes. A later trusted reply event or a publisher job rerun can reconcile them;
an open main/release PR can also receive a new manual request.

A PR that merges between scans may never be requested. An already requested
analysis may finish after merging; its result still covers the recorded input,
not an audit of the final merge tree.

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
still issues one request. Assess known defect detection and repair false positives
separately; a narrow repair control does not establish general accuracy.

The deterministic tests simulate GitHub. A real AI reply and a real Check write
are separate validation layers and must be reported as such. Historical cases
are already merged; replay them explicitly without broadening the production
candidate filter.
