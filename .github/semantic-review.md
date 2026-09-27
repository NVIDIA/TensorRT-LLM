<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Semantic conflict review

The workflow asks CodeRabbit to inspect the combined behavior of a PR and its
target branch. Its **Semantic conflict with target branch** check is advisory:
keep it **non-required** in branch protection/rulesets. AI can miss defects and
report false positives; reviewers should read the linked evidence.

## Request policy

Every two hours, scan open, non-draft PRs targeting `main` or `release/**` that
have `ci: full pre-merge approved` or auto-merge enabled. PR activity does not
immediately request analysis. GitHub scheduled runs can be delayed.

- Read the current head, target and merge-base. Skip only previously requested
  head/target/branch combinations. A head that already contains target still
  needs compatibility analysis: rebase or merge can incorporate semantic bugs.
- Select candidates by descending PR number, starting with the newest each scan.
  Select at most 30 requests after deduplication. Failed selection or delivery
  attempts consume slots; a failed POST can still have reached CodeRabbit.
  Workers recheck eligibility, revisions and deduplication under the PR's lock.
- Stop requesting when either token's observed REST quota remaining is 1,000 or
  less, or when rate limited. `GITHUB_TOKEN` and the service PAT have separate
  quota checks. Concurrent API users can spend quota between observations.
- Do not automatically retry an unchanged version, including a missing reply.
  Maintainers may use **Run workflow** with an open main/release `pull_number`,
  including drafts, without an approval label or auto-merge. Manual requests
  bypass version deduplication but retain revision and quota checks. An ordinary
  workflow rerun still follows its original inputs.

The budget permits up to 360 scheduled requests per day; manual retries are
additional. Newest-first selection can defer older PRs indefinitely when target
keeps advancing and there are more than 30 actionable candidates. Monitor actual
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

Replies publish without waiting for the next scheduled scan. Request and
publication jobs for the same PR share a concurrency queue, so switching the
current request cannot race an older reply. Different PRs can run independently;
each scan has up to four concurrent request workers. Scheduled batches run one
at a time; manual requests and publication do not share that batch lock. Jobs do
not wait for AI analysis while holding a queue. A scan's success only means its
requests were processed, not that AI approved those PRs.

When main advances, a reply still describes its requested snapshot. The next
scan requests the newer combination. Switching requests clears the earlier
verdict; an old reply cannot update the new request's check. With a new head,
the check is attached to that head. Editing or deleting the published source
reply must revoke a conclusion that is no longer valid.

A PR that merges between scans may never be requested. An already requested
analysis may finish after merging; its result still covers the recorded input,
not an audit of the final merge tree.

## Deployment and permissions

Install the workflows and scripts on the repository's default branch. Privileged
jobs always check out that branch and never execute PR-controlled code. The
read-only automation test workflow runs against the proposed changes.

`TRTLLM_AGENT_SHARED_TOKEN` must belong to `trtllm-agent` (user ID `296075020`)
and permit posting issue comments. It is used only to send CodeRabbit commands;
reads and Check publication use `GITHUB_TOKEN`. The publisher recognizes
`coderabbitai[bot]` (user ID `136622811`). Keep these checks non-required.

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
