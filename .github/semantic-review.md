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

- Read the current head, target and merge-base. Skip if head already contains
  the target or the same head/target/branch combination was requested before.
- Issue at most 20 new requests per scan, rotating the starting PR. Failed
  attempts count against the budget because their delivery can be uncertain.
- Stop when approaching the GitHub REST quota reserve or receiving rate limits.
- Do not automatically retry an unchanged version, including a missing reply.
  Maintainers may explicitly retry an eligible PR with **Run workflow** and its
  `pull_number`. An ordinary workflow rerun still follows its original inputs.

The budget permits up to 240 scheduled requests per day; manual retries are
additional. Monitor actual throughput and backlog before changing this limit.

## Results

Requests have a unique ID and fixed head/target/merge-base SHAs. CodeRabbit
replies carry those fields. The result job validates the bot identity, request,
revisions, and presence of source citations before publishing:

| Result | PR check |
| --- | --- |
| PASS | Success: no conflict found for the recorded revisions |
| FAIL | Failure: possible conflict; inspect linked evidence |
| Missing or inconclusive | Neutral: no verified verdict |

Replies publish without waiting for the next scheduled scan. The request and
publication jobs share a concurrency queue, so switching the current request
cannot race an older reply. They do not wait for AI analysis while holding the
queue. A scan's success only means its requests were processed, not that AI
approved those PRs.

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

`semantic_review_cases.js` freezes three real divergent histories for analysis
replay. Build each request with the same `command()` used by the workflow:

```sh
node -e "const {randomUUID}=require('node:crypto'); const {command}=require('./.github/scripts/semantic_review'); const cases=require('./.github/scripts/semantic_review_cases'); console.log(command({...cases[0],id:randomUUID()}));"
```

Replay all three with the final prompt, retaining request IDs, input SHAs, raw
replies, concrete findings and timings, including missed/inconclusive results.
Use an independent context without giving the incident explanation or a repair.
If the hosting discussion reveals the answer, label the run as a replay rather
than a blind evaluation. Fixed repeats characterize variability; production
still issues one request. Validate compatible repair controls separately.

The deterministic tests simulate GitHub. A real AI reply and a real Check write
are separate validation layers and must be reported as such. Historical cases
are already merged; replay them explicitly without broadening the production
candidate filter.
