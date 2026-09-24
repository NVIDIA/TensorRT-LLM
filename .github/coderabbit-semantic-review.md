<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advisory semantic conflict review

The `CodeRabbit Semantic Conflict Review` workflow performs best-effort semantic
compatibility analysis for open, non-draft PRs targeting `main` or `release/**`.
It compares both branches from their merge base and
follows affected callers, contracts, configuration, and tests across files.

- PR creation, reopening, commit updates, becoming ready, target-branch changes
  and six-hour scans evaluate the same threshold.
- The first analysis needs new target commits and either 24 hours since the
  merge-base commit or at least 30 target commits beyond that base. After a
  completed PASS/FAIL analysis, a changed head or target qualifies after 24 hours
  from that reply or 30 additional target commits. A head-only change therefore
  qualifies after 24 hours even when the target has not advanced. If the latest
  request has no valid verdict, use its request time and target as the baseline.
  Rewritten target history falls back to the merge-base threshold.
- An authorized `ci: full pre-merge approved` label or enabling auto-merge
  bypasses the threshold. Approval-label authors are checked against the existing
  `trt-llm-ci-approvers` team using its existing token. Automatic pre-merge requests
  for each PR and target branch share a one-hour cooldown, even across SHA changes. A signal
  during cooldown is skipped; ordinary scans and the post-merge audit remain.
- Exact revision pairs are deduplicated. When a reply is missing, incomplete or
  Inconclusive, a scheduled scan may retry that version once, at least six hours
  after its latest request. This applies to pre-merge analysis and audits. A
  verified FAIL is a completed analysis, not a reason to retry. After the one
  automatic retry, the same version needs a manual retry; changed versions are
  evaluated under the ordinary threshold and get their own retry allowance.
  Workflow dispatch accepts one PR number and bypasses the thresholds, cooldown
  and deduplication for that PR only.
- Merge events request a post-merge audit regardless of thresholds or cooldown.
  The six-hour scan recovers merges from the preceding 24 hours. The audit pins
  the actual merge commit and historical target, including for release PRs;
  later target updates do not invalidate it. Squash-only rules or the two-parent
  merge must establish the historical target; ambiguous rebase history is rejected.
  A pre-merge result/request can be reused only when its head/target pair and
  GitHub's recorded test-merge tree match the actual merged tree. Otherwise the
  audit includes the actual merged code in a new analysis request.

## Request transport and scan limits

The workflow reads the semantic instructions from `.coderabbit.yaml`; the native
custom check remains `off` during ordinary reviews. It requests an ordinary
CodeRabbit PR chat reply with `SEMANTIC_REVIEW_V3`, the full revision record and
source links. Native custom-check PASS table cells can truncate those details.
Legacy table replies remain readable but truncated evidence never becomes PASS.
The request explicitly forbids formal review submission or Request Changes.

Commands use the existing `TRTLLM_AGENT_SHARED_TOKEN` and require the
`trtllm-agent` User account (ID `296075020`), verified before scanning. There is
no fallback to `github-actions[bot]`, whose commands may be ignored. Repository
reads and Check publication still use `GITHUB_TOKEN`; neither token executes PR
code. Acceptance of the service-account commands, especially on merged/release
PRs, requires a deployment pilot. Missing replies never imply semantic success.

Scheduled scans run at minute 23 every six hours (UTC). Each scan sends at most
20 new AI requests, including audits, to avoid an initial burst across all open
PRs. Recent merged PRs are considered first; open PRs use a rotating starting
point. The single automatic retry also counts toward the 20-request limit.
Scans restart from live state, not a saved cursor. Saturation can delay PRs;
use a manual dispatch for a specific PR rather than relying on a resume guarantee.

Each token's remaining REST budget is read once, then tracked from response
headers, including pagination. A scan stops below a 100-request reserve or on
primary/secondary rate limiting and reports processed PRs and new requests.
Ordinary permission errors remain failures. The 20-request cap and 100-request
reserve apply only to scheduled scans; GitHub's own rate limits can affect every
API operation, including single-PR events, manual requests and result publication.
Frequency reduction does not reduce the peak cost of one scan. A rate-limited or missed scan can also delay
post-merge recovery beyond its 24-hour lookback.

## Result verification

The `Semantic conflict with target branch` Check starts neutral. Stale results
become neutral when the PR event or scheduled scan observes a version change.
This rule applies to every pre-merge verdict; historical audit verdicts stay
attached to their fixed merged revisions. Only formatted CodeRabbit analysis
replies update the Check. Receiving a reply publishes a result, not another AI
request.
The verifier checks the bot identity, most recent trusted request, exact revision
record, and GitHub merge base. Publication and preview select the newest applicable
reply after that request; delayed events cannot restore an older verdict, and
pending manual retries cannot reuse an earlier PASS. Before deployment, the preview
also accepts manual evaluations when no trusted request exists for the pair.
PASS becomes success; FAIL makes the Check and
publishing job red; Inconclusive remains neutral. A successful request job only
means orchestration succeeded. Checks on the actual merge SHA use the distinct
`Semantic conflict audit (post-merge)` name, with a receipt linking the analysis
on the original PR. Evidence includes code locations and regression scenarios.
The instructions first discover cross-branch interactions, then verify their
contracts, including test replacements and the production paths they exercise.
Replies must cite immutable source links
with full SHAs and line numbers from both head and target. A PASS/FAIL without
those citations becomes Inconclusive, including in the preview; an older PASS
cannot substitute for that incomplete reply. Citation presence does not prove
the AI's reasoning or the cited code is correct.

CodeRabbit can make mistakes, including false positives. Keep these checks and
workflows non-required: their failures then do not block merging. No required
waiting gate is added, and auto-merge does not wait for this analysis. Audit does
not revert code or modify branches. Repository rules remain unchanged.
The thresholds and cooldown limit frequency, not total calls per PR.

Changes to this automation run the separate read-only `CodeRabbit Semantic
Review Preview` workflow, including fork drafts. `Automation tests and result
lookup (not AI approval)` runs Node tests and reads actual CodeRabbit replies;
`AI verdict (advisory; skipped = unavailable)` runs only for a verified current
PASS/FAIL and is otherwise gray/skipped. `precommit-check.yml` is unchanged.

Both workflows use the same verifier. Privileged jobs load only trusted default
branch scripts, never PR code. The preview has read-only permissions. Before
merge, a maintainer can use the exported `command(pair)` function in
`.github/scripts/coderabbit_semantic_review_request.js` to prepare a chat request
with fixed `head`, `target`, `mergeBase` and `branch`, then post it on the PR.
After the reply arrives, rerun the **tests and result lookup** job; rerunning only
the AI job reuses old outputs. Preview tests do not establish production trigger,
permission, command-acceptance or post-merge behavior.
