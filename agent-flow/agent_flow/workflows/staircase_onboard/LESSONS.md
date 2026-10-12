<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lessons from real runs

What two full onboarding runs taught this workflow. Both runs onboarded a
large MoE checkpoint with features onto multi-GPU Blackwell nodes. Each
lesson gives the **symptom** an operator or agent saw, the **cause**, and the
mechanism that now **prevents** it. Where the prevention is not in the
workflow yet, **open** names the gap.

Most of these presented as a model or target defect while they were
happening, and none of them were. Read this before debugging a target. For
verbatim error text, use [TROUBLESHOOTING.md](TROUBLESHOOTING.md) instead.

## Substrate and preflight

### 1. Prove the host before any agent exists
- **Symptom.** Hours into a run, a turn or gate fails with `MPI_ERR_SPAWN`,
  a kvCache that cannot fit a sequence, or NVRTC missing `cuda.h`, and the
  failure reads as a target defect.
- **Cause.** The substrate was first exercised by the target itself.
- **Prevented by.**
  - `preflight.run_preflight` claims the devices (`claim_devices`,
    `probe_device_alone`) and checks the checkpoint and the accuracy anchors
    (`check_checkpoint`, `check_accuracy_anchors`).
  - It also checks `TMPDIR` (`check_tmpdir`) and the multi-rank substrate
    (`check_multirank_substrate`).
  - `substrate_gate` cold-starts the stock path under the run's own runtime
    block.
  - `agent_plane_gate` proves an agent turn can start at all.

### 2. A new architecture has no stock path, so the substrate check needs a proxy
- **Symptom.** Preflight, or a stock-path gate leg, stops on "unknown
  architecture". That is the condition onboarding exists for.
- **Cause.** The check booted the run's own checkpoint. "Unsupported" was
  matched by a marker list that missed refusals from other layers: the
  registry, `transformers`' unknown `model_type`, custom code, and
  registration shims.
- **Prevented by.**
  - `environment.substrate_checkpoint` boots a proxy checkpoint instead.
  - `UNSUPPORTED_ARCHITECTURE_MARKERS` / `reports_unsupported_architecture`
    cover each refusal path.
  - The acceptance gate falls back to a declared `min_absolute` floor and
    records the weaker claim.

### 3. A multi-rank launch inside someone else's allocation has rules; generate it in one place
- **Symptom.** "Not enough slots"; N ranks that each believe they are the
  leader and hang without an error; a ZeroMQ IPC path longer than 107
  characters.
- **Cause.** A session is one Slurm task. The launcher prefers `SLURM_*` over
  `OMPI_*`. The RPC socket lives under `TMPDIR`, and `sun_path` caps at 107.
- **Prevented by.**
  - `environment.resolve_environment` emits the one launch line every prompt
    carries, with `--oversubscribe` and the `MULTI_TASK_SLURM_SCRUB` /
    `SINGLE_TASK_MPI_SCRUB` scrubs.
  - `check_tmpdir` bounds `TMPDIR` at `MAX_TMPDIR_LEN` (61) and shows the
    arithmetic.

### 4. Work that is not an engine must not inherit the engine's launcher
- **Symptom.** A probe exits 127 with no output. A leg that is itself a
  launcher runs only on rank 0, and its child engines hang on the outer proxy.
- **Cause.** Every command was wrapped for multi-rank. A spec `PATH`
  replaced the inherited one. Launcher variables leaked into child engines.
- **Prevented by.**
  - Single-process commands for the survey probe and for witness contrast
    legs (`single_process=True`).
  - Launcher variables scrubbed from contrast legs.
  - `PATH` composed with the inherited value.

### 5. A skipped or failed measurement must never print like an empty or passing one
- **Symptom.** "0/0 entries certified ✔" against a populated catalog;
  "preflight passed" when the substrate gate never ran.
- **Cause.** Summaries did not separate "measured zero" from "did not
  measure", and test fixtures used a different file shape from the real
  index.
- **Prevented by.** The survey states an inventory-probe error in the block
  the plan drafter reads, and renders whatever did land, so a measurement is
  never confused with its absence.
- **Open.** The dry run should print that the substrate gate was skipped.

### 6. Environment knowledge is an artifact, not something passed by hand
- **Symptom.** Every agent re-derives the same host facts (pytest's exit
  code, the `TMPDIR` bound, which log line is noise). Hand-edited entry
  scripts point at the wrong spool. The container image is imported again
  for every allocation.
- **Prevented by.**
  - `host_facts.md`, seeded by preflight and appended by roles, which
    survives resumes.
  - `corrections.md`, for findings that land after the plan.
  - `image_import_warning` and `shares_allocation_with_spool` notes.
- **Open.** Generate the session entry from `environment.spool` (plan 4.4).

## Plan and survey

### 7. Every structural config key is a demand, and the demand map is closed
- **Symptom.** The plan omits obligations that surface hours later as
  `BLOCKED: vocabulary_gap`, or describes a drafter as the wrong kind.
- **Cause.** The survey had no probe for hybrid layer patterns, attention
  sinks, asymmetric head dimensions or routing groups.
- **Prevented by.**
  - `survey.read_checkpoint_surface`, with probes such as `attention.hybrid`
    and `attention.sink`; unrecognized keys are reported, not dropped.
  - `runtime_index.yaml` and a test that every demand an area declares is one
    some probe can produce.

### 8. Anchors are phase zero
- **Symptom.** A unit stalls at its accuracy gate because the reference row
  does not exist, or anchor jobs spend hours rediscovering the reference
  engine's recipe.
- **Prevented by.** `preflight.check_accuracy_anchors` checks each unit's
  `reference_yaml` row for its model key and discriminators before the plan
  phase.

### 9. An independent plan review is worth its cost, and must stay independent
- **Symptom.** The review was run on the drafter's own model family and
  recorded as cross-model.
- **Cause.** The reviewer's backend did not travel with an externally
  dispatched turn.
- **Prevented by.**
  - Per-role routing (`roles.py`; `plan_reviewer` defaults to Codex).
  - The CLI warns when the reviewer resolves to the drafter's model.
  - Each `dispatch/<id>.request.json` names the role's backend and model.
- **Open.** Record in the run's own record when the answering model cannot
  be verified.

## Smith and catalog

### 10. Schedule a smith row by the unit that needs it
- **Symptom.** The run closes before core over an entry only a later feature
  needs.
- **Prevented by.** `unit:` on queue rows; a feature-only failure is
  `deferred`, retried once before its unit, and closes the run as
  `smith-blocked:<unit>` only if the retry fails too.

### 11. Certify against a declared serving envelope, never against observed values
- **Symptom.** Review loops that never converge. UNCOVERED rows found only
  at the final audit. Crashes or silent corruption beyond the configuration
  the gates ran.
- **Cause.**
  - Coverage was keyed on values the gates happened to dump.
  - Capture sets were derived per entry rather than per call site.
  - Target capacity constants and kernel integer widths were not tied to
    engine limits.
- **Prevented by.**
  - `envelope.py` (`serving_envelope`, `feature_matrix`, `derive_certification`).
  - `smith_brief.py`: call-site row multiples, and integer-width probes at
    2^31 and 2^31 / row width.
  - `capacity_audit.py`: capacity constants bound to engine arguments.
  - The bounded remediation cycle for `UNCOVERED`.
  - The typed `certification_gap` decision.

### 12. Prove a test can fail
- **Symptom.** Tests and floors pass a deliberately wrong implementation. A
  hypothesis is "confirmed" by a count that both hypotheses predict.
- **Cause.** No negative control, and thresholds chosen without measurement.
- **Open.** The kernel skills should refuse a receipt without a negative
  control, and floors should sit above the best wrong control.

### 13. Recurring outcomes get typed decisions, not prose
- **Symptom.** The operator picks redirects, verify requests and spec fixes
  out of prose and dispatches them by hand.
- **Prevented by.** `certification_gap` fanned out to verify smiths;
  `corrections.md`, which takes precedence over `vocabulary.yaml`.
- **Open.** A typed `spec_change` the orchestrator applies and records.

## Gates

### 14. Gate every configuration a target claims
- **Symptom.** A target serving several parallel configurations is green,
  but its expert-parallel and attention-DP branches never ran.
- **Cause.** The gate folded a unified target to tensor parallelism, and the
  member set was prose.
- **Prevented by.** `members:` with `members_for`; each verdict is named
  `<gate>@<member>`; `order_members_by_cost` runs the cheapest member first;
  `--dry-run` prints the gate count.
- **Open.** Run witnesses on every member, not only the first (plan 1.5).

### 15. A gate records, and owns, the code it measured
- **Symptom.** A number is about to be credited to a commit it was not
  measured on. Rank 0 runs one checkout while the workers run another.
- **Cause.** Gates import whatever is on disk, and MPI workers resolve the
  package from their working directory ahead of `PYTHONPATH`.
- **Prevented by.** `gates.inspect_tree` records the tree in every verdict
  and refuses a dirty target path unless `STAIRCASE_ALLOW_DIRTY_GATE` is set,
  which marks the verdict as such. A member's overlay is never written into
  the source tree.
- **Open.** `--tree`, with a per-rank import-origin check (plan 4.2).

### 16. Gates run `TRTLLM_MODELING_V2=require`, and each unit's gates take its own path
- **Symptom.** A feature unit's boot never exercises the feature. The
  serving path returns errors no gate saw. A number measured through
  fallback reads as the target's.
- **Prevented by.**
  - Every gate runs under `=require`.
  - The speculator depth comes from the unit's own options YAML
    (`resolve_max_draft_len`).
  - The unit's own LLM-API options file (`accuracy.llm_api_options`).
  - `preflight.check_speculation_support` refuses a speculation algorithm the
    gates' launcher has no branch for, instead of letting it run without a
    speculator.
- **Open.** A boot workload per unit (images for a multimodal unit),
  accuracy through each unit's deployment entry (server, disaggregated
  proxy), and boot keywords matched only in generated text (plan 1.4).

### 17. Read failures from the whole log, and results from explicit markers
- **Symptom.** The gate shows only a generic re-raise. A keyword matches a
  log line rather than the answer. Reviewers hand-parse interleaved
  multi-rank stdout.
- **Prevented by.**
  - Per-rank structured runtime-value dumps (`TRTLLM_MODELING_V2_VALUE_DUMP`).
  - Witness token ids and text printed on marked lines
    (`TRTLLM_WITNESS_FULL_OUTPUT`).
- **Open.** Classify from the first traceback in the full output, not from
  the display tail.

### 18. An acceptance gate needs a workload that drafts, and a floor that means something
- **Symptom.** Acceptance pinned at 1.00 for every drafter, working or
  broken. Floors that pass wrong drafters. Regressions flagged on noise.
- **Cause.**
  - The workload's answers ended in 2–3 tokens.
  - The floor was taken from the ceiling.
  - A 4-prompt comparison used a fixed delta.
- **Prevented by.**
  - `accuracy.acceptance` declares `prompts`/`prompts_file`, `max_tokens`
    and `min_draft_iterations`.
  - A workload too short to measure fails as such, not as a low rate.
- **Open.** At least 32 prompts with a paired test; gate every unit that
  keeps speculation on, not a fixed list of unit names (plan 1.2).

### 19. Accuracy floors are statistical, and generation steps need a golden check
- **Symptom.** A defect present in every member passes every gate.
- **Cause.** The floor sat several standard errors below the anchor, and
  golden checks covered context steps only.
- **Open.** Floor = anchor − 2·SE; "below the anchor on every member" is a
  finding; a generation-step check under overlap and speculation against the
  reference (plans 1.1 and 1.3).

### 20. Add a cheap performance gate
- **Symptom.** An accuracy-neutral change regresses the serving path by
  double digits, and only a profile catches it.
- **Open.** One prefill chunk plus one decode step, compared with the
  previous tree; fail on more than 5% unless the commit declares a trade-off
  (plan 1.6).

## Witnesses

### 21. A feature the standing gates cannot see needs its own kind, skill and witness
- **Symptom.** An output-identical feature, or one trained to stay inside
  the accuracy tolerance, passes every gate whether or not it was built. The
  nearest existing kind builds the wrong thing.
- **Prevented by.**
  - A closed `features:` list, each feature with a skill.
  - `WITNESS_GATED_UNITS`: residency, equivalence, replay and kv_quant
    witnesses.
  - Counters the target must print (`MODELING_V2_WITNESS` under
    `TRTLLM_MODELING_V2_WITNESS=1`), where a missing counter is a failure.
  - Conflicting kinds refused by the schema (`MTP_NAMESPACE_FEATURES`).

### 22. A contrast must be real and differ in exactly one variable
- **Symptom.** A witness compares a run with itself. Desynchronized legs
  read as transfer defects. Identity fails by construction.
- **Prevented by.**
  - The equivalence witness requires `contrast_command`.
  - Leg B gets leg A's workload through `STAIRCASE_WITNESS_*` variables, with
    absolute paths.
  - `legs_env` pins what both legs share.
- **Open.** Assemblers should call the workflow's own gate entry rather than
  reconstruct it.

### 23. A token comparison must be able to see a difference, and say where
- **Symptom.** Witnesses on 2–3-token continuations; prefix-only text
  matches; near-tie flips caused by batching; guessed timeouts.
- **Prevented by.**
  - `witness.compare_continuations` compares token ids, then text, then
    prefix, and quotes the first difference from both legs.
  - `max_batch_size: 1` on both legs.
  - A per-leg `timeout`.
- **Open.** A concurrent leg beside the batch-1 one (plan 1.5).

## Transport, resume and concurrency

### 24. When a host cannot authenticate an agent, move the turn and nothing else
- **Symptom.** Every turn fails with zero tokens, as `EROFS ...
  credentials.lock` or `Not logged in` after about forty minutes, which
  reads as a model refusal.
- **Cause.** A read-only `$HOME`, then a single-use refresh token shared
  between copies.
- **Prevented by.**
  - `agent_home.ensure_agent_home`.
  - `--agent-transport external`: the stage machine, decision validation and
    gates stay in the orchestrator.
  - Decision files when MCP servers cannot be registered.

### 25. Resume and re-entry must not cost work or budget
- **Symptom.** A restart re-dispatches finished or in-flight turns, charges
  turns nobody answered, or closes a run a later repair could reopen.
- **Prevented by.**
  - A resumed process adopts a smith's decision file, and waits on an
    unanswered twin request instead of asking again.
  - `REMEDIABLE_VERDICTS` (including `assembler-blocked:knowledge_gap` /
    `:environment`) re-enter with the turn budget restored.
  - A `smith-blocked` run whose failures are feature-only re-enters.
  - A delivered run whose spec grew re-enters at the new unit.
  - The spec is re-read on every launch (`_refresh_workspace_spec`).
- **Open.** Checkpoint each gate verdict with its tree, and skip it on
  resume when unchanged (plan 3.1).

### 26. Liveness is per turn, timeouts are per role, and the driver must see the run end
- **Symptom.** Blocking for hours on a turn that committed and then died. A
  long role times out while a short one hangs. The driver idles after the
  orchestrator exits.
- **Prevented by.** A stale `driver.heartbeat` fails the turn fast, and
  `--agent-turn-timeout` bounds a turn.
- **Open.** Per-turn heartbeats, per-role timeouts, and orchestrator-exit
  detection in the driver (plan 4.6).

### 27. Cap concurrency by what binds, and let one scheduler own the devices
- **Symptom.** Agent streams stall at high parallelism. GPU jobs collide,
  serialize or starve while devices sit idle.
- **Prevented by.** `--max-parallel-smiths` (bounded by default), and
  concurrent dispatch turns never sharing a request id.
- **Open.** Device slots and a priority queue in the spool watcher (plan 4.1).

### 28. No single job, kill or build may take down the allocation or another session
- **Symptom.** One crashed, killed or out-of-memory job takes the whole
  session, and every queued job, with it. A rebuild segfaults processes in
  another allocation.
- **Cause.**
  - Jobs inherited the allocation's PMI context.
  - Jobs had no memory or time limit.
  - Libraries were rewritten in place.
- **Open.**
  - Scrub the PMI and Slurm environment per job, and give each job time and
    memory limits (plan 4.1).
  - Atomic native rebuilds (plan 4.3).
  - Run the orchestrator outside the session it dispatches to; preflight
    warns about this today.

## Commits, delivery and records

### 29. The parallelism boundary is the file or the worktree, never a shared index
- **Symptom.** Concurrent smiths' commits cross-attribute. A gate is queued
  against a file another agent is rewriting.
- **Prevented by.**
  - `bin/commit.sh` (`committer.install_committer`): commits explicit paths
    under a lock and verifies the file list.
  - `--overlap-units` runs the next unit's assembler in its own worktree, and
    adopts the result only if it rebases cleanly.
- **Open.** Index fragments per entry (plan 3.4).

### 30. Commits meet the target repository's policy before the first one
- **Symptom.** Dozens of unsigned commits found at delivery. A re-sign under
  the wrong identity. Sandbox placeholder files committed.
- **Prevented by.**
  - `commit.sh` authors, commits and signs off as the repository's
    configured user.
  - It refuses untracked symlinks, sandbox placeholders, co-author trailers
    and paths outside `commit_policy.allowed_paths`.
  - The CLI runs `committer.trial_commit` before the run starts, and
    delivery audits every commit since the merge base
    (`delivery.check_commit_hygiene`).

### 31. Run every delivery check that needs no finished target early
- **Symptom.** Every unit is green, then delivery fails on test naming, a
  missing accuracy test and a missing test-db entry.
- **Prevented by.** `early_checks.run_early_checks`, as warnings, at the end
  of the plan phase and after each unit.

### 32. Evidence goes stale; key it to the tree and settings it was measured on
- **Symptom.** Delivery accepts a verdict older than the code and settings
  under it, and `HEAD` never receives a full regression.
- **Open.** A results ledger that invalidates on change, and a final
  regression bound to the merged tree's native build (plans 3.1 and 3.2).

### 33. Records are never overwritten, and every claim is traceable
- **Symptom.** Earlier units' decisions overwritten by later ones; duplicate
  dispatch numbers; commit hashes in records that no longer resolve after a
  re-sign.
- **Open.** Decision files keyed by unit, role and turn; atomic journal
  ordinals; reports generated from a ledger (plans 3.3 and 3.4).

### 34. An embargo must be able to name every reachable source of the answer
- **Symptom.** Another engine's implementation of the checkpoint, or this
  workflow's own worked example for it, is one read away from every agent,
  and the spec cannot name it.
- **Prevented by.**
  - `source_embargo` takes repo-relative paths, absolute paths outside the
    repo (as trees), and `package:<path>` inside this package.
  - `embargo.py` enforces it on Claude Code roles with PreToolUse and
    PostToolUse hooks.
  - Roles that hooks cannot reach are given the list as an instruction:
    Codex roles, and turns a driver runs.
  - Every attempt is logged to `<workspace>/embargo/attempts.jsonl`.
  - Running another implementation as an oracle stays allowed.
