<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lessons from real runs

What full onboarding runs of large MoE checkpoints with features, on
multi-GPU Blackwell nodes, taught this workflow. Each lesson gives the
**symptom** an operator or agent saw, the **cause**, and the mechanism that
now **prevents** it. Where the prevention is not in the workflow, **open**
names the gap.

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
  - `preflight.run_preflight` claims a local run's devices (`claim_devices`,
    `probe_device_alone`) and checks the checkpoint and the accuracy anchors
    (`check_checkpoint`, `check_accuracy_anchors`).
  - It also checks `TMPDIR` (`check_tmpdir`) and the multi-rank substrate
    (`check_multirank_substrate`).
  - `substrate_gate` cold-starts the stock path under the run's own runtime
    block.
- **Open.** Nothing proves an agent turn can start before the first one is
  dispatched (see lesson 24).

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
  - Launcher variables unset in contrast legs (`witness.LAUNCHER_EXPORTS`).
  - A session's `PATH` composed with the inherited value
    (`infra/substrate/sessions.py`, `MERGED_ENV_VARS`).

### 5. A skipped or failed measurement must never print like an empty or passing one
- **Symptom.** "0/0 entries certified ✔" against a populated catalog;
  "preflight passed" when the substrate gate never ran; a supplied
  vocabulary that dispatched nothing, read as "no catalog work".
- **Cause.** Summaries did not separate "measured zero" from "did not
  measure", and test fixtures used a different file shape from the real
  index.
- **Prevented by.**
  - The survey's inventory probe runs on a dry run too, states its error in
    the block the plan drafter reads, and renders whatever did land.
  - A supplied `--vocabulary` is validated, and its queue rows are
    dispatched exactly as a drafted one's; a malformed one stops the run.
- **Open.** The dry run should print that the substrate gate was skipped.

### 6. Environment knowledge is an artifact, not something passed by hand
- **Symptom.** Every agent re-derives the same host facts (pytest's exit
  code, the `TMPDIR` bound, which log line is noise). Hand-edited entry
  scripts point at the wrong spool. The container image is imported again
  for every allocation.
- **Prevented by.**
  - `host_facts.md`, seeded by preflight and appended by roles, which
    survives resumes.
  - Lessons (`agents/lessons.py`): what a failed task taught, shown to every
    later dispatch in its scope, in this run and later ones. A lesson that
    works around the workflow's own code (`workflow_defect`) also files a
    flow issue in the workspace's `flow_issues.yaml`, the operator's fix
    list, and is retired, shown to no one, once the commit that fixed it
    (`fixed_by`) is in the running code.
  - Session entry scripts rendered from `environment.sessions`
    (`trtllm-modeling-v2 session render`, `sessions.render_session_entry`).
  - `image_import_warning` and `shares_allocation_with_spool` notes.

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

### 11. Certify against the serving envelope, never against observed values
- **Symptom.** Review loops that never converge. UNCOVERED rows found only
  at the final audit. Crashes or silent corruption beyond the configuration
  the gates ran.
- **Cause.**
  - Coverage was keyed on values the gates happened to dump.
  - Capture sets were derived per entry rather than per call site.
  - Target capacity constants and kernel integer widths were not tied to
    engine limits.
- **Prevented by.**
  - The serving envelope is what `task.yaml`'s runtime block declares
    (`max_batch_size`, `max_num_tokens`, `max_seq_len`, the CUDA-graph
    capture batches). A smith reads its step's `call_sites` in
    `vocabulary.yaml`, multiplies them by every capture batch, makes every
    resulting row a cell, and records the integer-width boundaries the
    entry's index arguments cross as hard limits (the smith prompt).
  - The reviewer checks each capacity constant against the envelope.
- **Not worth its cost: a delivery audit of runtime values.** An auditor
  compared every value the gates dumped against the certified cells, and a
  remediation cycle sent what it found uncovered to verify smiths. Across
  four runs it handled 85 uncovered rows and changed no target, kernel or
  engine behaviour: every fix was a widened cell, a note or a test. It also
  cannot see what never reaches a gate. The crashes that mattered (top-k
  above 8192, an int32 indexer overflow, a token count above
  `max_num_tokens`) were found by running at the envelope's limits by hand,
  and by the smiths' own boundary probes at build time.
  - The envelope boot (`envelope_boot.py`) serves the delivered target at
    those limits on every member.
- **Open.** No check ties a target's capacity constants to engine limits;
  that rests on the reviewer.

### 12. Prove a test can fail
- **Symptom.** Tests and floors pass a deliberately wrong implementation. A
  hypothesis is "confirmed" by a count that both hypotheses predict.
- **Cause.** No negative control, and thresholds chosen without measurement.
- **Open.** The kernel skills should refuse a receipt without a negative
  control, and floors should sit above the best wrong control.

### 13. Recurring outcomes get typed decisions, not prose
- **Symptom.** The operator picks redirects, verify requests and spec fixes
  out of prose and dispatches them by hand.
- **Prevented by.** Planner events: a finding reaches the planner's next turn,
  which turns it into a task.
- **Open.** A typed `spec_change` the orchestrator applies and records.

## Gates

### 14. Gate every configuration a target claims
- **Symptom.** A target serving several parallel configurations is green,
  but its expert-parallel and attention-DP branches never ran.
- **Cause.** The gate folded a unified target to tensor parallelism, and the
  member set was prose.
- **Prevented by.** `members:` with `members_for`; each verdict is named
  `<gate>@<member>`; boot, accuracy, the witness, acceptance and the envelope
  boot run at every member, cheapest first (`order_members_by_cost`).

### 15. A gate records, and owns, the code it measured
- **Symptom.** A number is about to be credited to a commit it was not
  measured on. Rank 0 runs one checkout while the workers run another.
- **Cause.** Gates import whatever is on disk, and MPI workers resolve the
  package from their working directory ahead of `PYTHONPATH`.
- **Prevented by.**
  - `gates.inspect_tree` records the tree in every verdict and refuses a
    dirty target path unless `MV2FLOW_ALLOW_DIRTY_GATE` is set, which marks
    the verdict as such. A member's overlay is never written into the
    source tree.
  - `trtllm-modeling-v2 gate --tree <checkout>` measures another tree. The
    scheduler's resolve hook records which checkout each process imported,
    and refuses a result whose ranks disagree or that did not import the
    declared tree (`reason=mixed_checkout`).

### 16. Gates run with `modeling_v2=require`, and each unit's gates take its own path
- **Symptom.** A feature unit's boot never exercises the feature. The
  serving path returns errors no gate saw. A number measured through
  fallback reads as the target's.
- **Cause.** Target selection is the LLM-API argument `modeling_v2`
  (default `"off"`); the engine reads no environment variable for it, so
  exporting one selects nothing and every gate serves the built-in model.
- **Prevented by.**
  - The workflow carries the mode to its own harness as
    `TRTLLM_MODELING_V2`, and the harness turns it into the argument:
    `eval_command` merges `modeling_v2: <mode>` into the member's options
    YAML, and the engine driver passes it to the quickstart's `LLM(...)`
    (test `test_the_accuracy_command_selects_the_target_through_the_llm_api`).
  - The speculator depth comes from the unit's own options YAML
    (`resolve_max_draft_len`), and the unit's own LLM-API options file
    (`accuracy.llm_api_options`) is used.
  - Each unit has its own boot workload (`gates.resolve_boot_spec`); a
    multimodal unit is sent an image, and keywords are matched on generated
    text only.
  - `preflight.check_speculation_support` refuses a speculation algorithm the
    gates' launcher has no branch for, instead of letting it run without a
    speculator.

### 17. Read failures from the whole log, and results from explicit markers
- **Symptom.** The gate shows only a generic re-raise. A keyword matches a
  log line rather than the answer. Reviewers hand-parse interleaved
  multi-rank stdout.
- **Prevented by.**
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
  - `accuracy.acceptance` declares the workload (`num_prompts`, default 32,
    from the accuracy dataset; or `prompts`/`prompts_file`), `max_tokens`
    and `min_draft_iterations`. Fewer than `min_prompts` prompts, or too few
    draft iterations, fail as such, not as a low rate.
  - Each run is compared prompt by prompt with the previous verdict through
    a paired test (`accuracy.acceptance.regression`).
  - A floor with no measured source (`floor_source` or
    `measured_reference`) is warned about.
  - Every unit whose target keeps speculation on is gated, not only the unit
    that adds it.

### 19. Accuracy floors are statistical, and generation steps need a golden check
- **Symptom.** A defect present in every member passes every gate.
- **Cause.** The floor sat several standard errors below the anchor, and
  golden checks covered context steps only.
- **Prevented by.**
  - `accuracy.floor` with `rule: statistical` (the default): the floor sits
    `k` (default 2) standard errors below the anchor.
  - A unit whose every member scores more than 1 SE below the anchor is
    recorded as a finding (`gates.note_systematic_shift`), and delivery
    refuses it (`delivery-failed:findings`) until
    `accuracy.floor.acknowledge.<unit>` gives the reason.
- **Not worth its cost: two gates.** A scheduler-equivalence gate (overlap
  scheduler on vs off, token-identical at batch 1) and a launch-path smoke
  through `trtllm-serve` ran 17 and 16 times across the historical runs,
  caught no target defect, and each failed once falsely (a Triton autotune
  difference, a scorer reading the last number), costing about 2.7 h of
  stale-verdict waves. Both are removed. The defects such a comparison did
  find were found by agents comparing legs while developing.

### 20. Add a cheap performance gate
- **Symptom.** An accuracy-neutral change regresses the serving path by
  double digits, and only a profile catches it.
- **Open.** One prefill chunk plus one decode step, compared with the
  previous tree; fail on more than 5% unless the commit declares a
  trade-off. The engine driver's timing mode exists; no gate uses it.

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
  - The residency reading is the target's to print (`__DEVICE_MEMORY_MIB__`,
    before the KV pool is allocated); a leg without it fails the witness.
  - Conflicting kinds refused by the schema (`MTP_NAMESPACE_FEATURES`).

### 22. A contrast must be real and differ in exactly one variable
- **Symptom.** A witness compares a run with itself. Desynchronized legs
  read as transfer defects. Identity fails by construction.
- **Prevented by.**
  - The equivalence witness's leg B is a real disaggregated hop: the flow's
    own `checks/harness/disagg_leg_b.py` unless the spec names a
    `contrast_command`, which then needs its `contrast_env`.
  - Leg B gets leg A's workload through `MV2FLOW_WITNESS_*` variables, with
    absolute paths.
  - `legs_env` pins what both legs share.
  - The assembler's turn prompt names the workflow's own entry for
    reproducing a gate or witness (`gate_cli.invocation`:
    `trtllm-modeling-v2 gate ...`), never a reconstruction of its command.

### 23. A token comparison must be able to see a difference, and say where
- **Symptom.** Witnesses on 2–3-token continuations; prefix-only text
  matches; near-tie flips caused by batching; guessed timeouts. An exact
  scheduler-equivalence pair failed, and the verdict blamed stale state,
  while the same configuration gave two outputs across processes because
  each process timed its own `@triton.autotune` configs.
- **Prevented by.**
  - `witness.compare_continuations` compares token ids, then text, then
    prefix, and quotes the first difference from both legs.
  - Only token ids can pass (`witness.missing_token_ids`), so
    `min_generated_tokens` holds whatever the legs print.
  - A witness block's `max_batch_size` applies to both legs; at 1, requests
    go one at a time.
  - A per-leg `timeout`.
  - `gates.pin_tuning` makes every leg of a gate run share the AutoTuner's
    tactics and Triton's autotuning, emptied at the start of each gate run.
    The token-comparing witnesses use it, so a task-level Triton cache is
    not needed.
- **Open.** A concurrent, batch > 1 leg beside the batch-1 one.

## Transport, resume and concurrency

### 24. When a host cannot authenticate an agent, move the turn and nothing else
- **Symptom.** Every turn fails with zero tokens, as `EROFS ...
  credentials.lock` or `Not logged in` after about forty minutes, which
  reads as a model refusal.
- **Cause.** A read-only `$HOME`, then a single-use refresh token shared
  between copies.
- **Prevented by.**
  - `--agent-transport external`: an already-authenticated session runs the
    turns, and the stage machine, decision validation and gates stay in the
    orchestrator. Staging a writable home with a copied credential store
    lasted minutes, for the token reason above.
  - Decisions travel as files on both transports, so no turn depends on a
    session to host a tool in.

### 25. Resume and re-entry must not cost work or budget
- **Symptom.** A restart re-dispatches finished or in-flight turns, charges
  turns nobody answered, or closes a run a later repair could reopen. A gate
  whose compute session died closes the run as if the target had failed.
- **Prevented by.**
  - A resumed process adopts a smith's decision file, and waits on an
    unanswered twin request instead of asking again. On the external
    transport it also adopts every answered turn of a task that had no
    outcome yet, and collects the decision where that turn wrote it;
    `driver status` marks those turns `ADOPTED`. A turn is consumed once its
    task is done, failed or abandoned, so a task reopened later is asked
    afresh.
  - Verdicts are kept in `contract_status.jsonl` under the fingerprint of
    the tree they measured (lesson 32), and a check the contract already
    counts as passed on the current tree is not run again.
  - A tune task run again, after a restart or as a retry, takes its campaign
    back from `state.tune` and asks no turn whose answer is on disk. It never
    adopts a failed job receipt, and from its second attempt queues its jobs
    under names of its own.
  - A unit whose `UNIT_READY` still describes the target
    (`_unit_work_unchanged`, git asked from the orchestrator's own process,
    never through the gate runner) goes straight to its gates.
  - `REMEDIABLE_VERDICTS` (including `assembler-blocked:knowledge_gap` /
    `:environment` and `session-lost`) re-enter with the turn budget
    restored.
  - A gate lost to its session is re-run once a session heart-beats again,
    bounded by `environment.session_wait` and `SESSION_LOSS_RERUNS`; past
    either bound the run closes `session-lost:<unit>`.
  - A `smith-blocked` run whose failures are feature-only re-enters.
  - A delivered run whose spec grew re-enters at the new unit.
  - The spec is re-read on every launch (`_refresh_workspace_spec`).

### 26. Liveness is per turn, timeouts are per role, and the driver must see the run end
- **Symptom.** Blocking for hours on a turn that committed and then died. A
  long role times out while a short one hangs. The driver idles after the
  orchestrator exits.
- **Prevented by.**
  - A stale `driver.heartbeat` fails the turn fast.
  - Each role has its own turn timeout (`DEFAULT_ROLE_TURN_TIMEOUTS`);
    `--agent-turn-timeout` overrides them all.
  - The orchestrator beats `orchestrator.heartbeat`, and `driver wait`
    returns `ORCHESTRATOR_EXITED` when it stops.
- **Open.** Liveness is per driver, not per turn.

### 27. Cap concurrency by what binds, and let one scheduler own the devices
- **Symptom.** Agent streams stall at high parallelism. GPU jobs collide,
  serialize or starve while devices sit idle. Tasks wait for devices that
  only gate jobs hold.
- **Prevented by.**
  - `driver wait` hands out at most 16 worker turns at once
    (`DEFAULT_MAX_INFLIGHT`), whatever `--max-parallel-tasks` (default 16)
    allows, the planner's and the critic's beside them, and concurrent
    dispatch turns never share a request id.
  - Tasks are admitted by capacity: the live sessions' devices less the
    `gpus` the running tasks declare, not the devices free at that moment.
    An agent turn holds no device; its gate jobs queue in the spools.
  - Checks run on one bounded pool shared by every task, and the planner's
    turn and the critic's read on threads of their own, so none of them
    holds tasks from starting.
  - The spool scheduler (`infra/substrate/scheduler.py`): one priority queue,
    device slots exported as `CUDA_VISIBLE_DEVICES`, and a waiting
    multi-device job holding the devices it needs.

### 28. No single job, kill or build may take down the allocation or another session
- **Symptom.** One crashed, killed or out-of-memory job takes the whole
  session, and every queued job, with it. A rebuild segfaults processes in
  another allocation.
- **Cause.**
  - Jobs inherited the allocation's PMI context.
  - Jobs had no memory or time limit.
  - Libraries were rewritten in place.
- **Prevented by.**
  - The scheduler scrubs the PMI and Slurm environment from every job
    (`SCRUB_PATTERN`), and gives each a wall limit and a resident-memory
    limit. It never signals a process that still carries the allocation's
    MPI context (the safe-kill rule).
  - `infra/scripts/incremental_build.sh` announces the rebuild to every session
    and gives every installed library a fresh inode.
  - Preflight warns when the orchestrator shares the allocation it
    dispatches to (`shares_allocation_with_spool`).

## Commits, delivery and records

### 29. The parallelism boundary is the file or the worktree, never a shared index
- **Symptom.** Concurrent smiths' commits cross-attribute. A gate is queued
  against a file another agent is rewriting. Unrelated tasks run one after
  another because their `touches` name a shared directory.
- **Prevented by.**
  - Each task works on its own branch in its own worktree and is rebased
    onto the checkout when it finishes; a rebase that conflicts goes back to
    its worker to resolve (`integration.rebase_block`). So two tasks are
    kept apart only when both name the very same file in `touches`, or
    either is `exclusive`.
  - `bin/commit.sh` (`committer.install_committer`): commits explicit paths
    under a lock and verifies the file list.
  - A gate refuses a dirty target path (`gates.inspect_tree`).
  - The catalog is its entry files: modeling_v2 keeps no catalog index, so a
    smith's commit is one entry and its test, and no file two smiths share.
  - Each tune candidate is its own worktree (`tune/worktrees/<slug>`).

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
- **Open.** The delivery checks that need no finished target (test naming,
  the accuracy test and its test-db listing) run only at delivery.

### 32. Evidence goes stale; key it to the tree and settings it was measured on
- **Symptom.** Delivery accepts a verdict older than the code and settings
  under it, and `HEAD` never receives a full regression. The opposite costs
  too: a commit that only splits a test-list entry makes every verdict
  stale, and checks re-run on a tree no gate sees differently.
- **Prevented by.**
  - Every verdict records a fingerprint of the tree the gates measure
    (`workflow._target_marker`): the commit the checkout is at, less
    `tests/`, `docs/`, `.github/`, `examples/` and `*.md`
    (`gate_relevant`), but with the files a gate reads there
    (`gate_inputs`), and with the native libraries' sizes and modification
    times. A catalog entry or a native rebuild makes every verdict stale; a
    test list does not.
  - A check measures a read-only snapshot of that commit, never the
    checkout integrations move while it runs (`run/snapshots.py`), and its
    verdict records what each command imported: one whose legs imported two
    trees, or a checkout no commit pins, is void. A two-leg witness taken on
    the live checkout once ran its legs on two commits and passed.
  - Delivery evaluates the contract against the fingerprint of the tree as
    it is, and a verdict under another is missing
    (`delivery-failed:contract`). The delivery tier runs once, on the final
    tree.
  - Boot and accuracy checks that run one command on one engine against one
    criterion share a pass (`contract.equivalence_key`); failures are never
    shared.
- **Open.** Nothing invalidates a verdict when a value the spec's
  `environment` block sets (a `TRTLLM_*` variable, say) changes.

### 33. Records are never overwritten, and every claim is traceable
- **Symptom.** Earlier units' decisions overwritten by later ones; duplicate
  dispatch numbers; commit hashes in records that no longer resolve after a
  re-sign.
- **Prevented by.**
  - Decision files are keyed by unit, role and turn
    (`decisions/<unit>-<role>-<turn>.yaml`, `progress.decision_path`); a
    reused key gets `.2`, `.3` and so on, claimed atomically.
  - Journal dispatch ordinals are claimed atomically
    (`journal.next_ordinal`).
- **Open.** Commit hashes in the journal and reports are not re-resolved
  after a history rewrite.

### 34. An embargo must be able to name every reachable source of the answer
- **Symptom.** Another engine's implementation of the checkpoint, or this
  workflow's own worked example for it, is one read away from every agent,
  and the spec cannot name it.
- **Prevented by.**
  - `source_embargo` takes repo-relative paths, absolute paths outside the
    repo (as trees), and `package:<path>` inside this package; the schema
    refuses an entry that does not exist (`task_schema.py`).
  - Every role prompt restates the list
    (`prompts._common.source_embargo_block`).
  - Running another implementation as an oracle stays allowed.
- **Open.** The embargo is a rule in the role prompts only: no tool hook
  refuses a read of an embargoed path, and no read is logged.

## Performance (the tune stage)

### 35. Hosts differ by more than knobs do; judge on the same host
- **Symptom.** Six unrelated config rungs (a GC threshold, early first
  token, stream interval, a token budget, graph sizes, ...) all measured 3%
  to 4% worse at concurrency 1, and the ladder rejected a +4.8% rung as a
  loss.
- **Cause.** Each rung had run on one of the nodes that were slower for
  identical configs; the noise floor was the cross-host spread.
- **Prevented by.** `campaign.Campaign._control_for(host)`: every measurement is judged
  against a control of the same config measured on the same host; the noise
  floor is the worst same-host spread, never the cross-host one.

### 36. A profiler that cannot finish is worse than none
- **Symptom.** nsys captures that never wrote a report; the server idle at
  0% GPU forever after `Profiling started at iteration N`.
- **Cause.** The in-process `torch.cuda.cudart().cudaProfilerStop()` holds
  the GIL while CUPTI waits for GPU work that other Python threads must
  complete (gdb: every other thread in `PyGILState_Ensure`). A driver-level
  `cuProfilerStop` did not return on that server either, with node- and
  graph-level graph tracing alike.
- **Prevented by.** `bench.py --nsys` defaults to session mode: `nsys launch`
  injects, `nsys start`/`stop` from outside the server time the window, and
  nothing in the server calls the profiler API.

### 37. A job's processes include the ones that left its session
- **Symptom.** A server left by a cancelled capture held most of a GPU's
  memory during the next measurement.
- **Cause.** `nsys profile` and a server started with `start_new_session`
  are re-parented to init when the job's leader exits, and the scheduler
  found a job's processes by session and ancestry only.
- **Prevented by.**
  - `scheduler.processes_with_marker`: the job's environment markers
    (spool, watcher and job) survive `setsid`, so a job's leftovers are
    found and stopped with it.
  - `bench.py` turns SIGTERM into its teardown and refuses a device that
    already has memory in use.

### 38. Module knobs never reach a target
- **Symptom.** A config rung that selects a kernel (`moe_config.backend`,
  `attn_backend`, `use_cute_dsl_*`) measures as noise on the target.
- **Cause.** A modeling_v2 target pins its kernels through the catalog.
- **Prevented by.** The serving config is derived from the workload and
  carries no module knob; a kernel choice for a target is a catalog change,
  made by a tuner candidate.

### 39. Measure candidates stacked on the current best
- **Symptom.** Candidates written in parallel from one base, each measured
  against the base, would be judged against a control that no longer
  exists once the first keep moves the best tree.
- **Prevented by.** `campaign.Campaign._stack`: an `eval-<slug>` worktree is
  the best tree plus the candidate's commits; a keep that moves the best
  tree re-stacks and re-measures the others before their verdict.
