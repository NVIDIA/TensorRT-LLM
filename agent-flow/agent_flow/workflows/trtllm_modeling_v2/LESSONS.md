<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lessons from real runs

What full onboarding runs of large MoE checkpoints with features, on multi-GPU
Blackwell nodes, taught this workflow. Each lesson gives the **symptom** an
operator or agent saw, the **cause**, and the mechanism that **prevents** it;
where the prevention is not in the workflow, **open** names the gap. Most of
these presented as a model or target defect, and none were. For verbatim error
text, use [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Substrate and preflight

### 1. Prove the host before any agent exists
- **Symptom.** Hours into a run, a turn or gate fails with `MPI_ERR_SPAWN`,
  a kvCache that cannot fit a sequence, or NVRTC missing `cuda.h`, and it reads
  as a target defect.
- **Cause.** The substrate was first exercised by the target itself.
- **Prevented by.**
  - `preflight.run_preflight` checks the checkpoint and the accuracy anchors
    (`check_checkpoint`, `check_accuracy_anchors`), `TMPDIR` (`check_tmpdir`),
    a local run's devices (`claim_devices`, `probe_device_alone`) and the
    multi-rank substrate (`check_multirank_substrate`).
  - `substrate_gate` cold-starts the stock path under the run's own runtime
    block.
  - On the in-process transport, the agent probe (`probe_agents`) proves every
    agent can take a turn here and run its Bash tool.
- **Open.** Nothing proves an external driver can take a turn before the first
  one is dispatched.

### 2. A new architecture has no stock path, so the substrate check needs a proxy
- **Symptom.** Preflight, or a stock-path gate leg, stops on "unknown
  architecture", the condition onboarding exists for.
- **Cause.** The check booted the run's own checkpoint, and "unsupported" was
  matched by a marker list that missed refusals from the registry,
  `transformers`' unknown `model_type`, custom code and registration shims.
- **Prevented by.**
  - `environment.substrate_checkpoint` boots a proxy checkpoint instead.
  - `UNSUPPORTED_ARCHITECTURE_MARKERS` / `reports_unsupported_architecture`
    cover each refusal path.
  - The acceptance gate falls back to a declared `min_absolute` floor and
    records the weaker claim.

### 3. A multi-rank launch inside someone else's allocation has rules; generate it in one place
- **Symptom.** "Not enough slots"; N ranks that each believe they are the
  leader and hang; a ZeroMQ IPC path longer than 107 characters.
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
- **Cause.** Every command was wrapped for multi-rank; a spec `PATH` replaced
  the inherited one; launcher variables leaked into child engines.
- **Prevented by.**
  - Single-process commands for the survey probe and witness contrast legs
    (`single_process=True`).
  - Launcher variables unset in contrast legs (`witness.LAUNCHER_EXPORTS`).
  - A session's `PATH` composed with the inherited value
    (`infra/substrate/sessions.py`, `MERGED_ENV_VARS`).

### 5. A skipped or failed measurement must never print like an empty or passing one
- **Symptom.** "0/0 entries certified ✔" against a populated catalog;
  "preflight passed" when the substrate gate never ran; a vocabulary that
  dispatched nothing, read as "no catalog work".
- **Cause.** Summaries did not separate "measured zero" from "did not
  measure", and test fixtures used a different file shape from the real index.
- **Prevented by.**
  - The survey's inventory probe runs on a dry run too, states its error in the
    block the surveyor reads, and renders whatever did land.
  - A survey's `vocabulary.yaml` is validated before its queue rows become
    kernel tasks; a malformed one fails the survey task.
- **Open.** The dry run should print that the substrate gate was skipped.

### 6. Environment knowledge is an artifact, not something passed by hand
- **Symptom.** Every agent re-derives the same host facts (pytest's exit code,
  the `TMPDIR` bound, which log line is noise). Hand-edited entry scripts point
  at the wrong spool. The container image is imported again for every
  allocation.
- **Prevented by.**
  - `host_facts.md`, seeded by preflight and appended by roles, which survives
    resumes.
  - Session entry scripts rendered from `environment.sessions`
    (`trtllm-modeling-v2 session render`, `sessions.render_session_entry`).
  - `image_import_warning` and `shares_allocation_with_spool` notes.
- **Open.** Nothing carries what one run learned to the next: `host_facts.md`
  lives in its run's workspace.

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
  `reference_yaml` row for its model key and discriminators before any agent
  turn.

### 9. A role's backend must travel with its turn
- **Symptom.** A review was run on the surveyor's own model family and recorded
  as cross-model.
- **Cause.** The reviewer's backend did not travel with an externally
  dispatched turn.
- **Prevented by.** Each `dispatch/<id>.request.json` names the role's backend
  and model, and per-role routing (`roles.py`) can send a role to another
  provider.
- **Open.** Record in the run's own record when the answering model cannot be
  verified.

## Kernel writer and catalog

### 10. Schedule a kernel row by the unit that needs it
- **Symptom.** The run closes before core over an entry only a later feature
  needs.
- **Prevented by.** `unit:` on queue rows. Every row's kernel task starts when
  the survey lands, a failed one goes back to the planner rather than closing
  the run, and a row only an optional unit needs is `nice_to_have`.

### 11. Certify against the serving envelope, never against observed values
- **Symptom.** Review loops that never converge; uncovered rows found only at
  the final audit; crashes or silent corruption beyond the configuration the
  gates ran (top-k above 8192, an int32 indexer overflow, a token count above
  `max_num_tokens`).
- **Cause.** Coverage was keyed on values the gates happened to dump; capture
  sets were derived per entry rather than per call site; capacity constants and
  kernel integer widths were not tied to engine limits.
- **Prevented by.**
  - The serving envelope is what `task.yaml`'s runtime block declares
    (`max_batch_size`, `max_num_tokens`, `max_seq_len`, the CUDA-graph capture
    batches). A kernel writer multiplies its step's `call_sites` in
    `vocabulary.yaml` by every capture batch, makes every resulting row a cell,
    and records the integer-width boundaries the entry's index arguments cross
    as hard limits.
  - The `assemble` tag has each capacity constant derived from an engine
    argument and checked against `runtime_facts.yaml`.
  - The envelope boot (`envelope_boot.py`) serves the delivered target at those
    limits on every member.
- **Open.** No check ties a target's capacity constants to engine limits; that
  rests on the worker that writes them, and the envelope boot.

### 12. Prove a test can fail
- **Symptom.** Tests and floors pass a deliberately wrong implementation. A
  hypothesis is "confirmed" by a count that both hypotheses predict.
- **Cause.** No negative control, and thresholds chosen without measurement.
- **Open.** The entry skill asks a widened band for its wrong variants'
  distances; nothing refuses a receipt without a negative control, and floors
  should sit above the best wrong control.

### 13. Recurring outcomes get typed decisions, not prose
- **Symptom.** The operator picks redirects, verify requests and spec fixes
  out of prose and dispatches them by hand.
- **Prevented by.** Planner events: a finding reaches the planner's next turn,
  which turns it into a task.
- **Open.** A typed `spec_change` the orchestrator applies and records.

## Gates

### 14. Gate every configuration a target claims
- **Symptom.** A target serving several parallel configurations is green, but
  its expert-parallel and attention-DP branches never ran.
- **Cause.** The gate folded a unified target to tensor parallelism, and the
  member set was prose.
- **Prevented by.** `members:` with `members_for`; each verdict is named
  `<gate>@<member>`; boot, the logit gate, the witness, acceptance and the
  envelope boot run at every member, cheapest first (`order_members_by_cost`),
  and accuracy at every member the logit gate cannot stand for.

### 15. A gate records, and owns, the code it measured
- **Symptom.** A number is about to be credited to a commit it was not measured
  on. Rank 0 runs one checkout while the workers run another.
- **Cause.** Gates import whatever is on disk, and MPI workers resolve the
  package from their working directory ahead of `PYTHONPATH`.
- **Prevented by.**
  - `gates.inspect_tree` records the tree in every verdict and refuses a dirty
    target path unless `MV2FLOW_ALLOW_DIRTY_GATE` is set, which marks the
    verdict as such. A member's overlay is never written into the source tree.
  - `trtllm-modeling-v2 gate --tree <checkout>` measures another tree. The
    scheduler's resolve hook records which checkout each process imported, and
    refuses a result whose ranks disagree or that did not import the declared
    tree (`reason=mixed_checkout`).

### 16. Gates run with `modeling_v2=require`, and each unit's gates take its own path
- **Symptom.** A feature unit's boot never exercises the feature. The serving
  path returns errors no gate saw. A number measured through fallback reads as
  the target's.
- **Cause.** Target selection is the LLM-API argument `modeling_v2` (default
  `"off"`); the engine reads no environment variable for it, so exporting one
  selects nothing.
- **Prevented by.**
  - The workflow carries the mode to its own harness as `TRTLLM_MODELING_V2`,
    and the harness turns it into the argument: `eval_command` merges
    `modeling_v2: <mode>` into the member's options YAML, and the engine driver
    passes it to the quickstart's `LLM(...)`
    (`test_gates.py::test_the_accuracy_gate_hands_require_to_the_engine`).
  - The speculator depth comes from the unit's own options YAML
    (`resolve_max_draft_len`), and each unit uses its own LLM-API options file
    (`accuracy.llm_api_options`).
  - Each unit has its own boot workload (`gates.resolve_boot_spec`); a
    multimodal unit is sent an image, and keywords are matched on generated
    text only.
  - `preflight.check_speculation_support` refuses a speculation algorithm the
    gates' launcher has no branch for.

### 17. Read failures from the whole log, and results from explicit markers
- **Symptom.** The gate shows only a generic re-raise. A keyword matches a log
  line rather than the answer. Reviewers hand-parse interleaved multi-rank
  stdout.
- **Prevented by.** Witness token ids and text printed on marked lines
  (`TRTLLM_WITNESS_FULL_OUTPUT`).
- **Open.** Classify from the first traceback in the full output, not from the
  display tail.

### 18. An acceptance gate needs a workload that drafts, and a floor that means something
- **Symptom.** Acceptance pinned at 1.00 for every drafter, working or broken.
  Floors that pass wrong drafters. Regressions flagged on noise.
- **Cause.** The workload's answers ended in 2-3 tokens; the floor was taken
  from the ceiling; a 4-prompt comparison used a fixed delta.
- **Prevented by.**
  - `accuracy.acceptance` declares the workload (`num_prompts`, default 32,
    from the accuracy dataset; or `prompts`/`prompts_file`), `max_tokens` and
    `min_draft_iterations`. Fewer than `min_prompts` prompts, or too few draft
    iterations, fail as such, not as a low rate.
  - Each run is compared prompt by prompt with the previous verdict through a
    paired test (`accuracy.acceptance.regression`).
  - A floor with no measured source (`floor_source` or `measured_reference`) is
    warned about.
  - Every unit whose target keeps speculation on is gated, not only the unit
    that adds it.

### 19. Accuracy floors are statistical, and generation steps need a golden check
- **Symptom.** A defect present in every member passes every gate.
- **Cause.** The floor sat several standard errors below the anchor, and golden
  checks covered context steps only.
- **Prevented by.**
  - `accuracy.floor` with `rule: statistical` (the default): the floor sits
    `k` (default 2) standard errors below the anchor.
  - A unit whose every member scores more than 1 SE below the anchor is recorded
    as a finding (`gates.note_systematic_shift`), and delivery refuses it
    (`delivery-failed:findings`) until `accuracy.floor.acknowledge.<unit>` gives
    the reason.
  - The logit gate and the boot's greedy tokens hold generation steps to the HF
    reference.

### 20. Add a cheap performance gate
- **Symptom.** An accuracy-neutral change regresses the serving path by double
  digits, and only a profile catches it.
- **Open.** One prefill chunk plus one decode step, compared with the previous
  tree; fail on more than 5% unless the commit declares a trade-off. The engine
  driver's timing mode exists; no gate uses it.

## Witnesses

### 21. A feature the standing gates cannot see needs its own kind and witness
- **Symptom.** An output-identical feature, or one trained to stay inside the
  accuracy tolerance, passes every gate whether or not it was built. The nearest
  existing kind builds the wrong thing.
- **Prevented by.**
  - A closed `features:` list (`FEATURE_FAMILIES`), each family's checks wired
    by name.
  - `WITNESS_GATED_UNITS`: residency, equivalence, replay and kv_quant
    witnesses, owed when `accuracy.witness.enabled` (or `profile: certified`)
    turns them on.
  - Counters the target must print (`MODELING_V2_WITNESS` under
    `TRTLLM_MODELING_V2_WITNESS=1`), where a missing counter is a failure.
  - The residency reading is the target's to print (`__DEVICE_MEMORY_MIB__`,
    before the KV pool is allocated); a leg without it fails the witness.
  - Conflicting kinds refused by the schema (`MTP_NAMESPACE_FEATURES`).

### 22. A contrast must be real and differ in exactly one variable
- **Symptom.** A witness compares a run with itself. Desynchronized legs read
  as transfer defects. Identity fails by construction.
- **Prevented by.**
  - The equivalence witness's leg B is a disaggregated hop the run launches
    itself (`contrast_command`), and the spec is refused without it and the
    `contrast_env` that sets it apart (`EQUIVALENCE_WITNESS_REQUIRED`).
  - Leg B gets leg A's workload through `MV2FLOW_WITNESS_*` variables, with
    absolute paths.
  - `legs_env` pins what both legs share.
  - The assembler's turn prompt names the workflow's own entry for reproducing
    a gate or witness (`gate_cli.invocation`: `trtllm-modeling-v2 gate ...`),
    never a reconstruction of its command.

### 23. A token comparison must be able to see a difference, and say where
- **Symptom.** Witnesses on 2-3-token continuations; prefix-only text matches;
  near-tie flips caused by batching; guessed timeouts; one configuration giving
  two outputs across processes because each process timed its own
  `@triton.autotune` configs.
- **Prevented by.**
  - `witness.compare_continuations` compares token ids, then text, then prefix,
    and quotes the first difference from both legs.
  - Only token ids can pass (`witness.missing_token_ids`), so
    `min_generated_tokens` holds whatever the legs print.
  - A witness block's `max_batch_size` applies to both legs; at 1, requests go
    one at a time.
  - A per-leg `timeout`.
  - `gates.pin_tuning` makes every leg of a gate run share the AutoTuner's
    tactics and Triton's autotuning, emptied at the start of each gate run.
- **Open.** A concurrent, batch > 1 leg beside the batch-1 one.

## Transport, resume and concurrency

### 24. When a host cannot authenticate an agent, move the turn and nothing else
- **Symptom.** Every turn fails with zero tokens, as `EROFS ...
  credentials.lock` or, after about forty minutes, `Not logged in`, which reads
  as a model refusal.
- **Cause.** A read-only `$HOME`, then a single-use refresh token shared
  between copies.
- **Prevented by.**
  - `--agent-transport external`: an already-authenticated session runs the
    turns, and decision validation and the gates stay in the orchestrator.
  - Decisions travel as files on both transports, so no turn depends on a
    session to host a tool in.

### 25. Resume and re-entry must not cost work or budget
- **Symptom.** A restart re-dispatches finished or in-flight turns, charges
  turns nobody answered, or closes a run a later repair could reopen. A gate
  whose compute session died closes the run as if the target had failed.
- **Prevented by.**
  - A resumed process adopts a kernel writer's decision file, and waits on an
    unanswered twin request instead of asking again. On the external transport
    it also adopts every answered turn of a task that had no outcome yet
    (`ADOPTED` in `driver status`); a turn is consumed once its task is done,
    failed or abandoned, so a task reopened later is asked afresh.
  - Verdicts are kept in `contract_status.jsonl` under the fingerprint of the
    tree they measured (lesson 32), and a check the contract already counts as
    passed on the current tree is not run again.
  - A tune task run again takes its campaign back from `state.tune` and asks no
    turn whose answer is on disk; it never adopts a failed job receipt, and from
    its second attempt queues its jobs under names of its own.
  - A run closed with a verdict in `REMEDIABLE_VERDICTS` (`delivery-failed`,
    `session-lost`, `escalated`, `planner-stalled`, `planner-failed`, ...)
    re-enters when the same command runs again.
  - A gate lost to its session is re-run once a session heart-beats again,
    bounded by `environment.session_wait` and `SESSION_LOSS_RERUNS`; past
    either bound the run closes `session-lost:<unit>`.
  - The spec is re-read on every launch (`_refresh_workspace_spec`).

### 26. Liveness is per turn, timeouts are per role, and the driver must see the run end
- **Symptom.** Blocking for hours on a turn that committed and then died. A
  long role times out while a short one hangs. The driver idles after the
  orchestrator exits.
- **Prevented by.**
  - Each role has its own turn timeout (`DEFAULT_ROLE_TURN_TIMEOUTS`);
    `--agent-turn-timeout` overrides them all, and `driver status` marks a
    claimed turn past it `OVERDUE`.
  - The orchestrator beats `orchestrator.heartbeat`, and `driver wait` returns
    `ORCHESTRATOR_EXITED` when it stops.
- **Open.** A turn whose driver died is caught only by its role's timeout.

### 27. Cap concurrency by what binds, and let one scheduler own the devices
- **Symptom.** Agent streams stall at high parallelism. GPU jobs collide,
  serialize or starve while devices sit idle. Tasks wait for devices that only
  gate jobs hold.
- **Prevented by.**
  - `driver wait` hands out at most 8 worker turns at once
    (`DEFAULT_MAX_INFLIGHT`), whatever `--max-parallel-tasks` (default 8)
    allows, the planner's beside them, and concurrent dispatch turns never
    share a request id.
  - Tasks are admitted by capacity: the live sessions' devices less the `gpus`
    the running tasks declare. An agent turn holds no device; its gate jobs
    queue in the spools.
  - Checks run on one bounded pool shared by every task, and the planner's
    turn on a thread of its own, so neither holds tasks from starting.
  - The spool scheduler (`infra/substrate/scheduler.py`): one priority queue,
    device slots exported as `CUDA_VISIBLE_DEVICES`, and a waiting
    multi-device job holding the devices it needs.

### 28. No single job, kill or build may take down the allocation or another session
- **Symptom.** One crashed, killed or out-of-memory job takes the whole session,
  and every queued job, with it. A rebuild segfaults processes in another
  allocation.
- **Cause.** Jobs inherited the allocation's PMI context; jobs had no memory or
  time limit; libraries were rewritten in place.
- **Prevented by.**
  - The scheduler scrubs the PMI and Slurm environment from every job
    (`SCRUB_PATTERN`), gives each a wall limit and a resident-memory limit, and
    never signals a process that still carries the allocation's MPI context
    (the safe-kill rule).
  - `infra/scripts/incremental_build.sh` announces the rebuild to every session
    and gives every installed library a fresh inode.
  - Preflight warns when the orchestrator shares the allocation it dispatches
    to (`shares_allocation_with_spool`).

## Commits, delivery and records

### 29. The parallelism boundary is the file or the worktree, never a shared index
- **Symptom.** Concurrent kernel writers' commits cross-attribute. A gate is
  queued against a file another agent is rewriting. Unrelated tasks run one
  after another because their `touches` name a shared directory.
- **Prevented by.**
  - Each task works on its own branch in its own worktree and is rebased onto
    the checkout when it finishes; a conflict goes back to its worker
    (`integration.rebase_block`). Two tasks are kept apart only when both name
    the very same file in `touches`, or either is `exclusive`.
  - `bin/commit.sh` (`committer.install_committer`) commits explicit paths
    under a lock and verifies the file list.
  - A gate refuses a dirty target path (`gates.inspect_tree`).
  - The catalog is its entry files, with no index, so a kernel writer's commit
    is one entry and its test.
  - Each tune candidate is its own worktree (`tune/worktrees/<slug>`).

### 30. Commits meet the target repository's policy before the first one
- **Symptom.** Dozens of unsigned commits found at delivery. A re-sign under
  the wrong identity. Sandbox placeholder files committed.
- **Prevented by.**
  - `commit.sh` authors, commits and signs off as the repository's configured
    user, and refuses untracked symlinks, sandbox placeholders, co-author
    trailers and paths outside `commit_policy.allowed_paths`.
  - The CLI runs `committer.trial_commit` before the run starts, and delivery
    audits every commit since the merge base (`delivery.check_commit_hygiene`).

### 31. Run every delivery check that needs no finished target early
- **Symptom.** Every unit is green, then delivery fails on test naming, a
  missing accuracy test and a missing test-db entry.
- **Open.** The delivery checks that need no finished target (test naming, the
  accuracy test and its test-db listing) run only at delivery.

### 32. Evidence goes stale; key it to the tree and settings it was measured on
- **Symptom.** Delivery accepts a verdict older than the code and settings under
  it. The opposite costs too: a commit that only splits a test-list entry makes
  every verdict stale, and checks re-run on a tree no gate sees differently.
- **Prevented by.**
  - Every verdict records a fingerprint of the tree the gates measure
    (`workflow._target_marker`): the commit the checkout is at, less `tests/`,
    `docs/`, `.github/`, `examples/` and `*.md` (`gate_relevant`), but with the
    files a gate reads there (`gate_inputs`), and with the native libraries'
    sizes and modification times; and the Python modules its check imported,
    so a verdict holds while what it read is unchanged
    (`workflow._narrow_marker`). A native rebuild makes every verdict stale; a
    test list, none.
  - A check measures a read-only snapshot of that commit, never the checkout
    integrations move while it runs (`run/snapshots.py`), and a verdict whose
    legs imported two trees, or a checkout no commit pins, is void. A two-leg
    witness taken on the live checkout once ran its legs on two commits and
    passed.
  - What a check reads from outside the checkout is a digest on its verdict
    (`contract.check_inputs`), so a changed floor or option re-runs the checks
    that read it.
  - Delivery evaluates the contract on the tree as it is, and a verdict under
    another state is missing (`delivery-failed:contract`). The delivery tier
    runs on the final tree.
  - Checks that run one command on one engine against one criterion share a
    pass (`contract.equivalence_key`); failures are never shared.
- **Open.** Nothing invalidates a verdict when a value the spec's `environment`
  block sets (a `TRTLLM_*` variable, say) changes.

### 33. Records are never overwritten, and every claim is traceable
- **Symptom.** Earlier units' decisions overwritten by later ones; duplicate
  dispatch numbers; commit hashes in records that no longer resolve after a
  re-sign.
- **Prevented by.**
  - Decision files are keyed by unit, role and turn
    (`decisions/<unit>-<role>-<turn>.yaml`, `progress.decision_path`); a reused
    key gets `.2`, `.3` and so on, claimed atomically.
  - Journal dispatch ordinals are claimed atomically (`journal.next_ordinal`).
- **Open.** Commit hashes in the journal and reports are not re-resolved after a
  history rewrite.

### 34. An embargo must be able to name every reachable source of the answer
- **Symptom.** Another engine's implementation of the checkpoint, or this
  workflow's own worked example for it, is one read away from every agent, and
  the spec cannot name it.
- **Prevented by.**
  - `source_embargo` takes repo-relative paths, absolute paths outside the repo
    (as trees), and `package:<path>` inside this package; the schema refuses an
    entry that does not exist (`task_schema.py`).
  - Every role prompt restates the list (`prompts._common.source_embargo_block`).
  - Running another implementation as an oracle stays allowed.
- **Open.** The embargo is a rule in the role prompts only: no tool hook refuses
  a read of an embargoed path, and no read is logged.

## Performance (the tune stage)

### 35. Hosts differ by more than knobs do; judge on the same host
- **Symptom.** Six unrelated config rungs (a GC threshold, early first token,
  stream interval, a token budget, graph sizes, ...) all measured 3-4% worse at
  concurrency 1, and a +4.8% rung was rejected as a loss.
- **Cause.** Each rung had run on one of the nodes that were slower for
  identical configs; the noise floor was the cross-host spread.
- **Prevented by.** `campaign.Campaign._controls(host)`: every measurement is
  judged against controls of the same config measured on the same host, and the
  noise floor is the same-host spread, never the cross-host one.

### 36. A profiler that cannot finish is worse than none
- **Symptom.** nsys captures that never wrote a report; the server idle at 0%
  GPU after `Profiling started at iteration N`.
- **Cause.** The in-process `torch.cuda.cudart().cudaProfilerStop()` holds the
  GIL while CUPTI waits for GPU work that other Python threads must complete
  (gdb: every other thread in `PyGILState_Ensure`); a driver-level
  `cuProfilerStop` did not return either.
- **Prevented by.** `bench.py --nsys` defaults to session mode: `nsys launch`
  injects, `nsys start`/`stop` from outside the server time the window, and
  nothing in the server calls the profiler API.

### 37. A job's processes include the ones that left its session
- **Symptom.** A server left by a cancelled capture held most of a GPU's memory
  during the next measurement.
- **Cause.** `nsys profile` and a server started with `start_new_session` are
  re-parented to init when the job's leader exits, and the scheduler found a
  job's processes by session and ancestry only.
- **Prevented by.**
  - `scheduler.processes_with_marker`: the job's environment markers (spool,
    watcher and job) survive `setsid`, so a job's leftovers are found and
    stopped with it.
  - `bench.py` turns SIGTERM into its teardown and refuses a device that
    already has memory in use.

### 38. Module knobs never reach a target
- **Symptom.** A config rung that selects a kernel (`moe_config.backend`,
  `attn_backend`, `use_cute_dsl_*`) measures as noise on the target.
- **Cause.** A modeling_v2 target pins its kernels through the catalog.
- **Prevented by.** The serving config is derived from the workload and carries
  no module knob; a kernel choice for a target is a catalog change, made by a
  tuner candidate.

### 39. Measure candidates stacked on the current best
- **Symptom.** Candidates written in parallel from one base, each measured
  against the base, would be judged against a control that no longer exists
  once the first keep moves the best tree.
- **Prevented by.** `campaign.Campaign._stack`: an `eval-<slug>` worktree is the
  best tree plus the candidate's commits; a keep that moves the best tree
  re-stacks and re-measures the others before their verdict.
