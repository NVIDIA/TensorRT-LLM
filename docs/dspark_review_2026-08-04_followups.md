# Review: the five follow-up commits of 2026-08-04 evening

Scope: `72dd803141..fab6291e2d` on `confidence_head` —

| commit | title |
|---|---|
| `72dd803141` | [fix] Interpolate the SPS cost table between breakpoints |
| `c117be89a1` | [feat] Calibrate DSpark confidence and measure the planner's inputs |
| `50675cb181` | [fix] Make the ragged fit's consistency checks able to fail |
| `0b524435fc` | [feat] Pin the DSpark verify length at runtime |
| `fab6291e2d` | [chore] Measure the SPS table against the deployment, not a rebuild |

Method: seven independent review passes (one per commit plus two cross-cutting
lenses: CUDA-graph capture/replay safety and overlap-scheduler
write-after-read — the two bug families the IMA series just closed), then an
adversarial verification pass per finding whose default stance was that the
finding is wrong until the code proves it. 38 findings raised, **34 confirmed,
4 refuted**; the refuted ones are omitted. Several verifications executed the
cited code paths rather than reading them (noted inline).

Severity is scoped to the feature: nothing below corrupts served tokens
(verification guarantees them). "High" means a feature's primary artifact or
purpose silently fails.

How to read an entry: *mechanism* is what the code does, *failure* is a
concrete input→wrong-outcome chain, *fix* is the smallest correct change the
reviewers could see. Line numbers are as of `fab6291e2d`.

---

## A. STS calibration pairs confidence with the wrong labels — HIGH (as a group)

Four findings converge on one conclusion: **the fitted temperature table —
`c117be89a1`'s primary artifact — is fit on mispaired samples in every
supported collection mode**, and nothing downstream can detect it.

### A1. The label is one (or two) draft passes behind the confidence — deterministic

`spec_sampler_base.py:311`, `dspark_sts.py:199` · medium

Mechanism: one forward(i) first *verifies* the block drafted by the previous
forward (B_{i-1}; `dspark.py:822` computes the accepted count), then *drafts*
the new block B_i and stashes its confidence C_i (`dspark.py:789-791` →
`dspark_sts.py:199`, which unconditionally replaces the single mutable
`_stash_logits`). `record()` runs later, inside `update_requests`
(`spec_sampler_base.py:311`), pairing `accepted = A(B_{i-1})` with whatever
the stash holds — which is always C_i. Non-overlap loop: off by one
(`py_executor.py:4634` forward before `:4673` update). Overlap loop: off by
two (`:5132` forward before `:5139` update *of the previous batch's state*).
Because the stash overwrite happens inside the very forward that produces the
label, **no execution order leaves the correct C_{i-1} in the stash**.

The window axis got exactly the missing treatment in the same commit
(`verify_lens_snapshot`; the `_verified_len` docstring at
`spec_sampler_base.py:226-233` names this identical hazard). The confidence
axis did not.

Why it is invisible: the fitter (`dspark_fit_sts.py:171`, `--source auto`)
*prefers* the `logits_at_draft` column, whose docstring calls it "the correct
pairing"; the built-in stash-vs-buffer diagnostic compares two columns that
are both C_i, so they agree with each other while both are mispaired; the
82.2%-rows-differed measurement quoted in `load_shards` compares the columns
against each other, never against the label. The commit's own measurement of
one-pass-apart rows (correlation 0.14, mean |delta| 7.35) says what the fit
converges to: temperatures that flatten survival toward the marginal
acceptance rate, which then silently biases the serving planner's budget
argmax wherever the table is deployed.

Fix: join on the draft-pass axis, not the wall-clock axis — ride the
confidence rows for block B in the SampleState produced by the forward that
verifies B (the `verify_lens_snapshot` pattern), or keep a two-deep ring
keyed by the existing draft sequence counter and have `record()` select by
the verified block's sequence.

### A2. The recorder reads the confidence buffer with the wrong slot key

`spec_sampler_base.py:312` · medium

Mechanism: `record(slot=req.py_seq_slot, device_logits=self.sts_logits_buffer)`
indexes the raw `worker._confidence_logits` buffer by **py_seq_slot**, but
that buffer's rows are assigned by the DSpark worker's *private* allocator
(`_assign_slot`, a FIFO deque keyed by `py_request_id`,
`dspark.py:591-614`). `py_seq_slot` comes from the separate `SlotManager`
(set-based, immediate reuse; `resource_manager.py:2332-2344`). Nothing ever
synchronizes the two. They coincide only until the first request completes:
the set hands a freed slot back immediately while the deque defers it, so any
departure+arrival churn diverges them permanently. Every other reader
correctly goes through `confidence_row_for(py_request_id)`
(`dspark.py:319-334`); the recorder alone uses the raw index. Bounds always
hold (buffer is `[max_batch+2, K]` vs `py_seq_slot < max_batch`), so nothing
crashes — request A's accepted count is paired with request B's logits and
the shards look healthy. The late-read diagnostic reads back with the same
wrong key, so it is structurally blind to this too. The docstrings claiming
both halves are "indexed by the same py_seq_slot"
(`spec_sampler_base.py:203-206`, `dspark_sts.py:137-138`) assert an invariant
the code never establishes.

Fix: wire a resolver instead of a raw buffer — e.g.
`self.sampler.sts_row_for = worker.confidence_row_for` at
`py_executor.py:3421-3427` and have `record()` take `py_request_id`.

### A3. The draft stash survives CUDA-graph replay steps unbounded

`dspark_sts.py:203` · low (verified with the full replay chain)

Mechanism: `begin_step` invalidates `_host_logits`/`_stash_host` but never
`_stash_logits`/`_stash_slots`. Under graph replay, the Python inside the
captured draft body does not run, so the stash retains clones frozen at the
last *eager* step; `record()` re-materializes `_stash_host` from them
(`dspark_sts.py:236-244`) and appends stale-but-finite rows into
`logits_at_draft` as if they were this step's data (continuing requests keep
their slots, so lookups hit). The capture-time warning fires once during
warmup and any later eager step (mixed batch, uncaptured shape — routine
under in-flight batching) silently re-arms the stash for all subsequent
replays. `make_recorder_from_env` refuses cost-table and compact-ragged
regimes but never checks `cuda_graph_config`.

Fix: tag the stash with the worker's draft sequence when armed
(`_confidence_stamp` already solves this class for the device buffer,
`dspark.py:781`) and treat a sequence mismatch as a missing entry.

### A4. Recorder declines are silent, against the branch's own rule

`dspark_sts.py:226` · high

`record()` returns without recording on an out-of-range slot — no counter, no
log — and the capture-time stash refusal warns once but is not counted. This
module family's own documented rule (`dspark_verify.py:326-329`: even the
benign empty-batch decline is counted, because "this feature's failures are
all silent") is violated at exactly the call sites that A2/A3 make load-bearing:
a systematically wrong slot or a fully-graphed run yields empty or poisoned
shards with nothing in the stats to say why. Additionally, the construction
-time censoring guards can be bypassed mid-run: `adopt_verify_len_pin` can
lower the pin below `block_size` (or clear it) *while a collection run is
writing shards*, reintroducing the censored-label regime the constructor's
ValueError exists to refuse.

Fix: count dropped slots and stash refusals in a recorder stats dict surfaced
next to the planner's; refuse (or count) pin adoption below `block_size`
while a recorder is active.

---

## B. The runtime verify-length pin silently never applies on tp_size == 1 — HIGH

`py_executor.py:3516` (adoption), `:3341` (endpoint), `dspark_verify.py:263`
(queue) · from `0b524435fc`

Mechanism: the only production call to `planner.adopt_verify_len_pin` sits
inside `if is_distributed:` (requires `tp_size > 1`; `py_executor.py:3433,
:3497`) — there is no else branch. The endpoint chain
(`openai_server.py` → `base_worker.py:875` → `PyExecutor
.set_dspark_verify_len_pin`) only *queues* the pin
(`request_verify_len_pin`, `dspark_verify.py:238-247` — docstring: "NOT
applied here") and returns 200 with the validated value.
`decide_verify_lens` reads only `_forced_verify_len` (`:393`), which nothing
else sets. The `base_worker.py:880` design comment — "rank 0 alone receiving
the RPC is sufficient" because "the step's own allgather" broadcasts it — is
false at tp_size == 1: there is no allgather.

Failure: a single-GPU profiling server (the profiler's *default*,
`--tp-size 1`) accepts POST `/dspark/verify_len_pin` with 200; the pin sits
queued forever; `forced_steps` stays 0; and the server sweep
(`dspark_sps_profiler.py:1887-1890`) explicitly trusts the 200 and never
reads the effective length back — so every cell of the produced SPS cost
table is timed at planner-chosen lengths (full block, since sweeps run
without a cost table) but *labelled* with the pinned length. The fitted
table comes out flat in L and then silently mis-drives every planner that
consumes it. This is precisely the mislabelled-cell corruption the pin
feature exists to prevent, and `test_dspark_verify_len_pin.py` exercises
only the planner in isolation, never the executor wiring.

Two adjacent silent modes (confirmed separately):
- **Static-mode engines** accept and adopt the pin, log "set", but
  `decide_verify_lens` is never called (`compute_windows` is False) — the
  endpoint should 400 the way it already does for a missing planner
  (`py_executor.py:3341`).
- **Queue race**: `adopt_verify_len_pin` unconditionally resets
  `_pending_verify_len_pin = -1` (`dspark_verify.py:263`), so a second pin
  queued by the RPC thread between the decode loop's payload read (`:3504`)
  and adopt (`:3516`, an allgather apart) is wiped after its endpoint already
  returned 200. Fix: compare-and-clear.

Fix (primary): `else: planner.adopt_verify_len_pin(planner.pending_verify_len_pin())`
after the `is_distributed` block — on a single-rank world the local queue *is*
the group's agreement. Have the sweep read back the effective value.

---

## C. The table fingerprint records CLI flags, not engine facts — hard-rejects correct tables — HIGH

`dspark_sps_profiler.py:2071` (write), `dspark.py:456-462` (check, from
`c117be89a1`) · from `fab6291e2d`

Mechanism: the fingerprint dict is built from `args` (`"ep": int(args.ep_size
or 0)`, `"tp": args.tp_size` default 1, `"attention_dp"` store_true default
False) and attached whenever the run was not `--fit-only`. But:

- The two **new deployment-measuring sources** (`--from-iter-log`,
  `--base-url`) build no engine at all, so there is no reason to pass those
  flags; a table swept over HTTP against a tp8/ep8/attention-DP deployment
  carries `{tp:1, ep:0, attention_dp:False}`.
- Even the in-process sweep false-rejects whenever `--ep-size` is unset:
  the write side records 0, while the loader compares
  `mapping.moe_ep_size`, which `Mapping` normalizes to >= 1
  (`llm_args.py:4739` maps None→−1; `mapping.py:114` resolves −1→1 — chain
  verified line by line).

`check_table_fingerprint` (`dspark_planner.py:206`, raise at `:244`) then
refuses with "profiled on a different engine configuration… Re-profile on
this configuration" — and re-running the same profiler command reproduces the
refusal. This defeats the commit's own title (measure against the
deployment: the tables measured ON the deployment are exactly the ones
rejected), and violates `check_table_fingerprint`'s own documented mechanism
for facts the consumer cannot verify: key ABSENCE (compared keys must be
present in both dicts; absent ones go to the check-manually INFO line) — the
`ep: 0` encoding fabricates a fact instead of omitting it.

Fix: record resolved engine facts when an engine exists (read the mapping
after construction); omit the key entirely when the fact is unknown
(HTTP/log sources, unset flags).

---

## D. `padded_bs_hist` pollution is not actually fixed, and the fit publishes before validating — HIGH

`model_engine.py:3789` (clear), `py_executor.py:3633-3644` (read),
`model_engine.py:3908-3986` (publish-before-validate) · from `50675cb181`

Mechanism (pollution): the commit clears `_dspark_last_padded_bs` at
`fit_ragged_verify_lens` **entry** and its comment claims this prevents a
stale value from polluting "exactly the fallback steps the histogram exists
to distinguish". But repo-wide the attribute has three references: cleared at
fit entry, set on fit success, and **read unconditionally into
`record_step` on every stats-bearing step** (`py_executor.py:3640`). The
eager-ragged branch (`:3582-3599`) never enters the fit (it clears
`agreed_ragged_bucket` and `ragged_pad_verify_len` — but not this), and the
cap-accept branch never reaches the fit either. Both produce non-empty
`verify_lens`, so `record_step` passes its early-out and
`dspark_observability.py:327-328` books the LAST fitted step's padded row
count into `padded_bs_hist` for a step that was never padded. Eager-ragged
steps are frequent by the code's own measurement ("46 of the first 64
steps"), so the histogram — added by this very range to reconcile the
negative-trim_ratio bug — can be dominated by phantom entries on exactly the
steps it exists to explain. Execution behavior is unaffected (the ceiling
rebase requires `bucket`), so this is observability-only — but this branch's
stated stance is that these counters are the sole detection channel for the
feature's silent failures.

Fix (total, verified): pass `padded_bs` to `record_step` only when
`bucket is not None` — bucket is non-None exactly when the fit delivered.

Mechanism (publish-before-validate): inside the fit,
`_dspark_last_total_tokens` (`:3908`), `_dspark_last_padded_bs` (`:3914`),
`runner.agreed_ragged_bucket` (`:3928`) and `runner.ragged_pad_verify_len`
(`:3971`) are all published **before** three validation-failure returns
(`:3960`, `:3969`, `:3986`). A failed fit therefore leaves every one of them
stale-set while the step falls back to uniform. A stale published bucket is
the exact state family behind the just-fixed IMA; today the wrong-key replay
is prevented only by the window check in `_ragged_verify_bucket` plus the
`py_verify_len` clearing — one line of defense where there should be two.
The layout assert's uniform branch (`model_engine.py:4076-4085`) does not
check `agreed_ragged_bucket is None`, so the stale state is invisible even
with `TLLM_DSPARK_ASSERT_LAYOUT=1`.

Fix: move all four publications below the last failure check, next to the
`py_verify_len` stamping.

Related (medium): the commit's central rationale — that the old assert was a
tautology — is false against its own parent: at `50675cb181^`,
`_ragged_verify_bucket` already returned `agreed_ragged_bucket` rather than
re-summing, so the old assert already compared the batch walk against the
fit's allgathered value and could fire. The change is a no-op refactor, and
the comment at `model_engine.py:4103-4109` encodes the incorrect history.
Rewrite it to state what the check actually does.

---

## E. Profiler: the default sweep geometry hits a device-side assert, and the promised drain observability does not exist — HIGH (×2)

From `72dd803141` / `fab6291e2d`, `tests/microbenchmarks/dspark_sps_profiler.py`.

### E1. Defaults trap: `block_follows_verify_len=False` runs the asserting regime

`:1036` (default), `:1219-1223` (the commit's own comment), `:1581-1585`
(flag help) · high

The default sweep geometry is constant-block; the same commit's comment says
that regime "currently asserts (Repeat.cu output_size vs sum(repeats))"
whenever `max(verify_len) < block_size` — which is **every swept cell except
L == max_draft_len** under the default `verify_lens = 1..max_draft_len`
(compact ragged is forced on by `_prepare_environment`,
`TLLM_DSPARK_RAGGED_VERIFY_MODE=COMPACT`). `SweepConfig.validate()` — whose
docstring is literally "catch the geometry mistakes that would otherwise
fail mid-sweep" — does not check it. A user running the documented default
builds an engine for minutes, then dies on a CUDA assert at the first
L < block cell. Verification also checked the timeline: the "currently
asserts" comment was written hours *after* the IMA fix landed and was
re-worded at HEAD, so it is not stale; and no recorded sweep artifact
(`dspark-runs/sps_samples_L*.jsonl`, `sps_real_final.json`) has ever run the
constant-block geometry to completion.

Fix: either fix the Repeat.cu packing for `max(lens) < block_size` and delete
the comment, or flip the default to `block_follows_verify_len=True` and make
`validate()` refuse the known-asserting combination.

### E2. The mid-cell drain promises a counter that does not exist, and the drain can latch shut

`:1344` (comment), `executor/result.py` (`IterationResult.get_results`) · high

The comment says "the count of empty reads is reported so a cell that lost
its tail this way is visible rather than merely short" — grep of the file at
the commit and at HEAD finds only the comment; the drainer reports only
exception counts. Worse, on the in-process path `IterationResult.get_results`
latches `_done = True` on the first Empty and only `mark_undone()` (new
prompt submission) resets it — one longer-than-timeout gap in stats
production makes **every subsequent `get_stats` call, including the final
post-cell drain, return `[]` instantly**. Cells are silently truncated with
no signal, in a tool whose output feeds the planner.

Fix: implement the promised empty-read counter and surface it per cell;
`mark_undone()` (or bypass the latch) before the final drain.

---

## F. Interpolation switch: three loose ends — MEDIUM

From `72dd803141`.

1. **Truncation eats shelf-close points first**
   (`dspark_sps_profiler.py:756`, medium — verified by executing the numeric
   example). `compress_to_risers` now encodes a flat shelf as a breakpoint
   PAIR (required by the interp consumer), but the `max_breakpoints`
   truncation drops the interior point with the smallest incoming jump —
   which is by construction always a shelf-CLOSE point (its jump is below
   `min_riser_ms`), never a real riser. Dropping it turns the measured shelf
   into a ramp: with counts `[16..128]` and a 2.0→3.0 riser, cap=5 bills
   M=80 (measured 2.004) at interp 2.5 — the mid-shelf over-bill the
   function's own docstring warns about. The pair encoding also roughly
   doubles kept points while the default cap stayed 8, so truncation now
   fires on sweeps that never used to hit it — silently (no warning, no
   `_meta` note), and the only truncation test uses `min_riser_ms=0.0` so
   the interplay is untested. Fix: drop by introduced interpolation error
   (or treat a shelf pair as atomic), warn when truncation fires, add a test
   with `min_riser_ms > 0`.

2. **No format marker for the semantic reversal**
   (`dspark_planner.py:113`, low). A table emitted by the OLD compressor
   (which dropped shelf-closing breakpoints *by design*, for the floor
   consumer) is silently reinterpreted as a continuous ramp by the new
   interp consumer — the "over-trim mirror" the new code names but cannot
   detect. Old tables lack `_meta.engine` so they get only the generic
   missing-fingerprint warning. Fix: stamp `lookup: "interp"` (or a format
   version) into the payload meta; warn-or-refuse tables without it.

3. **`fixed_overhead_ms` escapes the finiteness validation the commit
   message claims** (`dspark_planner.py:148`, low). The check is
   `if self.fixed_overhead_ms < 0.0`, which NaN and +inf both pass; a
   non-finite overhead silently collapses the planner to maximal trimming.
   Fix: `if not math.isfinite(...) or ... < 0.0: raise`.

---

## G. Documentation drift (new feature doc contradicts shipped code) — LOW

`docs/source/features/dspark-confidence-scheduling.md`, from `50675cb181`:

- `:195` — asserts the cost lookup is a staircase that "绝不插值", the exact
  opposite of the code as of three commits earlier (`72dd803141`), while the
  commit message claims the doc "caught up with the code".
- `:261-268` and `:283` vs `:307` — self-contradictory on the window floor:
  two places say `min_verify_len` defaults to 0 (with a worked example headed
  "min_verify_len=0（默认）"), one place (and the code) says the floor is
  fixed at 1.
- `:400` — states in present tense that draft_len "is not in the
  attention-DP consistency allgather", documenting as live a hazard the
  shipped shape gate already closed (it allgathers and compares draft_len).

Fix: one editing pass against the code; recast the `:400` diagram as the
motivating failure.

---

## H. Profiler tooling correctness (deployment-measurement paths) — MEDIUM/LOW

From `fab6291e2d`, `tests/microbenchmarks/dspark_sps_profiler.py`:

1. `:1778` (low, verified by executing the parser on a real mixed-step log
   line) — `samples_from_iter_log` files mixed context+generation steps
   under a wrong `(padded_bs, verify_len)` cell with prefill-inflated eager
   step times; `num_ctx_requests` is on the same log line and ignored. The
   ladder-resolution path converts ambiguous-therefore-dropped into
   confidently-mislabelled. Fix: capture `num_ctx_requests` and skip ctx>0
   steps, counting the drops.
2. `:1966` (medium) — the server sweep's load duration
   (`max_tokens = max(64, measure_steps)`) is uncoupled from the measurement
   window (`settle_s + poll_s`); under defaults the synthetic requests can
   finish before the window opens, collecting nothing or ramp-down rows
   filed under smaller batch sizes. Fix: derive load length from the window,
   or re-post until the drain completes.
3. `:1878` (medium) — the completion-POST worker threads discard responses
   and swallow all errors; a wrong `--served-model-name` produces an idle
   server, empty cells, and a SystemExit blaming `/metrics` and
   `enable_iter_perf_stats`. Fix: check status codes, surface the first
   failure in the error message.
4. `:1866` (low) — a pin refused mid-sweep raises out of the whole sweep and
   discards every already-collected sample (`_write_samples` runs only after
   the sweep returns); with default `verify_lens = 1..max_draft_len` against
   a server whose ladder lacks a rung, a refusal is near-certain. Fix: catch
   per rung, count skips, write samples incrementally.
5. `:1917` (low) — `warmup_steps` is accepted by the server path but never
   applied; the progress line prints `kept - warmup_steps` while all `kept`
   rows were appended. Fix: apply it or drop it.
6. `:1788` (low) — `rank_prefix='0:'` silently drops all lines of logs
   without an mpirun/srun rank tag (single-process deployments); not exposed
   on the CLI; the error blames `print_iter_log`. Fix: fall back to
   prefix-less matching when zero lines match; expose `--rank-prefix`.
7. `:1473` (medium) — the moved file still advertises the old module path in
   its own `--help` prog string and docstring run command; copy-pasting the
   tool's documented invocation fails with ModuleNotFoundError (the commit
   updated the `llm_args.py` error message but not the tool itself).

---

## Suggested order of attack

C → B → D → A → E1/E2 → F → H → G. C and B block anyone producing a usable
table with the new tooling and are one-liners; D's histogram fix is a
one-line condition plus a code motion; A decides whether the calibration
feature produces a meaningful artifact at all and is the largest change
(snapshot-per-block pairing + request-id keyed reads); E1 is a defaults flip
or a kernel fix; the rest are local.

A recurring theme worth internalizing: most of these violate the branch's
own explicitly documented rules — "every silent decline must be counted"
(A4, B, E2, H), "facts the consumer cannot see must be absent, not
fabricated" (C), "clear stale published state so it cannot key a later step"
(D) — rather than rules imported from outside. The rules are right; the new
code just needs to keep following them.
