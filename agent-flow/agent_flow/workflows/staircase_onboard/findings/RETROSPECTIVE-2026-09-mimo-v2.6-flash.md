# Staircase workflow retrospective — MiMo-V2.6-Flash-RL run

Written 2026-09-23, after the run closed with all eleven line items delivered.
Everything here is measured from this run, not proposed in the abstract.

## The numbers the rest of this document argues from

| | |
|---|---|
| wall clock | ~26 h |
| orchestrator launches | **7** |
| assembler dispatches | **10**, for **4** units |
| total role dispatches | 27 (25 answered, 2 orphaned) |
| gate runs | 138 |
| workflow defects found | 11 (F001-F011), 5 fixed with tests mid-run |
| compute sessions destroyed | **2** |
| agents stalled and killed | **3** |
| GPU-h on anchors that did not exist | ~12 |

**Six of ten assembler dispatches did no construction.** That is the single
largest measurable waste and it has one cause, below.

---

## 1. Execution efficiency

### 1.1 The restart tax — the biggest single loss

`task.yaml` is validated and loaded into `self.task_data` **once**, at
construction. Every gate afterwards reads that in-memory copy. So any spec fix
— and this run needed four — requires killing and restarting the orchestrator,
and **every restart re-enters the `assemble` stage and dispatches a fresh
assembler turn**.

Measured: core went to turn 5 of a 6-turn budget with only turn 1 doing work.
MTP went to turn 3 the same way. The unit came within one turn of dying
`assembly-exhausted` with a green, twice-approved target sitting in `HEAD`.

Three fixes, increasing in cost:

1. **Re-read `task.yaml` at each stage boundary**, and log when the on-disk
   copy differs from the loaded one. The spec is small; validation is
   milliseconds.
2. **A resume must not consume a turn.** On re-entering `assemble`, check
   whether the unit's decision file and `.done` already exist for the current
   turn index. If the last turn produced `UNIT_READY` and nothing under the
   target has changed since, advance to the gates instead of re-dispatching.
3. **Count turns by *work*, not by dispatch.** `unit_turns` exists to bound a
   self-heal loop. It currently counts restarts, which is a different thing.

Item 2 alone would have removed all six wasted dispatches.

### 1.2 The container import is 4-8 min normally and 25-35 min under load

Measured both. The run paid it on every one-shot job after the session died:
three jobs × ~30 min ≈ 1.5 h of pure waiting.

`container_name` reuse works **per node**, so it does not help a one-shot job
that lands elsewhere. Two things would:

- **Save a `.sqsh` once per run** and use the local path thereafter. The
  cluster already has `/lustre/.../containers/*.sqsh` from earlier work; the
  workflow should produce and register one as a preflight side effect.
- **Keep the session alive harder** (see §3.2), since a live session pays the
  import exactly once.

### 1.3 Gates are ~60% of wall clock and mostly correct

138 gate runs. The member-set change (F002) tripled the gate bill deliberately
and it was the right call — it is what produced the 0.11-point agreement across
tp4/tep4/dep4 that justifies the `unified` claim. No complaint here, but two
cheap wins:

- **Order members by cost.** `tep4` consistently booted in ~180 s against
  `tp4`'s ~285-438 s. Running the cheapest member first surfaces a broken unit
  sooner.
- **Let a boot failure skip the remaining members.** It already stops the
  unit; it should not have been worth running member 2 and 3 first.

---

## 2. Shortening the flow

### 2.1 Anchors are on the critical path and should be phase 0

Neither the model card nor the technical report publishes a GSM8K, MMLU, MMMU
or VideoMME number. Both anchors had to be **measured on an independent
engine**, and both landed late:

- GSM8K: needed before the core unit's accuracy gate. Made it, barely.
- MMMU: **discovered missing** only when I checked `mmmu.yaml` while MTP was
  gating. Had I not looked, the multimodal unit would have stalled at its
  accuracy gate hours later.

**Preflight should check that every unit's `reference_yaml` actually contains
the row its `model_key` + `discriminators` select, and fail loudly if not.**
That check is pure Python, costs milliseconds, and would have converted a
late surprise into a startup error. It is the same class as F001 and F005:
something knowable in seconds, discovered in hours.

Then: **dispatch anchor measurement concurrently with the plan phase.** It
needs no plan, no vocabulary and no target — only the checkpoint and an
engine. This run did that for MMMU by accident and it worked.

### 2.2 The engine recipe is an artifact, and reusing it is worth 2.5 hours

The first MMMU agent ran **2.5 h and produced nothing**. The second produced
the full 900-sample result in **13 min**. The only difference was that the
second was handed a *proven* recipe: the exact container
(`vllm/vllm-openai:mimo-v26-cu129`, the build that parses `store_dtype:
mxfp4`), the serve flags, the `--limit-mm-per-prompt` requirement, and the
short-`TMPDIR` constraint.

**The run should produce a `reference_engine.yaml` the first time it gets an
external engine serving the checkpoint, and every later anchor job should read
it.** This run passed it hand-to-hand through my dispatch briefs, which worked
only because I was watching.

### 2.3 `--max-plan-rounds 1` is right; the review is worth more than the re-draft

Two plan rounds cost ~1 h and the reviewer returned **REJECT** both times with
substantive findings — the disagg engine limitation, the refutation that
`kv_cache_reuse` was inert, and the world-size-1 geometry error that would have
had smiths certifying cells no rank ever passes.

That is a good trade. But note **the round-2 reviewer refuted one of my own
rulings** (R1's per-section slicing) by reading raw FP8 codes rather than
counting blocks — which the block count could not distinguish. The lesson is
not about the cap, it is §4.1 below.

---

## 3. Coordination between agents

### 3.1 The parallelism boundary is the file, and only the smith phase respects it

The smith batch was the run's best-executed phase: **9 dispatches, 8
concurrent, 9 disjoint catalog entries, zero collisions.** It works because a
catalog entry is one file plus one test file, and the queue is built from
disjoint surfaces by construction.

The assembler phase has **one `modeling.py`**, so it cannot parallelize at
all — and when I tried to run other work alongside it, I caused **F011**:
I dispatched the audio agent in parallel with the multimodal unit's gating,
onto the same file. `modeling.py` went 2827 → 3199 lines *while the reviewer
was reading it*, and three MMMU gates were queued against a half-rewritten
vision path.

The reviewer caught it only because it happened to re-read the line count.

**What the workflow should enforce:**

- **A gate records the tree state it measured** — `git describe --dirty` in
  every `GateResult`. One subprocess call. Without it, no number is
  attributable, and a *pass* on a dirty tree is worse than a failure.
- **A gate refuses to run when the target's own path is dirty**, unless
  explicitly overridden. At gate time the assembler has already committed;
  dirty means someone else is writing.
- **Say the invariant in the role prompts.** The commit-discipline block warns
  about `git add -A` sweeping in a sibling's half-written files. This is the
  same hazard one level up and is not mentioned anywhere.

### 3.2 A destroyed session takes the orchestrator with it — twice

Both session deaths had the same shape: an agent cleaning up its own work
killed the watcher, because `session_entry.sh` ends in
`exec bash spool_watcher.sh` — **the watcher *is* the allocation's main
process**.

1. The disagg probe's teardown SIGINT'd an `mpirun`-rooted worker; the step
   died at the same second.
2. The audio agent ran `pkill -9 -f smoke_audio.py` to clear a hung job.

Both took the orchestrator and every queued job with them.

**Fixes:**
- **The orchestrator must not live in the session it dispatches to.** It was
  submitted as `free-staircase*` inside the same allocation, so a session death
  is also an orchestrator death. Running it on the login node failed for other
  reasons, which is why it ended up there — but the coupling should be named
  and, ideally, the watcher should be a *child* the job supervises rather than
  the job itself.
- **`pkill` by pattern is banned inside a session job.** Clear a hung job by
  writing `STOP` or by killing a PID read from the log. This belongs in
  `TURN_INSTRUCTIONS.md` as a hard rule.
- **A probe that must signal multi-rank workers gets its own allocation.** The
  disagg probe adopted this after the first incident and it was right.

### 3.3 A turn can commit its work and never hand back (F009)

Dispatch 006 committed 380 lines with a passing receipt, then died without
writing its decision file. The orchestrator sat blocked **58 minutes** on a
missing zero-byte file while the work sat in `HEAD`, and would have waited the
full 4-hour timeout.

**A per-turn heartbeat** — mirroring the existing `driver.heartbeat` one level
down — turns a four-hour silent block into a named failure in five minutes.
And a stalled turn should fail *that request*, not the batch.

### 3.4 Findings must be able to reach in-flight dispatches

`SMITH_CORRECTIONS.md` was invented mid-run because the round-2 plan review
landed *after* the queue was written, and one of its corrections (C1: the FP8
GEMM widths) would otherwise have had a smith certify `N=192`, a width **no
rank ever passes**.

It worked — dispatch 009 certified `N=320` instead and proved it by doubling a
`b_scale` row and watching exactly those output columns move.

**This should be a first-class mechanism**: a corrections file the orchestrator
writes and every dispatch prompt includes by reference, with an explicit
precedence rule ("where this and `vocabulary.yaml` disagree, this wins").

---

## 4. Agents' own trial-and-error

### 4.1 Arithmetic that agrees is not evidence

The most expensive intellectual error in this run was mine. R1 concluded the
fused `qkv_proj` could be sliced per-rank **and then per-section**, because
both hypotheses predict 27 scale blocks per rank and 27 × 4 = the observed 108.
I treated the agreement as confirmation.

The round-2 reviewer read the **raw FP8 code values** and found block 25
straddles `K[128:192]` and `V[0:64]` — `scale[25,0] = 7.75e-4` against
`scale[26,0] = 7.36e-5`, an order of magnitude apart, which only happens across
a mixed span.

**The rule worth encoding: when two hypotheses predict the same count, the
count is not the discriminator.** Find the quantity that differs. This belongs
in the kernel-feasibility and modeling-core skills as a named trap, because it
is exactly the shape of reasoning an agent finds satisfying and is wrong.

### 4.2 "Prove your instrument can fail" was emergent — make it mandatory

Several smiths did this without being told, and it is the single most valuable
behaviour the run produced:

| control | correct | deliberately wrong |
|---|---|---|
| `mxfp8_quantize` layout | 1.97 ulp | **203.32 ulp** (swizzled) |
| MoE expert offset | inside gate | **>100 ulp**, still finite and populated |
| MTP concat order | 2.59 acceptance | **1.00** — total rejection, text still correct |
| ViT sink semantics | 0.467 of gate | **294.1×** outside (denominator-side sink) |
| uniform-KV replication | `max|d| = 0` | tile control separates at rel ≈ 1.0 |

On a checkpoint where **a wrong MXFP4 MoE does not raise** — it serves fluent
token soup — a test that cannot distinguish right from wrong bytes certifies
nothing.

**`staircase-kernel-onboard` and `-verify` should require a negative control
and refuse a receipt without one.** It is cheap, and it converts "the op ran"
into "the test can tell".

### 4.3 Agents over-polish the wrong artifact

Two clear cases:

- The multimodal equivalence agent spent **~2 h perfecting PNG rendering**
  (triangle orientation, digit spacing) and produced **zero** reference
  outputs. The images it had were already adequate — they exercise the tower's
  geometry, which was the point.
- The first MMMU agent spent **2.5 h** on benchmark selection and produced no
  artifact.

Both needed the same instruction, which I added to their replacements and
which worked immediately: **name the minimum artifact and produce it first,
then improve.** The replacement brief said "if you find yourself editing a
PNG, you have gone wrong" and it delivered in one job.

**Dispatch briefs should carry a first-artifact checkpoint**: what file must
exist, and by roughly when. An agent with no intermediate deliverable will
optimise whatever it is currently touching.

### 4.4 A probe anchors to its first hypothesis

The disagg probe was excellent work — it measured two engine walls, refuted a
claim the plan review had made, and correctly recommended descope **given the
framing it inherited**: "what would the engine need to tolerate this?"

It never asked the other question: **"what would have to be true for this to
work, and can the target make it true?"** Replicating KV heads 4→8 removes both
walls with no engine change at all, and it was found only because I posed that
question afterwards.

**A probe that concludes BLOCKED should be required to enumerate the
preconditions of the thing it is blocking, and say which are under the
target's control.** Here the precondition was "uniform per-layer KV geometry",
and the target owned it entirely.

### 4.5 Host facts were re-derived by agent after agent

`pytest` exits 4 here even when green. `TMPDIR` must be ≤ 61 characters. A
4-rank launch needs the `SLURM_*` scrub plus `--oversubscribe`. `UCX ERROR
uct_iface_open` is noise. `SMOKE_RC=$?` after a pipe is the pipe's exit code.
`.venv-3.12/bin/python` resolves only inside the container.

**I injected these by hand into roughly fifteen dispatch briefs.** Every one
was learned the expensive way by someone.

`TROUBLESHOOTING.md` exists but is keyed on symptoms *after* failure. What is
missing is a **preventive, machine-readable `host_facts.md` in the workspace
that every role prompt includes automatically** — written by preflight, which
already discovers most of it, and appendable by any role that learns a new one.

---

## The five changes with the best return

1. **A resume must not consume an assembler turn** (§1.1) — removes 6 of 10
   wasted dispatches.
2. **Gates record and require a clean tree** (§3.1) — F011 nearly attached an
   MMMU number to a commit it was not measured on.
3. **Preflight validates that every unit's anchor row exists** (§2.1) — turns
   a late stall into a startup error.
4. **Negative controls are mandatory for a receipt** (§4.2) — on this
   checkpoint it is the difference between a test and a formality.
5. **A `host_facts.md` every role gets automatically** (§4.5) — stops the same
   six facts being rediscovered fifteen times.
