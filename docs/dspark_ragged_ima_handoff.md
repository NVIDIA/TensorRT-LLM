# DSpark ragged verify — handoff

Written 2026-08-04. Supersedes nothing; read alongside
`docs/dspark_p0_task_prompt.md` (the original task) and
`docs/dspark_ragged_verify_status.md` (design/status).

Branch `dspark-p0`, fork `origin` = `github.com/lancelly/TensorRT-LLM`,
PR branch `origin/confidence_head` (NVIDIA/TensorRT-LLM#17056).

---

## 0. The one thing to read first

**There is an open CUDA illegal memory access on the ragged verify path, and my
framing of it was wrong until the last experiment of the session.** I spent most
of a day believing it was confined to a path I had just added (running ragged on
steps with no CUDA graph). The final run disproved that: with that path gated
OFF, the default configuration still faults.

So the correct statement is:

> Planner-driven ragged verification with real trimming has **never** completed a
> GSM8K run without `CUDA_LAUNCH_BLOCKING=1`.

Everything in §4 was designed under the wrong hypothesis. The falsifications
are still valid as facts, but the *set* of candidates they were drawn from was
too narrow. Re-derive the candidate list from "ragged trimming is unsafe", not
from "eager is unsafe".

---

## 1. Goal

Make DSpark confidence-head verification scheduling work, and be
CUDA-graph-compatible, on DeepSeek-V4-Pro-DSpark at DEP8 (TP8 + EP8 +
attention-DP, 8x B300), `max_draft_len=5`, tiers `{1,3,5}`, overlap scheduler ON.

Acceptance, all required:
1. Real checkpoint, not a stub.
2. GSM8K no worse than baseline. Scheduling changes only how many drafted
   positions are submitted, never which token is emitted — **accuracy loss is a
   bug, not a tradeoff**.
3. attention-DP verified (not just TP).
4. Ragged demonstrably *live*, not silently falling back.
5. Kernels compatible (indexer top-k, DSv4 compressor).
6. CUDA graphs actually replaying, not forced eager.
7. Overlap scheduler ON — that is the shipping configuration.

Requirement 4 exists because **every failure mode of this feature is silent**:
the planner declining, a flat cost model, a batch missing its captured shape, a
partially-windowed batch — each still produces correct output and baseline
accuracy. GSM8K passing proves nothing on its own.

---

## 2. Where it stands

### Works, with evidence

| | Evidence |
|---|---|
| K1/K2 unblocked, **zero C++ changes** | token-major presentation: each generation token becomes a length-1 row, so `seq_len == 1` divides any row count and `num_tokens % num_seqs == 0` passes trivially |
| GSM8K on the ragged path | TRTLLM **96.4367** (uniform baseline 96.2092) |
| cap-accept implemented | **96.096** TRTLLM + MEGAMOE_DEEPGEMM both pass |
| static reference | **96.171** |
| compact | **96.323** — but see the caveat in §3 |
| Uniform tier ladder removed | three states collapsed to two, −350 lines |
| Capture set no longer builds dead tiers | 57 graphs → 19 on the non-ragged branch |
| Planner counters readable | `decisions` was a dead counter; now increments |

### Open

| # | Item |
|---|---|
| **IMA** | §3. Blocks everything else. |
| #19 | Verify budget is quantised to the tier ladder. Measured cost up to **73%** of the planner's own objective. Not implemented. |
| #11 | Sync control never completed. **Do not cite any sync-related conclusion.** |
| #20 | Three-way differential incomplete (compact leg needs a non-blocking run) |

---

## 3. The IMA

### Symptom

One rank reports `CUDA error: an illegal memory access was encountered`, peers
report `unspecified launch failure`, at decode step ~75–128 of GSM8K on DEP8
with the steep synthetic cost table. Surfaces at `dspark.py:117`, but CUDA
errors are asynchronous — that is the next sync point, not the origin. Ranks
that throw stop issuing collectives, so the survivors deadlock and the hang
detector kills the job at 300s. **The hang is a consequence; the IMA is the
cause.**

### The only configuration that completes

`CUDA_LAUNCH_BLOCKING=1` — 128 steps, GSM8K 96.323 (`compact_ima.log`). That is
where the 96.323 in §2 comes from, so treat it as "the numbers are right when
the race is suppressed", not as a clean pass.

### Scope — read this carefully

The last run of the session (`compact_gated.log`) used the **default**
configuration with the eager-ragged path gated off:

```
fallbacks: {'no_graph_this_step': 46, ...}   <- gating worked
steps_ragged: 43                              <- all of these were GRAPH steps
IMA: 1, hang: 5
```

So the fault occurs on the ordinary graph ragged path — `fit_ragged_verify_lens`,
a captured bucket, a replayed graph. It is not specific to any path added on
this branch.

Corollary: `commit 4f81cc5534` ("Gate the eager-ragged path off until its IMA is
understood") does **not** make the branch safe. Its stated justification is
wrong. Keep the gate or drop it on its own merits, but do not rely on it.

### Reproduce

```bash
export LLM_MODELS_ROOT=/lustre/fsw/coreai_comparch_trtllm/laliao/llm-models
SPS_TABLE=/lustre/fsw/coreai_comparch_trtllm/laliao/dspark-runs/steep_sps.json \
  tmp/run_dspark_e2e.sh <jobid> ragged TRTLLM <logname>
```

A steep table is **required**: at this checkpoint's ~90% acceptance the planner
correctly declines to trim against the real table, so nothing is exercised.
`tests/microbenchmarks/dspark_make_steep_sps_table.py` generates one; it carries
`SYNTHETIC: true` and **no throughput number taken with it means anything**.

Useful switches, all wired through `tmp/run_dspark_e2e.sh`:

| env | effect |
|---|---|
| `LAUNCH_BLOCKING=1` | `CUDA_LAUNCH_BLOCKING=1` — the only thing that passes |
| `NO_OVERLAP=1` | `disable_overlap_scheduler=True` |
| `EAGER_RAGGED=1` | re-enable the gated eager-ragged path |
| `ASSERT_ROWS=1` | row-map staleness assertion (skipped during capture) |
| `SANITIZER=1` | compute-sanitizer — see §5, not currently viable |

---

## 4. Falsified hypotheses

Each by its own run unless marked. **Caveat**: several were designed to
discriminate "eager vs graph", which §3 shows was the wrong axis. The
observations stand; their relevance may not.

By experiment:
- stale `ragged_pad_verify_len` (`compact_confirm`)
- temporary pinned buffers as async H2D sources (`compact_pinned`)
- ragged on a mixed batch (`compact_genonly`; `steps_ragged` rose 35→69 and it
  still faulted)
- cross-step overlap (`compact_nooverlap`)
- bucket alignment and the fit (`compact_fitnokey`: full fit, only the graph key
  dropped)
- per-step buffer allocation (`compact_eagerpool`: a persistent pool made it
  **worse**, 18 faults)
- window raggedness (`compact_eageruni`: identical windows, 12 faults)
- **trimming itself** (`compact_eagerfull`: every window set to `max_draft_len`,
  making the step semantically identical to static — still faulted)
- a barrier at the scheduling point (`compact_syncfinal`, `compact_wait`)
- token-major row-map staleness (`compact_assert2`: IMA reproduced 15x, the
  assertion never fired, and `steps_ragged: 74` confirms it had every chance to)

By reading:
- DeepGEMM 32-token alignment — `scheduler_metadata_buffer_expanded` is
  `(num_sms+1, 2)`, independent of token count
- row-buffer overflow — `rows <= cap` is asserted (`dsa.py:2024`)
- buffer-pool use-after-free — eager metadata has `cuda_graph_buffers=None`, so
  `get_empty` returns `torch.zeros` and never touches the pool
- metadata rebuilt per step — `_set_up_attn_metadata` caches `self.attn_metadata`
- MoE aux-stream overlap — `with_multi_stream(True)` appears only in
  `cuda_graph_runner` capture/replay, so multi-stream is OFF on eager steps

---

## 5. Traps — I fell into each of these

**"It stopped without error" is not "it passed."** Three times a run exited
early for an unrelated reason and I read the absence of a fault as a fix:

- `compact_eagersync` reported IMA 0 — it died at step 67 on the
  `assert_ragged_active` gate, before the step range where the fault occurs. I
  wrote a task claiming the fix was found. Re-run to completion: IMA 3.
- `compact_trtllm4` (conservative fallback) showed no fault at step 64 and I
  concluded the eager path was to blame. It never completed either. This is what
  produced the wrong framing in §0.
- My own row-map assertion did a `.max()` device-to-host read inside the CUDA
  graph capture body, raising `cudaErrorStreamCaptureInvalidated` and killing the
  engine before any real step ran. That run's "IMA 0" meant nothing.

**Always check the run reached the failure window** (step ≥ ~128, or a
non-zero `steps_ragged` in the periodic summary) before believing a negative.

**Post-hoc mechanism is not evidence.** Twice I saw a symptom, found code that
could explain it, and reported it as the cause without checking:
- the hang: I blamed rank-local `enable_spec_decode`; `git diff` showed this
  branch never touched it. The real trigger was that ranks 5 and 6 *threw* (my
  own layout assertion) and stopped issuing collectives. **The ranks that did
  not print a stack were the informative ones.**
- the workspace: I claimed `68 MiB → 604 MiB` was the token-major cost and
  changed `predicted_tokens_per_seq` accordingly. Measured: byte-for-byte
  identical. The 604 MiB is the *prefill* workspace at `max_num_tokens`, and
  `getWorkspaceSize` returns `max(context, generation)`, so the generation term
  is invisible. Change reverted.

**Run the full test selection.** Removing the uniform tier ladder broke 4 tests
in `tests/unittest/llmapi/test_llm_args.py` and 1 integration test. I reported
"all green" from a sweep that never executed either file.

**Capture is a systematic blind spot.** Three bugs this session had the same
shape — runtime path updated, capture path not:

| | runtime | capture |
|---|---|---|
| capture set keyed on config flag | by mode | by flag |
| dead tier ladder | draft_len fixed | still built 3x graphs |
| `agreed_ragged_bucket` | set by the fit | set by nobody |

Unit tests cannot reach capture; it needs a real pass.

**attention-DP turns a rank-local raise into a cluster deadlock.** A diagnostic
assertion that fires on 2 of 8 ranks costs 300 seconds and produces a hang
report that names nothing. Consider reducing across ranks before raising.

**compute-sanitizer is not currently viable** — it instruments the ~40-minute
weight load; the log grew zero bytes in 55 minutes. Restricting it to a single
rank is untried and is the obvious next move.

---

## 6. Next steps

1. **Re-derive the candidate list** from "ragged trimming is unsafe", not "eager
   is unsafe". The sharpest surviving constraint is `compact_eagerfull`: with
   every window at `max_draft_len` the step is semantically identical to static,
   yet it faults — so the trigger is that `is_ragged_verify` is True, not what
   the windows contain. Audit every `is_ragged_verify` consumer. Two remain
   unexamined: `refresh_token_major_block_table` (a 4-D `index_select` that has
   produced a device-side out-of-bounds before) and the four branches in
   `sparse_attn_indexer`.
2. **Single-rank compute-sanitizer.** It names the kernel and the address; that
   ends the guessing. Everything else in this section is another hypothesis.
3. Only after the IMA: #19 (budget quantisation, the largest measured gap),
   then #11 and #20.

### Do not

- Do not resolve by declining ragged whenever a graph is unavailable. Those steps
  were eager before this feature existed (`can_run_cuda_graph` is
  `num_context_requests == 0`), so declining buys no graph replay — it only
  forfeits trimming, measured at 46 of the first 64 steps, `trim_ratio` 0.62→0.
- Do not cite `docs/dspark_p0_task_prompt.md` §10 sync numbers; the control never
  ran.
- Do not trust a throughput number taken with the synthetic steep table.

---

## 7. Unpushed work

`origin/confidence_head` is at `f033e3cf5b`. Three local commits, none
end-to-end verified:

| commit | what | note |
|---|---|---|
| `59e2d47b41` | clear the ragged pad window when the fit is skipped | correct on its own merits |
| `4f81cc5534` | gate the eager-ragged path off | **stated justification is now known wrong** (§3) |
| `cdc4538616` | row-map staleness assertion | hypothesis it tested is falsified; assertion is cheap and may still be worth keeping |

Also uncommitted: `dsa.py` carries the capture-skip fix for that assertion.

Two fixes worth keeping regardless of the IMA, already in the above:
`_pinned_host` (persistent staging buffers — PyTorch does not extend an async
H2D source's lifetime, and three sites were passing temporaries) and the
`TLLM_DSPARK_DISABLE_OVERLAP` diagnostic switch.

---

## 8. Environment

- 8x B300 SXM6 (SM103, 275 GB/card), x86_64, node pool `bia*`, partition `batch`
- **Queue on both SLURM accounts in parallel** — `coreai_comparch_trtllm` and
  `coreai_comparch_aarwlt`. Their fairshare is independent and differs
  enormously: on 2026-08-03 one sat PENDING for over two hours with reason
  `Priority` while the other, submitted to the same full partition minutes
  later, was RUNNING in 30 seconds.
- Weights: `~/llm-models/DeepSeek-V4-Pro-DSpark` (symlink; `LLM_MODELS_ROOT`
  must be exported *before* pytest starts — `MODEL_PATH` is evaluated at import)
- Run logs: `/lustre/fsw/coreai_comparch_trtllm/laliao/dspark-runs/`
- Every number in this document is x86 DGX B300. **GB300 is aarch64 with
  NVLink-C2C unified memory — all timing, memory and load-path figures must be
  re-measured there.** The structural conclusions transfer.
