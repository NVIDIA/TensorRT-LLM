# DSpark ragged verify — handoff

Written 2026-08-04. Supersedes nothing; read alongside
`docs/dspark_p0_task_prompt.md` (the original task; §10 is the results record).

Branch `dspark-p0`, fork `origin` = `github.com/lancelly/TensorRT-LLM`,
PR branch `origin/confidence_head` (NVIDIA/TensorRT-LLM#17056).

---

## 0. The one thing to read first

**The CUDA illegal memory access this document hands off has since been root-caused and FIXED** — see §3's dated entries: a ragged graph key derived over a window-less step replayed the previous ragged step's row state; fixed at the key derivation (`6a59b01e53`, with the WAR-staging hardening in `c7e6d78ac2`/`74f1a82d06`/`5c38441a00`), validated 131 steps / zero IMA with ragged fully active. The rest of this section is preserved as the record of how the framing went wrong: I spent most of a day believing the fault was confined to a path I had just added (running ragged on steps with no CUDA graph); the final run disproved that with the path gated OFF. Everything in §4 was designed under that wrong hypothesis — the falsifications stand as facts, but re-derive any future candidate list from "ragged trimming is unsafe", not "eager is unsafe".

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
| ~~IMA~~ | **Closed** — root cause and fix in §3 (2026-08-04 ~16:30 entry); validated 131 steps, zero IMA/hang/tripwire, ragged fully active. |
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

### Root cause (best-evidenced theory; coarse-guard fix in validation 2026-08-04)

GPU coredump-on-exception (`COREDUMP=1` in the runner) ended the guessing that
§6 used to recommend compute-sanitizer for. Three independent runs produced
corefiles naming the same kernel:
`applyMLARopeAndAssignQKVKernelGeneration<bf16, 256, 448, 64, KVBlockArray>`,
Warp MMU Fault, and disassembly at the trigger PC shows the faulting
instruction is the 128-bit **KV-cache store** (`STG.E.128`) whose address is
computed from a block offset — i.e. the kernel was handed a block table / KV
extent pair that points outside mapped memory.

The mechanism is a **write-after-read race on persistent pinned staging**: the
base class already fixed exactly this bug for `kv_cache_block_offsets` (nvbug
6293536 — see the long comment in `resource_manager.py
copy_batch_block_offsets` and `_stage_block_offsets_for_copy`): under the
overlap scheduler the CPU runs an iteration ahead, so an in-place refill of a
pinned host buffer can clobber the source of the previous iteration's
still-queued `non_blocking=True` H2D. The base class's fix was to snapshot into
a fresh pinned buffer per copy. **`dsa.py` re-introduces the unfixed pattern
wholesale**: `prepare_for_mla_rope_append` (four indptr buffers — direct
inputs of the faulting rope kernel), `prepare_for_indexer_k_cache`
(`host_indexer_k_cache_block_offsets`), `prepare_for_indices_conversion`
(`host_req_idx_per_token`), `kv_lens_expanded_host`, and the ragged row maps.
Ragged is what turns the latent race into an MMU fault: uniform decode's
next-step values differ by +1 token, but ragged verify's per-row extents and
row→request mapping change shape step to step, so a one-step-stale mix of
extents and block table walks off the allocated blocks.

Two narrower fixes failed before the mechanism was fully mapped: per-buffer
events on the ragged row maps only (fault persisted — the indptr staging was
still exposed), and skipping the copies during graph capture (a no-op:
`_forward_step` is captured on **pre-built inputs**, `prepare()` never runs
inside the capture region — verified at `model_engine.py:7322`, the captured
body is `_preprocess_inputs` + `model_forward` only).

The fix has two layers, because the racy buffers have two different *writers*:

1. **Buffers rewritten inside `prepare()`** (the DSA indptr/indexer/row-map
   staging): one coarse guard in `DSAtrtllmAttentionMetadata.prepare` —
   synchronize on an event recorded at the end of the previous `prepare()`
   before any pinned buffer is rewritten, record it after this step's copies
   are enqueued. Steady-state cost is zero. Trap discovered while validating:
   `DeepseekV4TrtllmAttentionMetadata` overrode `prepare()` and skipped
   straight to the grandparent, silently bypassing the guard — the V4 override
   is now `_prepare_impl` so the wrapper covers it. Any future subclass must
   override `_prepare_impl`, never `prepare`.
2. **`host_kv_cache_block_offsets` under KVCacheManagerV2** (the run config
   uses `use_kv_cache_manager_v2=True`): its rows are live page-index buffers
   the allocator rewrites at *scheduling* time — outside any prepare-level
   fence — while the previous step's H2D from the same buffer can still be
   queued behind an in-flight forward. Neither of V2's two copy branches had
   V1's nvbug-6293536 snapshot fix. A prepare-scope guard provably does not
   cover this: a run with layer-1 alone (fix5) still faulted at step ≥64.
   Fixed by `_stage_host_block_offsets_for_copy` in `kv_cache_manager_v2.py`
   — snapshot to a fresh pinned buffer per copy, V1 parity. This is the layer
   that matches the coredump: the block table is what the faulting store's
   address is computed from, and ragged trimming is what makes roster churn
   (block free/reassign at scheduling time) frequent enough to hit the window.

Validation note: default-gated runs now die at step ~67 on the runner's own
activity gate (`trim_ratio` ≈ −0.02) because the `4f81cc5534` gate declines
eager-ragged on no-graph steps — a config artifact, not the IMA. Use
`EAGER_RAGGED=1` runs for validation; they exercise BOTH eager-ragged and
graph-ragged (bucketed) steps in one run.

### 2026-08-04 ~07:00 — the tripwire verdict: VALUE corruption, confirmed live

The two staging-fix layers above were *necessary but not sufficient*: with both
in, 4 identical runs gave 1 pass / 3 faults. The remaining hypothesis space was
split by a zero-sync device tripwire (`TLLM_DSPARK_VALIDATE_ROWS`,
`torch._assert_async` on the exact (extent, block id) pair the rope KV-append
computes its store address from — placed after the prepare-time gather and
inside the capture body after the extent rebuild) run under
`CUDA_DEVICE_WAITS_ON_EXCEPTION`:

**The tripwire FIRED** (`CUDA_EXCEPTION_12, Warp Assert` in
`_assert_async_cuda_kernel`, frozen live on the faulting rank) — the value
pair is out of bounds *before* the rope kernel consumes it. This kills the
VMM/page-mapping family (suspend/resume, free-vs-replay unmap) outright and
reduces the problem to: **who writes a bad extent or block id, between two
validated points**. The launch-blocking algebra had already excluded
deterministic value bugs and graph-internal races (a replay is ONE launch —
CUDA_LAUNCH_BLOCKING does not serialize inside it, yet 79 bucketed steps
passed under it), so the writer is a host-side actor racing in-flight device
work on ragged-only row state.

Forensics protocol that finally worked (traps everywhere else):
- Attach from the host via `enroot exec <pid> cuda-gdb -p <pid>` — a sibling
  pyxis container cannot ptrace, the host has no cuda-gdb, and gdb through
  `/proc/<pid>/root` cannot map the target's libraries.
- NEVER wrap an attached cuda-gdb in `timeout` — killing it mid-handshake
  kills the frozen inferior (lost one scene to this).
- Do not `p *(@parameter ...)` on the assert kernel — cuda-gdb 13.2 crashes
  (out_of_range in print_cuda_exception_string) and the crash cascade tears
  the run down.
- The faulting rank is named by the peers' `timed out waiting for completion
  flag from rank R` printfs, which flush minutes after the freeze.

The tripwire now also STASHES (site, rows, bad row, extent, block id, bound,
capacity, ok, req idx, kv_len) into a plain-cudaMalloc int64[10] whose address
is logged at allocation (`DSPARK VALIDATE debug buffer @ 0x…`) — after a
freeze, `x/10gd <addr>` in the attached debugger reads the verdict directly.
site 1 = prepare-time gather produced the bad pair (bad inputs from the host
side); site 2 = it appeared only at the in-graph rebuild (a concurrent writer
touched row state between prepare and replay).

### 2026-08-04 ~09:15 — ROOT CAUSE, read off two frozen scenes

Two independent stash readouts, identical signature:

| scene | bucket | row | ext | block id | req | kv_lens_cuda | drift vs allocation |
|---|---|---|---|---|---|---|---|
| i / rank6 | 256 | 60 | 1221 | −1 | 10 | 1226 | ≈ +70 |
| j / rank2 | 96 | 36 | 926 | −1 | 6 | 931 | ≈ +35 |

Both **site 2** with a clean site 1 in the same step. Since both sites use the
same formula on the same table, site1-clean + site2-trip forces one
conclusion: `kv_lens_cuda[slot]` moved between the prepare H2D reset and the
in-graph rebuild — and the only mutation in that interval is the captured
overlap correction `kv_lens_cuda += previous_kv_lens_offsets_cuda`. Its legal
values are bounded by one verify window (≤7); the observed drift is tens of
tokens.

The unbounded source: `previous_kv_lens_offsets` is built as
`new_tokens_lens_device[previous_slots] − tokens_per_previous_request`
(`model_engine.py`, ragged branch), and `new_tokens_lens` is the **spec
sampler's slot-indexed store, allocated with `torch.empty`** and rewritten
each step only for the slots actually sampled. A slot the sampler never wrote
reads as arbitrary bytes; one garbage offset pushes the device KV length past
the host's block allocation, the in-graph extent rebuild follows it, and the
rope KV-append store walks into a −1 block entry → unmapped address → Warp
MMU Fault.

Every historical constraint checks out: **uniform never faults** because a
stable roster rewrites every slot every step (even a stale read is bounded);
**ragged faults** because trimming churns completions and the fault window
(step ~64–90) is exactly when batch concurrency reaches new peaks, i.e. when
slots are read before their first write; **CUDA_LAUNCH_BLOCKING passes**
because it serializes the sampler's store writes ahead of the next step's
staging gather; the staging-fix layers never touched it because the offsets
gather is device-side, not a pinned-host copy.

Fix (`sampler.py int_tensor` + `model_engine.py` ragged offsets):
1. Slot-indexed sampler stores are **zeros, not empty** — an unwritten slot
   reads 0.
2. The ragged offsets gather is **clamped to ±(window)** — a stale value is
   then at worst one window of KV length, never an out-of-bounds address.

Note the residual: a *stale-but-clamped* offset is still a small KV-length
error on roster-churn steps (silent, bounded). The clean long-term fix is an
explicit ordering guarantee (event) between the sampler's store writes and
the next step's previous-batch gather; not attempted yet — measure first.

### 2026-08-04 ~16:30 — THE ROOT CAUSE, self-described by the stamp tripwire

The kv-lens-drift reading above was a *symptom* misread (the apparent drift
was a stale row→request map gathering the new roster's kv_lens). The final
mechanism, caught by a prepare-tick/staged-tick stamp pair raising on the
step itself:

```
DSpark ragged staleness: prepare tick 559, staged tick 558, skip reason None
```

`skip reason None` = the token-major staging chain was NEVER INVOKED that
step: `_ragged_token_lens` returned None (a generation request without a
window), so `_attach_ragged_verify_layout` set `ragged_verify_lens = None`
and prepare skipped every row-map/table refresh — while the graph key still
carried the fit's `agreed_ragged_bucket`, so the step REPLAYED a
ragged-bucket graph over the PREVIOUS ragged step's row state: stale
row→request mapping, stale extents, stale per-row block tables. Wherever the
retired roster's allocation was shorter than a current extent, the rope
KV-append read a −1 block entry and stored to an unmapped address.

`_ragged_verify_bucket`'s docstring always promised "None unless *every*
generation request carries a window"; the code returned the published bucket
without checking. **Fix: make the code match the docstring** — a step whose
windows and bucket disagree derives the uniform key or falls back to eager
(`no_captured_shape`), never a ragged replay it staged no rows for.

Why every earlier observation fit a race: the divergence between the fit's
bucket and the requests' windows is timing/roster dependent (planner or peer
declining after the fit, completions between fit and key derivation), so the
fault was intermittent, followed the first captured buckets by a few dozen
steps, never touched uniform (no bucket key), and vanished under
CUDA_LAUNCH_BLOCKING's pacing. Every WAR/staging fix left it untouched
because nothing was racing — the replay was reading last step's rows by
design once the staging chain was skipped.

First validation (fix11_a): **131 steps, zero IMA/hang/tripwire**, ragged
fully active (`steps_ragged` 38, `trim_ratio` 0.5375, 527 replays), the
consistency decline fired only 6 times (`no_captured_shape: 6`). GSM8K
95.944 vs the 96.0 gate — within one σ (≈0.54 at n=1319) of the historical
96.1–96.4 band; two more samples running to separate noise from a real
regression (suspect list if real: the offsets clamp, the trim-schedule
shift from declined steps).

The investigation scaffolding (zero-sync device tripwire + forensic stash,
prepare/staged tick stamps, instance markers, pre-replay staleness assert)
was stripped from the tree once the fix was validated — this document and
`tmp/autopsy_recipe.md` (frozen-scene attach protocol, kernel param-space
offsets, the never-wrap-cuda-gdb-in-timeout rule) are the durable record.
Rebuilding any piece takes an hour with this doc; keeping them all in-tree
was judged not worth the noise.

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

1. #19 (budget quantisation — the largest measured gap, up to 73% of the planner's own objective), then #11 (sync control; until it completes, do not cite any sync conclusion) and #20 (the three-way static / cap-accept / compact differential). All IMA work has landed: coarse prepare() WAR guard with the `_prepare_impl` subclass rule (`c7e6d78ac2`), KVCacheManagerV2 snapshot staging (`74f1a82d06`), bounded/zeroed overlap staging inputs (`5c38441a00`), window-consistent ragged graph key (`6a59b01e53`), and the eager-ragged gate reverted on its merits (`35a1ac4c4a`).

### Do not

- Do not resolve by declining ragged whenever a graph is unavailable. Those steps
  were eager before this feature existed (`can_run_cuda_graph` is
  `num_context_requests == 0`), so declining buys no graph replay — it only
  forfeits trimming, measured at 46 of the first 64 steps, `trim_ratio` 0.62→0.
- Do not cite `docs/dspark_p0_task_prompt.md` §10 sync numbers; the control never
  ran.
- Do not trust a throughput number taken with the synthetic steep table.

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
