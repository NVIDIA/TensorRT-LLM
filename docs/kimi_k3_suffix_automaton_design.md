# Kimi K3 ("golden prairie") suffix automaton spec dec — design & implementation plan

Status: design, SA-0 rebase complete (2026-07-21). Base: the rebased NGram
verify foundation — `brnguyen/kimi-k3-specdec-eval` @ `b3cf9a039d`
(19 commits on `feat/golden_prairie` @ `6f1af56d1c`); branch
`brnguyen/suffix-automaton` stacks on it.
Companion docs: `kimi_k3_specdec_phase0.md` / `kimi_k3_specdec_phase1_design.md`
(removed from the MR branch tip in `d63d509422`; in its git history).
NGram MR !9 (closed, verification-side context + validation record):
https://gitlab-master.nvidia.com/ftp/tekit-golden-prairie/-/merge_requests/9
Scope: enable Suffix Automaton (SA) speculative decoding for Kimi K3, building
on the NGram-branch verify foundation. **Decision (Sharan): SA-only — NGram
will not be a supported K3 spec-dec mode** (SA dominates it: GPU-native,
overlap-scheduler and CUDA-graph capable, dynamic draft length, global pool).
NGram survives only as a bringup diagnostic on this branch and is excised
before the final MR.

## TL;DR

1. **SA is fully implemented in-tree** (`SADecodingConfig`, GPU-native
   C++/CUDA automaton kernels, nanobind bindings) — but it is a **one-engine
   in-forward worker** (`SAWorker`, like MTP/Eagle3), *not* a host-side
   drafter like NGram. This changes the K3 integration surface vs the NGram
   plan: K3 must adopt the `SpecDecOneEngineForCausalLM` base class, and the
   recurrent-state promote call site moves *into* `SAWorker.forward`
   (the dflash/eagle3 pattern) instead of the host-drafter
   `update_resources` hook.
2. **The verification machinery from the NGram branch is the hard part and
   carries over unchanged** — KDA multi-token verify + MLA multi-token
   metadata are draft-source-agnostic. That work exists on
   `brnguyen/kimi-k3-specdec-eval` (rebased onto ToT; full-model text
   parity in flight, job 2646769 — result posted to MR !9); this branch
   stacks directly on it. The SA-only decision *improves* MTP
   transferability: SA shares MTP's one-engine skeleton (base class,
   in-forward worker, in-forward promote), whereas NGram's one
   non-transferable piece — the host-drafter promote hook in
   `update_resources` (`203a867326`) — now never ships.
3. **Rollback strategy (Michael Iovine's guidance)**: naive rollback would
   require caching every intermediate recurrent state; the smarter "replay"
   approach reconstructs from checkpoints. At this ToT, **replay exists only
   for Mamba2/Nemotron-H** (`replay_selective_state_update`); GDN/Qwen3-Next
   still uses full intermediate buffering (PR #16464 not merged here), and
   the replay kernel math is Mamba2-specific — not directly applicable to
   KDA's delta rule. **Phase 1 uses full intermediate buffering
   (verify-and-promote)**, exactly like the NGram plan; a KDA replay kernel
   is a scoped future optimization.
4. K3's spec-dec assert is still present at ToT
   (`modeling_kimi_linear.py:986-987`); no spec-dec support was merged.
   Tali's new fused KDA decode CUDA op (`fd36b17d27`) is **T=1 only**
   (`_kda_decode.py:110`) — multi-token verify keeps the FLA sequential
   path for now.

## What's in-tree: SA architecture (ToT survey)

### Config & mode

- `SADecodingConfig` (`llmapi/llm_args.py:2314-2371`): PyTorch backend only,
  linear tree (`max_total_draft_tokens = max_draft_len`),
  `max_matching_ngram_size` (-1 = longest match, ≥1 = fixed),
  `enable_global_pool` + `global_pool_size` (cross-request pattern reuse,
  FIFO eviction, must be ≥ `max_batch_size`, ≤1024 slots, ngram ≤64).
- `SpeculativeDecodingMode.SA` (`speculative/interface.py:272`), properties:
  `use_one_engine()` **True**, `has_spec_drafter()` **False** (vs NGram
  True), `has_draft_model()` False, `extend_ctx()` **False** (one-engine
  early return), `needs_kv_cache_rewind()` True,
  `support_overlap_scheduler()` **True** (vs NGram False),
  `support_dynamic_draft_len()` **True**, `needs_kv_cache_recompute()`
  False. Rejection sampling explicitly rejected
  (`llm_args.py:5125-5161`) — greedy acceptance only (no proposal
  distribution), so spec dec stays lossless and parity-checkable.
- There is also `SAEnhancerConfig` (`sa_config=` on Eagle3/MTP/PARD,
  `sa_enhancer.py`): SA overrides neural drafts when match_len ≥ threshold,
  computed on a side stream. Irrelevant until K3 has MTP weights — noted
  under Future work.

### Runtime pieces

- `SAWorker(SpecWorkerBase)` (`speculative/sa_worker.py:105+`): runs in the
  model's epilogue inside forward. Samples target logits, accepts/rejects
  prior drafts via `_sample_and_accept_draft_tokens_base` (on-device
  `num_accepted` — exactly what `update_mamba_states` wants), then
  generates next drafts via the automaton (`_generate_draft_tokens:270-321`,
  rows gated by `match_len > 0`, zeroed otherwise — shapes stay uniform,
  CUDA-graph safe). `runtime_draft_len == 0` → `skip_drafting`.
- `SuffixAutomatonManager(BaseResourceManager)`
  (`speculative/suffix_automaton.py`): per-request automata built on host
  (`build_automaton_host`), copied to fixed GPU slots on `prepare()`;
  native kernels `invoke_extend` / `invoke_extend_ngram` /
  `invoke_global_search`
  (`cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/`).
  Dummy slot at index `pool_size` for CUDA-graph padding.
- Model attachment: `get_spec_worker` (`speculative/utils.py:468-469`)
  creates `SAWorker` **only for models subclassing
  `SpecDecOneEngineForCausalLM`** (`modeling_speculative.py:1880-1965`;
  SA skips draft-model creation, worker appended to epilogue). Llama and
  Qwen3-Next subclass it; **`KimiLinearForCausalLM` subclasses plain
  `DecoderModelForCausalLM` (`modeling_kimi_linear.py:979`) — base-class
  swap required.**
- Executor gating: FLASHINFER attention raises for one-engine modes
  (`py_executor_creator.py:431-445`) — K3 uses TrtllmAttention, fine.
  Unlike NGram, SA does **not** auto-disable the overlap scheduler; K3
  requires it off, so it must be disabled in config (or a K3-specific gate
  added).

## The rollback problem and where replay stands (ToT facts)

Michael's framing: linear KV caches discard rejected draft tokens by
pointer rewind; SSM/recurrent layers need the *state* rolled back. Two
in-tree strategies, selected by `use_replay_state_update` on the mamba
cache manager (`mamba_cache_manager.py:240-270`):

| | Legacy full buffering | Replay (checkpoint) |
|---|---|---|
| Buffers | `intermediate_ssm` `[slots, draft_len+1, <state>]` + `intermediate_conv_window` | compact `old_x/old_B/old_dt/old_dA_cumsum` double-buffered checkpoints + `prev_num_accepted_tokens` + `cache_buf_idx` |
| Verify kernel | writes every per-step state | updates state in-place, reconstructs from checkpoint on demand (`replay_selective_state_update`) |
| Commit (`update_mamba_states:787-841`) | gather accepted step → scatter into live state | host mirrors kernel's checkpoint predicate, flips buffer index |
| Memory | O(draft_len+1) × state per slot | O(history) × compact inputs |
| Who uses it | GDN/Qwen3-Next verify (`gdn_mixer.py:384-535`, FlashInfer MTP kernel or Triton fallback) | Mamba2/Nemotron-H (`mamba2_mixer.py:409-566`), gated `sm≥80`, env `TRTLLM_USE_MAMBA_REPLAY` |

Facts constraining the K3 choice:

- **Replay math is Mamba2-specific.** `replay_selective_state_update`
  reconstructs via the Mamba2 selective-state-update recurrence; KDA's
  gated delta rule needs its own replay kernel. PR #16464 (GDN replay,
  `fla/cached_replay.py`) is **not** at this ToT and is GDN-shaped anyway.
- **Replay asserts fixed draft length** (`mamba2_mixer.py:427-431`) —
  in tension with SA's `support_dynamic_draft_len()`. Any future KDA
  replay must resolve this or pin the draft length.
- Kimi's manager construction (`_util.py:1855-1915`) does not pass
  `use_replay_state_update` → K3 defaults to **legacy buffering**, and
  `MixedMambaHybridCacheManager` already allocates `SpeculativeState`
  buffers when `spec_config` is set (`:1087-1091`) — the infrastructure
  Phase 1 needs is present.

**Decision: full intermediate buffering for enablement.** Cost estimate:
per KDA layer per slot, intermediate SSM is
`(draft_len+1) × H_per_rank × 128 × 128 × 4 B` fp32 (+ bf16 conv windows),
× 69 KDA layers × active slots. At small draft lengths (2-4) this is
manageable; it is the lever that motivates a KDA replay kernel if SA wants
long drafts (SA's longest-match mode profits from larger `max_draft_len`
on repetitive workloads — measure first, optimize after).

## Gap analysis: what breaks for SA-on-K3

Inherited from the NGram branch (already designed/implemented on
`brnguyen/kimi-k3-specdec-eval` — see `kimi_k3_specdec_phase1_design.md`):

1. `assert spec_config is None` (`modeling_kimi_linear.py:986-987`).
   The NGram branch admits `is_ngram()` (`a705fcbd3a`); **final form
   admits only `is_sa()`** (NGram admission is bringup scaffolding,
   removed at the history rewrite).
2. KDA decode is strictly T=1 (`kimi_kda_mixer.py` q_len==1 assert; in-place
   `index_copy_` state writes would be corrupted by rejected tokens) →
   multi-token verify writing per-step states into `SpeculativeState`
   scratch, commit deferred to `update_mamba_states`.
3. MLA runtime assumes 1 token/request → per-request `q_len = draft_len+1`
   metadata on the existing MTP-style generation FMHA (analysis says zero
   kernel changes).

SA-specific deltas (this effort):

4. **Base class**: `KimiLinearForCausalLM` → `SpecDecOneEngineForCausalLM`
   so `get_spec_worker` attaches `SAWorker` + `SASampler` + `SASpecMetadata`
   wiring works. SA creates no draft model, so the swap is low-risk — the
   forward epilogue and logits path are the observable changes; verify
   baseline parity (no spec config) after the swap alone.
5. **Promote call site**: the NGram branch added a host-drafter hook in
   `MixedMambaHybridCacheManager.update_resources`, constructor-gated on
   `decoding_type == 'NGram'` (`203a867326` + freed-slot fix
   `a81c83787c`) — **SA won't hit it**, and under SA-only it is
   scaffolding that never ships (dropped at the history rewrite). SA
   follows the one-model pattern instead: call
   `kv_cache_manager.update_mamba_states(...)` inside `SAWorker.forward`
   right after acceptance, guarded by
   `isinstance(kv_cache_manager, MambaHybridCacheManager)` — mirroring
   `dflash.py:414-419` / `eagle3.py:717-721`. `num_accepted` is already an
   on-device tensor there (no host assembly needed, unlike the NGram hook).
6. **Dynamic draft length**: SA honors `runtime_draft_len` (uniform across
   the batch each step; per-request variation is expressed as zeroed draft
   rows, later rewound). The KDA verify loop must take T from spec
   metadata rather than assuming `max_draft_len`, and tolerate
   `skip_drafting` steps (T=1 fast path). The GDN precedent already reads
   `spec_metadata.runtime_draft_len`.
7. **Executor config**: disable overlap scheduler explicitly for K3 (SA
   does not auto-disable it, NGram did); CUDA graphs stay off (K3 eager) —
   SA's graph-safety is a bonus for later, not a requirement.
8. **SA state feeding**: `SuffixAutomatonManager.add_request` builds the
   automaton from prompt tokens and `extend*` appends accepted tokens each
   step — request lifecycle is handled generically by the resource manager
   (`utils.py:348-350`); no K3-specific work expected, verify in Phase 2.

## Dependencies & branch strategy

- `brnguyen/kimi-k3-specdec-eval` @ `b3cf9a039d` carries items 1-3: guard
  relaxation, KDA `_forward_verify`, promote plumbing, the matured parity
  harness (logits mode, calibrated tolerances, multi-prompt statistics,
  CI integration test + L0 entries), KDA unit test, spec stats reporting.
- **Branch plan (agreed with Sharan's SA-only call):**
  1. ✅ **NGram MR !9 closed** without merging; validation record lives in
     the MR (unit test 4/4; 3-layer and 4-layer e2e; 4-layer logits
     parity PASS post-rebase, job 2646634; 48-prompt statistics PASS,
     job 2646765 — 52/52 divergences near-tie, 0 drift; full-model text
     comparison 2/4 bit-exact with divergences forensically benign,
     job 2646769). Full-model logits-parity **certification** + NGram
     acceptance/tokens-per-step baseline job (2646841) was CANCELLED
     (not worth 16 GPUs for a closed MR) — SA-3b regenerates its own
     baseline + ngram-diagnostic numbers on a consistent FUSED_MOE
     setting.
  2. ✅ **This branch rebased onto `brnguyen/kimi-k3-specdec-eval`**
     (`b3cf9a039d`, itself rebased onto ToT `6f1af56d1c`). All 19 commits
     ride along, including NGram-only ones — useful bringup scaffolding
     (diagnostic drafter, parity-harness structure).
  3. **Once SA is working end-to-end, rewrite history before the new MR**:
     drop the NGram-only commits (host-drafter promote hook
     `203a867326` + `a81c83787c`'s hook parts, NGram guard admission from
     `a705fcbd3a`), keep the shared verify core (`2f89d48a1e`,
     `23a43a95ff`) and the parity/CI harness (retargeted to SA), and
     squash the SA work into a reviewable sequence. The final MR contains
     zero NGram support.
- Non-dependency: KV cache reuse (#16427/#16598) — orthogonal, unchanged
  from the Phase 0 analysis.

## Implementation plan (phases)

### Phase SA-0 — foundation sync (✅ complete except final drain)

- ✅ NGram MR !9 closed; validation recorded there.
- ✅ Rebase complete: this branch = `b3cf9a039d` + design doc. The
  NGram-diagnostic parity already ran green **on the rebased base**
  (jobs 2646634 logits parity, 2646765 48-prompt statistics) — proving
  KDA verify + MLA metadata + promote machinery against ToT (incl. the
  KDA thop op and MoE integration) independent of SA.
- ⏳ Full-model 16-GPU logits-parity certification + NGram
  acceptance/tokens-per-step baseline run cancelled (2646841; closed-MR
  economics) — SA-2.5 is the first full-model certification to actually
  run, and SA-3 generates its own perf baselines. (Text-comparison run 2646769: 2/4
  bit-exact, divergences forensically benign — motivated the shift to
  certification; see MR !9.)
- The NGram path stays available on this branch as the triangulation
  tool (SA fails + NGram passes → SA integration bug; both fail →
  verify bug) until the pre-MR history rewrite removes it.

### Phase SA-1 — one-engine adoption

1. ✅ Base-class swap to `SpecDecOneEngineForCausalLM` (`7af0f970d6`;
   guard admits `is_sa()` + NGram-diagnostic).
2. ✅ **No-spec regression gate** (jobs 2647196 baseline-dump / 2647240
   compare / 2647262 control, 52 prompts, 4-layer): PASS. Finding:
   **bitwise logit identity does not exist on this stack even
   run-to-run** — the control (pre-swap code vs its own dump, tol=0)
   failed with the same noise as the swap comparison
   (48 vs 49 drift events; medians 0.0013 vs 0.0014; no directional
   bias; no positional growth). FUSED_MOE=1 kernels are
   nondeterministic run-to-run. Revised criterion, met: swap-run noise
   statistically indistinguishable from a same-code control. Any future
   "identity" gate on K3 must be control-calibrated, not tol=0.
3. Wire `SASpecMetadata`/`SASampler` path end-to-end on the truncated
   model; expect failure at the promote step (next phase).

### Phase SA-2 — recurrent-state promote for one-engine

1. Add the `update_mamba_states` call in `SAWorker.forward` post-acceptance
   (dflash pattern, isinstance-guarded). Handle the finished-request /
   freed-slot case (the NGram branch hit this: freed mamba slots must
   redirect to the padding slot — commit `a81c83787c`; check whether the
   one-engine timing avoids it or needs the same fix).
2. Dynamic-T handling in KDA verify (T from spec metadata, T=1 fast path
   on skip_drafting).
3. Gate: 3-layer pure-KDA, then 4-layer KDA+MLA, **logits-parity mode**
   with `SADecodingConfig` (`KIMI_K3_SPEC_MODE=sa` retargeting the
   harness knob). The NGram MR established that bit-identical *text*
   parity is provably the wrong test at truncated scale (the verify
   forward batches T× tokens through the MoE, changing reduction order;
   noise logits flip greedy argmax on rounding) — the truncated-scale
   criterion is the matured logits methodology: drift-free shared
   prefixes = hard fail on drift, divergences must be near-ties, plus
   the aggregate multi-prompt systemic check (~50 prompts).
4. **SA-specific harness adaptations** (base harness landed and
   validated on the NGram branch: `22501a7f6f` + calibration/statistics
   follow-ups, CI test in `test_kimi_k3_specdec.py` + L0 entries):
   (a) log acceptance events alongside per-position drift — a promote
   off-by-one corrupts recurrent state cumulatively and may not flip an
   argmax for many steps, while drift onset aligned with the first
   acceptance event fingerprints the promote path directly;
   (b) include deliberately repetitive prompts — SA only drafts on
   automaton matches (`match_len > 0`), so non-repetitive prompts can
   pass parity without exercising the machinery at all (fail loudly on
   zero acceptance events, don't warn);
   (c) ~~adapt logprob extraction to the `SASampler` one-engine path~~
   **RESOLVED as a structural limitation (2026-07-21, job 2647516)**:
   one-engine spec samplers (SA/MTP/Eagle3) do not emit per-token
   logprobs at all (`SpecSamplerBase.Store` holds tokens only; the
   mtp/eagle3 "logprob" code is internal acceptance math). SA
   certification is therefore **one-sided**: shared-text-prefix
   comparison + near-tie classification from the *baseline's* top-5
   logprobs (`b7b5c36b53` harness fallback). The drift-trend check is
   lost on the spec side — compensated by the exact KDA verify unit
   test (kernel-level precision) and the systematic-corruption
   signature that one-sided classification still catches. Per-token
   logprob support in `SpecSamplerBase` is a candidate upstream
   feature (would also serve MTP/Eagle3 users).

### Phase SA-2.5 — full-model logits-parity certification (pre-GSM8K gate)

Sanity-scale full-model run (16-GPU): **logits-parity certification** —
drift-free shared prefixes, near-tie gaps at every divergence, plus
acceptance-rate / tokens-per-step stats — SA enabled, **>50 prompts**
(match the NGram certification's 52-prompt set for comparability;
statistical near-tie classification needs the sample size at full scale
just as it did truncated), including the repetitive-prompt subset. Strict text equality is *not* the criterion even at full scale:
the NGram full-model run (MR !9) showed 2/4 bit-exact with the
divergences forensically identified as benign near-ties (one-token
whitespace flip with post-flip reconvergence — impossible under state
corruption). Batched verification changes MoE reduction order, so the
losslessness guarantee is **distributional, not bitwise**; the
certification methodology (52-prompt cross-process logits parity) is
the standard this branch inherits. Same
weights/parallelism as GSM8K at a fraction of the cost — catches
full-model-only numerics (EP, real MoE routing) before committing to
the multi-hour eval, and is the cheap iteration loop if SA-3a fails.

### Phase SA-3 — full-model validation & measurement

1. Full-model 16-GPU sanity with SA; GSM8K parity (96.74 ±0.49; partial
   scores via `TLLM_EVAL_PARTIAL_SCORES_EVERY` for early signal).
2. Measure: acceptance rate, mean accepted length, tokens/step, e2e
   speedup vs baseline and vs a fresh NGram-diagnostic run at matched
   `max_draft_len` (the MR !9 cert baseline was cancelled; regenerate
   both on one FUSED_MOE setting for comparability); sweep
   `max_matching_ngram_size` (-1 vs fixed) and `max_draft_len` ∈ {2,4,8}.
   GSM8K's repetitive arithmetic phrasing should favor SA/NGram; also run
   a code-generation sample (higher repetition) for the upside case.
3. Try `enable_global_pool` (size ≥ max_batch_size) — cross-request reuse
   on homogeneous eval workloads is SA's differentiator over NGram.

### Phase SA-3.5 — history rewrite & MR

Once SA-3 gates pass: rewrite branch history per the branch plan — excise
NGram-only commits/scaffolding (guard admission, host-drafter promote
hook, ngram harness knob), keep the shared verify core + SA work as a
clean reviewable sequence, then open the MR → `feat/golden_prairie`.
Post-rewrite, re-run the SA-2 truncated-model gates once as a
squash-correctness check before submitting.

Dependent branches (decided): the MTP-prep follow-ups —
`brnguyen/kda-decode-mtp-kernel` (CuTe verify kernel) and
`brnguyen/kimi-k3-mtp-scaffold` (structural MTP wiring) — hold their MRs
until the SA MR exists, then **target the SA branch** and retarget to
`feat/golden_prairie` once SA merges. The rewrite invalidates their
bases: each must `rebase --onto <rewritten-SA-branch> <old-base>`
immediately after the rewrite, before opening its MR. The MTP
head-analysis doc has no code and rides this MR's `docs/`.

### Upstream spin-off (post SA-2, standalone GitHub PR)

The SAWorker promote fix (`0cf831efa3`) is model-agnostic and fixes a
real upstream bug: on GitHub main, standalone SA on any hybrid
(Qwen3-Next subclasses `SpecDecOneEngineForCausalLM`, so SA attaches)
runs without ever promoting accepted recurrent states — silent state
corruption; uncaught because the SA test matrix is llama-only. File as
its own PR to `main` after SA-2a/2b pass (K3 parity = the empirical
evidence), bundled with an SA-on-hybrid accuracy test (parameterize
`test_suffix_automaton` onto Qwen3-Next/Nemotron-H) and lifting any
gate that currently blocks SA+hybrid. Interface (`update_mamba_states`)
survives the #16598 rewrite per the Phase-0 analysis.

### Phase SA-4 — perf follow-ups (measure-first, each optional)

- **KDA replay = adopt Moonshot's `KDA_decode_mtp` kernel** (VALIDATED
  2026-07-21 on `brnguyen/kda-decode-mtp-kernel`, 3-way parity vs the
  drop's CPU golden and the FLA sequential reference, jobs
  2647504/2647518/2647547): the kernel implements replay-style state
  management *internally* — pre-verification state pool + per-request
  replay of `num_accepted` cached tokens + self-commit after the golden
  token. Adopting it **replaces** the `SpeculativeState`
  intermediate-buffer + `update_mamba_states` flow for KDA layers (the
  hand-written replay port this bullet previously described is
  unnecessary). Caveats: (a) shipped replay-path bug (CuTe DSL tracing;
  two-line hoist, fixed in our test harness — upstream to Moonshot);
  (b) head coverage H ∈ {2, 12, 32} only — **H=8 (state-TP4 truncated
  sanity) hard-fails**, so truncated-scale gates keep the FLA path;
  (c) NUM_SPEC is a JIT Constexpr (only 2 benchmark-tuned) — SA dynamic
  draft length needs per-length recompiles or pinning. No thop op
  needed: CuTe DSL JITs from Python like FLA's Triton kernels.
- **CUDA graphs**: SA is graph-safe by design; K3's eager constraint is the
  blocker — revisit when K3 gains graph support.
- **SA enhancer on MTP**: when MTP-head weights drop, `sa_config=` on the
  MTP config combines neural drafting with automaton override — the
  natural endgame; everything in SA-1..3 is prerequisite to it.

## Transferability to MTP (the endgame)

Division of labor across the two efforts, and what remains for MTP when
draft-head weights drop:

| Piece | Plumbed by | Transfers to MTP? |
|---|---|---|
| KDA multi-token verify, `SpeculativeState` scratch, `update_mamba_states`, MLA multi-token metadata | NGram branch (`2f89d48a1e`) | ✅ draft-source-agnostic |
| Parity/certification methodology + CI harness | NGram branch | ✅ reruns with a new draft source |
| Host-drafter promote hook in `update_resources` | NGram branch (`203a867326`) | ❌ host-drafter-only — dropped at history rewrite |
| One-engine base class on K3 + composite-config fix (`7af0f970d6`) | SA (this effort) | ✅ hard MTP prerequisite (same attach path); SA-1 identity gate certifies the swap, so MTP starts from a validated base |
| In-forward promote + freed-slot timing on `MixedMambaHybridCacheManager` | SA (SA-2) | ✅ the dflash/eagle3/MTP pattern, debugged once |
| `spec_metadata` plumbing, epilogue logits path, spec-sampler integration | SA | ✅ shared one-engine machinery |

MTP-only remainder: draft-head weight loading, hidden-state capture, and
the rejection-sampling executor path (SA/NGram are greedy-acceptance
only). Net: NGram plumbed verification, SA plumbs one-engine
integration — MTP reduces to weight loading plus the sampling-path
switch, and `sa_config=` on `MTPDecodingConfig` composes the automaton
override with neural drafting on top of everything built here.

### Pre-weights MTP prep (parallelizable, none on this branch's critical path)

Work that needs no draft-head checkpoint, in descending value/effort:

1. ✅ **`KDA_decode_mtp` CuTe kernel — VALIDATED** (2026-07-21, branch
   `brnguyen/kda-decode-mtp-kernel`, 3-way parity: CuTe vs the drop's
   self-contained CPU golden vs the FLA sequential reference; jobs
   2647504/2647518/2647547, 5 pass + 1 strict xfail). Key findings:
   the kernel does **replay-style state management internally** (see the
   SA-4 bullet — it replaces the intermediate-buffer + promote flow for
   KDA, not just the verify math); a shipped replay-path bug was found
   and fixed (two-line CuTe-DSL hoist — upstream to Moonshot); coverage
   H ∈ {2, 12, 32} (H=8 / state-TP4 sanity shape unsupported);
   NUM_SPEC=2 tuned. No thop op needed — CuTe DSL JITs from Python.
2. **MTP-head input convention under attn_res** — K3's per-token snapshot
   mixing makes "which hidden state feeds the MTP head" ambiguous
   (pre-norm? post-attn_res mix?). Decided by Moonshot's training;
   discoverable now from HF `modeling_kimi.py` + the kernel-drop wrapper.
   Classic silent-quality-loss bug if guessed wrong; pure analysis.
3. **Structural wiring vs a synthetic head** — `mtp_layers` construction,
   `spec_metadata.maybe_capture_hidden_states` plumbing, and the
   streaming-loader `checkpoint_name_plan` extension for MTP-head keys,
   tested against a tiny random-weight safetensors in the anticipated
   schema. Verifies shapes/naming/capture mechanically; keep thin — the
   schema is a guess until the drop.
4. **Rejection-sampling mechanics under promote** — MTP uses one-model
   rejection sampling (TRTLLM-13321), not greedy acceptance; the promote
   machinery consumes `num_accepted` source-agnostically but that is
   untested on K3. A random-weight head stresses reject-all; a
   copy-of-target-layer fake head yields partial accepts at T>0.
5. **Combination constraint to settle** — `sa_config` on
   `MTPDecodingConfig` hard-disqualifies rejection sampling
   (`llm_args.py:5125-5161`; the automaton has no proposal
   distribution). "MTP + SA enhancer" therefore implies greedy
   acceptance — decide the posture before building on both.
6. **Structural perf prereqs** — CUDA graphs + overlap scheduler are off
   for K3; one-model MTP's latency win partly assumes them. Independent,
   chunky effort; schedule awareness only.

Standing Moonshot asks: MTP checkpoint schema (de-risks item 3) and the
kernel reference (item 1 mostly removes the need).

## Prior art: how in-tree spec-dec validation works, and how this plan maps

In-tree spec-dec validation has two tiers; neither does full-scale text
parity:

- **Tier 1 — functional smoke**
  (`tests/unittest/_torch/speculative/hw_agnostic/test_sa.py`): asserts
  `text_spec == text_ref` plus behavioral facts (drafts produced,
  acceptances occurred, multi-token acceptance) — but under conditions
  where near-tie flips are vanishingly rare: dense Llama-3.1-8B, one
  prompt, 64 tokens, greedy. The exact-match assert is an artifact of
  that easy regime, not a general losslessness claim. K3 is the opposite
  regime (MXFP4 MoE; batched verify changes expert-sum reduction order —
  MR !9 forensics showed exactly this flipping near-ties).
- **Tier 2 — the standard full-model evaluation**
  (`tests/integration/defs/accuracy/`): for Qwen3-Next, SA, Eagle3,
  PARD, MTP alike, the gate is task accuracy (GSM8K/MMLU) **with spec
  dec enabled** clearing a one-tailed hypothesis-test threshold vs a
  reference score (`accuracy_core.py:57`, α=0.05). No text comparison
  against a non-spec run. "Lossless" is operationalized as
  *statistically indistinguishable task accuracy* — distributional by
  construction.

This plan is a superset of both tiers applied to K3+SA, with two honest
footnotes: (a) Tier 1's exact-text assert is *replaced* by the logits
certification — strictly more sensitive where both are valid (catches
sub-argmax drift), and the only well-defined form on K3; the behavioral
asserts carry over as-is into the K3 CI test. (b) Tier 1's
overlap-scheduler × CUDA-graph matrix shrinks by constraint (K3 pins
both off), not omission. On top: the exact kernel-level KDA verify unit
test, the no-spec identity gate (SA-1), truncated-scale logits parity
(SA-2), full-model >50-prompt certification (SA-2.5), and the NGram
speedup baseline (SA-3b). Tier 2 is included verbatim as SA-3a.

## Validation matrix (gates, cumulative)

| Gate | Config | Pass criterion |
|---|---|---|
| SA-0 | 3/4-layer, NGram (diagnostic only) | ✅ logits parity + 48-prompt stats green on rebased base (jobs 2646634, 2646765); full-model cert run cancelled — first full-scale cert is SA-2.5 |
| SA-1 | 4-layer, no spec, 52 prompts | ✅ swap noise ≈ same-code control noise (bitwise identity impossible: FUSED_MOE=1 is run-to-run nondeterministic; jobs 2647240/2647262) |
| SA-2a | 3-layer (pure KDA), SA, logits mode | ✅ job 2647576: 52 prompts, 3 identical, 49 divergences, 0 non-tie (one-sided mode); tokens/step 1.016 (SA fired, promote exercised) |
| SA-2b | 4-layer (KDA+MLA), SA, logits mode | ✅ job 2647598: 52 divergences, 6 non-tie (borderline 0.31-0.56), 0 drift — confirmed mode-independent noise by ngram triangulation (2647620: 9 non-tie, same band, 0 drift with two-sided check, tokens/step 1.080) |
| SA-2.5 | full model, SA, >50 prompts | logits-parity certification (drift-free prefixes, near-tie divergences) + acceptance stats (pre-GSM8K gate) |
| SA-3a | full model, SA | sanity clean; GSM8K 96.74 ±0.49 |
| SA-3b | full model, SA | acceptance/speedup report; ≥1.0× e2e (no regression), speedup target set after first measurement |
| SA-3.5 | 3/4-layer, SA (post-rewrite) | SA-2 gates green on rewritten history |

## Open questions

1. Freed-slot timing under one-engine promote (Phase SA-2.1): does the
   in-forward call site dodge the freed-mamba-slot KeyError the
   host-drafter hook hit, or is the padding-slot redirect needed here too?
2. `SuffixAutomatonManager` slot lifecycle under K3's
   `MixedMambaHybridCacheManager` coexistence — both are resource managers
   keyed by request; confirm ordering in `ResourceManager.update_resources`
   (SA's `update_resources:609` is state-append only, expected benign).
3. Overlap scheduler: leave "disabled via config" or add a K3 model gate
   (as `py_executor_creator.py` does for NGram)? Config-only is fine for
   bringup; a gate prevents silent misconfiguration for external users.
4. Does `SASampler`'s d2t/logits handling interact with K3's MTP-less lm
   head in any surprising way? (Expected no — SA has no draft model.)
