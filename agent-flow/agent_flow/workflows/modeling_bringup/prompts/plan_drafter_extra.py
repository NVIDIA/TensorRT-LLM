"""Modeling-bringup-specific guidance appended to the PlanDrafter prompt."""

from ._common import (
    ACCURACY_GAP_PARITY_POLICY,
    ACCURACY_GATE_FRAMEWORK,
    ATTENTION_SCOPE,
    ATTENTION_VALIDATION_POLICY,
    BUILD_VALIDATION_POLICY,
    DESIGN_REVIEW_POLICY,
    DOMAIN_PRIMING,
    FULL_MODEL_SCOPE,
    GSM8K_REFERENCE_CONFIG_POLICY,
    HF_REFERENCE_GOLDEN_POLICY,
    MOE_VALIDATION_POLICY,
    REFERENCE_TEST_POLICY,
    SOURCE_BOUNDARY,
    TRTLLM_TEST_SPECIALIST_INVOCATION,
    VALIDATION_EVIDENCE_LABELS,
)

_PLANNER_GUIDANCE = """\
## PlanDrafter guidance for TensorRT-LLM bring-up

Bring-up plans naturally make many prescriptions — backend choice, KV
layout, quant scheme, kernel path. Those are *starting hypotheses*
recorded in `plan.md`. Bring-up criteria stay **outcome-bound** and
derived from `task.yaml` (model correctness, parity, accuracy thresholds,
runtime contracts that the user explicitly named). A criterion that
names "uses k_bmm_quantizer metadata" or "uses backend X" when
`task.yaml` did not ask for that specific means is over-prescribed;
rephrase it as the underlying outcome (correctness, parity, hard-path
evidence) instead.

**Bring-up project-level required mechanisms (treated as outcomes).**
This project requires a fixed set of mechanism names in every bring-up
acceptance-criteria checklist regardless of how `task.yaml` is phrased,
because they encode the production contracts every bring-up must prove:
`KVCacheManagerV2`; the selected attention backend (TRTLLM or
FlashInfer) at checkpoint-scale dimensions; the CUDA-graph hard-path
matrix (`cuda_graph=false` baseline plus `cuda_graph=true` enabled with
`CudaGraphConfig()` or equivalent); and the `source_activation_replay`
/ `source_logit_replay` / `generation_parity` evidence labels defined
above. These labels are not built-in `agent-flow` test functions, but
they are required names for the outcomes the criteria must prove.
Encoding them in criteria is **not** a leaked plan prescription — it is
the bring-up project's contract. Helper names, file paths, function
signatures, quant modes, or other knobs the user did not ask for still
must not leak into criteria.

`plan.md` must give the Coder enough specificity to execute and the
Reviewer enough specificity to detect drift. For TensorRT-LLM bring-up,
cover:

1. **Source semantics.** Atomic source-side contracts that matter for this
   task: Q/K/V geometry, Q/K/V norms, positional encoding, masks/windows,
   KV-cache layout, router/expert behavior, normalization, weight loading,
   etc. Split convenience bundles by primary concern, owner boundary,
   activation condition, variant coverage, and proof path. Include a
   variant inventory when checkpoint/config differences can affect the
   contract; otherwise state `Unknown` and resolve before planning.
2. **TensorRT-LLM mapping.** Where each piece lands in module / backend /
   runtime / KV layers, with likely files, classes, or functions to inspect
   or modify. These are guidance for the Coder, not binding paths.
3. **Architecture decision.** For any model / backend / runtime / cache /
   ModelConfig / binding contract mismatch, list plausible directions, mark
   each accept/reject, explain why rejected directions are architecturally
   invalid, cite concrete feasibility evidence or contract mismatches, and
   name the invariants/evidence that prove the selected direction keeps a
   coherent end-to-end TensorRT-LLM abstraction. Compare directions by future-
   feature composability, whether they add unnecessary entities, and
   expected code-path/LOC cost. If no contract-level mismatch is present,
   say so explicitly.
4. **Implementation steps.** Contract-level source and test changes in
   order: required capability, owner area, likely files, invariants to
   preserve, proof obligations. For attention work, prefer the TensorRT-LLM or
   FlashInfer attention backends over a vanilla backend whenever a
   production backend can support the semantics. **Do not use VANILLA
   evidence as a substitute for TRTLLM or FlashInfer when either is the
   declared target path.** If existing backends are missing required
   kernel behavior, plan a native torch op or an OpenAI Triton kernel
   that fills the gap under the existing backend per the Python-first
   rule. The architecture decision must name the chosen path explicitly;
   vague phrasing like "add the missing kernel" is layer drift.
   Backend/runtime/ModelConfig/KV-cache plumbing changes (Python side)
   are still in scope. In the attention stage, keep non-attention modules
   as minimal HF-style Python/Torch scaffolding only; do not add fused
   MoE/CUTLASS/C++ kernel work or broad global runtime/import changes
   there.
5. **Risks.** Contracts that could silently fail even when tests pass —
   e.g. shared helpers between reference and implementation, hard paths
   that get bypassed by config, loose tolerances.
6. **Validation matrix (`acceptance-criteria.md`).** Every high-risk
   contract must map to a concrete pass/fail item, or be named as a
   remaining weak contract in `plan.md`'s risk register. Each criterion
   should state the risk it covers, the independent reference, the
   **reference_tier** (`static` / `minimal_golden` / `reduced_source`
   / `real_source`), the
   **validation_tier** (`static` / `unit` / `integration` /
   `real_runtime`), the device/runtime path that must be exercised
   (e.g. CUDA/GPU rather than CPU), hard config, prefill/decode
   coverage, state-dict accounting if weights matter,
   expected failure signal, and the command.
   `reference_tier` describes reference quality; `validation_tier`
   describes execution shape — `reference_tier=real_source` alone does
   **not** prove `real_runtime`. Required `acceptance-criteria.md`
   items for attention and full-model bring-up are listed in the
   validation policies above; encode them as the outcome they prove.

Do not defer required code changes with phrasing like *future work*,
*follow-up*, *out-of-scope*, *wait for runtime support*, or *after
backend/runtime support is fixed*. Required model, backend, runtime,
ModelConfig/cpp conversion, KV-cache, and test changes must appear as
concrete steps in the current plan.

**Backend feasibility on the real target.** Attention backend selection
must be driven by the real target model configuration. Read the
checkpoint/config files named in the user spec **before** choosing the
backend, and include at least one required test using checkpoint-scale
attention dimensions and settings from that config — attention variants,
`head_dim`, projection topology, KV layout, position semantics,
mask/window behavior, and cache/runtime path. Toy/tiny configs are useful
for focused goldens but cannot be the only pass evidence for
backend/runtime feasibility, because kernel limits and dispatch failures
can appear only at real dimensions.

Required pass-critical evidence to encode in `acceptance-criteria.md`.
Phrase each item as the **outcome** it proves; specific kernels,
backends, or schemes named below are *examples* of common starting
hypotheses recorded in `plan.md`, not text to copy verbatim into
criteria unless `task.yaml` itself names them:
- For attention bring-up: include coverage for Q/K/V geometry, Q/K/V
  norms, RoPE/position semantics, mask/window semantics, KV-cache
  layout, context plus decode, and CUDA/GPU evidence for the
  KV-cache-manager and attention backend the plan chose, whenever
  runtime or cache semantics are touched. Pass-critical items must
  include evidence labeled `source_activation_replay`,
  `source_logit_replay`, `generation_parity` (>=32 tokens, >=5
  prompts, deterministic greedy), and at least one `real_runtime`
  evidence item exercising the
  KV-cache-manager plus the selected attention backend at
  checkpoint-scale dimensions. Every pass-critical non-static
  attention validation item must declare `cuda_graph=false` baseline
  plus `cuda_graph=true` hard-path coverage. Source-replay evidence
  should report `max_abs`, `mean_abs`, cosine similarity, and the
  prompt/layer/config used.
- For full-model bring-up: `source_logit_replay` and `generation_parity` are
  pass-critical evidence labels and must each cover both
  `cuda_graph=false` and `cuda_graph=true` runs with hard-path evidence.
- If accuracy gates are configured in the user spec, attention must
  include pass-critical evidence labeled `accuracy_canary`; full-model
  must run the configured benchmarks. Both use the `(cuda_graph=false,
  overlap_scheduler=false)` baseline and `(cuda_graph=true,
  overlap_scheduler=true)` enabled matrix, preceded by a short LLM
  API smoke for both configurations. **If `task.yaml` requires a
  long-input or long-decode benchmark, criteria must also include a
  cheaper long-horizon canary (multi-thousand-token decode parity or
  a small-N benchmark slice) that exercises the same failure mode.**
  A short replay test plus the full benchmark is not sufficient —
  short replays pass while long decode silently diverges, and the
  full benchmark is too expensive to be the only signal.
- If implementation will touch `cpp/` (C++/CUDA/header/CMake), encode
  the required rebuild type and the evidence that validation uses the
  rebuilt package as an explicit acceptance item.

CPU-only tests may be supplemental but cannot be the only pass-critical
evidence for bring-up — require CUDA/GPU execution for unit and focused
parity tests.
"""

_STAGE_GOAL_PLAN_SCHEMA = """\
## Stage/Goal plan schema (mandatory for `plan.md`)

`plan.md`'s `## Implementation Steps` section must use a two-level
**Stage/Goal** hierarchy. A **Stage** is a milestone version of the
model where a defined accuracy bar is met; a **Goal** is a concrete
module-implementation or focused-debugging task whose closure
contributes to the Stage's accuracy bar. The expected progression for
TensorRT-LLM bring-up is:

- Stage 1 — accuracy converges on the configured dataset using
  simple/known-correct backends (e.g. VANILLA attention, Python MoE
  fallback) at the bring-up accuracy target.
- Stage 2 — swap to the performance-targeted backends named in
  `task.yaml` (e.g. TRTLLM or FlashInfer attention, fused MoE) while
  accuracy regresses back to the bar.
- Stage 3 — cuda_graph + overlap_scheduler compatibility with
  hard-path evidence, again with accuracy preserved.

If `task.yaml` is silent on the later Stages (e.g. an attention-only
bring-up), drop them — but Stage 1 (accuracy convergence) is always
required.

### `plan.md` — `## Implementation Steps` format

```markdown
## Implementation Steps

### Stage 1: <accuracy convergence>
Exit criterion: <one-line pointer to `acceptance-criteria.md`'s
                "Stage 1" subsection>

- Goal 1.1: <attention module implementation — scope / output>
- Goal 1.2: <MoE module implementation — scope / output>
- Goal 1.3: <full-model wiring + first source_logit_replay>
- Goal 1.4: <debug accuracy on the configured benchmark until target>

### Stage 2: <swap to performance backends, accuracy regression>
Exit criterion: ...
- Goal 2.1: ...

### Stage 3: <cuda_graph + overlap_scheduler, accuracy regression>
Exit criterion: ...
- Goal 3.1: ...
```

Goal IDs use the form `<Stage>.<Goal>` (e.g. `1.3`). Stages appear in
execution order; Goal order within a Stage is a suggested sequence the
Reviewer may reorder at runtime.

### `acceptance-criteria.md` — Stage-partitioned checklist

Partition `acceptance-criteria.md` into one `## Stage N — <label>`
subsection per Stage in `plan.md`, each a flat `- [ ] ...` checklist.
The Stage labels in `acceptance-criteria.md` must match the Stage
labels in `plan.md`'s `## Implementation Steps`. Example:

```markdown
## Stage 1 — accuracy convergence
- [ ] source_logit_replay max_abs < 1e-3 ...
- [ ] gsm8k score >= 0.40 (baseline, VANILLA attention)

## Stage 2 — performance backend + accuracy regression
- [ ] TRTLLM attention backend passes source_activation_replay
- [ ] gsm8k score >= 0.40 (TRTLLM backend)

## Stage 3 — cuda_graph + overlap_scheduler
- [ ] cuda_graph=true passes generation_parity (hard-path evidence)
- [ ] gsm8k score >= 0.40 (cuda_graph=true, overlap_scheduler=true)
```

QA parses these `## Stage N — ...` headers to scope its verification,
so the exact format matters — do not nest Stage subsections under
other headers, do not change the `## Stage N — ...` prefix.

The Coder, Reviewer, and QA never edit these two files. You (the
PlanDrafter) own both: write them in the draft phase, and revise them
in the replan phase under the lock-matrix rules below.
"""

_STAGE_GOAL_REPLAN_LOCK_MATRIX = """\
## Replan lock matrix (Stage/Goal mode)

The replan phase of this workflow allows you to revise `plan.md` and
`acceptance-criteria.md` based on QA findings — but execution history
constrains what you may change. Before any replan-turn edit, read
`status.md` (use the generic `Read` tool — `status.md` is otherwise
maintained by Coder/Reviewer) and identify each Stage's status from
the `## Stages & Goals` table at the top of the file. Then enforce the
matrix below on yourself:

| Object | Status in `status.md` table | Allowed edits |
|---|---|---|
| Stage `— CLOSED` | Locked | No edit to Stage title, exit criterion, or Goal list in `plan.md`; no edit to the matching `## Stage N — ...` subsection in `acceptance-criteria.md`. |
| Stage `— IN_PROGRESS` | Partially locked | Goals already `[Done]` or `[Failed]` are locked; new Goals may be **appended** to cover gaps left by a `[Failed]` goal; Stage title and exit criterion are locked. Existing acceptance items in the matching subsection are locked; new items may be appended. |
| Stage `— PENDING` | Fully editable | Title, exit criterion, Goal list, and matching acceptance subsection may all change. |
| New Stage (forward-looking) | — | Append at the tail (preserve temporal ordering). Insert in `plan.md`'s `## Implementation Steps` and a matching subsection in `acceptance-criteria.md`. |
| New Stage (QA-REJECT gap-fix) | — | Insert **immediately after the failing CLOSED Stage** (i.e., right after the Stage QA REJECT'd, before the next existing Stage). All subsequent Stages — which must all be `— PENDING` per the workflow ordering — shift down by one and are **renumbered** in both `plan.md` and `acceptance-criteria.md`. CLOSED Stages keep their original numbers. |

Bookkeeping after QA APPROVE on Stage N:

- Update `status.md` (overwrite via the generic `Write` tool,
  preserving every line below the `## Stages & Goals` block as-is):
  flip Stage N's header to `— CLOSED`, mark the next `PENDING` Stage
  `— IN_PROGRESS`, and promote its first `[Undo]` goal to
  `[Doing] (iterations=0)`.
- This is the **only** bookkeeping path that mutates the Stage/Goal
  table during replan; do not let the Coder/Reviewer infer the
  transition from prose.

Bookkeeping after QA REJECT on Stage N:

- **Do not demote, reopen, or otherwise mutate Stage N or any other
  `— CLOSED` Stage.** The lock matrix above is authoritative: CLOSED
  Stages stay CLOSED. The failing acceptance items in QA's summary
  stay as-is (still `- [ ]`) in their original Stage's locked
  subsection — the failure is part of the immutable execution
  record, and the Stage's Goal table already reflects the `[Failed]`
  closure that drove the REJECT.
- Instead, **insert a new gap-fix Stage immediately after the
  failing CLOSED Stage N** in `plan.md`'s `## Implementation Steps`
  and a matching `## Stage (N+1) — <label>` subsection at the same
  position in `acceptance-criteria.md`. All originally-numbered
  Stages after position N — which must all be `— PENDING` at this
  point in the workflow — shift down by one and are **renumbered**
  (Stage K → Stage K+1) in both `plan.md`, `acceptance-criteria.md`,
  and `status.md`'s `## Stages & Goals` table. CLOSED Stages keep
  their original numbers; only PENDING Stages downstream are
  renumbered. The gap-fix Stage:
  - Has a **single Goal** whose objective is to resolve the
    QA-flagged gap. For a score-gate failure, that is "tune
    <metric> above <new-threshold>" where `<new-threshold>` is your
    evidence-grounded judgement of what is empirically achievable
    given the QA report — cite the best landed score, the
    HF-parity / model-capability ceiling, and any
    prompt-variant-exhaustion data the Coder accumulated. For a
    non-score failure (e.g., a missing runtime evidence item), it
    is the narrowest Goal that closes that specific gap.
  - Carries its own `## Stage (N+1) — ...` acceptance subsection
    holding the new, achievable criterion. The original Stage N's
    locked items remain unchanged; the gap-fix Stage's items are
    where the new pass/fail line lives.
  - Is marked `— IN_PROGRESS` in `status.md`, with its single
    Goal promoted to `[Doing] (iterations=0)`. Downstream Stages
    stay `— PENDING` and get their numbers shifted by one
    (K → K+1).
- The renumber is **mandatory and mechanical**; revising the
  content of those downstream PENDING Stages is **optional and
  judgement-driven**. Per the lock matrix above, PENDING Stages
  are fully editable, so during this same replan turn you may
  also rewrite a downstream Stage's title, exit criterion, Goal
  list, or acceptance subsection when the QA evidence or the
  newly-inserted gap-fix Stage makes the original plan stale or
  unreasonable. For each such content edit, justify it in your
  `summary` (cite the QA finding or the gap-fix Stage's new bar
  that made the prior PENDING Stage stale); a content edit that
  silently relaxes acceptance criteria or removes a still-relevant
  Goal without justification is a hard no.
- The next Coder turn picks up Goal `(N+1).1` of the new gap-fix
  Stage. Only after the gap-fix Stage is itself QA-APPROVE'd does
  the workflow proceed to the renumbered next Stage (originally
  Stage N+1, now Stage N+2), in whatever form that Stage now
  takes after your optional content revisions.
- Justify the new threshold (or new criterion) in your `summary`:
  cite the QA evidence, explain what makes the new value
  empirically achievable, and explain why the original gate could
  not be satisfied within any locked Stage's scope. A new
  threshold below an already-demonstrated landed score, or a
  vague non-mechanical criterion, is a hard no.

Every acceptance-criteria edit, every new Goal, and every new Stage
must be justified in your `summary` so the replan-reviewer can audit
the change. Relaxing a locked acceptance item in place — as opposed
to appending a gap-fix Stage with its own item — is a hard no; if
the original criterion no longer reflects `task.yaml`, say so
explicitly in your gap-fix Stage's summary and cite the conflict.
"""

_STAGE_GOAL_REPLAN_DECISION_MAPPING = """\
## Replan decision mapping in Stage/Goal mode

Override the generic decision table in the base PlanDrafter prompt with
the Stage/Goal-aware mapping below when invoked in `replan` mode. Read
`status.md` first to identify the current Stage states, then choose:

- `DONE` — every Stage in `plan.md` is marked `— CLOSED` in
  `status.md`, and either (a) every Stage subsection in
  `acceptance-criteria.md` has been APPROVE'd by a prior QA turn,
  or (b) any QA-REJECT'd Stage has since been followed by a
  gap-fix Stage inserted immediately after it that itself was
  QA-APPROVE'd (the gap-fix Stage's APPROVE resolves the earlier
  Stage's REJECT — it is the contractual replacement for the
  unmet item). The latest QA's `weighted_score` must be at or
  above ``min_score``.
  **Hard rule unchanged**: if the latest `weighted_score` is
  below ``min_score`` you may not return `DONE`; downgrade to
  `POLISHING` directly.
- `POLISHING` — QA APPROVE'd the just-closed Stage and at least one
  Stage remains `PENDING`. Bump `status.md` (Stage N → `— CLOSED`,
  next Stage → `— IN_PROGRESS`, first Goal → `[Doing] (iterations=0)`)
  and apply any low-risk plan touch-ups (sharpen a Goal description,
  add a pitfall note, clarify a future-Stage criterion). The Coder
  runs again immediately; no PlanReviewer.
- `DRAFT_READY` — QA REJECT'd the just-closed Stage, OR you need a
  substantive plan rewrite. After a QA REJECT the **only** allowed
  remediation is inserting a new gap-fix Stage immediately after
  the failing CLOSED Stage (and renumbering downstream PENDING
  Stages) per the QA-REJECT bookkeeping above; never demote a
  `— CLOSED` Stage or edit its locked subsection. Other
  DRAFT_READY-worthy edits (forward-looking only): rewrite a
  `— PENDING` Stage, or append a new Stage at the tail for a
  not-yet-attempted concern. The replan PlanReviewer reviews the
  revised files before the Coder runs again.

Never return `HUMAN_APPROVED` from a top-level replan turn; that value
is reserved for the human-review sub-loop the orchestrator enters
after `DRAFT_READY` when `--plan-human-review` is set.

The whole Stage/Goal protocol — the plan schema, the lock matrix, and
this decision mapping — is only active when the workflow runs with
`--replan-on-qa`. Without the flag none of these blocks reach any
agent's prompt: the workflow is single-shot QA against the full
`acceptance-criteria.md`, and `plan.md` uses the base (flat) plan
format.
"""

_REFERENCE_LADDER_GUIDANCE = """\
## Accuracy debugging — build a verified reference ladder before TensorRT-LLM parity

Plan the bring-up so every later parity comparison has an already-trusted
target, which isolates "my reference is wrong" from "my TensorRT-LLM port
is wrong". Build the reference ladder in this order:

1. **Native baseline.** First run the whole model end-to-end through the
   native HF interface (`AutoModelForCausalLM.from_pretrained`) and pin
   3-5 fixed smoke-test prompts. This is the ground-truth reference;
   capture its outputs for those prompts.
2. **Pure-PyTorch module reference.** Implement each module as a local
   pure-PyTorch implementation and align its output to the native
   `from_pretrained` model on the pinned prompts. Only once the
   pure-PyTorch output matches the native baseline is that module a
   trusted reference — this proves the reference itself is correct.
3. **TensorRT-LLM parity.** Implement each module in TensorRT-LLM and run
   parity against the verified pure-PyTorch implementation. Because the
   pure-PyTorch tier is already aligned to the native baseline, a
   remaining TensorRT-LLM parity gap points at the port, not the
   reference.

Encode this in the Goals: a module's pure-PyTorch reference (aligned to
`from_pretrained`) must be pinned before its TensorRT-LLM parity item,
not after.
"""

_PARITY_VS_DATASET_GUIDANCE = """\
## Accuracy debugging — read parity and dataset accuracy together

Do not gate the *start* of dataset-accuracy testing (e.g. gsm8k) on a
perfectly-tight parity result. Parity max-abs/cosine can look
"insufficient" purely from benign numerical jitter (different kernels,
accumulation order, dtype), so a loose-but-stable parity number is not by
itself a defect. Judge correctness from parity **and** dataset accuracy
together:

- If dataset accuracy lands at a reasonable score for the model, a
  loose-but-stable parity result is acceptable — do not burn iterations
  chasing the last digits of max-abs.
- If dataset accuracy is low, parity is the diagnostic that localizes
  where the divergence enters.

A not-yet-tight parity number is not, by itself, a gate on starting
dataset-accuracy testing. Cost ordering still applies only to the
*expensive* full benchmark — run the cheap LLM-API smoke and accuracy
canary before it, and the model must actually produce coherent output
first. But do not block the dataset-accuracy run on parity: lay out the
Goals so it can begin as soon as the model runs coherently, in parallel
with parity tightening, rather than strictly after every parity item is
green.
"""

_MULTIMODAL_TEXT_FIRST = """\
## Multimodal models — bring up the text path first

For a multimodal model, split the TensorRT-LLM implementation into two
phases. First bring up the text (language-model) path and validate it
through the configured gsm8k accuracy gate; only then debug the
multimodal (vision / audio / other encoder) path. Sequence the plan's
Stages/Goals so the multimodal front-end is attempted only after the
text path has passed its accuracy gate — a text-path accuracy bug is far
cheaper to localize without the multimodal encoder in the loop.
"""

SYSTEM_PROMPT_EXTENSION = "\n".join(
    [
        DOMAIN_PRIMING,
        SOURCE_BOUNDARY,
        DESIGN_REVIEW_POLICY,
        VALIDATION_EVIDENCE_LABELS,
        ATTENTION_SCOPE,
        FULL_MODEL_SCOPE,
        REFERENCE_TEST_POLICY,
        HF_REFERENCE_GOLDEN_POLICY,
        BUILD_VALIDATION_POLICY,
        ATTENTION_VALIDATION_POLICY,
        MOE_VALIDATION_POLICY,
        ACCURACY_GATE_FRAMEWORK,
        GSM8K_REFERENCE_CONFIG_POLICY,
        ACCURACY_GAP_PARITY_POLICY,
        _PLANNER_GUIDANCE,
        _REFERENCE_LADDER_GUIDANCE,
        _PARITY_VS_DATASET_GUIDANCE,
        _MULTIMODAL_TEXT_FIRST,
        TRTLLM_TEST_SPECIALIST_INVOCATION,
    ]
)

# Stage/Goal control flow is only wired when the workflow runs with
# --replan-on-qa; ``build_modeling_bringup_prompts`` appends this block
# on top of ``SYSTEM_PROMPT_EXTENSION`` in that mode only.
STAGE_GOAL_EXTENSION = "\n".join(
    [
        _STAGE_GOAL_PLAN_SCHEMA,
        _STAGE_GOAL_REPLAN_LOCK_MATRIX,
        _STAGE_GOAL_REPLAN_DECISION_MAPPING,
    ]
)
