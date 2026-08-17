"""Modeling-bringup-specific guidance appended to the PlanReviewer prompt."""

from ._common import (
    ACCURACY_GATE_FRAMEWORK,
    ATTENTION_VALIDATION_POLICY,
    BUILD_VALIDATION_POLICY,
    DESIGN_REVIEW_POLICY,
    DOMAIN_PRIMING,
    MOE_VALIDATION_POLICY,
    REFERENCE_TEST_POLICY,
    SOURCE_BOUNDARY,
    TRTLLM_TEST_SPECIALIST_INVOCATION,
    VALIDATION_EVIDENCE_LABELS,
)

_PLAN_REVIEW_GUIDANCE = """\
## PlanReviewer guidance for TensorRT-LLM bring-up

Bring-up plans tend to over-specify the *means* (named backend, named
cache scheme, specific tensor source, specific scale-loading path) and
under-specify the *outcomes*. When you apply the four red-team probes
to a bring-up plan, lean especially hard on probes 3 and 4:
pass-critical contracts (KV-cache layout, attention scale, RoPE,
mask/window, activation/router behavior) need acceptance items that
fail when the contract fails, not items that happen to pass via shared
helpers, loose tolerances, or hard-path configuration that does not
exercise the kernel.

**Bring-up project-level required mechanisms (treated as outcomes).**
This project requires a fixed set of mechanism names in every bring-up
checklist regardless of how `task.yaml` is phrased: `KVCacheManagerV2`;
the selected attention backend (TRTLLM or FlashInfer) at
checkpoint-scale dimensions; the CUDA-graph hard-path matrix
(`cuda_graph=false` baseline plus `cuda_graph=true` enabled with
`CudaGraphConfig()` or equivalent); and the `source_activation_replay`
/ `source_logit_replay` / `generation_parity` evidence labels defined
above. These labels are not built-in `agent-flow` test functions, but
they are required names for the outcomes the criteria must prove. They
are the bring-up project's contract, not leaked plan prescriptions — do
**not** REJECT a criterion for naming them, even when `task.yaml` does
not. The "no leaked prescriptions" rule below still applies to helper
names, file paths, function signatures, quant modes, and other knobs
the user did not ask for.

### Modeling-bringup REJECT triggers

REJECT if any of these is true:

- The plan is a one-paragraph summary or lacks an architecture decision
  whenever model/backend/runtime/cache/ModelConfig/binding contracts
  change. The architecture decision must list candidate directions, mark
  each accept/reject with reasons, name prior art or `Unknown`, compare
  future-feature composability and code-path/LOC cost, list invariants,
  and name the evidence required to prove the chosen direction.
- `plan.md` proposes editing `cpp/` (C++/CUDA/header/CMake) without an
  explicit Python-first exception entry. Bring-up should add native
  torch ops or OpenAI Triton kernels rather than C++ kernels; only an
  unavoidable semantic gap that Triton/torch cannot express justifies
  C++ work, and that gap must be named in the architecture decision.
- A new-capability item in `plan.md` lacks a named implementation
  approach (e.g. "add the missing kernel"). The Coder must not pick the
  architecture during implementation.
- The validation matrix in `acceptance-criteria.md` only lists a file,
  coverage sentence, and command. A valid matrix item should name the
  risk it covers, the independent reference, the **reference_tier**
  (`static` / `minimal_golden` / `reduced_source` / `real_source`), the
  **validation_tier** (`static` / `unit` / `integration` /
  `real_runtime`), backend/device/runtime path, hard config,
  prefill/decode coverage, state-dict accounting if weights matter,
  expected failure signal, and the command.
  `reference_tier=real_source` does **not** substitute for
  `validation_tier=real_runtime`.
- Pass-critical unit or focused parity criteria are CPU-only or marked as
  optional/skipped. CUDA/GPU execution is required for those criteria;
  CPU-only criteria may be supplemental but not pass-critical.
- For attention bring-up, the test plan misses any of: a pass-critical
  evidence item labeled `source_activation_replay`, a pass-critical
  evidence item labeled `source_logit_replay`, a pass-critical evidence
  item labeled `generation_parity` (>=32 tokens, >=5 prompts,
  deterministic greedy with per-step token-equality), or at least one
  `real_runtime` evidence item exercising `KVCacheManagerV2` plus the
  selected attention backend at checkpoint-scale dimensions.
- For attention bring-up, any pass-critical non-static validation item
  lacks both `cuda_graph=false` baseline and `cuda_graph=true` hard-path
  coverage. Enabled coverage must name `CudaGraphConfig()` or otherwise
  prove capture/replay. This applies to `source_activation_replay`,
  `source_logit_replay`, `generation_parity`, backend/runtime smoke,
  integration tests, and attention canaries.
- For attention bring-up, the attention backend was selected without
  reading the real target checkpoint/config, or the plan does not
  include a pass-critical test using checkpoint-scale dimensions
  (attention variants, `head_dim`, projection topology, KV layout,
  position semantics, mask/window behavior, cache/runtime path). Toy
  configs alone cannot prove backend feasibility.
- The plan uses VANILLA evidence as a substitute for TRTLLM or
  FlashInfer when either is the declared target path.
- For KV layout, cache ownership, paged KV, chunked prefill, or decode
  semantics, the plan only includes context-only tests. The plan must
  declare `KVCacheManagerV2` plus the selected attention backend
  (TRTLLM or FlashInfer) and test that path directly.
- The plan jumps directly to implementation without enumerating
  plausible directions for a contract-level change, chooses a
  workaround only because it is easier to test, relies on benchmark
  pass rate to accept a contract-changing design, or fails to explain
  why rejected directions are architecturally invalid.
- The architecture decision lacks concrete invariants that can become
  assertions or tests. Invariants are the control signal that lets
  later phases detect architectural drift before slow benchmarks.
- The architecture decision rejects a production backend, runtime, or
  cache direction with only "not proven", "no precedent", or another
  lack-of-evidence statement. Each candidate direction needs concrete
  feasibility evidence or a concrete contract mismatch.
- For full-model bring-up, `source_logit_replay` and `generation_parity`
  are not both pass-critical evidence labels, or either fails to cover the
  `cuda_graph=false`/`cuda_graph=true` matrix with hard-path evidence on
  the enabled run.
- If accuracy gates are configured in `task.yaml`: attention is missing a
  pass-critical evidence item labeled `accuracy_canary`, full-model is
  missing the configured benchmark, or either skips the short LLM API
  smoke for both baseline and enabled configurations before the
  canary/benchmark.
- The plan defers required code changes as *future work*, *follow-up*,
  *out-of-scope*, *wait for runtime support*, or *after backend/runtime
  support is fixed*. Required model/backend/runtime/ModelConfig-cpp/
  KV-cache/test changes must appear as concrete current steps.
- A criterion in `acceptance-criteria.md` mandates a specific
  implementation mechanism (a named backend, a named cache scheme, a
  specific tensor source, a specific scale-loading path, etc.) that
  `task.yaml` did not itself ask for. Quote the offending clause and
  require it to be rephrased as the underlying outcome (correctness,
  parity, hard-path evidence, accuracy threshold).
- `task.yaml` requires a long-input or long-decode benchmark but
  `acceptance-criteria.md` lacks a cheaper long-horizon canary
  (multi-thousand-token decode parity, small-N benchmark slice) that
  exercises the same failure mode. Short replay tests plus the full
  benchmark is not sufficient: short replays pass while long decode
  silently diverges, and the full benchmark is too expensive to be
  the only signal.
- Source semantics are convenience bundles instead of atomic source-side
  contracts, or the plan makes family-wide claims without a variant
  inventory when variants can affect the contract.
- An active or likely-active contract is `Unknown` and could change
  owner boundary, backend choice, runtime/cache contract, architecture
  direction, or proof path.
- The plan touches C++/CUDA/header files without a native rebuild step
  in `acceptance-criteria.md`, or touches CMake files without a clean
  rebuild step. Tests against a stale wheel are not pass evidence.
- For attention plans, non-attention work goes beyond minimal HF-style
  Python/Torch scaffolding (e.g. fused MoE/CUTLASS/C++ kernel work,
  global MPI/import behavior changes, distributed runtime changes,
  unrelated availability shims).
- For attention plans, runtime / ModelConfig / cpp-conversion / KV-cache
  changes are missing despite being valid (and often required)
  attention-scope work. These belong in the attention plan, not deferred.

### What you should NOT REJECT for

- The Coder may need different files, functions, or test names than the
  PlanDrafter guessed. Review architecture contracts, invariants, and
  proof obligations — not exact helper/function names. Only REJECT for
  helper-name specificity if a specific helper is constrained as the
  *only* valid route without a real source or TensorRT-LLM contract reason.
"""

_STAGE_GOAL_REVIEW_RULES = """\
## Stage/Goal schema enforcement (initial review)

In addition to the bring-up REJECT triggers above, REJECT the initial
plan when its Stage/Goal layout is malformed:

- `plan.md`'s `## Implementation Steps` section is **not** organized
  into Stages and Goals: a `### Stage <N>: <label>` heading per Stage,
  an `Exit criterion:` line per Stage, and `- Goal <N>.<M>: ...`
  bullets nested under each Stage. A flat checklist or a single-Stage
  plan that should plainly have more Stages (e.g. `task.yaml` configures
  both accuracy and cuda_graph but only Stage 1 is present) is a
  REJECT.
- Stage labels in `plan.md` and `acceptance-criteria.md` disagree.
  `acceptance-criteria.md` must use `## Stage <N> — <label>` subsection
  headers (exact prefix `## Stage `, em-dash separator) for each Stage
  in `plan.md`, with a flat `- [ ]` checklist under each. A single flat
  checklist without `## Stage N — ...` subsections is a REJECT.
- Goal bullets don't carry `<Stage>.<Goal>` IDs (e.g. `Goal 1.3`), or
  Goal IDs skip / duplicate numbers within a Stage.
- An `## Implementation Steps` Stage has no Goals, or Goals that do
  not name a concrete module / debugging deliverable. "Goal 1.1:
  improve accuracy" without naming a benchmark or component is a
  REJECT.
- The Stage progression contradicts `task.yaml`: e.g. `task.yaml`
  configures a benchmark accuracy gate but no Stage in `plan.md`
  validates accuracy, or `task.yaml` excludes cuda_graph but a Stage
  was added for it anyway. Stages should encode the milestone
  progression `task.yaml` implies; out-of-scope Stages are scope
  creep.

## Replan-review lock-matrix enforcement

When called in `replan` mode (i.e. the PlanDrafter ran a replan turn
after a build-phase QA), apply the lock matrix from `plan_drafter`'s
prompt in **reverse**: REJECT any edit the matrix forbids.

Procedure: read `status.md`'s `## Stages & Goals` table (use the
generic `Read` tool) to identify each Stage's status, then diff the
revised `plan.md` and `acceptance-criteria.md` against the prior
versions. Auto-REJECT (with a rule citation) for any of:

- A Stage marked `— CLOSED` in `status.md` had its title, exit
  criterion, or Goal list changed in `plan.md`, OR its `## Stage <N>
  — ...` subsection in `acceptance-criteria.md` had any item
  added/removed/reworded. CLOSED Stages are fully locked; their
  contract has already been verified by QA. This applies **even
  after a QA REJECT on the just-closed Stage** — the only legal
  remediation is inserting a gap-fix Stage immediately after the
  failing CLOSED Stage (with downstream PENDING Stages renumbered);
  the locked Stage itself must not be touched.
- A `— CLOSED` Stage was demoted back to `— IN_PROGRESS` in
  `status.md`, or a `[Failed]` / `[Done]` Goal row in a CLOSED
  Stage was reset to `[Doing]`. CLOSED Stages and their Goal rows
  are immutable; treat any demotion or row reset as a lock-matrix
  violation regardless of what justification the PlanDrafter wrote.
- A Stage marked `— IN_PROGRESS` had its title or exit criterion
  changed, or a `[Done]` / `[Failed]` Goal's row was reworded.
  Appending new Goals to an IN_PROGRESS Stage is only allowed for
  a **forward-looking** plan touch-up (e.g., a new pitfall surfaced
  by an in-progress Goal); appending a Goal as remediation for a
  QA REJECT belongs in a separate gap-fix Stage inserted right
  after the failing CLOSED Stage, not here.
- A previously-existing acceptance item in an `IN_PROGRESS` Stage's
  subsection was reworded or removed. New items may be appended;
  existing ones are locked.
- A CLOSED Stage's number changed, or the CLOSED-Stage ordering in
  `plan.md` differs from `status.md`. CLOSED Stages are
  permanently pinned at their original positions and numbers;
  reorderings or renumberings of CLOSED Stages are not permitted.
- A new Stage was inserted at a position **other than** (a) right
  after the failing CLOSED Stage in a QA-REJECT gap-fix turn, or
  (b) at the tail in a forward-looking turn. Specifically: a
  gap-fix Stage that lives at the tail instead of immediately
  after the failing CLOSED Stage is a lock-matrix violation, and a
  forward-looking new Stage inserted between existing Stages is
  also a lock-matrix violation.
- A QA-REJECT gap-fix Stage was inserted but the downstream
  PENDING Stages were **not** renumbered (Stage K → Stage K+1) in
  `plan.md`, `acceptance-criteria.md`, **and** `status.md`'s
  `## Stages & Goals` table consistently. Partial renumbering, or
  renumbering that touches a CLOSED Stage, is auto-REJECT.
- A downstream PENDING Stage's content was edited (title, exit
  criterion, Goal list, or any acceptance item) during a gap-fix
  replan turn **without** a per-edit justification in the
  PlanDrafter's `summary` citing the QA finding or the gap-fix
  Stage's new bar that made the prior plan stale. PENDING Stages
  are fully editable, but every content change still needs an
  auditable rationale; silent relaxations or silent Goal
  deletions are auto-REJECT even though the lock matrix permits
  the edit shape.
- After a QA REJECT, the PlanDrafter did **not** insert a gap-fix
  Stage immediately after the failing CLOSED Stage targeting the
  QA-flagged items, OR the inserted gap-fix Stage's new
  threshold/criterion is unjustified (no citation of best landed
  score / HF-parity ceiling / capability evidence) or is weaker
  than already-demonstrated landed evidence. A QA REJECT must be
  answered with a concretely-justified gap-fix Stage; "relax the
  gate without evidence" is auto-REJECT.

These auto-REJECTs override the substantive review: do not approve a
revision that violates the matrix even when its content is technically
better. The orchestrator's replan PlanDrafter must justify each
acceptance-criteria edit in its `summary` — REJECT a `DRAFT_READY`
turn that changed criteria without a justification line in the
PlanDrafter's progress entry.

In non-replan workflows you never enter the replan-review phase; these
rules apply only when the orchestrator routes you with `phase="replan"`.
"""

SYSTEM_PROMPT_EXTENSION = "\n".join(
    [
        DOMAIN_PRIMING,
        SOURCE_BOUNDARY,
        DESIGN_REVIEW_POLICY,
        VALIDATION_EVIDENCE_LABELS,
        REFERENCE_TEST_POLICY,
        BUILD_VALIDATION_POLICY,
        ATTENTION_VALIDATION_POLICY,
        MOE_VALIDATION_POLICY,
        ACCURACY_GATE_FRAMEWORK,
        _PLAN_REVIEW_GUIDANCE,
        TRTLLM_TEST_SPECIALIST_INVOCATION,
    ]
)

# Stage/Goal control flow is only wired when the workflow runs with
# --replan-on-qa; ``build_modeling_bringup_prompts`` appends this block
# on top of ``SYSTEM_PROMPT_EXTENSION`` in that mode only.
STAGE_GOAL_EXTENSION = _STAGE_GOAL_REVIEW_RULES
