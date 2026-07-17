"""Modeling-bringup-specific guidance appended to the Coder system prompt."""

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
    STATUS_DONE_TODO_RUBRIC,
    VALIDATION_EVIDENCE_LABELS,
)

_CODER_GUIDANCE = """\
## Coder guidance for TensorRT-LLM bring-up

- Reason about module / backend / runtime contract / KV-cache semantics
  whenever the change is in those layers. Source and test edits are
  allowed, including runtime, ModelConfig, cpp conversion, backend, and
  KV-cache changes when required by the task.
- Cover what `acceptance-criteria.md` requires, including the matching
  CUDA/GPU unit or focused parity tests. Do not make GPU tests
  skipped, optional, or conditional pass evidence.
- **Python-first kernel rule.** Bring-up is parity-first. New kernels
  default to a native torch op or an OpenAI Triton kernel, not C++ /
  CUDA / header / CMake. Touch `cpp/` only if Triton/torch fundamentally
  cannot express the required semantics, the plan's architecture decision
  documents that constraint, and the plan budgeted for the rebuild cost.
- A missing or unsupported kernel is worker-fixable: add a native torch op
  or an OpenAI Triton kernel under the existing backend choice. Only flag
  an architecture-level conflict in your final summary when no Python-
  kernel path exists under the current plan.
- Translate plan invariants into assertions or focused
  tests where practical. Invariants are the cheap signal that catches
  drift before slow benchmarks.
- **Tier order for bring-up validation:** source replay → runtime smoke →
  focused integration → accuracy canary → long benchmark. Run the
  cheapest tier that covers your changes first, and do not start a long
  benchmark while a focused source replay, runtime smoke, or canary is
  still failing.
- **When a bring-up command repeatedly hits the same blocker** (aborted,
  killed, OOM, timed out, hung, segfault, did not complete), pick a
  different approach: lighter fixture, smaller config, alternate kernel
  path, or a Python-fallback kernel under the current backend. Quote the
  blocker when you describe what you tried.
- For attention work, declare and exercise the selected attention backend
  (TRTLLM or FlashInfer) plus `KVCacheManagerV2`. Pass-critical non-static
  items must cover the CUDA graph matrix (one `cuda_graph=false` baseline
  and one `cuda_graph=true` enabled run that exercises the **CUDA graph
  hard path** via `CudaGraphConfig()` or equivalent). A `cuda_graph=true`
  run that silently falls back to a non-graph path is **not** valid
  hard-path evidence.
- Do not use VANILLA backend evidence as a substitute for TRTLLM or
  FlashInfer when either is the declared target path. The VANILLA
  backend bypasses the production runtime contracts (KV cache,
  attention dispatch, CUDA graph capture), so a passing VANILLA test
  does not prove the target backend works.
- For full-model work that includes MoE, name the selected MoE backend
  (e.g. `CUTLASS`, `VANILLA`, `TRTLLMGen`), the activation
  implementation (e.g. `gelu`, `gelu_tanh`, `swiglu`), and the op path
  (e.g. `torch.ops.trtllm.fused_moe`). Generic "MoE parity passed" is not
  pass evidence.
- If your changes touch C++/CUDA/header/CMake files, rebuild TensorRT-LLM and
  ensure validation uses the rebuilt package before reporting tests as
  passing. A stale wheel is not pass evidence.
- Do not read, cite, or use `auto_deploy/` as a technical source. Do not
  edit `auto_deploy/` or `tests/.../auto_deploy/`.
"""

_STAGE_GOAL_CODER_PROTOCOL = """\
## Stage/Goal protocol — working a single Goal per turn

The bring-up workflow organizes `plan.md`'s `## Implementation Steps`
into Stages and Goals. `status.md` carries a `## Stages & Goals` table
at the top of the file that is the live state machine for those Stages
and Goals. Read it via `read_status` at the start of every turn and
locate the single `[Doing]` Goal — that is the **only Goal you work
on this turn**.

State tokens you will see in the table:

- Stage states: `PENDING` (not yet started), `IN_PROGRESS` (active),
  `CLOSED` (verified by QA), and the transient `CLOSED (pending QA)`
  the Reviewer uses while submitting to QA.
- Goal states: `[Undo]` (not yet started), `[Doing]` (active, with an
  `(iterations=N)` suffix the Reviewer maintains), `[Done]` (closed
  with evidence), `[Failed]` (Reviewer judged unrecoverable). At most
  one `[Doing]` Goal per Stage.

### Per-turn rules

1. **Stay inside the active Goal.** Do not implement, debug, or
   refactor anything outside the `[Doing]` Goal's scope as described
   in `plan.md`. Work that belongs to a future `[Undo]` Goal or a
   different Stage is out of bounds this turn — the Reviewer will
   REJECT for it.
2. **Make tangible progress, then describe it.** Each turn should
   produce new evidence: a new test run, a new diagnosis, a different
   approach attempted. Re-running the same failing command without
   trying anything new is iteration noise.
3. **Do not flip Goal or Stage state in the table.** Only the
   Reviewer decides `[Done]` / `[Failed]` and only the Reviewer (or
   the replan-mode PlanDrafter) flips Stages. Rewrite the
   `## Stages & Goals` block in `status.md` verbatim from the prior
   turn's content — preserve every `(iterations=N)` count exactly,
   preserve every `[Done] ... closed iter <n>` annotation. You may
   only append a short evidence pointer to your own `[Doing]` Goal's
   row if a new artifact (test name, log path) became available this
   turn. Everything else is Reviewer-owned.
4. **`update_status` always rewrites the whole file.** Put the
   `## Stages & Goals` block first (unchanged from the prior turn
   plus any allowed evidence pointer), then the `Current status` /
   `Execution path` / `Done & TODO` sections, refreshed to reflect
   this turn.

### When you are truly stuck — the `BLOCKER:` line

A Goal can be marked `[Failed]` by the Reviewer only when **both**
conditions hold simultaneously:

1. Your most recent `append_coder_progress` `summary` contains a
   line starting `BLOCKER:` that names the dead end you have hit, and
2. The Reviewer independently agrees by re-checking the evidence you
   cite.

There is no minimum-iterations floor: the Reviewer may mark
`[Failed]` on iteration 1 if you write a genuine `BLOCKER:` line
and the Reviewer independently confirms no untried approach exists.
The `(iterations=N)` counter in the table (Reviewer maintains the count)
is visibility for the human reader, not a gate.

Format the line as:

```
BLOCKER: <one-line description of why no approach works>
```

Plus a short rationale paragraph in the same summary listing every
approach you tried, what evidence ruled it out, and which acceptance
criterion that closes the loop on. Without that rationale the
Reviewer cannot confirm and will not mark the Goal `[Failed]`.

**Do not write `BLOCKER:` casually.** It is the explicit
worker-side signal that the current Goal cannot be closed under any
approach you have tried. Writing it when you actually still have
ideas to try is the failure mode this protocol is most prone to:
the Goal would be marked `[Failed]` prematurely and the Stage would
close with a gap. Write it only when you have genuinely exhausted
the search space.

After a Goal is marked `[Done]` or `[Failed]`, the Reviewer promotes
the next `[Undo]` Goal in the same Stage to `[Doing] (iterations=0)`
and REJECTs back to you. Your next turn picks up that Goal.

### Stages, dataset accuracy, and progression

The Stage you are in tells you what kind of bring-up work is
acceptable evidence for the active Goal:

- Inside Stage 1 (accuracy convergence), simple/known-correct
  backends (e.g. VANILLA attention, Python MoE fallback) are
  acceptable. The Stage closes when the configured dataset accuracy
  bar is met under that simpler stack.
- Inside Stage 2 (performance backends), the production backends
  named in `task.yaml` (TRTLLM/FlashInfer attention, fused MoE) must
  pass `source_activation_replay` / `source_logit_replay` /
  `generation_parity` and re-hit the accuracy bar.
- Inside Stage 3 (cuda_graph + overlap_scheduler), the hard-path
  matrix must hold while accuracy stays at the bar.

These match `plan.md`'s Stage labels — re-read them at the start of
each turn so you understand the bar the Stage closes against.
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
        STATUS_DONE_TODO_RUBRIC,
        GSM8K_REFERENCE_CONFIG_POLICY,
        ACCURACY_GAP_PARITY_POLICY,
        _CODER_GUIDANCE,
    ]
)

# Stage/Goal control flow is only wired when the workflow runs with
# --replan-on-qa; ``build_modeling_bringup_prompts`` appends this block
# on top of ``SYSTEM_PROMPT_EXTENSION`` in that mode only.
STAGE_GOAL_EXTENSION = _STAGE_GOAL_CODER_PROTOCOL
