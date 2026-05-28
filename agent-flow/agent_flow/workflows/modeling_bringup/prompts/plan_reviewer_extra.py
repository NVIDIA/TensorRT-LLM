"""Modeling-bringup-specific guidance appended to the PlanReviewer prompt."""

from ._common import (ACCURACY_GATE_FRAMEWORK, ATTENTION_VALIDATION_POLICY,
                      BUILD_VALIDATION_POLICY, DESIGN_REVIEW_POLICY,
                      DOMAIN_PRIMING, MOE_VALIDATION_POLICY,
                      REFERENCE_TEST_POLICY, SOURCE_BOUNDARY,
                      VALIDATION_EVIDENCE_LABELS)

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
  negative control or mutation check, expected failure signal, and the
  command. `reference_tier=real_source` does **not** substitute for
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

SYSTEM_PROMPT_EXTENSION = "\n".join([
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
])
