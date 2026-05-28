"""Modeling-bringup-specific guidance appended to the PlanDrafter prompt."""

from ._common import (ACCURACY_GATE_FRAMEWORK, ATTENTION_SCOPE,
                      ATTENTION_VALIDATION_POLICY, BUILD_VALIDATION_POLICY,
                      DESIGN_REVIEW_POLICY, DOMAIN_PRIMING, FULL_MODEL_SCOPE,
                      MOE_VALIDATION_POLICY, REFERENCE_TEST_POLICY,
                      SOURCE_BOUNDARY, VALIDATION_EVIDENCE_LABELS)

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
   that get bypassed by config, loose tolerances, missing negative
   controls.
6. **Validation matrix (`acceptance-criteria.md`).** Every high-risk
   contract must map to a concrete pass/fail item, or be named as a
   remaining weak contract in `plan.md`'s risk register. Each criterion
   should state the risk it covers, the independent reference, the
   **reference_tier** (`static` / `minimal_golden` / `reduced_source`
   / `real_source`), the
   **validation_tier** (`static` / `unit` / `integration` /
   `real_runtime`), the device/runtime path that must be exercised
   (e.g. CUDA/GPU rather than CPU), hard config, prefill/decode
   coverage, state-dict accounting if weights matter, negative control
   or mutation check, expected failure signal, and the command.
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
  prompt/layer/config used. Plan negative controls for wrong RoPE /
  position handling, wrong V or K=V materialization, wrong score
  scale, wrong mask/window behavior, and fake KV geometry when those
  contracts exist.
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

SYSTEM_PROMPT_EXTENSION = "\n".join([
    DOMAIN_PRIMING,
    SOURCE_BOUNDARY,
    DESIGN_REVIEW_POLICY,
    VALIDATION_EVIDENCE_LABELS,
    ATTENTION_SCOPE,
    FULL_MODEL_SCOPE,
    REFERENCE_TEST_POLICY,
    BUILD_VALIDATION_POLICY,
    ATTENTION_VALIDATION_POLICY,
    MOE_VALIDATION_POLICY,
    ACCURACY_GATE_FRAMEWORK,
    _PLANNER_GUIDANCE,
])
