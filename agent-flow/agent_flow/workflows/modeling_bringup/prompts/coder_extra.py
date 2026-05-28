"""Modeling-bringup-specific guidance appended to the Coder system prompt."""

from ._common import (ACCURACY_GATE_FRAMEWORK, ATTENTION_SCOPE,
                      ATTENTION_VALIDATION_POLICY, BUILD_VALIDATION_POLICY,
                      DESIGN_REVIEW_POLICY, DOMAIN_PRIMING, FULL_MODEL_SCOPE,
                      MOE_VALIDATION_POLICY, REFERENCE_TEST_POLICY,
                      SOURCE_BOUNDARY, STATUS_DONE_TODO_RUBRIC,
                      VALIDATION_EVIDENCE_LABELS)

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
- Translate plan invariants into assertions, negative controls, or focused
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
    STATUS_DONE_TODO_RUBRIC,
    _CODER_GUIDANCE,
])
