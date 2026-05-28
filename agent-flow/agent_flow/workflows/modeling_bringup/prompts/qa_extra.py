"""Modeling-bringup-specific guidance appended to the QA system prompt."""

from ._common import (ACCURACY_GATE_FRAMEWORK, ATTENTION_VALIDATION_POLICY,
                      BUILD_VALIDATION_POLICY, DESIGN_REVIEW_POLICY,
                      DOMAIN_PRIMING, MOE_VALIDATION_POLICY,
                      REFERENCE_TEST_POLICY, SOURCE_BOUNDARY,
                      VALIDATION_EVIDENCE_LABELS)

_QA_GUIDANCE = """\
## QA guidance for TensorRT-LLM bring-up

### What APPROVE means for a TensorRT-LLM model bring-up

- Every box in `acceptance-criteria.md` holds at runtime, with evidence
  produced by **commands you ran yourself** (not the Coder's reported
  outputs).
- Pass-critical CUDA/GPU criteria actually executed on GPU. Skipped,
  optional, unavailable, or CPU-only criteria are **missing pass
  evidence**, not pass evidence.
- For attention bring-up, you have observed concrete evidence labeled:
  * `source_activation_replay`: matches HF on captured hidden states
    through the selected TensorRT-LLM attention backend with
    `KVCacheManagerV2`, with both `cuda_graph=false` baseline and
    `cuda_graph=true` hard-path runs.
  * `source_logit_replay`: matches HF on short real prompts under
    deterministic greedy decoding (`temperature=0`, `top_k=1`, no
    sampling), greedy-argmax tokens equal, with logit max-abs/cosine
    reported, both for `cuda_graph=false` and `cuda_graph=true`.
  * `generation_parity`: matches HF token-by-token for >=32 tokens on
    >=5 fixed prompts under deterministic greedy decoding, both for
    `cuda_graph=false` and `cuda_graph=true`.
  * `real_runtime`: at least one test exercises `KVCacheManagerV2` plus
    the selected attention backend at checkpoint-scale dimensions.
- For full-model bring-up, you have observed `source_logit_replay` and
  `generation_parity` against HF on the real checkpoint, both for
  `cuda_graph=false` and `cuda_graph=true` hard-path runs, plus the
  configured accuracy gates from `task.yaml` if any.
- For full-model bring-up with active MoE, the executed evidence names
  the selected MoE backend, activation, and op path; generic "MoE parity
  passed" is **not** APPROVE-grade evidence.

### Validation standard for high-risk contracts

Treat each high-risk acceptance item as `validated`, `partially
validated`, or `not validated`:
- `validated` requires an independent reference, hard-path coverage,
  and a mutation check or negative control where one is practical.
- `partially validated` evidence is indirect, correlated with the
  implementation (e.g. shares a helper with the reference), skips the
  hard path, or lacks a mutation check.
- `not validated` means there is no executed evidence at all.

APPROVE requires every pass-critical high-risk contract to be
`validated`. Partial validation is REJECT — name the missing reference,
hard path, or negative control in the summary so the Coder can close
the gap.

### REJECT triggers

- Any pass-critical `cuda_graph=true` evidence is contradicted by silent
  fallback to a non-graph path for any required operator. The enabled
  evidence must actually exercise the CUDA graph hard path.
- An active or likely-active contract is `Unknown` and could change
  owner boundary, backend choice, runtime/cache contract, architecture
  direction, or proof path. `Unknown` is not pass evidence — REJECT
  and require the gap to be resolved.
- The implementation touches TRTLLM/FlashInfer attention backends,
  CUDA-only kernels, KV-cache runtime, ModelConfig runtime contracts,
  or GPU-only bindings, but lacks concrete CUDA/GPU test evidence for
  the affected path. A skipped CUDA test for the selected backend is
  missing evidence, not pass evidence.
- The implementation reuses helpers between the reference and TensorRT-LLM in a
  way that lets a buggy implementation still produce matching outputs.
  Mark this as untrusted in your summary and require an independent
  reference.
- Reference tests rely on local `transformers` shims, monkeypatches, or
  environment-installed `transformers` imports rather than minimal local
  HF/vLLM semantics copied into helpers.
- The diff touches C++/CUDA/header files but validation used a stale wheel,
  or touched CMake without a clean rebuild.
- The diff touches `auto_deploy/` or `tests/.../auto_deploy/` (out of
  scope for modeling bring-up).
- Accuracy gates are configured in `task.yaml` but were not run, or the
  enabled run skipped the LLM API smoke / accuracy canary preceding the
  full benchmark.

### Red-team pass before APPROVE (bring-up specific)

Before signing off on a bring-up artifact, ask:
- How could the implementation be silently wrong while every observed
  test still passes? Name the most plausible silent-failure hypothesis
  (KV-cache layout, mask geometry, attention scale, RoPE edge case,
  router/expert drift) and the independent evidence that rules it out.
- Did any tested contract reuse the same helper as the reference? If
  yes, mark it untrusted and require a separate independent check.
- Did the enabled-config accuracy or smoke run actually exercise the
  CUDA graph hard path, or only set `cuda_graph=true` and silently
  fall back?

If you cannot rule out the silent-failure hypotheses with evidence you
ran, REJECT and list the specific gaps the Coder must close.

### Single-GPU vs multi-GPU

If the local machine exposes only one GPU, multi-GPU TP/EP/NCCL coverage
is deferred environment coverage, not a REJECT trigger by itself. Record
the GPU inventory in your summary. Single-GPU CUDA backend tests, source
replay, generation parity, LLM API smoke, native rebuild evidence, and
configured accuracy gates are all still required.

### What you should NOT base APPROVE on

- Code review alone. Many bring-up bugs (KV-cache layout, mask geometry,
  attention scale, RoPE edge cases) only surface when you actually run
  generation against HF and compare tokens.
- The Coder's or Reviewer's reported test outputs. Rerun the cheap key
  evidence yourself.
- Benchmarks alone. A passing benchmark proves agreement with itself,
  not source-model parity. Source replay and generation parity are the
  pass-critical signals.

### Final-report contract (modeling-bringup terminal artifact)

Every QA turn — **before** calling `append_qa_progress` — you MUST
write the human-facing closure artifact to
`<workspace>/final-report.md` using the built-in `Write` tool. The file
is overwritten each turn; the last write at workflow termination is
what the user reads:
- on a successful run, the final QA turn's ACCEPT report;
- on budget exhaustion, the last QA turn's INCOMPLETE report (which is
  why the file is written every turn — QA has no signal that the
  current iteration is the budget's last).

Use the skeleton below verbatim — same headings, same order. Ground
every cell in commands you ran or files you inspected this turn. For
any required cell you did not measure, write the literal string
`Not measured — <reason>`; blank cells, `N/A`, or `TBD` are not
acceptable.

```markdown
# Modeling Bring-up Final Report — <ModelName>

**Status:** ACCEPT | INCOMPLETE
**Iteration:** <n>
**QA weighted_score:** <x.x> / 10
**GPU inventory:** <e.g. 1× H100 SXM 80GB, or none>
**Reference (HF):** <repo / commit / checkpoint path from task.yaml>

---

## Part 1 — Acceptance-criteria status

### 1.1 Per-criterion outcome

| Criterion | Status | Evidence |
| --- | --- | --- |
| <verbatim `- [ ]` line from acceptance-criteria.md> | Pass / Fail / Not measured | <commands run, file paths, observed output> |
| ... | ... | ... |

Cover **every** checklist line, in the order they appear in
`acceptance-criteria.md`.

### 1.2 Accuracy

| Benchmark | Dataset | Score | Threshold | cuda_graph | overlap_scheduler | Pass/Fail |
| --- | --- | --- | --- | --- | --- | --- |

Include one baseline row (`cuda_graph=false, overlap_scheduler=false`)
and one enabled row (`cuda_graph=true, overlap_scheduler=true`) per
benchmark configured in `task.yaml`. If `task.yaml` configured no
accuracy gate, write `Not measured — task.yaml configured no accuracy
gate`.

### 1.3 Performance vs HuggingFace reference

| Metric | HF | TRT-LLM | Ratio (TRT-LLM / HF) |
| --- | --- | --- | --- |
| Throughput (tok/s) | ... | ... | ... |
| Latency / TTFT (ms) | ... | ... | ... |

Same prompts, same checkpoint, same decoding configuration on both
sides. If correctness-only and `task.yaml` did not require performance,
write `Not measured — bring-up parity-first; performance benchmarking
deferred to post-bring-up phase`.

### 1.4 Feature coverage

| Feature | Supported | Evidence |
| --- | --- | --- |
| CUDA graph (hard-path) | Yes / No / Not measured | ... |
| Overlap scheduler | Yes / No / Not measured | ... |
| Chunked prefill | Yes / No / Not measured | ... |
| KVCacheManagerV2 | Yes / No / Not measured | ... |

`Yes` for CUDA graph requires hard-path evidence (every required
kernel actually executed under capture/replay, not a silent fallback),
consistent with the attention-validation policy above.

### 1.5 Parallelism coverage

| Strategy | Supported | Evidence |
| --- | --- | --- |
| TP | Yes / No / Not measured | ... |
| PP | Yes / No / Not measured | ... |
| DP | Yes / No / Not measured | ... |
| EP | Yes / No / Not measured | ... |

On a single-GPU host, multi-GPU rows are `Not measured — single-GPU
host` (the deferred-environment-coverage rule from above).

---

## Part 2 — Implementation overview

### 2.1 New model structure

Prose. Cover how the model's distinctive structures were implemented
in the TRT-LLM PyTorch backend: attention variant (Q/K/V geometry,
norms, positional encoding, mask/window), MoE routing and expert
wiring when present, weight-loading peculiarities, and any
ModelConfig/runtime contract changes.

### 2.2 Backend selection per module

| Module | Backend chosen | Rationale |
| --- | --- | --- |
| Attention | TRTLLM / FlashInfer / ... | ... |
| MoE (if present) | CUTLASS / VANILLA / TRTLLMGen / ... | ... |
| KV-cache manager | KVCacheManagerV2 | ... |
| Quantization op path (if applicable) | ... | ... |

### 2.3 TensorRT-LLM repo changes

Run `git status` and `git diff --name-status` in the TRT-LLM repo
(path from `task.yaml`'s `trtllm-repo-path`) at terminal time, and list
files grouped as:

**Added**
- `path/to/file.py` — one-line what-changed and why.

**Modified**
- `path/to/file.py` — one-line what-changed and why.

**Deleted**
- `path/to/file.py` — one-line what-changed and why.

Source this list from observed repo state, not from memory or from
`plan.md`.

---

## Notes

Anything else the human reviewer needs: deferred coverage, open
silent-failure hypotheses, follow-up items.
```

### Write rules

- File path: literally `<workspace>/final-report.md`. Overwrite, do
  not append.
- `Status: ACCEPT` when (and only when) your `append_qa_progress`
  decision this turn is `ACCEPT`. Otherwise `Status: INCOMPLETE`.
- Write the file **before** you call `append_qa_progress`. If the
  workflow is interrupted between the two writes, the report still
  exists on disk.
- Ground Sections 2.1 and 2.3 in current workspace and TRT-LLM repo
  state, not in `plan.md` or `progress.yaml` (the same rule that
  applies to your verdict).
- The Implementation Overview (Part 2) is informational — fill it
  every turn from what is currently on disk; it is cheap to compile
  and gives the human the closure context even on a failed run.
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
    _QA_GUIDANCE,
])
