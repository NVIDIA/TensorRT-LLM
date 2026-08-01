---
name: ad-accuracy-debug
description: >
  Debug AutoDeploy accuracy regressions vs a reference score (PyTorch backend or published
  baseline). Use when an AutoDeploy model's eval score is significantly below the reference
  and the root cause is unknown.
license: Apache-2.0
tags:
  - tensorrt-llm
  - autodeploy
  - accuracy
  - debugging
  - evaluation
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy Accuracy Debugging

## Where this skill applies

This file is part of **trtllm-agent-toolkit**. Paths such as `tensorrt_llm/`, `tests/`, and
`examples/auto_deploy/` are relative to a **TensorRT-LLM source checkout** on the user's machine,
not the plugin repository.

## Related skills in this plugin

- `trtllm-agent-toolkit:ad-graph-dump` — inspect per-transform FX graph snapshots when Phase 2
  suggests a transform was applied incorrectly or is corrupting activations.
- `trtllm-agent-toolkit:ad-conf-check` — verify that precision or config settings (FP8, sharding,
  chunked prefill, etc.) were actually applied at runtime before attributing an accuracy gap to a
  kernel or weight bug.

**Input:** model name, failing accuracy score, reference score, eval task (e.g. MMLU, GSM8K).
**Output:** identified root cause, minimal reproducer, and a code fix.

## Situation Assessment

Before debugging, confirm:

1. **What is the reference score?** Is it from the PyTorch backend test, a published leaderboard, or set manually?
2. **How large is the gap?** A 1-2% gap may be within statistical noise; a 5%+ gap is a real bug.
3. **Is the eval framework itself suspect?** Run the same eval on the PyTorch backend to validate the harness before blaming AutoDeploy.

## Abbreviations

- **AD** — AutoDeploy, TRT-LLM with `_autodeploy` backend
- **PT** — PyTorch, TRT-LLM with `pytorch` backend (manual deployment)

## Phase 0 — Validate the Test Harness

Run the equivalent PyTorch backend test on the same model and same eval task. If PT also fails or scores lower than expected, the issue is in the eval framework (prompt format, chat template, sampling params), not AD-specific.

Key things to verify in the eval harness:
- **Prompt format / `apply_chat_template`**: does the evaluator send raw prompts or apply a chat template?
  The relationship is two-sided for reasoning/chat models:
  - Applying `apply_chat_template` to a concatenated few-shot prompt (without `fewshot_as_multiturn`)
    collapses the examples into a malformed single turn and can produce 0% accuracy.
  - Omitting `apply_chat_template` for a chat-first model can be equally wrong.
  For chat models on few-shot benchmarks, consider whether `apply_chat_template=True` paired with
  `fewshot_as_multiturn=True` is appropriate — the latter turns each few-shot example into an
  explicit user/assistant exchange before the template is applied.
  (Reference: Qwen3.5-MoE accuracy fix in `test_llm_api_autodeploy.py`.)
- **`max_output_len` for generation tasks**: for benchmarks where the model must generate a full
  reasoning chain before the answer (e.g. GSM8K with a reasoning model), the default `MAX_OUTPUT_LEN`
  may truncate the response before the final answer is reached. Consider patching it up (e.g. 512)
  if outputs appear cut off. This is distinct from capping `max_tokens` for classification tasks
  like MMLU where you want to *prevent* long generations.
- **`max_tokens` for classification tasks**: must be capped (e.g. 2 for MMLU) to prevent the model
  generating a full reasoning chain.
- **Dataset path**: confirm `LLM_MODELS_ROOT` is set correctly and the dataset directory exists.

If PT passes: the harness is fine. Proceed to Phase 1.

## Phase 1 — Quick Diagnostic with a Small Sample

Write a standalone diagnostic script that:
1. Loads the AD model directly via `from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM`
2. Reproduces the **exact** prompt format the evaluator uses (not a simplified variant), including few-shot examples if any
3. Runs ~50-100 samples
4. Prints per-sample `(ref, output, correct)` and overall accuracy

**Critical:** reproduce the evaluator's exact prompt format. Deviating — for example, using a 0-shot prompt when the evaluator uses 5-shot — can cause thinking models to produce "Okay" or other meta-responses instead of the expected answer, making results uninterpretable. Verify the first printed prompt matches what the evaluator sends.

Typical evaluator sources:
- `tensorrt_llm/evaluate/mmlu.py` — 5-shot format with dev examples
- `tensorrt_llm/evaluate/gsm8k.py` — few-shot with CoT references
- `tests/integration/defs/accuracy/accuracy_core.py` — `MAX_INPUT_LEN`, `MAX_OUTPUT_LEN`, `NUM_SAMPLES` per task

## Phase 2 — Classify the Error Pattern

From the diagnostic output, determine what the model is generating:

| Output Pattern | Likely Root Cause |
|---|---|
| Coherent but consistently wrong letter / answer | Numerical accuracy bug (attention, FP8 kernel, weight corruption) |
| Generates meta-text ("The user wants...", "The answer is...", "Let me think...") | Prompt format issue — model not primed to answer directly |
| Outputs empty string or EOS immediately | KV cache garbage (uninitialized cache, scale overflow), or `end_id` matching first token |
| Completely random tokens / gibberish | Transformation applied incorrectly, load hook missing or applied twice, corrupted weights |
| Correct on easy subjects, wrong on hard subjects | Subtle numerical precision bug (FP8 kernel mismatch, attention scale wrong) |
| NaN in logits, especially on prefill | FX graph transform produced a node without shape metadata — enable `AD_DUMP_GRAPHS_DIR` and look for nodes missing `meta["val"]`; often caused by an opaque Python closure inside a transform |
| Passes at `world_size=1`, fails at `world_size>1` | Sharding bug — see Phase 4c |

## Phase 3 — Configuration Isolation

Narrow down which part of the setup is responsible by reducing the environment to its simplest
form, then re-enabling components one at a time until the regression reappears.

**Step 1 — Strip to a minimal configuration:**

Where feasible, reduce complexity along each axis, re-running the Phase 1 diagnostic after each
change:

- Remove sharding or reduce to TP=1 / single GPU
- Disable multi-streaming
- Disable non-default transform passes in the YAML config (`enabled: false`)
- **Revert to `torch-simple` `compile_backend`**: AutoDeploy currently supports two backends —
  `torch-cudagraph` (CUDA graphs, the typical production setting) and `torch-simple` (no CUDA
  graphs, significantly slower). If the model is configured with `torch-cudagraph`, revert to
  `torch-simple` and check whether the accuracy issue persists. Note the slower throughput will
  make the validation loop take longer. If accuracy recovers at `torch-simple`, CUDA graph capture
  or replay is the suspect.

If the issue disappears when a component is removed, that component is the suspect — note it and
proceed to Step 2 targeting it. If the issue persists even at minimal config, the bug is in a core
path (weight loading, attention, KV cache) — proceed to Phase 4.

**Step 2 — Re-enable one component at a time:**

Starting from the stripped-down configuration that still reproduces the issue, re-enable the
suspected components individually — one per diagnostic run. Stop as soon as accuracy drops: the
last re-enabled component is the offending pass or backend. Carry this finding into Phase 4 to
investigate the root cause.

## Phase 4 — Root Cause Investigation

> This phase contains targeted investigation paths for known root-cause categories. Add
> model-specific or error-pattern-specific steps here as they are discovered.

### 4a — Quantized Model Accuracy

If the failing model is quantized (e.g. FP8, NVFP4), first verify whether the issue
is in the quantization itself or in the quantized kernel path:

**Step 1 — Test an unquantized baseline.**

Ask the user for an unquantized (BF16/FP16) version of the same model. Run the Phase 1 diagnostic
against it with an identical configuration (same `compile_backend`, same TP, same eval format).

- If the unquantized model **also fails**: the bug is not quantization-related — the issue is in a
  transform pass, attention implementation, or weight loading. Continue with Phase 3 isolation
  against the unquantized model.
- If the unquantized model **passes**: the accuracy gap is introduced by quantization or the
  quantized kernel path. Proceed to Step 2.

**Step 2 — Suspect classification.**

When quantization is confirmed as the source, the likely causes are (in rough order of severity):

| Suspect | Symptom | How to isolate |
|---|---|---|
| Missing scale during dequantization | Near-zero or astronomically large logits; catastrophic accuracy loss (≈ random chance or worse) | Log a few raw logits; they will be wildly out of range |
| Inverted scale (multiplied instead of divided, or vice versa) | Similarly catastrophic; outputs plausible tokens but systematically wrong | Same logit inspection; compare scale values in the checkpoint vs what the kernel receives |
| Incorrect block-scale computation | Major but not catastrophic degradation; typically 5–20% below unquantized reference | Compare per-block scales against a reference quantizer on a few weight tensors |
| `.to(dtype)` used instead of `.view(dtype)` for packed format reinterpretation | Wrong scale or weight values without an error — `.to()` converts values numerically while `.view()` reinterprets the raw bits | Grep the quantization transform for `.to(` on quantized weight/scale tensors of packed types (FP4, FP8); the intent is bit-level reinterpretation, which requires `.view()` |
| Quantized kernel bug (wrong accumulation, wrong cast) | Non-catastrophic; may be input-dependent or shape-dependent | Step 3 below |

**Step 3 — Isolate quantized kernels via fake quantization.**

AutoDeploy's transform pipeline has a built-in fake-quantization path that implements exactly
Q→DQ→high-precision-matmul. Understanding the two stages helps:

- **Stage 1 (`pattern_matcher`)**: Replaces `nn.Linear` nodes with
  `torch.ops.auto_deploy.torch_fake_quant_fp8_linear` /
  `torch_fake_quant_nvfp4_linear` etc. These ops quantize the input, immediately dequantize
  both input and weight back to BF16/FP16, then run a standard `torch.matmul`. Scales are
  exercised but all arithmetic is in high precision. Implementation:
  `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py`, lines 178–286.

- **Stage 2 (`post_load_fusion`)**: `fuse_fp8_linear`, `fuse_nvfp4_linear`, and
  `fuse_finegrained_fp8_linear` transforms **replace** the fake-quant ops with optimized
  low-precision kernels. This is where a kernel bug would be introduced.

To run inference in fake-quantization mode (bypassing the low-precision kernels), add the
following to the YAML config file the test is already using (passed via `--config` or
`--extra_llm_api_options`):

```yaml
transforms:
  fuse_fp8_linear:
    enabled: false
  fuse_nvfp4_linear:
    enabled: false
  fuse_finegrained_fp8_linear:
    enabled: false
  # For MoE models also add:
  fuse_fp8_moe:
    enabled: false
  fuse_finegrained_fp8_moe:
    enabled: false
  fuse_nvfp4_moe:
    enabled: false
```

If there is no existing config file, create one with only the above content and pass it via
`--extra_llm_api_options /path/to/fake_quant_debug.yaml`. The `transforms` key maps directly to
`LlmArgs.transforms`; any transform not listed inherits its default from
`tensorrt_llm/_torch/auto_deploy/config/default.yaml`.

If accuracy recovers with fake quantization, the quantized kernel (not the scales) is the bug.
If accuracy is still wrong, the scales or weight data are the likely culprit.

### 4b — Kernel Wrapper Hardcoded Assumptions

**Symptom:** coherent but systematically wrong output on one specific model family (or one model
variant within a family), while a structurally similar model is fine.

**Root cause pattern:** The C++ kernel or its Python wrapper has a constant where it should read
from the model config or from the actual tensor. Two common forms:

- **Config value hardcoded**: a kernel wrapper assumes a default value instead of reading it
  from the HF config loaded at runtime.
- **Tensor stride/shape hardcoded**: a C++ kernel assumes a specific memory layout that differs
  from what AutoDeploy actually passes. The kernel then reads the wrong memory locations silently.

**How to investigate:**

1. Identify which kernel is dispatched for the failing op (search for the op's Python entry point
   in `tensorrt_llm/_torch/auto_deploy/`).
2. Compare every constant or default in the kernel wrapper call against the corresponding field
   in the model's HF config (`config.json`). Flag any value that is not read from the config.
3. For stride bugs: print the actual strides of the tensors being passed to the kernel and
   compare against what the kernel expects (check the C++ kernel source for stride assumptions
   or parameters).
4. If a mismatch is found, the fix is to pass the config or tensor property as an explicit
   parameter rather than using a hardcoded constant.

### 4c — Sharding-Related Accuracy (world_size > 1)

**First step:** reproduce the issue at `world_size=1`. If accuracy recovers, the bug is in the
sharding path. If it fails at `world_size=1` too, sharding is not the cause — return to Phase 3.

To run at `world_size=1`, set `world_size: 1` in the model's YAML config or pass
`--extra_llm_api_options` with `world_size: 1`.

**Known sharding bug patterns (check in order):**

| Suspect | Symptom | How to isolate |
|---|---|---|
| Wrong allreduce strategy | Non-deterministic or rank-dependent outputs; may appear only at TP≥4 | Set `allreduce_strategy: NCCL` in the sharding transform config; the `AUTO` default has caused correctness issues in the past |
| Double `all_reduce` in MoE | MoE output doubled in magnitude; accuracy catastrophic | Inspect the exported graph; there should be exactly one `all_reduce` after the sum of routed and shared expert outputs, not one per branch |
| Head reshape with wrong stride after TP | Attention output garbage at TP>1, correct at TP=1 | Reshapes that use concrete head counts from `torch.export` become wrong after TP splits the head dimension; these must use `torch.ops.auto_deploy.view` with `tp_scaled_dim` |
| Sharding a projection that must not be sharded | Dim-1 gating projections or latent projections sharded → wrong results | Check `tp_mode` on small-output projections (e.g. MoE router, MLA latent q_a/kv_a); they must be `"none"` |
| Nested parameter deletion breaking weight loading | Some weights missing after sharding, silently defaulting to zero or random | If sharding deletes parent module params and child params are looked up by the old path, the load hook may silently skip them |

**Validating a sharding fix:**

If model size permits, run it at `world_size=1` (baseline), then `world_size=2`, then the target `world_size`.
If accuracy is correct at TP=1 and TP=2 but wrong at TP=8, the bug is likely a head-count divisibility
assumption (head dim must be divisible by the TP degree). If it is wrong at all TP>1, it is a
structural sharding bug (missing allreduce, wrong split point, wrong stride).

## Phase 5 — Per-Subject / Per-Category Breakdown

When the overall score is lower than expected but not catastrophically wrong, look at per-subject or per-category breakdowns in the eval logs. Patterns to look for:

| Pattern | Implication |
|---|---|
| All subjects uniformly ~N% below reference | Uniform precision loss — suspect FP8 kernel or attention scale |
| Specific subjects near 25% (random chance for 4-choice MCQ) | Those subjects have a systematic error — suspect subject length or chunked prefill |
| Easy subjects correct, hard subjects wrong | Near-decision-boundary sensitivity — suspect subtle numerical error |
| Subject-correlated errors | Prompt-length correlation — verify truncation behavior |

For MCQ tasks like MMLU, random chance is 25%. Subjects scoring 25-35% may be genuinely hard for the model even in the PT backend — verify against PT per-subject scores before concluding an AD-specific bug.

## Phase 6 — Iterative Ablation

Once a hypothesis is formed, verify it by toggling one change at a time and re-running the diagnostic (50-100 samples is sufficient for 5%+ gaps).

Each ablation should be a separate diagnostic run. Do not batch multiple hypotheses in one run — it makes results ambiguous.

## Anti-Patterns

- **Do not use 0-shot prompts to diagnose a 5-shot evaluator.** Meta-responses ("Okay", "The answer is...") from a 0-shot run are a prompt-format artifact, not an AD inference bug.
- **Do not invoke `torchrun` for AD tests.** The AD LLM API spawns MPI workers internally; `torchrun` adds a second layer of distributed init that deadlocks.
- **Do not override `LLM_MODELS_ROOT`.** If it is already set in the environment (CI sets it to `/path/to/llm-models`), unsetting or overriding it breaks dataset lookups. Check `echo $LLM_MODELS_ROOT` before assuming it needs to be set.
- **Do not lower the reference threshold as a "fix."** The reference value must be validated against the PT backend before being accepted. If PT also fails, re-examine the harness, not the threshold.
- **Do not apply the same load hook twice.** If a hook converts interleaved → NeoX, applying it again corrupts the weights (it is not idempotent). Check the git log for reverts/restores before adding a hook that might already exist elsewhere in the call chain.

## Keeping This Skill Up to Date

Whenever this skill is used and the debugging session uncovers a new root cause or error pattern
that is not yet described here, update the skill before closing the session.

**Where to add new findings:**

- **Phase 2 — Classify the Error Pattern**: add a new row to the table if the session revealed a
  symptom → root cause mapping that is not already listed. Keep the "Output Pattern" column
  observable (something visible in diagnostic output), and the "Likely Root Cause" column
  actionable (points to a concrete next step or Phase 4 subsection).

- **Phase 4 — Root Cause Investigation**: add the investigation steps under the most fitting
  existing subsection (4a quantization, 4b kernel wrapper assumptions, 4c sharding). If the
  finding does not fit any existing subsection, create a new one numbered sequentially (4d, 4e,
  …). Each subsection should follow the same structure: symptom, root cause pattern, and a
  numbered investigation procedure.

**What is worth capturing:**

- A root cause that is not already represented in Phase 4.
- A symptom pattern that allows earlier classification (Phase 2).
- A configuration or environment condition that reproduces or masks the bug (Phase 3).
- A new anti-pattern discovered during the session.

**What is not worth capturing:**

- Model-specific quirks with no generalization potential.
- Findings that duplicate what is already written.
- Workarounds that paper over a bug rather than identifying it.
