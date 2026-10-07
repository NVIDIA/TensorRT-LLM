---
name: visual-gen-component-test
description: >
  Design, implement, calibrate, and run deterministic TensorRT-LLM VisualGen
  L1 operation tests and L2 component-by-feature parity tests. Use for DiT,
  VAE, text-encoder, scheduler, parallelism, quantization, caching, compile,
  or model-specific component correctness; not for perceptual model evaluation
  or full-pipeline LPIPS quality gates.
---

# VisualGen component tests

Use this skill to turn a VisualGen behavior or feature into a cheap,
deterministic test that identifies the component which diverged. Follow the
repository's `CODING_GUIDELINES.md` and
`tensorrt_llm/_torch/visual_gen/ENGINEERING_CRITERIA.md` before editing tests.

## Classify the test before writing it

- **L1:** one operation or a unit smaller than a component, such as RoPE,
  normalization, a fused kernel, mask construction, or latent layout logic.
- **L2:** one component under one feature or feature combination. Components
  include a DiT block or step, VAE encode/decode, text encoding, and a scheduler
  step. Features include parallelism, a kernel/backend swap, compile, caching,
  quantization, and sparsity.
- **Not L2:** shape-only smoke tests, full-pipeline generation, LPIPS against a
  rendered output, and tests that merely assert tensors are finite.

If a full pipeline is required to reach the component, intercept and compare
the component's input and output tensors. Do not turn the test into an E2E
pixel comparison.

## Define the paired experiment

Write down these four items in the test docstring or nearby helper names:

1. The component boundary and emitted tensor.
2. The feature or model-specific behavior under test.
3. The trusted reference.
4. Whether the expected numerical change is transparent (T1) or lossy (T2).

Choose the reference in this order:

1. Feature-off in the same process, with the same weights and inputs.
2. An official or otherwise trusted external implementation run alongside.
3. An eager or FP32 mathematical oracle for a small L1 operation.

Do not commit a golden tensor for every model-by-feature cell. When dependency
conflicts require subprocesses, capture both sides during the test run or pass
the reference tensors through temporary/scratch files created for that run.

## Make both sides the same experiment

Hold constant everything except the feature being tested:

- GPU architecture and nominal precision;
- model/component weights and loaded weight layout;
- tensor shapes, layouts, masks, conditioning, timesteps, and scheduler state;
- initial latents and every stochastic input.

Inject identical noise or latent tensors into both sides. Matching seeds is not
enough, and matching two frameworks' RNG streams is not a requirement. Avoid
unnecessary device copies when arranging the comparison.

For parallel tests, compare against the single-GPU path from the same job and
assert every output stream affected by sharding or collectives. For an external
reference, verify that both frameworks actually loaded the intended checkpoint
and dtype before comparing outputs.

## Assert useful numerical evidence

Report at least:

- relative L2: `norm(actual - expected) / norm(expected)`;
- cosine similarity over flattened tensors;
- maximum absolute error.

Add p99 absolute error when a large tensor can hide localized failures. Add
per-channel mean shift for quantization. Handle a zero-norm reference explicitly
rather than hiding it behind an arbitrary denominator.

Also add a **feature-active assertion**: prove the feature or alternate path ran
and was not silently a no-op. Prefer an observable path flag, call count,
sharding state, or nonzero feature delta over inspecting implementation details.

Use statistical/tolerance gates, never bit-exactness as a requirement:

- **T1, transparent:** start with `rtol=atol=1e-3`, relative L2 at most `1e-2`,
  and cosine at least `0.9999`.
- Relax T1 only for a named accumulation cause such as reduction reordering,
  a different backend/kernel, compile fusion, or a parallel partial-sum merge.
  The outer reference band is relative L2 `5e-2` and cosine `0.999`.
- **T2, lossy:** measure the same statistics, calibrate the bar from the
  feature's observed drift plus headroom on supported GPU architectures, and
  record that evidence. Do not copy a universal T2 tolerance. Cosine `0.99` is
  a floor that requires investigation, not an automatic default.

If a supposedly transparent feature needs a lossy-width tolerance, treat that
as a finding. Do not loosen the threshold merely to make the test pass.

## Keep the test diagnostic and affordable

- Compare the component's native output: latent tensors for DiT/VAE encode,
  pixel tensors for VAE decode, and embeddings for a text encoder.
- Use the smallest realistic shape that exercises the production path. Do not
  replace a layout-sensitive or model-specific path with an unrelated toy
  architecture.
- Keep one case per distinct code path. Remove parameter combinations that
  exercise identical behavior.
- On failure, include the component, feature, dtype, shape, and measured
  statistics in the assertion message.
- Skip only when a required GPU capability, checkpoint, or trusted reference is
  absent. A missing reference must not turn a parity test into a smoke test.
- Do not use LPIPS, PSNR, or SSIM at L1/L2.

Place internal tests under `tests/unittest/_torch/visual_gen/`. Put shared test
helpers next to the owning model tests unless multiple model families genuinely
reuse them. Use existing metric, checkpoint, distributed-run, and model-loading
helpers before adding another implementation.

## Execute and calibrate

Run CPU-only logic locally when possible. Run GPU tests from the authoritative
Mac worktree through `gpu-run` and its native workspace; never borrow another
worktree's virtual environment. Set `LLM_MODELS_ROOT` for tests that load model
weights, and keep Hugging Face checkpoints under remote `$SCRATCH/models`.

For T1, record the observed statistics in the test output or change summary and
set the threshold above the measured deterministic floor with modest headroom.
For T2, test every supported GPU-architecture stratum used to justify the bar.
Use `python scripts/test_to_stage_mapping.py --tests "<test id>"` when adding a
new CI test or changing its placement.

Before handing off the change:

1. Run the focused reference-paired test twice to detect nondeterminism.
2. Run adjacent component tests that share the modified helpers.
3. Run pre-commit on the explicitly changed files.
4. Report exact commands, hardware, precision, observed statistics, thresholds,
   skips, and anything not run.
