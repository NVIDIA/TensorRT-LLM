# Blog 27 Pre-Merge TODO

> **Temporary PR working document. This is not final blog content. Resolve or
> explicitly disposition the items below, then delete this file before the PR
> is merged.**

## Merge gate

- [ ] Resolve or explicitly defer every open item below.
- [ ] Delete this temporary file from the PR before merge.

## External reproduction package

- [ ] **Waiting on ModelOpt:** publish the calibrated Skip Softmax configuration
  in a public location. Once available, link it from the reproduction package
  and verify that it contains the component-specific
  `sparse_attention_config` blocks, calibration formulas, and ignore lists for
  both transformers. The internal checkpoint itself is not a publication
  artifact.
- [x] Correct the configuration guidance: one direct
  `threshold_scale_factor` does not reproduce this dual-transformer result,
  because the two transformer components use different calibration formulas.
- [ ] Publish the seven exact prompts, their seeds, the default negative
  prompt, scheduler settings, and `boundary_ratio`.
- [ ] Publish a runnable sweep harness that records the full warmup, CUDA
  synchronization, timed-forward, dense-anchor, eager-reference, and
  multi-GPU scheduling procedures.
- [ ] Record the TensorRT-LLM release image used for the sweep and the commands
  needed to run it. The final blog should identify the release image only,
  without source commit or feature-PR identifiers.
- [ ] Publish the LPIPS evaluator with pinned PyTorch, torchvision, LPIPS, and
  AlexNet weight versions, including frame normalization, batching, and
  aggregation behavior.
- [ ] Publish aggregate and per-prompt timing and LPIPS CSV files together
  with the scripts used to generate the article's tables and figures. Raw
  frame arrays are optional if generation and evaluation are fully specified.
- [ ] For the component breakdown, publish the Nsight Systems command and
  version plus the kernel-classification script or a reproducible trace
  summary.

## Validation backlog

- [ ] Repeat timing and quality runs sufficiently to report run-to-run
  variation, confidence intervals, and a justified external pass tolerance.
- [ ] Measure achieved Skip Softmax sparsity by layer and denoising step for
  each requested `target_sparsity` operating point.
- [ ] Decide whether the quality claim requires prompt adherence, motion,
  temporal-consistency, or human-preference evaluation in addition to LPIPS.
- [ ] If claiming behavior beyond this workload, validate additional models,
  resolutions, schedulers, seeds, and GPU generations. Otherwise keep the
  final claim scoped to Wan 2.2 T2V-A14B on B200.
- [ ] Define a stable same-build quality reference and quantify eager versus
  compiled BF16 drift before comparing quantized families across builds.
- [ ] Profile more prompts and configurations if the component percentages
  are intended as a general workload breakdown; otherwise label the current
  profile as one representative BF16 run.

## Final-blog cleanup

- [x] Replace the six-item `Limitations` checklist with a short audience-facing
  scope statement.
- [x] Remove repeated reviewer-facing defenses around family anchors,
  end-to-end measurement, single-GPU attribution, and the interpretation of
  small LPIPS deltas.
- [x] Change “previous post” to “earlier post” and describe SAGE quantization
  as occurring in the attention path rather than necessarily inside one
  kernel.
- [x] State the RC21 provenance of the representative profile in one concise
  sentence without adding the internal cross-GPU validation narrative.
- [x] In the runtime table, identify only the release image used for the main
  sweep; omit source commit and feature-PR identifiers.
