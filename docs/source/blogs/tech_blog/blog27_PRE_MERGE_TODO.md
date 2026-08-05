# Blog 27 Pre-Merge TODO

> **Temporary PR working document. This is not final blog content. Resolve or
> explicitly disposition the items below, then delete this file before the PR
> is merged.**

## Merge gate

- [ ] Resolve or explicitly defer every open item below.
- [ ] Delete this temporary file from the PR before merge.

## External reproduction package

- [ ] **Waiting on ModelOpt:** publish the calibrated Skip Softmax configuration
  in a public location. Prefer
  `examples/visual_gen/skip_softmax/wan2.2-t2v-a14b/` so the article can use a
  repository-local, component-aware merge helper. Once available, link it from
  the reproduction section and verify that it contains the component-specific
  `sparse_attention_config` blocks, calibration formulas, and ignore lists for
  both transformers. Test the documented helper from a clean Hugging Face
  download. The internal checkpoint itself is not a publication artifact.
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
- [ ] Add a synchronized compiled-BF16 versus NVFP4 + SAGE + Skip Softmax
  results video using the same prompt, seed, resolution, frame count, and
  denoising steps. Host it through an approved NVIDIA media service and keep
  only its poster image in the repository; do not commit a raw MP4.

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

- [x] Remove the six-item `Limitations` checklist and weave practical tuning
  scope into the results narrative.
- [x] Remove repeated reviewer-facing defenses around family anchors,
  end-to-end measurement, single-GPU attribution, and the interpretation of
  small LPIPS deltas.
- [x] Change “previous post” to “earlier post” and describe SAGE quantization
  as occurring in the attention path rather than necessarily inside one
  kernel.
- [x] Keep the representative profile framing concise and omit the internal
  cross-GPU validation narrative.
- [x] Identify only the release image in Reproduction; omit source commit and
  feature-PR identifiers.
- [x] Remove single-GPU framing from the article and latency-chart titles while
  retaining B200 as factual evaluation context.
- [x] Keep the pipeline chart title to “Pipeline Breakdown,” use dark green for
  attention, label the remainder “Others,” and remove its subtitle.
- [x] Merge the quality-anchor and operating-point discussion into the results
  narrative, and align points ①–③ with the table immediately below Figure 3.
- [ ] Remove every `TODO(blog27, remove before merge)` HTML comment after its
  corresponding publication asset is linked and verified.
