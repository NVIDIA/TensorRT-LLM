# Qwen-Image-2.1 VisualGen example

This page documents the TRTLLM VisualGen enablement artifacts for
`qwen-image-2.1`.  The implementation is intentionally split between the
native TRTLLM candidate runtime and a frozen Diffusers reference/oracle used
only for parity evidence.

## Checked-in artifacts

- Example entrypoint: `examples/visual_gen/models/qwen_image_21.py`
- Example config: `examples/visual_gen/configs/qwen-image-2.1-bf16-1gpu.yaml`
- VisualGen README index: `examples/visual_gen/README.md`
- Focused production-artifact test:
  `tests/unittest/visual_gen/test_qwen_image_21_production_artifacts.py`
- Perf-sanity/test-db wiring:
  `tests/scripts/perf-sanity/visual_gen/qwen_image_21_blackwell.yaml`

## Runtime expectations

The candidate runtime must not import the upstream Diffusers Qwen Image
pipeline, transformer/DiT, attention implementation, or video/image generation
pipeline as a runtime component.  Selective Diffusers reuse is limited to the
items declared in `design/external_component_reuse.yaml` for the run, with the
pinned version, reason, and parity evidence recorded there.

The production example uses the same denoising step count as the frozen
reference manifest.  When a source-specified value is unavailable, the VisualGen
SDK default is 30 denoising steps, represented in the checked-in config as
`num_inference_steps: 30`.

## Validation artifacts

For a production-readiness run, record the following run artifacts before
claiming the model ready:

- `validation/production_readiness_report.json` with compared VisualGen model
  patterns, changed files, validation commands, artifact hashes, unresolved
  gaps, and Diffusers dependency/CI coverage.
- `validation/verification_report.json` with M8 latency/memory/runtime config
  and artifact hashes.
- `validation/lpips_report.json` produced by
  `scripts/visualgen_eval/visual_gen_lpips_score_eval.py --dataset ...
  --output-json ... --threshold ...`.
- `validation/quality_parity_report.json` and
  `validation/module_parity_report.json` for scheduler, conditioning,
  attention/block, VAE/postprocess, and checkpoint/reference-backed parity
  checks where practical.

Do not use black, NaN/non-finite, or null media for LPIPS or candidate/reference
comparison.  Treat those outputs as invalid generation artifacts and debug the
candidate or reference run first.
