<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LTX-2.5 IC-LoRA video editing (BF16)

This pipeline implements the single-control-video subset of Lightricks'
[ICLoraPipeline](https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py):
half-resolution generation with IC-LoRA, spatial latent upsampling, then
full-resolution refinement with the original transformer weights and no control
video tokens. It uses the distilled 8+3 Euler schedule and generates audio.

## Run

Install TensorRT-LLM, then install the model-specific dependencies in a dedicated
LTX-2.5 environment:

```sh
python -m pip install -r examples/visual_gen/models/requirements.ltx2_5.txt
```

This explicitly replaces the base Transformers pin with the qualified 5.14.1
version for this environment only, following the model-specific install pattern.
It is not a resolver-compatible extra: `pip check` can report the base pin
mismatch. Do not use this environment for ModelOpt HF quantization, whose version
constraints may conflict. Other models retain TensorRT-LLM's base requirements.
The encoder fails early with installation guidance for unqualified versions.
Control-video decoding uses the shared OpenCV `media_io` path, without PyAV or
source-audio extraction. MP4 output with audio requires `ffmpeg` on `PATH`.
Use the official split BF16 pack with the convolutional video VAE. Edit component
paths in `examples/visual_gen/configs/ltx2_5-iclora-bf16-1gpu.yaml`, then run from
the repository root:

```sh
python examples/visual_gen/models/ltx2_5.py \
  --model /models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --visual_gen_args examples/visual_gen/configs/ltx2_5-iclora-bf16-1gpu.yaml \
  --source /data/control.mp4 --height 256 --width 256 --num_frames 9 \
  --seed 42 --output_path edited.mp4
```

Required components are the matching LTX-tuned Gemma4 encoder (including tokenizer
and readout projections), convolutional video VAE, audio VAE/vocoder, x2 spatial
upsampler, and IC-LoRA. The reference recipe uses
`ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`; the adapter's name does not
imply a dependency on an LTX-2.3 TensorRT-LLM implementation. Checkpoint metadata
supplies its spatial and temporal reference scales. Missing or incompatible
weights fail loading instead of using random projections.

Use one GPU, BF16, `VANILLA` attention, and disable CUDA graphs, compilation and
step caching. Fused AdaLN and QK-normalization/RoPE are disabled to preserve
the reference BF16 rounding boundaries. No performance improvement is claimed.
Use a 96 GB GPU and provision at least approximately 100 GB of host RAM for this
recipe. Measured small cases used about 68 GiB of allocated GPU memory and
52 GiB peak host RSS; loading needs additional headroom. CPU copies of adapted
base and merged weights enable exact stage restoration and transfer about
52 GB of weights per request. These measurements do not qualify the API defaults
of 768x1280 / 65 frames, which have not been tested. Use the explicit validated
256x256 / 9-frame example first.
The pipeline currently keeps all components resident. Dimensions must align to
`64 * reference_downscale_factor`; frame count must be `8n+1`. Provide at least
that many SDR control frames. Frame-rate conversion, HDR, masks, image references,
multiple videos, prompt enhancement and one-stage generation are outside this
initial subset. Supply a control clip already at the requested frame rate.

The public API uses `VisualGen`, `VisualGenArgs`, and
`VisualGenParams.video_reference=MediaRef(content=path, format="path")`.
`extra_params.reference_strength` controls reference latent conditioning (default
1.0); `pipeline_config.iclora_scale` controls the stage-1 adapter. Stage 2 always
restores unmodified base weights. NVFP4, DiffVAE and customer-specific reference
semantics are not enabled by this configuration.

## Validation

Unit tests are in `tests/unittest/_torch/visual_gen/test_ltx2_5_*.py`. The public
example smoke test and independent BF16 LPIPS gate are in
`tests/integration/defs/examples/visual_gen/test_ltx2_5.py`. Run them in the
model-specific environment. Integration tests and the Gemma4 encoder tests are
local qualification tests until the model-specific CI environment and assets are
provisioned. They are not currently pre-merge gates; missing assets are failures.
`LTX25_MODEL_ROOT`, `LTX25_V2V_SOURCE`, and `LTX25_V2V_BF16_GOLDEN` can override the
shared-store paths.

The independent golden must come from the pinned official implementation above,
with the same six checkpoint files, input, prompt, 256x256, 9 frames, 24 fps,
seed 42, BF16 VAE, no prompt enhancement, default tiling and 8+3 sigmas. Record
SHA256 hashes, Torch/Transformers/CUDA versions, attention backend and raw tensor
traces alongside the golden. Never regenerate a golden from the candidate.
The initial LPIPS acceptance limit is strictly below 0.05; failures require
investigation. Smoke success alone does not establish numerical parity. Compare
prompt contexts, reference VAE latents, initial noise, every Euler step, upsampled
latents and decoded video/audio before accepting quality. Audio padding for muxing
is excluded from waveform comparisons. Audio waveform repeatability is not
bitwise exact on the tested GPU, including the unchanged baseline; audio RMSE
is diagnostic and does not establish perceptual audio quality.

Local validation on an RTX PRO 6000 Blackwell with Torch 2.14 and Transformers
5.14.1 passed the installed-wheel public example and independent MP4 LPIPS gate
(0.020812 < 0.05). At 256x256, seed 42, both 9- and 65-frame runs matched all
22 official Euler input/next tensors bitwise; raw-frame LPIPS was 0.000258 and
0.000285. Repeating the 9-frame request after a 65-frame request reproduced its
video and step latents exactly. These results cover one control clip and seed.
Upstream CI assets, broader quality/memory coverage, release-build qualification
and cross-model dependency regressions remain required for full qualification.

Generate reference assets in the separately provisioned official environment:

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/visualgen_eval/generate_ltx2_5_reference.py \
  --ltx-root /path/to/pinned/LTX-2 --model-root /models/LTX-2.5 \
  --source /data/control-canny-65f.mp4 --output-directory /data/new-official-golden
```

The generator verifies the source commit and tracked cleanliness, records hashes
for all six components and the control clip, and refuses to overwrite a golden.
Keep `manifest.json` next to the MP4; the quality gate checks its recipe and hashes.
This utility requires the official LTX dependencies in that separate environment;
it is not a runtime dependency of the TensorRT-LLM pipeline.


### Extended qualification

The default integration case stays at 256x256 / 9 frames / seed 42. Run the same
public example and strict manifest/LPIPS checks with `LTX25_V2V_CASE=portrait`
(384x256 / 17 frames / seed 17), `square` (512x512 / 9 frames / seed 123), or
`long` (256x256 / 65 frames / seed 7), or `temporal`
(256x256 / 129 frames / seed 42), and set `LTX25_V2V_BF16_GOLDEN` to the
corresponding independently generated MP4. Set `LTX25_V2V_SOURCE` to the same
control used for that golden; the temporal case requires a 129-frame source.
A mismatched golden recipe fails.
These expanded cases are offline qualification, not additional pre-merge jobs.
The reference generator accepts `--height`, `--width`, `--num-frames`, and `--seed`.
Keep each recipe in its own new output directory.

For intermediate and memory qualification, run in a dedicated process using the
installed candidate wheel:

```sh
python scripts/visualgen_eval/ltx25_bf16_trace.py \
  --model-root /models/LTX-2.5 --source /data/control-canny-65f.mp4 \
  --height 384 --width 256 --num-frames 17 --seed 17 \
  --max-allocated-gib 90 --output candidate-trace.pt
python scripts/visualgen_eval/compare_ltx2_5_traces.py \
  --candidate candidate-trace.pt --reference /data/new-official-golden/official-bf16.pt \
  --report parity.json
```

The 90 GiB allocation budget is for the 96 GB development workstation; choose an
explicit budget for another GPU. The trace command checks both loading and
inference allocation peaks and reports reserved CUDA memory and process peak RSS.
It uses internal loader hooks only for instrumentation and does not replace public
API testing. Exact tensor comparison requires matching Torch, Transformers,
attention selection and hardware. LPIPS remains strictly below 0.05; audio RMSE
is diagnostic rather than a perceptual acceptance gate. Existing reports and
goldens are never overwritten by these commands.

The expanded local matrix passed exact contexts and Euler tensors, with raw-frame
LPIPS 0.000301 (portrait), 0.000396 (square), 0.000176 (long), and 0.000229
(temporal). The temporal case used a new 129-frame loop of the original control
to exercise VAE temporal tiling; it does not establish natural long-video quality.
Inference peak allocation was approximately 67.97 GiB, reserved memory 69.78 GiB, and host peak
RSS 51.60 GiB across the sequential matrix. These measurements include tracing
and describe this workload, not production resolution limits or a speedup claim.
