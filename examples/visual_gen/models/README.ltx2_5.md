<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native LTX-2.5 IC-LoRA video editing

The standalone [example](ltx2_5.py) uses the public VisualGen API for one SDR
control video. Stage 1 runs eight Euler steps at half resolution with IC-LoRA
and clean reference tokens. An x2 latent upsampler precedes three full-resolution
Stage 2 steps with pristine distilled weights and no reference tokens. Audio is
generated in both stages. Runtime uses native TensorRT-LLM components.

## Setup and assets

Use a dedicated environment with TensorRT-LLM and its matching NVIDIA
PyTorch/CUDA build, then install the [model requirements](requirements.ltx2_5.txt):

```bash
python -m pip install -r examples/visual_gen/models/requirements.ltx2_5.txt
export HF_HOME="$PWD/.cache/ltx2_5/huggingface"
hf auth login
python scripts/visualgen_eval/download_ltx25_assets.py \
  --model-root /models/LTX-2.5 --run-root ./ltx2_5-download-records
```

Gemma4 requires Transformers 5.14.1, overriding the base package's pin. This is
not a resolver-compatible extra; preserve environments used by other models.
Do not use this environment for unrelated ModelOpt HF quantization. OpenCV
loads control videos; FFmpeg must be installed for MP4 output.

The downloader fixes the model revision to
`Lightricks/LTX-2.5@49b093a39c19169a8a81721dcc541e07e945232f` and the adapter to
`Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control@20a580c8606c06a0988d40375a61309e0fe5893d`.
It records file sizes and SHA256 hashes. All three native precision modes start
from the **BF16 distilled** transformer. Official prequant checkpoints are not
accepted by this pipeline. The matching LTX-tuned Gemma4-with-proj, convolutional
video VAE, audio VAE/vocoder and spatial x2 upsampler are required; generic Gemma4
and the diffusion VAE are not substitutes. Checkpoint loading rejects missing,
duplicate, mismatched and unexplained extra tensors.

Set the five component paths in the [BF16 YAML](../configs/ltx2_5-iclora-bf16-1gpu.yaml).
Then run on one visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use the GPU assigned by your scheduler.
python examples/visual_gen/models/ltx2_5.py \
  --model /models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --visual_gen_args examples/visual_gen/configs/ltx2_5-iclora-bf16-1gpu.yaml \
  --source /data/control.mp4 --height 256 --width 256 --num_frames 9 \
  --seed 42 --output_path edited.mp4
```

Use an SDR control already at the requested frame rate, with enough frames.
Frames are consumed in order without FPS resampling. The downloader includes a
raw example clip; preparing a Canny control is a separate preprocessing choice.
For independent comparisons, use the identical prepared control on both sides.

## Numerical contract

BF16 with VANILLA attention is the verified configuration. Normalization
parameters and RoPE buffers preserve their FP32 boundaries. LoRA uses the given
coefficient directly, without PEFT alpha/rank scaling. Separately stored pristine
and conditioned weights prevent Stage 1 fusion from leaking into Stage 2 or the
next request. The `ref0.5` adapter variant has spatial reference factor 2 and
temporal factor 1; runtime LoRA scale and reference strength remain 1.

The example uses one prompt, one video, no CFG and the distilled 8+3 schedule.
Dimensions must align to `64 * reference_downscale_factor`; frames must be
`8n+1`. Multiple outputs, image/audio references, step caches, compile and CUDA
Graph are rejected. The delivered configurations disable compile and graphs.
HDR and alternate attention masks are unqualified.

Public audio retains its raw generated length and volume. MP4 saving pads only
its encoding copy with trailing silence. Score raw audio from the output object,
never AAC or the padded preview. The 9-frame case produces 15,840 raw stereo
samples at 48 kHz; the 24 fps preview spans 18,000 samples.

## Precision support and quality limits

| Configuration | Transformer Linear | Attention | Status |
|---|---|---|---|
| [BF16](../configs/ltx2_5-iclora-bf16-1gpu.yaml) | BF16 | VANILLA | Verified on single B200 |
| [Experimental FP8](../configs/ltx2_5-iclora-fp8-experimental-1gpu.yaml) | Dynamic FP8_BLOCK_SCALES | VANILLA | Functional; quality UNQUALIFIED |
| [Experimental NVFP4](../configs/ltx2_5-iclora-nvfp4-vanilla-experimental-1gpu.yaml) | Dynamic NVFP4 | VANILLA | Functional; quality UNQUALIFIED |
| [Experimental NVFP4 attention](../configs/ltx2_5-iclora-nvfp4-attention-experimental-1gpu.yaml) | Dynamic NVFP4 | Video NVFP4 Q/K + FP8 V; other roles dense FlashInfer | Functional; quality UNQUALIFIED |

Quantization is opt-in through the corresponding YAML. Gemma4, connectors,
VAEs, upsampler and audio decoding retain the verified precision boundaries.
Stage 1 merges the original BF16 IC-LoRA effective weights before quantization;
Stage 2 restores separately quantized pristine weights and matching scales.
Bank switches invalidate derived layouts/caches. Activation scales are computed
for the current input. FP8 uses 128-element blocks and FP32 CuTe scales on B200;
unsupported dimensions retain BF16. NVFP4 uses packed E2M1, block16 E4M3 scales
and FP32 global scales. Its measured coverage is 1,660 Transformer Linear layers,
including all 480 LoRA targets. Original checkpoint FP32 tensors retain their
values and dtype; added quantization scales are separate state.

Prior independent B200 validation used fixed official BF16 references:

| Shape (width x height x frames) | BF16 raw LPIPS | NVFP4 + VANILLA raw LPIPS | NVFP4 + quantized attention raw LPIPS |
|---|---:|---:|---:|
| 256 x 256 x 9 | 0.0001066 | 0.4187472 | 0.4666124 |
| 256 x 256 x 81 | 0.0001723 | 0.3658614 | 0.3706720 |
| 1024 x 768 x 9 | 0.0002113 | 0.2475870 | 0.2914341 |

FP8 also completed these shapes, but its measured quality remains UNQUALIFIED.
The quantized results exceed the unchanged provisional video LPIPS target of
0.05 and fail required raw-audio spectral observations. Their operator/bank
checks and A -> B -> A request reuse do not qualify generated media. These
experiments cover the fixed prompt/control/seed cases, not arbitrary content.
No speedup, steady-state latency, compile or CUDA Graph result is claimed.

## Validation and CI

Focused tests live in `tests/unittest/_torch/visual_gen/test_ltx2_5_*.py`, including
independent FP8/FP4 payload/scale references and stage-bank/request-state checks.
The public example smoke and independent official BF16 LPIPS gate live in
`tests/integration/defs/examples/visual_gen/test_ltx2_5.py`, registered on B200.
The job requires the dedicated model environment and provisioned assets:

```bash
export LLM_MODELS_ROOT=/models
export LTX25_MODEL_ROOT=/models/LTX-2.5
export LTX25_V2V_SOURCE=/data/control-canny-9f-24fps.mp4
export LTX25_V2V_BF16_GOLDEN=/references/ltx2_5/official-bf16-256x256-9f.mp4
python -m pytest -q tests/unittest/_torch/visual_gen/test_ltx2_5_*.py
python -m pytest -q -s tests/integration/defs/examples/visual_gen/test_ltx2_5.py
```

The BF16 golden must have its generated `manifest.json` beside it. The test
verifies the recipe, control hash and all six component hashes before LPIPS.
Missing required assets fail. Generate a new independent reference in a separate
official environment, using clean Lightricks source at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`:

```bash
/path/to/official-python scripts/visualgen_eval/generate_ltx2_5_reference.py \
  --ltx-root /path/to/LTX-2 --model-root /models/LTX-2.5 \
  --source /data/control-canny-9f-24fps.mp4 --output-directory /references/ltx2_5
```

That environment needs the fixed source's `ltx-core` and `ltx-pipelines`, matching
Gemma4 Transformers and media dependencies. Keep it separate from native tests.
`scripts/visualgen_eval/compare_ltx2_5_traces.py` checks supplied tensor captures;
`compare_ltx25_quantized.py` reports raw media and spectral drift with an explicit,
frozen policy. Reports without a policy are diagnostic, not certification.
Independent quantized references are distinct recipes and never replace the
BF16 quality baseline. Lossy quality acceptance requires maintainer review.

Derived Lightricks components retain their original attribution and the
`LicenseRef-LTX-2x` terms reproduced in the repository LICENSE.
