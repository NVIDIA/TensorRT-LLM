<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Visual generation evaluation

**Table of contents**

- [Image generation evaluation](#image-generation-evaluation)
  - [Qwen Image Bench](#qwen-image-bench)
- [Video generation evaluation](#video-generation-evaluation)
  - [LPIPS](#lpips)
  - [Latent RMSE](#latent-rmse)

`image_generation_eval` and `video_generation_eval` use two separate configurations:

* `--config generator.yaml` configures the VisualGen generator and its
  `generation_params` (for example, seed, resolution, frame count, and inference steps).
* `--evaluator-options evaluator.yaml` configures the scoring evaluator.
  Its accepted keys depend on the selected evaluator; unknown keys are rejected.

Both commands require `--prompts` and `--output-dir`. The prompt file accepts
one prompt per line, a JSON list in a file ending in `.json`, or JSONL records such as:

```json
{"id": 1, "prompt": "A red fox walking through snow"}
{"id": 2, "prompt": "A cat sitting on a windowsill"}
```

Integer IDs are accepted and converted to strings internally. IDs label results;
video filenames follow zero-based prompt order, not the ID value.

Use `--num-samples N` to evaluate only the first N prompts.

## Image generation evaluation

### Qwen Image Bench

`image_generation_eval` generates images and scores them with Qwen Image Bench.
Pass the generator checkpoint with `--model` and the evaluator checkpoint with
`--evaluator`. The evaluator currently requires a local checkpoint directory
containing `config.json`; its directory name must contain `qwen-image-bench`
(case-insensitive, with underscores also accepted in place of hyphens).

```bash
trtllm-eval --model /path/to/image-model image_generation_eval \
  --config generator.yaml \
  --evaluator /path/to/Qwen-Image-Bench \
  --evaluator-options qwen-image-bench.yaml \
  --prompts prompts.jsonl \
  --output-dir results
```

The evaluator YAML is optional; omit `--evaluator-options` to use all defaults.
For example, this file specifies the default token budgets and sampling settings:

```yaml
max_tokens: 4096
max_num_tokens: 8192
max_seq_len: 8192
kv_cache_max_tokens: 8192
max_batch_size: 1
temperature: 0.0
top_k: 1
top_p: 1.0
repetition_penalty: 1.05
```

All supported evaluator YAML keys are listed below. These configure the evaluator,
not the image generator.

| Key | Default | Meaning |
| --- | --- | --- |
| `image_data_format` | `pt` | Format passed to the image loader for image paths; the default loads PyTorch image tensors. |
| `max_tokens` | `4096` | Maximum generated tokens per evaluator response for a dimension. |
| `max_num_tokens` | `8192` | Evaluator runtime token budget per scheduling iteration. |
| `max_seq_len` | `8192` | Maximum evaluator sequence length, including input and generated tokens. |
| `kv_cache_max_tokens` | `8192` | Maximum tokens allocated to the evaluator's KV cache. |
| `max_batch_size` | `1` | Evaluator runtime batch capacity; this does not change the number of images generated per prompt. |
| `temperature` | `0.0` | Evaluator sampling temperature. |
| `top_k` | `1` | Evaluator top-k sampling setting. |
| `top_p` | `1.0` | Evaluator nucleus-sampling setting. |
| `repetition_penalty` | `1.05` | Evaluator sampling repetition penalty. |
| `use_fast_processor` | `false` | Request the fast Hugging Face processor implementation. |
| `trust_remote_code` | `false` | Allow custom checkpoint code when loading the evaluator model and processor. |
| `enable_block_reuse` | `false` | Enable evaluator KV-cache block reuse. |
| `include_raw_outputs` | `false` | Retain raw dimension responses in the evaluator's Python result objects. The image-generation CLI report does not currently serialize these raw responses. |

Do not put `model`, `type`, or `backend` in this YAML; use `--evaluator`
for the checkpoint. LPIPS options such as `reference_dir`, `generated_dir`,
`net`, and `device` are not Qwen Image Bench options. This evaluator scores
the generated image against its prompt and evaluation criteria, without a
reference image directory.

Select dimensions with repeated `--criteria` flags, not YAML entries. Available
values are `Quality`, `Aesthetics`, `Alignment`, `Real-world Fidelity`, and
`Creative Generation`. All five are evaluated when `--criteria` is omitted;
quote dimension names containing spaces on the command line.

## Video generation evaluation

### LPIPS

Select LPIPS with `video_generation_eval --evaluator lpips`. An
`--evaluator-options` YAML file is required:

```yaml
reference_dir: baselines
generated_dir: generated  # Optional: score existing videos instead of generating them.
net: alex                 # Optional; default: alex.
device: cpu               # Optional; default: cpu.
```

| Key | Default | Meaning |
| --- | --- | --- |
| `reference_dir` | Required | Existing directory containing the baseline MP4 videos. |
| `generated_dir` | Omitted | Existing directory containing videos to score. When supplied, generation is skipped. The root CLI still requires `--model`, but does not load it in this mode. |
| `net` | `alex` | LPIPS backbone: `alex`, `vgg`, or `squeeze`. Keep the backbone consistent when comparing scores across runs. |
| `device` | `cpu` | PyTorch device used for scoring, such as `cpu`, `cuda`, or `cuda:0`. |

Directory paths may be absolute or relative to the evaluator YAML's directory.
Files in both directories must be named by zero-based prompt order:
`0000.mp4`, `0001.mp4`, and so on. The first prompt in the example above maps
to `0000.mp4`, regardless of its `id`. Every selected prompt needs a numbered
reference and, in existing-video mode, a numbered generated video. Missing files
fail validation before model loading. Keep prompt order consistent with both directories.

To generate and score videos, omit `generated_dir` from the evaluator YAML
(the minimal file is just `reference_dir: baselines`), then supply the generator:

```bash
trtllm-eval --model /path/to/video-model video_generation_eval \
  --config generator.yaml \
  --evaluator lpips \
  --evaluator-options lpips.yaml \
  --prompts prompts.jsonl \
  --output-dir results
```

Generation defaults to seed `42` when `generation_params.seed` is omitted or null;
an explicitly configured seed is preserved and used for every prompt.
Produce baselines with the same seed and generation parameters (including resolution,
frame count, and inference steps); matching the seed alone does not ensure comparable outputs.
The effective generation parameters, including the seed, are recorded in `results.json`.

To score existing videos using the YAML above:

```bash
trtllm-eval --model /path/to/video-model video_generation_eval \
  --evaluator lpips \
  --evaluator-options lpips.yaml \
  --prompts prompts.jsonl \
  --output-dir results
```

New videos are saved as `results/generated_videos/0000.mp4`, etc. A supplied
`generated_dir` is an input directory, not a destination for new generation.
Both modes write `results.json` and `results.jsonl` under `--output-dir`.
Completed scores are flushed to `results.jsonl` after each video, so they survive
a later failure; `results.json` is written only when the entire run succeeds.
Generation rejects output paths that overlap reference videos.

LPIPS requires the `lpips` and OpenCV packages, plus cached or downloadable
backbone weights. Generating MP4s also requires an encoder such as FFmpeg.
The evaluator scores decoded frame pairs individually and averages their distances;
lower is better. Reference and generated videos must have equal frame rates,
frame counts, and dimensions. Empty videos and non-finite scores are rejected;
frames are not resized, trimmed, or temporally aligned. The reported aggregate is
the mean of the per-video scores; the command does not apply a quality threshold.

### Latent RMSE

`--evaluator rmse` measures numerical drift in **final denoising latents**, before
VAE decoding. It reports `video_latent_rmse` and, when present, `audio_latent_rmse`
separately: `sqrt(mean((candidate - reference)**2))`. Arithmetic uses CPU float64;
the summary is the arithmetic mean of per-prompt RMSE values, not a pooled RMSE.
Lower is closer to the reference, but this is not a perceptual quality score.

The generator must support `extra_params.output_type: latent` (including MiniMax
H3). The evaluator selects this option automatically during generation. Latent
files use prompt order (`0000.safetensors`, `0001.safetensors`, ...), with named
`video_latent` and optional `audio_latent` tensors. Decoded videos and tensors
saved from normal decoded outputs are not valid latent baselines. Both sides
must contain the same modalities and exact shapes; no resizing or truncation is
performed. Empty tensors and nonfinite values are rejected.

For example, to compare H3 dynamic FP8 against a BF16 baseline:

```yaml
# rmse.yaml
reference_dir: /path/to/bf16_latents
# Optional: score already generated latent files instead of generating them.
# generated_dir: /path/to/fp8_latents
```

```yaml
# fp8.yaml
quant_config:
  quant_algo: FP8_BLOCK_SCALES
  dynamic: true
generation_params:
  seed: 42
  height: 768
  width: 1344
  num_frames: 124
  frame_rate: 24
  num_inference_steps: 50
```

To generate and score:

```bash
trtllm-eval --model /path/to/MiniMax-H3 video_generation_eval \
  --evaluator rmse --config fp8.yaml --evaluator-options rmse.yaml \
  --prompts prompts.json --output-dir results/fp8
```

New tensors are saved in `results/fp8/generated_latents/`, with per-prompt scores
and generation parameters in the JSON reports. To score existing tensors, set
`generated_dir` in `rmse.yaml` and omit `--config`; the root command still requires
`--model`, although it does not load that model in existing-file mode.

Generate the BF16 reference once using the same checkpoint and generation
settings, without `quant_config`. For example, save the matching settings above
as `bf16.yaml` and use the evaluation helpers:

```python
from pathlib import Path

from tensorrt_llm.evaluate.visual_gen.config import load_prompt_records
from tensorrt_llm.evaluate.visual_gen.generators import build_visual_generator
from tensorrt_llm.evaluate.visual_gen.rmse import save_latents

_, prompts = load_prompt_records("prompts.json", None)
reference_dir = Path("bf16_latents")
reference_dir.mkdir(parents=True, exist_ok=True)
generator, params = build_visual_generator("/path/to/MiniMax-H3", "bf16.yaml")
try:
    params.extra_params = {**(params.extra_params or {}), "output_type": "latent"}
    for index, prompt in enumerate(prompts):
        output = generator.generate([prompt], params)[0]
        if output.error:
            raise RuntimeError(output.error)
        save_latents(output, reference_dir / f"{index:04d}.safetensors")
finally:
    generator.close()
```

Use the same prompt order, seed, dimensions, frame count, scheduler settings,
conditioning, model revision, and runtime settings on both sides, changing only
the intended quantization setting. File pairing validates shapes and modalities;
it cannot prove that independently supplied baselines used matching settings.
H3 latent output excludes fixed conditioning rows and retains the final scheduler
outputs for video and audio. It does not compare intermediate denoising steps or
teacher-forced trajectories.
