# Visual generation evaluation

**Table of contents**

- [Image generation evaluation](#image-generation-evaluation)
  - [Qwen Image Bench](#qwen-image-bench)
- [Video generation evaluation](#video-generation-evaluation)
  - [LPIPS](#lpips)

`image_generation_eval` and `video_generation_eval` use two separate configurations:

* `--config generator.yaml` configures the VisualGen generator and its
  `generation_params` (for example, seed, resolution, frame count, and inference steps).
* `--evaluator-options evaluator.yaml` configures the scoring evaluator.
  Its accepted keys depend on the selected evaluator; unknown keys are rejected.

Both commands require `--prompts` and `--output-dir`. The prompt file accepts
one prompt per line, a JSON list, or JSONL records such as:

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

LPIPS requires the `lpips` and OpenCV packages, plus cached or downloadable
backbone weights. Generating MP4s also requires an encoder such as FFmpeg.
The evaluator scores decoded frame pairs individually and averages their distances;
lower is better. Reference and generated videos must have equal frame rates,
frame counts, and dimensions. Empty videos and non-finite scores are rejected;
frames are not resized, trimmed, or temporally aligned. The reported aggregate is
the mean of the per-video scores; the command does not apply a quality threshold.
