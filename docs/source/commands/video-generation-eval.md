# Video generation evaluation

Generate videos with VisualGen and evaluate them against fixed reference videos:

```bash
trtllm-eval --model /path/to/model video_generation_eval \
  --config generator.yaml \
  --evaluator lpips \
  --evaluator-options lpips_video.yaml \
  --prompts prompts.jsonl \
  --output-dir results
```

The generator YAML uses the same VisualGen arguments and `generation_params`
as `image_generation_eval`. Keep the model, prompt, seed, generation parameters,
precision, and video encoding consistent with the reference baseline.

`prompts.jsonl` associates stable IDs with prompts:

```json
{"id": "fox", "prompt": "A red fox walking through snow"}
```

`lpips_video.yaml` configures the metric and the reference locations:

```yaml
references:
  fox: baselines/fox.mp4
net: alex
device: cpu
```

Reference paths are absolute or relative to the evaluator YAML directory. Every
selected prompt needs a reference; duplicate prompt IDs are rejected. Quote
numeric YAML keys (for example `"0"` for the first plain-text prompt).

LPIPS requires `lpips`, `opencv-python-headless`, and an MP4 encoder such as
`ffmpeg`. Supported LPIPS backbones are `alex`, `vgg`, and `squeeze`. The default
device is CPU; set `device: cuda` for GPU scoring. Pretrained backbone weights
must be cached or downloadable by the LPIPS dependency.

The evaluator converts decoded frames from BGR to RGB, normalizes them to
[-1, 1], scores matching frames, and averages their LPIPS distances. Lower is
better. It processes one frame pair at a time and rejects empty videos,
mismatched frame counts, and mismatched dimensions. It does not resize, trim,
or temporally align videos; reference and generated frame rates must match.

Generated MP4s are saved under `generated_videos/`. `results.json` contains
per-prompt metric scores and their mean; `results.jsonl` contains per-prompt
records. Generation or scoring failures fail the command. Scores do not impose
a universal quality threshold.

Video metrics implement `VideoEvaluator.evaluate(prompt_id, prompt, video)`
and `close()`, returning named scores. Only LPIPS is implemented; other metrics
can provide their own options and reference requirements through the evaluator
factory without changing generation.
