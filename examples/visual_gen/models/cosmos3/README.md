# Cosmos3 Text(+Image)-to-Video(+Audio) generation

Cosmos3 supports the following generation modes from a single checkpoint:

- **T2V** — text-to-video (`prompts/t2v.json`).
- **T2I** — text-to-image (`prompts/t2i.json`); emits a still frame (use `--output_type image` / a non-video `--output_path`).
- **I2V / TI2V** — image-conditioned video (`prompts/i2v.json`). Condition on a reference frame via the prompt file's `vision_path` or `--image_path`. The image may be a local path, a `file://` / `http(s)://` URL, or a `data:` URI.
- **V2V** — video-conditioned video (`prompts/v2v.json`). Condition on a reference video via `--video_path` (a local MP4/AVI file). Only the first (or last, per `condition_video_keep`) `max(condition_video_latent_indexes) * 4 + 1` input frames condition the output (5 by default); the encoded bytes pass through and each worker decodes just that window on NVDEC (see [Media I/O dependencies](#media-io-dependencies)). Validated for Nano / Super only.
- **Transfer** — control-video conditioning (`edge`/`blur`/`depth`/`seg`/`wsm` hints via `--extra_params`). The control constrains structure frame by frame; the prompt supplies appearance. `edge` and `blur` are auto-computed from `--video_path`; any hint also accepts a precomputed control clip (`{"edge": "control.mp4"}` — the example reads it and sends encoded bytes, the same contract as the `video` reference). Multiple hints compose (each adds a full control-token copy of the video sequence); long videos run chunked (93 frames/chunk, stitched on overlap frames) — but only past the first chunk, so raise `num_frames` above the pipeline default to generate one: it bounds how many frames are decoded from the inputs, and so how long the output can be. A single-hint request picks up that hint's tuned sampling preset — guidance scale, control guidance and flow shift — for any of those the request leaves unset; requests with several hints fall back to the generic video defaults. The active hint names are also appended to the prompt as a one-sentence control-adherence directive; pass `"emphasize_control_in_prompt": false` to suppress it for clean baselines or ablations.
- **T2AV** — text-to-video with synchronized audio (`prompts/t2av.json` with `enable_audio: true`, or pass `--enable_audio`). Combine with a `vision_path` for image-conditioned audio-video (TI2AV).
- **Action** — policy / forward dynamics / inverse dynamics generation (pass `--action_mode`); `inverse_dynamics` reads its observation clip from `--video_path` (MP4/AVI, decoded on worker NVDEC like V2V). Action and audio generation are mutually exclusive. A predicted trajectory has no representation in a video container, so action runs are saved as `safetensors` or `pt`, keeping the rollout and the action tensor in one payload — over `trtllm-serve` the default `format=auto` selects that payload automatically, and an explicit `mp4`/`avi` is rejected.

## Checkpoints

Pass the Hub ID or local path via `--model`:

- [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano)
- [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super)
- [`nvidia/Cosmos3-Super-Text2Image-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step) — DMD2-distilled text-to-image: fixed 4-step schedule with classifier-free guidance baked into the weights. Steps/guidance are read from the checkpoint; conflicting request values are rejected. Use with `configs/cosmos3-t2i-1gpu.yaml`.
- [`nvidia/Cosmos3-Super-Image2Video-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) — DMD2-distilled image-to-video: same fixed 4-step, guidance-baked-in contract. The default omni video shape (720p × 189 frames) is the deployed shape, so no dedicated config is needed. This checkpoint declares `default_use_system_prompt: true` in its `model_index.json`, which the pipeline applies automatically (override with `--use_system_prompt` / `--no-use_system_prompt`).
- [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) — 4B Nemotron-dense backbone supporting **T2I / T2V / I2V only**: no audio tower, and the checkpoint's action weights are not supported by this pipeline yet. 480p-native defaults (832×480 × 121 frames, 50 UniPC steps on the checkpoint-declared native flow schedule with shift 3.0, guidance 5.0; T2I defaults to 640×640), so no dedicated config is needed. The model card validates 256p/480p, 50–150 frames, and 12–30 FPS; requests outside that envelope run with an advisory log.

## Guardrails

Guardrails are enabled by default (required by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)). Install and authenticate as follows:

```bash
pip install cosmos_guardrail==0.3.0 && pip uninstall opencv-python
```

Accept the terms for the guardrail checkpoint at https://huggingface.co/nvidia/Cosmos-1.0-Guardrail and set a valid `HF_TOKEN` (the checkpoint is downloaded automatically on first run).

To run without guardrails (you are responsible for safe deployment):

```bash
export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1
```

## Media I/O dependencies

- Saving `.mp4` output requires the `ffmpeg` CLI on `PATH` (`apt-get install -y ffmpeg`); without it the encoder falls back to `.avi`.
- Decoding MP4/AVI reference videos (V2V) happens in the worker processes on NVDEC via PyNvVideoCodec, a declared TensorRT-LLM dependency — nothing extra to install. Tested combinations: H.264 in MP4 and H.264 in AVI; other containers/codecs/profiles depend on the demuxer and the GPU's NVDEC capabilities and are best-effort.
- Transfer's `edge`/`blur` controls are derived on the GPU from the reference video — nothing extra to install. Precomputed controls (`depth`/`seg`/`wsm`, or a precomputed `edge`/`blur`) are decoded like any other reference video.

## Deployment configs

See `examples/visual_gen/configs/`:

- `cosmos3-nano-1gpu.yaml` — 1 GPU
- `cosmos3-super-4gpu.yaml` — 4 GPU, CFG + Ulysses + parallel VAE
- `cosmos3-t2i-1gpu.yaml` — 1 GPU, text-to-image deployments (base or distilled): warms the deployed 1024×1024 single-frame shape instead of the omni video shape.

Example prompts live under `prompts/` (mirroring `cosmos3-internal/inputs/omni`).

### Prompt inputs

`--prompt` and `--negative_prompt` each accept **either literal text or a path to a
prompt file**, chosen by whether the value names an existing file. `--prompt_file`
and `--negative_prompt_file` accept a path only and fail if the file is missing, so
use those when a silent fallback to literal text would be a bug (scripts, CI).

A prompt file may hold any of three shapes:

| Shape | Example | Notes |
|---|---|---|
| Omni prompt object | `prompts/t2v.json` | `prompt` plus optional `model_mode`, `vision_path`, `enable_audio`, which supply defaults for the matching flags |
| Structured caption | a checkpoint's `assets/example_i2v_prompt.json` | the object *is* the caption; carries no options |
| Plain text | any `.txt` | used verbatim |

Structured captions are what the model cards ship and what the checkpoints were
tuned on; they give noticeably cleaner output than a one-line summary.
`--negative_prompt` defaults to `cosmos3_negative_prompt.json` in this directory.

## Usage

```bash
# T2V: text-to-video
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2v.json \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# I2V/TI2V: image-conditioned video (vision_path is read from the prompt file;
# local path, file://, http(s):// URL, or data: URI are all accepted)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/i2v.json \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# I2V with an explicit conditioning image (overrides the prompt file)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/i2v.json \
    --image_path https://example.com/frame.jpg \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# V2V: video-conditioned video (continues the first frames of --video_path).
# Best results when the prompt describes the input video — e.g. continue a
# T2V output reusing its original prompt. Output size follows the source's
# aspect ratio (the closest supported bucket) unless the request sets
# height/width; the reference is center-cropped to whatever size is chosen.
python cosmos3.py --model /path/to/Cosmos3-Nano \
    --prompt_file prompts/v2v.json \
    --video_path /path/to/Cosmos3-Nano/assets/example_i2v_output.mp4 \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# T2AV: text-to-video with synchronized audio
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2av.json \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# T2I: text-to-image
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2i.json \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml \
    --output_path output.png

# T2I, distilled 4-step checkpoint (use the T2I config so warmup runs the
# image shape; steps/guidance come from the checkpoint automatically)
python cosmos3.py --model nvidia/Cosmos3-Super-Text2Image-4Step \
    --prompt_file prompts/t2i.json \
    --visual_gen_args ../../configs/cosmos3-t2i-1gpu.yaml \
    --output_type image \
    --output_path output.png

# I2V, distilled 4-step checkpoint (steps/guidance and the system-prompt
# default come from the checkpoint automatically; defaults are the deployed
# 720p x 189-frame shape, so no config is required)
python cosmos3.py --model nvidia/Cosmos3-Super-Image2Video-4Step \
    --prompt "The camera slowly pans right across the scene" \
    --image_path https://example.com/frame.jpg \
    --output_path output.mp4

# Transfer: control-video conditioning — structure from the control video,
# appearance from the prompt. edge/blur are computed from --video_path.
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "The same scene rendered as a photorealistic video, sharp detail." \
    --video_path /path/to/reference.mp4 \
    --extra_params '{"edge": true}' \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# Transfer with a fully synthetic control (no assets): generate an edge-map
# video of a bouncing ball, then let the prompt paint it photoreal.
# Keep synthetic controls edge-style: the blur hint expects the low
# frequencies of natural video, and flat synthetic color fields degrade
# generation. Temporal exposure swings (e.g. pulsing global light) do not
# transfer — express lighting spatially or in the prompt instead.
python generate_bouncing_ball_control.py --out_dir ./ball_control
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "A photorealistic beach ball with colorful panels bouncing between the walls of an enclosed room, studio lighting." \
    --extra_params '{"edge": "./ball_control/control.mp4"}' \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# Multi-hint transfer: edge pins the layout, blur pins the palette/lighting.
# Hints must describe the same underlying video as each other and the prompt.
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "The same scene, ultra sharp, professional photography." \
    --video_path /path/to/reference.mp4 \
    --extra_params '{"edge": true, "blur": true}' \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# Cosmos3-Edge image-to-video (480p-native defaults: 832x480 x 121 frames).
# Reproduces the model-card sample: the checkpoint ships a structured prompt and
# its own negative prompt alongside the conditioning image. Fetch them with
#   hf download nvidia/Cosmos3-Edge --local-dir Cosmos3-Edge
python cosmos3.py --model nvidia/Cosmos3-Edge \
    --prompt Cosmos3-Edge/assets/example_i2v_prompt.json \
    --negative_prompt Cosmos3-Edge/assets/negative_prompt.json \
    --image_path Cosmos3-Edge/assets/example_i2v_input.jpg \
    --output_path output.mp4

# Inline prompt (--prompt or a JSON file path)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "A cute puppy playing with a ball in a park" \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml

# Action — policy (first frame + instruction -> predicted action + rollout video)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/action_policy.json \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml \
    --action_mode policy \
    --domain_name bridge_orig_lerobot \
    --raw_action_dim 10 \
    --output_path policy_rollout.safetensors \
    --action_output_path policy_action.json

# Action — forward dynamics (first frame + action trajectory -> rollout video)
# action_trajectory.json is a [T, D] list of lists; D is the embodiment's action
# width (9 for av) and a mismatch is rejected.
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/action_forward_dynamics.json \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml \
    --action_mode forward_dynamics \
    --domain_name av \
    --action_json action_trajectory.json \
    --output_path forward_dynamics.safetensors

# Action — inverse dynamics (video -> predicted action)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/action_inverse_dynamics.json \
    --video_path /path/to/observation_clip.mp4 \
    --visual_gen_args ../../configs/cosmos3-nano-1gpu.yaml \
    --action_mode inverse_dynamics \
    --domain_name bridge_orig_lerobot \
    --raw_action_dim 10 \
    --output_path inverse_video.safetensors \
    --action_output_path inverse_action.json
```
