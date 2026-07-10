# Cosmos3 Text(+Image)-to-Video(+Audio) generation

Cosmos3 supports the following generation modes from a single checkpoint:

- **T2V** — text-to-video (`prompts/t2v.json`).
- **T2I** — text-to-image (`prompts/t2i.json`); emits a still frame (use `--output_type image` / a non-video `--output_path`).
- **I2V / TI2V** — image-conditioned video (`prompts/i2v.json`). Condition on a reference frame via the prompt file's `vision_path` or `--image_path`. The image may be a local path, a `file://` / `http(s)://` URL, or a `data:` URI.
- **V2V** — video-conditioned video (`prompts/v2v.json`). Condition on a reference video via `--video_path` (a local frame directory or `.mp4`/`.avi` file). Only the first (or last, per `condition_video_keep`) `max(condition_video_latent_indexes) * 4 + 1` input frames condition the output (5 by default); `.mp4`/`.avi` decode uses OpenCV (see [Media I/O dependencies](#media-io-dependencies)).
- **Transfer** — control-video conditioning (`edge`/`blur`/`depth`/`seg`/`wsm` hints via `--extra_params`). The control constrains structure frame by frame; the prompt supplies appearance. `edge` and `blur` are auto-computed from `--video_path`; any hint accepts a precomputed `{"control_path": ...}` (server-/machine-local media). Multiple hints compose (each adds a full control-token copy of the video sequence); long videos run chunked (93 frames/chunk, stitched on overlap frames).
- **T2AV** — text-to-video with synchronized audio (`prompts/t2av.json` with `enable_audio: true`, or pass `--enable_audio`). Combine with a `vision_path` for image-conditioned audio-video (TI2AV).

## Checkpoints

Pass the Hub ID or local path via `--model`:

- [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano)
- [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super)

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
- Decoding `.mp4`/`.avi` reference videos (V2V, transfer controls) uses OpenCV — the same optional decoder as the multimodal video path. It is **not** bundled with TensorRT-LLM — install it yourself: `pip install opencv-python-headless`. Frame directories work without it.
- Transfer's `edge`/`blur` auto-computation also uses OpenCV (same `opencv-python-headless`).

## Deployment configs

See `examples/visual_gen/configs/`:

- `cosmos3-nano-1gpu.yaml` — 1 GPU
- `cosmos3-super-4gpu.yaml` — 4 GPU, CFG + Ulysses + parallel VAE

Example prompts live under `prompts/` (mirroring `cosmos3-internal/inputs/omni`).

## Usage

```bash
# T2V: text-to-video
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2v.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# I2V/TI2V: image-conditioned video (vision_path is read from the prompt file;
# local path, file://, http(s):// URL, or data: URI are all accepted)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/i2v.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# I2V with an explicit conditioning image (overrides the prompt file)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/i2v.json \
    --image_path https://example.com/frame.jpg \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# V2V: video-conditioned video (continues the first frames of --video_path).
# Best results when the prompt describes the input video — e.g. continue a
# T2V output reusing its original prompt. Output size is fixed (1280x720
# default); inputs are center-cropped, not aspect-matched.
python cosmos3.py --model /path/to/Cosmos3-Nano \
    --prompt_file prompts/v2v.json \
    --video_path /path/to/Cosmos3-Nano/assets/example_i2v_output.mp4 \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# T2AV: text-to-video with synchronized audio
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2av.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# T2I: text-to-image
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2i.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
    --output_path output.png

# Transfer: control-video conditioning — structure from the control video,
# appearance from the prompt. edge/blur are computed from --video_path.
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "The same scene rendered as a photorealistic video, sharp detail." \
    --video_path /path/to/reference.mp4 \
    --extra_params '{"edge": true}' \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# Transfer with a fully synthetic control (no assets): generate an edge-map
# video of a bouncing ball, then let the prompt paint it photoreal.
# Keep synthetic controls edge-style: the blur hint expects the low
# frequencies of natural video, and flat synthetic color fields degrade
# generation. Temporal exposure swings (e.g. pulsing global light) do not
# transfer — express lighting spatially or in the prompt instead.
python generate_bouncing_ball_control.py --out_dir ./ball_control
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "A photorealistic beach ball with colorful panels bouncing between the walls of an enclosed room, studio lighting." \
    --extra_params '{"edge": {"control_path": "./ball_control/control.mp4"}}' \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# Multi-hint transfer: edge pins the layout, blur pins the palette/lighting.
# Hints must describe the same underlying video as each other and the prompt.
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "The same scene, ultra sharp, professional photography." \
    --video_path /path/to/reference.mp4 \
    --extra_params '{"edge": true, "blur": true}' \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# Inline prompt (--prompt or a JSON file path)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "A cute puppy playing with a ball in a park" \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml
```
