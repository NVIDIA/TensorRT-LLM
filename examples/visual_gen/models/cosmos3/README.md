# Cosmos3 Text(+Image)-to-Video(+Audio) generation

Cosmos3 supports four generation modes from a single checkpoint:

- **T2V** — text-to-video (`prompts/t2v.json`).
- **T2I** — text-to-image (`prompts/t2i.json`); emits a still frame (use `--output_type image` / a non-video `--output_path`).
- **I2V / TI2V** — image-conditioned video (`prompts/i2v.json`). Condition on a reference frame via the prompt file's `vision_path` or `--image_path`. The image may be a local path, a `file://` / `http(s)://` URL, or a `data:` URI.
- **T2AV** — text-to-video with synchronized audio (`prompts/t2av.json` with `enable_audio: true`, or pass `--enable_audio`). Combine with a `vision_path` for image-conditioned audio-video (TI2AV). Audio requires an audio-capable checkpoint (Nano / Super); Cosmos3-Edge has no audio tower.

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

## Deployment configs

See `examples/visual_gen/configs/`:

- `cosmos3-nano-1gpu.yaml` — 1 GPU
- `cosmos3-super-4gpu.yaml` — 4 GPU, CFG + Ulysses + parallel VAE
- `cosmos3-t2i-1gpu.yaml` — 1 GPU, text-to-image deployments (base or distilled): warms the deployed 1024×1024 single-frame shape instead of the omni video shape.

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

# T2AV: text-to-video with synchronized audio
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2av.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# T2I: text-to-image
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2i.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
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

# Cosmos3-Edge image-to-video (480p-native defaults: 832x480 x 121 frames)
python cosmos3.py --model nvidia/Cosmos3-Edge \
    --prompt "The camera slowly pans right across the scene" \
    --image_path https://example.com/frame.jpg \
    --output_path output.mp4

# Inline prompt (--prompt or a JSON file path)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "A cute puppy playing with a ball in a park" \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml
```
