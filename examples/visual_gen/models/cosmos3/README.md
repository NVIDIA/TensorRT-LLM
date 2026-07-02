# Cosmos3 Text(+Image)-to-Video(+Audio) generation

Cosmos3 supports four generation modes from a single checkpoint:

- **T2V** — text-to-video (`prompts/t2v.json`).
- **T2I** — text-to-image (`prompts/t2i.json`); emits a still frame (use `--output_type image` / a non-video `--output_path`).
- **I2V / TI2V** — image-conditioned video (`prompts/i2v.json`). Condition on a reference frame via the prompt file's `vision_path` or `--image_path`. The image may be a local path, a `file://` / `http(s)://` URL, or a `data:` URI.
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

# T2AV: text-to-video with synchronized audio
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2av.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# T2I: text-to-image
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt_file prompts/t2i.json \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
    --output_path output.png

# Inline prompt (--prompt or a JSON file path)
python cosmos3.py --model nvidia/Cosmos3-Nano \
    --prompt "A cute puppy playing with a ball in a park" \
    --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml
```
