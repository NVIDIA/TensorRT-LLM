# Visual Generation Examples

See [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
for feature details.

## Layout

| Path | Purpose |
|---|---|
| [`quickstart_example.py`](quickstart_example.py) | Minimal VisualGen API example |
| [`models/`](models/) | Per-model example scripts |
| [`configs/`](configs/) | Shared `VisualGenArgs` YAMLs (used by `--visual_gen_args` and `trtllm-serve`) |
| [`serve/`](serve/) | `trtllm-serve` usage, benchmarking, and clients |

## Usage

```bash
python quickstart_example.py
python models/glm_image.py

# With engine config (quant, parallelism, etc.)
# Omit --visual_gen_args and its YAML path to use default settings.
python models/wan_t2v.py --visual_gen_args configs/wan2.2-t2v-fp4-1gpu.yaml
python models/wan_i2v.py --visual_gen_args configs/wan2.2-i2v-fp4-1gpu.yaml --image /path/to/image.png
python models/ltx2.py --visual_gen_args configs/ltx2-1gpu.yaml
# FP8 blockwise config: configs/minimax-h3-fp8-blockwise-1gpu.yaml.
# 8-GPU Ulysses config: configs/minimax-h3-bf16-8gpu.yaml.
python models/minimax_h3.py --model <approved-checkpoint> --visual_gen_args configs/minimax-h3-bf16-1gpu.yaml
python models/flux1.py --visual_gen_args configs/flux1-dev-fp4-1gpu.yaml
python models/flux2.py --visual_gen_args configs/flux2-dev-fp4-1gpu.yaml
python models/cosmos3_ti2v.py --visual_gen_args configs/cosmos3-nano-1gpu.yaml --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py --visual_gen_args configs/qwen-image-fp8-1gpu.yaml
python models/qwen_image_layered.py --visual_gen_args configs/qwen-image-layered-1gpu.yaml --image /path/to/image.png
python models/qwen_image_edit.py --visual_gen_args configs/qwen-image-edit-2511-fp4-1gpu.yaml --image /path/to/source.png --prompt "Make the image look like a watercolor painting"
python models/hunyuan_t2v.py --visual_gen_args configs/hunyuan-t2v-fp8-1gpu.yaml
```

See the [MiniMax-H3 notes](../../docs/source/models/visual-generation.md#minimax-h3-notes)
for supported tasks, the TRTLLM attention restriction, and checkpoint licensing.

Install deps from the repo root: `pip install -r requirements-dev.txt`.

Output: `.png` for image models; `.mp4` for video models when FFmpeg is installed (otherwise `.avi`).

### MiniMax-H3 tiled VAE

MiniMax-H3 uses the checkpoint's spatial tiling by default. Its video decoder
can distribute independent tiles across the existing Ulysses group; the
eight-GPU BF16 configuration enables this. Tile geometry, overlap blending,
and temporal chunking match the sequential tiled decoder. Small canvases
with fewer tiles than ranks use sequential decoding on each rank.

Use `VisualGen.pipeline_config("MiniMaxAI/MiniMax-H3")` to discover the knobs.
They can be set through `VisualGenArgs.pipeline_config` or the YAML
`pipeline_config` mapping:

```yaml
pipeline_config:
  vae_use_tiling: true
  vae_tile_parallel: true
  vae_tile_size: 256
  vae_tile_overlap: 64
```

Tile size and overlap are positive pixel counts aligned to the checkpoint's
spatial compression ratio, with overlap smaller than tile size. Setting
`vae_tile_parallel: false` retains sequential tiling. Setting both booleans
to `false` disables spatial tiling for encode and decode, which changes the
checkpoint's reference output and can increase memory and latency. Audio
VAE processing is unaffected. Tile parallelism requires all ranks to decode
the same video latents and uses the supported pure Ulysses configuration.
