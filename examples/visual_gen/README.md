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
# Defaults
python quickstart_example.py
python models/wan_t2v.py
python models/ltx2.py
python models/flux1.py
python models/flux2.py
python models/cosmos3_ti2v.py --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py

# With engine config (quant, parallelism, etc.)
python models/wan_t2v.py --visual_gen_args configs/wan2.2-t2v-fp4-1gpu.yaml
python models/wan_i2v.py --visual_gen_args configs/wan2.2-i2v-fp4-1gpu.yaml --image /path/to/image.png
python models/ltx2.py --visual_gen_args configs/ltx2-t2v-fp8-1-gpu.yaml
python models/flux1.py --visual_gen_args configs/flux1-dev-fp4-1gpu.yaml
python models/flux2.py --visual_gen_args configs/flux2-dev-fp4-1gpu.yaml
python models/cosmos3_ti2v.py --visual_gen_args configs/cosmos3-nano-1gpu.yaml --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py --visual_gen_args configs/qwen-image-fp8-1gpu.yaml
python models/qwen_image.py \
  --model Qwen/Qwen-Image-Layered \
  --visual_gen_args configs/qwen-image-layered-fp8-blockscale-edge-bf16-sage-fp8-1gpu.yaml \
  --image /path/to/input.png \
  --prompt ""
```

Install deps from the repo root: `pip install -r requirements-dev.txt`.

Output: `.png` for image models; `.mp4` for video models when FFmpeg is installed (otherwise
`.avi`). Qwen-Image-Layered saves per-layer `.png` files plus a white-background composite.
