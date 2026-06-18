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
python models/qwen_image.py

# With engine config (quant, parallelism, etc.)
python models/wan_t2v.py --visual_gen_args configs/wan2.2-t2v-fp4-1gpu.yaml
python models/wan_i2v.py --visual_gen_args configs/wan2.2-i2v-fp4-1gpu.yaml --image /path/to/image.png
python models/ltx2.py --visual_gen_args configs/ltx2-t2v-fp8-1-gpu.yaml
python models/flux1.py --visual_gen_args configs/flux1-dev-fp4-1gpu.yaml
python models/flux2.py --visual_gen_args configs/flux2-dev-fp4-1gpu.yaml

# Qwen-Image NVFP4: point --model at a ModelOpt-quantized checkpoint; the NVFP4
# config is read from the checkpoint. (Use a BF16 checkpoint for the baseline.)
python models/qwen_image.py --model <qwen-image-nvfp4> --visual_gen_args configs/qwen-image-fp4-1gpu.yaml
```

Install deps from the repo root: `pip install -r requirements-dev.txt`.

Output: `.png` for image models; `.mp4` for video models when FFmpeg is installed (otherwise `.avi`).
