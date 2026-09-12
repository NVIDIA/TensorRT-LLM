# Visual Generation Examples

See [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
for feature details.

## Layout

| Path | Purpose |
|---|---|
| [`quickstart_example.py`](quickstart_example.py) | Minimal VisualGen API example |
| [`models/`](models/) | Per-model example scripts |
| [`configs/`](configs/) | Curated showcase `VisualGenArgs` recipes — multi-GPU / deployment (used by `--visual_gen_args` and `trtllm-serve`) |
| [`serve/`](serve/) | `trtllm-serve` usage, benchmarking, and clients |

## Usage

Each script in [`models/`](models/) runs one model end-to-end on defaults.
[`configs/`](configs/) holds a small set of curated showcase recipes — multi-GPU
parallelism layouts and full deployment configs — passed with `--visual_gen_args`.
Using Wan 2.2 text-to-video as the worked example:

```bash
# Model defaults (single GPU)
python models/wan_t2v.py

# With a curated multi-GPU recipe (4-GPU NVFP4)
torchrun --nproc_per_node=4 models/wan_t2v.py \
    --visual_gen_args configs/wan2.2-t2v-fp4-4gpu.yaml
```

The same shape applies to every script in [`models/`](models/); run one with
`--help` for model-specific inputs (e.g. `--image`, `--prompt`). Some models
document mode-specific usage in their own directory (e.g.
[`models/cosmos3/`](models/cosmos3/)). A minimal API-level example lives in
[`quickstart_example.py`](quickstart_example.py).

Install deps from the repo root: `pip install -r requirements-dev.txt`.

Output: `.png` for image models; `.mp4` for video models when FFmpeg is installed (otherwise `.avi`).
