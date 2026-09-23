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

### MiniMax-H3 reference-to-video-and-audio

Ref2VA loads `transformer_ref/`; FL2VA and text-only generation use `transformer/`.
Use the converted checkpoint containing both partitions and the shared encoders/VAEs:

```bash
python models/minimax_h3.py --model <approved-checkpoint> \
  --visual_gen_args configs/minimax-h3-ref2va-bf16-1gpu.yaml \
  --image_reference subject.png --audio_reference voice.wav \
  --prompt 'The person in <Picture 1> speaks with the voice in <Audio 1>.' \
  --output_path ref2va.mp4
```

Repeat `--image_reference`, `--video_reference`, or `--audio_reference` for multiple
references (up to 9 images, 3 videos, 3 audio clips, 12 total). Audio alone is not
supported. Video references include their soundtrack when present. Media files
retain their frame/sample rates; PyAV decodes video/audio, and torchaudio is needed
only when audio must be resampled to 32 kHz. References default to images, then
videos, then audio, preserving each list's order. Set `--reference_order video:0
image:0` to change cross-modality order; include every reference exactly once.

The Python API uses existing `MediaRef` slots with role `reference`,
`VisualGenArgs.pipeline_config={"workflow": "ref2va"}`, and optional
`VisualGenParams.extra_params={"reference_order": ["video:0", "image:0"]}`.
The BF16 Ref2VA example disables `torch_compile_config.enable` to preserve
the reference rounding boundaries; compilation can introduce numerical drift.
Ref2VA references condition content rather than anchoring the first/last frame;
the default output canvas is 768×1344 regardless of reference dimensions.
