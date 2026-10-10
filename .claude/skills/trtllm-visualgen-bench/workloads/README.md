# Per-model request workloads

One file per distinct model/mode workload shape, keyed by **HF model id** — not by pipeline class. Several workloads may share an id (the three Cosmos3-Super mode splits), and one workload may cover several ids (Wan 2.2-T2V bf16/FP8/NVFP4). `WanPipeline` alone resolves to several default sets through `get_wan_default_params()`'s version → size → name cascade, so keying by class would collapse Wan 2.2-TI2V-5B (704x1280, 121 frames, 50 steps, guidance 5.0, 24 fps) onto Wan 2.2-A14B (720x1280, 81, 40, 4.0, 16 fps).

## The rule: pin what the workload chooses

**Pin everything that changes executed work.** A workload writes out each field that determines how much compute the run does — even when it equals the pipeline default. The workload is the workload's identity; if a value can change under it without the file changing, two runs of "the same workload" are not comparable.

| pinned | why |
|---|---|
| `width` `height` `num_frames` `frame_rate` `num_inference_steps` | the shape, which sets the executed work directly. |
| `guidance_scale` | **gates CFG**: `do_cfg = guidance_scale > 1.0` (`pipeline.py` `do_cfg`). Non-parallel CFG is **2 transformer calls per step** instead of 1. A default drifting across 1.0 silently doubles the work. |
| `max_sequence_length` | the tokenizer runs `padding="max_length"` in `pipeline_wan.py`, so this is fixed text-encoder work independent of prompt length, and it sets downstream attention dimensions. |
| `seed` | `params.py` default is `None` — the engine draws one and the run is not reproducible. |
| cost-bearing `extra_params` | QwenLayered `layers` (output images, or one tiled grid under `save_layers_to_grid`); Cosmos3 `output_type` (selects the entire mode table) and `enable_audio` (runs the audio tower). |

Left out: `negative_prompt` (except Qwen-Edit's `" "`, which participates in enabling the CFG branch) and every `extra_params` knob that only moves quality.

A header comment is not a substitute for a pinned field: nothing checks a comment against the values below it, so it drifts. Neither is `--save-detailed`, which is optional and which default output omits.

**One exception:** distilled Cosmos3 (`-4Step`) leaves `num_inference_steps` unset — `validate_request()` accepts only the checkpoint-baked `len(fixed_sigmas)`, which is unknowable statically. `guidance_scale` is pinned to `1.0` (`DISTILLED_GUIDANCE_SCALE`), which the validator does accept.

## The values

Every pinned value is the pipeline's own `default_generation_params`, with two deltas:

| Delta | Pipeline default |
|---|---|
| LTX-2 `1280x768` | `768x512`, a smoke size. |
| `seed: 42` | `None` |

## Cosmos3: one workload per checkpoint, not per family

Cosmos3 resolves `COSMOS3_GENERATION_DEFAULTS[(family, mode)]` — a 2-D table no other pipeline has. That table decides **where the defaults come from**; it does not decide what a workload is. Cosmos3-Nano and Cosmos3-Super share the `(qwen3, video)` bucket and therefore the same shape, but they are **33G and 126G** checkpoints — roughly 4x apart. One workload for both would label two very different workloads as the same benchmark, so they are split by checkpoint.

Four ship: `cosmos3-nano-t2i` is the cheap single-GPU smoke test, `cosmos3-super-t2v` the heavy text-to-video baseline, `cosmos3-super-i2v` the same shape conditioned on an image, and `cosmos3-super-t2av` the same shape again with `extra_params.enable_audio: true`, so the three Super files price conditioning and the audio tower against one baseline. That tower ships in Nano and Super; Edge declares `sound_gen: false`.

Workload names use the **checkpoint**, never the internal `backbone_type`: a name like `cosmos3-qwen3-*` puts an implementation detail in the filename and collides visually with the unrelated `qwen-image*.yaml` beside it.

## Naming: the mode goes in the filename

`<checkpoint>-<mode>.yaml`, and the mode is written even when a checkpoint has only one workload today. An unsuffixed file reads as "the whole checkpoint" when it is one mode of several, and the checkpoint name does not imply the mode — TI2V-5B does t2v and i2v, LTX-2 does both, FLUX.2-dev does t2i and reference edits.

Where a checkpoint has more than one mode, the extra workloads hold the shape of the text-only one and differ by the reference alone, so the pair prices what conditioning costs. A reference mode is a second measurement, which is why a bare model name is answered with the list.

`extra_params.output_type` selects the bucket but only ever restates `backend` — `openai-images` implies `image`, `openai-videos` implies `video` — and the loader rejects a disagreement. It stays explicit because `output_type` is a per-pipeline extra param: LTX-2 declares the same name for `pt`/`pil`, a tensor format carrying no modality.

## Layout

```yaml
backend: openai-videos          # or openai-images / openai-image-edits

common_params:                  # shared by every request; a request may override
  width: 1280
  height: 720
  seed: 42

requests:
  - prompt: "..."
    image_reference:
      content: ../media/cat_piano.png
      format: path
```

Every field, the merge order, and the reference and prompt-file forms are defined in
[`tensorrt_llm/serve/scripts/BENCHMARKING_VISUAL_GEN.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/BENCHMARKING_VISUAL_GEN.md)
in the TensorRT-LLM repository, next to the loader that reads them.

**Give a reference `format: path` wherever the server can read the file.** A `base64`
reference travels in the request body, so its transfer is inside `e2e_latency`. A `path` keeps it
off the wire entirely.

The server opens that path itself, so the run directory has to sit on storage the server reads at
the same path — one filesystem, or a share both sides mount alike. A client-only path reaches the
server as a name it cannot open.

## Checking a workload against a loaded model

```python
from tensorrt_llm import VisualGen
vg = VisualGen(model="<hf id>")
print(vg.default_params)      # universal fields + all extra_params defaults
print(vg.extra_param_specs)   # valid extra_params keys, types, ranges
```

`extra_param_specs` is the authority for which `extra_params` keys a pipeline accepts. Use it before adding a key — the surface ranges from zero (Flux2, QwenImage) to 34 (Cosmos3).

`tests/test_workloads.py` runs over every file here: every path a workload names — `prompt_file`, a reference, `extra_params.action_file` — must resolve to the file itself rather than a git-lfs pointer, and each must yield a warmup config from `--warmup-from`. No hook or CI job runs it: run it after changing a workload, passing paths to check only those files.

## The prompt files

A workload carries `prompt` inline or names a `prompt_file`; setting both is an error. Cosmos3 ships
one prompt per mode, and its workloads point at those, so the text has one source.

| file | copied from | md5 |
|---|---|---|
| `cosmos3-t2i.json` | `examples/visual_gen/models/cosmos3/prompts/t2i.json` | `dead38cba06272ba9ff3a057b6bd5b39` |
| `cosmos3-t2v.json` | `examples/visual_gen/models/cosmos3/prompts/t2v.json` | `0f499c5094f12b674ff581b44affc628` |
| `cosmos3-t2av.json` | `examples/visual_gen/models/cosmos3/prompts/t2av.json` | `71cac763fb4c0b255e18383e02d20b2a` |
| `cosmos3-i2v.json` | `examples/visual_gen/models/cosmos3/prompts/i2v.json` | `51229cdeace59526b759b37b10e1db23` |

The loader reads three shapes, the ones `cosmos3.py` `load_prompt_file` accepts: a JSON object with a `prompt`
field yields that field; an object without one is a structured caption and goes out serialized; and
anything that is not JSON is plain text. The other keys such a file may carry — `vision_path`,
`model_mode`, `enable_audio` — are ignored here. A reference is the workload's own `image_reference`
and audio is the workload's own `extra_params.enable_audio`, so what a run sends is visible in the
workload rather than reached through a second file.

## The reference media

Everything under `../media/` is vendored so a workload runs with nothing but this directory. In a TensorRT-LLM checkout, `landscape.png` and `image_with_red_background.png` are git-LFS objects, being over that repository's 500 KB file limit: a clone without `git lfs pull` holds small pointer files in their place, which the server cannot decode and `tests/test_workloads.py` reports. The originals sit in three places — this repository, a checkpoint, and the `NVIDIA/cosmos` cookbook — each with the md5 to check the copy against:

| file | from | where that lives | md5 |
|---|---|---|---|
| `cat_piano.png` | `examples/visual_gen/cat_piano.png` | the repo | `0882b6f0f2bbe80023ecbb8206b35235` |
| `robot_153.jpg` | the `vision_path` in `examples/visual_gen/models/cosmos3/prompts/i2v.json` | a URL the repo names | `ac58d0298649ffc5969e9f4427fac88a` |
| `landscape.png` | the Qwen-Image-Edit demo image | that model's PR | `70f8024fd06ce9f627f7928fae9b0ac2` |
| `image_with_red_background.png` | the Qwen-Image-Layered demo image | that model's PR | `4f2bbee64349d4e006c240fe1d46294d` |

Each is the picture its own workload's prompt describes. `robot_153.jpg` is the frame `prompts/i2v.json` points at, so `cosmos3-super-i2v.yaml` carries that file's prompt and that file's image together. A model shipping no i2v example of its own takes `wan_i2v.py`'s prompt, the one written for `cat_piano.png` — LTX-2 and Wan 2.2-TI2V-5B both do. Under a pinned `max_sequence_length` the tokenizer pads to it, so the prompt text costs the same whichever one a workload carries.

### The reference's own shape

Each file is the original, at whatever size its source publishes. A reference whose aspect
differs from the workload's pinned output is resized by the pipeline on the way in: LTX-2 and
Wan resize straight to the target, so the subject is stretched, while Cosmos3 scales to cover
and centre-crops, so it is reframed. Neither changes what the run measures.
