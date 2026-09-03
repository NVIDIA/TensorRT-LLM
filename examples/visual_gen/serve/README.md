# Visual Generation API Examples

This directory contains example scripts that demonstrate how to use the TensorRT-LLM Visual Generation API endpoints for image and video generation.

## Overview

These examples show how to interact with the visual generation server using both the OpenAI Python SDK and standard HTTP requests. The API provides endpoints for:

- **Image Generation**: Text-to-image generation (T2I)
- **Video Generation**:
  - Text-to-video generation (T2V) - generate videos from text prompts only
  - Text+Image-to-video generation (TI2V) - generate videos from text + reference image
  - Both synchronous and asynchronous modes supported
  - Multipart/form-data support for file uploads
- **Video Management**: Retrieving and deleting generated videos

## Prerequisites

Before running these examples, ensure you have:

1. **Install modules**: Install optional dependency:

   **Optional**: For better video compression (H.264/MP4), install [ffmpeg](https://ffmpeg.org/):
   ```bash
   # Ubuntu/Debian
   apt-get install ffmpeg
   ```
   If ffmpeg is not available, the server will use a pure Python encoder that outputs MJPEG/AVI format. See [FFmpeg download page](https://ffmpeg.org/download.html) for installation instructions on other platforms.

2. **Server Running**: The TensorRT-LLM visual generation server must be running
   ```bash
   trtllm-serve <path to your model> --visual_gen_args <path to config yaml>
   ```

   e.g.

   ```bash
   trtllm-serve $LLM_MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers --visual_gen_args ./configs/wan21.yml
   trtllm-serve $LLM_MODEL_DIR/Wan2.2-T2V-A14B-Diffusers --visual_gen_args ./configs/wan22.yml
   trtllm-serve $LLM_MODEL_DIR/FLUX.1-dev --visual_gen_args ./configs/flux1.yml
   trtllm-serve $LLM_MODEL_DIR/FLUX.2-dev --visual_gen_args ./configs/flux2.yml
   trtllm-serve $LLM_MODEL_DIR/LTX-2/ --visual_gen_args ./configs/ltx2.yml
   trtllm-serve $LLM_MODEL_DIR/Qwen-Image --visual_gen_args ./configs/qwen_image.yml

   # Cosmos3 — configs live one directory up, shared with the offline examples
   trtllm-serve nvidia/Cosmos3-Nano --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

   # Run server on background:
   trtllm-serve $LLM_MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers --visual_gen_args ./configs/wan21.yml > /tmp/serve.log 2>&1 &

   ## Check if the server is setup
   tail -f /tmp/serve.log

   ```
   For LTX-2, you need to provide a proper text_encoder_path in `./configs/ltx2.yml`.

## Examples

Current supported & tested models:

1. WAN T2V/I2V for video generation (t2v, ti2v, delete_video)
2. FLUX.1 for image generation (t2i)
3. FLUX.2 for image generation (t2i)
4. LTX-2 for video generation with audio (t2v, ti2v)
5. Qwen-Image for image generation (t2i)
6. Cosmos3 for video (t2v, i2v/ti2v, v2v), video with synchronized audio (t2av), and image (t2i) generation — see [Cosmos3](#cosmos3-t2v--i2v--v2v--t2av--t2i)

### 1. Synchronous Image Generation (`sync_image_gen.py`)

Demonstrates synchronous text-to-image generation using the OpenAI SDK. Supports FLUX.1 and FLUX.2.

**Features:**
- Generates images from text prompts
- Supports configurable model and image size
- Returns base64-encoded images or URLs
- Saves generated images to disk

**Usage:**
```bash
# FLUX.2 (default)
python sync_image_gen.py

# FLUX.1
python sync_image_gen.py --model flux1

# Custom server and prompt
python sync_image_gen.py --base-url http://your-server:8000/v1 --prompt "A sunset"
```

**API Endpoint:** `POST /v1/images/generations`

**Output:** Saves generated image to `output_generation.png` (or numbered files for multiple images)

---

### 2. Synchronous Video Generation with T2V and TI2V Modes (`sync_video_gen.py`)

Demonstrates synchronous video generation using direct HTTP requests. Waits for completion and returns the video file directly.

**Features:**
- **T2V Mode**: Generate videos from text prompts only
- **TI2V Mode**: Generate videos from text + reference image (multipart/form-data)
- Waits for video generation to complete before returning
- Returns video file directly in response
- Command-line interface for easy testing

**Usage:**

```bash
# Text-to-Video (T2V) - No reference image
python sync_video_gen.py --mode t2v \
    --prompt "A cute cat playing with a ball in the park" \
    --duration 4.0 --fps 24 --size 256x256

# Text+Image-to-Video (TI2V) - With reference image
## Note: longer duration and higher size will lead to much longer waiting time
python sync_video_gen.py --mode ti2v \
    --prompt "She turns around and smiles, then slowly walks out of the frame" \
    --image ./media/woman_skyline_original_720p.jpeg \
    --duration 4.0 --fps 24 --size 512x512

# Custom parameters
python sync_video_gen.py --mode t2v \
    --prompt "A serene sunset over the ocean" \
    --duration 5.0 --fps 30 --size 512x512 \
    --output my_video.mp4

# LTX-2: Text-to-Video (generates video with audio)
python sync_video_gen.py --mode t2v \
    --model ltx2 \
    --prompt "A cute cat playing with a ball in the park" \
    --duration 5.0 --fps 24 --size 1280x720

# LTX-2: Image-to-Video
python sync_video_gen.py --mode ti2v \
    --model ltx2 \
    --prompt "She turns around and smiles, then slowly walks out of the frame" \
    --image ./media/woman_skyline_original_720p.jpeg \
    --duration 5.0 --fps 24 --size 1280x720
```

**Command-Line Arguments:**
- `--mode` - Generation mode: `t2v` or `ti2v` (default: t2v)
- `--prompt` - Text prompt for video generation (required)
- `--image` - Path to reference image (required for ti2v mode)
- `--base-url` - API server URL (default: http://localhost:8000/v1)
- `--model` - Model name (default: wan). Use `ltx2` for LTX-2.
- `--duration` - Video duration in seconds (default: 4.0)
- `--fps` - Frames per second (default: 24)
- `--size` - Video resolution in WxH format (default: 256x256)
- `--output` - Output video file path (default: output_sync.mp4)

**API Endpoint:** `POST /v1/videos/sync`

**API Details:**
- T2V uses JSON `Content-Type: application/json`
- TI2V uses multipart/form-data `Content-Type: multipart/form-data` with file upload

**Output:** Saves generated video to specified output file

---

### 3. Async Video Generation with T2V and TI2V Modes (`async_video_gen.py`)

**NEW**: Enhanced async video generation supporting both Text-to-Video (T2V) and Text+Image-to-Video (TI2V) modes.

**Features:**
- **T2V Mode**: Generate videos from text prompts only (JSON request)
- **TI2V Mode**: Generate videos from text + reference image (multipart/form-data with file upload)
- Command-line interface for easy testing
- Automatic mode detection
- Comprehensive parameter control

**Usage:**

```bash
# Text-to-Video (T2V) - No reference image
python async_video_gen.py --mode t2v \
    --prompt "A cool cat on a motorcycle in the night" \
    --duration 4.0 --fps 24 --size 256x256

# Text+Image-to-Video (TI2V) - With reference image
python async_video_gen.py --mode ti2v \
    --prompt "She turns around and smiles, then slowly walks out of the frame" \
    --image ./media/woman_skyline_original_720p.jpeg \
    --duration 4.0 --fps 24 --size 512x512

# Custom parameters
python async_video_gen.py --mode t2v \
    --prompt "A serene sunset over the ocean" \
    --duration 5.0 --fps 30 --size 512x512 \
    --output my_video.mp4

# LTX-2: Async Text-to-Video (generates video with audio)
python async_video_gen.py --mode t2v \
    --model ltx2 \
    --prompt "A cool cat on a motorcycle in the night" \
    --duration 5.0 --fps 24 --size 1280x720

# LTX-2: Async Image-to-Video
python async_video_gen.py --mode ti2v \
    --model ltx2 \
    --prompt "She turns around and smiles, then slowly walks out of the frame" \
    --image ./media/woman_skyline_original_720p.jpeg \
    --duration 5.0 --fps 24 --size 1280x720
```

**Command-Line Arguments:**
- `--mode` - Generation mode: `t2v` or `ti2v` (default: t2v)
- `--prompt` - Text prompt for video generation (required)
- `--image` - Path to reference image (required for ti2v mode)
- `--base-url` - API server URL (default: http://localhost:8000/v1)
- `--model` - Model name (default: wan). Use `ltx2` for LTX-2.
- `--duration` - Video duration in seconds (default: 4.0)
- `--fps` - Frames per second (default: 24)
- `--size` - Video resolution in WxH format (default: 256x256)
- `--output` - Output video file path (default: output_async.mp4)

**API Details:**
- T2V uses JSON `Content-Type: application/json`
- TI2V uses multipart/form-data `Content-Type: multipart/form-data` with file upload

**Output:** Saves generated video to specified output file

---

### 4. Video Deletion (`delete_video.py`)

Demonstrates the complete lifecycle of video generation and deletion.

**Features:**
- Creates a test video generation job
- Waits for completion
- Deletes the generated video
- Verifies deletion by attempting to retrieve the deleted video
- Tests error handling for non-existent videos

**Usage:**
```bash
# Use default localhost server
python delete_video.py

# Specify custom server URL
python delete_video.py http://your-server:8000/v1
```

**API Endpoints:**
- `POST /v1/videos` - Create video job
- `GET /v1/videos/{video_id}` - Check video status
- `DELETE /v1/videos/{video_id}` - Delete video

**Test Flow:**
1. Create video generation job
2. Wait for completion
3. Delete the video
4. Verify video returns `NotFoundError`
5. Test deletion of non-existent video

---

## Cosmos3 (T2V / I2V / V2V / T2AV / T2I)

Cosmos3 serves every generation mode from a single checkpoint. Nothing selects
the mode but the request itself:

| Mode | Endpoint | What selects it |
|------|----------|-----------------|
| T2V — text-to-video | `/v1/videos/generations`, `/v1/videos` | default |
| I2V / TI2V — image-conditioned video | same | `input_reference` whose content is an image |
| V2V — video-conditioned video | same | `input_reference` whose content is a video |
| T2AV — video with synchronized audio | same | `extra_params: {"enable_audio": true}` |
| T2I — text-to-image | `/v1/images/generations` | `extra_params: {"output_type": "image"}` |
| Reasoner — chat | `/v1/chat/completions` | starting the server **without** `--visual_gen_args` |

`input_reference` is classified by content, not by filename or field — image
bytes route to I2V, video bytes to V2V. This section is the served analogue of
the offline usage block in
[`../models/cosmos3/README.md`](../models/cosmos3/README.md).

### Starting the server

Cosmos3 deployment configs live in `../configs/` (one level up from here) and
are shared with the offline examples:

```bash
# 1 GPU (Nano or Super)
trtllm-serve nvidia/Cosmos3-Nano --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

# 4 GPU / 8 GPU (Super): trtllm-serve spawns the workers itself, no torchrun/mpirun
trtllm-serve nvidia/Cosmos3-Super --visual_gen_args ../configs/cosmos3-super-4gpu.yaml
trtllm-serve nvidia/Cosmos3-Super --visual_gen_args ../configs/cosmos3-super-8gpu.yaml

# 1 GPU text-to-image deployment: warms the 1024x1024 single-frame shape
# instead of the omni video shape
trtllm-serve nvidia/Cosmos3-Super-Text2Image-4Step --visual_gen_args ../configs/cosmos3-t2i-1gpu.yaml
```

`nvidia/Cosmos3-Super-Image2Video-4Step` and `nvidia/Cosmos3-Edge` need no
config: their defaults already are the deployed shape. A local checkpoint
directory works in place of any Hub ID.

Guardrails are enabled by default and the server will not start without them —
install and authenticate per
[Guardrails](../models/cosmos3/README.md#guardrails) before serving.

### Per-mode requests

Cosmos3 supplies its own defaults for every field, so `prompt` alone is a
complete request; the examples pass the shape explicitly to show what is being
asked for. Query the loaded pipeline for the values it will use:

```python
generator = VisualGen(model="nvidia/Cosmos3-Nano")
print(generator.default_params)      # resolution, frames, fps, steps, guidance
print(generator.extra_param_specs)   # accepted extra_params keys
```

**T2V — text-to-video**

```bash
curl -X POST "http://localhost:8000/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute puppy playing with a ball in a park",
    "size": "1280x720",
    "num_frames": 189,
    "fps": 24
  }' \
  -o cosmos3_t2v.mp4
```

To reuse an offline prompt file verbatim:

```bash
curl -X POST "http://localhost:8000/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d "$(jq '{prompt: .prompt, size: "1280x720", num_frames: 189, fps: 24}' ../models/cosmos3/prompts/t2v.json)" \
  -o cosmos3_t2v.mp4
```

The bundled scripts drive this mode as-is (`seconds * fps` derives the frame
count; 7.875 x 24 = 189):

```bash
python sync_video_gen.py --mode t2v --model nvidia/Cosmos3-Nano \
    --prompt "A cute puppy playing with a ball in a park" \
    --size 1280x720 --duration 7.875 --fps 24
```

**I2V / TI2V — image-conditioned video**

```bash
curl -X POST "http://localhost:8000/v1/videos/generations" \
  -F "prompt=She turns around and smiles, then slowly walks out of the frame" \
  -F "input_reference=@./media/woman_skyline_original_720p.jpeg" \
  -F "size=1280x720" \
  -F "num_frames=189" \
  -F "fps=24" \
  -o cosmos3_i2v.mp4
```

```bash
python sync_video_gen.py --mode ti2v --model nvidia/Cosmos3-Nano \
    --prompt "She turns around and smiles, then slowly walks out of the frame" \
    --image ./media/woman_skyline_original_720p.jpeg \
    --size 1280x720 --duration 7.875 --fps 24
```

**V2V — video-conditioned video**

Upload an MP4/AVI instead of an image; see
[Video-to-Video](#video-to-video-multipart-with-file-upload-cosmos3) in the curl
quick reference for the conditioning knobs. Only the first (or last, per
`condition_video_keep`) `max(condition_video_latent_indexes) * 4 + 1` reference
frames condition the output — 5 frames with the defaults. Output size is fixed
by the request; references are center-cropped, not aspect-matched. Validated for
Nano / Super only.

**T2AV — video with synchronized audio**

```bash
curl -X POST "http://localhost:8000/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute puppy playing with a ball in a park",
    "size": "1280x720",
    "num_frames": 189,
    "fps": 24,
    "extra_params": {"enable_audio": true}
  }' \
  -o cosmos3_t2av.mp4
```

Add an image `input_reference` for image-conditioned audio-video (TI2AV). In
multipart form-data, `extra_params` is sent as a JSON-encoded string:
`-F 'extra_params={"enable_audio": true}'`.

Audio needs an audio-capable checkpoint (Nano / Super; Cosmos3-Edge has no audio
tower). On a checkpoint without one, `enable_audio` is accepted and video-only
output is returned. The audio track is muxed into the container only by the
FFmpeg encoder — the pure-Python AVI fallback drops it. Request
`"format": "safetensors"` (or `"pt"`) to get the video and audio tensors plus
`frame_rate` / `audio_sample_rate` in one payload instead.

**T2I — text-to-image**

Text-to-image goes to the image endpoint. Cosmos3 switches to its image-mode
defaults and generates a single frame:

```bash
curl -s -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A medium shot of a modern robotics research laboratory",
    "size": "1024x1024",
    "response_format": "b64_json",
    "extra_params": {"output_type": "image"}
  }' \
  | python3 -c "import base64,json,sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" \
  > cosmos3_t2i.png
```

With the default `"response_format": "url"` the response carries a
`/v1/images/{image_id}/content` URL to download instead.

**Reasoner — chat**

A Cosmos3 checkpoint holds two models: the **Reasoner** (a Qwen3-VL-based VLM)
and the **Generator** (video / image diffusion). `--visual_gen_args` selects
which one `trtllm-serve` loads — omit it to serve the Reasoner:

```bash
trtllm-serve nvidia/Cosmos3-Nano --port 8000
```

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Cosmos3-Nano",
    "messages": [{"role": "user", "content": "Describe what a robot arm does."}],
    "max_tokens": 60
  }'
```

The two are mutually exclusive: a Reasoner server returns 404 on `/v1/videos/*`
and `/v1/images/*`, and a generation server has no `/v1/chat/completions`.

### Notes

- **A guardrail block surfaces as HTTP 500** (`Video generation failed` /
  `Image generation failed`): the pipeline returns an empty result rather than a
  distinguishable error.
- **Distilled checkpoints** (`...-4Step`) fix their step count and guidance;
  simplest is to omit `num_inference_steps` and `guidance_scale`. A value that
  matches the checkpoint is accepted, a conflicting one returns HTTP 400 naming
  the value it requires.
- **Prompt quality**: the checkpoints were tuned on the structured captions the
  model cards ship, which give noticeably cleaner output than a one-line summary.
  Send one as the `prompt` string, and the offline example's
  [negative prompt](../models/cosmos3/cosmos3_negative_prompt.json) as
  `negative_prompt` — JSON-shaped prompts are re-parsed, so formatting does not
  matter.

---

## API Configuration

All examples use the following default configuration:

- **Base URL**: `http://localhost:8000/v1`
- **API Key**: `"tensorrt_llm"` (authentication token)
- **Timeout**: 300 seconds for async operations

You can customize these by:
1. Passing the base URL as a command-line argument
2. Modifying the default parameters in each script's function

## Common Parameters

### Image Generation
- `prompt`: Text description (required)
- `n`: Number of images to generate
- `size`: Image dimensions in `WxH` format (e.g., `"512x512"`, `"1024x1024"`) — or use the structured pair `width` + `height` (both required when sent)
- `seed`: Random seed; `null` / omitted means the engine draws a fresh seed
- `num_inference_steps`, `guidance_scale`, `max_sequence_length`, `negative_prompt`: per-request denoise controls (override pipeline defaults when sent)
- `extra_params`: model-specific overflow as a JSON object (see "Model-Specific `extra_params`" below). Unknown keys are rejected by the executor.
- `response_format`: `"url"` (default; HTTP URL to `/content`), `"b64_json"` (inline base64), or `"path"` (server-side on-disk path, for co-located clients)
- `format`: Generation content encoding. Image encoders: `"png"`, `"webp"`, `"jpeg"`. Tensor formats: `"safetensors"`, `"pt"`.
- Accept-and-warn OpenAI-shape fields (no engine semantic): `model`, `quality`, `style`, `user`. Sending `quality`/`style` logs a server-side WARNING; sending `model` warns on mismatch. None of these change generation behavior.

### Video Generation
- `prompt`: Text description (required)
- `size` / `width` / `height`: same convention as image
- `seconds`: Duration in seconds (engine multiplies by `frame_rate` to derive `num_frames` when the latter is absent)
- `frame_rate` (canonical) or `fps` (alias): frames per second
- `num_frames`: when set, wins over the `seconds * frame_rate` derivation
- `seed`, `num_inference_steps`, `guidance_scale`, `max_sequence_length`, `negative_prompt`: per-request denoise controls
- `image_reference`, `video_reference`, `audio_reference`: reference image(s) for I2V/TI2V, video(s) for V2V, audio(s). In JSON each takes a `{content, format, role}` object or a list of them; a multipart file upload needs no `format`.
  - `format` declares how to read `content`: `"path"` (a file readable by the server, or a `file://` URI), `"url"` (`http(s)`), or `"base64"` (or a `data:` URI). It is required in JSON, where `"bytes"` is rejected — upload the file instead.

    ```json
    {"image_reference": {"content": "iVBORw0KGgoAAAANSUhEUg...", "format": "base64"}}
    ```

  - `"path"` reads a file on the *server*, so it is only meaningful for a co-located client; set `TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1` to reject it (the same switch also disables `response_format="path"`). `"url"` is fetched through the SSRF-guarded loader (private-address block, redirect re-validation, timeout, size cap).
  - `format` here is the *input* wire form; the top-level `format` selects the *output* encoding.
  - `role` disambiguates a model that accepts the same modality in more than one role — Wan 2.1 I2V takes a first frame and an optional last frame. Roles and lists need a JSON body; a multipart upload is a single file with no role.

    ```json
    {"image_reference": [
      {"content": "<base64>", "format": "base64", "role": "first_frame"},
      {"content": "<base64>", "format": "base64", "role": "last_frame"}
    ]}
    ```

  - **Supported formats**: PNG and JPEG images; MP4 and AVI video, with H.264 the tested codec and others best-effort. HEIF/AVIF are not supported.
- `input_reference` (deprecated): a single image or video reference, routed by content signature to I2V or V2V. A JSON request carries base64 bytes and a multipart request uploads the file; it is ignored when `image_reference` / `video_reference` is also given. Prefer the typed fields.
- `extra_params`: model-specific overflow (see below)
- `response_format`: `"file"` (default; `FileResponse` byte download) or `"path"` (server-side output path JSON, for co-located clients)
- `format`: Generation content encoding. Video encoders: `"mp4"`, `"avi"`, `"auto"`. Tensor formats: `"safetensors"`, `"pt"` (carries video + audio + scalar metadata in one payload for LTX-2).

> **`response_format="path"`** (image and video) returns absolute server-side file paths under the server's media-storage directory (`TRTLLM_MEDIA_STORAGE_PATH`), for clients co-located with the server (shared filesystem). Enabled by default; set `TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1` to reject `path` requests with HTTP 400. One switch covers both directions: it also rejects a reference sent with `format="path"`.

#### Tensor-format consumer contract

When `format="safetensors"` or `format="pt"`, the payload bundles every populated media tensor (`image` / `video` / `audio`) and the scalar metadata (`frame_rate`, `audio_sample_rate`) into one file.

- **`pt`**: `torch.load(buf, weights_only=True)` returns a dict with the tensor keys and the scalars as native Python values.
- **`safetensors`**: `safetensors.torch.load(bytes)` returns a dict with the tensor keys and each scalar as a 0-d tensor under the same key — call `.item()` to unbox (e.g. `loaded["frame_rate"].item()`). The same scalars are also written to the safetensors file header as strings; `safe_open(path, framework="pt").metadata()` exposes them in that form for consumers that prefer header access.

#### Unknown-field policy

The visual-gen endpoints reject unknown top-level fields with HTTP 422 (`extra="forbid"`). Anything model-specific belongs inside `extra_params`. Sending `output_format`, top-level `guidance_rescale`, or — for video — top-level `n` returns 422 with the offending field named in the error body.

#### Model-specific `extra_params`

Use the Python API to discover accepted keys for a loaded pipeline:

```python
generator = VisualGen(model="...")
print(generator.extra_param_specs)   # {key: ExtraParamSchema(type=..., range=..., default=..., description=...)}
```

Examples:
- **LTX-2**: `stg_scale`, `stg_blocks`, `modality_scale`, `guidance_rescale`, `output_type`, ...
- **Wan 2.2 A14B**: `guidance_scale_2`, `boundary_ratio`
- **Wan 2.1 / Flux**: no model-specific `extra_params` declared
- **Cosmos3**: `condition_video_latent_indexes`, `condition_video_keep` (V2V conditioning), `flow_shift`, `use_system_prompt`, and the transfer hints `edge`/`blur`/`depth`/`seg`/`wsm` with `control_guidance`, `control_guidance_interval`, `num_video_frames_per_chunk`, ... (see below)

##### Cosmos3 transfer hints

`extra_params` is JSON, so a control clip travels as a **base64-encoded** MP4/AVI
string under `<hint>.control`; the server decodes it at the HTTP boundary. Only
`edge` and `blur` can be auto-computed — pass `true` and supply a `video`
reference for them to derive from. `depth`/`seg`/`wsm` have no generator, so
they always need a control clip.

```json
{
  "prompt": "a city street at dusk",
  "extra_params": {
    "video": "<base64 MP4/AVI>",
    "edge": {"preset_edge_threshold": "medium"},
    "blur": {"preset_blur_strength": "medium"},
    "depth": {"control": "<base64 MP4/AVI>"},
    "control_guidance": 1.5
  }
}
```

`preset_edge_threshold` and `preset_blur_strength` accept
`none`/`very_low`/`low`/`medium`/`high`/`very_high` and default to `medium`; a
bare `true` (or `"<base64>"`) is shorthand for the object form. Individual
values are validated before the job is queued, so a bad preset or an
unsupported frame count fails fast; combinations that only make sense together
— a transfer option with no hint selected, or `edge`/`blur` asked to
auto-compute with no `video` — are still reported by the worker, as a client
error, once the request is running.

> **Note:** LTX-2 generates video **with audio**. The `ltx2.yml` config must include
> `text_encoder_path` pointing to a Gemma3 model (e.g., `google/gemma-3-12b-it`).

## Quick Reference - curl Examples

### Text-to-Video (JSON)
```bash
curl -X POST "http://localhost:8000/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cool cat on a motorcycle",
    "seconds": 4.0,
    "fps": 24,
    "size": "256x256"
  }'
```

### Text-to-Video with LTX-2 (JSON, generates video with audio)
```bash
curl -X POST "http://localhost:8000/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ltx2",
    "prompt": "A cool cat on a motorcycle",
    "seconds": 5.0,
    "fps": 24,
    "size": "1280x720"
  }'
```

### Text+Image-to-Video (Multipart with File Upload)
```bash
curl -X POST "http://localhost:8000/v1/videos" \
  -F "prompt=She turns around and smiles" \
  -F "image_reference=@./media/woman_skyline_original_720p.jpeg" \
  -F "seconds=4.0" \
  -F "fps=24" \
  -F "size=256x256" \
  -F "guidance_scale=5.0"
```

### Video-to-Video (Multipart with File Upload, Cosmos3)
```bash
# Modality comes from the field name: image_reference -> I2V, video_reference -> V2V.
# V2V conditioning knobs ride in extra_params (values below are the defaults).
curl -X POST "http://localhost:8000/v1/videos" \
  -F "prompt=Continue the same scene with smooth natural motion and consistent subjects." \
  -F "video_reference=@./media/reference.mp4" \
  -F "num_frames=189" \
  -F "fps=24" \
  -F 'extra_params={"condition_video_latent_indexes": [0, 1], "condition_video_keep": "first"}'
```

### Check Video Status
```bash
curl -X GET "http://localhost:8000/v1/videos/{video_id}"
```

The async job's `status` advances `queued` → `generating` (model inference) → `postprocessing` (encode the media and/or write the output file) → `completed`. The `generating` → `postprocessing` transition marks the end of inference; poll for `completed` to download via `/content`.

### Download Video
```bash
# The server returns either MP4 (with ffmpeg) or AVI (without ffmpeg)
# Check the Content-Type header to determine the format
curl -X GET "http://localhost:8000/v1/videos/{video_id}/content" -o output.mp4

# Or use -J -O to let curl use the server-provided filename
curl -X GET "http://localhost:8000/v1/videos/{video_id}/content" -J -O
```

### Delete Video
```bash
curl -X DELETE "http://localhost:8000/v1/videos/{video_id}"
```

## API Endpoints Summary

| Endpoint | Method | Mode | Content-Type | Purpose |
|----------|--------|------|--------------|---------|
| `/v1/videos` | POST | Async | JSON or Multipart | Create video job (T2V/TI2V) |
| `/v1/videos/sync` | POST | Sync | JSON or Multipart | Generate video sync (T2V/TI2V) |
| `/v1/videos/{id}` | GET | - | - | Get video status/metadata |
| `/v1/videos/{id}/content` | GET | - | - | Download video file |
| `/v1/videos/{id}` | DELETE | - | - | Delete video |
| `/v1/videos` | GET | - | - | List all videos |
| `/v1/images/generations` | POST | - | JSON | Generate images (T2I) |

**Note:** Both `/v1/videos` (async) and `/v1/videos/sync` (sync) support:
- **JSON**: Standard text-to-video (T2V)
- **Multipart/Form-Data**: Text+image-to-video (TI2V) with file upload

## Error Handling

All examples include comprehensive error handling:

- Connection errors (server not running)
- API errors (invalid parameters, model not found)
- Timeout errors (generation taking too long)
- Resource errors (video not found for deletion)

Errors are displayed with full stack traces for debugging.

## Output Files

Generated files are saved to the current working directory:

- `output_generation.png` - Synchronous image generation (`sync_image_gen.py`)
- `output_sync.mp4` or `output_sync.avi` - Synchronous video generation (`sync_video_gen.py`)
- `output_async.mp4` or `output_async.avi` - Asynchronous video generation (`async_video_gen.py`)

**Note:** You can customize output filenames using the `--output` parameter in all scripts.

## Video Encoding

The server supports two video encoding modes:

| Encoder | Format | Requirements | Features |
|---------|--------|--------------|----------|
| **FFmpeg (H.264)** | MP4 | ffmpeg installed | Better compression, audio support |
| **Pure Python (MJPEG)** | AVI | None (built-in) | No external dependencies |

The server automatically selects the best available encoder. The example scripts detect the actual format from the server response and adjust the output filename extension accordingly.
