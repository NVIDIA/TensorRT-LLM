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

1. **Install modules**: Install required dependencies before running examples:

   ```bash
   pip install git+https://github.com/huggingface/diffusers.git
   pip install av
   ```

2. **Server Running**: The TensorRT-LLM visual generation server must be running
   ```bash
   trtllm-serve <path to your model> --extra_visual_gen_options <path to config yaml>
   ```

   e.g.

   ```bash
   trtllm-serve $LLM_MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers --extra_visual_gen_options ./configs/wan.yml

   # Run server on background:
   trtllm-serve $LLM_MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers --extra_visual_gen_options ./configs/wan.yml > /tmp/serve.log 2>&1 &

   ## Check if the server is setup
   tail -f /tmp/serve.log

   ```

## Examples

Current supported & tested models:

1. WAN T2V/I2V for video generation (t2v, ti2v, delete_video)

### 1. Synchronous Image Generation (`sync_t2i.py`)

Demonstrates synchronous text-to-image generation using the OpenAI SDK.

**Features:**
- Generates images from text prompts
- Supports configurable image size and quality
- Returns base64-encoded images or URLs
- Saves generated images to disk

**Usage:**
```bash
# Use default localhost server
python sync_image_gen.py

# Specify custom server URL
python sync_image_gen.py http://your-server:8000/v1
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
```

**Command-Line Arguments:**
- `--mode` - Generation mode: `t2v` or `ti2v` (default: t2v)
- `--prompt` - Text prompt for video generation (required)
- `--image` - Path to reference image (required for ti2v mode)
- `--base-url` - API server URL (default: http://localhost:8000/v1)
- `--model` - Model name (default: wan)
- `--duration` - Video duration in seconds (default: 4.0)
- `--fps` - Frames per second (default: 24)
- `--size` - Video resolution in WxH format (default: 256x256)
- `--output` - Output video file path (default: output_sync.mp4)

**API Endpoint:** `POST /v1/videos/generations`

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
```

**Command-Line Arguments:**
- `--mode` - Generation mode: `t2v` or `ti2v` (default: t2v)
- `--prompt` - Text prompt for video generation (required)
- `--image` - Path to reference image (required for ti2v mode)
- `--base-url` - API server URL (default: http://localhost:8000/v1)
- `--model` - Model name (default: wan)
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
- `model`: Model identifier (e.g., "wan")
- `prompt`: Text description
- `n`: Number of images to generate
- `size`: Image dimensions (e.g., "512x512", "1024x1024")
- `quality`: "standard" or "hd"
- `response_format`: "b64_json" or "url"

### Video Generation
- `model`: Model identifier (e.g., "wan")
- `prompt`: Text description
- `size`: Video resolution (e.g., "256x256", "512x512")
- `seconds`: Duration in seconds
- `fps`: Frames per second
- `input_reference`: Reference image file (for TI2V mode)

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

### Text+Image-to-Video (Multipart with File Upload)
```bash
curl -X POST "http://localhost:8000/v1/videos" \
  -F "prompt=She turns around and smiles" \
  -F "input_reference=@./media/woman_skyline_original_720p.jpeg" \
  -F "seconds=4.0" \
  -F "fps=24" \
  -F "size=256x256" \
  -F "guidance_scale=5.0"
```

### Check Video Status
```bash
curl -X GET "http://localhost:8000/v1/videos/{video_id}"
```

### Download Video
```bash
curl -X GET "http://localhost:8000/v1/videos/{video_id}/content" -o output.mp4
```

### Delete Video
```bash
curl -X DELETE "http://localhost:8000/v1/videos/{video_id}"
```

## API Endpoints Summary

| Endpoint | Method | Mode | Content-Type | Purpose |
|----------|--------|------|--------------|---------|
| `/v1/videos` | POST | Async | JSON or Multipart | Create video job (T2V/TI2V) |
| `/v1/videos/generations` | POST | Sync | JSON or Multipart | Generate video sync (T2V/TI2V) |
| `/v1/videos/{id}` | GET | - | - | Get video status/metadata |
| `/v1/videos/{id}/content` | GET | - | - | Download video file |
| `/v1/videos/{id}` | DELETE | - | - | Delete video |
| `/v1/videos` | GET | - | - | List all videos |
| `/v1/images/generations` | POST | - | JSON | Generate images (T2I) |

**Note:** Both `/v1/videos` (async) and `/v1/videos/generations` (sync) support:
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
- `output_sync.mp4` - Synchronous video generation (`sync_video_gen.py`)
- `output_async.mp4` - Asynchronous video generation (`async_video_gen.py`)
- `output_multipart.mp4` - Multipart example output (`multipart_example.py`)

**Note:** You can customize output filenames using the `--output` parameter in all scripts.
