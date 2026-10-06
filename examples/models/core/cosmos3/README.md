<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Cosmos3 Reasoner

The Cosmos3 Reasoner accepts text, image and video inputs and produces text
through `/v1/chat/completions`. It uses the LLM runtime, not VisualGen.
For image and video generation, see the
[Cosmos3 generator serving examples](../../../visual_gen/serve/README.md#cosmos3-t2v--i2v--v2v--transfer--action--t2av--t2i).

## Serving

A Cosmos3 checkpoint holds both the Reasoner and the Generator. Omit
`--visual_gen_args` and `--enable_visual_gen` to serve the Reasoner:

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

## Static FP8 Reasoner

Nano and Super static-FP8 checkpoints also support the standalone Reasoner:
text, image and video inputs produce text through `/v1/chat/completions`.
Use a local FP8 checkpoint directory containing a root `hf_quant_config.json`
with the ModelOpt FP8 quantization configuration, alongside the weights and
their calibrated scales.

Start the server in terminal 1. Set `MODEL_DIR` to either your Nano or Super
FP8 checkpoint directory:

```bash
MODEL_DIR=/path/to/Cosmos3-Nano-FP8
trtllm-serve "$MODEL_DIR" --host 127.0.0.1 --port 8000 \
    --max_num_tokens 32768
```

Once `curl -f http://127.0.0.1:8000/health` returns HTTP 200, run the client
in terminal 2 (`pip install openai` if needed). Replace the image and video
paths with files accessible to the server; `file://` URLs refer to the
server's filesystem, including its container mounts when applicable:

```python
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
model = client.models.list().data[0].id

# Text understanding
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Describe what a robot arm does."}],
    max_tokens=4096,
)
print(response.choices[0].message.content)

# Image understanding
image_url = Path("/path/to/image.jpg").resolve().as_uri()
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "Caption the image in detail."},
    ]}],
    max_tokens=4096,
)
print(response.choices[0].message.content)

# Video understanding: decode at 4 FPS without a second sampling pass
video_url = Path("/path/to/video.mp4").resolve().as_uri()
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": video_url}},
        {"type": "text", "text": "Describe the video in detail."},
    ]}],
    max_tokens=4096,
    extra_body={
        "media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}},
        "mm_processor_kwargs": {"do_sample_frames": False},
    },
)
print(response.choices[0].message.content)
```
