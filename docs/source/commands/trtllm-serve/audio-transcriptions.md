<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Whisper audio transcriptions

When serving a supported Whisper checkpoint, `trtllm-serve` accepts short audio
uploads at `POST /v1/audio/transcriptions`.

Use the encoder-decoder runtime settings from the
[Whisper guide](../../models/encoder-decoder.md#transcribe-audio-with-whisper).
For example, save the following as `whisper.yaml`:

```yaml
attn_backend: TRTLLM
dtype: float16
max_batch_size: 4
max_input_len: 1500
max_seq_len: 2048
max_num_tokens: 3000
enable_chunked_prefill: false
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
  cross_kv_cache_fraction: 0.5
scheduler_config:
  use_python_scheduler: true
```

`max_input_len` must cover Whisper's 1500 encoder positions even for a short
recording. This endpoint does not change engine or cross-KV cache sizing.

```bash
trtllm-serve openai/whisper-large-v3 --backend pytorch --config whisper.yaml --port 8000
```

Submit a multipart request using the model identifier reported by `/v1/models`:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@short.wav \
  -F language=en \
  -F response_format=json
```

The JSON response is `{"text":"..."}`. Set `response_format=text` for a plain
text response. OpenAI-compatible clients can use the same endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with open("short.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=audio,
        language="en",
    )
print(result.text)
```

Uploads are limited to 25 MiB and to the checkpoint's input window (normally
30 seconds). Audio is decoded with SoundFile, downmixed to mono, and resampled
to the checkpoint's sampling rate. WAV and FLAC are supported; other formats
depend on the installed libsndfile codecs. Malformed, empty, nonfinite, or
overlong audio is rejected. Split longer recordings on the client side.

`language` selects the language prefix on multilingual checkpoints. Omitting it
uses the checkpoint's current English-transcription default; automatic language
detection is not implemented. English-only checkpoints accept `language=en`.
`temperature` accepts values from 0 to 1 and defaults to greedy decoding (0).
There is no temperature fallback/retry schedule.

This endpoint currently supports JSON/text output, one clip per request, and
non-streaming transcription. Nonempty `prompt`, `stream=true`, timestamp
granularities, other response formats, and unknown form fields are rejected
explicitly. Translation, timestamps, long-form windowing, and hotword biasing
are not implemented. The endpoint requires a Whisper model and does not route
through the disaggregated-serving proxy.
