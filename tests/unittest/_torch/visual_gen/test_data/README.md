<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisualGen test media fixtures

## `cosmos3_v2v_ref_9f_bframes.mp4`

9-frame 64×64 H.264-in-MP4 V2V reference fixture (2,786 bytes), encoded once
offline with ffmpeg/libx264 so decode tests exercise **cross-encoder** interop
(x264 encodes, NVDEC decodes; the H.264 spec makes decoded YUV
bit-exact for conformant decoders; the YUV->RGB conversion is this stack's
(PyNvVideoCodec), so decoded RGB is stable for this decode path). B-frames are forced — with only 9 frames x264
would otherwise skip them — and verified present (`I B B P I B B P I`).

Each frame encodes its own display index three ways, so tests recover ordering
from content alone (catching B-frame reorder bugs):
- red channel: solid ramp, `R = 20 + 25 * i`
- green channel: horizontal bar at rows `[7*i, 7*i + 7)`
- blue channel: vertical bar at columns `[7*i, 7*i + 7)`

Regeneration (exact provenance; ffmpeg 6.1.1 / libx264):

```python
import numpy as np
from PIL import Image

for i in range(9):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 20 + i * 25
    frame[7 * i : 7 * i + 7, :, 1] = 255
    frame[:, 7 * i : 7 * i + 7, 2] = 255
    Image.fromarray(frame).save(f"frame_{i:02d}.png")
```

```bash
ffmpeg -y -framerate 24 -i frame_%02d.png \
  -c:v libx264 -pix_fmt yuv420p -g 4 -bf 2 \
  -x264-params b_adapt=0:scenecut=0 \
  -movflags +faststart cosmos3_v2v_ref_9f_bframes.mp4

# verify B-frames survived:
ffprobe -v error -select_streams v:0 -show_entries frame=pict_type \
  -of csv=p=0 cosmos3_v2v_ref_9f_bframes.mp4
```

## `cosmos3_v2v_ref_9f_bframes.avi`

The **same 9 frames** as the MP4 fixture, re-muxed as H.264-in-AVI (7,842
bytes) so the second supported container is exercised through the real decode
path. Same `frame_%02d.png` source as above; only the container differs:

```bash
ffmpeg -y -framerate 24 -i frame_%02d.png \
  -c:v libx264 -pix_fmt yuv420p -g 4 -bf 2 \
  -x264-params b_adapt=0:scenecut=0 \
  cosmos3_v2v_ref_9f_bframes.avi
```
