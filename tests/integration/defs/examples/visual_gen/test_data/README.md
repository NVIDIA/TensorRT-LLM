<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisualGen integration-test input fixtures

## `cosmos3_v2v_lpips_reference.mp4`

5-frame 720p H.264/MP4 conditioning reference for the Cosmos3-Nano V2V LPIPS
gate (2,729 bytes). Deterministic content: gray (30, 30, 30) background with a
(200, 120, 40) block moving 40 px/frame. the H.264 spec makes decoded YUV bit-exact
for conformant decoders; the YUV->RGB conversion is this stack's
(PyNvVideoCodec), so decoded RGB is stable for this decode path.

Regeneration (exact provenance; ffmpeg 6.1.1 / libx264):

```python
import numpy as np
from PIL import Image

for i in range(5):
    frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    x = 100 + i * 40
    frame[200:520, x : x + 200] = (200, 120, 40)
    Image.fromarray(frame).save(f"frame_{i:02d}.png")
```

```bash
ffmpeg -y -framerate 24 -i frame_%02d.png \
  -c:v libx264 -pix_fmt yuv420p -g 4 -bf 2 \
  -x264-params b_adapt=0:scenecut=0 \
  -movflags +faststart cosmos3_v2v_lpips_reference.mp4
```
