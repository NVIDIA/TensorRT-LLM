<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FastH3 quality references

The B200 post-merge smoke loads the released eight-step checkpoint, selects the FastH3 VSA target and generates video and audio. The separate SM100 quality tests compare each backend with a baseline capture from the same checkpoint, request and GPU architecture. Set `FASTH3_CHECKPOINT` to an immutable checkpoint snapshot or stage `FastVideo-FastH3-8-Step-V2` under `LLM_MODELS_ROOT`.

Capture fixed references using the baseline installation on SM100.

```bash
mkdir -p "$BASELINE_OUTPUT"
for backend in TRTLLM CUTEDSL; do
    "$BASELINE_PYTHON" "$CASE/quality_case.py" --checkpoint "$FASTH3_CHECKPOINT" --backend "$backend" --output "$BASELINE_OUTPUT/$backend.pt"
done
```

The input manifest records `baseline_commit`, `checkpoint_revision` and the accepted `max_lpips` and `max_audio_log_stft` under each backend name. Install the captured artifacts into the QA model store once per immutable reference release. The installer checks that the captures have matching metadata and writes their SHA-256 digests into `sm100/8-step-v2.json`.

```bash
python "$CASE/stage_references.py" --manifest "$BASELINE_OUTPUT/8-step-v2.json" --trtllm "$BASELINE_OUTPUT/TRTLLM.pt" --cute "$BASELINE_OUTPUT/CUTEDSL.pt" --output "$LLM_MODELS_ROOT/FastH3-QA-references"
```

For a separate reference store, set `FASTH3_QA_REFERENCES` to the parent of `sm100`. Run `test_quality.py::test_video_audio_against_baseline` against the candidate installation.
