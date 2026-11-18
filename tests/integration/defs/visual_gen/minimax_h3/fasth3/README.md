<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FastH3 quality references

Use the released FastVideo-FastH3-8-Step-V2 checkpoint for both runs. Set `FASTH3_CHECKPOINT` to the immutable model snapshot and `FASTH3_QA_REFERENCES` to the QA reference directory. Run the capture script with the baseline installation on the same GPU architecture used for candidate validation.

```bash
for backend in TRTLLM CUTEDSL; do
    "$BASELINE_PYTHON" "$CASE/quality_case.py" --checkpoint "$FASTH3_CHECKPOINT" --backend "$backend" --output "$FASTH3_QA_REFERENCES/$backend.pt"
done
```

Store the reference metadata in `8-step-v2.json`. Record the baseline Git revision under `baseline_commit` and the immutable checkpoint revision under `checkpoint_revision`. Each backend entry (`TRTLLM` and `CUTEDSL`) supplies an `artifact` filename, a `max_lpips` video threshold and a `max_audio_log_stft` audio threshold. Calibrate those limits on repeated baseline runs and have the model owner approve them before the first QA run. Retain the tensors outside the source tree. Run `test_quality.py` using the candidate installation.
