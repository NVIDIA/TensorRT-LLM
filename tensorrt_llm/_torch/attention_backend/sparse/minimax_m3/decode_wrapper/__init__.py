# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-safe decode planner for MiniMax-M3 sparse attention.

Drives the external MSA (`fmha_sm100`) SM100 kernels with launch
arguments assembled from device tensors, so decode steps can be captured
into CUDA graphs and replayed correctly as sequence state advances.

Public surface:

* `M3DecodeGeometry`: compile/alloc-time constants for one layer family.
* `M3DecodePlanner`: per-attention-metadata-instance planner + launcher
  (proxy MQA max-score pass, top-k block selection, block-sparse GQA).
"""

from .dispatch import M3DecodeGeometry, M3DecodePlanner

__all__ = [
    "M3DecodeGeometry",
    "M3DecodePlanner",
]
