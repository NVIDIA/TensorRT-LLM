# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-safe decode wrapper for MiniMax-M3 sparse attention.

Drives the external MSA (`fmha_sm100`) SM100 kernels with launch
arguments assembled from device tensors, so decode steps can be captured
into CUDA graphs and replayed correctly as sequence state advances.

Public surface:

* `M3DecodeGeometry`: compile/alloc-time geometry key for one layer family.
* `M3DecodeState`: persistent decode buffers + compiled kernels, owned by
  the attention metadata (no separate driver object).
* `build_m3_decode_state` / `resolve_decode_state`: build the state (in the
  metadata's `prepare()`) or return the one the metadata already owns.
* `decode_proxy_max_score` / `decode_select_blocks` / `decode_sparse_attention`:
  the proxy MQA, top-k block selection, and block-sparse GQA launches.
"""

from .dispatch import (
    M3DecodeGeometry,
    M3DecodeState,
    build_m3_decode_state,
    decode_proxy_max_score,
    decode_select_blocks,
    decode_sparse_attention,
    resolve_decode_state,
)

__all__ = [
    "M3DecodeGeometry",
    "M3DecodeState",
    "build_m3_decode_state",
    "decode_proxy_max_score",
    "decode_select_blocks",
    "decode_sparse_attention",
    "resolve_decode_state",
]
