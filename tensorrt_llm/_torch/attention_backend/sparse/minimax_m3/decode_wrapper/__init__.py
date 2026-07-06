# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-safe decode wrapper for MiniMax-M3 sparse attention.

Drives the external MSA (``fmha_sm100``) SM100 kernels with launch
arguments assembled from device tensors, so decode steps can be
captured into CUDA graphs and replayed correctly as sequence state
advances.  Public surface:

* :func:`proxy_mqa_decode` — proxy MQA (indexer) pass, one KV head,
  per-KV-block max-score output.
* :func:`sparse_gqa_decode` — block-sparse GQA main pass over the
  top-k selected KV blocks.
"""

from .dispatch import proxy_mqa_decode, sparse_gqa_decode

__all__ = [
    "proxy_mqa_decode",
    "sparse_gqa_decode",
]
