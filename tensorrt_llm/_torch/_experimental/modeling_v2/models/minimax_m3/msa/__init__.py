# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 MSA sparse attention code shared by the family's targets.

* ``msa_backend``, ``msa_indexer``, ``msa_prefill``, ``msa_decode``: the family's MSA attention backend, its indexer
  and its context / generation FMHA libraries. Each class derives from main's MiniMax-M3 MSA class of the same role
  and runs the family's paths (vendored ``kernels/msa_kernels/fmha_sm100``, lazy proxy and dense plans, NVFP4 readers,
  prefill scorer hook, split prefill workspace, k2q CSR schedule).
* ``cache_manager``: main's MiniMax-M3 KV cache manager plus the NVFP4 index-K cache.
* ``engine``: how the targets select these classes (``RuntimeHooks.sparse_attention``).
* ``prewarm``: compile every MSA kernel variant serving can select during the decoder warmup.
* ``kernels``: helpers of the above and the index-K producers the targets call.
"""
