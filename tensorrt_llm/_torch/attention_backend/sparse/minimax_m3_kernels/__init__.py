# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 attention kernels and the cache plumbing they need.

Split out of the sibling minimax_m3 package, whose backends import
attention_backend.trtllm and so cannot be reached from an FMHA library without
closing a cycle. Nothing here may import minimax_m3.

Submodules are not re-exported: triton_sparse_decode pulls in Triton, and this
package sits on the import path of every attention_backend.trtllm import.
"""
