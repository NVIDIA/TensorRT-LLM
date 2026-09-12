# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 attention kernels and the cache plumbing they need.

Nothing here may import a sibling module of the parent package: those modules
import attention.backends.trtllm, so reaching them from an FMHA library would
close an import cycle.

Submodules are not re-exported: triton_sparse_decode pulls in Triton, and this
package sits on the import path of every attention.backends.trtllm import.
"""
