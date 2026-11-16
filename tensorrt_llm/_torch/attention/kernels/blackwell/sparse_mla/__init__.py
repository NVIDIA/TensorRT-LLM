# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA generation attention over gathered E4M3 rows for 64 query heads, CuTe DSL, sm_103.

``kernel_gb300`` is the persistent per-token kernel, launched through ``launch_gb300``; ``kernel`` holds its
inline-PTX helpers. Nothing is re-exported, so importing the package does not load CUTLASS.
"""
