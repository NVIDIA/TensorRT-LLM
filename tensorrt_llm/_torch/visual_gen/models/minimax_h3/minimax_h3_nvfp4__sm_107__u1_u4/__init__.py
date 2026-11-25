# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 scenario ``minimax_h3_nvfp4__sm_107__u1_u4``: pre-quantized NVFP4 Linears, FP8 trtllm-gen
attention, Ulysses 1 or 4 on SM 107. Selected by ``models/minimax_h3/scenarios.py``; see README.md.

Only :mod:`.config` is imported here: the kernels, modeling and pipeline modules read the config at
import time, so the resolver applies the caller's overrides before importing them.
"""

from . import config  # noqa: F401

__all__ = ["config"]
