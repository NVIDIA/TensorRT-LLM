# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin source-copy shim for the compatible Blackwell TopK reduction."""

# <<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>
from ....blackwell.inference.mega.topk_reduce import TopkReduce

__all__ = ["TopkReduce"]
