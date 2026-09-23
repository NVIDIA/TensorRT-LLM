# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin Genphase MegaMoE shim for the shared Gather4 primitive."""

# <<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>
from ..local_mega.tma_gather import sm107_tma_gather4_load

__all__ = ["sm107_tma_gather4_load"]
