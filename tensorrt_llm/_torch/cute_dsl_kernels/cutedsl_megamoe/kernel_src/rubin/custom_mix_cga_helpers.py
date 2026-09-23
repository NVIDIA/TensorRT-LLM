# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin source-copy shim for shared mixed-CGA helpers."""

# <<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>
from ..blackwell.custom_mix_cga_helpers import (
    PipelineTmaUmmaMixedCga,
    TmaAtomOrPair,
    bind_executable_tma_load_fields,
    make_executable_tma_atom_or_pair,
    pipeline_init_arrive_mixed_cga,
    pipeline_init_wait_mixed_cga,
    select_tma_atom_for_cluster,
)

__all__ = [
    "PipelineTmaUmmaMixedCga",
    "TmaAtomOrPair",
    "bind_executable_tma_load_fields",
    "make_executable_tma_atom_or_pair",
    "pipeline_init_arrive_mixed_cga",
    "pipeline_init_wait_mixed_cga",
    "select_tma_atom_for_cluster",
]
