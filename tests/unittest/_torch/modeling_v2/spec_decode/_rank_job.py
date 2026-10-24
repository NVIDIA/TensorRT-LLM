# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the existing four-rank launcher wrapper for spec-decode matrices."""

import importlib.util
from pathlib import Path


def run(entry: str, world_size: int) -> None:
    if world_size != 4:
        raise ValueError("This collected launcher supports WORLD4 only")
    directory = Path(__file__).resolve().parent
    path = directory.parent / "comm" / "_rank_job.py"
    spec = importlib.util.spec_from_file_location("_spec_decode_comm_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(entry, launcher_dir=directory)
