# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This directory's collective matrices, run through modeling_v2's shared rank launcher (``comm/_rank_job.py``).

The shared launcher starts ``_<entry>_op_matrix.py`` from the directory it is given, so the matrices stay beside the
tests of the entries they certify.
"""

import importlib.util
from pathlib import Path

# tests/unittest/_torch/modeling_v2/models/minimax_m3/ops/<dir>/ -> tests/unittest/_torch/modeling_v2/comm/
_SHARED_PATH = Path(__file__).resolve().parents[4] / "comm" / "_rank_job.py"
_spec = importlib.util.spec_from_file_location("_modeling_v2_comm_rank_job", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

WORLD_SIZE = _shared.WORLD_SIZE
adopt_srun_env = _shared.adopt_srun_env


def run(entry: str, world_size: int = WORLD_SIZE) -> None:
    """Run ``_<entry>_op_matrix.py`` beside this file over ``world_size`` devices of this node."""
    _shared.run(entry, world_size, directory=Path(__file__).resolve().parent)
