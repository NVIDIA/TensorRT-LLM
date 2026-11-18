# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry points for the family comm ops' certification matrices (``_<entry>_op_matrix.py`` beside this
file, each driving its entry's ``.CELLS``).

- ``allgather_cast``: `ops/comm/allgather_cast`.
- ``dspark_markov_chain``: `ops/dspark/dspark_markov_chain`.
- ``dspark_markov_chain_ll``: its LL exchange variant (LL tokens == the S4B8 batch-1 kernel's, bitwise, on real
  ranks, including forced cross-rank ties, back-to-back launches and CUDA-graph replays).
- ``m31_allreduce``: `ops/comm/m31_allreduce` (MiniMax-M3.1-Flash, sm_107 only).

World 4 is collected here. The world-8 cells span two GB300 trays and run as one srun task per rank (see each
matrix's docstring), like ``comm/_mnnvl_allreduce_op_matrix.py``.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (103, 107),
    reason="these comm ops are certified on sm_103 and sm_107",
)


def _rank_job():
    # By path: other test directories ship a module of the same name, and `import _rank_job` would get whichever
    # one a collected sibling imported first.
    path = Path(__file__).resolve().with_name("_rank_job.py")
    spec = importlib.util.spec_from_file_location("_minimax_m3_comm_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Spawns `world` rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize(
    "entry,world",
    [
        ("allgather_cast", 4),
        ("dspark_markov_chain", 4),
        ("dspark_markov_chain_ll", 4),
        pytest.param(
            "m31_allreduce",
            4,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or get_sm_version() != 107,
                reason="m31_allreduce is certified on sm_107",
            ),
        ),
    ],
)
def test_op_matrix(entry: str, world: int) -> None:
    _rank_job().run(entry, world)
