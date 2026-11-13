# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for `ops/dspark/dspark_markov_chain`'s certification matrix
(``_dspark_markov_chain_op_matrix.py`` beside this file, which drives the entry's ``.CELLS``).

World 4 is collected here. The world-8 cells span two GB300 trays and run as one srun task per rank (see the
matrix's docstring), like ``comm/_mnnvl_allreduce_op_matrix.py``.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (103, 107),
    reason="dspark_markov_chain is certified on sm_103 and sm_107",
)


def _rank_job():
    # By path: other test directories ship a module of the same name, and `import _rank_job` would get whichever
    # one a collected sibling imported first.
    path = Path(__file__).resolve().with_name("_rank_job.py")
    spec = importlib.util.spec_from_file_location("_minimax_m3_comm_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Spawns 4 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_dspark_markov_chain_op_matrix() -> None:
    _rank_job().run("dspark_markov_chain", 4)
