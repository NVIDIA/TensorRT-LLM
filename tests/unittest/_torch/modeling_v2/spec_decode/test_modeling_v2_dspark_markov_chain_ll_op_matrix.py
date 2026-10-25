# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for `spec_decode/dspark_markov_chain`'s LL exchange variant
(``_dspark_markov_chain_ll_op_matrix.py`` beside this file: LL tokens == the S4B8 batch-1 kernel's, bitwise, on real
ranks, including forced cross-rank ties, back-to-back launches and CUDA-graph replays).

World 4 is collected here. World 8 spans two GB300 trays and runs as one srun task per rank (see the matrix's
docstring).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

assert torch.cuda.is_available(), "dspark_markov_chain requires CUDA devices"


def _rank_job():
    # By path: other test directories ship a module of the same name.
    path = Path(__file__).resolve().with_name("_rank_job.py")
    spec = importlib.util.spec_from_file_location("_spec_decode_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Spawns 4 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_dspark_markov_chain_ll_op_matrix() -> None:
    _rank_job().run("dspark_markov_chain_ll", 4)
