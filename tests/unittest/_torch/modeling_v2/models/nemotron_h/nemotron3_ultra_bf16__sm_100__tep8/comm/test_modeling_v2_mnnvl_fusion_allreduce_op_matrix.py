# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for this target's MNNVL fused all-reduce certification matrix.

The matrix is ``_mnnvl_fusion_allreduce_op_matrix.py`` beside this file: one MPI process per rank,
because the engine builds an MNNVL workspace from an MPI communicator. This starts four such ranks
with the suite's ``mpi_pool_executor`` fixture (the way ``multi_gpu/test_mnnvl_allreduce.py`` does)
and has each run the matrix; a rank that fails exits hard and the pool reports it. The same file runs
directly under ``mpirun`` or ``srun`` for any other group size -- see its docstring.

Only on a system the engine itself would use MNNVL on: aarch64 with every NVLink up. On one node the
ranks run the engine's single-node test switches; the production configuration, a group spanning
nodes, is what ``srun`` over two nodes runs.
"""

import pickle
import platform
import sys
from pathlib import Path

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

assert torch.cuda.is_available(), "the MNNVL all-reduce requires CUDA devices"

_MATRIX = Path(__file__).resolve().with_name("_mnnvl_fusion_allreduce_op_matrix.py")
_RANKS = 4


def _rank_main(matrix_path: str) -> bool:
    """Run the matrix in one pool worker: load it by path (this directory is not a package) and
    run every check. Top level and self-contained so cloudpickle ships it by value."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_mnnvl_fusion_allreduce_op_matrix", matrix_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_rank()


# Spawns its own rank processes over the node's devices. Under xdist several workers would fight
# for the same GPUs, so this must run alone.
@pytest.mark.no_xdist
@pytest.mark.timeout(1800)
@pytest.mark.skipif(
    platform.machine().lower() != "aarch64"
    or torch.cuda.device_count() < _RANKS
    or not MnnvlMemory.supports_mnnvl(),
    reason=f"MNNVL all-reduce needs an NVL72-class aarch64 node with {_RANKS} devices",
)
@pytest.mark.parametrize("mpi_pool_executor", [_RANKS], indirect=True)
def test_mnnvl_fusion_allreduce_op_matrix(mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_main, [str(_MATRIX)] * _RANKS))
    assert results == [True] * _RANKS, results
