#!/usr/bin/env python3
import logging

from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD

# For multi-node MPI, the worker nodes should launch MPICommExecutor to accept tasks sent from rank0
with MPICommExecutor(COMM_WORLD) as executor:
    if executor is not None:
        raise RuntimeError(f"rank{COMM_WORLD.rank} should not have executor")

logging.warning(f"worker rank{COMM_WORLD.rank} quited")
