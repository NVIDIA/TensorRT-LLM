#!/usr/bin/env python3
import logging

import torch
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm._utils import global_mpi_rank, local_mpi_size

# For multi-node MPI, the worker nodes should launch MPICommExecutor to accept tasks sent from rank0

device_id = global_mpi_rank() % local_mpi_size()
torch.cuda.set_device(device_id)
with MPICommExecutor(COMM_WORLD) as executor:
    if executor is not None:
        raise RuntimeError(f"rank{COMM_WORLD.rank} should not have executor")

logging.warning(f"worker rank{COMM_WORLD.rank} quited")
