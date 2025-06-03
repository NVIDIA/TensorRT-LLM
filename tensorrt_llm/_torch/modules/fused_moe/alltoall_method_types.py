import os
from enum import IntEnum

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._utils import local_mpi_size, mpi_rank
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


# The type of alltoall method
class AlltoallMethodType(IntEnum):
    # Not available
    NotAvailable = 0
    # MNNVL
    MNNVL = 1
    # DeepEP intranode or internode: no CUDA Graphs support, IBGDA is required by internode
    DeepEP = 2
    # DeepEP low latency: CUDA Graphs are supported, IBGDA is required
    DeepEPLowLatency = 3


def log_once(func):
    func._logged_args = set()

    def wrapped(*args):
        result = func(*args)
        if args not in func._logged_args and mpi_rank() == 0:
            logger.info(f"{func.__name__} returns {result!r}")
            func._logged_args.add(args)
        return result

    return wrapped


@log_once
def select_alltoall_method_type(mapping: Mapping) -> AlltoallMethodType:
    if MnnvlMemory.supports_mnnvl():
        return AlltoallMethodType.MNNVL

    if os.environ.get("TRTLLM_ALLOW_NO_CUDA_GRAPHS", "0") == "1":
        intranode = mapping.moe_ep_size <= local_mpi_size()
        if intranode:
            return AlltoallMethodType.DeepEP
        else:
            # TODO: Detect IBGDA support
            if os.environ.get("TRTLLM_CAN_USE_IBGDA", "0") == "1":
                return AlltoallMethodType.DeepEP
    else:
        if os.environ.get("TRTLLM_CAN_USE_IBGDA", "0") == "1":
            return AlltoallMethodType.DeepEPLowLatency

    return AlltoallMethodType.NotAvailable
