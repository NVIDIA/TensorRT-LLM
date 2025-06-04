import os
from enum import IntEnum

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._utils import local_mpi_size
from tensorrt_llm.mapping import Mapping

from .deep_ep_utils import deep_ep_installed


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


def select_alltoall_method_type(mapping: Mapping,
                                use_cuda_graph: bool) -> AlltoallMethodType:
    if MnnvlMemory.supports_mnnvl():
        return AlltoallMethodType.MNNVL

    if deep_ep_installed:
        intranode = mapping.moe_ep_size <= local_mpi_size()
        ibgda = os.environ.get("TRTLLM_CAN_USE_IBGDA",
                               "0") == "1"  # TODO: Auto detect IBGDA support
        if use_cuda_graph:
            # Here we can only choose DeepEPLowLatency since only this method supports CUDA Graphs.
            if ibgda:
                return AlltoallMethodType.DeepEPLowLatency
        else:
            # Here we can choose DeepEP or DeepEPLowLatency if both are available. Now DeepEP is faster.
            if intranode or ibgda:
                return AlltoallMethodType.DeepEP

    return AlltoallMethodType.NotAvailable
