import os
import platform
import traceback

import torch

from tensorrt_llm._utils import mpi_barrier, mpi_comm, mpi_disabled, torch_comm
from tensorrt_llm.mapping import Mapping

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl():
    return os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"


ENABLE_PDL = get_env_enable_pdl()
if ENABLE_PDL:
    logger.info("PDL is enabled")

if platform.system() != "Windows":
    try:
        import flashinfer
        import flashinfer.comm as flashinfer_comm
        from flashinfer.comm.cuda_ipc import cudart
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )


class FlashInferAllReduceWorkspace:

    def __init__(self, mapping: Mapping):
        if not IS_FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "flashinfer is not installed properly, please try pip install or building from source codes"
            )

        if not mpi_disabled():
            new_group = mpi_comm().group.Incl(mapping.tp_group)
            self.tp_comm = mpi_comm().Create_group(new_group)
        else:
            self.tp_comm = torch_comm()

        self.mapping = mapping

        max_size = 8192 * 8192 * 2  # 2 bytes for bfloat16
        self.meta_ptrs = self.create_shared_buffer(
            flashinfer_comm.vllm_meta_size() + max_size)

        # Create rank data buffer (8MB as in test)
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device=f"cuda:{mapping.local_rank}")

        # Create buffer pointers for IPC communication
        self.buffer_ptrs = self.create_shared_buffer(max_size)

        # Initialize custom allreduce
        self.fa = flashinfer_comm.vllm_init_custom_ar(
            ipc_tensors=self.meta_ptrs,
            rank_data=self.rank_data,
            rank=mapping.rank,
            full_nvlink=True)

        # Register buffer - this is crucial!
        flashinfer_comm.vllm_register_buffer(self.fa, self.buffer_ptrs)

    def create_shared_buffer(self, size_in_bytes: int):
        # Allocate local memory and get IPC handle
        _, pointer = cudart.cudaMalloc(size_in_bytes)
        _, handle = cudart.cudaIpcGetMemHandle(pointer)

        tp_rank = self.mapping.tp_rank
        handles = self.tp_comm.allgather(handle)

        # Open IPC handles
        pointers = []
        for i, h in enumerate(handles):
            if i == tp_rank:
                pointers.append(pointer.value)
            else:
                _, ptr = cudart.cudaIpcOpenMemHandle(h)
                pointers.append(ptr.value)

        if mpi_disabled():
            self.tp_comm.barrier()
        else:
            mpi_barrier()

        return pointers

    def free_shared_buffer(self):
        pass

    def destroy(self):
        pass


flashinfer_allreduce_workspace = None


def init_flashinfer_allreduce_workspace(mapping: Mapping):
    global flashinfer_allreduce_workspace
    flashinfer_allreduce_workspace = FlashInferAllReduceWorkspace(mapping)


def current_flashinfer_allreduce_workspace():
    global flashinfer_allreduce_workspace
    assert flashinfer_allreduce_workspace is not None, "You must call `init_flashinfer_allreduce_workspace` first"
    return flashinfer_allreduce_workspace
