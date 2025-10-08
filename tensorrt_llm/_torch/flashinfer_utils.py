import contextlib
import os
import platform
import traceback

import torch

from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm._utils import mpi_comm, mpi_disabled, torch_comm
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

        self.mapping = mapping

        max_size = 8192 * 8192 * 2  # 2 bytes for bfloat16
        logger.info(
            f"Opening IPC memory for meta with size {flashinfer_comm.vllm_meta_size() + max_size}"
        )
        try:
            self.meta_ptrs_ipc = IpcMemory(
                mapping,
                flashinfer_comm.vllm_meta_size() + max_size)

            # Create rank data buffer (8MB as in test)
            self.rank_data = torch.empty(8 * 1024 * 1024,
                                         dtype=torch.uint8,
                                         device=f"cuda:{mapping.local_rank}")

            # Create buffer pointers for IPC communication
            self.buffer_ptrs_ipc = IpcMemory(mapping, max_size)

            # Initialize custom allreduce
            self.fa = flashinfer_comm.vllm_init_custom_ar(
                ipc_tensors=self.meta_ptrs_ipc.peer_ptrs,
                rank_data=self.rank_data,
                rank=mapping.tp_rank,
                full_nvlink=True)

            flashinfer_comm.vllm_register_buffer(self.fa,
                                                 self.buffer_ptrs_ipc.peer_ptrs)
        except Exception:
            traceback.print_exc()
            logger.error(f"Error initializing FlashInferAllReduceWorkspace")
            raise

        self._is_capturing = False
        self._graph_registered = False
        logger.info(
            f"FlashInferAllReduceWorkspace initialized for rank {mapping.tp_rank}")

    @contextlib.contextmanager
    def capture(self):
        try:
            self._is_capturing = True
            logger.info(
                f"Rank {self.mapping.tp_rank}: Starting CUDA graph capture")
            yield
        finally:
            self._is_capturing = False
            # Register graph buffers after capture if not already done
            if not self._graph_registered:
                self.register_graph_buffers()
            logger.info(
                f"Rank {self.mapping.tp_rank}: Finished CUDA graph capture")

    def register_graph_buffers(self):
        # add error handling
        handle, offsets = flashinfer_comm.vllm_get_graph_buffer_ipc_meta(self.fa)
        logger.info(
            f"Rank {self.mapping.tp_rank}: Registering {len(handle)} graph buffer(s)"
        )

        world_size = self.mapping.tp_size
        tp_rank = self.mapping.tp_rank
        tp_group_ranks = sorted(self.mapping.tp_group)

        all_data = [[None, None] for _ in range(world_size)]
        all_data[tp_rank] = [handle, offsets]

        comm = mpi_comm().Split(
            self.mapping.pp_rank * self.mapping.cp_size + self.mapping.cp_rank, self.mapping.tp_rank)

        for i in range(world_size):
            all_data[i] = comm.bcast(all_data[i], i)

        handles = [d[0] for d in all_data]
        offsets_list = [d[1] for d in all_data]

        flashinfer_comm.vllm_register_graph_buffers(self.fa, handles, offsets_list)
        self._graph_registered = True
        logger.info(
            f"Rank {self.mapping.tp_rank}: Registered {len(handle)} graph buffer(s)"
        )


flashinfer_allreduce_workspace = None


def init_flashinfer_allreduce_workspace(mapping: Mapping):
    global flashinfer_allreduce_workspace
    if flashinfer_allreduce_workspace is None:
        flashinfer_allreduce_workspace = FlashInferAllReduceWorkspace(mapping)


def current_flashinfer_allreduce_workspace():
    global flashinfer_allreduce_workspace
    assert flashinfer_allreduce_workspace is not None, "You must call `init_flashinfer_allreduce_workspace` first"
    return flashinfer_allreduce_workspace
