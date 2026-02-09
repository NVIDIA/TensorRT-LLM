import contextlib
import os
import platform
import traceback

import torch

from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer
from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.mapping import Mapping

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl():
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


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

_MiB = 1024 * 1024


class FlashInferAllReduceWorkspace:

    def __init__(self, mapping: Mapping):
        if not IS_FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "flashinfer is not installed properly, please try pip install or building from source codes"
            )

        self.mapping = mapping

        if mapping.tp_size not in (2, 4, 6, 8):
            raise ValueError(
                f"FlashInfer vLLM custom allreduce only supports tp_size in (2,4,6,8), "
                f"got {mapping.tp_size}")

        if not can_access_peer(mapping):
            raise RuntimeError(
                "FlashInfer vLLM custom allreduce requires NVLink peer access "
                "between all TP ranks")

        self.max_size = self._get_max_size()
        meta_alloc_size = flashinfer_comm.vllm_meta_size() + self.max_size
        logger.info(f"FlashInfer AR: meta_alloc={meta_alloc_size}, "
                    f"buffer_alloc={self.max_size}, rank_data=8MB")
        try:
            self.meta_ptrs_ipc = IpcMemory(
                mapping,
                flashinfer_comm.vllm_meta_size() + self.max_size)

            # Create rank data buffer. 8MB is for 131072 elements.
            # Most models use < 10000 elements.
            self.rank_data = torch.empty(8 * 1024 * 1024,
                                         dtype=torch.uint8,
                                         device=f"cuda:{mapping.local_rank}")

            # Create buffer pointers for IPC communication
            self.buffer_ptrs_ipc = IpcMemory(mapping, self.max_size)

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
        logger.info(
            f"FlashInferAllReduceWorkspace initialized for rank {mapping.tp_rank}"
        )

    def _get_max_size(self):
        # Per-architecture max input sizes for custom allreduce.
        # Keys are (SM major, SM minor).
        # Values are dicts of {tp_size: max_bytes}.
        # Based on empirical testing.
        custom_ar_max_sizes = {
            (8, 0): {
                2: 8 * _MiB,
                4: 8 * _MiB,
                6: 8 * _MiB,
                8: 8 * _MiB
            },
            (8, 9): {
                2: 8 * _MiB,
                4: 8 * _MiB,
                6: 8 * _MiB,
                8: 8 * _MiB
            },
            (9, 0): {
                2: 8 * _MiB,
                4: 8 * _MiB,
                6: 4 * _MiB,
                8: 4 * _MiB
            },
            (10, 0): {
                2: 4 * _MiB,
                4: 4 * _MiB,
                6: 4 * _MiB,
                8: 4 * _MiB
            },
        }

        default_max_size = 8 * _MiB  # 8 MB — safe default for unknown architectures

        if not torch.cuda.is_available():
            return default_max_size

        device = torch.device(f"cuda:{self.mapping.tp_rank}")
        cap = torch.cuda.get_device_capability(device)
        arch_sizes = custom_ar_max_sizes.get(cap)
        if arch_sizes is not None and self.mapping.tp_size in arch_sizes:
            size = arch_sizes[self.mapping.tp_size]
            return min(size, default_max_size)
        return default_max_size

    @contextlib.contextmanager
    def capture(self):
        try:
            self._is_capturing = True
            yield
        finally:
            self._is_capturing = False
            # Register graph buffers after capture if not already done
            self.register_graph_buffers()

    def register_graph_buffers(self):
        try:
            handle, offsets = flashinfer_comm.vllm_get_graph_buffer_ipc_meta(
                self.fa)
        except Exception as e:
            logger.error(f"Failed to get graph buffer IPC meta: {e}")
            raise e

        world_size = self.mapping.tp_size
        tp_rank = self.mapping.tp_rank

        all_data = [[None, None] for _ in range(world_size)]
        all_data[tp_rank] = [handle, offsets]

        comm = mpi_comm().Split(
            self.mapping.pp_rank * self.mapping.cp_size + self.mapping.cp_rank,
            self.mapping.tp_rank)

        for i in range(world_size):
            all_data[i] = comm.bcast(all_data[i], i)

        handles = [d[0] for d in all_data]
        offsets_list = [d[1] for d in all_data]

        flashinfer_comm.vllm_register_graph_buffers(self.fa, handles,
                                                    offsets_list)


flashinfer_allreduce_workspace = None


def init_flashinfer_allreduce_workspace(mapping: Mapping):
    global flashinfer_allreduce_workspace
    if flashinfer_allreduce_workspace is None and IS_FLASHINFER_AVAILABLE:
        flashinfer_allreduce_workspace = FlashInferAllReduceWorkspace(mapping)


def get_current_flashinfer_allreduce_workspace(
) -> FlashInferAllReduceWorkspace:
    global flashinfer_allreduce_workspace
    assert flashinfer_allreduce_workspace is not None, "You must call `init_flashinfer_allreduce_workspace` first"
    return flashinfer_allreduce_workspace


def cleanup_flashinfer_allreduce_workspace():
    global flashinfer_allreduce_workspace
    if flashinfer_allreduce_workspace is not None and IS_FLASHINFER_AVAILABLE:
        try:
            flashinfer_comm.vllm_dispose(flashinfer_allreduce_workspace.fa)
        except Exception as e:
            logger.warning(f"Error disposing FlashInfer AR workspace: {e}")
        flashinfer_allreduce_workspace = None
