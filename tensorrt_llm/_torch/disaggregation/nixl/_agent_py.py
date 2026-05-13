import time
from enum import Enum

from nixl import nixl_agent, nixl_agent_config, nixl_xfer_handle

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

# Import base classes for type compatibility
from ..base.agent import BaseTransferAgent, RegMemoryDescs, TransferRequest, TransferStatus


class TransferState(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROC"
    DONE = "DONE"
    ERROR = "ERROR"


class NixlTransferStatus(TransferStatus):
    def __init__(self, agent: nixl_agent, handle: nixl_xfer_handle):
        self.agent = agent
        self.handle = handle

    def is_completed(self):
        status = TransferState(self.agent.check_xfer_state(self.handle))
        return status == TransferState.DONE

    def wait(self, timeout_ms=None):
        start_time = time.time()
        status = TransferState.PENDING
        sleep_time = 0.0001  # 0.1ms in seconds
        max_sleep_time = 0.01  # 10ms in seconds

        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        while status in (TransferState.PENDING, TransferState.PROCESSING):
            status = TransferState(self.agent.check_xfer_state(self.handle))
            if status == TransferState.ERROR:
                logger.error("NIXL transfer entered ERROR state (agent=%s).", self.agent.name)
                return False
            if timeout is not None and (time.time() - start_time > timeout):
                logger.warning("NIXL transfer wait timed out after %s ms.", timeout_ms)
                return False
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, max_sleep_time)
        return status == TransferState.DONE


class NixlTransferAgent(BaseTransferAgent):
    """NixlTransferAgent using Python nixl library."""

    def __init__(self, name: str, use_prog_thread: bool = True, num_threads: int = 1, **kwargs):
        """
        Initialize NixlTransferAgent.
        :param name: Name of the agent.
        :param use_prog_thread: Whether to enable the progress thread, if available.
        :param num_workers: Specify number of threads for the supported multi-threaded backends.
        """
        self.name = name
        self.backends = ["UCX"]
        agent_config = nixl_agent_config(
            enable_prog_thread=use_prog_thread, backends=self.backends, num_threads=num_threads
        )
        self.agent = nixl_agent(name, agent_config)

    def shutdown(self):
        if getattr(self, "agent", None) is None:
            return
        self.agent = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()

    def _get_validated_reg_descs(self, descs: RegMemoryDescs):
        if not descs.descs:
            raise ValueError("descs.descs must not be empty")
        if isinstance(descs.descs[0], tuple) and len(descs.descs[0]) != 4:
            raise ValueError(
                f"Expected 4 elements per desc, got {len(descs.descs[0])}: {descs.descs[0]}"
            )
        reg_descs = self.agent.get_reg_descs(descs.descs, descs.type)
        if reg_descs is None:
            raise RuntimeError(
                f"nixl get_reg_descs returned None for type={descs.type}, count={len(descs.descs)}"
            )
        return reg_descs

    def register_memory(self, descs: RegMemoryDescs):
        self.agent.register_memory(self._get_validated_reg_descs(descs))

    def deregister_memory(self, descs: RegMemoryDescs):
        self.agent.deregister_memory(self._get_validated_reg_descs(descs))

    def load_remote_agent(self, name: str, agent_desc: bytes):
        self.agent.add_remote_agent(agent_desc)

    def get_local_agent_desc(self):
        return self.agent.get_agent_metadata()

    def invalidate_remote_agent(self, name: str):
        self.agent.remove_remote_agent(name)

    def check_remote_descs(self, name: str, memory_descs: list[int]) -> bool:
        raise NotImplementedError

    def notify_sync_message(self, name: str, sync_message: str):
        raise NotImplementedError

    @nvtx_range("NixlTransferAgent.submit_transfer_requests")
    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        src_xfer_descs = self.agent.get_xfer_descs(request.src_descs.descs, request.src_descs.type)
        if src_xfer_descs is None:
            raise RuntimeError(
                f"nixl get_xfer_descs returned None for src type={request.src_descs.type}"
            )
        dst_xfer_descs = self.agent.get_xfer_descs(request.dst_descs.descs, request.dst_descs.type)
        if dst_xfer_descs is None:
            raise RuntimeError(
                f"nixl get_xfer_descs returned None for dst type={request.dst_descs.type}"
            )
        sync_message = "" if request.sync_message is None else request.sync_message
        handle = self.agent.initialize_xfer(
            request.op,
            src_xfer_descs,
            dst_xfer_descs,
            request.remote_name,
            sync_message,
        )
        status = self.agent.transfer(handle)
        if status == "ERROR":
            raise RuntimeError(
                f"NIXL transfer failed: op={request.op}, remote={request.remote_name}"
            )
        return NixlTransferStatus(self.agent, handle)
