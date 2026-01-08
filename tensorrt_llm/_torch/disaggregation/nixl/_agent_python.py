import time

from nixl import nixl_agent, nixl_agent_config, nixl_xfer_handle

from tensorrt_llm._utils import nvtx_range

from ..base.agent import BaseTransferAgent, BaseTransferStatus, RegMemoryDescs, TransferRequest


class NixlTransferStatus(BaseTransferStatus):
    """TransferStatus using the Python NIXL library."""

    def __init__(self, agent: nixl_agent, handle: nixl_xfer_handle):
        self.agent = agent
        self.handle = handle

    def is_completed(self) -> bool:
        status = self.agent.check_xfer_state(self.handle)
        return status == "DONE"

    def wait(self) -> bool:
        status = "PROC"
        sleep_time = 0.0001  # 0.1ms
        max_sleep_time = 0.01  # 10ms
        while status == "PROC":
            status = self.agent.check_xfer_state(self.handle)
            if status == "ERROR":
                return False  # Transfer failed
            time.sleep(sleep_time)  # Sleep to release GIL
            sleep_time = min(sleep_time * 2, max_sleep_time)
        return status == "DONE"


class NixlTransferAgent(BaseTransferAgent):
    """Python-based TransferAgent using the NIXL library."""

    def __init__(self, name: str, use_prog_thread: bool, num_workers: int = 1):
        self.name = name
        agent_config = nixl_agent_config(
            enable_prog_thread=use_prog_thread,
            backends=["UCX"],
            num_threads=num_workers,
        )
        self.agent = nixl_agent(name, agent_config)

    def register_memory(self, descs: RegMemoryDescs):
        reg_descs = self.agent.get_reg_descs(descs.descs, descs.type)
        self.agent.register_memory(reg_descs)

    def deregister_memory(self, descs: RegMemoryDescs):
        self.agent.deregister_memory(descs.descs, descs.type)

    def load_remote_agent(self, name: str, agent_desc: bytes):
        self.agent.add_remote_agent(agent_desc)

    def get_local_agent_desc(self) -> bytes:
        return self.agent.get_agent_metadata()

    def invalidate_remote_agent(self, name: str):
        self.agent.remove_remote_agent(name)

    def check_remote_descs(self, name: str, memory_descs: list[int]) -> bool:
        raise NotImplementedError("check_remote_descs is not implemented.")

    def notify_sync_message(self, name: str, sync_message: str):
        raise NotImplementedError("notify_sync_message is not implemented.")

    @nvtx_range("NixlTransferAgent.submit_transfer_requests")
    def submit_transfer_requests(self, request: TransferRequest) -> BaseTransferStatus:
        src_xfer_descs = self.agent.get_xfer_descs(request.src_descs.descs, request.src_descs.type)
        dst_xfer_descs = self.agent.get_xfer_descs(request.dst_descs.descs, request.dst_descs.type)
        handle = self.agent.initialize_xfer(
            request.op,
            src_xfer_descs,
            dst_xfer_descs,
            request.remote_name,
            request.sync_message,
        )
        status = self.agent.transfer(handle)
        assert status != "ERROR", "Transfer failed during initialization."
        return NixlTransferStatus(self.agent, handle)
