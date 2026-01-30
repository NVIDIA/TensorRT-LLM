from dataclasses import dataclass
from enum import Enum
from typing import List

from tensorrt_llm._torch.disaggregation.native import (
    Receiver,
    RxSession,
    Sender,
    State,
    TaskIdType,
    TxSession,
)
from tensorrt_llm._torch.disaggregation.nixl import NixlTransferAgent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


class BackendType(Enum):
    NIXL = "Nixl"


nixl_classes = {
    BackendType.NIXL: {
        "transfer_agent": NixlTransferAgent,
        "tx_session": TxSession,
        "rx_session": RxSession,
        "receiver": Receiver,
        "sender": Sender,
    },
}


@dataclass
class TaskQuery:
    llm_request: LlmRequest
    task_id: List[TaskIdType]


class CacheTransceiver:
    def __init__(self, backend_type: BackendType, kv_cache_manager: KVCacheManager):
        classes = self._get_transfer_classes(backend_type)
        self.transfer_agent_class = classes["transfer_agent"]
        self.tx_session_class = classes["tx_session"]
        self.rx_session_class = classes["rx_session"]
        self.receiver_class = classes["receiver"]
        self.sender_class = classes["sender"]
        self.kv_cache_manager = kv_cache_manager

    def _get_transfer_classes(backend_type: BackendType):
        if backend_type in nixl_classes:
            return nixl_classes[backend_type]
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def create_transfer_agent(self, *args, **kwargs):
        self.transfer_agent = self.transfer_agent_class(*args, **kwargs)
        return self.transfer_agent

    def create_sender(self, *args, **kwargs):
        self.sender = self.sender_class(*args, **kwargs)
        return self.sender

    def create_receiver(self, *args, **kwargs):
        self.receiver = self.receiver_class(*args, **kwargs)
        return self.receiver

    def create_tx_session(self, *args, **kwargs):
        self.tx_session = self.tx_session_class(*args, **kwargs)
        return self.tx_session

    def create_rx_session(self, *args, **kwargs):
        self.rx_session = self.rx_session_class(*args, **kwargs)
        return self.rx_session

    @staticmethod
    def aggregate_sessions_state(llm_requests: List[LlmRequest]) -> List[State]: ...

    @staticmethod
    def aggregate_tasks_state(query: List[TaskQuery]) -> List[State]: ...
