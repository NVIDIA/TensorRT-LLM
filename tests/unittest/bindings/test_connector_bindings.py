from typing import List, Optional

from tensorrt_llm._torch.pyexecutor.connector import KvCacheConnector
from tensorrt_llm.bindings.internal.batch_manager import (KvCacheConnectorRole,
                                                          LlmRequest)


class BasicConnector(KvCacheConnector):

    def __init__(self, role: KvCacheConnectorRole):
        super().__init__(role)

    def build_connector_metadata(self) -> Optional[object]:
        return {"test": "test"}

    def start_load_kv(self):
        pass

    def wait_for_layer_load(self, layer_idx: int):
        pass

    def save_kv_layer(self, layer_idx: int):
        pass

    def wait_for_save(self):
        pass

    def get_finished(
            self, finished_req_ids: List[int]) -> tuple[List[int], List[int]]:
        return [42], [7]

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        return 16, True


def test_basic_init():
    connector = BasicConnector(KvCacheConnectorRole.Scheduler)

    assert connector.role == KvCacheConnectorRole.Scheduler

    assert connector.build_connector_metadata() == {"test": "test"}

    assert connector.get_finished([]) == ([42], [7])

    connector.save_kv_layer(0)
    connector.wait_for_save()

    connector_worker = BasicConnector(KvCacheConnectorRole.Worker)

    assert connector_worker.role == KvCacheConnectorRole.Worker
