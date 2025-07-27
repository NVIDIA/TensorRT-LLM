from typing import List

from tensorrt_llm.bindings.internal.batch_manager import (
    KvCacheConnectorPoolsData, KvCacheConnectorScheduler,
    KvCacheConnectorWorker, LlmRequest)


class BasicConnectorWorker(KvCacheConnectorWorker):

    def register_kv_caches(self, kv_cache_data: KvCacheConnectorPoolsData):
        pass

    def start_load_kv(self):
        pass

    def wait_for_save(self):
        pass

    def wait_for_layer_load(self, layer_idx: int):
        pass

    def save_kv_layer(self, layer_idx: int):
        pass

    def get_finished(
            self, finished_req_ids: List[int]) -> tuple[List[int], List[int]]:
        return [42], [7]


class BasicConnectorScheduler(KvCacheConnectorScheduler):

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        return 16, True

    def update_state_after_alloc(self):
        pass


def test_basic_init():
    connector_scheduler = BasicConnectorScheduler()

    connector_scheduler.update_state_after_alloc()

    connector_worker = BasicConnectorWorker()

    assert connector_worker.get_finished([]) == ([42], [7])

    connector_worker.save_kv_layer(0)
    connector_worker.wait_for_save()
