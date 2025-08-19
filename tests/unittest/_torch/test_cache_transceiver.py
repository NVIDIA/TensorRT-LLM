import unittest

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import \
    create_kv_cache_transceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType


class TestCacheTransceiver(unittest.TestCase):

    def _create_kv_cache_manager(self, mapping):
        return KVCacheManager(
            trtllm.KvCacheConfig(
                max_tokens=1024,
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=4,
            num_kv_heads=1,
            head_dim=2,
            tokens_per_block=8,
            max_seq_len=64,
            max_batch_size=4,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.FLOAT,
        )

    def _create_requests(self, request_ids):
        requests = []
        for request_id in request_ids:
            requests.append(
                LlmRequest(
                    request_id=request_id,
                    max_new_tokens=1,
                ))
        return requests

    #def _get_context_phase_params(self):
    #return None

    #def _prepare_

    def setUp(self):
        self.mapping = Mapping(world_size=2, tp_size=2, rank=0)
        self.kv_cache_manager_0 = self._create_kv_cache_manager(self.mapping)
        self.kv_cache_manager_1 = self._create_kv_cache_manager(self.mapping)

        cache_transceiver_config = trtllm.CacheTransceiverConfig(
            backend=trtllm.CacheTransceiverBackendType.DEFAULT,
            max_tokens_in_buffer=1024)

        self.kv_cache_transceiver_0 = create_kv_cache_transceiver(
            self.mapping, self.kv_cache_manager_0, AttentionTypeCpp.DEFAULT,
            cache_transceiver_config)
        self.kv_cache_transceiver_1 = create_kv_cache_transceiver(
            self.mapping, self.kv_cache_manager_1, AttentionTypeCpp.DEFAULT,
            cache_transceiver_config)

    def test_cache_transceiver(self):
        sampling_params = SamplingParams()
        request = LlmRequest(
            request_id=0,
            max_new_tokens=1,
            input_tokens=[1, 2, 3, 4, 5],
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

        self.kv_cache_manager_0.impl.add_sequence(request.py_request_id,
                                                  request.prompt_len, 1,
                                                  request)
        self.kv_cache_transceiver_0.respond_and_send_async(request)

        request_1 = LlmRequest(
            request_id=1,
            max_new_tokens=1,
            input_tokens=[1, 2, 3, 4, 5],
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            context_phase_params=request.context_phase_params)
        self.kv_cache_manager_1.impl.add_sequence(request_1.py_request_id,
                                                  request_1.prompt_len, 1,
                                                  request_1)
        self.kv_cache_transceiver_1.request_and_receive_async(request_1)

        if self.kv_cache_transceiver_0.check_context_transfer_status(1):
            raise Exception("Context transfer status is not 0")
        if self.kv_cache_transceiver_1.check_gen_transfer_status(1):
            raise Exception("Gen transfer status is not 0")


if __name__ == "__main__":
    unittest.main()
