import unittest
from dataclasses import dataclass
from typing import List, Optional, Union

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.distributed import *
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             ResourceManager)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.executor import ExecutorConfig, KvCacheConfig
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.mapping import Mapping


@dataclass(frozen=True, kw_only=True)
class KVCacheParameters:
    num_layers: int
    num_heads: int
    num_kv_heads: Union[int, List[Optional[int]]]
    head_dim: int
    max_seq_len: int


@dataclass
class Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class DummyModel(torch.nn.Module):

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=Config(
            torch_dtype=dtype))
        self.recorded_position_ids = None

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        self.recorded_position_ids = kwargs["position_ids"]
        batch_size = input_ids.size(0)
        return {"logits": torch.randn((batch_size, 10), device='cuda')}


class DummyModelEngine(PyTorchModelEngine):

    def __init__(self,
                 pytorch_backend_config: PyTorchConfig,
                 batch_size: int,
                 dtype: torch.dtype,
                 max_seq_len: int = 32) -> None:
        self.dtype = dtype
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          rank=tensorrt_llm.mpi_rank())
        super().__init__("",
                         pytorch_backend_config,
                         batch_size,
                         max_seq_len=max_seq_len,
                         mapping=mapping)

    def _load_model(self, mode_path: str, **kwargs) -> torch.nn.Module:
        return DummyModel(self.dtype)


def _create_request(num_tokens, req_id: int):
    sampling_params = SamplingParams()
    kwargs = {
        "request_id":
        req_id,
        "max_new_tokens":
        1,
        "input_tokens": [0] * num_tokens,
        "sampling_config":
        tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        "is_streaming":
        False,
    }
    result = LlmRequest(**kwargs)
    result.paged_kv_block_ids = []
    return result


def create_model_engine_and_kvcache():
    max_num_requests = 15
    executor_config = ExecutorConfig(max_batch_size=max_num_requests)
    tokens_per_block = 1
    max_tokens = 130
    num_layers = 1

    config = PyTorchConfig(use_cuda_graph=True, cuda_graph_padding_enabled=True)
    test_batches = (5, 13)
    for batch_size in test_batches:
        assert batch_size not in config.cuda_graph_batch_sizes

    assert (8 in config.cuda_graph_batch_sizes
            and 16 in config.cuda_graph_batch_sizes)

    model_engine = DummyModelEngine(config, max_num_requests, torch.half)

    kv_cache_config = KvCacheConfig(max_tokens=max_tokens)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF, num_layers,
        model_engine.model.config.num_attention_heads,
        model_engine.model.config.num_key_value_heads,
        model_engine.model.config.head_dim, tokens_per_block, max_tokens,
        batch_size, mapping, tensorrt_llm.bindings.DataType.HALF)

    return model_engine, kv_cache_manager


class PyTorchModelEngineTestCase(unittest.TestCase):

    def test_pad_generation_requests(self) -> None:
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()

        resources = {"kv_cache_manager": kv_cache_manager}

        seqlens_and_batch_sizes = [
            (5, 1),
            (13, 1),
            (5, 25),
        ]
        for (batch_size, max_seq_len) in seqlens_and_batch_sizes:
            requests = [
                _create_request(max_seq_len, i) for i in range(batch_size)
            ]
            batch = ScheduledRequests()
            batch.context_requests = requests
            batch.generation_requests = []

            pages_before = kv_cache_manager.get_num_free_blocks()
            with model_engine._maybe_pad_batch(
                    batch, kv_cache_manager) as padded_batch:
                # No padding for prefill
                self.assertIs(batch, padded_batch)
            self.assertEqual(kv_cache_manager.get_num_free_blocks(),
                             pages_before)

            batch = ScheduledRequests()
            batch.context_requests = []
            batch.generation_requests = requests
            pages_before = kv_cache_manager.get_num_free_blocks()
            with model_engine._maybe_pad_batch(
                    batch, kv_cache_manager) as padded_batch:
                if batch_size < 8 and max_seq_len < 25:
                    self.assertEqual(
                        len(padded_batch.generation_requests) % 8, 0)
                else:
                    # No padding if it would create too many concurrent requests.
                    # This requirement is not strictly required, but we should probably
                    # respect the requirement?
                    # The seqlen check makes sure we don't exceed the KV cache memory
                    # budget.
                    self.assertIs(batch, padded_batch)
            self.assertEqual(kv_cache_manager.get_num_free_blocks(),
                             pages_before)

        kv_cache_manager.shutdown()

    def test_position_id_preparation(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {"kv_cache_manager": kv_cache_manager})

        prompt_len = 256
        requests = [_create_request(prompt_len, 0)]

        # Prefill run
        batch = ScheduledRequests()
        batch.context_requests = requests
        batch.generation_requests = []
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        expected_prefill_pos_ids = torch.arange(0,
                                                prompt_len,
                                                dtype=torch.int32,
                                                device='cuda').unsqueeze(0)
        torch.testing.assert_close(model_engine.model.recorded_position_ids,
                                   expected_prefill_pos_ids,
                                   atol=0,
                                   rtol=0)

        # Simulate decoding one token after prefill
        requests[-1].add_new_token(42, 0)

        # Generation run
        batch = ScheduledRequests()
        batch.context_requests = []
        batch.generation_requests = requests
        kv_cache_manager.prepare_resources(batch)

        model_engine.forward(batch, resource_manager)
        expected_gen_pos_id = torch.tensor([prompt_len],
                                           dtype=torch.int64,
                                           device='cuda').unsqueeze(0)
        torch.testing.assert_close(model_engine.model.recorded_position_ids,
                                   expected_gen_pos_id,
                                   atol=0,
                                   rtol=0)

        kv_cache_manager.shutdown()

    def test_warmup(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {"kv_cache_manager": kv_cache_manager})

        # Test with a huge batch size. The warmup run should bail out of
        # warmup instead of crashing (there's not enough KV cache space for this).
        model_engine._cuda_graph_batch_sizes.append(1000000000)

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)
        # Make sure we don't leak any blocks.
        self.assertEqual(num_free_before,
                         kv_cache_manager.get_num_free_blocks())

        kv_cache_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
