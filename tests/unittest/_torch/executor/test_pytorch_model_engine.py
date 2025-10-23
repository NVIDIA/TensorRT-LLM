import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import torch

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import \
    KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

# isort: off
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             ResourceManager,
                                                             ResourceManagerType
                                                             )
# isort: on
from utils.util import skip_ray

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import CudaGraphConfig, LlmArgs, SamplingParams
from tensorrt_llm.mapping import CpType, Mapping


@dataclass
class Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list[str] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class DummyKvCacheConnectorWorker(KvCacheConnectorWorker):

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        pass


class DummyModel(torch.nn.Module):

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=Config(
            torch_dtype=dtype))
        self.recorded_position_ids = None

    def infer_max_seq_len(self):
        return 2048

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
        model = DummyModel(self.dtype)
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

        model_path = "dummy"
        llm_args = TorchLlmArgs(model=model_path,
                                max_batch_size=batch_size,
                                max_seq_len=max_seq_len)
        super().__init__(model_path=model_path,
                         pytorch_backend_config=pytorch_backend_config,
                         mapping=mapping,
                         model=model,
                         llm_args=llm_args)


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


def create_model_engine_and_kvcache(config: PyTorchConfig = None):
    max_num_requests = 15
    tokens_per_block = 1
    max_tokens = 258  # Atleast 1 more than the max seq len
    num_layers = 1
    batch_size = 13

    config = config if config else PyTorchConfig(
        use_cuda_graph=True, cuda_graph_padding_enabled=True)
    config.cuda_graph_batch_sizes = [
        1, 2, 4, 8, 16, 32, 64, 128
    ] if config.cuda_graph_batch_sizes is None else config.cuda_graph_batch_sizes
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
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=model_engine.model.config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_tokens,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )

    return model_engine, kv_cache_manager


class PyTorchModelEngineTestCase(unittest.TestCase):

    def test_pad_generation_requests(self) -> None:
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

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
            with model_engine.cuda_graph_runner.pad_batch(
                    batch, resource_manager) as padded_batch:
                # No padding for prefill
                self.assertIs(batch, padded_batch)
            self.assertEqual(kv_cache_manager.get_num_free_blocks(),
                             pages_before)

            batch = ScheduledRequests()
            batch.context_requests = []
            batch.generation_requests = requests
            pages_before = kv_cache_manager.get_num_free_blocks()
            new_dummy_block = 1 if model_engine.cuda_graph_runner.padding_dummy_request is None else 0
            with model_engine.cuda_graph_runner.pad_batch(
                    batch, resource_manager) as padded_batch:
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
            self.assertEqual(
                kv_cache_manager.get_num_free_blocks() + new_dummy_block,
                pages_before)

        kv_cache_manager.shutdown()

    def test_position_id_preparation(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

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
                                           dtype=torch.int32,
                                           device='cuda').unsqueeze(0)
        torch.testing.assert_close(model_engine.model.recorded_position_ids,
                                   expected_gen_pos_id,
                                   atol=0,
                                   rtol=0)

        kv_cache_manager.shutdown()

    def test_warmup(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # Test with a huge batch size. The warmup run should bail out of
        # warmup instead of crashing (there's not enough KV cache space for this).
        model_engine._cuda_graph_batch_sizes.append(1000000000)

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)
        # Make sure we don't leak any blocks.
        self.assertEqual(num_free_before,
                         kv_cache_manager.get_num_free_blocks())

        kv_cache_manager.shutdown()

    def test_layerwise_nvtx_marker(self):
        config = PyTorchConfig(use_cuda_graph=True,
                               cuda_graph_padding_enabled=True,
                               enable_layerwise_nvtx_marker=True)
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(config)
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

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

        kv_cache_manager.shutdown()

    def test_cuda_graph_padding_filters_huge_batch_size(self):
        config = PyTorchConfig(
            use_cuda_graph=True,
            cuda_graph_padding_enabled=True,
            cuda_graph_batch_sizes=[1, 2, 3, 1000000000000000000000000])
        model_engine = DummyModelEngine(config, 32, torch.half)

        self.assertEqual(model_engine._cuda_graph_batch_sizes,
                         [1, 2, 3, model_engine.max_seq_len])

    def test_cuda_graph_enable(self):
        # Test 1: Default behavior (no cuda_graph_config specified)
        llm_args_default = LlmArgs.from_kwargs(model="dummy_model")
        pytorch_config_default = llm_args_default.get_pytorch_backend_config()
        self.assertTrue(pytorch_config_default.use_cuda_graph,
                        "CUDA graphs should be enabled by default")

        # Test 2: Explicit CudaGraphConfig()
        llm_args_explicit = LlmArgs.from_kwargs(
            model="dummy_model", cuda_graph_config=CudaGraphConfig())
        pytorch_config_explicit = llm_args_explicit.get_pytorch_backend_config()
        self.assertTrue(
            pytorch_config_explicit.use_cuda_graph,
            "CUDA graphs should be enabled when CudaGraphConfig() is provided")

        # Test 3: cuda_graph_config=None (explicitly disabled)
        llm_args_disabled = LlmArgs.from_kwargs(model="dummy_model",
                                                cuda_graph_config=None)
        pytorch_config_disabled = llm_args_disabled.get_pytorch_backend_config()
        self.assertFalse(
            pytorch_config_disabled.use_cuda_graph,
            "CUDA graphs should be disabled when cuda_graph_config=None")

        # Test 4: Custom CudaGraphConfig with specific settings
        custom_config = CudaGraphConfig(max_batch_size=256, enable_padding=True)
        llm_args_custom = LlmArgs.from_kwargs(model="dummy_model",
                                              cuda_graph_config=custom_config)
        pytorch_config_custom = llm_args_custom.get_pytorch_backend_config()
        self.assertTrue(pytorch_config_custom.use_cuda_graph,
                        "CUDA graphs should be enabled with custom config")
        self.assertEqual(pytorch_config_custom.cuda_graph_max_batch_size, 256,
                         "Custom max_batch_size should be respected")
        self.assertTrue(pytorch_config_custom.cuda_graph_padding_enabled,
                        "Custom enable_padding should be respected")

    def test_forward_pass_callable_on_cuda_graph_on(self):
        config = PyTorchConfig(use_cuda_graph=True,
                               cuda_graph_padding_enabled=True)
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(config)

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests = requests
        batch.generation_requests = []
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_forward_pass_callable_on_cuda_graph_off(self):
        config = PyTorchConfig(use_cuda_graph=False)
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(config)

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests = requests
        batch.generation_requests = []
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_foward_pass_callable_off(self):
        config = PyTorchConfig(use_cuda_graph=False)
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(config)
        self.assertTrue(model_engine.forward_pass_callable is None,
                        "forward_pass_callback should be None by default")

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests = requests
        batch.generation_requests = []
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

    def test_foward_pass_callable_backward_compat(self):
        config = PyTorchConfig(use_cuda_graph=False)
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(config)
        self.assertTrue(model_engine.forward_pass_callable is None,
                        "forward_pass_callback should be None by default")

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests = requests
        batch.generation_requests = []
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

    @skip_ray
    def test_prepare_tp_inputs_with_helix_parallelism(self) -> None:
        """Test _prepare_tp_inputs function with helix parallelism."""

        # Create model engine with helix parallelism.
        config = PyTorchConfig(use_cuda_graph=False)
        model_engine = DummyModelEngine(config, batch_size=4, dtype=torch.half)

        # Provide mapping for model engine.
        cp_size = 2
        cp_rank = 0
        cp_config = {"cp_type": CpType.HELIX, "tokens_per_block": 4}
        mapping = Mapping(world_size=cp_size,
                          tp_size=1,
                          pp_size=1,
                          cp_size=cp_size,
                          cp_config=cp_config,
                          rank=cp_rank)
        model_engine.mapping = mapping

        # Mock model_engine's dist and its cp_allgather to return different values per CP rank.
        mock_dist = MagicMock()

        def mock_cp_allgather(obj):
            # Simulate allgather across CP ranks: [past_seen_token_num_rank0, past_seen_token_num_rank1]
            if cp_rank == 0:
                return [obj, obj + 10]  # Rank 0 sees tokens [obj, obj+10]
            else:
                return [obj - 10, obj]  # Rank 1 sees tokens [obj-10, obj]

        mock_dist.cp_allgather.side_effect = mock_cp_allgather
        model_engine.dist = mock_dist

        # Create scheduled requests with two generation requests.
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests = []
        prompt_lens = [20, 15]
        gen_requests = []
        for idx in range(len(prompt_lens)):
            req = _create_request(num_tokens=prompt_lens[idx], req_id=idx + 1)
            req.py_prompt_len = prompt_lens[idx]
            req.py_batch_idx = None
            req.is_dummy_request = False
            req.py_seq_slot = idx
            req.sampling_config.beam_width = 1
            req.py_multimodal_data = {}
            gen_requests.append(req)
        scheduled_requests.generation_requests = gen_requests

        # Create KV cache manager for attention metadata.
        kv_cache_config = KvCacheConfig(max_tokens=512)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=16,
            head_dim=16,
            tokens_per_block=1,
            max_seq_len=512,
            max_batch_size=4,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF,
        )
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=512,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        # Initialize model engine buffers.
        max_num_tokens = 512
        model_engine.max_num_tokens = max_num_tokens
        model_engine.input_ids_cuda = torch.zeros(max_num_tokens,
                                                  dtype=torch.int32,
                                                  device='cuda')
        model_engine.position_ids_cuda = torch.zeros(max_num_tokens,
                                                     dtype=torch.int32,
                                                     device='cuda')
        model_engine.previous_batch_indices_cuda = torch.zeros(
            max_num_tokens, dtype=torch.int32, device='cuda')

        result, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata)

        # Verify expected keys are present.
        self.assertIsNotNone(result)
        self.assertIn('input_ids', result)
        self.assertIn('position_ids', result)
        self.assertIn('attn_metadata', result)

        # Check that cp_allgather was called for position calculation.
        # Also, verify that position_ids are properly calculated.
        self.assertTrue(mock_dist.cp_allgather.called)
        position_ids = result['position_ids']
        self.assertIsInstance(position_ids, torch.Tensor)
        # For cp_rank=0, the expected position_ids should be:
        # req1: past_seen_token_num=19 (prompt_len0-1), allgather=[19, 29], sum=48.
        # req2: past_seen_token_num=14 (prompt_len1-1), allgather=[14, 24], sum=38.
        expected_positions = [48, 38]
        actual_positions = position_ids.squeeze(0).cpu().tolist()[:2]
        self.assertEqual(
            actual_positions, expected_positions,
            f"Position IDs should reflect CP allgather results. Expected: {expected_positions}, Got: {actual_positions}"
        )

        # Verify attention metadata is properly configured.
        self.assertEqual(attn_metadata.request_ids, [1, 2])
        self.assertEqual(attn_metadata.prompt_lens, [20, 15])
        self.assertEqual(attn_metadata.num_contexts, 0)

        # Verify KV cache parameters
        self.assertIsNotNone(attn_metadata.kv_cache_params)
        self.assertTrue(attn_metadata.kv_cache_params.use_cache)

        # Verify sequence lengths are correct.
        expected_seq_lens = [1, 1]
        if hasattr(attn_metadata,
                   'seq_lens') and attn_metadata.seq_lens is not None:
            actual_seq_lens = attn_metadata.seq_lens.cpu().tolist()
            self.assertEqual(actual_seq_lens, expected_seq_lens)


if __name__ == "__main__":
    unittest.main()
