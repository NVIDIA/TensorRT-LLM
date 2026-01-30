from typing import List, Optional, Type

import pytest
import torch
import torch.nn as nn

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine
from tensorrt_llm._torch.auto_deploy.shim.demollm import DemoEngine
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig


class TransformerLikeModelwithFakeCachePool(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TransformerLikeModelwithFakeCachePool, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        embeddings = self.embedding(input_ids)
        hidden_states = self.mlp(embeddings)
        logits = self.output_projection(hidden_states)
        return [logits]


def get_inference_model(cache_seq_interface):
    vocab_size = 128
    embed_dim = 32
    hidden_dim = 64
    device = "cuda"

    model = TransformerLikeModelwithFakeCachePool(vocab_size, embed_dim, hidden_dim)
    model.eval().to(device)

    return model


@pytest.mark.parametrize("engine_cls", [ADEngine, DemoEngine])
@pytest.mark.parametrize("tokens_per_block", [0, 2, 0])
def test_engine(engine_cls: Type[ADEngine], tokens_per_block: int):
    """Test the SimpleEngine functionality."""

    seed = 42  # Set random seed for model param init
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8

    # Create KvCacheConfig with specified tokens_per_block (use 32 as default if 0)
    kv_cache_config = KvCacheConfig(tokens_per_block=tokens_per_block or max_seq_len)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        kv_cache_config=kv_cache_config,
    )
    cache_seq_interface.to(device)

    engine = engine_cls(get_inference_model, cache_seq_interface)

    # Test basic token generation
    with torch.inference_mode():
        # Test logits
        input_ids = [torch.tensor([0, 1, 2], device=device)]
        cache_seq_interface.info.reset()
        cache_seq_interface.info.nest_sequences(input_ids)
        logits = engine._compute_logits()
        assert logits is not None, "Logits are None"

        mock_input = None
        original_logits = get_inference_model(mock_input)(input_ids[0].unsqueeze(0))[0]
        assert torch.allclose(logits, original_logits, atol=1e-5), "Generated Token ID mismatch"

    cache_seq_interface.shutdown()


@pytest.mark.parametrize("tokens_per_block", [0, 2])
def test_demo_engine_sampling(tokens_per_block: int):
    """Test sampling logic specific to DemoEngine."""
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8

    # Create KvCacheConfig with specified tokens_per_block (use 32 as default if 0)
    kv_cache_config = KvCacheConfig(tokens_per_block=tokens_per_block or 32)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        kv_cache_config=kv_cache_config,
    )
    cache_seq_interface.to(device)

    engine = DemoEngine(get_inference_model, cache_seq_interface)

    with torch.inference_mode():
        input_ids = [torch.tensor([1, 2, 3, 4], device=device)]
        cache_seq_interface.info.reset()
        cache_seq_interface.info.nest_sequences(input_ids)
        logits = engine._compute_logits()

        vocab_size = logits.size(-1)
        sampling_params = SamplingParams(top_k=5, temperature=1.0)

        token_ids, _ = engine._sample(logits, sampling_params)
        expected_shape = logits.shape[:-1]

        assert token_ids.shape == expected_shape, (
            f"Unexpected shape for sampled token IDs, expected {expected_shape}, but got {token_ids.shape}"
        )
        assert torch.all((token_ids >= 0) & (token_ids < vocab_size)), (
            "Sampled indices out of range"
        )

        # Test that top_k=1 (greedy) matches top_k=None (argmax fallback)
        sampling_params_greedy = SamplingParams(top_k=1)
        sampling_params_none = SamplingParams(top_k=None)

        token_ids_1, _ = engine._sample(logits, sampling_params_greedy)
        token_ids_2, _ = engine._sample(logits, sampling_params_none)

        torch.testing.assert_close(token_ids_1, token_ids_2)

    cache_seq_interface.shutdown()


class _DummyKVCacheManager:
    def __init__(self, tokens_per_block: int):
        self.tokens_per_block = tokens_per_block

    def get_cache_indices(self, request):
        # Return many dummy page IDs; ADEngine will truncate as needed
        return list(range(1024))

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        if self.tokens_per_block and self.tokens_per_block > 0:
            return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block
        return num_tokens


class _DummyResourceManager:
    def __init__(self, kv_cache_manager: _DummyKVCacheManager):
        self._kv = kv_cache_manager

    def get_resource_manager(self, _):
        return self._kv


class _DummyRequest:
    def __init__(self, tokens: List[int], begin: int, size: int, seq_slot: int = 0):
        self._tokens = tokens
        self.context_current_position = begin
        self.context_chunk_size = size
        self.seq_slot = seq_slot
        self.py_batch_idx = None
        self.py_multimodal_data = None

    def get_tokens(self, _beam: int) -> List[int]:
        return self._tokens


@pytest.mark.parametrize("tokens_per_block", [256, 2])
def test_ad_engine_chunked_prefill_equivalence(tokens_per_block: int):
    """Verify ADEngine logits match between chunked and non-chunked prefill.

    We simulate chunking by splitting a single context request into two chunks and
    compare the final-step logits to processing the full prompt at once.
    """
    seed = 123
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8

    # Create KvCacheConfig with specified tokens_per_block
    kv_cache_config = KvCacheConfig(tokens_per_block=tokens_per_block)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        kv_cache_config=kv_cache_config,
    )
    cache_seq_interface.to(device)

    engine = ADEngine(get_inference_model, cache_seq_interface)

    # A simple prompt; model is position-wise so last token dominates the last logit
    tokens = [1, 2, 3, 4, 5, 6]

    kv_manager = _DummyKVCacheManager(tokens_per_block=tokens_per_block)
    resource_manager = _DummyResourceManager(kv_manager)

    # No-chunk: whole prompt in one request
    req_full = _DummyRequest(tokens=tokens, begin=0, size=len(tokens), seq_slot=0)
    scheduled_requests = ScheduledRequests()
    scheduled_requests.context_requests.append(req_full)
    logits_full_last = engine.forward(scheduled_requests, resource_manager)["logits"][-1]

    # Chunked: split into two context chunks
    split = len(tokens) // 2
    req_part1 = _DummyRequest(tokens=tokens, begin=0, size=split, seq_slot=0)
    req_part2 = _DummyRequest(tokens=tokens, begin=split, size=len(tokens) - split, seq_slot=0)

    scheduled_requests_part1 = ScheduledRequests()
    scheduled_requests_part1.context_requests.append(req_part1)
    scheduled_requests_part2 = ScheduledRequests()
    scheduled_requests_part2.context_requests.append(req_part2)

    # Run first chunk (ignored output), then compare second chunk logits to full
    _ = engine.forward(scheduled_requests_part1, resource_manager)
    logits_chunked_last = engine.forward(scheduled_requests_part2, resource_manager)["logits"][-1]

    torch.testing.assert_close(logits_full_last, logits_chunked_last)  # , atol=1e-5)

    cache_seq_interface.shutdown()


# =============================================================================
# Hybrid Cache Manager Integration Tests
# =============================================================================


class _DummyHybridKVCacheManager:
    """Simulates MambaHybridCacheManager with mamba_cache_index."""

    def __init__(self, tokens_per_block: int, num_slots: int = 8):
        self.tokens_per_block = tokens_per_block
        # mamba_cache_index maps request_id to slot_idx
        self.mamba_cache_index = {i: num_slots - 1 - i for i in range(num_slots)}
        self.mamba_cache_free_blocks = num_slots

    def get_cache_indices(self, request):
        return list(range(1024))

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        if self.tokens_per_block and self.tokens_per_block > 0:
            return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block
        return num_tokens

    def get_num_free_blocks(self):
        return 100


class _DummyRequestWithRequestId:
    """Request with py_request_id for hybrid cache manager testing."""

    def __init__(
        self,
        tokens: List[int],
        begin: int,
        size: int,
        seq_slot: int = 0,
        request_id: int = 0,
    ):
        self._tokens = tokens
        self.context_current_position = begin
        self.context_chunk_size = size
        self.seq_slot = seq_slot
        self.py_request_id = request_id
        self.py_batch_idx = None
        self.py_multimodal_data = None

    def get_tokens(self, _beam: int) -> List[int]:
        return self._tokens


def test_ad_engine_prepare_inputs_with_hybrid_cache_manager():
    """Test ADEngine _prepare_inputs uses mamba_cache_index when available."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8
    tokens_per_block = 16

    kv_cache_config = KvCacheConfig(tokens_per_block=tokens_per_block)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        kv_cache_config=kv_cache_config,
    )
    cache_seq_interface.to(device)

    engine = ADEngine(get_inference_model, cache_seq_interface)

    # Create hybrid KV cache manager with specific mamba_cache_index mapping
    hybrid_manager = _DummyHybridKVCacheManager(tokens_per_block=tokens_per_block)

    class _HybridResourceManager:
        def __init__(self, kv_mgr):
            self._kv = kv_mgr

        def get_resource_manager(self, _):
            return self._kv

    resource_manager = _HybridResourceManager(hybrid_manager)

    # Create request with specific request_id
    request_id = 3
    tokens = [1, 2, 3, 4]
    req = _DummyRequestWithRequestId(
        tokens=tokens,
        begin=0,
        size=len(tokens),
        seq_slot=0,
        request_id=request_id,
    )

    scheduled = ScheduledRequests()
    scheduled.context_requests.append(req)

    # Call _prepare_inputs
    engine._prepare_inputs(scheduled, resource_manager, new_tokens=None)

    # Verify slot_idx was taken from mamba_cache_index, not seq_slot
    expected_slot_idx = hybrid_manager.mamba_cache_index[request_id]
    actual_slot_idx = cache_seq_interface.info._args_list["slot_idx"][0]
    assert actual_slot_idx == expected_slot_idx

    cache_seq_interface.shutdown()


def test_ad_engine_prepare_inputs_generation_with_hybrid_cache():
    """Test ADEngine _prepare_inputs handles generation requests with hybrid cache."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8
    tokens_per_block = 16

    kv_cache_config = KvCacheConfig(tokens_per_block=tokens_per_block)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        kv_cache_config=kv_cache_config,
    )
    cache_seq_interface.to(device)

    engine = ADEngine(get_inference_model, cache_seq_interface)

    # Create hybrid KV cache manager
    hybrid_manager = _DummyHybridKVCacheManager(tokens_per_block=tokens_per_block)

    class _HybridResourceManager:
        def __init__(self, kv_mgr):
            self._kv = kv_mgr

        def get_resource_manager(self, _):
            return self._kv

    resource_manager = _HybridResourceManager(hybrid_manager)

    # Create generation request
    class _GenRequest:
        def __init__(self, request_id: int, seq_slot: int, num_tokens: int):
            self.py_request_id = request_id
            self.seq_slot = seq_slot
            self.py_batch_idx = None
            self.is_dummy = False
            self.py_draft_tokens = []

            # Mock methods for generation request
            def get_token(beam, idx):
                return 42  # Dummy token

            self.get_token = get_token
            self.get_num_tokens = lambda beam: num_tokens
            self.max_beam_num_tokens = num_tokens

        def get_draft_token_length(self):
            return 0

    # Create a generation request with specific request_id
    request_id = 2
    gen_req = _GenRequest(request_id=request_id, seq_slot=5, num_tokens=10)

    scheduled = ScheduledRequests()
    scheduled.generation_requests.append(gen_req)

    # Call _prepare_inputs
    engine._prepare_inputs(scheduled, resource_manager, new_tokens=None)

    # Verify slot_idx was taken from mamba_cache_index
    expected_slot_idx = hybrid_manager.mamba_cache_index[request_id]
    actual_slot_idx = cache_seq_interface.info._args_list["slot_idx"][0]
    assert actual_slot_idx == expected_slot_idx

    cache_seq_interface.shutdown()


def test_ad_engine_with_regular_kv_cache_manager():
    """Test ADEngine falls back to seq_slot when mamba_cache_index not available."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8
    tokens_per_block = 16

    kv_cache_config = KvCacheConfig(tokens_per_block=tokens_per_block)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        kv_cache_config=kv_cache_config,
    )
    cache_seq_interface.to(device)

    engine = ADEngine(get_inference_model, cache_seq_interface)

    # Use regular (non-hybrid) KV cache manager without mamba_cache_index
    regular_manager = _DummyKVCacheManager(tokens_per_block=tokens_per_block)
    resource_manager = _DummyResourceManager(regular_manager)

    # Create request with specific seq_slot
    expected_seq_slot = 3
    tokens = [1, 2, 3, 4]
    req = _DummyRequest(
        tokens=tokens,
        begin=0,
        size=len(tokens),
        seq_slot=expected_seq_slot,
    )

    scheduled = ScheduledRequests()
    scheduled.context_requests.append(req)

    # Call _prepare_inputs
    engine._prepare_inputs(scheduled, resource_manager, new_tokens=None)

    # Verify slot_idx falls back to seq_slot
    actual_slot_idx = cache_seq_interface.info._args_list["slot_idx"][0]
    assert actual_slot_idx == expected_seq_slot

    cache_seq_interface.shutdown()
