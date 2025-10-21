from types import SimpleNamespace
from typing import List, Optional, Type

import pytest
import torch
import torch.nn as nn

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine
from tensorrt_llm._torch.auto_deploy.shim.demollm import DemoEngine


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
@pytest.mark.parametrize("attn_page_size", [0, 2, 0])
def test_engine(engine_cls: Type[ADEngine], attn_page_size: int):
    """Test the SimpleEngine functionality."""

    seed = 42  # Set random seed for model param init
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8

    sequence_info = SequenceInfo(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        page_size=attn_page_size,
    )
    sequence_info.to(device)

    engine = engine_cls(get_inference_model, sequence_info, device)

    # Test basic token generation
    with torch.inference_mode():
        # Test logits
        input_ids = [torch.tensor([0, 1, 2], device=device)]
        sequence_info.reset()
        sequence_info.nest_sequences(input_ids)
        logits = engine._compute_logits()
        logits = torch.stack(logits)
        assert logits is not None, "Logits are None"

        mock_input = None
        original_logits = get_inference_model(mock_input)(input_ids[0].unsqueeze(0))[0]
        assert torch.allclose(logits, original_logits, atol=1e-5), "Generated Token ID mismatch"


@pytest.mark.parametrize("attn_page_size", [0, 2])
def test_demo_engine_sampling(attn_page_size: int):
    """Test sampling logic specific to DemoEngine."""
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    max_seq_len = 64
    max_batch_size = 8

    sequence_info = SequenceInfo(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        page_size=attn_page_size,
    )
    sequence_info.to(device)

    engine = DemoEngine(get_inference_model, sequence_info, device)

    with torch.inference_mode():
        input_ids = [torch.tensor([1, 2, 3, 4], device=device)]
        sequence_info.reset()
        sequence_info.nest_sequences(input_ids)
        logits = engine._compute_logits()
        logits = torch.stack(logits)

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


@pytest.mark.parametrize("attn_page_size", [256, 2])
def test_ad_engine_chunked_prefill_equivalence(attn_page_size: int):
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

    sequence_info = SequenceInfo(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        page_size=attn_page_size,
    )
    sequence_info.to(device)

    engine = ADEngine(get_inference_model, sequence_info, device)

    # A simple prompt; model is position-wise so last token dominates the last logit
    tokens = [1, 2, 3, 4, 5, 6]

    kv_manager = _DummyKVCacheManager(tokens_per_block=attn_page_size)
    resource_manager = _DummyResourceManager(kv_manager)

    # No-chunk: whole prompt in one request
    req_full = _DummyRequest(tokens=tokens, begin=0, size=len(tokens), seq_slot=0)
    scheduled_full = SimpleNamespace(context_requests=[req_full], generation_requests=[])
    logits_full = engine.forward(scheduled_full, resource_manager)["logits"]

    # Chunked: split into two context chunks
    split = len(tokens) // 2
    req_part1 = _DummyRequest(tokens=tokens, begin=0, size=split, seq_slot=0)
    req_part2 = _DummyRequest(tokens=tokens, begin=split, size=len(tokens) - split, seq_slot=0)

    scheduled_part1 = SimpleNamespace(context_requests=[req_part1], generation_requests=[])
    scheduled_part2 = SimpleNamespace(context_requests=[req_part2], generation_requests=[])

    # Run first chunk (ignored output), then compare second chunk logits to full
    _ = engine.forward(scheduled_part1, resource_manager)
    logits_chunked_last = engine.forward(scheduled_part2, resource_manager)["logits"]

    assert logits_full.shape == logits_chunked_last.shape
    assert torch.allclose(logits_full, logits_chunked_last, atol=1e-5)
