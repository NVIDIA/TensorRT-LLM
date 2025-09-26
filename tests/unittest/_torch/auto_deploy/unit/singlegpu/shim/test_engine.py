from typing import Type, Union

import pytest
import torch
import torch.nn as nn

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine
from tensorrt_llm._torch.auto_deploy.shim.demollm import DemoEngine
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface


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

    def forward(self, cm_or_input_ids: Union[CachedSequenceInterface, torch.Tensor]):
        if isinstance(cm_or_input_ids, CachedSequenceInterface):
            input_ids = cm_or_input_ids.args[0]
        else:
            input_ids = cm_or_input_ids
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
@pytest.mark.parametrize(
    "attn_backend, attn_page_size", [("triton", 0), ("flashinfer", 2), ("torch", 0)]
)
def test_engine(engine_cls: Type[ADEngine], attn_backend: str, attn_page_size: int):
    """Test the SimpleEngine functionality."""

    seed = 1234  # Set random seed for model param init
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
