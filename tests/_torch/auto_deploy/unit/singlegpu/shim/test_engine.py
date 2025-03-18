from typing import Type

import pytest
import torch
import torch.nn as nn

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

    def forward(self, input_ids, *cache_args):
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
    "attn_backend, page_size", [("TritonWithFlattenedInputs", 0), ("FlashInfer", 2)]
)
def test_engine(engine_cls: Type[ADEngine], attn_backend: str, page_size: int):
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
        page_size=page_size,
    )
    sequence_info.to(device)

    engine = engine_cls(get_inference_model, sequence_info, device)

    # Test basic token generation
    with torch.inference_mode():
        # Test logits
        input_ids = [torch.tensor([0, 1, 2], device=device)]
        sequence_info.reset()
        sequence_info.nest_sequences(input_ids)
        engine.cache_seq_interface.info.sync(sequence_info)
        logits = engine._compute_logits()
        logits = torch.stack(logits)
        assert logits is not None, "Logits are None"

        mock_input = None
        original_logits = get_inference_model(mock_input)(input_ids[0].unsqueeze(0))[0]
        assert torch.allclose(logits, original_logits, atol=1e-5), "Generated Token ID mismatch"
