import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    filter_mm_token_from_input_ids, fuse_input_embeds)
from tensorrt_llm._torch.modules.embedding import Embedding


def make_embedding(num_embeddings: int = 100,
                   hidden_size: int = 16,
                   device: str = "cpu") -> Embedding:
    torch.manual_seed(0)
    emb = Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)
    emb.weight.data.normal_(mean=0.0, std=0.02)
    return emb.to(device)


@pytest.mark.parametrize("device", ["cpu"] +
                         (["cuda"] if torch.cuda.is_available() else []))
def test_filter_mm_token_from_input_ids_oov(device):
    vocab_size = 10
    # input_ids contains text (< vocab) and OOV mm tokens (>= vocab)
    input_ids = torch.tensor([1, 2, 11, 3, 12, 9, 15],
                             dtype=torch.long,
                             device=device)
    text_idx, mm_idx = filter_mm_token_from_input_ids(input_ids,
                                                      vocab_size=vocab_size,
                                                      mm_token_ids=None)

    torch.testing.assert_close(text_idx.cpu(),
                               torch.tensor([0, 1, 3, 5], dtype=torch.long))
    torch.testing.assert_close(mm_idx.cpu(),
                               torch.tensor([2, 4, 6], dtype=torch.long))


@pytest.mark.parametrize("device", ["cpu"] +
                         (["cuda"] if torch.cuda.is_available() else []))
def test_filter_mm_token_from_input_ids_explicit_ids(device):
    vocab_size = 100
    # All ids are < vocab; mm tokens are explicitly specified
    input_ids = torch.tensor([1, 2, 55, 3, 77, 9, 88],
                             dtype=torch.long,
                             device=device)
    mm_token_ids = torch.tensor([55, 77, 88], dtype=torch.long)
    text_idx, mm_idx = filter_mm_token_from_input_ids(input_ids,
                                                      vocab_size=vocab_size,
                                                      mm_token_ids=mm_token_ids)

    torch.testing.assert_close(text_idx.cpu(),
                               torch.tensor([0, 1, 3, 5], dtype=torch.long))
    torch.testing.assert_close(mm_idx.cpu(),
                               torch.tensor([2, 4, 6], dtype=torch.long))

    # Even with some ids > vocab, mm indices should still only match the given mm_token_ids
    input_ids = torch.tensor([1, 2, 55, 3, 77, 9, 88, 101, 102, 103],
                             dtype=torch.long,
                             device=device)
    _, mm_idx = filter_mm_token_from_input_ids(input_ids,
                                               vocab_size=vocab_size,
                                               mm_token_ids=mm_token_ids)
    torch.testing.assert_close(mm_idx.cpu(),
                               torch.tensor([2, 4, 6], dtype=torch.long))


@pytest.mark.parametrize("device", ["cpu"] +
                         (["cuda"] if torch.cuda.is_available() else []))
def test_fuse_input_embeds_empty_mm_returns_ids(device):
    emb = make_embedding(num_embeddings=20, hidden_size=8, device=device)
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)

    out_ids, out_embeds = fuse_input_embeds(emb,
                                            input_ids,
                                            mm_embeds=[],
                                            mm_token_ids=None)

    # No mm embeddings => passthrough ids, no embeds fused
    assert out_embeds is None
    torch.testing.assert_close(out_ids, input_ids)


@pytest.mark.parametrize("device", ["cpu"] +
                         (["cuda"] if torch.cuda.is_available() else []))
def test_fuse_input_embeds_mismatch_raises(device):
    emb = make_embedding(num_embeddings=50, hidden_size=8, device=device)

    # Mix text (< vocab) and mm (>= vocab) tokens. Here vocab_size == 50
    input_ids = torch.tensor([1, 51, 2, 52, 3, 53],
                             dtype=torch.long,
                             device=device)

    # Identify indices first to drive the lower-level CUDA fuse directly for mismatch
    text_idx, mm_idx = filter_mm_token_from_input_ids(
        input_ids, vocab_size=emb.num_embeddings)

    # Provide the wrong number of mm embeddings (e.g., one short)
    hidden = 8
    true_mm_count = mm_idx.shape[0]
    wrong_mm = torch.randn(true_mm_count - 1, hidden, device=device)

    with pytest.raises(ValueError, match="Multimodal token count mismatch"):
        fuse_input_embeds(emb,
                          input_ids, [wrong_mm],
                          mm_token_ids=None,
                          text_token_indices=text_idx,
                          mm_token_indices=mm_idx)


@pytest.mark.parametrize("device", ["cpu"] +
                         (["cuda"] if torch.cuda.is_available() else []))
def test_fuse_input_embeds_success_oov_path(device):
    hidden = 8
    emb = make_embedding(num_embeddings=40, hidden_size=hidden, device=device)

    # input ids: mix of text (<40) and mm (>=40)
    input_ids = torch.tensor([0, 1, 41, 2, 42, 3, 43, 4],
                             dtype=torch.long,
                             device=device)

    # Build mm embeddings to match number of OOV positions
    text_idx, mm_idx = filter_mm_token_from_input_ids(
        input_ids, vocab_size=emb.num_embeddings)
    mm_emb = torch.randn(mm_idx.shape[0], hidden, device=device)

    # kwargs path to produce fused embeddings
    out_ids, out_embeds = fuse_input_embeds(emb,
                                            input_ids,
                                            mm_embeds=[mm_emb],
                                            mm_token_ids=None,
                                            text_token_indices=text_idx,
                                            mm_token_indices=mm_idx)
    # integrated filtering path to produce fused embeddings (not kwargs path)
    out_ids_v2, out_embeds_v2 = fuse_input_embeds(emb,
                                                  input_ids,
                                                  mm_embeds=[mm_emb],
                                                  mm_token_ids=None)

    assert out_ids is None
    assert out_embeds is not None
    assert out_embeds.shape == (input_ids.numel(), hidden)

    # Validate that text positions equal embedding lookup, and mm positions equal provided mm_emb
    text_idx, mm_idx2 = filter_mm_token_from_input_ids(
        input_ids, vocab_size=emb.num_embeddings)
    torch.testing.assert_close(mm_idx2, mm_idx)

    ref_text = emb(input_ids[text_idx])
    torch.testing.assert_close(out_embeds[text_idx], ref_text)
    torch.testing.assert_close(
        out_embeds[mm_idx],
        mm_emb.to(dtype=out_embeds.dtype, device=out_embeds.device))
    torch.testing.assert_close(out_embeds_v2, out_embeds)
    torch.testing.assert_close(out_ids_v2, out_ids)


@pytest.mark.parametrize("device", ["cpu"] +
                         (["cuda"] if torch.cuda.is_available() else []))
def test_fuse_input_embeds_kwargs_precedence_over_sentinel_and_ids(device):
    """
    Ensure that when kwargs provide precomputed indices, they take precedence
    over both OOV-sentinel filtering and explicit mm_token_ids.
    """
    hidden = 8
    vocab_size = 40
    emb = make_embedding(num_embeddings=vocab_size,
                         hidden_size=hidden,
                         device=device)

    # Use vocab_size+1 as OOV sentinel
    oov_sentinel = vocab_size + 1
    input_ids = torch.tensor([0, oov_sentinel, 1, oov_sentinel, 2],
                             dtype=torch.long,
                             device=device)

    # Precompute correct indices (kwargs path)
    text_idx, mm_idx = filter_mm_token_from_input_ids(input_ids,
                                                      vocab_size=vocab_size,
                                                      mm_token_ids=None)
    mm_emb = torch.randn(mm_idx.shape[0], hidden, device=device)

    # Provide a deliberately incorrect mm_token_ids to ensure it is ignored
    bad_mm_token_ids = torch.tensor(
        [0], dtype=torch.long,
        device=device)  # would misclassify index 0 as mm if used

    out_ids, out_embeds = fuse_input_embeds(
        emb,
        input_ids,
        mm_embeds=[mm_emb],
        mm_token_ids=
        bad_mm_token_ids,  # should be ignored because indices are provided
        text_token_indices=text_idx,
        mm_token_indices=mm_idx,
    )

    # Validate outputs
    assert out_ids is None
    assert out_embeds is not None
    assert out_embeds.shape == (input_ids.numel(), hidden)

    ref_text = emb(input_ids[text_idx])
    torch.testing.assert_close(out_embeds[text_idx], ref_text)
    torch.testing.assert_close(
        out_embeds[mm_idx],
        mm_emb.to(dtype=out_embeds.dtype, device=out_embeds.device),
    )
