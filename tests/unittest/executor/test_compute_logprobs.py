"""Unit tests for tensorrt_llm.executor.result.compute_logprobs.

These tests pin the numerical contract of ``compute_logprobs`` (values, ranks,
dict shape) against a straightforward per-token reference implementation, so
that the vectorized fast path cannot silently drift from the original
semantics.

``compute_logprobs`` moves its input to CUDA internally, so these tests need a
GPU.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm.executor.result import Logprob, compute_logprobs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compute_logprobs runs on CUDA"
)


def _reference_topk_logprobs(
    logits: torch.Tensor, top_k: int, tokens: list[int] | None, simple: bool
):
    """Per-token reference implementation (slow but obviously correct)."""
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if tokens is not None and logits.size(0) > len(tokens):
        logits = logits[: len(tokens)]
    logprobs = F.log_softmax(logits.to("cuda", dtype=torch.float32), dim=-1)

    if top_k == 0:
        if simple:
            return (
                []
                if tokens is None
                else [logprobs[t, tokens[t]].item() for t in range(logprobs.size(0))]
            )
        results = []
        if tokens is not None:
            for t in range(logprobs.size(0)):
                token_id = tokens[t]
                token_logprob = logprobs[t, token_id].item()
                rank = (logprobs[t] > token_logprob).sum().item() + 1
                results.append({token_id: Logprob(logprob=token_logprob, rank=rank)})
        return results

    topk_vals, topk_indices = torch.topk(logprobs, k=top_k, dim=-1)
    results = []
    for t in range(logprobs.size(0)):
        token_dict = {
            idx.item(): Logprob(logprob=val.item(), rank=r + 1)
            for r, (val, idx) in enumerate(zip(topk_vals[t], topk_indices[t]))
        }
        if tokens is not None:
            token_id = tokens[t]
            if token_id not in token_dict:
                token_logprob = logprobs[t, token_id].item()
                rank = (logprobs[t] > token_logprob).sum().item() + 1
                token_dict[token_id] = Logprob(logprob=token_logprob, rank=rank)
        results.append(token_dict)
    return results


@pytest.mark.parametrize("num_tokens,vocab_size", [(1, 64), (17, 257), (64, 1024)])
@pytest.mark.parametrize("top_k", [0, 1, 5])
@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("tokens_in_topk", [False, True])
@pytest.mark.parametrize("batched_logits", [False, True])
def test_generation_logprobs_match_reference(
    num_tokens, vocab_size, top_k, simple, tokens_in_topk, batched_logits
):
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, vocab_size)
    if tokens_in_topk:
        tokens = logits.argmax(dim=-1).tolist()
    else:
        tokens = logits.argmin(dim=-1).tolist()
    if batched_logits:
        logits = logits.unsqueeze(0)

    expected = _reference_topk_logprobs(logits, top_k, tokens, simple)
    actual = compute_logprobs(None, top_k, None, logits, tokens, simple_logprobs=simple).generation
    assert actual == expected


@pytest.mark.parametrize("top_k", [0, 3])
@pytest.mark.parametrize("simple", [False, True])
def test_prompt_logprobs_match_reference(top_k, simple):
    torch.manual_seed(1)
    logits = torch.randn(37, 512)
    tokens = torch.randint(0, 512, (37,)).tolist()

    expected = _reference_topk_logprobs(logits, top_k, tokens, simple)
    actual = compute_logprobs(
        top_k, None, logits, None, None, tokens, simple_prompt_logprobs=simple
    ).prompt
    assert actual == expected


def test_more_tokens_than_logits_is_truncated():
    """WAR for nvbug 5324291: logits may be shorter than the token list."""
    torch.manual_seed(2)
    logits = torch.randn(4, 128)
    tokens = torch.randint(0, 128, (9,)).tolist()

    result = compute_logprobs(None, 0, None, logits, tokens, simple_logprobs=True).generation
    assert result == _reference_topk_logprobs(logits, 0, tokens, True)
    assert len(result) == 4


def test_no_tokens_returns_topk_only():
    torch.manual_seed(3)
    logits = torch.randn(5, 64)

    result = compute_logprobs(None, 3, None, logits, None).generation
    assert result == _reference_topk_logprobs(logits, 3, None, False)
    assert all(len(d) == 3 for d in result)
