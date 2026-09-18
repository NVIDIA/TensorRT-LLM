"""Split-row Triton kernels behind Fusions.gather_log_softmax_with_output / determine_sampled_rank
must match the torch reference at fp32 level, including padded -inf vocab columns,
non-contiguous logits views, int32/int64 indices and 1-row batches."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops import logprobs_triton
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import Fusions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _logits(rows, vocab, pad=0, generator=None):
    x = torch.randn(rows, vocab, generator=generator, device="cuda") * 3.0
    x[:, :5] += 8.0  # a few peaked tokens
    if pad:
        x[:, vocab - pad :] = float("-inf")
    return x


@pytest.mark.parametrize("rows_total,m", [(1, 1), (2, 1), (13, 13), (27, 9), (64, 64)])
@pytest.mark.parametrize("vocab,pad", [(4096, 0), (248320, 0), (248320, 37)])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_gather_log_softmax_matches_torch(rows_total, m, vocab, pad, index_dtype):
    g = torch.Generator(device="cuda").manual_seed(rows_total * 1000 + vocab + pad)
    logits = _logits(rows_total, vocab, pad, g)
    idx = torch.randperm(rows_total, generator=g, device="cuda")[:m].to(index_dtype)
    out = torch.empty((m, vocab), dtype=torch.float32, device="cuda")
    Fusions.gather_log_softmax_with_output(logits, idx, out=out)
    ref = torch.log_softmax(logits[idx.long()].double(), dim=-1)
    assert out.dtype == torch.float32 and out.shape == (m, vocab) and out.is_contiguous()
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out), finite)
    assert (out.double()[finite] - ref[finite]).abs().max().item() < 1e-5
    # the token logprobs that RL consumes: tight tolerance
    tok = torch.randint(0, vocab - pad, (m,), generator=g, device="cuda")
    got = out[torch.arange(m, device="cuda"), tok].double()
    exp = ref[torch.arange(m, device="cuda"), tok]
    assert (got - exp).abs().max().item() < 5e-6


def test_gather_log_softmax_non_contiguous_view():
    g = torch.Generator(device="cuda").manual_seed(7)
    wide = torch.randn(16, 4096 + 64, generator=g, device="cuda")
    view = wide[:, :4096]  # row stride 4160 elements: not a contiguous matrix
    idx = torch.arange(16, device="cuda")
    out = torch.empty((16, 4096), dtype=torch.float32, device="cuda")
    Fusions.gather_log_softmax_with_output(view, idx, out=out)
    ref = torch.log_softmax(view.double(), dim=-1)
    assert (out.double() - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("rows,vocab", [(1, 4096), (13, 248320), (32, 248320)])
def test_determine_sampled_rank_matches_reference(rows, vocab):
    g = torch.Generator(device="cuda").manual_seed(rows + vocab)
    logprobs = torch.log_softmax(_logits(rows, vocab, 0, g), dim=-1)
    tok = torch.randint(0, vocab, (rows,), generator=g, device="cuda")
    sampled = logprobs[torch.arange(rows, device="cuda"), tok].unsqueeze(-1)
    out = Fusions.determine_sampled_rank(logprobs, sampled)
    ref = logprobs.greater(sampled).count_nonzero(dim=-1).to(torch.int32)
    assert out.dtype == torch.int32 and out.shape == (rows,)
    assert torch.equal(out, ref)
    # the sampled token itself is never counted, and the top token has rank 0
    top = logprobs.argmax(dim=-1)
    top_rank = Fusions.determine_sampled_rank(
        logprobs, logprobs[torch.arange(rows, device="cuda"), top].unsqueeze(-1)
    )
    assert torch.equal(top_rank, torch.zeros(rows, dtype=torch.int32, device="cuda"))


def test_kernels_are_deterministic():
    g = torch.Generator(device="cuda").manual_seed(3)
    logits = _logits(24, 248320, 0, g)
    idx = torch.arange(24, device="cuda")
    a = logprobs_triton.gather_log_softmax(logits, idx)
    b = logprobs_triton.gather_log_softmax(logits, idx)
    assert torch.equal(a, b)
