# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SFP_MC_FI on an SM107 GPU: ``sfp_mc_fi.sampling_from_probs_mc_fi`` returns FlashInfer's
``sampling_from_probs(deterministic=True)`` tokens bit for bit (int64 seed / offset tensors), on spread and
frequency-skewed rows at DEP's row counts, with the drawn token in the first or the last chunk, on FlashInfer's
fallbacks (rows summing below u, all-zero rows), on padding rows, on denormal probabilities, at every ctas_per_row and
under CUDA-graph replay."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    sfp_mc_fi,
)


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="SM107 (Rubin) only")

VOCAB = 200064
CHUNK = 4096


def _probs(rows, seed, mode="spread", temp=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, VOCAB, generator=gen, device="cuda") * 2.0
    if mode == "skewed":
        logits -= 0.3 * torch.log1p(torch.arange(VOCAB, device="cuda", dtype=torch.float32))
    elif mode in ("first", "last"):
        lo = 0 if mode == "first" else (VOCAB - 1) // CHUNK * CHUNK
        keep = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
        keep[lo : lo + CHUNK] = True
        logits = logits.masked_fill(~keep, float("-inf"))
    return torch.softmax(logits / temp, -1).contiguous()


def _seeds(rows, k):
    gen = torch.Generator(device="cuda").manual_seed(1000 + k)
    s = torch.randint(0, 2**62, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    o = torch.randint(0, 2**40, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    return s, o


def _flashinfer(probs, s, o):
    import flashinfer

    return flashinfer.sampling.sampling_from_probs(
        probs, deterministic=True, check_nan=False, generator=None, seed=s, offset=o
    ).to(torch.int32)


def _assert_same(probs, s, o):
    got = sfp_mc_fi.sampling_from_probs_mc_fi(probs, s, o)
    assert got is not None
    assert torch.equal(got, _flashinfer(probs, s, o))


@pytest.mark.parametrize("rows", [1, 8, 12, 16, 24, 48])
@pytest.mark.parametrize("mode", ["spread", "skewed"])
def test_bitwise_flashinfer_at_deps_row_counts(rows, mode):
    for k in range(6):
        probs = _probs(rows, 10 * rows + k, mode, temp=(0.6, 0.8, 1.0)[k % 3])
        _assert_same(probs, *_seeds(rows, k))


@pytest.mark.parametrize("mode", ["first", "last"])
def test_the_drawn_token_in_the_first_or_the_last_chunk(mode):
    probs = _probs(12, 500 + len(mode), mode)
    for k in range(4):
        _assert_same(probs, *_seeds(12, 100 + k))


def test_flashinfers_fallbacks():
    base = _probs(12, 600)
    half = (base * 0.5).contiguous()
    zero = base.clone()
    zero[[2, 5, 9]] = 0.0
    last = torch.zeros_like(base)
    last[:, VOCAB - 1] = 1.0
    for probs in (half, zero, last):
        for k in range(4):
            _assert_same(probs, *_seeds(12, 200 + k))
    got = sfp_mc_fi.sampling_from_probs_mc_fi(zero, *_seeds(12, 0))
    assert bool((got[[2, 5, 9]] == 0).all())


@pytest.mark.parametrize("mode", ["onehot", "uniform"])
def test_padding_rows(mode):
    probs = _probs(12, 700 + len(mode))
    pad = torch.tensor([3, 7, 10], device="cuda")
    if mode == "onehot":
        probs[pad] = 0.0
        probs[pad, 0] = 1.0
    else:
        probs[pad] = 1.0 / VOCAB
    for k in range(4):
        _assert_same(probs, *_seeds(12, 300 + k))


def test_denormal_probabilities():
    probs = _probs(12, 800)
    gen = torch.Generator(device="cuda").manual_seed(4321)
    probs[torch.rand(probs.shape, generator=gen, device="cuda") < 0.01] = 1.0e-39
    rows = torch.arange(12, device="cuda")
    top = probs.argmax(-1)
    probs[rows, (top + 1).clamp(max=VOCAB - 1)] = 1.0e-39
    probs[rows, (top - 1).clamp(min=0)] = 2.0e-39
    for k in range(4):
        _assert_same(probs, *_seeds(12, 400 + k))


@pytest.mark.parametrize("rows", [1, 12, 48])
def test_the_output_does_not_depend_on_ctas_per_row(rows):
    probs = _probs(rows, 55)
    ws = sfp_mc_fi.prebuild(torch.device("cuda", torch.cuda.current_device()))
    op = sfp_mc_fi.sfp_mc_fi_op()
    s, o = _seeds(rows, 5)
    ref = _flashinfer(probs, s, o)
    for ctas in (1, 2, 4, 8, 16, 32, 64):
        assert torch.equal(op(probs, s, o, ws, ctas), ref)


def test_the_override_replays_inside_a_cuda_graph():
    probs = _probs(12, 66)
    s, o = _seeds(12, 6)
    sfp_mc_fi.sampling_from_probs_mc_fi(probs, s, o)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sfp_mc_fi.sampling_from_probs_mc_fi(probs, s, o)
    for k in range(3):
        new = _probs(12, 600 + k, "skewed")
        probs.copy_(new)
        ns, no = _seeds(12, 60 + k)
        s.copy_(ns)
        o.copy_(no)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _flashinfer(new, ns, no))
