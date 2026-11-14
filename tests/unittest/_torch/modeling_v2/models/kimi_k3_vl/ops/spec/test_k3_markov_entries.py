# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark greedy Markov chain entries (``ops/spec``) over their certified cells.

The one-launch chains run as one rank of the helix group on one GPU: the other ranks' pairs
(``k3_markov_chain``) or per-CTA keys (``k3_dspark_markov_greedy``) of every block position are
written into the exchange before the launch, computed from the reference chain, so the kernel
polls the group's exchange as it does on the deployment and reduces it with its own rank's.
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec import (  # noqa: E402, E501
    _reference,
    k3_dspark_markov_greedy,
    k3_fused_markov_chain_logits,
    k3_markov_chain,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

VOCAB = 163840
MARKOV_RANK = 256
_ARCH = "sm_{}{}".format(*torch.cuda.get_device_capability())
_U32 = 0xFFFFFFFF


def _require_arch(op) -> None:
    if _ARCH not in {a.value for a in op.ARCHS}:
        pytest.skip(f"{type(op).__name__} is not certified on {_ARCH}")


def _gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _weights(gen) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``markov_w1`` / ``markov_w2`` ``[vocab, R]`` with entries of 0.25: a bias of unit
    scale, of the order of the logits' spread, so that it moves the argmax."""
    w = 0.25 * torch.randn(2, VOCAB, MARKOV_RANK, device="cuda", generator=gen)
    return w[0].to(torch.bfloat16), w[1].to(torch.bfloat16)


def _anchors(reqs: int, dtype: str, gen, out_of_vocab: bool) -> torch.Tensor:
    anchors = torch.randint(0, VOCAB, (reqs,), device="cuda", generator=gen)
    if out_of_vocab:
        anchors[reqs - 1] = -1
    return anchors.to(getattr(torch, dtype))


def _dot_slack(row: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """How far an FP32 dot over the Markov rank may sit from the exact one, in any summation
    order: ``R * 2**-24 * sum |w1 w2|`` (the standard bound for a length-R sum), doubled."""
    return (MARKOV_RANK * 2.0**-23 * (row.abs() @ w2.abs().t())).float()


def _bias_tol(bias: torch.Tensor, slack: torch.Tensor) -> torch.Tensor:
    """How far an op's BF16 bias may sit from the reference's: the dot's slack, then one BF16
    rounding (an ulp is at most 2**-7 of the value) on either side."""
    return 2.0**-7 * bias.abs() + slack


def _chain_scores(block, anchors, markov_w1, markov_w2, tokens):
    """``(scores, bias, slack)`` FP32 ``[B, K, vocab]`` of the reference chain that produced
    ``tokens``: position ``i``'s logits plus the BF16 bias of ``prev_i``, and the slack of the
    FP32 dot behind each bias (``_dot_slack``)."""
    w2 = markov_w2.double()
    prev = anchors.to(torch.int64)
    scores, biases, slacks = [], [], []
    for i in range(block.shape[1]):
        valid = (prev >= 0) & (prev < VOCAB)
        row = markov_w1[prev.clamp(0, VOCAB - 1)].double() * valid[:, None]
        bias = (row @ w2.t()).to(torch.bfloat16).float()
        scores.append(block[:, i].float() + bias)
        biases.append(bias)
        slacks.append(_dot_slack(row, w2))
        prev = tokens[:, i].to(torch.int64)
    return torch.stack(scores, 1), torch.stack(biases, 1), torch.stack(slacks, 1)


def _assert_chain(tokens, ref_tokens, scores, bias, slack) -> None:
    """Tokens equal up to a near-tie. The kernels round an FP32 dot over R to the BF16 bias, the
    reference an FP64 product: the two biases differ by up to ``_bias_tol``. Where a request's
    chain first differs, the kernel's token must score within the two candidates' tolerances
    of the reference's; the chain is not compared past that position."""
    assert tokens.dtype == torch.int32 and tokens.shape == ref_tokens.shape
    diverged = 0
    for b in range(tokens.shape[0]):
        diff = (tokens[b] != ref_tokens[b]).nonzero()
        if diff.numel() == 0:
            continue
        i = int(diff[0])
        k, r = int(tokens[b, i]), int(ref_tokens[b, i])
        assert 0 <= k < VOCAB, f"request {b} position {i}: token {k} outside the vocab"
        tol = float(_bias_tol(bias[b, i, [k, r]], slack[b, i, [k, r]]).sum())
        gap = float(scores[b, i, r] - scores[b, i, k])
        assert gap <= tol, f"request {b} position {i}: {k} vs {r}, gap {gap:.3e} > {tol:.3e}"
        diverged += 1
    assert diverged <= 1, f"{diverged} requests' chains differ"


def _block_logits(reqs: int, block: int, gen) -> torch.Tensor:
    return (2.0 * torch.randn(reqs, block, VOCAB, device="cuda", generator=gen)).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# k3_markov_chain (Triton, one launch)
# ---------------------------------------------------------------------------------------------

_CHAIN = k3_markov_chain.k3_markov_chain


@pytest.mark.parametrize("cell", _CHAIN.CELLS, ids=[c.why[:48] for c in _CHAIN.CELLS])
def test_markov_chain_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_spec.vocab_parallel_kernels import (  # noqa: E501
        _CHAIN_MAX_REQS,
        _CHAIN_MAX_STEPS,
        _MARKOV_BLOCK_V,
    )

    _require_arch(_CHAIN)
    spec, gen = cell.spec, _gen(cell)
    tp, rank, reqs, block = spec["tp"], spec["rank"], spec["reqs"], spec["block"]
    shard = VOCAB // tp
    programs = -(-shard // _MARKOV_BLOCK_V)
    if programs > torch.cuda.get_device_properties(0).multi_processor_count:
        pytest.skip(f"{programs} programs must all be resident")
    certifying(_CHAIN)
    markov_w1, markov_w2 = _weights(gen)
    full = _block_logits(reqs, block, gen)
    anchors = _anchors(reqs, spec["anchor"], gen, out_of_vocab=reqs > 1)
    blocks = [full[..., s * shard : (s + 1) * shard].contiguous() for s in range(tp)]
    w2_shards = [markov_w2[s * shard : (s + 1) * shard] for s in range(tp)]
    tokens = torch.full((reqs, block), -7, dtype=torch.int32, device="cuda")
    set_floats = _CHAIN_MAX_STEPS * _CHAIN_MAX_REQS * tp * 2
    partial = torch.empty(
        (_CHAIN_MAX_STEPS * programs * _CHAIN_MAX_REQS * 2,), dtype=torch.float32, device="cuda"
    )
    tickets = torch.zeros((_CHAIN_MAX_STEPS,), dtype=torch.int32, device="cuda")
    launch = torch.zeros((1,), dtype=torch.int32, device="cuda")
    exchange = torch.full((3 * set_floats,), -0.0, dtype=torch.float32, device="cuda")
    scratch = (partial, tickets, launch, exchange, exchange)
    ref = _CHAIN.reference(
        blocks, anchors, markov_w1, w2_shards, tokens, *scratch, set_floats, VOCAB, 0, tp, rank
    )
    scores, bias, slack = _chain_scores(full, anchors, markov_w1, markov_w2, ref)

    # The other ranks' pair of every position, in set 0 (launch 0): [step][request][rank][2].
    cols = torch.arange(VOCAB, device="cuda").view(tp, shard).expand(reqs, block, tp, shard)
    value, index = _reference.argmax_pairs(scores.view(reqs, block, tp, shard), cols)
    value = torch.where(value == 0, 0.0, value)  # -0.0 marks an empty slot
    region = exchange[: block * _CHAIN_MAX_REQS * tp * 2].view(block, _CHAIN_MAX_REQS, tp, 2)
    peers = [s for s in range(tp) if s != rank]
    region[:, :reqs, peers, 0] = value.transpose(0, 1)[:, :, peers]
    region[:, :reqs, peers, 1] = index.transpose(0, 1)[:, :, peers].float()

    out = _CHAIN(
        blocks[rank],
        anchors,
        markov_w1,
        w2_shards[rank],
        tokens,
        *scratch,
        set_floats=set_floats,
        vocab=VOCAB,
        col_base=rank * shard,
        tp=tp,
        tp_rank=rank,
    )
    torch.cuda.synchronize()
    assert out is tokens and int(launch) == 1
    _assert_chain(tokens, ref, scores, bias, slack)


# ---------------------------------------------------------------------------------------------
# k3_dspark_markov_greedy (CuTe DSL, one launch)
# ---------------------------------------------------------------------------------------------

_GREEDY = k3_dspark_markov_greedy.k3_dspark_markov_greedy


def _u32_to_i32(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def _keys(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``(low, high)`` int32 words of the kernel's key of FP32 ``value`` at global ``index``:
    high the value's bits in unsigned order (-0 as +0, every NaN on top), low ``~index``."""
    b = value.contiguous().view(torch.int32).to(torch.int64) & _U32
    b = torch.where(b == 0x80000000, 0, b)
    high = torch.where(b >= 0x80000000, ~b & _U32, b | 0x80000000)
    high = torch.where((b & 0x7FFFFFFF) > 0x7F800000, _U32, high)
    low = ~index.to(torch.int64) & _U32
    return torch.stack([_u32_to_i32(low), _u32_to_i32(high)], dim=-1)


@pytest.mark.parametrize("cell", _GREEDY.CELLS, ids=[c.why[:48] for c in _GREEDY.CELLS])
def test_dspark_markov_greedy_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_spec.dspark_markov_greedy import (  # noqa: E501
        NUM_SETS,
        ROWS,
        STATE_WORDS,
        ctas_for,
        set_bytes,
        width_for,
    )

    _require_arch(_GREEDY)
    certifying(_GREEDY)
    spec, gen = cell.spec, _gen(cell)
    tp, rank, reqs, block = spec["tp"], spec["rank"], spec["reqs"], spec["block"]
    shard = VOCAB // tp
    ctas, width = ctas_for(shard), width_for(reqs)
    markov_w1, markov_w2 = _weights(gen)
    full = _block_logits(reqs, block, gen)
    anchors = _anchors(reqs, spec["anchor"], gen, out_of_vocab=reqs > 1)
    blocks = [full[..., s * shard : (s + 1) * shard].contiguous() for s in range(tp)]
    w2_shards = [markov_w2[s * shard : (s + 1) * shard] for s in range(tp)]
    tokens = torch.full((reqs, block), -7, dtype=torch.int32, device="cuda")
    state = torch.zeros((STATE_WORDS,), dtype=torch.int32, device="cuda")
    box = torch.zeros((NUM_SETS * set_bytes(tp * ctas) // 4,), dtype=torch.int32, device="cuda")
    kwargs = dict(
        vocab=VOCAB,
        tp=tp,
        rank=rank,
        state=state,
        box_uc=box.data_ptr(),
        box_mc=box.data_ptr(),
        multicast=False,
    )
    ref = _GREEDY.reference(blocks, anchors, markov_w1, w2_shards, tokens, **kwargs)
    scores, bias, slack = _chain_scores(full, anchors, markov_w1, markov_w2, ref)

    # Every other rank's CTA keys of every position in set 0 (launch 0):
    # [position][rank][CTA][request slot][low, high]; padding slots hold a key of 0.0.
    cols = torch.arange(VOCAB, device="cuda").view(tp, ctas, ROWS)
    value, index = _reference.argmax_pairs(
        scores.view(reqs, block, tp, ctas, ROWS), cols.expand(reqs, block, tp, ctas, ROWS)
    )
    words = torch.empty((block, tp, ctas, width, 2), dtype=torch.int32, device="cuda")
    words[:, :, :, :reqs] = _keys(value.float(), index).permute(1, 2, 3, 0, 4)
    pad_index = cols[..., 0].expand(block, tp, ctas).unsqueeze(-1).expand(-1, -1, -1, width - reqs)
    words[:, :, :, reqs:] = _keys(torch.zeros(pad_index.shape, device="cuda"), pad_index)
    region = box[: block * tp * ctas * width * 2].view(block, tp, ctas, width, 2)
    peers = [s for s in range(tp) if s != rank]
    region[:, peers] = words[:, peers]

    _GREEDY(blocks[rank], anchors, markov_w1, w2_shards[rank], tokens, **kwargs)
    torch.cuda.synchronize()
    assert int(state[0]) == 1, "the launch ordinal advances once"
    _assert_chain(tokens, ref, scores, bias, slack)


# ---------------------------------------------------------------------------------------------
# k3_fused_markov_chain_logits (full vocab)
# ---------------------------------------------------------------------------------------------

_FUSED = k3_fused_markov_chain_logits.k3_fused_markov_chain_logits


@pytest.mark.parametrize("cell", _FUSED.CELLS, ids=[c.why[:48] for c in _FUSED.CELLS])
def test_fused_markov_chain_logits_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    from tensorrt_llm._torch.models.modeling_speculative import (
        dspark_markov_chain,
        dspark_markov_step_bias,
    )

    _require_arch(_FUSED)
    certifying(_FUSED)
    spec, gen = cell.spec, _gen(cell)
    reqs, block = spec["reqs"], spec["block"]
    markov_w1, markov_w2 = _weights(gen)
    base = _block_logits(reqs, block, gen).float()
    anchors = _anchors(reqs, spec["anchor"], gen, out_of_vocab=spec["anchor"] == "int32")
    base_before = base.clone()
    out = _FUSED(base, anchors, markov_w1, markov_w2)
    assert torch.equal(base, base_before), "the op must not write the logits"
    assert out.shape == base.shape and out.dtype == torch.float32

    # Cross-check: the chain of the drafter's own head (the same F.linear bias), bit for bit.
    _, chained = dspark_markov_chain(
        base,
        anchors,
        lambda prev, _hidden: dspark_markov_step_bias(prev, markov_w1, markov_w2),
        cast_bias_to_logits=True,
    )
    assert torch.equal(out, chained)

    # The reference rounds an FP64 bias to BF16, the op an FP32 GEMV: the biases differ by up
    # to ``_bias_tol``, and both FP32 sums with the logits round once (2**-24 of the result
    # each, and the bias read back from the reference's sum once more). The positions are
    # compared until the two chains first pick different tokens, which must be a near-tie
    # within the candidates' tolerances (every later position has another previous token).
    ref = _FUSED.reference(base, anchors, markov_w1, markov_w2)
    ref_bias = ref - base
    cols = torch.arange(VOCAB, device="cuda").expand(reqs, -1)
    w2 = markov_w2.double()
    prev = anchors.to(torch.int64)
    for i in range(block):
        valid = (prev >= 0) & (prev < VOCAB)
        row = markov_w1[prev.clamp(0, VOCAB - 1)].double() * valid[:, None]
        tol = _bias_tol(ref_bias[:, i], _dot_slack(row, w2))
        band = tol + 2.0**-22 * ref[:, i].abs()
        assert bool(((out[:, i] - ref[:, i]).abs() <= band).all()), f"position {i}"
        _, k_tok = _reference.argmax_pairs(out[:, i], cols)
        _, r_tok = _reference.argmax_pairs(ref[:, i], cols)
        if not torch.equal(k_tok, r_tok):
            b = int((k_tok != r_tok).nonzero()[0])
            k, r = int(k_tok[b]), int(r_tok[b])
            gap = float(ref[b, i, r] - ref[b, i, k])
            limit = float(tol[b, k] + tol[b, r])
            assert gap <= limit, f"position {i}: {k} vs {r}, gap {gap:.3e} > {limit:.3e}"
            break
        prev = r_tok
