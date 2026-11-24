# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_selector_draft entry: every cell bitwise against the torch path, and
``dflash2_sample_draft``'s call of it."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_draft import (  # noqa: E501
    k31_dflash2_selector_draft as op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the K3.1 DFlash2 selector entry is certified on sm_103 and sm_107",
)

#: K3.1 DFlash2 selector: rank 256 codebooks over the 163,840-token vocabulary.
VOCAB = 163840
RANK = 256

_CODEBOOKS = {}


def _codebooks(dtype):
    """One pair of full-size codebooks per dtype for the whole module (80 MB each in BF16)."""
    if dtype not in _CODEBOOKS:
        g = torch.Generator(device="cuda").manual_seed(7)
        pred = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).to(dtype)
        succ = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).to(dtype)
        _CODEBOOKS[dtype] = (pred, succ)
    return _CODEBOOKS[dtype]


def _temperatures(kind, n, g):
    if kind == "hot":
        return 0.5 + 2.5 * torch.rand(n, generator=g, device="cuda")
    if kind == "edges":
        threshold = torch.tensor(GREEDY_TEMPERATURE_THRESHOLD, dtype=torch.float32)
        values = torch.stack(
            [
                threshold,
                torch.nextafter(threshold, torch.tensor(1.0)),
                torch.tensor(1e-5),
                torch.tensor(1e-6),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(2e-5),
            ]
        )
        return values.repeat(n)[:n].cuda()
    temps = 1.5 * torch.rand(n, generator=g, device="cuda")
    temps[::3] = 0.0
    return temps


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots, rows = spec["n"], spec["slots"], spec["rows"]
    pred_cb, succ_cb = _codebooks(spec["codebook"])
    candidate_ids = torch.randint(0, VOCAB, (n, slots, 32), generator=g, device="cuda")
    unary = 10.0 + 2.0 * torch.randn(n, slots, 32, generator=g, device="cuda")
    unary = unary.sort(dim=-1, descending=True).values
    if spec["scores"] == "ties":
        # Repeated candidates with equal logits score identically: the argmax and the
        # cumulative sums meet exact ties.
        candidate_ids[..., 16:] = candidate_ids[..., :16]
        unary[..., 16:] = unary[..., :16]
        unary[..., 1] = unary[..., 0]
        candidate_ids[..., 1] = candidate_ids[..., 0]
    unary = unary.to(spec["unary"])
    gate = torch.randn(n, slots, RANK, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int32)
    seed_t = torch.randint(0, 2**62, (n,), generator=g, device="cuda")
    offset = torch.randint(0, 2**40, (n,), generator=g, device="cuda")
    temps = _temperatures(spec["temps"], n, g)
    store = (
        torch.zeros(rows, slots, 32, dtype=torch.int64, device="cuda"),
        torch.zeros(rows, slots, 32, dtype=torch.float32, device="cuda"),
        torch.full((rows, slots), -1, dtype=torch.int64, device="cuda"),
    )
    slot_ids = torch.randperm(rows, generator=g, device="cuda")[:n]
    return [
        pred_cb,
        succ_cb,
        candidate_ids,
        unary,
        gate,
        anchor,
        seed_t,
        offset,
        temps,
        GREEDY_TEMPERATURE_THRESHOLD,
        *store,
        slot_ids,
    ]


def _with_fresh_store(args):
    out = list(args)
    out[10:13] = [t.clone() for t in args[10:13]]
    return out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    got_args = _with_fresh_store(args)
    ref_args = _with_fresh_store(args)
    tokens = op(*got_args)
    expected = op.reference(*ref_args)
    assert tokens.dtype == torch.int32 and tokens.shape == expected.shape
    assert torch.equal(tokens, expected)
    for got, ref in zip(got_args[10:13], ref_args[10:13]):
        assert torch.equal(got, ref)


def test_cuda_graph_replay_matches_eager() -> None:
    """Captured after one eager call, a replay with new seeds, scores and temperatures gives the
    eager result bit for bit."""
    spec = op.CELLS[2].spec
    args = _build(spec, seed=11)
    op(*_with_fresh_store(args))  # compile outside capture
    static = _with_fresh_store(args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        op(*static)
        with torch.cuda.graph(graph, stream=stream):
            out = op(*static)
    torch.cuda.current_stream().wait_stream(stream)
    fresh = _build(spec, seed=12)
    for i in (2, 3, 4, 5, 6, 7, 8):
        static[i].copy_(fresh[i])
    graph.replay()
    expected = op.reference(*_with_fresh_store(fresh))
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


def test_guard_refuses_another_top_k() -> None:
    args = _build(op.CELLS[0].spec, seed=3)
    args[2] = args[2][..., :16].contiguous()
    args[3] = args[3][..., :16].contiguous()
    with pytest.raises(AssertionError, match="top_k must be 32"), validating(op):
        op(*args)


def test_guard_refuses_an_out_of_range_id() -> None:
    args = _build(op.CELLS[0].spec, seed=4)
    args[2][0, 0, 3] = VOCAB
    with pytest.raises(AssertionError, match="id out of range"), validating(op):
        op(*args)


@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_dflash2_sample_draft_runs_this_entry(logits_dtype) -> None:
    """``dflash2_sample_draft`` on the K3.1 worker drafts with this entry, and its operands are wired
    right.

    At the selector's top-32 and rank 256 the sampling batch takes the worker's global top-32, the
    selector's ``hidden_projection`` of the block hidden states, and each generation request's RNG
    state, temperature and sequence slot (context rows first in the batch). The drafted tokens and the
    proposal-store rows must equal this entry's torch reference on the same operands, bit for bit, on
    a real ``K31DFlash2CandidateSelector`` and full-vocab drafter logits (FP32 as the logits processor
    returns them, BF16-valued; and BF16)."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31DFlashWorker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (  # noqa: E501
        DFlash2ProposalStore,
        dflash2_sample_draft,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_path import (  # noqa: E501
        K31DFlash2CandidateSelector,
    )
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    num_contexts, n, slots, hidden = 2, 8, 7, 7168
    batch = num_contexts + n
    g = torch.Generator(device="cuda").manual_seed(17)
    pred_cb, succ_cb = _codebooks(torch.bfloat16)
    projection = (0.02 * torch.randn(RANK, hidden, generator=g, device="cuda")).to(torch.bfloat16)
    selector = K31DFlash2CandidateSelector(pred_cb, succ_cb, projection, 32, VOCAB, RANK)
    draft_model = SimpleNamespace(candidate_selector=selector)
    gen_logits = (10.0 + 3.0 * torch.randn(n, slots, VOCAB, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    gen_logits = gen_logits.to(logits_dtype)
    block_hidden = torch.randn(n, slots, hidden, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int32)
    rows = batch + 1
    metadata = SimpleNamespace(
        request_seeds=torch.randint(0, 2**62, (batch,), generator=g, device="cuda"),
        request_offsets=torch.randint(0, 2**40, (batch,), generator=g, device="cuda"),
        request_temperatures=_temperatures("mixed", batch, g),
        batch_slot_ids=torch.randperm(rows, generator=g, device="cuda")[:batch],
        rng_draft_slot=lambda step: 0,
    )

    worker = K31DFlashWorker(DFlashDecodingConfig(max_draft_len=slots), Mapping())
    worker._dflash2_proposals = DFlash2ProposalStore(rows, slots, 32, "cuda")
    tokens = dflash2_sample_draft(
        worker, draft_model, gen_logits, block_hidden, anchor, metadata, num_contexts, batch
    )
    store = worker._dflash2_proposals

    candidate_ids, unary, _ = worker._dflash2_global_top_k(gen_logits, metadata, 32, VOCAB)
    expected_store = DFlash2ProposalStore(rows, slots, 32, "cuda")
    expected = op.reference(
        selector.predecessor_codebook,
        selector.successor_codebook,
        candidate_ids,
        unary,
        selector.hidden_projection(block_hidden),
        anchor,
        metadata.request_seeds[num_contexts:],
        metadata.request_offsets[num_contexts:],
        metadata.request_temperatures[num_contexts:],
        GREEDY_TEMPERATURE_THRESHOLD,
        expected_store.candidate_ids,
        expected_store.q,
        expected_store.tokens,
        metadata.batch_slot_ids[num_contexts:],
    )
    assert torch.equal(tokens, expected)
    assert torch.equal(store.candidate_ids, expected_store.candidate_ids)
    assert torch.equal(store.q, expected_store.q)
    assert torch.equal(store.tokens, expected_store.tokens)
