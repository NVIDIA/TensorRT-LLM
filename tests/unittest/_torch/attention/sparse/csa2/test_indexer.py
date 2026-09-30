# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 indexer contracts, candidate selection, native kernels and runtime workflows."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2CompressionBatch
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import (
    _HAS_SPARSE_MQA_LOGITS,
    CSA2Indexer,
    _ChunkInputs,
    _QueryChunk,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2ForwardState,
    CSA2Layout,
    CSA2Params,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    pack_rows,
    read_index_rows,
    unpack_rows,
    write_packed_index_rows,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm.mapping import Mapping

from ._utils import _prediction_reference, _run_indexer, _run_modes, _selection_state


def _reference_select_candidate_positions(
    scores: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Return bounded candidate positions [queries, blocks * block_size], int64.

    Scores are float32 [queries, positions]. Unreachable positions are masked
    before block reduction. The newest visible block is always retained, as
    required by the trained hierarchical indexer. Invalid slots contain -1.
    """
    if scores.ndim != 2 or visible_lengths.shape != scores.shape[:1]:
        raise ValueError("Candidate scores and visible lengths must have matching query rows")
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("Candidate block count and size must be positive")
    width = scores.shape[-1]
    positions = torch.arange(width, device=scores.device)
    reachable = positions[None, :] < visible_lengths[:, None]
    scores = scores.masked_fill(~reachable, -torch.inf)
    padded = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = padded.reshape(scores.shape[0], padded.shape[-1] // block_size, block_size).amax(-1)
    block_ids = torch.arange(blocks.shape[-1], device=scores.device)
    latest = (visible_lengths - 1) // block_size
    blocks = blocks.masked_fill(block_ids[None, :] == latest[:, None], torch.inf)
    values, selected = blocks.topk(min(topk_blocks, blocks.shape[-1]), dim=-1, sorted=False)
    candidates = selected[:, :, None] * block_size + torch.arange(block_size, device=scores.device)
    valid = (values[:, :, None] > -torch.inf) & (candidates < visible_lengths[:, None, None])
    valid &= candidates < width
    return torch.where(valid, candidates, -1).flatten(1)


def _reference_select_topk_positions(
    scores: torch.Tensor,
    positions: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select sorted logical positions; pad unreachable selections with -1.

    ``positions`` is either [positions] or [queries, candidates]. Selection
    never publishes candidate-array offsets or cache addresses to Reuse layers.
    """
    if scores.ndim != 2 or visible_lengths.shape != scores.shape[:1]:
        raise ValueError("Top-k scores and visible lengths must have matching query rows")
    if topk <= 0 or positions.shape[-1] != scores.shape[-1]:
        raise ValueError("Invalid top-k size or candidate positions")
    positions = positions.long().expand_as(scores)
    valid = (positions >= 0) & (positions < visible_lengths[:, None])
    scores = scores.masked_fill(~valid, -torch.inf)
    values, offsets = scores.topk(min(topk, scores.shape[-1]), dim=-1, sorted=False)
    selected = positions.gather(1, offsets)
    # Sort invalid entries last. In particular, an unreachable candidate with a
    # low logical index must not become a valid selection after the gather.
    sentinel = torch.iinfo(torch.int64).max
    selected = torch.where(values > -torch.inf, selected, sentinel).sort(dim=-1).values
    selected = torch.where(selected == sentinel, -1, selected).to(torch.int32)
    return F.pad(selected, (0, topk - selected.shape[-1]), value=-1)


def _reference_index_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Rectified indexer scores with already-scaled per-head weights.

    Q is [queries, heads, dim], K is [positions, dim] or
    [queries, candidates, dim], weights is [queries, heads]. The gathered-K
    form bounds Reindex work by candidate count, independently of context size.
    """
    if k.ndim == 2:
        dots = torch.einsum("qhd,kd->qhk", q, k)
    elif k.ndim == 3:
        dots = torch.einsum("qhd,qkd->qhk", q, k)
    else:
        raise ValueError("Indexer keys must be shared or gathered per query")
    return (dots.relu() * weights.unsqueeze(-1)).sum(1).float()


def _prepared_indexer(heads=32, topk=32):
    return CSA2Indexer(CSA2Layout((1,), (0,), (0,), index_topk=topk), 0, heads, 128)


def _prepared(indexer, q, keys, weights, lengths, positions, hook=None):
    if not q.is_cuda:
        indexer.top_k.prefill_implementation = TopKImplementation.TORCH
        indexer.top_k.decode_implementation = TopKImplementation.TORCH
    count = q.shape[0]
    width = keys.shape[-2]
    starts = torch.arange(count, dtype=torch.int32, device=q.device)
    starts = starts * width if keys.ndim == 3 else torch.zeros_like(starts)
    keys = keys.reshape(-1, 68)
    out = torch.empty((count, indexer.index_topk), dtype=torch.int32, device=q.device)
    indexer.forward_prepared(
        q[..., :64].contiguous().view(torch.int8),
        keys[:, :64].contiguous().view(torch.int8),
        keys[:, 64:].contiguous().view(torch.int32).squeeze(-1),
        weights,
        starts,
        starts + width,
        out,
        q[..., 64:].contiguous().view(torch.int32).squeeze(-1),
        logical_positions=positions,
        visible_lengths=lengths,
        score_hook=hook,
    )
    return out


@pytest.mark.cpu_only
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_projection_free_lifecycle(layer_idx):
    layout = CSA2Layout((1, 1), (0,), (0, 1), index_topk=32)
    indexer = CSA2Indexer(layout, layer_idx, 32, 128)
    assert isinstance(indexer, Indexer)
    assert CSA2Indexer._call_mqa_logits is Indexer._call_mqa_logits
    assert CSA2Indexer._call_paged_mqa_logits is Indexer._call_paged_mqa_logits
    assert isinstance(indexer.top_k, TopK)
    assert list(indexer.parameters()) == []
    assert indexer.rotary_emb is None
    assert indexer.wq_b is indexer.wk is indexer.weights_proj is indexer.k_norm is None
    indexer.cache_derived_state()
    indexer.post_load_weights()
    with pytest.raises(RuntimeError, match="forward_prepared"):
        indexer.pre_indexer_proj(None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("per_query", [False, True])
@pytest.mark.parametrize("heads", [2, 8, 32, 64])
def test_prepared_packed_selection(per_query, heads):
    indexer = _prepared_indexer(heads)
    torch.manual_seed(1701)
    q = pack_rows(torch.randn(3, heads, 128, device="cuda"), "index")
    shape = (3, 257, 128) if per_query else (257, 128)
    packed = pack_rows(torch.randn(shape, device="cuda"), "index")
    owner = torch.zeros((*packed.shape[:-1], 356), dtype=torch.uint8, device="cuda")
    owner[..., 288:].copy_(packed)
    keys = owner[..., 288:]
    weights = torch.rand(3, heads, device="cuda")
    lengths = torch.tensor([257, 39, 0], device="cuda", dtype=torch.int32)
    positions = torch.arange(257, device="cuda").expand(3, -1).clone()
    positions[:, 5::11] = -1
    actual = _prepared(indexer, q, keys, weights, lengths, positions)
    scores = _reference_index_scores(
        unpack_rows(q, 128, "index", torch.float32),
        unpack_rows(keys, 128, "index", torch.float32),
        weights,
    )
    expected = _reference_select_topk_positions(scores, positions, lengths, 32)
    # Low head counts can tie at zero; compare selected score multisets, with
    # exact logical output for the unambiguous 32/64-head cases.
    if heads >= 32:
        torch.testing.assert_close(actual, expected)
    else:
        valid = actual >= 0
        torch.testing.assert_close(valid.sum(-1), (expected >= 0).sum(-1))
        torch.testing.assert_close(
            scores.gather(1, actual.clamp_min(0).long())
            .masked_fill(~valid, -torch.inf)
            .sort(-1)
            .values,
            scores.gather(1, expected.clamp_min(0).long())
            .masked_fill(expected < 0, -torch.inf)
            .sort(-1)
            .values,
        )


def test_prepared_hierarchy_and_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    indexer = CSA2Indexer(
        CSA2Layout(
            (1,),
            (0,),
            (0,),
            candidate_source_layer_id=0,
            candidate_topk_blocks=3,
            candidate_block_size=32,
            index_topk=32,
        ),
        0,
        32,
        128,
    )
    indexer._configure_candidate_topk(257)
    torch.manual_seed(59)
    q = pack_rows(torch.randn(3, 32, 128, device="cuda"), "index")
    keys = pack_rows(torch.randn(3, 257, 128, device="cuda"), "index")
    weights = torch.rand(3, 32, device="cuda")
    lengths = torch.tensor([257, 129, 0], device="cuda", dtype=torch.int32)
    positions = torch.arange(257, device="cuda").expand(3, -1).clone()
    candidates = torch.empty((3, 96), dtype=torch.int64, device="cuda")

    def hook(scores):
        blocks = (
            torch.nn.functional.pad(scores, (0, 31), value=-torch.inf).reshape(3, 9, 32).amax(-1)
        )
        latest = (lengths - 1) // 32
        blocks = blocks.masked_fill(
            torch.arange(9, device="cuda")[None, :] == latest[:, None], torch.inf
        )
        selected = torch.empty((3, 3), dtype=torch.int32, device="cuda")
        indexer.select_prepared_scores(
            blocks, selected, torch.zeros_like(lengths), torch.full_like(lengths, 9), candidate=True
        )
        values = selected.long()[:, :, None] * 32 + torch.arange(32, device="cuda")
        valid = (values < lengths[:, None, None]) & (values < 257)
        valid = valid & (blocks.gather(1, selected.long())[:, :, None] > -torch.inf)
        candidates.copy_(torch.where(valid, values, -1).flatten(1))

    def run():
        return _prepared(indexer, q, keys, weights, lengths, positions, hook)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for visible in ([257, 129, 0], [0, 17, 256], [256, 0, 3]):
        lengths.copy_(torch.tensor(visible, device="cuda", dtype=torch.int32))
        weights.mul_(0.9)
        keys[..., :64].bitwise_xor_(0x88)
        graph.replay()
        scores = _reference_index_scores(
            unpack_rows(q, 128, "index", torch.float32),
            unpack_rows(keys, 128, "index", torch.float32),
            weights,
        )
        expected_candidates = _reference_select_candidate_positions(scores, lengths, 3, 32)
        expected = _reference_select_topk_positions(scores, positions, lengths, 32)
        torch.testing.assert_close(candidates.sort(-1).values, expected_candidates.sort(-1).values)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prepared_bmm_fallback(monkeypatch):
    indexer = _prepared_indexer(8)
    torch.manual_seed(7)
    q = pack_rows(torch.randn(2, 8, 128, device="cuda"), "index")
    k = pack_rows(torch.randn(2, 33, 128, device="cuda"), "index")
    weights = torch.rand(2, 8, device="cuda")
    lengths = torch.tensor([33, 3], dtype=torch.int32, device="cuda")
    positions = torch.arange(33, device="cuda").expand(2, -1)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.indexer.get_sm_version", lambda: 90
    )
    scores = []
    actual = _prepared(indexer, q, k, weights, lengths, positions, scores.append)
    dots = torch.bmm(unpack_rows(q, 128, "index"), unpack_rows(k, 128, "index").transpose(1, 2))
    expected_scores = (dots.relu() * weights.bfloat16().unsqueeze(-1)).sum(1).float()
    expected = _reference_select_topk_positions(expected_scores, positions, lengths, 32)
    torch.testing.assert_close(actual, expected)
    for _ in range(3):
        _prepared(indexer, q, k, weights, lengths, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = _prepared(indexer, q, k, weights, lengths, positions)
    lengths.copy_(torch.tensor([0, 17], dtype=torch.int32, device="cuda"))
    k[..., :64].bitwise_xor_(0x88)
    graph.replay()
    dots = torch.bmm(unpack_rows(q, 128, "index"), unpack_rows(k, 128, "index").transpose(1, 2))
    expected_scores = (dots.relu() * weights.bfloat16().unsqueeze(-1)).sum(1).float()
    expected = _reference_select_topk_positions(expected_scores, positions, lengths, 32)
    torch.testing.assert_close(replayed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_fp4", [False, True])
def test_prepared_default_row_local(use_fp4):
    indexer = _prepared_indexer()
    indexer.use_fp4 = use_fp4
    torch.manual_seed(913)
    q = torch.randn(2, 32, 128, device="cuda")
    k = torch.randn(258, 128, device="cuda")
    weights = torch.rand(2, 32, device="cuda")
    if use_fp4:
        qr, kr = pack_rows(q, "index"), pack_rows(k, "index")
        qd, kd = (
            qr[..., :64].contiguous().view(torch.int8),
            kr[:, :64].contiguous().view(torch.int8),
        )
        qs, ks = (
            qr[..., 64:].contiguous().view(torch.int32),
            kr[:, 64:].contiguous().view(torch.int32),
        )
        q, k = (
            unpack_rows(qr, 128, "index", torch.float32),
            unpack_rows(kr, 128, "index", torch.float32),
        )
    else:
        qd, kd = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
        qs, ks = None, torch.ones(258, device="cuda")
        q, k = qd.float(), kd.float()
    starts = torch.tensor([0, 129], dtype=torch.int32, device="cuda")
    ends = starts + 129
    out = torch.empty((2, 32), dtype=torch.int32, device="cuda")
    Indexer.forward_prepared(indexer, qd, kd, ks, weights, starts, ends, out, qs)
    scores = _reference_index_scores(q, k, weights)
    expected = torch.stack((scores[0, :129].topk(32).indices, scores[1, 129:].topk(32).indices))
    torch.testing.assert_close(out.long().sort(-1).values, expected.sort(-1).values)


@pytest.mark.cpu_only
def test_prepared_cpu_reference():
    indexer = _prepared_indexer(8, 4)
    q = pack_rows(torch.randn(2, 8, 128), "index")
    k = pack_rows(torch.randn(2, 17, 128), "index")
    weights = torch.rand(2, 8)
    lengths = torch.tensor([17, 3], dtype=torch.int32)
    positions = torch.arange(17).expand(2, -1)
    actual = _prepared(indexer, q, k, weights, lengths, positions)
    scores = _reference_index_scores(
        unpack_rows(q, 128, "index", torch.float32),
        unpack_rows(k, 128, "index", torch.float32),
        weights,
    )
    expected = _reference_select_topk_positions(scores, positions, lengths, 4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_hook", [False, True])
def test_short_sequence_skip_preserves_candidate_scores(monkeypatch, has_hook):
    layout = CSA2Layout((1,), (0,), (0,), index_topk=8)
    indexer = CSA2Indexer(layout, 0, 8, 128)
    indexer.top_k.prefill_implementation = TopKImplementation.TORCH
    indexer.top_k.decode_implementation = TopKImplementation.TORCH
    q = pack_rows(torch.randn(2, 8, 128), "index")
    k = pack_rows(torch.randn(4, 128), "index")
    weights = torch.rand(2, 8)
    positions = torch.tensor([[0, -1, 2, 3], [0, 1, 2, 3]])
    visible = torch.tensor([3, 0], dtype=torch.int32)
    calls = []
    native = indexer._call_mqa_logits

    def record(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(indexer, "_call_mqa_logits", record)
    hook_scores = []
    output = torch.empty(2, 8, dtype=torch.int32)
    indexer.forward_prepared(
        q[..., :64].contiguous().view(torch.int8),
        k[:, :64].contiguous().view(torch.int8),
        k[:, 64:].contiguous(),
        weights,
        torch.zeros(2, dtype=torch.int32),
        torch.full((2,), 4, dtype=torch.int32),
        output,
        q[..., 64:].contiguous(),
        logical_positions=positions,
        visible_lengths=visible,
        score_hook=hook_scores.append if has_hook else None,
    )
    assert len(calls) == int(has_hook)
    assert len(hook_scores) == int(has_hook)
    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0, 2, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32
        ),
    )


@pytest.mark.parametrize("implementation", ["dsl", "self_sampling"])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_topk_opt_in_native_and_graph(implementation, is_prefill):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("TopK optimizations require SM100")
    torch.manual_seed(6301)
    options = CSA2Params(
        use_cute_dsl_topk=implementation == "dsl",
        enable_heuristic_topk=implementation == "self_sampling",
    )
    indexer = CSA2Indexer(CSA2Layout((1,), (0,), (0,), index_topk=512), 0, 64, 128, options)
    scores = torch.rand(2, 4096, device="cuda")
    starts = torch.zeros(2, dtype=torch.int32, device="cuda")
    ends = torch.tensor([4096, 2048], dtype=torch.int32, device="cuda")
    output = torch.empty(2, 512, dtype=torch.int32, device="cuda")
    aux_indices = torch.empty(2, 10, 512, dtype=torch.int32, device="cuda")
    aux_logits = torch.empty_like(aux_indices, dtype=torch.float32)
    selector = indexer.top_k
    expected_impl = (
        TopKImplementation.CUTE_DSL_RADIX
        if implementation == "dsl"
        else TopKImplementation.CUTE_DSL_GVR
    )
    assert (
        selector.prefill_implementation if is_prefill else selector.decode_implementation
    ) == expected_impl

    def run():
        return indexer.select_prepared_scores(
            scores,
            output,
            starts,
            ends,
            is_prefill=is_prefill,
            radix_aux_indices=aux_indices,
            radix_aux_logits=aux_logits,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths in ([4096, 2048], [512, 1024]):
        ends.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        scores.copy_(torch.rand_like(scores))
        graph.replay()
        for row, length in enumerate(lengths):
            expected = scores[row, :length].topk(512).indices
            # These deterministic, continuous scores have no boundary ties.
            torch.testing.assert_close(output[row].long().sort().values, expected.sort().values)


def test_self_sampling_mqa_selection_and_graph():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Self-sampling GVR requires SM100")
    torch.manual_seed(7303)
    indexer = CSA2Indexer(
        CSA2Layout((1,), (0,), (0,), index_topk=512),
        0,
        64,
        128,
        options=CSA2Params(enable_heuristic_topk=True),
    )
    q = pack_rows(torch.randn(2, 64, 128, device="cuda"), "index")
    k = pack_rows(torch.randn(2, 4096, 128, device="cuda"), "index")
    weights = torch.rand(2, 64, device="cuda") / 64
    positions = torch.arange(4096, device="cuda").expand(2, -1).clone()
    positions[:, 7::11] = -1
    lengths = torch.tensor([4096, 3072], dtype=torch.int32, device="cuda")
    for _ in range(3):
        _prepared(indexer, q, k, weights, lengths, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _prepared(indexer, q, k, weights, lengths, positions)
    for visible in ([4096, 3072], [0, 256], [2048, 4096]):
        lengths.copy_(torch.tensor(visible, dtype=torch.int32, device="cuda"))
        weights.mul_(0.99)
        graph.replay()
        scores = _reference_index_scores(
            unpack_rows(q, 128, "index", torch.float32),
            unpack_rows(k, 128, "index", torch.float32),
            weights,
        )
        expected = _reference_select_topk_positions(scores, positions, lengths, 512)
        torch.testing.assert_close(actual, expected)


def _reference_candidate_mask(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """The reference algorithm, transcribed. Returns a dense bool mask."""
    width = logits.size(-1)
    pad = -width % block_size
    padded = F.pad(logits, (0, pad), value=float("-inf"))
    scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1).div(block_size, rounding_mode="floor")
    block_ar = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_ar.unsqueeze(0) == last.unsqueeze(1), float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _candidate_indexer(width, block_size, topk_blocks):
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        candidate_source_layer_id=0,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=block_size,
        index_topk=4,
    )
    indexer = CSA2Indexer(layout, 0, 8, 128)
    indexer.top_k.prefill_implementation = TopKImplementation.TORCH
    indexer._configure_candidate_topk(width)
    return indexer


def _effective_mask(positions, width):
    columns = torch.arange(width)
    return (positions[:, :, None] == columns[None, None, :]).any(1)


def _publish_candidate_rows(indexer, scores, lengths):
    results = {}
    indexer._publish_candidates(scores, lengths, results, 0, len(lengths))
    return results[0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width,block_size", [(32, 4), (30, 4), (33, 8), (64, 8), (17, 3)])
@pytest.mark.parametrize("topk_blocks", [1, 2, 3, 5, 99])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_candidates_match_reference_effective_mask(device, width, block_size, topk_blocks, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for the CuTe candidate publication")
    torch.manual_seed(0x4109 ^ width ^ (block_size << 8) ^ (topk_blocks << 16))
    lengths = torch.tensor([0, 1, block_size, block_size + 1, width - 1, width], dtype=torch.int32)
    # Integer scores remain unique in BF16, avoiding backend-specific top-k tie order.
    scores = torch.stack([torch.randperm(width).to(dtype) for _ in lengths])
    reachable = torch.arange(width)[None, :] < lengths[:, None]
    scores = scores.masked_fill(~reachable, -torch.inf)
    indexer = _candidate_indexer(width, block_size, topk_blocks)
    positions = _publish_candidate_rows(indexer, scores.to(device), lengths.to(device)).cpu()
    expected = _reference_candidate_mask(scores, lengths, topk_blocks, block_size) & reachable
    assert torch.equal(_effective_mask(positions, width), expected)
    assert torch.all((positions == -1) | ((positions >= 0) & (positions < lengths[:, None])))


@pytest.mark.cpu_only
def test_latest_partial_block_is_pinned_despite_lowest_score():
    indexer = _candidate_indexer(10, 4, 1)
    scores = torch.tensor([[100.0, 99.0, 98.0, 97.0, 50.0, 49.0, 48.0, 47.0, -20.0, -30.0]])
    positions = _publish_candidate_rows(indexer, scores, torch.tensor([9], dtype=torch.int32))
    assert positions.tolist() == [[8, -1, -1, -1]]


@pytest.mark.cpu_only
@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 95, 96, 97])
def test_inert_boundary_and_first_dropped_block(length):
    width, block_size, blocks = 97, 32, 3
    indexer = _candidate_indexer(width, block_size, blocks)
    scores = torch.arange(width, dtype=torch.float32).neg()[None, :]
    scores[:, length:] = -torch.inf
    positions = _publish_candidate_rows(indexer, scores, torch.tensor([length], dtype=torch.int32))
    mask = _effective_mask(positions, width)[0]
    if length <= block_size * blocks:
        assert torch.equal(mask, torch.arange(width) < length)
    else:
        # The newest block replaces the worst older block at the first active length.
        assert mask[96]
        assert mask[:64].all()
        assert not mask[64:96].any()


@pytest.mark.cpu_only
def test_ragged_rows_and_tiled_publication_keep_query_order():
    indexer = _candidate_indexer(17, 4, 2)
    lengths = torch.tensor([17, 0, 3, 9], dtype=torch.int32)
    scores = torch.arange(68, dtype=torch.float32).reshape(4, 17)
    scores = scores.masked_fill(torch.arange(17)[None, :] >= lengths[:, None], -torch.inf)
    whole = _publish_candidate_rows(indexer, scores, lengths)
    tiled = {}
    for begin, end in ((0, 1), (1, 3), (3, 4)):
        indexer._publish_candidates(scores[begin:end], lengths[begin:end], tiled, begin, 4)
    assert torch.equal(tiled[0], whole)
    assert (whole[1] == -1).all()
    with pytest.raises(ValueError, match="full-query shape"):
        indexer._publish_candidates(scores[1:2], lengths[1:2], {}, 1, 4)


@pytest.mark.cpu_only
def test_candidate_selector_survives_warmup_and_draft_bounds():
    indexer = _candidate_indexer(17, 4, 8)
    selector = indexer.candidate_top_k
    assert selector.top_k == 8
    for width in (17, 33, 5, 0, 65, 17):
        indexer._configure_candidate_topk(width)
        assert indexer.candidate_top_k is selector
        lengths = torch.tensor([width, max(0, width - 3)], dtype=torch.int32)
        scores = torch.arange(width).float().expand(2, -1)
        reachable = torch.arange(width)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~reachable, -torch.inf)
        actual = _publish_candidate_rows(indexer, scores, lengths)
        assert actual.shape == (2, min(8, (width + 3) // 4) * 4)
        if width:
            expected = _reference_candidate_mask(scores, lengths, 8, 4) & reachable
            torch.testing.assert_close(_effective_mask(actual, width), expected)
        else:
            assert actual.numel() == 0

    with pytest.raises(ValueError, match="nonnegative"):
        indexer._configure_candidate_topk(-1)
    with pytest.raises(ValueError, match="cannot truncate"):
        indexer._publish_candidates(torch.ones(1, 17), torch.tensor([17]), {}, 0, 1, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_candidate_selector_keeps_captured_graphs_across_metadata_bounds():
    layout = CSA2Layout(
        (1,),
        (0,),
        (0,),
        candidate_source_layer_id=0,
        candidate_topk_blocks=8,
        candidate_block_size=4,
        index_topk=4,
    )
    indexer = CSA2Indexer(layout, 0, 8, 128)
    captures = []
    for width in (17, 33):
        indexer._configure_candidate_topk(width)
        scores = torch.arange(width, dtype=torch.float32, device="cuda").expand(2, -1).clone()
        lengths = torch.tensor([width, width - 3], dtype=torch.int32, device="cuda")
        published_width = min(8, (width + 3) // 4) * 4

        def run():
            result = {}
            # Native MQA may return aligned storage wider than its logical
            # domain. Deliberately large padding must never reach the hook.
            logits = F.pad(scores, (0, 64 - width % 64), value=1e6)
            positions = torch.arange(width, device="cuda").expand(2, -1)
            output = torch.empty(2, layout.index_topk, dtype=torch.int32, device="cuda")

            def publish(masked):
                assert masked.shape == scores.shape
                indexer._publish_candidates(masked, lengths, result, 0, 2, published_width)

            indexer._select_mapped_logits(
                logits, torch.zeros_like(lengths), lengths, positions, lengths, output, publish
            )
            return result[0]

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        captures.append((graph, scores, lengths, output, width, indexer.candidate_top_k))
    # Revisit the old graph after another metadata bound has used the layer.
    for entry in (captures[0], captures[1], captures[0]):
        graph, scores, lengths, output, width, selector = entry
        indexer._configure_candidate_topk(width)
        assert indexer.candidate_top_k is selector
        scores.neg_()
        lengths[1] = 1
        pointer = output.data_ptr()
        graph.replay()
        reachable = torch.arange(width)[None, :] < lengths.cpu()[:, None]
        expected = (
            _reference_candidate_mask(
                scores.cpu().masked_fill(~reachable, -torch.inf), lengths.cpu(), 8, 4
            )
            & reachable
        )
        torch.testing.assert_close(_effective_mask(output.cpu(), width), expected)
        assert output.data_ptr() == pointer


@pytest.mark.cpu_only
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("next_n", [1, 4, 17])
def test_verification_queries_pin_their_own_latest_candidate(ratio, next_n):
    """Future verification rows cannot steal an earlier query's pinned block."""
    endpoints = torch.tensor([33, 65], dtype=torch.int32)
    positions = endpoints[:, None] - next_n + torch.arange(next_n)[None, :]
    visible = ((positions.flatten() + 1) // ratio).int()
    scores = torch.arange(80).float().expand(visible.numel(), -1)
    actual = _publish_candidate_rows(_candidate_indexer(80, 8, 1), scores, visible)
    latest = ((visible - 1) // 8)[:, None] * 8
    expected = latest + torch.arange(8)[None, :]
    expected = torch.where(expected < visible[:, None], expected, -1)
    torch.testing.assert_close(actual, expected.to(actual.dtype))


def _index_pages_from_rows(rows):
    """Native page-footer index pages holding packed [n, 68] rows at slots 0..n-1."""
    pages = (rows.shape[0] + 63) // 64
    storage = torch.zeros(pages * 64 * 68, dtype=torch.uint8, device=rows.device)
    write_packed_index_rows(storage, torch.arange(rows.shape[0], device=rows.device), rows)
    return storage.view(pages, 64, 1, 68)


class _StridedOwner:
    """GLOBAL main/index fixture; exercise the manager's actual native gather and pages."""

    gather_indexer_keys = CSA2CacheManager.gather_indexer_keys
    tokens_per_block = 128

    def __init__(self, layout, rows):
        self.layout = layout
        # Main records live in their own buffer; index rows live in the native
        # page-footer pages the paged kernels read in place.
        self.storage = torch.full((rows.shape[0], 288), 77, dtype=torch.uint8, device=rows.device)
        self.index_pages = _index_pages_from_rows(rows)

    def get_index_pages(self, layer):
        assert self.layout.layer(layer).kv_source == 0
        return self.index_pages

    def index_pages_per_global_page(self, layer):
        return (self.tokens_per_block // self.layout.compress_ratios[layer]) // 64


def _workflow_indexer():
    return CSA2Indexer(CSA2Layout((1,), (0,), (0,), index_topk=32), 0, 32, 128)


def _projected(count):
    q = torch.randn(count, 32, 128, device="cuda", dtype=torch.bfloat16)
    packed = pack_rows(q, "index")
    weights = torch.rand(count, 32, device="cuda") / 32
    return q, packed, weights


def _assert_selection(actual, scores, valid):
    """Accept boundary ties, but require the correct valid set size and ranking."""
    for row in range(actual.shape[0]):
        selected = actual[row][actual[row] >= 0].long()
        count = min(actual.shape[1], int(valid[row].sum()))
        assert selected.numel() == count
        assert selected.unique().numel() == count
        assert bool(valid[row, selected].all())
        assert bool((selected[1:] >= selected[:-1]).all())
        assert bool((actual[row, count:] == -1).all())
        if count:
            cutoff = scores[row].masked_fill(~valid[row], -torch.inf).topk(count).values[-1]
            assert bool((scores[row, selected] >= cutoff - 0.005).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cached_prefix_chunks_gather_once_and_preserve_offsets(monkeypatch):
    torch.manual_seed(4301)
    indexer = _workflow_indexer()
    q, packed, weights = _projected(5)
    keys = pack_rows(torch.randn(80, 128, device="cuda", dtype=torch.bfloat16), "index")
    visible = torch.tensor([33, 34, 35, 66, 67], device="cuda", dtype=torch.int32)
    loads = []
    calls = []
    original = indexer.forward_prepared
    topk_phases = []
    original_topk = indexer.top_k.forward

    def topk(*args, **kwargs):
        topk_phases.append(kwargs["is_prefill"])
        return original_topk(*args, **kwargs)

    monkeypatch.setattr(indexer.top_k, "forward", topk)

    def forward(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(indexer, "forward_prepared", forward)

    def load(begin, end, width):
        loads.append((begin, end, width))
        starts = torch.zeros(end - begin, device="cuda", dtype=torch.int32)
        return _ChunkInputs(
            keys[:width, :64].contiguous().view(torch.int8),
            keys[:width, 64:].contiguous(),
            starts,
            torch.full_like(starts, width),
            torch.arange(width, device="cuda").expand(end - begin, -1),
            visible[begin:end],
        )

    chunks = [
        _QueryChunk(0, 3, 35, load=lambda: load(0, 3, 35), max_query_tokens=1),
        _QueryChunk(3, 5, 67, load=lambda: load(3, 5, 67), max_query_tokens=1),
    ]
    out = torch.empty(5, 32, dtype=torch.int32, device="cuda")
    indexer._run_csa2_chunks(
        chunks,
        packed[..., :64].contiguous().view(torch.int8),
        weights,
        packed[..., 64:].contiguous(),
        out,
        Mapping(),
        -1,
    )
    assert loads == [(0, 3, 35), (3, 5, 67)]
    assert calls == [1] * 5
    assert topk_phases == [True] * 5
    decoded_q = unpack_rows(packed, 128, "index").float()
    decoded_k = unpack_rows(keys, 128, "index").float()
    scores = (torch.einsum("qhd,kd->qhk", decoded_q, decoded_k).relu() * weights[..., None]).sum(1)
    _assert_selection(out, scores, torch.arange(80, device="cuda")[None] < visible[:, None])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [1, 2])
@torch.inference_mode()
def test_native_paged_owner_indexer_and_graph(monkeypatch, ratio):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native FP4 paged CSA2 integration requires SM100 family")
    torch.manual_seed(4302)
    layout = CSA2Layout(
        (ratio,),
        (0,),
        (0,),
        index_topk=32,
        candidate_source_layer_id=0 if ratio == 1 else None,
        candidate_topk_blocks=2,
    )
    packed_keys = pack_rows(torch.randn(512, 128, device="cuda", dtype=torch.bfloat16), "index")
    manager = _StridedOwner(layout, packed_keys)
    # One request carries two verification queries; declare that arena width.
    manager.max_total_draft_tokens = 1
    q, _, weights = _projected(3)
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=3)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.is_cuda_graph = True
    metadata.csa2_kv_sources = {0: 0}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, 2), (2, 3)]
    metadata.csa2_request_start_positions = [125, 127]
    metadata.csa2_request_lengths = [2, 1]
    metadata.csa2_global_max_positions = {0: 128}
    page_size = 128 // ratio
    metadata.csa2_global_page_sizes = {0: page_size}
    table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    visible = torch.tensor([63, 63, 64], device="cuda", dtype=torch.int32)
    metadata.csa2_global_page_tables = {0: table}
    metadata.csa2_visible_lengths = {0: visible}
    metadata.csa2_token_requests = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int64)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(3, 512, device="cuda", dtype=torch.bfloat16),
        index_q=q,
        index_weights=weights,
    )
    indexer = CSA2Indexer(layout, 0, 32, 128)
    native_calls = []
    topk_calls = []
    original_topk = TopK.forward

    def topk(selector, *args, **kwargs):
        topk_calls.append((selector.top_k, kwargs["is_prefill"]))
        return original_topk(selector, *args, **kwargs)

    monkeypatch.setattr(TopK, "forward", topk)
    original = indexer._call_paged_mqa_logits

    def paged(*args, **kwargs):
        native_calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(indexer, "_call_paged_mqa_logits", paged)
    monkeypatch.setattr(
        indexer, "_call_mqa_logits", lambda *a, **kw: pytest.fail("Dense path used")
    )
    for _ in range(3):
        output = indexer(state, 0, 3)
    assert native_calls and all(shape[:2] == (3, 1) for shape in native_calls)
    assert (32, False) in topk_calls
    if ratio == 1:
        assert metadata.csa2_candidates[0].shape[0] == 3  # Candidate blocks were published.
    bound = metadata.prepare_indexer(0)
    ptrs = (
        bound.csa2_indexer_k_cache.data_ptr(),
        bound.csa2_indexer_block_table.data_ptr(),
        bound.csa2_indexer_scheduler_metadata.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 3)
    for lengths, pages in (
        ([0, 1, 32], [[0, 1], [2, 3]]),
        ([64, 65, 96], [[2, 3], [0, -1]]),
        ([63, 63, 64], [[1, 0], [3, 2]]),
    ):
        visible.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        table.copy_(torch.tensor(pages, device="cuda", dtype=torch.int32))
        q.neg_()
        graph.replay()
        torch.cuda.synchronize()
        selected = output.clone()
        refreshed = metadata.prepare_indexer(0)
        assert ptrs == (
            refreshed.csa2_indexer_k_cache.data_ptr(),
            refreshed.csa2_indexer_block_table.data_ptr(),
            refreshed.csa2_indexer_scheduler_metadata.data_ptr(),
        )
        positions = torch.arange(128, device="cuda").expand(3, -1)
        req = metadata.csa2_token_requests
        physical = table[req[:, None], positions // page_size]
        valid = (physical >= 0) & (positions < visible[:, None])
        rows = read_index_rows(
            manager.get_index_pages(0),
            (physical.clamp_min(0) * page_size + positions % page_size).long(),
        )
        decoded_k = unpack_rows(rows, 128, "index").float()
        decoded_q = unpack_rows(pack_rows(q, "index"), 128, "index").float()
        scores = (
            torch.einsum("qhd,qkd->qhk", decoded_q, decoded_k).relu() * weights[..., None]
        ).sum(1)
        _assert_selection(selected, scores, valid)
    assert bool((manager.storage == 77).all())
    with pytest.raises(ValueError, match="complete model query batch"):
        indexer(state, 1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [5, 1])
# chunk_size 2 makes the source and consumer layers split queries across ranks differently.
@pytest.mark.parametrize("chunk_size", [16, 2])
@pytest.mark.parametrize("mixed", [False, True])
@torch.inference_mode()
def test_multi_rank_candidate_publication_and_reindex(count, chunk_size, mixed, monkeypatch):
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    world_size = mpi_world_size()
    if world_size not in (2, 4):
        pytest.skip("Run with two or four MPI ranks and one CUDA device per rank")
    for budget in (8, 32):
        torch.manual_seed(4303)
        layout = CSA2Layout(
            (1, 1),
            (0,),
            (0, 1),
            0,
            candidate_topk_blocks=budget,
            candidate_block_size=8,
            index_topk=32,
        )
        keys = pack_rows(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16), "index")
        manager = _StridedOwner(layout, keys)
        total = count + int(mixed)
        q, _, weights = _projected(total)
        metadata = CSA2TrtllmMetadata(max_num_requests=1 + int(mixed), max_num_tokens=total)
        metadata.kv_cache_manager = manager
        metadata.mapping = Mapping(world_size=world_size, rank=mpi_rank(), tp_size=world_size)
        metadata.csa2_kv_sources = {0: 0, 1: 0}
        metadata.csa2_num_context_requests = 1
        metadata.csa2_request_query_ranges = [(0, count)]
        metadata.csa2_request_start_positions = [59]
        metadata.csa2_request_lengths = [count]
        metadata.csa2_global_max_positions = {0: 128}
        metadata.csa2_global_page_sizes = {0: 128}
        metadata.csa2_global_page_tables = {0: torch.zeros(1, 1, device="cuda", dtype=torch.int32)}
        visible = torch.arange(60, 60 + count, device="cuda", dtype=torch.int32)
        metadata.csa2_visible_lengths = {0: visible, 1: visible}
        metadata.csa2_token_requests = torch.zeros(count, device="cuda", dtype=torch.int64)
        if mixed:
            metadata.csa2_request_query_ranges.append((count, total))
            metadata.csa2_request_start_positions.append(42)
            metadata.csa2_request_lengths.append(1)
            metadata.csa2_global_page_tables[0] = torch.zeros(
                2, 1, device="cuda", dtype=torch.int32
            )
            visible = torch.cat((visible, torch.tensor([43], device="cuda", dtype=torch.int32)))
            metadata.csa2_visible_lengths = {0: visible, 1: visible}
            metadata.csa2_token_requests = torch.cat(
                (metadata.csa2_token_requests, torch.ones(1, device="cuda", dtype=torch.int64))
            )
        metadata.csa2_indices = {}
        metadata.csa2_candidates = {}
        metadata.csa2_candidate_blocks = {}
        metadata.csa2_candidate_counts = {}
        metadata.indexer_max_chunk_size = chunk_size
        metadata.indexer_q_split_threshold = 0
        state = CSA2ForwardState(
            metadata=metadata,
            swa_kv=torch.zeros(total, 512, device="cuda", dtype=torch.bfloat16),
            index_q=q,
            index_weights=weights,
        )
        source = CSA2Indexer(layout, 0, 32, 128)
        source_result = source(state, 0, total).clone()
        gathered_candidates = metadata.csa2_candidates[0].clone()
        gathered_blocks = metadata.csa2_candidate_blocks[0].clone()
        gathered_counts = metadata.csa2_candidate_counts[0].clone()
        consumer = CSA2Indexer(layout, 1, 32, 128)
        state.index_q = -q
        shared_calls = []
        original_forward = consumer.forward_prepared

        def record_shared(*args, **kwargs):
            shared_calls.append(kwargs.get("sparse_indices") is not None)
            return original_forward(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(consumer, "forward_prepared", record_shared)
            consumer_result = consumer(state, 0, total).clone()
        local_count = (mpi_rank() + 1) * count // world_size - mpi_rank() * count // world_size
        # Sparse prefill consumers score the shared prefix once per rank.
        sparse = consumer.use_sparse_candidates
        if mixed:
            assert sum(shared_calls) == int(sparse and local_count > 0)
        else:
            assert shared_calls == ([sparse] if local_count else [])
        # Execute the same requests without query splitting through the same
        # scoring path, including the rank with zero local Q.
        metadata.mapping = Mapping()
        metadata.csa2_indices = {}
        metadata.csa2_candidates = {}
        metadata.csa2_candidate_blocks = {}
        metadata.csa2_candidate_counts = {}
        state.index_q = q
        expected_source = source(state, 0, total).clone()
        expected_candidates = metadata.csa2_candidates[0].clone()
        expected_blocks = metadata.csa2_candidate_blocks[0].clone()
        expected_counts = metadata.csa2_candidate_counts[0].clone()
        state.index_q = -q
        expected_consumer = consumer(state, 0, total).clone()
        torch.testing.assert_close(source_result, expected_source, atol=0, rtol=0)
        torch.testing.assert_close(gathered_candidates, expected_candidates, atol=0, rtol=0)
        torch.testing.assert_close(gathered_blocks, expected_blocks, atol=0, rtol=0)
        torch.testing.assert_close(gathered_counts, expected_counts, atol=0, rtol=0)
        torch.testing.assert_close(consumer_result, expected_consumer, atol=0, rtol=0)
        assert bool((gathered_candidates >= 0).any(1).all())


def _single_request_state(heads=8, two_owners=False):
    layout = (
        CSA2Layout((2, 1), (0, 1), (0, 1), index_topk=32)
        if two_owners
        else CSA2Layout((2,), (0,), (0,), index_topk=32)
    )
    keys = torch.full((512, 128), -1.0, device="cuda", dtype=torch.bfloat16)
    keys[64:128].fill_(1)
    keys[128:].fill_(3)
    manager = _StridedOwner(layout, pack_rows(keys, "index"))
    if two_owners:
        manager.owner_buffers = {
            0: manager.index_pages,
            1: _index_pages_from_rows(pack_rows(-keys, "index")),
        }
        manager.get_index_pages = lambda layer: manager.owner_buffers[layout.layer(layer).kv_source]
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=1)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.csa2_kv_sources = {i: i for i in layout.kv_source_layer_ids}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, 1)]
    metadata.csa2_request_start_positions = [3]
    metadata.csa2_request_lengths = [1]
    metadata.csa2_global_max_positions = {
        i: 256 // layout.compress_ratios[i] for i in layout.kv_source_layer_ids
    }
    metadata.csa2_global_page_sizes = {
        i: 128 // layout.compress_ratios[i] for i in layout.kv_source_layer_ids
    }
    metadata.csa2_global_page_tables = {
        i: torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        for i in layout.kv_source_layer_ids
    }
    metadata.csa2_visible_lengths = {
        i: torch.tensor([4 // layout.compress_ratios[i]], device="cuda", dtype=torch.int32)
        for i in layout.kv_source_layer_ids
    }
    metadata.csa2_token_requests = torch.zeros(1, device="cuda", dtype=torch.int64)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16),
        index_q=torch.ones(1, heads, 128, device="cuda", dtype=torch.bfloat16),
        index_weights=torch.full((1, heads), 1.0 / heads, device="cuda"),
    )
    return layout, manager, metadata, state


def _assert_state_selection(output, state, owner):
    metadata = state.metadata
    manager = metadata.kv_cache_manager
    width = metadata.csa2_global_max_positions[owner]
    positions = torch.arange(width, device="cuda").unsqueeze(0)
    slots = metadata.global_slot_tile(owner, 0, 1, positions)
    valid = (slots >= 0) & (positions < metadata.csa2_visible_lengths[owner][:, None])
    rows = read_index_rows(manager.get_index_pages(owner), slots.clamp_min(0))
    keys = unpack_rows(rows, 128, "index").float()
    query = unpack_rows(pack_rows(state.index_q, "index"), 128, "index").float()
    scores = (
        torch.einsum("qhd,qkd->qhk", query, keys).relu() * state.index_weights[..., None]
    ).sum(1)
    _assert_selection(output, scores, valid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_gathered_decode_graph_grows_past_warmup_prefix(monkeypatch):
    layout, manager, metadata, state = _single_request_state()
    metadata.is_cuda_graph = True
    indexer = CSA2Indexer(layout, 0, 8, 128)
    calls = []
    original = manager.gather_indexer_keys

    def gather(owner, slots):
        calls.append(slots.numel())
        return original(owner, slots)

    monkeypatch.setattr(manager, "gather_indexer_keys", gather)
    # Exercise the bounded fallback independently of native head coverage.
    monkeypatch.setattr(metadata, "prepare_indexer", lambda layer: None)
    monkeypatch.setattr(
        indexer,
        "_call_paged_mqa_logits",
        lambda *a, **kw: pytest.fail("Unavailable native staging requires gathered fallback"),
    )
    for _ in range(3):
        output = indexer(state, 0, 1)
    assert calls == [128] * 3
    torch.testing.assert_close(
        output[0, :2], torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 1)
    for length, second_page in ((96, 1), (96, 2), (0, -1), (65, 1)):
        metadata.csa2_visible_lengths[0].fill_(length)
        metadata.csa2_global_page_tables[0][0, 1] = second_page
        graph.replay()
        torch.cuda.synchronize()
        _assert_state_selection(output, state, 0)
        if length > 64:
            assert bool((output >= 64).any())
    assert bool((manager.storage == 77).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_paged_metadata_eager_growth_reuses_bounded_arena():
    layout, manager, metadata, state = _single_request_state(32, two_owners=True)
    metadata.is_cuda_graph = False
    previous = None
    previous_capacity = 0
    for visible in (1, 2, 31, 32, 63, 64, 65, 66, 96, 127, 128):
        metadata.csa2_request_start_positions = [visible * 2 - 1]
        metadata.csa2_visible_lengths[0].fill_(visible)
        metadata.csa2_visible_lengths[1].fill_(visible * 2)
        for owner in layout.kv_source_layer_ids:
            binding = metadata.prepare_indexer(owner)
            assert len(metadata._csa2_indexer_workspaces) == 1
            # Index pages are read in place; only descriptors are retained,
            # and not per exact sequence length or per owner.
            assert (
                binding.csa2_indexer_k_cache.data_ptr() == manager.get_index_pages(owner).data_ptr()
            )
            capacity = binding.csa2_indexer_block_table.numel()
            pointer = binding.csa2_indexer_block_table.data_ptr()
            if previous is not None and capacity == previous_capacity:
                assert pointer == previous
            previous, previous_capacity = pointer, capacity
    assert previous_capacity <= 4
    # A changed packed-query geometry must replace its eager arena too.
    metadata.csa2_request_query_ranges = [(0, 2)]
    metadata.csa2_request_lengths = [2]
    metadata.csa2_request_start_positions = [0]
    metadata.csa2_token_requests = torch.zeros(2, device="cuda", dtype=torch.int64)
    metadata.csa2_visible_lengths = {
        owner: torch.ones(2, device="cuda", dtype=torch.int32)
        for owner in layout.kv_source_layer_ids
    }
    metadata.prepare_indexer(0)
    assert len(metadata._csa2_indexer_workspaces) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_native_graph_switches_ratio_two_and_one_owner():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native FP4 paged CSA2 integration requires SM100 family")
    layout, manager, metadata, state = _single_request_state(32, two_owners=True)
    metadata.is_cuda_graph = True
    indexers = [CSA2Indexer(layout, owner, 32, 128) for owner in layout.kv_source_layer_ids]

    def run():
        # Each output is independently owned even though both calls reuse
        # the same descriptor buffers; each owner reads its own index pages.
        return tuple(indexer(state, 0, 1) for indexer in indexers)

    for _ in range(3):
        run()
    pointers = []
    for owner in layout.kv_source_layer_ids:
        metadata.prepare_indexer(owner)
        pointers.append(
            (
                metadata.csa2_indexer_k_cache.data_ptr(),
                metadata.csa2_indexer_block_table.data_ptr(),
                metadata.csa2_indexer_scheduler_metadata.data_ptr(),
            )
        )
    assert pointers[0][1] == pointers[1][1]
    assert pointers[0][0] == manager.get_index_pages(0).data_ptr()
    assert pointers[1][0] == manager.get_index_pages(1).data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for first, second in ((65, 129), (1, 2), (0, 0), (96, 191)):
        metadata.csa2_visible_lengths[0].fill_(first)
        metadata.csa2_visible_lengths[1].fill_(second)
        state.index_q.neg_()
        graph.replay()
        torch.cuda.synchronize()
        for owner, output in enumerate(outputs):
            _assert_state_selection(output, state, owner)
            binding = metadata.prepare_indexer(owner)
            assert pointers[owner] == (
                binding.csa2_indexer_k_cache.data_ptr(),
                binding.csa2_indexer_block_table.data_ptr(),
                binding.csa2_indexer_scheduler_metadata.data_ptr(),
            )
    assert len(metadata._csa2_indexer_workspaces) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_paged_descriptors_refresh_when_the_manager_changes():
    """Target and draft managers share the descriptor arena and advance serials in lockstep."""
    layout, manager, metadata, state = _single_request_state()
    metadata.is_cuda_graph = True
    metadata.csa2_visible_lengths[0].fill_(96)
    assert metadata.prepare_indexer(0) is metadata
    assert metadata.csa2_indexer_block_table.tolist() == [[0, 1]]
    keys = torch.ones(512, 128, device="cuda", dtype=torch.bfloat16)
    draft = _StridedOwner(layout, pack_rows(keys, "index"))
    metadata.kv_cache_manager = draft
    metadata.csa2_global_page_tables[0].copy_(
        torch.tensor([[3, 2]], device="cuda", dtype=torch.int32)
    )
    # Same forward serial, different manager: descriptors must follow the draft.
    assert metadata.prepare_indexer(0) is metadata
    assert metadata.csa2_indexer_k_cache.data_ptr() == draft.index_pages.data_ptr()
    assert metadata.csa2_indexer_block_table.tolist() == [[3, 2]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [2, 8, 16, 32, 64])
@pytest.mark.parametrize("use_dsl", [False, True])
@torch.inference_mode()
def test_native_paged_head_coverage_and_optional_dsl(monkeypatch, heads, use_dsl):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native FP4 paged variants require SM100")
    layout, _, metadata, state = _single_request_state(heads=heads)
    metadata.is_cuda_graph = True
    metadata.csa2_visible_lengths[0].fill_(96)
    indexer = CSA2Indexer(
        layout, 0, heads, 128, options=CSA2Params(use_cute_dsl_paged_mqa_logits=use_dsl)
    )
    calls = []
    native = indexer._paged_mqa_logits

    def record(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(indexer, "_paged_mqa_logits", record)
    for _ in range(3):
        output = indexer(state, 0, 1)
    assert len(calls) == 3
    _assert_state_selection(output, state, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 1)
    for length, page in ((65, 1), (96, 2), (0, -1)):
        metadata.csa2_visible_lengths[0].fill_(length)
        metadata.csa2_global_page_tables[0][0, 1] = page
        graph.replay()
        _assert_state_selection(output, state, 0)


def _temporal_gvr_case(monkeypatch, emission, *, page_holes=False, candidate_source=False):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Temporal GVR requires SM100")
    torch.manual_seed(7302)
    width, heads = 65536, 64
    layout = CSA2Layout(
        (1,),
        (0,),
        (0,),
        index_topk=512,
        candidate_source_layer_id=0 if candidate_source else None,
        candidate_topk_blocks=4,
    )
    manager = _StridedOwner(
        layout, pack_rows(torch.randn(width, 128, device="cuda", dtype=torch.bfloat16), "index")
    )
    manager.epoch = 1
    manager.request_epoch = lambda request: manager.epoch
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=1)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.request_ids = [13]
    metadata._num_ctx_tokens = 0
    metadata._num_contexts = 0
    metadata._num_tokens = 1
    metadata._num_generations = 1
    metadata.csa2_kv_sources = {0: 0}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, 1)]
    metadata.csa2_request_start_positions = [32767]
    metadata.csa2_request_lengths = [1]
    metadata.csa2_global_max_positions = {0: width}
    metadata.csa2_global_page_sizes = {0: 128}
    metadata.csa2_global_page_tables = {
        0: torch.arange(width // 128, device="cuda", dtype=torch.int32)[None, :]
    }
    if page_holes:
        metadata.csa2_global_page_tables[0][0, 2] = -1
    metadata.csa2_visible_lengths = {0: torch.tensor([32768], device="cuda", dtype=torch.int32)}
    metadata.csa2_token_requests = torch.zeros(1, device="cuda", dtype=torch.int64)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    metadata._csa2_forward_serial = 0
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16),
        index_q=torch.randn(1, heads, 128, device="cuda", dtype=torch.bfloat16),
        index_weights=torch.rand(1, heads, device="cuda") / heads,
    )
    if page_holes:
        state.index_weights.neg_()
    indexer = CSA2Indexer(
        layout,
        0,
        heads,
        128,
        options=CSA2Params(
            enable_heuristic_topk=True,
            use_self_sampling_topk=False,
            use_cute_dsl_paged_mqa_logits=emission,
            use_gvr_emission=emission,
        ),
    )
    assert indexer.top_k.needs_gvr_prior
    resets, emitted = [], []
    original_reset = indexer.top_k.reset_gvr_emission_rows
    original_logits = indexer._paged_mqa_logits

    def reset(rows):
        resets.append(True)
        original_reset(rows)

    def logits(*args, **kwargs):
        emitted.append(bool(kwargs.get("emission_kwargs")))
        return original_logits(*args, **kwargs)

    monkeypatch.setattr(indexer.top_k, "reset_gvr_emission_rows", reset)
    monkeypatch.setattr(indexer, "_paged_mqa_logits", logits)
    metadata.is_cuda_graph = True
    for _ in range(3):
        output = indexer(state, 0, 1)
    _assert_state_selection(output, state, 0)
    assert any(emitted) == emission
    metadata.is_cuda_graph = True
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 1)
    graph.replay()
    for start, epoch, expect_hint in ((32768, 1, True), (32769, 2, False), (511, 2, False)):
        metadata.csa2_request_start_positions = [start]
        metadata.csa2_visible_lengths[0].fill_(start + 1)
        manager.epoch = epoch
        metadata._csa2_forward_serial += 1
        resets_before = len(resets)
        prior = metadata.prepare_indexer_prior(0, 512)
        assert bool((prior >= 0).any()) == expect_hint
        if not expect_hint:
            assert len(resets) > resets_before
        state.index_q.copy_(torch.randn_like(state.index_q))
        if page_holes:
            metadata.csa2_global_page_tables[0][0, 2] = 2 if expect_hint else -1
        graph.replay()
        _assert_state_selection(output, state, 0)
        if candidate_source:
            candidates = metadata.csa2_candidates[0]
            assert candidates.shape == (1, 32)
            slots = metadata.global_slot_tile(0, 0, 1, candidates)
            assert bool(((candidates < 0) | (slots >= 0)).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("emission", [False, True])
@torch.inference_mode()
def test_temporal_gvr_prior_rewind_epoch_and_graph(monkeypatch, emission):
    _temporal_gvr_case(monkeypatch, emission)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("candidate_source", [False, True])
@torch.inference_mode()
def test_emission_masks_page_holes_and_signed_scores(monkeypatch, candidate_source):
    _temporal_gvr_case(monkeypatch, True, page_holes=True, candidate_source=candidate_source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_packed_query_state_avoids_requantization(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Packed-query native integration requires SM100")
    layout, _, metadata, state = _single_request_state(heads=64)
    metadata.is_cuda_graph = True
    metadata.csa2_visible_lengths[0].fill_(96)
    indexer = CSA2Indexer(layout, 0, 64, 128)
    expected = indexer(state, 0, 1)
    packed = pack_rows(state.index_q, "index")
    state.index_q = packed[..., :64].contiguous().view(torch.int8)
    state.index_q_scale = packed[..., 64:].contiguous().view(torch.int32).squeeze(-1)
    calls = []
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as implementation

    native_pack = implementation.pack_index_queries_split

    def record(*args, **kwargs):
        calls.append(True)
        return native_pack(*args, **kwargs)

    monkeypatch.setattr(implementation, "pack_index_queries_split", record)
    for _ in range(3):
        actual = indexer(state, 0, 1)
    torch.testing.assert_close(actual, expected)
    assert not calls
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = indexer(state, 0, 1)
    graph.replay()
    torch.testing.assert_close(actual, expected)
    assert not calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["eager", "bounded_graph", "growing_graph"])
@torch.inference_mode()
def test_decode_short_skip_respects_graph_admission(monkeypatch, mode):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native decode routing check requires SM100")
    layout, manager, metadata, state = _single_request_state(heads=64)
    metadata.is_cuda_graph = mode != "eager"
    if mode == "bounded_graph":
        metadata.csa2_global_max_positions[0] = 16
    metadata.csa2_visible_lengths[0].fill_(12)
    metadata.csa2_request_start_positions = [23]
    indexer = CSA2Indexer(layout, 0, 64, 128)
    calls = []
    native = indexer._paged_mqa_logits

    def record(*args, **kwargs):
        if mode != "growing_graph":
            pytest.fail("Eligible short decode must skip paged MQA")
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(indexer, "_paged_mqa_logits", record)
    if mode != "growing_graph":

        def unexpected(*args, **kwargs):
            pytest.fail("Eligible short decode must not gather/repack indexer cache")

        monkeypatch.setattr(manager, "gather_indexer_keys", unexpected)
        monkeypatch.setattr(indexer, "_call_mqa_logits", unexpected)
    for _ in range(3):
        output = indexer(state, 0, 1)
    _assert_state_selection(output, state, 0)
    if mode == "eager":
        metadata.csa2_global_page_tables[0][0, 0] = -1
        output = indexer(state, 0, 1)
        assert bool((output == -1).all())
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 1)
    metadata.csa2_visible_lengths[0].fill_(96 if mode == "growing_graph" else 16)
    graph.replay()
    _assert_state_selection(output, state, 0)
    assert bool(calls) == (mode == "growing_graph")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_short_candidate_source_still_computes_block_scores(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native candidate-source routing check requires SM100")
    _, manager, metadata, state = _single_request_state(heads=64)
    layout = CSA2Layout(
        (1,),
        (0,),
        (0,),
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_topk=32,
    )
    manager.layout = layout
    metadata.csa2_global_page_sizes[0] = 128
    metadata.csa2_global_max_positions[0] = 16
    metadata.csa2_visible_lengths[0].fill_(4)
    metadata.csa2_request_start_positions = [3]
    indexer = CSA2Indexer(layout, 0, 64, 128)
    calls = []
    native = indexer._paged_mqa_logits

    def record(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(indexer, "_paged_mqa_logits", record)
    output = indexer(state, 0, 1)
    assert len(calls) == 1
    assert metadata.csa2_candidates[0].shape == (1, 8)
    _assert_state_selection(output, state, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("query_width", [3, 17])
@torch.inference_mode()
def test_live_acceptance_graph_refreshes_native_paged_indexer(monkeypatch, query_width):
    """Real native FP4 logits/scheduler follow changed accepted device endpoints."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native FP4 paged logits require SM100 family")
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    layout, manager, metadata, state = _single_request_state(32, two_owners=True)
    manager.max_total_draft_tokens = query_width - 1
    metadata.is_cuda_graph = True
    metadata._num_tokens = query_width
    metadata.csa2_request_query_ranges = [(0, query_width)]
    metadata.csa2_request_lengths = [query_width]
    metadata.csa2_request_start_positions = [127]
    device = torch.device("cuda")
    metadata._csa2_ready_for_kv_update = True
    metadata._csa2_query_base = torch.tensor([127], dtype=torch.int32, device=device)
    metadata._csa2_query_lengths = torch.tensor([query_width], dtype=torch.int32, device=device)
    metadata._csa2_query_offsets = torch.arange(query_width, dtype=torch.int32, device=device)
    metadata.csa2_token_requests = torch.zeros(query_width, dtype=torch.int64, device=device)
    metadata.csa2_positions = torch.empty(query_width, dtype=torch.int32, device=device)
    metadata.csa2_replay_start_positions = torch.zeros(1, dtype=torch.int64, device=device)
    metadata._csa2_swa_page_tables = {
        owner: torch.tensor([[0, 1]], dtype=torch.int32, device=device) for owner in (0, 1)
    }
    metadata._csa2_buffers = {}
    metadata._csa2_swa_descriptors = metadata._swa_table_descriptors(2)
    metadata._csa2_swa_ratios = metadata._swa_ratio_tensor(layout.compress_ratios)
    # The forward resolves the SWA slots; keep them for the checks.
    kept = {
        owner: tuple(torch.empty(query_width, dtype=torch.int64, device=device) for _ in range(2))
        for owner in (0, 1)
    }
    metadata._csa2_source_geometry = {
        owner: (metadata._csa2_query_base, metadata._csa2_query_lengths) for owner in (0, 1)
    }
    metadata.csa2_main_write_slots = {
        owner: torch.empty(query_width, dtype=torch.int64, device=device) for owner in (0, 1)
    }
    metadata._csa2_compressed_positions = {
        owner: torch.empty(query_width, dtype=torch.int32, device=device) for owner in (0, 1)
    }
    metadata._csa2_compression = {
        0: CSA2CompressionBatch(
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            torch.empty(1, dtype=torch.int32, device=device),
            torch.empty(1, dtype=torch.int32, device=device),
            torch.tensor([0, query_width], dtype=torch.int32, device=device),
            torch.empty(2, dtype=torch.int32, device=device),
            query_width,
            128,
            query_width,
        )
    }
    metadata._csa2_owner_descriptors = metadata._owner_slot_descriptors()
    state.index_q = state.index_q.expand(query_width, -1, -1).clone()
    state.index_weights = state.index_weights.expand(query_width, -1).clone()
    state.swa_kv = state.swa_kv.expand(query_width, -1).clone()
    indexers = [CSA2Indexer(layout, owner, 32, 128) for owner in (0, 1)]
    for indexer in indexers:
        monkeypatch.setattr(
            indexer,
            "_call_mqa_logits",
            lambda *a, **kw: pytest.fail("Expected native paged logits, not gathered fallback"),
        )

    def run():
        metadata.on_update_kv_lens()
        metadata._ensure_swa_slots()  # As entering the first layer does.
        for owner in (0, 1):
            kept[owner][0].copy_(metadata.csa2_swa_write_slots[owner])
            kept[owner][1].copy_(metadata.csa2_visible_lengths[owner])
        return tuple(indexer(state, 0, query_width) for indexer in indexers)

    metadata.kv_lens_cuda[0] = 130
    for _ in range(3):
        run()
    pointers = (
        metadata.csa2_positions.data_ptr(),
        metadata.csa2_main_write_slots[0].data_ptr(),
        metadata.csa2_indexer_scheduler_metadata.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for endpoint in (129, 130, 128, 131):
        metadata.kv_lens_cuda[0] = endpoint
        state.index_q.neg_()
        graph.replay()
        torch.cuda.synchronize()
        positions = torch.arange(endpoint - query_width, endpoint, device=device)
        torch.testing.assert_close(metadata.csa2_positions.long(), positions)
        torch.testing.assert_close(kept[0][0], positions)
        for owner, selected in enumerate(output):
            ratio = layout.compress_ratios[owner]
            visible = (positions + 1) // ratio
            torch.testing.assert_close(kept[owner][1], visible)
            groups = list(range((endpoint - query_width) // ratio, endpoint // ratio))
            assert metadata.csa2_main_write_slots[owner].tolist() == groups + [-1] * (
                query_width - len(groups)
            )
            width = metadata.csa2_global_max_positions[owner]
            logical = torch.arange(width, device=device).expand(query_width, -1)
            slots = metadata.global_slot_tile(owner, 0, query_width, logical)
            keys = unpack_rows(
                read_index_rows(manager.get_index_pages(owner), slots.clamp_min(0)), 128, "index"
            ).float()
            query = unpack_rows(pack_rows(state.index_q, "index"), 128, "index").float()
            scores = (
                torch.einsum("qhd,qkd->qhk", query, keys).relu() * state.index_weights[..., None]
            ).sum(1)
            _assert_selection(selected, scores, (slots >= 0) & (logical < visible[:, None]))
        batch = metadata._csa2_compression[0]
        assert batch.start_positions.item() == endpoint - query_width
        assert batch.kv_lengths.item() == endpoint
        assert batch.cu_compressed_lengths.tolist() == [
            0,
            endpoint // 2 - (endpoint - query_width) // 2,
        ]
        # The serialized ratio-one owner ran last, so its native scheduling
        # inputs and refreshed schedule occupy the shared arena now.
        expected_context = (positions + 1).int().clamp_min(1)[:, None]
        torch.testing.assert_close(metadata.csa2_indexer_context_lengths, expected_context)
        expected_schedule = get_paged_mqa_logits_metadata(
            expected_context,
            64,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        torch.testing.assert_close(metadata.csa2_indexer_scheduler_metadata, expected_schedule)
        assert pointers == (
            metadata.csa2_positions.data_ptr(),
            metadata.csa2_main_write_slots[0].data_ptr(),
            metadata.csa2_indexer_scheduler_metadata.data_ptr(),
        )


@pytest.mark.cpu_only
def test_candidates_pin_latest_block_and_mask_future():
    keys = torch.tensor([6.0, 4.0, 3.0, 2.0, 0.5, 6.0], dtype=torch.bfloat16)[:, None].expand(
        -1, 128
    )
    layout, state = _selection_state(1, keys, torch.tensor([5]))
    _run_indexer(layout, 0, state)
    assert state.metadata.csa2_candidates[0].tolist() == [[4, -1]]
    state.main_kv = state.index_k = None
    _run_indexer(layout, 1, state)
    assert state.metadata.csa2_indices[1].tolist() == [[4, -1, -1]]


@pytest.mark.cpu_only
@pytest.mark.parametrize("queries,width", [(0, 0), (0, 8), (2, 0), (2, 8)])
def test_empty_visibility(queries, width):
    layout, state = _selection_state(
        queries,
        torch.zeros(width, 128, dtype=torch.bfloat16),
        torch.zeros(queries, dtype=torch.int32),
    )
    _run_indexer(layout, 0, state)
    assert state.metadata.csa2_indices[0].shape == (queries, 3)
    assert torch.all(state.metadata.csa2_indices[0] == -1)
    assert torch.all(state.metadata.csa2_candidates[0] == -1)


@pytest.mark.cpu_only
def test_full_reuse_reindex_private_swa_and_shared_sink():
    outputs, routing, cache, batch = _run_modes("cpu")
    assert routing.csa2_indices[0].tolist() == [[0]]
    assert routing.csa2_indices[2].tolist() == [[1]]
    # q=0 means selected global KV, private SWA, and sink each get 1/3.
    for actual, expected in zip(outputs, [2.0, 3.0, 5.0]):
        torch.testing.assert_close(
            actual.float(), torch.full_like(actual.float(), expected), atol=0.02, rtol=0
        )
    layout = CSA2Layout((1, 1), (0,), (0,))
    with pytest.raises(ValueError, match="across forwards"):
        routing.enter_layer(layout.layer(0))
    q = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="source did not run"):
        batch.reset_routing()
        _prediction_reference(layout, 1, q, q[:, 0], torch.zeros(1), batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_modes_cuda_graph_changed_routing():
    # Preserve the independent analytical oracle before exercising replay changes.
    outputs, routing, _, _ = _run_modes("cuda")
    torch.cuda.synchronize()
    assert routing.csa2_indices[2].tolist() == [[1]]
    torch.testing.assert_close(outputs[-1], torch.full_like(outputs[-1], 5.0), atol=0.02, rtol=0)
    _run_modes("cuda", graph_replay=True)


# Candidate blocks (of 8 positions) hand-published by the decode fixtures: an
# unordered list with a duplicate and an out-of-range block, as a source would
# never publish them but the publication helper below must normalize them.
_MANUAL_BLOCKS = [22, 0, 15, 16, 23, 22, 40, 4]


def _publish_manual_candidates(metadata, layer, block_ids, block_size=8):
    """Publish hand-written candidate blocks the way a candidate source does.

    Per query row the blocks are filtered to visible, allocated positions,
    sorted and de-duplicated; the positions form a valid prefix per row, the
    sparse block ids repeat the last valid block, and ``counts`` holds the valid
    column count. Existing publications are updated in place so captured graphs
    keep reading the same buffers. Also binds the generation-row -> request map.
    """
    visible = metadata.csa2_visible_lengths[layer + 1].to(torch.int64)
    count = visible.shape[0]
    device = visible.device
    max_positions = metadata.csa2_global_max_positions[0]
    table = metadata.csa2_global_page_tables[0]
    page_size = metadata.csa2_global_page_sizes[0]
    requests = metadata.csa2_token_requests.long()
    width = -(-len(block_ids) // 4) * 4 * block_size
    ids = torch.tensor(sorted(set(block_ids)), dtype=torch.int64, device=device)
    first = ids * block_size
    page = table[requests[:, None], (first // page_size).clamp(0, table.shape[1] - 1)]
    reachable = (first[None, :] < visible[:, None]) & (first[None, :] < max_positions)
    reachable &= page >= 0
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.where(reachable, ids, sentinel).sort(dim=1).values
    positions = ordered[..., None] * block_size + torch.arange(block_size, device=device)
    valid = (ordered != sentinel)[..., None] & (positions < visible[:, None, None])
    candidates = F.pad(
        torch.where(valid, positions, -1).flatten(1),
        (0, width - ids.numel() * block_size),
        value=-1,
    )
    num_valid = (ordered != sentinel).sum(1, keepdim=True)
    last = torch.where(num_valid > 0, ordered.gather(1, (num_valid - 1).clamp_min(0)), 0)
    blocks = torch.where(ordered == sentinel, last, ordered)
    blocks = torch.cat((blocks, last.expand(-1, width // block_size - blocks.shape[1])), 1).int()
    counts = (candidates >= 0).sum(1, keepdim=True).int()
    for store, value in (
        (metadata.csa2_candidates, candidates),
        (metadata.csa2_candidate_blocks, blocks),
        (metadata.csa2_candidate_counts, counts),
    ):
        if layer in store and store[layer].shape == value.shape:
            store[layer].copy_(value)
        else:
            store[layer] = value.clone()
    contexts = metadata.csa2_num_context_requests
    ranges = metadata.csa2_request_query_ranges
    decode_start = ranges[contexts][0] if contexts < len(ranges) else count
    rows = (requests[decode_start:] - contexts).int()
    existing = getattr(metadata, "csa2_decode_row_requests", None)
    # Captured graphs read this map by address as well: update it in place.
    if existing is not None and existing.shape == rows.shape:
        existing.copy_(rows)
    else:
        metadata.csa2_decode_row_requests = rows


def _candidate_decode_fixture(*, mixed=False, random=False):
    layout = CSA2Layout(
        (1, 1), (0,), (0, 1), candidate_source_layer_id=0, candidate_topk_blocks=32, index_topk=4
    )
    torch.manual_seed(7413)
    identifiers = ((torch.arange(512, device="cuda") * 37) % 512) ^ 173
    identifiers[52], identifiers[436] = 511, 510
    keys = torch.zeros(512, 128, dtype=torch.bfloat16, device="cuda")
    keys[:, :9] = ((identifiers[:, None] >> torch.arange(9, device="cuda")) & 1).bfloat16()
    if random:
        keys.normal_()
    manager = _StridedOwner(layout, pack_rows(keys, "index"))
    manager.max_total_draft_tokens = 1
    count = 4 if mixed else 3
    metadata = CSA2TrtllmMetadata(max_num_requests=3 if mixed else 2, max_num_tokens=count)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.is_cuda_graph = True
    metadata.csa2_kv_sources = {0: 0, 1: 0}
    metadata.csa2_num_context_requests = int(mixed)
    metadata.csa2_request_query_ranges = [(0, 1), (1, 3), (3, 4)] if mixed else [(0, 2), (2, 3)]
    metadata.csa2_request_start_positions = [191, 191, 192] if mixed else [191, 192]
    metadata.csa2_request_lengths = [1, 2, 1] if mixed else [2, 1]
    metadata.csa2_global_max_positions = {0: 256}
    metadata.csa2_global_page_sizes = {0: 128}
    metadata.csa2_global_page_tables = {
        0: torch.tensor(
            [[3, 2], [1, 0], [2, 3]] if mixed else [[1, 0], [2, 3]],
            dtype=torch.int32,
            device="cuda",
        )
    }
    metadata.csa2_visible_lengths = {1: torch.full((count,), 192, dtype=torch.int64, device="cuda")}
    metadata.csa2_token_requests = torch.tensor([0, 1, 1, 2] if mixed else [0, 0, 1], device="cuda")
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    _publish_manual_candidates(metadata, 0, _MANUAL_BLOCKS)
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    q = torch.zeros(count, 32, 128, dtype=torch.bfloat16, device="cuda")
    q[:, :9, :9] = torch.eye(9, dtype=torch.bfloat16, device="cuda")
    weights = torch.zeros(count, 32, device="cuda")
    for row in range(count):
        weights[row, :9] = 2.0 ** (
            torch.arange(9, device="cuda") if row % 2 else torch.arange(8, -1, -1, device="cuda")
        )
    if random:
        q.normal_()
        weights.uniform_(0, 1.0 / 32)
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(count, 512, dtype=torch.bfloat16, device="cuda"),
        index_q=q,
        index_weights=weights,
    )
    indexer = CSA2Indexer(
        layout, 1, 32, 128, options=CSA2Params(layout=layout, skip_indexer_for_short_seqs=False)
    )
    return indexer, state


def _candidate_decode_run(indexer, state, monkeypatch, *, old):
    """Run the consumer; ``old`` forces the per-query candidate gathers.

    Returns the selection, the masked scores of every query in the order of its
    published candidate columns (whatever column order the scoring path used),
    and the number of dense / sparse kernel calls.
    """
    metadata = state.metadata
    captured, calls = [], {"dense": 0, "sparse_prefill": 0, "sparse_paged": 0}
    select = indexer._select_mapped_logits
    dense = indexer._call_mqa_logits
    sparse = indexer._sparse_candidate_logits
    candidates = metadata.csa2_candidates[0]
    width = max(metadata.csa2_global_max_positions[0], int(candidates.max()) + 1)
    rows_done = [0]

    def collect(logits, starts, ends, positions, visible, *args, **kwargs):
        # ``None`` bounds mean every logits column is the row's domain.
        if starts is None:
            starts = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
        if ends is None:
            ends = torch.full_like(starts, logits.shape[1])
        columns = starts[:, None].long() + torch.arange(positions.shape[1], device=logits.device)
        scores = logits.gather(1, columns.clamp(0, logits.shape[1] - 1))
        valid = (columns < ends[:, None]) & (columns < logits.shape[1])
        valid &= (positions >= 0) & (positions < visible[:, None])
        scores = scores.masked_fill(~valid, -torch.inf)
        # Re-key the scores from the path's column order to the candidate order.
        by_position = torch.full((logits.shape[0], width + 1), -torch.inf, device=logits.device)
        by_position.scatter_(1, torch.where(valid, positions.long(), width), scores)
        rows = slice(rows_done[0], rows_done[0] + logits.shape[0])
        rows_done[0] += logits.shape[0]
        ordered = by_position.gather(1, candidates[rows].clamp_min(0))
        captured.append(ordered.masked_fill(candidates[rows] < 0, -torch.inf))
        return select(logits, starts, ends, positions, visible, *args, **kwargs)

    def count_calls(name, function):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapped

    with monkeypatch.context() as context:
        context.setattr(indexer, "_select_mapped_logits", collect)
        context.setattr(indexer, "_call_mqa_logits", count_calls("dense", dense))

        def count_sparse(*args, **kwargs):
            calls["sparse_paged" if kwargs.get("paged") is not None else "sparse_prefill"] += 1
            return sparse(*args, **kwargs)

        context.setattr(indexer, "_sparse_candidate_logits", count_sparse)
        if old:
            context.setattr(indexer, "use_sparse_candidates", False)
        output = indexer(state, 0, state.swa_kv.shape[0]).clone()
    return output, torch.cat(captured), calls


def _bf16_tolerance(scores):
    """Absolute tolerance for bf16 sparse logits against fp32 reference scores."""
    finite = scores[torch.isfinite(scores)]
    return float(finite.abs().max()) * 2**-7 if finite.numel() else 0.0


def _check_candidate_multiset(output, scores, candidates, tolerance):
    from collections import Counter

    for row in range(output.shape[0]):
        ids = output[row][output[row] >= 0].tolist()
        valid = scores[row] > -torch.inf
        possible = Counter(candidates[row][valid].tolist())
        assert not (Counter(ids) - possible)
        values = [scores[row][candidates[row] == identity][0] for identity in ids]
        actual = torch.stack(values).sort(descending=True).values if values else scores.new_empty(0)
        expected = scores[row][valid].topk(min(output.shape[1], int(valid.sum()))).values
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixed", [False, True])
@torch.inference_mode()
def test_candidate_decode_batched_exact_and_mixed(monkeypatch, mixed):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native paged FP4 requires SM100 family")
    indexer, state = _candidate_decode_fixture(mixed=mixed)
    # Keep active queries on both requests, plus an empty first query and a hole.
    state.metadata.csa2_visible_lengths[1][0] = 0
    state.metadata.csa2_global_page_tables[0][0, 0] = -1
    _publish_manual_candidates(state.metadata, 0, _MANUAL_BLOCKS)
    candidate = state.metadata.csa2_candidates[0].clone()
    if not _sparse_kernels_available():
        pytest.skip("DeepGEMM sparse MQA logits required")
    old, scores, control = _candidate_decode_run(indexer, state, monkeypatch, old=True)
    actual, new_scores, calls = _candidate_decode_run(indexer, state, monkeypatch, old=False)
    assert control == {
        "dense": len(state.metadata.csa2_request_query_ranges),
        "sparse_prefill": 0,
        "sparse_paged": 0,
    }
    assert calls == {"dense": 0, "sparse_prefill": int(mixed), "sparse_paged": 1}
    # The sparse kernels emit bf16 logits; the per-query gather reference is fp32.
    torch.testing.assert_close(new_scores, scores, atol=0, rtol=2**-7)
    torch.testing.assert_close(state.metadata.csa2_candidates[0], candidate, atol=0, rtol=0)
    _check_candidate_multiset(actual, scores, candidate, _bf16_tolerance(scores))
    if mixed:
        # Without paged sparse descriptors the decode rows fall back to the
        # per-query candidate gathers without row-offset loss.
        with monkeypatch.context() as context:
            context.setattr(state.metadata, "prepare_sparse_indexer", lambda *a: None)
            fallback, fallback_scores, fallback_calls = _candidate_decode_run(
                indexer, state, monkeypatch, old=False
            )
        assert fallback_calls == {"dense": 2, "sparse_prefill": 1, "sparse_paged": 0}
        torch.testing.assert_close(fallback, old, atol=0, rtol=0)
        torch.testing.assert_close(fallback_scores, scores, atol=0, rtol=2**-7)
    # An admitted domain wider than the candidate pool changes nothing: only
    # the published candidate blocks are scored.
    state.metadata.csa2_global_max_positions[0] = 257
    wider, wider_scores, wider_calls = _candidate_decode_run(indexer, state, monkeypatch, old=False)
    assert wider_calls == calls
    _check_candidate_multiset(wider, scores, candidate, _bf16_tolerance(scores))
    torch.testing.assert_close(wider_scores, scores, atol=0, rtol=2**-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_candidate_decode_batched_graph_refresh(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native paged FP4 requires SM100 family")
    indexer, state = _candidate_decode_fixture()
    metadata = state.metadata
    for _ in range(3):
        indexer(state, 0, 3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = indexer(state, 0, 3)
    pointers = (
        metadata.csa2_indexer_k_cache.data_ptr(),
        metadata.csa2_indexer_block_table.data_ptr(),
    )
    for step, lengths in enumerate(([192, 193, 190], [0, 0, 0], [129, 191, 192])):
        metadata.csa2_visible_lengths[1].copy_(torch.tensor(lengths, device="cuda"))
        metadata.csa2_global_page_tables[0][0, 1] = -1 if step == 0 else 0
        # A source republishes candidates for the new visibility/pages every step
        _publish_manual_candidates(
            metadata, 0, _MANUAL_BLOCKS[step:] + [40 + i for i in range(step)]
        )
        state.index_q[:, :9, :9].copy_(torch.eye(9, device="cuda").roll(step, dims=1))
        expected, scores, _ = _candidate_decode_run(indexer, state, monkeypatch, old=True)
        graph.replay()
        torch.cuda.synchronize()
        _check_candidate_multiset(
            actual, scores, metadata.csa2_candidates[0], _bf16_tolerance(scores)
        )
        assert pointers == (
            metadata.csa2_indexer_k_cache.data_ptr(),
            metadata.csa2_indexer_block_table.data_ptr(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("phase", ["decode", "prefill"])
@torch.inference_mode()
def test_candidate_batched_random_logits(monkeypatch, phase):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native paged FP4 requires SM100 family")
    if phase == "decode":
        indexer, state = _candidate_decode_fixture(random=True)
        old, scores, _ = _candidate_decode_run(indexer, state, monkeypatch, old=True)
        actual, new_scores, _ = _candidate_decode_run(indexer, state, monkeypatch, old=False)
    else:
        indexer, state = _candidate_prefill_fixture(random=True)
        old, scores, _, _, _ = _candidate_prefill_run(indexer, state, monkeypatch, True)
        actual, new_scores, _, _, _ = _candidate_prefill_run(indexer, state, monkeypatch, False)
    valid = torch.isfinite(scores)
    assert torch.equal(valid, torch.isfinite(new_scores))
    difference = torch.where(valid, new_scores - scores, 0)
    top = scores.topk(5).values
    gaps = top[:, 3] - top[:, 4]
    # Existing native selection tests allow only 0.005 absolute cutoff error.
    torch.testing.assert_close(new_scores[valid], scores[valid], atol=0.005, rtol=2**-7)
    _check_candidate_multiset(
        actual, scores, state.metadata.csa2_candidates[0], 0.005 + _bf16_tolerance(scores)
    )
    separated = gaps > 2 * difference.abs().amax(1)
    torch.testing.assert_close(actual[separated], old[separated], atol=0, rtol=0)


def _candidate_prefill_fixture(*, random=False):
    indexer, state = _candidate_decode_fixture(mixed=True, random=random)
    metadata = state.metadata
    count = 206
    rows = torch.arange(count, device="cuda") % state.index_q.shape[0]
    state.index_q = state.index_q[rows]
    state.index_weights = state.index_weights[rows]
    state.swa_kv = torch.zeros(count, 512, dtype=torch.bfloat16, device="cuda")
    metadata.is_cuda_graph = False
    metadata.csa2_num_context_requests = 2
    metadata.csa2_request_query_ranges = [(0, 137), (137, 204), (204, 206)]
    metadata.csa2_request_start_positions = [32, 144, 192]
    metadata.csa2_request_lengths = [137, 67, 2]
    metadata.csa2_token_requests = torch.tensor([0] * 137 + [1] * 67 + [2] * 2, device="cuda")
    metadata.csa2_global_page_tables = {
        0: torch.tensor([[1, 0], [2, 3], [0, 1]], device="cuda", dtype=torch.int32)
    }
    metadata.csa2_visible_lengths[1] = torch.cat(
        (
            torch.arange(33, 170, device="cuda"),
            torch.arange(145, 212, device="cuda"),
            torch.tensor([193, 194], device="cuda"),
        )
    )
    metadata.csa2_visible_lengths[1][5] = 0
    metadata.csa2_global_page_tables[0][0, 1] = -1
    _publish_manual_candidates(metadata, 0, _MANUAL_BLOCKS)
    return indexer, state


def _candidate_prefill_run(indexer, state, monkeypatch, old):
    shapes, gathers = [], []
    manager = state.metadata.kv_cache_manager
    original_sparse = indexer._sparse_candidate_logits
    original_gather = manager.gather_indexer_keys

    def sparse(*args, **kwargs):
        if kwargs.get("paged") is None:
            shapes.append((args[0].shape[0], kwargs["keys"][0].shape[0]))
        return original_sparse(*args, **kwargs)

    def gather(layer, slots):
        gathers.append(slots.numel())
        return original_gather(layer, slots)

    with monkeypatch.context() as context:
        context.setattr(indexer, "_sparse_candidate_logits", sparse)
        context.setattr(manager, "gather_indexer_keys", gather)
        result = _candidate_decode_run(indexer, state, monkeypatch, old=old)
    return (*result, shapes, gathers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_candidate_shared_prefill_exact_and_mixed(monkeypatch):
    if not _sparse_kernels_available():
        pytest.skip("DeepGEMM sparse MQA logits required")
    indexer, state = _candidate_prefill_fixture()
    candidates = state.metadata.csa2_candidates[0].clone()
    old, scores, old_calls, _, old_gathers = _candidate_prefill_run(
        indexer, state, monkeypatch, True
    )
    actual, new_scores, calls, shapes, gathers = _candidate_prefill_run(
        indexer, state, monkeypatch, False
    )
    assert calls == {"dense": 0, "sparse_prefill": 2, "sparse_paged": 1}
    # One sparse logits call per prefill request over its whole gathered prefix.
    assert shapes == [(137, 169), (67, 211)]
    # Decode reads index pages in place; only the two prefill tiles gather rows.
    assert gathers == [169, 211]
    torch.testing.assert_close(new_scores, scores, atol=0, rtol=2**-7)
    _check_candidate_multiset(actual, scores, candidates, _bf16_tolerance(scores))
    torch.testing.assert_close(state.metadata.csa2_candidates[0], candidates, atol=0, rtol=0)


@pytest.mark.cpu_only
def test_candidate_shared_prefill_tile_bounds(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as implementation

    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        candidate_source_layer_id=0,
        candidate_topk_blocks=1 << 20,
        index_topk=4,
    )
    indexer = CSA2Indexer(layout, 1, 32, 128)
    # The DSV4.1 pool (16K positions) gets a positive tile inside the native logits budget.
    assert 0 < indexer._candidate_prefill_tile_size(16384) <= 512
    # Very wide pools still tile, bounded by the per-query transient budget.
    assert 0 < indexer._candidate_prefill_tile_size(2000000) < 32
    with monkeypatch.context() as limited:
        limited.setattr(implementation, "_INDEXER_MQA_LOGITS_ELEM_BUDGET", 8191)
        assert indexer._candidate_prefill_tile_size(16384) == 1
    assert indexer._candidate_prefill_tile_size(0) == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "count,layout_name",
    [
        (5, "contiguous"),
        (7, "strided"),
    ],
    ids=[
        "5-contiguous-bf16",
        "7-strided-bf16",
    ],
)
@torch.inference_mode()
def test_query_pack_reuses_exact_index_quantizer(count, layout_name, monkeypatch):
    from types import SimpleNamespace

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Exact fused query packing requires SM100 family")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(91)
    heads = 32
    shape = (count, heads, 128)
    dtype = torch.bfloat16
    if layout_name == "strided":
        query = torch.empty(count, heads, 256, dtype=dtype, device=device)[..., ::2]
    else:
        query = torch.empty(shape, dtype=dtype, device=device)
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            6 * 2.0**-126,
            2.0**-126,
            2.0**-125,
            1.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            torch.finfo(torch.bfloat16).max,
        ],
        dtype=dtype,
        device=device,
    )
    values = torch.randn(shape, device=device, dtype=dtype)
    # Separate quantization groups exercise nonfinite and finite edges.
    values.flatten()[: edges.numel() * 32].copy_(edges.repeat_interleave(32))
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device, dtype=dtype)
    adjacent = torch.cat(
        (
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf"))),
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
        )
    )
    offset = edges.numel() * 32
    values.flatten()[offset : offset + adjacent.numel()].copy_(adjacent)
    values.flatten()[offset + 31] = 6  # Force scale=1 for midpoint groups.

    query.copy_(values)
    layer = CSA2Layout((2,), (0,), (0,), index_topk=32)
    indexer = CSA2Indexer(layer, 0, heads, 128)
    state = SimpleNamespace(
        metadata=SimpleNamespace(kv_cache_manager=object()),
        index_q=query,
        index_q_scale=None,
        index_weights=torch.ones(count, heads, device=device),
        swa_kv=torch.empty(count, 512, device=device, dtype=torch.bfloat16),
    )

    def observe(metadata, swa, data, *args, q_scale):
        assert data.is_contiguous() and q_scale.is_contiguous()
        assert data.shape == (count, heads, 64) and q_scale.shape == (count, heads, 4)
        return torch.cat((data.view(torch.uint8), q_scale), dim=-1)

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    calls = []
    quantize = kernel.quantize_index_queries

    def witnessed_quantize(values, data, scales):
        calls.append((values.shape, data.shape, scales.shape))
        return quantize(values, data, scales)

    monkeypatch.setattr(kernel, "quantize_index_queries", witnessed_quantize)
    monkeypatch.setattr(indexer, "sparse_attn_indexer", observe)
    expected = pack_rows(query, "index")
    actual = indexer(state, 0, count)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # The fused quantizer writes the split data/scale outputs directly.
    assert calls == [((count * heads, 128), (count * heads, 64), (count * heads, 4))]
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=device)
    assert capture_stream.device == query.device
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = indexer(state, 0, count)
    query.copy_(-values)
    graph.replay()
    torch.testing.assert_close(captured, pack_rows(query, "index"), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_constant_rows_retained_only_for_captured_graphs():
    """Eager constants are transient; rows a graph captured stay alive across geometries."""
    indexer = _prepared_indexer()
    device = torch.device("cuda")
    eager = indexer._constant_rows(5, 0, device)
    assert indexer._constant_rows(5, 0, device) is not eager
    assert not indexer.__dict__.get("_constant_row_cache")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = indexer._constant_rows(7, 3, device)
        assert indexer._constant_rows(7, 3, device) is captured
    # Many eager geometries never evict the rows a graph replays against.
    for count in range(1, 200):
        indexer._constant_rows(count, 0, device)
    assert indexer._constant_rows(7, 3, device) is captured
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, torch.full((7,), 3, dtype=torch.int32, device=device))


# ---- DeepGEMM sparse (candidate-only) MQA logits ----


def _sparse_candidate_layout(topk_blocks=4, block_size=8, topk=32):
    """Source layer 0 publishes candidates; Reindex layer 1 consumes them."""
    return CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        0,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=block_size,
        index_topk=topk,
    )


def _sparse_kernels_available():
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and _HAS_SPARSE_MQA_LOGITS
    )


def _publish_sparse(indexer, scores, lengths, candidate_width=None):
    """Run the source publication on a device and return (positions, blocks, counts)."""
    positions, blocks, counts = {}, {}, {}
    if not scores.is_cuda:
        indexer.top_k.prefill_implementation = TopKImplementation.TORCH
    indexer._configure_candidate_topk(scores.shape[1])
    indexer._publish_candidates(
        scores, lengths, positions, 0, scores.shape[0], candidate_width, blocks, counts
    )
    return positions[indexer.layer_idx], blocks[indexer.layer_idx], counts[indexer.layer_idx]


@pytest.mark.cpu_only
@pytest.mark.parametrize("block_size,sparse_block", [(8, 8), (16, 16), (32, 16)])
def test_publish_candidates_sparse_blocks(block_size, sparse_block):
    source = CSA2Indexer(_sparse_candidate_layout(4, block_size), 0, 32, 128)
    assert source.sparse_block == sparse_block
    ratio = block_size // sparse_block
    torch.manual_seed(4310)
    width = 12 * block_size + 5
    lengths = torch.tensor([width, 3 * block_size + 1, 1, 0], dtype=torch.int32)
    scores = torch.rand(4, width)
    reference = _reference_select_candidate_positions(scores, lengths, 4, block_size)
    # The source scores arrive masked to the visible prefix (see _select_mapped_logits)
    scores = scores.masked_fill(torch.arange(width)[None, :] >= lengths[:, None], -torch.inf)
    positions, blocks, counts = _publish_sparse(source, scores, lengths, 4 * block_size)
    assert blocks.dtype == torch.int32 and blocks.shape == (4, 4 * ratio)
    assert counts.dtype == torch.int32 and counts.shape == (4, 1)
    assert positions.shape == (4, 4 * block_size)
    for row in range(4):
        expected = set(reference[row][reference[row] >= 0].tolist())
        assert set(positions[row][positions[row] >= 0].tolist()) == expected
        # Valid columns are a prefix, in ascending block order.
        valid = positions[row] >= 0
        assert int(counts[row]) == int(valid.sum()) and bool(valid[: int(counts[row])].all())
        assert bool((positions[row][valid].diff() > 0).all())
        # Every sparse sub-block of a selected candidate block is listed, in order.
        expected_blocks = sorted(
            {(p // block_size) * ratio + j for p in expected for j in range(ratio)}
        )
        assert blocks[row][: len(expected_blocks)].tolist() == expected_blocks
        # Padding repeats the last candidate block (all of its sparse sub-blocks).
        last = expected_blocks[-1] // ratio if expected_blocks else 0
        assert set(blocks[row][len(expected_blocks) :].tolist()) <= set(
            range(last * ratio, (last + 1) * ratio)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_size,topk_blocks", [(8, 4), (16, 3), (8, 99)])
@torch.inference_mode()
def test_publish_candidates_sparse_blocks_cuda_matches_torch(block_size, topk_blocks, monkeypatch):
    """The one-launch publication writes the same positions, sparse ids and counts as PyTorch."""
    import tensorrt_llm._torch.attention.backends.sparse.csa2.indexer as implementation

    torch.manual_seed(4311)
    width = 12 * block_size + 5
    source = CSA2Indexer(_sparse_candidate_layout(topk_blocks, block_size), 0, 32, 128)
    lengths = torch.tensor([width, 3 * block_size + 1, 1, 0, width - 2], dtype=torch.int32)
    scores = torch.rand(5, width)
    scores = scores.masked_fill(torch.arange(width)[None, :] >= lengths[:, None], -torch.inf)
    scores[0, block_size : 3 * block_size] = -torch.inf  # unreachable middle blocks
    published_width = min(topk_blocks, -(-width // block_size)) * block_size
    with monkeypatch.context() as context:
        context.setattr(implementation, "dsl_available", lambda: False)
        expected = _publish_sparse(source, scores.cuda(), lengths.cuda(), published_width)
    actual = _publish_sparse(source, scores.cuda(), lengths.cuda(), published_width)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.skipif(not _sparse_kernels_available(), reason="DeepGEMM sparse MQA on SM100")
@torch.inference_mode()
def test_sparse_candidate_paged_decode_and_graph(monkeypatch):
    torch.manual_seed(4312)
    layout = _sparse_candidate_layout(4, 8)
    packed_keys = pack_rows(torch.randn(512, 128, device="cuda", dtype=torch.bfloat16), "index")
    manager = _StridedOwner(layout, packed_keys)
    manager.max_total_draft_tokens = 1
    q, _, weights = _projected(3)
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=3)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.is_cuda_graph = True
    metadata.csa2_kv_sources = {0: 0, 1: 0}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, 2), (2, 3)]
    metadata.csa2_request_start_positions = [125, 127]
    metadata.csa2_request_lengths = [2, 1]
    metadata.csa2_global_max_positions = {0: 128}
    metadata.csa2_global_page_sizes = {0: 128}
    table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    visible = torch.tensor([63, 63, 64], device="cuda", dtype=torch.int32)
    metadata.csa2_global_page_tables = {0: table}
    metadata.csa2_visible_lengths = {0: visible, 1: visible}
    metadata.csa2_token_requests = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int64)
    metadata.csa2_decode_row_requests = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(3, 512, device="cuda", dtype=torch.bfloat16),
        index_q=q,
        index_weights=weights,
    )
    source = CSA2Indexer(layout, 0, 32, 128)
    consumer = CSA2Indexer(layout, 1, 32, 128)
    sparse_calls = []
    original = consumer._sparse_candidate_logits

    def sparse(*args, **kwargs):
        sparse_calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(consumer, "_sparse_candidate_logits", sparse)
    monkeypatch.setattr(
        consumer, "_paged_mqa_logits", lambda *a, **kw: pytest.fail("Dense paged path used")
    )
    monkeypatch.setattr(
        consumer, "_call_mqa_logits", lambda *a, **kw: pytest.fail("Dense path used")
    )

    def forward():
        metadata._csa2_forward_serial = getattr(metadata, "_csa2_forward_serial", 0) + 1
        source(state, 0, 3)
        return consumer(state, 0, 3)

    for _ in range(2):
        output = forward()
    assert sparse_calls and all(shape == (3, 32, 64) for shape in sparse_calls)
    descriptors = metadata.prepare_sparse_indexer(1, consumer.sparse_block)
    assert descriptors is not None
    pages, block_table, context_lens, rows, schedule = descriptors
    assert schedule is not None and schedule.is_cuda
    # The schedule is shared by every consumer layer of a forward
    assert metadata.prepare_sparse_indexer(1, consumer.sparse_block)[4] is schedule
    assert pages.shape[1:] == (64, 1, 68)
    assert block_table.shape[0] == 3 and context_lens.shape == (3,)
    assert rows.tolist() == [0, 0, 1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    for lengths, pages_ in (
        ([40, 41, 100], [[0, 1], [2, 3]]),
        ([64, 65, 96], [[2, 3], [0, -1]]),
        ([63, 63, 64], [[1, 0], [3, 2]]),
    ):
        visible.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        table.copy_(torch.tensor(pages_, device="cuda", dtype=torch.int32))
        q.neg_()
        graph.replay()
        torch.cuda.synchronize()
        selected = output.clone()
        candidates = metadata.csa2_candidates[0].clone()
        positions = torch.arange(128, device="cuda").expand(3, -1)
        req = metadata.csa2_token_requests
        physical = table[req[:, None], positions // 128]
        valid = (physical >= 0) & (positions < visible[:, None])
        rows_ = read_index_rows(
            manager.get_index_pages(0),
            (physical.clamp_min(0) * 128 + positions % 128).long(),
        )
        decoded_k = unpack_rows(rows_, 128, "index").float()
        decoded_q = unpack_rows(pack_rows(q, "index"), 128, "index").float()
        scores = (
            torch.einsum("qhd,qkd->qhk", decoded_q, decoded_k).relu() * weights[..., None]
        ).sum(1)
        # The source selects its blocks over every visible position...
        expected = _reference_select_candidate_positions(
            scores.masked_fill(~valid, -torch.inf), visible, 4, 8
        )
        assert set(candidates[0][candidates[0] >= 0].tolist()) == set(
            expected[0][expected[0] >= 0].tolist()
        )
        # ...and the consumer's Top-K stays inside the published candidates.
        in_pool = torch.zeros_like(valid)
        in_pool.scatter_(1, candidates.clamp_min(0), candidates >= 0)
        _assert_selection(selected, scores, valid & in_pool)
    assert bool((manager.storage == 77).all())

    # A holey prefix has fewer candidates than visible KV blocks. Repeated
    # padding must not enter DeepGEMM's paired-query merge (CI124 regression).
    layout = _sparse_candidate_layout(1024, 8)
    manager.layout = layout
    source = CSA2Indexer(layout, 0, 32, 128)
    consumer = CSA2Indexer(layout, 1, 32, 128)
    metadata.is_cuda_graph = False
    metadata.csa2_global_max_positions[0] = 8192
    metadata.csa2_request_start_positions = [3246, 0]
    table = torch.full((2, 64), -1, dtype=torch.int32, device="cuda")
    table[:, 0] = torch.tensor([0, 2], device="cuda")
    table[:, 25] = torch.tensor([1, 3], device="cuda")
    metadata.csa2_global_page_tables[0] = table
    visible.copy_(torch.tensor([3247, 3248, 0], dtype=torch.int32, device="cuda"))

    def check_holey(actual: torch.Tensor, counts: list[int]) -> None:
        assert metadata.csa2_candidate_counts[0].flatten().tolist() == counts
        _, scores, _ = _candidate_decode_run(consumer, state, monkeypatch, old=True)
        _check_candidate_multiset(
            actual, scores, metadata.csa2_candidates[0], _bf16_tolerance(scores)
        )

    for num_contexts, decode_start in ((0, 0), (1, 2), (2, 3)):
        metadata.csa2_num_context_requests = num_contexts
        metadata.csa2_decode_row_requests = (
            metadata.csa2_token_requests[decode_start:].int() - num_contexts
        )
        actual = forward().clone()
        torch.cuda.synchronize()
        check_holey(actual, [175, 176, 0])
        if num_contexts < 2:
            counts = metadata.prepare_sparse_indexer(1, consumer.sparse_block)[2]
            assert counts.tolist() == [175, 176, 0][decode_start:]

    metadata.csa2_num_context_requests = 0
    metadata.csa2_decode_row_requests = metadata.csa2_token_requests.int()
    metadata.is_cuda_graph = True
    for _ in range(2):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for lengths, counts in (
        ([0, 9, 13], [0, 9, 13]),
        ([3247, 3248, 0], [175, 176, 0]),
    ):
        visible.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        check_holey(captured.clone(), counts)
