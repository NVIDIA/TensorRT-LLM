# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 DEP decode select with vLLM's ranker (minimax_m3_sm107_dep_vllm_topk, sm_107).

* the op against a Python model of vLLM's 16-bit key, on the NVFP4 reader's native score layout;
* ascending ids, -1 past short rows, the local block always taken, determinism and CUDA graph replay;
* the served call: MsaIndexer._select with rows at the KV manager's max_blocks_per_seq bound (and one past it)
  agrees with the exact select (minimax_m3_select_blocks).
"""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa import msa_indexer
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    select_blocks_from_maxscore,
)

CUDA = torch.cuda.is_available()
SM107 = CUDA and torch.cuda.get_device_capability() == (10, 7)

needs_op = pytest.mark.skipif(
    not (SM107 and msa_indexer._family_op("minimax_m3_sm107_dep_vllm_topk") is not None),
    reason="minimax_m3_sm107_dep_vllm_topk needs sm_107",
)


def _native_scores(heads, rows, blocks, *, seed, equal=False):
    """Scores in the reader's native layout [rows, heads, blocks], seen as [heads, blocks, rows] (strides
    (blocks, 1, heads * blocks)), as nvfp4_index_k.score_decode_rows hands them to the select."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    native = (torch.randn(rows, heads, 128, blocks, generator=g, device="cuda") * 5.7).amax(dim=2)
    if equal:
        native.fill_(0.25)
    return native.as_strided((heads, blocks, rows), (blocks, 1, heads * blocks))


def _reference_row(row: torch.Tensor, n: int) -> list:
    """vLLM's selection for one row of n candidates: block n - 1 forced, 15 more ranked by the 16-bit key
    (q = trunc(fma(v, scale, bias)) clamped to 65535, ties to the lower block), ascending; n <= 16 gives
    0 .. n - 1 then -1. The fp32 FMA is evaluated in float64 and rounded once to fp32."""
    if n <= MSA_REQUIRED_TOPK:
        return list(range(n)) + [-1] * (MSA_REQUIRED_TOPK - n)
    nvm = n - 1
    v = row[:nvm].float()
    mx, mn = v.max(), v.min()
    scale = (torch.tensor(65534.0) / (mx - mn)).float() if bool(mx > mn) else torch.tensor(0.0)
    bias = ((-mn).double() * scale.double() + 1.5).float()
    q = (v.double() * scale.double() + bias.double()).float()
    q = torch.clamp(torch.trunc(q.double()), 0, 65535).long()
    ids = torch.arange(nvm)
    top = torch.topk(q * 65536 + (65535 - ids), MSA_REQUIRED_TOPK - 1).indices
    return sorted(ids[top].tolist() + [nvm])


def _reference(scores: torch.Tensor, nvb: torch.Tensor) -> torch.Tensor:
    heads, blocks, rows = scores.shape
    s, n = scores.cpu(), nvb.cpu()
    table = [
        [_reference_row(s[h, :, t], min(int(n[t]), blocks)) for h in range(heads)]
        for t in range(rows)
    ]
    return torch.tensor(table, dtype=torch.int32)


def _lengths(rows, blocks, *, seed, short=False):
    g = torch.Generator().manual_seed(seed)
    lo = 1 if short else max(17, blocks // 2)
    n = torch.randint(lo, blocks + 1, (rows,), generator=g, dtype=torch.int32)
    if short:
        n[0], n[-1] = 0, 16  # a padded row and the longest row the select leaves unranked
    return n.cuda()


@needs_op
@torch.inference_mode()
@pytest.mark.parametrize(
    "heads,rows,blocks,short,equal",
    [
        (4, 96, 1300, False, False),  # a 96-row decode step at a 160K context
        (4, 128, 1600, False, False),  # a 128-row decode step
        (4, 48, 9217, False, False),  # the DEP build's bound (kMaxNvm 9216 + the row's own block)
        (4, 24, 600, True, False),  # rows of 0 .. 16 candidates among the rest
        (2, 16, 300, False, True),  # equal scores: the ids decide
        (4, 511, 200, False, False),  # 2044 rows, the op's row bound
    ],
)
def test_matches_vllm_key_reference(heads, rows, blocks, short, equal):
    scores = _native_scores(heads, rows, blocks, seed=rows + blocks, equal=equal)
    nvb = _lengths(rows, blocks, seed=blocks, short=short)
    table = msa_indexer._dep_vllm_topk_select(scores, nvb, 0, 1)
    assert table is not None
    assert table.shape == (rows, heads, MSA_REQUIRED_TOPK)
    assert table.permute(1, 0, 2).is_contiguous()
    assert torch.equal(table.cpu(), _reference(scores, nvb))


@needs_op
@torch.inference_mode()
def test_ascending_local_taken_deterministic_and_graph_replay():
    heads, rows, blocks = 4, 96, 1300
    scores = _native_scores(heads, rows, blocks, seed=3)
    nvb = _lengths(rows, blocks, seed=4)
    first = msa_indexer._dep_vllm_topk_select(scores, nvb, 0, 1).clone()
    ids = first.long()
    assert bool((ids[..., 1:] > ids[..., :-1]).all()), "ids not strictly ascending"
    assert torch.equal(first[..., -1].cpu(), (nvb.cpu() - 1).unsqueeze(1).expand(rows, heads).int())
    assert torch.equal(msa_indexer._dep_vllm_topk_select(scores, nvb, 0, 1), first)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = msa_indexer._dep_vllm_topk_select(scores, nvb, 0, 1)
    fresh = _native_scores(heads, rows, blocks, seed=5)
    scores.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.cpu(), _reference(fresh, nvb))


# DEP generation shapes, derived as the executor and KV cache manager V2 do for DSpark K7 (max_draft_len 7,
# tokens_per_gen_step 8, 6 extra-KV tokens): the engine max_seq_len is 1048576 + 7 (overlap scheduler) + 6 + 7, the
# manager adds a generation headroom of 6 + 8 tokens, and pads its block count to a multiple of 4 (copy_block_offsets):
# max_blocks_per_seq 8193 -> 8196. The proxy plan rounds that up to a multiple of 128 for the score width.
_TOKENS_PER_GEN_STEP, _EXTRA_KV = 8, 6
_KV_MAX_SEQ_LEN = 1048576 + (_TOKENS_PER_GEN_STEP - 1) + _EXTRA_KV + (_TOKENS_PER_GEN_STEP - 1)
_KV_BLOCKS = math.ceil((_KV_MAX_SEQ_LEN + _EXTRA_KV + _TOKENS_PER_GEN_STEP) / 128)
SERVED_MAX_ROW_BLOCKS = (_KV_BLOCKS + 3) // 4 * 4
SERVED_TILES = math.ceil(SERVED_MAX_ROW_BLOCKS / 128) * 128
DECODE_QUERY_LEN = 8
# Per-request context lengths of long-context DEP ranks (median 120K, max 838K).
SERVED_CTX = [
    157813, 75751, 110443, 63984, 85040, 61785, 76120, 117952, 94542, 3454, 103334, 54845, 65736, 48911, 53268,
    146674, 15267, 219615, 3143, 103742, 86454, 122038, 171645, 102212, 75296, 99830, 79838, 84046, 81731, 87200,
    79963, 72901, 3305, 3294, 82175, 64117, 3859, 27066, 152663, 71282, 223472, 112797, 84976, 54740, 91777, 86577,
    127210, 837900,
]  # fmt: skip


def _served_lengths(requests: int) -> torch.Tensor:
    """n_valid_blocks of each decode row (8 per request): ceil(position / 128). The last request sits at the KV
    manager's bound, so its last row has SERVED_MAX_ROW_BLOCKS candidates."""
    ctx = SERVED_CTX[: requests - 1] + [SERVED_MAX_ROW_BLOCKS * 128]
    n = [
        math.ceil((c - DECODE_QUERY_LEN + 1 + t) / 128)
        for c in ctx
        for t in range(DECODE_QUERY_LEN)
    ]
    return torch.tensor(n, dtype=torch.int32, device="cuda")


def _served_indexer(max_row_blocks):
    config = MiniMaxM3SparseConfig(
        num_q_heads=16, num_kv_heads=4, head_dim=128, num_index_heads=4, sparse_index_dim=128, block_size=128,
        topk=MSA_REQUIRED_TOPK,
    )  # fmt: skip
    indexer = msa_indexer.MsaIndexer(config)
    indexer.max_row_blocks = max_row_blocks
    return indexer


def _spy(monkeypatch):
    taken, declined = [], []
    real = msa_indexer._dep_vllm_topk_select

    def select(*args, **kwargs):
        out = real(*args, **kwargs)
        taken.append(out is not None)
        return out

    monkeypatch.setattr(msa_indexer, "_dep_vllm_topk_select", select)
    monkeypatch.setattr(
        msa_indexer, "_dep_vllm_topk_decline", lambda reason, detail="": declined.append(reason)
    )
    return taken, declined


def _distinct_scores(heads, rows, blocks, *, seed):
    """Native-layout scores whose every row is a permutation of 0 .. blocks - 1: distinct integers stay distinct
    under the 16-bit key (scale = 65534 / (blocks - 1) > 7 at 8320 blocks), so the port ranks like the exact select."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.rand(rows, heads, blocks, generator=g, device="cuda")
    native = keys.argsort(dim=-1).float()
    return native.as_strided((heads, blocks, rows), (blocks, 1, heads * blocks))


@needs_op
@torch.inference_mode()
# The served bound (8196) and one past it (8197).
@pytest.mark.parametrize("bound", [SERVED_MAX_ROW_BLOCKS, SERVED_MAX_ROW_BLOCKS + 1])
def test_rows_at_the_row_bound_are_taken_and_match_the_exact_select(monkeypatch, bound):
    heads, requests = 4, 2
    rows = requests * DECODE_QUERY_LEN
    scores = _distinct_scores(heads, rows, SERVED_TILES, seed=bound)
    nvb = _served_lengths(requests)
    nvb[-DECODE_QUERY_LEN:] = bound  # the last request's rows sit at the bound
    taken, declined = _spy(monkeypatch)
    table = _served_indexer(bound)._select(scores, nvb)
    assert taken == [True] and declined == []
    assert bool((table[-1, :, -1] == bound - 1).all())
    exact = select_blocks_from_maxscore(
        scores, topk=16, n_valid_blocks=nvb, init_blocks=0, local_blocks=1, head_major_output=True
    )
    assert torch.equal(table, exact)
