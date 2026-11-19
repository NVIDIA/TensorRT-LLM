# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family DSA metadata overrides only what the K3.1 Rubin targets run differently.

CPU checks of ``rubin_dsa_metadata`` against the engine's indexer preparation helpers: the
long-prompt step's indexer plan (TP query split, none under attention DP, query tiles under the
per-call budget, one MQA-logits schedule per tile); the verify window's single DSL atom on graph
and eager metadata, and the refusal of a wider window; the backend classes the targets pick and
their refusals.
"""

import types

import pytest
import torch

__extra_import_path__ = ["."]
from _rubin_dsa_metadata_util import bare  # noqa: E402 — needs the path declared above

from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer as EngineIndexer
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import (
    rubin_dsa_backend,
    rubin_dsa_metadata,
)

pytestmark = pytest.mark.cpu_only

Metadata = rubin_dsa_metadata.RubinDSAMetadata
MQA_BUDGET = rubin_dsa_metadata._INDEXER_MQA_LOGITS_ELEM_BUDGET


class _Called(Exception):
    pass


def _raise(*args, **kwargs):
    raise _Called


# prefill plan


def _mapping(tp_size=8, tp_rank=0, attention_dp=False):
    return types.SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, enable_attention_dp=attention_dp)


def _ctx_step(
    q_lens,
    cached,
    mapping,
    *,
    cached_kv=True,
    chunked_prefill=True,
    chunk_size=16384,
):
    """A bare metadata of a step with context requests ``q_lens`` behind ``cached`` tokens.

    ``cached_kv`` and ``chunked_prefill``: the server's context MLA with cached KV and its
    chunked prefill (both on: the MLA chunked-prefill plan)."""
    nc = len(q_lens)
    cached_indptr = torch.tensor([0, *torch.tensor(cached).cumsum(0).tolist()], dtype=torch.int64)
    kv_indptr = torch.tensor(
        [0, *torch.tensor([c + q for c, q in zip(cached, q_lens)]).cumsum(0).tolist()],
        dtype=torch.int64,
    )
    md = bare(
        Metadata,
        sparse_metadata_params=types.SimpleNamespace(q_split_threshold=8192),
        enable_context_mla_with_cached_kv=cached_kv,
        runtime_features=types.SimpleNamespace(chunked_prefill=chunked_prefill),
        compress_ratios=[1],
        host_ctx_cached_token_indptr=cached_indptr,
        host_ctx_kv_indptr=kv_indptr,
        mapping=mapping,
        max_num_tokens=16384,
        kv_cache_manager=types.SimpleNamespace(max_seq_len=1 << 20),
        indexer_prefill_q_split=None,
        indexer_max_chunk_size=chunk_size,
        skip_indexer_for_ctx_reqs=False,
        cu_seqlen_ks=torch.zeros(1, dtype=torch.int32),
        num_contexts=nc,
    )
    params = types.SimpleNamespace(
        num_contexts=nc,
        seq_lens=torch.tensor(q_lens, dtype=torch.int32),
        kv_lens=torch.tensor([c + q for c, q in zip(cached, q_lens)], dtype=torch.int32),
    )
    return md, params


@pytest.fixture
def plan_stubs(monkeypatch):
    calls = []

    def meta(ks, ke, num_kv_tokens, num_heads):
        calls.append((ks.clone(), ke.clone(), num_kv_tokens, num_heads))
        return ("meta", len(calls) - 1)

    monkeypatch.setattr(rubin_dsa_metadata, "rubin_mqa_logits_metadata", meta)
    monkeypatch.setattr(EngineIndexer, "recompute_context_kv_gather_mappings", lambda md, p: None)
    monkeypatch.setattr(EngineIndexer, "prepare_for_chunked_prefill", _raise)
    return calls


@pytest.mark.parametrize(
    "tokens, mapping, expected",
    [
        (8191, _mapping(tp_rank=3), None),
        (8192, _mapping(tp_rank=3), [1024] * 8),
        (16384, _mapping(tp_size=16, tp_rank=5, attention_dp=True), None),
    ],
    ids=["tep-8191", "tep-8192", "dep16-16384"],
)
def test_tep_query_split_equals_the_reference_split(plan_stubs, tokens, mapping, expected):
    md, params = _ctx_step([tokens], [0], mapping)
    rubin_dsa_metadata._plan_prefill(md, params)
    prev = engine_indexer._split_prefill_queries(tokens, mapping, 8192)
    assert prev[2] == expected
    if mapping.enable_attention_dp:
        # DEP16 attention DP never splits the queries.
        assert prev == (0, tokens, None)
        assert md.indexer_prefill_q_split is None
        assert all(c.q_sizes is None for c in md.indexer_prefill_chunks)
        assert md.indexer_prefill_chunks[-1].token_end == tokens
    elif expected is None:
        assert md.indexer_prefill_q_split is None
        rows = [
            (c.token_start + c.q_start, c.token_start + c.q_end) for c in md.indexer_prefill_chunks
        ]
        assert rows[0][0] == 0 and rows[-1][1] == tokens
    else:
        assert md.indexer_prefill_q_split == prev
        q_begin, q_end, _ = prev
        for c in md.indexer_prefill_chunks:
            assert q_begin <= c.token_start and c.token_end <= q_end
            assert c.q_sizes is None  # the batch plan already split the rows


def test_tiles_stay_within_the_per_call_budget_and_get_one_schedule_each(plan_stubs):
    # A DEP16 16k chunk behind a 283k prefix: rows x keys exceed the per-call budget.
    cached = 282624
    md, params = _ctx_step([16384], [cached], _mapping(tp_size=16, attention_dp=True))
    rubin_dsa_metadata._plan_prefill(md, params)
    (chunk,) = md.indexer_prefill_chunks
    num_k = chunk.k_token_end - chunk.k_token_start
    assert num_k == cached + 16384
    # The reference tiling (FP8 keys): q_tile = budget // keys.
    q_tile = MQA_BUDGET // num_k
    assert chunk.tiles == tuple((a, min(a + q_tile, 16384)) for a in range(0, 16384, q_tile))
    assert all((c1 - c0) * num_k <= MQA_BUDGET for c0, c1 in chunk.tiles)
    assert len(plan_stubs) == len(chunk.tiles) == len(chunk.mqa_meta)
    for (c0, c1), (ks, ke, n_k, heads) in zip(chunk.tiles, plan_stubs):
        assert torch.equal(ks, chunk.cu_seqlen_ks[c0:c1]) and torch.equal(
            ke, chunk.cu_seqlen_ke[c0:c1]
        )
        assert (n_k, heads) == (num_k, 32)


# one DSL atom


def _spec_md(cls, is_cuda_graph, next_n, index_k=None, num_generations=2, max_blocks=4):
    """A verify step's metadata; with ``index_k`` the default target's, over a cache of those
    index keys, after its index-key post-init."""
    rows = num_generations * next_n
    manager = object()
    if index_k is not None:
        manager = types.SimpleNamespace(
            index_head_dim=128, indexer_k_dtype=index_k, use_fp4=index_k == "fp4"
        )
    md = bare(
        cls,
        is_cuda_graph=is_cuda_graph,
        use_cute_dsl_paged_mqa_logits=True,
        is_spec_decoding_enabled=False,
        ragged_verify_lens=None,
        enable_helix=False,
        _helix_spec_tokens_valid=False,
        max_draft_tokens=next_n - 1,
        runtime_tokens_per_gen_step=0,
        kv_cache_manager=manager,
        num_contexts=0,
        num_generations=num_generations,
        num_seqs=num_generations,
        num_sms=212,
        _indexer_compress_ratio=1,
        dsl_atom_ctx_offsets=None,
        kv_lens_expanded_host=torch.full((rows,), -7, dtype=torch.int32),
        kv_lens_expanded_cuda=torch.full((rows,), -7, dtype=torch.int32),
        host_block_table_expanded=torch.full((rows, max_blocks), -7, dtype=torch.int32),
        block_table_expanded=torch.full((rows, max_blocks), -7, dtype=torch.int32),
    )
    if index_k is not None:
        md._fp4_post_init()
    return md


# A k7 verify window (7 drafts) of two requests: the shared metadata on graph and eager steps.
@pytest.mark.parametrize("is_cuda_graph", [True, False], ids=["graph", "eager"])
def test_a_verify_window_is_one_atom_whatever_the_lengths(monkeypatch, is_cuda_graph):
    monkeypatch.setattr(engine_indexer, "_pick_dsl_split", _raise)
    next_n = 8
    index_k = None
    cls = Metadata
    md = _spec_md(cls, is_cuda_graph, next_n, index_k)
    for lens in ([131072, 131072], [3000, 3000], [2049, 60000]):
        cls.prepare_for_spec_decode(md, torch.tensor(lens))
        assert (md.expand_for_dsl, md.dsl_expand_factor, md.dsl_atom, md.dsl_kernel_atoms) == (
            False,
            1,
            next_n,
            1,
        )
        assert not md.use_expanded_buffers_for_mtp
    # No atom-offset tensor is built and no split table is written.
    assert md.dsl_atom_ctx_offsets is None
    assert (md.kv_lens_expanded_host == -7).all() and (md.block_table_expanded == -7).all()


def test_a_window_wider_than_one_atom_is_refused():
    cls = Metadata
    md = _spec_md(cls, False, 9, None)
    with pytest.raises(ValueError, match="one DSL logits atom"):
        cls.prepare_for_spec_decode(md, torch.tensor([5000, 5000]))


# backend classes


def test_dep16_and_tep8_pick_the_plain_backend():
    cls = rubin_dsa_backend.rubin_dsa_backend_cls(
        kv_lens_refresh_once=False, helix_fused_meta=False
    )
    assert cls is rubin_dsa_backend.RubinDSABackend
    # The engine refreshes the KV lengths after every correction for this metadata.
    assert not getattr(cls.Metadata, "kv_lens_refresh_once", False)
    helix = rubin_dsa_backend.rubin_dsa_backend_cls(
        kv_lens_refresh_once=True, helix_fused_meta=True
    )
    assert helix is rubin_dsa_backend.RubinDSAHelixBackend
    assert helix.Metadata is rubin_dsa_metadata.RubinDSAHelixMetadata
    for args in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="no Kimi K3.1 Rubin DSA backend"):
            rubin_dsa_backend.rubin_dsa_backend_cls(
                kv_lens_refresh_once=args[0], helix_fused_meta=args[1]
            )
