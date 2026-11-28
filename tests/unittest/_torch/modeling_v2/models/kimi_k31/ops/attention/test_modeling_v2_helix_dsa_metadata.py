# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix CP on the engine's DSA metadata: ``HelixDSAMetadataMixin`` (``dsa_metadata.py``) and the general target's
Helix pieces (``helix.py``: ``K31HelixDSAMetadata``, the decode Top-K).

CPU: the rank-local lengths under Helix, in the indexer K-cache table and in every length-taking preparation step;
the token-sparse view; the decode Top-K's Helix bounds and CP merge.
"""

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]
from _rubin_dsa_metadata_util import bare  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import helix
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.helix import K31HelixDSAMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_merge_helix_topk import (
    rubin_merge_helix_topk,
)

cpu_only = pytest.mark.cpu_only

#: A step of one context request and two generation requests of 3-token verify groups.
_CACHED = [5, 6, 9]
_SEQ_LENS_KV = [4, 3, 3]
#: The new generation tokens this rank owns, per generation request (verify groups).
_OWNED = [1, 2]
#: Whether this rank is not the active one, per generation request (no speculation).
_INACTIVE = [True, False]
#: The KV page size of the indexer K-cache table checks: the local lengths above differ in pages.
_PAGE = 2


def _step(cls, *, helix: bool, spec: bool, **attrs):
    """A bare ``cls`` metadata object holding the step above."""
    return bare(
        cls,
        num_contexts=1,
        num_generations=2,
        num_seqs=3,
        enable_helix=helix,
        _helix_spec_tokens_valid=spec,
        helix_owned_new_tokens_cpu=torch.tensor(_OWNED, dtype=torch.int32),
        helix_is_inactive_rank_cpu=torch.tensor(_INACTIVE),
        seq_lens_kv=torch.tensor(_SEQ_LENS_KV, dtype=torch.int32),
        kv_cache_params=SimpleNamespace(
            use_cache=True, num_cached_tokens_per_seq=list(_CACHED), num_extra_kv_tokens=0
        ),
        **attrs,
    )


def _helix_kv_lens(spec: bool) -> torch.Tensor:
    """The rank-local KV lengths of the step: the context appends all its tokens; a generation request appends its
    owned tokens (verify groups) or its one token on the active rank."""
    cached = torch.tensor(_CACHED)
    ctx = cached[:1] + torch.tensor(_SEQ_LENS_KV[:1])
    if spec:
        gen = cached[1:] + torch.tensor(_OWNED)
    else:
        gen = cached[1:] + torch.tensor(
            [0 if i else n for i, n in zip(_INACTIVE, _SEQ_LENS_KV[1:])]
        )
    return torch.cat([ctx, gen]).int()


@cpu_only
@pytest.mark.parametrize("spec, expected", [(False, [9, 6, 12]), (True, [9, 7, 11])])
def test_helix_local_kv_lens(spec, expected):
    """The context appends its 4 tokens; without speculation the inactive generation request appends nothing and
    the active one its 3 tokens; with speculation each appends the tokens it owns (1 and 2)."""
    md = _step(K31HelixDSAMetadata, helix=True, spec=spec)
    kv_lens = md._helix_local_kv_lens(torch.tensor(_CACHED, dtype=torch.int32))
    assert kv_lens.tolist() == expected


# --- the preparation steps under Helix ---


def _indexer_k_cache_step(cls, *, helix: bool, spec: bool = False):
    pool = 100 + torch.arange(3 * 8, dtype=torch.int32).view(3, 8)
    md = _step(
        cls,
        helix=helix,
        spec=spec,
        kv_cache_manager=SimpleNamespace(tokens_per_block=_PAGE),
        block_table=torch.full((4, 8), -9, dtype=torch.int32),
    )
    md._update_indexer_k_cache_block_offsets = lambda: pool.clone()
    return md


@cpu_only
@pytest.mark.parametrize("spec", [False, True])
def test_prepare_for_indexer_k_cache_under_helix_takes_the_local_lengths(spec):
    md = _indexer_k_cache_step(K31HelixDSAMetadata, helix=True, spec=spec)
    md.prepare_for_indexer_k_cache()
    pages = (_helix_kv_lens(spec) + _PAGE - 1) // _PAGE
    width = int(pages.max())
    expected = 100 + torch.arange(3 * 8, dtype=torch.int32).view(3, 8)[:, :width]
    expected = expected.masked_fill(torch.arange(width)[None] >= pages[:, None], -1)
    assert torch.equal(md.block_table[:3, :width], expected)
    assert bool((md.block_table[:3, width:] == -9).all())


_PREPARE_STEPS = (
    "_validate_ragged_verification",
    "_invalidate_pool_view_cache",
    "_prepare_device_window_kv_lens",
    "prepare_for_mla_rope_append",
    "prepare_for_skip_indexer",
    "prepare_for_indices_conversion",
    "prepare_for_indexer_k_cache",
    "prepare_for_spec_decode",
    "_prepare_helix_sparse_rows",
)


def _prepare_step(cls, monkeypatch, calls, *, helix: bool, spec: bool = False):
    md = _step(cls, helix=helix, spec=spec, sparse_offload_state=None)

    def recorder(name):
        def record(*args):
            calls.append((name, tuple(a.clone() if torch.is_tensor(a) else a for a in args)))
            return args[0] if name == "_prepare_device_window_kv_lens" else None

        return record

    for name in _PREPARE_STEPS:
        setattr(md, name, recorder(name))
    return md


@pytest.fixture
def prepare_stubs(monkeypatch):
    calls = []
    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", lambda self: calls.append(("base", ())))
    monkeypatch.setattr(
        Indexer, "prepare", staticmethod(lambda metadata: calls.append(("indexer", ())))
    )
    return calls


@cpu_only
@pytest.mark.parametrize("spec", [False, True])
def test_prepare_under_helix_takes_the_local_lengths_and_lays_out_the_sparse_rows(
    prepare_stubs, monkeypatch, spec
):
    md = _prepare_step(K31HelixDSAMetadata, monkeypatch, prepare_stubs, helix=True, spec=spec)
    md.prepare()
    names = [name for name, _ in prepare_stubs]
    assert names[-1] == "_prepare_helix_sparse_rows"
    expected = _helix_kv_lens(spec)
    for name, args in prepare_stubs:
        if name in (
            "prepare_for_skip_indexer",
            "prepare_for_spec_decode",
            "_prepare_helix_sparse_rows",
        ):
            assert torch.equal(args[0].int(), expected), name


# --- the token-sparse view ---


def _view_step(num_rows: int, *, ragged: bool = False):
    rows = 8
    return bare(
        K31HelixDSAMetadata,
        is_ragged_verify=ragged,
        _attn_num_rows=0,
        _helix_num_rows=num_rows,
        helix_row_kv_lens_cuda=torch.arange(rows, dtype=torch.int32),
        helix_row_kv_lens_host=torch.arange(rows, dtype=torch.int32),
        helix_row_prompt_lens_cpu=torch.ones(rows, dtype=torch.int32),
        helix_row_prompt_lens_cuda=torch.ones(rows, dtype=torch.int32),
        helix_row_request_types_host=torch.ones(rows, dtype=torch.int32),
        helix_row_block_offsets=torch.zeros(1, rows, 2, 4, dtype=torch.int32),
    )


@cpu_only
def test_token_major_gen_view_presents_the_helix_rows_only_to_token_sparse_calls():
    md = _view_step(5)
    assert md.token_major_gen_view() is None
    view = md.token_major_gen_view(token_sparse=True)
    assert view.num_rows == 5 and view.max_num_rows == 8
    assert torch.equal(view.sequence_length, torch.arange(5, dtype=torch.int32))
    with md.token_sparse_call():
        assert md.token_major_gen_view().num_rows == 5
    assert md.token_major_gen_view() is None
    # No Helix rows this step, or a ragged step: the engine's view.
    assert _view_step(0).token_major_gen_view(token_sparse=True) is None
    assert _view_step(5, ragged=True).token_major_gen_view(token_sparse=True) is None


# --- the decode Top-K ---


def _merge_reference(scores, local_rows, other_scores, cp_rank):
    """The merge of this rank's candidates with another rank's, through the merge op's reference."""
    valid = local_rows >= 0
    mine = torch.gather(scores.float(), 1, local_rows.clamp(min=0).long()).masked_fill(
        ~valid, float("-inf")
    )
    ranks = [mine, other_scores] if cp_rank == 0 else [other_scores, mine]
    return rubin_merge_helix_topk.reference(torch.stack(ranks), local_rows, cp_rank)


@cpu_only
@pytest.mark.parametrize("bounded", [False, True])
def test_decode_top_k_merges_over_the_cp_ranks(monkeypatch, bounded):
    torch.manual_seed(0)
    rows, width, k = 4, 16, 3
    scores = torch.randn(rows, width)
    local = torch.stack([torch.randperm(width)[:k] for _ in range(rows)]).int()
    other = torch.randn(rows, k)
    bounds = torch.tensor([5, 0, 7, 2], dtype=torch.int32)
    selected = {}

    def select(scores_, out, *, is_prefill, row_kv_lens=None, next_n=1, **kwargs):
        selected.update(row_kv_lens=row_kv_lens, next_n=next_n, is_prefill=is_prefill)
        out.copy_(local)
        return out

    gathered = {}

    def allgather(t, mapping, dim):
        gathered["shape"] = tuple(t.shape)
        return torch.cat([t, other.unsqueeze(0)], dim=dim)

    monkeypatch.setattr(helix, "cp_allgather", allgather)
    kept = {}
    md = SimpleNamespace(
        enable_helix=True,
        mapping=SimpleNamespace(cp_size=2, cp_rank=0),
        helix_spec_row_kv_lens=lambda n: bounds[:n] if bounded else None,
        set_helix_sparse_row_lens=lambda r: kept.update(rows=r.clone()),
    )
    out = torch.full((rows, k), -9, dtype=torch.int32)
    helix._helix_decode_top_k(
        select, {"metadata": md}, scores, out, is_prefill=False, row_kv_lens=None, next_n=3
    )
    assert selected["is_prefill"] is False
    if bounded:
        assert selected["row_kv_lens"] is not None and selected["next_n"] == 1
    else:
        assert selected["row_kv_lens"] is None and selected["next_n"] == 3
    expected_local = local.clone()
    if bounded:
        expected_local[bounds == 0] = -1
    assert torch.equal(out, _merge_reference(scores, expected_local, other, 0))
    assert gathered["shape"] == (1, rows, k)
    assert torch.equal(kept["rows"], out)
