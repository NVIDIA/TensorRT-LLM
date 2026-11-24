# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family DSA metadata overrides only what the K3.1 Rubin targets run differently.

CPU checks of ``rubin_dsa_metadata`` against the engine class it inherits and against the
engine's indexer preparation helpers: the override list; the step's indexer plan (TP query split,
query tiles under the per-call budget, one MQA-logits schedule per tile, the dense Top-K for short
prompts); the verify window's single DSL atom on graph and eager metadata; the decode metadata
through the family entries; the draft step that skips the device rebuild.
"""

import ast
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path

import pytest
import torch
from _rubin_dsa_metadata_util import bare

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
    rubin_dsa_backend,
    rubin_dsa_metadata,
    rubin_gvr_topk_decode,
    rubin_mqa_logits_metadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer as EngineIndexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata as EngineMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only

Metadata = rubin_dsa_metadata.RubinDSAMetadata
MQA_BUDGET = rubin_dsa_metadata._INDEXER_MQA_LOGITS_ELEM_BUDGET
PACKED_BUDGET = rubin_dsa_metadata._INDEXER_PACKED_LOGITS_ELEM_BUDGET
TARGETS = ("kimi_k31_pro_nvfp4__sm_107__tep8", "kimi_k31_pro_nvfp4__sm_107__dep16")


class _Called(Exception):
    pass


def _raise(*args, **kwargs):
    raise _Called


# override list


def test_the_metadata_defines_exactly_its_override_list():
    defined = {
        name
        for name, value in vars(Metadata).items()
        if inspect.isfunction(value) or isinstance(value, (property, staticmethod, classmethod))
    }
    assert defined == set(Metadata.OVERRIDES)
    assert issubclass(Metadata, EngineMetadata)


def test_every_other_method_is_the_engine_class():
    for name in dir(EngineMetadata):
        engine_value = inspect.getattr_static(EngineMetadata, name)
        if not (inspect.isfunction(engine_value) or isinstance(engine_value, property)):
            continue
        if name in Metadata.OVERRIDES:
            continue
        assert inspect.getattr_static(Metadata, name) is engine_value, name


def test_overrides_reach_trtllm_metadata_not_the_engine_dsa_body(monkeypatch):
    # update_for_spec_dec and on_update_kv_lens call TrtllmAttentionMetadata directly.
    monkeypatch.setattr(EngineMetadata, "on_update_kv_lens", _raise)
    monkeypatch.setattr(EngineMetadata, "update_for_spec_dec", _raise)
    seen = []
    monkeypatch.setattr(TrtllmAttentionMetadata, "update_for_spec_dec", lambda self: seen.append(1))
    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: seen.append(2))
    md = bare(Metadata, _pool_cache_valid=True)
    md._invalidate_pool_view_cache = lambda: seen.append(3)
    Metadata.update_for_spec_dec(md, refresh_device_metadata=False)
    assert seen == [1, 2, 3]


# runtime helpers


def _module_tree(module) -> ast.Module:
    return ast.parse(inspect.getsource(module))


@pytest.mark.parametrize(
    "module",
    [rubin_dsa_metadata, rubin_dsa_backend],
    ids=["metadata", "backend"],
)
def test_the_runtime_helpers_launch_nothing_but_entries(module):
    tree = _module_tree(module)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "ops":
            if isinstance(node.value, ast.Name) and node.value.id == "torch":
                found.append(f"torch.ops at {node.lineno}")
        if isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            if "deep_gemm" in name or name.lstrip(".").startswith("kernels"):
                found.append(f"import of {name} at {node.lineno}")
            if node.level == 0 and ".kimi_k31.kernels" in name:
                found.append(f"import of {name} at {node.lineno}")
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if "deep_gemm" in a.name]
    assert not found


@pytest.mark.parametrize(
    "module",
    [rubin_dsa_metadata, rubin_dsa_backend, rubin_mqa_logits_metadata],
    ids=["metadata", "backend", "mqa_logits_metadata"],
)
def test_the_runtime_helpers_read_no_environment(module):
    source = inspect.getsource(module)
    tree = ast.parse(source)
    reads = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"))
        or (isinstance(node, ast.Name) and node.id in ("environ", "getenv"))
    ]
    imports_os = any(
        isinstance(node, ast.Import) and any(a.name == "os" for a in node.names)
        for node in ast.walk(tree)
    )
    assert not reads and not imports_os


# budgets


def test_the_packed_budget_never_exceeds_one_requests_largest_logits_tile():
    assert rubin_dsa_metadata.packed_logits_elem_budget(16384, 1 << 20) == PACKED_BUDGET
    assert rubin_dsa_metadata.packed_logits_elem_budget(8, 1024) == 8 * 1024


@pytest.mark.parametrize("target", TARGETS)
def test_the_targets_log_the_per_call_budget_the_metadata_tiles_with(target):
    # The target's startup line names the per-call budget; it must be the one the plan uses.
    family = Path(inspect.getfile(rubin_dsa_metadata)).parents[2]
    text = (family / target / "modeling.py").read_text()
    values = {int(v) for v in re.findall(r'"indexer_logits_elem_budget": (\d+)', text)}
    assert values == {MQA_BUDGET}


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
    skip_ctx=False,
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
        skip_indexer_for_ctx_reqs=skip_ctx,
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


@pytest.mark.parametrize("tokens", [8191, 8192, 16384])
@pytest.mark.parametrize("tp_rank", [0, 3, 7])
def test_tep_query_split_equals_the_reference_split(plan_stubs, tokens, tp_rank):
    mapping = _mapping(tp_rank=tp_rank)
    md, params = _ctx_step([tokens], [0], mapping)
    rubin_dsa_metadata._plan_prefill(md, params)
    prev = engine_indexer._split_prefill_queries(tokens, mapping, 8192)
    expected = {8191: None, 8192: [1024] * 8, 16384: [2048] * 8}[tokens]
    assert prev[2] == expected
    if expected is None:
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


@pytest.mark.parametrize("tokens", [8192, 16384])
def test_dep_attention_dp_never_splits(plan_stubs, tokens):
    mapping = _mapping(tp_size=16, tp_rank=5, attention_dp=True)
    md, params = _ctx_step([tokens], [0], mapping)
    rubin_dsa_metadata._plan_prefill(md, params)
    assert engine_indexer._split_prefill_queries(tokens, mapping, 8192) == (0, tokens, None)
    assert md.indexer_prefill_q_split is None
    assert all(c.q_sizes is None for c in md.indexer_prefill_chunks)
    assert md.indexer_prefill_chunks[-1].token_end == tokens


@pytest.mark.parametrize("tokens", [8191, 8192, 16384])
def test_tep_chunk_split_without_mla_chunked_prefill_equals_the_reference_split(plan_stubs, tokens):
    mapping = _mapping(tp_rank=5)
    md, params = _ctx_step(
        [tokens, tokens], [0, 0], mapping, chunked_prefill=False, chunk_size=8192
    )
    rubin_dsa_metadata._plan_prefill(md, params)
    assert len(md.indexer_prefill_chunks) >= 2
    for c in md.indexer_prefill_chunks:
        rows = c.token_end - c.token_start
        assert (c.q_start, c.q_end, c.q_sizes) == engine_indexer._split_prefill_queries(
            rows, mapping, 8192
        )


@pytest.mark.parametrize("cached", [86016, 282624], ids=["86k", "283k"])
def test_tiles_stay_within_the_per_call_budget_and_get_one_schedule_each(plan_stubs, cached):
    # A DEP16 16k chunk behind a long prefix: rows x keys exceed the per-call budget.
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


def test_tep_tiles_cover_this_ranks_rows_only(plan_stubs):
    md, params = _ctx_step([16384], [131072], _mapping(tp_rank=2), chunked_prefill=False)
    rubin_dsa_metadata._plan_prefill(md, params)
    for c in md.indexer_prefill_chunks:
        covered = [r for c0, c1 in c.tiles for r in range(c0, c1)]
        assert covered == list(range(c.q_start, c.q_end))


@pytest.mark.parametrize(
    "q_lens, skip",
    [([2048], True), ([1000, 900], True), ([100, 3000], False), ([2049], False)],
    ids=["one_2048", "two_short", "short_long_mix", "one_2049"],
)
def test_short_prompts_take_the_dense_top_k_and_build_no_schedules(plan_stubs, q_lens, skip):
    # The engine's skip rule over the step's context requests, then the plan.
    md = bare(
        Metadata,
        num_contexts=len(q_lens),
        num_generations=0,
        num_seqs=len(q_lens),
        enable_indexer_skip=True,
        enable_helix=False,
        num_sparse_topk=2048,
        kv_cache_params=types.SimpleNamespace(num_extra_kv_tokens=0),
    )
    md.prepare_dense_topk_indices = lambda kv_lens, device=False: None
    Metadata.prepare_for_skip_indexer(md, torch.tensor(q_lens, dtype=torch.int32))
    assert md.skip_indexer_for_ctx_reqs is skip
    plan, params = _ctx_step(q_lens, [0] * len(q_lens), _mapping(), skip_ctx=skip)
    rubin_dsa_metadata._plan_prefill(plan, params)
    assert (not plan_stubs) is skip
    assert all((not c.mqa_meta) is skip for c in plan.indexer_prefill_chunks)


def test_the_dense_top_k_rows_are_causal():
    md = bare(Metadata, num_sparse_topk=8)
    rows = Metadata._get_dense_topk_indices(md, torch.tensor([3, 2]), torch.tensor([3, 5]), 5)
    assert rows.tolist() == [
        [0, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, -1, -1, -1, -1, -1, -1],
        [0, 1, 2, -1, -1, -1, -1, -1],
        [0, 1, 2, 3, -1, -1, -1, -1],
        [0, 1, 2, 3, 4, -1, -1, -1],
    ]


# one DSL atom


def _spec_md(is_cuda_graph, next_n, num_generations=2, max_blocks=4):
    rows = num_generations * next_n
    return bare(
        Metadata,
        is_cuda_graph=is_cuda_graph,
        use_cute_dsl_paged_mqa_logits=True,
        is_spec_decoding_enabled=False,
        ragged_verify_lens=None,
        enable_helix=False,
        _helix_spec_tokens_valid=False,
        max_draft_tokens=next_n - 1,
        runtime_tokens_per_gen_step=0,
        kv_cache_manager=object(),
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


# b12-k7: a DEP16 generation step of 12 requests verifying 7 drafts (96 rows).
@pytest.mark.parametrize(
    "num_generations, next_n", [(2, 6), (2, 8), (12, 8)], ids=["k5", "k7", "b12-k7"]
)
@pytest.mark.parametrize("is_cuda_graph", [True, False], ids=["graph", "eager"])
def test_a_verify_window_is_one_atom_whatever_the_lengths(
    monkeypatch, num_generations, next_n, is_cuda_graph
):
    monkeypatch.setattr(engine_indexer, "_pick_dsl_split", _raise)
    md = _spec_md(is_cuda_graph, next_n, num_generations)
    for lens in ([131072, 131072], [3000, 3000], [2049, 60000]):
        lens = torch.tensor(lens).repeat_interleave(num_generations // 2)
        Metadata.prepare_for_spec_decode(md, lens)
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
    with pytest.raises(ValueError, match="one DSL logits atom"):
        Metadata.prepare_for_spec_decode(_spec_md(False, 9), torch.tensor([5000, 5000]))


def test_helix_verify_bounds_take_per_token_rows():
    md = _spec_md(True, 6)
    md.enable_helix = True
    md._helix_spec_tokens_valid = True
    md.host_indexer_k_cache_block_offsets = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    Metadata.prepare_for_spec_decode(md, torch.tensor([300, 700], dtype=torch.int32))
    assert md.use_expanded_buffers_for_mtp
    assert md.kv_lens_expanded_host.tolist() == [300] * 6 + [700] * 6
    assert md.block_table_expanded[:, 0].tolist() == [10] * 6 + [20] * 6


# decode metadata


def _gen_md(num_contexts, seq_lens, kv_lens, *, max_draft=0, helix=False):
    num_seqs = len(seq_lens)
    num_tokens = sum(seq_lens)
    md = bare(
        Metadata,
        num_contexts=num_contexts,
        num_generations=num_seqs - num_contexts,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        num_ctx_tokens=sum(seq_lens[:num_contexts]),
        ragged_verify_lens=None,
        kv_cache_manager=types.SimpleNamespace(index_head_dim=128),
        use_fp8_ds_mla=False,
        enable_helix=helix,
        enable_ragged_verification=False,
        use_expanded_buffers_for_mtp=False,
        max_draft_tokens=max_draft,
        runtime_tokens_per_gen_step=0,
        num_sms=212,
        _tokens_per_block=64,
        indexer_quant_block_size=128,
        _indexer_compress_ratio=1,
        seq_lens_cuda=torch.tensor(seq_lens, dtype=torch.int32),
        kv_lens_cuda=torch.tensor(kv_lens, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.arange(num_seqs * 8, dtype=torch.int32).view(
            num_seqs, 8
        ),
        req_idx_per_token=torch.full((num_tokens,), -1, dtype=torch.int32),
        slot_mapping_fp8=torch.zeros(num_tokens, dtype=torch.int64),
        slot_mapping_scale=torch.zeros(num_tokens, dtype=torch.int64),
        gen_kv_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        gen_cached_token_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        kv_lens_cuda_2d=torch.zeros(num_seqs, 1 + max_draft, dtype=torch.int32),
        scheduler_metadata_buffer=torch.zeros(4, 2, dtype=torch.int32),
        scheduler_metadata_buffer_full_next_n=torch.zeros(4, 2, dtype=torch.int32),
    )
    md._invalidate_pool_view_cache = lambda: None
    md._compute_kv_lens_row_reorder = lambda: None
    md.prepare_dense_topk_indices = lambda kv_lens, device=False: None
    return md


@pytest.fixture
def decode_stubs(monkeypatch):
    calls = {"fused": [], "schedule": []}
    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    monkeypatch.setattr(
        rubin_dsa_metadata,
        "fused_dsa_decode_metadata",
        lambda *args, **kwargs: calls["fused"].append(kwargs),
    )
    monkeypatch.setattr(
        rubin_dsa_metadata,
        "get_paged_mqa_logits_metadata",
        lambda lens, block_kv, num_sms, out: calls["schedule"].append((lens.clone(), out)),
    )
    return calls


# b12: a DEP16 generation step of 12 requests verifying 7 drafts (96 rows).
@pytest.mark.parametrize(
    "kv_lens", [[4000, 9000], [2000 + 9000 * i for i in range(12)]], ids=["b2", "b12"]
)
def test_a_pure_decode_step_takes_the_fused_entry_and_one_schedule(
    monkeypatch, decode_stubs, kv_lens
):
    monkeypatch.setattr(rubin_dsa_metadata, "build_req_idx_per_token", _raise)
    md = _gen_md(0, [8] * len(kv_lens), kv_lens, max_draft=7)
    Metadata.on_update_kv_lens(md)
    assert len(decode_stubs["fused"]) == 1
    assert decode_stubs["fused"][0]["max_query_len"] == 8
    # The DSL logits read the request-major schedule only (no full-window schedule).
    ((lens, out),) = decode_stubs["schedule"]
    assert lens.view(-1).tolist() == kv_lens and out is md.scheduler_metadata_buffer
    assert md.kv_lens_cuda_2d.tolist() == [[n] * 8 for n in kv_lens]


@pytest.mark.parametrize(
    "seq_lens, kv_lens, max_draft, slot, kv_indptr, cached_indptr",
    [
        # Token 5 is request 1's position 69: block 9 (page 1 of its table), row 5.
        ([5, 1, 1], [5, 70, 130], 0, (5, 9, 5), [70, 200], [69, 198]),
        # Attention DP: a context rank's dummy k7 verify request at positions 100..107; token 4
        # is its position 100: block 9, row 36.
        ([4, 8], [4, 108], 7, (4, 9, 36), [108], [100]),
    ],
    ids=["decode", "dummy-verify"],
)
def test_a_mixed_step_takes_the_eager_chain(
    decode_stubs, seq_lens, kv_lens, max_draft, slot, kv_indptr, cached_indptr
):
    md = _gen_md(1, seq_lens, kv_lens, max_draft=max_draft)
    Metadata.on_update_kv_lens(md)
    assert not decode_stubs["fused"]
    assert md.req_idx_per_token.tolist() == [i for i, n in enumerate(seq_lens) for _ in range(n)]
    token, block, row = slot
    assert md.slot_mapping_fp8[token].item() == block * 64 * (128 + 4) + row * 128
    assert md.gen_kv_indptr[1 : 1 + len(kv_indptr)].tolist() == kv_indptr
    assert md.gen_cached_token_indptr[1 : 1 + len(cached_indptr)].tolist() == cached_indptr


@pytest.mark.parametrize("refresh", [True, False])
def test_the_draft_step_rebuilds_the_device_metadata_only_when_asked(monkeypatch, refresh):
    seen = []
    monkeypatch.setattr(TrtllmAttentionMetadata, "update_for_spec_dec", lambda self: None)
    monkeypatch.setattr(
        TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: seen.append("base")
    )
    md = bare(Metadata)
    md.on_update_kv_lens = lambda: seen.append("rebuild")
    md._invalidate_pool_view_cache = lambda: seen.append("invalidate")
    Metadata.update_for_spec_dec(md, refresh_device_metadata=refresh)
    assert seen == (["rebuild"] if refresh else ["base", "invalidate"])
    assert (md.max_ctx_kv_len, md.num_ctx_cached_tokens, md.max_gen_seq_len) == (0, 0, 1)


def test_prepare_runs_the_family_plan_not_the_engine_indexer(monkeypatch):
    for name in ("prepare", "prepare_for_chunked_prefill", "prepare_scheduler_metadata"):
        monkeypatch.setattr(EngineIndexer, name, _raise)
    order = []
    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", lambda self: order.append("base"))
    monkeypatch.setattr(rubin_dsa_metadata, "_prepare_indexer", lambda md: order.append("plan"))
    md = bare(
        Metadata,
        ragged_verify_lens=None,
        enable_ragged_verification=False,
        kv_cache_params=types.SimpleNamespace(use_cache=True, num_cached_tokens_per_seq=[3]),
        enable_helix=False,
        seq_lens_kv=torch.tensor([1], dtype=torch.int),
    )
    for name in (
        "_invalidate_pool_view_cache",
        "prepare_for_mla_rope_append",
        "prepare_for_skip_indexer",
        "prepare_for_indices_conversion",
        "prepare_for_indexer_k_cache",
        "prepare_for_spec_decode",
    ):
        setattr(md, name, lambda *args, name=name: order.append(name))
    md._prepare_device_window_kv_lens = lambda kv_lens: kv_lens
    Metadata.prepare(md)
    assert order == [
        "base",
        "_invalidate_pool_view_cache",
        "prepare_for_mla_rope_append",
        "prepare_for_skip_indexer",
        "prepare_for_indices_conversion",
        "prepare_for_indexer_k_cache",
        "prepare_for_spec_decode",
        "plan",
    ]


# the GVR gate


@pytest.mark.parametrize("config_keys", [True, False], ids=["keys-on", "keys-off"])
def test_the_self_sampling_gvr_runs_on_sm107(monkeypatch, config_keys):
    """Only the engine's capability gate decides: the targets always run the self-sampling GVR,
    so the sparse attention config's Top-K keys do not."""
    params = types.SimpleNamespace(
        enable_heuristic_topk=config_keys, use_self_sampling_topk=config_keys, max_sparse_topk=2048
    )
    monkeypatch.setattr(rubin_dsa_metadata, "IS_CUTLASS_DSL_AVAILABLE", True)
    for sm in (100, 103, 107):
        assert rubin_dsa_metadata.self_sampling_gvr(params, 1, sm)
    assert not rubin_dsa_metadata.self_sampling_gvr(params, 1, 90)
    assert not rubin_dsa_metadata.self_sampling_gvr(params, 2, 107)
    monkeypatch.setattr(rubin_dsa_metadata, "IS_CUTLASS_DSL_AVAILABLE", False)
    assert not rubin_dsa_metadata.self_sampling_gvr(params, 1, 107)


@pytest.mark.parametrize("dsl", [True, False], ids=["dsl-logits", "dense-logits"])
@pytest.mark.parametrize("config_det", [True, False], ids=["config-det", "config-not-det"])
def test_the_warmup_compiles_the_deterministic_gvr_the_targets_run(monkeypatch, dsl, config_det):
    """The warmup compiles the deterministic variant through the GVR entry's warmup, whatever the
    config's determinism key says; the decode rows are the eager ones and every graph batch's."""
    calls = []
    monkeypatch.setattr(rubin_dsa_metadata, "rubin_gvr_topk_warmup", lambda *a: calls.append(a))
    md = bare(
        Metadata,
        use_self_sampling_topk=True,
        kv_cache_manager=object(),
        sparse_mla_topk=2048,
        _indexer_compress_ratio=1,
        use_cute_dsl_paged_mqa_logits=dsl,
        sparse_metadata_params=types.SimpleNamespace(gvr_deterministic_topk=config_det),
    )
    md.get_indexer_max_seq_len = lambda: 262146
    md.warmup_selfsampling_topk(8, batch_sizes=[1, 2, 12])
    rows = tuple(sorted({8, 16, 24, 32, 96}))
    stride = 262400 if dsl else None  # whole 256-column rows; else 262146 is no whole float4
    assert calls == [(2048, 262146, 1, 8, rows, stride, True)]
    md.use_self_sampling_topk = False
    md.warmup_selfsampling_topk(8)
    assert len(calls) == 1


def test_the_targets_run_the_deterministic_gvr_the_metadata_warms():
    """Each target's indexer Top-K order is the one the metadata's warmup compiles."""
    family = Path(rubin_dsa_metadata.__file__).resolve().parents[2]
    for target in ("tep8", "dep16", "helix8"):
        path = family / f"kimi_k31_pro_nvfp4__sm_107__{target}" / "modeling.py"
        values = [
            node.value.value
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            and [getattr(t, "id", None) for t in node.targets] == ["_IDX_GVR_DETERMINISTIC"]
            and isinstance(node.value, ast.Constant)
        ]
        assert values == [rubin_dsa_metadata.GVR_DETERMINISTIC], (target, values)


# the GVR warmup entry

#: The engine package whose self-sampling host module the warmup imports when it runs.
_GVR_HOST = "tensorrt_llm._torch.kernels.blackwell.top_k"


def _engine_warmup(name: str):
    """A function with the signature of the engine host module's ``name`` (read from its source:
    importing the module needs the CuTe DSL); calling it checks that the arguments bind."""
    root = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0])
    path = root / "_torch/kernels/blackwell/top_k/gvr_topk_decode_self_sampling_host.py"
    body = ast.parse(path.read_text()).body
    (node,) = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    a = node.args
    for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
        if arg is not None:
            arg.annotation = None
    node.decorator_list, node.returns, node.body = [], None, [ast.Pass()]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module([node], [])), path.name, "exec"), namespace)
    return namespace[name]


def _gvr_host(monkeypatch, calls: list, oom=()):
    """The engine's host module as a recording stand-in; the parts in ``oom`` run out of memory."""
    host = types.ModuleType(f"{_GVR_HOST}.gvr_topk_decode_self_sampling_host")
    for part, name in (("prefill", "warmup_prefill"), ("decode", "warmup_varlen")):

        def warmup(*args, _part=part, _signature=_engine_warmup(name), **kwargs):
            _signature(*args, **kwargs)
            calls.append((_part, args, kwargs))
            if _part in oom:
                raise torch.cuda.OutOfMemoryError(f"stand-in: {_part}")

        setattr(host, name, warmup)
    package = types.ModuleType(_GVR_HOST)
    package.gvr_topk_decode_self_sampling_host = host
    monkeypatch.setitem(sys.modules, _GVR_HOST, package)


@pytest.mark.parametrize(
    "args, prefill, decode",
    [
        (
            (2048, 262146, 1, 8, (8, 16, 96), 262400, True),
            ((2048, 262146), {"deterministic": True}),
            (
                (2048, 262146),
                dict(
                    compress_ratio=1,
                    next_n=8,
                    num_rows_list=(8, 16, 96),
                    row_stride=262400,
                    deterministic=True,
                ),
            ),
        ),
        (
            # A short context: the prefill compiles at least 32768 columns. Compressed index keys.
            (512, 4096, 4, 2, (2, 4), 4096, False),
            ((512, 32768), {}),
            ((512, 16384), dict(compress_ratio=4, next_n=2, num_rows_list=(2, 4), row_stride=4096)),
        ),
        (
            # No row stride of whole float4s: the decode takes the radix Top-K, nothing to compile.
            (2048, 262146, 1, 8, (8,), None, True),
            ((2048, 262146), {"deterministic": True}),
            None,
        ),
    ],
    ids=["decode-rows", "short-context", "no-row-stride"],
)
def test_the_gvr_warmup_compiles_the_engines_the_targets_run(monkeypatch, args, prefill, decode):
    """The prefill engine, then the decode engine for the rows at the logits row stride, with the
    engine warmups' own keywords; the deterministic variant where the targets run it."""
    calls = []
    _gvr_host(monkeypatch, calls)
    rubin_gvr_topk_decode.rubin_gvr_topk_warmup(*args)
    assert calls == [("prefill", *prefill)] + ([("decode", *decode)] if decode else [])


@pytest.mark.parametrize("oom", [("prefill",), ("decode",), ("prefill", "decode")])
def test_out_of_memory_skips_only_that_warmup(monkeypatch, oom):
    calls, logged = [], []
    _gvr_host(monkeypatch, calls, oom)
    monkeypatch.setattr(logger, "warning", logged.append)
    rubin_gvr_topk_decode.rubin_gvr_topk_warmup(2048, 65536, 1, 4, (4, 8), 65536, True)
    assert [part for part, _, _ in calls] == ["prefill", "decode"]
    assert len(logged) == len(oom) and all("out of memory" in line for line in logged), logged


def test_a_missing_engine_module_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, _GVR_HOST, None)
    with pytest.raises(ImportError):
        rubin_gvr_topk_decode.rubin_gvr_topk_warmup(2048, 65536, 1, 4, (4,), 65536, True)


# graph safety

_PERSISTENT = (
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
    "kv_lens_cuda_2d",
    "scheduler_metadata_buffer",
)


@pytest.mark.parametrize(
    "num_contexts, seq_lens, kv_lens, max_draft",
    [(0, [8, 8], [4000, 9000], 7), (1, [5, 1, 1], [5, 70, 130], 0)],
    ids=["decode", "mixed"],
)
def test_the_decode_rebuild_writes_the_captured_buffers_in_place(
    monkeypatch, num_contexts, seq_lens, kv_lens, max_draft
):
    # A captured graph reads these buffers at the addresses it recorded: every rebuild writes
    # into them and rebinds none.
    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    monkeypatch.setattr(
        rubin_dsa_metadata,
        "fused_dsa_decode_metadata",
        lambda *a, **k: [t.fill_(1) for t in a[3:8]],
    )
    monkeypatch.setattr(
        rubin_dsa_metadata,
        "get_paged_mqa_logits_metadata",
        lambda lens, block_kv, num_sms, out: out.fill_(int(lens.sum()) % 7),
    )
    md = _gen_md(num_contexts, seq_lens, kv_lens, max_draft=max_draft)
    before = {name: (getattr(md, name), getattr(md, name).data_ptr()) for name in _PERSISTENT}
    for _ in range(2):
        Metadata.on_update_kv_lens(md)
    for name, (tensor, ptr) in before.items():
        assert getattr(md, name) is tensor and tensor.data_ptr() == ptr, name
    # The step's request-major lengths are a view of the persistent KV lengths.
    assert (
        md.gen_indexer_kv_lens_cuda_runtime.data_ptr() == md.kv_lens_cuda[num_contexts:].data_ptr()
    )


def test_sharded_dkv_index_keys_skip_the_engine_index_slot_mappings(monkeypatch):
    # DKV's sharded index keys: the cep model builds its own insert / gather slots (one table
    # column per DKV block); the engine's per-page mappings would read past that table.
    calls = []
    monkeypatch.setattr(
        EngineIndexer, "prepare_for_update_k_cache", staticmethod(lambda md, p: calls.append("ins"))
    )
    monkeypatch.setattr(
        EngineIndexer,
        "recompute_context_kv_gather_mappings",
        staticmethod(lambda md, p: calls.append("gather")),
    )
    step = types.SimpleNamespace(
        new_kv_tokens=torch.tensor([10]),
        num_contexts=1,
        _num_past_tokens_tensor=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.tensor([10]),
        kv_lens=torch.tensor([10]),
    )
    monkeypatch.setattr(EngineIndexer, "build_indexer_params", staticmethod(lambda md: step))
    monkeypatch.setattr(
        rubin_dsa_metadata,
        "compute_cu_seqlen_kv_bounds_with_cache",
        lambda *a: (torch.zeros(10, dtype=torch.int32),) * 2,
    )
    for cache, expected in (
        (types.SimpleNamespace(dkv_index_sharded=True), []),
        (types.SimpleNamespace(dkv_index_sharded=False), ["ins", "gather"]),
        (types.SimpleNamespace(), ["ins", "gather"]),
    ):
        calls.clear()
        md = types.SimpleNamespace(
            kv_cache_manager=cache,
            num_contexts=1,
            num_ctx_tokens=10,
            num_generations=0,
            compress_ratios=[1],
            seq_lens=torch.tensor([10]),
            cu_seqlen_ks=torch.zeros(10, dtype=torch.int32),
            cu_seqlen_ke=torch.zeros(10, dtype=torch.int32),
            sparse_metadata_params=types.SimpleNamespace(q_split_threshold=1024),
            enable_context_mla_with_cached_kv=False,
            indexer_max_chunk_size=1024,
            slot_mapping_fp8_fullkv="stale",
            slot_mapping_scale_fullkv="stale",
        )
        rubin_dsa_metadata._prepare_indexer(md)
        assert calls == expected
        if expected == []:
            assert md.slot_mapping_fp8_fullkv is None and md.slot_mapping_scale_fullkv is None


def test_the_rubin_cache_states_sharded_index_keys():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
        RubinKdaDsaHybridCacheManager,
    )

    sharded = RubinKdaDsaHybridCacheManager.dkv_index_sharded.fget
    assert sharded(types.SimpleNamespace(dkv=object(), _dkv_index_whole=False)) is True
    assert sharded(types.SimpleNamespace(dkv=object(), _dkv_index_whole=True)) is False
    assert sharded(types.SimpleNamespace(dkv=None, _dkv_index_whole=False)) is False
