# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix8 DSA metadata: Helix CP on top of the family metadata.

CPU checks: the helix class's override list; the KV-length refresh rule on the helix class
only; the key-sharded Top-K's configuration and its per-step width (uniform, ragged and
bound-less steps; a CUDA-graph step that cannot take it raises); the one-launch decode metadata
of a verify step against the eager chain with the Helix ownership rewrite, bit for bit; the step
inputs the target reads.
"""

import inspect
import types

import pytest
import torch
from _rubin_dsa_metadata_util import bare

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (
    rubin_dsa_key_shard,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
    rubin_dsa_metadata,
    rubin_fused_dsa_helix_decode_metadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata as EngineMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

pytestmark = pytest.mark.cpu_only

Base = rubin_dsa_metadata.RubinDSAMetadata
Helix = rubin_dsa_metadata.RubinDSAHelixMetadata


def test_the_helix_class_defines_exactly_its_override_list():
    own = {
        name
        for name, value in vars(Helix).items()
        if inspect.isfunction(value) or isinstance(value, (property, staticmethod, classmethod))
    }
    assert own == set(Helix.OVERRIDES)
    for name in dir(EngineMetadata):
        value = inspect.getattr_static(Helix, name)
        if not (inspect.isfunction(value) or isinstance(value, property)):
            continue
        if name in Helix.OVERRIDES:
            continue
        if name in Base.OVERRIDES:
            assert value is inspect.getattr_static(Base, name), name
        else:
            assert value is inspect.getattr_static(EngineMetadata, name), name


def test_the_kv_length_refresh_once_rule_is_the_helix_class_only():
    assert Helix.kv_lens_refresh_once is True
    assert not getattr(Base, "kv_lens_refresh_once", False)


def test_the_key_shard_page_size_is_the_kernels():
    assert (
        rubin_dsa_metadata._KEY_SHARD_TOKENS_PER_BLOCK
        == rubin_dsa_key_shard.KEY_SHARD_TOKENS_PER_BLOCK
    )


# key shard


def _config_md(**overrides):
    base = dict(
        mapping=types.SimpleNamespace(cp_size=8, tp_size=1, enable_attention_dp=False),
        kv_cache_manager=types.SimpleNamespace(
            indexer_k_dtype="fp8", max_draft_len=5, max_total_draft_tokens=5, max_blocks_per_seq=256
        ),
        use_cute_dsl_paged_mqa_logits=True,
        use_self_sampling_topk=True,
        _indexer_compress_ratio=1,
        _tokens_per_block=64,
        max_num_sequences=16,
        max_draft_tokens=0,
    )
    base.update(overrides)
    return bare(Helix, **base)


def test_helix8_takes_the_key_sharded_top_k():
    world, rows, width = rubin_dsa_metadata._key_shard_config(_config_md())
    # helix8 generation, batch 16, DFlash2 k5 (the cache manager's draft ceiling; the eager metadata has none yet).
    assert (world, rows, width) == (8, 16 * 6, 256 * 64)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mapping": types.SimpleNamespace(cp_size=8, tp_size=2, enable_attention_dp=False)},
        {"mapping": types.SimpleNamespace(cp_size=6, tp_size=1, enable_attention_dp=False)},
        {"use_self_sampling_topk": False},
        {"_tokens_per_block": 32},
    ],
    ids=["tp2", "cp6", "radix", "page32"],
)
def test_a_layout_the_key_shard_cannot_serve_merges_over_cp_every_step(overrides):
    # A layout the key shard cannot serve merges over CP every step; it is not refused.
    assert rubin_dsa_metadata._key_shard_config(_config_md(**overrides)) == (1, 0, 0)


def _step_md(num_gen, rows, *, ragged=False, bounds=True, graph=False, rows_max=96):
    md = bare(
        Helix,
        num_generations=num_gen,
        num_tokens=rows,
        num_ctx_tokens=0,
        is_ragged_verify=ragged,
        is_cuda_graph=graph,
        key_shard_world=8,
        key_shard_rows_max=rows_max,
    )
    md.helix_spec_row_kv_lens = lambda n: torch.ones(n, dtype=torch.int32) if bounds else None
    return md


@pytest.mark.parametrize(
    "num_gen, rows, ragged, bounds, width",
    [
        (16, 96, False, True, 6),  # k5 verify with Helix bounds
        (12, 96, False, True, 8),  # k7
        (16, 16, False, False, 1),  # plain decode, no bounds needed
        (2, 5, False, True, 0),  # rows do not divide by the requests
        (2, 8, True, True, 0),  # ragged verify
        (2, 8, False, False, 0),  # a 4-row window without bounds
    ],
    ids=["k5", "k7", "decode", "uneven", "ragged", "boundless"],
)
def test_the_key_shard_width(num_gen, rows, ragged, bounds, width):
    md = _step_md(num_gen, rows, ragged=ragged, bounds=bounds)
    assert rubin_dsa_metadata._key_shard_next_n(md) == width


def test_more_rows_than_the_workspace_raise():
    with pytest.raises(RuntimeError, match="exceed the merge workspace"):
        rubin_dsa_metadata._key_shard_next_n(_step_md(16, 112, rows_max=96))


@pytest.mark.parametrize(
    "case", [dict(ragged=True), dict(bounds=False)], ids=["ragged", "boundless"]
)
def test_a_graph_step_the_key_shard_cannot_serve_raises(case):
    md = _step_md(2, 8, graph=True, **case)
    with pytest.raises(RuntimeError, match="CUDA-graph Helix step"):
        rubin_dsa_metadata._key_shard_next_n(md)
    # The same step served by the key shard is fine under a graph.
    assert rubin_dsa_metadata._key_shard_next_n(_step_md(2, 8, graph=True)) == 4


def test_without_the_key_shard_every_step_merges():
    md = _step_md(16, 96)
    md.key_shard_world = 1
    assert rubin_dsa_metadata._key_shard_next_n(md) == 0


# decode metadata


def _helix_gen_md(seq_lens, kv_lens, local_slots, bounds, *, max_blocks=8):
    num_seqs = len(seq_lens)
    num_tokens = sum(seq_lens)
    next_n = seq_lens[0]
    return bare(
        Helix,
        num_contexts=0,
        num_generations=num_seqs,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        num_ctx_tokens=0,
        ragged_verify_lens=None,
        enable_ragged_verification=False,
        enable_helix=True,
        _helix_spec_tokens_valid=True,
        use_expanded_buffers_for_mtp=True,
        use_fp8_ds_mla=False,
        max_draft_tokens=next_n - 1,
        runtime_tokens_per_gen_step=0,
        num_sms=212,
        is_cuda_graph=False,
        _tokens_per_block=64,
        indexer_quant_block_size=128,
        _indexer_compress_ratio=1,
        kv_cache_manager=types.SimpleNamespace(index_head_dim=128),
        key_shard_world=8,
        key_shard_rows_max=256,
        seq_lens_cuda=torch.tensor(seq_lens, dtype=torch.int32),
        kv_lens_cuda=torch.tensor(kv_lens, dtype=torch.int32),
        helix_local_slots=torch.tensor(local_slots, dtype=torch.int32),
        helix_kv_bounds=torch.tensor(bounds, dtype=torch.int32),
        helix_is_inactive_rank=torch.zeros(num_seqs, dtype=torch.bool),
        indexer_k_cache_block_offsets=(
            100 + torch.arange(num_seqs * max_blocks, dtype=torch.int32)
        ).view(num_seqs, max_blocks),
        req_idx_per_token=torch.full((num_tokens,), -9, dtype=torch.int32),
        slot_mapping_fp8=torch.full((num_tokens,), -9, dtype=torch.int64),
        slot_mapping_scale=torch.full((num_tokens,), -9, dtype=torch.int64),
        gen_kv_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        gen_cached_token_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        kv_lens_cuda_2d=torch.zeros(num_seqs, next_n, dtype=torch.int32),
        kv_lens_expanded_cuda=torch.full((num_tokens,), -9, dtype=torch.int32),
        scheduler_metadata_buffer=torch.zeros(4, 2, dtype=torch.int32),
        scheduler_metadata_buffer_expanded=torch.zeros(4, 2, dtype=torch.int32),
        _helix_num_rows=num_tokens,
        helix_row_kv_lens_cuda=torch.zeros(num_tokens + 4, dtype=torch.int32),
    )


@pytest.fixture
def helix_stubs(monkeypatch):
    calls = {"fused": 0, "schedules": []}

    def fused(*args, **kwargs):
        calls["fused"] += 1
        rubin_fused_dsa_helix_decode_metadata.rubin_fused_dsa_helix_decode_metadata.reference(
            *args, **kwargs
        )

    monkeypatch.setattr(rubin_dsa_metadata, "rubin_fused_dsa_helix_decode_metadata", fused)
    monkeypatch.setattr(
        rubin_dsa_metadata,
        "get_paged_mqa_logits_metadata",
        lambda lens, block_kv, num_sms, out: calls["schedules"].append(
            (lens.clone().view(-1), out)
        ),
    )
    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    return calls


def _decode_case():
    # Two requests of a 6-token verify (k5); request 1's window straddles a page edge; tokens
    # another CP rank owns have local slot -1 and bound 0 on this rank.
    seq_lens = [6, 6]
    kv_lens = [70, 129]
    local = [64, 65, -1, -1, 66, 67, 126, 127, 128, -1, -1, -1]
    bounds = [65, 66, 66, 66, 67, 68, 127, 128, 129, 129, 129, 0]
    return seq_lens, kv_lens, local, bounds


def _run(monkeypatch, helix_stubs, fused: bool):
    md = _helix_gen_md(*_decode_case())
    for name in ("_invalidate_pool_view_cache", "_compute_kv_lens_row_reorder"):
        setattr(md, name, lambda *a: None)
    md.prepare_dense_topk_indices = lambda kv_lens, device=False: None
    if not fused:
        monkeypatch.setattr(rubin_dsa_metadata, "_helix_fused_applies", lambda md: False)
    Helix.on_update_kv_lens(md)
    return md


def test_the_one_launch_helix_metadata_equals_the_eager_chain_bit_for_bit(monkeypatch, helix_stubs):
    fused = _run(monkeypatch, helix_stubs, fused=True)
    assert helix_stubs["fused"] == 1
    fused_schedules = [(lens.tolist(), out) for lens, out in helix_stubs["schedules"]]
    helix_stubs["schedules"].clear()
    eager = _run(monkeypatch, helix_stubs, fused=False)
    assert helix_stubs["fused"] == 1
    for name in (
        "req_idx_per_token",
        "slot_mapping_fp8",
        "slot_mapping_scale",
        "gen_kv_indptr",
        "gen_cached_token_indptr",
        "kv_lens_cuda_2d",
        "kv_lens_expanded_cuda",
        "gen_indexer_kv_lens_cuda_runtime",
    ):
        assert torch.equal(getattr(fused, name), getattr(eager, name)), name
    # The rows another rank owns are not written here.
    assert fused.slot_mapping_fp8[2].item() == -1 and fused.slot_mapping_fp8[11].item() == -1
    assert fused.kv_lens_expanded_cuda.tolist()[-1] == 1  # a zero bound scores one key
    # Both paths leave the request-major and per-token schedules of the same lengths.
    eager_schedules = [(lens.tolist(), out) for lens, out in helix_stubs["schedules"]]
    assert (
        fused_schedules[0][0] == [70, 129]
        and fused_schedules[0][1] is fused.scheduler_metadata_buffer
    )
    assert fused_schedules[-1][0] == eager_schedules[-1][0]
    assert eager_schedules[-1][1] is eager.scheduler_metadata_buffer_expanded
    assert fused.key_shard_next_n == eager.key_shard_next_n == 6


def test_the_target_reads_the_key_shard_inputs_and_the_counts_buffer():
    md = _helix_gen_md(*_decode_case())
    md.gen_indexer_kv_lens_cuda_runtime = md.kv_lens_cuda
    md.helix_spec_row_kv_lens = lambda n: md.helix_kv_bounds[:n]
    table, lens, sched, row_lens = Helix.helix_key_shard_inputs(md, 12)
    assert table.data_ptr() == md.indexer_k_cache_block_offsets.data_ptr() and table.shape[0] == 2
    assert lens is md.kv_lens_cuda and sched is md.scheduler_metadata_buffer
    assert torch.equal(row_lens, md.helix_kv_bounds)
    out = Helix.helix_sparse_row_lens_out(md, 12)
    assert out.data_ptr() == md.helix_row_kv_lens_cuda.data_ptr() and out.shape[0] == 12
    md._helix_num_rows = 0
    assert Helix.helix_sparse_row_lens_out(md, 12) is None
