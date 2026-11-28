# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix8 DSA metadata: Helix CP on top of the family metadata.

CPU checks: the key-sharded Top-K's per-step width (uniform, ragged and unsharded steps; a
CUDA-graph step that cannot take it raises); the one-launch decode metadata of a verify step
against the eager chain with the Helix ownership rewrite, bit for bit, on the shared class with
FP8 index keys.
"""

import pytest
import torch

__extra_import_path__ = ["."]
from _rubin_dsa_metadata_util import (  # noqa: E402 — needs the path declared above
    PAGE,
    bare,
    data_bytes,
    index_key_manager,
    index_key_slots,
)

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import (
    rubin_dsa_metadata,
    rubin_fused_dsa_helix_decode_metadata,
)

pytestmark = pytest.mark.cpu_only

Helix = rubin_dsa_metadata.RubinDSAHelixMetadata


# key shard


def _step_md(num_gen, rows, *, ragged=False, bounds=True, graph=False, rows_max=96, world=8):
    md = bare(
        Helix,
        num_generations=num_gen,
        num_tokens=rows,
        num_ctx_tokens=0,
        is_ragged_verify=ragged,
        is_cuda_graph=graph,
        key_shard_world=world,
        key_shard_rows_max=rows_max,
    )
    md.helix_spec_row_kv_lens = lambda n: torch.ones(n, dtype=torch.int32) if bounds else None
    return md


@pytest.mark.parametrize(
    "num_gen, rows, ragged, bounds, world, width",
    [
        (12, 96, False, True, 8, 8),  # k7 verify with Helix bounds
        (16, 16, False, False, 8, 1),  # plain decode, no bounds needed
        (2, 8, True, True, 8, 0),  # ragged verify
        (16, 96, False, True, 1, 0),  # no key shard: every step merges
    ],
    ids=["k7", "decode", "ragged", "no-shard"],
)
def test_the_key_shard_width(num_gen, rows, ragged, bounds, world, width):
    md = _step_md(num_gen, rows, ragged=ragged, bounds=bounds, world=world)
    assert rubin_dsa_metadata._key_shard_next_n(md) == width


def test_a_graph_step_the_key_shard_cannot_serve_raises():
    md = _step_md(2, 8, graph=True, ragged=True)
    with pytest.raises(RuntimeError, match="CUDA-graph Helix step"):
        rubin_dsa_metadata._key_shard_next_n(md)
    # The same step served by the key shard is fine under a graph.
    assert rubin_dsa_metadata._key_shard_next_n(_step_md(2, 8, graph=True)) == 4


# decode metadata


def _helix_gen_md(cls, index_k, seq_lens, kv_lens, local_slots, bounds, *, max_blocks=8):
    num_seqs = len(seq_lens)
    num_tokens = sum(seq_lens)
    next_n = seq_lens[0]
    md = bare(
        cls,
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
        _tokens_per_block=PAGE,
        indexer_quant_block_size=128,
        _indexer_compress_ratio=1,
        # The family cache manager's index-key facts.
        kv_cache_manager=index_key_manager(index_k),
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
    for name in ("_invalidate_pool_view_cache", "_compute_kv_lens_row_reorder"):
        setattr(md, name, lambda *a: None)
    md.prepare_dense_topk_indices = lambda kv_lens, device=False: None
    # The default target's latent-KV hook, which an FP4-index-key step runs after the metadata.
    md._nvfp4_on_update_kv_lens = lambda: None
    return md


def _helix_stubs(monkeypatch, modules):
    """The one-launch entry runs its reference (CPU) and the schedules are recorded. ``modules``
    are the decode metadata's modules; the last one calls the one-launch entry."""
    calls = {"fused": [], "schedules": []}

    def fused(*args, **kwargs):
        calls["fused"].append(kwargs)
        rubin_fused_dsa_helix_decode_metadata.rubin_fused_dsa_helix_decode_metadata.reference(
            *args, **kwargs
        )

    def schedule(lens, block_kv, num_sms, out):
        calls["schedules"].append((lens.clone().view(-1), out))

    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    monkeypatch.setattr(modules[-1], "rubin_fused_dsa_helix_decode_metadata", fused)
    for module in modules:
        monkeypatch.setattr(module, "get_paged_mqa_logits_metadata", schedule)
    return calls


def _decode_case():
    # Two requests of a 6-token verify (k5); request 1's window straddles a page edge; tokens
    # another CP rank owns have local slot -1 and bound 0 on this rank.
    seq_lens = [6, 6]
    kv_lens = [70, 129]
    local = [64, 65, -1, -1, 66, 67, 126, 127, 128, -1, -1, -1]
    bounds = [65, 66, 66, 66, 67, 68, 127, 128, 129, 129, 129, 0]
    return seq_lens, kv_lens, local, bounds


def _run(monkeypatch, cls, module, index_k, fused: bool):
    md = _helix_gen_md(cls, index_k, *_decode_case())
    if not fused:
        monkeypatch.setattr(module, "_helix_fused_applies", lambda md: False)
    cls.on_update_kv_lens(md)
    return md


def test_the_one_launch_helix_metadata_equals_the_eager_chain_bit_for_bit(monkeypatch):
    """The shared helix8 class with FP8 index keys."""
    index_k = "fp8"
    cls, modules = Helix, (rubin_dsa_metadata,)
    helix_stubs = _helix_stubs(monkeypatch, modules)
    fused = _run(monkeypatch, cls, modules[-1], index_k, fused=True)
    ((kwargs,),) = (helix_stubs["fused"],)
    assert kwargs["data_bytes_per_token"] == data_bytes(index_k)
    fused_schedules = [(lens.tolist(), out) for lens, out in helix_stubs["schedules"]]
    helix_stubs["schedules"].clear()
    eager = _run(monkeypatch, cls, modules[-1], index_k, fused=False)
    assert len(helix_stubs["fused"]) == 1
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
    # This rank's slot 64 of request 0 is row 0 of its page 1 (block 101); request 1's slot 128
    # is row 0 of its page 2 (block 110). The rows another rank owns are not written here.
    assert (
        fused.slot_mapping_fp8[0].item(),
        fused.slot_mapping_scale[0].item(),
    ) == index_key_slots(index_k, 101, 0)
    assert fused.slot_mapping_fp8[8].item() == index_key_slots(index_k, 110, 0)[0]
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
