# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DSA indexer prefill share across attention-DP ranks (dsa_prefill_share): the rows a
rank scores for a shared chunk are bit for bit the rows the owner writes alone.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import dsa_prefill_share as share

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402


@pytest.fixture(autouse=True)
def _aggregated_server(monkeypatch):
    """No disaggregated worker role, wherever the test runs."""
    monkeypatch.delenv(share._DISAGG_ROLE_ENV, raising=False)


@inputs.on_sm103
@pytest.mark.parametrize(
    "world,tile,piece_rows", [(16, 300, 128)], ids=["tier2-tiles-128-row-pieces"]
)
def test_shared_rows_match_the_owner(monkeypatch, world, tile, piece_rows):
    """Every piece of a long-prefix chunk, scored alone with its owner tile's Top-K plan, equals the owner's own
    rows bit for bit (FP8 index keys, the perf targets' GVR prefill Top-K)."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import dsa_indexer
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
        K31DSAParams,
    )

    params = K31DSAParams(
        index_n_heads=32, index_head_dim=128, index_topk=2048, indexer_k_dtype="fp8"
    )
    # forward_prepared runs on prequantized inputs: the projection weights stay unused.
    mla = SimpleNamespace(hidden_size=128, q_lora_rank=128)
    indexer = dsa_indexer.K31Indexer(mla, params, torch.bfloat16)
    indexer._k31_perf = True

    rows, cached = 1024, 40_000
    gen = torch.Generator().manual_seed(1000 * world + tile)

    ks, ke, num_keys = inputs.chunk_windows([(cached, rows)])
    q = inputs.queries(rows, "randn", gen)
    k, k_scale = inputs.keys(num_keys, "randn", gen)
    k, k_scale = k.cuda(), k_scale.cuda()
    weights = inputs.weights(rows, "randn", gen)
    monkeypatch.setattr(dsa_indexer, "_INDEXER_MQA_LOGITS_ELEM_BUDGET", tile * num_keys)
    assert indexer.prefill_q_tile(rows, num_keys) == min(tile, rows)

    owner = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    for c0 in range(0, rows, tile):
        c1 = min(c0 + tile, rows)
        indexer.forward_prepared(
            q[c0:c1], k, k_scale, weights[c0:c1], ks[c0:c1], ke[c0:c1], owner[c0:c1]
        )
    # Every piece on its own (as some rank scores it), written back at its rows.
    shared = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    for lo, hi, tier in share.chunk_pieces(rows, num_keys, piece_rows):
        out = torch.full((hi - lo, 2048), -7, dtype=torch.int32, device="cuda")
        share.score_piece(indexer, q, k, k_scale, weights, ks, ke, out, lo, hi, tier)
        shared[lo:hi] = out
    assert torch.equal(shared, owner)
