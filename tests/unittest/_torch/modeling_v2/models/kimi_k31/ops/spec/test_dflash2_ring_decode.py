# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 MLA drafter's block decode over a window-bounded context ring (ops/spec/dflash2_ring):
the same decode with each request's context at ring rows (positions mod R) as over a dense table."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    DraftModel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_chunk_start,
    dflash2_ring_positions,
    dflash2_ring_rows,
)

# The drafter's synthetic checkpoint, config and un-absorbed reference live with its own tests.
__extra_import_path__ = ["../../kimi_k31_pro_mxfp4__sm_10x__default"]
from test_spec_dflash2_mla import (  # noqa: E402
    _H,
    _KVL,
    _L,
    _ROPE_D,
    _WINDOW,
    _draft_config,
    _reference_block,
    _weights,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("block_size", [5, 8])
def test_block_decode_on_a_context_ring_matches_the_dense_table(block_size):
    """The same decode with each request's context in a window-bounded ring (positions mod R), for
    contexts shorter than, equal to and several times the ring: bitwise the dense table's output,
    and the un-absorbed reference's."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    draft = DraftModel(_draft_config(block_size=5), block_size=block_size)
    assert draft._ctx_ring_window == _WINDOW
    w = _weights()
    draft.load_weights({k: v.clone() for k, v in w.items()})
    draft = draft.to(device)
    scale = draft.softmax_scale

    page_size = 32  # the TRTLLM-Gen window attention reads 32- or 64-token pages
    ring = dflash2_ring_rows(_WINDOW, block_size, block_size, page_size)
    ctx_lens = [0, 3, ring, ring + 1, 3 * ring + 7]
    batch = len(ctx_lens)
    dim = _KVL + _ROPE_D
    ring_pages = ring // page_size
    dense_pages = (max(ctx_lens) + block_size + page_size - 1) // page_size
    ring_table = torch.randperm(batch * ring_pages, device=device).view(batch, ring_pages)
    dense_table = torch.randperm(batch * dense_pages, device=device).view(batch, dense_pages)

    def pools(pages_per_slot):
        # Stale garbage everywhere a context row is not stored.
        return [
            torch.randn(batch * pages_per_slot, 1, 1, page_size, dim, device=device).to(
                torch.bfloat16
            )
            for _ in range(_L)
        ]

    ring_pools, dense_pools = pools(ring_pages), pools(dense_pages)
    ctx_projs = []
    for b, n in enumerate(ctx_lens):
        proj = torch.randn(n, _H, device=device).to(torch.bfloat16)
        ctx_projs.append(proj)
        if n == 0:
            continue
        lat, _ = draft.precompute_context_kv(proj, torch.arange(n, device=device))
        pos = torch.arange(n, device=device)
        keep = dflash2_ring_chunk_start(n, ring)
        ring_pos = dflash2_ring_positions(pos[keep:], ring)
        for layer in range(_L):
            dense_pools[layer][:, 0, 0][dense_table[b, pos // page_size], pos % page_size] = lat[
                :, layer, 0
            ]
            ring_pools[layer][:, 0, 0][
                ring_table[b, ring_pos // page_size], ring_pos % page_size
            ] = lat[keep:, layer, 0]
    noise = torch.randn(batch, block_size, _H, device=device).to(torch.bfloat16)
    ctx_len = torch.tensor(ctx_lens, device=device)

    def decode(ring_rows, table, pool):
        draft._ctx_ring_rows = ring_rows
        return draft.dflash_forward(
            noise_embedding=noise,
            query_positions=ctx_len.view(-1, 1) + torch.arange(block_size, device=device),
            num_ctx_per_req=ctx_len,
            ctx_k_cache=None,
            ctx_v_cache=None,
            ctx_cache_batch_idx=torch.arange(batch, device=device),
            ctx_kv_cache=pool,
            ctx_page_table=table.to(torch.int32),
        ).view(batch, block_size, _H)

    out = decode(ring, ring_table, ring_pools)
    dense = decode(None, dense_table, dense_pools)
    assert torch.equal(out, dense)
    for b, n in enumerate(ctx_lens):
        ref = _reference_block(w, noise[b], ctx_projs[b], n, block_size, scale)
        # BF16 absorbed decode vs the FP32 un-absorbed reference (the dense test's bound, plus
        # headroom for this test's wider set of contexts).
        torch.testing.assert_close(out[b].float(), ref, atol=8e-2, rtol=5e-2)
