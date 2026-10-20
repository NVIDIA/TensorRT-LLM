# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_MTP_FUSED_INPUT: the MTP fc input (masked embedding gather + both pre-fc Gemma RMSNorms + concat +
row-parallel chunk) in one launch vs the stock modules."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _modules(V, H, eps, seed):
    from tensorrt_llm._torch.modules.embedding import Embedding
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    g = torch.Generator(device="cuda").manual_seed(seed)
    emb = Embedding(V, H, dtype=torch.bfloat16).cuda()
    emb.weight.data.copy_((torch.randn((V, H), generator=g, device="cuda") * 0.05).to(torch.bfloat16))
    n_e = RMSNorm(hidden_size=H, eps=eps, dtype=torch.bfloat16, use_gemma=True).cuda()
    n_h = RMSNorm(hidden_size=H, eps=eps, dtype=torch.bfloat16, use_gemma=True).cuda()
    n_e.weight.data.copy_((torch.randn(H, generator=g, device="cuda") * 0.3).to(torch.bfloat16))
    n_h.weight.data.copy_((torch.randn(H, generator=g, device="cuda") * 0.3).to(torch.bfloat16))
    return emb, n_e, n_h, g


def _stock(ids, hidden, emb, n_e, n_h, tp, rank):
    x = torch.concat([n_e(emb(ids)), n_h(hidden)], dim=-1)
    return torch.chunk(x, tp, dim=-1)[rank] if tp > 1 else x


@pytest.mark.parametrize("rows", [1, 7, 28])
@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_fused_input_matches_stock(rows, tp):
    from tensorrt_llm._torch.modules.q35_mtp_glue import mtp_fused_input

    V, H, eps = 4096, 4096, 1e-6
    emb, n_e, n_h, g = _modules(V, H, eps, seed=rows * 31 + tp)
    ids = torch.randint(0, V, (rows, ), generator=g, device="cuda", dtype=torch.int32)
    if rows > 2:
        ids[1] = -1  # rejection-padding id -> zero embedding (stock masking)
        ids[2] = V  # out of range -> zero embedding
    hidden = (torch.randn((rows, H), generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    with torch.inference_mode():
        for rank in range(tp):
            ref = _stock(ids, hidden, emb, n_e, n_h, tp, rank)
            got = mtp_fused_input(ids, emb.weight, hidden, n_e.weight, n_h.weight, eps, eps, tp, rank)
            assert got.shape == ref.shape and got.is_contiguous()
            # fp32 math of the stock kernels; reduction order may differ -> at most 1 bf16 ulp
            torch.testing.assert_close(got.float(), ref.float(), rtol=8e-3, atol=1e-5)
            exact = (got == ref).float().mean().item()
            assert exact > 0.97, exact
            # The embedding-half columns of this chunk are zero for the masked rows.
            lo, hi = rank * (2 * H // tp), (rank + 1) * (2 * H // tp)
            emb_cols = max(0, min(hi, H) - lo)
            if rows > 2 and emb_cols:
                assert torch.all(got[1, :emb_cols] == 0) and torch.all(got[2, :emb_cols] == 0)


def test_negative_controls():
    """A wrong Gemma offset or swapped halves must not pass the comparison."""
    from tensorrt_llm._torch.modules.q35_mtp_glue import mtp_fused_input

    V, H, eps = 1024, 4096, 1e-6
    emb, n_e, n_h, g = _modules(V, H, eps, seed=7)
    ids = torch.randint(0, V, (7, ), generator=g, device="cuda", dtype=torch.int64)
    hidden = torch.randn((7, H), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode():
        ref = _stock(ids, hidden, emb, n_e, n_h, 8, 0)
        got = mtp_fused_input(ids, emb.weight, hidden, n_e.weight, n_h.weight, eps, eps, 8, 0)
        torch.testing.assert_close(got.float(), ref.float(), rtol=8e-3, atol=1e-5)
        bad = mtp_fused_input(ids, emb.weight, hidden, n_e.weight - 1.0, n_h.weight, eps, eps, 8, 0)
        assert not torch.allclose(bad.float(), ref.float(), rtol=8e-3, atol=1e-5)
        swapped = mtp_fused_input(ids, emb.weight, hidden, n_e.weight, n_h.weight, eps, eps, 8, 4)
        assert not torch.allclose(swapped.float(), ref.float(), rtol=8e-3, atol=1e-5)


def test_strided_hidden_rows():
    """Draft steps pass row slices of a larger buffer."""
    from tensorrt_llm._torch.modules.q35_mtp_glue import mtp_fused_input

    V, H, eps = 512, 4096, 1e-6
    emb, n_e, n_h, g = _modules(V, H, eps, seed=11)
    big = torch.randn((12, H), generator=g, device="cuda").to(torch.bfloat16)
    hidden = big[3:7]
    ids = torch.randint(0, V, (4, ), generator=g, device="cuda", dtype=torch.int32)
    with torch.inference_mode():
        ref = _stock(ids, hidden, emb, n_e, n_h, 2, 1)
        got = mtp_fused_input(ids, emb.weight, hidden, n_e.weight, n_h.weight, eps, eps, 2, 1)
        torch.testing.assert_close(got.float(), ref.float(), rtol=8e-3, atol=1e-5)


@pytest.mark.parametrize("tp", [1, 8])
def test_fused_input_applies_the_deferred_kv_increment_once(tp):
    """TRTLLM_Q35_DRAFT_BOOKKEEPING folds the draft loop's kv_lens[:rows] += 1 into this launch: every rank
    increments each of the first kv_rows entries exactly once, whichever half its chunk lies in."""
    from tensorrt_llm._torch.modules.q35_mtp_glue import mtp_fused_input

    V, H, eps = 512, 4096, 1e-6
    emb, n_e, n_h, g = _modules(V, H, eps, seed=13)
    ids = torch.randint(0, V, (4, ), generator=g, device="cuda", dtype=torch.int32)
    hidden = torch.randn((4, H), generator=g, device="cuda").to(torch.bfloat16)
    with torch.inference_mode():
        for rank in range(tp):
            kv = torch.arange(100, 108, device="cuda", dtype=torch.int32)
            got = mtp_fused_input(ids, emb.weight, hidden, n_e.weight, n_h.weight, eps, eps, tp, rank,
                                  kv_lens=kv, kv_rows=3)
            ref = _stock(ids, hidden, emb, n_e, n_h, tp, rank)
            torch.testing.assert_close(got.float(), ref.float(), rtol=8e-3, atol=1e-5)
            assert kv.tolist() == [101, 102, 103, 103, 104, 105, 106, 107], (rank, kv.tolist())
