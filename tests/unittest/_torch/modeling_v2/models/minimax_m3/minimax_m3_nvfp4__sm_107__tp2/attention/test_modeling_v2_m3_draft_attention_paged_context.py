# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter layer's context attention on SM107 as the server runs it: paged context FMHA (chunked prefill
and speculative draft tokens set ``use_paged_context_fmha``) over the shared manager's FP8 draft view at its
128-token pages, which SM107 serves through the thop paged-context kernel (no TRTLLM-Gen FMHA there).

Each case is checked against the exact float64 reference of ``m3_draft_attention`` (``test_modeling_v2_m3_draft_
attention.py``'s band), with fresh contexts, chunked appends over a cached prefix (page-aligned or not), two contexts
in one step and a mixed step with generation rows beside the context. The appended K / V rows and a shifted-reference
discriminator are checked as there.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.attention.m3_draft_attention import (  # noqa: E501
    m3_draft_attention as op,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
import _m3_attn_runtime_tp2 as rt  # noqa: E402 — needs the path declared above
from _validating import ULP, assert_within_ulp  # noqa: E402

assert torch.cuda.is_available(), "m3_draft_attention requires a CUDA device"
pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the SM107 thop paged-context path (no TRTLLM-Gen FMHA)"
)

# test_modeling_v2_m3_draft_attention.py's band (BF16 ulps of each row's largest magnitude / of the relative RMS).
_ELEMENT_ULP = 30.0
_RMS_ULP = 12.0
_PAD = 3

# (case id, cached tokens per request, new rows per request, context requests first).
_CASES = [
    ("fresh_ctx_512", (0,), (512,), 1),
    ("append_1000_unaligned", (1000,), (700,), 1),
    ("append_1024_aligned", (1024,), (300,), 1),
    ("two_ctx_prefix_384", (0, 384), (257, 129), 2),
    ("prefix_ctx_beside_verify", (500, 200, 1000), (200, 4, 4), 1),
    ("fresh_ctx_beside_decode", (0, 200, 1000), (200, 1, 1), 1),
    ("fresh_ctx_beside_verify", (0, 200, 1000), (200, 4, 4), 1),
]


def _ulp_stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    o, r = out.float(), ref.float()
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / scale).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ULP[torch.bfloat16]
    return element, rms


def _paged_draft_metadata(manager) -> TrtllmAttentionMetadata:
    """``rt.make_draft_metadata`` with the server's runtime features (chunked prefill, Eagle3 draft tokens)."""
    view = manager.get_draft_kv_cache_view()
    metadata = TrtllmAttentionMetadata(
        max_num_requests=rt.MAX_BATCH,
        max_num_tokens=rt.MAX_NUM_TOKENS,
        kv_cache_manager=view,
        mapping=rt.mapping(),
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=True, has_speculative_draft_tokens=True
        ),
    )
    assert metadata.use_paged_context_fmha
    return metadata


def _prepare(metadata, request_ids, qlens, cached, num_contexts: int) -> None:
    """``rt.prepare`` as the serving runner prepares a step: a context request's prompt length is its CURRENT CHUNK's
    token count (``prompt_lengths.append(len(prompt_tokens))``) with the prefix in ``num_cached_tokens_per_seq``, so
    the chunk is appended at the cached length; generation rows as ``rt.prepare``."""
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata.seq_lens = torch.tensor(list(qlens), dtype=torch.int)
    metadata.num_contexts = num_contexts
    metadata.request_ids = list(request_ids)
    metadata.prompt_lens = [
        n if row < num_contexts else c for row, (c, n) in enumerate(zip(cached, qlens))
    ]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached))
    metadata.prepare()


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_paged_context_at_128_token_pages(case) -> None:
    name, cached, qlens, contexts = case
    rids = list(range(len(qlens)))
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(name.encode()) % 2**31)
    manager = rt.make_manager(draft=True)
    try:
        backend = rt.make_draft_backend()
        metadata = _paged_draft_metadata(manager)
        assert metadata.tokens_per_block == rt.PAGE == 128
        assert 128 in manager.get_draft_kv_cache_view().trtllm_gen_extra_tokens_per_block
        rt.allocate(manager, rids, qlens, cached)
        for rid, c in zip(rids, cached):
            if c:
                hist = torch.randn(2, c, rt.KV_HEADS * rt.HEAD_DIM, generator=gen, device="cuda")
                hist = hist.to(torch.float8_e4m3fn)
                rt.write_fp8_rows(manager, rt.DRAFT, rid, torch.arange(c), hist[0], hist[1])
        pool = manager.get_buffers(rt.DRAFT, "HND")
        prefix_slots = {
            rid: rt.flat_slots(manager, rt.DRAFT, rid, torch.arange(c, device="cuda")).long()
            for rid, c in zip(rids, cached)
            if c
        }
        prefix_before = {
            rid: pool[sl // rt.PAGE, :, :, sl % rt.PAGE].clone() for rid, sl in prefix_slots.items()
        }
        _prepare(metadata, rids, qlens, cached, contexts)
        total = sum(qlens)
        width = (rt.HEADS + 2 * rt.KV_HEADS) * rt.HEAD_DIM
        qkv = torch.randn(total + _PAD, width, generator=gen, device="cuda").to(torch.bfloat16)

        out = op(backend, qkv, metadata)
        # The cached prefix's K / V rows are left as they were (the call appends only the step's rows).
        torch.cuda.synchronize()
        changed = {
            rid: int(
                (
                    pool[sl // rt.PAGE, :, :, sl % rt.PAGE].view(torch.uint8)
                    != prefix_before[rid].view(torch.uint8)
                )
                .any(dim=-1)
                .sum()
            )
            for rid, sl in prefix_slots.items()
        }
        print(f"\n[{name}] cached prefix rows x heads x K/V changed by the call: {changed}")
        ref = op.reference(backend, qkv, metadata)
        assert out.shape == (total, rt.HEADS * rt.HEAD_DIM) and out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()
        element, rms = _ulp_stats(out, ref)
        print(f"\n[{name}] max element {element:.2f} ulp, rms {rms:.3f} ulp")
        # Every live row's K / V landed in its page and slot (the append of a chunk past a cached prefix).
        q_size = rt.HEADS * rt.HEAD_DIM
        first = 0
        for rid, c, n in zip(rids, cached, qlens):
            positions = torch.arange(c, c + n, device="cuda")
            slots = rt.flat_slots(manager, rt.DRAFT, rid, positions).long()
            rows = qkv[first : first + n]
            for h in range(rt.KV_HEADS):
                v_cols = q_size + (rt.KV_HEADS + h) * rt.HEAD_DIM
                want_v = rows[:, v_cols : v_cols + rt.HEAD_DIM].to(torch.float8_e4m3fn)
                got_v = pool[slots // rt.PAGE, 1, h, slots % rt.PAGE]
                assert torch.equal(got_v.view(torch.uint8), want_v.view(torch.uint8)), f"V head {h}"
            first += n

        assert not any(changed.values()), f"the call overwrote cached prefix K / V rows: {changed}"
        assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)

        shifted = op.reference(backend, qkv.roll(1, 0), metadata)
        with pytest.raises(AssertionError):
            assert_within_ulp(out, shifted, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    finally:
        manager.shutdown()
