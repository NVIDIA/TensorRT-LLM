# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MSA-backed MiniMax-M3 sparse attention runtime.

The MSA stack (`fmha_sm100` package) is SM100-only and not installed in
the standard CI runners, so these tests are GPU-free structural checks:

  * config to params to backend-class wiring (`sparse_use_msa`),
  * the MSA backend is a `TrtllmAttention` subclass and does not inherit
    the Triton reference backend,
  * the backend-neutral helpers in `common` (cache-layout adapters,
    per-request page-table and length derivation),
  * the FMHA wiring: `MsaSparseGqaFmha` is a dispatch-participating `Fmha`
    registered under `msa_sparse_gqa`,
  * regression guards that the removed proxy-wrapper and duplicate-resolver
    symbols stay gone.

End-to-end kernel parity is covered by the MiniMax-M3 integration
accuracy test (SM100 + fmha_sm100 + weights).
"""

from __future__ import annotations

from typing import List

import pytest
import torch

# Module-under-test imports are kept lazy so the file imports cleanly on
# hosts where heavy TRT-LLM C++ extensions are absent.
sparse_minimax_m3 = pytest.importorskip("tensorrt_llm._torch.attention_backend.sparse.minimax_m3")
common = pytest.importorskip("tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common")
msa_backend = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend"
)


# ---------------------------------------------------------------------------
# Param + config plumbing
# ---------------------------------------------------------------------------


def test_sparse_attention_config_threads_use_msa_into_params():
    """`sparse_use_msa=True` must land on the lowered SparseParams."""
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    cfg = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True)
    assert cfg.to_sparse_params().use_msa is True
    assert MiniMaxM3SparseAttentionConfig().to_sparse_params().use_msa is False


def test_backend_dispatch_picks_msa_when_flag_set():
    """`utils._resolve_minimax_m3_backend_cls` honours sparse_use_msa."""
    from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    triton_cls = sparse_minimax_m3.get_minimax_m3_attention_backend_cls()
    msa_cls = sparse_minimax_m3.get_minimax_m3_msa_attention_backend_cls()

    # _resolve_minimax_m3_backend_cls consumes the lowered
    # MiniMaxM3SparseParams (.use_msa), not the user-facing config.
    triton_params = MiniMaxM3SparseAttentionConfig(sparse_use_msa=False).to_sparse_params()
    msa_params = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True).to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(triton_params) is triton_cls
    assert _resolve_minimax_m3_backend_cls(msa_params) is msa_cls


def test_msa_backend_is_trtllm_subclass_not_triton_subclass():
    """The MSA backend mimics DSATrtllmAttention.

    It must subclass `TrtllmAttention` (to reuse the FMHA dispatch loop)
    and must not inherit the Triton reference backend.
    """
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

    msa_cls = sparse_minimax_m3.get_minimax_m3_msa_attention_backend_cls()
    triton_cls = sparse_minimax_m3.get_minimax_m3_attention_backend_cls()

    assert issubclass(msa_cls, TrtllmAttention)
    assert not issubclass(msa_cls, triton_cls)
    # Its metadata rides the TrtllmAttentionMetadata stack.
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

    assert issubclass(msa_cls.Metadata, TrtllmAttentionMetadata)


def test_duplicate_resolver_helper_removed():
    """The duplicate `get_..._with_msa` resolver must be gone (single resolver)."""
    assert not hasattr(msa_backend, "get_minimax_m3_attention_backend_cls_with_msa")
    assert not hasattr(sparse_minimax_m3, "get_minimax_m3_attention_backend_cls_with_msa")


# ---------------------------------------------------------------------------
# Cache layout adapters (backend-neutral helpers in common.py)
# ---------------------------------------------------------------------------


def test_cache_view_to_msa_paged_4d_layout():
    num_pages, tokens_per_block, num_kv_heads, head_dim = 3, 128, 2, 128
    cache_view = torch.randn(
        num_pages, tokens_per_block, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    paged = common.cache_view_to_msa_paged(cache_view)
    assert paged.shape == (num_pages, num_kv_heads, tokens_per_block, head_dim)
    assert paged.is_contiguous()
    assert torch.equal(paged, cache_view.permute(0, 2, 1, 3).contiguous())


def test_cache_view_to_msa_paged_3d_flat_slot_layout():
    num_slots, num_kv_heads, head_dim = 64, 2, 128
    cache_view = torch.randn(num_slots, num_kv_heads, head_dim, dtype=torch.bfloat16)
    paged = common.cache_view_to_msa_paged(cache_view)
    assert paged.shape == (1, num_kv_heads, num_slots, head_dim)
    assert torch.equal(paged, cache_view.permute(1, 0, 2).unsqueeze(0).contiguous())


def test_idx_cache_to_msa_paged_handles_both_ranks():
    num_pages, tokens_per_block, sparse_index_dim = 2, 128, 128
    paged_4d = torch.randn(num_pages, tokens_per_block, 1, sparse_index_dim, dtype=torch.bfloat16)
    out_4d = common.idx_cache_to_msa_paged(paged_4d)
    assert out_4d.shape == (num_pages, 1, tokens_per_block, sparse_index_dim)
    assert torch.equal(out_4d, paged_4d.permute(0, 2, 1, 3).contiguous())

    flat_3d = torch.randn(num_pages * tokens_per_block, 1, sparse_index_dim, dtype=torch.bfloat16)
    out_3d = common.idx_cache_to_msa_paged(flat_3d)
    assert out_3d.shape == (1, 1, flat_3d.shape[0], sparse_index_dim)


def test_cache_view_to_msa_paged_rejects_other_ranks():
    with pytest.raises(ValueError, match="rank"):
        common.cache_view_to_msa_paged(torch.empty(4, 8))
    with pytest.raises(ValueError, match="rank"):
        common.idx_cache_to_msa_paged(torch.empty(2))


# ---------------------------------------------------------------------------
# Metadata derivation (whole-batch lens + page table)
# ---------------------------------------------------------------------------


def _build_metadata_for_test(
    *,
    is_prefill: bool,
    seq_lens: List[int],
    extend_seq_lens: List[int] | None = None,
    page_size: int = 128,
):
    """Construct a minimal `MiniMaxM3SparseAttentionMetadata`."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseAttentionMetadata,
    )

    batch = len(seq_lens)
    max_kv_len = max(((s + page_size - 1) // page_size) * page_size for s in seq_lens)
    req_to_token = torch.arange(batch * max_kv_len, dtype=torch.int32).view(batch, max_kv_len)
    slot_ids = torch.arange(batch, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

    if is_prefill:
        assert extend_seq_lens is not None
        prefix_lens = torch.tensor(
            [seq_lens[b] - extend_seq_lens[b] for b in range(batch)], dtype=torch.int32
        )
        cu_q = [0]
        for x in extend_seq_lens:
            cu_q.append(cu_q[-1] + x)
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=True,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_t,
            seq_lens_cpu=seq_lens_t,
            prefix_lens=prefix_lens,
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32),
            extend_seq_lens_cpu=list(extend_seq_lens),
        )
    else:
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=False,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_t,
            seq_lens_cpu=seq_lens_t,
        )
    meta.prepare()
    return meta


def test_whole_batch_lens_prefill_uses_extend_lens_and_prefix():
    meta = _build_metadata_for_test(
        is_prefill=True, seq_lens=[256, 384], extend_seq_lens=[128, 128]
    )
    qo, kv, qo_off, _ = msa_backend._whole_batch_lens(meta, None, 128)
    assert qo.dtype == torch.int32
    assert qo.tolist() == [128, 128]
    assert kv.tolist() == [256, 384]
    assert qo_off.tolist() == [128, 256]


def test_whole_batch_lens_decode_uses_unit_q_and_kv_minus_one():
    meta = _build_metadata_for_test(is_prefill=False, seq_lens=[200, 300])
    qo, kv, qo_off, _ = msa_backend._whole_batch_lens(meta, None, 128)
    assert qo.tolist() == [1, 1]
    assert kv.tolist() == [200, 300]
    assert qo_off.tolist() == [199, 299]


def test_build_kv_indices_packs_per_request_pages():
    page_size = 128
    meta = _build_metadata_for_test(
        is_prefill=False,
        seq_lens=[page_size * 2, page_size * 3 - 5],
        page_size=page_size,
    )
    kv_indices, kv_lens = common.build_kv_indices_and_lens(meta, page_size)
    assert kv_indices.dtype == torch.int32
    assert kv_indices.tolist() == [0, 1, 3, 4, 5]
    assert kv_lens.tolist() == [256, 379]


# ---------------------------------------------------------------------------
# FMHA wiring: MsaSparseGqaFmha as a registered Fmha
# ---------------------------------------------------------------------------


def test_msa_sparse_gqa_registered_as_fmha():
    """The block-sparse main FMHA is a registered dispatch-participating Fmha.

    It inherits `Fmha` directly (not `PhasedFmha`): MSA does its own
    whole-batch dispatch, so there is no ctx/gen phase split to reuse.
    """
    from tensorrt_llm._torch.attention_backend.fmha import FMHA_LIBS, MsaSparseGqaFmha, PhasedFmha
    from tensorrt_llm._torch.attention_backend.fmha.interface import Fmha

    assert "msa_sparse_gqa" in FMHA_LIBS
    assert FMHA_LIBS["msa_sparse_gqa"] is MsaSparseGqaFmha
    assert issubclass(MsaSparseGqaFmha, Fmha)
    assert not issubclass(MsaSparseGqaFmha, PhasedFmha)


def test_msa_sparse_gqa_is_supported_gates_on_sparse_prediction():
    """is_supported must reject non-M3-MSA requests (no sparse block indices)."""
    from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha

    class _DummyAttn:
        pass

    fmha = MsaSparseGqaFmha(_DummyAttn())
    # No forward_args / no sparse_prediction -> not an MSA sparse request.
    assert fmha.is_supported(None, None, None, None, None) is False


def test_proxy_wrapper_and_block_sparse_bases_removed():
    """The Fmha-lib proxy wrapper + opt-out bases must be gone (indexer calls fmha_sm100 directly)."""
    import tensorrt_llm._torch.attention_backend.fmha as fmha_pkg

    for removed in ("MsaProxyMqaFmha", "IndexerProxyFmha", "BlockSparseFmha"):
        assert not hasattr(fmha_pkg, removed), f"{removed} should have been removed"
    from tensorrt_llm._torch.attention_backend.fmha import FMHA_LIBS

    assert "msa_proxy_mqa" not in FMHA_LIBS


# ---------------------------------------------------------------------------
# Lazy fmha_sm100 import guard
# ---------------------------------------------------------------------------


def _make_import_blocklist(blocked, *, fallback):
    def _shim(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked or name.split(".")[0] in blocked:
            raise ImportError(f"blocked: {name}")
        return fallback(name, globals, locals, fromlist, level)

    return _shim


def test_require_msa_module_raises_when_absent(monkeypatch):
    """The lazy import wrapper produces a descriptive error."""
    import sys

    original = sys.modules.pop("fmha_sm100", None)
    try:
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocklist({"fmha_sm100"}, fallback=__import__),
        )
        with pytest.raises(RuntimeError, match="fmha_sm100"):
            common.require_msa_module()
    finally:
        if original is not None:
            sys.modules["fmha_sm100"] = original
