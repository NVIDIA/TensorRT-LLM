# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MSA-backed MiniMax-M3 sparse attention runtime.

The MSA stack (``fmha_sm100`` package) is SM100-only and not installed
in the standard CI runners. These tests therefore split into two
tiers:

  * **Static plumbing tests** (always run): config -> params -> backend
    class wiring; layout adapter shapes; metadata derivation. These do
    not touch the MSA kernels and run anywhere PyTorch + the TRT-LLM
    Python imports succeed.
  * **Live kernel tests** (skipped when ``fmha_sm100`` is unavailable
    or no SM100 GPU is present): end-to-end forward parity check
    against the in-tree Triton reference path.
"""

from __future__ import annotations

import importlib
from typing import List

import pytest
import torch

# Module-under-test imports are kept lazy so the file imports cleanly
# on hosts where heavy TRT-LLM C++ extensions are absent (the rest of
# the testsuite uses the same pattern).
sparse_minimax_m3 = pytest.importorskip("tensorrt_llm._torch.attention_backend.sparse.minimax_m3")
msa_backend = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend"
)
m3_backend = pytest.importorskip("tensorrt_llm._torch.attention_backend.sparse.minimax_m3.backend")


# ---------------------------------------------------------------------------
# Param + config plumbing
# ---------------------------------------------------------------------------


def test_sparse_attention_config_threads_use_msa_into_params():
    """``sparse_use_msa=True`` must land on the lowered SparseParams."""
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    cfg = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True)
    params = cfg.to_sparse_params()
    assert params.use_msa is True

    cfg_default = MiniMaxM3SparseAttentionConfig()
    params_default = cfg_default.to_sparse_params()
    assert params_default.use_msa is False


def test_backend_dispatch_picks_msa_when_flag_set():
    """``utils._resolve_minimax_m3_backend_cls`` honours sparse_use_msa."""
    from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    triton_cls = sparse_minimax_m3.get_minimax_m3_attention_backend_cls()
    msa_cls = sparse_minimax_m3.get_minimax_m3_msa_attention_backend_cls()

    triton_cfg = MiniMaxM3SparseAttentionConfig(sparse_use_msa=False)
    assert _resolve_minimax_m3_backend_cls(triton_cfg) is triton_cls

    msa_cfg = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True)
    resolved = _resolve_minimax_m3_backend_cls(msa_cfg)
    assert resolved is msa_cls
    # MSA class subclasses the Triton class so the model layer's
    # isinstance check still accepts it.
    assert issubclass(resolved, triton_cls)


def test_get_with_msa_helper_routes_on_params_flag():
    """``get_minimax_m3_attention_backend_cls_with_msa`` matches the dispatch."""
    triton_cls = sparse_minimax_m3.get_minimax_m3_attention_backend_cls()
    msa_cls = sparse_minimax_m3.get_minimax_m3_msa_attention_backend_cls()

    no_msa = sparse_minimax_m3.MiniMaxM3SparseConfig  # noqa: F841 (just exercise import)
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseParams,
    )

    triton_params = MiniMaxM3SparseParams(use_msa=False)
    msa_params = MiniMaxM3SparseParams(use_msa=True)
    assert msa_backend.get_minimax_m3_attention_backend_cls_with_msa(triton_params) is triton_cls
    assert msa_backend.get_minimax_m3_attention_backend_cls_with_msa(msa_params) is msa_cls


# ---------------------------------------------------------------------------
# Cache layout adapter
# ---------------------------------------------------------------------------


def test_cache_view_to_msa_paged_4d_layout():
    """4-D pool views permute (page_size, num_kv_heads) -> HND."""
    num_pages, tokens_per_block, num_kv_heads, head_dim = 3, 128, 2, 128
    cache_view = torch.randn(
        num_pages, tokens_per_block, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    paged = msa_backend._cache_view_to_msa_paged(cache_view)
    assert paged.shape == (num_pages, num_kv_heads, tokens_per_block, head_dim)
    assert paged.is_contiguous()
    # Values must agree element-wise: paged[p, h, t, d] == view[p, t, h, d].
    expected = cache_view.permute(0, 2, 1, 3).contiguous()
    assert torch.equal(paged, expected)


def test_cache_view_to_msa_paged_3d_flat_slot_layout():
    """3-D flat-slot caches collapse into a single virtual page."""
    num_slots, num_kv_heads, head_dim = 64, 2, 128
    cache_view = torch.randn(num_slots, num_kv_heads, head_dim, dtype=torch.bfloat16)
    paged = msa_backend._cache_view_to_msa_paged(cache_view)
    assert paged.shape == (1, num_kv_heads, num_slots, head_dim)
    assert paged.is_contiguous()
    expected = cache_view.permute(1, 0, 2).unsqueeze(0).contiguous()
    assert torch.equal(paged, expected)


def test_idx_cache_to_msa_paged_handles_both_ranks():
    """Index cache adapter mirrors the K-cache one but for single-head data."""
    num_pages, tokens_per_block, sparse_index_dim = 2, 128, 128
    paged_4d = torch.randn(num_pages, tokens_per_block, 1, sparse_index_dim, dtype=torch.bfloat16)
    out_4d = msa_backend._idx_cache_to_msa_paged(paged_4d)
    assert out_4d.shape == (num_pages, 1, tokens_per_block, sparse_index_dim)
    assert torch.equal(out_4d, paged_4d.permute(0, 2, 1, 3).contiguous())

    flat_3d = torch.randn(num_pages * tokens_per_block, 1, sparse_index_dim, dtype=torch.bfloat16)
    out_3d = msa_backend._idx_cache_to_msa_paged(flat_3d)
    assert out_3d.shape == (1, 1, flat_3d.shape[0], sparse_index_dim)


def test_cache_view_to_msa_paged_rejects_other_ranks():
    with pytest.raises(ValueError, match="rank"):
        msa_backend._cache_view_to_msa_paged(torch.empty(4, 8))
    with pytest.raises(ValueError, match="rank"):
        msa_backend._idx_cache_to_msa_paged(torch.empty(2))


# ---------------------------------------------------------------------------
# Metadata derivation
# ---------------------------------------------------------------------------


def _build_metadata_for_test(
    *,
    is_prefill: bool,
    seq_lens: List[int],
    extend_seq_lens: List[int] | None = None,
    page_size: int = 128,
):
    """Construct a minimal :class:`MiniMaxM3SparseAttentionMetadata`.

    Builds a stable, contiguous slot layout so the page-index gather in
    :func:`_build_kv_indices_and_lens` produces deterministic block ids.
    """
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
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32)
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=True,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_t,
            seq_lens_cpu=seq_lens_t,
            prefix_lens=prefix_lens,
            cu_seqlens_q=cu_seqlens_q,
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


def test_qo_lens_offsets_prefill_uses_extend_lens_and_prefix():
    meta = _build_metadata_for_test(
        is_prefill=True,
        seq_lens=[256, 384],
        extend_seq_lens=[128, 128],
    )
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = msa_backend._qo_lens_offsets_from_metadata(meta)
    assert qo_lens_cpu.dtype == torch.int32
    assert qo_lens_cpu.tolist() == [128, 128]
    assert kv_lens_cpu.tolist() == [256, 384]
    # Causal offset is per-request prefix length (kv - extend).
    assert qo_offset_cpu.tolist() == [128, 256]


def test_qo_lens_offsets_decode_uses_unit_q_and_kv_minus_one():
    meta = _build_metadata_for_test(is_prefill=False, seq_lens=[200, 300])
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = msa_backend._qo_lens_offsets_from_metadata(meta)
    assert qo_lens_cpu.tolist() == [1, 1]
    assert kv_lens_cpu.tolist() == [200, 300]
    # Decode causal offset = kv_len - 1 (the new token's position).
    assert qo_offset_cpu.tolist() == [199, 299]


def test_build_kv_indices_packs_per_request_pages():
    """``kv_indices`` is the concatenated per-request page table."""
    page_size = 128
    meta = _build_metadata_for_test(
        is_prefill=False,
        seq_lens=[page_size * 2, page_size * 3 - 5],
        page_size=page_size,
    )
    kv_indices, kv_lens = msa_backend._build_kv_indices_and_lens(meta, page_size)

    # Request 0 owns slots [0, 384) so its pages are {0, 1, 2}, but only
    # 2 pages are needed (seq_len=256). Request 1 starts at slot 384,
    # so its first page is page id 3.
    # The helper builds page indices by ``req_to_token[b, page_starts]
    # // page_size`` where the slot ids are contiguous arange.
    # For request 0 (seq=256): page_starts=[0, 128], slots=[0, 128],
    # page ids=[0, 1].
    # For request 1 (seq=379, ceil=3 pages): page_starts=[0, 128, 256]
    # mapped to req_rows[1] = [384..767], slots=[384, 512, 640],
    # page ids=[3, 4, 5].
    assert kv_indices.dtype == torch.int32
    assert kv_indices.tolist() == [0, 1, 3, 4, 5]
    assert kv_lens.tolist() == [256, 379]


# ---------------------------------------------------------------------------
# Backend class construction
# ---------------------------------------------------------------------------


def test_msa_backend_rejects_disable_index_value_false():
    """Layer construction surfaces the unsupported-mode error early."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseParams,
    )

    msa_cls = sparse_minimax_m3.get_minimax_m3_msa_attention_backend_cls()
    bad_params = MiniMaxM3SparseParams(disable_index_value=False, use_msa=True)
    with pytest.raises(NotImplementedError, match="disable_index_value"):
        msa_cls(
            layer_idx=0,
            num_heads=8,
            head_dim=128,
            num_kv_heads=1,
            sparse_params=bad_params,
        )


def test_msa_backend_validates_required_dims():
    """Layer construction surfaces head_dim / sparse_index_dim mismatches."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseParams,
    )

    msa_cls = sparse_minimax_m3.get_minimax_m3_msa_attention_backend_cls()
    bad_params_head_dim = MiniMaxM3SparseParams(use_msa=True)
    # head_dim != 128 must trip the precondition (sparse_index_dim is
    # the default 128 in MiniMaxM3SparseParams).
    with pytest.raises(NotImplementedError, match="head_dim"):
        msa_cls(
            layer_idx=0,
            num_heads=8,
            head_dim=64,
            num_kv_heads=1,
            sparse_params=bad_params_head_dim,
        )

    bad_params_topk = MiniMaxM3SparseParams(use_msa=True, topk=8)
    with pytest.raises(NotImplementedError, match="topk"):
        msa_cls(
            layer_idx=0,
            num_heads=8,
            head_dim=128,
            num_kv_heads=1,
            sparse_params=bad_params_topk,
        )


def test_msa_proxy_mqa_registered_in_fmha_libs():
    """The proxy MQA FMHA must live in the standard ``FMHA_LIBS`` registry.

    Keeps the indexer's dispatch path symmetric with the main-attention
    FMHA backends (FlashInfer trtllm-gen, fallback): the same env var
    (``TLLM_FMHA_LIBS``) governs which proxy implementations are
    reachable, and a regression that drops the class from the registry
    breaks the M3 sparse forward at runtime.
    """
    from tensorrt_llm._torch.attention_backend.fmha import (
        FMHA_LIBS,
        IndexerProxyFmha,
        MsaProxyMqaFmha,
    )

    assert "msa_proxy_mqa" in FMHA_LIBS
    assert FMHA_LIBS["msa_proxy_mqa"] is MsaProxyMqaFmha
    assert issubclass(MsaProxyMqaFmha, IndexerProxyFmha)


def test_msa_proxy_mqa_opts_out_of_main_dispatch():
    """is_supported must return False for the main-attention dispatch loop.

    Confirms the indexer-style FMHA never claims work intended for
    main FMHA backends: a regression here would cause TrtllmAttention
    to call ``MsaProxyMqaFmha.forward`` and crash with
    ``NotImplementedError``.
    """
    from tensorrt_llm._torch.attention_backend.fmha import MsaProxyMqaFmha

    fmha = MsaProxyMqaFmha()  # constructed without an owning attn
    # Pass placeholder args -- is_supported should reject regardless.
    assert fmha.is_supported(None, None, None, None, None) is False
    with pytest.raises(NotImplementedError, match="forward_proxy"):
        fmha.forward(None, None, None, None, None)


def test_select_proxy_fmha_class_picks_msa_when_available(monkeypatch):
    """``_select_proxy_fmha_class`` returns MsaProxyMqaFmha when it is_available."""
    from tensorrt_llm._torch.attention_backend.fmha import MsaProxyMqaFmha

    # Force is_available to True and confirm the lookup picks MsaProxyMqaFmha.
    monkeypatch.setattr(MsaProxyMqaFmha, "is_available", classmethod(lambda cls, attn=None: True))
    # Clear the lru_cache on the selector so monkeypatch takes effect.
    msa_backend._select_proxy_fmha_class.cache_clear()
    try:
        cls = msa_backend._select_proxy_fmha_class()
    finally:
        msa_backend._select_proxy_fmha_class.cache_clear()
    assert cls is MsaProxyMqaFmha


def test_msa_sparse_gqa_registered_in_fmha_libs():
    """The block-sparse main FMHA must live in the standard ``FMHA_LIBS`` registry.

    Same rationale as the proxy registration test: ``TLLM_FMHA_LIBS``
    is the single env var governing which sparse-FMHA implementations
    are reachable, and a regression that drops the class breaks the
    M3 sparse forward at runtime.
    """
    from tensorrt_llm._torch.attention_backend.fmha import (
        FMHA_LIBS,
        BlockSparseFmha,
        MsaSparseGqaFmha,
    )

    assert "msa_sparse_gqa" in FMHA_LIBS
    assert FMHA_LIBS["msa_sparse_gqa"] is MsaSparseGqaFmha
    assert issubclass(MsaSparseGqaFmha, BlockSparseFmha)


def test_msa_sparse_gqa_opts_out_of_main_dispatch():
    """is_supported must return False for the main-attention dispatch loop."""
    from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha

    fmha = MsaSparseGqaFmha()  # no owning attn
    assert fmha.is_supported(None, None, None, None, None) is False
    with pytest.raises(NotImplementedError, match="forward_block_sparse"):
        fmha.forward(None, None, None, None, None)


def test_select_block_sparse_fmha_class_picks_msa_when_available(monkeypatch):
    """``_select_block_sparse_fmha_class`` returns MsaSparseGqaFmha when available."""
    from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha

    monkeypatch.setattr(MsaSparseGqaFmha, "is_available", classmethod(lambda cls, attn=None: True))
    msa_backend._select_block_sparse_fmha_class.cache_clear()
    try:
        cls = msa_backend._select_block_sparse_fmha_class()
    finally:
        msa_backend._select_block_sparse_fmha_class.cache_clear()
    assert cls is MsaSparseGqaFmha


def test_select_block_sparse_fmha_class_returns_none_when_unavailable(monkeypatch):
    """``_select_block_sparse_fmha_class`` returns None when no backend is available."""
    from tensorrt_llm._torch.attention_backend.fmha import (
        BlockSparseFmha,
        get_enabled_fmha_lib_classes,
    )

    for cls in get_enabled_fmha_lib_classes():
        if issubclass(cls, BlockSparseFmha):
            monkeypatch.setattr(cls, "is_available", classmethod(lambda cls, attn=None: False))
    msa_backend._select_block_sparse_fmha_class.cache_clear()
    try:
        assert msa_backend._select_block_sparse_fmha_class() is None
    finally:
        msa_backend._select_block_sparse_fmha_class.cache_clear()


def test_select_proxy_fmha_class_returns_none_when_unavailable(monkeypatch):
    """``_select_proxy_fmha_class`` returns None when no backend is_available."""
    # Force every indexer-style class in the registry to report unavailable.
    from tensorrt_llm._torch.attention_backend.fmha import (
        IndexerProxyFmha,
        get_enabled_fmha_lib_classes,
    )

    for cls in get_enabled_fmha_lib_classes():
        if issubclass(cls, IndexerProxyFmha):
            monkeypatch.setattr(cls, "is_available", classmethod(lambda cls, attn=None: False))
    msa_backend._select_proxy_fmha_class.cache_clear()
    try:
        assert msa_backend._select_proxy_fmha_class() is None
    finally:
        msa_backend._select_proxy_fmha_class.cache_clear()


def test_require_msa_module_raises_when_absent(monkeypatch):
    """The lazy import wrapper produces a descriptive error."""

    # Force the import to fail by sabotaging ``sys.modules``.
    import sys

    original = sys.modules.pop("fmha_sm100", None)
    try:
        # Also prevent re-import from a possible install.
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocklist({"fmha_sm100"}, fallback=__import__),
        )
        with pytest.raises(RuntimeError, match="fmha_sm100"):
            msa_backend._require_msa_module()
    finally:
        if original is not None:
            sys.modules["fmha_sm100"] = original


def _make_import_blocklist(blocked, *, fallback):
    """Return an ``__import__`` shim that raises for names in ``blocked``."""

    def _shim(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked or name.split(".")[0] in blocked:
            raise ImportError(f"blocked: {name}")
        return fallback(name, globals, locals, fromlist, level)

    return _shim


# ---------------------------------------------------------------------------
# Live kernel parity (SM100 + fmha_sm100 only)
# ---------------------------------------------------------------------------


def _msa_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major != 10:  # SM100 family
        return False
    try:
        importlib.import_module("fmha_sm100")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _msa_available(), reason="MSA fmha_sm100 + SM100 GPU required")
def test_msa_prefill_runs_end_to_end():
    """Smoke test: the MSA prefill path executes without raising.

    A bit-exact parity check vs the Triton reference is left to the
    Minimax-M3 integration tests because the two paths differ in their
    top-k semantics (union-OR vs amax-then-topk).  This smoke test
    confirms the public entry points compose correctly: cache layout
    adapters, ``fmha_sm100_plan``, ``sparse_topk_select``, and the
    second-stage ``fmha_sm100`` call wire up to a kernel launch and
    return a tensor of the expected shape.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseConfig,
    )

    device = torch.device("cuda")
    page_size = 128
    seq_lens = [256, 384]
    extend_seq_lens = [128, 128]
    num_q_heads, num_kv_heads, head_dim = 8, 1, 128
    num_index_heads, sparse_index_dim = 4, 128

    config = MiniMaxM3SparseConfig(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_index_heads=num_index_heads,
        sparse_index_dim=sparse_index_dim,
        block_size=page_size,
        topk=16,
        init_blocks=0,
        local_blocks=1,
    )

    meta = _build_metadata_for_test(
        is_prefill=True,
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        page_size=page_size,
    )
    # Move tensors to device for the live kernel run.
    meta.req_to_token = meta.req_to_token.to(device)
    meta.slot_ids = meta.slot_ids.to(device)
    meta.seq_lens = meta.seq_lens.to(device)
    meta.prefix_lens = meta.prefix_lens.to(device)
    meta.cu_seqlens_q = meta.cu_seqlens_q.to(device)
    if meta.q_batch_row is not None:
        meta.q_batch_row = meta.q_batch_row.to(device)
    if meta.q_positions is not None:
        meta.q_positions = meta.q_positions.to(device)

    total_q = sum(extend_seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    idx_q = torch.randn(
        total_q, num_index_heads, sparse_index_dim, dtype=torch.bfloat16, device=device
    )
    num_pages = max(((s + page_size - 1) // page_size) for s in seq_lens) * len(seq_lens)
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    idx_k_cache = torch.randn(
        num_pages, page_size, 1, sparse_index_dim, dtype=torch.bfloat16, device=device
    )

    out = msa_backend.minimax_m3_msa_sparse_prefill(
        q,
        k_cache,
        v_cache,
        idx_q,
        idx_k_cache,
        meta,
        config,
    )
    assert out.shape == (total_q, num_q_heads * head_dim)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
