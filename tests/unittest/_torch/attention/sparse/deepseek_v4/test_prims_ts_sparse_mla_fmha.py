# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PrimTS native sparse MLA FMHA library (DeepSeek-V4, SM100)."""

import gc
import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import registry
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
    PrimsTSSparseMlaFmha,
    get_prims_ts_toolchain_unsupported_reason,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._utils import get_sm_version

from ..test_sparse_mla_forward import HAS_FLASH_MLA
from ..test_sparse_mla_forward import (
    test_forward_sparse_mla_unified as _run_unified_sparse_mla_forward,
)

LIB_NAME = "prims_ts_sparse_mla"
LATENT = 512
ROPE = 64
BLOCK = 128


@pytest.fixture(autouse=True)
def _strict_precompile_workers(monkeypatch):
    """A precompile worker failure is an error here, not a logged fallback."""
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import _workspace_size

    monkeypatch.setenv("TLLM_PRIMTS_WARM_STRICT", "1")
    _workspace_size.cache_clear()
    yield
    _workspace_size.cache_clear()


pytestmark = pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="PrimTS sparse MLA requires SM100 or SM103"
)


def _skip_unless_toolchain() -> None:
    """Skip the tests that need the library to actually serve a batch."""
    if get_sm_version() != 100:
        pytest.skip("PrimTS sparse MLA serves SM100 only (SM103 is not yet qualified)")
    reason = get_prims_ts_toolchain_unsupported_reason()
    if reason is not None:
        pytest.skip(reason)


def test_registry_entry_is_default_before_fallback(monkeypatch) -> None:
    """The library ships enabled and ahead of the TRTLLM-Gen fallback.

    Its ``is_available`` restricts it to DeepSeek-V4 on SM100, so being
    in the default set changes nothing for other models or architectures.
    """
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    assert registry.FMHA_LIBS[LIB_NAME] is PrimsTSSparseMlaFmha
    assert LIB_NAME in registry.DEFAULT_FMHA_LIBS
    names = tuple(registry.FMHA_LIBS)
    assert names.index(LIB_NAME) < names.index("flashinfer_trtllm_gen")
    assert names.index(LIB_NAME) < names.index("fallback")
    assert PrimsTSSparseMlaFmha in registry.get_enabled_fmha_lib_classes()
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"-{LIB_NAME}")
    assert PrimsTSSparseMlaFmha not in registry.get_enabled_fmha_lib_classes()


def test_sm103_stays_on_the_fallback_until_qualified(monkeypatch) -> None:
    """B300/GB300 report an SM reason; SM100 proceeds to the shape and toolchain checks.

    The vendored kernels admit SM103 through their architecture guard, but they
    are signed off on SM100 only, so the adapter keeps SM103 on TRTLLM-Gen.
    """
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod

    attn = SimpleNamespace(
        use_fp8_ds_mla=False,
        kv_cache_dtype="auto",
        _uses_nvfp4_compress=False,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
        head_dim=512,
        num_heads=16,
    )
    monkeypatch.setattr(mod, "get_sm_version", lambda: 103)
    assert "SM103" in PrimsTSSparseMlaFmha._unavailable_reason(attn)
    monkeypatch.setattr(mod, "get_sm_version", lambda: 90)
    assert "requires SM100" in PrimsTSSparseMlaFmha._unavailable_reason(attn)
    monkeypatch.setattr(mod, "get_sm_version", lambda: 100)
    reason = PrimsTSSparseMlaFmha._unavailable_reason(attn)
    assert reason is None or reason == get_prims_ts_toolchain_unsupported_reason()


def test_softmax_scale_follows_qk_head_dim() -> None:
    """The scale uses ``qk_nope + qk_rope`` like attentionOp.cpp, not the 512 latent.

    DeepSeek-V4 checkpoints have 448 + 64 = 512, so only a narrower nope width
    (the 128 + 64 scenario of ``test_deepseek_v4_sparse_mla``) tells the two apart.
    """
    attn = SimpleNamespace(qk_nope_head_dim=128, qk_rope_head_dim=64, q_scaling=0.5336)
    expected = 1.0 / (0.5336 * math.sqrt(192.0))
    assert PrimsTSSparseMlaFmha._softmax_scale(attn) == pytest.approx(expected, rel=1e-12)
    attn = SimpleNamespace(qk_nope_head_dim=448, qk_rope_head_dim=64, q_scaling=1.0)
    assert PrimsTSSparseMlaFmha._softmax_scale(attn) == pytest.approx(LATENT**-0.5, rel=1e-12)


def test_pool_view_cache_follows_the_owning_manager() -> None:
    """A new cache manager gets fresh pool views, even at the old manager's address.

    The views wrap the raw pool base pointer, so a view cached for a freed
    manager would alias released memory. The cache is owned through a weak
    reference and dropped whenever the owner changes.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )

    class _Owner:  # weak-referenceable stand-ins for TrtllmAttention / the manager
        pass

    def make_manager(rows: int) -> _Owner:
        manager = _Owner()
        manager.pool = torch.empty(rows, 1, LATENT, dtype=torch.bfloat16, device="cuda")
        manager.swa_pool_ptr = manager.pool.data_ptr()
        manager.compress_pool_ptrs = {}
        manager.pp_layers = [0]
        manager._compress_ratios = {0: 4}
        manager.dtype = torch.bfloat16
        manager.get_buffers = lambda layer, attn_type: manager.pool
        return manager

    attn = _Owner()
    fmha = PrimsTSSparseMlaFmha(attn)
    first = make_manager(8)
    view = fmha._pool_view(first, DeepseekV4AttentionType.SWA, 1)
    assert view.data_ptr() == first.swa_pool_ptr
    assert tuple(view.shape) == (8, 1, LATENT)
    assert fmha._pool_view(first, DeepseekV4AttentionType.SWA, 1) is view

    second = make_manager(16)
    other = fmha._pool_view(second, DeepseekV4AttentionType.SWA, 1)
    assert other is not view
    assert other.data_ptr() == second.swa_pool_ptr
    assert tuple(other.shape) == (16, 1, LATENT)
    assert fmha._pool_views_owner() is second

    del second
    gc.collect()
    assert fmha._pool_views_owner() is None
    third = make_manager(4)
    assert tuple(fmha._pool_view(third, DeepseekV4AttentionType.SWA, 1).shape) == (4, 1, LATENT)


def test_plans_are_shared_through_the_metadata_cache(monkeypatch) -> None:
    """Layers with one static profile share one plan via ``metadata.fmha_plan_caches``.

    The plan owns the kernel workspace, so caching it per layer would keep one
    workspace per layer per CUDA-graph geometry; the metadata cache (the
    convention of the block-sparse and FlashInfer wrappers) keeps one per
    profile, and layers sharing a metadata object run serially.
    """
    import tensorrt_llm._torch.attention.backends.prims_ts.mla_decode as mla_decode

    planned = []

    class _FakeWrapper:
        def __init__(self, workspace_buffer=None):
            self.workspace_buffer = workspace_buffer

        def plan(self, device, rows, heads, **kwargs):
            planned.append((device, rows, heads, kwargs, self.workspace_buffer))

    def fake_resolve(device, rows, heads, **kwargs):
        return 1024 * (1 + kwargs["max_extra_topk"] // 256)

    monkeypatch.setattr(mla_decode, "BatchSparseMLADecodePagedTSWrapper", _FakeWrapper)
    monkeypatch.setattr(mla_decode, "get_prims_ts_sparse_mla_decode_workspace_size", fake_resolve)

    class _Attn:  # weak-referenceable stand-in for TrtllmAttention
        num_heads = 16
        sparse_attention_config = SimpleNamespace(
            window_size=128, enable_kv_cache_offload=False, enable_heuristic_topk=False
        )

    attn_a, attn_b = _Attn(), _Attn()
    layer_a, layer_b = PrimsTSSparseMlaFmha(attn_a), PrimsTSSparseMlaFmha(attn_b)
    metadata = SimpleNamespace(fmha_plan_caches={})
    profile = dict(
        metadata=metadata,
        device=torch.device("cuda"),
        rows=4,
        extra=256,
        q_dtype=torch.bfloat16,
        has_sinks=False,
        fused_groups=None,
    )
    plan_a = layer_a._get_plan(**profile)
    plan_b = layer_b._get_plan(**profile)
    assert plan_a is plan_b
    assert len(planned) == 1
    assert planned[0][3]["max_topk"] == 128 and planned[0][3]["max_extra_topk"] == 256
    # DeepSeek-V4's index producers emit valid prefixes, so the plan promises
    # them unless KV-cache offload (which masks non-resident pages to -1 in
    # place) or heuristic top-k is on; see test_plan_keeps_the_hole_checking_path.
    assert planned[0][3]["assume_valid_prefix"] is True
    cache = metadata.fmha_plan_caches[PrimsTSSparseMlaFmha.PLAN_CACHE_KEY]
    assert list(cache.values()) == [plan_a]
    workspace = metadata.fmha_plan_caches[PrimsTSSparseMlaFmha.WORKSPACE_CACHE_KEY]["buffer"]
    assert planned[0][4] is workspace and workspace.numel() == 2048

    # A smaller profile reuses the shared workspace; a larger one grows it.
    smaller = layer_a._get_plan(**{**profile, "extra": 0})
    assert smaller is not plan_a and planned[1][4] is workspace
    larger = layer_b._get_plan(**{**profile, "extra": 512})
    grown = metadata.fmha_plan_caches[PrimsTSSparseMlaFmha.WORKSPACE_CACHE_KEY]["buffer"]
    assert larger is not plan_a and planned[2][4] is grown and grown.numel() == 3072
    assert len(planned) == 3 and len(cache) == 3


@pytest.mark.parametrize(
    "compress_ratio, offload, has_arena, rejected",
    [
        (4, True, True, False),
        (4, True, False, True),
        (4, False, False, False),
        (128, True, False, False),
    ],
)
def test_offloaded_compressed_rows_need_their_arena(
    compress_ratio, offload, has_arena, rejected
) -> None:
    """An offloaded ratio-4 layer is served from its sparse cache arena, so it is
    rejected only when the offload state has no arena for it. Other layers keep
    their compress pool whether or not offload is on."""

    class _Attn:
        pass

    attn = _Attn()
    attn.compress_ratio = compress_ratio
    attn.layer_idx = 3
    library = PrimsTSSparseMlaFmha(attn)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(swa_pool_ptr=0, get_sparse_compress_buffers=None),
        sparse_offload_state=(
            SimpleNamespace(layers={3: object()} if has_arena else {}) if offload else None
        ),
        # Rejected by the next check, so a layer that passes the arena check stops there.
        beam_width=2,
    )
    reason = library._unsupported_reason(
        SimpleNamespace(device=torch.device("cuda"), ndim=2),
        None,
        None,
        metadata,
        SimpleNamespace(attention_input_type=AttentionInputType.generation_only),
    )
    if rejected:
        assert reason == "the offloaded layer has no sparse cache arena."
    else:
        assert reason == "beam search is not supported."


@pytest.mark.parametrize("offload, heuristic_topk", [(True, False), (False, True)])
def test_plan_keeps_the_hole_checking_path(monkeypatch, offload, heuristic_topk) -> None:
    """Offload (pages masked to -1 in place) and the heuristic GVR top-k (pads from its
    own candidate count) can leave -1 inside a live prefix, so no prefix promise."""
    import tensorrt_llm._torch.attention.backends.prims_ts.mla_decode as mla_decode

    planned = []

    class _FakeWrapper:
        def __init__(self, workspace_buffer=None):
            pass

        def plan(self, device, rows, heads, **kwargs):
            planned.append(kwargs)

    monkeypatch.setattr(mla_decode, "BatchSparseMLADecodePagedTSWrapper", _FakeWrapper)
    monkeypatch.setattr(
        mla_decode, "get_prims_ts_sparse_mla_decode_workspace_size", lambda *a, **k: 1024
    )

    class _Attn:
        num_heads = 16
        sparse_attention_config = SimpleNamespace(
            window_size=128, enable_kv_cache_offload=offload, enable_heuristic_topk=heuristic_topk
        )

    attn = _Attn()  # the library only holds a weak reference to its attention module
    PrimsTSSparseMlaFmha(attn)._get_plan(
        metadata=SimpleNamespace(fmha_plan_caches={}),
        device=torch.device("cuda"),
        rows=4,
        extra=256,
        q_dtype=torch.bfloat16,
        has_sinks=False,
        fused_groups=None,
    )
    assert planned[0]["assume_valid_prefix"] is False


def _rope_table(max_positions: int, device: torch.device) -> torch.Tensor:
    """Forward [cos(32), sin(32)] rows, the layout both kernels consume."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE)
    )
    angles = torch.outer(torch.arange(max_positions, dtype=torch.float32, device=device), inv_freq)
    return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()


def _random_indices(rows: int, capacity: int, pool_rows: int, device: torch.device):
    lengths = torch.randint(1, capacity + 1, (rows,), dtype=torch.int32, device=device)
    indices = torch.randint(0, pool_rows, (rows, capacity), dtype=torch.int32, device=device)
    inactive = torch.arange(capacity, device=device)[None, :] >= lengths[:, None]
    indices[inactive] = -1
    return indices, lengths


def _two_source_case(rows: int, heads: int, topk: int, extra_topk: int, device, dtype):
    """Pools, query and two index sets for one two-source BF16/FP8 attention."""
    pool_rows, extra_rows = 2048, 1024
    pool = torch.randn(pool_rows, 1, LATENT, dtype=torch.bfloat16, device=device).to(dtype)
    extra_pool = torch.randn(extra_rows, 1, LATENT, dtype=torch.bfloat16, device=device).to(dtype)
    query = (torch.randn(rows, 1, heads, LATENT, dtype=torch.bfloat16, device=device) * 0.5).to(
        dtype
    )
    sets = []
    for _ in range(2):
        indices, lengths = _random_indices(rows, topk, pool_rows, device)
        extra_indices, extra_lengths = _random_indices(rows, extra_topk, extra_rows, device)
        sets.append((indices, lengths, extra_indices, extra_lengths))
    return pool, extra_pool, query, sets


class _OffloadAttn:
    """A ratio-4 BF16 DeepSeek-V4 layer, as much of it as the adapter's forward reads."""

    num_heads = 16
    compress_ratio = 4
    layer_idx = 0
    qk_nope_head_dim = 448
    qk_rope_head_dim = ROPE
    q_scaling = 1.0

    def __init__(self, offload: bool):
        self.sparse_attention_config = SimpleNamespace(
            window_size=BLOCK, enable_kv_cache_offload=offload, enable_heuristic_topk=False
        )


class _WeakrefableNamespace(SimpleNamespace):
    """A manager stand-in; the adapter's pool-view cache holds its owner weakly."""


def _run_adapter(attn, manager, query, indices, lengths, extra_lengths, offload_layers):
    """Run the adapter's forward on one generation-only BF16 batch and return the output."""
    rows, heads = query.shape[0], attn.num_heads
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        num_ctx_tokens=0,
        num_tokens=rows,
        fmha_plan_caches={},
        token_positions_cuda=torch.zeros(rows, dtype=torch.int32, device=query.device),
        sparse_mla_swa_lens=lengths,
        sparse_mla_extra_lens={attn.compress_ratio: extra_lengths},
        max_compressed_indices={attn.compress_ratio: indices.shape[1] - BLOCK},
        sparse_offload_state=(
            SimpleNamespace(layers=offload_layers) if offload_layers is not None else None
        ),
    )
    output = torch.empty(rows, heads * LATENT, dtype=torch.bfloat16, device=query.device)
    fwd = SimpleNamespace(
        sparse_runtime_params=SimpleNamespace(sparse_attn_indices=indices),
        output=output,
        enable_dsv4_epilogue_fusion=False,
        quant_q_buffer=None,
        attention_sinks=None,
        output_sf=None,
        out_scale=None,
        dsv4_inv_rope_cos_sin_cache=None,
    )
    PrimsTSSparseMlaFmha(attn).forward(query, None, None, metadata, fwd)
    torch.cuda.synchronize()
    return output


def test_offloaded_layer_reads_the_sparse_cache_arena(monkeypatch):
    """With offload, the compressed slots are rows of the layer's sparse cache arena.

    The same compressed rows are scattered into an arena whose other rows are
    noise, and the indices are remapped to their arena rows. PrimTS must read the
    arena through the manager's view and match the same attention over the
    compress pool. One live arena slot is a -1 hole, which offload produces and
    the kernel must mask; the reference drops that slot instead.
    """
    _skip_unless_toolchain()
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )

    # Request validation is covered elsewhere; this test
    # pins down which pool the compressed indices address.
    monkeypatch.setattr(PrimsTSSparseMlaFmha, "_check_request", lambda *args: None)

    torch.manual_seed(11)
    device = torch.device("cuda")
    rows, extra_topk, pool_rows, arena_rows = 6, 256, 1024, 1536
    heads = _OffloadAttn.num_heads
    swa = torch.randn(2048, 1, LATENT, dtype=torch.bfloat16, device=device)
    compress = torch.randn(pool_rows, 1, LATENT, dtype=torch.bfloat16, device=device)
    # Noise an order of magnitude larger than the real rows, so reading the
    # wrong arena row cannot go unnoticed.
    arena = torch.randn(arena_rows, 1, LATENT, dtype=torch.bfloat16, device=device) * 8
    arena_row_of = torch.randperm(arena_rows, device=device)[:pool_rows].to(torch.int32)
    arena[arena_row_of.long()] = compress
    query = torch.randn(rows, heads * LATENT, dtype=torch.bfloat16, device=device) * 0.5

    swa_indices, lengths = _random_indices(rows, BLOCK, swa.shape[0], device)
    extra_indices, extra_lengths = _random_indices(rows, extra_topk, pool_rows, device)
    extra_lengths.clamp_(min=2)
    extra_indices[:, :2] = torch.randint(0, pool_rows, (rows, 2), dtype=torch.int32, device=device)

    manager = _WeakrefableNamespace(
        dtype=torch.bfloat16,
        swa_pool_ptr=swa.data_ptr(),
        compress_pool_ptrs={4: compress.data_ptr()},
        pp_layers=[0],
        _compress_ratios={0: 4},
        get_buffers=lambda layer, attn_type: (
            swa if attn_type == DeepseekV4AttentionType.SWA else compress
        ),
        get_sparse_compress_buffers=lambda layer: (arena, None),
    )

    # Arena run: remap every live compressed slot, then punch a hole in slot 0
    # of row 0, which stays inside its live prefix.
    arena_extra = torch.where(
        extra_indices >= 0, arena_row_of[extra_indices.clamp(min=0).long()], extra_indices
    )
    arena_extra[0, 0] = -1
    offload_attn = _OffloadAttn(offload=True)
    got = _run_adapter(
        offload_attn,
        manager,
        query,
        torch.cat([swa_indices, arena_extra], dim=1).contiguous(),
        lengths,
        extra_lengths,
        offload_layers={0: object()},
    )

    # Reference: the compress pool, with row 0's hole removed from its prefix.
    ref_extra = extra_indices.clone()
    ref_extra[0, :-1] = extra_indices[0, 1:].clone()
    ref_extra[0, -1] = -1
    ref_extra_lengths = extra_lengths.clone()
    ref_extra_lengths[0] -= 1
    pool_attn = _OffloadAttn(offload=False)
    ref = _run_adapter(
        pool_attn,
        manager,
        query,
        torch.cat([swa_indices, ref_extra], dim=1).contiguous(),
        lengths,
        ref_extra_lengths,
        offload_layers=None,
    )
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_reloaded_kernel_accepts_the_same_launch():
    """A kernel reloaded from the persistent compile cache runs the adapter's launch.

    CI processes always compile cold, while every later process (and
    production restarts) reload the exported TVM-FFI kernels, whose positional
    entry has no Python defaults. Drop the in-process kernel memo after the
    first run so the second plan takes the reload path, and require identical
    output.
    """
    _skip_unless_toolchain()
    from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

    torch.manual_seed(1)
    device = torch.device("cuda")
    rows, heads, topk, extra_topk = 5, 64, 128, 256
    pool, extra_pool, query, sets = _two_source_case(
        rows, heads, topk, extra_topk, device, torch.bfloat16
    )
    indices, lengths, extra_indices, extra_lengths = sets[0]

    def run_fresh_plan() -> torch.Tensor:
        wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper()
        kwargs = dict(max_topk=topk, max_extra_topk=extra_topk)
        cache_info = mla_decode._get_compiled_mla_decode.cache_info()
        wrapper.plan(device, rows, heads, compile=False, **kwargs)
        spec = wrapper._state["compile_spec"]
        assert wrapper._state["compiled"] is None
        assert mla_decode._get_compiled_mla_decode.cache_info() == cache_info
        with pytest.raises(RuntimeError, match="plan\\(compile=True\\)"):
            wrapper.run(None, None, None, None)
        wrapper.plan(device, rows, heads, **kwargs)
        compiled = wrapper._state["compiled"]
        assert compiled is not None and wrapper._state["compile_spec"] == spec
        # A planning-only replan must not retain an old executable kernel.
        wrapper.plan(device, rows, heads, compile=False, **kwargs)
        assert wrapper._state["compiled"] is None
        wrapper.plan(device, rows, heads, **kwargs)
        assert wrapper._state["compiled"] is compiled
        out = torch.empty(rows, 1, heads, LATENT, dtype=torch.bfloat16, device=device)
        wrapper.run(
            query,
            pool,
            indices,
            lengths,
            extra_pool,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            softmax_scale=LATENT**-0.5,
            output_scale=1.0,
            out=out,
        )
        torch.cuda.synchronize()
        return out

    first = run_fresh_plan()
    mla_decode._get_compiled_mla_decode.cache_clear()
    second = run_fresh_plan()
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_graph_replay_follows_new_indices_and_lengths():
    """A captured run reads the live index and length buffers on replay.

    Warm up eagerly, capture one run with ``validate=False``, swap in a second
    index set and compare the replayed output with an eager run on that set.
    """
    _skip_unless_toolchain()
    from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

    torch.manual_seed(2)
    device = torch.device("cuda")
    rows, heads, topk, extra_topk = 8, 64, 128, 256
    pool, extra_pool, query, sets = _two_source_case(
        rows, heads, topk, extra_topk, device, torch.bfloat16
    )
    buffers = [tensor.clone() for tensor in sets[0]]  # fixed-address launch inputs
    indices, lengths, extra_indices, extra_lengths = buffers
    out = torch.empty(rows, 1, heads, LATENT, dtype=torch.bfloat16, device=device)
    wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(device, rows, heads, max_topk=topk, max_extra_topk=extra_topk)

    def launch(validate: bool) -> None:
        wrapper.run(
            query,
            pool,
            indices,
            lengths,
            extra_pool,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            softmax_scale=LATENT**-0.5,
            output_scale=1.0,
            out=out,
            validate=validate,
        )

    launch(validate=True)  # eager warmup on set 0, also prepares the scalar cache
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(validate=False)
    for buffer, source in zip(buffers, sets[1]):
        buffer.copy_(source)
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    launch(validate=True)  # eager reference on set 1 through the same plan
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, out, rtol=0, atol=0)
    assert not torch.equal(replayed, torch.zeros_like(replayed))


def test_request_check_runs_once_per_geometry(monkeypatch) -> None:
    """The full request re-check runs on the first call and again only when the
    request geometry changes; repeated steps of one shape skip it."""
    adapter = PrimsTSSparseMlaFmha.__new__(PrimsTSSparseMlaFmha)
    adapter._checked_key = None
    reasons = []

    def fake_reason(self, q, k, v, metadata, fwd):
        reasons.append(1)
        return None

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "_unsupported_reason", fake_reason)

    def make(rows: int, width: int = 640, fusion: bool = False, ctx: int = 0):
        metadata = SimpleNamespace(num_tokens=ctx + rows, num_ctx_tokens=ctx)
        indices = torch.zeros(rows, width, dtype=torch.int32)
        output = torch.empty(rows, 64 * LATENT, dtype=torch.bfloat16)
        fwd = SimpleNamespace(
            sparse_runtime_params=SimpleNamespace(sparse_attn_indices=indices),
            output=output,
            enable_dsv4_epilogue_fusion=fusion,
            quant_q_buffer=None,
            attention_sinks=None,
            output_sf=None,
            out_scale=None,
            dsv4_inv_rope_cos_sin_cache=None,
        )
        return torch.empty(rows, 64 * LATENT, dtype=torch.bfloat16), metadata, fwd

    q, metadata, fwd = make(8)
    adapter._check_request(q, None, None, metadata, fwd)
    adapter._check_request(q, None, None, metadata, fwd)
    adapter._check_request(q, None, None, metadata, fwd)
    assert len(reasons) == 1
    q2, _, fwd2 = make(16)  # new row count on the same metadata object
    adapter._check_request(q2, None, None, metadata, fwd2)
    assert len(reasons) == 2
    adapter._check_request(q2, None, None, metadata, fwd2)
    assert len(reasons) == 2
    _, _, fwd3 = make(16, fusion=True)
    adapter._check_request(q2, None, None, metadata, fwd3)
    assert len(reasons) == 3
    adapter._check_request(q2, None, None, metadata, fwd3)
    assert len(reasons) == 3
    _, metadata4, fwd4 = make(16, fusion=True)  # only the metadata object differs
    adapter._check_request(q2, None, None, metadata4, fwd4)
    assert len(reasons) == 4

    def rejecting_reason(self, q, k, v, metadata, fwd):
        return "nope"

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "_unsupported_reason", rejecting_reason)
    adapter._check_request(q2, None, None, metadata4, fwd4)  # same geometry: skipped
    q5, metadata5, fwd5 = make(4)
    with pytest.raises(RuntimeError, match="nope"):
        adapter._check_request(q5, None, None, metadata5, fwd5)


def test_sinks_pass_through_or_are_prepared_once_per_layer() -> None:
    """FP32 per-head sinks are passed through at a stable address; other
    dtypes are converted once per source tensor, so the kernel key keyed by the
    sinks address stays stable for every layer served by one adapter."""
    from tensorrt_llm._torch.attention.backends.fmha import prims_ts_sparse_mla as mod

    adapter = PrimsTSSparseMlaFmha.__new__(PrimsTSSparseMlaFmha)
    adapter._sinks_cache = {}
    fp32 = torch.nn.Parameter(torch.randn(64, device="cuda"), requires_grad=False)
    assert adapter._sinks(SimpleNamespace(attention_sinks=fp32)) is fp32
    assert not adapter._sinks_cache
    bf16_a = torch.randn(64, device="cuda").to(torch.bfloat16)
    bf16_b = torch.randn(64, device="cuda").to(torch.bfloat16)
    prep_a = adapter._sinks(SimpleNamespace(attention_sinks=bf16_a))
    prep_b = adapter._sinks(SimpleNamespace(attention_sinks=bf16_b))
    assert prep_a.dtype == torch.float32 and prep_a is not prep_b
    assert adapter._sinks(SimpleNamespace(attention_sinks=bf16_a)) is prep_a
    assert adapter._sinks(SimpleNamespace(attention_sinks=bf16_b)) is prep_b
    torch.testing.assert_close(prep_a, bf16_a.float())
    assert adapter._sinks(SimpleNamespace(attention_sinks=None)) is None
    assert mod._SINKS_CACHE_LIMIT >= 61


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "rows,heads,groups,topk,extra_topk",
    [
        (1, 128, 16, 128, 0),
        (5, 128, 16, 128, 512),
        (64, 128, 16, 128, 512),
        (16, 64, 8, 128, 256),
        (3, 64, 8, 128, 0),
    ],
)
def test_fused_output_matches_trtllm_inverse_rope_quant(
    dtype, rows, heads, groups, topk, extra_topk
):
    """The kernel's fused epilogue must reproduce TRT-LLM's inverse-RoPE + FP8 quant.

    Reference: the same attention with a BF16 output, followed by
    ``trtllm.fused_inv_rope_fp8_quant_vllm_port`` (the op the unfused DSv4
    path runs). The fused kernel quantizes from FP32 accumulators instead of
    the BF16 output, so compare dequantized values up to FP8 rounding.
    """
    _skip_unless_toolchain()
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    pool_rows, extra_rows, max_positions = 4096, 2048, 4096

    def make_pool(n):
        return (torch.randn(n, 1, LATENT, dtype=torch.bfloat16, device=device)).to(dtype)

    pool = make_pool(pool_rows)
    extra_pool = make_pool(extra_rows) if extra_topk else None
    indices, lengths = _random_indices(rows, topk, pool_rows, device)
    extra_indices, extra_lengths = (
        (None, None) if extra_topk == 0 else _random_indices(rows, extra_topk, extra_rows, device)
    )
    query = (torch.randn(rows, 1, heads, LATENT, dtype=torch.bfloat16, device=device) * 0.5).to(
        dtype
    )
    positions = torch.randint(0, max_positions, (rows,), dtype=torch.int32, device=device)
    cos_sin = _rope_table(max_positions, device)

    softmax_scale = LATENT**-0.5
    descale = 0.75 if dtype == torch.float8_e4m3fn else 1.0
    # The kernel takes the combined scales (attention x Q descale x KV descale
    # on the softmax, the KV descale on the output), as the adapter passes them.
    run_scales = dict(softmax_scale=softmax_scale * descale * descale, output_scale=descale)
    sources = dict(extra_indices=extra_indices, extra_lengths=extra_lengths) if extra_topk else {}
    plan_args = dict(max_topk=topk, max_extra_topk=extra_topk, max_q_len=1, q_dtype=dtype)

    reference = BatchSparseMLADecodePagedTSWrapper()
    reference.plan(device, rows, heads, **plan_args)
    out_bf16 = reference.run(query, pool, indices, lengths, extra_pool, **sources, **run_scales)
    heads_per_group = heads // groups
    ref_fp8, ref_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
        out_bf16.view(rows, heads, LATENT),
        positions,
        cos_sin,
        groups,
        heads_per_group,
        LATENT - ROPE,
        ROPE,
        BLOCK,
        False,
    )

    fused = BatchSparseMLADecodePagedTSWrapper()
    fused.plan(
        device,
        rows,
        heads,
        output_quant_block_size=BLOCK,
        out_scale_format="fp32",
        num_output_groups=groups,
        **plan_args,
    )
    channels = heads_per_group * LATENT
    padded_rows = (rows + 3) // 4 * 4
    # The exact buffers TRT-LLM's fused DSv4 epilogue allocates.
    out_buf = torch.empty((groups, rows, channels), dtype=torch.float8_e4m3fn, device=device)
    scale_buf = torch.empty(
        (groups, channels // BLOCK, padded_rows), dtype=torch.float32, device=device
    )
    fused_out, fused_scale = fused.run(
        query,
        pool,
        indices,
        lengths,
        extra_pool,
        **sources,
        out=out_buf.transpose(0, 1),
        out_scale=scale_buf.permute(2, 0, 1)[:rows],
        token_positions=positions.view(rows, 1),
        cos_sin_cache=cos_sin,
        **run_scales,
    )
    assert fused_out.data_ptr() == out_buf.data_ptr()
    assert fused_scale.data_ptr() == scale_buf.data_ptr()
    assert ref_fp8.shape == out_buf.shape and ref_scale.shape == scale_buf.shape

    ref_scales = ref_scale[..., :rows].permute(0, 2, 1)  # [G, T, blocks]
    got_scales = scale_buf[..., :rows].permute(0, 2, 1)
    torch.testing.assert_close(got_scales, ref_scales, rtol=2e-2, atol=1e-6)

    def dequant(values, scales):
        blocks = values.float().view(groups, rows, channels // BLOCK, BLOCK)
        return (blocks * scales[..., None]).flatten(-2)

    ref = dequant(ref_fp8, ref_scales)
    got = dequant(out_buf, got_scales)
    # One E4M3 ulp is 1/8 of the magnitude. The reference rotates the
    # BF16-rounded output, so near-cancelling RoPE channels (D[448:512]) carry
    # an absolute error of a few BF16 ulps of their O(1) inputs, which the
    # kernel (rotating FP32 accumulators) does not have; that shows up as up to
    # about one quantization step of the block. Allow two steps there and half
    # a step elsewhere, on top of two relative ulps.
    block_scales = ref_scales.repeat_interleave(BLOCK, dim=-1)
    is_rope = torch.zeros(channels, dtype=torch.bool, device=device)
    is_rope.view(-1, LATENT)[:, LATENT - ROPE :] = True
    step_allowance = torch.where(is_rope, 2.0, 0.5) * block_scales
    ulp = ref.abs() * 0.125 + block_scales * 2.0**-6
    diff = (got - ref).abs()
    assert (diff <= 2.0 * ulp + step_allowance + 1e-6).all(), (
        f"max fused-vs-reference gap {diff.max().item()}"
    )
    assert diff.mean().item() <= 0.02 * ref.abs().mean().item() + 1e-6


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.parametrize("library", ["prims_ts", "fallback"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "batch_name",
    [
        "small_decode",
        "medium_decode",
        "single_decode",
        "small_mixed",
        "mixed_heavy_decode",
    ],
)
def test_forward_sparse_mla_unified_prims_ts(monkeypatch, batch_name, kv_cache_dtype, library):
    """Run the DeepSeek-V4 module test on the default library set and on the opt-out.

    ``prims_ts``: no ``TLLM_FMHA_LIBS`` override; the PrimTS library is in the
    default set and must win the generation phase on its own. ``fallback``:
    ``TLLM_FMHA_LIBS=-prims_ts_sparse_mla``, the documented opt-out, must hand
    generation back to the TRTLLM-Gen fallback (``FallbackFmha``) with PrimTS
    never entered, so a regression on that path (what SM103, ``fp8_ds_mla`` and
    NVFP4 deployments run) still fails here.
    """
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType

    if library == "prims_ts":
        _skip_unless_toolchain()
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", f"-{LIB_NAME}")
    prims_calls: list = []
    fallback_calls: list = []
    original_forward = PrimsTSSparseMlaFmha.forward
    original_fallback = FallbackFmha.forward

    def counting_forward(self, *args, **kwargs):
        prims_calls.append(1)
        return original_forward(self, *args, **kwargs)

    def counting_fallback(self, q, k, v, metadata, forward_args, *args, **kwargs):
        if (
            metadata.num_generations > 0
            and forward_args.attention_input_type != AttentionInputType.context_only
        ):
            fallback_calls.append(1)
        return original_fallback(self, q, k, v, metadata, forward_args, *args, **kwargs)

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "forward", counting_forward)
    monkeypatch.setattr(FallbackFmha, "forward", counting_fallback)
    _run_unified_sparse_mla_forward(batch_name, kv_cache_dtype, "deepseek_v4")
    if library == "prims_ts":
        assert prims_calls, "the PrimTS sparse MLA FMHA did not serve the generation phase"
    else:
        assert not prims_calls, "TLLM_FMHA_LIBS=-prims_ts_sparse_mla did not disable PrimTS"
        assert fallback_calls, "the TRTLLM-Gen fallback did not serve the generation phase"


@pytest.mark.parametrize(
    "generation_seq_len_q",
    [2, 6, [1, 2, 4]],
    ids=["mtp_next_n_2", "mtp_next_n_6", "ragged_1_2_4"],
)
def test_speculative_decoding_rows_run_on_prims_ts(monkeypatch, generation_seq_len_q):
    """MTP generation (``max_draft_tokens > 0``) on the default library set.

    ``test_deepseek_v4_sparse_mla`` drives the DeepSeek-V4 attention module with
    several query tokens per request (uniform ``next_n`` widths and a ragged
    batch) and checks the output against its reference; here it is run with the
    library default and the spy asserts that PrimTS served those steps with one
    row per query token (``rows = sum(generation_lens)``, not one per request),
    which is the per-token-row contract the kernel is planned for
    (``max_q_len = 1``).
    """
    _skip_unless_toolchain()
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    from . import test_deepseek_v4_sparse_mla as module_test

    context_lengths = [14, 508, 3947]
    generation_lens = (
        generation_seq_len_q
        if isinstance(generation_seq_len_q, list)
        else [generation_seq_len_q] * len(context_lengths)
    )
    rows_seen: list[int] = []
    original_forward = PrimsTSSparseMlaFmha.forward

    def spy_forward(self, q, k, v, metadata, forward_args):
        rows_seen.append(int(q.shape[0]))
        return original_forward(self, q, k, v, metadata, forward_args)

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "forward", spy_forward)
    module_test.test_deepseek_v4_sparse_mla(context_lengths, generation_seq_len_q)
    assert rows_seen, "PrimTS did not serve the speculative-decoding generation steps"
    assert set(rows_seen) == {sum(generation_lens)}, rows_seen
    assert sum(generation_lens) > len(context_lengths)  # more rows than requests: draft tokens


def _dequant_fused(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize a fused DSv4 epilogue buffer.

    ``values`` is ``[G, T, C]`` E4M3 and ``scales`` is ``[G, C // 128, pad4(T)]``
    FP32, exactly as ``_create_dsv4_epilogue_buffers`` allocates them.
    """
    groups, rows, channels = values.shape
    sc = scales[..., :rows].permute(0, 2, 1)  # [G, T, blocks]
    blocks = values.float().view(groups, rows, channels // BLOCK, BLOCK)
    return (blocks * sc[..., None]).flatten(-2), sc


def _assert_fused_close(got, got_sc, ref, ref_sc, channels, device):
    """Same tolerance as the kernel-level fused test: two relative E4M3 ulps plus
    half a quantization step, two steps on the RoPE channels D[448:512]."""
    torch.testing.assert_close(got_sc, ref_sc, rtol=2e-2, atol=1e-6)
    block_scales = ref_sc.repeat_interleave(BLOCK, dim=-1)
    is_rope = torch.zeros(channels, dtype=torch.bool, device=device)
    is_rope.view(-1, LATENT)[:, LATENT - ROPE :] = True
    step_allowance = torch.where(is_rope, 2.0, 0.5) * block_scales
    diff = (got - ref).abs()
    ulp = ref.abs() / 8.0
    assert (diff <= 2.0 * ulp + step_allowance + 1e-6).all(), (
        f"max diff {diff.max().item():.4g} vs allowance "
        f"{(2.0 * ulp + step_allowance).max().item():.4g}"
    )
    assert diff.mean().item() <= 0.02 * ref.abs().mean().item() + 1e-6


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("batch_name", ["small_decode", "medium_decode", "single_decode"])
def test_forward_sparse_mla_unified_fused_epilogue(monkeypatch, batch_name, kv_cache_dtype):
    """Module-level coverage of the fused DSv4 epilogue (production ADP path).

    ``prepare_sparse_attn_outputs`` only hands out the 3D fused buffer under
    attention-DP with 128 heads, an FP8 cache and an FP8 ``o_a_proj``, which the
    module test harness does not configure. The generation entry point
    ``forward_generation_sparse_attn`` takes ``enable_dsv4_epilogue_fusion``
    directly, so wrap the harness's call to it and, on the same prepared cache,
    metadata and inputs, additionally run the fused mode on the default library
    set: PrimTS must serve it with ``forward_args.enable_dsv4_epilogue_fusion``
    set, and its fused output must match the unfused BF16 output pushed through
    ``trtllm.fused_inv_rope_fp8_quant_vllm_port`` (the path the module takes when
    fusion is off). The harness's own reference check still runs on the unfused
    call.

    The TRTLLM-Gen fallback is not run in fused mode here: the module only
    enables the fused epilogue for it at 128 heads, and at the harness's 64
    heads its C++ op aborts the process rather than raising.
    """
    _skip_unless_toolchain()
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)

    from .. import test_sparse_mla_forward as harness

    original_gen = harness.forward_generation_deepseek_v4_mla
    original_forward = PrimsTSSparseMlaFmha.forward
    seen_fused_flags: list[bool] = []
    results: dict[str, object] = {}

    def spy_forward(self, q, k, v, metadata, forward_args):
        seen_fused_flags.append(bool(forward_args.enable_dsv4_epilogue_fusion))
        return original_forward(self, q, k, v, metadata, forward_args)

    _INPLACE_INPUTS = ("q", "compressed_kv", "k_pe", "latent_cache")

    def run_fused(mla, kwargs, pristine):
        # The RoPE/KV-write prologue rotates ``q`` and ``latent_cache`` in
        # place, so every call needs the inputs as they were before the first.
        fused_kwargs = dict(kwargs)
        fused_kwargs.update({k: v.clone() for k, v in pristine.items()})
        fused_kwargs["output"] = None
        fused_kwargs["enable_dsv4_epilogue_fusion"] = True
        return original_gen(mla, **fused_kwargs)

    def wrapped_gen(mla, **kwargs):
        pristine = {k: kwargs[k].clone() for k in _INPLACE_INPUTS if kwargs.get(k) is not None}
        # 1. The harness's unfused call, untouched (it validates against the
        #    Python reference afterwards).
        out = original_gen(mla, **kwargs)
        output = kwargs["output"]
        rows = output.shape[0]
        heads = mla.num_heads_tp
        groups = mla.n_local_groups
        positions = kwargs["position_ids"].view(-1).to(torch.int32)
        cos_sin = mla.inverse_rotary_emb.rotary_cos_sin
        torch.cuda.synchronize()

        # Path B: the standalone inverse-RoPE + FP8 quant op on the BF16 output.
        ref_fp8, ref_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
            output.view(rows, heads, LATENT).clone(),
            positions,
            cos_sin,
            groups,
            heads // groups,
            LATENT - ROPE,
            ROPE,
            BLOCK,
            False,
        )

        # 2. Path A on the default library set: PrimTS serves the fused epilogue.
        #    (The module hands out the 3D buffer only under ADP + 128 heads + FP8,
        #    which is what production throughput configs run.)
        before = len(seen_fused_flags)
        prims_fp8, prims_scale = run_fused(mla, kwargs, pristine)
        torch.cuda.synchronize()
        assert seen_fused_flags[before:] == [True], (
            "the PrimTS sparse MLA FMHA did not serve the fused generation call"
        )

        results.update(
            rows=rows,
            heads=heads,
            groups=groups,
            ref=(ref_fp8.clone(), ref_scale.clone()),
            prims=(prims_fp8.clone(), prims_scale.clone()),
        )
        return out

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "forward", spy_forward)
    monkeypatch.setattr(harness, "forward_generation_deepseek_v4_mla", wrapped_gen)
    _run_unified_sparse_mla_forward(batch_name, kv_cache_dtype, "deepseek_v4")

    assert results, "the generation phase never ran"
    assert seen_fused_flags.count(False) >= 1, "the unfused call did not reach PrimTS"
    device = results["prims"][0].device
    channels = results["heads"] // results["groups"] * LATENT

    ref, ref_sc = _dequant_fused(*results["ref"])
    got, got_sc = _dequant_fused(*results["prims"])
    assert got.shape == ref.shape
    _assert_fused_close(got, got_sc, ref, ref_sc, channels, device)


def test_warmup_collects_only_enabled_layer_profiles(monkeypatch) -> None:
    import builtins
    from unittest.mock import Mock

    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import module
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
        DeepseekV4TrtllmAttention,
    )
    from tensorrt_llm._torch.attention.mla import MLA

    real_import = builtins.__import__

    def no_vendor_import(name, *args, **kwargs):
        assert name != "tensorrt_llm._torch.attention.backends.prims_ts"
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_vendor_import)
        mod._warmup_profiles([], SimpleNamespace())

    layers, owners = [], []
    for enabled, fused, sinks, dtype in (
        (True, True, True, torch.bfloat16),
        (True, False, False, torch.bfloat16),
        (False, False, False, torch.bfloat16),
        (True, False, False, torch.float16),
    ):
        attn = Mock(spec=DeepseekV4TrtllmAttention)
        attn.compress_ratio = 4
        attn.attn_sink = object() if sinks else None
        attn._fmha_manager = SimpleNamespace(
            fmha_libs=[PrimsTSSparseMlaFmha(attn)] if enabled else []
        )
        layer = Mock(spec=MLA)
        layer.mqa, layer.dtype = attn, dtype
        layer.o_a_proj = object()
        layer.n_local_groups, layer.fused = 8, fused
        layers.append(layer)
        owners.append(attn)  # FMHA keeps its attention owner weakly.
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        token_positions_cuda=SimpleNamespace(device=torch.device("cuda", 0)),
        max_compressed_indices={4: 512},
    )
    captured = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(mod.PrimsTSSparseMlaFmha, "_is_fp8_cache", staticmethod(lambda _: True))
    monkeypatch.setattr(module, "supports_dsv4_epilogue_fusion", lambda layer: layer.fused)
    monkeypatch.setattr(mod, "_warmup_profiles", lambda profiles, meta: captured.extend(profiles))
    mod.warmup_prims_ts_sparse_mla(SimpleNamespace(modules=lambda: iter(layers)), metadata)
    assert [(p.extra, p.q_dtype, p.has_sinks, p.fused_groups) for p in captured] == [
        (512, torch.float8_e4m3fn, True, 8),
        (512, torch.float8_e4m3fn, False, None),
    ]
    metadata.is_cuda_graph = True
    with pytest.raises(RuntimeError, match="outside graph capture"):
        mod.warmup_prims_ts_sparse_mla(SimpleNamespace(modules=lambda: iter(layers)), metadata)


def test_warmup_combines_profiles_and_loads_largest_workspace_first(monkeypatch) -> None:
    """One worker group serves all distinct layer profiles before local plan loading."""
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile as pp
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod

    planned, fanned, workspace_calls = [], [], []
    mla_decode = _fake_sparse_plan(monkeypatch, planned)

    def workspace_size(device, rows, heads, **kwargs):
        workspace_calls.append((rows, kwargs["max_extra_topk"]))
        return rows * kwargs["max_extra_topk"]

    monkeypatch.setattr(mla_decode, "get_prims_ts_sparse_mla_decode_workspace_size", workspace_size)
    monkeypatch.setattr(pp, "precompile_payloads", lambda payloads, **kw: fanned.append(payloads))
    attn, profile, eager = _attn_and_profile()
    lib = PrimsTSSparseMlaFmha(attn)
    profiles = [mod._WarmupProfile(lib, extra=extra, **profile) for extra in (32, 512, 32)]
    mod._warmup_profiles(profiles, eager)
    assert len(fanned) == 1
    assert [(p["rows"], p["plan_kwargs"]["max_extra_topk"]) for p in fanned[0]] == [
        (20, 32),
        (15, 32),
        (7, 32),
        (20, 512),
        (15, 512),
        (7, 512),
    ]
    assert planned == [(20, 512), (15, 512), (7, 512), (20, 32), (15, 32), (7, 32)]
    workspace = eager.fmha_plan_caches[lib.WORKSPACE_CACHE_KEY]["buffer"]
    assert workspace.numel() == 20 * 512
    assert lib._get_plan(metadata=eager, rows=15, extra=512, **profile) is not None
    assert len(planned) == 6
    assert len(workspace_calls) == 40  # One public size query per distinct shape.

    # Workspace requirements need not grow with rows within one compile spec.
    monkeypatch.setattr(
        mla_decode,
        "get_prims_ts_sparse_mla_decode_workspace_size",
        lambda device, rows, heads, **kw: 100000 if rows == 6 else rows * 64,
    )
    planned.clear()
    mod._warmup_profiles([mod._WarmupProfile(lib, extra=64, **profile)], eager)
    assert planned == [(6, 64), (20, 64), (15, 64)]
    assert eager.fmha_plan_caches[lib.WORKSPACE_CACHE_KEY]["buffer"].numel() == 100000


def test_warm_set_covers_the_speculative_decoding_rows(monkeypatch) -> None:
    """With MTP the generation step carries ``requests * (1 + draft)`` rows; the warm-up
    plans the largest row count of every variant up to that ceiling (bounded by the
    token budget), so the MTP row counts never compile during the first serving step."""
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod

    planned: list = []
    _fake_sparse_plan(monkeypatch, planned, signature_div=8)
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "0")
    attn, profile, _ = _attn_and_profile()
    lib = PrimsTSSparseMlaFmha(attn)

    mtp = SimpleNamespace(
        fmha_plan_caches={},
        max_num_tokens=256,
        max_num_requests=4,
        max_total_draft_tokens=5,  # DeepSeek-V4 MTP: next_n = 6 rows per request
        mapping=None,
        is_cuda_graph=False,
    )
    mod._warmup_profiles([mod._WarmupProfile(lib, extra=512, **profile)], mtp)
    rows = [r for r, _ in planned]
    assert rows[0] == 4 * (1 + 5), rows
    assert rows == sorted(rows, reverse=True)
    assert set(rows) == {24, 23, 15, 7}  # the largest row count of each variant

    # The step's token budget bounds the rows even when the MTP ceiling is larger.
    planned.clear()
    budget = SimpleNamespace(
        fmha_plan_caches={},
        max_num_tokens=10,
        max_num_requests=4,
        max_total_draft_tokens=5,
        mapping=None,
        is_cuda_graph=False,
    )
    mod._warmup_profiles([mod._WarmupProfile(lib, extra=512, **profile)], budget)
    assert [r for r, _ in planned][0] == 10

    # A later warmup with a larger capacity includes every signature beyond the
    # old 128-row dense scan, including a non-power-of-two maximum.
    planned.clear()
    larger = SimpleNamespace(
        **{**vars(mtp), "max_num_requests": 32, "max_num_tokens": 183, "fmha_plan_caches": {}}
    )
    mod._warmup_profiles([mod._WarmupProfile(lib, extra=512, **profile)], larger)
    assert [row for row, _ in planned] == [183, *range(175, 0, -8)]

    # A metadata class that does not declare the fields fails loudly.
    renamed = SimpleNamespace(
        fmha_plan_caches={}, max_num_tokens=10, max_num_requests=4, is_cuda_graph=False
    )
    with pytest.raises(AttributeError):
        mod._warmup_profiles([mod._WarmupProfile(lib, extra=512, **profile)], renamed)


def test_index_contract_is_validated_once_per_process(monkeypatch) -> None:
    """One validated eager launch per contract (producers + pools); the same contract never
    re-validates, because that synchronizes the host inside the context ramp."""
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod

    monkeypatch.setattr(mod, "_VALIDATED_CONTRACTS", set())
    key = ("cuda:0", 128, 512, 4, False, "torch.bfloat16", 1, 4096, 2, 4096)
    assert PrimsTSSparseMlaFmha._claim_contract_validation(key) is True
    assert PrimsTSSparseMlaFmha._claim_contract_validation(key) is False
    other_pools = key[:6] + (3, 4096, 4, 4096)
    assert PrimsTSSparseMlaFmha._claim_contract_validation(other_pools) is True


def _fake_sparse_plan(monkeypatch, planned, *, signature_div=8):
    """Record executable plans and expose specs from planning-only calls."""
    import tensorrt_llm._torch.attention.backends.prims_ts.mla_decode as mla_decode

    class _FakeWrapper:
        discovery_rows = []

        def __init__(self, workspace_buffer=None):
            self._state = None

        def plan(self, device, rows, heads, *, compile=True, **kwargs):
            self._state = {
                "compile_spec": (rows // signature_div, kwargs["max_extra_topk"]),
                "compiled": "kernel" if compile else None,
            }
            if compile:
                planned.append((rows, kwargs["max_extra_topk"]))
            else:
                self.discovery_rows.append(rows)

    monkeypatch.setattr(mla_decode, "BatchSparseMLADecodePagedTSWrapper", _FakeWrapper)
    monkeypatch.setattr(
        mla_decode, "get_prims_ts_sparse_mla_decode_workspace_size", lambda *a, **kw: 1024
    )
    return mla_decode


def _attn_and_profile():
    class _Attn:
        num_heads = 16
        sparse_attention_config = SimpleNamespace(
            window_size=128, enable_kv_cache_offload=False, enable_heuristic_topk=False
        )

    attn = _Attn()  # the library holds a weak reference to its owner
    profile = dict(
        device=torch.device("cuda", 0), q_dtype=torch.bfloat16, has_sinks=False, fused_groups=None
    )
    eager = SimpleNamespace(
        fmha_plan_caches={},
        max_num_tokens=20,
        max_num_requests=20,
        max_total_draft_tokens=None,
        mapping=None,
        is_cuda_graph=False,
    )
    return attn, profile, eager


def test_warmup_loads_after_workers_and_falls_back_on_failure(monkeypatch) -> None:
    """Workers see only representative plans; their failure falls back to the parent."""
    import json

    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile as pp
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod

    planned, fanned = [], []
    mla_decode = _fake_sparse_plan(monkeypatch, planned)
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "2")
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    def fake_workers(payloads, workers):
        assert not planned  # Discovery must not compile serially in the parent.
        fanned.append((list(payloads), workers))
        return True

    monkeypatch.setattr(pp, "run_precompile_workers", fake_workers)
    attn, profile, eager = _attn_and_profile()
    lib = PrimsTSSparseMlaFmha(attn)
    mod._warmup_profiles([mod._WarmupProfile(lib, extra=32, **profile)], eager)
    assert planned == [(20, 32), (15, 32), (7, 32)]
    assert mla_decode.BatchSparseMLADecodePagedTSWrapper.discovery_rows == list(range(1, 21))
    assert len(fanned) == 1
    payloads, workers = fanned[0]
    assert workers == 2
    assert [p["rows"] for p in payloads] == [20, 15, 7]
    assert payloads[0]["plan_kwargs"]["q_dtype"] == "bfloat16"
    assert payloads[0]["device_index"] == 0
    json.dumps(payloads)

    def broken_workers(payloads, workers):
        raise OSError("no subprocesses here")

    monkeypatch.setattr(pp, "run_precompile_workers", broken_workers)
    monkeypatch.setenv("TLLM_PRIMTS_WARM_STRICT", "0")
    planned.clear()
    fresh = SimpleNamespace(**{**vars(eager), "fmha_plan_caches": {}})
    mod._warmup_profiles([mod._WarmupProfile(lib, extra=32, **profile)], fresh)
    assert planned == [(20, 32), (15, 32), (7, 32)]
    monkeypatch.setenv("TLLM_PRIMTS_WARM_STRICT", "1")
    with pytest.raises(OSError):
        mod._warmup_profiles([mod._WarmupProfile(lib, extra=32, **profile)], eager)


def test_precompile_worker_count_follows_the_env(monkeypatch) -> None:
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile as pp

    monkeypatch.delenv("TLLM_PRIMTS_WARM_WORKERS", raising=False)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    assert pp.worker_count(12) == 4  # default for a single rank
    assert pp.worker_count(3) == 3  # never more workers than variants
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "0")
    assert pp.worker_count(12) == 0
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "8")
    assert pp.worker_count(12) == 8  # explicit values are never capped
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "many")
    assert pp.worker_count(12) == 4  # unparsable: default with a warning
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    assert pp.worker_count(12) == 0  # nothing to persist into
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE")
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    assert pp.worker_count(12) == 0  # compiling is forbidden


def test_default_worker_count_is_capped_by_the_ranks_on_the_host(monkeypatch) -> None:
    """Under the default launcher every rank has a private FlashInfer workspace and fans
    out on its own, so the per-rank default shrinks with the ranks on the host: the
    host never starts more than 16 compile subprocesses at the default, and an
    explicit ``TLLM_PRIMTS_WARM_WORKERS`` still wins."""
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile as pp

    monkeypatch.delenv("TLLM_PRIMTS_WARM_WORKERS", raising=False)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    expected = {1: 4, 2: 4, 4: 4, 8: 2, 16: 1, 32: 1}
    for ranks, workers in expected.items():
        assert pp.default_worker_count(ranks) == workers, ranks
        assert pp.worker_count(12, local_ranks=ranks) == workers, ranks
        assert ranks * workers <= 16 or workers == 1
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "4")
    assert pp.worker_count(12, local_ranks=8) == 4  # explicit values are never capped

    # The rank count comes from the model's Mapping (no MPI call inside the
    # forward), else from the launcher's environment, else 1.
    for name in pp._LOCAL_RANK_ENV:
        monkeypatch.delenv(name, raising=False)
    assert pp.local_rank_count(SimpleNamespace(world_size=8, gpus_per_node=8)) == 8
    assert pp.local_rank_count(SimpleNamespace(world_size=16, gpus_per_node=8)) == 8
    assert pp.local_rank_count(SimpleNamespace(world_size=4, gpus_per_node=8)) == 4
    assert pp.local_rank_count(None) == 1
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "8")
    assert pp.local_rank_count(None) == 8
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "many")
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "4")
    assert pp.local_rank_count(None) == 4


def test_precompile_workers_compile_disjoint_shards(monkeypatch) -> None:
    """Workers use normal plan() and together handle every payload exactly once."""
    import json

    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile as pp

    planned = []
    mla_decode = _fake_sparse_plan(monkeypatch, planned)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    payloads = [
        pp.encode_payload(
            0, rows, 16, dict(max_topk=128, max_extra_topk=512, q_dtype=torch.bfloat16)
        )
        for rows in (20, 15, 7, 3)
    ]
    assert pp._main(["worker", json.dumps(payloads), "1", "2"]) == 0
    assert [r for r, _ in planned] == [15, 3]
    assert pp._main(["worker", json.dumps(payloads), "0", "2"]) == 0
    assert [r for r, _ in planned] == [15, 3, 20, 7]
    assert pp._main(["worker", json.dumps(payloads), "5", "8"]) == 0
    assert len(planned) == 4
    assert not mla_decode.BatchSparseMLADecodePagedTSWrapper.discovery_rows


def test_precompile_workers_get_a_clean_environment_and_bounded_wait(monkeypatch, tmp_path) -> None:
    """The runner strips the MPI launcher environment, forwards sys.path and the
    worker count 0, assigns one slot per worker, and kills workers on timeout."""
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile as pp

    spawned = []

    class _FakeProc:
        def __init__(self, argv, env, stdout, returncode=0, text="", hang=False):
            self.argv, self.env, self.returncode, self._hang = argv, env, returncode, hang
            stdout.write(text)
            self._done = False

        def wait(self, timeout=None):
            if self._hang and not getattr(self, "killed", False):
                raise subprocess.TimeoutExpired(self.argv, timeout)
            self._done = True  # like Popen.wait() after kill()
            return self.returncode

        def poll(self):
            return None if (self._hang and not self._done) else self.returncode

        def kill(self):
            self._done = True
            self.killed = True

    import subprocess

    def make_popen(**behaviour):
        def fake_popen(argv, env, stdout, stderr, text):
            proc = _FakeProc(argv, env, stdout, **behaviour)
            spawned.append(proc)
            return proc

        return fake_popen

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "4")
    monkeypatch.setattr(subprocess, "Popen", make_popen())
    payloads = [{"rows": r} for r in (20, 15, 7)]
    assert pp.run_precompile_workers(payloads, 2) is True
    assert len(spawned) == 2
    env = spawned[0].env
    assert "OMPI_COMM_WORLD_RANK" not in env and "SLURM_PROCID" not in env
    assert env["TLLM_PRIMTS_WARM_WORKERS"] == "0"
    import json
    import sys

    assert json.loads(env["TLLM_PRIMTS_PRECOMPILE_SYS_PATH"]) == sys.path
    assert [p.argv[-2:] for p in spawned] == [["0", "2"], ["1", "2"]]
    assert spawned[0].argv[0] == sys.executable and spawned[0].argv[1] == "-c"
    assert json.loads(spawned[0].argv[3]) == payloads

    spawned.clear()
    monkeypatch.setattr(subprocess, "Popen", make_popen(returncode=3, text="boom"))
    assert pp.run_precompile_workers(payloads, 2) is False

    spawned.clear()
    monkeypatch.setenv("TLLM_PRIMTS_WARM_TIMEOUT", "0.01")
    monkeypatch.setattr(subprocess, "Popen", make_popen(hang=True))
    assert pp.run_precompile_workers(payloads, 2) is False
    assert all(getattr(p, "killed", False) for p in spawned)  # nothing left running
    assert pp.run_precompile_workers(payloads, 1) is False  # one worker: nothing to fan out
