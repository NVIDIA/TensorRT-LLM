# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimTS DSA sparse MLA over NVFP4 KV records: the gather-skip decision, the writer, the kernel."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import prims_ts_dsa_sparse_mla, registry
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_dsa_sparse_mla import (
    PrimsTSDsaSparseMlaFmha,
    may_use_nvfp4_mla_record_cache,
    prims_ts_reads_nvfp4_mla_records,
)
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
    get_prims_ts_toolchain_unsupported_reason,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    NVFP4_MLA_KV_CACHE_LAYOUT_ENV,
    NVFP4_MLA_RECORD_LAYOUT,
    NVFP4_MLA_RESIDUAL_LAYOUT,
    DSACacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode

_LATENT_DIM = 512
_ROPE_DIM = 64
_QK_DIM = _LATENT_DIM + _ROPE_DIM
_NOPE_HEAD_DIM = 128
_RECORD_STORAGE_EXTRA_DIM = 2 * _ROPE_DIM
_TOKENS_PER_BLOCK = 64
_SEQ_LENS = (300, 200)
# The global scale a DSA layer keeps for the residual layout (amax 100 over 448 * 6). The layer
# passes it to the cache ops whatever the layout; records do not take it.
_RESIDUAL_LAYOUT_SCALE = 100.0 / (448 * 6)

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="NVFP4 KV caches require Blackwell",
)


class _Layer:
    """Stand-in for a TrtllmAttention layer (the libraries hold their layer weakly)."""

    def __init__(self, lists_library: bool = True, num_heads: int = 64):
        library = (
            PrimsTSDsaSparseMlaFmha.__new__(PrimsTSDsaSparseMlaFmha) if lists_library else None
        )
        self._fmha_manager = SimpleNamespace(fmha_libs=[library] if lists_library else [object()])
        self.num_heads = num_heads
        self.qk_nope_head_dim = _NOPE_HEAD_DIM
        self.qk_rope_head_dim = _ROPE_DIM
        self.kv_lora_rank = _LATENT_DIM
        self.q_scaling = 1.0


def _metadata(layout=NVFP4_MLA_RECORD_LAYOUT, beam_width=1, helix=False, helix_offsets=None):
    return SimpleNamespace(
        kv_cache_manager=SimpleNamespace(mla_kv_cache_layout=layout),
        beam_width=beam_width,
        mapping=SimpleNamespace(has_cp_helix=lambda: helix),
        helix_position_offsets=helix_offsets,
    )


@pytest.mark.cpu_only
def test_layer_skips_the_gather_only_when_the_library_serves():
    assert prims_ts_reads_nvfp4_mla_records(_metadata(), _Layer())
    # Skip-correction and unsupported head counts keep the library out of the layer's FMHA list.
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(), _Layer(lists_library=False))
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(helix=True), _Layer())
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(helix_offsets=torch.zeros(1)), _Layer())
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(beam_width=2), _Layer())
    assert not prims_ts_reads_nvfp4_mla_records(
        _metadata(layout=NVFP4_MLA_RESIDUAL_LAYOUT), _Layer()
    )


@pytest.mark.cpu_only
def test_metadata_leaves_out_the_gather_rows_only_when_the_library_can_run(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_fmha_lib_classes", lambda: [PrimsTSDsaSparseMlaFmha])
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 103)
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla, "get_prims_ts_toolchain_unsupported_reason", lambda: None
    )
    assert prims_ts_reads_nvfp4_mla_records(_metadata())
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(helix=True))
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(beam_width=2))
    assert not prims_ts_reads_nvfp4_mla_records(_metadata(layout=NVFP4_MLA_RESIDUAL_LAYOUT))

    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 90)
    assert not prims_ts_reads_nvfp4_mla_records(_metadata())
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 103)
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla,
        "get_prims_ts_toolchain_unsupported_reason",
        lambda: "no CuTe DSL",
    )
    assert not prims_ts_reads_nvfp4_mla_records(_metadata())
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla, "get_prims_ts_toolchain_unsupported_reason", lambda: None
    )
    monkeypatch.setattr(registry, "get_enabled_fmha_lib_classes", lambda: [])
    assert not prims_ts_reads_nvfp4_mla_records(_metadata())


@pytest.mark.cpu_only
def test_record_reader_is_known_before_the_layers_and_the_cache(monkeypatch):
    """What the cache manager asks to settle the ``auto`` layout."""
    reason = prims_ts_dsa_sparse_mla.get_nvfp4_mla_record_reader_unsupported_reason
    monkeypatch.setattr(registry, "get_enabled_fmha_lib_classes", lambda: [PrimsTSDsaSparseMlaFmha])
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 103)
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla, "get_prims_ts_toolchain_unsupported_reason", lambda: None
    )
    assert reason() is None
    assert reason(SimpleNamespace(has_cp_helix=lambda: False)) is None
    assert "Helix" in reason(SimpleNamespace(has_cp_helix=lambda: True))

    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 120)
    assert "SM120" in reason()
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 103)
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla,
        "get_prims_ts_toolchain_unsupported_reason",
        lambda: "no CuTe DSL",
    )
    assert reason() == "no CuTe DSL"
    monkeypatch.setattr(registry, "get_enabled_fmha_lib_classes", lambda: [])
    assert "TLLM_FMHA_LIBS" in reason()


@pytest.mark.cpu_only
def test_sm107_needs_the_rubin_helpers_of_the_cute_dsl(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_fmha_lib_classes", lambda: [PrimsTSDsaSparseMlaFmha])
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla, "get_prims_ts_toolchain_unsupported_reason", lambda: None
    )
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "get_sm_version", lambda: 107)
    layer = _Layer()
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)
    assert prims_ts_reads_nvfp4_mla_records(_metadata())
    assert PrimsTSDsaSparseMlaFmha._unavailable_reason(layer) is None
    monkeypatch.setattr(prims_ts_dsa_sparse_mla, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", False)
    assert not prims_ts_reads_nvfp4_mla_records(_metadata())
    assert "Rubin helpers" in PrimsTSDsaSparseMlaFmha._unavailable_reason(layer)


@pytest.mark.cpu_only
def test_library_rejects_the_requests_the_gather_skip_rejects():
    """The library's request check and the gather skip share the metadata conditions."""
    layer = _Layer()
    library = PrimsTSDsaSparseMlaFmha(layer)
    q = SimpleNamespace(device=SimpleNamespace(type="cuda"), ndim=2)
    fwd = SimpleNamespace()
    for metadata, expected in (
        (_metadata(helix=True), "Helix"),
        (_metadata(beam_width=2), "beam search"),
        (_metadata(layout=NVFP4_MLA_RESIDUAL_LAYOUT), "layout"),
    ):
        assert not prims_ts_reads_nvfp4_mla_records(metadata, _Layer())
        assert expected in library._unsupported_reason(q, None, None, metadata, fwd)


@pytest.mark.cpu_only
def test_layer_may_hold_records_unless_the_residual_layout_is_requested(monkeypatch):
    layer = SimpleNamespace(has_fp4_kv_cache=True)
    monkeypatch.delenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, raising=False)
    assert may_use_nvfp4_mla_record_cache(layer)
    assert not may_use_nvfp4_mla_record_cache(SimpleNamespace(has_fp4_kv_cache=False))
    monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, NVFP4_MLA_RECORD_LAYOUT)
    assert may_use_nvfp4_mla_record_cache(layer)
    monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, NVFP4_MLA_RESIDUAL_LAYOUT)
    assert not may_use_nvfp4_mla_record_cache(layer)


@pytest.fixture
def written_records(monkeypatch):
    """A record cache that the native MLA writer filled from known latents."""
    monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, NVFP4_MLA_RECORD_LAYOUT)
    manager = DSACacheManagerV2(
        KvCacheConfig(max_tokens=4096, enable_block_reuse=False),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=_QK_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=1024,
        max_batch_size=len(_SEQ_LENS),
        mapping=Mapping(),
        dtype=DataType.NVFP4,
        sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=128),
        pretrained_config=SimpleNamespace(
            num_hidden_layers=1, kv_lora_rank=_LATENT_DIM, qk_rope_head_dim=_ROPE_DIM
        ),
    )
    try:
        assert manager.mla_kv_cache_layout == NVFP4_MLA_RECORD_LAYOUT
        assert manager.mla_kv_cache_residual_dim == _RECORD_STORAGE_EXTRA_DIM
        request_ids = list(range(len(_SEQ_LENS)))
        seq_lens = list(_SEQ_LENS)
        manager.add_dummy_requests(request_ids, token_nums=seq_lens)
        metadata = TrtllmAttentionMetadata(
            max_num_requests=len(seq_lens),
            max_num_tokens=8192,
            kv_cache_manager=manager,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        metadata.num_contexts = len(seq_lens)
        metadata.request_ids = request_ids
        metadata.prompt_lens = seq_lens
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[0] * len(seq_lens)
        )
        metadata.prepare()

        generator = torch.Generator(device="cuda").manual_seed(0)
        num_tokens = sum(seq_lens)
        # Every 16-value NoPE block gets its own magnitude: a misplaced block scale shows.
        block_magnitude = 2.0 ** (
            torch.rand(num_tokens, _LATENT_DIM // 16, 1, device="cuda", generator=generator) * 5 - 3
        )
        nope = torch.randn(num_tokens, _LATENT_DIM // 16, 16, device="cuda", generator=generator)
        nope = (nope * block_magnitude).flatten(1)
        rope = torch.randn(num_tokens, _ROPE_DIM, device="cuda", generator=generator) * 2
        latent = torch.cat((nope, rope), dim=-1).to(torch.bfloat16)
        written = latent.clone()

        # An identity rotation table: the writer stores the RoPE part as given.
        _, rotary_cos_sin = RopeParams(
            dim=_ROPE_DIM, theta=10000.0, max_positions=1024, duplicate_data=True
        ).create_rope_const_params()
        table = rotary_cos_sin.reshape(-1)
        table[0::2] = 1.0
        table[1::2] = 0.0
        num_heads = 2
        q = torch.zeros(
            num_tokens,
            num_heads * (_NOPE_HEAD_DIM + _ROPE_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            written,
            len(seq_lens),
            metadata.ctx_cached_token_indptr,
            metadata.ctx_kv_indptr,
            int(metadata.max_ctx_seq_len),
            rotary_cos_sin,
            num_heads,
            _NOPE_HEAD_DIM,
            _ROPE_DIM,
            _LATENT_DIM,
            metadata.kv_cache_block_offsets,
            manager.kv_cache_pool_pointers,
            manager.kv_cache_pool_mapping,
            torch.tensor([1.0 / _RESIDUAL_LAYOUT_SCALE], dtype=torch.float32, device="cuda"),
            manager.mla_kv_cache_residual_dim,
            0,
            manager.tokens_per_block,
            manager.max_seq_len,
            1,
            int(QuantMode.from_quant_algo(kv_cache_quant_algo=QuantAlgo.NVFP4)),
        )
        torch.cuda.synchronize()

        page_scale, layer_offset = manager.get_primary_pool_page_index_params(0)
        rows = []
        for request_id, seq_len in zip(request_ids, seq_lens):
            blocks = list(manager.get_batch_cache_indices([request_id])[0])
            rows += [
                (blocks[position // _TOKENS_PER_BLOCK] * page_scale + layer_offset)
                * _TOKENS_PER_BLOCK
                + position % _TOKENS_PER_BLOCK
                for position in range(seq_len)
            ]
        pool = manager.get_unique_primary_pool()
        yield SimpleNamespace(
            manager=manager,
            latent=latent.float(),
            rows=torch.tensor(rows, dtype=torch.int32, device="cuda"),
            num_pool_tokens=pool.shape[0] * pool.shape[1] * manager.tokens_per_block,
        )
    finally:
        manager.shutdown()


def _gather(records, indices: torch.Tensor) -> torch.Tensor:
    """E4M3 rows of the selected records, as the TRTLLM-Gen generation fallback reads them."""
    output = torch.zeros((*indices.shape, _QK_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    compact_indices = torch.empty_like(indices)
    torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
        records.manager.kv_cache_pool_pointers.cpu(),
        records.manager.kv_cache_pool_mapping.cpu(),
        indices,
        output,
        compact_indices,
        torch.tensor([_RESIDUAL_LAYOUT_SCALE], dtype=torch.float32, device="cuda"),
        0,
        records.manager.mla_kv_cache_residual_dim,
        records.num_pool_tokens,
    )
    return output


@requires_blackwell
def test_record_writer_round_trips_through_the_gather(written_records):
    """What the writer stores with residual_dim = 128 is what the gather reads back."""
    gathered = _gather(written_records, written_records.rows.view(1, -1))[0].float()
    latent = written_records.latent

    # RoPE is stored as E4M3: one rounding of the BF16 input.
    expected_rope = latent[:, _LATENT_DIM:].to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(gathered[:, _LATENT_DIM:], expected_rope, rtol=0.07, atol=2.0**-9)

    # NoPE is E2M1 against an E4M3 block scale of about amax / 6. The scale may round down by
    # 6%, the widest E2M1 step is two scales, and the dequantized value is rounded to E4M3.
    nope = latent[:, :_LATENT_DIM].unflatten(-1, (-1, 16))
    error = (gathered[:, :_LATENT_DIM].unflatten(-1, (-1, 16)) - nope).abs()
    block_amax = nope.abs().amax(-1, keepdim=True)
    assert bool((error <= 0.26 * block_amax + 1e-6).all())
    relative_rms = error.pow(2).mean().sqrt() / nope.pow(2).mean().sqrt()
    assert relative_rms < 0.15


@requires_blackwell
def test_prims_ts_matches_the_gather(written_records, monkeypatch):
    """PrimTS reading the records in place agrees with attention over the gathered rows."""
    if get_sm_version() not in (100, 103, 107):
        pytest.skip("PrimTS DSA sparse MLA requires SM100, SM103 or SM107")
    reason = get_prims_ts_toolchain_unsupported_reason()
    if reason is not None:
        pytest.skip(reason)
    # Compile in this process; the DeepSeek-V4 tests cover the precompile worker fan-out.
    monkeypatch.setenv("TLLM_PRIMTS_WARM_WORKERS", "0")

    layer = _Layer(num_heads=64)
    heads, num_rows, topk = layer.num_heads, 3, 256
    generator = torch.Generator(device="cuda").manual_seed(1)
    written = written_records.rows
    picks = torch.stack(
        [
            torch.randperm(written.numel(), device="cuda", generator=generator)[:topk]
            for _ in range(num_rows)
        ]
    )
    indices = written[picks].contiguous()
    # The indexer pads short selections with -1 anywhere in the row.
    indices[1, 5::7] = -1
    indices[2, topk // 2 :] = -1

    query = torch.randn(num_rows, heads, _QK_DIM, device="cuda", generator=generator)
    query[..., :_LATENT_DIM] *= 0.25
    query = query.to(torch.float8_e4m3fn)
    output = torch.empty(num_rows, heads * _LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    metadata = SimpleNamespace(
        kv_cache_manager=written_records.manager,
        fmha_plan_caches={},
        num_tokens=num_rows,
        num_ctx_tokens=0,
        beam_width=1,
        mapping=Mapping(),
        helix_position_offsets=None,
        max_num_requests=num_rows,
        max_total_draft_tokens=0,
        max_num_tokens=num_rows,
        is_cuda_graph=False,
    )
    forward_args = SimpleNamespace(
        attention_input_type=AttentionInputType.generation_only,
        sparse_runtime_params=SimpleNamespace(
            aux_kv_cache_pool_ptr=None, sparse_attn_indices=indices
        ),
        quant_q_buffer=query.view(torch.uint8).reshape(-1),
        output=output,
        attention_mask_data=None,
        relative_attention_bias=None,
        softmax_stats_tensor=None,
        attention_sinks=None,
        out_scale=None,
        output_sf=None,
    )
    fused_q = torch.empty(num_rows, heads * _QK_DIM, dtype=torch.bfloat16, device="cuda")
    assert prims_ts_reads_nvfp4_mla_records(metadata, layer)
    library = layer._fmha_manager.fmha_libs[0]
    library.__init__(layer)
    assert library._is_supported(fused_q, None, None, metadata, forward_args)
    library.forward(fused_q, None, None, metadata, forward_args)
    torch.cuda.synchronize()

    keys = _gather(written_records, indices).float()
    logits = torch.einsum("rhd,rkd->rhk", query.float(), keys)
    logits = logits * library._softmax_scale(layer)
    logits = logits.masked_fill((indices < 0)[:, None, :], float("-inf"))
    reference = torch.einsum("rhk,rkd->rhd", logits.softmax(-1), keys[..., :_LATENT_DIM])
    got = output.view(num_rows, heads, _LATENT_DIM).float()
    relative_error = (got - reference).norm() / reference.norm()
    print(f"PrimTS vs gather relative error: {float(relative_error):.4f}")
    assert relative_error < 0.05
