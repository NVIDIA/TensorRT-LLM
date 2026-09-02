# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM FP4 MLA helper tests."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.attention_backend.fp4_mla as fp4_mla_backend
from tensorrt_llm._torch.attention_backend.fp4_mla import (
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q_GLOBAL_SCALE,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _cutedsl_backend_available,
    _fp4_mla_attention_backend,
    _get_fp4_mla_global_scale,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
)
from tensorrt_llm._torch.attention_backend.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as LlmKvCacheConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _swizzled_sf_offset(row_idx: int, col_idx: int, sf_per_token: int) -> int:
    padded_cols = ((sf_per_token + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


def _is_cutedsl_unavailable() -> bool:
    return (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or not _cutedsl_backend_available()
    )


def _reset_triton_allocator() -> None:
    import triton

    def _null_allocator(size: int, alignment: int, stream):
        raise RuntimeError("Triton kernel requested scratch memory without an allocator.")

    triton.set_allocator(_null_allocator)


def test_fp4_mla_generation_hp_page_ids_skip_mixed_batch_context_rows() -> None:
    metadata = SimpleNamespace(
        num_contexts=2,
        num_seqs=5,
        _fp4_mla_device_page_table=True,
        fp4_mla_page_table_stride=3,
        _fp4_mla_hp_page_indices=torch.arange(15, dtype=torch.int32),
    )

    page_ids = fp4_mla_backend._fp4_mla_generation_hp_page_ids(metadata, 3)

    torch.testing.assert_close(page_ids, torch.arange(6, 15, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
def test_fp4_mla_cuda_graph_generation_lengths_records_capture_once(monkeypatch) -> None:
    corrected_kv_lens = torch.empty(2, dtype=torch.int32, device="cuda")
    generation_lens = torch.empty(2, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        num_contexts=1,
        num_seqs=3,
        kv_lens_cuda_runtime=torch.tensor([9, 17, 25], dtype=torch.int32, device="cuda"),
        prompt_lens_cuda_runtime=torch.tensor([9, 1, 1], dtype=torch.int32, device="cuda"),
        fp4_mla_generation_kv_lens=corrected_kv_lens,
        fp4_mla_generation_append_lens=generation_lens,
        fp4_mla_generation_lengths_num_tokens=4,
        fp4_mla_generation_lengths_num_seqs=2,
        fp4_mla_generation_lengths_num_contexts=1,
        _fp4_mla_generation_lengths_capture_recorded=False,
        is_cuda_graph=True,
    )
    populate_calls = []

    def populate_generation_lengths(*args, **kwargs) -> None:
        populate_calls.append((args, kwargs))
        args[2].fill_(19)
        args[3].fill_(2)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        fp4_mla_backend,
        "populate_fp4_mla_generation_lengths",
        populate_generation_lengths,
    )

    first = fp4_mla_backend._fp4_mla_uniform_generation_lengths(metadata, 4, 2)
    second = fp4_mla_backend._fp4_mla_uniform_generation_lengths(metadata, 4, 2)

    assert len(populate_calls) == 1
    args, kwargs = populate_calls[0]
    assert args[0].data_ptr() == metadata.kv_lens_cuda_runtime[1:].data_ptr()
    assert args[1].data_ptr() == metadata.prompt_lens_cuda_runtime[1:].data_ptr()
    assert args[2].data_ptr() == corrected_kv_lens.data_ptr()
    assert args[3].data_ptr() == generation_lens.data_ptr()
    assert kwargs == {"num_gen_tokens": 4, "num_gen": 2}
    assert metadata._fp4_mla_generation_lengths_capture_recorded
    assert first[0].data_ptr() == second[0].data_ptr() == corrected_kv_lens.data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr() == generation_lens.data_ptr()


@pytest.mark.parametrize("layer_offset", [0, 1, 3])
def test_fp4_mla_v2_encoded_page_capacity_is_layer_invariant(layer_offset: int) -> None:
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    manager.impl = SimpleNamespace(
        get_page_index_converter=lambda *_: SimpleNamespace(
            scale=4,
            expansion=1,
            layer_offset=layer_offset,
        ),
        get_page_index_upper_bound=lambda *_: 20 - layer_offset,
    )

    assert manager._role_encoded_page_capacity(0, Role.KEY) == 17


def test_fp4_mla_v2_cache_size_accounts_for_hp_intercept(monkeypatch) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
    )

    slope, intercept = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(world_size=2, tp_size=1, pp_size=2, rank=0),
        num_layers=2,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=3,
        spec_config=SimpleNamespace(tokens_per_gen_step=4),
    )

    assert slope == 2 * (320 + 40 + 32)
    assert intercept == 3 * 2 * (HP_BLOCK_SIZE + 3) * 576 * 2 * 2


def test_fp4_mla_v2_runtime_sizing_accounts_for_pipeline_slots() -> None:
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    manager.max_batch_size = 3
    manager.mapping = SimpleNamespace(pp_size=2)
    manager.max_num_tokens = 100
    manager.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    manager.enable_swa_scratch_reuse = False
    manager._has_cp_helix = False
    manager._get_runtime_cache_size_layer_components = lambda: ([10, 8], [None, 19])

    quota = manager._get_quota_from_max_tokens(manager.max_num_tokens)

    hp_page_bytes = FP4_MLA_TOKENS_PER_BLOCK * 8
    expected_quota = manager.max_num_tokens * (10 + 8) + 6 * hp_page_bytes
    assert quota == expected_quota
    assert manager._get_max_tokens_from_quota(quota) == manager.max_num_tokens


@pytest.mark.parametrize(
    ("fused_v_transpose", "expected_v_head_dim"),
    [(False, 512), (True, None)],
    ids=["mufu16-packed-v", "fused-v-canonical-cache"],
)
def test_fp4_mla_manager_v2_packed_v_matches_cutedsl_variant(
    monkeypatch,
    fused_v_transpose: bool,
    expected_v_head_dim: int | None,
) -> None:
    class _ConstructionCaptured(Exception):
        pass

    captured = {}

    def capture_base_init(manager, *args, **kwargs) -> None:
        captured["manager"] = manager
        raise _ConstructionCaptured

    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused_v_transpose)))
    monkeypatch.setattr(KVCacheManagerV2, "__init__", capture_base_init)

    with pytest.raises(_ConstructionCaptured):
        Fp4MlaKVCacheManagerV2(
            LlmKvCacheConfig(dtype="nvfp4", enable_partial_reuse=False),
            _CacheType.SELFKONLY,
            head_dim=576,
            tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
            dtype=_DataType.NVFP4,
            pretrained_config=SimpleNamespace(kv_lora_rank=512),
        )

    assert captured["manager"].mla_v_head_dim == expected_v_head_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_manager_v2_registers_native_cache_and_hp_roles(monkeypatch) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    manager = Fp4MlaKVCacheManagerV2(
        LlmKvCacheConfig(
            max_tokens=512,
            dtype="nvfp4",
            enable_block_reuse=True,
            enable_partial_reuse=True,
            host_cache_size=0,
        ),
        _CacheType.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=512,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=_DataType.NVFP4,
        max_num_tokens=512,
        pretrained_config=SimpleNamespace(kv_lora_rank=512),
    )
    try:
        config = manager.kv_cache_manager_py_config
        assert not config.enable_partial_reuse
        assert len(config.layers) == 4
        assert [buffer.role for buffer in config.layers[0].buffers] == [
            Role.KEY,
            Role.KEY_BLOCK_SCALE,
            Role.MLA_V_SCALE,
        ]
        assert [buffer.role for buffer in config.layers[2].buffers] == [Role.MLA_HP_TAIL]
        assert config.layers[2].sliding_window_size == HP_BLOCK_SIZE

        page_spec = manager.get_fp4_mla_page_table_spec(0)
        assert page_spec.cache_pool_id != page_spec.hp_pool_id
        assert page_spec.cache_page_index_scale == 1
        assert page_spec.hp_page_index_scale == 1

        kv_cache, sf_cache = manager.get_fp4_mla_cache_buffers(0)
        v_scale_pool = manager.get_mla_v_scale_pool()
        hp_pool = manager.get_fp4_mla_hp_pool()
        assert kv_cache.shape[0] == sf_cache.shape[0] == v_scale_pool.shape[1]
        assert v_scale_pool.dtype == torch.float8_e4m3fn
        assert manager.get_mla_v_scale_pool_base().dtype == torch.uint8
        assert hp_pool.shape[1:] == (2, 1, HP_BLOCK_SIZE * 576)
    finally:
        manager.shutdown()


def _dequant_fp4_swizzled(
    fp4_tensor: torch.Tensor,
    sf_tensor: torch.Tensor,
    *,
    logical_dim: int,
    sf_per_token: int,
    global_scale: float,
) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=fp4_tensor.device,
    )
    fp4_bytes = fp4_tensor.view(torch.uint8)
    sf_flat = sf_tensor.view(torch.float8_e4m3fn).reshape(-1)
    out = torch.empty(
        (fp4_bytes.shape[0], logical_dim),
        dtype=torch.float32,
        device=fp4_tensor.device,
    )

    for row_idx in range(fp4_bytes.shape[0]):
        for sf_col in range(sf_per_token):
            start = sf_col * FP4_BLOCK_SIZE
            packed = fp4_bytes[row_idx, start // 2 : start // 2 + 8]
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            vals = torch.empty(FP4_BLOCK_SIZE, dtype=torch.float32, device=fp4_tensor.device)
            low_sign = torch.where(
                (low & 0x08) != 0,
                -torch.ones_like(low, dtype=torch.float32),
                torch.ones_like(low, dtype=torch.float32),
            )
            high_sign = torch.where(
                (high & 0x08) != 0,
                -torch.ones_like(high, dtype=torch.float32),
                torch.ones_like(high, dtype=torch.float32),
            )
            vals[0::2] = fp4_values[(low & 0x07).long()] * low_sign
            vals[1::2] = fp4_values[(high & 0x07).long()] * high_sign
            sf_offset = _swizzled_sf_offset(row_idx, sf_col, sf_per_token)
            out[row_idx, start : start + FP4_BLOCK_SIZE] = (
                vals * sf_flat[sf_offset].float() / global_scale
            )

    return out


def _duplicate_tail_groups(tensor: torch.Tensor, residual_dim: int) -> torch.Tensor:
    prefix = tensor[..., :-residual_dim]
    tail = tensor[..., -residual_dim:].reshape(
        *tensor.shape[:-1], residual_dim // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE
    )
    duplicated_tail = tail.repeat_interleave(2, dim=-2).reshape(
        *tensor.shape[:-1],
        residual_dim * 2,
    )
    return torch.cat((prefix, duplicated_tail), dim=-1)


def _expand_qk_residual_terms(
    q: torch.Tensor,
    k: torch.Tensor,
    k_residual: torch.Tensor,
    residual_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [Q, Q_r, Q] x [K, K, K_r] after the shared prefix."""
    prefix_dim = k.shape[-1] - residual_dim
    residual_groups = residual_dim // FP4_BLOCK_SIZE
    q_tail = q[..., prefix_dim:].reshape(*q.shape[:-1], residual_groups, 2, FP4_BLOCK_SIZE)
    q_main = q_tail[..., 0, :].reshape(*q.shape[:-1], residual_dim)
    q_residual = q_tail[..., 1, :].reshape(*q.shape[:-1], residual_dim)
    k_main = k[..., prefix_dim:]
    return (
        torch.cat((q[..., :prefix_dim], q_main, q_residual, q_main), dim=-1),
        torch.cat((k[..., :prefix_dim], k_main, k_main, k_residual), dim=-1),
    )


def _create_fp4_mla_v2_manager(
    *,
    max_tokens: int,
    max_seq_len: int,
    max_batch_size: int,
    spec_config=None,
    enable_block_reuse: bool = False,
) -> Fp4MlaKVCacheManagerV2:
    return Fp4MlaKVCacheManagerV2(
        LlmKvCacheConfig(
            max_tokens=max_tokens,
            dtype="nvfp4",
            enable_block_reuse=enable_block_reuse,
            host_cache_size=0,
        ),
        _CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=_DataType.NVFP4,
        spec_config=spec_config,
        max_num_tokens=max_tokens,
        pretrained_config=SimpleNamespace(kv_lora_rank=512),
    )


def _build_multi_seq_metadata(kv_cache_manager, *, seq_lens, page_size):
    device = torch.device("cuda")
    num_seqs = len(seq_lens)
    request_ids = list(range(num_seqs))
    block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(request_ids, layer_idx=0)
    page_spec = kv_cache_manager.get_fp4_mla_page_table_spec(0)
    hp_block_ids_per_seq = kv_cache_manager._get_batch_cache_indices_by_pool_id(
        request_ids,
        pool_id=page_spec.hp_pool_id,
        is_kv_aggregate=False,
    )
    num_blocks = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]

    max_blocks_per_seq = max(num_blocks)
    page_rows = []
    hp_page_rows = []
    for seq_idx, seq_blocks in enumerate(block_ids_per_seq):
        active_blocks = seq_blocks[: num_blocks[seq_idx]]
        page_rows.extend(active_blocks + [0] * (max_blocks_per_seq - len(active_blocks)))
        active_hp_blocks = hp_block_ids_per_seq[seq_idx][: num_blocks[seq_idx]]
        hp_page_rows.extend(active_hp_blocks + [0] * (max_blocks_per_seq - len(active_hp_blocks)))
    paged_kv_indices = torch.tensor(
        page_rows,
        dtype=torch.int32,
        device=device,
    )
    paged_kv_indptr = (
        torch.arange(num_seqs + 1, dtype=torch.int32, device=device) * max_blocks_per_seq
    )
    hp_page_indices = torch.tensor(hp_page_rows, dtype=torch.int32, device=device)
    batch_indices = torch.cat(
        [
            torch.full(
                (seq_len,),
                seq_idx,
                dtype=torch.int32,
                device=device,
            )
            for seq_idx, seq_len in enumerate(seq_lens)
        ]
    )
    positions = torch.cat(
        [torch.arange(seq_len, dtype=torch.int32, device=device) for seq_len in seq_lens]
    )

    hp_pool = kv_cache_manager.get_fp4_mla_hp_pool()
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prompt_lens_cuda = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prompt_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    kv_global_scale = torch.tensor([FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=device)
    q_global_scale = torch.tensor([FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=device)

    return SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_indices=paged_kv_indices,
        _paged_kv_indices=paged_kv_indices,
        _fp4_mla_hp_page_indices=hp_page_indices,
        paged_kv_indptr=paged_kv_indptr,
        _paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_decode=paged_kv_indptr.clone(),
        _fp4_mla_device_page_table=True,
        _fp4_mla_device_page_table_valid=True,
        fp4_mla_page_table_stride=max_blocks_per_seq,
        fp4_mla_context_repack_max_touched_pages=max_blocks_per_seq,
        page_size=page_size,
        num_context_blocks=num_seqs * max_blocks_per_seq,
        num_generation_blocks=0,
        num_contexts=num_seqs,
        num_seqs=num_seqs,
        num_blocks=None,
        high_precision_kv_pool=hp_pool,
        fp4_mla_v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        fp4_mla_generation_kv_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
        fp4_mla_generation_append_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
        fp4_mla_generation_lengths_num_tokens=-1,
        fp4_mla_generation_lengths_num_seqs=-1,
        fp4_mla_generation_lengths_num_contexts=-1,
        _fp4_mla_generation_lengths_capture_recorded=False,
        _fp4_mla_q_global_scale=q_global_scale,
        _fp4_mla_kv_global_scale=kv_global_scale,
        request_ids=request_ids,
        runtime_features=SimpleNamespace(has_speculative_draft_tokens=False),
        is_cuda_graph=False,
        is_warmup=False,
    )


def _materialize_reference_cache_storage(metadata, layer_idx: int, head_dim: int) -> torch.Tensor:
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    storage_head_dim = kv_cache.shape[-1] * 2
    static_global_scale = float(_get_fp4_mla_global_scale(metadata, kv_cache.device).item())
    num_generation_sequences = metadata.num_seqs - metadata.num_contexts
    page_stride = metadata.fp4_mla_page_table_stride
    page_rows = metadata.paged_kv_indices.view(
        metadata.num_seqs,
        page_stride,
    )
    generation_page_rows = page_rows[metadata.num_contexts : metadata.num_seqs]
    indptr = metadata.paged_kv_indptr_decode[: num_generation_sequences + 1].cpu().tolist()
    expected_indptr = [seq_idx * page_stride for seq_idx in range(num_generation_sequences + 1)]
    assert indptr == expected_indptr
    assert generation_page_rows.shape == (num_generation_sequences, page_stride)

    active_blocks = (
        (
            metadata.kv_lens_cuda_runtime[metadata.num_contexts : metadata.num_seqs]
            + metadata.page_size
            - 1
        )
        // metadata.page_size
    ).tolist()
    for seq_idx, block_count in enumerate(active_blocks):
        assert block_count <= page_stride
        assert indptr[seq_idx + 1] - indptr[seq_idx] == page_stride
        assert block_count * metadata.page_size >= int(
            metadata.kv_lens_cuda_runtime[metadata.num_contexts + seq_idx].item()
        )

    pages = []
    dequantized_pages = {}
    for page_id in generation_page_rows.reshape(-1).cpu().tolist():
        assert 0 <= page_id < kv_cache.shape[0]
        if page_id not in dequantized_pages:
            fp4_page = kv_cache[page_id, 0, :, 0, :]
            sf_page = sf_cache[page_id]
            dequantized_pages[page_id] = _dequant_fp4_swizzled(
                fp4_page,
                sf_page,
                logical_dim=storage_head_dim,
                sf_per_token=storage_head_dim // FP4_BLOCK_SIZE,
                global_scale=static_global_scale,
            )
        pages.append(dequantized_pages[page_id])
    if num_generation_sequences == 0:
        return torch.empty(
            (0, metadata.page_size, storage_head_dim),
            dtype=torch.float32,
            device=kv_cache.device,
        )
    storage = torch.stack(pages, dim=0)
    assert storage.shape[0] == indptr[-1] == num_generation_sequences * page_stride
    return storage


def _materialize_reference_cache_tokens(
    metadata,
    layer_idx: int,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    storage_head_dim = kv_cache.shape[-1] * 2
    static_global_scale = float(_get_fp4_mla_global_scale(metadata, kv_cache.device).item())
    page_rows = metadata.paged_kv_indices.view(
        metadata.num_seqs,
        metadata.fp4_mla_page_table_stride,
    )
    dequantized_pages = {}
    tokens = []
    for batch_idx, position in zip(
        batch_indices.cpu().tolist(),
        positions.cpu().tolist(),
    ):
        page_idx, page_position = divmod(position, metadata.page_size)
        physical_page = int(page_rows[batch_idx, page_idx].item())
        if physical_page not in dequantized_pages:
            dequantized_pages[physical_page] = _dequant_fp4_swizzled(
                kv_cache[physical_page, 0, :, 0, :],
                sf_cache[physical_page],
                logical_dim=storage_head_dim,
                sf_per_token=storage_head_dim // FP4_BLOCK_SIZE,
                global_scale=static_global_scale,
            )
        tokens.append(dequantized_pages[physical_page][page_position, :head_dim])
    return torch.stack(tokens, dim=0)


def _build_fp4_mla_attention_decode_case(
    *,
    seq_lens,
    num_heads,
    seed,
    query_len_per_seq=1,
    enable_block_reuse=False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_blocks = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]
    num_pages = sum(num_blocks)
    max_seq_len = max(page_size, max(seq_lens))
    max_tokens = max(page_size, num_pages * page_size)
    context_seq_lens = [seq_len - query_len_per_seq for seq_len in seq_lens]
    if min(context_seq_lens) <= 0:
        raise ValueError("FP4 MLA decode cases require a non-empty context for every sequence.")
    spec_config = MTPDecodingConfig(max_draft_len=3) if query_len_per_seq == 4 else None

    kv_cache_manager = _create_fp4_mla_v2_manager(
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        max_batch_size=len(seq_lens),
        spec_config=spec_config,
        enable_block_reuse=enable_block_reuse,
    )
    kv_cache_manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
    expected_storage_head_dim = head_dim + (
        FP4_MLA_K_RESIDUAL_DIM if _fp4_mla_attention_backend() in ("triton", "cutedsl") else 0
    )
    kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(0)
    kv_cache.zero_()
    sf_cache.zero_()
    assert kv_cache.shape[-1] * 2 == expected_storage_head_dim
    assert sf_cache.shape[-1] == expected_storage_head_dim // FP4_BLOCK_SIZE

    metadata = _build_multi_seq_metadata(
        kv_cache_manager,
        seq_lens=seq_lens,
        page_size=page_size,
    )
    assert metadata.fp4_mla_v_scale_pool is not None
    persistent_pool_base = kv_cache_manager.get_mla_v_packed_pool_base()
    if persistent_pool_base is not None:
        persistent_pool_base.zero_()
    metadata.fp4_mla_v_scale_pool.zero_()

    metadata.kv_lens_cuda_runtime = torch.tensor(
        context_seq_lens,
        dtype=torch.int32,
        device=device,
    )
    metadata.prompt_lens_cuda_runtime = metadata.kv_lens_cuda_runtime.clone()
    metadata.prompt_lens_cpu_runtime = torch.tensor(context_seq_lens, dtype=torch.int32)
    metadata.batch_indices = torch.cat(
        [
            torch.full((seq_len,), seq_idx, dtype=torch.int32, device=device)
            for seq_idx, seq_len in enumerate(context_seq_lens)
        ]
    )
    metadata.positions = torch.cat(
        [torch.arange(seq_len, dtype=torch.int32, device=device) for seq_len in context_seq_lens]
    )
    context_latent = (
        torch.randn(sum(context_seq_lens), head_dim, dtype=torch.bfloat16, device=device) * 0.25
    ).clamp_(-1.0, 1.0)
    scatter_fp4_mla_kv_cache(
        metadata,
        context_latent,
        layer_idx=0,
        token_offset=0,
        phase="context",
        local_layer=0,
        v_head_dim=kv_lora_rank,
    )
    torch.cuda.synchronize()

    metadata.num_contexts = 0
    metadata.num_context_blocks = 0
    metadata.num_generation_blocks = len(seq_lens) * metadata.fp4_mla_page_table_stride
    metadata.kv_lens_cuda_runtime = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    metadata.prompt_lens_cuda_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32, device=device
    )
    metadata.prompt_lens_cpu_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32
    )
    num_queries = len(seq_lens) * query_len_per_seq
    metadata.batch_indices = torch.arange(
        len(seq_lens),
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(query_len_per_seq)
    metadata.positions = torch.cat(
        [
            torch.arange(context_len, seq_len, dtype=torch.int32, device=device)
            for context_len, seq_len in zip(context_seq_lens, seq_lens)
        ]
    )
    metadata.num_tokens = num_queries
    metadata.num_ctx_tokens = 0
    metadata.max_num_tokens = num_queries
    metadata.max_num_sequences = len(seq_lens)
    metadata.max_total_draft_tokens = query_len_per_seq - 1
    metadata.runtime_features.has_speculative_draft_tokens = query_len_per_seq > 1
    generation_latent = (
        torch.randn(num_queries, head_dim, dtype=torch.bfloat16, device=device) * 0.25
    ).clamp_(-1.0, 1.0)
    generation_latent[:, :FP4_BLOCK_SIZE] = 0.75
    generation_latent[:, 1:FP4_BLOCK_SIZE:2] = -0.75
    q_nope = (
        torch.randn(num_queries, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
        * 0.25
    ).clamp_(-1.0, 1.0)
    q_pe = (
        torch.randn(
            num_queries,
            num_heads,
            qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).clamp_(-1.0, 1.0)
    q_quant_input = torch.cat((q_nope, torch.zeros_like(q_pe)), dim=-1).contiguous()
    q_rope_out = q_quant_input[..., kv_lora_rank:]
    rotary_cos_sin = torch.zeros(
        (max_seq_len, qk_rope_head_dim, 2),
        dtype=torch.float32,
        device=device,
    )
    rotary_cos_sin[..., 0] = 1.0

    assert scatter_fp4_mla_kv_cache(
        metadata,
        generation_latent,
        layer_idx=0,
        token_offset=0,
        phase="generation",
        local_layer=0,
        v_head_dim=kv_lora_rank,
        rotary_cos_sin=rotary_cos_sin,
        q_pe=q_pe,
        q_rope_out=q_rope_out,
        q_quant_input=q_quant_input,
    )
    torch.cuda.synchronize()
    assert metadata._fp4_mla_prequantized_q is not None
    assert metadata._fp4_mla_prequantized_q_sf is not None
    assert metadata._fp4_mla_q_batch_capacity == num_queries
    torch.testing.assert_close(q_rope_out, q_pe, rtol=0, atol=0)

    canonical_generation = _materialize_reference_cache_tokens(
        metadata,
        layer_idx=0,
        batch_indices=metadata.batch_indices,
        positions=metadata.positions,
        head_dim=head_dim,
    )
    torch.testing.assert_close(
        canonical_generation,
        generation_latent.float(),
        rtol=0.25,
        atol=0.2,
        msg="generation scatter did not write the distinct BF16 latent into canonical KV",
    )

    # Decode reference exercises the quantized cache without an HP overlay.
    metadata.high_precision_kv_pool.zero_()
    torch.cuda.synchronize()

    return kv_cache_manager, metadata, q_quant_input, kv_lora_rank, qk_rope_head_dim


def _fp4_mla_attention_decode_reference(
    metadata,
    q_nope,
    q_pe,
    *,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
) -> torch.Tensor:
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage = _materialize_reference_cache_storage(metadata, 0, head_dim)
    dequant_cache = storage[..., :head_dim]
    dequant_k_residual = None
    if _fp4_mla_attention_backend() in ("triton", "cutedsl"):
        dequant_k_residual = storage[..., head_dim : head_dim + FP4_MLA_K_RESIDUAL_DIM]
    num_heads = q_nope.shape[1]
    q_full = torch.cat((q_nope, q_pe), dim=-1).reshape(-1, head_dim)
    global_scale = metadata._fp4_mla_q_global_scale
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
        q_full,
        global_scale,
        FP4_MLA_Q_RESIDUAL_DIM,
        is_act=True,
    )
    q_logical_dim = head_dim + FP4_MLA_Q_RESIDUAL_DIM
    q_dequant = _dequant_fp4_swizzled(
        q_fp4,
        q_sf.view(torch.float8_e4m3fn),
        logical_dim=q_logical_dim,
        sf_per_token=q_logical_dim // FP4_BLOCK_SIZE,
        global_scale=float(global_scale.item()),
    )

    p_dequant = None
    if hasattr(metadata, "_fp4_mla_attention_p_buf"):
        p_dequant = _dequant_fp4_swizzled(
            metadata._fp4_mla_attention_p_buf,
            metadata._fp4_mla_attention_p_sf_buf,
            logical_dim=metadata.page_size,
            sf_per_token=metadata.page_size // FP4_BLOCK_SIZE,
            global_scale=FP4_MLA_P_GLOBAL_SCALE,
        )

    indptr = metadata.paged_kv_indptr_decode.cpu().tolist()
    num_seqs = metadata.num_seqs - metadata.num_contexts
    indptr = indptr[: num_seqs + 1]
    kv_lens = (
        metadata.kv_lens_cuda_runtime[metadata.num_contexts : metadata.num_seqs].cpu().tolist()
    )
    assert len(kv_lens) == num_seqs
    assert len(indptr) == num_seqs + 1
    assert indptr[0] == 0
    assert indptr[-1] == storage.shape[0]
    query_len_per_seq = q_nope.shape[0] // num_seqs
    max_pages = max(indptr[seq_idx + 1] - indptr[seq_idx] for seq_idx in range(num_seqs))
    outputs = []
    for seq_idx in range(num_seqs):
        kv_len = kv_lens[seq_idx]
        page_count = indptr[seq_idx + 1] - indptr[seq_idx]
        assert page_count == metadata.fp4_mla_page_table_stride
        assert page_count * metadata.page_size >= kv_len
        full_cache = dequant_cache[indptr[seq_idx] : indptr[seq_idx + 1]].reshape(-1, head_dim)
        assert full_cache.shape[0] >= kv_len
        full_v_cache = full_cache[:, :kv_lora_rank]
        for query_offset in range(query_len_per_seq):
            query_idx = seq_idx * query_len_per_seq + query_offset
            effective_kv_len = kv_len - (query_len_per_seq - 1 - query_offset)
            cache = full_cache[:effective_kv_len]
            v_cache = full_v_cache[:effective_kv_len]
            q_start = query_idx * num_heads
            q = q_dequant[q_start : q_start + num_heads]
            if dequant_k_residual is None:
                logical_q = q
                logical_k = _duplicate_tail_groups(cache.float(), FP4_MLA_Q_RESIDUAL_DIM)
            else:
                full_k_residual = dequant_k_residual[indptr[seq_idx] : indptr[seq_idx + 1]].reshape(
                    -1, FP4_MLA_K_RESIDUAL_DIM
                )
                logical_q, logical_k = _expand_qk_residual_terms(
                    q,
                    cache.float(),
                    full_k_residual[:effective_kv_len],
                    FP4_MLA_Q_RESIDUAL_DIM,
                )
            probs = torch.softmax(
                torch.matmul(logical_q, logical_k.transpose(0, 1)) * sm_scale,
                dim=-1,
            )

            if p_dequant is None:
                p = probs
            else:
                p_pages = []
                for page_rel in range(indptr[seq_idx + 1] - indptr[seq_idx]):
                    page_start = page_rel * metadata.page_size
                    valid_tokens = max(min(effective_kv_len - page_start, metadata.page_size), 0)
                    if valid_tokens == 0:
                        continue
                    p_page = query_idx * max_pages + page_rel
                    p_start = p_page * num_heads
                    p_pages.append(p_dequant[p_start : p_start + num_heads, :valid_tokens])
                p = torch.cat(p_pages, dim=-1)

            outputs.append(torch.matmul(p, v_cache.float()))
    return torch.stack(outputs, dim=0)


def _assert_fp4_mla_attention_decode_accuracy(
    monkeypatch,
    *,
    backend: str,
    fused_v_transpose: bool,
    num_heads: int,
    seq_lens: list[int],
    seed: int,
    query_len_per_seq: int = 1,
    enable_block_reuse: bool = False,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    monkeypatch.setenv(
        FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
        str(int(fused_v_transpose)),
    )
    (
        kv_cache_manager,
        metadata,
        q,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        seed=seed,
        query_len_per_seq=query_len_per_seq,
        enable_block_reuse=enable_block_reuse,
    )
    try:
        v_packed_pool = fp4_mla_backend._get_fp4_mla_v_packed_pool(metadata, 0)
        assert (v_packed_pool is None) is fused_v_transpose

        output = torch.empty_like(q[..., :kv_lora_rank])
        sm_scale = 0.1
        run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q=q,
            output=output,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            prequantized_q=metadata._fp4_mla_prequantized_q,
            prequantized_q_sf=metadata._fp4_mla_prequantized_q_sf,
            q_batch_capacity=metadata._fp4_mla_q_batch_capacity,
        )
        torch.cuda.synchronize()

        ref_output = _fp4_mla_attention_decode_reference(
            metadata,
            q[..., :kv_lora_rank],
            q[..., kv_lora_rank:],
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        output_float = output.float()
        error = output_float - ref_output
        output_rows = output_float.reshape(output_float.shape[0], -1)
        reference_rows = ref_output.reshape(ref_output.shape[0], -1)
        error_rows = error.reshape(error.shape[0], -1)
        output_norms = torch.linalg.vector_norm(output_rows, dim=1)
        reference_norms = torch.linalg.vector_norm(reference_rows, dim=1)
        relative_l2 = torch.linalg.vector_norm(error_rows, dim=1) / reference_norms
        cosine = torch.sum(output_rows * reference_rows, dim=1) / (output_norms * reference_norms)
        assert bool(torch.all(reference_norms > 1e-3).item())
        assert bool(torch.all(output_norms > 1e-3).item())
        max_relative_l2 = float(relative_l2.max().item())
        min_cosine = float(cosine.min().item())
        assert max_relative_l2 < 0.15, (
            f"{backend} FP4 MLA attention relative L2 error is too large: {max_relative_l2}"
        )
        assert min_cosine > 0.99, (
            f"{backend} FP4 MLA attention cosine similarity is too small: {min_cosine}"
        )
        max_abs_error = torch.max(torch.abs(error)).item()
        torch.testing.assert_close(
            output_float,
            ref_output,
            atol=1.5e-1,
            rtol=1.5e-1,
            msg=(
                f"{backend} FP4 MLA attention decode output diverged from reference; "
                f"max_abs_error={max_abs_error}, max_relative_l2={max_relative_l2}, "
                f"min_cosine={min_cosine}"
            ),
        )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CTM/CuTeDSL runtime",
)
@pytest.mark.parametrize(
    "fused_v_transpose",
    [False, True],
    ids=["mufu16", "mufu16-fused-v-transpose"],
)
@pytest.mark.parametrize("query_len_per_seq", [1, 4], ids=["mtp0", "mtp3"])
def test_fp4_mla_attention_decode_cutedsl_matches_reference(
    monkeypatch,
    fused_v_transpose: bool,
    query_len_per_seq: int,
) -> None:
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutedsl",
        fused_v_transpose=fused_v_transpose,
        num_heads=128,
        seq_lens=[131, 512],
        seed=29,
        query_len_per_seq=query_len_per_seq,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CTM/CuTeDSL runtime",
)
def test_fp4_mla_attention_decode_cutedsl_block_reuse_repack_matches_reference(
    monkeypatch,
) -> None:
    original_repack = fp4_mla_backend._repack_cutedsl_v_packed_cache
    repack_calls = []

    def record_repack(*args, **kwargs) -> None:
        original_repack(*args, **kwargs)
        repack_calls.append(
            {
                "page_indptr": kwargs["page_indptr"].detach().cpu().clone(),
                "kv_lens": kwargs["kv_lens"].detach().cpu().clone(),
                "generation_lens": kwargs["generation_lens"].detach().cpu().clone(),
                "max_touched_pages": kwargs["max_touched_pages"],
            }
        )

    monkeypatch.setattr(fp4_mla_backend, "_repack_cutedsl_v_packed_cache", record_repack)

    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutedsl",
        fused_v_transpose=False,
        num_heads=128,
        seq_lens=[131, 512],
        seed=29,
        query_len_per_seq=1,
        enable_block_reuse=True,
    )

    assert len(repack_calls) == 2
    expected_indptr = torch.tensor([0, 4, 8], dtype=torch.int32)
    torch.testing.assert_close(repack_calls[0]["page_indptr"], expected_indptr)
    torch.testing.assert_close(repack_calls[1]["page_indptr"], expected_indptr)
    torch.testing.assert_close(
        repack_calls[0]["kv_lens"],
        torch.tensor([130, 511], dtype=torch.int32),
    )
    torch.testing.assert_close(
        repack_calls[0]["generation_lens"],
        torch.tensor([130, 511], dtype=torch.int32),
    )
    torch.testing.assert_close(
        repack_calls[1]["kv_lens"],
        torch.tensor([131, 512], dtype=torch.int32),
    )
    torch.testing.assert_close(
        repack_calls[1]["generation_lens"],
        torch.ones(2, dtype=torch.int32),
    )
    assert repack_calls[0]["max_touched_pages"] == 4
    assert repack_calls[1]["max_touched_pages"] == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires Rubin SM107",
)
@pytest.mark.parametrize(
    ("seq_len", "expected_slot"),
    [(17, 16), (33, 13)],
    ids=["tile-boundary", "ring-wrap"],
)
def test_fp4_mla_context_tail_uses_draft_slack_ring(
    monkeypatch,
    seq_len: int,
    expected_slot: int,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    ring_size = HP_BLOCK_SIZE + 3
    spec_config = MTPDecodingConfig(max_draft_len=3)
    kv_cache_manager = _create_fp4_mla_v2_manager(
        max_tokens=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=1,
        spec_config=spec_config,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [seq_len])
        metadata = _build_multi_seq_metadata(
            kv_cache_manager,
            seq_lens=[seq_len],
            page_size=FP4_MLA_TOKENS_PER_BLOCK,
        )
        latent = torch.randn(
            seq_len,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

        scatter_fp4_mla_kv_cache(
            metadata,
            latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=kv_lora_rank,
        )
        torch.cuda.synchronize()

        hp_page = int(metadata._fp4_mla_hp_page_indices[0].item())
        hp_ring = metadata.high_precision_kv_pool.view(
            metadata.high_precision_kv_pool.shape[0], 1, 1, ring_size, head_dim
        )
        torch.testing.assert_close(
            hp_ring[hp_page, 0, 0, expected_slot],
            latent[-1],
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(hp_ring[hp_page, 0, 0, 0]).item() == 0
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


_V_REPACK_PAGE_SIZE = 128
_V_REPACK_HEAD_DIM = 512
_V_REPACK_PACKED_DIM = _V_REPACK_HEAD_DIM // 2


def _random_v_repack_production_cache(num_pages: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(20260817)
    return torch.randint(
        0,
        256,
        (num_pages, 1, _V_REPACK_PAGE_SIZE, 1, _V_REPACK_PACKED_DIM),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )


def _v_repack_sentinel_output(num_pages: int, sentinel: int) -> torch.Tensor:
    return torch.full(
        (num_pages * _V_REPACK_HEAD_DIM, _V_REPACK_PAGE_SIZE // 2),
        sentinel,
        dtype=torch.uint8,
        device="cuda",
    )


def _resolve_v_repack_generation_page_ids(
    page_ids: list[int],
    page_indptr: list[int],
    kv_lens: list[int],
    generation_lens: list[int],
    *,
    num_pages: int,
    max_touched_pages: int,
) -> list[int]:
    resolved = []
    for sequence_idx, (kv_len, generation_len) in enumerate(zip(kv_lens, generation_lens)):
        if kv_len <= 0 or generation_len <= 0:
            continue
        first_page = max(kv_len - generation_len, 0) // _V_REPACK_PAGE_SIZE
        last_page = (kv_len - 1) // _V_REPACK_PAGE_SIZE
        page_begin = page_indptr[sequence_idx]
        page_count = page_indptr[sequence_idx + 1] - page_begin
        for touched_page in range(max_touched_pages):
            logical_page = first_page + touched_page
            if logical_page > last_page or logical_page < 0 or logical_page >= page_count:
                continue
            physical_page = page_ids[page_begin + logical_page]
            if 0 <= physical_page < num_pages:
                resolved.append(physical_page)
    return resolved


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CTM/CuTeDSL runtime",
)
def test_fp4_mla_v_repack_full_pages_matches_reference() -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    num_pages = 3
    kv_cache = _random_v_repack_production_cache(num_pages)
    actual = _v_repack_sentinel_output(num_pages, sentinel=0xA5)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CTM/CuTeDSL runtime",
)
def test_fp4_mla_v_repack_selected_pages_preserves_untouched_page() -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    num_pages = 3
    sentinel = 0x5A
    kv_cache = _random_v_repack_production_cache(num_pages)
    page_ids = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    actual = _v_repack_sentinel_output(num_pages, sentinel)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.all(actual[_V_REPACK_HEAD_DIM : 2 * _V_REPACK_HEAD_DIM] == sentinel)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CTM/CuTeDSL runtime",
)
def test_fp4_mla_v_repack_generation_csr_crosses_page_boundary() -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    num_pages = 4
    max_touched_pages = 2
    sentinel = 0xC3
    page_ids_host = [2, 0, 1]
    page_indptr_host = [0, 2, 3]
    kv_lens_host = [130, 128]
    generation_lens_host = [4, 0]
    resolved_page_ids_host = _resolve_v_repack_generation_page_ids(
        page_ids_host,
        page_indptr_host,
        kv_lens_host,
        generation_lens_host,
        num_pages=num_pages,
        max_touched_pages=max_touched_pages,
    )
    assert resolved_page_ids_host == [2, 0]

    kv_cache = _random_v_repack_production_cache(num_pages)
    page_ids = torch.tensor(page_ids_host, dtype=torch.int32, device="cuda")
    page_indptr = torch.tensor(page_indptr_host, dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor(kv_lens_host, dtype=torch.int32, device="cuda")
    generation_lens = torch.tensor(generation_lens_host, dtype=torch.int32, device="cuda")
    resolved_page_ids = torch.tensor(
        resolved_page_ids_host,
        dtype=torch.int32,
        device="cuda",
    )
    actual = _v_repack_sentinel_output(num_pages, sentinel)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        resolved_page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
        page_indptr=page_indptr,
        kv_lens=kv_lens,
        generation_lens=generation_lens,
        max_touched_pages=max_touched_pages,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.all(actual[_V_REPACK_HEAD_DIM : 2 * _V_REPACK_HEAD_DIM] == sentinel)
    assert torch.all(actual[3 * _V_REPACK_HEAD_DIM : 4 * _V_REPACK_HEAD_DIM] == sentinel)
