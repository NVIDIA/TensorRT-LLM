from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_multi_ctas_kv_counter_size,
)


def _get_total_num_blocks(manager: SimpleNamespace, kv_factor: int = 2) -> int:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = kv_factor
    return fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=manager))


def test_flashinfer_uses_v2_page_index_upper_bound_directly() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=50_000_000,
        impl=SimpleNamespace(get_page_index_upper_bound=lambda *_: 50_000_000),
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager) == 50_000_000


def test_flashinfer_preserves_legacy_pool_scaling() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=1024,
        impl=SimpleNamespace(),
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager, kv_factor=2) == 1024 * 36 * 2


def test_multi_ctas_kv_counter_size_covers_beam_expanded_batch() -> None:
    # The kernel keeps one counter per head per decoder sequence. Sizing off the
    # request count alone under-allocates under beam search, but only once the
    # product clears the multi-processor floor, so pick a case that does.
    num_heads, batch, beam, sm_count = 6, 16, 2, 148
    needed = num_heads * batch * beam * torch.int32.itemsize
    assert _get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) < needed
    assert _get_multi_ctas_kv_counter_size(num_heads, batch * beam, sm_count) >= needed


def test_multi_ctas_kv_counter_size_keeps_multi_processor_floor() -> None:
    num_heads, batch, sm_count = 6, 1, 148
    assert _get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) >= (
        sm_count * torch.int32.itemsize
    )
