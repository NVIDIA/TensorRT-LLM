# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract tests for the CuTe DSL MLA FMHA library (FP8 scales, split-KV sizing)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm.bindings import DataType


def test_fp8_mla_scales_remain_tensors() -> None:
    """FP8 scales reach the kernel as the same device tensors, never as host floats."""
    q = torch.empty(1)
    bmm1_scale = torch.tensor([0.25, 0.25 * 1.442695], dtype=torch.float32)
    bmm2_scale = torch.tensor([4.0], dtype=torch.float32)

    actual_bmm1_scale, actual_bmm2_scale = CuteDslMlaFmha._get_fp8_scales(
        q,
        SimpleNamespace(mla_bmm1_scale=bmm1_scale, mla_bmm2_scale=bmm2_scale),
    )

    assert actual_bmm1_scale is bmm1_scale
    assert actual_bmm2_scale is bmm2_scale


@pytest.mark.parametrize(
    "bmm1_scale, bmm2_scale, match",
    [
        (None, torch.tensor([1.0]), "requires mla_bmm1_scale and mla_bmm2_scale"),
        (torch.tensor([1.0], dtype=torch.float16), torch.tensor([1.0]), "float32"),
        (torch.tensor([1.0]), torch.empty(0), "float32"),
    ],
)
def test_fp8_mla_scales_are_validated(bmm1_scale, bmm2_scale, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        CuteDslMlaFmha._get_fp8_scales(
            torch.empty(1),
            SimpleNamespace(mla_bmm1_scale=bmm1_scale, mla_bmm2_scale=bmm2_scale),
        )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
def test_fp8_mla_custom_op_accepts_tensor_scales() -> None:
    # Importing the module registers the CuTe DSL custom ops.
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401

    schema = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell.default._schema
    argument_types = {argument.name: str(argument.type) for argument in schema.arguments}

    assert argument_types["softmax_scale"] == "Tensor"
    assert argument_types["output_scale"] == "Tensor"


def _mla_decode_runner_cls():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
def test_mla_decode_split_kv_is_a_bounded_power_of_two() -> None:
    runner_cls = _mla_decode_runner_cls()
    max_active_blocks = 296
    previous = None
    for batch_size in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        split_kv = runner_cls.get_default_split_kv(batch_size, 1, max_active_blocks)
        assert 1 <= split_kv <= runner_cls._MAX_SPLIT_KV
        assert split_kv & (split_kv - 1) == 0
        # More requests already fill more CTAs, so the split never grows.
        if previous is not None:
            assert split_kv <= previous
        previous = split_kv
    # Persistent round-robin is never a candidate: skewed decode KV balances
    # better on the hardware scheduler.
    assert runner_cls.get_is_persistent_candidates() == [False]


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
@pytest.mark.parametrize("num_heads", [8, 128])
def test_mla_decode_split_kv_workspace_rows_cover_every_bucket(num_heads: int) -> None:
    """The CUDA-graph workspace is sized once, so its row bound must cover every
    batch bucket and query length below the configured maxima, and must not
    shrink when either maximum grows."""
    runner_cls = _mla_decode_runner_cls()
    max_active_blocks = 296
    max_seq_len_q, max_batch_size = 4, 64

    rows = runner_cls._max_split_kv_rows(
        num_heads, max_seq_len_q, max_batch_size, max_active_blocks
    )
    for seq_len_q in range(1, max_seq_len_q + 1):
        num_q_tiles = runner_cls.get_num_q_tiles(num_heads, seq_len_q)
        assert 1 <= num_q_tiles <= seq_len_q * max(1, num_heads // runner_cls._MMA_TILE_M)
        bucket = 1
        while bucket <= max_batch_size:
            split_kv = runner_cls.get_default_split_kv(bucket, num_q_tiles, max_active_blocks)
            if split_kv > 1:
                assert bucket * num_q_tiles * split_kv <= rows
            bucket *= 2

    assert (
        runner_cls._max_split_kv_rows(
            num_heads, max_seq_len_q + 1, max_batch_size, max_active_blocks
        )
        >= rows
    )
    assert (
        runner_cls._max_split_kv_rows(
            num_heads, max_seq_len_q, 2 * max_batch_size, max_active_blocks
        )
        >= rows
    )


class _TrtllmGenStub:
    """Stands in for FlashInfer TRTLLM-Gen, the next MLA library on SM100/SM103."""

    def is_supported(self, *args, **kwargs) -> bool:
        return True


def _select_mla_generation_fmha(
    monkeypatch,
    *,
    sm: int,
    num_heads: int,
    seq_len_q: int,
    fp8_kv: bool,
    helix: bool,
    batch_size: int,
):
    """Run FmhaManager._select_mla over [CuteDslMlaFmha, TRTLLM-Gen stub] on CPU."""
    from tensorrt_llm._torch.attention.backends.fmha import cute_dsl_mla
    from tensorrt_llm._torch.attention.backends.fmha.manager import FmhaManager
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.autotuner import AutoTuner

    monkeypatch.setattr(cute_dsl_mla, "get_sm_version", lambda: sm)
    # AutoTuner.get() builds the singleton from CUDA device properties.
    monkeypatch.setattr(
        AutoTuner, "get", staticmethod(lambda: SimpleNamespace(is_tuning_mode=False))
    )
    # The kernel's can_implement needs nvidia-cutlass-dsl; this test covers only
    # the host-side routing in front of it.
    monkeypatch.setattr(
        CuteDslMlaFmha, "_kernel_can_implement", staticmethod(lambda *args, **kwargs: (True, ""))
    )

    num_tokens = batch_size * seq_len_q
    attn = SimpleNamespace(
        num_heads=num_heads,
        has_fp8_kv_cache=fp8_kv,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_idx=0,
    )
    meta = SimpleNamespace(
        num_contexts=0,
        num_generations=batch_size,
        beam_width=1,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        helix_position_offsets=torch.zeros(num_tokens, dtype=torch.int32) if helix else None,
        _helix_spec_tokens_valid=helix,
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda layer_idx: torch.empty(0, dtype=torch.bfloat16)
        ),
        tokens_per_block=64,
    )
    fwd = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        output=torch.empty(num_tokens, num_heads, 512, dtype=torch.bfloat16),
        softmax_stats_tensor=(
            torch.empty(num_tokens, num_heads, 2, dtype=torch.float32) if helix else None
        ),
    )
    q = torch.empty(num_tokens, num_heads, 576, dtype=torch.bfloat16)

    cute = CuteDslMlaFmha.__new__(CuteDslMlaFmha)
    cute._attn_ref = lambda: attn
    trtllm_gen = _TrtllmGenStub()
    manager = FmhaManager.__new__(FmhaManager)
    manager.fmha_libs = [cute, trtllm_gen]
    selected = manager._select_mla(q, None, None, meta, fwd)
    return "cute_dsl_mla" if selected is cute else "trtllm_gen"


@pytest.mark.parametrize(
    "sm, num_heads, seq_len_q, fp8_kv, helix, expected",
    [
        # Non-helix FP8 verify group at H=96 (DSpark under attention DP): the
        # TRTLLM-Gen MLA decode rejects 64 < H < 128, so it must not be chosen.
        (103, 96, 8, True, False, "cute_dsl_mla"),
        (100, 96, 8, True, False, "cute_dsl_mla"),
        (103, 96, 2, True, False, "cute_dsl_mla"),
        # SM107 has no TRTLLM-Gen FMHA; the request keeps falling through.
        (107, 96, 8, True, False, "trtllm_gen"),
        # Helix H=96 verify groups are unchanged.
        (103, 96, 4, True, True, "cute_dsl_mla"),
        (107, 96, 4, True, True, "cute_dsl_mla"),
        # Single-token H=96 is unchanged (measured perf-table entry).
        (103, 96, 1, True, False, "cute_dsl_mla"),
        # Head counts TRTLLM-Gen serves keep the perf gate.
        (103, 128, 8, True, False, "trtllm_gen"),
        (103, 64, 8, True, False, "trtllm_gen"),
        # BF16 KV is unchanged (H=96 is already admitted for any seq_len_q).
        (103, 96, 8, False, False, "cute_dsl_mla"),
    ],
)
def test_mla_generation_routes_wide_head_fp8_verify_away_from_trtllm_gen(
    monkeypatch, sm, num_heads, seq_len_q, fp8_kv, helix, expected
) -> None:
    # The perf gate admits single-token H=96 FP8 decode only from batch_size 8.
    # The multi-token routes must not depend on the batch size, so they stay
    # below that floor.
    batch_size = 8 if seq_len_q == 1 else 4
    assert (
        _select_mla_generation_fmha(
            monkeypatch,
            sm=sm,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            fp8_kv=fp8_kv,
            helix=helix,
            batch_size=batch_size,
        )
        == expected
    )


def test_mla_generation_single_token_wide_head_fp8_below_perf_floor_falls_through(
    monkeypatch,
) -> None:
    """Below its measured batch floor, single-token H=96 FP8 decode is not a
    CuTe DSL perf win, so selection falls through to the next MLA library."""
    assert (
        _select_mla_generation_fmha(
            monkeypatch,
            sm=103,
            num_heads=96,
            seq_len_q=1,
            fp8_kv=True,
            helix=False,
            batch_size=4,
        )
        == "trtllm_gen"
    )


# kv_cache_block_offsets geometry with an odd page count per sequence, e.g. a
# max_seq_len of 8320 over 128-token pages: each [num_seqs, 2, max_blocks] row
# is 8 * 65 bytes, so an odd row offset leaves an 8-byte aligned slice.
_NUM_POOLS = 3
_MAX_NUM_SEQUENCES = 7
_MAX_BLOCKS_PER_SEQ = 65


@pytest.mark.parametrize(
    "pool_idx, num_contexts, layer_in_pool, slice_is_aligned",
    [
        (0, 0, 0, True),
        # Mixed batch: generation rows start past an odd number of contexts.
        (0, 1, 0, False),
        # A non-first pool behind an odd sequence capacity.
        (1, 0, 0, False),
        (2, 3, 0, False),
        # KVCacheManagerV1 folds the layer slot into a fresh page table.
        (0, 1, 2, False),
    ],
)
def test_generation_page_table_keeps_the_runner_alignment(
    pool_idx: int, num_contexts: int, layer_in_pool: int, slice_is_aligned: bool
) -> None:
    """The decode runner compiles the page table with assumed_align=16."""
    batch_size = 4
    block_offsets = torch.arange(
        _NUM_POOLS * _MAX_NUM_SEQUENCES * 2 * _MAX_BLOCKS_PER_SEQ, dtype=torch.int32
    ).view(_NUM_POOLS, _MAX_NUM_SEQUENCES, 2, _MAX_BLOCKS_PER_SEQ)
    page_table_layer = block_offsets[pool_idx, :, 0, :]
    rows = page_table_layer[num_contexts : num_contexts + batch_size]
    assert (rows.data_ptr() % 16 == 0) == slice_is_aligned

    page_table = CuteDslMlaFmha._generation_page_table(
        page_table_layer, num_contexts, batch_size, layer_in_pool
    )

    assert page_table.data_ptr() % 16 == 0
    assert page_table.dtype == torch.int32
    assert page_table.shape == (_MAX_BLOCKS_PER_SEQ, batch_size)
    assert page_table.stride(0) == 1
    torch.testing.assert_close(page_table, rows.transpose(0, 1) + layer_in_pool)
    if slice_is_aligned and not layer_in_pool:
        # The common decode-only case launches on the metadata buffer itself.
        assert page_table.data_ptr() == rows.data_ptr()


@pytest.mark.parametrize("num_contexts", [0, 1, 3, 4])
def test_generation_kv_lengths_keep_the_runner_alignment(num_contexts: int) -> None:
    sequence_lengths = torch.arange(1, 17, dtype=torch.int32)
    generation_lengths = sequence_lengths[num_contexts:]

    aligned = CuteDslMlaFmha._aligned_kernel_input(generation_lengths)

    assert aligned.data_ptr() % 16 == 0
    torch.testing.assert_close(aligned, generation_lengths)
    if generation_lengths.data_ptr() % 16 == 0:
        assert aligned is generation_lengths


@pytest.mark.parametrize("num_contexts", [0, 1, 3, 4])
def test_decode_kv_lengths_keep_one_pointer_per_step(num_contexts: int) -> None:
    """Every layer of a step passes the same pointer, so the split plan runs once."""
    meta = SimpleNamespace(max_num_requests=16)
    sequence_lengths = torch.arange(1, 17, dtype=torch.int32)
    generation_lengths = sequence_lengths[num_contexts:]

    first_layer = CuteDslMlaFmha._decode_kv_lengths(meta, generation_lengths)
    second_layer = CuteDslMlaFmha._decode_kv_lengths(meta, generation_lengths)

    assert first_layer.data_ptr() % 16 == 0
    assert second_layer.data_ptr() == first_layer.data_ptr()
    torch.testing.assert_close(second_layer, generation_lengths)
    if generation_lengths.data_ptr() % 16 == 0:
        assert first_layer is generation_lengths
        assert not hasattr(meta, "_cute_dsl_mla_kv_lens_staging")


@pytest.mark.parametrize(
    "num_contexts, num_generations, exclusive_page_size, admitted",
    [
        (2, 4, True, True),
        (2, 4, False, False),
        (2, 0, True, False),
    ],
    ids=["mixed_exclusive_page_size", "mixed_shared_page_size", "no_generations"],
)
def test_mixed_batch_gate_admits_only_page_sizes_exclusive_to_cute_dsl(
    num_contexts: int, num_generations: int, exclusive_page_size: bool, admitted: bool
) -> None:
    fmha = SimpleNamespace(_page_size_exclusive_to_cute_dsl=lambda meta: exclusive_page_size)
    meta = SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        beam_width=1,
        helix_position_offsets=None,
        # Stops the check right after the batch-shape gates.
        kv_cache_manager=None,
    )
    fwd = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    # 128 heads stay off the wide-head verify carve-outs; an exclusive page
    # size also skips the perf gate.
    attn = SimpleNamespace(num_heads=128)
    q = torch.empty(max(num_generations, 1), 8)

    supported, reason = CuteDslMlaFmha._is_supported_with_reason(fmha, q, attn, meta, fwd)

    assert not supported
    if admitted:
        assert reason == "KV cache manager is required."
    else:
        assert reason == "CuTe DSL MLA FMHA only supports decode-only batches."


@pytest.mark.parametrize(
    "sm, tokens_per_block, kv_dtype, extra_tokens_per_block, exclusive",
    [
        # Only the FP4 MLA 128-token pages can be exclusive.
        (100, 32, DataType.FP8, (), False),
        (107, 32, DataType.NVFP4, (), False),
        (100, 64, DataType.FP8, (), False),
        (103, 16, DataType.BF16, (), False),
        (107, 64, DataType.FP8, (), False),
        (107, 64, DataType.NVFP4, (), False),
        (100, 256, DataType.FP8, (), False),
        (107, 256, DataType.NVFP4, (), False),
        # FlashInfer TRTLLM-Gen reads the manager's extra page sizes.
        (107, 128, DataType.FP8, (128,), False),
        (100, 128, DataType.NVFP4, (128,), False),
        # PrimTS reads 128-token pages of FP8 or BF16 managers on SM100/SM103.
        (100, 128, DataType.FP8, (), False),
        (103, 128, DataType.BF16, (), False),
        (107, 128, DataType.FP8, (), True),
        # An NVFP4 manager's FP8 fallback layers on the FP4 MLA pages.
        (100, 128, DataType.NVFP4, (), True),
        (103, 128, DataType.NVFP4, (), True),
        (107, 128, DataType.NVFP4, (), True),
    ],
)
def test_page_size_exclusive_to_cute_dsl(
    monkeypatch,
    sm: int,
    tokens_per_block: int,
    kv_dtype: DataType,
    extra_tokens_per_block: tuple,
    exclusive: bool,
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha import cute_dsl_mla

    monkeypatch.setattr(cute_dsl_mla, "get_sm_version", lambda: sm)
    kv_cache_manager = SimpleNamespace(dtype=kv_dtype)
    if extra_tokens_per_block:
        kv_cache_manager.trtllm_gen_extra_tokens_per_block = frozenset(extra_tokens_per_block)
    meta = SimpleNamespace(tokens_per_block=tokens_per_block, kv_cache_manager=kv_cache_manager)

    assert CuteDslMlaFmha._page_size_exclusive_to_cute_dsl(meta) is exclusive


def test_page_size_exclusive_to_cute_dsl_needs_a_page_size() -> None:
    meta = SimpleNamespace(tokens_per_block=None, kv_cache_manager=None)

    assert not CuteDslMlaFmha._page_size_exclusive_to_cute_dsl(meta)
