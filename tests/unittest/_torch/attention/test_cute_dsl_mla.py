# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract tests for the CuTe DSL MLA FMHA library (FP8 scales, split-KV sizing)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


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
    tokens_per_block: int = 64,
    kernel_supported: bool = True,
    num_contexts: int = 0,
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
        CuteDslMlaFmha,
        "_kernel_can_implement",
        staticmethod(lambda *args, **kwargs: (kernel_supported, "")),
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
        num_contexts=num_contexts,
        num_generations=batch_size,
        beam_width=1,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        helix_position_offsets=torch.zeros(num_tokens, dtype=torch.int32) if helix else None,
        _helix_spec_tokens_valid=helix,
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda layer_idx: torch.empty(0, dtype=torch.bfloat16)
        ),
        tokens_per_block=tokens_per_block,
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


@pytest.mark.cpu_only
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
        # Helix verify groups also require CuTe DSL with fewer TP-local heads.
        (103, 48, 6, True, True, "cute_dsl_mla"),
        (100, 48, 6, True, True, "cute_dsl_mla"),
        (103, 48, 6, False, True, "cute_dsl_mla"),
        # The same verify shape without Helix keeps the perf gate.
        (103, 48, 6, True, False, "trtllm_gen"),
        # H=48 page-64 single-token Helix needs the coverage path on SM103.
        (103, 48, 1, True, True, "cute_dsl_mla"),
        # Single-token H=96 is unchanged (measured perf-table entry).
        (103, 96, 1, True, False, "cute_dsl_mla"),
        # Head counts TRTLLM-Gen serves keep the perf gate.
        (103, 128, 8, True, False, "trtllm_gen"),
        (103, 64, 8, True, False, "trtllm_gen"),
        # BF16 KV is unchanged (H=96 is already admitted for any seq_len_q).
        (103, 96, 8, False, False, "cute_dsl_mla"),
    ],
)
def test_mla_generation_routes_verify_groups_to_supported_fmha(
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "page_size, fp8_kv, helix, sm, kernel_supported, expected",
    [
        (64, True, True, 103, True, "cute_dsl_mla"),
        (32, True, True, 103, True, "trtllm_gen"),
        (128, True, True, 103, True, "trtllm_gen"),
        (64, False, True, 103, True, "trtllm_gen"),
        (64, True, False, 103, True, "trtllm_gen"),
        (64, True, True, 100, True, "trtllm_gen"),
        (64, True, True, 103, False, "trtllm_gen"),
    ],
)
def test_mla_generation_helix_fp8_page64_coverage(
    monkeypatch, page_size, fp8_kv, helix, sm, kernel_supported, expected
) -> None:
    assert (
        _select_mla_generation_fmha(
            monkeypatch,
            sm=sm,
            num_heads=48,
            seq_len_q=1,
            fp8_kv=fp8_kv,
            helix=helix,
            batch_size=1,
            tokens_per_block=page_size,
            kernel_supported=kernel_supported,
        )
        == expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "page, seq_len_q, fp8_kv, helix, supported, expected",
    [
        (64, 1, True, True, True, "cute_dsl_mla"),
        (64, 1, True, True, False, "trtllm_gen"),
        (32, 1, True, True, True, "trtllm_gen"),
        (64, 1, False, True, True, "trtllm_gen"),
        (64, 1, True, False, True, "trtllm_gen"),
        (64, 6, True, True, True, "cute_dsl_mla"),
        (32, 6, True, True, True, "cute_dsl_mla"),
    ],
)
def test_mla_generation_mixed_helix_coverage(
    monkeypatch, page, seq_len_q, fp8_kv, helix, supported, expected
) -> None:
    assert (
        _select_mla_generation_fmha(
            monkeypatch,
            sm=103,
            num_heads=48,
            seq_len_q=seq_len_q,
            fp8_kv=fp8_kv,
            helix=helix,
            batch_size=2,
            tokens_per_block=page,
            kernel_supported=supported,
            num_contexts=2,
        )
        == expected
    )


@pytest.mark.cpu_only
def test_mla_generation_mixed_page_table_offset(monkeypatch) -> None:
    page, heads, latent, rope = 64, 48, 512, 64
    cache = torch.zeros((12, 1, page, 1, latent + rope), dtype=torch.float8_e4m3fn)
    blocks = torch.arange(12).view(1, 4, 1, 3)
    lengths = torch.tensor([999, 888, 5, 7], dtype=torch.int32)
    meta = SimpleNamespace(
        num_contexts=2,
        max_num_requests=4,
        helix_position_offsets=torch.zeros(2, dtype=torch.int32),
        _helix_spec_tokens_valid=False,
        kv_cache_manager=SimpleNamespace(get_buffers=lambda _: cache),
        kv_cache_block_offsets=blocks,
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]]),
    )
    attn = SimpleNamespace(
        kv_lora_rank=latent,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=128,
        num_heads=heads,
        q_scaling=1,
        layer_idx=0,
        get_local_layer_idx=lambda _: 0,
    )
    params = SimpleNamespace(
        attn=attn,
        meta=meta,
        num_requests=2,
        tokens_per_block=page,
        sequence_lengths=lengths[2:],
        workspace=torch.empty(0, dtype=torch.int8),
        fwd=SimpleNamespace(
            quant_q_buffer=None,
            mla_bmm1_scale=torch.ones(2, dtype=torch.float32),
            mla_bmm2_scale=torch.ones(1, dtype=torch.float32),
            softmax_stats_tensor=torch.empty((2, heads, 2), dtype=torch.float32),
        ),
    )
    library = CuteDslMlaFmha.__new__(CuteDslMlaFmha)
    monkeypatch.setattr(library, "_required_workspace_size", lambda *args: 0)
    monkeypatch.setattr(library, "_split_kv_plan", lambda *args: None)
    calls = []
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_mla_decode_fp8_blackwell",
        lambda *args: calls.append(args),
        raising=False,
    )
    library._run_mla_decode(
        torch.zeros((2, heads, latent + rope), dtype=torch.bfloat16),
        torch.empty((2, heads, latent), dtype=torch.bfloat16),
        params,
        torch.float8_e4m3fn,
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][4], blocks[0, 2:, 0].T.to(torch.int32))
    torch.testing.assert_close(calls[0][5], lengths[2:])
    assert calls[0][14] is params.fwd.softmax_stats_tensor
