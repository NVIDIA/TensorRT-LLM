# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap
from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeAlias

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_multi_ctas_kv_counter_size,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import _CuteDslMlaStagingKey
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.trtllm import (
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.autotuner import AutoTuner


class _AttentionStub:
    def __init__(
        self,
        *,
        is_mla_enable: bool,
        has_fp8_kv_cache: bool,
        flashinfer_mla_backend: str | None = None,
    ) -> None:
        self.is_mla_enable = is_mla_enable
        self.has_fp8_kv_cache = has_fp8_kv_cache
        self.flashinfer_mla_backend = flashinfer_mla_backend
        self.kv_lora_rank = 512 if is_mla_enable else None
        self.head_dim = 576
        self.v_head_dim = 512 if is_mla_enable else None
        self.qk_rope_head_dim = 64 if is_mla_enable else None
        self.rope_append = True if is_mla_enable else None

    def out_head_size(self, is_gen_only: bool) -> int:
        return TrtllmAttention.out_head_size(self, is_gen_only)


_MlaBackendPolicy: TypeAlias = Callable[[str, SimpleNamespace, int], str]


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


@pytest.mark.parametrize("value, expected", [(None, 0), (0, 0), (8192, 8192)])
def test_attention_chunk_size_uses_zero_for_native_disabled_value(
    value: int | None,
    expected: int,
) -> None:
    assert FlashInferTrtllmGenFmha._get_attention_chunk_size(value) == expected


def test_prepare_workspace_sizes_counter_for_max_num_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_heads, max_num_requests, beam_width, sm_count = 6, 16, 2, 148
    max_num_sequences = max_num_requests * beam_width

    def check_counter_size_args(
        actual_num_heads: int,
        actual_max_num_sequences: int,
        actual_sm_count: int,
    ) -> int:
        assert (actual_num_heads, actual_max_num_sequences, actual_sm_count) == (
            num_heads,
            max_num_sequences,
            sm_count,
        )
        raise RuntimeError("counter size arguments observed")

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen."
        "_get_multi_ctas_kv_counter_size",
        check_counter_size_args,
    )

    fmha = SimpleNamespace(
        attn=SimpleNamespace(num_heads=num_heads),
        _multi_processor_count=sm_count,
    )
    metadata = SimpleNamespace(
        max_num_requests=max_num_requests,
        beam_width=beam_width,
        max_num_sequences=max_num_sequences,
    )
    with pytest.raises(RuntimeError, match="counter size arguments observed"):
        FlashInferTrtllmGenFmha.prepare_workspace(
            fmha,
            params=SimpleNamespace(
                qkv_or_q=SimpleNamespace(),
                fwd=SimpleNamespace(),
                workspace=SimpleNamespace(),
            ),
            metadata=metadata,
        )


def test_flashinfer_cute_dsl_mla_backend_rejects_fp8_kv_cache() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=True,
        flashinfer_mla_backend="cute-dsl",
    )

    with pytest.raises(ValueError, match="does not support FP8 KV cache"):
        FlashInferTrtllmGenFmha(attn)


@pytest.mark.parametrize("configured_backend", ["cute-dsl", "trtllm-gen"])
def test_standalone_cute_dsl_mla_defers_to_explicit_flashinfer_backend(
    configured_backend: str,
) -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=False,
        flashinfer_mla_backend=configured_backend,
    )

    assert not CuteDslMlaFmha.is_available(attn)


def _cute_dsl_mla_helix_support(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seq_len_q: int = 1,
    softmax_stats: torch.Tensor | None,
) -> tuple[bool, str]:
    batch_size, num_heads = 2, 96
    q = torch.empty(
        (batch_size * seq_len_q, num_heads * (512 + 64)),
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (batch_size * seq_len_q, num_heads * 512),
        dtype=torch.bfloat16,
    )
    attn = SimpleNamespace(
        num_heads=num_heads,
        has_fp8_kv_cache=False,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_idx=0,
    )
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=batch_size,
        beam_width=1,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        helix_position_offsets=torch.zeros(batch_size, dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda _layer_idx: torch.empty(0, dtype=torch.bfloat16)
        ),
        tokens_per_block=64,
    )
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        softmax_stats_tensor=softmax_stats,
    )

    monkeypatch.setattr(
        AutoTuner,
        "get",
        classmethod(lambda _cls: SimpleNamespace(is_tuning_mode=False)),
    )
    monkeypatch.setattr(
        CuteDslMlaFmha,
        "_kernel_can_implement",
        staticmethod(lambda *_args: (True, "")),
    )
    fmha = object.__new__(CuteDslMlaFmha)
    return fmha._is_supported_with_reason(q, attn, metadata, forward_args)


def test_cute_dsl_mla_accepts_single_token_helix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = torch.empty((2, 96, 2), dtype=torch.float32)

    supported, reason = _cute_dsl_mla_helix_support(monkeypatch, softmax_stats=stats)

    assert supported, reason


@pytest.mark.parametrize(
    ("seq_len_q", "softmax_stats", "reason"),
    [
        (2, torch.empty((4, 96, 2), dtype=torch.float32), "single-token decode"),
        (1, None, "requires softmax_stats_tensor"),
        (1, torch.empty((2, 95, 2), dtype=torch.float32), "shape"),
        (1, torch.empty((2, 96, 2), dtype=torch.bfloat16), "float32"),
    ],
)
def test_cute_dsl_mla_rejects_invalid_helix_contract(
    monkeypatch: pytest.MonkeyPatch,
    seq_len_q: int,
    softmax_stats: torch.Tensor | None,
    reason: str,
) -> None:
    supported, actual_reason = _cute_dsl_mla_helix_support(
        monkeypatch,
        seq_len_q=seq_len_q,
        softmax_stats=softmax_stats,
    )

    assert not supported
    assert reason in actual_reason


def test_flashinfer_mla_backend_defaults_to_trtllm_gen() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=False,
    )

    assert FlashInferTrtllmGenFmha(attn)._mla_backend == "trtllm-gen"


def test_mla_scheduler_invalidation_resets_cute_dsl_staging_key() -> None:
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata._mla_scheduler_buffers_valid = True
    metadata._mla_ctx_cu_seqlens_valid = True
    metadata._cute_dsl_mla_staging_key = _CuteDslMlaStagingKey(
        is_capturing=True,
        workspace_ptr=1,
        block_tables_ptr=2,
        block_tables_shape=(3, 4),
        sequence_lengths_ptr=5,
        sequence_lengths_offset=6,
        batch_beam=7,
        padded_num_pages=8,
    )

    metadata._invalidate_mla_scheduler_buffers()

    assert not metadata._mla_scheduler_buffers_valid
    assert not metadata._mla_ctx_cu_seqlens_valid
    assert metadata._cute_dsl_mla_staging_key is None


def test_flashinfer_mla_backend_rejects_unknown_backend() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=False,
        flashinfer_mla_backend="cutedsl",
    )

    with pytest.raises(ValueError, match="flashinfer_mla_backend must be one of"):
        FlashInferTrtllmGenFmha(attn)


def _make_fmha(
    requested_backend: str,
    mla_backend_policy: _MlaBackendPolicy | None,
) -> FlashInferTrtllmGenFmha:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._mla_backend = requested_backend
    # ``Fmha.attn`` is a read-only property that dereferences ``_attn_ref``
    # (normally a weakref to the owning TrtllmAttention). SimpleNamespace is
    # not weak-referenceable, so stand in with a closure of the same shape.
    attn = SimpleNamespace(mla_backend_policy=mla_backend_policy)
    fmha._attn_ref = lambda: attn
    return fmha


@pytest.mark.parametrize("requested_backend", ["cute-dsl", "trtllm-gen"])
@pytest.mark.parametrize(
    ("num_contexts", "num_generations", "num_gen_tokens"),
    [
        (0, 4, 4),  # generation-only, one token per request
        (1, 3, 3),  # mixed context/generation batch
        (0, 4, 8),  # multi-token generation (speculative verification)
    ],
)
def test_flashinfer_mla_backend_default_matches_static_selection(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_gen_tokens: int,
) -> None:
    """Without an installed policy the static backend is used for every batch
    composition, matching the behavior before the policy hook existed."""
    fmha = _make_fmha(requested_backend, mla_backend_policy=None)

    assert (
        fmha._get_effective_mla_backend(
            SimpleNamespace(
                num_contexts=num_contexts,
                num_generations=num_generations,
            ),
            num_gen_tokens,
        )
        == requested_backend
    )


def test_flashinfer_mla_backend_policy_hook_is_consulted() -> None:
    calls: list[tuple[str, SimpleNamespace, int]] = []

    def policy(
        requested_backend: str,
        meta: SimpleNamespace,
        num_gen_tokens: int,
    ) -> str:
        calls.append((requested_backend, meta, num_gen_tokens))
        return "trtllm-gen"

    fmha = _make_fmha("cute-dsl", mla_backend_policy=policy)
    meta = SimpleNamespace(num_contexts=0, num_generations=4)

    assert fmha._get_effective_mla_backend(meta, 4) == "trtllm-gen"
    assert calls == [("cute-dsl", meta, 4)]


# The six tests below guard the MLA generation perf gate that #15300 removed as
# refactoring collateral, costing ~3% output token throughput on DeepSeek-V3-family
# and Kimi-K2 MLA decode at the default tokens_per_block. They deliberately call the
# checker instead of asserting on SLOWER_MLA_GENERATION_KERNELS itself: a test that
# pins the literal set would be deleted along with the constant by the next
# mechanical refactor, whereas these turn a dropped parameter into a TypeError and a
# dropped constant into an AttributeError.


def test_mla_generation_declines_slower_trtllm_gen_decode_kernel() -> None:
    # DeepSeek-V3 / Kimi-K2 shape at the default tokens_per_block=32: the trtllm-gen
    # MLA decode kernel is slower here than the thop.attention fallback, so this
    # backend must decline and let selection fall through.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=32,
        mla_backend="trtllm-gen",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert not supported
    assert "slower" in reason
    assert "headDimQk=576" in reason
    assert "headDimV=512" in reason
    assert "tokens_per_block=32" in reason


@pytest.mark.parametrize("tokens_per_block", [16, 64])
def test_mla_generation_gate_is_scoped_to_one_page_size(tokens_per_block: int) -> None:
    # The gate must stay narrow: the same head dims at other page sizes are still
    # served by this backend. Real configs run tokens_per_block=64.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=tokens_per_block,
        mla_backend="trtllm-gen",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert supported, reason
    assert reason == ""


def test_mla_generation_gate_is_scoped_to_the_trtllm_gen_backend() -> None:
    # The gated kernel is the trtllm-gen one; the cute-dsl MLA decode path shares
    # this class and these head dims, and must stay selectable.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=32,
        mla_backend="cute-dsl",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert supported, reason
    assert reason == ""


def test_mla_generation_gate_declines_a_policy_downgrade_to_trtllm_gen() -> None:
    # A cute-dsl config whose per-batch policy downgrades to trtllm-gen (K3 does so
    # for mixed batches and for speculative verification) still runs the gated
    # kernel, so the gate must fire on the *effective* backend. Reading the static
    # self._mla_backend here would let the slower kernel through.
    fmha = _make_fmha("cute-dsl", mla_backend_policy=lambda *_: "trtllm-gen")
    meta = SimpleNamespace(num_contexts=1, num_generations=3)
    effective = fmha._get_effective_mla_backend(meta, 3)
    assert effective == "trtllm-gen"

    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=32,
        mla_backend=effective,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert not supported
    assert "slower" in reason


def test_mla_generation_gate_reads_the_effective_mla_backend() -> None:
    # The composition above is only load-bearing if the production call site feeds
    # the gate the effective backend. Assert that structurally: reverting to the
    # static self._mla_backend is a one-word change, and no behavioural test here
    # would catch it because driving _is_supported_with_reason needs a full
    # metadata/forward-args stub.
    source = textwrap.dedent(inspect.getsource(FlashInferTrtllmGenFmha._is_supported_with_reason))
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_check_mla_generation_support"
    ]
    assert len(calls) == 1, "expected exactly one MLA generation gate call site"
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    passed = kwargs.get("mla_backend")
    assert passed is not None, "gate call site lost its mla_backend argument"
    assert (
        isinstance(passed, ast.Call)
        and getattr(passed.func, "attr", None) == "_get_effective_mla_backend"
    ), f"gate must receive the effective backend, got {ast.dump(passed)}"


def test_mla_generation_allows_other_supported_head_dims() -> None:
    # (320, 256) is unaffected at every page size.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=320,
        tokens_per_block=32,
        mla_backend="trtllm-gen",
        kv_lora_rank=256,
        qk_rope_head_dim=64,
    )
    assert supported, reason
    assert reason == ""
