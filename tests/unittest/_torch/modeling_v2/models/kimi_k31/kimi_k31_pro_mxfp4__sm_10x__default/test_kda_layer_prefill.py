# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Kimi K3.1 KDA layer's prefill."""

import math
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("fla")

from kimi_k31_kda_test_utils import KimiKDAReference

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    modeling as kda_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
    K31KdaLayer,
)
from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
    is_kda_optimized_supported,
)
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import (  # noqa: E402
    KimiKDALinearAttention as EngineKimiKDALinearAttention,
)

NUM_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HIDDEN_SIZE = 7168


def _has_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return is_kda_optimized_supported()


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)


def _make_kda(
    source_state_dict: Mapping[str, torch.Tensor] | None = None,
    *,
    num_heads: int = NUM_HEADS,
    layer_cls: type = K31KdaLayer,
) -> K31KdaLayer:
    """The KDA layer (or, with ``layer_cls``, the engine's) with finalized weights and FP32 gate parameters."""
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        kda_fp32_gate_params=True,
    )
    kda = layer_cls(cfg, layer_idx=0).to("cuda")
    # FP32 A_log / dt_bias, as Kimi K3.1 keeps them (the engine layer rounds them to BF16).
    for name in ("A_log", "dt_bias"):
        getattr(kda, name).data = getattr(kda, name).data.float()
    with torch.no_grad():
        # The layer leaves its checkpoint-loaded weights uninitialized: draw the projections and convolutions as
        # nn.Linear / nn.Conv1d initialize theirs.
        for param in kda.parameters():
            if param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        kda.dt_bias.zero_()
    if source_state_dict is not None:
        kda.load_state_dict(source_state_dict)
    kda.finalize_decode_weights()
    return kda


def _make_reference(
    source_state_dict: Mapping[str, torch.Tensor],
    *,
    num_heads: int = NUM_HEADS,
    use_full_rank_gate: bool = True,
) -> KimiKDAReference:
    reference = KimiKDAReference(
        hidden_size=HIDDEN_SIZE,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL_SIZE,
        use_full_rank_gate=use_full_rank_gate,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        dtype=torch.bfloat16,
    ).to("cuda")
    reference.load_state_dict(source_state_dict)
    return reference


def _run_production_prefill(
    attention: K31KdaLayer,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    conv_pool: torch.Tensor | None = None,
    state_pool: torch.Tensor | None = None,
    slot_indices: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, sequence_length, _ = hidden_states.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch_size + 1, device="cuda", dtype=torch.long) * sequence_length
    else:
        batch_size = cu_seqlens.numel() - 1

    projection_size = attention.proj_size
    if conv_pool is None:
        conv_pool = torch.zeros(
            batch_size,
            3 * projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
    if state_pool is None:
        state_pool = torch.zeros(
            batch_size,
            attention.num_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
    if slot_indices is None:
        slot_indices = torch.arange(batch_size, device="cuda", dtype=torch.long)
    use_initial_states = has_initial_states is not None
    if has_initial_states is None:
        has_initial_states = torch.zeros(batch_size, device="cuda", dtype=torch.bool)
    # The packed prefill schedule the model metadata prepares once per step: longest first, 16-row chunks.
    lengths = cu_seqlens.diff().tolist()
    order = sorted(range(batch_size), key=lambda i: (-lengths[i], i))
    cu_chunks = [0]
    for length in lengths:
        cu_chunks.append(cu_chunks[-1] + math.ceil(length / 16))
    metadata = SimpleNamespace(
        use_initial_states=use_initial_states,
        has_initial_states=has_initial_states,
        state_indices=slot_indices.to(torch.int32),
        # causal_conv1d wants int32 offsets, unlike the int64 cu_seqlens above.
        query_start_loc=cu_seqlens.to(torch.int32),
        kda_max_seqlen=max(lengths),
        kda_seq_order=torch.tensor(order, device="cuda", dtype=torch.int32),
        kda_cu_chunks=torch.tensor(cu_chunks, device="cuda", dtype=torch.int32),
    )
    core = attention.forward_prefill(
        hidden_states.reshape(-1, HIDDEN_SIZE),
        cu_seqlens.to(torch.int64),
        metadata,
        batch_size,
        conv_pool,
        state_pool,
        slot_indices,
    )
    output = attention._project_output(core)
    return output.reshape_as(hidden_states)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@torch.no_grad()
@pytest.mark.parametrize("num_heads", [NUM_HEADS, 16], ids=["h96", "h16"])
def test_optimized_prefill_matches_fla_reference(num_heads: int) -> None:
    torch.manual_seed(0)
    use_full_rank_gate = True
    optimized = _make_kda(num_heads=num_heads)
    reference = _make_reference(
        optimized.state_dict(), num_heads=num_heads, use_full_rank_gate=use_full_rank_gate
    )

    # Reuse the same module across changing token counts and batch sizes.
    for batch_size, sequence_length in [(2, 256), (2, 512), (1, 1024)]:
        hidden_states = (
            torch.randn(
                batch_size,
                sequence_length,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = _run_production_prefill(optimized, hidden_states)
        expected = reference.forward_prefill(hidden_states)
        _assert_close(actual, expected)

    hidden_states = torch.randn(1, 300, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    actual = _run_production_prefill(optimized, hidden_states)
    expected = reference.forward_prefill(hidden_states)
    _assert_close(actual, expected)

    sequence_lengths = [128, 256, 192]
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(
            1,
            sum(sequence_lengths),
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


@torch.no_grad()
def test_kda_mixer_empty_prefill():
    """An empty token payload still publishes the conv state for its slot."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    # Seed the pool with a sentinel: the output shape alone would still match if the
    # zero-token state write-back were skipped, so only the pool can catch that.
    conv_pool = torch.full(
        (1, 3 * NUM_HEADS * HEAD_DIM, CONV_KERNEL_SIZE - 1),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = _run_production_prefill(optimized, hidden_states, conv_pool=conv_pool)
    assert out.shape == (1, 0, HIDDEN_SIZE)
    # Without an initial state a zero-token sequence leaves an all-zero causal window.
    assert torch.equal(conv_pool, torch.zeros_like(conv_pool))


@torch.no_grad()
def test_kda_mixer_empty_prefill_preserves_initial_conv_state():
    """A zero-token payload carrying an initial state leaves that state untouched."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    conv_seed = (
        torch.randn(
            1,
            3 * NUM_HEADS * HEAD_DIM,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    conv_pool = conv_seed.clone()
    out = _run_production_prefill(
        optimized,
        hidden_states,
        conv_pool=conv_pool,
        has_initial_states=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    assert out.shape == (1, 0, HIDDEN_SIZE)
    # No token shifts into the window, so the write-back must republish it verbatim.
    assert torch.equal(conv_pool, conv_seed)


@torch.no_grad()
def test_kda_prefill_never_selects_small_bh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The layer's prefill pins the BT16 routing: the prefill op's "auto" mode can pick Small-BH, which is wrong on
    sm_107 at 96 heads."""
    modes = []
    prefill = kda_module.kda_prefill

    def _capturing_prefill(*args):
        modes.append(args[12])
        return prefill(*args)

    monkeypatch.setattr(kda_module, "kda_prefill", _capturing_prefill)
    hidden_states = torch.randn(1, 40, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    _run_production_prefill(_make_kda(num_heads=2), hidden_states)
    assert modes == ["bt16"]


@torch.no_grad()
def test_kda_prefill_op_partial_final_chunk_large_batch():
    """Large packed batches with short final sequences match the FLA reference.

    Cover the autotuner warmup split [8191, 1] and a mixed-length 8192-token batch.
    """
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
    for sequence_lengths in ([8191, 1], [8000, 150, 42]):
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=torch.long,
            device="cuda",
        )
        hidden_states = (
            torch.randn(
                1,
                sum(sequence_lengths),
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        _assert_close(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize(
    "sequence_lengths",
    ([1], [30], [65], [6, 12], [64, 64], [1, 2, 3], [64, 64, 63], [6, 12, 20, 25]),
    ids=(
        "single-token",
        "short-single-seq",
        "partial-single-seq",
        "two-short-seqs",
        "two-aligned-seqs",
        "three-short-seqs",
        "three-seqs-partial-tail",
        "four-short-seqs",
    ),
)
def test_kda_prefill_small_varlen_stays_on_optimized_op(
    monkeypatch: pytest.MonkeyPatch,
    sequence_lengths: list[int],
) -> None:
    """Short packed prompts stay on the optimized prefill path.

    Routing these short-prompt shapes to FLA instead would make a real serving
    run pay FLA's cold Triton autotune on its first short context.
    """
    import fla.ops.kda

    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(
            1,
            sum(sequence_lengths),
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )

    # The FLA fallback resolves ``chunk_kda`` at call time, so this counts it;
    # the reference binds it at import time and is unaffected.
    fla_spy = MagicMock(side_effect=fla.ops.kda.chunk_kda)
    monkeypatch.setattr(fla.ops.kda, "chunk_kda", fla_spy)
    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    assert fla_spy.call_count == 0, (
        f"prefill fell back to FLA for sequence lengths {sequence_lengths}; "
        "the optimized op must handle short packed prompts"
    )

    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    "misaligned_indices", [False, True], ids=["aligned-indices", "offset-indices"]
)
@torch.no_grad()
def test_kda_prefill_state_pool_matches_fallback_with_mixed_initial_states(
    monkeypatch: pytest.MonkeyPatch,
    misaligned_indices: bool,
) -> None:
    """Aligned and offset int32 slot views both match the explicit FLA fallback."""
    torch.manual_seed(3)
    optimized = _make_kda()
    # The engine's KDA layer without the in-tree prefill op runs the FLA prefill.
    monkeypatch.setattr(_kda_kernels, "is_intree_prefill_available", lambda: False)
    fallback = _make_kda(
        source_state_dict=optimized.state_dict(), layer_cls=EngineKimiKDALinearAttention
    )
    assert fallback._dispatch.prefill_kernel_path == "fla"

    sequence_lengths = [64, 64, 64, 64]
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(1, sum(sequence_lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    slots = 5
    slot_indices = torch.tensor([3, 0, 4, 1], dtype=torch.int32, device="cuda")
    if misaligned_indices:
        slot_storage = torch.tensor([-1, 3, 0, 4, 1], dtype=torch.int32, device="cuda")
        slot_indices = slot_storage[1:]
        assert slot_indices.is_contiguous()
        assert slot_indices.data_ptr() % 16 != 0
    has_initial_states = torch.tensor([True, False, True, False], device="cuda")
    projection_size = NUM_HEADS * HEAD_DIM
    conv_seed = (
        torch.randn(
            slots,
            3 * projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    state_seed = (
        torch.randn(
            slots,
            NUM_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.05
    )

    optimized_conv, optimized_state = conv_seed.clone(), state_seed.clone()
    optimized_output = _run_production_prefill(
        optimized,
        hidden_states,
        cumulative_lengths,
        conv_pool=optimized_conv,
        state_pool=optimized_state,
        slot_indices=slot_indices,
        has_initial_states=has_initial_states,
    )
    fallback_conv, fallback_state = conv_seed.clone(), state_seed.clone()
    fallback_output = _run_production_prefill(
        fallback,
        hidden_states,
        cumulative_lengths,
        conv_pool=fallback_conv,
        state_pool=fallback_state,
        slot_indices=slot_indices,
        has_initial_states=has_initial_states,
    )

    _assert_close(fallback_output, optimized_output)
    _assert_close(
        fallback_conv.index_select(0, slot_indices),
        optimized_conv.index_select(0, slot_indices),
    )
    _assert_close(
        fallback_state.index_select(0, slot_indices),
        optimized_state.index_select(0, slot_indices),
    )


@torch.no_grad()
def test_kda_prefill_op_shape_growth_and_cu_dtype_transitions():
    """Changing token counts and caller offset dtypes preserve reference parity.

    The production adapter prepares int64 packed offsets for FlashInfer from
    either caller dtype. Repeated growth and shrinkage exercise cached kernels.
    """
    torch.manual_seed(0)
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
    cases = [
        ([517, 654], torch.long),  # small batch, int64 cu (dump-replay-like)
        ([8191], torch.int32),  # growth + dtype conversion
        ([1171], torch.long),  # shrink + flip back
        ([150, 900, 333, 640], torch.int32),  # multi-seq int32
    ]
    for sequence_lengths, cu_dtype in cases:
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=cu_dtype,
            device="cuda",
        )
        hidden_states = (
            torch.randn(
                1,
                sum(sequence_lengths),
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        _assert_close(actual, expected)
