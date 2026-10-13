# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity coverage for host-derived KDA prefill metadata."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import (
    KimiK3MambaMetadata,  # noqa: E402
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (  # noqa: E402
    BaseMambaCacheManager,
)

NUM_HEADS = 6
HEAD_DIM = 128
LOWER_BOUND = -5.0


def _has_supported_gpu() -> bool:
    # SM100/SM103 (Blackwell) and SM107 (Rubin) -- must track
    # _kda_kernels.is_kda_optimized_supported(), which accepts all three.
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in {(10, 0), (10, 3), (10, 7)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell SM100/SM103 or Rubin SM107",
)


@pytest.mark.parametrize("on_device", [False, True])
def test_replay_metadata_copies_counts_and_clears_padding(on_device: bool) -> None:
    counts = [4, 2, 2, 1, 0, 3]
    previous_lengths = torch.tensor([2, 99, 1, 3], dtype=torch.int32, device="cuda")
    # Reuse a view of the engine's slot buffer. Slot zero is valid, while
    # recycled slot one contains stale data and is excluded from the view.
    slot_buffer = torch.tensor([99, 3, 0, 2, 99], dtype=torch.int32, device="cuda")
    previous_slots = slot_buffer[1:4]
    state_indices = torch.arange(8, dtype=torch.int32, device="cuda")
    manager = Mock(spec=BaseMambaCacheManager)
    manager.use_kda_replay_update = True
    manager.get_state_indices.side_effect = lambda request_ids, is_padding: state_indices[
        : len(request_ids)
    ]
    attn_metadata = AttentionMetadata(
        max_num_requests=8,
        max_num_tokens=24,
        mamba_chunk_size=64,
        mamba_metadata_cls=KimiK3MambaMetadata,
        num_contexts=0,
        seq_lens=torch.ones(6, dtype=torch.int32),
        request_ids=[10, 11, 12, 13, 14, 15],
        kv_cache_manager=manager,
    )
    # Initialize through the general attention path before checking that
    # subsequent input preparation avoids synchronization.
    attn_metadata.prepare()
    metadata = attn_metadata.mamba_metadata
    old_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attn_metadata.prepare()
        metadata.prepare_prev_num_accepted_tokens(
            previous_slots=previous_slots if on_device else None,
            # Stale host counts for rows with slots must lose to GPU results.
            accepted_draft_tokens=[4, 2, 9, 9, 9, 3] if on_device else counts,
            previous_new_tokens_lens=previous_lengths if on_device else None,
            generation_start=2 if on_device else 0,
        )
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    assert metadata.prev_num_accepted_tokens.tolist() == [4, 2, 2, 1, 0, 3, 0, 0]

    # A later, smaller batch must clear replay counts left in padding rows.
    attn_metadata.request_ids = [10, 15]
    attn_metadata.seq_lens = attn_metadata.seq_lens[:2]
    attn_metadata.prepare()
    metadata.prepare_prev_num_accepted_tokens(
        previous_slots=previous_slots[:0] if on_device else None,
        accepted_draft_tokens=[1, 2],
        previous_new_tokens_lens=previous_lengths if on_device else None,
    )
    assert metadata.prev_num_accepted_tokens.tolist() == [1, 2, 0, 0, 0, 0, 0, 0]


@pytest.fixture(scope="module")
def dispatch_pair() -> tuple[KDAKernelDispatch, KDAKernelDispatch]:
    optimized = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)
    reference = KDAKernelDispatch(use_optimized_prefill=False, use_optimized_decode=False)
    assert optimized.prefill_kernel_path == "optimized"
    assert reference.prefill_kernel_path == "fla"
    return optimized, reference


@pytest.fixture(scope="module")
def gate_params() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    a_log = torch.randn(NUM_HEADS, generator=generator, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = (
        torch.randn(
            NUM_HEADS * HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )
    return a_log, dt_bias


def _make_inputs(total_tokens: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random_tensor(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32, device="cuda").to(
            dtype
        )

    q = random_tensor(1, total_tokens, NUM_HEADS, HEAD_DIM)
    k = random_tensor(1, total_tokens, NUM_HEADS, HEAD_DIM)
    v = random_tensor(1, total_tokens, NUM_HEADS, HEAD_DIM)
    g = random_tensor(1, total_tokens, NUM_HEADS, HEAD_DIM)
    beta = random_tensor(1, total_tokens, NUM_HEADS, dtype=torch.float32)
    return q, k, v, g, beta


def _run(
    dispatch: KDAKernelDispatch,
    gate_params: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor,
    *,
    chunk_indices: torch.Tensor | None = None,
    varlen_is_aligned: bool | None = None,
    single_sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q, k, v, g, beta = inputs
    a_log, dt_bias = gate_params
    return dispatch.prefill_chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=a_log,
        dt_bias=dt_bias,
        scale=HEAD_DIM**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        varlen_is_aligned=varlen_is_aligned,
        single_sequence_length=single_sequence_length,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@pytest.mark.parametrize(
    "sequence_lengths",
    ([128, 256, 192], [100, 257, 219], [300]),
    ids=("aligned", "mixed", "single_unaligned"),
)
@torch.no_grad()
def test_host_metadata_bypasses_device_inference(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    sequence_lengths: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fla.ops.utils.index as fla_index

    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops

    optimized, reference = dispatch_pair
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    cache_key = id(cu_seqlens)
    assert cache_key not in cute_dsl_kimi_k3_custom_ops._varlen_pure_cache
    assert cache_key not in cute_dsl_kimi_k3_custom_ops._varlen_single_seqlen_cache

    inputs = _make_inputs(sum(sequence_lengths), seed=700 + len(sequence_lengths))
    chunk_indices = torch.tensor(
        [
            [sequence_index, chunk_index]
            for sequence_index, length in enumerate(sequence_lengths)
            for chunk_index in range((length + 63) // 64)
        ],
        dtype=torch.long,
        device="cuda",
    )
    varlen_is_aligned = all(length % 64 == 0 for length in sequence_lengths)
    single_sequence_length = sequence_lengths[0] if len(sequence_lengths) == 1 else None
    with monkeypatch.context() as context:
        context.setattr(
            fla_index,
            "prepare_chunk_indices",
            lambda *_args, **_kwargs: pytest.fail(
                "host metadata should bypass FLA chunk-index preparation"
            ),
        )
        actual_output, actual_state = _run(
            optimized,
            gate_params,
            inputs,
            cu_seqlens,
            chunk_indices=chunk_indices,
            varlen_is_aligned=varlen_is_aligned,
            single_sequence_length=single_sequence_length,
        )
    reference_output, reference_state = _run(
        reference,
        gate_params,
        inputs,
        cu_seqlens.clone(),
    )

    assert actual_state is not None
    assert reference_state is not None
    _assert_close(actual_output, reference_output)
    _assert_close(actual_state, reference_state)
    assert cache_key not in cute_dsl_kimi_k3_custom_ops._varlen_pure_cache
    assert cache_key not in cute_dsl_kimi_k3_custom_ops._varlen_single_seqlen_cache
