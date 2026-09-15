# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bf16 KDA recurrent-state pool must round-trip through the fp32-only paths.

The fused T=1 decode kernel and the indexed prefill kernel read and write the
state in fp32 only, so a bf16 pool goes through an fp32 copy of the addressed
rows: plain decode takes ``forward_decode_fallback`` (gather, widen, run the
kernel, round back) and prefill stages the rows for the same indexed kernel the
fp32 pool uses. The fp32 pool is the reference in both cases: decode must agree
up to bf16 rounding of the committed state, prefill must be bit-identical
(same kernel, same fp32 inputs) with the state rounded once, and rows the batch
does not address must stay untouched.

Needs 1 GPU + fla-core; runs with random weights (no checkpoint).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
    is_intree_prefill_available,
    is_kda_optimized_supported,
)

HIDDEN_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 128  # the fused decode kernel only supports 128
CONV_WIDTH = 4


class _Cfg:
    hidden_size = HIDDEN_SIZE
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "short_conv_kernel_size": CONV_WIDTH,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


class _LayerCache:
    """Only what the plain-decode path reads; no fused-verify replay caches."""

    kda_kg_cache = None


class _MambaMetadata:
    def __init__(self, max_batch_size: int, device: str) -> None:
        self._arange_buffer = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)


def _make_runtime() -> KimiKDALinearAttention:
    if not torch.cuda.is_available() or not is_kda_optimized_supported():
        pytest.skip("the fused KDA decode kernel needs a supported CUDA device")
    runtime = KimiKDALinearAttention(_Cfg(), layer_idx=0).to("cuda")
    # dt_bias expects a checkpoint; uninitialized values overflow the decay.
    torch.nn.init.normal_(runtime.dt_bias, std=0.1)
    runtime.finalize_decode_weights()
    if runtime._qkvg_proj_weight is None:
        pytest.skip("decode fast path unavailable for this build")
    return runtime


def _decode_steps(runtime, x, conv_pool, ssm_pool, slot_indices):
    metadata = _MambaMetadata(conv_pool.shape[0], "cuda")
    return [
        runtime.forward_decode(
            x[:, step],
            conv_pool,
            ssm_pool,
            slot_indices.long(),
            metadata,
            _LayerCache(),
            ssm_state_indices=slot_indices,
        )
        for step in range(x.shape[1])
    ]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


@torch.no_grad()
def test_bf16_pool_decode_matches_fp32_pool():
    torch.manual_seed(0)
    runtime = _make_runtime()
    batch, slots, steps = 3, 6, 4
    dim = NUM_HEADS * HEAD_DIM
    slot_indices = torch.tensor([4, 1, 5], dtype=torch.int32, device="cuda")
    addressed = slot_indices.long()
    untouched = torch.tensor([0, 2, 3], device="cuda")
    # The live pool holds the W - 1 raw inputs of each short convolution.
    conv_pool = torch.randn(slots, 3 * dim, CONV_WIDTH - 1, dtype=torch.bfloat16, device="cuda")
    conv_pool *= 0.05
    ssm_bf16 = (torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    ssm_fp32 = ssm_bf16.float()
    initial = ssm_bf16.clone()
    x = torch.randn(batch, steps, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1

    expected = _decode_steps(runtime, x, conv_pool.clone(), ssm_fp32, slot_indices)
    actual = _decode_steps(runtime, x, conv_pool.clone(), ssm_bf16, slot_indices)

    for step, (got, want) in enumerate(zip(actual, expected)):
        assert _relative_error(got, want) < 2e-2, f"decode output diverged at step {step}"
    assert not torch.equal(ssm_bf16[addressed], initial[addressed]), "state was not written back"
    assert _relative_error(ssm_bf16[addressed], ssm_fp32[addressed]) < 1e-2
    torch.testing.assert_close(ssm_bf16[untouched], initial[untouched], rtol=0, atol=0)


def _prefill_metadata(
    sequence_lengths, use_initial_states, has_initial_states, cu_seqlens, slot_indices
):
    """The Kimi K3 runtime metadata forward_prefill reads (chunk indices precomputed)."""
    chunk_indices = torch.tensor(
        [
            [sequence_index, chunk_index]
            for sequence_index, length in enumerate(sequence_lengths)
            for chunk_index in range((length + 63) // 64)
        ],
        device="cuda",
        dtype=torch.long,
    )
    return SimpleNamespace(
        use_initial_states=use_initial_states,
        has_initial_states=torch.tensor(has_initial_states, device="cuda", dtype=torch.bool),
        kda_chunk_indices=chunk_indices,
        kda_varlen_is_aligned=all(length % 64 == 0 for length in sequence_lengths),
        kda_single_sequence_length=(sequence_lengths[0] if len(sequence_lengths) == 1 else None),
        # the packed causal convolution takes int32 sequence starts and the conv-pool slots
        query_start_loc=cu_seqlens.to(torch.int32),
        state_indices=slot_indices,
        # the runtime metadata's preallocated 0..max_batch range (staged rows are dense)
        _arange_buffer=torch.arange(len(sequence_lengths) + 1, dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize(
    "sequence_lengths,use_initial_states,has_initial_states",
    [
        ([17, 31, 64], False, [False, False, False]),
        ([1, 129], True, [True, False]),
    ],
)
@torch.no_grad()
def test_bf16_pool_prefill_matches_fp32_pool(
    sequence_lengths, use_initial_states, has_initial_states, monkeypatch
):
    torch.manual_seed(0)
    runtime = _make_runtime()
    if not is_intree_prefill_available() or runtime._dispatch.prefill_kernel_path != "optimized":
        pytest.skip("the indexed KDA prefill kernel is unavailable")
    num_prefills = len(sequence_lengths)
    num_tokens = sum(sequence_lengths)
    slots = num_prefills + 3
    dim = NUM_HEADS * HEAD_DIM
    slot_indices = torch.arange(2, 2 + num_prefills, device="cuda", dtype=torch.int32)
    addressed = slot_indices.long()
    untouched = torch.tensor([0, 1, slots - 1], device="cuda")
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()], device="cuda", dtype=torch.long
    )
    metadata = _prefill_metadata(
        sequence_lengths, use_initial_states, has_initial_states, cu_seqlens, slot_indices
    )
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    conv_seed = torch.randn(slots, 3 * dim, CONV_WIDTH - 1, dtype=torch.bfloat16, device="cuda")
    conv_seed *= 0.02
    ssm_bf16 = (torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.01).to(
        torch.bfloat16
    )
    # The reference pool holds the same bf16-representable values in fp32.
    ssm_fp32 = ssm_bf16.float()
    initial = ssm_bf16.clone()

    indexed_calls = []
    prefill_chunk_kda = runtime._dispatch.prefill_chunk_kda

    def _record(**kwargs):
        indexed_calls.append(kwargs["state_pool"] is not None)
        return prefill_chunk_kda(**kwargs)

    monkeypatch.setattr(runtime._dispatch, "prefill_chunk_kda", _record)

    ref_conv = conv_seed.clone()
    expected = runtime.forward_prefill(
        hidden, cu_seqlens, metadata, num_prefills, ref_conv, ssm_fp32, slot_indices
    )
    actual_conv = conv_seed.clone()
    actual = runtime.forward_prefill(
        hidden, cu_seqlens, metadata, num_prefills, actual_conv, ssm_bf16, slot_indices
    )

    assert indexed_calls == [True, True], "both pools must take the indexed prefill kernel"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_conv, ref_conv, rtol=0, atol=0)
    assert not torch.equal(ssm_bf16[addressed], initial[addressed]), "state was not written back"
    torch.testing.assert_close(
        ssm_bf16[addressed], ssm_fp32[addressed].to(torch.bfloat16), rtol=0, atol=0
    )
    torch.testing.assert_close(ssm_bf16[untouched], initial[untouched], rtol=0, atol=0)
