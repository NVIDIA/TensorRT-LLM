# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN decode must keep its arithmetic when unrelated prefill joins the batch."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _make_layer() -> Qwen3NextGatedDeltaNet:
    layer = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.attn_tp_size = 1
    layer.num_k_heads = layer.num_k_heads_per_tp = 4
    layer.num_v_heads = layer.num_v_heads_per_tp = 8
    layer.head_k_dim = layer.head_v_dim = 128
    layer.key_dim_per_tp = 512
    layer.value_dim_per_tp = 1024
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(2048, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
        bias=None,
    )
    layer.A_log = torch.randn(8, device="cuda") * 0.1
    layer.dt_bias = torch.randn(8, device="cuda") * 0.1
    return layer


@pytest.fixture(params=["flashinfer", "triton"])
def prefill_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    monkeypatch.setenv(
        "TLLM_USE_FLASHINFER_GDN_PREFILL", "1" if request.param == "flashinfer" else "0"
    )
    if request.param == "flashinfer" and not gdn_mixer._use_flashinfer_gdn_prefill():
        pytest.skip("FlashInfer GDN prefill is unavailable on this GPU")
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
    yield
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()


@pytest.mark.parametrize("prefill_lengths", [(1,), (127,), (128,), (129,), (7, 129)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("preallocated", [False, True])
@pytest.mark.usefixtures("prefill_backend")
def test_mixed_decode_matches_decode_only(
    prefill_lengths: tuple[int, ...], state_dtype: torch.dtype, preallocated: bool
) -> None:
    torch.manual_seed(42)
    layer = _make_layer()
    num_decodes, num_slots = 3, 7
    num_prefills = len(prefill_lengths)
    prefill_length = sum(prefill_lengths)
    num_tokens = prefill_length + num_decodes
    # Preserve the noncontiguous views produced by the real fused projections.
    projected = torch.randn(num_tokens, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    original_projected = projected.clone()
    mixed_qkv = projected[:, :2048]
    gates = torch.randn(num_tokens, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    b, a = gates[:, :8], gates[:, 8:]
    conv = torch.randn(num_slots, 2048, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(num_slots, 8, 128, 128, device="cuda", dtype=state_dtype) * 0.1
    # Nonsequential slots and an unaligned int32 decode-index slice.
    active_slots = [5, 4][:num_prefills] + [3, 0, 6]
    indices = torch.tensor(active_slots, device="cuda", dtype=torch.int32)
    lengths = [0, *prefill_lengths, *([1] * num_decodes)]
    positions = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    common = {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "batch_size": num_prefills + num_decodes,
        "has_initial_states": torch.ones(
            num_prefills + num_decodes, device="cuda", dtype=torch.bool
        ),
        "cache_indices": indices,
        "query_start_loc": positions,
        "query_start_loc_long": positions.long(),
        "num_prefill_tokens": prefill_length,
        "num_decode_tokens": num_decodes,
        "state_indices_p": indices[:num_prefills],
        "state_indices_d": indices[num_prefills:],
        "num_prefill": num_prefills,
        "num_decodes": num_decodes,
    }

    expected_conv, expected_state = conv.clone(), state.clone()
    expected_qkv = projected.clone()[:, :2048]
    expected_decode = layer.forward_decode(
        expected_conv,
        expected_state,
        query_start_loc_long=torch.arange(num_decodes + 1, device="cuda", dtype=torch.int64),
        mixed_qkv=expected_qkv[prefill_length:],
        a=a[prefill_length:],
        b=b[prefill_length:],
        cache_indices=indices[num_prefills:],
        num_decodes=num_decodes,
    )
    prefill_kwargs = {
        **common,
        "mixed_qkv": expected_qkv[:prefill_length],
        "a": a[:prefill_length],
        "b": b[:prefill_length],
        "batch_size": num_prefills,
        "num_decode_tokens": 0,
        "num_decodes": 0,
        "cache_indices": indices[:num_prefills],
        "state_indices_d": indices[:0],
        "query_start_loc": positions[: num_prefills + 1],
        "query_start_loc_long": positions[: num_prefills + 1].long(),
    }
    # Warm chunk tuning with disposable state; cold-autotune state preservation
    # has its own regression, independent of mixed-batch dispatch.
    layer.forward_extend(conv.clone(), state.clone(), **prefill_kwargs)
    expected_prefill = layer.forward_extend(expected_conv, expected_state, **prefill_kwargs)
    output = (
        torch.empty(1, num_tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
        if preallocated
        else None
    )
    actual_conv, actual_state = conv.clone(), state.clone()
    actual = layer.forward_extend(actual_conv, actual_state, output=output, **common)

    torch.testing.assert_close(actual[:, prefill_length:], expected_decode, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_state[indices[num_prefills:]], expected_state[indices[num_prefills:]], rtol=0, atol=0
    )
    torch.testing.assert_close(actual[:, :prefill_length], expected_prefill, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
    unused_slots = sorted(set(range(num_slots)) - set(active_slots))
    torch.testing.assert_close(actual_state[unused_slots], state[unused_slots], rtol=0, atol=0)
    if output is not None:
        assert actual.data_ptr() == output.data_ptr()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projected.copy_(original_projected)
            actual_conv.copy_(conv)
            actual_state.copy_(state)
            graph_output = layer.forward_extend(actual_conv, actual_state, output=output, **common)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(
                graph_output[:, prefill_length:], expected_decode, rtol=0, atol=0
            )
            torch.testing.assert_close(
                graph_output[:, :prefill_length], expected_prefill, rtol=0, atol=0
            )
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
            torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
