# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN decode must keep its arithmetic when unrelated prefill joins the batch."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._utils import is_sm_100f

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


@pytest.fixture(params=["flashinfer", "triton", "flashinfer_workspace"])
def prefill_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    use_flashinfer = request.param != "triton"
    # Deterministic mode splits decode from prefill; a state workspace does so in any mode.
    monkeypatch.setenv(
        "FORCE_DETERMINISTIC", "0" if request.param == "flashinfer_workspace" else "1"
    )
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1" if use_flashinfer else "0")
    if use_flashinfer and not gdn_mixer._use_flashinfer_gdn_prefill():
        pytest.skip("FlashInfer GDN prefill is unavailable on this GPU")
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
    yield request.param
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()


@pytest.mark.parametrize("prefill_lengths", [(1,), (127,), (128,), (129,), (7, 129)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("preallocated", [False, True])
def test_mixed_decode_matches_decode_only(
    prefill_backend: str,
    prefill_lengths: tuple[int, ...],
    state_dtype: torch.dtype,
    preallocated: bool,
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
    if prefill_backend == "flashinfer_workspace":
        # One-sequence scratch, as on SM90/SM120: two prefills run in groups and
        # decode never stages its state there.
        workspace_dtype = state_dtype if is_sm_100f() else torch.float32
        common["state_workspace"] = tuple(
            torch.empty(1, 8, 128, 128, device="cuda", dtype=workspace_dtype) for _ in range(2)
        )
        common["cu_seqlens_cpu"] = torch.tensor([0, *prefill_lengths]).cumsum(0)

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
    with patch.object(
        gdn_mixer, "chunk_gated_delta_rule", wraps=gdn_mixer.chunk_gated_delta_rule
    ) as prefill_call:
        actual = layer.forward_extend(actual_conv, actual_state, output=output, **common)
    # Only the prefill requests go through the prefill kernel.
    assert prefill_call.call_count == 1
    assert prefill_call.call_args.kwargs["cu_seqlens"].shape[0] == num_prefills + 1

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


def test_forward_core_runs_prefill_groups_from_host_offsets(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch with more prefill requests than the state workspace holds runs in groups."""
    monkeypatch.setenv("FORCE_DETERMINISTIC", "0")
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    if not gdn_mixer._use_flashinfer_gdn_prefill() or is_sm_100f():
        pytest.skip("Only FlashInfer GDN prefill on SM90/SM120 uses a state workspace")
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
    request.addfinalizer(gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear)
    torch.manual_seed(42)
    layer = _make_layer()
    layer.layer_idx = 0
    prefill_lengths, num_decodes, num_slots = (7, 129, 64), 2, 8
    lengths = [*prefill_lengths, *([1] * num_decodes)]
    num_prefills, num_tokens = len(prefill_lengths), sum(lengths)
    projected = torch.randn(num_tokens, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    gates = torch.randn(num_tokens, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    conv = torch.randn(num_slots, 2048, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(num_slots, 8, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    indices = torch.tensor([6, 1, 4, 3, 0], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, *lengths], device="cuda").cumsum(0, dtype=torch.int32)
    mamba_metadata = SimpleNamespace(
        has_initial_states=torch.ones(len(lengths), device="cuda", dtype=torch.bool),
        state_indices=indices,
        query_start_loc=positions,
        query_start_loc_long=positions.long(),
        replay_work_items=None,
        replay_n_writes=None,
    )
    state_bytes = state[0].numel() * torch.float32.itemsize

    def run(workspace_states: int):
        monkeypatch.setattr(
            gdn_mixer, "_PREFILL_STATE_WORKSPACE_BYTES", workspace_states * state_bytes
        )
        layer.model_config = SimpleNamespace(extra_attrs={})
        cache = SimpleNamespace(conv=conv.clone(), temporal=state.clone())
        attn_metadata = SimpleNamespace(
            num_contexts=num_prefills,
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            num_ctx_tokens=sum(prefill_lengths),
            num_tokens=num_tokens,
            max_num_sequences=len(lengths),
            max_num_requests=len(lengths),
            max_num_tokens=num_tokens,
            kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda _layer_idx: cache),
        )
        # The decode convolution writes into its input, so each run gets fresh inputs.
        mixed_qkv = projected.clone()[:, :2048]
        b, a = gates[:, :8].clone(), gates[:, 8:].clone()
        with patch.object(
            gdn_mixer, "chunk_gated_delta_rule", wraps=gdn_mixer.chunk_gated_delta_rule
        ) as prefill_call:
            out = layer.forward_core(mixed_qkv, a, b, attn_metadata, mamba_metadata)
        assert prefill_call.call_count == 1
        return out, cache, prefill_call.call_args.kwargs

    # Warm chunk tuning with disposable state, as in the mixed-batch test above.
    run(1)
    grouped, grouped_cache, grouped_call = run(1)
    single, single_cache, single_call = run(num_prefills)
    # Only a batch the workspace cannot hold needs the host offsets of its prefill requests.
    assert grouped_call["state_workspace"][0].shape[0] == 1
    torch.testing.assert_close(
        grouped_call["cu_seqlens_cpu"], torch.tensor([0, *prefill_lengths]).cumsum(0)
    )
    assert single_call["state_workspace"][0].shape[0] == num_prefills
    assert single_call["cu_seqlens_cpu"] is None
    # Decode requests never use the workspace; prefill groups match one call to accumulation error.
    prefill_length = sum(prefill_lengths)
    torch.testing.assert_close(
        grouped[:, prefill_length:], single[:, prefill_length:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        grouped[:, :prefill_length], single[:, :prefill_length], atol=1e-4, rtol=1.6e-2
    )
    decode_slots, prefill_slots = indices[num_prefills:], indices[:num_prefills]
    torch.testing.assert_close(
        grouped_cache.temporal[decode_slots], single_cache.temporal[decode_slots], rtol=0, atol=0
    )
    torch.testing.assert_close(
        grouped_cache.temporal[prefill_slots],
        single_cache.temporal[prefill_slots],
        atol=2e-3,
        rtol=1.6e-2,
    )
    torch.testing.assert_close(grouped_cache.conv, single_cache.conv, rtol=0, atol=0)
