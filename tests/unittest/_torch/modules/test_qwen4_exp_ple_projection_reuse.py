# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact run reuse with full PLE forward/state and conservative fallback guards."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod
from tensorrt_llm._torch.modules.qwen4_exp import ple
from tensorrt_llm._torch.modules.qwen4_exp.ple import (
    PLEMetadata,
    Qwen4ExpPLE,
    _build_ple_projection_run_plan,
)


def _make_module(dtype: torch.dtype, device: str = "cpu") -> Qwen4ExpPLE:
    config = SimpleNamespace(
        hidden_size=32,
        hc_count=4,
        ngram_size=3,
        heads_per_ngram=8,
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        vocab_size=248320,
        eos_token_id=248044,
        seed=1234,
        ngram_vocab_size_base=257,
        make_ngram_vocab_size_divisible_by=128,
        rms_norm_eps=1e-6,
    )
    module = Qwen4ExpPLE(config, dtype=dtype).to(device).eval()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name:
                parameter.normal_(0.0, 0.01)
            else:
                parameter.normal_(0.0, 0.02)
    return module


def _metadata(
    sequences: list[list[int]],
    slots: list[int],
    *,
    num_contexts: int,
    decode: bool = False,
    padding: int = 0,
    device: str = "cpu",
) -> PLEMetadata:
    lengths = [len(sequence) for sequence in sequences]
    tokens = torch.tensor([token for sequence in sequences for token in sequence], device=device)
    return PLEMetadata.build(
        tokens,
        torch.tensor(lengths, device=device),
        torch.tensor(slots, device=device),
        is_decode=decode,
        eos_token_id=248044,
        physical_tokens=len(tokens) + padding,
        num_contexts=num_contexts,
        host_seq_lens=lengths,
    )


@torch.inference_mode()
def test_run_certificate_preserves_complete_heads_and_request_boundaries() -> None:
    module = _make_module(torch.float32)
    sequences = [[248057] * 8192, [], [17, 248044] + [248056] * 8190, [248056]]
    lengths = [len(sequence) for sequence in sequences]
    metadata = _metadata(sequences, [3, 0, 1, 2], num_contexts=3, padding=7)
    packed = torch.tensor([token for sequence in sequences for token in sequence])
    original = packed.clone()
    plan = _build_ple_projection_run_plan(packed, lengths, module.ngram_size)
    assert plan is not None
    histories = torch.tensor([[19, 23], [71, 83], [11, 41], [43, 63]])
    _, ids = module._prepare_ngram_lookup(metadata, histories)
    assert torch.equal(ids, ids.index_select(0, plan.starts).index_select(0, plan.inverse))
    assert set((0, 1, 2, 8192, 8193, 8194, 16384)).issubset(set(plan.starts.tolist()))
    assert torch.equal(packed, original)
    assert torch.equal(packed, metadata.padded_tokens[metadata.req_indices, metadata.token_offsets])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_normal_forward_reuses_projection_and_preserves_state(dtype: torch.dtype) -> None:
    torch.manual_seed(1811)
    baseline = _make_module(dtype)
    candidate = copy.deepcopy(baseline)
    candidate._experimental_projection_reuse = True
    states = torch.zeros(6, *baseline.conv_state_shape, dtype=dtype)
    histories = torch.full((6, 2), 248044, dtype=torch.long)
    candidate_states, candidate_histories = states.clone(), histories.clone()
    rows: list[int] = []
    handle = candidate.key_proj.register_forward_pre_hook(
        lambda _module, inputs: rows.append(inputs[0].shape[0])
    )
    cases = [
        ([[248057] * 8192, [11, 248044] + [248056] * 126], [3, 1], 2, False, 7, False),
        ([[248057] * 8192, [248056] * 128], [3, 1], 2, False, 0, False),
        ([[248057] * 8192, [248057]], [4, 1], 1, False, 3, False),
        ([[248057], [248056]], [1, 3], 0, True, 2, False),
        ([[248044] * 8192], [3], 1, False, 0, True),
    ]
    for sequences, slots, contexts, decode, padding, reset in cases:
        if reset:
            states[slots] = 0
            candidate_states[slots] = 0
            histories[slots] = 248044
            candidate_histories[slots] = 248044
        metadata = _metadata(
            sequences, slots, num_contexts=contexts, decode=decode, padding=padding
        )
        original_states, original_histories = states.clone(), histories.clone()
        query = torch.randn(metadata.physical_tokens, baseline.conv_channels).to(dtype)
        expected = baseline(query, metadata, states, histories)
        actual = candidate(query, metadata, candidate_states, candidate_histories)
        tolerance = 1e-4 if dtype == torch.float32 else 0.04
        torch.testing.assert_close(expected, actual, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(states, candidate_states, atol=tolerance, rtol=tolerance)
        assert torch.equal(histories, candidate_histories)
        unused = [index for index in range(6) if index not in slots]
        assert torch.equal(original_states[unused], candidate_states[unused])
        assert torch.equal(original_histories[unused], candidate_histories[unused])
        assert torch.count_nonzero(actual[metadata.processed_tokens :]) == 0
        assert rows[-1] == metadata.processed_tokens if decode else rows[-1] < 16
    handle.remove()


@pytest.mark.parametrize("kind", ["small", "unique", "mild_repeat"])
@torch.inference_mode()
def test_negative_certificate_keeps_dense_projection(kind: str) -> None:
    count = 1024 if kind == "small" else 8192
    tokens = torch.full((count,), 248057)
    if kind == "unique":
        tokens = torch.arange(count)
    elif kind == "mild_repeat":
        tokens = torch.arange(count) // 8
    assert _build_ple_projection_run_plan(tokens, [count], 3) is None
    module = _make_module(torch.float32)
    module._experimental_projection_reuse = True
    metadata = _metadata([tokens.tolist()], [0], num_contexts=1)
    rows: list[int] = []
    handle = module.key_proj.register_forward_pre_hook(
        lambda _module, inputs: rows.append(inputs[0].shape[0])
    )
    module._project_key_value(torch.randn(count, 32), metadata)
    assert rows == [count]
    handle.remove()


@pytest.mark.parametrize(
    "fallback",
    [
        "disabled",
        "decode",
        "small",
        "graph",
        "compile",
        "speculative",
        "adp",
        "gradient",
        "quantized",
        "lora",
        "linear_subclass",
        "tp",
    ],
)
def test_unsupported_steps_do_not_plan(fallback: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _make_module(torch.float32)
    module._experimental_projection_reuse = fallback != "disabled"
    metadata = _metadata([[248057] * 8192], [0], num_contexts=1)
    if fallback == "decode":
        metadata.is_decode = True
    elif fallback == "small":
        metadata.processed_tokens = 1024
    elif fallback == "graph":
        metadata.is_cuda_graph = True
    elif fallback == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif fallback == "speculative":
        metadata.use_spec_decoding = True
    elif fallback == "adp":
        module.ple_embedding.use_attention_dp_sharding = True
    elif fallback == "lora":
        module.key_proj.lora = SimpleNamespace()
    elif fallback == "tp":
        module.key_proj.tp_mode = "column"
    elif fallback == "linear_subclass":

        class CustomLinear(type(module.key_proj)):
            pass

        module.key_proj.__class__ = CustomLinear
    elif fallback == "quantized":

        class QuantizedSubclass(UnquantizedLinearMethod):
            pass

        module.key_proj.quant_method = QuantizedSubclass()

    def unexpected_plan(*args: object) -> None:
        raise AssertionError("unsupported steps must not copy tokens or build a plan")

    monkeypatch.setattr(ple, "_build_ple_projection_run_plan", unexpected_plan)
    with torch.set_grad_enabled(fallback == "gradient"):
        assert not module._can_reuse_projection(metadata, torch.randn(8192, 32))
        module._project_key_value(torch.randn(8192, 32), metadata)


@torch.inference_mode()
def test_prefetch_uses_same_fresh_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _make_module(torch.float32)
    candidate = copy.deepcopy(baseline)
    candidate._experimental_projection_reuse = True
    metadata = _metadata([[248057] * 8192], [2], num_contexts=1, padding=5)
    histories = torch.full((4, 2), 248044, dtype=torch.long)
    histories[2] = torch.tensor([19, 248057])
    states = torch.randn(4, *baseline.conv_state_shape) * 0.01
    combined, ids = candidate._prepare_ngram_lookup(metadata, histories)
    embeddings = candidate.ple_embedding.embed(ids).flatten(start_dim=-2)

    def consume(_metadata: PLEMetadata) -> tuple[torch.Tensor, torch.Tensor]:
        candidate._prefetch_state = None
        return embeddings, combined

    monkeypatch.setattr(candidate, "_consume_prefetched_embeddings", consume)
    candidate._prefetch_state = (embeddings, 8192, 8197, combined)
    candidate_states, candidate_histories = states.clone(), histories.clone()
    query = torch.randn(8197, baseline.conv_channels)
    expected = baseline(query, metadata, states, histories)
    actual = candidate(query, metadata, candidate_states, candidate_histories)
    torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(states, candidate_states, atol=1e-4, rtol=1e-4)
    assert torch.equal(histories, candidate_histories)
