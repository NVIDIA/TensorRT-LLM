# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4.1 Engram prefill, disaggregated handoff and graphed decode."""

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.models import modeling_deepseekv41 as v41
from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram import engram as engram_module
from tensorrt_llm._torch.modules.engram import functional as engram_functional
from tensorrt_llm._torch.modules.engram.projection import EngramFp8Projection
from tensorrt_llm._torch.modules.linear import (
    flashinfer_mxfp8_autotune,
    flashinfer_mxfp8_decode_graph_capture,
)


def _reference_v41_engram_gate(
    hidden: torch.Tensor,
    kv: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    add_residual: bool,
) -> torch.Tensor:
    """Keys-first gate with a single cast after the FP32 residual addition."""
    _, hc, dim = hidden.shape
    keys, value = kv.float().split([hc * dim, dim], dim=-1)
    keys = keys.reshape(-1, hc, dim)
    h = hidden.float()
    weight = query_weight.float() * key_weight.float()
    rstd = (h.square().mean(-1) + eps).rsqrt() * (keys.square().mean(-1) + eps).rsqrt()
    dot = (h * weight * keys).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    value = gate.unsqueeze(-1) * value.unsqueeze(-2)
    return (h + value if add_residual else value).to(hidden.dtype)


def _hashes(
    provider: EngramHashProvider, tokens: list[int], positions: Sequence[int]
) -> torch.Tensor:
    return provider.compute_hashes(
        torch.tensor(tokens, dtype=torch.long, device="cuda"),
        position_ids=torch.tensor(positions, dtype=torch.long, device="cuda"),
        request_ids=[41],
        seq_lens_host=torch.tensor([len(tokens)], dtype=torch.int32),
        max_seq_len=64,
    )[1].clone()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Engram FP8 execution requires CUDA")
@pytest.mark.parametrize("add_residual", [False, True])
@torch.no_grad()
def test_v41_engram_gate_parity(monkeypatch: pytest.MonkeyPatch, add_residual: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10 or not hasattr(
        torch.ops.trtllm, "mxfp8_mxfp8_gemm"
    ):
        pytest.skip("Native MXFP8 Engram WKV requires a compiled Blackwell backend")
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    torch.manual_seed(151)
    buckets = [101, 103, 107, 109]
    config = EngramConfig(
        layer_ids=[1],
        max_ngram_size=3,
        n_head_per_ngram=2,
        n_embed_per_ngram=256,
        hidden_size=256,
        hc_mult=4,
        norm_eps=1e-20,
        dtype=torch.bfloat16,
    )
    # Fix tokenizer metadata, not the hashing or request-history implementation.
    mapping = SimpleNamespace(
        compressed_tokenizer=SimpleNamespace(lookup_table=torch.arange(256) // 2),
        pad_id=0,
        layer_multipliers={1: torch.tensor([17, 31, 47])},
        vocab_size_across_layers={1: [buckets[:2], buckets[2:]]},
    )
    monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
    aggregate, generation = EngramHashProvider(config), EngramHashProvider(config)
    with torch.device("cuda"):
        module = v41.DeepseekV41Engram(1, config, vocab_sizes_flat=buckets)
    table = module.multi_head_embedding
    table_pointers = (table.weight.data_ptr(), table.scale.data_ptr())
    module.cuda()
    table.to(device="cuda", dtype=torch.bfloat16)
    assert table._requires_standard_hf_loading
    assert table.weight.device.type == table.scale.device.type == "cpu"
    assert table.weight.is_pinned() and table.scale.is_pinned()
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    assert table_pointers == (table.weight.data_ptr(), table.scale.data_ptr())
    assert module.short_conv is None
    assert isinstance(module.kv_proj, EngramFp8Projection)
    method = module.kv_proj.quant_method
    assert method.use_cutlass, "dequantized WKV is not native FP8 execution"
    assert method.backend == "trtllm"
    assert module.kv_proj._use_flashinfer_mxfp8_decode_graph_default

    table_weight = torch.randn(sum(buckets), 128).to(torch.float8_e4m3fn)
    table_scale = torch.randint(125, 130, (sum(buckets), 4), dtype=torch.uint8)
    table.load_weights([{"weight": table_weight, "scale": table_scale.view(torch.float8_e8m0fnu)}])
    weight = torch.randn(1280, 512).to(torch.float8_e4m3fn)
    scale = torch.randint(123, 130, (40, 16), dtype=torch.uint8)
    checkpoint = {"layers.1.engram.wkv.weight": weight, "layers.1.engram.wkv.scale": scale}
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(checkpoint, num_hidden_layers=2)
    stem = "model.layers.1.engram.kv_proj"
    assert forwarded[stem + ".weight"] is weight
    assert forwarded[stem + ".scale"] is scale
    assert forwarded.census.folded == 0
    module.kv_proj.load_weights(
        [{"weight": forwarded[stem + ".weight"], "scale": forwarded[stem + ".scale"]}]
    )
    torch.testing.assert_close(
        module.kv_proj.weight.view(torch.uint8).cpu(), weight.view(torch.uint8), rtol=0, atol=0
    )
    module.query_norm_weight.copy_(torch.randn_like(module.query_norm_weight))
    module.key_norm_weight.copy_(torch.randn_like(module.key_norm_weight))
    module.post_load_weights()
    assert module._gate_norm_product.dtype == torch.float32
    pointers = (
        table.weight.data_ptr(),
        table.scale.data_ptr(),
        module._gate_norm_product.data_ptr(),
    )
    module.warmup_kernels()

    if add_residual:
        # Release gate geometry, with a small unused projection and table.
        gate_config = replace(config, hidden_size=5120, n_embed_per_ngram=64)
        with torch.device("cuda"):
            gate_module = v41.DeepseekV41Engram(1, gate_config, vocab_sizes_flat=buckets)
        gate_module.query_norm_weight.copy_(torch.randn_like(gate_module.query_norm_weight))
        gate_module.key_norm_weight.copy_(torch.randn_like(gate_module.key_norm_weight))
        gate_module.post_load_weights()
        launch = Mock(wraps=engram_functional._engram_gate_kernel.run)
        with monkeypatch.context() as gate_patch:
            gate_patch.setattr(engram_functional._engram_gate_kernel, "run", launch)
            gate_module.warmup_kernels()
        assert [call.kwargs["PRECOMPUTED_WEIGHT"] for call in launch.call_args_list] == [
            False,
            torch.cuda.get_device_capability() == (10, 3),
        ]
        gate_hidden = torch.randn(3, 4, 5120, dtype=torch.bfloat16, device="cuda")
        gate_kv = torch.randn(3, 25600, dtype=torch.bfloat16, device="cuda")
        expected_gate = _reference_v41_engram_gate(
            gate_hidden,
            gate_kv,
            gate_module.query_norm_weight,
            gate_module.key_norm_weight,
            gate_config.norm_eps,
            True,
        )
        for product in (None, gate_module._gate_norm_product):
            actual_gate = v41.engram_gate(
                gate_hidden,
                gate_kv,
                gate_module.query_norm_weight,
                gate_module.key_norm_weight,
                gate_config.norm_eps,
                add_residual=True,
                norm_weight_product=product,
            )
            torch.testing.assert_close(actual_gate, expected_gate, rtol=8e-3, atol=2e-6)

    table_reference = (
        table_weight.float() * torch.exp2(table_scale.float() - 127).repeat_interleave(32, 1)
    ).cuda()
    reference_offsets = torch.tensor(
        [sum(buckets[:head]) for head in range(len(buckets))], device="cuda"
    )
    expanded_scale = (
        torch.exp2(scale.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    )
    projection_reference = (weight.float() * expanded_scale).cuda().t()

    def check_output(
        hidden: torch.Tensor,
        hashes: torch.Tensor,
        embeddings: torch.Tensor,
        kv: torch.Tensor,
        actual: torch.Tensor,
    ) -> None:
        expected_embeddings = (
            table_reference[hashes + reference_offsets].flatten(1).to(hidden.dtype)
        )
        torch.testing.assert_close(embeddings, expected_embeddings, rtol=0, atol=0)
        reference_kv = expected_embeddings.float() @ projection_reference
        assert torch.isfinite(kv).all() and reference_kv.norm() > 0
        # Preserve the native W8A8 tolerance; checkpoint bytes and lookup are exact.
        assert (kv.float() - reference_kv).norm() / reference_kv.norm() < 0.04
        expected = _reference_v41_engram_gate(
            hidden,
            kv,
            module.query_norm_weight,
            module.key_norm_weight,
            config.norm_eps,
            add_residual,
        )
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-6)

    prompt = [11, 13, 15]
    hashes = _hashes(aggregate, prompt, range(len(prompt)))
    hidden = torch.randn(3, 4, 256, dtype=torch.bfloat16, device="cuda")
    embeddings = module.precompute(hashes)
    kv = module.precompute_kv(embeddings)
    actual = module(hidden, embeddings, add_residual=add_residual)
    check_output(hidden, hashes, embeddings, kv, actual)

    input_ids = torch.tensor([17], device="cuda")
    position_ids = torch.tensor([3], device="cuda")
    lengths = torch.tensor([1], dtype=torch.int32)
    hidden = torch.randn(1, 4, 256, dtype=torch.bfloat16, device="cuda")
    generation.compute_hashes(input_ids)
    captured_hashes = generation._cached_hashes[1]
    hash_pointer = captured_hashes.data_ptr()

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = module.precompute(generation.compute_hashes(input_ids)[1])
        kv = module.precompute_kv(embeddings)
        return (
            embeddings,
            kv,
            module(hidden, embeddings, add_residual=add_residual, precomputed_kv=kv),
        )

    assert method.enable_flashinfer_auto(), (
        "the pinned Blackwell runtime must provide MXFP8 FlashInfer"
    )
    with flashinfer_mxfp8_autotune():
        forward()
    method.mark_flashinfer_autotuned()
    flashinfer = Mock(wraps=method._flashinfer_mxfp8)
    monkeypatch.setattr(method, "_flashinfer_mxfp8", flashinfer)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    assert flashinfer.call_count == 0, "eager execution must retain its native backend"
    graph = torch.cuda.CUDAGraph()
    with flashinfer_mxfp8_decode_graph_capture(), torch.cuda.graph(graph):
        embeddings, kv, actual = forward()
    capture_calls = flashinfer.call_count
    assert capture_calls > 0, "decode capture must use the tuned FlashInfer kernel"

    reads = []

    def get_tokens_range(beam: int, start: int, end: int) -> list[int]:
        reads.append((beam, start, end))
        return prompt[start:end]

    model = SimpleNamespace(model=SimpleNamespace(use_engram=True, engram_hash_provider=generation))
    request = SimpleNamespace(py_request_id=41, prompt_len=3, get_tokens_range=get_tokens_range)
    v41.DeepseekV41ForCausalLM.prepare_disagg_generation_request(model, request)
    assert reads == [(0, 1, 3)]
    assert 41 in generation._pending_history_seeds
    previous_hashes = None
    for position, token in enumerate([21, 23, 25], start=len(prompt)):
        input_ids.fill_(token)
        position_ids.fill_(position)
        if position < 5:
            hidden.copy_(torch.randn_like(hidden))
        else:
            hidden.zero_()
        refreshed = generation.refresh_captured_hashes(
            input_ids,
            position_ids=position_ids,
            request_ids=[41],
            seq_lens_host=lengths,
            max_seq_len=64,
        )[1]
        assert refreshed.data_ptr() == hash_pointer
        expected_hashes = _hashes(aggregate, [token], [position])
        torch.testing.assert_close(refreshed, expected_hashes, rtol=0, atol=0)
        if previous_hashes is not None:
            assert not torch.equal(refreshed, previous_hashes)
        previous_hashes = refreshed.clone()
        graph.replay()
        torch.cuda.synchronize()
        check_output(hidden, expected_hashes, embeddings, kv, actual)
        native_kv = module.precompute_kv(embeddings)
        assert flashinfer.call_count == capture_calls
        assert (kv.float() - native_kv.float()).norm() / native_kv.float().norm() < 1e-3
        assert not generation._pending_history_seeds
        assert pointers == (
            table.weight.data_ptr(),
            table.scale.data_ptr(),
            module._gate_norm_product.data_ptr(),
        )
    assert reads == [(0, 1, 3)], "later decode steps must not reseed prompt history"
    v41.DeepseekV41ForCausalLM.release_request_state(model, 41)
    assert 41 not in generation._pending_history_seeds
    assert 41 not in generation._history_row_of


def _multi_token_hash_providers(
    monkeypatch: pytest.MonkeyPatch, vocab: int
) -> tuple[EngramHashProvider, EngramHashProvider]:
    """Two providers with the same constants; the reference always takes the general path."""
    generator = torch.Generator().manual_seed(1234)
    config = EngramConfig(
        layer_ids=[1, 3],
        max_ngram_size=4,
        n_head_per_ngram=2,
        n_embed_per_ngram=256,
        hidden_size=256,
        hc_mult=4,
        norm_eps=1e-20,
        dtype=torch.bfloat16,
    )
    mapping = SimpleNamespace(
        compressed_tokenizer=SimpleNamespace(
            lookup_table=torch.randint(0, vocab - 3, (vocab,), generator=generator)
        ),
        pad_id=2,
        layer_multipliers={
            1: torch.tensor([3, 1000003, 999999937, 2147483647]),
            3: torch.tensor([5, 7919, 104729, 1299709]),
        },
        vocab_size_across_layers={
            1: [[100003, 100019], [100043, 100049], [100057, 100069]],
            3: [[99991, 99989], [99971, 99961], [99929, 99923]],
        },
    )
    monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
    reference, candidate = EngramHashProvider(config), EngramHashProvider(config)
    monkeypatch.setattr(reference, "_compute_history_hashes", lambda *args, **kwargs: None)
    return reference, candidate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Engram hash kernels require CUDA")
@torch.no_grad()
def test_multi_token_history_hashes_match_general_path(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab, max_seq_len = 1000, 512
    reference, candidate = _multi_token_hash_providers(monkeypatch, vocab)
    taken = []
    history_hashes = candidate._compute_history_hashes

    def record(*args, **kwargs):
        result = history_hashes(*args, **kwargs)
        taken.append(result is not None)
        return result

    monkeypatch.setattr(candidate, "_compute_history_hashes", record)
    generator = torch.Generator().manual_seed(4321)

    def tokens(count: int, low: int = 0, high: int = vocab) -> list[int]:
        return torch.randint(low, high, (count,), generator=generator).tolist()

    def forward(
        request_ids: list[int],
        lengths: list[int],
        positions: list[int],
        token_ids: list[int],
        *,
        padded: int | None = None,
        mask: list[bool] | None = None,
        dtype: torch.dtype = torch.long,
    ) -> dict[int, int]:
        input_ids = torch.tensor(token_ids, dtype=dtype, device="cuda")
        position_ids = torch.tensor(positions, dtype=dtype, device="cuda")
        seq_lens_host = torch.tensor(lengths, dtype=torch.int32)
        token_mask = None if mask is None else torch.tensor(mask, device="cuda")
        calls = len(taken)
        outputs = [
            provider.compute_hashes(
                input_ids,
                seq_lens=seq_lens_host.cuda(),
                position_ids=position_ids,
                request_ids=list(request_ids),
                seq_lens_host=seq_lens_host,
                max_seq_len=max_seq_len,
                token_mask=token_mask,
                padded_num_tokens=padded,
            )
            for provider in (reference, candidate)
        ]
        assert taken[calls:] == [True], "the multi-token history path must handle this forward"
        expected, actual = outputs
        assert expected.keys() == actual.keys()
        for layer_id in expected:
            torch.testing.assert_close(actual[layer_id], expected[layer_id], rtol=0, atol=0)
        torch.testing.assert_close(candidate._history, reference._history, rtol=0, atol=0)
        assert candidate._history_row_of == reference._history_row_of
        assert sorted(candidate._history_free_rows) == sorted(reference._history_free_rows)
        assert candidate._cached_hashes_store.keys() == reference._cached_hashes_store.keys()
        return {layer_id: value.data_ptr() for layer_id, value in actual.items()}

    seeds = {101: tokens(3), 102: tokens(2)}
    for provider in (reference, candidate):
        provider.queue_history_seed(101, 47, seeds[101])
        provider.queue_history_seed(102, 10, seeds[102])
    # Context chunks, one of them continuing a seeded prompt.
    forward([100, 101], [37, 20], list(range(37)) + list(range(50, 70)), tokens(57))
    forward([102], [5], list(range(12, 17)), tokens(5))
    forward([103], [9], list(range(9)), tokens(9))
    next_position = {100: 37, 101: 70, 102: 17, 103: 9}
    request_ids = [100, 101, 102, 103]

    def draft_positions(offset: int = 0) -> list[int]:
        return [
            position
            for request_id in request_ids
            for position in range(
                next_position[request_id] + offset, next_position[request_id] + offset + 6
            )
        ]

    # Speculative steps with 6 tokens per request and CUDA graph padding. Accepting
    # 1-6 tokens makes the next step overwrite the history cells of rejected drafts.
    pointers = None
    for step in range(8):
        step_pointers = forward(request_ids, [6] * 4, draft_positions(), tokens(24), padded=32)
        assert pointers is None or step_pointers == pointers, "graph buffers must be reused"
        pointers = step_pointers
        for index, request_id in enumerate(request_ids):
            next_position[request_id] += 1 + (step + index) % 6
    forward(request_ids, [6] * 4, draft_positions(), tokens(24))
    forward(request_ids, [6] * 4, draft_positions(), tokens(24), padded=32, dtype=torch.int32)
    out_of_vocab = tokens(12) + [-5, vocab + 7, 0, vocab - 1] + tokens(8)
    forward(request_ids, [6] * 4, draft_positions(6), out_of_vocab, padded=32)
    mask = [index not in (3, 10, 17) for index in range(24)]
    forward(request_ids, [6] * 4, draft_positions(), tokens(24), padded=32, mask=mask)
    aliased = list(range(next_position[100], next_position[100] + 6)) * 2
    aliased += list(range(next_position[101], next_position[101] + 6)) * 2
    forward([100, 100, 101, 101], [6] * 4, aliased, tokens(24), padded=32)
    for provider in (reference, candidate):
        provider.release_request_state(102)
        provider.queue_history_seed(200, 30, seeds[101])
    recycled = list(range(next_position[100], next_position[100] + 6))
    recycled += list(range(next_position[101], next_position[101] + 6)) + list(range(33, 39))
    recycled += list(range(next_position[103], next_position[103] + 6))
    forward([100, 101, 200, 103], [6] * 4, recycled, tokens(24), padded=32)
    mixed = list(range(next_position[100], next_position[100] + 6)) + list(range(50))
    forward([100, 300], [6, 50], mixed, tokens(56))
