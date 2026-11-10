# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.watermark import WatermarkHandler, philox_uniforms
from tensorrt_llm.watermarking import WatermarkConfig, WatermarkDetector
from tensorrt_llm.watermarking.prf import philox4x32_10, token_uniform


@pytest.mark.cpu_only
def test_random123_published_vectors():
    # Random123's Philox4x32-10 known-answer vectors, independent of our mapping.
    assert philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )
    assert philox4x32_10((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2) == (
        0x408F276D,
        0x41C83B0E,
        0xA20BC7C6,
        0x6D5451FD,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("key", [0, 42, (1 << 64) - 1])
@pytest.mark.parametrize("domain", [0, 1, 2])
def test_tensor_and_scalar_prf_match(key, domain):
    contexts = torch.tensor([[0, 1, 2, 3], [123456789, -1, 0, 2**32 - 1]])
    tokens = torch.tensor([0, 42, 65535, 2**31, 2**32 - 1])
    expected = torch.tensor(
        [
            [token_uniform(key, context, token, domain) for token in tokens.tolist()]
            for context in contexts.tolist()
        ],
        dtype=torch.float64,
    )
    assert torch.equal(philox_uniforms(key, contexts, tokens, domain), expected)


@pytest.mark.cpu_only
def test_paired_prf_matches_scalar_without_cross_product():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [8, 9, 10, 11]])
    positions = torch.tensor([7, 100, 2**32 - 1])
    paired = philox_uniforms(123, contexts, positions, domain=2, paired=True)
    assert paired.shape == (3, 1)
    assert paired[:, 0].tolist() == [
        token_uniform(123, context, position, domain=2)
        for context, position in zip(contexts.tolist(), positions.tolist())
    ]


def _request(slot, request_id, prompt, watermarking=None):
    return SimpleNamespace(
        py_seq_slot=slot,
        py_request_id=request_id,
        py_orig_prompt_len=len(prompt),
        py_watermarking=watermarking,
        is_dummy=False,
        get_tokens=lambda beam: prompt,
    )


def _generate(config, length=256, vocabulary=32):
    prompt = [11, 7, 19, 4]
    tokens = []
    history = []
    context = prompt.copy()
    for position in range(length):
        suffix = tuple(context[-config.context_width :])
        if suffix in history:
            token = random.Random(position).randrange(vocabulary)
        else:
            domain = 0
            if config.algorithm == "dual_key_gumbel":
                domain = int(token_uniform(config.key, suffix, position, 2) < config.alpha)
            token = max(
                range(vocabulary),
                key=lambda candidate: (
                    -math.log(-math.log(token_uniform(config.key, suffix, candidate, domain)))
                ),
            )
        history.append(suffix)
        if len(history) > config.deduplicate_contexts_max_history:
            history.pop(0)
        tokens.append(token)
        context.append(token)
    return prompt, tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("algorithm", ["gumbel", "dual_key_gumbel"])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_detector_recovers_watermark_and_rejects_wrong_key(algorithm, alpha):
    config = WatermarkConfig(algorithm=algorithm, key=123456789, alpha=alpha)
    prompt, tokens = _generate(config)
    matched = WatermarkDetector(config).detect(tokens, prompt_token_ids=prompt)
    assert matched.detected
    assert matched.p_value < 1e-20
    wrong_key = config.model_copy(update={"key": 987654321})
    assert not WatermarkDetector(wrong_key).detect(tokens, prompt_token_ids=prompt).detected
    # Detection also works when the prompt is unavailable, after an unscored prefix.
    assert WatermarkDetector(config).detect(tokens).detected


@pytest.mark.cpu_only
def test_null_p_values_and_categorical_distribution():
    # Repeated fixed text under unrelated keys is an independent null experiment.
    rng = random.Random(2026)
    tokens = [rng.randrange(10000) for _ in range(128)]
    false_positives = sum(
        WatermarkDetector(WatermarkConfig(key=key)).detect(tokens).detected for key in range(64)
    )
    assert false_positives <= 4

    # Across keys, Gumbel-max retains the source categorical distribution.
    probabilities = [0.1, 0.3, 0.6]
    counts = [0, 0, 0]
    for key in range(1024):
        choice = max(
            range(3),
            key=lambda token: (
                math.log(probabilities[token])
                - math.log(-math.log(token_uniform(key, [1, 2, 3, 4], token)))
            ),
        )
        counts[choice] += 1
    for count, probability in zip(counts, probabilities):
        assert abs(count - 1024 * probability) < 5 * math.sqrt(
            1024 * probability * (1 - probability)
        )


@pytest.mark.cpu_only
def test_device_history_does_not_read_stale_host_outputs_and_resets_slots():
    config = WatermarkConfig(key=123, deduplicate_contexts_max_history=8)
    handler = WatermarkHandler(config, device="cpu")
    stale_prompt = [1, 2, 3, 4]
    request = _request(0, 100, stale_prompt)
    handler.setup_request(request)
    probabilities = torch.full((1, 32), 1 / 32)
    generated = []
    for _ in range(16):
        generated.append(handler.sample(probabilities, torch.tensor([0]), [0]).item())
    # The host prompt has not been changed or consulted for any generated token.
    assert stale_prompt == [1, 2, 3, 4]
    prompt, expected = _generate(config, length=16)
    other = WatermarkHandler(config, device="cpu")
    other.setup_request(_request(7, 101, prompt))
    actual = [other.sample(probabilities, torch.tensor([0]), [7]).item() for _ in range(16)]
    assert actual == expected
    assert handler._states[0].context.tolist() == generated[-config.context_width :]

    replacement = _request(0, 200, [9, 8, 7, 6])
    handler.setup_request(replacement)
    handler.release_request(request)
    assert handler._states[0].request_id == 200
    assert handler._states[0].context.tolist() == [9, 8, 7, 6]
    assert handler._states[0].history_count == 0
    handler.release_request(replacement)
    assert not handler._states


@pytest.mark.cpu_only
def test_repeated_context_falls_back_to_stock_and_detector_skips_it():
    config = WatermarkConfig(key=0, deduplicate_contexts_max_history=8)
    repeated_token = max(range(2), key=lambda token: token_uniform(0, [1, 1, 1, 1], token))
    prompt = [repeated_token] * 4
    # Select a key whose first winner preserves the repeated context.
    key = next(
        candidate
        for candidate in range(100)
        if max(range(2), key=lambda token: token_uniform(candidate, prompt, token))
        == repeated_token
    )
    config = config.model_copy(update={"key": key})
    handler = WatermarkHandler(config, device="cpu")
    handler.setup_request(_request(0, 0, prompt))
    probabilities = torch.tensor([[0.5, 0.5]])
    stock = torch.tensor([1 - repeated_token])
    first = handler.sample(probabilities, stock, [0]).item()
    second = handler.sample(probabilities, stock, [0]).item()
    assert first == repeated_token
    assert second == stock.item()
    result = WatermarkDetector(config).detect([first, second], prompt_token_ids=prompt)
    assert result.num_scored_tokens == 1


@pytest.mark.cpu_only
def test_short_prompt_optout_and_batch_invariance():
    config = WatermarkConfig(key=(1 << 64) - 1)
    handler = WatermarkHandler(config, device="cpu")
    request = _request(0, 0, [7])
    assert handler.enabled(request)
    assert not handler.enabled(_request(1, 1, [7], watermarking=False))
    assert not WatermarkHandler(None, device="cpu").enabled(request)
    handler.setup_request(request)
    probabilities = torch.full((1, 8), 0.125)
    for token in [1, 2, 3]:
        assert handler.sample(probabilities, torch.tensor([token]), [0]).item() == token
    assert handler._states[0].context.tolist() == [7, 1, 2, 3]

    one = WatermarkHandler(config, device="cpu")
    batch = WatermarkHandler(config, device="cpu")
    one.setup_request(request)
    batch.setup_request(request)
    batch.setup_request(_request(2, 2, [6, 5, 4, 3]))
    for _ in range(8):
        expected = one.sample(probabilities, torch.tensor([0]), [0])
        actual = batch.sample(probabilities.repeat(2, 1), torch.tensor([0, 0]), [0, 2])
        assert actual[0] == expected[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("algorithm", ["gumbel", "dual_key_gumbel"])
def test_chunked_selector_matches_whole_batch_exactly(algorithm, monkeypatch):
    config = WatermarkConfig(key=2718281828, algorithm=algorithm, alpha=0.5)
    whole = WatermarkHandler(config, device="cpu")
    chunked = WatermarkHandler(config, device="cpu")
    chunked._max_candidate_elements = 64
    slots = list(range(9))
    for slot in slots:
        request = _request(slot, slot + 100, [slot, slot + 3, slot + 7, slot + 9])
        whole.setup_request(request)
        chunked.setup_request(request)
    probabilities = torch.arange(1, 33).to(torch.float32).repeat(9, 1)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    stock = torch.zeros(9, dtype=torch.int64)
    chunk_shapes = []
    original_select = chunked._select_chunk

    def record_chunk(probs, *args):
        chunk_shapes.append(probs.shape)
        return original_select(probs, *args)

    monkeypatch.setattr(chunked, "_select_chunk", record_chunk)
    for _ in range(8):
        assert torch.equal(
            whole.sample(probabilities, stock, slots),
            chunked.sample(probabilities, stock, slots),
        )
    assert len(chunk_shapes) == 8 * 5
    assert all(rows * vocabulary <= 64 for rows, vocabulary in chunk_shapes)


@pytest.mark.cpu_only
def test_reprefill_reconstructs_full_accepted_prefix_and_preserves_live_state():
    config = WatermarkConfig(key=123, deduplicate_contexts_max_history=4)
    prompt, tokens = _generate(config, length=12)
    uninterrupted = WatermarkHandler(config, device="cpu")
    request = _request(0, 19, prompt)
    uninterrupted.setup_request(request)
    probabilities = torch.full((1, 32), 1 / 32)
    stock = torch.tensor([0])
    for _ in range(12):
        uninterrupted.sample(probabilities, stock, [0])
    original = uninterrupted._states[0]
    uninterrupted.setup_request(request)
    assert uninterrupted._states[0] is original
    request.py_seq_slot = 1
    uninterrupted.setup_request(request)
    assert 0 not in uninterrupted._states
    assert uninterrupted._states[1] is original

    resumed = WatermarkHandler(config, device="cpu")
    resumed_request = _request(7, 19, prompt + tokens)
    resumed_request.py_orig_prompt_len = len(prompt)
    resumed.setup_request(resumed_request)
    assert resumed._states[7].context.tolist() == tokens[-config.context_width :]
    assert resumed._states[7].history_count == 4
    for _ in range(8):
        assert torch.equal(
            uninterrupted.sample(probabilities, stock, [1]),
            resumed.sample(probabilities, stock, [7]),
        )


@pytest.mark.cpu_only
def test_detector_deduplicates_beyond_generation_history():
    config = WatermarkConfig(key=123, context_width=1, deduplicate_contexts_max_history=1)
    result = WatermarkDetector(config).detect([1, 2, 1, 2, 1, 2], prompt_token_ids=[2])
    assert result.num_scored_tokens == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("token", [True, -1, 1.0, 2**32])
def test_detector_rejects_invalid_token_ids(token):
    detector = WatermarkDetector(WatermarkConfig(key=0))
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        detector.detect([token])
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        detector.detect([], prompt_token_ids=[token])


@pytest.mark.cpu_only
@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 0, 1])
def test_detector_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        WatermarkDetector(WatermarkConfig(key=0)).detect([], threshold=threshold)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_prf_and_sampling_match_cpu():
    config = WatermarkConfig(key=0xFEDCBA9876543210, algorithm="dual_key_gumbel", alpha=0.3)
    contexts = torch.tensor([[0, 1, 2, 3], [65535, 2**31, 7, 42]])
    tokens = torch.arange(1000)
    assert torch.equal(
        philox_uniforms(config.key, contexts, tokens),
        philox_uniforms(config.key, contexts.cuda(), tokens.cuda()).cpu(),
    )
    cpu = WatermarkHandler(config, device="cpu")
    gpu = WatermarkHandler(config)
    for handler in (cpu, gpu):
        handler.setup_request(_request(0, 1, [1, 2, 3, 4]))
    probabilities = torch.full((1, 128), 1 / 128)
    stock = torch.tensor([0])
    for _ in range(32):
        assert torch.equal(
            cpu.sample(probabilities, stock, [0]),
            gpu.sample(probabilities.cuda(), stock.cuda(), [0]).cpu(),
        )
