# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for text watermarking: PRF, keyed sampling, detection and sampler state."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.watermark import (
    WatermarkHandler,
    request_wants_watermark,
)
from tensorrt_llm.watermark import GumbelWatermarkDetector, WatermarkConfig
from tensorrt_llm.watermark.detector import _gamma_upper_tail, _max_pair_sum_tail
from tensorrt_llm.watermark.prf import (
    PAD_TOKEN,
    context_seed,
    context_seeds,
    context_seeds_torch,
    philox_uniform,
    philox_uniform_torch,
    secondary_key,
)
from tensorrt_llm.watermark.sampling import (
    gumbel_sample_from_probs,
    gumbel_sample_reference,
    keyed_recovery,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _uniform_from_word(word: int) -> float:
    return ((word >> 9) * 2 + 1) / float(1 << 24)


class TestPrf:
    def test_philox_known_answer(self):
        # Random123 known-answer test: Philox4x32-10 with counter 0 and key 0 outputs
        # 6627e8d5 e169c58d bc57ac4c 9b00dbd8; the PRF uses the first word.
        assert philox_uniform(0, np.array([0]))[0] == np.float32(_uniform_from_word(0x6627E8D5))

    def test_uniform_range_and_spread(self):
        values = philox_uniform(0x1234_5678_9ABC_DEF0, np.arange(100_000))
        assert values.dtype == np.float32
        assert values.min() > 0.0 and values.max() < 1.0
        assert abs(values.mean() - 0.5) < 0.01
        assert abs(values.var() - 1 / 12) < 0.005

    def test_numpy_matches_torch(self):
        seeds = np.array([0, 1, 2**63 + 5, 2**64 - 1], dtype=np.uint64)
        tokens = np.array([0, 1, 7, 151_935, 2**31 - 1])
        expected = np.stack([philox_uniform(int(seed), tokens) for seed in seeds])
        signed = torch.tensor(seeds.astype(np.int64))
        actual = philox_uniform_torch(signed, torch.tensor(tokens)).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_per_token_seed_array(self):
        tokens = np.arange(6)
        seeds = np.array([3, 3, 9, 9, 2**64 - 1, 0], dtype=np.uint64)
        batched = philox_uniform(seeds, tokens)
        single = [philox_uniform(int(s), tokens[i : i + 1])[0] for i, s in enumerate(seeds)]
        np.testing.assert_array_equal(batched, np.array(single, dtype=np.float32))

    def test_context_seed_depends_on_key_and_every_token(self):
        base = context_seed(7, [1, 2, 3, 4])
        assert base == context_seed(7, [1, 2, 3, 4])
        assert base != context_seed(8, [1, 2, 3, 4])
        for position in range(4):
            changed = [1, 2, 3, 4]
            changed[position] += 1
            assert base != context_seed(7, changed)
        assert context_seed(7, [PAD_TOKEN, 1, 2, 3]) != context_seed(7, [0, 1, 2, 3])

    def test_context_seeds_torch_matches_numpy(self):
        rng = np.random.default_rng(0)
        contexts = rng.integers(0, 150_000, size=(64, 4))
        contexts[0, :2] = PAD_TOKEN
        key = 2**63 + 12345
        expected = context_seeds(key, contexts)
        actual = context_seeds_torch(key, torch.tensor(contexts))
        np.testing.assert_array_equal(actual.numpy().astype(np.uint64), expected)


class TestGammaTail:
    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 30.0])
    def test_exponential_special_case(self, x):
        assert _gamma_upper_tail(1.0, x) == pytest.approx(math.exp(-x), rel=1e-9)

    @pytest.mark.parametrize("x", [0.5, 3.0, 12.0])
    def test_erlang_two(self, x):
        assert _gamma_upper_tail(2.0, x) == pytest.approx((1 + x) * math.exp(-x), rel=1e-9)

    def test_large_shape_normal_limit(self):
        n = 400.0
        assert _gamma_upper_tail(n, n) == pytest.approx(0.5, abs=0.02)
        assert _gamma_upper_tail(n, n + 3 * math.sqrt(n)) < 0.01
        assert _gamma_upper_tail(n, 0.0) == 1.0


def _generate(config, probs, *, num_tokens, prompt, watermark=True, seed=0):
    """Toy autoregressive loop with a fixed next-token distribution."""
    rng = np.random.default_rng(seed)
    width = config.context_width
    tokens = list(prompt)
    seen = set()
    out = []
    probs_row = torch.tensor(probs, dtype=torch.float32).unsqueeze(0)
    for _ in range(num_tokens):
        context = tokens[-width:]
        context = [PAD_TOKEN] * (width - len(context)) + context
        key = tuple(context)
        repeated = config.deduplicate_contexts == "single_turn" and key in seen
        seen.add(key)
        if watermark and not repeated:
            seed_t = context_seeds_torch(config.key, torch.tensor([context]))
            token = int(gumbel_sample_reference(probs_row, seed_t)[0])
        else:
            token = int(rng.choice(len(probs), p=probs))
        tokens.append(token)
        out.append(token)
    return out


class TestMaxPairTail:
    def test_matches_simulation(self):
        rng = np.random.default_rng(0)
        n = 40
        sums = np.maximum(
            rng.exponential(size=(200_000, n)), rng.exponential(size=(200_000, n))
        ).sum(1)
        for quantile in (0.5, 0.9, 0.99):
            threshold = float(np.quantile(sums, quantile))
            assert _max_pair_sum_tail(n, threshold) == pytest.approx(
                1 - quantile, rel=0.08, abs=2e-3
            )

    def test_is_a_valid_tail(self):
        assert _max_pair_sum_tail(10, 0.0) == 1.0
        assert _max_pair_sum_tail(10, 1e6) < 1e-100 or _max_pair_sum_tail(10, 1e6) == 0.0
        values = [_max_pair_sum_tail(30, t) for t in (30.0, 45.0, 60.0, 80.0)]
        assert values == sorted(values, reverse=True)


class TestDetector:
    @staticmethod
    def _zipf(vocab=2000):
        weights = 1.0 / np.arange(1, vocab + 1) ** 0.8
        return weights / weights.sum()

    def test_detects_watermarked_text(self):
        config = WatermarkConfig(key=42)
        probs = self._zipf()
        prompt = [5, 6, 7, 8, 9]
        text = _generate(config, probs, num_tokens=300, prompt=prompt)
        result = GumbelWatermarkDetector(config).detect(text, prompt_token_ids=prompt)
        assert result.is_watermarked
        assert result.p_value < 1e-6

    def test_does_not_flag_plain_text_or_wrong_key(self):
        config = WatermarkConfig(key=42)
        probs = self._zipf()
        prompt = [5, 6, 7, 8, 9]
        plain = _generate(config, probs, num_tokens=300, prompt=prompt, watermark=False)
        assert (
            not GumbelWatermarkDetector(config)
            .detect(plain, prompt_token_ids=prompt)
            .is_watermarked
        )
        marked = _generate(config, probs, num_tokens=300, prompt=prompt)
        other = WatermarkConfig(key=43)
        assert (
            not GumbelWatermarkDetector(other)
            .detect(marked, prompt_token_ids=prompt)
            .is_watermarked
        )

    def test_false_positive_rate_on_plain_text(self):
        config = WatermarkConfig(key=7)
        probs = self._zipf()
        detector = GumbelWatermarkDetector(config, p_value_threshold=0.05)
        flagged = 0
        trials = 200
        for trial in range(trials):
            text = _generate(
                config, probs, num_tokens=120, prompt=[1, 2, 3], watermark=False, seed=trial
            )
            flagged += detector.detect(text, prompt_token_ids=[1, 2, 3]).is_watermarked
        assert flagged / trials < 0.12

    def test_without_prompt_skips_first_context_width_positions(self):
        config = WatermarkConfig(key=1, context_width=4)
        text = _generate(config, self._zipf(), num_tokens=50, prompt=[1, 2, 3])
        result = GumbelWatermarkDetector(config).detect(text)
        assert result.num_scored <= 50 - 4

    def test_dual_key_detector_finds_either_key(self):
        config = WatermarkConfig(key=11, algorithm="gumbel_dual_key", deduplicate_contexts="none")
        probs = self._zipf()
        prompt = [5, 6, 7, 8, 9]
        # Tokens drawn with the primary key, then with the secondary key.
        primary = _generate(config, probs, num_tokens=150, prompt=prompt)
        secondary_config = WatermarkConfig(key=secondary_key(11), deduplicate_contexts="none")
        secondary = _generate(secondary_config, probs, num_tokens=150, prompt=prompt)
        detector = GumbelWatermarkDetector(config)
        assert detector.detect(primary, prompt_token_ids=prompt).is_watermarked
        assert detector.detect(secondary, prompt_token_ids=prompt).is_watermarked
        # A single-key detector does not know the secondary key.
        single = GumbelWatermarkDetector(WatermarkConfig(key=11, deduplicate_contexts="none"))
        assert not single.detect(secondary, prompt_token_ids=prompt).is_watermarked

    def test_repeated_contexts_are_scored_once(self):
        config = WatermarkConfig(key=3, context_width=1)
        # Context of width one repeats constantly; only distinct contexts may be scored.
        text = _generate(config, self._zipf(50), num_tokens=200, prompt=[0])
        scored = GumbelWatermarkDetector(config).detect(text, prompt_token_ids=[0]).num_scored
        assert scored == len({0, *text[:-1]})
        config_none = WatermarkConfig(key=3, context_width=1, deduplicate_contexts="none")
        assert (
            GumbelWatermarkDetector(config_none).detect(text, prompt_token_ids=[0]).num_scored
            == 200
        )


class TestConfig:
    def test_key_is_not_in_repr(self):
        assert "12345" not in repr(WatermarkConfig(key=12345))

    @pytest.mark.parametrize(
        "kwargs", [{"key": -1}, {"key": 2**64}, {"key": 1, "context_width": 0}]
    )
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            WatermarkConfig(**kwargs)

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError):
            WatermarkConfig(key=1, algorithm="synthid")


@requires_cuda
class TestKernel:
    def test_matches_reference(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(48, 5000, device="cuda") * 3, dim=-1)
        probs[3, 100:] = 0  # exercise zero-probability tokens
        probs[3] /= probs[3].sum()
        seeds = torch.randint(-(2**62), 2**62, (48,), device="cuda", dtype=torch.int64)
        actual = gumbel_sample_from_probs(probs, seeds)
        expected = gumbel_sample_reference(probs, seeds)
        # float32 log may differ from float64 in the last place, which can flip a near tie.
        assert (actual == expected).float().mean() >= 0.97
        assert (probs.gather(1, actual.long().unsqueeze(1)) > 0).all()

    @pytest.mark.parametrize("num_rows", [3, 40])
    def test_large_vocabulary_is_split_across_programs(self, num_rows):
        torch.manual_seed(1)
        vocab = 151936
        probs = torch.softmax(torch.randn(num_rows, vocab, device="cuda") * 4, dim=-1)
        seeds = torch.randint(-(2**62), 2**62, (num_rows,), device="cuda", dtype=torch.int64)
        active = torch.arange(num_rows, device="cuda") % 3 != 1
        # A map that shifts every column, as a draft-to-target vocabulary map does.
        id_map = torch.arange(vocab, device="cuda") + 7
        actual = gumbel_sample_from_probs(probs, seeds, active, id_map)
        expected = gumbel_sample_reference(probs, seeds, id_map)
        assert (actual[active] == expected[active]).float().mean() >= 0.97
        assert (actual[~active] == -1).all()
        assert torch.equal(
            gumbel_sample_from_probs(probs, seeds, id_map=id_map)[active], actual[active]
        )

    def test_deterministic_and_seed_dependent(self):
        probs = torch.softmax(torch.randn(8, 3000, device="cuda"), dim=-1)
        seeds = torch.arange(8, device="cuda", dtype=torch.int64)
        first = gumbel_sample_from_probs(probs, seeds)
        assert torch.equal(first, gumbel_sample_from_probs(probs, seeds))
        assert not torch.equal(first, gumbel_sample_from_probs(probs, seeds + 1000))

    def test_inactive_rows_are_skipped(self):
        probs = torch.softmax(torch.randn(4, 100, device="cuda"), dim=-1)
        seeds = torch.arange(4, device="cuda", dtype=torch.int64)
        active = torch.tensor([True, False, True, False], device="cuda")
        out = gumbel_sample_from_probs(probs, seeds, active)
        assert out[1] == -1 and out[3] == -1
        assert out[0] == gumbel_sample_from_probs(probs[:1], seeds[:1])[0]

    def test_vocab_not_multiple_of_block(self):
        probs = torch.softmax(torch.randn(3, 1025, device="cuda"), dim=-1)
        seeds = torch.arange(3, device="cuda", dtype=torch.int64) + 99
        out = gumbel_sample_from_probs(probs, seeds)
        assert ((out >= 0) & (out < 1025)).all()

    def test_does_not_distort_the_distribution(self):
        # Averaged over random seeds the keyed draw must be an exact sample from p.
        probs = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625, 0.0, 0.0, 0.0], device="cuda")
        rows = 400_000
        seeds = torch.randint(-(2**62), 2**62, (rows,), device="cuda", dtype=torch.int64)
        tokens = gumbel_sample_from_probs(probs.unsqueeze(0).expand(rows, -1).contiguous(), seeds)
        counts = torch.bincount(tokens.long(), minlength=8).double().cpu()
        expected = probs.double().cpu() * rows
        mask = expected > 0
        chi2 = (((counts - expected) ** 2)[mask] / expected[mask]).sum().item()
        assert counts[~mask].sum() == 0
        # 4 degrees of freedom; 18.5 is the 0.001 critical value.
        assert chi2 < 18.5


def _chi_square(tokens, probs, rows):
    counts = torch.bincount(tokens.long(), minlength=probs.numel()).double().cpu()
    expected = probs.double().cpu() * rows
    mask = expected > 0
    assert counts[~mask].sum() == 0
    return (((counts - expected) ** 2)[mask] / expected[mask]).sum().item()


@requires_cuda
class TestKeyedRecovery:
    """Rejection sampling with keyed draws must keep the target distribution exact.

    The drafts are drawn under one key, the token after the accepted prefix under a second,
    independent key, and the kernel's accept coins come from a third independent stream.
    """

    P = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
    Q = [0.45, 0.05, 0.10, 0.20, 0.05, 0.15]
    ROWS = 600_000

    def _run(self, num_draft, *, draft_keyed=True, recovery_key="secondary"):
        p = torch.tensor(self.P, device="cuda")
        q = torch.tensor(self.Q, device="cuda")
        rows, vocab = self.ROWS, p.numel()
        random_seeds = lambda *shape: torch.randint(  # noqa: E731
            -(2**62), 2**62, shape, device="cuda", dtype=torch.int64
        )
        seeds_draft = random_seeds(rows, num_draft)
        if draft_keyed:
            drafts = torch.stack(
                [
                    gumbel_sample_from_probs(q.expand(rows, -1).contiguous(), seeds_draft[:, i])
                    for i in range(num_draft)
                ],
                dim=1,
            )
        else:
            drafts = torch.multinomial(q.expand(rows * num_draft, -1), 1).view(rows, num_draft)
        drafts = drafts.to(torch.int32).contiguous()
        draft_probs = q.expand(rows, num_draft, vocab).contiguous()
        target_probs = p.expand(rows, num_draft + 1, vocab).contiguous()
        accepted, num_accepted = torch.ops.trtllm.chain_speculative_sampling(
            draft_probs,
            drafts,
            target_probs,
            random_seeds(rows),
            torch.zeros(rows, dtype=torch.int64, device="cuda"),
        )
        if recovery_key == "secondary":
            seeds = random_seeds(rows, num_draft + 1)
        else:  # the same stream that keyed the drafts (wrong on purpose)
            seeds = torch.cat([seeds_draft, seeds_draft[:, -1:]], dim=1)
        active = torch.ones(rows, num_draft + 1, dtype=torch.bool, device="cuda")
        out = keyed_recovery(accepted, num_accepted, target_probs, draft_probs, seeds, active)
        return out, num_accepted, p

    @pytest.mark.parametrize("num_draft", [1, 3])
    def test_dual_key_is_exact(self, num_draft):
        out, _, p = self._run(num_draft)
        # 5 degrees of freedom; 20.5 is the 0.001 critical value.
        assert _chi_square(out[:, 0], p, self.ROWS) < 20.5

    @pytest.mark.parametrize("num_draft", [1, 3])
    def test_target_only_is_exact(self, num_draft):
        out, _, p = self._run(num_draft, draft_keyed=False)
        assert _chi_square(out[:, 0], p, self.ROWS) < 20.5

    def test_reusing_the_draft_key_distorts(self):
        out, _, p = self._run(1, recovery_key="primary")
        assert _chi_square(out[:, 0], p, self.ROWS) > 1000

    def test_last_committed_token_is_the_only_one_redrawn(self):
        out, num_accepted, _ = self._run(3)
        accepted, n = torch.ops.trtllm.chain_speculative_sampling(
            torch.full((4, 3, 6), 1 / 6, device="cuda"),
            torch.zeros(4, 3, dtype=torch.int32, device="cuda"),
            torch.full((4, 4, 6), 1 / 6, device="cuda"),
            torch.arange(4, device="cuda", dtype=torch.int64),
            torch.zeros(4, dtype=torch.int64, device="cuda"),
        )
        seeds = torch.arange(16, device="cuda", dtype=torch.int64).view(4, 4)
        active = torch.ones(4, 4, dtype=torch.bool, device="cuda")
        redrawn = keyed_recovery(
            accepted,
            n,
            torch.full((4, 4, 6), 1 / 6, device="cuda"),
            torch.full((4, 3, 6), 1 / 6, device="cuda"),
            seeds,
            active,
        )
        for row in range(4):
            last = int(n[row]) - 1
            assert torch.equal(redrawn[row, :last], accepted[row, :last])
            assert torch.equal(redrawn[row, last + 1 :], accepted[row, last + 1 :])

    def test_inactive_positions_keep_the_kernel_token(self):
        probs = torch.full((2, 2, 4), 0.25, device="cuda")
        draft = torch.full((2, 1, 4), 0.25, device="cuda")
        accepted = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32, device="cuda")
        n = torch.tensor([2, 1], device="cuda")
        seeds = torch.tensor([[5, 6], [7, 8]], device="cuda")
        active = torch.zeros(2, 2, dtype=torch.bool, device="cuda")
        assert torch.equal(keyed_recovery(accepted, n, probs, draft, seeds, active), accepted)

    def test_secondary_key_is_independent_of_the_primary(self):
        assert secondary_key(7) != 7
        assert secondary_key(7) == secondary_key(7)
        assert secondary_key(7) != secondary_key(8)


def _request(request_id, slot, tokens, watermarking=None, greedy=False):
    return SimpleNamespace(
        py_request_id=request_id,
        py_seq_slot=slot,
        py_watermarking=watermarking,
        greedy=greedy,
        get_tokens=lambda beam: list(tokens),
    )


def _can_sample(request):
    return not request.greedy


def _device(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device="cuda")


def _seed_of(config, context):
    return int(context_seeds(config.key, np.array([context]))[0])


def _as_uint(tensor):
    return tensor.cpu().numpy().astype(np.uint64)


@requires_cuda
class TestHandler:
    def _handler(self, **kwargs):
        config = WatermarkConfig(key=99, context_width=3, **kwargs)
        return config, WatermarkHandler(config, max_num_sequences=4, max_seq_len=64)

    def test_seeds_follow_the_token_stream(self):
        config, handler = self._handler()
        handler.observe([_request(1, 2, [10, 11, 12, 13])], _can_sample)
        history = [11, 12, 13]
        for token in (7, 8, 9):
            rows = handler.begin([2])
            assert _as_uint(rows.seeds)[0, 0] == _seed_of(config, history[-3:])
            handler.commit(rows, _device([token], torch.int32))
            history.append(token)

    @pytest.mark.parametrize("dedupe", ["none", "single_turn"])
    @pytest.mark.parametrize("from_probs", [False, True])
    def test_step_matches_plan_draw_commit(self, dedupe, from_probs):
        config = WatermarkConfig(key=99, context_width=3, deduplicate_contexts=dedupe)
        handlers = [WatermarkHandler(config, max_num_sequences=4, max_seq_len=64) for _ in range(2)]
        requests = [_request(i, slot, [5, 6, 7, i]) for i, slot in enumerate([3, 0, 2, 1])]
        requests[2].py_watermarking = False  # one row carries no watermark
        for handler in handlers:
            handler.observe(requests, _can_sample)
        reference, fused = handlers
        slots = _device([3, 0, 2, 1])
        enabled = _device([True, True, False, True], torch.bool)
        generator = torch.Generator(device="cuda").manual_seed(0)
        for _ in range(4):
            logits = torch.randn(4, 3001, device="cuda", generator=generator) * 3
            temperature = _device([0.7, 1.0, 1.3, 0.9], torch.float32)
            probs = torch.softmax(logits / temperature[:, None], dim=-1)
            # A tiny vocabulary of recurring contexts makes the repeat ring fire.
            fallback = torch.randint(0, 3, (4,), device="cuda", dtype=torch.int32)
            rows = reference.plan_reference(slots, enabled)
            keyed = gumbel_sample_reference(probs, rows.seeds[:, 0])
            expected = torch.where(rows.active[:, 0], keyed.to(torch.int32), fallback)
            reference.commit_reference(rows, expected)
            actual = fused.sample_step(
                probs if from_probs else logits,
                None if from_probs else temperature,
                slots,
                enabled,
                fallback,
                from_probs=from_probs,
            )
            assert torch.equal(actual, expected)
        assert torch.equal(fused._tail, reference._tail)
        assert torch.equal(fused._count, reference._count)
        if dedupe == "single_turn":
            size = fused._ring_size  # the last column is a trash column
            assert torch.equal(fused._ring[:, :size], reference._ring[:, :size])

    def test_short_prompt_is_padded(self):
        config, handler = self._handler()
        handler.observe([_request(1, 0, [5])], _can_sample)
        rows = handler.begin([0])
        assert _as_uint(rows.seeds)[0, 0] == _seed_of(config, [PAD_TOKEN, PAD_TOKEN, 5])

    def test_new_occupant_resets_the_slot(self):
        _, handler = self._handler()
        handler.observe([_request(1, 1, [1, 2, 3])], _can_sample)
        rows = handler.begin([1])
        handler.commit(rows, _device([4], torch.int32))
        handler.observe([_request(2, 1, [1, 2, 3])], _can_sample)
        again = handler.begin([1])
        assert torch.equal(again.seeds, rows.seeds)
        assert again.active.all()  # the repeat history was cleared with the slot

    def test_repeated_context_falls_back(self):
        _, handler = self._handler()
        handler.observe([_request(1, 0, [1, 2, 3])], _can_sample)
        first = handler.begin([0])
        assert first.active.all()
        # Return to the same context: feed the tokens that rebuild [1, 2, 3].
        for token in (1, 2, 3):
            rows = first if token == 1 else handler.begin([0])
            handler.commit(rows, _device([token], torch.int32))
        repeated = handler.begin([0])
        assert torch.equal(repeated.seeds, first.seeds)
        assert not repeated.active.any()

    def test_no_dedup_when_disabled(self):
        _, handler = self._handler(deduplicate_contexts="none")
        handler.observe([_request(1, 0, [1, 2, 3])], _can_sample)
        first = handler.begin([0])
        for token in (1, 2, 3):
            rows = first if token == 1 else handler.begin([0])
            handler.commit(rows, _device([token], torch.int32))
        assert handler.begin([0]).active.all()

    def test_opt_out_and_greedy_rows_are_inactive(self):
        _, handler = self._handler()
        handler.observe(
            [
                _request(1, 0, [1, 2, 3]),
                _request(2, 1, [1, 2, 3], watermarking=False),
                _request(3, 2, [1, 2, 3], greedy=True),
            ],
            _can_sample,
        )
        rows = handler.begin([0, 1, 2])
        assert rows.active[:, 0].tolist() == [True, False, False]
        assert handler.begin([1, 2]) is None

    def test_window_contexts_include_the_drafts(self):
        config, handler = self._handler()
        handler.observe([_request(1, 0, [1, 2, 3])], _can_sample)
        rows = handler.plan(_device([0]), _device([True], torch.bool), _device([[7, 8]]))
        expected = [[1, 2, 3], [2, 3, 7], [3, 7, 8]]
        assert [int(v) for v in _as_uint(rows.seeds)[0]] == [_seed_of(config, c) for c in expected]

    def test_window_dedupes_repeats_inside_the_window(self):
        _, handler = self._handler()
        handler.observe([_request(1, 0, [5, 5, 5])], _can_sample)
        # Drafts 5, 5 keep the context [5, 5, 5]: positions 1 and 2 repeat position 0.
        rows = handler.plan(_device([0]), _device([True], torch.bool), _device([[5, 5]]))
        assert rows.active.tolist() == [[True, False, False]]

    @pytest.mark.parametrize("dedupe", ["none", "single_turn"])
    def test_speculative_windows_match_plain_decoding(self, dedupe):
        # Whatever the draft quality, the seeds of the committed positions and the final state
        # must equal those of one-token-at-a-time decoding of the same output.
        rng = np.random.default_rng(1)
        prompt = [3, 1, 4]
        output = [int(t) for t in rng.integers(0, 6, size=40)]  # small vocab forces repeats
        config, plain = self._handler(deduplicate_contexts=dedupe)
        _, spec = self._handler(deduplicate_contexts=dedupe)
        plain.observe([_request(1, 0, prompt)], _can_sample)
        spec.observe([_request(1, 0, prompt)], _can_sample)
        plain_seeds, plain_active = [], []
        for token in output:
            rows = plain.begin([0])
            plain_seeds.append(int(_as_uint(rows.seeds)[0, 0]))
            plain_active.append(bool(rows.active[0, 0]))
            plain.commit(rows, _device([token], torch.int32))

        spec_seeds, spec_active = [], []
        position = 0
        while position < len(output):
            k = 3
            target = (output + [0] * k)[position : position + k + 1]
            drafts = list(target[:k])
            mismatch = int(rng.integers(0, k + 2))  # k + 1 means every draft matches
            if mismatch <= k - 1:
                drafts[mismatch] = (drafts[mismatch] + 1) % 6
            matched = min(mismatch, k)
            accepted = matched + 1
            rows = spec.plan(_device([0]), _device([True], torch.bool), _device([drafts]))
            take = min(accepted, len(output) - position)
            spec_seeds += [int(v) for v in _as_uint(rows.seeds)[0, :take]]
            spec_active += [bool(v) for v in rows.active[0, :take]]
            window_tokens = _device([target], torch.int32)
            spec.commit(rows, window_tokens, _device([take]))
            position += take
        assert spec_seeds == plain_seeds
        assert spec_active == plain_active
        assert torch.equal(spec._tail, plain._tail)
        assert torch.equal(spec._count, plain._count)

    def test_disabled_request_state_is_untouched(self):
        _, handler = self._handler()
        handler.observe([_request(1, 0, [1, 2, 3])], _can_sample)
        before = handler._tail.clone()
        rows = handler.plan(_device([0]), _device([False], torch.bool), _device([[4, 5]]))
        handler.commit(rows, _device([[4, 5, 6]], torch.int32), _device([3]))
        assert torch.equal(handler._tail, before)
        assert int(handler._count[0]) == 0


def _twin_handlers(**config_kwargs):
    config = WatermarkConfig(key=2**63 + 17, context_width=3, **config_kwargs)
    return config, [
        WatermarkHandler(config, max_num_sequences=6, max_seq_len=40, device="cuda")
        for _ in range(2)
    ]


@requires_cuda
class TestFusedKernels:
    """The fused plan and commit kernels must match the plain-torch reference exactly."""

    @pytest.mark.parametrize("dedupe", ["none", "single_turn"])
    @pytest.mark.parametrize("algorithm", ["gumbel", "gumbel_dual_key"])
    @pytest.mark.parametrize("num_drafts", [0, 1, 3])
    def test_matches_reference_over_random_steps(self, dedupe, algorithm, num_drafts):
        rng = np.random.default_rng(num_drafts)
        _, (fused, reference) = _twin_handlers(deduplicate_contexts=dedupe, algorithm=algorithm)
        # A tiny vocabulary makes repeated contexts (and in-window repeats) common, and a ring
        # shorter than the generation makes entries expire.
        for slot in range(5):
            for handler in (fused, reference):
                handler.observe([_request(100 + slot, slot, [1, 2, 3])], _can_sample)
        for _ in range(60):
            slots = _device(rng.permutation(5).tolist() + [5])  # slot 5 stands in for padding
            enabled = _device([True, True, False, True, True, False], torch.bool)
            drafts = _device(rng.integers(0, 3, size=(6, num_drafts))) if num_drafts else None
            rows = fused.plan(slots, enabled, drafts)
            expected = reference.plan_reference(slots, enabled, drafts)
            assert torch.equal(rows.seeds, expected.seeds)
            assert torch.equal(rows.seeds_secondary, expected.seeds_secondary)
            assert torch.equal(rows.active, expected.active)
            accepted = _device(rng.integers(0, 3, size=(6, num_drafts + 1)), torch.int32)
            num_accepted = _device(rng.integers(1, num_drafts + 2, size=6))
            fused.commit(rows, accepted, num_accepted)
            reference.commit_reference(expected, accepted, num_accepted)
            assert torch.equal(fused._tail, reference._tail)
            assert torch.equal(fused._count, reference._count)
            if dedupe == "single_turn":
                assert torch.equal(
                    fused._ring[:, : fused._ring_size], reference._ring[:, : reference._ring_size]
                )

    def test_inactive_and_padding_rows_keep_their_state(self):
        _, (fused, reference) = _twin_handlers()
        fused.observe([_request(1, 0, [4, 5, 6])], _can_sample)
        before = fused._tail.clone(), fused._count.clone()
        slots = _device(
            [0, 5, 5]
        )  # a disabled request and two padding rows sharing the scratch slot
        enabled = _device([False, False, False], torch.bool)
        rows = fused.plan(slots, enabled, _device([[1, 2], [3, 4], [5, 6]]))
        assert not rows.active.any()
        fused.commit(rows, _device([[1, 2, 3]] * 3, torch.int32), _device([3, 3, 3]))
        assert torch.equal(fused._tail, before[0]) and torch.equal(fused._count, before[1])


class TestRequestWantsWatermark:
    def test_semantics(self):
        on = WatermarkConfig(key=1)
        forced = WatermarkConfig(key=1, allow_request_opt_out=False)
        assert not request_wants_watermark(None, _request(1, 0, []))
        assert request_wants_watermark(on, _request(1, 0, []))
        assert request_wants_watermark(on, _request(1, 0, [], watermarking=True))
        assert not request_wants_watermark(on, _request(1, 0, [], watermarking=False))
        assert request_wants_watermark(forced, _request(1, 0, [], watermarking=False))
