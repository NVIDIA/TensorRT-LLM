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
"""Config-time budget checks for DFlash pooled-context buffers.

The pooled-context K/V buffers are allocated lazily on the first forward. With
KV-cache estimation (the default) that is the estimation forward, before the
pool is sized, so the buffers compete with weights and activations for the
device; with estimation skipped it is warmup, after the pool has committed,
so they get ``(1 - free_gpu_memory_fraction)`` of the post-load memory. In
either mode an oversized ``max_batch_size`` used to fail late with a bare
CUDA OOM (and a violated token budget with a memory corruption at engine
init). These tests pin the config-time errors that replace both failure
modes, the two budget derivations, and the trained-width warning.
"""

import json
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.speculative import dflash
from tensorrt_llm._torch.speculative.dflash import (
    compute_dflash_ctx_buffer_bytes,
    derive_dflash_ctx_memory_budget_bytes,
    estimate_checkpoint_weight_bytes,
    validate_dflash_ctx_buffer_budget,
)

pytestmark = pytest.mark.cpu_only

GIB = 1024**3
OVERHEAD = dflash._DFLASH_BUDGET_RUNTIME_OVERHEAD_BYTES
RESERVE = dflash._DFLASH_BUDGET_ACTIVATION_RESERVE_BYTES
SAFETY = dflash._DFLASH_BUDGET_SAFETY_FACTOR
SKIP_EST_ENV = "TRTLLM_SKIP_KV_CACHE_ESTIMATION"

# Geometry of a small GQA drafter for a large target: 5 attention layers,
# 16 KV heads (4 per rank at tp4), head_dim 64, bf16, 1M advertised
# positions. With no serve-time max_seq_len the capacity falls back to the
# 1M positions: at max_draft_len 7 (block width 8) each slot's K+V pool is
# 2 * 5 * (1048576 + 8) * 4 * 64 * 2 bytes = 5,368,750,080 bytes (~5 GiB),
# so max_batch_size 32 (33 slots) wants ~165 GiB per GPU. A configured
# max_seq_len caps the capacity instead (the pooled context cannot outgrow
# the sequence cap), so the same batch at max_seq_len 8192 wants ~1.3 GiB.
DRAFT_CONFIG = {
    "block_size": 8,
    "num_hidden_layers": 5,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "hidden_size": 8192,
    "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16",
}
PER_SLOT_BYTES = 2 * 5 * (1048576 + 8) * 4 * 64 * 2
PER_SLOT_BYTES_MSL = 2 * 5 * (8192 + 8) * 4 * 64 * 2
BUDGET = 72 * 10**9  # ~a quarter of a large device, outside the KV pool


def _validate(**overrides):
    kwargs = dict(
        max_batch_size=8,
        max_num_tokens=8192,
        max_seq_len=8192,
        max_draft_len=7,
        attention_backend="VANILLA",
        draft_config=DRAFT_CONFIG,
        tp_size=4,
        memory_budget_bytes=BUDGET,
    )
    kwargs.update(overrides)
    return validate_dflash_ctx_buffer_budget(**kwargs)


class TestComputeCtxBufferBytes:
    def test_vanilla_matches_allocation_shape(self):
        # Two contiguous [slots, L, capacity, Hkv, D] tensors.
        assert (
            compute_dflash_ctx_buffer_bytes(
                max_batch_size=32,
                max_ctx_len=1048576,
                block_size=8,
                num_attn_layers=5,
                num_kv_heads_per_rank=4,
                head_dim=64,
                dtype_bytes=2,
                attention_backend="VANILLA",
            )
            == 33 * PER_SLOT_BYTES
        )

    def test_trtllm_rounds_capacity_to_pages(self):
        # One paged tensor; per-slot capacity rounds up to whole 32-pages.
        capacity = 100 + 8
        paged_capacity = ((capacity + 31) // 32) * 32  # 4 pages -> 128
        slots, kv, layers, kv_heads, head_dim, dtype_bytes = 2, 2, 2, 1, 16, 2
        assert (
            compute_dflash_ctx_buffer_bytes(
                max_batch_size=1,
                max_ctx_len=100,
                block_size=8,
                num_attn_layers=layers,
                num_kv_heads_per_rank=kv_heads,
                head_dim=head_dim,
                dtype_bytes=dtype_bytes,
                attention_backend="TRTLLM",
            )
            == slots * kv * layers * paged_capacity * kv_heads * head_dim * dtype_bytes
        )


class TestCtxBufferBudget:
    def test_oversized_batch_is_a_config_error_naming_the_fitting_value(self):
        # A batch that overruns the buffer budget (here a tight budget at the
        # serve-time max_seq_len) is refused, and the message names the largest
        # batch that fits.
        budget = 10 * PER_SLOT_BYTES_MSL
        with pytest.raises(ValueError) as excinfo:
            _validate(max_batch_size=32, memory_budget_bytes=budget)
        message = str(excinfo.value)
        required_gib = 33 * PER_SLOT_BYTES_MSL / GIB
        fitting = budget // PER_SLOT_BYTES_MSL - 1
        assert f"{required_gib:.2f} GiB" in message
        assert f"max_batch_size estimated to fit is {fitting}" in message
        assert "max_batch_size + 1 slots" in message  # the formula

    def test_unset_max_seq_len_skips_the_buffer_check(self):
        # max_seq_len is unset at config time and resolved later from the model
        # config; falling back to the drafter's 1M positions would refuse every
        # batch, so the buffer-fit check is skipped when max_seq_len is None.
        _validate(max_batch_size=32, max_seq_len=None)  # no raise

    def test_trtllm_over_budget_warns_not_raises(self):
        # TRTLLM normally binds the managed pool and reserves no private arena,
        # so an over-budget estimate is a warning, not a hard refusal -- but it
        # is not silent, because the pool is not guaranteed and the private
        # arena fallback would allocate exactly this.
        with patch.object(dflash.logger, "warning") as mock_warning:
            _validate(
                max_batch_size=32,
                attention_backend="TRTLLM",
                memory_budget_bytes=PER_SLOT_BYTES_MSL,
            )  # no raise
        assert mock_warning.call_count == 1
        assert "managed KV-cache pool" in mock_warning.call_args[0][0]

    @pytest.mark.parametrize("bypass", [{"max_seq_len": None}, {"attention_backend": "TRTLLM"}])
    def test_token_budget_checked_before_buffer_bypass(self, bypass):
        # The token-budget check must run before the max_seq_len-None and
        # TRTLLM buffer-check bypasses, so a violated token budget still raises.
        with pytest.raises(ValueError, match="max_num_tokens >= 256"):
            _validate(max_batch_size=32, max_num_tokens=100, **bypass)

    def test_max_seq_len_caps_the_capacity(self):
        # The same batch the 1M fallback refuses fits easily once the
        # serve-time sequence cap bounds the pooled context.
        _validate(max_batch_size=32)  # max_seq_len=8192

    def test_refusal_formula_uses_max_seq_len_when_set(self):
        with pytest.raises(ValueError) as excinfo:
            _validate(max_batch_size=32, memory_budget_bytes=PER_SLOT_BYTES_MSL)
        message = str(excinfo.value)
        assert "(8192 + 8)" in message  # capacity from max_seq_len, not 1M
        required_gib = 33 * PER_SLOT_BYTES_MSL / GIB
        assert f"{required_gib:.2f} GiB" in message

    def test_token_budget_violation_is_a_config_error(self):
        # 32 * (1 + 7) = 256 > 100; the fitting batch size is 100 // 8 = 12.
        with pytest.raises(ValueError) as excinfo:
            _validate(max_batch_size=32, max_num_tokens=100)
        message = str(excinfo.value)
        assert "256" in message and "max_num_tokens >= 256" in message
        assert "at most 12" in message

    def test_healthy_config_passes_without_warning(self):
        with patch.object(dflash.logger, "warning") as mock_warning:
            _validate()  # batch 8: 9 slots at msl 8192, ~0.4 GiB <= budget
        mock_warning.assert_not_called()

    def test_narrower_than_trained_block_warns_once(self):
        with patch.object(dflash.logger, "warning") as mock_warning:
            _validate(max_draft_len=3)  # trained block_size 8 -> width 7
        assert mock_warning.call_count == 1
        assert "max_draft_len=7" in mock_warning.call_args[0][0]

    def test_no_budget_skips_the_memory_check(self):
        _validate(max_batch_size=32, memory_budget_bytes=None)

    def test_missing_geometry_skips_the_memory_check(self):
        _validate(max_batch_size=32, draft_config={"block_size": 8})

    def test_trained_width_reads_nested_dflash_config(self):
        nested = {"dflash_config": {"block_size": 8}}
        with patch.object(dflash.logger, "warning") as mock_warning:
            _validate(max_draft_len=3, draft_config=nested)
        assert mock_warning.call_count == 1


class TestWeightEstimate:
    def test_prefers_safetensors_over_bin(self, tmp_path):
        # Repos that ship both .safetensors and .bin copies of the same weights
        # must not be double-counted: prefer .safetensors, ignore the .bin.
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"b" * 50)
        (tmp_path / "pytorch_model.bin").write_bytes(b"c" * 7)
        (tmp_path / "config.json").write_text("{}")  # not a shard
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        assert estimate_checkpoint_weight_bytes(str(tmp_path)) == 150

    def test_falls_back_to_bin_without_safetensors(self, tmp_path):
        (tmp_path / "pytorch_model-00001-of-00002.bin").write_bytes(b"c" * 40)
        (tmp_path / "pytorch_model-00002-of-00002.bin").write_bytes(b"d" * 20)
        (tmp_path / "config.json").write_text("{}")

        assert estimate_checkpoint_weight_bytes(str(tmp_path)) == 60

    def test_walks_subdirectory_shards(self, tmp_path):
        sub = tmp_path / "weights"
        sub.mkdir()
        (sub / "model-00001-of-00001.safetensors").write_bytes(b"a" * 80)

        assert estimate_checkpoint_weight_bytes(str(tmp_path)) == 80

    def test_no_shards_is_none_not_zero(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        # None (estimate unavailable) rather than 0: a zero-weight estimate
        # would silently restore the total-memory bound this fix removes.
        assert estimate_checkpoint_weight_bytes(str(tmp_path)) is None

    def test_non_directory_is_none(self, tmp_path):
        assert estimate_checkpoint_weight_bytes(str(tmp_path / "absent")) is None
        assert estimate_checkpoint_weight_bytes("hf-org/hf-model") is None
        assert estimate_checkpoint_weight_bytes(None) is None


def _pool_first(total, fraction, weights=None):
    return derive_dflash_ctx_memory_budget_bytes(total, fraction, weights, arena_before_pool=False)


def _arena_first(total, fraction, weights=None):
    return derive_dflash_ctx_memory_budget_bytes(total, fraction, weights, arena_before_pool=True)


class TestBudgetDerivation:
    """Two derivations, selected by when the arena lands relative to the pool."""

    def test_pool_first_subtracts_weights_before_the_fraction(self):
        # Estimation skipped: the pool commits first as 0.8 of the post-load
        # memory, so the arena gets (100 - 50 weights - 4 overhead) * 0.2 * safety.
        total = 100 * GIB
        expected = int((total - 50 * GIB - OVERHEAD) * 0.2 * SAFETY)
        assert _pool_first(total, 0.8, 50 * GIB) == expected

    def test_pool_first_without_weights_still_tightens_the_total(self):
        total = 100 * GIB
        expected = int((total - OVERHEAD) * 0.2 * SAFETY)
        assert _pool_first(total, 0.8, None) == expected
        # And the weight-aware bound is strictly tighter.
        assert _pool_first(total, 0.8, 50 * GIB) < expected

    def test_arena_first_ignores_the_fraction(self):
        # Estimation runs: the arena is allocated before the pool is sized, so
        # free_gpu_memory_fraction bounds the pool, not the arena. The budget
        # is the post-weight headroom less an additive reserve.
        total = 100 * GIB
        expected = total - OVERHEAD - 50 * GIB - RESERVE
        assert _arena_first(total, 0.8, 50 * GIB) == expected
        assert _arena_first(total, 0.9, 50 * GIB) == expected
        assert _arena_first(total, 0.5, 50 * GIB) == expected

    def test_arena_first_without_weights_subtracts_overhead_and_reserve(self):
        total = 100 * GIB
        assert _arena_first(total, 0.8, None) == total - OVERHEAD - RESERVE
        assert _arena_first(total, 0.8, 50 * GIB) < _arena_first(total, 0.8, None)

    def test_arena_first_is_wider_than_pool_first(self):
        # The pool-first form on the same device is a small fraction of the
        # headroom; the arena-first form is most of it.
        total, weights = 100 * GIB, 50 * GIB
        assert _arena_first(total, 0.9, weights) > 5 * _pool_first(total, 0.9, weights)

    @pytest.mark.parametrize("arena_before_pool", [True, False])
    def test_weights_larger_than_device_floor_at_zero(self, arena_before_pool):
        assert (
            derive_dflash_ctx_memory_budget_bytes(
                8 * GIB, 0.8, 100 * GIB, arena_before_pool=arena_before_pool
            )
            == 0
        )

    def test_arena_first_reserve_floors_at_zero(self):
        # Headroom smaller than the reserve is a zero budget, not negative.
        assert _arena_first(OVERHEAD + RESERVE // 2, 0.8, None) == 0

    def test_mode_is_keyword_only(self):
        with pytest.raises(TypeError):
            derive_dflash_ctx_memory_budget_bytes(100 * GIB, 0.8, None)


# The L0 accuracy configs the check must accept: a single-GPU bf16 Laguna-XS
# target on B200 and its FP8 counterpart on H100, each with the Laguna-XS
# DFlash drafter (5 layers, 8 KV heads, head_dim 128, bf16), max_seq_len 4096,
# max_batch_size 128, max_draft_len 4 and free_gpu_memory_fraction 0.9. The
# arena is 2 x 129 x 5 x (4096 + 5) x 8 x 128 x 2 B = 10.09 GiB.
L0_DRAFT_CONFIG = {
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_attention_heads": 32,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
    "dflash_config": {"block_size": 16},
}
L0_ARENA_BYTES = 2 * 129 * 5 * (4096 + 5) * 8 * 128 * 2
L0_CASES = {
    # name: (device total, target weight bytes)
    "b200_bf16": (178 * GIB, 62 * GIB),
    "h100_fp8": (79 * GIB, 33 * GIB),
}


class TestL0ConfigsFitTheArenaFirstBudget:
    def _validate_l0(self, budget, arena_before_pool):
        validate_dflash_ctx_buffer_budget(
            max_batch_size=128,
            max_num_tokens=4096,
            max_seq_len=4096,
            max_draft_len=4,
            attention_backend="VANILLA",
            draft_config=L0_DRAFT_CONFIG,
            tp_size=1,
            memory_budget_bytes=budget,
            arena_before_pool=arena_before_pool,
        )

    def test_arena_size_matches_the_l0_drafter(self):
        assert (
            compute_dflash_ctx_buffer_bytes(
                128, 4096, 5, 5, 8, 128, dtype_bytes=2, attention_backend="VANILLA"
            )
            == L0_ARENA_BYTES
        )
        assert 10.0 < L0_ARENA_BYTES / GIB < 10.2

    @pytest.mark.parametrize("name", sorted(L0_CASES))
    def test_accepted_under_the_arena_first_derivation(self, name):
        total, weights = L0_CASES[name]
        budget = _arena_first(total, 0.9, weights)
        assert budget >= L0_ARENA_BYTES
        self._validate_l0(budget, arena_before_pool=True)  # no raise

    @pytest.mark.parametrize("name", sorted(L0_CASES))
    def test_pool_first_form_would_have_refused_it(self, name):
        # The same numbers under (1 - 0.9) x headroom x 0.7 budget only
        # 7-8 GiB (B200) / ~3 GiB (H100) against a 10.09 GiB arena, which is
        # the false refusal the arena-first derivation removes.
        total, weights = L0_CASES[name]
        budget = _pool_first(total, 0.9, weights)
        assert budget < L0_ARENA_BYTES
        with pytest.raises(ValueError, match="max_batch_size estimated to fit"):
            self._validate_l0(budget, arena_before_pool=False)


class TestRefusalWording:
    def test_arena_first_refusal_does_not_point_at_the_fraction(self):
        # free_gpu_memory_fraction bounds the pool, not the arena, in this
        # mode, so the remedy must not suggest lowering it.
        with pytest.raises(ValueError) as excinfo:
            _validate(
                max_batch_size=32,
                memory_budget_bytes=10 * PER_SLOT_BYTES_MSL,
                arena_before_pool=True,
            )
        message = str(excinfo.value)
        assert "before the pool is sized" in message
        assert "free_gpu_memory_fraction of what remains" in message
        assert "lower kv_cache_config.free_gpu_memory_fraction" not in message
        assert "lower max_batch_size or max_seq_len" in message

    def test_pool_first_refusal_points_at_the_fraction(self):
        with pytest.raises(ValueError) as excinfo:
            _validate(
                max_batch_size=32,
                memory_budget_bytes=10 * PER_SLOT_BYTES_MSL,
                arena_before_pool=False,
            )
        message = str(excinfo.value)
        assert "KV-cache pool commits" in message
        assert "lower kv_cache_config.free_gpu_memory_fraction" in message


class TestLlmArgsWiring:
    """The validator must run from TorchLlmArgs with the drafter's config."""

    def _args(self, tmp_path, max_batch_size, max_seq_len=8192, model="unused", **extra):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, TorchLlmArgs

        drafter_dir = tmp_path / "drafter"
        drafter_dir.mkdir(exist_ok=True)
        (drafter_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))
        # model_construct skips model-path validation; only the DFlash
        # budget helper is under test here.
        return TorchLlmArgs.model_construct(
            model=model,
            speculative_config=DFlashDecodingConfig(
                max_draft_len=7, speculative_model=str(drafter_dir)
            ),
            max_batch_size=max_batch_size,
            max_num_tokens=8192,
            max_seq_len=max_seq_len,
            tensor_parallel_size=4,
            **extra,
        )

    def _fake_device(self, total_memory):
        from types import SimpleNamespace

        from tensorrt_llm.llmapi import llm_args as llm_args_module

        return (
            patch.object(llm_args_module.torch.cuda, "is_available", return_value=True),
            patch.object(
                llm_args_module.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=total_memory),
            ),
        )

    def _target_with_weights(self, tmp_path, weight_bytes):
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        with open(target_dir / "model.safetensors", "wb") as f:
            f.truncate(weight_bytes)
        return target_dir

    def test_estimation_runs_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        assert self._args(tmp_path, max_batch_size=8)._kv_cache_estimation_runs() is True

    def test_estimation_skipped_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(SKIP_EST_ENV, "1")
        assert self._args(tmp_path, max_batch_size=8)._kv_cache_estimation_runs() is False
        monkeypatch.setenv(SKIP_EST_ENV, "0")
        assert self._args(tmp_path, max_batch_size=8)._kv_cache_estimation_runs() is True

    def test_estimation_skipped_for_vanilla_target_attention(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        args = self._args(tmp_path, max_batch_size=8, attn_backend="VANILLA")
        assert args._kv_cache_estimation_runs() is False

    def test_estimation_skipped_for_context_parallelism(self, tmp_path, monkeypatch):
        from tensorrt_llm.llmapi.llm_args import CpConfig

        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        args = self._args(tmp_path, max_batch_size=8, cp_config=CpConfig())
        assert args._kv_cache_estimation_runs() is False

    def test_oversized_batch_rejected_through_llm_args(self, tmp_path):
        # A batch that overruns a tight buffer budget at the serve-time
        # max_seq_len is refused through the TorchLlmArgs entry point.
        args = self._args(tmp_path, max_batch_size=32)
        with pytest.raises(ValueError, match="max_batch_size estimated to fit"):
            args._validate_dflash_ctx_budget(memory_budget_bytes=10 * PER_SLOT_BYTES_MSL)

    def test_default_batch_is_clamped_not_rejected(self, tmp_path):
        # max_batch_size left at its default overruns a tight token budget;
        # rather than failing construction it is clamped to what fits, with a
        # warning. An explicitly-set max_batch_size would hard-fail instead.
        from tensorrt_llm.llmapi import llm_args as llm_args_module
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig, TorchLlmArgs

        drafter_dir = tmp_path / "drafter"
        drafter_dir.mkdir()
        (drafter_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))
        args = TorchLlmArgs.model_construct(
            model="unused",
            speculative_config=DFlashDecodingConfig(
                max_draft_len=7, speculative_model=str(drafter_dir)
            ),
            max_num_tokens=100,
            max_seq_len=8192,
            tensor_parallel_size=4,
        )
        # An explicit max_tokens cap skips the GPU-based budget derivation, so
        # the clamp is exercised without touching a device.
        args.kv_cache_config = KvCacheConfig(max_tokens=1024)
        assert "max_batch_size" not in args.model_fields_set
        with patch.object(llm_args_module.logger, "warning") as mock_warning:
            args._validate_dflash_ctx_budget(memory_budget_bytes=None)
        # 100 // (1 + 7) = 12
        assert args.max_batch_size == 12
        assert mock_warning.call_count == 1
        assert "clamping max_batch_size to 12" in mock_warning.call_args[0][0]

    def test_max_seq_len_admits_the_same_batch_through_llm_args(self, tmp_path):
        # Identical shape to the rejected case, plus the serve-time cap.
        args = self._args(tmp_path, max_batch_size=32, max_seq_len=8192)
        args._validate_dflash_ctx_budget(memory_budget_bytes=BUDGET)

    def test_healthy_config_accepted_through_llm_args(self, tmp_path):
        args = self._args(tmp_path, max_batch_size=8)
        args._validate_dflash_ctx_budget(memory_budget_bytes=BUDGET)

    def test_pool_first_budget_subtracts_target_weights(self, tmp_path, monkeypatch):
        """Estimation skipped, a large target whose weights halve the device.

        With a VANILLA target attention backend the pool commits first, so
        the budget the refusal reports must be (1 - fraction) of the memory
        left AFTER the estimated per-rank weights (200 GiB / tp4 = 50 GiB)
        are subtracted -- not of the 100 GiB device total, which is how a
        "fits" verdict OOM'd at warmup.
        """
        import re

        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        target_dir = self._target_with_weights(tmp_path, 200 * GIB)
        # Batch large enough to overrun the weight-subtracted budget even at
        # the capped max_seq_len (each slot is ~40 MiB at msl 8192).
        args = self._args(
            tmp_path,
            max_batch_size=200,
            max_seq_len=8192,
            model=str(target_dir),
            attn_backend="VANILLA",
        )
        args.kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        is_available, props = self._fake_device(100 * GIB)
        with is_available, props:
            with pytest.raises(ValueError) as excinfo:
                args._validate_dflash_ctx_budget()

        expected = _pool_first(100 * GIB, 0.8, (200 * GIB) // 4)
        message = str(excinfo.value)
        assert f"estimated {expected / GIB:.2f} GiB" in message
        assert "KV-cache pool commits" in message
        # The old total-memory bound would have reported 18.00 GiB
        # ((100) * 0.2 * 0.9 less overhead); prove we are below it.
        reported = float(re.search(r"estimated ([0-9.]+) GiB", message).group(1))
        no_weight_bound = _pool_first(100 * GIB, 0.8, None)
        assert reported < no_weight_bound / GIB

    def test_arena_first_budget_admits_what_pool_first_refuses(self, tmp_path, monkeypatch):
        """Default path (estimation runs): the same config is accepted.

        201 slots at max_seq_len 8192 are ~7.9 GiB; the arena-first budget is
        100 - 4 - 50 - reserve = 40 GiB, so no refusal, while the pool-first
        form above budgets ~6.4 GiB and refuses.
        """
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        target_dir = self._target_with_weights(tmp_path, 200 * GIB)
        args = self._args(tmp_path, max_batch_size=200, max_seq_len=8192, model=str(target_dir))
        args.kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        assert 201 * PER_SLOT_BYTES_MSL > _pool_first(100 * GIB, 0.8, 50 * GIB)
        assert 201 * PER_SLOT_BYTES_MSL <= _arena_first(100 * GIB, 0.8, 50 * GIB)
        is_available, props = self._fake_device(100 * GIB)
        with is_available, props:
            args._validate_dflash_ctx_budget()  # no raise

    def test_arena_first_refusal_reports_the_arena_first_budget(self, tmp_path, monkeypatch):
        # 401 slots at msl 8192 are ~15.7 GiB; a 70 GiB device with 50 GiB of
        # per-rank weights leaves 70 - 4 - 50 - reserve = 10 GiB, so the
        # default path refuses and reports the arena-first budget.
        import re

        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        target_dir = self._target_with_weights(tmp_path, 200 * GIB)
        args = self._args(tmp_path, max_batch_size=400, max_seq_len=8192, model=str(target_dir))
        args.kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        is_available, props = self._fake_device(70 * GIB)
        with is_available, props:
            with pytest.raises(ValueError) as excinfo:
                args._validate_dflash_ctx_budget()
        message = str(excinfo.value)
        expected = _arena_first(70 * GIB, 0.8, 50 * GIB)
        assert expected == 70 * GIB - OVERHEAD - 50 * GIB - RESERVE
        reported = float(re.search(r"estimated ([0-9.]+) GiB", message).group(1))
        assert reported == pytest.approx(expected / GIB, abs=0.01)
        assert "before the pool is sized" in message

    def test_skip_env_selects_the_pool_first_budget(self, tmp_path, monkeypatch):
        # TRTLLM_SKIP_KV_CACHE_ESTIMATION=1 makes the pool commit first, so
        # the config the default path admits is refused with the pool-first
        # budget in the message.
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        monkeypatch.setenv(SKIP_EST_ENV, "1")
        target_dir = self._target_with_weights(tmp_path, 200 * GIB)
        args = self._args(tmp_path, max_batch_size=200, max_seq_len=8192, model=str(target_dir))
        args.kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        is_available, props = self._fake_device(100 * GIB)
        with is_available, props:
            with pytest.raises(ValueError) as excinfo:
                args._validate_dflash_ctx_budget()
        expected = _pool_first(100 * GIB, 0.8, 50 * GIB)
        assert f"estimated {expected / GIB:.2f} GiB" in str(excinfo.value)


class TestHubIdRevalidation:
    """A Hub repo id has no readable config.json at validation time.

    The buffer-fit check then has no drafter geometry and only the token
    budget is checked, so CachedModelLoader must re-run the budget check
    once the drafter is downloaded. A local path is fully checked at
    validation time and must not be checked twice.
    """

    def _loader(self, speculative_model):
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig, TorchLlmArgs
        from tensorrt_llm.llmapi.llm_utils import CachedModelLoader

        args = TorchLlmArgs.model_construct(
            model="unused",
            speculative_config=DFlashDecodingConfig(
                max_draft_len=7, speculative_model=speculative_model
            ),
            max_batch_size=32,
            max_num_tokens=8192,
            max_seq_len=8192,
            tensor_parallel_size=4,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.8),
        )
        return CachedModelLoader(args)

    def test_hub_id_budget_checked_after_download(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from tensorrt_llm.llmapi import llm_args as llm_args_module
        from tensorrt_llm.llmapi.llm_utils import CachedModelLoader

        monkeypatch.delenv(SKIP_EST_ENV, raising=False)
        drafter_dir = tmp_path / "drafter"
        drafter_dir.mkdir()
        (drafter_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))
        loader = self._loader(speculative_model="hf-org/dflash-drafter")

        # A device too small for 33 slots at max_seq_len 8192 (~1.3 GiB): with
        # no weight estimate the arena-first budget is total - overhead -
        # reserve = 1 GiB, so the re-run must refuse the batch on the
        # downloaded drafter's geometry (and still name a fitting batch).
        fake_props = SimpleNamespace(total_memory=OVERHEAD + RESERVE + 1 * GIB)
        with (
            patch.object(
                CachedModelLoader, "_download_hf_model_if_needed", return_value=drafter_dir
            ) as mock_download,
            patch.object(llm_args_module.torch.cuda, "is_available", return_value=True),
            patch.object(
                llm_args_module.torch.cuda, "get_device_properties", return_value=fake_props
            ),
        ):
            with pytest.raises(ValueError, match="max_batch_size estimated to fit"):
                loader()
        mock_download.assert_called_once()
        assert loader.llm_args.speculative_config.speculative_model == drafter_dir

    def test_local_path_is_not_checked_twice(self, tmp_path):
        from tensorrt_llm.llmapi import llm_utils as llm_utils_module
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

        drafter_dir = tmp_path / "drafter"
        drafter_dir.mkdir()
        (drafter_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))
        loader = self._loader(speculative_model=str(drafter_dir))

        class _Stop(Exception):
            pass

        # ModelLoader is constructed right after the speculative-model step;
        # stopping there bounds the test to the code under test.
        with (
            patch.object(TorchLlmArgs, "_validate_dflash_ctx_budget") as mock_budget,
            patch.object(llm_utils_module, "ModelLoader", side_effect=_Stop),
        ):
            with pytest.raises(_Stop):
                loader()
        mock_budget.assert_not_called()
