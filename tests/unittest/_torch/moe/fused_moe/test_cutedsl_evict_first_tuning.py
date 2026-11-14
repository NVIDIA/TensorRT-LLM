# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL evict-first FC2 finalize runner tunes up to the token bound its caller selects it for
(``MoEDecodeOptions.fc2_weights_evict_first_rows``), not only up to the tuning call's tokens; where the tuning call
is at that bound (the Qwen3.5 DEP4 GEN reference) the tuned profiles and every cache key are unchanged."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as cute_dsl_custom_ops
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="Requires CUTLASS DSL"),
]

# Qwen3.5-397B NVFP4 under DEP4: 512 experts, top-10, EP4 (128 local experts), hidden 4096, expert I 1024.
_EXPERTS, _TOP_K, _LOCAL, _HIDDEN, _INTER = 512, 10, 128, 4096, 1024
# DEP4 GEN reference: CUDA graphs up to 48 requests x 7 rows (MTP 6) per rank. The evict-first bound is
# those rows x dp_size; the generation-only autotuner warmup runs 48 x 7 rows per rank, which the EP4
# dispatch receives from every rank.
_DEP4_BOUND = 48 * 7 * 4
_DEP4_TUNING_TOKENS = 4 * 48 * 7


def _runner(monkeypatch, tile_size: int, evict_first: bool, max_tokens: int):
    monkeypatch.setattr(cute_dsl_custom_ops, "get_sm_version", lambda: 103)
    return cute_dsl_custom_ops.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
        _EXPERTS,
        _TOP_K,
        _LOCAL,
        0,
        tile_size,
        torch.bfloat16,
        weights_evict_first=evict_first,
        weights_evict_first_max_tokens=max_tokens,
    )


def _helper(tile_size: int):
    return cute_dsl_custom_ops.GroupedGemmInputsHelper(_EXPERTS, _TOP_K, _LOCAL, 0, tile_size)


def _inputs(tokens: int, tile_size: int) -> list:
    """Shapes of the finalize op's inputs for a call of ``tokens`` tokens (meta tensors: only shapes matter)."""
    rows = _helper(tile_size).get_max_num_permuted_tokens(tokens)
    meta = dict(device="meta")
    return [
        torch.empty((rows, _INTER // 2), **meta),
        torch.empty((_LOCAL, _HIDDEN, _INTER // 2), **meta),
        torch.empty((rows * _INTER // 16,), **meta),
        torch.empty((_LOCAL, _HIDDEN, _INTER // 16), **meta),
        torch.empty((_LOCAL,), **meta),
        torch.empty((tokens, _HIDDEN), **meta),
        torch.empty((rows // tile_size,), **meta),
        torch.empty((rows // tile_size,), **meta),
        torch.empty((rows,), **meta),
        torch.empty((1,), **meta),
        torch.empty((tokens, _TOP_K), **meta),
    ]


def _profiles(config, inputs) -> list:
    return [p.get_opt_shapes() for p in AutoTuner.get()._optimization_profiles(config, inputs)]


@pytest.mark.parametrize("tile_size", [128, 256])
def test_evict_first_runner_tunes_up_to_its_bound(monkeypatch, tile_size) -> None:
    helper = _helper(tile_size)
    evict = _runner(monkeypatch, tile_size, True, _DEP4_BOUND).get_tuning_config()
    assert evict.tune_max_num_tokens == helper.get_max_num_permuted_tokens(_DEP4_BOUND)
    # Without the evict-first policy (or a bound) the runner keeps tuning up to the tuning call.
    assert (
        _runner(monkeypatch, tile_size, False, _DEP4_BOUND).get_tuning_config().tune_max_num_tokens
        is None
    )
    assert _runner(monkeypatch, tile_size, True, 0).get_tuning_config().tune_max_num_tokens is None
    # A tuning call below the bound (e.g. fewer warmup tokens than the evict-first rows) still tunes every
    # bucket that a call the runner serves (1..bound tokens) looks up; without the bound the larger ones are
    # missing, so those calls ran the default tactic.
    small = _inputs(_DEP4_BOUND // 4, tile_size)
    untuned = _runner(monkeypatch, tile_size, True, 0).get_tuning_config()
    looked_up = {
        helper.map_to_tuning_buckets(helper.get_max_num_permuted_tokens(t))
        for t in range(1, _DEP4_BOUND + 1)
    }
    tuned = lambda config: {p[0][0] for p in _profiles(config, small)}  # noqa: E731
    assert looked_up <= tuned(evict)
    assert not looked_up <= tuned(untuned)


@pytest.mark.parametrize("tile_size", [128, 256])
def test_dep4_reference_tuning_is_unchanged(monkeypatch, tile_size) -> None:
    """DEP4 GEN reference: the tuning call is at the evict-first bound, so the bound adds no profile, and every
    runtime call the evict-first runner serves (1..bound tokens) reads the same cache key."""
    assert _DEP4_TUNING_TOKENS == _DEP4_BOUND
    with_bound = _runner(monkeypatch, tile_size, True, _DEP4_BOUND).get_tuning_config()
    without = _runner(monkeypatch, tile_size, True, 0).get_tuning_config()
    tuning_inputs = _inputs(_DEP4_TUNING_TOKENS, tile_size)
    assert _profiles(with_bound, tuning_inputs) == _profiles(without, tuning_inputs)

    def key(config, shapes: tuple, at_runtime: bool) -> tuple:
        return AutoTuner._find_nearest_profile(
            tuple(torch.Size(s) for s in shapes),
            config.dynamic_tensor_specs,
            config.constraint_specs,
            config.tune_max_num_tokens,
            at_runtime,
        )

    # Lookups at runtime (bucketed) ...
    for tokens in range(1, _DEP4_BOUND + 1):
        shapes = tuple(tuple(t.shape) for t in _inputs(tokens, tile_size))
        assert key(with_bound, shapes, True) == key(without, shapes, True), tokens
    # ... and the keys the tuned profiles are stored under.
    for profile in _profiles(without, tuning_inputs):
        assert key(with_bound, profile, False) == key(without, profile, False), profile


def test_finalize_op_takes_the_bound() -> None:
    schema = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell.default._schema
    assert "weights_evict_first_max_tokens" in [a.name for a in schema.arguments]
