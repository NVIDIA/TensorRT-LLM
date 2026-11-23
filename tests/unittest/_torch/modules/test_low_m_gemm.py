# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests in this module use mocks for GPU operations and run on any platform.
# Apply ``_skip_no_cuda`` / ``_skip_non_sm10x`` to tests that actually
# invoke the kernel hardware rather than the Python dispatch logic.

import dataclasses
import functools
import gc
import sys
import threading
import time
import weakref
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

import tensorrt_llm._torch.modules.low_m_gemm as _mod
from tensorrt_llm._torch.autotuner import AutoTuner, TunableRunner, autotune
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture
from tensorrt_llm._torch.modules.low_m_gemm import (
    _BACKEND_ENV,
    LOW_M_GEMM_OPT_IN_ATTR,
    LowMGemmDispatcher,
    _parse_enabled,
    bf16_lowm_linear,
    fp8_grouped_lowm,
    lowm_fp8_supported,
    lowm_tiles,
    mxfp8_lowm_index_query,
    mxfp8_lowm_linear,
)
from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight
from tensorrt_llm._utils import is_blackwell_datacenter, is_sm_100f

# ---------------------------------------------------------------------------
# Skip helpers — apply to tests that need hardware, not to mocked unit tests.
# ---------------------------------------------------------------------------

_skip_no_cuda = pytest.mark.skipif(
    torch.cuda.device_count() == 0,
    reason="requires at least one CUDA GPU",
)
_skip_non_sm10x = pytest.mark.skipif(
    torch.cuda.device_count() == 0 or not is_sm_100f(),
    reason="requires SM10x GPU (SM100/SM103)",
)
_skip_no_lowm_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not lowm_fp8_supported(),
    reason="needs SM100/SM103",
)

# ---------------------------------------------------------------------------
# Backend env-var parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", False),  # canonical disabled value
        ("auto", True),  # canonical enabled value
        ("flashinfer", True),  # legacy enabled alias
        ("cublaslt", False),  # legacy disabled alias
    ],
)
def test_parse_enabled(value: str, expected: bool, monkeypatch) -> None:
    monkeypatch.setenv(_BACKEND_ENV, value)
    assert _parse_enabled() is expected


def test_parse_enabled_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv(_BACKEND_ENV, "split-all-shapes")
    with pytest.raises(ValueError, match=_BACKEND_ENV):
        _parse_enabled()


# ---------------------------------------------------------------------------
# Dispatcher prepare()
# ---------------------------------------------------------------------------


def test_prepare_labels_modules(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)

    module = torch.nn.Linear(8, 8)
    dispatcher = LowMGemmDispatcher()
    dispatcher.prepare(module)

    assert dispatcher._prepared
    assert module._low_m_gemm_name == ""
    # apply_low_m_gemm reads _low_m_gemm_dispatcher to reach the per-engine instance;
    # a missing binding silently falls back to the shared global _DISPATCHER.
    assert module._low_m_gemm_dispatcher is dispatcher


def test_attach_reaches_the_router_gate(monkeypatch) -> None:
    """The gate sets the attribute literally; attach() reads it by constant."""
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate

    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)
    root = torch.nn.Module()
    root.gate = Qwen3NextGate(hidden_size=8, num_experts=4, top_k=2, dtype=torch.bfloat16)
    dispatcher = LowMGemmDispatcher()
    dispatcher.attach(root)

    assert getattr(Qwen3NextGate, LOW_M_GEMM_OPT_IN_ATTR, False)
    assert root.gate._low_m_gemm_dispatcher is dispatcher
    assert root.gate._low_m_gemm_name == "gate"


def test_a_later_attach_keeps_the_models_own_dispatcher(monkeypatch) -> None:
    """The loader's prepare_low_m_gemm after load must not replace a dispatcher the model attached itself
    (Qwen3.5: limit_rows, whole_bucket_direct_tiles, per_rank_tuning); new modules still get the later one."""
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)
    root = torch.nn.Module()
    root.own = torch.nn.Linear(8, 8)
    model_dispatcher = _mod.prepare_low_m_gemm(root, force=True, whole_bucket_direct_tiles=True)
    root.later = torch.nn.Linear(8, 8)
    loader_dispatcher = _mod.prepare_low_m_gemm(root)

    assert root.own._low_m_gemm_dispatcher is model_dispatcher
    assert root.later._low_m_gemm_dispatcher is loader_dispatcher


# ---------------------------------------------------------------------------
# linear.py fast pre-filter
# ---------------------------------------------------------------------------


def test_linear_fast_rejects_m_above_max_m(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)

    assert _mod._should_apply_low_m_gemm(torch.empty((32, 128)))
    assert not _mod._should_apply_low_m_gemm(torch.empty((33, 128)))


def test_takes_rows_follows_the_module_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", False)
    dispatcher = LowMGemmDispatcher()
    plain, limited = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
    for module in (plain, limited):
        module._low_m_gemm_dispatcher = dispatcher
    dispatcher.limit_rows(limited, 14)

    assert not _mod.low_m_gemm_takes_rows(torch.nn.Linear(8, 8), 4)
    assert _mod.low_m_gemm_takes_rows(plain, _mod._MAX_M)
    assert not _mod.low_m_gemm_takes_rows(plain, _mod._MAX_M + 1)
    assert _mod.low_m_gemm_takes_rows(limited, 14)
    assert not _mod.low_m_gemm_takes_rows(limited, 15)


def test_static_fp8_input_scale_is_one_decision(monkeypatch) -> None:
    """Linear.static_fp8_input_scale: the scale where an FP8 QDQ layer takes an E4M3 input at these rows, else
    None (another quantization, LoRA, bias, a non-scalar scale, or rows on the low-M path)."""
    from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod, Linear, W8A16FP8LinearMethod

    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)
    scale = torch.tensor([0.5], dtype=torch.float32)

    def layer(**overrides):
        fields = dict(
            quant_method=FP8QDQLinearMethod(),
            lora=None,
            bias=None,
            input_scale=scale,
            force_dynamic_quantization=False,
        )
        fields.update(overrides)
        return MagicMock(
            spec=[*fields, "_low_m_gemm_dispatcher"], _low_m_gemm_dispatcher=None, **fields
        )

    rows = _mod._MAX_M + 1
    assert Linear.static_fp8_input_scale(layer(), rows) is scale
    assert Linear.static_fp8_input_scale(layer(), _mod._MAX_M) is None
    assert Linear.static_fp8_input_scale(layer(quant_method=W8A16FP8LinearMethod()), rows) is None
    assert Linear.static_fp8_input_scale(layer(lora=object()), rows) is None
    assert Linear.static_fp8_input_scale(layer(bias=torch.zeros(4)), rows) is None
    assert Linear.static_fp8_input_scale(layer(force_dynamic_quantization=True), rows) is None
    assert Linear.static_fp8_input_scale(layer(input_scale=scale.half()), rows) is None


# ---------------------------------------------------------------------------
# apply() — shape routing and weight transpose
# ---------------------------------------------------------------------------


def test_apply_routes_correct_shapes(monkeypatch) -> None:
    """apply() must flatten input, transpose weight, and restore the batch shape."""
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")

    dispatcher = LowMGemmDispatcher()
    dispatcher._prepared = True
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", staticmethod(lambda *_: True))

    from tensorrt_llm._torch.modules.low_m_gemm import _SplitKGemmRunner

    captured: dict = {}

    def fake_run_splitk_dense(a, b_t, bias, out, pdl, tactic):
        captured.update(a_shape=a.shape, b_t_shape=b_t.shape, out_shape=out.shape)
        out.fill_(1.0)
        return out

    splitk_module = MagicMock()
    splitk_module.run_splitk_dense = fake_run_splitk_dense
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk",
        splitk_module,
    )

    fake_runner = _SplitKGemmRunner(has_bias=True, pdl=False)
    mock_at = MagicMock()
    mock_at.choose_one.return_value = (fake_runner, (64, 8, 4, 6))
    monkeypatch.setattr(_mod, "AutoTuner", MagicMock(get=staticmethod(lambda: mock_at)))

    dispatcher._runner_no_bias = _SplitKGemmRunner(has_bias=False, pdl=False)
    dispatcher._runner_with_bias = fake_runner
    from tensorrt_llm._torch.autotuner import TuningConfig
    from tensorrt_llm._torch.modules.low_m_gemm import _M_DIM_SPEC

    dispatcher._tuning_config = TuningConfig(
        dynamic_tensor_specs=(_M_DIM_SPEC,), use_cold_l2_cache=True, use_cuda_graph=False
    )

    input_tensor = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    weight = torch.empty((256, 128), dtype=torch.bfloat16)
    bias = torch.empty((256,), dtype=torch.bfloat16)

    # Give the module a stable name so the buffer-key assertion is deterministic.
    test_linear = torch.nn.Linear(1, 1)
    test_linear._low_m_gemm_name = "test_linear"

    with torch.inference_mode():
        output = dispatcher.apply(test_linear, input_tensor, weight, bias)

    assert output.shape == (2, 2, 256)
    assert captured["a_shape"] == torch.Size([4, 128])  # flattened [M, K]
    assert captured["b_t_shape"] == torch.Size([128, 256])  # transposed [K, N]
    assert captured["out_shape"] == torch.Size([4, 256])
    splitk_module.SplitKTactic.assert_called_with(64, 8, 4, 6)
    # Verify the pre-allocated output buffer was cached for (module, m_bucket=4, n=256).
    # The key now includes the module name to prevent aliasing across modules with
    # identical output shapes (e.g. gate_proj / up_proj in a SwiGLU MLP).
    buf_key = ("test_linear", 4, 256)
    assert buf_key in dispatcher._output_buffers
    assert dispatcher._output_buffers[buf_key].shape == torch.Size([4, 256])


# ---------------------------------------------------------------------------
# Force-active path (implicit activation via use_cute_dsl_bf16_gemm)
# ---------------------------------------------------------------------------


def test_prepare_force_initialises_without_env_var(monkeypatch) -> None:
    """prepare(force=True) must initialise runners even with LOW_M_GEMM_ACTIVE=False."""
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", False)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")

    dispatcher = LowMGemmDispatcher()
    dispatcher.prepare(torch.nn.Linear(8, 8), force=True)

    assert dispatcher._prepared
    # Runners must be populated even without the env var.
    assert dispatcher._runner_no_bias is not None
    assert dispatcher._runner_with_bias is not None


def test_apply_force_active_bypasses_low_m_gemm_active(monkeypatch) -> None:
    """apply(force_active=True) must execute split-K even when LOW_M_GEMM_ACTIVE=False."""
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", False)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")

    dispatcher = LowMGemmDispatcher()
    dispatcher._prepared = True
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", staticmethod(lambda *_: True))

    from tensorrt_llm._torch.modules.low_m_gemm import _SplitKGemmRunner

    reached: dict = {}

    def fake_run(a, b_t, bias, out, pdl, tactic):
        reached["called"] = True
        out.fill_(1.0)
        return out

    splitk_module = MagicMock()
    splitk_module.run_splitk_dense = fake_run
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk",
        splitk_module,
    )

    fake_runner = _SplitKGemmRunner(has_bias=False, pdl=False)
    mock_at = MagicMock()
    mock_at.choose_one.return_value = (fake_runner, (64, 8, 4, 6))
    monkeypatch.setattr(_mod, "AutoTuner", MagicMock(get=staticmethod(lambda: mock_at)))

    dispatcher._runner_no_bias = fake_runner
    dispatcher._runner_with_bias = fake_runner
    from tensorrt_llm._torch.autotuner import TuningConfig
    from tensorrt_llm._torch.modules.low_m_gemm import _M_DIM_SPEC

    dispatcher._tuning_config = TuningConfig(
        dynamic_tensor_specs=(_M_DIM_SPEC,), use_cold_l2_cache=True, use_cuda_graph=False
    )

    input_tensor = torch.empty((4, 128), dtype=torch.bfloat16)
    weight = torch.empty((256, 128), dtype=torch.bfloat16)

    with torch.inference_mode():
        output = dispatcher.apply(
            torch.nn.Linear(1, 1), input_tensor, weight, None, force_active=True
        )

    assert output is not None, "force_active=True should bypass LOW_M_GEMM_ACTIVE=False"
    assert reached.get("called"), "split-K kernel was not invoked"


def test_attach_force_survives_a_lazy_init_by_an_ordinary_apply(monkeypatch) -> None:
    """attach(force=True), then a first apply() without force_active (a BF16 Linear call): the
    lazy init keeps the force and builds the runners and the tuning config the race reads."""
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", False)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", staticmethod(lambda *_: True))

    module = torch.nn.Linear(128, 256, bias=False)
    dispatcher = LowMGemmDispatcher()
    dispatcher.attach(module, force=True)
    assert not dispatcher._prepared

    def fake_runner(inputs, tactic, bias, out):
        return out.fill_(1.0)

    def choose_one(op, runners, tuning_config, inputs, **kwargs):
        assert tuning_config is not None
        assert all(runner is not None for runner in runners)
        return fake_runner, -1

    mock_at = MagicMock()
    mock_at.choose_one.side_effect = choose_one
    monkeypatch.setattr(_mod, "AutoTuner", MagicMock(get=staticmethod(lambda: mock_at)))

    with torch.inference_mode():
        output = _mod.apply_low_m_gemm(
            module, torch.empty((4, 128), dtype=torch.bfloat16), module.weight, None
        )

    assert dispatcher._force and dispatcher._prepared
    assert output is not None and mock_at.choose_one.call_count == 1


@torch.inference_mode()
def test_single_cta_k_fp8_race(monkeypatch) -> None:
    """single_cta_k races the warp-MMA tactics without split-K alone (own runner id); an untuned shape takes the
    heuristic tactic, so the fused user always gets one."""
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(_mod.low_m_mma_gemm, "warp_mma_supported", lambda: True)
    dispatcher = LowMGemmDispatcher()
    dispatcher._init_runners()
    plain, single = dispatcher._fp8_runners[True][1], dispatcher._fp8_runners_single_cta_k[True][0]
    assert dispatcher._fp8_runners_single_cta_k[True] == [single]
    assert single.unique_id() != plain.unique_id()

    tactics = [(16, 2, 1, 0, 1), (8, 4, 2, 0, 1), (8, 2, 4, 1, 1)]
    monkeypatch.setattr(_mod, "_warp_mma_tactics", lambda *args, **kwargs: tactics)
    x = torch.empty((7, 4096), dtype=torch.bfloat16)
    w = torch.empty((1024, 4096), dtype=torch.float8_e4m3fn)
    scale = torch.ones((1,), dtype=torch.float32)
    assert single.get_valid_tactics([x, w], None) == [(16, 2, 1, 0, 1)]
    assert plain.get_valid_tactics([x, w], None) == tactics

    mock_at = MagicMock(is_tuning_mode=False)
    mock_at.choose_one.return_value = (single, (16, 2, 1, 0, 1))
    mock_at.profiling_cache.search_cache.return_value = (False, 0, -1, None)
    monkeypatch.setattr(_mod, "AutoTuner", MagicMock(get=staticmethod(lambda: mock_at)))
    runner, tactic, _ = dispatcher._fp8_choice(
        x, w, scale, scale, torch.bfloat16, single_cta_k=True
    )
    assert mock_at.choose_one.call_args[0][1] == [single] and runner is single
    assert dispatcher.warp_mma_tactic(
        7, w, scale, scale, single_cta_k=True
    ) == _mod.low_m_mma_gemm.default_tactic(7, 1024, 4096)
    assert dispatcher.warp_mma_tactic(7, w, scale, scale) is None


# ---------------------------------------------------------------------------
# _SplitKGemmRunner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _DirectGemmRunner
# ---------------------------------------------------------------------------


def test_direct_runner_no_tactics_when_bias_present(monkeypatch) -> None:
    """_DirectGemmRunner must return [] when bias is provided."""
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    from tensorrt_llm._torch.modules.low_m_gemm import _DirectGemmRunner

    direct_module = MagicMock()
    direct_module.default_tactic = MagicMock(return_value=MagicMock())
    direct_module.autotune_tactics = MagicMock(return_value=[])
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.kernels.dense_bf16_gemm_direct",
        direct_module,
    )

    runner = _DirectGemmRunner(pdl=False)
    bias = torch.empty((256,), dtype=torch.bfloat16)
    tactics = runner.get_valid_tactics(
        [torch.empty((4, 128)), torch.empty((128, 256))],
        MagicMock(),
        bias=bias,
    )
    assert tactics == [], "direct runner must return [] for biased calls"
    direct_module.default_tactic.assert_not_called()


def test_direct_runner_tactics_serialisable(monkeypatch) -> None:
    """_DirectGemmRunner tactics are JSON-serialisable (block, outputs, rows) 3-tuples."""
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    import dataclasses as _dc

    from tensorrt_llm._torch.modules.low_m_gemm import _DirectGemmRunner

    direct_module = MagicMock()
    direct_module.autotune_tactics = lambda m, n, k: []

    @_dc.dataclass(frozen=True)
    class FakeTactic:
        block_size: int = 256
        outputs_per_block: int = 2
        rows_per_block: int = 4

    direct_module.default_tactic = lambda m, n, k: FakeTactic()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.kernels.dense_bf16_gemm_direct",
        direct_module,
    )

    runner = _DirectGemmRunner(pdl=False)
    tactics = runner.get_valid_tactics(
        [torch.empty((4, 128)), torch.empty((128, 256))],
        MagicMock(),
    )

    assert isinstance(tactics, list) and len(tactics) >= 1
    for t in tactics:
        assert isinstance(t, tuple) and len(t) == 3
        assert all(isinstance(v, int) for v in t)


@_skip_non_sm10x
@torch.inference_mode()
def test_cached_direct_tactic_uses_exact_small_m_and_validates_large_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip exact M=3 and reject an incompatible M=16-bucket tactic."""
    from tensorrt_llm._torch.autotuner import TuningConfig
    from tensorrt_llm._torch.modules.low_m_gemm import (
        _M_DIM_SPEC,
        _M_TUNING_BUCKETS,
        _CuBLASGemmRunner,
        _DirectGemmRunner,
        _SplitKGemmRunner,
    )

    n, k = 256, 8192
    tactics = {3: (256, 2, 3), 10: (256, 2, 8)}
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    direct = _DirectGemmRunner(pdl=False)
    splitk = _SplitKGemmRunner(has_bias=False, pdl=False)
    cublas = _CuBLASGemmRunner(has_bias=False)
    direct.forward = MagicMock(wraps=direct.forward)
    splitk.forward = MagicMock(wraps=splitk.forward)
    cublas.forward = MagicMock(wraps=cublas.forward)

    tuning_config = TuningConfig(dynamic_tensor_specs=(_M_DIM_SPEC,))
    tuner = _mod.AutoTuner()
    profiles = tuner._optimization_profiles(
        tuning_config,
        [torch.empty((1, k), dtype=torch.bfloat16, device="cuda"), weight.t()],
    )
    assert tuple(int(profile.get_opt_shapes()[0][0]) for profile in profiles) == _M_TUNING_BUCKETS

    custom_op = "test::low_m_exact_cache_round_trip"
    input_shapes = (torch.Size((3, k)), torch.Size((k, n)))
    cache_key = tuner.profiling_cache.get_cache_key(
        custom_op,
        direct,
        input_shapes,
        tuning_config,
        apply_map_to_tuning_buckets=False,
    )
    tuner.profiling_cache[cache_key] = (2, tactics[3], 0.0)
    cache_hit, runner_id, cached_tactic, _ = tuner.profiling_cache.search_cache(
        custom_op,
        [cublas, splitk, direct],
        input_shapes,
        tuning_config,
    )
    assert (cache_hit, runner_id, cached_tactic) == (True, 2, tactics[3])

    dispatcher = LowMGemmDispatcher()
    dispatcher._prepared = True
    dispatcher._direct = direct
    dispatcher._runner_no_bias = splitk
    dispatcher._runner_with_bias = _SplitKGemmRunner(has_bias=True, pdl=False)
    dispatcher._cublas_no_bias = cublas
    dispatcher._cublas_with_bias = _CuBLASGemmRunner(has_bias=True)
    dispatcher._tuning_config = tuning_config

    mock_at = MagicMock()
    mock_at.choose_one.side_effect = lambda _op, _runners, _config, inputs, **_: (
        direct,
        tactics[int(inputs[0].shape[0])],
    )
    monkeypatch.setattr(_mod, "AutoTuner", MagicMock(get=staticmethod(lambda: mock_at)))

    assert all(_M_DIM_SPEC.map_to_tuning_buckets(m) == m for m in range(1, 9))
    assert _M_DIM_SPEC.map_to_tuning_buckets(10) == 16

    module = torch.nn.Linear(1, 1)
    module._low_m_gemm_name = "direct_tactic_cache"
    # M=3 retrieves its exact tactic. M=10 shares the M=16 cache bucket and
    # must reject that bucket's incompatible row-8 direct tactic.
    for m in (3, 10):
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        actual = dispatcher.apply(module, a, weight, None, force_active=True)
        torch.testing.assert_close(actual, torch.mm(a, weight.t()), rtol=1e-2, atol=5e-3)

    assert direct.forward.call_args.kwargs["tactic"] == tactics[3]
    assert direct.forward.call_count == 1
    assert cublas.forward.call_args.kwargs["tactic"] == -1
    assert cublas.forward.call_count == 1
    splitk.forward.assert_not_called()


# ---------------------------------------------------------------------------
# _SplitKGemmRunner
# ---------------------------------------------------------------------------


def test_runner_get_valid_tactics_returns_serialisable_tuples(monkeypatch) -> None:
    from tensorrt_llm._torch.modules.low_m_gemm import _SplitKGemmRunner

    runner = _SplitKGemmRunner(has_bias=False, pdl=False)

    splitk_module = MagicMock()
    splitk_module.default_tactic = lambda m, n, k: object()
    splitk_module.autotune_tactics = lambda m, n, k: []
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk",
        splitk_module,
    )

    import dataclasses as _dc

    monkeypatch.setattr(_dc, "astuple", lambda _: (64, 8, 4, 6))

    tactics = runner.get_valid_tactics(
        [torch.empty((4, 128)), torch.empty((128, 256))], MagicMock()
    )

    assert isinstance(tactics, list) and len(tactics) >= 1
    for t in tactics:
        assert isinstance(t, tuple) and len(t) == 4
        assert all(isinstance(v, int) for v in t)


# Small-M FP8 warp-MMA GEMM (CuTe DSL) against a dequantized reference. Its split-output, grouped and
# RoPE variants serve DSpark's TP attention, which test_dspark_cute_dsl_attention.py covers.


@_skip_no_lowm_fp8
@pytest.mark.parametrize("m", [1, 64])
def test_fp8_lowm_matches_dequant_reference(m):
    n, k = 1792, 5120
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(115, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ dequant_mxfp8_weight(w, s).t()
    # BF16 output rounding only: the kernel accumulates the exact dequantized products in FP32.
    torch.testing.assert_close(
        mxfp8_lowm_linear(x, w, s).float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()
    )


@_skip_no_lowm_fp8
@pytest.mark.parametrize("m", [1, 16])
def test_fp8_lowm_rms_norm_matches_norm_then_gemm(m):
    """The A RMS norm projects ``rms_norm(x) * gamma`` with the row scale applied in FP32 after
    the MMA; a separate BF16 norm and the same kernel differ by BF16 rounding of A only."""
    n, k, eps = 4096, 1280, 1e-6
    g = torch.Generator(device="cuda").manual_seed(m)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(115, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    x = (torch.randn(m, k, device="cuda", generator=g) * 3).to(torch.bfloat16)
    gamma = (torch.rand(k, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    ref = (normalized * gamma.float()) @ dequant_mxfp8_weight(w, s).t()
    actual = mxfp8_lowm_linear(x, w, s, rms_norm=(gamma, eps)).float()
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item())
    separate = mxfp8_lowm_linear((normalized * gamma.float()).to(torch.bfloat16), w, s).float()
    torch.testing.assert_close(actual, separate, rtol=1e-2, atol=1e-2 * ref.abs().max().item())
    # Output RoPE rotates each pair with the row scale on both partners.
    positions = torch.randint(0, 64, (m,), device="cuda", dtype=torch.int32, generator=g)
    rope = (positions, torch.randn(64, 2, 32, device="cuda", generator=g), 128)
    roped = mxfp8_lowm_linear(x, w, s, rms_norm=(gamma, eps), rope=rope).float()
    separate = mxfp8_lowm_linear(
        (normalized * gamma.float()).to(torch.bfloat16), w, s, rope=rope
    ).float()
    torch.testing.assert_close(roped, separate, rtol=1e-2, atol=1e-2 * separate.abs().max().item())


@_skip_no_lowm_fp8
@torch.inference_mode()
@pytest.mark.parametrize("m", [1, 5, 8])
def test_fp8_lowm_bf16_dequant_and_swap_ab_bitwise(m, monkeypatch):
    """The BF16 weight conversion and, at <= 8 rows, two tiles per CTA with swapped MMA operands give
    the plain kernel's bits (MXFP8 with and without the RMS norm, and the grouped FP8 GEMM)."""
    g = torch.Generator(device="cuda").manual_seed(m)

    def run(options, fn):
        monkeypatch.setattr(_mod, "lowm_mma_options", lambda: options)
        return fn()

    def bits(y):
        return y.view(torch.int16)

    plain, dequant, both = (False, False), (True, False), (True, True)
    n, k, edge = 4096, 1280, 64
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(110, 125, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    # The first output columns take the NaN codes and full-range scale bytes; the rest stay finite.
    w.view(torch.uint8)[: edge // 2, ::97] = 0x7F
    w.view(torch.uint8)[: edge // 4, 1::97] = 0xFF
    s[edge // 2 : edge] = torch.randint(
        0, 256, (edge // 2, k // 32), device="cuda", dtype=torch.uint8, generator=g
    )
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    gamma = (torch.rand(k, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    tiles = 2 if m <= 8 else 1
    for norm in (None, (gamma, 1e-6)):
        ref = run(plain, lambda: mxfp8_lowm_linear(x, w, s, rms_norm=norm))
        assert torch.isfinite(ref[:, edge:]).all()
        assert torch.equal(
            bits(run(dequant, lambda: mxfp8_lowm_linear(x, w, s, rms_norm=norm))), bits(ref)
        )
        swapped = run(both, lambda: mxfp8_lowm_linear(x, w, s, rms_norm=norm, n_tiles=tiles))
        assert torch.equal(bits(swapped), bits(ref))
    # Split outputs whose boundary falls inside a CTA's two tiles.
    split = run(both, lambda: torch.cat(mxfp8_lowm_linear(x, w, s, split_n=1032, n_tiles=tiles), 1))
    assert torch.equal(bits(split), bits(run(plain, lambda: mxfp8_lowm_linear(x, w, s))))

    groups, rank, head_dim, heads = 2, 1024, 512, 8
    depth = heads * head_dim
    wa = (torch.randn(groups * rank, depth, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    sa = torch.exp2(
        torch.randint(-12, -4, (groups * rank // 128, depth // 128), device="cuda", generator=g)
    ).float()
    sa[0, :4] = torch.tensor([0.3, 1.0 + 2**-9, 3e-3, 2.0**-130], device="cuda")
    xa = torch.randn(m, groups * depth, device="cuda", generator=g).to(torch.bfloat16)
    positions = torch.randint(0, 4096, (m,), device="cuda", dtype=torch.int32, generator=g)
    cos_sin = torch.randn(4096, 2, 32, device="cuda", generator=g)
    rope = (positions, cos_sin, head_dim)
    ref = run(plain, lambda: fp8_grouped_lowm(xa, wa, sa, rank, inverse_rope=rope))
    swapped = run(
        both, lambda: fp8_grouped_lowm(xa, wa, sa, rank, inverse_rope=rope, n_tiles=tiles)
    )
    assert torch.equal(bits(swapped), bits(ref))


@_skip_no_lowm_fp8
@torch.inference_mode()
@pytest.mark.parametrize("m", [1, 12])
def test_fp8_lowm_index_query_matches_packed_rope_projection(m):
    """The packed index queries are ``pack_index_queries_split`` of the RoPE'd projection, bit for
    bit, with zero, tiny, huge, NaN and infinite rows."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_index_queries_split,
    )

    n, k = 4096, 1280
    g = torch.Generator(device="cuda").manual_seed(m)
    w, s = torch.ops.trtllm.mxfp8_quantize(
        (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16), True
    )
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    for row, scale in zip(range(1, m), (0.0, 1e-30, 1e30, float("nan"), float("inf"))):
        x[row] *= scale
    positions = torch.randint(0, 64, (m,), device="cuda", dtype=torch.int32, generator=g)
    cos_sin = torch.randn(64, 2, 32, device="cuda", generator=g)
    query = mxfp8_lowm_linear(x, w, s, rope=(positions, cos_sin, 128), scale_swizzled=True)
    expected = pack_index_queries_split(query.view(m, n // 128, 128))
    for got, want in zip(mxfp8_lowm_index_query(x, w, s, positions, cos_sin), expected):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_blackwell_datacenter(),
    reason="the shared expert stores the SwiGLU on SM100/SM103",
)
@torch.inference_mode()
@pytest.mark.parametrize("m", [6])
def test_shared_expert_piece_stores_the_swiglu(m, monkeypatch):
    """The full-TP shared-expert piece's gate_up GEMM stores the clamped SwiGLU: the bits of the
    GEMM, the separate SwiGLU launch and the down GEMM, eagerly and in graph replays."""
    from tensorrt_llm._torch.models.modeling_deepseekv4 import full_tp_shared_expert_forward
    from tensorrt_llm._torch.modules import swiglu as swiglu_module

    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    limit, k, inter = 10.0, 5120, 320
    g = torch.Generator(device="cuda").manual_seed(m * 1000 + inter)

    def weight(n, kk):
        w = (torch.randn(n, kk, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(115, 128, (n, kk // 32), device="cuda", dtype=torch.uint8, generator=g)
        return w, s

    (w13, s13), (w2, s2) = weight(2 * inter, k), weight(k, inter)
    x = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)

    def new_input(scale):
        x.copy_(torch.randn(m, k, device="cuda", generator=g) * scale)
        x[0] *= 16  # past the clamp limit

    def separate():
        gate_up = mxfp8_lowm_linear(x, w13, s13)
        return mxfp8_lowm_linear(swiglu_module.swiglu(gate_up, swiglu_limit=limit), w2, s2)

    def no_swiglu(*args, **kwargs):
        raise AssertionError("the shared expert launched the separate SwiGLU")

    def folded():
        with monkeypatch.context() as patch:
            patch.setattr(swiglu_module, "swiglu", no_swiglu)
            return full_tp_shared_expert_forward(x, (w13, s13, w2, s2), limit)

    new_input(2)
    assert torch.equal(folded(), separate())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = folded()
    for scale in (1, 4, 8):
        new_input(scale)
        graph.replay()
        assert torch.equal(out, separate())


def _lowm_call(variant: str, m: int) -> Callable[[], Any]:
    """A small-M MMA GEMM call of ``variant`` on ``m`` rows, as a no-argument function. Calls
    with ``lowm_tiles`` tiles swap their MMA operands at <= 8 rows where the GPU does."""
    g = torch.Generator(device="cuda").manual_seed(m)
    n, k = 512, 1024
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(115, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    positions = torch.randint(0, 64, (m,), device="cuda", dtype=torch.int32, generator=g)
    cos_sin = torch.randn(64, 2, 32, device="cuda", generator=g)
    if variant == "rope":
        return lambda: mxfp8_lowm_linear(x, w, s, rope=(positions, cos_sin, 128))
    if variant == "rms_norm":
        gamma = (torch.rand(k, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
        return lambda: mxfp8_lowm_linear(x, w, s, rms_norm=(gamma, 1e-6), n_tiles=lowm_tiles(n, m))
    if variant == "split_n":
        return lambda: mxfp8_lowm_linear(x, w, s, split_n=128, n_tiles=lowm_tiles(n, m))
    if variant == "swiglu_out":
        return lambda: mxfp8_lowm_linear(x, w, s, swiglu_out_limit=10.0)
    if variant == "index_query":
        ws, ss = torch.ops.trtllm.mxfp8_quantize(
            (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16), True
        )
        return lambda: mxfp8_lowm_index_query(x, ws, ss, positions, cos_sin)
    if variant == "grouped_inverse_rope":
        groups = 2
        xg = torch.randn(m, groups * k, device="cuda", generator=g).to(torch.bfloat16)
        block = torch.exp2(torch.randint(-3, 3, (n // 128, k // 128), device="cuda", generator=g))
        tiles = lowm_tiles(n, m, n // groups)
        return lambda: fp8_grouped_lowm(
            xg, w, block.float(), n // groups, (positions, cos_sin, 128), n_tiles=tiles
        )
    wb = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
    return lambda: bf16_lowm_linear(x, wb)


@_skip_no_lowm_fp8
@pytest.mark.parametrize("m", [3, 5, 13, 17, 40, 50])
@pytest.mark.parametrize(
    "variant",
    [
        "rope",
        "rms_norm",
        "split_n",
        "swiglu_out",
        "index_query",
        "grouped_inverse_rope",
        "bf16",
    ],
)
def test_eager_rows_run_the_row_bucket_kernel_bitwise(
    variant: str, m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An eager call runs the kernel of its row bucket on rows padded with an earlier call's, and a
    CUDA graph captured after it runs that kernel without compiling; a decode-graph call runs the
    kernel of its exact rows, which later eager calls reuse. The real rows of all four results carry
    the same bits."""
    kernel_rows = []
    mma_kernel = _mod._mma_kernel
    compiled_mma = _mod._compiled_mma

    def record(m: int, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        fn, rows = mma_kernel(m, *args, **kwargs)
        kernel_rows.append(rows)
        return fn, rows

    monkeypatch.setattr(_mod, "_MMA_SIGNATURES", {})
    monkeypatch.setattr(_mod, "_MMA_BUILT", set())
    monkeypatch.setattr(_mod, "_MMA_ROUTES", {})
    bucket = {3: 3, 5: 5, 13: 16, 17: 32, 40: 48, 50: 64}[m]
    if bucket > m:
        # Leaves other rows in the staging rows this call pads with.
        _lowm_call(variant, bucket - 1)()
    monkeypatch.setattr(_mod, "_mma_kernel", record)
    call = _lowm_call(variant, m)
    eager = call()
    misses = compiled_mma.cache_info().misses
    # A padded call under capture keeps its buffer to the capture, out of the staging cache.
    monkeypatch.setattr(_mod, "_A_STAGING", threading.local())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    assert compiled_mma.cache_info().misses == misses
    assert not _mod._staging_buffers()
    graph.replay()
    torch.cuda.synchronize()
    with flashinfer_mxfp8_decode_graph_capture():
        exact = call()
    reused = call()
    assert kernel_rows == [bucket, bucket, m, m]
    results = [r if isinstance(r, tuple) else (r,) for r in (eager, captured, exact, reused)]
    if variant == "split_n":
        assert [t.shape for t in results[0]] == [(m, 128), (m, 384)]
    if variant == "index_query":
        data, scales = results[0]
        assert data.shape == (m, 4, 64) and data.dtype == torch.int8
        assert scales.shape == (m, 4, 4) and scales.dtype == torch.uint8
    assert len({len(r) for r in results}) == 1
    for outputs in zip(*results):
        for t in outputs:
            assert t.shape == outputs[0].shape and t.shape[0] == m and t.is_contiguous()
            assert torch.equal(t.view(torch.uint8), outputs[0].view(torch.uint8))
    assert len(_mod._MMA_SIGNATURES) == 1


@_skip_no_lowm_fp8
def test_prewarm_compiles_every_row_bucket_of_eager_calls_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prewarm compiles the eleven buckets an eager call did not reach and none of a GEMM that
    only a decode graph ran. Afterwards no eager row count compiles, and eager calls of the decode
    graph's row count run its exact kernel."""
    compiled_mma = functools.lru_cache(maxsize=None)(_mod._compiled_mma.__wrapped__)
    monkeypatch.setattr(_mod, "_compiled_mma", compiled_mma)
    monkeypatch.setattr(_mod, "_MMA_SIGNATURES", {})
    monkeypatch.setattr(_mod, "_MMA_BUILT", set())
    monkeypatch.setattr(_mod, "_MMA_ROUTES", {})

    def rows(m: int) -> torch.Tensor:
        return torch.zeros(m, 512, device="cuda", dtype=torch.bfloat16)

    eager_w = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    graph_w = torch.randn(72, 512, device="cuda").to(torch.bfloat16)
    bf16_lowm_linear(rows(3), eager_w)
    with flashinfer_mxfp8_decode_graph_capture():
        bf16_lowm_linear(rows(3), graph_w)
    assert compiled_mma.cache_info().misses == 2
    assert _mod.prewarm_mma_row_buckets() == (11, 1)
    assert compiled_mma.cache_info().misses == 13
    for m in range(1, 65):
        bf16_lowm_linear(rows(m), eager_w)
    bf16_lowm_linear(rows(3), graph_w)
    assert compiled_mma.cache_info().misses == 13
    bf16_lowm_linear(rows(13), graph_w)
    assert compiled_mma.cache_info().misses == 14


@_skip_no_lowm_fp8
def test_prewarm_stops_at_the_bucket_that_holds_the_most_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a row bound, the prewarm compiles the buckets up to the first one that holds it."""
    compiled_mma = functools.lru_cache(maxsize=None)(_mod._compiled_mma.__wrapped__)
    monkeypatch.setattr(_mod, "_compiled_mma", compiled_mma)
    for name, fresh in (("_MMA_SIGNATURES", {}), ("_MMA_BUILT", set()), ("_MMA_ROUTES", {})):
        monkeypatch.setattr(_mod, name, fresh)
    w = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    bf16_lowm_linear(torch.zeros(3, 512, device="cuda", dtype=torch.bfloat16), w)
    assert _mod.prewarm_mma_row_buckets(8) == (7, 1)
    assert _mod.prewarm_mma_row_buckets(20) == (2, 1)
    assert _mod.prewarm_mma_row_buckets() == (2, 1)
    assert compiled_mma.cache_info().misses == 12


@_skip_no_lowm_fp8
def test_a_bucket_route_yields_to_an_exact_kernel_built_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eager call that compiles its bucket's kernel while another thread builds the exact one
    leaves later eager calls of its row count on the exact kernel."""
    for name, fresh in (("_MMA_SIGNATURES", {}), ("_MMA_BUILT", set()), ("_MMA_ROUTES", {})):
        monkeypatch.setattr(_mod, name, fresh)
    compiled_mma = _mod._compiled_mma
    compiling_bucket, exact_built = threading.Event(), threading.Event()

    def bucket_after_exact(rows: int, *args: Any, **kwargs: Any) -> Any:
        if rows == 16:
            compiling_bucket.set()
            assert exact_built.wait(timeout=60)
        return compiled_mma(rows, *args, **kwargs)

    monkeypatch.setattr(_mod, "_compiled_mma", bucket_after_exact)
    w = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    x = torch.randn(13, 512, device="cuda").to(torch.bfloat16)
    eager = threading.Thread(target=bf16_lowm_linear, args=(x, w))
    eager.start()
    try:
        assert compiling_bucket.wait(timeout=60)
        with flashinfer_mxfp8_decode_graph_capture():
            bf16_lowm_linear(x, w)
    finally:
        exact_built.set()
        eager.join(timeout=60)
    routes = [r for (m, _, graph), (_, r) in _mod._MMA_ROUTES.items() if m == 13 and not graph]
    assert routes == [13]


@_skip_no_lowm_fp8
def test_padded_calls_run_in_and_out_of_inference_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A padded call's staging buffer is a normal tensor: calls in and out of inference mode both
    write it, and both match the exact-row kernel."""
    for name, fresh in (
        ("_A_STAGING", threading.local()),
        ("_MMA_SIGNATURES", {}),
        ("_MMA_BUILT", set()),
    ):
        monkeypatch.setattr(_mod, name, fresh)
    monkeypatch.setattr(_mod, "_MMA_ROUTES", {})
    w = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    x = torch.randn(13, 512, device="cuda").to(torch.bfloat16)
    with torch.inference_mode():
        inside = bf16_lowm_linear(x, w)
    outside = bf16_lowm_linear(x, w)
    with flashinfer_mxfp8_decode_graph_capture():
        exact = bf16_lowm_linear(x, w)
    assert torch.equal(inside, exact) and torch.equal(outside, exact)


@_skip_no_lowm_fp8
def test_padded_calls_keep_the_staging_buffer_out_of_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A padded call copies its rows detached: the staging buffer it reuses never joins the autograd
    graph of an input that requires gradients."""
    for name, fresh in (
        ("_A_STAGING", threading.local()),
        ("_MMA_SIGNATURES", {}),
        ("_MMA_BUILT", set()),
        ("_MMA_ROUTES", {}),
    ):
        monkeypatch.setattr(_mod, name, fresh)
    w = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    with torch.enable_grad():
        for _ in range(2):
            x = torch.randn(13, 512, device="cuda").to(torch.bfloat16).requires_grad_()
            bf16_lowm_linear(x, w)
    (buf,) = _mod._staging_buffers().values()
    assert not buf.requires_grad and buf.grad_fn is None


@_skip_no_lowm_fp8
def test_padded_calls_of_two_threads_on_one_stream_keep_their_own_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each host thread pads into its own staging buffer: a padded call that another thread submits
    to the same stream between this call's copy and its launch cannot overwrite this launch's rows."""
    for name, fresh in (
        ("_A_STAGING", threading.local()),
        ("_MMA_SIGNATURES", {}),
        ("_MMA_BUILT", set()),
        ("_MMA_ROUTES", {}),
    ):
        monkeypatch.setattr(_mod, name, fresh)
    w = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    x = {name: torch.randn(13, 512, device="cuda").to(torch.bfloat16) for name in "AB"}
    expected = {name: bf16_lowm_linear(x[name], w) for name in "AB"}
    mma_kernel = _mod._mma_kernel
    copied, submitted = threading.Event(), threading.Event()

    def launch_after_the_other_submits(m: int, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        fn, rows = mma_kernel(m, *args, **kwargs)
        if threading.current_thread().name != "A":
            return fn, rows

        def launch(*launch_args: Any, **launch_kwargs: Any) -> Any:
            copied.set()
            assert submitted.wait(timeout=60)
            return fn(*launch_args, **launch_kwargs)

        return launch, rows

    monkeypatch.setattr(_mod, "_mma_kernel", launch_after_the_other_submits)
    out = {}
    a = threading.Thread(target=lambda: out.update(A=bf16_lowm_linear(x["A"], w)), name="A")
    a.start()
    try:
        assert copied.wait(timeout=60)
        out["B"] = bf16_lowm_linear(x["B"], w)
    finally:
        submitted.set()
        a.join(timeout=60)
    torch.cuda.synchronize()
    assert torch.equal(out["A"], expected["A"]) and torch.equal(out["B"], expected["B"])


@_skip_no_lowm_fp8
def test_prewarm_keeps_going_past_a_failed_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bucket that fails to compile is reported and left to its first call; the other buckets of
    its GEMM and the other GEMMs still compile."""
    compiled_mma = _mod._compiled_mma

    def flaky(rows: int, *args: Any, **kwargs: Any) -> Any:
        if rows == 16 and args[0] == 64:
            raise RuntimeError("no compiler")
        return compiled_mma(rows, *args, **kwargs)

    monkeypatch.setattr(_mod, "_compiled_mma", flaky)
    for name, fresh in (("_MMA_SIGNATURES", {}), ("_MMA_BUILT", set()), ("_MMA_ROUTES", {})):
        monkeypatch.setattr(_mod, name, fresh)
    x = torch.zeros(3, 512, device="cuda", dtype=torch.bfloat16)
    for n in (64, 72):
        bf16_lowm_linear(x, torch.randn(n, 512, device="cuda").to(torch.bfloat16))
    with patch.object(_mod.logger, "warning") as warning:
        assert _mod.prewarm_mma_row_buckets() == (2 * 11 - 1, 2)
    warning.assert_called_once()
    for signature in _mod._MMA_SIGNATURES:
        for rows in _mod._MMA_ROW_BUCKETS:
            failed = rows == 16 and signature[0][0] == 64
            assert ((rows, signature) in _mod._MMA_BUILT) is not failed


@_skip_no_lowm_fp8
def test_cute_views_are_reused_per_tensor_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A base tensor's CuTe view is exported once per host thread and reused by that thread; another
    thread, new storage, a view, or a dead tensor does not reuse it. A thread's cache goes with the
    thread while the tensor lives on."""
    monkeypatch.setattr(_mod, "_CUTE_VIEWS", threading.local())
    t = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
    first = _mod._cute_view(t)
    assert _mod._cute_view(t) is first
    other = []

    def use_in_worker(u: torch.Tensor) -> None:
        other.extend((_mod._cute_view(u), _mod._cute_view(u), weakref.ref(_mod._views())))

    worker = threading.Thread(target=use_in_worker, args=(t,))
    worker.start()
    worker.join()
    assert other[0] is other[1] and other[0] is not first
    worker_cache = other[2]
    other.clear()
    gc.collect()
    assert worker_cache() is None
    t.data = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
    assert _mod._cute_view(t) is not first
    assert _mod._cute_view(t[:4]) is not _mod._cute_view(t[:4])
    key = id(t)
    del t
    gc.collect()
    assert key not in _mod._views()


# ---------------------------------------------------------------------------
# Race timing
# ---------------------------------------------------------------------------


class _HostDeviceCostRunner(TunableRunner):
    """Returns its input after ``host_s`` of host work and a ``cycles`` device sleep."""

    def __init__(self, name: str, host_s: float, cycles: int) -> None:
        self.name, self.host_s, self.cycles = name, host_s, cycles

    def unique_id(self) -> tuple:
        return (self.name,)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return [0]

    def forward(self, inputs, *, tactic=-1, do_preparation=False, **kwargs):
        time.sleep(self.host_s)
        torch.cuda._sleep(self.cycles)
        return inputs[0]


@_skip_no_cuda
@pytest.mark.parametrize("pdl", ["1", "0"])
@torch.inference_mode()
def test_race_times_device_work_not_host_launch(pdl: str, monkeypatch) -> None:
    """The decode CUDA graphs pay device time only, so the race times inside a CUDA graph, PDL on or off: a
    runner with a slow host launch and a short kernel beats one with a fast launch and a longer kernel. Eager
    timing (positive control) picks the other one."""
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", pdl)
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)
    dispatcher = LowMGemmDispatcher()
    dispatcher._init_runners()
    config = dispatcher._tuning_config
    assert config.use_cuda_graph
    # ~2 us vs ~40 us of device sleep at ~1-2 GHz; 2 ms of host time per launch.
    slow_launch = _HostDeviceCostRunner("slow_launch", host_s=2e-3, cycles=2_000)
    slow_kernel = _HostDeviceCostRunner("slow_kernel", host_s=0.0, cycles=60_000)
    x = torch.zeros((4, 128), dtype=torch.bfloat16, device="cuda")
    w_t = torch.zeros((128, 64), dtype=torch.bfloat16, device="cuda")
    tuner = AutoTuner.get()
    winners = {}
    for graph in (True, False):
        tuner.clear_cache()
        with autotune(skip_dynamic_tuning_buckets=True):
            runner, _ = tuner.choose_one(
                f"test::low_m_race_timing_{graph}",
                [slow_kernel, slow_launch],
                dataclasses.replace(config, use_cuda_graph=graph),
                [x, w_t],
            )
        winners[graph] = runner
    tuner.clear_cache()
    assert winners[True] is slow_launch
    assert winners[False] is slow_kernel
