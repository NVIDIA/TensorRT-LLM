# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests in this module use mocks for GPU operations and run on any platform.
# Apply ``_skip_no_cuda`` / ``_skip_non_sm10x`` to tests that actually
# invoke the kernel hardware rather than the Python dispatch logic.

import dataclasses
import sys
import time
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm._torch.modules.low_m_gemm as _mod
from tensorrt_llm._torch.autotuner import AutoTuner, TunableRunner, autotune
from tensorrt_llm._torch.modules.low_m_gemm import (
    _BACKEND_ENV,
    LOW_M_GEMM_OPT_IN_ATTR,
    LowMGemmDispatcher,
    _parse_enabled,
)
from tensorrt_llm._utils import is_sm_100f

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


# ---------------------------------------------------------------------------
# linear.py fast pre-filter
# ---------------------------------------------------------------------------


def test_linear_fast_rejects_m_above_max_m(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "LOW_M_GEMM_ACTIVE", True)

    assert _mod._should_apply_low_m_gemm(torch.empty((32, 128)))
    assert not _mod._should_apply_low_m_gemm(torch.empty((33, 128)))


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
