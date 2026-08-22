# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests in this module use mocks for GPU operations and run on any platform.
# Apply ``_skip_no_cuda`` / ``_skip_non_sm10x`` to tests that actually
# invoke the kernel hardware rather than the Python dispatch logic.

import sys
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm._torch.modules.low_m_gemm as _mod
from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.low_m_gemm import _BACKEND_ENV, LowMGemmDispatcher, _parse_enabled
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


# ---------------------------------------------------------------------------
# linear.py fast pre-filter
# ---------------------------------------------------------------------------


def test_linear_fast_rejects_m_above_max_m(monkeypatch) -> None:
    monkeypatch.setattr(linear_module, "LOW_M_GEMM_ACTIVE", True)

    assert linear_module._should_apply_low_m_gemm(torch.empty((32, 128)))
    assert not linear_module._should_apply_low_m_gemm(torch.empty((33, 128)))


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
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk",
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
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk",
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
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct",
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
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct",
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
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk",
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
