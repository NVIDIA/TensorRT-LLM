# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests in this module use mocks for GPU operations and run on any platform.
# Apply ``_skip_no_cuda`` / ``_skip_non_sm10x`` to tests that actually
# invoke the kernel hardware rather than the Python dispatch logic.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm._torch.modules.low_m_gemm as _mod
from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.low_m_gemm import (
    _BACKEND_ENV,
    _FUSED_EPILOGUE_ENV,
    LowMGemmDispatcher,
    _parse_enabled,
    low_m_gemm_fused_epilogue_enabled,
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


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("off", False),
        ("cublaslt", False),
        ("auto", True),
    ],
)
def test_parse_direct_enabled(value: str | None, expected: bool, monkeypatch) -> None:
    if value is None:
        monkeypatch.delenv(_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(_BACKEND_ENV, value)
    assert _mod._parse_direct_enabled() is expected


def test_fused_epilogue_env_is_explicit_and_strict(monkeypatch) -> None:
    monkeypatch.delenv(_FUSED_EPILOGUE_ENV, raising=False)
    assert not low_m_gemm_fused_epilogue_enabled()
    for enabled in ("auto", "on", "true", "1"):
        monkeypatch.setenv(_FUSED_EPILOGUE_ENV, enabled)
        assert low_m_gemm_fused_epilogue_enabled()
    monkeypatch.setenv(_FUSED_EPILOGUE_ENV, "off")
    assert not low_m_gemm_fused_epilogue_enabled()
    monkeypatch.setenv(_FUSED_EPILOGUE_ENV, "qwen-only")
    with pytest.raises(ValueError, match=_FUSED_EPILOGUE_ENV):
        low_m_gemm_fused_epilogue_enabled()


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


def test_linear_fast_rejects_m_above_max_m() -> None:
    assert linear_module._is_low_m_input(torch.empty((32, 128)))
    assert not linear_module._is_low_m_input(torch.empty((33, 128)))


def test_linear_uses_conservative_direct_path_without_autotuner(monkeypatch) -> None:
    monkeypatch.setattr(linear_module, "LOW_M_GEMM_ACTIVE", False)
    expected = torch.empty((1, 8))
    direct = MagicMock(return_value=expected)
    monkeypatch.setattr(linear_module, "apply_direct_low_m_gemm", direct)
    autotuned = MagicMock(side_effect=AssertionError("autotuner path must remain opt-in"))
    monkeypatch.setattr(linear_module, "apply_low_m_gemm", autotuned)
    module = SimpleNamespace(
        weight=torch.empty((8, 128)),
        use_custom_cublas_mm=False,
    )

    output = linear_module.UnquantizedLinearMethod().apply(module, torch.empty((1, 128)), None)

    assert output is expected
    direct.assert_called_once()
    autotuned.assert_not_called()


def test_linear_keeps_explicit_autotuner_priority(monkeypatch) -> None:
    monkeypatch.setattr(linear_module, "LOW_M_GEMM_ACTIVE", True)
    expected = torch.empty((1, 8))
    autotuned = MagicMock(return_value=expected)
    monkeypatch.setattr(linear_module, "apply_low_m_gemm", autotuned)
    direct = MagicMock(side_effect=AssertionError("direct policy bypassed explicit autotuning"))
    monkeypatch.setattr(linear_module, "apply_direct_low_m_gemm", direct)
    module = SimpleNamespace(
        weight=torch.empty((8, 128)),
        use_custom_cublas_mm=False,
    )

    output = linear_module.UnquantizedLinearMethod().apply(module, torch.empty((1, 128)), None)

    assert output is expected
    autotuned.assert_called_once()
    direct.assert_not_called()


# ---------------------------------------------------------------------------
# Conservative direct dispatch without AutoTuner state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        (1, 96, 2560, True),
        (1, 512, 2560, True),
        (1, 640, 2560, True),
        (1, 2048, 2560, True),
        (1, 320, 10240, True),
        (1, 2560, 6144, True),
        (1, 4608, 8192, True),
        (1, 4096, 2560, False),
        (1, 8192, 6144, False),
        (1, 13312, 2560, False),
        (1, 16384, 2560, False),
        (1, 248320, 2560, False),
        (4, 512, 8192, True),
        (4, 512, 2560, False),
        (8, 512, 8192, False),
    ],
)
@_skip_non_sm10x
def test_prefer_direct_bands(m: int, n: int, k: int, expected: bool) -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
        prefer_direct_bf16_gemm_sm100,
    )

    assert prefer_direct_bf16_gemm_sm100(m, n, k) is expected


def test_apply_direct_declines_biased_calls(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(
        LowMGemmDispatcher,
        "_is_candidate_shape",
        staticmethod(lambda *_: True),
    )
    a = torch.empty((1, 2560), dtype=torch.bfloat16)
    weight = torch.empty((512, 2560), dtype=torch.bfloat16)
    bias = torch.empty((512,), dtype=torch.bfloat16)

    assert _mod.apply_direct_low_m_gemm(a, weight, bias) is None


def test_apply_direct_honors_explicit_disabled_policy(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", False)
    candidate = MagicMock(
        side_effect=AssertionError("disabled policy must reject before shape checks")
    )
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", candidate)

    assert _mod.apply_direct_low_m_gemm(torch.empty((1, 128)), torch.empty((8, 128)), None) is None
    candidate.assert_not_called()


def test_apply_direct_declines_shapes_outside_the_bands(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(
        LowMGemmDispatcher,
        "_is_candidate_shape",
        staticmethod(lambda *_: True),
    )
    a = torch.empty((1, 2560), dtype=torch.bfloat16)
    weight = torch.empty((16384, 2560), dtype=torch.bfloat16)
    direct_module = SimpleNamespace(
        default_tactic=MagicMock(
            side_effect=AssertionError("excluded shape must not select a tactic")
        ),
        prefer_direct_bf16_gemm_sm100=lambda m, n, k: False,
        run_direct_dense=MagicMock(
            side_effect=AssertionError("excluded shape must not launch a kernel")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct",
        direct_module,
    )

    assert _mod.apply_direct_low_m_gemm(a, weight, None) is None
    direct_module.default_tactic.assert_not_called()
    direct_module.run_direct_dense.assert_not_called()


def test_apply_direct_declines_unsupported_tactic(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(
        LowMGemmDispatcher,
        "_is_candidate_shape",
        staticmethod(lambda *_: True),
    )
    # This is inside the narrow-N band, but no supported block-size/vector
    # product divides K=640 exactly.
    a = torch.empty((1, 640), dtype=torch.bfloat16)
    weight = torch.empty((512, 640), dtype=torch.bfloat16)
    direct_module = SimpleNamespace(
        default_tactic=MagicMock(side_effect=ValueError("unsupported tactic")),
        prefer_direct_bf16_gemm_sm100=lambda m, n, k: True,
        run_direct_dense=MagicMock(
            side_effect=AssertionError("unsupported tactic must not launch a kernel")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct",
        direct_module,
    )

    assert _mod.apply_direct_low_m_gemm(a, weight, None) is None
    direct_module.default_tactic.assert_called_once_with(1, 512, 640)
    direct_module.run_direct_dense.assert_not_called()


def test_apply_direct_preserves_leading_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(
        LowMGemmDispatcher,
        "_is_candidate_shape",
        staticmethod(lambda *_: True),
    )
    calls = {}

    def run_direct(a, weight_t, output, pdl, tactic):
        calls.update(
            a_shape=a.shape,
            weight_shape=weight_t.shape,
            output_shape=output.shape,
            pdl=pdl,
            tactic=tactic,
        )
        output.fill_(1)

    direct_module = SimpleNamespace(
        default_tactic=lambda m, n, k: (m, n, k),
        prefer_direct_bf16_gemm_sm100=lambda m, n, k: True,
        run_direct_dense=run_direct,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct",
        direct_module,
    )
    monkeypatch.setattr(_mod, "get_env_enable_pdl", lambda: False)
    a = torch.empty((1, 1, 640), dtype=torch.bfloat16)
    weight = torch.empty((512, 640), dtype=torch.bfloat16)

    output = _mod.apply_direct_low_m_gemm(a, weight, None)

    assert output is not None and output.shape == (1, 1, 512)
    assert calls == {
        "a_shape": torch.Size([1, 640]),
        "weight_shape": torch.Size([640, 512]),
        "output_shape": torch.Size([1, 512]),
        "pdl": False,
        "tactic": (1, 512, 640),
    }


@_skip_non_sm10x
@torch.inference_mode()
@pytest.mark.parametrize(
    "n,k",
    [
        (96, 2560),
        (512, 2560),
        (640, 2560),
        (1280, 2560),
        (2560, 6144),
        (320, 10240),
    ],
)
def test_apply_direct_matches_fp32_reference(n: int, k: int, monkeypatch) -> None:
    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    torch.manual_seed(0)
    a = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.02
    reference = (a.float() @ weight.float().t()).to(torch.bfloat16)

    output = _mod.apply_direct_low_m_gemm(a, weight, None)

    assert output is not None
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=5e-3)


@_skip_non_sm10x
@torch.inference_mode()
def test_qwen3_next_gate_direct_path_preserves_topk(monkeypatch) -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_bf16_direct
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate

    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    calls = []
    real_run_direct = low_m_bf16_direct.run_direct_dense
    monkeypatch.setattr(
        low_m_bf16_direct,
        "run_direct_dense",
        lambda *args, **kwargs: (
            calls.append(1),
            real_run_direct(*args, **kwargs),
        )[1],
    )
    torch.manual_seed(0)
    gate = Qwen3NextGate(
        hidden_size=2560,
        num_experts=512,
        top_k=10,
        dtype=torch.bfloat16,
    )
    gate.weight.data = torch.randn((512, 2560), dtype=torch.bfloat16, device="cuda") * 0.02
    hidden = torch.randn((1, 2560), dtype=torch.bfloat16, device="cuda")

    logits = gate(hidden)
    baseline = torch.ops.trtllm.cublas_mm(
        hidden,
        gate.weight.t(),
        bias=None,
        out_dtype=torch.bfloat16,
    )

    assert calls == [1]
    reference = (hidden.float() @ gate.weight.float().t()).to(torch.bfloat16)
    torch.testing.assert_close(logits, reference, rtol=1e-2, atol=5e-3)
    assert torch.equal(
        logits.topk(10, dim=-1).indices.sort(dim=-1).values,
        baseline.topk(10, dim=-1).indices.sort(dim=-1).values,
    )


@_skip_non_sm10x
@torch.inference_mode()
def test_qwen3_next_gate_prefill_falls_back(monkeypatch) -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_bf16_direct
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate

    monkeypatch.setattr(_mod, "_DIRECT_LOW_M_GEMM_ACTIVE", True)
    calls = []
    monkeypatch.setattr(
        low_m_bf16_direct,
        "run_direct_dense",
        lambda *args, **kwargs: calls.append(1),
    )
    gate = Qwen3NextGate(
        hidden_size=2560,
        num_experts=512,
        top_k=10,
        dtype=torch.bfloat16,
    )
    gate.weight.data = torch.randn((512, 2560), dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn((64, 2560), dtype=torch.bfloat16, device="cuda")

    logits = gate(hidden)

    assert not calls
    assert logits.shape == (64, 512)


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


def test_splitk_tactic_accepts_k_tail_only_without_split() -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        SplitKTactic,
        default_tactic,
        validate_tactic,
    )

    m, n, k = 1, 512, 320
    tactic = default_tactic(m, n, k)
    assert tactic.split_k == 1
    validate_tactic(tactic, m, n, k)
    with pytest.raises(ValueError, match="does not divide evenly"):
        validate_tactic(SplitKTactic(64, 8, 2, 2), m, n, k)


@_skip_non_sm10x
@torch.inference_mode()
def test_low_m_epilogues_match_materialized_bf16_results() -> None:
    import torch.nn.functional as F

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
        DirectTactic,
        get_direct_dense_epilogue_runner,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        default_tactic,
        get_splitk_dense_epilogue_runner,
        interleave_grouped_output_rows,
    )

    run_direct_dense = get_direct_dense_epilogue_runner("none")
    run_direct_dense_silu_prefix = get_direct_dense_epilogue_runner("silu_prefix")
    run_splitk_dense = get_splitk_dense_epilogue_runner("none")
    run_splitk_dense_sigmoid_grouped_reduce = get_splitk_dense_epilogue_runner(
        "sigmoid_grouped_reduce"
    )
    for lookup in (get_direct_dense_epilogue_runner, get_splitk_dense_epilogue_runner):
        with pytest.raises(ValueError, match="expected one of"):
            lookup("missing")
        with pytest.raises(ValueError, match="must be a string"):
            lookup(None)

    torch.manual_seed(42)
    # M=2 makes the activated prefix a padded row-major view with stride 336;
    # M=1 would hide that leading-stride contract behind PyTorch contiguity.
    rows, group_count, hidden_size, lowrank = 2, 2, 128, 320
    packed_size, input_size = 336, 1024
    dtype, device = torch.bfloat16, torch.device("cuda")

    a = torch.randn((rows, input_size), dtype=dtype, device=device)
    down_weight = torch.randn((packed_size, input_size), dtype=dtype, device=device)
    plain = torch.empty((rows, packed_size), dtype=dtype, device=device)
    fused = torch.empty_like(plain)
    direct_tactic = DirectTactic(128, 2, rows)
    run_direct_dense(a, down_weight.t(), plain, False, direct_tactic)
    run_direct_dense_silu_prefix(
        a,
        down_weight.t(),
        fused,
        False,
        direct_tactic,
        scale=1.0 / group_count,
        prefix=lowrank,
    )

    expected_prefix = F.silu(plain[:, :lowrank].float() / group_count).to(dtype)
    torch.testing.assert_close(fused[:, :lowrank], expected_prefix, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(fused[:, lowrank:], plain[:, lowrank:], rtol=0.0, atol=0.0)

    grouped_size = group_count * hidden_size
    up_weight = torch.randn((grouped_size, lowrank), dtype=dtype, device=device)
    interleaved_weight = interleave_grouped_output_rows(up_weight, group_count)
    reduction_input = torch.randn((rows, grouped_size), dtype=dtype, device=device)
    projected = torch.empty((rows, grouped_size), dtype=dtype, device=device)
    mixed = torch.empty((rows, hidden_size), dtype=dtype, device=device)
    splitk_tactic = default_tactic(rows, grouped_size, lowrank)
    run_splitk_dense(
        fused[:, :lowrank],
        interleaved_weight.t(),
        None,
        projected,
        False,
        splitk_tactic,
    )
    run_splitk_dense_sigmoid_grouped_reduce(
        fused[:, :lowrank],
        interleaved_weight.t(),
        reduction_input,
        mixed,
        False,
        splitk_tactic,
        reduction_scale=1.0 / group_count,
        group_count=group_count,
    )
    overlapping_prefix = fused.as_strided((rows, lowrank), (1, 1))
    with pytest.raises(ValueError, match="non-overlapping"):
        run_splitk_dense(
            overlapping_prefix,
            interleaved_weight.t(),
            None,
            projected,
            False,
            splitk_tactic,
        )

    gates = torch.sigmoid(projected.float()).unflatten(-1, (hidden_size, group_count))
    grouped_input = reduction_input.float().unflatten(-1, (group_count, hidden_size))
    grouped_input = grouped_input.transpose(-2, -1)
    expected_mixed = (gates * grouped_input).sum(dim=-1).div(group_count).to(dtype)
    torch.testing.assert_close(mixed, expected_mixed, rtol=1e-2, atol=5e-3)


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
