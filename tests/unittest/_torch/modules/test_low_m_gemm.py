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


def test_linear_fast_rejects_m_above_max_m() -> None:
    assert linear_module._is_low_m_input(torch.empty((32, 128)))
    assert not linear_module._is_low_m_input(torch.empty((33, 128)))


# ---------------------------------------------------------------------------
# Direct-kernel crossover predicate and its autotuner-free entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        # Decode projections measured to favour the direct kernel on GB300
        # with a cold L2 (see prefer_direct_bf16_gemm_sm100's docstring).
        (1, 96, 2560, True),  # GDN in_proj_ba
        (1, 512, 2560, True),  # MoE router
        (1, 640, 2560, True),  # QSA indexer index_qk_proj
        (1, 2048, 2560, True),  # band edge, narrow-N clause
        (1, 320, 10240, True),  # hyper-connection mixer down-projection
        (1, 2560, 6144, True),  # GDN out_proj / QSA o_proj, deep-K clause
        (1, 4608, 8192, True),  # pre-existing K=8192 band, preserved
        # Shapes where the two kernels land within ~25%: must fall through.
        (1, 4096, 2560, False),  # past the narrow-N band at shallow K
        (1, 8192, 6144, False),  # past the deep-K band
        (1, 13312, 2560, False),  # QSA qkv+gate
        (1, 16384, 2560, False),  # GDN in_proj_qkvz
        (1, 248320, 2560, False),  # lm_head
        # The small-batch bands stay restricted to K=8192.
        (4, 512, 8192, True),
        (4, 512, 2560, False),
        (8, 512, 8192, False),
    ],
)
def test_prefer_direct_bands(m: int, n: int, k: int, expected: bool) -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
        prefer_direct_bf16_gemm_sm100,
    )

    assert prefer_direct_bf16_gemm_sm100(m, n, k) is expected


def test_apply_direct_declines_biased_calls(monkeypatch) -> None:
    """The direct kernel has no bias epilogue, so biased calls must fall through."""
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", staticmethod(lambda *_: True))
    a = torch.empty((1, 2560), dtype=torch.bfloat16)
    w = torch.empty((512, 2560), dtype=torch.bfloat16)
    bias = torch.empty((512,), dtype=torch.bfloat16)

    assert _mod.apply_direct_low_m_gemm(a, w, bias) is None


def test_apply_direct_declines_shapes_outside_the_bands(monkeypatch) -> None:
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", staticmethod(lambda *_: True))
    a = torch.empty((1, 2560), dtype=torch.bfloat16)
    # N=16384 at K=2560 is a near-tie with cuBLAS; the predicate excludes it.
    w = torch.empty((16384, 2560), dtype=torch.bfloat16)

    assert _mod.apply_direct_low_m_gemm(a, w, None) is None


def test_apply_direct_declines_unsupported_k(monkeypatch) -> None:
    """K must be divisible by a supported block size x the vector width."""
    monkeypatch.setattr(LowMGemmDispatcher, "_is_candidate_shape", staticmethod(lambda *_: True))
    # K=640 is inside the predicate bands but no block size divides it evenly,
    # so default_tactic raises and the call must fall through rather than fail.
    a = torch.empty((1, 640), dtype=torch.bfloat16)
    w = torch.empty((2560, 640), dtype=torch.bfloat16)

    assert _mod.apply_direct_low_m_gemm(a, w, None) is None


@_skip_non_sm10x
@torch.inference_mode()
@pytest.mark.parametrize(
    "n,k",
    [
        (96, 2560),  # GDN in_proj_ba
        (512, 2560),  # MoE router
        (640, 2560),  # QSA indexer index_qk_proj
        (1280, 2560),  # shared-expert gate_up
        (2560, 6144),  # GDN out_proj / QSA o_proj
        (320, 10240),  # hyper-connection mixer down-projection
    ],
)
def test_apply_direct_matches_torch_on_decode_shapes(n: int, k: int) -> None:
    """Every routed decode shape must be no less accurate than cuBLAS.

    Both kernels accumulate in FP32 and round once to BF16, so they agree to
    within a rounding step but not bit-exactly; comparing each against an FP32
    reference is the assertion that does not depend on accumulation order.
    """
    torch.manual_seed(0)
    a = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.02
    reference = (a.float() @ weight.float().t()).to(torch.bfloat16)

    output = _mod.apply_direct_low_m_gemm(a, weight, None)

    assert output is not None, f"N={n} K={k} should be routed to the direct kernel"
    assert output.shape == (1, n)
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=5e-3)
    direct_err = (output.float() - reference.float()).abs().max()
    cublas_err = (torch.nn.functional.linear(a, weight).float() - reference.float()).abs().max()
    # One BF16 rounding step of slack, so the comparison does not depend on
    # which side of a rounding boundary a given accumulation order lands on.
    slack = reference.float().abs().max() * 2**-8
    assert direct_err <= cublas_err + slack


@_skip_non_sm10x
@torch.inference_mode()
def test_apply_direct_preserves_leading_batch_dims() -> None:
    torch.manual_seed(0)
    weight = torch.randn((512, 2560), dtype=torch.bfloat16, device="cuda") * 0.02

    a = torch.randn((1, 1, 2560), dtype=torch.bfloat16, device="cuda")
    output = _mod.apply_direct_low_m_gemm(a, weight, None)
    assert output is not None and output.shape == (1, 1, 512)
    torch.testing.assert_close(output, torch.nn.functional.linear(a, weight), rtol=1e-2, atol=5e-3)

    # M=8 is outside the predicate's single-row band and must fall through.
    batched = torch.randn((2, 4, 2560), dtype=torch.bfloat16, device="cuda")
    assert _mod.apply_direct_low_m_gemm(batched, weight, None) is None


# ---------------------------------------------------------------------------
# MoE router gate — builds its GEMM by hand, so it needs its own routing check
# ---------------------------------------------------------------------------


@_skip_non_sm10x
@torch.inference_mode()
def test_qwen3_next_gate_takes_the_direct_kernel_at_decode(monkeypatch) -> None:
    """Qwen3NextGate calls cublas_mm directly; check it reaches the low-m path.

    The gate does not go through ``Linear``, so predicate coverage alone does
    not tell us anything about this call site.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_bf16_direct
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate

    calls = []
    real_run_direct = low_m_bf16_direct.run_direct_dense
    monkeypatch.setattr(
        low_m_bf16_direct,
        "run_direct_dense",
        lambda *a, **kw: (calls.append(1), real_run_direct(*a, **kw))[1],
    )

    torch.manual_seed(0)
    gate = Qwen3NextGate(hidden_size=2560, num_experts=512, top_k=10, dtype=torch.bfloat16)
    gate.weight.data = torch.randn((512, 2560), dtype=torch.bfloat16, device="cuda") * 0.02
    hidden = torch.randn((1, 2560), dtype=torch.bfloat16, device="cuda")

    logits = gate(hidden)

    assert len(calls) == 1, "decode-shaped router GEMM did not reach the direct kernel"
    assert logits.shape == (1, 512) and logits.dtype == torch.bfloat16
    reference = (hidden.float() @ gate.weight.float().t()).to(torch.bfloat16)
    torch.testing.assert_close(logits, reference, rtol=1e-2, atol=5e-3)
    # Routing is a top-k over these logits, so the selected experts must match.
    baseline = torch.ops.trtllm.cublas_mm(
        hidden, gate.weight.t(), bias=None, out_dtype=torch.bfloat16
    )
    assert torch.equal(
        logits.topk(10, dim=-1).indices.sort(dim=-1).values,
        baseline.topk(10, dim=-1).indices.sort(dim=-1).values,
    )


@_skip_non_sm10x
@torch.inference_mode()
def test_qwen3_next_gate_falls_back_outside_the_bands(monkeypatch) -> None:
    """A prefill-shaped batch must keep the cuBLAS path."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_bf16_direct
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate

    calls = []
    monkeypatch.setattr(
        low_m_bf16_direct,
        "run_direct_dense",
        lambda *a, **kw: calls.append(1),
    )

    torch.manual_seed(0)
    gate = Qwen3NextGate(hidden_size=2560, num_experts=512, top_k=10, dtype=torch.bfloat16)
    gate.weight.data = torch.randn((512, 2560), dtype=torch.bfloat16, device="cuda") * 0.02
    hidden = torch.randn((64, 2560), dtype=torch.bfloat16, device="cuda")

    logits = gate(hidden)

    assert not calls, "M=64 is outside the single-row band and must not be routed"
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


@_skip_non_sm10x
@torch.inference_mode()
def test_splitk_kernel_generic_epilogue_matches_torch() -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        SplitKTactic,
        run_splitk_dense,
    )

    torch.manual_seed(42)
    a = torch.randn((4, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((4, 256), dtype=torch.bfloat16, device="cuda")
    run_splitk_dense(
        a,
        weight.t(),
        None,
        output,
        False,
        SplitKTactic(mma_m=64, mma_n=8, split_k=4, ab_stages=2),
    )
    torch.testing.assert_close(output, torch.nn.functional.linear(a, weight), rtol=1e-2, atol=5e-3)


@_skip_non_sm10x
@torch.inference_mode()
@pytest.mark.parametrize("prefix", [320, 336])
def test_direct_kernel_silu_prefix_epilogue_matches_unfused(prefix: int) -> None:
    """The fused activation must reproduce a separate kernel over the GEMM.

    The epilogue rounds to BF16 before activating, so it is bit-exact against
    the unfused sequence up to the transcendental's own last-ulp behaviour.
    The suffix past ``prefix`` must come out as a plain GEMM result.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
        DirectTactic,
        run_direct_dense,
        run_direct_dense_silu_prefix,
    )

    # The hyper-connection mix packs a 320-wide low-rank block and 4 injection
    # logits into one 336-row weight; only the low-rank block is activated.
    n, k, scale = 336, 10240, 0.25
    tactic = DirectTactic(block_size=128, outputs_per_block=2, rows_per_block=1)
    torch.manual_seed(0)
    a = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.02

    plain = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    run_direct_dense(a, weight.t(), plain, False, tactic)
    expected = plain.clone()
    expected[:, :prefix] = torch.nn.functional.silu(plain[:, :prefix] * scale)

    fused = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    run_direct_dense_silu_prefix(a, weight.t(), fused, False, tactic, scale, prefix)

    torch.testing.assert_close(fused, expected, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(fused[:, prefix:], plain[:, prefix:], rtol=0, atol=0)


@_skip_non_sm10x
@torch.inference_mode()
def test_direct_kernel_silu_prefix_rejects_out_of_range_prefix() -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
        DirectTactic,
        run_direct_dense_silu_prefix,
    )

    tactic = DirectTactic(block_size=128, outputs_per_block=2, rows_per_block=1)
    a = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((64, 1024), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((1, 64), dtype=torch.bfloat16, device="cuda")
    for prefix in (0, 65):
        with pytest.raises(ValueError, match="prefix must be in"):
            run_direct_dense_silu_prefix(a, weight.t(), out, False, tactic, 0.25, prefix)


def test_splitk_tactic_accepts_a_k_tail_only_without_a_split() -> None:
    """K need not fill whole CTA K tiles unless the tiles are split across ranks."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        SplitKTactic,
        default_tactic,
        validate_tactic,
    )

    # The hyper-connection mix-up projection: K is the 320-wide low-rank block.
    validate_tactic(SplitKTactic(128, 8, 1, 6), 1, 10240, 320)
    assert default_tactic(1, 10240, 320).split_k == 1
    with pytest.raises(ValueError, match="does not divide evenly"):
        validate_tactic(SplitKTactic(64, 8, 2, 6), 1, 10240, 320)


@_skip_non_sm10x
@torch.inference_mode()
def test_splitk_kernel_matches_torch_on_a_k_tail() -> None:
    """The residual K tile is zero-filled by the TMA load, so it adds nothing."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        default_tactic,
        run_splitk_dense,
    )

    torch.manual_seed(42)
    m, n, k = 1, 512, 320
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.02
    output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    run_splitk_dense(a, weight.t(), None, output, False, default_tactic(m, n, k))
    torch.testing.assert_close(
        output.float(),
        (a.float() @ weight.float().t()),
        rtol=1e-2,
        atol=5e-3,
    )


@_skip_non_sm10x
@torch.inference_mode()
@pytest.mark.parametrize("leading_stride", [320, 336])
def test_splitk_gate_epilogue_matches_the_unfused_sequence(leading_stride: int) -> None:
    """The gate epilogue must reproduce GEMM + sigmoid + grouped weighted mean.

    Shapes are the hyper-connection decode mix: a 320-wide low-rank gate
    projected to ``hc_count * hidden_size``, reduced over the 4 streams. The
    ``336`` case covers reading the gate as a slice of the packed down/injection
    projection's row, which carries a padded leading stride.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
        default_tactic,
        run_splitk_dense_gate,
    )

    torch.manual_seed(42)
    group, hidden, k = 4, 2560, 320
    n = group * hidden
    scale = 1.0 / group
    packed = torch.randn((1, leading_stride), dtype=torch.bfloat16, device="cuda") * 0.1
    a = packed[:, :k]
    # Row (j * group + s) holds stream s of hidden column j: the interleaved
    # layout the epilogue reads its per-stream gates from.
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.02
    x = torch.randn((1, n), dtype=torch.bfloat16, device="cuda")

    out = torch.empty((1, hidden), dtype=torch.bfloat16, device="cuda")
    run_splitk_dense_gate(a, weight.t(), x, out, False, default_tactic(1, n, k), scale, group)

    gates = torch.sigmoid((a.float() @ weight.float().t()).unflatten(-1, (hidden, group)))
    streams = x.float().unflatten(-1, (group, hidden)).transpose(1, 2)
    torch.testing.assert_close(
        out.float(),
        scale * (gates * streams).sum(dim=-1),
        rtol=1e-2,
        atol=5e-3,
    )
