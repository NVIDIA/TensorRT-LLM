# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission and control of the TP8 drafter down / o_proj split-K routes (``drafter_down_splitk``) on metadata
doubles: no CUDA storage, no kernel. The kernel's arithmetic is covered by test_m31_down_splitk_gpu.py."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

TARGET = Path(_family.__file__).resolve().parent / "minimax_m3_1_flash_nvfp4__sm_107__tp8"


def load_source(name):
    spec = importlib.util.spec_from_file_location(f"_down_splitk_{name}", TARGET / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorMetadata:
    """No CUDA storage: exercises metadata predicates, never kernel correctness."""

    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)
        self.is_cuda = device.startswith("cuda:")
        self.requires_grad = False
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def t(self):
        return self


class MethodMetadata:
    supports_nccl_symmetric_memory_window_output = True

    def apply(self, *args):
        raise AssertionError("a CPU test must not run a stock GPU operator")


class AllReduceMetadata:
    _disable_mpi = False

    def uses_nccl_symmetric_memory_window(self):
        return True

    def forward(self, *args):
        raise AssertionError("the scoped producer must not run a collective")


class LinearMetadata:
    def __init__(self):
        self.weight = TensorMetadata((6144, 1536))
        self.bias = None
        self.tp_mode = "row"
        self.reduce_output = True
        self.partition_plan = SimpleNamespace(enabled=False)
        self.use_custom_cublas_mm = self.use_cute_dsl_bf16_gemm = False
        self.mapping = SimpleNamespace(
            tp_size=8, pp_size=1, cp_size=1, enable_attention_dp=False, tp_group=list(range(8))
        )
        self.quant_method, self.all_reduce = MethodMetadata(), AllReduceMetadata()
        self.stock_calls, self.stock_result = [], object()

    def forward(self, input, **kwargs):
        self.stock_calls.append((input, kwargs))
        return self.stock_result


@pytest.fixture
def dispatch():
    return load_source("drafter_down_splitk")


@pytest.fixture
def armed(dispatch):
    linear = LinearMetadata()
    launched, taken = [], []
    torch = SimpleNamespace(
        Tensor=TensorMetadata,
        is_grad_enabled=lambda: False,
        empty=lambda shape, **kwargs: TensorMetadata(shape, **kwargs),
    )
    provider = SimpleNamespace(run_splitk_dense=lambda *args: launched.append(args))
    types = ("bf16", "row", MethodMetadata, AllReduceMetadata, "none")
    forward = dispatch._wrap_forward(
        linear,
        provider,
        (64, 8, 2, 12),
        torch,
        SimpleNamespace(LOW_M_GEMM_ACTIVE=False),
        types,
        3,
        lambda *args: taken.append(args),
    )
    return SimpleNamespace(
        linear=linear,
        forward=forward,
        launched=launched,
        taken=taken,
        torch=torch,
        provider=provider,
    )


@pytest.mark.parametrize("rows", [1, 7, 8])
def test_supported_rows_write_direct_output(armed, rows):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    x = TensorMetadata((rows, 1536))
    out = armed.forward(x, all_reduce_params=params)
    assert out.shape == (rows, 6144)
    assert armed.launched == [(x, armed.linear.weight, None, out, True, (64, 8, 2, 12))]
    assert not armed.linear.stock_calls
    armed.forward(x, all_reduce_params=params)
    assert armed.taken == [(3, rows)]


@pytest.mark.parametrize(
    "input",
    [
        TensorMetadata((0, 1536)),
        TensorMetadata((9, 1536)),
        TensorMetadata((7, 1024)),
        TensorMetadata((1, 7, 1536)),
        TensorMetadata((7, 1536), dtype="fp16"),
        TensorMetadata((7, 1536), device="cpu"),
        TensorMetadata((7, 1536), contiguous=False),
    ],
)
def test_unsupported_inputs_preserve_complete_stock_call(armed, input):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    out = armed.forward(input, all_reduce_params=params, layer_idx=17)
    assert out is armed.linear.stock_result and not armed.launched
    assert armed.linear.stock_calls == [
        (input, dict(all_reduce_params=params, lora_params=None, layer_idx=17))
    ]


@pytest.mark.parametrize("adapter", [{}, {"active": True}])
def test_adapter_arguments_always_use_stock(armed, adapter):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    out = armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params, lora_params=adapter)
    assert out is armed.linear.stock_result and not armed.launched
    assert armed.linear.stock_calls[0][1]["lora_params"] is adapter


@pytest.mark.parametrize(
    "field,value",
    [("tp_size", 4), ("pp_size", 2), ("cp_size", 2), ("enable_attention_dp", True)],
)
def test_parallelism_outside_scope_falls_back(armed, field, value):
    setattr(armed.linear.mapping, field, value)
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    assert not armed.launched


def test_weight_replacement_and_grad_mode_fall_back(armed):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    original = armed.linear.weight
    armed.linear.weight = TensorMetadata((6144, 1536))
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    armed.linear.weight = original
    armed.torch.is_grad_enabled = lambda: True
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    assert not armed.launched


@pytest.mark.parametrize(
    "params",
    [
        None,
        SimpleNamespace(enable_allreduce=True, fusion_op="none"),
        SimpleNamespace(enable_allreduce=False, fusion_op="other"),
    ],
)
def test_non_deferred_call_uses_stock(armed, params):
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    assert not armed.launched


def test_launch_error_never_retries_stock(armed):
    def fail(*args):
        raise RuntimeError("kernel launch failed")

    armed.provider.run_splitk_dense = fail
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
    assert not armed.linear.stock_calls


def _clear_switches(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TRTLLM_M31_DISABLE_"):
            monkeypatch.delenv(name)


def test_kill_latched_by_existing_switch_skips_all_imports(dispatch, monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv("TRTLLM_M31_DISABLE_DRAFTER_DOWN_SPLITK", "1")
    switches = load_source("m31_switches")
    assert switches.m31_disabled("DRAFTER_DOWN_SPLITK")
    # A disabled route returns before touching the model (None here) or importing anything.
    assert (
        dispatch.route_down_splitk(None, enabled=not switches.m31_disabled("DRAFTER_DOWN_SPLITK"))
        == 0
    )


def test_provider_without_the_entry_points_is_declined(dispatch, monkeypatch):
    partial = SimpleNamespace(SplitKTactic=lambda *a: a, run_splitk_dense=lambda *a: None)
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda name: partial)
    assert dispatch._load_provider() is None


def test_missing_provider_is_declined(dispatch, monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(dispatch.importlib, "import_module", missing)
    assert dispatch._load_provider() is None


def _provider(validate):
    return SimpleNamespace(
        SplitKTactic=lambda *a: a,
        run_splitk_dense=lambda *a: None,
        validate_tactic=validate,
        _get_compiled_splitk_kernel=lambda *a: None,
    )


def test_provider_with_the_entry_points_is_loaded(dispatch, monkeypatch):
    provider = _provider(lambda *a: None)
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda name: provider)
    assert dispatch._load_provider() is provider


def test_tactic_is_admitted_only_when_every_row_validates(dispatch):
    seen = []
    assert dispatch._tactic_admitted(
        _provider(lambda t, m, n, k: seen.append((t, m, n, k))),
        dispatch.TACTIC,
        dispatch.DOWN_SHAPE,
    )
    assert seen == [(dispatch.TACTIC, m, 6144, 1536) for m in dispatch.ROWS]

    def reject_above_4(t, m, n, k):
        if m > 4:
            raise ValueError("tactic does not serve this M")

    assert not dispatch._tactic_admitted(
        _provider(reject_above_4), dispatch.OPROJ_TACTIC, dispatch.OPROJ_SHAPE
    )

    def old_signature(t, m, n):
        return None

    assert not dispatch._tactic_admitted(
        _provider(old_signature), dispatch.TACTIC, dispatch.DOWN_SHAPE
    )


@pytest.fixture
def armed_oproj(dispatch):
    linear = LinearMetadata()
    linear.weight = TensorMetadata(dispatch.OPROJ_SHAPE)
    launched = []
    torch = SimpleNamespace(
        Tensor=TensorMetadata,
        is_grad_enabled=lambda: False,
        empty=lambda shape, **kwargs: TensorMetadata(shape, **kwargs),
    )
    forward = dispatch._wrap_forward(
        linear,
        SimpleNamespace(run_splitk_dense=lambda *args: launched.append(args)),
        dispatch.OPROJ_TACTIC,
        torch,
        SimpleNamespace(LOW_M_GEMM_ACTIVE=False),
        ("bf16", "row", MethodMetadata, AllReduceMetadata, "none"),
        2,
        lambda *args: None,
        dispatch.OPROJ_SHAPE,
    )
    return SimpleNamespace(linear=linear, forward=forward, launched=launched)


@pytest.mark.parametrize("rows", [1, 7, 8])
def test_oproj_rows_write_direct_output(armed_oproj, rows):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    x = TensorMetadata((rows, 1024))
    out = armed_oproj.forward(x, all_reduce_params=params)
    assert out.shape == (rows, 6144)
    assert armed_oproj.launched == [(x, armed_oproj.linear.weight, None, out, True, (64, 8, 1, 12))]
    assert not armed_oproj.linear.stock_calls


@pytest.mark.parametrize("input", [TensorMetadata((7, 1536)), TensorMetadata((9, 1024))])
def test_oproj_out_of_scope_inputs_use_stock(armed_oproj, input):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    assert armed_oproj.forward(input, all_reduce_params=params) is armed_oproj.linear.stock_result
    assert not armed_oproj.launched


def test_oproj_scope_declines_down_weight(armed_oproj):
    armed_oproj.linear.weight = TensorMetadata((6144, 1536))
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    out = armed_oproj.forward(TensorMetadata((7, 1024)), all_reduce_params=params)
    assert out is armed_oproj.linear.stock_result and not armed_oproj.launched


def test_oproj_kill_is_independent_of_down(dispatch, monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv("TRTLLM_M31_DISABLE_DRAFTER_OPROJ_SPLITK", "1")
    switches = load_source("m31_switches")
    assert switches.m31_disabled("DRAFTER_OPROJ_SPLITK")
    assert not switches.m31_disabled("DRAFTER_DOWN_SPLITK")
    assert (
        dispatch.route_oproj_splitk(None, enabled=not switches.m31_disabled("DRAFTER_OPROJ_SPLITK"))
        == 0
    )
