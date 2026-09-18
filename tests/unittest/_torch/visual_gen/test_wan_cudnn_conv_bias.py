# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convolution dispatch, metadata lifetime, and error-boundary controls."""

import contextlib
import gc
import importlib
import threading
import time
import weakref
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch
import torch.nn.functional as F

helper = importlib.import_module("tensorrt_llm._torch.visual_gen.models.wan.cudnn_conv_bias")


class Descriptor:
    def __init__(self, uid, dim, stride, dtype, virtual=False):
        self.uid, self.dim, self.stride = uid, dim, stride
        self.dtype, self.virtual = dtype, virtual

    def get_uid(self):
        return self.uid

    def get_dim(self):
        return self.dim

    def get_stride(self):
        return self.stride

    def get_data_type(self):
        return self.dtype

    def get_is_virtual(self):
        return self.virtual

    def set_data_type(self, value):
        self.dtype = value
        return self

    def set_dim(self, value):
        self.dim = value
        return self

    def set_stride(self, value):
        self.stride = value
        return self

    def set_output(self, value):
        self.virtual = not value
        return self


class Unsupported(Exception):
    pass


class Graph:
    def __init__(self, runtime, **kwargs):
        self.runtime = runtime
        self.context = SimpleNamespace(compute_data_type="float", intermediate_data_type="bf16")
        self.nodes, self._cpp_tensors, self._data_bindings = {}, {}, {}
        self.selected_engine = None
        self.in_execute = False
        self.calls = []

    def tensor(self, dim, stride, data_type, name):
        t = Descriptor(len(self._cpp_tensors) + 1, dim, stride, data_type)
        self._cpp_tensors[t.uid] = t
        return t

    def conv_fprop(self, **kwargs):
        self.conv = kwargs
        self.nodes[kwargs["name"]] = SimpleNamespace(compute_data_type=kwargs["compute_data_type"])
        t = self.tensor([], [], "bf16", "conv")
        t.virtual = True
        return t

    def add(self, **kwargs):
        self.nodes[kwargs["name"]] = SimpleNamespace(compute_data_type=kwargs["compute_data_type"])
        return self.tensor([], [], "bf16", "y")

    def get_node(self, name):
        return self.nodes[name]

    def build(self, modes):
        if self.runtime.build_error:
            raise self.runtime.build_error
        if self.runtime.corrupt_boundary:
            self._cpp_tensors[4].dtype = "float"

    def get_workspace_size(self):
        return self.runtime.workspace

    def execute(self, values, workspace, handle):
        assert not self.in_execute, "Concurrent graph mutation"
        self.in_execute = True
        try:
            if self.runtime.execute_error:
                raise self.runtime.execute_error
            self.calls.append(
                (self.runtime.streams[handle], tuple(t.data_ptr() for t in values.values()))
            )
            x, w, b, y = values.values()
            conv = F.conv2d if x.ndim == 4 else F.conv3d
            result = conv(
                x.float(),
                w.float(),
                None,
                self.conv["stride"],
                self.conv["pre_padding"],
                self.conv["dilation"],
            )
            y.copy_((result.bfloat16().float() + b.float()).bfloat16())
            time.sleep(0.002)
        finally:
            self.in_execute = False


class Runtime:
    data_type = SimpleNamespace(BFLOAT16="bf16", FLOAT="float")
    heur_mode = SimpleNamespace(A="a", FALLBACK="fallback")
    cudnnGraphNotSupportedError = Unsupported

    def __init__(self):
        self.created, self.destroyed, self.graphs = [], [], []
        self.streams = {}
        self.build_error = self.execute_error = None
        self.workspace, self.corrupt_boundary = 0, False

    def create_handle(self):
        handle = len(self.created) + 1
        self.created.append(handle)
        return handle

    def destroy_handle(self, handle):
        self.destroyed.append(handle)

    def set_stream(self, handle, stream):
        self.streams[handle] = stream

    def pygraph(self, **kwargs):
        graph = Graph(self, **kwargs)
        self.graphs.append(graph)
        return graph


@pytest.fixture
def fake_runtime():
    runtime = Runtime()
    helper._CACHE.clear()
    with (
        patch.object(helper, "_eligible", new=lambda *args: True),
        patch.object(helper, "_runtime", new=lambda: runtime),
        patch.object(torch.cuda, "device", side_effect=lambda *args: contextlib.nullcontext()),
        patch.object(
            torch.cuda,
            "current_stream",
            side_effect=lambda *args: SimpleNamespace(cuda_stream=threading.get_ident()),
        ),
    ):
        yield runtime
        helper._CACHE.clear()
        gc.collect()


def tiny_call(x=None, weight=None, bias=None):
    x = (
        torch.ones(1, 2, 4, 4, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        if x is None
        else x
    )
    weight = (
        torch.ones(2, 2, 3, 3, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        if weight is None
        else weight
    )
    bias = torch.ones(2, dtype=torch.bfloat16) if bias is None else bias
    return helper.try_conv_bias(x, weight, bias, (1, 1), (1, 1), (1, 1), 1, training=False)


def test_rebind_parameters_and_do_not_retain_tensors(fake_runtime):
    weight = torch.nn.Parameter(
        torch.ones(2, 2, 3, 3, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    )
    bias = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    with torch.no_grad():
        first = tiny_call(weight=weight, bias=bias)
        weight.mul_(2)
        bias.add_(3)
        second = tiny_call(weight=weight, bias=bias)
        replacement = torch.nn.Parameter(torch.zeros_like(weight))
        third = tiny_call(weight=replacement, bias=bias)
    assert not torch.equal(first, second)
    assert torch.equal(third, torch.full_like(third, 4))
    assert len(fake_runtime.created) == 1
    refs = [weakref.ref(t) for t in (weight, bias, replacement)]
    del weight, bias, replacement
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert helper._cache_info()["ready"] == 1


@pytest.mark.parametrize("failure", ["unsupported", "unexpected", "boundary", "workspace"])
def test_build_fallback_boundary_and_handle_cleanup(fake_runtime, failure):
    if failure == "unsupported":
        fake_runtime.build_error = Unsupported("unsupported")
    elif failure == "unexpected":
        fake_runtime.build_error = RuntimeError("backend fault")
    elif failure == "boundary":
        fake_runtime.corrupt_boundary = True
    else:
        fake_runtime.workspace = helper._WORKSPACE_LIMIT + 1
    if failure in ("unexpected", "boundary"):
        with pytest.raises(RuntimeError):
            tiny_call()
        assert not helper._CACHE
    else:
        assert tiny_call() is None and tiny_call() is None
        assert len(fake_runtime.created) == 1
    assert fake_runtime.destroyed == [1]


def test_execution_failure_propagates(fake_runtime):
    fake_runtime.execute_error = RuntimeError("device execution error")
    with pytest.raises(RuntimeError, match="device execution error"):
        tiny_call()


def test_bounded_cache_preserves_in_use_plan(fake_runtime):
    with patch.object(helper, "_CACHE_LIMIT", 1):
        tiny_call()
        held = next(iter(helper._CACHE.values()))
        tiny_call(
            x=torch.ones(2, 2, 4, 4, dtype=torch.bfloat16).contiguous(
                memory_format=torch.channels_last
            )
        )
        assert helper._cache_info()["size"] == 1
        assert fake_runtime.destroyed == []
        del held
        gc.collect()
        assert fake_runtime.destroyed == [1]


def test_each_thread_rebinds_stream_under_plan_lock(fake_runtime):
    tiny_call()
    errors = []

    def run():
        try:
            tiny_call()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert sorted(call[0] for call in fake_runtime.graphs[0].calls[1:]) == sorted(
        t.ident for t in threads
    )
    assert len(fake_runtime.created) == 1


@contextlib.contextmanager
def metadata_cuda():
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        patch.object(
            torch.Tensor, "device", new_callable=PropertyMock, return_value=torch.device("cuda:0")
        ),
        patch.object(torch.Tensor, "data_ptr", return_value=16),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA B200"),
        torch.backends.cudnn.flags(
            enabled=True, deterministic=False, benchmark=False, allow_tf32=True
        ),
        patch.dict("os.environ", {"CUDNN_FRONTEND_ENABLE_FROST_ENGINES": "0"}),
        torch.no_grad(),
    ):
        yield


def metadata_tensors():
    shape, wshape, *_ = sorted(helper._QUALIFIED)[0]
    x = torch.empty_strided(
        shape, helper._channels_last_stride(shape), dtype=torch.bfloat16, device="meta"
    )
    w = torch.nn.Parameter(
        torch.empty_strided(
            wshape, helper._channels_last_stride(wshape), dtype=torch.bfloat16, device="meta"
        )
    )
    b = torch.nn.Parameter(torch.empty(wshape[0], dtype=torch.bfloat16, device="meta"))
    return x, w, b


def test_no_grad_accepts_real_parameter_defaults():
    x, w, b = metadata_tensors()
    signature = sorted(helper._QUALIFIED)[0]
    with metadata_cuda():
        assert w.requires_grad and b.requires_grad
        assert helper._eligible(x, w, b, *signature[2:], 1, False)
        assert not helper._eligible(x, w, b, *signature[2:], 1, True)
        with torch.enable_grad():
            assert not helper._eligible(x, w, b, *signature[2:], 1, False)


@pytest.mark.parametrize(
    "shape,wshape",
    [
        ((1, 512, 4, 352, 640), (256, 512, 1, 1, 1)),
        ((1, 1024, 4, 176, 320), (512, 1024, 1, 1, 1)),
    ],
)
def test_singleton_weight_strides_preserve_addresses_and_eligibility(shape, wshape):
    x = torch.empty_strided(
        shape, helper._channels_last_stride(shape), dtype=torch.bfloat16, device="meta"
    )
    # contiguous() preserves this already-contiguous 1x1x1 weight layout.
    w = torch.nn.Parameter(
        torch.randn(wshape, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
    )
    canonical = w.as_strided(wshape, helper._channels_last_stride(wshape))
    assert tuple(w.stride()) == (wshape[1], 1, 1, 1, 1)
    assert w.stride() != canonical.stride()
    assert w.data_ptr() == canonical.data_ptr()
    assert torch.equal(w, canonical)
    b = torch.nn.Parameter(torch.empty(wshape[0], dtype=torch.bfloat16, device="meta"))
    args = ((1, 1, 1), (0, 0, 0), (1, 1, 1), 1, False)
    with metadata_cuda():
        assert helper._eligible(x, w, b, *args)
        # Keep the input guard strict even for an unused singleton batch stride.
        input_strides = list(x.stride())
        input_strides[0] += 1
        assert not helper._eligible(x.as_strided(shape, input_strides), w, b, *args)


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4])
def test_nonsingleton_weight_stride_changes_fall_back(axis):
    x, w, b = metadata_tensors()
    signature = sorted(helper._QUALIFIED)[0]
    assert w.shape[axis] > 1
    strides = list(w.stride())
    strides[axis] += 1
    changed = torch.nn.Parameter(
        torch.empty_strided(w.shape, strides, dtype=torch.bfloat16, device="meta")
    )
    with metadata_cuda():
        assert helper._eligible(x, w, b, *signature[2:], 1, False)
        assert not helper._eligible(x, changed, b, *signature[2:], 1, False)


@pytest.mark.parametrize(
    "guard", ["compile", "capture", "device", "arch", "autocast", "deterministic", "frost"]
)
def test_eligibility_fallbacks(guard):
    x, w, b = metadata_tensors()
    signature = sorted(helper._QUALIFIED)[0]
    controls = {
        "compile": patch.object(torch.compiler, "is_compiling", return_value=True),
        "capture": patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        "device": patch.object(torch.cuda, "current_device", return_value=1),
        "arch": patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        "autocast": patch.object(torch, "is_autocast_enabled", return_value=True),
        "deterministic": patch.object(
            torch, "are_deterministic_algorithms_enabled", return_value=True
        ),
        "frost": patch.dict("os.environ", {"CUDNN_FRONTEND_ENABLE_FROST_ENGINES": "1"}),
    }
    with metadata_cuda(), controls[guard]:
        assert not helper._eligible(x, w, b, *signature[2:], 1, False)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_lazy_negative_views_fall_back(which):
    values = list(metadata_tensors())
    values[which] = torch._neg_view(values[which])
    signature = sorted(helper._QUALIFIED)[0]
    with metadata_cuda():
        assert not helper._eligible(*values, *signature[2:], 1, False)


def test_runtime_rejects_mismatch_and_propagates_broken_dependency():
    helper._runtime.cache_clear()
    with (
        patch.object(torch.version, "cuda", "13.4"),
        patch.object(torch.backends.cudnn, "version", return_value=92501),
        patch.object(torch, "__version__", "2.14.0a0+4fdf77b940.nv26.08"),
        patch.object(
            helper.importlib, "import_module", return_value=SimpleNamespace(__version__="1.26.0")
        ),
    ):
        assert helper._runtime() is None
        helper._runtime.cache_clear()
        with patch.object(
            helper.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("dependency absent", name="internal_dependency"),
        ):
            with pytest.raises(ModuleNotFoundError, match="dependency absent"):
                helper._runtime()
    helper._runtime.cache_clear()


def test_native_module_eval_keeps_parameters_trainable():
    wan = importlib.import_module("tensorrt_llm._torch.visual_gen.models.wan.wan_vae")
    module = wan.WanConv2d(2, 2, 3, padding=1).eval()
    x = torch.randn(1, 2, 4, 4)
    assert module.weight.requires_grad and module.bias.requires_grad
    with torch.no_grad(), patch.object(helper, "try_conv_bias", return_value=None) as entry:
        output = module(x)
    assert entry.call_args.kwargs == {"training": False}
    torch.testing.assert_close(output, F.conv2d(x, module.weight, module.bias, padding=1))
    assert set(module.state_dict()) == {"weight", "bias"}


@pytest.mark.parametrize("override", [None, (0, 1)])
def test_causal_cache_and_spatial_padding_stay_native(override):
    wan = importlib.import_module("tensorrt_llm._torch.visual_gen.models.wan.wan_vae")
    module = wan.WanCausalConv3d(2, 3, 3, padding=1).eval()
    x = torch.randn(1, 2, 2, 5, 6)
    cache = torch.randn(1, 2, 1, 5, 6)
    prepared = F.pad(torch.cat((cache, x), dim=2), (0, 0, 0, 0, 1, 0))
    padding = (0, 1, 1) if override is None else (0, *override)
    with torch.no_grad(), patch.object(helper, "try_conv_bias", return_value=None) as entry:
        actual = module(x, cache, spatial_padding=override)
    torch.testing.assert_close(entry.call_args.args[0], prepared)
    assert entry.call_args.args[4] == padding
    torch.testing.assert_close(
        actual, F.conv3d(prepared, module.weight, module.bias, padding=padding)
    )


def test_native_subclass_does_not_enter_helper():
    wan = importlib.import_module("tensorrt_llm._torch.visual_gen.models.wan.wan_vae")

    class CustomConv(wan.WanConv2d):
        pass

    with (
        torch.no_grad(),
        patch.object(helper, "try_conv_bias", side_effect=AssertionError("subclass routed")),
    ):
        CustomConv(2, 2, 3, padding=1).eval()(torch.ones(1, 2, 4, 4))


@pytest.mark.parametrize(
    "shape,wshape",
    [
        ((1, 1024, 3, 88, 160), (1024, 1024, 3, 3, 3)),
        ((1, 1024, 88, 160), (1024, 1024, 3, 3)),
    ],
)
def test_cuda_actual_graph_parameter_rebinding(shape, wshape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (10, 0) or "B200" not in torch.cuda.get_device_name():
        pytest.skip("Qualified B200 geometry")
    rank = len(shape) - 2
    fmt = torch.channels_last if rank == 2 else torch.channels_last_3d
    padding = (1, 1) if rank == 2 else (0, 1, 1)
    x = torch.ones(shape, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=fmt)
    w = torch.nn.Parameter(
        torch.zeros(wshape, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=fmt)
    )
    b = torch.nn.Parameter(torch.full((wshape[0],), 0.5, device="cuda", dtype=torch.bfloat16))
    with (
        torch.no_grad(),
        torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=False, allow_tf32=True
        ),
    ):
        before = helper._cache_info()
        y = helper.try_conv_bias(x, w, b, (1,) * rank, padding, (1,) * rank, 1, training=False)
        assert y is not None, "Qualified graph unexpectedly fell back"
        assert bool((y == 0.5).all().item())
        built = helper._cache_info()
        assert built["executions"] == before["executions"] + 1
        b.fill_(1)
        w = torch.nn.Parameter(torch.zeros_like(w))
        z = helper.try_conv_bias(x, w, b, (1,) * rank, padding, (1,) * rank, 1, training=False)
        assert z is not None and bool((z == 1).all().item())
        assert helper._cache_info()["builds"] == built["builds"]


@pytest.mark.parametrize(
    "shape,wshape",
    [
        ((1, 1024, 1, 176, 320), (512, 1024, 1, 1, 1)),
        ((1, 512, 1, 352, 640), (256, 512, 1, 1, 1)),
    ],
)
def test_excluded_singleton_shortcuts_remain_native(shape, wshape):
    x = torch.empty_strided(
        shape, helper._channels_last_stride(shape), device="meta", dtype=torch.bfloat16
    )
    w = torch.nn.Parameter(
        torch.empty_strided(
            wshape, helper._channels_last_stride(wshape), device="meta", dtype=torch.bfloat16
        )
    )
    b = torch.nn.Parameter(torch.empty(wshape[0], device="meta", dtype=torch.bfloat16))
    with (
        metadata_cuda(),
        patch.object(
            torch.cuda,
            "current_device",
            side_effect=AssertionError("Excluded shape reached CUDA queries"),
        ),
    ):
        assert not helper._eligible(x, w, b, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, False)
