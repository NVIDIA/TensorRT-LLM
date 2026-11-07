# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks execute the real QKV host method with Python optimization enabled."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.cpu_only


@pytest.fixture(params=[0, 2], ids=["normal", "optimized"])
def runner(request, monkeypatch):
    spec = importlib.util.find_spec("tensorrt_llm")
    source = (
        Path(next(iter(spec.submodule_search_locations)))
        / "_torch/kernels/rubin/m3_qkv_fusion/kernel.py"
    )
    tree = ast.parse(source.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in ("_DeviceArgs", "M3QkvFusedRunner", "_cache_strides")
    ]
    driver = Mock(spec_set=["CUstream"])
    driver.CUstream.side_effect = int
    namespace = {"HEAD_DIM": 128, "PAGE_SIZE": 128, "torch": torch, "cuda_driver": driver}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(source), "exec", optimize=request.param
        ),
        namespace,
    )
    cls = namespace["M3QkvFusedRunner"]
    handle = cls.__new__(cls)
    handle.op = Mock(spec_set=["n_total", "num_heads_q", "num_heads_kv", "num_heads_index"])
    handle.op.n_total = 640
    handle.op.num_heads_q = handle.op.num_heads_kv = handle.op.num_heads_index = 1
    handle.debug = False
    runtime = ModuleType("cutlass.cute.runtime")
    runtime.from_dlpack = Mock(
        side_effect=AssertionError("invalid operands reached DLPack conversion")
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    return handle, runtime.from_dlpack


@pytest.fixture
def operands():
    return dict(
        act=torch.empty((2, 128), dtype=torch.float8_e4m3fn),
        act_sf=torch.empty(512, dtype=torch.uint8),
        weight=torch.empty((640, 128), dtype=torch.float8_e4m3fn),
        weight_sf=torch.empty(2560, dtype=torch.uint8),
        alpha=torch.ones(1),
        q_out=torch.empty((2, 1, 128), dtype=torch.float8_e4m3fn),
        index_q_out=torch.empty((2, 1, 128), dtype=torch.float8_e4m3fn),
        kv_data_cache=torch.full((2, 2, 1, 128, 64), 17, dtype=torch.uint8),
        kv_scale_cache=torch.full((2, 2, 1, 128, 8), 19, dtype=torch.uint8),
        index_k_cache=torch.full((2, 1, 128, 128), 23, dtype=torch.uint8).view(torch.float8_e4m3fn),
        out_cache_loc=torch.zeros(2, dtype=torch.int32),
        kv_quant_scale=torch.ones(3),
        rotary_cos_sin=torch.empty((128, 2, 32)),
        position_ids=torch.zeros(2, dtype=torch.int32),
        norm_weights=torch.ones(512, dtype=torch.bfloat16),
        eps=1e-6,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "weight_shape",
        "k_alignment",
        "q_shape",
        "index_q_shape",
        "data_shape",
        "scale_shape",
        "index_shape",
        "data_stride",
        "scale_stride",
        "index_stride",
        "data_alignment",
        "scale_alignment",
        "index_alignment",
        "norm_dtype",
        "norm_size",
        "debug_output",
        "gemm_dtype",
    ],
)
def test_invalid_qkv_arguments_fail_before_packing_or_cache_write(runner, operands, invalid):
    handle, pack = runner
    if invalid == "weight_shape":
        operands["weight"] = operands["weight"][:-1]
    elif invalid == "k_alignment":
        operands["act"] = torch.empty((2, 129), dtype=torch.float8_e4m3fn)
        operands["weight"] = torch.empty((640, 129), dtype=torch.float8_e4m3fn)
    elif invalid in ("q_shape", "index_q_shape"):
        key = "q_out" if invalid == "q_shape" else "index_q_out"
        operands[key] = operands[key][:1]
    elif invalid.endswith("_shape"):
        key = {
            "data_shape": "kv_data_cache",
            "scale_shape": "kv_scale_cache",
            "index_shape": "index_k_cache",
        }[invalid]
        operands[key] = operands[key][..., :-1]
    elif invalid.endswith("_stride"):
        key = {
            "data_stride": "kv_data_cache",
            "scale_stride": "kv_scale_cache",
            "index_stride": "index_k_cache",
        }[invalid]
        old = operands[key]
        backing = torch.empty((*old.shape[:-1], 2 * old.shape[-1]), dtype=old.dtype)
        operands[key] = backing[..., ::2]
    elif invalid.endswith("_alignment"):
        key = {
            "data_alignment": "kv_data_cache",
            "scale_alignment": "kv_scale_cache",
            "index_alignment": "index_k_cache",
        }[invalid]
        old = operands[key]
        operands[key] = torch.empty(old.numel() + 1, dtype=old.dtype)[1:].view(old.shape)
    elif invalid == "norm_dtype":
        operands["norm_weights"] = operands["norm_weights"].float()
    elif invalid == "norm_size":
        operands["norm_weights"] = operands["norm_weights"][:-1]
    elif invalid == "debug_output":
        handle.debug = True
    else:
        operands["gemm_out"] = torch.empty((2, 640), dtype=torch.float32)
    caches = {
        key: operands[key].view(torch.uint8).clone()
        for key in ("kv_data_cache", "kv_scale_cache", "index_k_cache")
    }
    with pytest.raises(ValueError, match="Unsupported fused QKV operands"):
        handle.make_args(**operands)
    pack.assert_not_called()
    for key, before in caches.items():
        assert torch.equal(operands[key].view(torch.uint8), before)


@pytest.mark.parametrize("explicit_stream", [None, 41])
def test_launch_looks_up_operand_device_stream_at_call_time(runner, monkeypatch, explicit_stream):
    handle, _ = runner
    args_type = handle.run_args.__globals__["_DeviceArgs"]
    args = args_type(("ffi_argument",), torch.device("cuda:1"))
    device_zero = Mock(spec_set=["cuda_stream"])
    device_zero.cuda_stream = 17
    device_one = Mock(spec_set=["cuda_stream"])
    device_one.cuda_stream = 29
    current = Mock(
        side_effect=lambda device=None: device_one if device == args.device else device_zero
    )
    monkeypatch.setattr(torch.cuda, "current_stream", current)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    handle.compiled = Mock()
    handle.run_args(args, stream=explicit_stream)
    if explicit_stream is None:
        current.assert_called_once_with(torch.device("cuda:1"))
        handle.compiled.assert_called_once_with("ffi_argument", 29)
    else:
        current.assert_not_called()
        handle.compiled.assert_called_once_with("ffi_argument", explicit_stream)
    # Prepared arguments keep the device, not a stream captured during argument preparation.
    current.reset_mock()
    handle.compiled.reset_mock()
    device_one.cuda_stream = 53
    handle.run_args(args)
    current.assert_called_once_with(torch.device("cuda:1"))
    handle.compiled.assert_called_once_with("ffi_argument", 53)


def test_prepared_arguments_retain_tensor_device_and_tuple_interface(runner, operands):
    handle, pack = runner
    pack.side_effect = lambda tensor, **kwargs: tensor
    args = handle.make_args(**operands)
    assert isinstance(args, tuple)
    assert args.device == operands["act"].device
    assert args[0] is operands["act"]
