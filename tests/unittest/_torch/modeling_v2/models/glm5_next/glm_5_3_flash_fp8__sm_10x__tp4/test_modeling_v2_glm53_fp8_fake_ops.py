# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's fakes (``fake_ops.py``) against its native ops (``csrc/glm53Fp8Ops.cpp``).

Every op the target's library defines has a fake, ``fake_<op>``, that takes the op's schema
arguments by name and in order, so an argument added to a schema without its fake fails here
rather than at the first trace. Both ops then run under ``FakeTensorMode``, directly and through
their op wrappers, and the wrappers' outputs have the shapes and dtypes the real ops write.

Nothing runs on a device: the tensors are fake. The target's library must be installed, though;
without it the module is skipped.
"""

import importlib
import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4"
)

try:
    _target = importlib.import_module(_TARGET)
except NoTarget as error:
    pytest.skip(f"this installation does not carry the target: {error}", allow_module_level=True)

native_ops = _target.native_ops
fake_ops = importlib.import_module(f"{_TARGET}.fake_ops")
decode_moe_module = importlib.import_module(f"{_TARGET}.ops.moe.decode_moe")
boundary_module = importlib.import_module(f"{_TARGET}.ops.comm.mhc_ar_boundary")

# The shapes the kernels are compiled for (csrc/decodeMoe.h, csrc/mhcArBoundary.h).
HIDDEN, INTER, EXPERTS, BLOCK = 4096, 512, 288, 128
MULT, MIX = boundary_module.MULT, (2 + boundary_module.MULT) * boundary_module.MULT
WORKSPACE_POINTERS = 3 * boundary_module.WORLD + 2


def _target_schemas() -> dict:
    """The ops the target's library registers in its namespace, by name."""
    namespace = namespace_of(_TARGET)
    return {
        schema.name.split("::", 1)[1]: schema
        for schema in torch._C._jit_get_all_schemas()
        if schema.name.split("::", 1)[0] == namespace
    }


def test_every_op_has_a_fake_taking_its_schema_arguments():
    schemas = _target_schemas()
    assert {"decode_moe", "mhc_ar_boundary"} <= set(schemas), sorted(schemas)
    for name, schema in schemas.items():
        fake = getattr(fake_ops, f"fake_{name}", None)
        assert fake is not None, f"fake_ops.py has no fake_{name} for {schema}"
        expected = [argument.name for argument in schema.arguments]
        actual = list(inspect.signature(fake).parameters)
        assert actual == expected, (
            f"fake_{name} takes {actual}; the schema in csrc/glm53Fp8Ops.cpp is {schema}"
        )


def _empty(*shape, dtype):
    return torch.empty(shape, dtype=dtype, device="cuda")


def _decode_moe_inputs(rows: int) -> tuple:
    """``DecodeMoe.__call__``'s arguments, at the shapes the kernel is compiled for."""
    fp8 = torch.float8_e4m3fn
    return (
        _empty(rows, HIDDEN, dtype=torch.bfloat16),
        _empty(rows, EXPERTS, dtype=torch.float32),
        _empty(EXPERTS, dtype=torch.float32),
        2.5,
        10.0,
        _empty(EXPERTS, 2 * INTER, HIDDEN, dtype=fp8),
        _empty(EXPERTS, 2 * INTER // BLOCK, HIDDEN // BLOCK, dtype=torch.float32),
        _empty(EXPERTS, HIDDEN, INTER, dtype=fp8),
        _empty(EXPERTS, HIDDEN // BLOCK, INTER // BLOCK, dtype=torch.float32),
        _empty(2 * INTER, HIDDEN, dtype=fp8),
        _empty(2 * INTER // BLOCK, HIDDEN // BLOCK, dtype=torch.float32),
        _empty(HIDDEN, INTER, dtype=fp8),
        _empty(HIDDEN // BLOCK, INTER // BLOCK, dtype=torch.float32),
    )


def _boundary_inputs(rows: int) -> tuple:
    """``MhcArBoundary.__call__``'s arguments, at the shapes the kernel is compiled for."""
    return (
        _empty(rows, HIDDEN, dtype=torch.bfloat16),
        _empty(WORKSPACE_POINTERS, dtype=torch.int64),
        0,
        _empty(rows, MULT, HIDDEN, dtype=torch.bfloat16),
        _empty(rows, MULT, dtype=torch.float32),
        _empty(rows, MULT, MULT, dtype=torch.float32),
        _empty(MIX, MULT * HIDDEN, dtype=torch.float32),
        _empty(3, dtype=torch.float32),
        _empty(MIX, dtype=torch.float32),
        _empty(HIDDEN, dtype=torch.bfloat16),
        20,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        1e-5,
    )


def _assert_like(actual: torch.Tensor, shape: tuple, dtype: torch.dtype) -> None:
    assert tuple(actual.shape) == shape and actual.dtype == dtype, (actual.shape, actual.dtype)
    assert actual.device.type == "cuda", actual.device


@pytest.mark.parametrize("rows", [1, decode_moe_module.MAX_ROWS])
def test_decode_moe_under_fake_tensor_mode(rows):
    with FakeTensorMode():
        inputs = _decode_moe_inputs(rows)
        out = _empty(rows, HIDDEN, dtype=torch.bfloat16)
        assert native_ops.decode_moe(*inputs, out, False) is None
        wrapped = decode_moe_module.decode_moe(*inputs)
    _assert_like(out, (rows, HIDDEN), torch.bfloat16)
    _assert_like(wrapped, (rows, HIDDEN), torch.bfloat16)


@pytest.mark.parametrize("rows", [1, boundary_module.MAX_ROWS])
def test_mhc_ar_boundary_under_fake_tensor_mode(rows):
    outputs = (
        ((rows, MULT, HIDDEN), torch.bfloat16),
        ((rows, MULT), torch.float32),
        ((rows, MULT, MULT), torch.float32),
        ((rows, HIDDEN), torch.bfloat16),
    )
    with FakeTensorMode():
        inputs = _boundary_inputs(rows)
        written = [_empty(*shape, dtype=dtype) for shape, dtype in outputs]
        assert native_ops.mhc_ar_boundary(*inputs, *written, False) is None
        wrapped = boundary_module.mhc_ar_boundary(*inputs)
    assert len(wrapped) == len(outputs)
    for (shape, dtype), out, returned in zip(outputs, written, wrapped):
        _assert_like(out, shape, dtype)
        _assert_like(returned, shape, dtype)
