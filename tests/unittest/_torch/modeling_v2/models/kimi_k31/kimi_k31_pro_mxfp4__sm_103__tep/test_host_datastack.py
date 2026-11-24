# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run_on_large_datastack_chunk`` of the tep and dep targets' ``modeling.py``: calls through a large trampoline frame
(no GPU)."""

import importlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


@pytest.fixture(
    params=["kimi_k31_pro_mxfp4__sm_103__tep", "kimi_k31_pro_mxfp4__sm_103__dep"],
    ids=["tep", "dep"],
)
def host_datastack(request):
    """The target's ``modeling`` module, which holds its copy of the trampoline."""
    return importlib.import_module(f"{_FAMILY}.{request.param}.modeling")


def _deep(n: int) -> int:
    return 0 if n == 0 else 1 + _deep(n - 1)


def test_returns_the_call_result_with_args_and_kwargs(host_datastack):
    def fn(a, b, *, c):
        return (a, b, c)

    assert host_datastack.run_on_large_datastack_chunk(fn, 1, 2, c=3) == (1, 2, 3)
    # "fn" can be passed through as a keyword argument of the wrapped call
    assert host_datastack.run_on_large_datastack_chunk(lambda **kw: kw, fn=4) == {"fn": 4}


def test_trampoline_frame_holds_the_configured_locals_and_deep_calls_run(host_datastack):
    trampoline = host_datastack._TRAMPOLINE
    assert trampoline.__code__.co_nlocals >= host_datastack.DATASTACK_KIB * 1024 // 8
    assert host_datastack.run_on_large_datastack_chunk(_deep, 600) == 600
    caller = host_datastack.run_on_large_datastack_chunk(
        lambda: sys._getframe(1).f_code.co_filename
    )
    assert caller == "<kimi_k31 host_datastack>"


def test_exceptions_propagate(host_datastack):
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        host_datastack.run_on_large_datastack_chunk(boom)
