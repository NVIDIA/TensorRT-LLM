# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CSA2 cache-kernel construction and memoization without CuTe compilation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

pytestmark = pytest.mark.cpu_only


def _identity(function):
    return function


@pytest.fixture
def cache_factory(monkeypatch):
    base = SimpleNamespace(
        cuda=SimpleNamespace(CUstream=object),
        cutlass=SimpleNamespace(
            Int32=int, Int64=int, Float32=float, Uint8=int, Uint32=int, Constexpr=object
        ),
        cute=SimpleNamespace(jit=_identity, kernel=_identity, Pointer=object),
        llvm=object(),
        T=object(),
        dsl_user_op=_identity,
        SmemAllocator=object(),
        tensor1d=object(),
        tensor2d=object(),
        bitcast_u32_f32=object(),
    )
    glue = Mock(return_value=base)
    monkeypatch.setattr(kernel, "_glue_kernels", glue)
    kernel._cache_kernels.cache_clear()
    try:
        yield kernel._cache_kernels, glue
    finally:
        kernel._cache_kernels.cache_clear()


def test_cache_kernel_namespace_is_built_once_until_cleared(cache_factory):
    factory, glue = cache_factory
    namespace = factory()
    expected_names = {
        "QuantizeScatterKernel",
        "GatherDequantKernel",
        "ScatterPackedKernel",
        "StageCompactKernel",
        "StageDequantKernel",
        "SharedCompactKernel",
        "SharedDecodeKernel",
        "StageScaleKernel",
        "StageScaleFinalizeKernel",
    }
    assert set(vars(namespace)) == expected_names
    for name, kernel_class in vars(namespace).items():
        assert kernel_class.__name__ == name
        assert callable(kernel_class.__call__)
        assert callable(kernel_class.kernel)
    assert factory() is namespace
    glue.assert_called_once_with()
    assert factory.cache_info().hits == 1

    factory.cache_clear()
    rebuilt = factory()
    assert rebuilt is not namespace
    assert set(vars(rebuilt)) == expected_names
    for name in expected_names:
        assert getattr(rebuilt, name) is not getattr(namespace, name)
    assert glue.call_count == 2


@pytest.mark.parametrize(
    "class_name,arguments,expected",
    [
        (
            "GatherDequantKernel",
            ("main", 512),
            {"cache_format": "main", "head_dim": 512, "threads": 128},
        ),
        (
            "GatherDequantKernel",
            ("index", 128),
            {"cache_format": "index", "head_dim": 128, "threads": 32},
        ),
        (
            "GatherDequantKernel",
            ("swa", 160),
            {"cache_format": "swa", "head_dim": 160, "threads": 64},
        ),
        (
            "ScatterPackedKernel",
            (True, 256, 32),
            {"footer": True, "data_bytes": 256, "scale_bytes": 32},
        ),
        (
            "StageCompactKernel",
            (True, False, True),
            {"fp8": True, "has_main": False, "premapped": True},
        ),
        ("StageDequantKernel", (False,), {"fp8": False}),
        (
            "SharedCompactKernel",
            (False, True, False),
            {"fp8": False, "has_main": True, "premapped": False},
        ),
        ("SharedDecodeKernel", ("swa", True), {"cache_format": "swa", "fp8": True}),
        ("StageScaleKernel", (16, 4096), {"groups": 16, "limit": 4096}),
        ("StageScaleFinalizeKernel", (True, False), {"has_q": True, "q_shared": False}),
    ],
)
def test_cache_kernel_constructor_configuration(cache_factory, class_name, arguments, expected):
    factory, _ = cache_factory
    instance = getattr(factory(), class_name)(*arguments)
    assert vars(instance) == expected


@pytest.mark.parametrize("job_count", [0, 1, 4, 5])
def test_quantize_kernel_job_limits_and_thread_selection(cache_factory, job_count):
    factory, _ = cache_factory
    geometries = [("index", 128), ("main", 512), ("swa", 160), ("main", 256), ("index", 128)]
    jobs = [
        kernel.QuantizeJobSpec(cache_format, head_dim, "row")
        for cache_format, head_dim in geometries[:job_count]
    ]
    kernel_class = factory().QuantizeScatterKernel
    if job_count in (0, 5):
        with pytest.raises(ValueError, match="one to four jobs"):
            kernel_class(jobs)
    else:
        instance = kernel_class(jobs)
        assert instance.jobs == tuple(jobs)
        assert instance.threads == (32 if job_count == 1 else 128)
