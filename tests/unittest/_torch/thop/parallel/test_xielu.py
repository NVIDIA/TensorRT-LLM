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
"""``trtllm::xielu`` must be bit-identical to the fp32 PyTorch expression.

The kernel evaluates the same fp32 operations in the same order as
``xielu_reference`` (no FMA contraction) and rounds once, so every comparison
here is exact.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.modules.xielu as xielu_mod
from tensorrt_llm._torch.modules.xielu import XIELU, xielu_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.ops.trtllm, "xielu"),
    reason="Requires CUDA and the trtllm::xielu op",
)

# softplus(0.3), 0.5 + softplus(-1.7), and the checkpoint's bf16 beta / eps.
COEFFS = dict(a_p=0.8543552756309509, a_n=0.6677860170602798, beta=0.5, eps=-9.98377799987793e-07)

DTYPES = [torch.bfloat16, torch.float16]


def _op(x: torch.Tensor, a_p: float, a_n: float, beta: float, eps: float) -> torch.Tensor:
    return torch.ops.trtllm.xielu(x, a_p, a_n, beta, eps)


def _assert_identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Same bit patterns (so +0.0 != -0.0); NaN only needs to be NaN in both."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    bits = torch.int16 if actual.element_size() == 2 else torch.int32
    torch.testing.assert_close(actual.view(bits)[~nan], expected.view(bits)[~nan], atol=0, rtol=0)


def _special_values(dtype: torch.dtype) -> torch.Tensor:
    finfo = torch.finfo(dtype)
    eps = COEFFS["eps"]
    values = [
        0.0,
        -0.0,
        eps,
        eps * 0.5,
        eps * 2,
        -5e-7,
        1e-7,
        finfo.tiny,
        -finfo.tiny,
        finfo.smallest_normal / 4,  # subnormal
        -finfo.smallest_normal / 4,
        1.0,
        -1.0,
        -30.0,
        -1e4,
        finfo.max,
        -finfo.max,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    return torch.tensor(values, dtype=torch.float32).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (7,),
        (8,),
        (9,),
        (1000003,),
        (1, 21504),
        (3, 21504),
        (128, 21504),
        (4, 5, 6),
    ],
)
def test_matches_reference(dtype, shape):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 4).to(dtype)
    _assert_identical(_op(x, **COEFFS), xielu_reference(x, **COEFFS))


@pytest.mark.parametrize("dtype", DTYPES)
def test_special_values(dtype):
    x = _special_values(dtype).cuda()
    _assert_identical(_op(x, **COEFFS), xielu_reference(x, **COEFFS))


@pytest.mark.parametrize("dtype", DTYPES)
def test_dense_sweep(dtype):
    """Every finite value of the 16-bit dtype."""
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32, device="cuda").to(torch.int16)
    x = bits.view(dtype)
    _assert_identical(_op(x, **COEFFS), xielu_reference(x, **COEFFS))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [1, 3, 7])
def test_misaligned_input(dtype, offset):
    """Views that are not 16-byte aligned take the scalar path."""
    x = (torch.randn(4096 + offset, device="cuda") * 4).to(dtype)[offset:]
    assert x.data_ptr() % 16 != 0
    _assert_identical(_op(x, **COEFFS), xielu_reference(x, **COEFFS))


@pytest.mark.parametrize("dtype", DTYPES)
def test_non_contiguous_input(dtype):
    x = (torch.randn(64, 256, device="cuda") * 4).to(dtype).t()
    out = _op(x, **COEFFS)
    assert out.shape == x.shape
    _assert_identical(out, xielu_reference(x, **COEFFS))


def test_output_layout_matches_fake():
    """Singleton dims let a non-standard stride count as contiguous."""
    x = (torch.randn(1, 40, device="cuda") * 4).to(torch.bfloat16).as_strided((2, 1), (1, 20))
    assert x.is_contiguous()
    out = _op(x, **COEFFS)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake_out = _op(mode.from_tensor(x), **COEFFS)
    assert out.stride() == fake_out.stride() == (1, 1)
    _assert_identical(out, xielu_reference(x, **COEFFS))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
def test_input_on_non_current_device():
    x = (torch.randn(4096, device="cuda:1") * 4).to(torch.bfloat16)
    with torch.cuda.device(0), torch.cuda.stream(torch.cuda.Stream(device=1)):
        out = _op(x, **COEFFS)
    torch.cuda.synchronize(1)
    assert out.device == x.device
    _assert_identical(out, xielu_reference(x, **COEFFS))


def test_empty():
    x = torch.empty(0, 21504, device="cuda", dtype=torch.bfloat16)
    assert _op(x, **COEFFS).shape == x.shape


def test_rejects_float32():
    with pytest.raises(RuntimeError):
        _op(torch.randn(16, device="cuda"), **COEFFS)


def test_coefficients_are_used():
    x = (torch.randn(1024, device="cuda") * 4).to(torch.bfloat16)
    base = _op(x, **COEFFS)
    for name in ("a_p", "a_n", "beta"):
        changed = _op(x, **{**COEFFS, name: COEFFS[name] * 1.5})
        assert not torch.equal(base, changed), name


def test_fake_registration():
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    torch.library.opcheck(
        torch.ops.trtllm.xielu.default,
        (x, *COEFFS.values()),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_cuda_graph():
    x = (torch.randn(8, 21504, device="cuda") * 4).to(torch.bfloat16)
    _op(x, **COEFFS)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _op(x, **COEFFS)
    x.copy_((torch.randn_like(x, dtype=torch.float32) * 4).to(x.dtype))
    graph.replay()
    _assert_identical(out, xielu_reference(x, **COEFFS))


@pytest.fixture
def op_calls(monkeypatch):
    """Record dispatches to the kernel so a silent fallback cannot pass."""
    calls = []
    real = torch.ops.trtllm.xielu

    class _Spy:
        def __call__(self, *args):
            calls.append(args[0].shape)
            return real(*args)

    monkeypatch.setattr(torch.ops.trtllm, "xielu", _Spy(), raising=False)
    xielu_mod._xielu_op_available.cache_clear()
    yield calls
    xielu_mod._xielu_op_available.cache_clear()


@pytest.mark.parametrize("dtype", DTYPES)
def test_module_uses_kernel(dtype, op_calls):
    act = XIELU().cuda()
    x = (torch.randn(4, 21504, device="cuda") * 4).to(dtype)
    out = act(x)
    assert op_calls == [x.shape]
    _assert_identical(out, xielu_reference(x, act.a_p, act.a_n, act.beta_value, act.eps_value))


def test_module_env_disables_kernel(monkeypatch, op_calls):
    monkeypatch.setenv("TRTLLM_XIELU_CUDA", "0")
    xielu_mod._xielu_op_available.cache_clear()
    act = XIELU().cuda()
    act(torch.randn(4, 64, device="cuda", dtype=torch.bfloat16))
    assert op_calls == []
