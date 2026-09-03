# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``fast_custom_op``.

``fast_custom_op`` trades ``@torch.library.custom_op``'s per-call validation
for a much cheaper dispatch path, so the properties that stop that trade from
becoming a correctness problem are worth pinning down:

* it registers on the same devices the ported op used to support,
* it accepts both device-type ("cuda") and dispatch-key ("CUDA") spellings,
* it is transparent to ``torch.compile`` (no graph break, same result), and
* ``TLLM_VALIDATE_CUSTOM_OPS=1`` really does restore the aliasing check that
  the fast path drops.

Everything here runs on CPU so the whole file is cheap enough for a
hardware-agnostic CI stage.
"""

import importlib
import itertools
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.custom_ops import fast_custom_op as fco_module
from tensorrt_llm._torch.custom_ops.fast_custom_op import fast_custom_op

_COUNTER = itertools.count()


def _unique_ns(prefix: str) -> str:
    """torch libraries are process-global; give every op a fresh namespace."""
    return f"{prefix}_{next(_COUNTER)}"


def test_device_agnostic_registration_runs_on_cpu():
    """``device_types=None`` must keep an op usable on every backend.

    ``@torch.library.custom_op`` registers a backend-agnostic kernel when
    ``device_types`` is omitted. Porting such an op must not silently narrow it
    to CUDA, which would turn a CPU call into a NotImplementedError.
    """
    ns = _unique_ns("fco_agnostic")

    @fast_custom_op(f"{ns}::add_one", mutates_args=(), device_types=None)
    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    @add_one.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    x = torch.arange(4, dtype=torch.float32)
    torch.testing.assert_close(getattr(torch.ops, ns).add_one(x), x + 1)


@pytest.mark.parametrize("device_types", ["cuda", "CUDA"])
def test_accepts_device_type_and_dispatch_key_spellings(device_types):
    """Both "cuda" and "CUDA" must register.

    ``custom_op`` takes a device type, ``Library.impl`` takes a dispatch key.
    Ops in the tree use both spellings, so rejecting either would break a port
    at import time.
    """
    ns = _unique_ns("fco_spelling")

    @fast_custom_op(f"{ns}::noop", mutates_args=(), device_types=device_types)
    def noop(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @noop.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    # Registration alone is the assertion; the op resolves off torch.ops.
    assert hasattr(getattr(torch.ops, ns), "noop")


def test_matches_custom_op_under_eager_and_compile():
    """Same function, both decorators, same answers -- and no graph break."""
    ns_fast, ns_ref = _unique_ns("fco_fast"), _unique_ns("fco_ref")

    def impl(x: torch.Tensor, scale: float, bias: Optional[torch.Tensor]) -> torch.Tensor:
        out = x * scale
        return out if bias is None else out + bias

    fast_op = fast_custom_op(f"{ns_fast}::scale", mutates_args=(), device_types=None)(impl)
    fast_op.register_fake(lambda x, scale, bias: torch.empty_like(x))

    ref_op = torch.library.custom_op(f"{ns_ref}::scale", mutates_args=())(impl)
    ref_op.register_fake(lambda x, scale, bias: torch.empty_like(x))

    x = torch.randn(8)
    bias = torch.randn(8)
    fast_call = lambda: getattr(torch.ops, ns_fast).scale(x, 2.0, bias)  # noqa: E731
    ref_call = lambda: getattr(torch.ops, ns_ref).scale(x, 2.0, bias)  # noqa: E731

    torch.testing.assert_close(fast_call(), ref_call())

    explained = torch._dynamo.explain(fast_call)()
    assert explained.graph_break_count == 0

    torch._dynamo.reset()
    torch.testing.assert_close(torch.compile(fast_call, dynamic=False)(), ref_call())


def test_validate_env_restores_aliasing_check(monkeypatch):
    """``TLLM_VALIDATE_CUSTOM_OPS=1`` must catch an undeclared alias.

    An op that declares ``mutates_args=()`` but returns an input aliases that
    input. The fast path cannot see this and would let a later in-place write
    corrupt the caller's tensor; the validating path must raise instead. This
    is the whole reason the switch exists, so assert both halves.
    """
    ns_fast = _unique_ns("fco_alias_fast")

    @fast_custom_op(f"{ns_fast}::identity", mutates_args=(), device_types=None)
    def identity(x: torch.Tensor) -> torch.Tensor:
        return x  # undeclared alias of the input

    @identity.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    # Fast path: no check, so the aliasing write lands on the caller's tensor.
    x = torch.zeros(3)
    getattr(torch.ops, ns_fast).identity(x).add_(1.0)
    assert torch.equal(x, torch.ones(3)), "fast path is expected to alias"

    # Validating path: same bug, now rejected.
    monkeypatch.setenv("TLLM_VALIDATE_CUSTOM_OPS", "1")
    validating = importlib.reload(fco_module)
    assert validating.VALIDATE_CUSTOM_OPS

    ns_val = _unique_ns("fco_alias_val")

    @validating.fast_custom_op(f"{ns_val}::identity", mutates_args=(), device_types=None)
    def identity_validated(x: torch.Tensor) -> torch.Tensor:
        return x

    @identity_validated.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    with pytest.raises(RuntimeError, match="alias"):
        getattr(torch.ops, ns_val).identity(torch.zeros(3))

    # Leave the module in its default (fast) state for other tests.
    monkeypatch.delenv("TLLM_VALIDATE_CUSTOM_OPS")
    importlib.reload(fco_module)
