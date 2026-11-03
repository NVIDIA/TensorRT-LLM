# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``_meta_safe_cast_dtype`` must stay usable under ``MetaInitMode``.

Kimi K3 builds its KDA mixer inside ``MetaInitMode``, which allows only
``empty``/``empty_like`` on meta tensors and raises ``MetaInitException`` for
anything else. Reading ``param.data`` detaches, so casting the mixer's own
parameters through ``.data`` aborted model construction before the cast
helper's own ``is_meta`` branch was reached. CPU only, no GPU required.
"""

import pytest
import torch
from torch import nn

pytest.importorskip("fla")  # kimi_kda_mixer imports fla.modules at load time

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import _meta_safe_cast_dtype  # noqa: E402

# The L0 CPU stages run `pytest -m cpu_only` and collect only files that carry
# this marker.
pytestmark = pytest.mark.cpu_only


class _OwnParamModule(nn.Module):
    """Module holding a direct parameter, like the KDA mixer's own weights."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 8, dtype=torch.float32))
        self.register_buffer("count", torch.zeros(2, dtype=torch.int32))


def test_cast_under_meta_init_does_not_detach() -> None:
    with MetaInitMode():
        module = _OwnParamModule()
        assert module.weight.is_meta, "expected MetaInitMode to place the parameter on meta"
        # Casting through ``param.data`` would detach and raise
        #   MetaInitException: Meta tensor used in unsupported function:
        #   aten.detach.default
        _meta_safe_cast_dtype(module, torch.bfloat16)

    assert module.weight.dtype == torch.bfloat16
    assert module.weight.is_meta


def test_cast_off_meta_still_converts_values() -> None:
    module = _OwnParamModule()
    with torch.no_grad():
        module.weight.fill_(1.5)

    _meta_safe_cast_dtype(module, torch.bfloat16)

    assert module.weight.dtype == torch.bfloat16
    assert torch.equal(module.weight.float(), torch.full((4, 8), 1.5, dtype=torch.float32))


def test_cast_leaves_integer_buffers_alone() -> None:
    module = _OwnParamModule()

    _meta_safe_cast_dtype(module, torch.bfloat16)

    # Only floating point storage is retyped; counters must keep their dtype.
    assert module.count.dtype == torch.int32


class _GateParamModule(nn.Module):
    """Own parameters shaped like the KDA mixer's A_log / dt_bias plus a weight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 8, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(4, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(8, dtype=torch.float32))


def test_cast_keeps_named_params_in_their_dtype() -> None:
    module = _GateParamModule()
    # Values a bf16 round trip would change.
    exact = torch.tensor([1.0 + 2.0**-12, -3.0 - 2.0**-10, 0.1, 5.5e-3], dtype=torch.float32)
    with torch.no_grad():
        module.weight.fill_(1.5)
        module.A_log.copy_(exact)
        module.dt_bias.copy_(torch.linspace(-7.0, -2.3, 8, dtype=torch.float32))
    dt_bias = module.dt_bias.detach().clone()

    _meta_safe_cast_dtype(module, torch.bfloat16, keep=("A_log", "dt_bias"))

    assert module.weight.dtype == torch.bfloat16
    assert module.A_log.dtype == torch.float32 and torch.equal(module.A_log, exact)
    assert module.dt_bias.dtype == torch.float32 and torch.equal(module.dt_bias, dt_bias)


def test_cast_keeps_named_params_under_meta_init() -> None:
    with MetaInitMode():
        module = _GateParamModule()
        _meta_safe_cast_dtype(module, torch.bfloat16, keep=("A_log", "dt_bias"))

    assert module.weight.dtype == torch.bfloat16 and module.weight.is_meta
    assert module.A_log.dtype == torch.float32 and module.A_log.is_meta
    assert module.dt_bias.dtype == torch.float32 and module.dt_bias.is_meta
