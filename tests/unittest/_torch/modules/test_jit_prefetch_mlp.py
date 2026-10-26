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
"""The relu2 MLP activation's JIT compile-ahead plan covers its real launches.

The provider plans the ``fused_relu2`` variants of a model's relu2 MLPs from
the layers alone; every launch of the kernel on an up projection's output, at
any token count, must be one of them.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _keys(calls):
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    keys = set()
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            _, kkc, _, _, binder = jit_fn.device_caches[driver.active.get_current_device()]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


def _mlp(width: int, activation):
    from tensorrt_llm._torch.modules.mlp import MLP

    mlp = object.__new__(MLP)
    torch.nn.Module.__init__(mlp)
    mlp.activation = activation
    mlp.up_proj = SimpleNamespace(out_features=width, dtype=torch.bfloat16)
    return mlp


def test_relu2_provider_covers_the_mlp_activation():
    from tensorrt_llm._torch.fused_relu2_triton import fused_relu2
    from tensorrt_llm._torch.modules.jit_prefetch_mlp import Relu2Provider
    from tensorrt_llm._torch.utils import relu2

    model = torch.nn.Sequential(_mlp(2048, relu2), _mlp(2048, torch.nn.functional.silu))
    provider = Relu2Provider(model)
    assert provider.shapes == [(2048, torch.bfloat16)]
    (batch,) = list(provider.early_batches())
    planned = _keys(provider(batch))
    assert planned
    for rows in (1, 7, 16, 100, 2048):
        x = torch.randn(rows, 2048, dtype=torch.bfloat16, device="cuda")
        with jp.shadow_launches() as rec:
            fused_relu2(x)
        assert _keys(rec) <= planned, rows
    assert not Relu2Provider(torch.nn.Sequential(_mlp(2048, torch.nn.functional.silu)))
