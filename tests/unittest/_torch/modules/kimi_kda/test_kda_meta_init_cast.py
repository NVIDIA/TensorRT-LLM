# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""_meta_safe_cast_dtype must stay usable under MetaInitMode.

Kimi K3 builds its KDA mixer inside MetaInitMode, which allows only
empty/empty_like on meta tensors and raises MetaInitException for anything
else. Reading ``param.data`` detaches, so casting the mixer's own parameters
through ``.data`` aborted model construction for every rank before the cast
helper's own is_meta branch was reached. CPU-only, no GPU required.
"""

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import _meta_safe_cast_dtype


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
        # Before the fix this raised:
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
