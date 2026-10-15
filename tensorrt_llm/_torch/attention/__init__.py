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
"""Attention modules for the PyTorch backend.

Holds the module-layer implementations -- standard attention, cross attention,
Multi-head Latent Attention, QK-norm attention and rotary embedding -- that
were previously mixed into the shared ``_torch/modules`` bucket. The backend
classes those modules dispatch to live in the ``backends`` sub-package here.

``ATTENTION_DEVELOPER_GUIDE.md`` in this directory is the contract document for
attention metadata, backend families and KV-cache behaviour; read it before
changing anything here.

The module-layer classes listed in ``__all__`` can be imported from this
package (``from tensorrt_llm._torch.attention import Attention``). They are
resolved lazily by the module-level ``__getattr__`` below, so importing this
package, or any sub-package such as ``backends``, loads none of the modules
beside it. Keep it that way: those modules pull in torch, the attention
backends and the distributed layer, so an eager import here would make every
backend-only caller pay for the whole module layer at import time. It would
also put this ``__init__`` on the ``backends`` -> ``rotary_embedding`` import
path, so a module on that path that imported a name from the package root would
find the package partially initialized and fail with ``ImportError``.

Code inside this package imports from the defining submodule
(``from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding``),
never from the package root.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .attention import Attention
    from .cross_attention import CrossAttention
    from .mla import MLA
    from .qk_norm_attention import QKNormRoPEAttention
    from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding

_LAZY_ATTRS = {
    "Attention": "attention",
    "CrossAttention": "cross_attention",
    "MLA": "mla",
    "QKNormRoPEAttention": "qk_norm_attention",
    "RotaryEmbedding": "rotary_embedding",
    "MRotaryEmbedding": "rotary_embedding",
}

__all__ = [
    "Attention",
    "CrossAttention",
    "MLA",
    "MRotaryEmbedding",
    "QKNormRoPEAttention",
    "RotaryEmbedding",
]


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        # Also resolve bare submodule access (attention.backends,
        # attention.rotary_embedding, ...).
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            # Only translate "no such submodule" into AttributeError; a
            # ModuleNotFoundError raised *inside* an existing submodule
            # (missing dependency) must propagate unchanged.
            if e.name != f"{__name__}.{name}":
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f".{module_name}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr  # cache: subsequent access skips __getattr__
    return attr


def __dir__():
    return sorted(set(__all__) | set(globals()))
