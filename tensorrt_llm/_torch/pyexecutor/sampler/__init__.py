# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampler package.

The upper-level orchestration (``Sampler`` / ``TorchSampler`` / ``TRTLLMSampler``)
lives in ``sampler.py`` and depends on operation-level APIs in
``sampling_utils.py``. Implementation-specific kernel providers (FlashInfer,
vanilla/PyTorch, TRT-LLM ops) live under ``ops/`` and are selected
internally, never exposed as interchangeable backends to callers.

Public symbols from ``sampler.py`` are re-exported here so existing
``pyexecutor.sampler`` import paths keep working. The re-export is lazy
(PEP 562 ``__getattr__``) so that importing lightweight submodules such as
``pyexecutor.sampler.sampling_utils`` does not eagerly pull in ``sampler.py``
and its heavy dependency chain (which would create import cycles with
``speculative.interface``).
"""

import importlib

# Submodules of this package — never forward these to sampler.py (that would
# recurse, since accessing e.g. `.sampler` before it is bound re-enters here).
_SUBMODULES = frozenset({"sampler", "sampling_utils", "ops"})


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    # Lazily forward everything else to the sampler orchestration module.
    _sampler = importlib.import_module(f"{__name__}.sampler")
    try:
        return getattr(_sampler, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
