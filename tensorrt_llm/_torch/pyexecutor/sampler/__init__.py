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

The upper-level orchestration (``Sampler`` / ``TorchSampler``)
lives in ``sampler.py``; ``sampling_resolution.py`` maps a request's sampling
params onto the fused sampling kernel, for it and for the one-model speculative
path. Implementation-specific kernel providers (FlashInfer,
vanilla/PyTorch, TRT-LLM ops) live under ``ops/`` and are selected
internally, never exposed as interchangeable backends to callers.

Layout -- imports only ever point downward; there are no cycles::

                              sampler.py
                            orchestration
                                  │
                    depends on every module below
                                  │
          ┌───────────┬───────────┼───────────┬───────────┐
          ▼           ▼           ▼           ▼           ▼
      beam_search  sampling_   logprobs   penalties      ...
                   resolution                          token_ban
                                                       top_p_decay
                                                       finish_reasons
                                                       seed_manager
                                                       sampler_features
          │           │           │           │           │
          └───────────┴───────────┴───────────┴───────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
           sampler_common                       ops/
       shared types, request queries,    vanilla + flashinfer
       constants, tensor helpers            sampling kernels
       ─────────── no intra-package imports ───────────

Feature modules are shown flat, but some depend on others -- notably
``beam_search`` -> ``logprobs`` -> ``sampler_features``. Each feature owns its ``*Store``
(persistent per-slot device state) and ``*Handler`` (lifecycle); ``TorchSampler``
holds one handler per feature and drives them. Types passed *between* modules
that no single feature owns live in ``sampler_common.py``.

Public symbols from ``sampler.py`` are re-exported here so existing
``pyexecutor.sampler`` import paths keep working. The re-export is lazy
(PEP 562 ``__getattr__``) so that importing lightweight submodules such as
``pyexecutor.sampler.sampling_resolution`` does not eagerly pull in ``sampler.py``
and its heavy dependency chain (which would create import cycles with
``speculative.interface``).
"""

import importlib

# Submodules of this package — never forward these to sampler.py (that would
# recurse, since accessing e.g. `.sampler` before it is bound re-enters here).
_SUBMODULES = frozenset({"sampler", "sampler_common", "sampling_resolution", "ops"})


def _import_submodule(name: str, submodule: str):
    try:
        return importlib.import_module(f"{__name__}.{submodule}")
    except AttributeError as e:
        # An AttributeError escaping the submodule is a bug inside it, not a
        # missing name here. Raised as is, hasattr() and getattr() with a
        # default would hide it, and ``from ... import`` would run the
        # submodule twice (losing the cause unless the name is a submodule).
        raise ImportError(
            f"cannot import {name!r} from {__name__!r}: importing "
            f"{__name__}.{submodule} raised {e!r}",
            name=__name__,
        ) from e


def __getattr__(name: str):
    if name in _SUBMODULES:
        return _import_submodule(name, name)
    # Lazily forward everything else to the sampler orchestration module.
    _sampler = _import_submodule(name, "sampler")
    try:
        return getattr(_sampler, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
