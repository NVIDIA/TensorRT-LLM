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
"""Import the MuseSparkOpen decoder module with or without a built TensorRT-LLM.

``load()`` returns ``(modeling_musespark_open, modeling_utils, Mapping)``. With
a built package (compiled bindings, Triton, ...) those are the installed
modules. Without one, as on a CPU-only checkout, those three source files and
the dependency-free ``disaggregation/pp_dflash_context.py`` that the decoder
reads its opt-in switch from are executed for real under their package names while every other module they
import is replaced by a permissive stand-in, and
``modeling_speculative.SpecDecOneEngineForCausalLM`` by a minimal subclass of
the real ``DecoderModelForCausalLM`` (its constructor and draft-weight load are
what a test patches; its ``external_drafter_mapping`` is the real one, the
target's mapping). The stand-ins are installed only while these modules
import; ``sys.modules`` is restored afterwards, so other tests in the session
see no trace of them.

The stand-in path covers the module-level definitions, the construction-time
refusals, the pipeline-stage hooks and the loader entry points; nothing in it
runs a kernel.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Generic, TypeVar

_ROOT = Path(__file__).resolve().parents[4]
_PKG = "tensorrt_llm"
_REAL_FILES = {
    "tensorrt_llm.mapping": _ROOT / "tensorrt_llm" / "mapping.py",
    "tensorrt_llm._torch.models.modeling_utils": _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "models"
    / "modeling_utils.py",
    "tensorrt_llm._torch.models.modeling_musespark_open": _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "models"
    / "modeling_musespark_open.py",
    "tensorrt_llm._torch.disaggregation.pp_dflash_context": _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "disaggregation"
    / "pp_dflash_context.py",
}
_SPECULATIVE = "tensorrt_llm._torch.models.modeling_speculative"
_STUBBED_TOP_LEVEL = (_PKG, "triton", "nvtx", "flashinfer", "cuda", "tensorrt")


class _AnyMeta(type):
    """Metaclass of the stand-in classes: any attribute or subscript is another stand-in."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stand_in(f"{cls.__qualname__}.{name}")

    def __getitem__(cls, item):
        return cls

    def __call__(cls, *args, **kwargs):
        # A stand-in used as a decorator returns what it decorates; a real
        # subclass of a stand-in constructs normally.
        if cls.__dict__.get("_stand_in", False):
            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
        return super().__call__(*args, **kwargs)

    def __or__(cls, other):
        return cls

    __ror__ = __or__


class _AnyBase(metaclass=_AnyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stand_in(f"{type(self).__qualname__}.{name}")

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _stand_in(f"{type(self).__qualname__}()")()


def _stand_in(qualname):
    return _AnyMeta(qualname.rsplit(".", 1)[-1], (_AnyBase,), {"_stand_in": True})


class _StandInModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _OVERRIDES.get(self.__name__, {}).get(name)
        if value is None:
            value = _stand_in(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


#: Names whose stand-in must be a specific object for the real files to import.
_OVERRIDES = {
    "tensorrt_llm._torch.model_config": {"TConfig": TypeVar("TConfig")},
    "tensorrt_llm._utils": {"mpi_disabled": lambda: False, "local_mpi_rank": lambda: 0},
}


class _StandInLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _StandInModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


class _SpeculativeLoader(importlib.abc.Loader):
    """``SpecDecOneEngineForCausalLM`` over the real ``DecoderModelForCausalLM``."""

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        utils = importlib.import_module("tensorrt_llm._torch.models.modeling_utils")
        model_t, config_t = TypeVar("TModel"), TypeVar("TConfig")

        class SpecDecOneEngineForCausalLM(
            utils.DecoderModelForCausalLM[model_t, config_t], Generic[model_t, config_t]
        ):
            def __init__(self, model, model_config, hidden_size=None, vocab_size=None):
                raise NotImplementedError("patch the constructor in the test")

            def load_draft_weights(self, weights, weight_mapper=None):
                raise NotImplementedError("patch the draft-weight load in the test")

            def external_drafter_mapping(self, model_config):
                return model_config.mapping

        module.SpecDecOneEngineForCausalLM = SpecDecOneEngineForCausalLM


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in _REAL_FILES:
            return importlib.util.spec_from_file_location(name, _REAL_FILES[name])
        if name == _SPECULATIVE:
            return importlib.machinery.ModuleSpec(name, _SpeculativeLoader())
        if name.split(".", 1)[0] in _STUBBED_TOP_LEVEL:
            return importlib.machinery.ModuleSpec(name, _StandInLoader(), is_package=True)
        return None


def _load_with_stand_ins():
    saved = dict(sys.modules)
    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in _STUBBED_TOP_LEVEL:
                del sys.modules[name]
        mapping = importlib.import_module("tensorrt_llm.mapping")
        utils = importlib.import_module("tensorrt_llm._torch.models.modeling_utils")
        builtin = importlib.import_module("tensorrt_llm._torch.models.modeling_musespark_open")
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)
    return builtin, utils, mapping.Mapping


def load():
    """``(modeling_musespark_open, modeling_utils, Mapping)``, see the module docstring."""
    try:
        from tensorrt_llm._torch.models import modeling_musespark_open, modeling_utils
        from tensorrt_llm.mapping import Mapping
    except ImportError:
        return _load_with_stand_ins()
    return modeling_musespark_open, modeling_utils, Mapping
