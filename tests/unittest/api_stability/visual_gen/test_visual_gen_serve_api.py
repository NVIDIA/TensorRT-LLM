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
"""CI gate for the VisualGen request and response models of ``trtllm-serve``.

Runs the checks of ``tests/unittest/api_stability/test_serve_api.py`` on the
models pinned in ``references/trtllm_serve_api.yaml`` that are defined in
``tensorrt_llm.serve.api.visual_gen.protocol``, and checks that
``tensorrt_llm.serve.openai_protocol`` keeps re-exporting them lazily under
their old names.

The file lives under a ``visual_gen/`` path so that CI selection for
VisualGen-only pull requests still runs it: the directory entry
``unittest/api_stability`` is not kept for those. The reference YAML is
unchanged and stays with the API review committee, which also owns this file.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and the stage collects nothing from the file.
pytestmark = pytest.mark.cpu_only

_VISUAL_GEN_PROTOCOL_MODULE = "tensorrt_llm.serve.api.visual_gen.protocol"

# The VisualGen models that `tensorrt_llm.serve.openai_protocol` re-exports.
_VISUAL_GEN_MODEL_NAMES = (
    "ImageGenerationRequest",
    "ImageEditRequest",
    "ImageObject",
    "ImageGenerationResponse",
    "MediaReferenceItem",
    "VideoGenerationRequest",
    "VideoJob",
    "VideoJobList",
)

# The VisualGen models the reference YAML pins; a model pinned later is covered
# automatically because the fixture selects by defining module.
_PINNED_VISUAL_GEN_MODELS = frozenset(
    {
        "ImageGenerationRequest",
        "ImageGenerationResponse",
        "MediaReferenceItem",
        "VideoGenerationRequest",
    }
)


def _load_serve_api_checks():
    """Load ``test_serve_api.py`` by path, without pytest collecting it twice.

    The module is kept in a private name: a public ``TestServeAPIStability``
    in this module's namespace would be collected a second time, with the
    whole reference instead of the VisualGen part of it.
    """
    path = Path(__file__).resolve().parents[1] / "test_serve_api.py"
    spec = importlib.util.spec_from_file_location("trtllm_serve_api_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_serve_api = _load_serve_api_checks()


class TestVisualGenServeAPIStability(_serve_api.TestServeAPIStability):
    """The serve API gate, restricted to the pinned VisualGen models."""

    @pytest.fixture(scope="class")
    def reference(self) -> dict[str, Any]:
        """The reference YAML with only the models defined in the VisualGen package.

        Every pinned model is resolved through ``tensorrt_llm.serve.openai_protocol``,
        as the full gate does; a model counts as VisualGen when the class it resolves
        to is defined in ``tensorrt_llm.serve.api.visual_gen.protocol``.
        """
        full = _serve_api._load_reference()
        facade = importlib.import_module(_serve_api._PROTOCOL_MODULE)
        models: dict[str, Any] = {}
        for model_name, model_spec in (full.get("models", {}) or {}).items():
            cls = getattr(facade, model_name, None)
            if cls is not None and cls.__module__ == _VISUAL_GEN_PROTOCOL_MODULE:
                models[model_name] = model_spec
        missing = sorted(_PINNED_VISUAL_GEN_MODELS - set(models))
        assert not missing, (
            f"trtllm_serve_api.yaml pins VisualGen models {missing} but they no longer "
            f"resolve to {_VISUAL_GEN_PROTOCOL_MODULE} through {_serve_api._PROTOCOL_MODULE}. "
            "Keep them in serve/api/visual_gen/protocol.py and in the lazy re-export of "
            "openai_protocol; changes to the YAML need the API review committee."
        )
        return {**full, "models": models}


def _run_in_fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a new interpreter and return the JSON it prints last."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestVisualGenProtocolLazyReexport:
    """``openai_protocol`` keeps the VisualGen names without importing their module."""

    def test_importing_openai_protocol_does_not_load_visual_gen(self) -> None:
        out = _run_in_fresh_interpreter(
            f"""
            import json, sys
            import tensorrt_llm.serve.openai_protocol as protocol

            target = {_VISUAL_GEN_PROTOCOL_MODULE!r}
            before = target in sys.modules
            importers = []
            if before:
                loaded = sys.modules[target]
                for name, module in list(sys.modules.items()):
                    if module is loaded or module is None:
                        continue
                    try:
                        values = list(vars(module).values())
                    except TypeError:
                        continue
                    if any(v is loaded or getattr(v, "__module__", None) == target for v in values):
                        importers.append(name)
            protocol.VideoJob
            after = target in sys.modules
            print(json.dumps({{"before": before, "after": after, "importers": sorted(importers)}}))
            """
        )
        assert out["before"] is False, (
            f"importing tensorrt_llm.serve.openai_protocol loaded {_VISUAL_GEN_PROTOCOL_MODULE} "
            f"eagerly; the modules holding its names are {out['importers']}. The VisualGen "
            "models are re-exported lazily through the module __getattr__ of openai_protocol; "
            "do not import them at module level there."
        )
        assert out["after"] is True, (
            "reading openai_protocol.VideoJob did not import "
            f"{_VISUAL_GEN_PROTOCOL_MODULE}; the lazy re-export no longer resolves the name there."
        )

    def test_reexported_names_are_listed_and_unknown_names_raise(self) -> None:
        protocol = importlib.import_module(_serve_api._PROTOCOL_MODULE)
        listed = set(dir(protocol))
        not_listed = [name for name in _VISUAL_GEN_MODEL_NAMES if name not in listed]
        assert not not_listed, (
            f"dir(tensorrt_llm.serve.openai_protocol) lacks {not_listed}; add the names to the "
            "lazy re-export table of openai_protocol."
        )
        with pytest.raises(AttributeError):
            getattr(protocol, "NotAVisualGenModel")
