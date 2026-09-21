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

import importlib.util
import pathlib
import sys
import types

import pytest
from cutlass import pipeline
from cutlass.pipeline import sm100


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "factory_kind", ["missing", "missing_class", "inherited_sm90", "non_callable", "supported"]
)
def test_custom_pipeline_sync_factory_import(
    monkeypatch: pytest.MonkeyPatch, factory_kind: str
) -> None:
    """Reject unsupported factories at import while preserving the SM100 factory."""
    if factory_kind == "missing":
        # Deleting the override alone would expose the inherited SM90 factory.
        monkeypatch.setattr(sm100, "PipelineTmaUmma", types.SimpleNamespace())
    elif factory_kind == "missing_class":
        monkeypatch.delattr(sm100, "PipelineTmaUmma")
    elif factory_kind == "inherited_sm90":
        monkeypatch.setattr(
            sm100.PipelineTmaUmma,
            "_make_sync_object",
            staticmethod(pipeline.PipelineAsync._make_sync_object),
        )
    elif factory_kind == "non_callable":
        monkeypatch.setattr(sm100.PipelineTmaUmma, "_make_sync_object", None)

    source = (
        pathlib.Path(__file__).resolve().parents[4]
        / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/custom_pipeline.py"
    )
    # Execute a fresh module without modifying the cached production module or
    # importing unrelated TensorRT-LLM dependencies. Dataclasses need sys.modules.
    spec = importlib.util.spec_from_file_location("_custom_pipeline_import_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    if factory_kind == "supported":
        spec.loader.exec_module(module)
        assert callable(module._sm100_make_sync)
        assert module._sm100_make_sync is sm100.PipelineTmaUmma._make_sync_object
        assert module._sm100_make_sync is not pipeline.PipelineAsync._make_sync_object
    else:
        with pytest.raises(ImportError) as exc_info:
            spec.loader.exec_module(module)
        message = str(exc_info.value)
        assert "cutlass.pipeline.sm100.PipelineTmaUmma._make_sync_object" in message
        assert "PipelineOp.TCGen05Mma support" in message
        assert "Install the CuTe DSL version specified" in message
        assert "requirements.txt" in message
