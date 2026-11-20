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
"""Tests of input processors registered by architecture name.

An architecture that no model class of the zoo implements, such as one only a modeling_v2
target serves, registers its processor and prompt placeholders with
`register_architecture_input_processor`. `create_input_processor` and the placeholder lookups
find them after importing the modeling_v2 families, which register theirs on import.
"""

import subprocess
import sys
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.model_config as model_config_mod
import tensorrt_llm._torch.models as models_mod
import tensorrt_llm.inputs.registry as registry_mod
from tensorrt_llm._errors import UnsupportedError
from tensorrt_llm.inputs.registry import (
    INPUT_PROCESSOR_REGISTRY,
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    DefaultInputProcessor,
    MultimodalPlaceholderMetadata,
    create_input_processor,
    register_architecture_input_processor,
)

pytestmark = pytest.mark.cpu_only

_ARCHITECTURE = "FakeForConditionalGeneration"
_MODEL_TYPE = "fake_architecture_mm"
_METADATA = MultimodalPlaceholderMetadata(placeholder_map={"image": "<fake_image>"})


class _ModelClass:
    """A model class of the zoo, as `get_model_architecture` resolves one."""


class _RecordingInputProcessor:
    """An input processor that records its constructor arguments."""

    def __init__(self, model_path, config, tokenizer, **kwargs) -> None:
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        self.kwargs = kwargs


class _ArchitectureInputProcessor(_RecordingInputProcessor):
    """The processor registered for `_ARCHITECTURE` by name."""


@pytest.fixture
def registries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Empty registration tables, a checkpoint config of `_ARCHITECTURE`, and family imports.

    Each import of the modeling_v2 families is recorded and imports nothing here.
    """
    monkeypatch.setattr(INPUT_PROCESSOR_REGISTRY, "_input_processors_cls_by_architecture", {})
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY, "_multimodal_placeholder_by_model_type", {}
    )
    imports: list[str] = []
    monkeypatch.setattr(
        registry_mod, "_import_modeling_v2_families", lambda: imports.append("families")
    )
    config = SimpleNamespace(architectures=[_ARCHITECTURE])
    monkeypatch.setattr(
        model_config_mod.ModelConfig,
        "from_pretrained",
        classmethod(lambda _cls, *_args, **_kwargs: SimpleNamespace(pretrained_config=config)),
    )
    monkeypatch.setattr(model_config_mod, "hf_remote_code_lock", nullcontext)
    return imports


def _no_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_model_architecture(config):
        raise UnsupportedError(f"Unknown model architecture: {config.architectures[0]}")

    monkeypatch.setattr(models_mod, "get_model_architecture", get_model_architecture)


def _register() -> None:
    register_architecture_input_processor(
        _ARCHITECTURE, model_type=_MODEL_TYPE, placeholder_metadata=_METADATA
    )(_ArchitectureInputProcessor)


def test_the_registration_records_the_processor_and_its_placeholders(registries):
    _register()
    table = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_architecture
    assert table == {_ARCHITECTURE: _ArchitectureInputProcessor}
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata(_MODEL_TYPE) is _METADATA
    assert _ArchitectureInputProcessor._registered_model_type == _MODEL_TYPE


def test_the_registration_requires_placeholder_metadata():
    with pytest.raises(ValueError, match="placeholder_metadata"):
        register_architecture_input_processor(
            _ARCHITECTURE, model_type=_MODEL_TYPE, placeholder_metadata=None
        )


def test_an_architecture_without_a_model_class_gets_its_registered_processor(
    registries, monkeypatch
):
    _no_model_class(monkeypatch)
    _register()
    tokenizer = object()

    processor = create_input_processor("/ckpt", tokenizer)

    assert type(processor) is _ArchitectureInputProcessor
    assert processor.model_path == "/ckpt"
    assert processor.tokenizer is tokenizer
    assert registries == ["families"]


def test_the_lookup_imports_the_families_that_register_the_processor(registries, monkeypatch):
    _no_model_class(monkeypatch)
    monkeypatch.setattr(registry_mod, "_import_modeling_v2_families", _register)

    assert type(create_input_processor("/ckpt", object())) is _ArchitectureInputProcessor


def test_an_architecture_registered_by_no_one_gets_the_default_processor(registries, monkeypatch):
    _no_model_class(monkeypatch)
    tokenizer = object()

    processor = create_input_processor("/ckpt", tokenizer)

    assert type(processor) is DefaultInputProcessor
    assert processor.tokenizer is tokenizer


def test_a_model_class_processor_takes_precedence(registries, monkeypatch):
    """With a model class of the zoo, its registration decides.

    The modeling_v2 families are not imported then.
    """
    _register()
    monkeypatch.setattr(
        models_mod, "get_model_architecture", lambda config: (_ModelClass, _ARCHITECTURE)
    )
    monkeypatch.setitem(
        INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type,
        _ModelClass,
        _RecordingInputProcessor,
    )

    assert type(create_input_processor("/ckpt", object())) is _RecordingInputProcessor
    assert registries == []


def test_a_placeholder_lookup_imports_the_families_for_a_model_type_the_zoo_lacks(
    registries, monkeypatch
):
    monkeypatch.setattr(registry_mod, "_import_modeling_v2_families", _register)

    assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(_MODEL_TYPE, "image")
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(_MODEL_TYPE, "image") == "<fake_image>"


def test_listing_the_model_types_imports_the_families(registries, monkeypatch):
    monkeypatch.setattr(registry_mod, "_import_modeling_v2_families", _register)
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY, "_ensure_provider_imported", lambda _: None
    )

    assert _MODEL_TYPE in MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types()


def test_importing_the_families_imports_no_model_of_the_zoo():
    """Importing the modeling_v2 families loads no model of the zoo.

    The placeholder lookups import the families for any model type the zoo lacks, text-only ones
    included. The zoo's own index and base modules may load: every placeholder lookup reads the
    index.
    """
    script = textwrap.dedent("""\
        import sys
        from tensorrt_llm.inputs import registry
        from tensorrt_llm._torch.models import _arch_index as index

        registry._import_modeling_v2_families()
        models = {
            f"tensorrt_llm._torch.models.{name}"
            for table in (index.MODEL_ARCH_TO_MODULE, index.MULTIMODAL_MODEL_TYPE_TO_MODULE)
            for name in table.values()
        }
        loaded = sorted(models & set(sys.modules))
        assert not loaded, f"importing the modeling_v2 families loaded {loaded}"
        print("FAMILIES-OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "FAMILIES-OK" in result.stdout, result.stderr
