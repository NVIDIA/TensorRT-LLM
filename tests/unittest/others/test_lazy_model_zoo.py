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
"""Guards for the lazily imported model zoo and top-level namespace.

The package surface is loaded via PEP 562 and the PyTorch model zoo resolves
through the static tables in ``_arch_index.py``; registration is an import
side effect that now runs on demand instead of at ``import tensorrt_llm``
time. These tests pin the three contracts that keep that scheme correct:

- ``import tensorrt_llm`` stays thin (no model zoo / visual_gen in a fresh
  process) while first attribute access still resolves and caches.
- The static index stays in sync with the ``@register_auto_model`` /
  ``register_input_processor`` decorators it mirrors (a new model that
  forgets its index entry fails here, not at model-load time in production).
- On-demand registration never overrides an existing (e.g. custom, via
  ``--custom_module_dirs``) registration, and multimodal placeholder lookups
  resolve their provider in a process that never loaded a model.
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_MODELS_DIR = Path(__file__).parents[3] / "tensorrt_llm" / "_torch" / "models"

_SENTINEL = "LAZY-OK"


def _run_fresh(body: str, timeout: int = 300) -> None:
    """Run ``body`` in a fresh interpreter and assert it prints the sentinel.

    Fresh subprocess on purpose: the pytest process has long since imported
    tensorrt_llm (and, through other tests, parts of the model zoo), so
    lazy-import claims are only observable in a new process.
    """
    script = body + f"\nprint({_SENTINEL!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    assert result.returncode == 0 and _SENTINEL in result.stdout, (
        f"fresh-process check failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_import_does_not_load_zoo_or_visual_gen():
    _run_fresh(
        textwrap.dedent("""\
        import sys
        import tensorrt_llm

        loaded = [
            m for m in sys.modules
            if m.startswith("tensorrt_llm._torch.models.modeling_")
            or m == "tensorrt_llm.visual_gen"
            or m.startswith("tensorrt_llm.visual_gen.")
        ]
        assert not loaded, f"import tensorrt_llm eagerly loaded: {loaded}"
        """)
    )


def test_lazy_attribute_access_resolves_and_caches():
    _run_fresh(
        textwrap.dedent("""\
        import tensorrt_llm

        sp = tensorrt_llm.SamplingParams
        assert sp is tensorrt_llm.SamplingParams  # cached in globals()
        assert "SamplingParams" in vars(tensorrt_llm)
        assert "SamplingParams" in dir(tensorrt_llm)

        try:
            tensorrt_llm.definitely_not_an_attribute
        except AttributeError:
            pass
        else:
            raise AssertionError("missing attribute did not raise")
        """)
    )


def test_placeholder_registry_resolves_in_fresh_process():
    # trtllm-bench dataset prep queries the placeholder registry by
    # model_type in a process that never loads a model; the registry must
    # import the provider on demand.
    _run_fresh(
        textwrap.dedent("""\
        from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

        assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid("llama4", "image")
        assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(
            "llama4", "image")
        assert "qwen2_vl" in MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types()
        """)
    )


def _decorated_registrations():
    """AST-scan the modeling files for the registrations the index mirrors."""
    arch_to_modules = {}
    model_type_to_modules = {}
    for path in sorted(_MODELS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "register_auto_model":
                if node.args and isinstance(node.args[0], ast.Constant):
                    arch_to_modules.setdefault(node.args[0].value, set()).add(path.stem)
            elif name in ("register_input_processor", "set_placeholder_metadata"):
                model_type = None
                if name == "register_input_processor":
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        model_type = node.args[1].value
                elif node.args and isinstance(node.args[0], ast.Constant):
                    model_type = node.args[0].value
                for kw in node.keywords:
                    if kw.arg == "model_type" and isinstance(kw.value, ast.Constant):
                        model_type = kw.value.value
                if model_type is not None:
                    model_type_to_modules.setdefault(model_type, set()).add(path.stem)
    return arch_to_modules, model_type_to_modules


def test_arch_index_matches_decorators():
    from tensorrt_llm._torch.models._arch_index import (
        MODEL_ARCH_TO_MODULE,
        MULTIMODAL_MODEL_TYPE_TO_MODULE,
    )

    arch_truth, model_type_truth = _decorated_registrations()

    # Architectures registered through a non-literal decorator argument
    # (e.g. register_auto_model(VilaConfig.model_architecture)); the AST scan
    # cannot see them, so they are exempt from the stale check only.
    dynamic_archs = {"LlavaLlamaModel"}

    missing = set(arch_truth) - set(MODEL_ARCH_TO_MODULE)
    assert not missing, f"architectures missing from _arch_index: {missing}"
    stale = set(MODEL_ARCH_TO_MODULE) - set(arch_truth) - dynamic_archs
    assert not stale, f"stale architectures in _arch_index: {stale}"
    wrong = {
        arch: (MODEL_ARCH_TO_MODULE[arch], arch_truth[arch])
        for arch in MODEL_ARCH_TO_MODULE
        if arch in arch_truth and MODEL_ARCH_TO_MODULE[arch] not in arch_truth[arch]
    }
    assert not wrong, f"index points at the wrong module: {wrong}"

    missing = set(model_type_truth) - set(MULTIMODAL_MODEL_TYPE_TO_MODULE)
    assert not missing, f"model types missing from _arch_index: {missing}"
    stale = set(MULTIMODAL_MODEL_TYPE_TO_MODULE) - set(model_type_truth)
    assert not stale, f"stale model types in _arch_index: {stale}"
    wrong = {
        mt: (MULTIMODAL_MODEL_TYPE_TO_MODULE[mt], model_type_truth[mt])
        for mt in MULTIMODAL_MODEL_TYPE_TO_MODULE
        if MULTIMODAL_MODEL_TYPE_TO_MODULE[mt] not in model_type_truth[mt]
    }
    assert not wrong, f"index points at the wrong module: {wrong}"


def _decorated_draft_model_registrations():
    """AST-scan the modeling files for ``@register_draft_model`` decorators.

    Source scan rather than a registry walk on purpose: the registry is only
    populated by importing a provider, and a built-in builder that lost its
    slot to an external registration would be missing from it entirely, so a
    registry walk would pass while the index is stale.
    """
    mode_to_modules = {}
    for path in sorted(_MODELS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "register_draft_model" or not node.args:
                continue
            arg = node.args[0]
            # ``register_draft_model(SpeculativeDecodingMode.DFLASH)``
            if isinstance(arg, ast.Attribute):
                mode_to_modules.setdefault(arg.attr, set()).add(path.stem)
    return mode_to_modules


def test_spec_mode_index_matches_decorators():
    from tensorrt_llm._torch.models._arch_index import SPEC_MODE_TO_MODULE

    mode_truth = _decorated_draft_model_registrations()

    missing = set(mode_truth) - set(SPEC_MODE_TO_MODULE)
    assert not missing, f"spec modes missing from _arch_index: {missing}"
    stale = set(SPEC_MODE_TO_MODULE) - set(mode_truth)
    assert not stale, f"stale spec modes in _arch_index: {stale}"
    wrong = {
        mode: (SPEC_MODE_TO_MODULE[mode], mode_truth[mode])
        for mode in SPEC_MODE_TO_MODULE
        if SPEC_MODE_TO_MODULE[mode] not in mode_truth[mode]
    }
    assert not wrong, f"index points at the wrong module: {wrong}"


def test_class_index_matches_package_all():
    # MODEL_CLASS_TO_MODULE is the one table with no decorator to mirror: it
    # backs PEP 562 attribute access on the models package. Every name in the
    # package __all__ that is not bound eagerly must have an index entry
    # (otherwise the first `from tensorrt_llm._torch.models import NewModel`
    # fails as a confusing AttributeError), the index must not carry names the
    # package no longer exports, and each mapped module must actually define
    # its class.
    from tensorrt_llm._torch.models._arch_index import MODEL_CLASS_TO_MODULE

    init_tree = ast.parse((_MODELS_DIR / "__init__.py").read_text())
    all_names, eager_names = set(), set()
    for node in init_tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "__all__":
            all_names = {elt.value for elt in node.value.elts}
        elif isinstance(node, ast.ImportFrom):
            eager_names |= {alias.asname or alias.name for alias in node.names}

    missing = all_names - eager_names - set(MODEL_CLASS_TO_MODULE)
    assert not missing, f"__all__ names missing from MODEL_CLASS_TO_MODULE: {missing}"
    stale = set(MODEL_CLASS_TO_MODULE) - all_names
    assert not stale, f"MODEL_CLASS_TO_MODULE entries not exported by __all__: {stale}"

    undefined = {}
    for class_name, module_name in MODEL_CLASS_TO_MODULE.items():
        path = _MODELS_DIR / f"{module_name}.py"
        if not path.exists():
            undefined[class_name] = f"{module_name}: no such module"
            continue
        tree = ast.parse(path.read_text())
        defined = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))} | {
            target.id
            for n in tree.body
            if isinstance(n, ast.Assign)
            for target in n.targets
            if isinstance(target, ast.Name)
        }
        if class_name not in defined:
            undefined[class_name] = f"{module_name}: name not defined at module level"
    assert not undefined, f"index maps classes to modules that do not define them: {undefined}"


def test_models_package_missing_submodule_is_attribute_error():
    # The PEP 562 fallback must translate only "no such submodule" into
    # AttributeError (so hasattr works), same as the top-level package.
    import tensorrt_llm._torch.models as torch_models

    assert not hasattr(torch_models, "modeling_definitely_not_a_model")


def test_resolver_keeps_existing_registration():
    # A registration made by user code (--custom_module_dirs) must win over
    # the built-in module, exactly as it does with the eager import order on
    # main where the zoo loads first and custom code overrides it. Fresh
    # process so the mutated registry and cached provider imports cannot
    # leak into other tests.
    _run_fresh(
        textwrap.dedent("""\
        from tensorrt_llm._torch.models.modeling_utils import (
            MODEL_CLASS_MAPPING, get_registered_model_class)

        arch = "MistralForCausalLM"

        class CustomStub:
            pass

        MODEL_CLASS_MAPPING[arch] = CustomStub
        assert get_registered_model_class(arch) is CustomStub, (
            "resolving an architecture overrode its existing registration")

        # Unknown architectures resolve to None, left to the caller.
        assert get_registered_model_class(
            "DefinitelyNotARegisteredArch") is None
        """)
    )


def test_builtin_decorator_does_not_override_external_registration():
    # The priority guarantee lives in the registry itself: a built-in module
    # may run its decorators after an external registration (direct imports,
    # sibling architectures from the same module), and must not clobber it.
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING, register_auto_model

    arch = "LazyZooTestOnlyArch"
    assert arch not in MODEL_CLASS_MAPPING

    class _External:
        pass

    class _Builtin:
        pass

    _Builtin.__module__ = "tensorrt_llm._torch.models.modeling_fake"

    try:
        register_auto_model(arch)(_External)
        register_auto_model(arch)(_Builtin)
        assert MODEL_CLASS_MAPPING[arch] is _External, (
            "built-in decorator overrode an external registration"
        )

        # The other direction stays last-wins: external code registering
        # over a built-in is exactly the --custom_module_dirs use case.
        del MODEL_CLASS_MAPPING[arch]
        register_auto_model(arch)(_Builtin)
        register_auto_model(arch)(_External)
        assert MODEL_CLASS_MAPPING[arch] is _External
    finally:
        MODEL_CLASS_MAPPING.pop(arch, None)


def test_custom_registration_survives_direct_provider_import():
    # Regression for the full production path: a custom implementation is
    # registered, then some other code path imports the built-in provider
    # directly (e.g. model_loader's post-transform profile registry imports
    # modeling_llama) -- the custom registration must survive the built-in
    # module's decorators. Fresh process so modeling_llama is genuinely not
    # imported yet when the custom registration happens.
    _run_fresh(
        textwrap.dedent("""\
        import importlib
        from tensorrt_llm._torch.models.modeling_utils import (
            MODEL_CLASS_MAPPING, register_auto_model)

        @register_auto_model("LlamaForCausalLM")
        class CustomLlama:
            pass

        importlib.import_module(
            "tensorrt_llm._torch.models.modeling_llama")
        assert MODEL_CLASS_MAPPING["LlamaForCausalLM"] is CustomLlama, (
            "direct import of the built-in provider overrode the custom "
            "registration")
        """)
    )


def test_external_multimodal_override_keeps_provider_importable():
    # An external override of a multimodal architecture must not break the
    # built-in provider's import: register_vision_encoder used to locate the
    # freshly decorated class in MODEL_CLASS_MAPPING by identity and raise
    # when the external registration had won the slot. The built-in vision
    # encoder still fills the empty sibling slot, like the eager import
    # order on main.
    _run_fresh(
        textwrap.dedent("""\
        import importlib
        from tensorrt_llm._torch.models.modeling_utils import (
            MODEL_CLASS_MAPPING, MODEL_CLASS_VISION_ENCODER_MAPPING,
            register_auto_model)

        arch = "Qwen3VLForConditionalGeneration"

        @register_auto_model(arch)
        class CustomQwen3VL:
            pass

        importlib.import_module(
            "tensorrt_llm._torch.models.modeling_qwen3vl")

        assert MODEL_CLASS_MAPPING[arch] is CustomQwen3VL, (
            "built-in provider import overrode the external registration")
        assert MODEL_CLASS_VISION_ENCODER_MAPPING.get(arch) is not None, (
            "built-in vision encoder did not fill the empty sibling slot")
        """)
    )


def test_external_sibling_registrations_not_clobbered():
    # When the external implementation brings its own vision encoder and
    # placeholder metadata, a later built-in import must not overwrite them.
    _run_fresh(
        textwrap.dedent("""\
        import importlib
        from tensorrt_llm._torch.models.modeling_utils import (
            MODEL_CLASS_VISION_ENCODER_MAPPING, register_auto_model,
            register_vision_encoder)
        from tensorrt_llm.inputs.registry import (
            MULTIMODAL_PLACEHOLDER_REGISTRY, MultimodalPlaceholderMetadata)

        arch = "Qwen3VLForConditionalGeneration"

        class CustomEncoder:
            pass

        @register_vision_encoder(CustomEncoder)
        @register_auto_model(arch)
        class CustomQwen3VL:
            pass

        custom_metadata = MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<custom_image>"})
        MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
            "qwen3_vl", custom_metadata, registrant_module=__name__)

        importlib.import_module(
            "tensorrt_llm._torch.models.modeling_qwen3vl")

        assert MODEL_CLASS_VISION_ENCODER_MAPPING[arch][0] is CustomEncoder, (
            "built-in import clobbered the external vision encoder")
        assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata(
            "qwen3_vl") is custom_metadata, (
            "built-in import clobbered the external placeholder metadata")
        """)
    )
