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
"""Dispatch contract of the draft-model builder registry.

``get_draft_model`` used to be one if/elif chain whose *branch order* carried
unwritten rules. Registry dispatch has no inherent order, so what the chain
implied is pinned here explicitly — breaking it silently swaps the draft model
for ``AutoModelForCausalLM``, which no accuracy test would attribute back to
this function.

The external-draft pre-check runs ahead of the registry without a mode guard,
which is only safe because ``uses_external_draft_model`` implies
``is_mtp_one_model()``. That mutual exclusion is an invariant of ``llm_args``, not
of this module, so it is asserted here rather than assumed.

Everything here asserts *which builder is selected*, never the object it
builds: constructing a real drafter needs GPUs and checkpoints, and the
selection is the whole contract of this layer.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models import modeling_speculative, modeling_utils
from tensorrt_llm._torch.models._arch_index import SPEC_MODE_TO_MODULE
from tensorrt_llm._torch.models.modeling_utils import (
    _REGISTERED_SPEC_MODES_ATTR,
    DRAFT_MODEL_BUILDER_MAPPING,
    get_registered_draft_model_builder,
    register_draft_model,
)
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

_EXTERNAL_DRAFT_SENTINEL = object()
_BUILDER_SENTINEL = object()


class _StubAutoModel:
    """Stands in for ``AutoModelForCausalLM`` on the external-draft path."""

    @staticmethod
    def from_config(draft_config):
        return _EXTERNAL_DRAFT_SENTINEL


def _model_config(mode, *, uses_external_draft_model=False, eagle3_model_arch="llama3"):
    """Minimal duck-typed ``ModelConfig`` for dispatch-only assertions."""
    spec_config = SimpleNamespace(
        spec_dec_mode=mode,
        uses_external_draft_model=uses_external_draft_model,
        eagle3_model_arch=eagle3_model_arch,
    )
    return SimpleNamespace(
        spec_config=spec_config,
        pretrained_config=SimpleNamespace(num_hidden_layers=4),
    )


def _stub_builder(monkeypatch, mode):
    """Replace ``mode``'s registered builder with a sentinel-returning stub."""
    monkeypatch.setitem(
        DRAFT_MODEL_BUILDER_MAPPING, mode, lambda *args, **kwargs: _BUILDER_SENTINEL
    )


def test_external_draft_model_bypasses_the_registry(monkeypatch):
    # An external draft model is loaded from its own checkpoint, so the
    # pre-check must short-circuit before the registry is consulted at all.
    monkeypatch.setattr(modeling_speculative, "AutoModelForCausalLM", _StubAutoModel)
    monkeypatch.setattr(
        modeling_speculative,
        "get_registered_draft_model_builder",
        lambda mode: pytest.fail(f"registry consulted for {mode.name} under external draft"),
    )

    result = modeling_speculative.get_draft_model(
        _model_config(SpeculativeDecodingMode.MTP, uses_external_draft_model=True),
        draft_config=object(),
        lm_head=None,
        model=None,
    )

    assert result is _EXTERNAL_DRAFT_SENTINEL


def test_eagle3_is_unaffected_by_the_external_draft_flag(monkeypatch):
    # `uses_external_draft_model` implies `is_mtp_one_model()`, so it can never
    # be true for EAGLE3. This pins the mutual exclusion that lets the pre-check run
    # without a mode guard: were the property ever widened, EAGLE3 would start
    # building an AutoModel drafter and this test would catch it.
    monkeypatch.setattr(modeling_speculative, "AutoModelForCausalLM", _StubAutoModel)
    _stub_builder(monkeypatch, SpeculativeDecodingMode.EAGLE3_ONE_MODEL)

    result = modeling_speculative.get_draft_model(
        _model_config(SpeculativeDecodingMode.EAGLE3_ONE_MODEL, uses_external_draft_model=True),
        draft_config=object(),
        lm_head=None,
        model=None,
    )

    assert result is _BUILDER_SENTINEL, "external-draft flag hijacked the EAGLE3 builder"


def test_external_draft_model_without_draft_config_raises(monkeypatch):
    monkeypatch.setattr(modeling_speculative, "AutoModelForCausalLM", _StubAutoModel)

    with pytest.raises(ValueError, match="requires its model config"):
        modeling_speculative.get_draft_model(
            _model_config(SpeculativeDecodingMode.MTP, uses_external_draft_model=True),
            draft_config=None,
            lm_head=None,
            model=None,
        )


def test_unregistered_mode_raises_not_implemented():
    # NGRAM is a drafter-loop mode with no one-engine draft model, so it is
    # absent from both the index and the registry.
    assert SpeculativeDecodingMode.NGRAM.name not in SPEC_MODE_TO_MODULE

    with pytest.raises(NotImplementedError, match="does not support speculative decoding mode"):
        modeling_speculative.get_draft_model(
            _model_config(SpeculativeDecodingMode.NGRAM),
            draft_config=object(),
            lm_head=None,
            model=None,
        )


def test_every_indexed_mode_resolves_to_a_declaring_builder():
    # Index -> decorator direction: each indexed mode must resolve through the
    # single entry point, and the builder must itself declare that mode. The
    # declaration is read off the function, never by scanning the mapping by
    # identity (a built-in overridden externally keeps the attribute but loses
    # its slot).
    for mode_name in SPEC_MODE_TO_MODULE:
        mode = getattr(SpeculativeDecodingMode, mode_name, None)
        assert mode is not None, f"{mode_name} is not a SpeculativeDecodingMode member"
        builder = get_registered_draft_model_builder(mode)
        assert builder is not None, f"no builder resolved for {mode_name}"
        assert mode in getattr(builder, _REGISTERED_SPEC_MODES_ATTR, set()), (
            f"{builder.__module__}.{builder.__qualname__} is registered for "
            f"{mode_name} but does not declare it"
        )


def test_no_builder_declares_a_mode_missing_from_the_index():
    # Decorator -> index direction: importing every indexed provider and
    # walking its builders catches a mode added to an already-indexed module
    # without its index entry. (A brand-new provider module is caught by the
    # AST scan in tests/unittest/others/test_lazy_model_zoo.py, which needs no
    # import and therefore sees modules this loop would never load.)
    import importlib

    declared = set()
    for module_name in set(SPEC_MODE_TO_MODULE.values()):
        module = importlib.import_module(f"tensorrt_llm._torch.models.{module_name}")
        for attr in vars(module).values():
            declared |= getattr(attr, _REGISTERED_SPEC_MODES_ATTR, set())

    missing = {mode.name for mode in declared} - set(SPEC_MODE_TO_MODULE)
    assert not missing, f"builders declare modes missing from _arch_index: {missing}"


def test_builtin_builder_does_not_override_external_registration():
    # Under lazy loading a built-in module may run its decorators *after* an
    # external registration (e.g. --custom_module_dirs), so built-ins only fill
    # empty slots. The reverse direction stays last-wins.
    mode = SpeculativeDecodingMode.NGRAM
    assert mode not in DRAFT_MODEL_BUILDER_MAPPING

    def external(model_config, draft_config, lm_head, model):
        return "external"

    def builtin(model_config, draft_config, lm_head, model):
        return "builtin"

    builtin.__module__ = "tensorrt_llm._torch.models.modeling_fake"

    try:
        register_draft_model(mode)(external)
        register_draft_model(mode)(builtin)
        assert DRAFT_MODEL_BUILDER_MAPPING[mode] is external, (
            "built-in builder overrode an external registration"
        )

        del DRAFT_MODEL_BUILDER_MAPPING[mode]
        register_draft_model(mode)(builtin)
        register_draft_model(mode)(external)
        assert DRAFT_MODEL_BUILDER_MAPPING[mode] is external
    finally:
        DRAFT_MODEL_BUILDER_MAPPING.pop(mode, None)


def test_stacked_decorators_share_one_builder():
    # Vanilla MTP and MTP_EAGLE_ONE_MODEL are one branch in
    # SpeculativeDecodingMode.is_mtp_one_model(); the registry expresses that
    # as two keys pointing at the same function.
    mtp = get_registered_draft_model_builder(SpeculativeDecodingMode.MTP)
    mtp_eagle_one = get_registered_draft_model_builder(SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL)

    assert mtp is mtp_eagle_one
    declared = getattr(mtp, _REGISTERED_SPEC_MODES_ATTR, set())
    assert {SpeculativeDecodingMode.MTP, SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL} <= declared


def test_registry_module_placement_matches_index():
    # Builders live next to the draft model they construct, never in the
    # factory file: that is what keeps get_draft_model free of concrete
    # imports (and what removed the DSpark lazy import).
    builder = get_registered_draft_model_builder(SpeculativeDecodingMode.DSPARK)
    assert builder.__module__ == "tensorrt_llm._torch.models.modeling_dspark"
    assert modeling_utils.DRAFT_MODEL_BUILDER_MAPPING is DRAFT_MODEL_BUILDER_MAPPING
