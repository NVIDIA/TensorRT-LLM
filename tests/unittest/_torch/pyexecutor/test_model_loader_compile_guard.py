# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reload() guard against loading into a torch.compile wrapper.

Models that inherit the generic ``DecoderModelForCausalLM.load_weights`` strip
the wrapper's ``_orig_mod`` path component themselves and may be reloaded
wrapped; a model with its own ``load_weights`` walk must still be unwrapped
first (or opt in) and fails loudly otherwise.
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.model_loader import (
    _check_compile_wrapper_before_reload,
    _load_weights_strips_compile_wrapper,
)


def _wrapped_instance(cls):
    # Skip the heavy constructors: only the module tree and the class matter.
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.model = torch.compile(nn.Linear(2, 2), backend="eager")
    assert any("_orig_mod" in n for n, _ in model.named_parameters())
    return model


class _Generic(DecoderModelForCausalLM):
    pass


class _CustomWalk(DecoderModelForCausalLM):
    def load_weights(self, weights, weight_mapper=None, **kwargs):  # noqa: ARG002
        pass


class _Delegating(_CustomWalk):
    load_weights_strips_compile_wrapper = True


def test_generic_loader_is_wrapper_tolerant():
    model = _wrapped_instance(_Generic)
    assert _load_weights_strips_compile_wrapper(model)
    _check_compile_wrapper_before_reload(model)


def test_custom_loader_still_guarded(monkeypatch):
    monkeypatch.delenv("TLLM_REFIT_SKIP_WRAPPER_CHECK", raising=False)
    model = _wrapped_instance(_CustomWalk)
    assert not _load_weights_strips_compile_wrapper(model)
    with pytest.raises(RuntimeError, match="_orig_mod"):
        _check_compile_wrapper_before_reload(model)
    # Unwrapped, the same model passes.
    model.model = model.model._orig_mod
    _check_compile_wrapper_before_reload(model)


def test_opt_in_and_escape_hatch(monkeypatch):
    _check_compile_wrapper_before_reload(_wrapped_instance(_Delegating))
    monkeypatch.setenv("TLLM_REFIT_SKIP_WRAPPER_CHECK", "1")
    _check_compile_wrapper_before_reload(_wrapped_instance(_CustomWalk))
