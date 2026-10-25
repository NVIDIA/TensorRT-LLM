# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the engine reads the step-loop hooks a model declares (``runtime_hooks_of``)."""

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    MsaRuntimePolicy,
    RuntimeHooks,
    runtime_hooks_of,
)

pytestmark = pytest.mark.cpu_only


class _PlainModel(torch.nn.Module):
    """A model that declares no hooks, like every model outside the opt-in."""


class _ClassHooksModel(torch.nn.Module):
    runtime_hooks = RuntimeHooks(loop_on_execution_stream=True, skip_greedy_rng_upload=True)


def test_model_without_hooks_gets_the_all_off_default():
    hooks = runtime_hooks_of(_PlainModel())

    assert hooks is DEFAULT_RUNTIME_HOOKS
    assert hooks == RuntimeHooks()
    assert not hooks.loop_on_execution_stream
    assert not hooks.merge_tuned_tactics_across_ranks
    assert not hooks.skip_greedy_rng_upload
    assert hooks.spec_step_fold is None
    assert hooks.msa == MsaRuntimePolicy()
    assert not hooks.msa.short_sequence_identity
    assert hooks.msa.kv_update is None
    assert not hooks.msa.stage_tables_on_host
    assert not hooks.msa.reuse_pool_views


def test_hooks_declared_as_none_mean_the_default():
    model = _PlainModel()
    model.runtime_hooks = None

    assert runtime_hooks_of(model) is DEFAULT_RUNTIME_HOOKS


def test_class_attribute_hooks_are_returned_as_declared():
    model = _ClassHooksModel()

    assert runtime_hooks_of(model) is _ClassHooksModel.runtime_hooks


def test_instance_attribute_hooks_are_returned_as_declared():
    hooks = RuntimeHooks(
        loop_on_execution_stream=True,
        msa=MsaRuntimePolicy(short_sequence_identity=True, reuse_pool_views=True),
    )
    model = _PlainModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


def test_instance_attribute_overrides_class_attribute():
    hooks = RuntimeHooks(skip_greedy_rng_upload=True)
    model = _ClassHooksModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


@pytest.mark.parametrize(
    "declared",
    [True, {"loop_on_execution_stream": True}, MsaRuntimePolicy(), "RuntimeHooks()"],
    ids=["bool", "dict", "msa_policy", "str"],
)
def test_hooks_of_the_wrong_type_raise(declared):
    model = _PlainModel()
    model.runtime_hooks = declared

    with pytest.raises(TypeError, match=r"_PlainModel\.runtime_hooks must be a RuntimeHooks"):
        runtime_hooks_of(model)


def test_hooks_are_immutable_once_declared():
    hooks = RuntimeHooks()

    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.loop_on_execution_stream = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.msa.reuse_pool_views = True
