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
    runtime_hooks = RuntimeHooks(pack_step_inputs=True, skip_unchanged_h2d=True)


def test_model_without_hooks_gets_the_all_off_default():
    hooks = runtime_hooks_of(_PlainModel())

    assert hooks is DEFAULT_RUNTIME_HOOKS
    assert hooks == RuntimeHooks()
    assert not hooks.pack_step_inputs
    assert not hooks.skip_unchanged_h2d
    assert not hooks.loop_on_execution_stream
    assert not hooks.merge_tuned_tactics_across_ranks
    assert hooks.first_content_emit_iters == 1
    assert not hooks.mxfp8_pinned_tactics
    assert not hooks.late_compile_detect
    assert not hooks.skip_greedy_rng_upload
    assert not hooks.pack_spec_rng_uploads
    assert hooks.spec_step_fold is None
    assert not hooks.sampler_d2h_side_stream
    assert not hooks.spec_small_h2d
    assert not hooks.iteration_timing_one_event
    assert not hooks.pack_parallel_scatter
    assert not hooks.pack_in_graph
    assert hooks.msa == MsaRuntimePolicy()
    assert not hooks.msa.short_sequence_identity
    assert hooks.msa.kv_update is None
    assert not hooks.msa.skip_unchanged_block_offsets
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
    [True, {"pack_step_inputs": True}, MsaRuntimePolicy(), "RuntimeHooks()"],
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
        hooks.pack_step_inputs = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.msa.reuse_pool_views = True


@pytest.mark.parametrize(
    "hooks, skips",
    [
        (RuntimeHooks(), False),
        (RuntimeHooks(pack_step_inputs=True, skip_greedy_rng_upload=True), False),
        (RuntimeHooks(skip_unchanged_h2d=True), True),
        (RuntimeHooks(msa=MsaRuntimePolicy(skip_unchanged_block_offsets=True)), True),
    ],
    ids=["default", "no_skip_of_metadata", "h2d_skip", "msa_block_offset_skip"],
)
def test_every_metadata_upload_skip_asks_for_private_metadata_buffers(hooks, skips):
    assert hooks.skips_unchanged_metadata_uploads is skips
