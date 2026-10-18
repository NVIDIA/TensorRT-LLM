# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executor hooks this target declares (``runtime.make_runtime_hooks``), host-only.

Both parts are always built, in allocation order; the step-input window sets
``step_input_stager`` on exactly the runner, attention metadata and speculative
metadata it was given and clears it at ``end`` and ``discard``, leaving another
runner's objects alone; after its first decline the window declines every later
write; an ineligible step sets nothing and is logged once per reason; the
sampler store writer declines outside its case before touching a staging
block. The pinned staging blocks and the launch kernel are stand-ins here; the
device side is in ``test_modeling_v2_step_commit.py``.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def staging_log(monkeypatch):
    """Replace the pinned staging block and the kernel warm-up with recorders."""
    log = []

    class _Staging:
        def __init__(self, *args, **kwargs):
            self.name = f"staging{sum(1 for e in log if e[0] == 'new')}"
            log.append(("new", self.name))
            self.num = 0

        def reset(self):
            log.append(("reset", self.name))

        def launch(self):
            log.append(("launch", self.name))
            return False

        def room(self, *args):
            return True

    monkeypatch.setattr(runtime._sc, "StepStaging", _Staging)
    monkeypatch.setattr(runtime._sc, "warm_up", lambda: log.append(("warm_up",)))
    return log


def _model(pp_size=1):
    return types.SimpleNamespace(_mapping=types.SimpleNamespace(pp_size=pp_size))


# --------------------------------------------------------------------------
# make_runtime_hooks
# --------------------------------------------------------------------------
def test_the_hooks_declare_both_parts_in_allocation_order(staging_log):
    hooks = runtime.make_runtime_hooks(_model())
    assert isinstance(hooks, RuntimeHooks)
    assert isinstance(hooks.step_input_stager, runtime.StepInputStaging)
    assert isinstance(hooks.sampler_store_writer, runtime.SampleStoreWriter)
    # The prepare block, then the two sample blocks, then the kernel compile.
    assert staging_log == [
        ("new", "staging0"),
        ("new", "staging1"),
        ("new", "staging2"),
        ("warm_up",),
    ]
    assert hooks.step_input_stager.staging.name == "staging0"


def test_the_hooks_read_nothing_but_the_pipeline_size(staging_log):
    hooks = runtime.make_runtime_hooks(types.SimpleNamespace())
    assert hooks.step_input_stager.pp_size == 1
    assert hooks.sampler_store_writer.pp_size == 1


def test_the_parts_carry_the_pipeline_size(staging_log):
    hooks = runtime.make_runtime_hooks(_model(pp_size=2))
    assert hooks.step_input_stager.pp_size == 2
    assert hooks.sampler_store_writer.pp_size == 2


# --------------------------------------------------------------------------
# The step-input window
# --------------------------------------------------------------------------
def _step(pp_size=1, attn_metadata=None, num_context_requests=0):
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec

    t = torch.zeros(1, dtype=torch.int32)
    runner = types.SimpleNamespace(
        _config=types.SimpleNamespace(
            use_mrope=False, enable_attention_dp=False, disable_overlap_scheduler=False
        ),
        guided_decoder=None,
        use_beam_search=False,
        mapping=types.SimpleNamespace(has_cp_helix=lambda: False, pp_size=pp_size),
    )
    if attn_metadata is None:
        attn_metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
    spec_metadata = types.SimpleNamespace()
    args = (
        runner,
        types.SimpleNamespace(num_context_requests=num_context_requests),
        attn_metadata,
        spec_metadata,
        SampleStateTensorsSpec(new_tokens=t, new_tokens_lens=t, next_draft_tokens=t),
    )
    kwargs = dict(graph_replay=True, is_dummy=False, enable_spec_decode=True, promoted=False)
    return args, kwargs


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(message)

    debug = info


def test_the_window_sets_the_stager_on_exactly_the_objects_it_was_given(staging_log):
    staging = runtime.StepInputStaging()
    args, kwargs = _step()
    runner, _, attn, spec, _ = args
    window = staging.begin_if_eligible(*args, **kwargs)
    assert window is not None
    assert runner.step_input_stager is window
    assert attn.step_input_stager is window
    assert spec.step_input_stager is window
    assert ("reset", "staging0") in staging_log

    window.end({})
    assert runner.step_input_stager is None
    assert attn.step_input_stager is None
    assert spec.step_input_stager is None
    assert staging.counts["prepare staged"] == 1
    assert staging_log[-1] == ("launch", "staging0")
    assert window.static_inputs() is None, "no static buffers to write"


def test_a_window_leaves_another_runner_and_its_metadata_alone(staging_log):
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    staging = runtime.StepInputStaging()
    args, kwargs = _step()
    other_runner, _, other_attn, _, _ = _step()[0]
    other_spec = SpecMetadata.__new__(SpecMetadata)
    window = staging.begin_if_eligible(*args, **kwargs)
    assert args[0].step_input_stager is window
    assert "step_input_stager" not in vars(other_runner)
    assert other_attn.step_input_stager is None
    assert other_spec.step_input_stager is None
    assert TrtllmAttentionMetadata.step_input_stager is None
    assert SpecMetadata.step_input_stager is None
    window.end({})
    assert other_attn.step_input_stager is None


def test_after_a_decline_the_window_declines_every_later_write(staging_log):
    staging = runtime.StepInputStaging()
    args, kwargs = _step()
    window = staging.begin_if_eligible(*args, **kwargs)
    host = torch.zeros(2, dtype=torch.int32)
    # A host destination: declined, and the writes recorded so far are launched first.
    assert window.copy(torch.zeros(2, dtype=torch.int32), host) is False
    assert staging_log[-1] == ("launch", "staging0")
    launches = len([e for e in staging_log if e[0] == "launch"])
    # Every later write is declined without being looked at (stand-ins without any of
    # the attributes a recording would read) and without another launch.
    assert window.copy(object(), object()) is False
    assert window.overlap_inputs(object(), **dict.fromkeys(_OVERLAP_KWARGS)) is False
    assert window.block_offsets(object(), object(), [0], 1, 0, 1) is False
    assert len([e for e in staging_log if e[0] == "launch"]) == launches
    assert len(window.declined) == 1
    window.end({})
    assert window.static_inputs() is None, "a declining step's replay copies its inputs"


_OVERLAP_KWARGS = (
    "new_tokens_device",
    "new_tokens_lens_device",
    "next_draft_tokens_device",
    "previous_batch_indices",
    "previous_pos_indices",
    "num_tokens",
    "num_draft_tokens",
    "tokens_per_gen_step",
    "num_extend_requests",
)


def test_a_discarded_window_clears_the_stager_and_launches_nothing(staging_log):
    staging = runtime.StepInputStaging()
    args, kwargs = _step()
    runner, _, attn, spec, _ = args
    window = staging.begin_if_eligible(*args, **kwargs)
    window.discard()
    assert (runner.step_input_stager, attn.step_input_stager, spec.step_input_stager) == (
        None,
        None,
        None,
    )
    assert staging.counts["prepare aborted"] == 1
    assert not [e for e in staging_log if e[0] == "launch"]
    assert window.static_inputs() is None


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda a, k: k.update(graph_replay=False), "no graph replay"),
        (lambda a, k: k.update(is_dummy=True), "dummy step"),
        (lambda a, k: k.update(enable_spec_decode=False), "no speculative decoding"),
        (lambda a, k: k.update(promoted=True), "promoted context rows"),
        (lambda a, k: setattr(a[1], "num_context_requests", 1), "context rows"),
        (lambda a, k: setattr(a[0].mapping, "pp_size", 2), "pipeline parallelism"),
        (lambda a, k: setattr(a[0], "guided_decoder", object()), "guided decoding"),
    ],
)
def test_an_ineligible_step_sets_nothing(staging_log, change, reason):
    staging = runtime.StepInputStaging()
    args, kwargs = _step()
    change(args, kwargs)
    assert staging.ineligible(*args, **kwargs) == reason
    assert staging.begin_if_eligible(*args, **kwargs) is None
    for target in (args[0], args[2], args[3]):
        assert "step_input_stager" not in vars(target)
    assert not [e for e in staging_log if e[0] == "reset"]


def test_the_stager_refuses_its_own_pipeline_size(staging_log):
    args, kwargs = _step()
    assert runtime.StepInputStaging(pp_size=4).ineligible(*args, **kwargs) == (
        "pipeline parallelism"
    )


def test_the_first_decline_per_reason_is_logged_once(staging_log, monkeypatch):
    log = _Log()
    monkeypatch.setattr(runtime, "logger", log)
    staging = runtime.StepInputStaging()
    args, kwargs = _step(attn_metadata=types.SimpleNamespace())
    for _ in range(3):
        assert staging.begin_if_eligible(*args, **kwargs) is None
    (line,) = log.lines
    assert "declined for the first time: attention metadata type" in line
    assert "types.SimpleNamespace" in line
    assert staging.counts["prepare declined: attention metadata type"] == 3


# --------------------------------------------------------------------------
# The sampler store writer declines outside its case
# --------------------------------------------------------------------------
def _sampler(async_worker=False):
    store = types.SimpleNamespace(
        new_tokens=torch.zeros((8, 2, 1), dtype=torch.int32),
        next_new_tokens=torch.zeros((8, 2, 1), dtype=torch.int32),
        new_tokens_lens=torch.zeros((2,), dtype=torch.int32),
        next_draft_tokens=torch.zeros((2, 7), dtype=torch.int32),
    )
    return types.SimpleNamespace(store=store, _async_worker_active=lambda: async_worker)


def _outputs():
    return (
        torch.zeros((1, 8), dtype=torch.int32),
        torch.zeros((1, 8), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((1, 7), dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "pp_size, verify_lens, async_worker, pinned, reason",
    [
        (2, None, False, True, "pipeline parallelism"),
        (1, torch.ones(1, dtype=torch.int32), False, True, "verify_lens"),
        (1, None, True, True, "async worker"),
        (1, None, False, False, "no pinned host memory"),
        (1, None, False, True, "dtypes"),  # host tensors: the launch writes device memory
    ],
)
def test_the_store_writer_declines_before_staging(
    staging_log, monkeypatch, pp_size, verify_lens, async_worker, pinned, reason
):
    monkeypatch.setattr(runtime, "prefer_pinned", lambda: pinned)
    writer = runtime.SampleStoreWriter(pp_size=pp_size)
    sampler = _sampler(async_worker)
    assert writer.store_and_copy_out(sampler, [0], *_outputs(), verify_lens) is None
    assert writer.counts == {f"sample declined: {reason}": 1}
    assert not [e for e in staging_log if e[0] != "new"], "no block touched"
    assert all(int(t.abs().sum()) == 0 for t in vars(sampler.store).values())
