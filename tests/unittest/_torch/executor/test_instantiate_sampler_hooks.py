# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""instantiate_sampler hands a one-model speculative sampler the model's host-tail hooks
(RuntimeHooks.sampler_d2h_side_stream / spec_small_h2d) and logs the D2H side-stream engagement line only when it
is on."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_D2H_LINE = "[M3 host] spec sampler store copies on the D2H side stream"


class _Fold:
    """A SpecStepFold stand-in; binding hands it the sampler's store."""

    store = None

    def bind_sampler_store(self, store):
        self.store = store


def _instantiate(monkeypatch, hooks, sampler):
    lines = []
    monkeypatch.setattr(_util, "confidential_compute_enabled", lambda: False)
    monkeypatch.setattr(_util, "resolve_max_num_sequences", lambda *args, **kwargs: 4)
    monkeypatch.setattr(_util, "get_spec_decoder", lambda sampler_args, spec_config: sampler)
    monkeypatch.setattr(_util.logger, "info", lines.append)

    class _Model(torch.nn.Module):
        runtime_hooks = hooks

    engine = SimpleNamespace(
        max_seq_len=64,
        model=_Model(),
        spec_config=SimpleNamespace(
            spec_dec_mode=SimpleNamespace(has_spec_decoder=lambda: True), max_draft_len=3
        ),
    )
    llm_args = SimpleNamespace(
        sampler_force_async_worker=False,
        disable_overlap_scheduler=False,
        enable_speculative_beam_history_d2h=False,
    )
    result = _util.instantiate_sampler(
        engine,
        llm_args,
        Mapping(),
        max_batch_size=4,
        max_beam_width=1,
        mm_encoder_only=False,
        speculative_config=None,
        max_num_sequences=4,
    )
    assert result is sampler
    return lines


def _spec_sampler():
    sampler = object.__new__(SpecSampler)
    sampler.store = object()
    return sampler


def test_default_hooks_leave_the_sampler_stock_and_log_nothing(monkeypatch):
    sampler = _spec_sampler()

    lines = _instantiate(monkeypatch, RuntimeHooks(), sampler)

    assert sampler.d2h_side_stream_copies is False
    assert sampler.static_slot_ids is False
    assert sampler.spec_step_fold is None
    assert lines == []


@pytest.mark.parametrize(
    "side, static",
    [(True, False), (False, True), (True, True)],
    ids=["d2h_side_stream", "spec_small_h2d", "both"],
)
def test_the_hooks_reach_the_sampler(monkeypatch, side, static):
    sampler = _spec_sampler()
    fold = _Fold()
    hooks = RuntimeHooks(sampler_d2h_side_stream=side, spec_small_h2d=static, spec_step_fold=fold)

    lines = _instantiate(monkeypatch, hooks, sampler)

    assert sampler.d2h_side_stream_copies is side
    assert sampler.static_slot_ids is static
    assert sampler.spec_step_fold is fold and fold.store is sampler.store
    # spec_small_h2d's line is DecoderMetadata.setup_speculative's, not this one.
    assert lines == ([_D2H_LINE] if side else [])


def test_a_sampler_that_is_not_a_spec_sampler_is_left_alone(monkeypatch):
    other = SimpleNamespace()

    lines = _instantiate(
        monkeypatch, RuntimeHooks(sampler_d2h_side_stream=True, spec_small_h2d=True), other
    )

    assert not hasattr(other, "d2h_side_stream_copies")
    assert not hasattr(other, "static_slot_ids")
    assert lines == []
