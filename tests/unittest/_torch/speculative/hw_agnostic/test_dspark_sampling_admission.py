# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unsupported confidence/sampler combinations fail before model execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import (
    supports_advanced_draft_sampling,
    sync_group_all_greedy_sample,
    validate_dspark_draft_sampling,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.speculative.dspark import validate_confidence_request
from tensorrt_llm._torch.speculative.interface import SpecMetadata

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("enabled,rejection", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize(
    "controls,greedy",
    [
        ({}, True),
        ({"temperature": 0.0}, True),
        ({"temperature": 1.0}, False),
        ({"temperature": 1.0, "top_k": 1}, True),
        ({"min_p": 0.2}, False),
        ({"min_p": 1.0}, True),
    ],
)
def test_request_admission_matches_actual_draft_sampling_controls(
    enabled, rejection, controls, greedy
):
    config = SimpleNamespace(enable_confidence_scheduling=enabled, use_rejection_sampling=rejection)
    sampling = SimpleNamespace(temperature=None, top_k=None, top_p=None, min_p=None)
    sampling.__dict__.update(controls)
    request = SimpleNamespace(sampling_config=sampling, py_logits_post_processors=None)
    if enabled and rejection and not greedy:
        with pytest.raises(ValueError, match="advanced rejection draft sampling"):
            validate_confidence_request(config, request)
    else:
        validate_confidence_request(config, request)


@pytest.mark.parametrize("enabled", [False, True])
def test_per_request_logit_processor_is_rejected_at_confidence_admission(enabled):
    config = SimpleNamespace(enable_confidence_scheduling=enabled, use_rejection_sampling=False)
    request = SimpleNamespace(sampling_config=None, py_logits_post_processors=[object()])
    if enabled:
        with pytest.raises(ValueError, match="per-request logits post-processors"):
            validate_confidence_request(config, request)
    else:
        validate_confidence_request(config, request)


@pytest.mark.parametrize(
    "enabled,rejection,flags",
    [
        (True, True, [True, True]),
        (True, True, [True, False]),
        (True, True, [False, True]),
        (True, True, [False]),
        (True, False, [False, True]),
        (False, True, [False, True]),
    ],
)
def test_host_sampler_guard_is_group_uniform_before_graph_lookup(enabled, rejection, flags):
    errors = []
    for local_flag in flags:
        runner = SimpleNamespace()
        runner._dspark_confidence_enabled = enabled
        runner.mapping = SimpleNamespace(enable_lm_head_tp_in_adp=False, tp_size=len(flags))
        exchange = Mock(return_value=torch.tensor([[flag] for flag in flags], dtype=torch.int64))
        runner.dist = SimpleNamespace(tp_allgather_int64=exchange)
        metadata = object.__new__(SpecMetadata)
        metadata.use_rejection_sampling = rejection
        metadata.is_all_greedy_sample = local_flag
        metadata.group_all_greedy_sample = None
        sync_group_all_greedy_sample(
            metadata, runner.mapping, runner.dist, confidence_enabled=enabled
        )
        if enabled and rejection and not all(flags):
            with pytest.raises(ValueError, match="advanced rejection draft sampling"):
                validate_dspark_draft_sampling(metadata, confidence_enabled=enabled)
            errors.append(True)
        else:
            validate_dspark_draft_sampling(metadata, confidence_enabled=enabled)
            errors.append(False)
        assert exchange.call_count == int(enabled and rejection and len(flags) > 1)
    assert len(set(errors)) == 1


@pytest.mark.parametrize(
    "enabled,rejection", [(False, False), (False, True), (True, False), (True, True)]
)
def test_capture_gate_retains_only_supported_advanced_variants(enabled, rejection):
    runner = SimpleNamespace()
    runner._dspark_confidence_enabled = enabled
    runner._config = SimpleNamespace(
        spec_config=SimpleNamespace(
            use_rejection_sampling=rejection, enable_confidence_scheduling=enabled
        )
    )
    assert supports_advanced_draft_sampling(runner._config.spec_config) is not (
        enabled and rejection
    )


@pytest.mark.parametrize("processors", [False, True])
def test_invalid_confidence_request_is_excluded_before_activation(processors):
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        spec_config=SimpleNamespace(enable_confidence_scheduling=True, use_rejection_sampling=True),
    )
    request = SimpleNamespace(
        sampling_config=SimpleNamespace(temperature=1.0, top_k=None, top_p=None, min_p=None),
        py_logits_post_processors=[object()] if processors else None,
    )
    already_active = object()
    executor.waiting_queue = object()
    executor.active_requests = [already_active]
    executor._fetch_new_requests = Mock(return_value=[request])
    executor._handle_errors = Mock()
    executor._configure_csa2_remote_tail = Mock()

    assert executor._fetch_and_activate_new_requests() == []
    assert executor.active_requests == [already_active]
    executor._configure_csa2_remote_tail.assert_not_called()
    executor._handle_errors.assert_called_once()
    assert executor._handle_errors.call_args.kwargs == {
        "requests": [request],
        "charge_budget": False,
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_feature_off_retains_existing_lm_head_tp_group_sync(enabled):
    runner = SimpleNamespace()
    runner._dspark_confidence_enabled = enabled
    runner.mapping = SimpleNamespace(enable_lm_head_tp_in_adp=True, tp_size=2)
    exchange = Mock(return_value=torch.tensor([[True], [False]], dtype=torch.int64))
    runner.dist = SimpleNamespace(tp_allgather_int64=exchange)
    metadata = object.__new__(SpecMetadata)
    metadata.use_rejection_sampling = True
    metadata.is_all_greedy_sample = True
    metadata.group_all_greedy_sample = None

    sync_group_all_greedy_sample(metadata, runner.mapping, runner.dist, confidence_enabled=enabled)

    exchange.assert_called_once_with([True])
    assert not metadata.group_all_greedy_sample
    assert not metadata.is_all_greedy_sample
