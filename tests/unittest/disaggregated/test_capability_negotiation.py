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

from dataclasses import FrozenInstanceError, replace
from uuid import uuid4

import pytest

from tensorrt_llm._torch.disaggregation.capability_negotiation import (
    LifecycleNegotiationError,
    negotiate_generation_safe_lifecycle,
)
from tensorrt_llm._torch.disaggregation.lifecycle import LifecycleCapability
from tensorrt_llm._torch.disaggregation.protocol import PROTOCOL_V1_REQUIRED_CAPABILITIES
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle, TransceiverLifecycleAdvertisement


def _advertisement(
    *,
    capabilities: set[str] | None = None,
    protocol_version: int = 1,
    qualified_legacy_mode: bool = False,
    attention_dp: bool = False,
) -> TransceiverLifecycleAdvertisement:
    supported = set(PROTOCOL_V1_REQUIRED_CAPABILITIES)
    supported.update(
        {
            LifecycleCapability.DIRECT_TRANSFER.value,
            LifecycleCapability.GENERATION_FIRST.value,
            LifecycleCapability.TENSOR_PARALLEL.value,
        }
    )
    if attention_dp:
        supported.add(LifecycleCapability.ATTENTION_DATA_PARALLEL.value)
    if capabilities is not None:
        supported = capabilities
    return TransceiverLifecycleAdvertisement(
        protocol_version=protocol_version,
        capabilities=tuple(sorted(supported)),
        qualified_legacy_mode=qualified_legacy_mode,
        backend="python",
        instance_id=str(uuid4()),
        world_size=4,
        tp_size=4,
        pp_size=1,
        cp_size=1,
        attention_dp=attention_dp,
    )


def test_lifecycle_advertisement_is_deeply_immutable() -> None:
    advertisement = _advertisement()

    with pytest.raises(FrozenInstanceError):
        advertisement.protocol_version = 0
    with pytest.raises(TypeError):
        advertisement.capabilities[0] = "OTHER"


def test_context_first_negotiates_complete_v1_contract() -> None:
    source = _advertisement()
    destination = _advertisement()

    negotiated = negotiate_generation_safe_lifecycle(
        source,
        destination,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        ctx_dp_rank=None,
    )

    assert negotiated.source is source
    assert negotiated.destination is destination
    assert PROTOCOL_V1_REQUIRED_CAPABILITIES <= negotiated.required_capabilities


def test_protocol_v0_is_rejected_for_generation_safe_request() -> None:
    source = _advertisement(
        protocol_version=0,
        qualified_legacy_mode=True,
    )

    with pytest.raises(LifecycleNegotiationError, match="protocol v1"):
        negotiate_generation_safe_lifecycle(
            source,
            _advertisement(),
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            ctx_dp_rank=None,
        )


def test_missing_required_capability_is_rejected() -> None:
    source = _advertisement()
    destination = replace(
        _advertisement(),
        capabilities=tuple(
            capability
            for capability in _advertisement().capabilities
            if capability != LifecycleCapability.SUBMISSION_FENCE.value
        ),
    )

    with pytest.raises(LifecycleNegotiationError, match="SUBMISSION_FENCE"):
        negotiate_generation_safe_lifecycle(
            source,
            destination,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            ctx_dp_rank=None,
        )


def test_generation_first_requires_generation_first_capability() -> None:
    source = _advertisement()
    destination = replace(
        _advertisement(),
        capabilities=tuple(
            capability
            for capability in _advertisement().capabilities
            if capability != LifecycleCapability.GENERATION_FIRST.value
        ),
    )

    with pytest.raises(LifecycleNegotiationError, match="GENERATION_FIRST"):
        negotiate_generation_safe_lifecycle(
            source,
            destination,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ctx_dp_rank=0,
        )


def test_generation_first_attention_dp_requires_selected_writer() -> None:
    source = _advertisement(attention_dp=True)
    destination = _advertisement(attention_dp=True)

    with pytest.raises(LifecycleNegotiationError, match="exact context writer rank"):
        negotiate_generation_safe_lifecycle(
            source,
            destination,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ctx_dp_rank=None,
        )

    negotiated = negotiate_generation_safe_lifecycle(
        source,
        destination,
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ctx_dp_rank=2,
    )
    assert LifecycleCapability.ATTENTION_DATA_PARALLEL.value in negotiated.required_capabilities
