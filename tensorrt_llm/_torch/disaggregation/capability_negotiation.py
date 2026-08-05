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
"""Pre-publication lifecycle capability negotiation.

The server-level advertisement is an early rejection mechanism. Exact endpoint
and allocation identities are still validated by the per-rank wire protocol;
passing this check never authorizes destination-address publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from tensorrt_llm._torch.disaggregation.lifecycle import LifecycleCapability
from tensorrt_llm._torch.disaggregation.protocol import (
    PROTOCOL_V1_REQUIRED_CAPABILITIES,
    ProtocolVersion,
    validate_protocol_advertisement,
)
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle, TransceiverLifecycleAdvertisement


class LifecycleNegotiationError(RuntimeError):
    """Raised before a request can create an incompatible handoff obligation."""


@dataclass(frozen=True, slots=True)
class NegotiatedLifecycle:
    """Validated source/destination contract for one schedule style."""

    source: TransceiverLifecycleAdvertisement
    destination: TransceiverLifecycleAdvertisement
    required_capabilities: frozenset[str]


def make_transceiver_lifecycle_advertisement(
    *,
    protocol_version: int,
    capabilities: Iterable[LifecycleCapability | str],
    qualified_legacy_mode: bool,
    backend: str,
    instance_id: str,
    world_size: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    attention_dp: bool,
) -> TransceiverLifecycleAdvertisement:
    """Build and validate the canonical immutable server advertisement."""

    capability_names = tuple(
        sorted(
            {
                capability.value if isinstance(capability, LifecycleCapability) else capability
                for capability in capabilities
            }
        )
    )
    advertisement = TransceiverLifecycleAdvertisement(
        protocol_version=protocol_version,
        capabilities=capability_names,
        qualified_legacy_mode=qualified_legacy_mode,
        backend=backend,
        instance_id=instance_id,
        world_size=world_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        attention_dp=attention_dp,
    )
    validate_protocol_advertisement(
        advertisement.protocol_version,
        endpoint_incarnation=None
        if advertisement.protocol_version == ProtocolVersion.QUALIFIED_LEGACY
        else _advertisement_incarnation(advertisement),
        capabilities=frozenset(advertisement.capabilities),
        qualified_legacy_mode=advertisement.qualified_legacy_mode,
    )
    return advertisement


def negotiate_generation_safe_lifecycle(
    source: TransceiverLifecycleAdvertisement,
    destination: TransceiverLifecycleAdvertisement,
    *,
    schedule_style: DisaggScheduleStyle,
    ctx_dp_rank: int | None,
) -> NegotiatedLifecycle:
    """Fail closed unless both endpoints implement the complete v1 contract."""

    source = TransceiverLifecycleAdvertisement.from_value(source)
    destination = TransceiverLifecycleAdvertisement.from_value(destination)
    for role, advertisement in (
        ("context", source),
        ("generation", destination),
    ):
        validate_generation_safe_advertisement(advertisement, role=role)

    required = set(PROTOCOL_V1_REQUIRED_CAPABILITIES)
    required.add(LifecycleCapability.DIRECT_TRANSFER.value)
    if source.tp_size > 1 or destination.tp_size > 1:
        required.add(LifecycleCapability.TENSOR_PARALLEL.value)
    if source.pp_size > 1 or destination.pp_size > 1:
        required.add(LifecycleCapability.PIPELINE_PARALLEL.value)
    if source.attention_dp or destination.attention_dp:
        required.add(LifecycleCapability.ATTENTION_DATA_PARALLEL.value)
    if schedule_style == DisaggScheduleStyle.GENERATION_FIRST:
        required.add(LifecycleCapability.GENERATION_FIRST.value)
        if source.attention_dp and ctx_dp_rank is None:
            raise LifecycleNegotiationError(
                "generation-first attention-DP requires an exact context writer "
                "rank before destination publication"
            )
    elif schedule_style != DisaggScheduleStyle.CONTEXT_FIRST:
        raise LifecycleNegotiationError(
            f"unsupported disaggregated schedule style {schedule_style!r}"
        )

    required_capabilities = frozenset(required)
    for role, advertisement in (
        ("context", source),
        ("generation", destination),
    ):
        missing = required_capabilities - frozenset(advertisement.capabilities)
        if missing:
            raise LifecycleNegotiationError(
                f"{role} transceiver is missing lifecycle capabilities: "
                + ", ".join(sorted(missing))
            )

    return NegotiatedLifecycle(
        source=source,
        destination=destination,
        required_capabilities=required_capabilities,
    )


def validate_generation_safe_advertisement(
    advertisement: TransceiverLifecycleAdvertisement,
    *,
    role: str,
) -> None:
    """Validate one endpoint before it can enter a protocol-v1 attempt."""
    advertisement = TransceiverLifecycleAdvertisement.from_value(advertisement)
    if advertisement.protocol_version != ProtocolVersion.GENERATION_SAFE:
        raise LifecycleNegotiationError(
            f"{role} transceiver does not advertise lifecycle protocol v1"
        )
    if advertisement.backend != "python":
        raise LifecycleNegotiationError(
            f"{role} transceiver backend {advertisement.backend!r} is not "
            "qualified for lifecycle protocol v1"
        )
    try:
        validate_protocol_advertisement(
            advertisement.protocol_version,
            endpoint_incarnation=_advertisement_incarnation(advertisement),
            capabilities=frozenset(advertisement.capabilities),
            qualified_legacy_mode=advertisement.qualified_legacy_mode,
        )
    except ValueError as error:
        raise LifecycleNegotiationError(
            f"{role} transceiver has an invalid lifecycle advertisement: {error}"
        ) from error


def _advertisement_incarnation(
    advertisement: TransceiverLifecycleAdvertisement,
) -> UUID:
    return UUID(advertisement.instance_id)
