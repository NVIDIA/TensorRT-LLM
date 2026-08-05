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

"""Executor-to-serving handoff events for generation-safe disaggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from tensorrt_llm._torch.disaggregation.protocol import (
    TransferProtocolIdentity,
    transfer_protocol_identity_from_params,
)


class HandoffEventState(str, Enum):
    """Terminal result of the receive-side handoff obligation."""

    HANDOFF_COMMITTED = "HANDOFF_COMMITTED"
    HANDOFF_FAILED = "HANDOFF_FAILED"
    HANDOFF_ABORTED = "HANDOFF_ABORTED"


@dataclass(frozen=True, slots=True)
class HandoffLifecycleEvent:
    """Immutable evidence emitted after distributed receive retirement.

    This event is a logical control-plane notification. Allocation reuse is
    separately authorized by allocator-lease settlement after physical drain.
    """

    session: TransferProtocolIdentity
    consumer_grant_id: UUID
    state: HandoffEventState
    reason: str = ""
    lifecycle_protocol_version: int = 1

    def __post_init__(self) -> None:
        if self.lifecycle_protocol_version != 1:
            raise ValueError("handoff events require lifecycle protocol version 1")
        if not isinstance(self.consumer_grant_id, UUID) or self.consumer_grant_id.int == 0:
            raise ValueError("consumer_grant_id must be a non-nil UUID")
        if self.consumer_grant_id in {
            self.session.attempt.prefill_artifact_id,
            self.session.attempt.handoff_attempt_uuid,
            self.session.transfer_session_id,
        }:
            raise ValueError("consumer_grant_id must be distinct from session UUIDs")
        if not isinstance(self.state, HandoffEventState):
            raise TypeError("state must be a HandoffEventState")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")

    @property
    def committed(self) -> bool:
        return self.state is HandoffEventState.HANDOFF_COMMITTED

    @classmethod
    def from_params(
        cls,
        params: object,
        state: HandoffEventState,
        *,
        reason: str = "",
    ) -> "HandoffLifecycleEvent":
        session = transfer_protocol_identity_from_params(params)
        consumer_grant_id = getattr(params, "consumer_grant_id", None)
        if consumer_grant_id is None:
            raise ValueError("consumer_grant_id is missing from lifecycle metadata")
        try:
            grant_uuid = UUID(str(consumer_grant_id))
        except (TypeError, ValueError) as error:
            raise ValueError("consumer_grant_id must be a UUID") from error
        return cls(
            session=session,
            consumer_grant_id=grant_uuid,
            state=state,
            reason=reason,
        )


__all__ = [
    "HandoffEventState",
    "HandoffLifecycleEvent",
]
