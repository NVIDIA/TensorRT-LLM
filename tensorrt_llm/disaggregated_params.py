# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

import numpy as np

from tensorrt_llm.bindings import executor as tllme

_LIFECYCLE_IDENTITY_FIELD_NAMES = frozenset(
    {
        "logical_request_id",
        "prefill_artifact_id",
        "artifact_version",
        "handoff_attempt_uuid",
        "consumer_grant_id",
        "transfer_session_id",
        "generation_endpoint_name",
        "generation_endpoint_rank",
        "generation_endpoint_incarnation",
        "context_control_endpoint",
        "context_transceiver_lifecycle",
    }
)

_TRANSCEIVER_LIFECYCLE_KEYS = frozenset(
    {
        "protocol_version",
        "capabilities",
        "qualified_legacy_mode",
        "backend",
        "instance_id",
        "world_size",
        "tp_size",
        "pp_size",
        "cp_size",
        "attention_dp",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransceiverLifecycleAdvertisement:
    """Immutable server-level transceiver lifecycle contract.

    This advertisement is selected before a lifecycle-v1 request can publish
    destination addresses. The per-rank wire handshake still binds exact
    endpoint incarnations; this value lets the coordinator reject an
    unsupported backend or topology before creating cross-side obligations.
    """

    protocol_version: int
    capabilities: tuple[str, ...]
    qualified_legacy_mode: bool
    backend: str
    instance_id: str
    world_size: int
    tp_size: int
    pp_size: int
    cp_size: int
    attention_dp: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version < 0
        ):
            raise ValueError(
                "transceiver lifecycle protocol_version must be a non-negative integer"
            )
        capabilities = tuple(self.capabilities)
        if any(
            not isinstance(capability, str) or not capability for capability in capabilities
        ) or capabilities != tuple(sorted(set(capabilities))):
            raise ValueError(
                "transceiver lifecycle capabilities must be a sorted tuple of "
                "unique non-empty strings"
            )
        object.__setattr__(self, "capabilities", capabilities)
        if not isinstance(self.qualified_legacy_mode, bool):
            raise ValueError("transceiver lifecycle qualified_legacy_mode must be a boolean")
        if self.backend not in ("python", "cpp"):
            raise ValueError("transceiver lifecycle backend must be 'python' or 'cpp'")
        _parse_canonical_non_nil_uuid(
            "transceiver lifecycle instance_id",
            self.instance_id,
        )
        for name in ("world_size", "tp_size", "pp_size", "cp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"transceiver lifecycle {name} must be a positive integer")
        if not isinstance(self.attention_dp, bool):
            raise ValueError("transceiver lifecycle attention_dp must be a boolean")

    @classmethod
    def from_value(
        cls,
        value: "TransceiverLifecycleAdvertisement | Mapping[str, Any]",
    ) -> "TransceiverLifecycleAdvertisement":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("context_transceiver_lifecycle must be a dictionary")
        actual_keys = frozenset(value)
        if actual_keys != _TRANSCEIVER_LIFECYCLE_KEYS:
            missing = sorted(_TRANSCEIVER_LIFECYCLE_KEYS - actual_keys)
            extra = sorted(actual_keys - _TRANSCEIVER_LIFECYCLE_KEYS)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"extra: {', '.join(extra)}")
            raise ValueError(
                "context_transceiver_lifecycle has an invalid schema"
                + (f" ({'; '.join(details)})" if details else "")
            )
        capabilities = value["capabilities"]
        if not isinstance(capabilities, (list, tuple)):
            raise ValueError(
                "transceiver lifecycle capabilities must be a sorted tuple of "
                "unique non-empty strings"
            )
        return cls(
            protocol_version=value["protocol_version"],
            capabilities=tuple(capabilities),
            qualified_legacy_mode=value["qualified_legacy_mode"],
            backend=value["backend"],
            instance_id=value["instance_id"],
            world_size=value["world_size"],
            tp_size=value["tp_size"],
            pp_size=value["pp_size"],
            cp_size=value["cp_size"],
            attention_dp=value["attention_dp"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "qualified_legacy_mode": self.qualified_legacy_mode,
            "backend": self.backend,
            "instance_id": self.instance_id,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "cp_size": self.cp_size,
            "attention_dp": self.attention_dp,
        }


class DisaggScheduleStyle(IntEnum):
    CONTEXT_FIRST = 0
    GENERATION_FIRST = 1


def _parse_canonical_non_nil_uuid(name: str, value: object) -> UUID:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical non-nil UUID string")
    try:
        parsed_uuid = UUID(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical non-nil UUID string") from error
    if parsed_uuid.int == 0 or str(parsed_uuid) != value:
        raise ValueError(f"{name} must be a canonical non-nil UUID string")
    return parsed_uuid


def _validate_lifecycle_identity_fields(
    *,
    logical_request_id: Optional[int],
    prefill_artifact_id: Optional[str],
    artifact_version: Optional[int],
    handoff_attempt_uuid: Optional[str],
    consumer_grant_id: Optional[str],
    transfer_session_id: Optional[str],
    generation_endpoint_name: Optional[str],
    generation_endpoint_rank: Optional[int],
    generation_endpoint_incarnation: Optional[str],
    context_control_endpoint: Optional[str],
    context_transceiver_lifecycle: Optional[TransceiverLifecycleAdvertisement],
) -> None:
    """Validate generation-safe request and endpoint identity metadata."""
    fields = {
        "logical_request_id": logical_request_id,
        "prefill_artifact_id": prefill_artifact_id,
        "artifact_version": artifact_version,
        "handoff_attempt_uuid": handoff_attempt_uuid,
        "consumer_grant_id": consumer_grant_id,
        "transfer_session_id": transfer_session_id,
    }
    present_fields = [name for name, value in fields.items() if value is not None]
    if present_fields:
        if len(present_fields) != len(fields):
            missing_fields = sorted(set(fields) - set(present_fields))
            raise ValueError(
                "Disaggregated lifecycle identity fields must be provided together; "
                f"missing: {', '.join(missing_fields)}"
            )

        if isinstance(logical_request_id, bool) or not isinstance(logical_request_id, int):
            raise ValueError("logical_request_id must be a non-negative integer")
        if logical_request_id < 0:
            raise ValueError("logical_request_id must be a non-negative integer")
        if isinstance(artifact_version, bool) or not isinstance(artifact_version, int):
            raise ValueError("artifact_version must be a non-negative integer")
        if artifact_version < 0:
            raise ValueError("artifact_version must be a non-negative integer")

        uuid_fields = {
            "prefill_artifact_id": prefill_artifact_id,
            "handoff_attempt_uuid": handoff_attempt_uuid,
            "consumer_grant_id": consumer_grant_id,
            "transfer_session_id": transfer_session_id,
        }
        parsed_uuids = [
            _parse_canonical_non_nil_uuid(name, value) for name, value in uuid_fields.items()
        ]
        if len(set(parsed_uuids)) != len(parsed_uuids):
            raise ValueError("Disaggregated lifecycle UUID fields must be distinct")

    endpoint_fields = {
        "generation_endpoint_name": generation_endpoint_name,
        "generation_endpoint_rank": generation_endpoint_rank,
        "generation_endpoint_incarnation": generation_endpoint_incarnation,
    }
    present_endpoint_fields = [name for name, value in endpoint_fields.items() if value is not None]
    if present_endpoint_fields:
        if len(present_endpoint_fields) != len(endpoint_fields):
            missing_fields = sorted(set(endpoint_fields) - set(present_endpoint_fields))
            raise ValueError(
                "Generation endpoint identity fields must be provided together; "
                f"missing: {', '.join(missing_fields)}"
            )
        if not isinstance(generation_endpoint_name, str) or not generation_endpoint_name.strip():
            raise ValueError("generation_endpoint_name must be a non-empty string")
        if isinstance(generation_endpoint_rank, bool) or not isinstance(
            generation_endpoint_rank, int
        ):
            raise ValueError("generation_endpoint_rank must be a non-negative integer")
        if generation_endpoint_rank < 0:
            raise ValueError("generation_endpoint_rank must be a non-negative integer")
        _parse_canonical_non_nil_uuid(
            "generation_endpoint_incarnation", generation_endpoint_incarnation
        )

    if context_control_endpoint is not None and (
        not isinstance(context_control_endpoint, str) or not context_control_endpoint.strip()
    ):
        raise ValueError("context_control_endpoint must be a non-empty string")

    if context_transceiver_lifecycle is not None and not isinstance(
        context_transceiver_lifecycle,
        TransceiverLifecycleAdvertisement,
    ):
        raise ValueError(
            "context_transceiver_lifecycle must be an immutable transceiver lifecycle advertisement"
        )


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
        draft_tokens (List[int]): The draft tokens of the generation request
        disagg_request_id (int): The disaggregated request id, if set, both context and generation requests will use it
         as underlying request id.
        first_gen_log_probs (List): The logprobs for first_gen_tokens, produced during prefill.
         Each entry is a list (one per beam) of either ``TokenLogprobs`` (``list[dict[int, Logprob]]``,
         default format) or ``SimpleTokenLogprobs`` (``list[float]``, simple format).
        first_gen_logits (List): The generation logits for first_gen_tokens, produced during prefill.
         Each entry is a torch.Tensor of shape [num_tokens, vocab_size] (one per beam/sequence).
        ctx_usage (Dict[str, Any]): The context usage payload to preserve exact
         usage accounting on the generation server.

        multimodal_embedding_handles (List[Dict[str, Any]]): The resulting multimodal embedding handles from ViT.
        multimodal_hashes (List[List[int]]): The multimodal hashes of each multimodal item in the request.
    """

    request_type: Optional[str] = None
    # P-D Disaggregated Params
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None
    # If disagg_request_id is set, both context and generation requests will use it as underlying request id.
    disagg_request_id: Optional[int] = None
    # Generation-safe request identity. These fields form one atomic tuple.
    logical_request_id: Optional[int] = None
    prefill_artifact_id: Optional[str] = None
    artifact_version: Optional[int] = None
    handoff_attempt_uuid: Optional[str] = None
    consumer_grant_id: Optional[str] = None
    transfer_session_id: Optional[str] = None
    # Accepted generation endpoint identity. These fields form a separate atomic tuple.
    generation_endpoint_name: Optional[str] = None
    generation_endpoint_rank: Optional[int] = None
    generation_endpoint_incarnation: Optional[str] = None
    # Optional callback endpoint for context-side lifecycle control.
    context_control_endpoint: Optional[str] = None
    # Immutable source-transceiver capability advertisement selected by the
    # coordinator before any destination address may be published.
    context_transceiver_lifecycle: Optional[TransceiverLifecycleAdvertisement] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None
    ctx_usage: Optional[Dict[str, Any]] = None
    # Multi-turn conversation id (from session headers such as X-Session-ID),
    # carried through so worker-side consumers (e.g. the ADP router) can see
    # the same id the disagg orchestrator routed on.
    conversation_id: Optional[str] = None

    # E-P Disaggregated Params
    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = (
        None  # multimodal embedding handles should be a list of cudaIPC handles for each mm_embedding
    )
    multimodal_hashes: Optional[List[List[int]]] = (
        None  # user provided mm hashes should be a list of 8 integers
    )
    mrope_position_ids_handle: Optional[Dict[str, Any]] = None
    mrope_position_deltas_handle: Optional[Dict[str, Any]] = None
    _lifecycle_identity_locked: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in _LIFECYCLE_IDENTITY_FIELD_NAMES
            and getattr(self, "_lifecycle_identity_locked", False)
            and value != getattr(self, name)
        ):
            raise AttributeError(f"{name} is part of an immutable lifecycle identity")
        object.__setattr__(self, name, value)

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        # Prefer disagg_request_id over ctx_request_id
        request_id = (
            self.disagg_request_id if self.disagg_request_id is not None else self.ctx_request_id
        )
        # `first_gen_tokens` is now required by bindings and cannot be None.
        first_gen_tokens = self.first_gen_tokens if self.first_gen_tokens is not None else []
        return tllme.ContextPhaseParams(
            first_gen_tokens,
            request_id,
            self.opaque_state,
            self.draft_tokens,
            self.ctx_dp_rank,
            self.ctx_info_endpoint,
        )

    def get_request_type(self) -> tllme.RequestType:
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )

    def __post_init__(self):
        if self.context_transceiver_lifecycle is not None:
            object.__setattr__(
                self,
                "context_transceiver_lifecycle",
                TransceiverLifecycleAdvertisement.from_value(self.context_transceiver_lifecycle),
            )
        _validate_lifecycle_identity_fields(
            logical_request_id=self.logical_request_id,
            prefill_artifact_id=self.prefill_artifact_id,
            artifact_version=self.artifact_version,
            handoff_attempt_uuid=self.handoff_attempt_uuid,
            consumer_grant_id=self.consumer_grant_id,
            transfer_session_id=self.transfer_session_id,
            generation_endpoint_name=self.generation_endpoint_name,
            generation_endpoint_rank=self.generation_endpoint_rank,
            generation_endpoint_incarnation=self.generation_endpoint_incarnation,
            context_control_endpoint=self.context_control_endpoint,
            context_transceiver_lifecycle=self.context_transceiver_lifecycle,
        )
        if self.request_type is not None:
            self.request_type = self.request_type.lower()
            if self.request_type not in [
                "context_only",
                "generation_only",
                "context_and_generation",
            ]:
                raise ValueError(
                    f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                    "context_and_generation"
                )
        if self.multimodal_embedding_handles is not None:
            if self.multimodal_hashes is not None:
                # if mm hashes are provided, kvcache reuse can be enabled
                assert len(self.multimodal_embedding_handles) == len(self.multimodal_hashes), (
                    "multimodal_embedding_handles and multimodal_hashes must have the same length"
                )
                for mm_hash in self.multimodal_hashes:
                    assert isinstance(mm_hash, list), "mm_hash must be a list"
                    assert len(mm_hash) == 8, "mm_hash must be a list of 8 integers"
                    assert all(isinstance(x, int) for x in mm_hash), "mm_hash must contain integers"
            else:
                # if user did not provide mm embedding handles, kvcache reuse will be disabled
                assert len(self.multimodal_embedding_handles) > 0, (
                    "multimodal_embedding_handles must be provided"
                )
                vals = np.random.randint(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=8, dtype=np.int32
                ).tolist()
                self.multimodal_hashes = [vals] * len(self.multimodal_embedding_handles)
        object.__setattr__(self, "_lifecycle_identity_locked", True)
