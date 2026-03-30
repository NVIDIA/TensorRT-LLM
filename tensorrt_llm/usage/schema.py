# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pydantic models for GXT Event Protocol v1.6 telemetry payloads.

These models define the wire format for TRT-LLM usage telemetry sent to
the NvTelemetry/GXT endpoint. The envelope follows the GXT Event Protocol v1.6
specification; the event parameters are TRT-LLM-specific.

Reference:
- GXT API Design: GX Telemetry API Design.md
- DataDesigner reference: DataDesigner telemetry.py
"""

import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UINT32_MAX = 4_294_967_295  # 2**32 - 1; NvTelemetry "PositiveInt"
_SHORT_STR = 128  # NvTelemetry "ShortString" maxLength
_LONG_STR = 256  # NvTelemetry "LongString" maxLength

CLIENT_ID = "616561816355034"
EVENT_PROTOCOL = "1.6"
EVENT_SCHEMA_VER = "0.1"
EVENT_SYS_VER = "trtllm-telemetry/1.0"
CLIENT_TYPE = "Native"
CLIENT_VARIANT = "Release"
CPU_ARCHITECTURE = platform.uname().machine


# ---------------------------------------------------------------------------
# TRT-LLM Event Parameters (inner payloads)
# ---------------------------------------------------------------------------


class TrtllmInitialReport(BaseModel):
    """TRT-LLM initial report event parameters.

    Sent once at startup with full environment and configuration details.
    All fields are required by the SMS schema (GXT convention: every declared
    property must be in ``required``). Fields use sentinel defaults (empty
    string for strings, 0 for ints) when the actual value is unavailable.

    Field constraints match the SMS JSON schema type definitions:
    - ShortString: maxLength=128
    - LongString: maxLength=256
    - PositiveInt: ge=0, le=_UINT32_MAX (2**32 - 1)
    """

    # TRT-LLM version (ShortString)
    trtllm_version: str = Field(default="", max_length=_SHORT_STR, alias="trtllmVersion")

    # System info
    platform_info: str = Field(default="", max_length=_LONG_STR, alias="platform")  # LongString
    python_version: str = Field(
        default="", max_length=_SHORT_STR, alias="pythonVersion"
    )  # ShortString
    cpu_architecture: str = Field(
        default="", max_length=_SHORT_STR, alias="cpuArchitecture"
    )  # ShortString
    cpu_count: int = Field(default=0, ge=0, le=_UINT32_MAX, alias="cpuCount")  # PositiveInt

    # GPU info
    gpu_count: int = Field(default=0, ge=0, le=_UINT32_MAX, alias="gpuCount")  # PositiveInt
    gpu_name: str = Field(default="", max_length=_LONG_STR, alias="gpuName")  # LongString
    gpu_memory_mb: int = Field(default=0, ge=0, le=_UINT32_MAX, alias="gpuMemoryMB")  # PositiveInt
    cuda_version: str = Field(default="", max_length=_SHORT_STR, alias="cudaVersion")  # ShortString

    # Model info (architecture class name only -- no raw config) (LongString)
    architecture_class_name: str = Field(
        default="", max_length=_LONG_STR, alias="architectureClassName"
    )

    # TRT-LLM config
    backend: str = Field(default="", max_length=_SHORT_STR, alias="backend")  # ShortString
    tensor_parallel_size: int = Field(default=1, ge=0, le=_UINT32_MAX, alias="tensorParallelSize")
    pipeline_parallel_size: int = Field(
        default=1, ge=0, le=_UINT32_MAX, alias="pipelineParallelSize"
    )
    context_parallel_size: int = Field(default=1, ge=0, le=_UINT32_MAX, alias="contextParallelSize")
    moe_expert_parallel_size: int = Field(
        default=0, ge=0, le=_UINT32_MAX, alias="moeExpertParallelSize"
    )
    moe_tensor_parallel_size: int = Field(
        default=0, ge=0, le=_UINT32_MAX, alias="moeTensorParallelSize"
    )
    dtype: str = Field(default="", max_length=_SHORT_STR, alias="dtype")  # ShortString
    quantization_algo: str = Field(
        default="", max_length=_SHORT_STR, alias="quantizationAlgo"
    )  # ShortString
    kv_cache_dtype: str = Field(
        default="", max_length=_SHORT_STR, alias="kvCacheDtype"
    )  # ShortString

    # Ingress point (how TRT-LLM was invoked) (ShortString)
    ingress_point: str = Field(default="", max_length=_SHORT_STR, alias="ingressPoint")

    # Disaggregated serving metadata (ShortString)
    disagg_role: str = Field(default="", max_length=_SHORT_STR, alias="disaggRole")
    deployment_id: str = Field(default="", max_length=_SHORT_STR, alias="deploymentId")

    # Feature flags (JSON-serialized dict of enabled features)
    features_json: str = Field(default="{}", alias="featuresJson")

    model_config = {"populate_by_name": True}


class TrtllmHeartbeat(BaseModel):
    """TRT-LLM heartbeat event parameters.

    Sent periodically to signal the session is still alive.
    Contains only a monotonically increasing sequence counter.
    """

    seq: int = Field(..., ge=0, le=_UINT32_MAX, alias="seq")  # PositiveInt

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# GXT Event Wrapper (single event in the events array)
# ---------------------------------------------------------------------------


class GxtEvent(BaseModel):
    """A single event entry in the GXT events array."""

    ts: str = Field(..., description="ISO 8601 timestamp")
    name: str = Field(default="trtllm_usage_event")
    parameters: Dict[str, Any] = Field(...)


# ---------------------------------------------------------------------------
# GXT Envelope (top-level payload)
# ---------------------------------------------------------------------------


class GxtPayload(BaseModel):
    """GXT Event Protocol v1.6 envelope.

    The GXT ingestion endpoint validates a fixed envelope schema and rejects
    payloads that omit any of the required top-level fields — even fields
    that are semantically irrelevant for this client.

    GXT was designed for consumer products (GeForce Experience, NVIDIA App)
    where device fingerprints, user IDs, and GDPR consent flags carry real
    values. TRT-LLM is a server-side SDK with no browser, device, or login,
    so those fields are hardcoded to sentinel values:

    - Privacy/identity fields → "undefined"
      Signals to downstream pipelines and auditors that these fields were
      *deliberately not collected*, not accidentally missing.

    - GDPR opt-in fields → "None" (the string, not Python None)
      In GXT's data-policy framework this means "no consent model applies".

    This pattern matches the DataDesigner reference implementation (another
    internal GXT client for server-side telemetry).
    """

    # Client identification
    client_id: str = Field(default=CLIENT_ID, alias="clientId")
    client_type: str = Field(default=CLIENT_TYPE, alias="clientType")
    client_variant: str = Field(default=CLIENT_VARIANT, alias="clientVariant")
    client_ver: str = Field(..., alias="clientVer")
    cpu_architecture: str = Field(default=CPU_ARCHITECTURE, alias="cpuArchitecture")

    # Protocol metadata
    event_protocol: str = Field(default=EVENT_PROTOCOL, alias="eventProtocol")
    event_schema_ver: str = Field(default=EVENT_SCHEMA_VER, alias="eventSchemaVer")
    event_sys_ver: str = Field(default=EVENT_SYS_VER, alias="eventSysVer")

    # Session
    sent_ts: str = Field(..., alias="sentTs")
    session_id: str = Field(..., alias="sessionId")

    # Required by GXT schema but unused by server-side SDK clients.
    # "undefined" = deliberately not collected (not accidentally missing).
    browser_type: str = Field(default="undefined", alias="browserType")
    device_id: str = Field(default="undefined", alias="deviceId")
    device_make: str = Field(default="undefined", alias="deviceMake")
    device_model: str = Field(default="undefined", alias="deviceModel")
    device_os: str = Field(default="undefined", alias="deviceOS")
    device_os_version: str = Field(default="undefined", alias="deviceOSVersion")
    device_type: str = Field(default="undefined", alias="deviceType")
    user_id: str = Field(default="undefined", alias="userId")
    external_user_id: str = Field(default="undefined", alias="externalUserId")
    idp_id: str = Field(default="undefined", alias="idpId")
    integration_id: str = Field(default="undefined", alias="integrationId")
    product_name: str = Field(default="undefined", alias="productName")
    product_version: str = Field(default="undefined", alias="productVersion")

    # Required by GXT schema but no consent model applies for opt-out
    # server-side telemetry. "None" (string) = no GDPR consent tracking.
    gdpr_beh_opt_in: str = Field(default="None", alias="gdprBehOptIn")
    gdpr_func_opt_in: str = Field(default="None", alias="gdprFuncOptIn")
    gdpr_tech_opt_in: str = Field(default="None", alias="gdprTechOptIn")
    device_gdpr_beh_opt_in: str = Field(default="None", alias="deviceGdprBehOptIn")
    device_gdpr_func_opt_in: str = Field(default="None", alias="deviceGdprFuncOptIn")
    device_gdpr_tech_opt_in: str = Field(default="None", alias="deviceGdprTechOptIn")

    # Events array
    events: List[GxtEvent] = Field(...)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Helper: build a ready-to-serialize payload
# ---------------------------------------------------------------------------


def get_iso_timestamp(dt: Optional[datetime] = None) -> str:
    """Return ISO 8601 timestamp with millisecond precision."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_gxt_payload(
    event: Union[TrtllmInitialReport, TrtllmHeartbeat],
    *,
    session_id: str,
    trtllm_version: str,
) -> dict:
    """Build a complete GXT payload dict ready for json.dumps().

    Args:
        event: The TRT-LLM event to send (initial report or heartbeat).
        session_id: Ephemeral session UUID (hex string).
        trtllm_version: TRT-LLM package version string.

    Returns:
        dict suitable for json.dumps() and HTTP POST body.
    """
    now = get_iso_timestamp()

    if isinstance(event, TrtllmInitialReport):
        event_name = "trtllm_initial_report"
    elif isinstance(event, TrtllmHeartbeat):
        event_name = "trtllm_heartbeat"
    else:
        raise TypeError(f"Unknown event type: {type(event).__name__}")

    gxt_event = GxtEvent(
        ts=now,
        name=event_name,
        parameters=event.model_dump(by_alias=True),
    )

    payload = GxtPayload(
        client_ver=trtllm_version,
        sent_ts=now,
        session_id=session_id,
        events=[gxt_event],
    )

    return payload.model_dump(by_alias=True)
