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
"""Tests for Pydantic schema models, GXT payload format, and SMS schema compliance."""

import json

import pytest
from pydantic import ValidationError

from tensorrt_llm.usage import schema, schemas

# ---------------------------------------------------------------------------
# Features JSON payload structure tests
# ---------------------------------------------------------------------------


class TestFeaturesJsonPayload:
    """Tests for featuresJson field in the GXT payload."""

    def test_initial_report_contains_features_json(self):
        """FeaturesJson appears in initial report event parameters."""
        report = schema.TrtllmInitialReport(
            featuresJson='{"lora":true}',
        )
        payload = schema.build_gxt_payload(
            event=report,
            session_id="test",
            trtllm_version="0.18.0",
        )
        params = payload["events"][0]["parameters"]
        assert "featuresJson" in params
        assert params["featuresJson"] == '{"lora":true}'

    def test_features_json_round_trips_through_gxt_payload(self):
        """FeaturesJson survives full JSON serialization round-trip."""
        features = '{"lora":true,"speculative_decoding":false,"data_parallel_size":4}'
        report = schema.TrtllmInitialReport(featuresJson=features)
        payload = schema.build_gxt_payload(
            event=report,
            session_id="test",
            trtllm_version="0.18.0",
        )
        json_str = json.dumps(payload)
        parsed = json.loads(json_str)
        inner = json.loads(parsed["events"][0]["parameters"]["featuresJson"])
        assert inner["lora"] is True
        assert inner["speculative_decoding"] is False
        assert inner["data_parallel_size"] == 4

    def test_features_json_default_is_empty_object(self):
        """Default featuresJson value is '{}'."""
        report = schema.TrtllmInitialReport()
        data = report.model_dump(by_alias=True)
        assert data["featuresJson"] == "{}"


# ---------------------------------------------------------------------------
# GXT payload format tests
# ---------------------------------------------------------------------------


class TestGxtPayload:
    def test_initial_report_format(self):
        """Initial report payload matches GXT protocol v1.6 envelope."""
        event = schema.TrtllmInitialReport(
            trtllmVersion="0.18.0",
            platform="Linux-5.15.0",
            pythonVersion="3.10.12",
            cpuArchitecture="x86_64",
            cpuCount=64,
            gpuCount=8,
            gpuName="NVIDIA H100 80GB HBM3",
            gpuMemoryMB=81920,
            cudaVersion="12.4",
            architectureClassName="LlamaForCausalLM",
            backend="pytorch",
            tensorParallelSize=4,
            pipelineParallelSize=2,
            contextParallelSize=1,
            moeExpertParallelSize=8,
            moeTensorParallelSize=2,
            dtype="float16",
            quantizationAlgo="fp8",
            kvCacheDtype="auto",
            featuresJson='{"lora":true,"speculative_decoding":false}',
        )
        payload = schema.build_gxt_payload(
            event=event, session_id="abc123", trtllm_version="0.18.0"
        )

        # Top-level envelope
        assert payload["clientId"] == schema.CLIENT_ID
        assert payload["eventProtocol"] == schema.EVENT_PROTOCOL
        assert payload["eventSchemaVer"] == schema.EVENT_SCHEMA_VER
        assert payload["eventSysVer"] == schema.EVENT_SYS_VER
        assert payload["clientType"] == "Native"
        assert payload["clientVariant"] == "Release"
        assert payload["clientVer"] == "0.18.0"
        assert payload["sessionId"] == "abc123"

        # Privacy fields
        assert payload["userId"] == "undefined"
        assert payload["deviceId"] == "undefined"
        assert payload["browserType"] == "undefined"
        assert payload["externalUserId"] == "undefined"

        # GDPR fields
        assert payload["gdprBehOptIn"] == "None"
        assert payload["gdprFuncOptIn"] == "None"
        assert payload["gdprTechOptIn"] == "None"
        assert payload["deviceGdprBehOptIn"] == "None"

        # Events array
        assert len(payload["events"]) == 1
        assert payload["events"][0]["name"] == "trtllm_initial_report"
        assert payload["events"][0]["ts"]  # non-empty timestamp

        # Event parameters
        params = payload["events"][0]["parameters"]
        assert params["trtllmVersion"] == "0.18.0"
        assert params["gpuCount"] == 8
        assert params["gpuName"] == "NVIDIA H100 80GB HBM3"
        assert params["moeExpertParallelSize"] == 8
        assert params["moeTensorParallelSize"] == 2

    def test_heartbeat_minimal(self):
        """Heartbeat payload contains only seq field."""
        event = schema.TrtllmHeartbeat(seq=0)
        payload = schema.build_gxt_payload(event=event, session_id="xyz", trtllm_version="0.18.0")

        assert payload["events"][0]["name"] == "trtllm_heartbeat"
        params = payload["events"][0]["parameters"]
        assert params == {"seq": 0}

    def test_json_serializable(self):
        """Entire payload is JSON-serializable (no enum, datetime issues)."""
        event = schema.TrtllmInitialReport(
            trtllmVersion="0.18.0",
            platform="Linux-5.15.0",
            pythonVersion="3.10.12",
            cpuArchitecture="x86_64",
            cpuCount=64,
            gpuCount=4,
            gpuName="NVIDIA A100",
            gpuMemoryMB=40960,
            cudaVersion="12.4",
            architectureClassName="LlamaForCausalLM",
            backend="pytorch",
            tensorParallelSize=4,
            pipelineParallelSize=1,
            contextParallelSize=1,
            moeExpertParallelSize=0,
            moeTensorParallelSize=0,
            dtype="float16",
            quantizationAlgo="none",
            kvCacheDtype="auto",
            featuresJson='{"lora":false}',
        )
        payload = schema.build_gxt_payload(event=event, session_id="test", trtllm_version="0.18.0")
        json_str = json.dumps(payload)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["clientId"] == schema.CLIENT_ID

    def test_timestamp_format(self):
        """ISO timestamp has millisecond precision and ends with Z."""
        from datetime import datetime, timezone

        ts = schema.get_iso_timestamp(datetime(2026, 2, 17, 10, 30, 0, 500000, tzinfo=timezone.utc))
        assert ts == "2026-02-17T10:30:00.500Z"

    def test_timestamp_default_utc(self):
        """Default timestamp is current UTC time."""
        ts = schema.get_iso_timestamp()
        assert ts.endswith("Z")
        assert "T" in ts


# ---------------------------------------------------------------------------
# Schema constants tests
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_client_id(self):
        """CLIENT_ID matches provisioned value."""
        assert schema.CLIENT_ID == "616561816355034"

    def test_event_protocol(self):
        """EVENT_PROTOCOL is v1.6."""
        assert schema.EVENT_PROTOCOL == "1.6"

    def test_event_schema_ver(self):
        """EVENT_SCHEMA_VER is 0.1 (matches SMS schema schemaVersion)."""
        assert schema.EVENT_SCHEMA_VER == "0.1"

    def test_event_sys_ver(self):
        """EVENT_SYS_VER identifies the telemetry subsystem."""
        assert schema.EVENT_SYS_VER == "trtllm-telemetry/1.0"


# ---------------------------------------------------------------------------
# Heartbeat event content tests
# ---------------------------------------------------------------------------


class TestHeartbeatContent:
    def test_heartbeat_contains_only_seq(self):
        """Heartbeat events contain only seq field."""
        event = schema.TrtllmHeartbeat(seq=0)
        payload = schema.build_gxt_payload(event=event, session_id="xyz", trtllm_version="0.18.0")
        params = payload["events"][0]["parameters"]
        assert params == {"seq": 0}
        assert payload["events"][0]["name"] == "trtllm_heartbeat"

    def test_heartbeat_seq_increments(self):
        """Heartbeat seq field supports incrementing values."""
        for i in range(5):
            event = schema.TrtllmHeartbeat(seq=i)
            payload = schema.build_gxt_payload(
                event=event, session_id="xyz", trtllm_version="0.18.0"
            )
            params = payload["events"][0]["parameters"]
            assert params["seq"] == i


# ---------------------------------------------------------------------------
# Ingress point field tests
# ---------------------------------------------------------------------------


class TestIngressPoint:
    """Tests for the ingressPoint field in telemetry events."""

    def test_initial_report_has_ingress_point_field(self):
        """TrtllmInitialReport has an ingressPoint field."""
        report = schema.TrtllmInitialReport()
        data = report.model_dump(by_alias=True)
        assert "ingressPoint" in data

    def test_ingress_point_default_empty_string(self):
        """Default ingressPoint is empty string."""
        report = schema.TrtllmInitialReport()
        data = report.model_dump(by_alias=True)
        assert data["ingressPoint"] == ""

    def test_ingress_point_set_to_cli_serve(self):
        """IngressPoint can be set to 'cli_serve'."""
        report = schema.TrtllmInitialReport(ingressPoint="cli_serve")
        data = report.model_dump(by_alias=True)
        assert data["ingressPoint"] == "cli_serve"

    def test_ingress_point_set_to_llm_class(self):
        """IngressPoint can be set to 'llm_class'."""
        report = schema.TrtllmInitialReport(ingressPoint="llm_class")
        data = report.model_dump(by_alias=True)
        assert data["ingressPoint"] == "llm_class"

    def test_ingress_point_set_to_cli_bench(self):
        """IngressPoint can be set to 'cli_bench'."""
        report = schema.TrtllmInitialReport(ingressPoint="cli_bench")
        data = report.model_dump(by_alias=True)
        assert data["ingressPoint"] == "cli_bench"

    def test_ingress_point_set_to_cli_eval(self):
        """IngressPoint can be set to 'cli_eval'."""
        report = schema.TrtllmInitialReport(ingressPoint="cli_eval")
        data = report.model_dump(by_alias=True)
        assert data["ingressPoint"] == "cli_eval"

    def test_ingress_point_in_gxt_payload(self):
        """IngressPoint appears in the full GXT payload."""
        report = schema.TrtllmInitialReport(ingressPoint="cli_serve")
        payload = schema.build_gxt_payload(
            event=report,
            session_id="test-session",
            trtllm_version="1.0.0",
        )
        params = payload["events"][0]["parameters"]
        assert params["ingressPoint"] == "cli_serve"


# ---------------------------------------------------------------------------
# UsageContext enum tests
# ---------------------------------------------------------------------------


try:
    from tensorrt_llm.llmapi import llm_args as _llm_args_mod

    _HAS_LLMAPI = True
except ImportError:
    _HAS_LLMAPI = False

_skip_no_llmapi = pytest.mark.skipif(
    not _HAS_LLMAPI,
    reason="Requires full tensorrt_llm.llmapi (C++ bindings not available)",
)


@_skip_no_llmapi
class TestUsageContextEnum:
    """Tests for UsageContext enum values."""

    def test_all_context_values_are_strings(self):
        """All UsageContext values are valid strings."""
        for ctx in _llm_args_mod.UsageContext:
            assert isinstance(ctx.value, str)
            assert len(ctx.value) > 0

    def test_expected_contexts_exist(self):
        """All expected UsageContext values are defined."""
        expected = {"unknown", "llm_class", "cli_serve", "cli_bench", "cli_eval"}
        actual = {ctx.value for ctx in _llm_args_mod.UsageContext}
        assert actual == expected

    def test_unknown_is_default(self):
        """UNKNOWN is the default UsageContext."""
        config = _llm_args_mod.TelemetryConfig()
        assert config.usage_context == _llm_args_mod.UsageContext.UNKNOWN


# ---------------------------------------------------------------------------
# Schema compliance tests (SMS Event Definition validation)
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    """Validate Pydantic models match the SMS Event Definition Schema.

    The ground truth is the SMS schema file which defines the event
    structure registered with the NvTelemetry Data Platform.  These tests
    catch accidental schema drift.
    """

    def _load_sms_schema(self) -> dict:
        """Load the checked-in SMS Event Definition Schema."""
        assert schemas.SMS_SCHEMA_PATH.exists(), (
            f"SMS schema file not found: {schemas.SMS_SCHEMA_PATH}\n"
            "This file is the ground-truth event definition."
        )
        return json.loads(schemas.SMS_SCHEMA_PATH.read_text())

    def _get_pydantic_aliases(self, model_cls) -> set:
        """Extract alias names from a Pydantic model."""
        aliases = set()
        for name, field_info in model_cls.model_fields.items():
            alias = field_info.alias if field_info.alias else name
            aliases.add(alias)
        return aliases

    # --- Schema metadata validation ---

    def test_schema_version_matches_code_constant(self):
        """SMS schemaVersion matches EVENT_SCHEMA_VER in code."""
        sms_schema = self._load_sms_schema()
        assert sms_schema["schemaMeta"]["schemaVersion"] == schema.EVENT_SCHEMA_VER

    def test_client_id_matches_code_constant(self):
        """SMS clientId matches CLIENT_ID in code."""
        sms_schema = self._load_sms_schema()
        assert sms_schema["schemaMeta"]["clientId"] == schema.CLIENT_ID

    def test_schema_has_draft07(self):
        """SMS schema uses JSON Schema draft-07."""
        sms_schema = self._load_sms_schema()
        assert sms_schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_schema_has_two_events(self):
        """SMS schema defines exactly two events."""
        sms_schema = self._load_sms_schema()
        events = sms_schema["definitions"]["events"]
        assert set(events.keys()) == {"trtllm_initial_report", "trtllm_heartbeat"}

    # --- TrtllmInitialReport <-> trtllm_initial_report field sync ---

    def test_initial_report_fields_match_sms(self):
        """Every TrtllmInitialReport field (by alias) exists in SMS initial_report."""
        sms_schema = self._load_sms_schema()
        sms_props = set(
            sms_schema["definitions"]["events"]["trtllm_initial_report"]["properties"].keys()
        )
        pydantic_aliases = self._get_pydantic_aliases(schema.TrtllmInitialReport)

        missing_in_sms = pydantic_aliases - sms_props
        assert not missing_in_sms, (
            f"Fields in TrtllmInitialReport but missing from SMS schema: {sorted(missing_in_sms)}"
        )

    def test_sms_initial_report_fields_match_pydantic(self):
        """Every SMS initial_report property exists in TrtllmInitialReport."""
        sms_schema = self._load_sms_schema()
        sms_props = set(
            sms_schema["definitions"]["events"]["trtllm_initial_report"]["properties"].keys()
        )
        pydantic_aliases = self._get_pydantic_aliases(schema.TrtllmInitialReport)

        missing_in_pydantic = sms_props - pydantic_aliases
        assert not missing_in_pydantic, (
            f"Fields in SMS schema but missing from TrtllmInitialReport: "
            f"{sorted(missing_in_pydantic)}"
        )

    def test_initial_report_has_all_expected_fields(self):
        """SMS trtllm_initial_report contains all expected telemetry fields."""
        sms_schema = self._load_sms_schema()
        props = set(
            sms_schema["definitions"]["events"]["trtllm_initial_report"]["properties"].keys()
        )

        expected_fields = {
            "trtllmVersion",
            "platform",
            "pythonVersion",
            "cpuArchitecture",
            "cpuCount",
            "gpuCount",
            "gpuName",
            "gpuMemoryMB",
            "cudaVersion",
            "architectureClassName",
            "backend",
            "tensorParallelSize",
            "pipelineParallelSize",
            "contextParallelSize",
            "moeExpertParallelSize",
            "moeTensorParallelSize",
            "dtype",
            "quantizationAlgo",
            "kvCacheDtype",
            "ingressPoint",
            "featuresJson",
            "disaggRole",
            "deploymentId",
        }

        missing = expected_fields - props
        assert not missing, f"Missing expected initial_report fields: {missing}"

        extra = props - expected_fields
        assert not extra, (
            f"Unexpected initial_report fields: {extra}. "
            "If intentional, add them to the expected set in this test."
        )

    # --- TrtllmHeartbeat <-> trtllm_heartbeat field sync ---

    def test_heartbeat_fields_match_sms(self):
        """Every TrtllmHeartbeat field (by alias) exists in SMS heartbeat."""
        sms_schema = self._load_sms_schema()
        sms_props = set(
            sms_schema["definitions"]["events"]["trtllm_heartbeat"]["properties"].keys()
        )
        pydantic_aliases = self._get_pydantic_aliases(schema.TrtllmHeartbeat)

        missing_in_sms = pydantic_aliases - sms_props
        assert not missing_in_sms, (
            f"Fields in TrtllmHeartbeat but missing from SMS schema: {sorted(missing_in_sms)}"
        )

    def test_sms_heartbeat_fields_match_pydantic(self):
        """Every SMS heartbeat property exists in TrtllmHeartbeat."""
        sms_schema = self._load_sms_schema()
        sms_props = set(
            sms_schema["definitions"]["events"]["trtllm_heartbeat"]["properties"].keys()
        )
        pydantic_aliases = self._get_pydantic_aliases(schema.TrtllmHeartbeat)

        missing_in_pydantic = sms_props - pydantic_aliases
        assert not missing_in_pydantic, (
            f"Fields in SMS schema but missing from TrtllmHeartbeat: {sorted(missing_in_pydantic)}"
        )

    def test_heartbeat_required_fields(self):
        """SMS trtllm_heartbeat requires exactly 'seq'."""
        sms_schema = self._load_sms_schema()
        required = sms_schema["definitions"]["events"]["trtllm_heartbeat"]["required"]
        assert required == ["seq"]

    def test_initial_report_required_fields(self):
        """SMS trtllm_initial_report requires all declared fields."""
        sms_schema = self._load_sms_schema()
        required = set(sms_schema["definitions"]["events"]["trtllm_initial_report"]["required"])
        all_props = set(
            sms_schema["definitions"]["events"]["trtllm_initial_report"]["properties"].keys()
        )
        assert required == all_props, (
            f"Expected all properties to be required.\n"
            f"  Missing from required: {all_props - required}\n"
            f"  Extra in required: {required - all_props}"
        )

    # --- Additional properties enforcement ---

    def test_initial_report_no_additional_properties(self):
        """SMS trtllm_initial_report has additionalProperties: false."""
        sms_schema = self._load_sms_schema()
        event = sms_schema["definitions"]["events"]["trtllm_initial_report"]
        assert event["additionalProperties"] is False

    def test_heartbeat_no_additional_properties(self):
        """SMS trtllm_heartbeat has additionalProperties: false."""
        sms_schema = self._load_sms_schema()
        event = sms_schema["definitions"]["events"]["trtllm_heartbeat"]
        assert event["additionalProperties"] is False

    # --- Privacy / PII guard ---

    def test_no_pii_fields_in_initial_report(self):
        """Initial report must NOT contain PII or sensitive fields."""
        sms_schema = self._load_sms_schema()
        props = set(
            sms_schema["definitions"]["events"]["trtllm_initial_report"]["properties"].keys()
        )

        forbidden_fields = {
            "num_layers",
            "numLayers",
            "hidden_size",
            "hiddenSize",
            "num_attention_heads",
            "numAttentionHeads",
            "model_type",
            "modelType",
            "userId",
            "userName",
            "hostName",
            "hostname",
            "macAddress",
            "ipAddress",
            "modelName",
            "modelPath",
            "ttft",
            "tpot",
            "latency",
            "throughput",
            "tokensPerSecond",
        }

        found = forbidden_fields & props
        assert not found, (
            f"Forbidden fields found in SMS schema: {found}. "
            "These fields violate privacy/compliance constraints."
        )

    # --- GDPR metadata ---

    def test_events_have_gdpr_metadata(self):
        """Both events have GDPR metadata in eventMeta."""
        sms_schema = self._load_sms_schema()
        for event_name in ("trtllm_initial_report", "trtllm_heartbeat"):
            event = sms_schema["definitions"]["events"][event_name]
            assert "eventMeta" in event, f"{event_name} missing eventMeta"
            assert "gdpr" in event["eventMeta"], f"{event_name} missing gdpr in eventMeta"
            gdpr = event["eventMeta"]["gdpr"]
            assert gdpr["category"] == "functional"
            assert "description" in gdpr

    # --- GXT envelope ---

    def test_envelope_contains_all_gxt_v16_keys(self):
        """GxtPayload model contains all GXT Event Protocol v1.6 envelope fields."""
        payload = schema.GxtPayload(
            clientVer="0.18.0",
            sentTs="2026-01-01T00:00:00.000Z",
            sessionId="test",
            events=[],
        )
        serialized = payload.model_dump(by_alias=True)
        props = set(serialized.keys())

        gxt_v16_fields = {
            "clientId",
            "clientType",
            "clientVariant",
            "clientVer",
            "cpuArchitecture",
            "eventProtocol",
            "eventSchemaVer",
            "eventSysVer",
            "sentTs",
            "sessionId",
            "browserType",
            "deviceId",
            "deviceMake",
            "deviceModel",
            "deviceOS",
            "deviceOSVersion",
            "deviceType",
            "userId",
            "externalUserId",
            "idpId",
            "integrationId",
            "productName",
            "productVersion",
            "gdprBehOptIn",
            "gdprFuncOptIn",
            "gdprTechOptIn",
            "deviceGdprBehOptIn",
            "deviceGdprFuncOptIn",
            "deviceGdprTechOptIn",
            "events",
        }

        missing = gxt_v16_fields - props
        assert not missing, (
            f"GXT v1.6 fields missing from envelope: {missing}. "
            "The GXT endpoint will reject payloads without these fields."
        )

        extra = props - gxt_v16_fields
        assert not extra, (
            f"Unexpected envelope fields: {extra}. "
            "If intentional, add them to gxt_v16_fields in this test."
        )

    # --- JSON schema validation (from collapsed TestSchemaDriftDetection) ---

    def test_initial_report_validates_against_json_schema(self):
        """A fully-populated TrtllmInitialReport must validate against the JSON schema."""
        import jsonschema

        sms_schema = json.loads(schemas.SMS_SCHEMA_PATH.read_text())
        report = schema.TrtllmInitialReport(
            trtllmVersion="1.0",
            platform="Linux",
            pythonVersion="3.10",
            cpuArchitecture="x86_64",
            cpuCount=8,
            gpuCount=1,
            gpuName="H100",
            gpuMemoryMB=81920,
            cudaVersion="12.0",
            architectureClassName="LlamaForCausalLM",
            backend="pytorch",
            tensorParallelSize=1,
            pipelineParallelSize=1,
            contextParallelSize=1,
            moeExpertParallelSize=0,
            moeTensorParallelSize=0,
            dtype="float16",
            quantizationAlgo="",
            kvCacheDtype="",
            ingressPoint="llm_class",
            featuresJson='{"lora":false}',
            disaggRole="",
            deploymentId="",
        )
        payload = report.model_dump(by_alias=True)
        initial_schema = sms_schema["definitions"]["events"]["trtllm_initial_report"].copy()
        initial_schema["definitions"] = sms_schema["definitions"]
        jsonschema.validate(instance=payload, schema=initial_schema)

    def test_heartbeat_validates_against_json_schema(self):
        """A TrtllmHeartbeat must validate against the JSON schema."""
        import jsonschema

        sms_schema = json.loads(schemas.SMS_SCHEMA_PATH.read_text())
        heartbeat = schema.TrtllmHeartbeat(seq=0)
        payload = heartbeat.model_dump(by_alias=True)
        hb_schema = sms_schema["definitions"]["events"]["trtllm_heartbeat"].copy()
        hb_schema["definitions"] = sms_schema["definitions"]
        jsonschema.validate(instance=payload, schema=hb_schema)


# ---------------------------------------------------------------------------
# Pydantic field constraint validation tests
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    """Pydantic field constraint enforcement (max_length, ge/le)."""

    def test_initial_report_rejects_overlength_short_string(self):
        """ShortString field (max_length=128) rejects 129-char value."""
        with pytest.raises(ValidationError):
            schema.TrtllmInitialReport(trtllmVersion="a" * 129)

    def test_initial_report_rejects_overlength_long_string(self):
        """LongString field (max_length=256) rejects 257-char value."""
        with pytest.raises(ValidationError):
            schema.TrtllmInitialReport(platform="a" * 257)

    def test_initial_report_accepts_max_length_string(self):
        """ShortString field accepts exactly 128 chars (boundary)."""
        report = schema.TrtllmInitialReport(trtllmVersion="a" * 128)
        assert len(report.trtllm_version) == 128

    def test_heartbeat_rejects_negative_seq(self):
        """TrtllmHeartbeat rejects seq < 0 (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            schema.TrtllmHeartbeat(seq=-1)

    def test_heartbeat_rejects_overflow_seq(self):
        """TrtllmHeartbeat rejects seq > uint32 max (le=4294967295)."""
        with pytest.raises(ValidationError):
            schema.TrtllmHeartbeat(seq=4294967296)

    def test_initial_report_rejects_negative_int(self):
        """PositiveInt field (ge=0) rejects negative value."""
        with pytest.raises(ValidationError):
            schema.TrtllmInitialReport(cpuCount=-1)
