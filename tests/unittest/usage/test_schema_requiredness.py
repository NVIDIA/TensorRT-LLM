# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def test_checked_in_schema_has_no_requiredness_drift() -> None:
    validator = importlib.import_module("tensorrt_llm.usage.schemas.__main__")

    assert validator.validate() == []


def test_validate_detects_property_missing_from_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = importlib.import_module("tensorrt_llm.usage.schemas.__main__")
    sms = json.loads(validator.SMS_SCHEMA_PATH.read_text(encoding="utf-8"))
    sms["definitions"]["events"]["trtllm_heartbeat"]["required"] = []

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(sms), encoding="utf-8")
    monkeypatch.setattr(validator, "SMS_SCHEMA_PATH", schema_path)

    errors = validator.validate()

    assert "trtllm_heartbeat: field 'seq' in SMS properties but missing from required" in errors


def test_validate_detects_required_field_missing_from_properties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = importlib.import_module("tensorrt_llm.usage.schemas.__main__")
    sms = json.loads(validator.SMS_SCHEMA_PATH.read_text(encoding="utf-8"))
    heartbeat = sms["definitions"]["events"]["trtllm_heartbeat"]
    heartbeat["required"].append("undeclared_field")

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(sms), encoding="utf-8")
    monkeypatch.setattr(validator, "SMS_SCHEMA_PATH", schema_path)

    errors = validator.validate()

    assert (
        "trtllm_heartbeat: field 'undeclared_field' in SMS required but missing from properties"
        in errors
    )
