# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import pytest

pytestmark = pytest.mark.cpu_only


def test_validate_detects_requiredness_drift(tmp_path, monkeypatch):
    validator = importlib.import_module("tensorrt_llm.usage.schemas.__main__")
    sms = json.loads(validator.SMS_SCHEMA_PATH.read_text(encoding="utf-8"))
    sms["definitions"]["events"]["trtllm_heartbeat"]["required"] = []

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(sms), encoding="utf-8")
    monkeypatch.setattr(validator, "SMS_SCHEMA_PATH", schema_path)

    errors = validator.validate()

    assert (
        "trtllm_heartbeat: field 'seq' required in Pydantic model "
        "TrtllmHeartbeat but not required in SMS schema"
    ) in errors
