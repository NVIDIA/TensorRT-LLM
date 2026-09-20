#!/usr/bin/env python3
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

import json

import pytest

__extra_import_path__ = ["~"]
from jenkins.scripts.pulse_in_pipeline_scanning.utils.triage import extract_ticket_refs

pytestmark = pytest.mark.cpu_only


def test_extract_ticket_refs_accepts_strict_json_string():
    ticket_payload = {
        "license_correction_ticket": None,
        "version_bump_tickets": [
            {
                "dependency_name": "aiohttp",
                "link": "https://nvbugs.nvidia.com/1234567",
                "description": "Bump aiohttp",
            }
        ],
    }
    response_value = json.dumps(ticket_payload)

    refs = extract_ticket_refs({"value": response_value})

    assert refs["vulnerability"] == [
        {
            "dependency_name": "aiohttp",
            "ticket_reference": "1234567",
            "ticket_url": "https://nvbugs.nvidia.com/1234567",
            "status": "CREATED",
            "notes": "Bump aiohttp",
        }
    ]


def test_extract_ticket_refs_rejects_dictionary_value():
    with pytest.raises(TypeError, match="must be a JSON string"):
        extract_ticket_refs(
            {
                "value": {
                    "license_correction_ticket": None,
                    "version_bump_tickets": [],
                }
            }
        )


def test_extract_ticket_refs_accepts_known_agent_prefix(capfd):
    response_value = (
        "All dependency actions have been submitted. Now I'll produce the final JSON output.\n\n"
        '{"license_correction_ticket": null, "version_bump_tickets": []}'
    )

    refs = extract_ticket_refs({"value": response_value})

    assert refs == {"vulnerability": []}
    error_log = capfd.readouterr().err
    assert "[Triage agent response prefix workaround]" in error_log
    assert repr(response_value) in error_log


def test_extract_ticket_refs_rejects_explanation_before_json(capfd):
    response_value = (
        'Triage completed.\n{"license_correction_ticket": null, "version_bump_tickets": []}'
    )

    with pytest.raises(ValueError, match="valid JSON object only"):
        extract_ticket_refs({"value": response_value})

    error_log = capfd.readouterr().err
    assert "[Invalid triage agent response]" in error_log
    assert repr(response_value) in error_log


def test_extract_ticket_refs_rejects_missing_required_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        extract_ticket_refs({"value": "{}"})
