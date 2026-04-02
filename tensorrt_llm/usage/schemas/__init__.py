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
"""Ground-truth SMS Event Definition Schema for TRT-LLM usage telemetry.

A single JSON Schema file (``trtllm_usage_event_schema.json``) captures the
SMS Event Definition -- the event structure registered with the NvTelemetry
Data Platform. This is the **ground truth** for the telemetry schema; the
Pydantic models in ``tensorrt_llm.usage.schema`` must stay in sync with it.

This file is hand-written (not auto-generated from Pydantic models) and
checked into version control as the canonical reference for:

1. **Drift detection** -- CI tests assert the Pydantic model fields match
   the SMS schema properties.  A field change in ``schema.py`` without
   updating the SMS schema (or vice versa) causes a test failure.

2. **NvTelemetry cross-reference** -- The SMS schema is the format expected
   by the NvTelemetry / Data Platform team for event registration.

3. **Auditability** -- Legal and privacy reviewers can inspect the schema
   without reading Python.

Validate after changing ``tensorrt_llm.usage.schema``::

    pytest tests/unittest/usage/test_schema.py -x -q
"""

from pathlib import Path

SCHEMAS_DIR = Path(__file__).parent

SMS_SCHEMA_PATH = SCHEMAS_DIR / "trtllm_usage_event_schema.json"
