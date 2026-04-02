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
"""Validate that the Pydantic models stay in sync with the SMS JSON schema.

Usage::

    python -m tensorrt_llm.usage.schemas
"""

import json
import sys
from typing import List

from tensorrt_llm.usage import schema
from tensorrt_llm.usage.schemas import SMS_SCHEMA_PATH


def validate() -> List[str]:
    """Check Pydantic model fields match the SMS JSON schema properties.

    Returns a list of human-readable error strings (empty = all good).
    """
    errors: List[str] = []

    if not SMS_SCHEMA_PATH.exists():
        errors.append(f"SMS schema file not found: {SMS_SCHEMA_PATH}")
        return errors

    sms = json.loads(SMS_SCHEMA_PATH.read_text())
    events = sms.get("definitions", {}).get("events", {})

    # Map SMS event name -> Pydantic model
    model_map = {
        "trtllm_initial_report": schema.TrtllmInitialReport,
        "trtllm_heartbeat": schema.TrtllmHeartbeat,
    }

    for event_name, model_cls in model_map.items():
        if event_name not in events:
            errors.append(f"SMS schema missing event definition: {event_name}")
            continue

        sms_props = set(events[event_name].get("properties", {}).keys())
        # Pydantic field aliases are the wire names (camelCase)
        pydantic_aliases = {f.alias or name for name, f in model_cls.model_fields.items()}

        missing_in_pydantic = sms_props - pydantic_aliases
        missing_in_sms = pydantic_aliases - sms_props

        for field in sorted(missing_in_pydantic):
            errors.append(
                f"{event_name}: field '{field}' in SMS schema but missing "
                f"from Pydantic model {model_cls.__name__}"
            )
        for field in sorted(missing_in_sms):
            errors.append(
                f"{event_name}: field '{field}' in Pydantic model "
                f"{model_cls.__name__} but missing from SMS schema"
            )

    return errors


if __name__ == "__main__":
    errs = validate()
    if errs:
        print("Validation FAILED:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Validation OK: Pydantic models match SMS schema.")
