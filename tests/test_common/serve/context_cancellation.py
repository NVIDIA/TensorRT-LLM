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
"""Worker HTTP payloads used by the context-cancellation tests."""

from typing import Any

from tensorrt_llm.disaggregated_params import DisaggScheduleStyle


def with_context_reservation(
    completion: dict[str, Any], *, pair_id: int, token: str
) -> dict[str, Any]:
    """Attach a context-first reservation in the worker's JSON wire format."""
    return {
        **completion,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": pair_id,
            "schedule_style": DisaggScheduleStyle.CONTEXT_FIRST.value,
            "context_cancellation_token": token,
        },
    }
