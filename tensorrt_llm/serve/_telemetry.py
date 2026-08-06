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
"""Serving-boundary telemetry helpers."""

from types import FrameType
from typing import Optional

import uvicorn

from tensorrt_llm.usage import record_observed_signal


class TelemetryUvicornServer(uvicorn.Server):
    """Uvicorn server that records handled signals without changing behavior."""

    def handle_exit(self, sig: int, frame: Optional[FrameType]) -> None:
        record_observed_signal(sig)
        super().handle_exit(sig, frame)
