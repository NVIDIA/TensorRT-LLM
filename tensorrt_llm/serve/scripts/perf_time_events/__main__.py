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
"""Entry point for running perf_time_events as a module.

Usage:
    python -m tensorrt_llm.serve.scripts.perf_time_events \
        --event-dir /tmp/perf_events --kv-csv-dir /tmp/kv_csv \
        -o /tmp/perf_events/combined.json --html
"""

from .perf_time_events import main

if __name__ == "__main__":
    main()
