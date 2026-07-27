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
"""Perf Time Events aggregator.

Offline convenience tool that stitches the per-rank ``time_events_*.jsonl``
files written live by the executor (gated by ``TRTLLM_PERF_TIME_EVENTS_PATH``)
together with the disaggregation KV-transfer CSVs
(``TRTLLM_KVCACHE_TIME_OUTPUT_PATH``) into a single combined JSON (and optional
HTML timeline). This is a convenience over the per-rank files, not the
load-bearing capture path.
"""

from .perf_time_events import PerfTimeEventsMerger, main, parse_event_dir, parse_kv_csv_dir

__all__ = [
    "PerfTimeEventsMerger",
    "parse_event_dir",
    "parse_kv_csv_dir",
    "main",
]
