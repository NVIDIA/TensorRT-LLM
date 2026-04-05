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
from enum import Enum


class MetricNames(Enum):
    TTFT = "ttft"
    TPOT = "tpot"
    E2E = "e2e"
    REQUEST_QUEUE_TIME = "request_queue_time"
    ARRIVAL_TIMESTAMP = 'arrival_timestamp'
    PREFILL_TIME = "prefill_time"
    DECODE_TIME = "decode_time"
    INFERENCE_TIME = "inference_time"
    PROMPT_TOKENS = "prompt_tokens"
    GENERATION_TOKENS = "generation_tokens"


class RequestEventTiming(Enum):
    ARRIVAL_TIME = "arrival_time"
    FIRST_TOKEN_TIME = "first_token_time"  # nosec: B105
    FIRST_SCHEDULED_TIME = "first_scheduled_time"
    LAST_TOKEN_TIME = "last_token_time"  # nosec: B105
    KV_CACHE_TRANSFER_START = "kv_cache_transfer_start"
    KV_CACHE_TRANSFER_END = "kv_cache_transfer_end"
    KV_CACHE_SIZE = "kv_cache_size"
