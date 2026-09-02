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

# Storage tier a reused KV cache block was served from, decided by the KV cache
# manager when the block is matched (before any onboarding copy). The tuple index
# is the wire value shared with the C++ ``KvCacheTier`` enum
# (cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h) and with the
# ``(tier, num_tokens)`` segments the managers attach to requests.
#   gpu    - block resident in GPU memory
#   host   - block in the host (secondary) pool
#   disk   - block in a file-backed secondary pool
#   remote - tokens provided by the KV cache connector (external store)
#   none   - tokens skipped by reuse but not loaded from any tier (e.g. blocks
#            outside every sliding attention window); only used for token counts
CACHE_TIER_LABELS = ("gpu", "host", "disk", "remote", "none")
# Tiers that describe where a *block* physically came from (block counters).
CACHE_TIER_BLOCK_LABELS = CACHE_TIER_LABELS[:4]


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
    PROMPT_CACHE_CACHED_TOKENS = "prompt_cache_cached_tokens"
    PROMPT_CACHE_CACHED_TOKENS_BY_TIER = "prompt_cache_cached_tokens_by_tier"
    SPEC_DEC_ACCEPTED_PER_POS = "spec_dec_accepted_per_pos"
    SPEC_DEC_DRAFTED_PER_POS = "spec_dec_drafted_per_pos"
    PREFILL_PERPLEXITY = "prefill_perplexity"
    GENERATION_PERPLEXITY = "generation_perplexity"


class RequestEventTiming(Enum):
    ARRIVAL_TIME = "arrival_time"
    FIRST_TOKEN_TIME = "first_token_time"  # nosec: B105
    FIRST_SCHEDULED_TIME = "first_scheduled_time"
    LAST_TOKEN_TIME = "last_token_time"  # nosec: B105
    KV_CACHE_TRANSFER_START = "kv_cache_transfer_start"
    KV_CACHE_TRANSFER_END = "kv_cache_transfer_end"
    KV_CACHE_SIZE = "kv_cache_size"
