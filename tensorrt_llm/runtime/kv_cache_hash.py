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

KV_CACHE_HASH_ALGO_AUTO = "auto"
KV_CACHE_HASH_ALGO_V1 = "v1_block_key"
KV_CACHE_HASH_ALGO_V2 = "v2_sha256"
KV_CACHE_HASH_ALGO_V2_SHA256_64 = "v2_sha256_64"
KV_CACHE_HASH_ALGO_DEFAULT = KV_CACHE_HASH_ALGO_V1


def get_effective_kv_cache_event_hash_algo(hash_algo: str, use_kv_cache_manager_v2: bool) -> str:
    if hash_algo != KV_CACHE_HASH_ALGO_AUTO:
        return hash_algo
    if use_kv_cache_manager_v2:
        return KV_CACHE_HASH_ALGO_V2
    return KV_CACHE_HASH_ALGO_V1


def truncate_sha256_hash_to_int64(block_hash: bytes) -> int:
    return int.from_bytes(block_hash[:8], "big", signed=False)
