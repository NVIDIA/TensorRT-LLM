# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Resource manager package for KV cache, PEFT, and auxiliary resource managers.

All public symbols are re-exported here for backward compatibility.
Existing imports of the form ``from .resource_manager import X`` continue
to work unchanged.
"""

import tensorrt_llm.bindings

from .base import (
    BaseResourceManager,
    ResourceManager,
    ResourceManagerType,
    Role,
    compute_page_count,
    request_context,
)
from .kv_cache_manager import KVCacheManager
from .kv_cache_manager_v2 import KVCacheManagerV2
from .kv_cache_spec_ops import _update_kv_cache_draft_token_location, get_pp_layers
from .peft_cache_manager import PeftCacheManager
from .simple_managers import BlockManager, SlotManager

# Re-export binding aliases that some consumers imported from the old monolith.
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
DataType = tensorrt_llm.bindings.DataType

__all__ = [
    # base
    "BaseResourceManager",
    "ResourceManager",
    "ResourceManagerType",
    "Role",
    "compute_page_count",
    "request_context",
    # kv cache managers
    "KVCacheManager",
    "KVCacheManagerV2",
    # spec-dec kv ops
    "get_pp_layers",
    "_update_kv_cache_draft_token_location",
    # peft
    "PeftCacheManager",
    # simple managers
    "BlockManager",
    "SlotManager",
    # binding aliases (backward compat for consumers that imported these from the old monolith)
    "CacheTypeCpp",
    "DataType",
]
