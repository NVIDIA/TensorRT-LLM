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
"""Physical KV ownership for NVFP4 key-only (FP4 MLA) caches on a ``KVCacheManagerV2``.

Behind ``TRTLLM_ENABLE_FP4_MLA_KV_OWNERSHIP_BRIDGE=1`` the transfer runs with physical KV ownership
(a writer retires only proven writes). The bridge is defined by the cache layout -- NVFP4,
``SELFKONLY`` -- not by a manager class.
"""

import os

from tensorrt_llm._torch.disaggregation.base.agent import use_pure_python_transfer_agent

OWNERSHIP_BRIDGE_ENV = "TRTLLM_ENABLE_FP4_MLA_KV_OWNERSHIP_BRIDGE"


def requires_physical_ownership(manager, mapping, transceiver_config) -> bool:
    """Select the bridge only for its explicit no-retry FP4-MLA deployment cell."""
    if os.getenv(OWNERSHIP_BRIDGE_ENV, "0") != "1":
        return False
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
    from tensorrt_llm.bindings import DataType

    fp4_mla_layout = (
        isinstance(manager, KVCacheManagerV2)
        and manager.is_disagg
        and manager.dtype == DataType.NVFP4
        and manager.kv_cache_type == CacheTypeCpp.SELFKONLY
    )
    cfg = transceiver_config
    supported = (
        fp4_mla_layout
        and os.getenv("TRTLLM_DISAGG_NO_RETRY", "0") == "1"
        and not use_pure_python_transfer_agent()
        and cfg.kv_transfer_timeout_ms is not None
        and cfg.kv_transfer_timeout_ms > 0
        and mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and cfg.kv_cache_bounce_size_mb == 0
        and not cfg.enable_pipelined_transfer
        and os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") != "1"
        and os.getenv("TRTLLM_DISAGG_LAYERWISE") != "1"
    )
    if not supported:
        raise ValueError(
            "FP4 MLA lifecycle bridge requires a disaggregated NVFP4 SELFKONLY "
            "KVCacheManagerV2, the C++ NIXL agent binding, a finite timeout, "
            "no-retry ADP, PP1/CP1, "
            "async monolithic non-layerwise transfer, and bounce disabled"
        )
    return True
