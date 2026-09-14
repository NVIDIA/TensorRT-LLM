# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prims-TS MoE integration helpers.
"""

from .config_mapper import (
    PrimsTsGemmPair,
    PrimsTsMoeGemmConfig,
    map_trtllm_bf16_moe_tactic,
)
from .support import is_prims_ts_bf16_supported

__all__ = [
    "PrimsTsGemmPair",
    "PrimsTsMoeGemmConfig",
    "is_prims_ts_bf16_supported",
    "map_trtllm_bf16_moe_tactic",
]
