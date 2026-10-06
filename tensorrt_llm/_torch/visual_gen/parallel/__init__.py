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
"""Parallelism for VisualGen transformers.

Token-sharded tensor parallelism (``parallel_config.tp_layout='token_sharded'``):

* ``token_sharded_tp``: the plan of which tokens each TP rank holds, the collectives and
  the NVFP4 scaling-factor regroup (``TokenShardedTP``, ``TokenShardPlan``);
* ``token_sharded_modules``: the adapters that convert a model's existing TP modules, and
  the rules that pick them.

See ``TOKEN_SHARDED_TP_DEVELOPER_GUIDE.md`` in this directory.
"""

from .token_sharded_modules import (
    TokenShardedAdapter,
    TokenShardedColumn,
    TokenShardedMLP,
    TokenShardedRow,
    classify,
    convert_to_token_sharded_tp,
    register_token_sharded_adapter,
)
from .token_sharded_tp import TokenShardedTP, TokenShardPlan, static_nvfp4_input_scale

__all__ = [
    "TokenShardPlan",
    "TokenShardedAdapter",
    "TokenShardedColumn",
    "TokenShardedMLP",
    "TokenShardedRow",
    "TokenShardedTP",
    "classify",
    "convert_to_token_sharded_tp",
    "register_token_sharded_adapter",
    "static_nvfp4_input_scale",
]
