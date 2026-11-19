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
"""How the bytes of a role's buffers spread over ranks: the lender's role layouts
(``pyexecutor/kv_cache/sharing/role_layout.py``), re-exported in the contract layer for the KV
sharing runtime's instance layout (``kv_sharing.contract.block_map``) and byte plan
(``kv_sharing.layout.byte_plan``) and for ``role_runs``, which read them from here, as
``base.optional`` re-exports the lender's vocabulary (``lending``). It defines nothing, so a layout
is of one class wherever it is built."""

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing.role_layout import (
    GRID_DIMENSIONS,
    ByHead,
    Own,
    Piece,
    RankTopology,
    Replicated,
    RingSlots,
    RoleLayoutV2,
    Sectioned,
    check_pieces,
    grid_place,
    role_layout_from_dict,
)

__all__ = [
    "GRID_DIMENSIONS",
    "ByHead",
    "Own",
    "Piece",
    "RankTopology",
    "Replicated",
    "RingSlots",
    "RoleLayoutV2",
    "Sectioned",
    "check_pieces",
    "grid_place",
    "role_layout_from_dict",
]
