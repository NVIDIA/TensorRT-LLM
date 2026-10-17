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
"""absorbed_query: target query/projection region (the query_front contract, token-wise part of the
index_selection contract).

Registered entry points (``dispatch.get_impl("absorbed_query", name)``):

* ``dsa_proj(mla, position_ids, hidden_states, attn_metadata)`` ->
  ``Optional[AbsorbedQueryPacket]`` for target layers (``absorbed_query``); ``None``
  means "run the native path".
* ``mtp_dsa_proj(mla, position_ids, hidden_states, attn_metadata)`` -> the
  same packet for the one-model MTP draft layer at every draft step
  (package ``draft_absorbed_query``, which extends absorbed_query to the draft layer); ``None`` for
  target layers and outside the support domain.
* ``shared_dsa_proj(mla, position_ids, hidden_states, attn_metadata)`` ->
  the preabsorbed packet of a shared-indexer target layer from one kernel
  (package ``shared_indexer_absorbed_query``, which extends absorbed_query's shared-layer path);
  ``None`` for full-indexer and draft layers and outside the support domain.
"""

from ...dispatch import register
from .query_proj import AbsorbedQueryPacket, AbsorbedQueryProjection, dsa_proj, mtp_dsa_proj
from .shared_indexer import SharedIndexerProjection, shared_dsa_proj

register("absorbed_query", "dsa_proj", dsa_proj)
register("absorbed_query", "mtp_dsa_proj", mtp_dsa_proj)
register("absorbed_query", "shared_dsa_proj", shared_dsa_proj)

__all__ = [
    "AbsorbedQueryPacket",
    "AbsorbedQueryProjection",
    "SharedIndexerProjection",
    "dsa_proj",
    "mtp_dsa_proj",
    "shared_dsa_proj",
]
