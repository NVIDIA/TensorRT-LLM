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
"""fused_kv_indexer_projection: one staged projection launch for ``kva`` and ``kw`` on the calls whose
hidden state is the native normalized tensor.

Requires ``absorbed_query`` (the consumer path). Registered entry point
(``dispatch.get_impl("fused_kv_indexer_projection", name)``):

* ``dsa_proj(mla, position_ids, hidden_states, attn_metadata)``: the
  query_front packet of a full-indexer target layer, the MTP draft layer (with
  ``draft_absorbed_query``) or a shared-indexer target layer, with the fused-A output ``kva``
  and the indexer ``wk | weights_proj`` output ``kw`` taken from one
  ``FusedKvProjection`` launch, or ``None`` to run the next path.

``kva`` keeps the native fused-A accumulation order and is bit-identical to
``dsv3_fused_a_gemm_op``; see ``proj_kernel.py`` for the order and
``projection.py`` for the launch configuration.
"""

from ... import dispatch
from .entry import fused_kv_dsa_proj
from .projection import FusedKvProjConfig, FusedKvProjection

dispatch.register("fused_kv_indexer_projection", "dsa_proj", fused_kv_dsa_proj)

__all__ = ["FusedKvProjConfig", "FusedKvProjection", "fused_kv_dsa_proj"]
