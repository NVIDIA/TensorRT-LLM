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
"""Package router_gemm: pre-MoE reduction/norm/router boundary (the router contract).

Registered entry points (looked up with ``dispatch.get_impl("router_gemm", name)``):

* ``router_logits(hidden, gate_weight) -> Tensor | None``
* ``pre_moe_hop(allreduce, hidden, residual, norm_weight, eps, gate_weight)
  -> (hidden, residual, Tensor | None)``
* ``make_moe_router(gate, layer_idx, model_config) -> Callable | None``
"""

from ... import dispatch
from .hop import engagement_counts, make_moe_router, pre_moe_hop, router_logits
from .router_kernel import router_gemm, router_gemm_supported

dispatch.register("router_gemm", "router_logits", router_logits)
dispatch.register("router_gemm", "pre_moe_hop", pre_moe_hop)
dispatch.register("router_gemm", "make_moe_router", make_moe_router)

__all__ = [
    "engagement_counts",
    "make_moe_router",
    "pre_moe_hop",
    "router_gemm",
    "router_gemm_supported",
    "router_logits",
]
