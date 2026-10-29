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
"""post_moe_hop: multi-node post-MoE finalize/reduction/norm (the post_moe_hop contract).

Registered entry points (``dispatch.get_impl("post_moe_hop", name)``):

* ``post_moe_hop``: :class:`PostMoeHop` factory, built once per target MoE
  layer at model construction (collective on first use).
* ``health_check``: :func:`check_health`, the host-side report of the device
  error words and the engagement counter; call it outside timed regions.

Two kernels implement the hop with the same workspace protocol and
arithmetic: a CuTe DSL kernel with one thread block cluster per token row
(``cute_kernel.py``, the default) and a Triton kernel (``kernels.py``) that
serves every shape the cluster kernel does not.
"""

from ... import dispatch
from .health import PostMoeHopError, PostMoeHopHealth, check_health
from .hop import PostMoeHop, PostMoeHopContractError, native_finalize_reference, producer_capability

dispatch.register("post_moe_hop", "post_moe_hop", PostMoeHop)
dispatch.register("post_moe_hop", "health_check", check_health)

__all__ = [
    "PostMoeHopContractError",
    "PostMoeHopHealth",
    "PostMoeHopError",
    "PostMoeHop",
    "check_health",
    "native_finalize_reference",
    "producer_capability",
]
