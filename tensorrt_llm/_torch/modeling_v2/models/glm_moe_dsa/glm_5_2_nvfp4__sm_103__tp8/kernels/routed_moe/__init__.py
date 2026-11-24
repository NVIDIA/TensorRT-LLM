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
"""routed_moe: routed NVFP4 MoE (the routed_moe contract) as one CuTe DSL launch.

Registered entry points (``dispatch.get_impl("routed_moe", name)``):

* ``routed_output(module, routing_bias, x, router_logits, do_finalize)`` ->
  finalized ``[R, 6144]`` BF16, the native unfinalized triple, or ``None``
  when the call is ineligible (the caller then runs the native path; the
  decision is taken before any device work). ``module`` is the served
  ``ConfigurableMoE`` (``Deepseekv3MoE.experts``) or its TRT-LLM-Gen NVFP4
  leaf.
* ``eligibility(module, x, router_logits)`` -> ``(bool, reason)``.
* ``routed_output_into_hop(module, routing_bias, x, router_logits, hop)``
  (used only when the ``routed_moe_into_hop`` feature is selected): the finalized routed
  output written into the post_moe_hop's routed-input buffer, as ``[routed]``, or
  ``None``.
"""

from ... import dispatch
from .routed_moe import eligibility, routed_output, routed_output_into_hop

dispatch.register("routed_moe", "routed_output", routed_output)
dispatch.register("routed_moe", "eligibility", eligibility)
dispatch.register("routed_moe", "routed_output_into_hop", routed_output_into_hop)

__all__ = ["eligibility", "routed_output", "routed_output_into_hop"]
