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
"""cross_layer_expert_prefetch: cross-layer speculative routed-expert L2 prefetch.

Registered entry points: ``link_layers(layers, num_hidden_layers)`` (from the
model's ``setup_aliases``) and ``join()`` (end of the target model's forward).
Per-layer hooks are methods of the ``ExpertPrefetchLink`` bound to each MoE layer
as ``_expert_prefetch_link``.
"""

from ... import dispatch
from .spec_experts import ExpertPrefetchLink, config, join, link_layers

dispatch.register("cross_layer_expert_prefetch", "link_layers", link_layers)
dispatch.register("cross_layer_expert_prefetch", "join", join)

__all__ = ["ExpertPrefetchLink", "config", "join", "link_layers"]
