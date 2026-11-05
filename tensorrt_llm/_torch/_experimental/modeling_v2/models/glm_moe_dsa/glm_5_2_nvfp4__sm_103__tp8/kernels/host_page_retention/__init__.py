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
"""host_page_retention: rank-skew reduction at the decode hops by removing its host-side cause.

On unless ``TRTLLM_STAIRCASE_GLM_LL_DISABLE`` names it or a path it needs
(``post_attention_hop``). Entry point (``dispatch.get_impl``):

* ``host_policy``: called once per rank after warmup; stops the rank's CPU
  allocator from returning freed pages to the OS while it serves, so host
  work during a decode step does not invalidate the GPU's address
  translations (``host_policy`` module docstring).
"""

from ...dispatch import register
from .host_policy import install_host_policy

register("host_page_retention", "host_policy", install_host_policy)

__all__ = ["install_host_policy"]
