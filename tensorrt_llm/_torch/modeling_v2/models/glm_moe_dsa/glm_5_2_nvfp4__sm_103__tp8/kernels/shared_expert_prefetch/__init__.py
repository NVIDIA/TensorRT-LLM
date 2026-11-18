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
"""Package shared_expert_prefetch: the MoE region END (routed END and shared END).

Entry point registered with ``dispatch``:

* ``routed_output_into_hop(module, bias, x, logits, routing, hop, shared)``:
  hop_routing's routed_moe fork as ``SharedPrefetchExpertClusterKernel`` (``routed_moe.py``,
  ``kernel_xc.py``). Before its dependency wait, while the pre-MoE hop runs, it prefetches the
  shared expert's weights into L2, so the native shared expert on the auxiliary stream
  reads them from L2 instead of competing with the routed kernel for DRAM.
  ``None`` means "run hop_routing's fork" (decided before any device work).

The shared expert itself stays native (``GatedMLP``); shared_expert_prefetch has no numerical
variant: given the same routing its expert rows are routed_moe's and its FP32
accumulator is in routed_moe_into_hop's ``fp32`` class.
"""

from ...dispatch import register
from .routed_moe import routed_output_into_hop

register("shared_expert_prefetch", "routed_output_into_hop", routed_output_into_hop)

__all__ = ["routed_output_into_hop"]
