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
"""Package draft_routed_moe: the MTP draft layer's routed experts (BF16) at the DRAM rate.

Entry point registered with ``dispatch``:

* ``routed_output(module, routing_bias, x, router_logits, do_finalize)``:
  the finalized routed output ``[R, 6144]`` of the BF16 FlashInfer
  TRT-LLM-Gen leaf, bit-identical to ``trtllm_bf16_moe``, or ``None`` (the
  native path; decided before any device work).
"""

from ...dispatch import register
from .routed_moe import eligibility, launch_config, routed_output

register("draft_routed_moe", "routed_output", routed_output)

__all__ = ["eligibility", "launch_config", "routed_output"]
