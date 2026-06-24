# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRT-LLM custom kernel backend for sampling.

Houses sampling kernels implemented natively in TRT-LLM (registered under
``torch.ops.trtllm``), as opposed to pure-PyTorch fallbacks (backend_torch)
or third-party fused kernels (backend_flashinfer).

Backend implementation module: no imports from sampling_utils, backend_torch, or backend_flashinfer.
"""

from typing import Optional

import torch


def compute_probs_from_logits_op(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Delegate to the TRT-LLM C++ op (CUDA, no flashinfer path)."""
    # The C++ op keeps a skip_temperature flag; temperature is always applied here.
    return torch.ops.trtllm.compute_probs_from_logits_op(logits, temperatures, top_k, top_p, False)
