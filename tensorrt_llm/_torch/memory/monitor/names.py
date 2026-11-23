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
"""Names that call sites and the report must spell the same way.

Region names are merged by name in the overhead row's breakdown, and tag
labels decide which row memory lands in, so each is defined once here and
imported by every site that uses it.
"""

__all__ = [
    "CUDA_GRAPHS",
    "REGION_KV_TRANSFER_BUFFERS",
    "REGION_LORA_DEVICE_CACHE",
    "REGION_TORCH_DISTRIBUTED_NCCL",
    "UNTAGGED",
    "WEIGHTS",
]

#: Label of the loaded model's parameters and buffers, claimed at load time.
WEIGHTS = "WEIGHTS"
#: Label for live blocks in a CUDA graph's private pool. Applied from the
#: pool, not from a tag: capture happens inside a creation step whose tag
#: would otherwise claim the graph's memory as that step's.
CUDA_GRAPHS = "CUDA_GRAPHS"
#: Label for live memory that no frozen tag claims. Always reported.
UNTAGGED = "untagged"

#: Communicators ``torch.distributed`` creates, which allocate outside PyTorch.
REGION_TORCH_DISTRIBUTED_NCCL = "NCCL communicators (torch.distributed)"
#: Buffers the KV cache transceiver allocates for disaggregated serving.
REGION_KV_TRANSFER_BUFFERS = "KV cache transfer buffers"
#: The LoRA adapters' device cache.
REGION_LORA_DEVICE_CACHE = "LoRA device cache"
