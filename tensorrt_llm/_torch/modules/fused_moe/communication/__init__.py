# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
MoE Communication Strategies Module

This module provides various communication strategies for expert parallelism in MoE models.

Available Communication Methods:
- AllGatherReduceScatter: Default fallback method, always available
- NVLinkTwoSided: NVLINK-optimized communication for latency (formerly MNNVLLatency)
- NVLinkOneSided: NVLINK-optimized communication for throughput (formerly MNNVLThroughput)
- DeepEP: Deep Expert Parallelism with support for large batches
- DeepEPLowLatency: Deep Expert Parallelism optimized for low latency

Factory:
- CommunicationFactory: Automatically selects the best communication method
"""

from .allgather_reducescatter import AllGatherReduceScatter
from .base import Communication
from .communication_factory import CommunicationFactory
from .deep_ep import DeepEP
from .deep_ep_low_latency import DeepEPLowLatency
from .nvlink_one_sided import NVLinkOneSided
from .nvlink_two_sided import NVLinkTwoSided

__all__ = [
    # Base classes and types
    "Communication",
    # Communication strategies
    "AllGatherReduceScatter",
    "NVLinkTwoSided",
    "NVLinkOneSided",
    "DeepEP",
    "DeepEPLowLatency",
    # Factory
    "CommunicationFactory",
]
