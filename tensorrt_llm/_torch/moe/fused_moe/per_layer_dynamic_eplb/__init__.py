# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-owned lifecycle for per-layer dynamic expert load balancing.

The standard EPLB implementation couples route updates to host weight
migration. Per-layer dynamic EPLB instead produces one device-side replica plan
for every MoE invocation and copies helper weights on a CUDA stream. It keeps
the existing load-balancer seams: ``manager`` owns the layers and their GPU
resources for the whole model, ``layer`` is the per-layer handle that plans each
route and releases its plan, ``arena`` allocates the live-weight storage,
``planner`` submits HALO-Q and the TMA copy of its plan, and ``halo_r_planner``
submits HALO-R and its relay copy.
"""

from .layer import PerLayerDynamicMoeLayerLoadBalancer
from .manager import PerLayerDynamicMoeLoadBalancer

__all__ = ["PerLayerDynamicMoeLayerLoadBalancer", "PerLayerDynamicMoeLoadBalancer"]
