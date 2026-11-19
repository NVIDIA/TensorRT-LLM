# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resource definitions for the BatchedGemm TS example.

This module is intentionally kept as a public facade. Concrete resource
implementations live in the domain-specific modules next to this file:
GMEM coordinate/output resources, SMEM staging/barrier resources, and TMEM
scale/accumulator resources.
"""

from .gmem_ab_resources import (
    GmemAResource,
    GmemBResource,
)
from .gmem_sf_resources import (
    GmemSfAResource,
    GmemSfBResource,
)
from .gmem_c_resources import (
    GmemCResource,
)
from .smem_ab_resources import (
    SmemAResource,
    SmemBResource,
    SmemGatherResource,
    SmemTmaGatherResource,
)
from .smem_sf_resources import (
    SmemSfAResource,
    SmemSfBResource,
    SmemSfGatherResource,
    SmemSfGatherAResource,
    SmemSfGatherBResource,
    SmemSfLdgstsResource,
    SmemSfLdgstsAResource,
    SmemSfLdgstsBResource,
)
from .smem_misc_resources import (
    BatchedGemmWorkQueue,
    ProxyClusterBarrierResource,
    WorkThrottleBarrierResource,
)
from .smem_deepseek_sf_resources import (
    SmemDeepSeekSfAbResource,
)
from .tmem_sf_resources import (
    TmemCastAResource,
    TmemSfAResource,
    TmemSfABResource,
    TmemSfBResource,
    TmemSfRouteResource,
    TmemSfRouteAResource,
    TmemSfRouteBResource,
)
from .tmem_c_resources import (
    TmemCResource,
)

__all__ = [
    "BatchedGemmWorkQueue",
    "GmemAResource",
    "GmemBResource",
    "GmemCResource",
    "GmemSfAResource",
    "GmemSfBResource",
    "ProxyClusterBarrierResource",
    "SmemAResource",
    "SmemBResource",
    "SmemDeepSeekSfAbResource",
    "SmemGatherResource",
    "SmemSfAResource",
    "SmemSfBResource",
    "SmemSfGatherResource",
    "SmemSfGatherAResource",
    "SmemSfGatherBResource",
    "SmemSfLdgstsResource",
    "SmemSfLdgstsAResource",
    "SmemSfLdgstsBResource",
    "SmemTmaGatherResource",
    "TmemCastAResource",
    "TmemCResource",
    "TmemSfAResource",
    "TmemSfABResource",
    "TmemSfBResource",
    "TmemSfRouteResource",
    "TmemSfRouteAResource",
    "TmemSfRouteBResource",
    "WorkThrottleBarrierResource",
]
