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

"""Resources for the throughput-latency 1CTA MLA task schedule."""

from .common import MlaResource, ScheduleTokenThrottleResource
from .smem_p import SmemPResource
from .smem_resources import (
    SmemKResource,
    SmemKvResource,
    SmemPageOffsetsResource,
    SmemQResource,
    SmemVResource,
)
from .tmem_corr import TmemCorrResource
from .tmem_o import TmemOResource
from .tmem_p import TmemPResource
from .tmem_s import TmemSKeepsResource, TmemSResource
from .tmem_softmax_stats import (
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
)

__all__ = [
    "MlaResource",
    "ScheduleTokenThrottleResource",
    "SmemKResource",
    "SmemKvResource",
    "SmemPageOffsetsResource",
    "SmemQResource",
    "SmemVResource",
    "SmemPResource",
    "TmemCorrResource",
    "TmemOResource",
    "TmemPResource",
    "TmemSKeepsResource",
    "TmemSResource",
    "TmemSoftmaxGlobalResource",
    "TmemSoftmaxLocalResource",
]
