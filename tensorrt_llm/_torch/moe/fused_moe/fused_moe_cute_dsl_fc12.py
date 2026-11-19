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
"""``CuteDslFc12FusedMoE``: :class:`.cutedsl.TrtllmCutedslFusedFc12Nvfp4Impl`.

The implementation lives in :mod:`.cutedsl.fused_fc12`. An alias rather than a
base class: the ``moe_backend="CUTEDSL_FC12"`` call sites and the ``issubclass``
dispatch in ``create_moe.py`` mean the leaf itself.
"""

from .cutedsl import TrtllmCutedslFusedFc12Nvfp4Impl

CuteDslFc12FusedMoE = TrtllmCutedslFusedFc12Nvfp4Impl

__all__ = ["CuteDslFc12FusedMoE"]
