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
"""CuteDSL fused FC12: FC1 and FC2 in one persistent kernel, one leaf per quant.

Importing this package registers both leaves.
"""

from .mxfp8 import TrtllmCutedslFusedFc12Mxfp8Impl
from .nvfp4 import TrtllmCutedslFusedFc12Nvfp4Impl

__all__ = [
    "TrtllmCutedslFusedFc12Mxfp8Impl",
    "TrtllmCutedslFusedFc12Nvfp4Impl",
]
