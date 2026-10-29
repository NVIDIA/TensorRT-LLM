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
"""dense_mlp: small-M dense MLP of the dense decoder layers.

Registered entry point: ``dense_mlp_forward(mlp, x, final_all_reduce_params,
lora_params) -> Optional[Tensor]`` (``None`` means "run the native module").
"""

from ... import dispatch
from .dense_mlp import check_eligibility, dense_mlp_forward, op_available

dispatch.register("dense_mlp", "dense_mlp_forward", dense_mlp_forward)

__all__ = ["check_eligibility", "dense_mlp_forward", "op_available"]
