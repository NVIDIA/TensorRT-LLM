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
"""Token-sharded tensor parallelism for VisualGen (``parallel_config.tp_layout='token_sharded'``).

``token_sharded_tp`` holds the token plan, the collectives and the model's sharder,
``token_sharded_modules`` the adapters that convert a model's tensor-parallel modules and
``token_sharded_ce_gather`` the copy-engine all-gather over a symmetric-memory pool and the
body of the fused ``trtllm::token_sharded_fp8_ce_gather_gemm`` op, and
``token_sharded_ce_reduce_scatter`` the copy-engine reduce-scatter over its own pool, its
fixed-order fp32 reduce and the body of the fused
``trtllm::token_sharded_fp8_ce_gemm_reduce_scatter`` op.
"""
