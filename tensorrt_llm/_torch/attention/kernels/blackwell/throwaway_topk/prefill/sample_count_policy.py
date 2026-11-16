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

"""Same stratified hash; explicit S must match the actual sampled GEMM tiles."""
import cutlass
import cutlass.cute as cute
from .policy_dsv4 import model_sample_key as native_key
from .policy_qwen import sample_hash_u32
@cute.jit
def shape_sample_key(sample_count:cutlass.Constexpr,pos,qb,lower,upper,k:cutlass.Constexpr=512,heads:cutlass.Constexpr=64,coverage:cutlass.Constexpr=8,common:cutlass.Constexpr=False):
    if cutlass.const_expr(sample_count==0):
        return native_key(pos,qb,lower,upper,k,heads,coverage,common)
    else:
        width=max(1,upper-lower)
        h=sample_hash_u32(pos,cutlass.Int32(0) if cutlass.const_expr(common and k==2048) else qb)
        begin=cutlass.Int32(cutlass.Int64(pos)*cutlass.Int64(width)//sample_count)
        finish=cutlass.Int32(cutlass.Int64(pos+1)*cutlass.Int64(width)//sample_count)
        offset=begin+cutlass.Int32((cutlass.Uint64(h)*cutlass.Uint64(max(1,finish-begin)))>>32)
        return max(cutlass.Int32(0),min(upper-1,lower+offset))
