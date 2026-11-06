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

"""Synchronize a subset of warp-specialized CTA participants at distinct PCs.

Use barrier.cta.sync; aligned bar.sync requires uniform participation."""
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm

@dsl_user_op
def subset_barrier(*,barrier_id,number_of_threads,loc=None,ip=None):
    bid=cutlass.Int32(barrier_id).ir_value(loc=loc,ip=ip)
    count=cutlass.Int32(number_of_threads).ir_value(loc=loc,ip=ip)
    nvvm.barrier_cta_sync(bid,thread_count=count,aligned=False,loc=loc,ip=ip)
