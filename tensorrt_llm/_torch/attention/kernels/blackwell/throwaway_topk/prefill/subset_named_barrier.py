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

import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import dsl_user_op
from .subset_barrier import subset_barrier

class SubsetNamedBarrier(pipeline.NamedBarrier):
    @dsl_user_op
    def arrive_and_wait(self,*,loc=None,ip=None):
        subset_barrier(barrier_id=self.barrier_id,number_of_threads=self.num_threads,loc=loc,ip=ip)

    @dsl_user_op
    def arrive(self,*,loc=None,ip=None):
        cute.arch.barrier_arrive(barrier_id=self.barrier_id,number_of_threads=self.num_threads,aligned=False,loc=loc,ip=ip)

    @dsl_user_op
    def wait_unaligned(self,*,loc=None,ip=None):
        subset_barrier(barrier_id=self.barrier_id,number_of_threads=self.num_threads,loc=loc,ip=ip)
