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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Order indexer shared reads without changing shared CuTe pipeline classes."""
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import dsl_user_op

_NativePipelineTmaAsync = pipeline.PipelineTmaAsync
_native_release = _NativePipelineTmaAsync.consumer_release


class PipelineTmaAsync(_NativePipelineTmaAsync):
    """Vendor-local pipeline with ordered shared-memory stage release."""

    @staticmethod
    def create(*args, **kwargs):
        result = _NativePipelineTmaAsync.create(*args, **kwargs)
        # The native pipeline is a frozen dataclass; preserve its factory fields.
        object.__setattr__(result, "__class__", PipelineTmaAsync)
        return result

    @dsl_user_op
    def consumer_release(self, state, *, loc=None, ip=None):
        cute.arch.fence_view_async_shared(loc=loc, ip=ip)
        return _native_release(self, state, loc=loc, ip=ip)
