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
"""The fmha_v2 FMHA library of the native attention op."""

from tensorrt_llm.bindings.internal import thop

from .native import NativeFmha
from .phased import FmhaParams


class FmhaV2Fmha(NativeFmha):
    """fmha_v2 kernels: context FMHA, and MLA generation over the paged latent cache."""

    SERVES_CONTEXT = True
    SERVES_GENERATION = True

    @staticmethod
    def native_is_supported(params: FmhaParams, is_gen: bool) -> bool:
        return thop.fmha_v2_is_supported(params, is_gen)

    @staticmethod
    def native_run_context(params: FmhaParams) -> None:
        thop.fmha_v2_run_context(params)

    @staticmethod
    def native_run_generation(params: FmhaParams) -> None:
        thop.fmha_v2_run_generation(params)


__all__ = ["FmhaV2Fmha"]
