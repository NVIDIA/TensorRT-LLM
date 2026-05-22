# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""A set of transforms to handle SSM cache transforms."""

from ..interface import TransformRegistry
from .kvcache import _InsertCachedOperator


# TODO: think about separating valid attention backends per transform better in the future
@TransformRegistry.register("insert_cached_ssm_attention")
class SSMCacheTransform(_InsertCachedOperator):
    """A transform to handle SSM cache operations."""


@TransformRegistry.register("insert_cached_causal_conv")
class InitializeCausalConvCache(_InsertCachedOperator):
    """A transform to handle causal conv cache operations."""


@TransformRegistry.register("insert_cached_delta_rule")
class InsertCachedDeltaRule(_InsertCachedOperator):
    """A transform to handle delta rule cache operations."""


@TransformRegistry.register("insert_cached_gated_delta_rule")
class InsertCachedGatedDeltaRule(_InsertCachedOperator):
    """A transform to handle gated delta rule cache operations."""
