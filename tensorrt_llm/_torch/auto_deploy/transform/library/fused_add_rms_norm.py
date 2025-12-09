# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformation for fusing Add + Cast + RMSNorm."""

from typing import Tuple

import torch
from torch.fx import GraphModule

from ...custom_ops.flashinfer_fused_add_rms_norm import flashinfer_fused_add_rms_norm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("fuse_add_rms_norm")
class FuseAddRMSNorm(BaseTransform):
    """Fuse (add + cast + RMSNorm) into one fused op.

    Matches:
        x = add(input, residual)
        y = x.to(dtype)
        z = flashinfer_rms_norm(y, weight, eps)

    Replaces with:
        z, x = flashinfer_fused_add_rms_norm(input, residual, weight, eps)
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        patterns = ADPatternMatcherPass()

        # Dummy shapes for tracing
        bsz, hidden = 2, 128
        dummy_args = [
            torch.randn(bsz, hidden, device="meta", dtype=torch.bfloat16),  # x (bf16)
            torch.randn(bsz, hidden, device="meta", dtype=torch.bfloat16),  # residual (bf16)
            torch.randn(hidden, device="meta", dtype=torch.bfloat16),  # weight
            1e-5,  # eps
        ]

        op_ignore_types = {torch.ops.aten.to.dtype: (torch.dtype,)}
        scalar_workaround = {"eps": 1e-5}

        def _fused_add_norm_pattern(x, residual, weight, eps):
            added = torch.ops.aten.add.Tensor(x, residual)
            cast = torch.ops.aten.to.dtype(added, torch.bfloat16)
            # Note: we assume flashinfer_rms_norm is the target
            norm = torch.ops.auto_deploy.flashinfer_rms_norm.default(cast, weight, eps)
            return norm, added

        def _fused_add_norm_replacement(x, residual, weight, eps):
            # Use the python wrapper directly, not via torch.ops.auto_deploy
            return flashinfer_fused_add_rms_norm(x, residual, weight, eps)

        # Register pattern
        register_ad_pattern(
            search_fn=_fused_add_norm_pattern,
            replace_fn=_fused_add_norm_replacement,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types=op_ignore_types,
            scalar_workaround=scalar_workaround,
        )

        num_matches = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
