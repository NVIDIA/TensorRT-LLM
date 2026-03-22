# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MLIR-based transformation for fusing Add + (optional Cast) + RMSNorm.

This is the MLIR equivalent of ``fuse_add_rms_norm.py``, using xDSL's
pattern rewriter infrastructure instead of direct FX graph manipulation.

Only one of ``fuse_add_rms_norm`` or ``mlir_fuse_add_rms_norm`` should be
enabled at a time.
"""

from typing import Literal, Tuple, Type

from pydantic import Field
from torch.fx import GraphModule

from ...mlir import HAS_XDSL
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class MLIRFuseAddRMSNormConfig(TransformConfig):
    """Configuration for the MLIR fuse_add_rms_norm transform."""

    codegen_mode: Literal["preexisting", "generate"] = Field(
        default="preexisting",
        description="Codegen mode: 'preexisting' maps to existing FlashInfer ops, "
        "'generate' emits Triton kernels from templates.",
    )


@TransformRegistry.register("mlir_fuse_add_rms_norm")
class MLIRFuseAddRMSNorm(BaseTransform):
    """Fuse (add + optional cast + RMSNorm) using MLIR pattern rewriting.

    This transform:
    1. Converts the FX graph to MLIR (xDSL) using the ``ad`` dialect
    2. Applies the ``FuseAddRMSNormPattern`` via xDSL's greedy rewrite driver
    3. Converts the fused MLIR back to FX, mapping fused ops to kernels

    Requires ``pip install xdsl``. Skipped gracefully if xDSL is not installed.
    Coexists with the FX-based ``fuse_add_rms_norm`` — only enable one at a time.
    """

    config: MLIRFuseAddRMSNormConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MLIRFuseAddRMSNormConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not HAS_XDSL:
            self._log_warning("xDSL not installed, skipping MLIR transform")
            return gm, TransformInfo(skipped=True)

        from ...mlir.fx_to_mlir import FXToMLIRConverter
        from ...mlir.mlir_to_fx import MLIRToFXConverter
        from ...mlir.patterns import run_fusion_patterns

        # Step 1: FX → MLIR
        converter = FXToMLIRConverter(gm)
        mlir_module = converter.convert()

        # Step 2: Apply MLIR fusion patterns
        num_matches = run_fusion_patterns(mlir_module)

        if num_matches == 0:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Step 3: MLIR → FX
        back_converter = MLIRToFXConverter(gm, codegen_mode=self.config.codegen_mode)
        new_gm = back_converter.convert(mlir_module, converter.metadata)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=False,
        )
        return new_gm, info
