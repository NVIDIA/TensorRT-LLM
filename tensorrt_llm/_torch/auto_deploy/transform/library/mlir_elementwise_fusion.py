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

"""MLIR-based unified elementwise fusion transform.

Runs the full decompose -> discover -> codegen -> replace pipeline:
1. FX -> MLIR conversion
2. Decompose high-level ops (e.g. RMSNorm) into elementwise primitives
3. Discover maximal fusible subgraphs among the primitives
4. Generate Triton kernels for each subgraph
5. Replace subgraph ops in MLIR with fused opaque ops
6. MLIR -> FX conversion (produces graph with generated kernel calls)
"""

from typing import List, Tuple, Type

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


class MLIRElementwiseFusionConfig(TransformConfig):
    """Configuration for the MLIR elementwise fusion transform."""

    bypass_ops: List[str] = Field(
        default_factory=list,
        description="Op names to skip during decomposition (reserved for future use).",
    )


@TransformRegistry.register("mlir_elementwise_fusion")
class MLIRElementwiseFusion(BaseTransform):
    """Unified MLIR elementwise fusion: decompose + discover + codegen + replace.

    This transform:
    1. Converts the FX graph to MLIR (xDSL) using the ``ad`` dialect
    2. Decomposes high-level ops into elementwise primitives
    3. Discovers maximal fusible subgraphs
    4. Generates Triton kernels for each discovered subgraph
    5. Replaces subgraph ops in MLIR with fused opaque ops
    6. Converts MLIR back to FX with generated kernel calls

    Requires ``pip install xdsl``. Skipped gracefully if xDSL is not installed.
    """

    config: MLIRElementwiseFusionConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MLIRElementwiseFusionConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not HAS_XDSL:
            self._log_warning("xDSL not installed, skipping MLIR elementwise fusion")
            return gm, TransformInfo(skipped=True)

        from ...mlir.codegen.kernel_cache import KernelCache
        from ...mlir.codegen.triton_emitter import generate_kernel_from_subgraph
        from ...mlir.decompose import run_decomposition
        from ...mlir.fusion.subgraph_discovery import discover_fusible_subgraphs
        from ...mlir.fusion.subgraph_replace import replace_subgraph_with_fused_op
        from ...mlir.fx_to_mlir import FXToMLIRConverter
        from ...mlir.mlir_to_fx import MLIRToFXConverter

        # Step 1: FX -> MLIR
        converter = FXToMLIRConverter(gm)
        mlir_module = converter.convert()

        # Step 2: Decompose high-level ops into primitives
        num_decomposed = run_decomposition(mlir_module)
        self._log_info(f"Decomposed {num_decomposed} high-level ops into primitives")

        # Step 3: Discover fusible subgraphs
        subgraphs = discover_fusible_subgraphs(mlir_module)
        self._log_info(
            f"Discovered {len(subgraphs)} fusible subgraphs "
            f"(total ops: {sum(len(sg.ops) for sg in subgraphs)})"
        )

        if not subgraphs:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Step 4: Generate Triton kernels and replace subgraphs in MLIR.
        # Skip subgraphs where all inputs are 1D or lower — these are pure
        # weight-space ops (e.g., weight + 1.0) that don't benefit from fusion
        # and the row-based Triton kernel can't handle them correctly.
        from xdsl.dialects.builtin import TensorType as _TT

        def _max_input_rank(sg):
            return max(
                (len(inp.type.get_shape()) for inp in sg.inputs if isinstance(inp.type, _TT)),
                default=0,
            )

        num_replaced = 0
        num_skipped = 0
        for sg in subgraphs:
            if _max_input_rank(sg) < 2:
                num_skipped += 1
                continue
            try:
                kernel_fn = generate_kernel_from_subgraph(sg)
                if kernel_fn is not None:
                    sg_hash = KernelCache.hash_subgraph(sg)
                    replace_subgraph_with_fused_op(sg, kernel_fn, sg_hash, converter.metadata)
                    num_replaced += 1
            except Exception as e:
                self._log_warning(f"Failed to fuse subgraph: {e}")

        self._log_info(
            f"Replaced {num_replaced}/{len(subgraphs)} subgraphs (skipped {num_skipped} low-rank)"
        )

        if num_replaced == 0:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Step 5: MLIR -> FX (with fused ops becoming kernel calls)
        back_converter = MLIRToFXConverter(gm, codegen_mode="generate")
        new_gm = back_converter.convert(mlir_module, converter.metadata)

        return new_gm, TransformInfo(
            skipped=False,
            num_matches=num_replaced,
            is_clean=False,
            has_valid_shapes=False,
        )
