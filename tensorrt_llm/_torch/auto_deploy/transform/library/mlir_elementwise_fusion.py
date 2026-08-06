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

        from ...mlir.decompose import run_decomposition
        from ...mlir.fusion.fuse import run_fusion
        from ...mlir.fx_to_mlir import FXToMLIRConverter
        from ...mlir.mlir_to_fx import MLIRToFXConverter

        # Step 1: FX -> MLIR
        converter = FXToMLIRConverter(gm)
        try:
            mlir_module = converter.convert()
        except ValueError as e:
            # Gracefully skip if the graph contains unsupported dtypes (e.g.
            # complex64 in rotary embeddings) or other FX→MLIR conversion issues.
            self._log_warning(f"Skipping MLIR fusion (FX→MLIR conversion failed): {e}")
            return gm, TransformInfo(skipped=True)

        # Step 2: Decompose high-level ops into primitives
        num_decomposed = run_decomposition(mlir_module)
        self._log_info(f"Decomposed {num_decomposed} high-level ops into primitives")

        # Step 3+4: Discover fusible subgraphs and replace each with a generated
        # fused op.  Both decomposition and fusion now go through xDSL's
        # PatternRewriteWalker (see run_decomposition / run_fusion).
        stats = run_fusion(mlir_module, converter.metadata, log_warning=self._log_warning)
        self._log_info(
            f"Discovered {stats.num_subgraphs} fusible subgraphs; "
            f"replaced {stats.num_replaced} (skipped {stats.num_skipped_low_rank} low-rank)"
        )

        if stats.num_replaced == 0:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Step 5: MLIR -> FX (with fused ops becoming kernel calls)
        back_converter = MLIRToFXConverter(gm)
        new_gm = back_converter.convert(mlir_module, converter.metadata)

        # Step 6: Defer graph cleanup and shape-prop to the framework. Running
        # fake-tensor shape-prop here is unsafe at post_load_fusion: the graph
        # may be in a deliberately invalid intermediate state (e.g.
        # fuse_rope_into_trtllm_attention rewires Q/K/V to a fused-QKV tensor
        # whose rank does not match torch_attention.register_fake; the op swap
        # happens later at cache_init).
        return new_gm, TransformInfo(
            skipped=False,
            num_matches=stats.num_replaced,
            is_clean=False,
            has_valid_shapes=False,
        )
