# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this license except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract token embedding weights from the full model and save to safetensors.

This is the single place that writes embedding.safetensors for EdgeLLM export.
run_per_gm=false so we see the full model. Two paths:

1. Top-level API: when mod still has get_input_embeddings() (e.g. VLM after export_to_gm:
   mod is the top-level wrapper, only text/vision submodules were replaced by GraphModules).
   Use mod.get_input_embeddings().weight.

2. GraphModule fallback: when mod is already a single GraphModule (pure LLM after export_to_gm:
   submodule_name was "", so the whole model was replaced by the exported GM; see
   export_to_gm.py). The GM has no get_input_embeddings(). Find aten.embedding.default
   in the graph and get the weight tensor via get_weight_tensor(embedding_node).

This transform must run before rewrite_embedding_to_inputs_embeds so that for pure LLM
the graph still contains the embedding node.
"""

from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule

from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import get_weight_tensor
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class ExtractEmbeddingToSafetensorsConfig(TransformConfig):
    """Configuration for the extract embedding to safetensors transform."""

    output_dir: Path = Field(
        description="The directory to save the extracted embedding weights.",
        default=Path("."),
    )


def _get_embedding_weight_from_module(mod: nn.Module) -> Optional[torch.Tensor]:
    """Get embedding weight via top-level get_input_embeddings() if available."""
    if not hasattr(mod, "get_input_embeddings"):
        return None
    embed = mod.get_input_embeddings()
    if embed is None:
        return None
    weight = getattr(embed, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor):
        return None
    return weight


def _get_embedding_weight_from_graph(gm: GraphModule) -> Optional[torch.Tensor]:
    """Get embedding weight from the first aten.embedding.default node in the graph."""
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.embedding.default:
            try:
                return get_weight_tensor(node)
            except Exception as e:
                ad_logger.debug("get_weight_tensor(embedding_node) failed: %s", e)
                return None
    return None


@TransformRegistry.register("extract_embedding_to_safetensors")
class ExtractEmbeddingToSafetensors(BaseTransform):
    """Extract token embedding weights and write to embedding.safetensors.

    Tries (1) top-level get_input_embeddings().weight, then (2) embedding node in
    the graph when mod is a GraphModule (pure LLM after export_to_gm). See module docstring.
    """

    config: ExtractEmbeddingToSafetensorsConfig

    @classmethod
    def get_config_class(cls):
        """Return the configuration class for this transform."""
        return ExtractEmbeddingToSafetensorsConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        _cm: CachedSequenceInterface,
        _factory: object,
        _shared_config: SharedConfig,
    ) -> tuple[nn.Module, TransformInfo]:
        weight = _get_embedding_weight_from_module(mod)
        if weight is None and isinstance(mod, GraphModule):
            weight = _get_embedding_weight_from_graph(mod)
        if weight is None:
            ad_logger.info(
                "Could not get embedding weight (no get_input_embeddings() and no "
                "embedding node in graph), skipping"
            )
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        output_path = self.config.output_dir / "embedding.safetensors"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file({"weight": weight.detach().cpu()}, output_path)
        ad_logger.info(f"Saved embedding to {output_path} (shape {weight.shape})")
        return mod, TransformInfo(
            skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True
        )
