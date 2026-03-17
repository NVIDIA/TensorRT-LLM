# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the vision GraphModule to ONNX (run_per_gm=true, only exports the visual GM).

When the pipeline has both text and vision submodules (VLM), export_to_gm produces
multiple GraphModules. This transform runs per-GM and exports only the vision GM
(identified by placeholders ``hidden_states`` and ``grid_thw``, e.g. Qwen3-VL)
to ``visual_output_dir/vision_model.onnx``. Non-vision GMs are skipped.
"""

from pathlib import Path
from typing import Optional, Tuple, Type

import torch
from pydantic import Field
from torch.export import Dim
from torch.fx import GraphModule

from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import sync_weight_meta_dtype
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from . import _onnx_schemas


def _is_vision_gm(gm: GraphModule) -> bool:
    """Return True if this GraphModule is the vision encoder (e.g. Qwen3-VL visual)."""
    names = {ph.name for ph in gm.graph.find_nodes(op="placeholder")}
    # Qwen3-VL visual forward(hidden_states, grid_thw); text has inputs_embeds, past_key_values.
    if "grid_thw" in names and "hidden_states" in names and "past_key_values" not in names:
        return True
    return False


class ExportVisionToONNXConfig(TransformConfig):
    """Configuration for exporting the vision GraphModule to ONNX."""

    visual_output_dir: Path = Field(
        description="Directory to save the exported vision ONNX model (e.g. vision_model.onnx).",
    )


@TransformRegistry.register("export_vision_to_onnx")
class ExportVisionToONNX(BaseTransform):
    """Export only the vision submodule to ONNX when run_per_gm=true.

    Skips non-vision GMs. Vision GM is detected by placeholders (hidden_states, grid_thw).
    """

    config: ExportVisionToONNXConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ExportVisionToONNXConfig

    def _apply(
        self,
        gm: GraphModule,
        _cm: CachedSequenceInterface,
        _factory: Optional[object],
        _shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not _is_vision_gm(gm):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        out_dir = self.config.visual_output_dir
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        sync_weight_meta_dtype(gm)

        placeholders = gm.graph.find_nodes(op="placeholder")
        kwargs = {ph.name: ph.meta["val"] for ph in placeholders}

        batch_dim = Dim("batch_size", min=0, max=64)
        dynamic_shapes = {
            "hidden_states": {
                0: batch_dim,
                1: Dim.DYNAMIC,
                2: Dim.DYNAMIC,
                3: Dim.DYNAMIC,
            },
            "grid_thw": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        }
        # Only include keys that exist in this graph
        dynamic_shapes = {k: v for k, v in dynamic_shapes.items() if k in kwargs}

        output_path = out_dir / "vision_model.onnx"
        output_node = gm.graph.find_nodes(op="output")[0]
        outputs = output_node.args[0]
        output_names = [t.name for t in outputs]

        _onnx_schemas.register_onnx_schemas()

        ad_logger.info("Exporting vision GraphModule to ONNX: %s", output_path)
        torch.onnx.export(
            gm,
            (),
            output_path,
            opset_version=20,
            kwargs=kwargs,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            report=False,
            output_names=output_names,
        )
        ad_logger.info("Exported vision ONNX to %s", output_path)

        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )
