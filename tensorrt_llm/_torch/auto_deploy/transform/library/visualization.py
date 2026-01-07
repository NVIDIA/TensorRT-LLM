"""Transformation to the graph to render nicely in model_explorer."""

from typing import Tuple

import torch.export as te
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

try:
    import model_explorer
except ImportError:
    model_explorer = None


@TransformRegistry.register("visualize_namespace")
class VisualizeNamespace(BaseTransform):
    """Transform to visualize the graph using Model Explorer.

    This transform exports the graph module to an ExportedProgram and launches
    Model Explorer for interactive visualization. The visualization helps debug
    and understand the graph structure after AutoDeploy transformations.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Export the graph and launch Model Explorer for visualization.

        Args:
            gm: The graph module to visualize.
            cm: The cached sequence interface with input arguments.
            factory: The model factory (unused).
            shared_config: Shared configuration across transforms (unused).

        Returns:
            A tuple of the unchanged graph module and transform info indicating
            whether visualization was successful or skipped.
        """
        if model_explorer is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        try:
            # Export graph module to ExportedProgram for visualization
            exported_program = te.export(gm, args=(), kwargs=cm.named_args, dynamic_shapes=None)

            ad_logger.info("Launching Model Explorer visualization...")
            model_explorer.visualize_pytorch("model-viz", exported_program)

            return gm, TransformInfo(
                skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True
            )

        except Exception as e:
            ad_logger.error(f"Failed to visualize graph with Model Explorer: {e}")
            # Don't fail the pipeline if visualization fails
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
