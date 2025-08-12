"""A simple wrapper transform to export a model to a graph module."""

from typing import List, Optional, Tuple, Type

from pydantic import Field
from torch.fx import GraphModule

from ...export import torch_export_to_gm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class ExportToGMConfig(TransformConfig):
    """Configuration for the export to graph module transform."""

    strict: bool = Field(
        description="Whether to export in strict mode. NOTE: we generally export in non-strict mode"
        "for now as it relaxes some assumptions around tracing. Strict mode uses torchdynamo"
        "(symbolic bytecode analysis), which can be brittle since it relies on the exact bytecode"
        "representation of the model see here as well: https://pytorch.org/docs/stable/export.html#non-strict-export",
        default=False,
    )
    clone_state_dict: bool = Field(
        description="Whether to clone the state_dict of the model. This is useful to avoid"
        "modifying the original state_dict of the model.",
        default=False,
    )
    patch_list: Optional[List[str]] = Field(
        description="List of patch names to apply with export. "
        "Default is to apply all registered patches.",
        default=None,
    )


@TransformRegistry.register("export_to_gm")
class ExportToGM(BaseTransform):
    """A simple wrapper transform to export a model to a graph module."""

    config: ExportToGMConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ExportToGMConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # at this point we assume the gm is just a dummy graph module
        assert len(gm.graph.nodes) == 0, "Expected empty graph module."

        # retrieve the actual model from the dummy graph module
        model = gm.get_submodule("factory_model")

        # set the example sequence
        cm.info.set_example_sequence()

        # export the model to a graph module
        gm = torch_export_to_gm(
            model,
            args=cm.args,
            dynamic_shapes=cm.dynamic_shapes,
            clone=self.config.clone_state_dict,
            strict=self.config.strict,
            patch_list=self.config.patch_list,
        )

        # this is a clean graph by definition since it was just exported
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
