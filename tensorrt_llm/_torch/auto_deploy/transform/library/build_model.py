"""A simple wrapper transform to build a model via the model factory."""

import types
from typing import Tuple, Type

from pydantic import Field
from torch.fx import GraphModule
from transformers import AutoModelForCausalLM

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class BuildModelConfig(TransformConfig):
    """Configuration for the build model transform."""

    device: str = Field(default="meta", description="The device to build the model on.")


@TransformRegistry.register("build_model")
class BuildModel(BaseTransform):
    """A simple wrapper transform to build a model via the model factory."""

    config: BuildModelConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return BuildModelConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # build the model
        model = factory.build_model(self.config.device)

        # as wrapper to satisfy the interface we will register the model as a submodule
        gm.add_module("factory_model", model)

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info


@TransformRegistry.register("build_and_load_factory_model")
class BuildAndLoadFactoryModel(BaseTransform):
    """A simple wrapper transform to build a model via the model factory."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # load model with auto sharding
        model = AutoModelForCausalLM.from_pretrained(
            factory.model,
            trust_remote_code=True,
            tp_plan="auto",
            torch_dtype="auto",
        )

        # patch forward method
        model.original_forward = model.forward
        model.forward = types.MethodType(factory._simple_forward, model)
        model.eval()

        # as wrapper to satisfy the interface we will register the model as a submodule
        gm.add_module("factory_model", model)

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
