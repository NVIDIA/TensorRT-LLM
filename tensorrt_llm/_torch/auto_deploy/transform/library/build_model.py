"""A simple wrapper transform to build a model via the model factory."""

from typing import Tuple, Type

from pydantic import Field
from torch.fx import GraphModule

from ...models import ModelFactory, hf
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
    use_strict_forward: bool = Field(
        default=True,
        description="If True, the forward pass will be patched to use a strict positional-only list"
        " of arguments. If False, the default with **kwargs can be used.",
    )


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

        assert self.config.use_strict_forward, "Only strict forward is supported."
        factory._set_strict_forward(model)

        # as wrapper to satisfy the interface we will register the model as a submodule
        gm.add_module("factory_model", model)

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info


@TransformRegistry.register("build_and_load_factory_model")
class BuildAndLoadFactoryModel(BuildModel):
    """A simple wrapper transform to build a model via the model factory."""

    config: BuildModelConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # load model with auto sharding
        assert isinstance(factory, hf.AutoModelFactory), "Only HF models are supported."

        # build and load the model
        model = factory.build_and_load_model(self.config.device)

        assert not self.config.use_strict_forward, "Only regular forward is supported."

        # as wrapper to satisfy the interface we will register the model as a submodule
        gm.add_module("factory_model", model)

        # this ensures that extra_args are passed in as they are received instead of enforcing the
        # registered extra_args
        cm.info.use_strict_args = False

        # we set the standard example sequence WITHOUT extra_args to set them to None so that
        # only the text portion of the model gets called.
        cm.info.set_example_sequence()

        # by convention, we say this fake graph module is always clean
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
