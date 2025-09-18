"""A simple wrapper transform to build a model via the model factory."""

from typing import Optional, Tuple, Type

from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...transformations._graph import move_to_device
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class MoveDeviceConfig(TransformConfig):
    """Configuration for the moving inputs/arguments to the device transform."""

    device: str = Field(default="meta", description="The device to load the weights on.")
    adconfig_checkpoint_device: Optional[str] = Field(
        default=None, description="Optional checkpoint device argument from adconfig."
    )


@TransformRegistry.register("load_weights")
class LoadWeightsToDevice(BaseTransform):
    """A simple wrapper transform to load weights into a model."""

    config: MoveDeviceConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MoveDeviceConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        factory.load_or_random_init(
            gm,
            device=self.config.adconfig_checkpoint_device or self.config.device,
        )
        move_to_device(gm, self.config.device)

        info = TransformInfo(skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True)

        return gm, info


@TransformRegistry.register("move_inputs_to_device")
class LoadFactoryModelWeights(BaseTransform):
    """Wrapper transform to move all inputs/arguments to the device."""

    config: MoveDeviceConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MoveDeviceConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cm.to(self.config.device)

        info = TransformInfo(skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True)

        return gm, info
