"""A simple wrapper transform to build a model via the model factory."""

from typing import Optional, Tuple, Type

import torch.nn as nn
from pydantic import Field

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import move_to_device
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class MoveDeviceConfig(TransformConfig):
    """Configuration for the moving inputs/arguments to the device transform."""

    checkpoint_device: Optional[str] = Field(
        default=None,
        description="Optional device to init checkpoint before move to shared_config.local_device.",
    )


@TransformRegistry.register("load_weights")
class LoadWeightsToDevice(BaseTransform):
    """A simple wrapper transform to load weights into a model."""

    config: MoveDeviceConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MoveDeviceConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        factory.load_or_random_init(
            mod,
            device=self.config.checkpoint_device or cm.device,
        )
        move_to_device(mod, cm.device)

        info = TransformInfo(skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True)

        return mod, info


@TransformRegistry.register("move_inputs_to_device")
class LoadFactoryModelWeights(BaseTransform):
    """Wrapper transform to move all inputs/arguments to the device."""

    config: MoveDeviceConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MoveDeviceConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # TODO (hg) This is weird but equivalent to previous code.
        # We does not seems to need this transform.
        cm.to(cm.device)

        info = TransformInfo(skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True)

        return mod, info
