"""Transform to wrap model forward to accept a CachedSequenceInterface as single argument.

This creates a small ``nn.Module`` wrapper so callers can run ``model(cache_seq_interface)``.
Under the hood, it invokes the original model with ``*cache_seq_interface.args`` (same as
``ad_executor.py`` does today). This enables future callers to simply pass the interface
and let the wrapper handle the argument unpacking.
"""

from typing import Tuple, Type

import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactory
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class ForwardWithCSIConfig(TransformConfig):
    """Configuration for the forward-with-CSI wrapper transform."""

    args_only: bool = Field(
        default=True,
        description=(
            "If True, the wrapper will call the underlying model with *cm.args. "
            "If False, it will pass the cm object directly as the first argument."
        ),
    )


class _ForwardWithCSIWrapper(nn.Module):
    """A lightweight wrapper to forward with a CachedSequenceInterface argument."""

    def __init__(self, model: nn.Module, args_only: bool = True) -> None:
        super().__init__()
        self.model = model
        self.args_only = args_only

    def forward(self, cm: CachedSequenceInterface):  # type: ignore[override]
        if self.args_only:
            return self.model(*cm.args)
        # Fallback path with kwargs
        return self.model(**cm.named_args)


@TransformRegistry.register("forward_with_cached_sequence_interface")
class ForwardWithCachedSequenceInterface(BaseTransform):
    """Wrap the model so forward accepts a single ``CachedSequenceInterface`` argument."""

    config: ForwardWithCSIConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ForwardWithCSIConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # ``gm`` is an nn.Module (GraphModule or compiled module). Wrap it so callers can do
        # ``wrapped(cm)`` and internally we expand to ``gm(*cm.args)``.
        wrapped = _ForwardWithCSIWrapper(gm, args_only=self.config.args_only)

        # No graph mutation; simply return wrapped module. Mark as clean with valid shapes preserved.
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)
        return wrapped, info
