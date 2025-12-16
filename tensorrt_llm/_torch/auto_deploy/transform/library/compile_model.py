from typing import List, Literal, Optional, Tuple, Type

import torch.nn as nn
from pydantic import Field

from ...compile import ArgsKwargs, CompileBackendRegistry
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class CompileModelConfig(TransformConfig):
    """Configuration for the compile model transform."""

    cuda_graph_batch_sizes: Optional[List[int]] = Field(
        default=None, description="The batch sizes to use for CUDA graphs."
    )
    num_batched_inputs: int = Field(
        default=2, description="The number of batched inputs to use for CUDA graphs."
    )
    backend: Literal["torch-simple", "torch-compile", "torch-cudagraph", "torch-opt"] = Field(
        description="The backend to use for compiling the model."
    )


@TransformRegistry.register("compile_model")
class CompileModel(BaseTransform):
    """A transform to compile the model."""

    config: CompileModelConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return CompileModelConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        cm.info.reset()

        def _get_args_kwargs(bs: int) -> ArgsKwargs:
            cm.info.set_generate_only_batch(bs)
            return (), cm.named_args

        compiler_backend = CompileBackendRegistry.get(self.config.backend)(
            mod,
            get_args_kwargs_for_compile=_get_args_kwargs,
            **self.config.model_dump(),
        )
        mod_compiled = compiler_backend.compile()

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return mod_compiled, info
