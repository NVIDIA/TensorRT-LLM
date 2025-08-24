from typing import List, Literal, Optional, Tuple, Type

from pydantic import Field
from torch.fx import GraphModule

from ...compile import compile_and_capture
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
    compile_backend: Literal["torch-simple", "torch-compile", "torch-cudagraph", "torch-opt"] = (
        Field(description="The backend to use for compiling the model.")
    )


@TransformRegistry.register("compile_model")
class CompileModel(BaseTransform):
    """A transform to compile the model."""

    config: CompileModelConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return CompileModelConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cm.info.set_generate_only_batch()
        egm_compiled = compile_and_capture(
            gm,
            self.config.compile_backend,
            args=cm.args,
            dynamic_shapes=cm.dynamic_shapes,
            compiler_kwargs={
                "cuda_graph_batch_sizes": self.config.cuda_graph_batch_sizes,
                "num_batched_inputs": self.config.num_batched_inputs,
            },
        )
        cm.info.reset()

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return egm_compiled, info
