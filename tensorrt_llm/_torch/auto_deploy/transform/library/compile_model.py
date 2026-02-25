from typing import List, Literal, Optional, Tuple, Type

import torch.nn as nn
from pydantic import Field

from ...compile import ArgsKwargs, CompileBackendRegistry
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _generate_default_piecewise_num_tokens(max_num_tokens: int) -> List[int]:
    """Generate default piecewise bucket sizes when none are specified.

    Uses powers-of-2 from 64 up to max_num_tokens. This provides ~log2(max/64)
    bucket sizes with at most 2x padding overhead per bucket.

    For example, max_num_tokens=8192 → [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    """
    if max_num_tokens <= 0:
        return []

    buckets = []
    nt = 64
    while nt <= max_num_tokens:
        buckets.append(nt)
        nt *= 2

    # Always include max_num_tokens as the largest bucket
    if not buckets or buckets[-1] != max_num_tokens:
        buckets.append(max_num_tokens)

    return sorted(buckets)


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
    piecewise_enabled: bool = Field(
        default=False,
        description="Enable piecewise CUDA graph for prefill/mixed batches (dual-mode).",
    )
    piecewise_num_tokens: Optional[List[int]] = Field(
        default=None,
        description=(
            "Total token counts to pre-capture piecewise CUDA graphs for. "
            "If null and piecewise_enabled=true, auto-generates power-of-2 buckets "
            "up to max_num_tokens (e.g. [64, 128, 256, ..., max_num_tokens])."
        ),
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

        def _get_mixed_args_kwargs(num_tokens: int) -> ArgsKwargs:
            """Generate synthetic mixed-batch args for piecewise CG capture.

            Always creates 1 prefill sequence + 1 decode sequence to exercise
            both code paths in dynamic ops (attention, SSM). The static CUDA
            graph segments are agnostic to the prefill/decode split -- they only
            see total_num_tokens.
            """
            assert num_tokens >= 3, (
                f"Piecewise bucket {num_tokens} too small for mixed batch. "
                f"Minimum is 3 (1 prefill seq with len>=2 + 1 decode seq)."
            )
            cm.info.set_example_sequence(
                input_ids=[
                    [1] * (num_tokens - 1),  # prefill: len > 1
                    [1],  # decode: len == 1
                ],
            )
            return (), cm.named_args

        extra_kwargs = {}
        config_overrides = {}

        if self.config.piecewise_enabled:
            extra_kwargs["get_mixed_args_kwargs_for_compile"] = _get_mixed_args_kwargs

            # Auto-generate piecewise_num_tokens if not explicitly specified
            if self.config.piecewise_num_tokens is None:
                max_num_tokens = cm.info.max_num_tokens
                auto_buckets = _generate_default_piecewise_num_tokens(max_num_tokens)
                config_overrides["piecewise_num_tokens"] = auto_buckets
                ad_logger.info(
                    f"Auto-generated piecewise_num_tokens from max_num_tokens={max_num_tokens}: "
                    f"{auto_buckets}"
                )
            else:
                # Filter out buckets < 3 (mixed batch needs at least 3 tokens)
                valid_buckets = [nt for nt in self.config.piecewise_num_tokens if nt >= 3]
                dropped = [nt for nt in self.config.piecewise_num_tokens if nt < 3]
                if dropped:
                    ad_logger.warning(
                        f"Dropping piecewise_num_tokens {dropped} (too small for mixed batch, "
                        f"minimum is 3). Remaining: {valid_buckets}"
                    )
                config_overrides["piecewise_num_tokens"] = valid_buckets

        # Merge config with any overrides
        config_dict = self.config.model_dump()
        config_dict.update(config_overrides)

        compiler_backend = CompileBackendRegistry.get(self.config.backend)(
            mod,
            get_args_kwargs_for_compile=_get_args_kwargs,
            **extra_kwargs,
            **config_dict,
        )
        mod_compiled = compiler_backend.compile()

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return mod_compiled, info
