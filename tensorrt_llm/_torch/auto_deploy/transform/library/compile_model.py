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


def _get_eagle_extend_capture_draft_len(spec_config) -> Optional[int]:
    """Return max_draft_len when cudagraph warmup should use extend-mode inputs."""
    if spec_config is None:
        return None

    max_draft_len = getattr(spec_config, "max_draft_len", None)
    if max_draft_len is None:
        raise ValueError(
            "Speculative cudagraph warmup requires spec_config.max_draft_len to be set"
        )

    return max_draft_len


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
        use_cache_seq_interface_kwarg = cm._spec_config is not None
        eagle_extend_capture_draft_len = _get_eagle_extend_capture_draft_len(cm._spec_config)

        def _get_args_kwargs(bs: int) -> ArgsKwargs:
            if eagle_extend_capture_draft_len is not None:
                cm.info.set_eagle_extend_batch(bs, eagle_extend_capture_draft_len)
            else:
                cm.info.set_generate_only_batch(bs)
            if use_cache_seq_interface_kwarg:
                return (), {**cm.named_args, "cache_seq_interface": cm}
            return (), cm.named_args

        extra_kwargs = {}
        config_overrides = {}
        extra_kwargs["spec_mode"] = cm._spec_config is not None

        if self.config.piecewise_enabled:
            extra_kwargs["piecewise_seq_info"] = cm.info
            extra_kwargs["piecewise_named_args_fn"] = lambda: cm.named_args

            max_seq = cm.info.max_seq_len
            max_batch = cm.info.max_batch_size
            batch_capacity = (max_batch - 1) * max_seq + 1

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

            # Filter out buckets that exceed the mixed-batch capacity
            buckets = config_overrides.get(
                "piecewise_num_tokens", self.config.piecewise_num_tokens or []
            )
            over = [nt for nt in buckets if nt > batch_capacity]
            if over:
                buckets = [nt for nt in buckets if nt <= batch_capacity]
                ad_logger.warning(
                    f"Dropping piecewise buckets {over} that exceed mixed-batch capacity "
                    f"({max_batch - 1} seqs * {max_seq} tokens + 1 decode = {batch_capacity}). "
                    f"Remaining: {buckets}"
                )
                config_overrides["piecewise_num_tokens"] = buckets

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
