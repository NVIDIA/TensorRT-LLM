# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight sources that read no checkpoint tensors: dummy and vision-only loads."""

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.utils.fp4_utils import float4_e2m1x2

from ...model_config import ModelConfig
from .base import WeightSource


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    This is similar to this function in SGLang with a few changes:
    https://github.com/sgl-project/sglang/blob/e074e76b31d4fff13e87a455dbc3acdaa92c537a/python/sglang/srt/model_loader/weight_utils.py#L577
    This method is used to initialize weights with dummy values for testing
    models without checkpoints. Unquantized (FP16/BF16/etc) values are generated
    from a uniform distribution over the interval (low, high).
    For some quantized types (FP8/NVFP4), torch has no built-in way to generate random values.
    We simply generate values uniformly across an interval that has been empirically verified
    to not generate NaNs/inf for these.
    """

    def _get_random_min_max(dtype: torch.dtype) -> tuple[int, int]:
        """Return safe (min, max) bounds for uniform sampling of `dtype`."""
        # These values are not necessarily the largest possible min/max,
        # they need to be small enough to avoid NaNs.
        if dtype in (torch.float8_e4m3fn, torch.int8):
            return (-3.0, 3.0)

        elif dtype == float4_e2m1x2:
            # These correspond to bits of 2 packed FP4 values.
            # Because we only go up to 64, the high 4 bits will
            # always be 0. But this is fine - we just need values
            # that won't generate NaNs.
            return (0, 64)

        else:
            raise NotImplementedError(f"Unknown quantized type: {dtype}.")

    # Calibration scalars (input_scale / inv_input_scale / kv_scales /
    # inv_kv_scales / alpha / scalar_alpha) must keep their create_weights
    # default (typically 1.0). Randomizing them breaks FP8 attention output
    # and KV cache scaling for any `load_format="dummy"` + IPC update_weights
    # flow when the checkpoint doesn't ship calibrated values (e.g., HF
    # FineGrainedFP8, which uses dynamic activation quantization by design).
    _SKIP_NAME_SUFFIXES = (
        ".input_scale",
        ".inv_input_scale",
        ".kv_scales",
        ".inv_kv_scales",
        ".alpha",
        ".scalar_alpha",
    )

    for _name, param in model.state_dict().items():
        if any(_name.endswith(_s) for _s in _SKIP_NAME_SUFFIXES):
            continue
        generator = torch.Generator(device=param.data.device)
        generator.manual_seed(seed)
        dtype = param.data.dtype

        if param.data.element_size() < 2:
            # We need to do a cast/round since torch doesn't have uniform_
            # support for these dtypes.
            tmp_param = torch.empty(param.data.shape, dtype=torch.float16, device=param.data.device)

            quant_min, quant_max = _get_random_min_max(dtype)
            tmp_param = tmp_param.uniform_(quant_min, quant_max, generator=generator)

            param.data.copy_(tmp_param.to(dtype))

        # Note: no need to to mess with int32 params, these are probably
        # constants and not weights.
        elif torch.is_floating_point(param):
            param.uniform_(low, high, generator=generator)

    # Non-persistent buffers are intentionally absent from state_dict(). Let
    # modules that need such buffers during dummy execution initialize them
    # explicitly after the regular state has been populated.
    for module in model.modules():
        initialize_dummy_state = getattr(module, "initialize_dummy_state", None)
        if initialize_dummy_state is not None:
            initialize_dummy_state(low=low, high=high, seed=seed)


class DummyWeightSource(WeightSource):
    """Fill the model with small random values, for ``load_format="dummy"``.

    Benchmarks and tests use this to run a model without its checkpoint. The
    weight mapper is still initialized, because reinforcement-learning
    frameworks often start with dummy weights and push real ones later
    through `ModelLoader.reload()` or weight updates, which need the mapper.
    """

    def load_weights(self, model: nn.Module, config: ModelConfig) -> bool:
        """Pick the weight mapper and fill every weight with random values."""
        self.weight_mapper = self.context.checkpoint_loader.get_initialized_weight_mapper(
            model, config
        )
        initialize_dummy_weights(model)
        return False

    def load_draft_weights(self, model: nn.Module) -> None:
        """Copy the shared weights (such as embeddings) from the target model."""
        model.draft_model.load_weights_from_target_model(model)


class VisionOnlyWeightSource(WeightSource):
    """Load nothing, for ``load_format="vision_only"``.

    The multimodal model loads its vision encoder weights while it is
    constructed, and this load skips the language model weights. The default
    `WeightSource.load_draft_weights()` loads no draft weights either.
    """

    def load_weights(self, model: nn.Module, config: ModelConfig) -> bool:
        """Log that weight loading is skipped."""
        # Vision weights are already loaded within the model.
        logger.info(
            "LoadFormat.VISION_ONLY: skipping weight loading; using preloaded vision weights."
        )
        return False
