# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunyuanDiT default generation parameters and extra-param schema."""

from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

_HUNYUANDIT_DEFAULT_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "max_sequence_length": 77,
}


def get_hunyuandit_default_params() -> dict:
    return dict(_HUNYUANDIT_DEFAULT_PARAMS)


def get_hunyuandit_extra_param_specs() -> dict:
    return {
        "negative_prompt": ExtraParamSchema(
            type="str",
            default="",
            description="Negative text prompt for classifier-free guidance.",
        ),
        "use_resolution_binning": ExtraParamSchema(
            type="bool",
            default=True,
            description=(
                "Snap resolution to the nearest HunyuanDiT training bucket "
                "(recommended for best quality)."
            ),
        ),
    }
