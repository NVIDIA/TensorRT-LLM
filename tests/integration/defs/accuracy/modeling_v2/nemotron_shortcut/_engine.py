# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version

from .._env import scoped_env_overrides


@contextmanager
def shortcut_text_engine(checkpoint, **options):
    if get_sm_version() != 103:
        pytest.skip("Four-rank text generation is qualified on GB300")
    if not checkpoint.is_dir():
        pytest.skip(f"Checkpoint unavailable: {checkpoint}")
    overrides = {**options.pop("env_overrides", {}), MODELING_V2_VALIDATE_ENV: "1"}
    with (
        scoped_env_overrides(overrides) as rank_env,
        LLM(
            str(checkpoint),
            modeling_v2="require",
            env_overrides=rank_env,
            trust_remote_code=True,
            disable_mm_encoder=options.pop("disable_mm_encoder", True),
            tensor_parallel_size=4,
            moe_tensor_parallel_size=2,
            moe_expert_parallel_size=2,
            enable_attention_dp=False,
            **options,
        ) as llm,
    ):
        yield llm
