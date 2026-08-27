# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm.llmapi.llm_args import TorchCompileConfig


def test_compile_generation_defaults_to_enabled():
    assert TorchCompileConfig().compile_generation


def test_compile_generation_can_be_disabled():
    assert not TorchCompileConfig(compile_generation=False).compile_generation
