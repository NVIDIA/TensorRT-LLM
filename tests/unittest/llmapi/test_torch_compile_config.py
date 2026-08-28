# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm.llmapi.llm_args import TorchCompileConfig


def test_compile_only_context_and_mixed_graphs_defaults_to_disabled():
    assert not TorchCompileConfig().compile_only_context_and_mixed_graphs


def test_compile_only_context_and_mixed_graphs_can_be_enabled():
    config = TorchCompileConfig(compile_only_context_and_mixed_graphs=True)
    assert config.compile_only_context_and_mixed_graphs
