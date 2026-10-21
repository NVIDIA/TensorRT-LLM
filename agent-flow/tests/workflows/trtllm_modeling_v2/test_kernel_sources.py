# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kernel-source registry loads and reaches the kernel writer."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import (
    format_kernel_sources,
    load_kernel_sources,
)


def test_the_registry_reaches_the_kernel_writer(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem

    from .test_workflow import _workflow

    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    workflow._kernel_sources_block = format_kernel_sources(load_kernel_sources())
    prompt = workflow._kernel_writer_prompt(QueueItem(n=1, mode="onboard", target="x", tier=1))

    assert "Where kernels come from" in prompt
    assert "cute_dsl" in prompt and "flashinfer" in prompt
    assert "registration planes" in prompt
