# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the MiniMax-M3 integration tests.

The fixtures in this conftest are the discovery glue between the
bring-up workspace (``workspace/<task>/reference/...``) and the
TensorRT-LLM integration tests under
``tests/integration/_torch/test_minimax_m3_*.py``. Each fixture either
yields the data the test needs or raises ``pytest.skip`` with a precise
blocker message naming exactly what artifact / runtime resource is
missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from ._m3_replay_helpers import (
    SGLangArtifactStatus,
    checkpoint_skip_reason,
    discover_sglang_artifacts,
    load_jsonl_outputs,
    reference_protocol,
    sglang_artifact_skip_reason,
    workspace_skip_reason,
)


@pytest.fixture(scope="session")
def m3_workspace_protocol():
    """The ``reference.protocol`` module from the bring-up workspace."""
    reason = workspace_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    proto = reference_protocol()
    if proto is None:
        pytest.skip("reference.protocol module not importable")
    return proto


@pytest.fixture(scope="session")
def m3_artifact_status() -> SGLangArtifactStatus:
    """Status of every SGLang artifact the integration tests consume."""
    return discover_sglang_artifacts()


@pytest.fixture(scope="session")
def m3_sglang_text_outputs(m3_artifact_status: SGLangArtifactStatus) -> List[Dict[str, Any]]:
    """SGLang-captured fixed-prompt outputs (token ids + text)."""
    reason = sglang_artifact_skip_reason("text_prompts_jsonl")
    if reason is not None:
        pytest.skip(reason)
    return load_jsonl_outputs(Path(m3_artifact_status.text_prompts_jsonl))


@pytest.fixture(scope="session")
def m3_sglang_gsm8k_outputs(m3_artifact_status: SGLangArtifactStatus) -> List[Dict[str, Any]]:
    """SGLang-captured GSM8K subset outputs (token ids + text)."""
    reason = sglang_artifact_skip_reason("gsm8k_outputs_jsonl")
    if reason is not None:
        pytest.skip(reason)
    return load_jsonl_outputs(Path(m3_artifact_status.gsm8k_outputs_jsonl))


@pytest.fixture(scope="session")
def m3_real_checkpoint_path(m3_workspace_protocol) -> Path:
    """Path to the real MiniMax-M3 checkpoint, skipping when unloadable.

    The headroom check uses a conservative 60 GiB threshold per GPU on
    the assumption of TP>=4 with the mxfp8 weight format documented in
    ``plan.md``. If a future deviation uses higher TP / cpu_offload the
    threshold can be relaxed in the calling test.
    """
    # 60 GiB headroom matches "230B params at mxfp8 ~ 230GB / 4 TP" with
    # generous overhead for activations and CUDA context. This is the
    # value that distinguishes "GPU contention is the blocker" from
    # "checkpoint is missing".
    reason = checkpoint_skip_reason(min_free_gb_per_gpu=60.0)
    if reason is not None:
        pytest.skip(reason)
    return Path(m3_workspace_protocol.CHECKPOINT_PATH)


@pytest.fixture(scope="session")
def m3_cuda_required() -> None:
    """Hard-require CUDA for tests that need GPU execution."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; integration tests require a GPU")
    return None
