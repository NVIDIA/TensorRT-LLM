# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_pre_hopper

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.compilation.utils import _PhaseSelectiveForward
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm.llmapi import CudaGraphConfig, PrefillCudaGraphBackend, TorchCompileConfig


def test_phase_selective_forward_uses_compiled_by_default() -> None:
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    assert forward("input") == "compiled"
    compiled_forward.assert_called_once_with("input")
    eager_forward.assert_not_called()


def test_phase_selective_forward_bypass_is_restored() -> None:
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    assert forward() == "compiled"
    with forward.bypass():
        assert forward() == "eager"
    assert forward() == "compiled"


@pytest.mark.parametrize(
    ("can_run_graph", "prefill_graph_eligible", "expected"),
    [
        pytest.param(False, True, "compiled", id="pcg-eligible-prefill"),
        pytest.param(False, False, "eager", id="pcg-ineligible-prefill"),
        pytest.param(True, True, "eager", id="ordinary-generation-capture"),
        pytest.param(False, False, "eager", id="generation-graph-miss"),
    ],
)
def test_model_engine_selects_torch_compile_forward(
    can_run_graph: bool,
    prefill_graph_eligible: bool,
    expected: str,
) -> None:
    # A phase-selective proxy is installed by ModelEngine initialization when
    # compile_only_piecewise_graphs is enabled.
    engine = object.__new__(PyTorchModelEngine)
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    engine._phase_selective_forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    with patch(
        "tensorrt_llm._torch.pyexecutor.model_engine.get_per_request_prefill_cuda_graph_flag",
        return_value=prefill_graph_eligible,
    ) as get_prefill_graph_flag:
        with engine._maybe_bypass_torch_compile(can_run_graph=can_run_graph):
            actual = engine._phase_selective_forward()

    assert actual == expected
    selected_forward = compiled_forward if expected == "compiled" else eager_forward
    other_forward = eager_forward if expected == "compiled" else compiled_forward
    selected_forward.assert_called_once_with()
    other_forward.assert_not_called()
    if can_run_graph:
        get_prefill_graph_flag.assert_not_called()
    else:
        get_prefill_graph_flag.assert_called_once_with()


def test_model_engine_explicit_bypass_skips_graph_eligibility() -> None:
    engine = object.__new__(PyTorchModelEngine)
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    engine._phase_selective_forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    with patch(
        "tensorrt_llm._torch.pyexecutor.model_engine.get_per_request_prefill_cuda_graph_flag",
        return_value=True,
    ) as get_prefill_graph_flag:
        with engine._maybe_bypass_torch_compile(bypass=True, can_run_graph=False):
            actual = engine._phase_selective_forward()

    assert actual == "eager"
    eager_forward.assert_called_once_with()
    compiled_forward.assert_not_called()
    get_prefill_graph_flag.assert_not_called()


@skip_pre_hopper
@pytest.mark.timeout(1800)
def test_piecewise_cuda_graph_compile_restriction_stability() -> None:
    """Compare compiled and eager decode after compiled piecewise prefill."""
    models_root = llm_models_root()
    if models_root is None:
        pytest.skip("LLM_MODELS_ROOT is not available")
    model_path = models_root / "Qwen3/Qwen3-0.6B"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(
        max_tokens=8, end_id=-1, temperature=0, return_generation_logits=True
    )

    def run(compile_only_piecewise_graphs: bool) -> tuple[list[int], torch.Tensor]:
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            compile_only_piecewise_graphs=compile_only_piecewise_graphs,
        )
        with LLM(
            model_path,
            max_seq_len=128,
            max_num_tokens=32,
            max_batch_size=1,
            disable_overlap_scheduler=True,
            gather_generation_logits=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=1),
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
            prefill_capture_num_tokens=[32],
            torch_compile_config=torch_compile_config,
        ) as llm:
            output = llm.generate(prompts, sampling_params=sampling_params)[0].outputs[0]
            assert output.generation_logits is not None
            return list(output.token_ids), output.generation_logits.clone()

    compiled_tokens, compiled_logits = run(False)
    restricted_tokens, restricted_logits = run(True)

    assert restricted_tokens == compiled_tokens
    torch.testing.assert_close(restricted_logits, compiled_logits, rtol=1e-2, atol=1e-2)
