# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch.pyexecutor import model_engine
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def initialize_engine_args(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    monkeypatch.setattr(model_engine, "_DEEP_GEMM_PDL_CONFIGURED", False)
    processor = object()
    # Tokenizer/checkpoint IO is outside this argument-initialization contract.
    monkeypatch.setattr(model_engine, "create_input_processor", lambda *args, **kwargs: processor)
    args = SimpleNamespace(
        encode_only=False,
        mm_encoder_only=False,
        get_runtime_sizes=lambda: (1, 8, 16, 2),
        encoder_max_batch_size=None,
        encoder_max_num_tokens=None,
        multimodal_config=SimpleNamespace(video_pruning_rate=None),
        checkpoint_format="HF",
        trust_remote_code=False,
    )

    def initialize() -> None:
        engine = object.__new__(ModelEngine)
        engine._cleanup_done = True
        budget = engine._init_from_args(
            model_path="unused",
            llm_args=args,
            mapping=Mapping(),
            attn_runtime_features=None,
            dist=None,
            spec_config=None,
        )
        assert budget == args.get_runtime_sizes()[1]
        assert engine.input_processor is processor
        assert engine.batch_size == 2 and engine.max_seq_len == 16

    return initialize


@pytest.mark.parametrize("value,enabled", [(None, True), ("0", False), ("1", True)])
def test_engine_configures_deep_gemm_once(
    monkeypatch: pytest.MonkeyPatch,
    initialize_engine_args: Callable[[], None],
    value: str | None,
    enabled: bool,
) -> None:
    if value is None:
        monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", value)
    configured = []
    monkeypatch.setattr(deep_gemm, "set_pdl", configured.append)

    initialize_engine_args()
    initialize_engine_args()

    assert configured == [enabled]


def test_engine_retries_deep_gemm_after_configuration_failure(
    monkeypatch: pytest.MonkeyPatch, initialize_engine_args: Callable[[], None]
) -> None:
    configured = []
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")

    def configure(enabled: bool) -> None:
        configured.append(enabled)
        if len(configured) == 1:
            raise RuntimeError("runtime setup failed")

    monkeypatch.setattr(deep_gemm, "set_pdl", configure)
    with pytest.raises(RuntimeError, match="runtime setup failed"):
        initialize_engine_args()
    initialize_engine_args()
    initialize_engine_args()

    assert configured == [True, True]
