# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA BCG initialization requirements."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention


@pytest.mark.parametrize(
    "unsupported,match",
    [
        (None, None),
        ("bfa", "finalized QKVG/BFA"),
        ("indexed", "TLLM_KDA_ENABLE_INDEXED_STATE_POOL=1"),
        ("decode", "optimized KDA decode"),
        ("gate", "use_full_rank_gate=True"),
        ("weights", "finalized QKVG/BFA"),
    ],
)
def test_engine_validates_kda_bcg_before_capture(unsupported, match):
    import weakref

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
    from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

    attention = KimiKDALinearAttention.__new__(KimiKDALinearAttention)
    torch.nn.Module.__init__(attention)
    attention._use_indexed_ssm_pool = unsupported != "indexed"
    attention._dispatch = SimpleNamespace(
        decode_kernel_path="fla" if unsupported == "decode" else "optimized"
    )
    attention.use_full_rank_gate = unsupported != "gate"
    attention._qkvg_proj_weight = None if unsupported == "weights" else torch.empty(1)
    attention.qkvg_proj = None
    attention._bfa_proj_weight = None if unsupported == "bfa" else torch.empty(1)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = SimpleNamespace(
        model_config=SimpleNamespace(extra_attrs={"kda_layers": {"0": weakref.ref(attention)}})
    )
    engine.input_processor = None
    engine.llm_args = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE, disable_mm_encoder=False
    )
    with pytest.raises(ValueError, match=match) if unsupported else nullcontext():
        engine._validate_breakable_cuda_graph_compatibility()
    # The restrictions must not change the eager or PCG fallback contracts.
    for backend in (PrefillCudaGraphBackend.DISABLED, PrefillCudaGraphBackend.PIECEWISE):
        engine.llm_args.prefill_cuda_graph_backend = backend
        engine._validate_breakable_cuda_graph_compatibility()
