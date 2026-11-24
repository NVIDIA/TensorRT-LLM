# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm::k31_create_mla_outputs``: the MLA output buffer of each Kimi K3.1 target, looked up through the model's
layer registry, is what the target MLA module's ``_create_outputs`` allocates (shape, dtype, stride, device), for
context and generation steps, and the op's fake implementation (piecewise CUDA graph tracing) agrees with it."""

import importlib
import weakref

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.mla_outputs import (
    create_mla_outputs,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.utils import model_extra_attrs

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
#: (target, per-rank o_proj input width): the attention TP shard (12 of 96 heads x 128) and the full 96 heads under
#: attention DP.
TARGETS = [
    ("kimi_k31_pro_mxfp4__sm_103__tep", 12 * 128),
    ("kimi_k31_pro_mxfp4__sm_103__dep", 96 * 128),
    ("kimi_k31_pro_mxfp4__sm_10x__default", 12 * 128),
    ("kimi_k31_pro_mxfp4__sm_10x__default", 96 * 128),
]
#: (phase, rows, num_contexts): a context chunk and a DFlash2 w5 verify step of 4 requests.
PHASES = [("context", 3000, 2), ("generation", 24, 0)]


class _StepMetadata(TrtllmAttentionMetadata):
    """A step's metadata as far as the output allocation reads it: the context count."""

    def __init__(self, num_contexts: int):
        object.__setattr__(self, "_k31_contexts", num_contexts)

    @property
    def num_contexts(self) -> int:
        return self._k31_contexts


def _mla(target: str, width: int) -> nn.Module:
    """The target's MLA module with only the state ``_create_outputs`` reads."""
    cls = importlib.import_module(f"{_FAMILY}.{target}.modeling").K31MLA
    mla = cls.__new__(cls)
    nn.Module.__init__(mla)
    mla.attention_output_hidden_size = width
    return mla


def _layout(t: torch.Tensor):
    return t.shape, t.dtype, t.stride(), t.device.type


@pytest.mark.parametrize("phase,rows,num_contexts", PHASES, ids=[p[0] for p in PHASES])
@pytest.mark.parametrize(
    "target,width", TARGETS, ids=[f"{t.rsplit('__', 1)[1]}-{w}" for t, w in TARGETS]
)
def test_create_mla_outputs_matches_the_module(target, width, phase, rows, num_contexts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mla = _mla(target, width)
    metadata = _StepMetadata(num_contexts)
    hidden = torch.empty(rows, 7168, dtype=torch.bfloat16, device=device)
    expected = mla._create_outputs(hidden, metadata)
    assert len(expected) == 1
    attrs = {"attention_metadata": weakref.ref(metadata), "mla_layers": {"7": weakref.ref(mla)}}
    with model_extra_attrs(attrs):
        out = create_mla_outputs(hidden, "7")
        assert (
            _layout(out)
            == _layout(expected[0])
            == ((rows, width), torch.bfloat16, (width, 1), device)
        )
        with FakeTensorMode():
            fake = create_mla_outputs(
                torch.empty(rows, 7168, dtype=torch.bfloat16, device=device), "7"
            )
        assert (fake.shape, fake.dtype, fake.stride()) == (
            expected[0].shape,
            expected[0].dtype,
            expected[0].stride(),
        )


def test_create_mla_outputs_needs_a_registered_layer():
    metadata = _StepMetadata(0)
    with model_extra_attrs({"attention_metadata": weakref.ref(metadata), "mla_layers": {}}):
        with pytest.raises(AssertionError, match="Cannot find attention layer"):
            create_mla_outputs(torch.empty(4, 8, dtype=torch.bfloat16), "0")
