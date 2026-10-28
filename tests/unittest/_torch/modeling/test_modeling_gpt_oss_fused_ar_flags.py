# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS TransformerBlock construction under the fused all-reduce flags (no GPU, no weights).

TLLM_GPT_OSS_MOE_FINALIZE_AR / TLLM_GPT_OSS_AR_MXFP8_QUANT decide eligibility in TransformerBlock.__init__, before
the MoE has created its weights (and its quant_method). The block must build in that state, resolve the MXFP8 input
alignment from the method the MoE will create, and stay ineligible for any other MoE implementation.

Runs under pytest, or without it (images that ship no pytest): ``python3 test_modeling_gpt_oss_fused_ar_flags.py``
runs every test with a minimal monkeypatch and exits nonzero on any failure.
"""

import sys
import traceback
from types import SimpleNamespace

import torch

try:
    import pytest
except ImportError:  # the __main__ runner below does not need it
    pytest = None
from torch import nn

import tensorrt_llm._torch.models.modeling_gpt_oss as gpt_oss
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_w4a8_mxfp4_mxfp8 import \
    TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
from tensorrt_llm.functional import AllReduceStrategy


class _Stub(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()


class _AllReduce:

    def __init__(self, *args, **kwargs):
        self.mnnvl_allreduce = None
        self.symm_mem_allreduce = None


def _mlp_with(experts):

    class _MLP(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.experts = experts

    return _MLP


def _config():
    mapping = SimpleNamespace(enable_attention_dp=False,
                              has_tp=lambda: True,
                              tp_size=4,
                              is_multi_node=lambda: False)
    pretrained = SimpleNamespace(hidden_size=2880,
                                 rms_norm_eps=1e-5,
                                 torch_dtype=torch.bfloat16)
    return SimpleNamespace(mapping=mapping,
                           pretrained_config=pretrained,
                           allreduce_strategy=AllReduceStrategy.AUTO)


def _unweighted_mxfp8_impl():
    # The leaf as it is while the decoder layers are built: no create_weights yet, hence no quant_method.
    impl = object.__new__(TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl)
    assert not hasattr(impl, "quant_method")
    return impl


def _stub(monkeypatch):
    monkeypatch.setattr(gpt_oss, "AttentionBlock", _Stub)
    monkeypatch.setattr(gpt_oss, "AllReduce", _AllReduce)
    monkeypatch.setattr(gpt_oss, "MoEAllReduce", _AllReduce)
    monkeypatch.setattr(gpt_oss, "get_sm_version", lambda: 103)
    monkeypatch.setattr(gpt_oss, "_fused_mlp_nvfp4_input_enabled",
                        lambda *a: False)
    return monkeypatch


_FLAG_CASES = [(True, False), (False, True), (True, True)]


def _parametrize(test):
    return pytest.mark.parametrize("finalize_ar,mxfp8_quant", _FLAG_CASES)(test) if pytest else test


if pytest is not None:
    stubbed = pytest.fixture(name="stubbed")(_stub)


@_parametrize
def test_block_builds_before_moe_weights(stubbed, finalize_ar, mxfp8_quant):
    stubbed.setattr(gpt_oss, "_GPT_OSS_MOE_FINALIZE_AR", finalize_ar)
    stubbed.setattr(gpt_oss, "_GPT_OSS_AR_MXFP8_QUANT", mxfp8_quant)
    stubbed.setattr(gpt_oss, "MLPBlock", _mlp_with(_unweighted_mxfp8_impl()))
    block = gpt_oss.TransformerBlock(_config(), layer_idx=3)
    assert block._fused_ar_eligible
    assert block._mxfp8_alignment == 512
    assert (block.moe_allreduce is not None) == finalize_ar


def test_alignment_prefers_created_quant_method():
    impl = _unweighted_mxfp8_impl()
    impl.quant_method = SimpleNamespace(input_hidden_alignment=256)
    assert gpt_oss._moe_input_hidden_alignment(impl) == 256


def test_other_moe_is_not_eligible(stubbed):
    stubbed.setattr(gpt_oss, "_GPT_OSS_MOE_FINALIZE_AR", True)
    stubbed.setattr(gpt_oss, "_GPT_OSS_AR_MXFP8_QUANT", True)
    stubbed.setattr(gpt_oss, "MLPBlock", _mlp_with(nn.Identity()))
    block = gpt_oss.TransformerBlock(_config(), layer_idx=3)
    assert not block._fused_ar_eligible
    assert block._mxfp8_alignment is None


def test_flags_off_unchanged(stubbed):
    stubbed.setattr(gpt_oss, "_GPT_OSS_MOE_FINALIZE_AR", False)
    stubbed.setattr(gpt_oss, "_GPT_OSS_AR_MXFP8_QUANT", False)
    stubbed.setattr(gpt_oss, "MLPBlock", _mlp_with(_unweighted_mxfp8_impl()))
    block = gpt_oss.TransformerBlock(_config(), layer_idx=3)
    assert not block._fused_ar_eligible
    assert block.moe_allreduce is None


class _MonkeyPatch:
    """setattr with undo, the part of pytest's monkeypatch these tests use."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, value in reversed(self._undo):
            setattr(target, name, value)
        self._undo.clear()


def _run_without_pytest() -> int:
    cases = [(f"test_block_builds_before_moe_weights[{fa}-{mq}]",
              lambda mp, fa=fa, mq=mq: test_block_builds_before_moe_weights(_stub(mp), fa, mq))
             for fa, mq in _FLAG_CASES]
    cases += [("test_alignment_prefers_created_quant_method", lambda mp: test_alignment_prefers_created_quant_method()),
              ("test_other_moe_is_not_eligible", lambda mp: test_other_moe_is_not_eligible(_stub(mp))),
              ("test_flags_off_unchanged", lambda mp: test_flags_off_unchanged(_stub(mp)))]
    failed = 0
    for name, fn in cases:
        mp = _MonkeyPatch()
        try:
            fn(mp)
            print(f"PASSED {name}", flush=True)
        except Exception:  # noqa: BLE001 - reported, counted
            failed += 1
            print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)
        finally:
            mp.undo()
    print(f"{len(cases) - failed} passed, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_without_pytest())
