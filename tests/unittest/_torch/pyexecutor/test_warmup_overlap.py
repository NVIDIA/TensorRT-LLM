# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel builds next to weight loading (pyexecutor/warmup_overlap)."""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer as sampler_ops
import tensorrt_llm._torch.pyexecutor.warmup_overlap.linear as overlap_linear
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.pyexecutor.warmup_overlap import warmup_overlap
from tensorrt_llm._torch.utils import deep_gemm_jit_warmup_buckets
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

FP8 = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
# Wider than one wave of 128-row tiles on SM100 / SM103: no low-M range.
NO_LOW_M = (32768, 512)


class _Mapping:
    def __init__(self, helix: bool, pp: bool = False):
        self.helix, self.pp = helix, pp

    def has_cp_helix(self) -> bool:
        return self.helix

    def has_pp(self) -> bool:
        return self.pp


def _llm_args(max_num_tokens, max_batch_size, max_seq_len=None, draft=0, **overrides):
    spec = SimpleNamespace(max_total_draft_tokens=draft, max_draft_len=draft) if draft else None
    args = dict(
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_beam_width=1,
        speculative_config=spec,
        cuda_graph_config=None,
        enable_autotuner=True,
        guided_decoding_backend=None,
        torch_compile_config=None,
    )
    return SimpleNamespace(**{**args, **overrides})


def _sweep(*token_counts):
    return tuple(sorted({m for t in token_counts for m in (t, *deep_gemm_jit_warmup_buckets(t))}))


def _fp8_linear(in_features, out_features):
    return Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, quant_config=FP8)


def test_tuning_token_counts():
    # A context forward of max_num_tokens (capped by the sequence budget) and one
    # generation request, as DecoderWarmup._run_autotuner_warmup tunes them.
    tuning_token_counts = overlap_linear._tuning_token_counts
    assert tuning_token_counts(_llm_args(32768, 16), _Mapping(False), 996672) == (32768, 1)
    assert tuning_token_counts(_llm_args(8192, 4, draft=3), _Mapping(False), 1025) == (4096, 4)
    # No generation request under pipeline parallelism or guided decoding.
    assert tuning_token_counts(_llm_args(8192, 4), _Mapping(False, pp=True), 1025) == (4096,)
    guided = _llm_args(8192, 4, guided_decoding_backend="xgrammar")
    assert tuning_token_counts(guided, _Mapping(False), 1025) == (4096,)
    # Helix CP generation workers tune one batch of generation requests.
    assert tuning_token_counts(_llm_args(14, 2, draft=6), _Mapping(True), None) == (14,)
    assert tuning_token_counts(_llm_args(192, 32, draft=5), _Mapping(True), None) == (192,)


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_collect_plans_the_swept_shapes():
    with torch.device("meta"):
        model = nn.Module()
        model.fp8 = _fp8_linear(7168, 2048)
        model.wide = _fp8_linear(NO_LOW_M[1], NO_LOW_M[0])
        model.bf16 = Linear(1024, 1024, bias=False, dtype=torch.bfloat16)
    ctx = overlap_linear.collect(model, _llm_args(8192, 8), _Mapping(False))
    shapes = {(n, k): sweep for n, k, sweep in ctx.fp8}
    assert set(shapes) == {(2048, 7168), NO_LOW_M}
    # The 8192-token forward runs fp8_swap_ab_gemm for every shape; the one-request
    # forward only where the low-M GEMM does not serve one row.
    for (n, k), sweep in shapes.items():
        expected = _sweep(8192, 1) if overlap_linear._low_m_max(n, k) < 1 else _sweep(8192)
        assert sweep == expected, (n, k)
    assert not ctx.flashinfer_mxfp8

    # 14-token generation steps: the shapes whose low-M GEMM serves them are not swept.
    gen = overlap_linear.collect(model, _llm_args(14, 2, draft=6), _Mapping(True))
    assert overlap_linear._low_m_max(*NO_LOW_M) < 14
    swept = {(n, k) for n, k in shapes if overlap_linear._low_m_max(n, k) < 14}
    assert {(n, k): sweep for n, k, sweep in gen.fp8} == {s: _sweep(14) for s in swept}


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_kimi_k3_load_time_gemms(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    class KdaAttention(nn.Module):
        """What the fusion rule of KimiKDALinearAttention reads."""

        _fp8_qkvg_modules = KimiKDALinearAttention._fp8_qkvg_modules

        def __init__(self):
            super().__init__()
            self._dispatch = SimpleNamespace(decode_kernel_path="optimized")
            self.use_full_rank_gate = True
            for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                setattr(self, name, _fp8_linear(7168, 4096))

    def kimi_k3(layers):
        # What the load-time rules read. Calling the class would build the whole model
        # (PostInitCaller runs __init__, then __post_init__).
        model = kimi.KimiLinearForCausalLM.__new__(kimi.KimiLinearForCausalLM)
        nn.Module.__init__(model)
        model.model = SimpleNamespace(layers=layers)
        model._fp8_weight_read_moe_mlp = False
        return model

    monkeypatch.setattr(kimi, "_rubin_mxfp8_is_available", lambda: False)
    with torch.device("meta"):
        mla = nn.Module()
        mla.kv_b_proj = _fp8_linear(512, 24576)
        mla.o_proj = nn.Linear(16384, 7168, bias=False)
        layers = [
            SimpleNamespace(is_kda=False, self_attn=SimpleNamespace(mixer=mla)),
            SimpleNamespace(is_kda=True, linear_attn=KdaAttention()),
            # A layer of another pipeline stage.
            SimpleNamespace(_weights_removed=True),
        ]
        # Found on a submodule, as the text model that KimiK25ForConditionalGeneration wraps.
        model = nn.Module()
        model.llm = kimi_k3(layers)

    # The FP8 attention projections that _finalize_weight_load rewraps (not the BF16
    # o_proj), and the qkvg_proj that finalize_decode_weights fuses from q, k, v and g.
    assert overlap_linear._load_time_fp8_shapes(model) == {
        (24576, 512),
        (4096, 7168),
        (16384, 7168),
    }


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_warmup_overlap_builds_before_the_with_ends(monkeypatch):
    monkeypatch.setenv("TLLM_OVERLAP_PREWARM", "1")
    monkeypatch.delenv("TLLM_OVERLAP_PREWARM_LINEAR", raising=False)
    sampled = []
    monkeypatch.setattr(sampler_ops, "warmup_sampling_module", lambda: sampled.append(True))
    with torch.device("meta"):
        model = nn.Module()
        model.wide = _fp8_linear(NO_LOW_M[1], NO_LOW_M[0])
    model_config = SimpleNamespace(pretrained_config=None)

    with warmup_overlap(model, model_config, _llm_args(14, 2, draft=6), _Mapping(True)) as groups:
        pass

    groups = {group.name: group for group in groups}
    assert sampled == [True]
    assert (groups["warmup_steps"].count, groups["warmup_steps"].failed) == (1, 0)
    deep_gemm = groups["deep_gemm"]
    assert (deep_gemm.count, deep_gemm.failed) == (1, 0)
    assert deep_gemm.launches == len(_sweep(14))
    assert groups["flashinfer_mxfp8"].count == 0
    assert not any(t.is_alive() for group in groups.values() for t in group.threads)


def test_warmup_overlap_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv("TLLM_OVERLAP_PREWARM", "0")
    with warmup_overlap(nn.Module(), None, _llm_args(14, 2), _Mapping(True)) as groups:
        pass
    assert groups == []


def _wide_model():
    with torch.device("meta"):
        model = nn.Module()
        model.wide = _fp8_linear(NO_LOW_M[1], NO_LOW_M[0])
    return model


@pytest.fixture
def overlap_on(monkeypatch):
    monkeypatch.setenv("TLLM_OVERLAP_PREWARM", "1")
    monkeypatch.delenv("TLLM_OVERLAP_PREWARM_LINEAR", raising=False)
    monkeypatch.setattr(sampler_ops, "warmup_sampling_module", lambda: None)


def _overlap():
    """``warmup_overlap`` around a model with one swept FP8 GEMM (14-token Helix steps)."""
    model_config = SimpleNamespace(pretrained_config=None)
    return warmup_overlap(_wide_model(), model_config, _llm_args(14, 2, draft=6), _Mapping(True))


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_a_failed_build_is_left_to_the_warmup(monkeypatch, overlap_on):
    def fail(item, device, stop, free_samples):
        raise RuntimeError("nvcc failed")

    monkeypatch.setattr(overlap_linear, "build_deep_gemm", fail)
    with _overlap() as groups:
        pass
    deep_gemm = next(group for group in groups if group.name == "deep_gemm")
    assert (deep_gemm.count, deep_gemm.failed, deep_gemm.launches) == (1, 1, 0)
    assert not any(group.alive() for group in groups)


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_no_deep_gemm_threads_without_free_memory(monkeypatch, overlap_on):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1 << 30, 1 << 40))
    with _overlap() as groups:
        pass
    deep_gemm = next(group for group in groups if group.name == "deep_gemm")
    assert (deep_gemm.count, deep_gemm.threads) == (0, [])


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_a_sweep_is_skipped_without_free_memory(monkeypatch):
    # The loader may have taken the memory after the threads were sized.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1 << 30, 1 << 40))
    free_samples = []
    item = (*NO_LOW_M, _sweep(14))
    stop = threading.Event()
    assert overlap_linear.build_deep_gemm(item, torch.device("cuda"), stop, free_samples) == 0
    assert free_samples == [1 << 30]


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_the_builds_stop_when_the_loading_raises(monkeypatch, overlap_on):
    started = threading.Event()

    def wait_for_stop(item, device, stop, free_samples):
        started.set()
        stop.wait(timeout=30)
        return 0

    monkeypatch.setattr(overlap_linear, "build_deep_gemm", wait_for_stop)
    with pytest.raises(RuntimeError, match="loading failed"):
        with _overlap() as groups:
            assert started.wait(timeout=30)
            raise RuntimeError("loading failed")
    assert all(group.stop.is_set() for group in groups)
    assert not any(group.alive() for group in groups)


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_no_plan_with_the_autotuner_off():
    llm_args = _llm_args(8192, 8, enable_autotuner=False)
    plan = overlap_linear.collect(_wide_model(), llm_args, _Mapping(False))
    assert (plan.fp8, plan.flashinfer_mxfp8) == ([], False)
