# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-K3 ULL MLA attention-side scheduling switches against today's path, one GPU.

``TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER``: the FP4 packed decode (K3's kernel module) whose split
reduction the fused Helix tail folds (``DeferredPackedReduce(release_early=False)``) releases the
tail when its producer warps finish the last KV tile (``epilogue``) instead of at its completion.
The tail, launched as its programmatic dependent, stages ``W_UV`` and the plan during the decode's
end and folds after ``griddepcontrol.wait``. Checks: the tail output (fold, V up-projection,
combine, MXFP8 o_proj operand) is bitwise that of the late-released chain, eagerly and over CUDA
graph replays, at n1 / n2 with FP16 and FP32 split partials; the switch parsing, which needs a
kernel module with the epilogue trigger.

``TRTLLM_K3_ULL3_MLA_GATE_AT``: the per-batch placement policy of the side-stream output gate
parses as documented. ``TRTLLM_K3_ULL3_MLA_LATE_TAIL_MAX_TOKENS`` widens the late tail release.

The FP4 cache and chain helpers are those of ``test_k3_ull2_mla.py`` / ``test_k3_mla_q_fused.py``
(same directory).
"""

import pytest
import test_k3_ull2_mla as u2
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

H, LORA, Q_LEN = u2.H, u2.LORA, u2.Q_LEN
TRIGGER_ENV = u2.TRIGGER_ENV


@pytest.fixture(autouse=True)
def _k3_decode_kernel(monkeypatch):
    u2.qf.use_k3_decode_kernel(monkeypatch)


class _Tail:
    """The fused tail on a local receive buffer (``P = 1``): fold, V up-projection, push into its
    own buffer, combine, MXFP8 operand of ``bf16(out * gate)``."""

    def __init__(self, num_tokens, seed):
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
            k3_helix_tail as tail_mod,
        )

        self.mod = tail_mod
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        self.w_uv = (torch.randn((H, 128, LORA), device="cuda", generator=gen) * 0.05).to(
            torch.bfloat16
        )
        k = H * 128
        self.gate = torch.sigmoid(torch.randn((num_tokens, k), device="cuda", generator=gen)).to(
            torch.bfloat16
        )
        self.out = torch.empty((num_tokens, k), dtype=torch.bfloat16, device="cuda")
        self.xq = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device="cuda")
        self.xs = torch.empty((num_tokens, k // 32), dtype=torch.uint8, device="cuda")
        self.half_bytes = num_tokens * tail_mod.record_bytes(H, 128) + 4096
        self.buffer = torch.full((2 * self.half_bytes // 4,), -1, dtype=torch.int32, device="cuda")
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")

    def launch(self, rows, stats, fold):
        self.mod.k3_helix_tail(
            rows,
            stats,
            self.w_uv,
            self.out,
            fold=fold,
            zero_kv_mask=None,
            cp_rank=0,
            cp_size=1,
            ws_addr=self.buffer.data_ptr(),
            rank_stride=self.buffer.numel() * 4,
            half_bytes=self.half_bytes,
            state=self.state,
            max_ctas=torch.cuda.get_device_properties(0).multi_processor_count,
            use_pdl=True,
            mx_out=(self.gate, self.xq, self.xs),
            ull2=True,
        )

    def result(self):
        torch.cuda.synchronize()
        return self.out.clone(), self.xq.view(torch.uint8).clone(), self.xs.clone()


def _assert_same(got, ref):
    """Tail output (BF16), MXFP8 codes and exponents bit for bit."""
    assert torch.equal(got[0].view(torch.int16), ref[0].view(torch.int16))
    assert torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])


def _layer(chain, tail):
    """Decode with the split reduction deferred (late release), then the fused tail on it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.helix_tail import (
        fold_inputs_of,
    )
    from tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce import (
        DeferredPackedReduce,
    )

    with DeferredPackedReduce(release_early=False) as deferred:
        chain.run(early_trigger=True)
    fold = fold_inputs_of(deferred, chain.out.view(chain.T, -1), chain.stats)
    assert fold is not None, "the split reduction was not deferred"
    tail.launch(chain.out.view(chain.T * H, LORA), chain.stats.view(chain.T * H, 2), fold)


@u2.qf.needs_k3_decode
@pytest.mark.parametrize("f16_partials", [True, False], ids=["fp16", "fp32"])
@pytest.mark.parametrize("n", [1, 2])
def test_tail_trigger_matches_late_release(n, f16_partials, monkeypatch, k3_features):
    from tensorrt_llm._torch.attention.backends import fp4_mla

    monkeypatch.setenv("TRTLLM_FP4_MLA_REDUCE_PDL", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "8")
    monkeypatch.setenv("TRTLLM_FP4_MLA_F16_PARTIALS", "1" if f16_partials else "0")
    k3_features(ull2_mla_tail=True)
    monkeypatch.setenv(u2.PDL_ENV, "1")
    module = fp4_mla._fp4_mla_cutedsl_kernel_module()
    assert getattr(module, "LAUNCH_DEPENDENTS_EPILOGUE", 0), module.__name__
    seen = []
    inner = module.run_trtllm_fp4_mla_decode_packed

    def recorder(*args, **kwargs):
        seen.append(kwargs.get("launch_dependents", False))
        return inner(*args, **kwargs)

    monkeypatch.setattr(module, "run_trtllm_fp4_mla_decode_packed", recorder)
    chain = u2._Chain(n, 6000 + 1500 * n, seed=70 + n)
    tail = _Tail(chain.T, seed=5 + n)
    try:
        monkeypatch.delenv(TRIGGER_ENV, raising=False)
        _layer(chain, tail)
        ref = tail.result()
        assert seen[-1] is False
        assert torch.isfinite(ref[0].float()).all() and ref[0].abs().max() > 0
        monkeypatch.setenv(TRIGGER_ENV, "epilogue")
        for _ in range(2):
            tail.out.fill_(torch.nan)
            _layer(chain, tail)
            _assert_same(tail.result(), ref)
            assert seen[-1] == module.LAUNCH_DEPENDENTS_EPILOGUE
        # CUDA graph: the programmatic edges decode -> tail as the model captures them.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _layer(chain, tail)
        torch.cuda.synchronize()
        for _ in range(3):
            tail.out.fill_(torch.nan)
            graph.replay()
            _assert_same(tail.result(), ref)
        assert bool((tail.buffer == -1).all()), "the receive buffer was not re-armed"
    finally:
        chain.mgr.shutdown()


@u2.qf.needs_k3_decode
def test_tail_trigger_switch(monkeypatch):
    from tensorrt_llm._torch.attention.backends import fp4_mla
    from tensorrt_llm._torch.attention.backends.fp4_mla import config as fp4_mla_config

    dec = fp4_mla._fp4_mla_cutedsl_kernel_module()
    for value, mode in [
        (None, 0),
        ("0", 0),
        ("epilogue", dec.LAUNCH_DEPENDENTS_EPILOGUE),
    ]:
        if value is None:
            monkeypatch.delenv(TRIGGER_ENV, raising=False)
        else:
            monkeypatch.setenv(TRIGGER_ENV, value)
        assert fp4_mla.ull3_mla_tail_trigger() == mode
    for bad in ("entry", "last"):
        monkeypatch.setenv(TRIGGER_ENV, bad)
        with pytest.raises(ValueError):
            fp4_mla.ull3_mla_tail_trigger()
    # The entry release of the early modes stays the boolean True.
    assert dec.LAUNCH_DEPENDENTS_ENTRY == True  # noqa: E712
    # The shared kernel module has no epilogue trigger: the switch is rejected, not ignored.
    monkeypatch.setattr(fp4_mla_config, "_CUTEDSL_KERNEL_MODULE_OVERRIDE", None)
    assert not getattr(fp4_mla._fp4_mla_cutedsl_kernel_module(), "LAUNCH_DEPENDENTS_EPILOGUE", 0)
    monkeypatch.setenv(TRIGGER_ENV, "epilogue")
    with pytest.raises(ValueError):
        fp4_mla.ull3_mla_tail_trigger()
    monkeypatch.delenv(TRIGGER_ENV)
    assert fp4_mla.ull3_mla_tail_trigger() == 0


def test_gate_place_policy():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla import (
        kimi_k3_mla_attention as k3mla,
    )

    assert k3mla._parse_ull3_gate_at("decode") == ("decode", 16, "decode")
    assert k3mla._parse_ull3_gate_at("exchange:16,start") == ("exchange", 16, "start")
    assert k3mla._parse_ull3_gate_at("exchange") == ("exchange", 16, "exchange")
    assert k3mla._parse_ull3_gate_at("exchange:8,start") == ("exchange", 8, "start")
    for bad in ("early", "exchange,late", "start:x", "query", "tail"):
        with pytest.raises(ValueError):
            k3mla._parse_ull3_gate_at(bad)


def test_late_tail_token_limit(k3_features):
    """``features.ull3_mla_late_tail_max_tokens`` widens the late tail release (``ull2_mla_tail``)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.helix_tail import (
        decode_releases_tail_early,
    )

    k3_features(ull2_mla_tail=True, ull3_mla_late_tail_max_tokens=16)
    assert not decode_releases_tail_early(16) and decode_releases_tail_early(32)
    k3_features(ull3_mla_late_tail_max_tokens=32)
    assert not decode_releases_tail_early(32) and decode_releases_tail_early(33)
    k3_features(ull2_mla_tail=False)
    assert decode_releases_tail_early(8)
