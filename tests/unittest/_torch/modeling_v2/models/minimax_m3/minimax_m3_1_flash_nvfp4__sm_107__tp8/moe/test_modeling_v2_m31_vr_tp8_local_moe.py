# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL_MOE on the sm_107 TP8 target: the 8-row decode MoE call runs the locality engine variant
(local_moe/: routed weights placed shard-major per locality domain, routed CTAs pinned to their
domain's SMs, the domain-major plan with the stock split stages). Its output is bit for bit the stock m3_moe engine's
at its default configuration (the engine the target builds when TRTLLM_M31_DISABLE_LOCAL_MOE=1), and its routed part
is bit for bit TRTLLM-Gen's NVFP4 MoE on the same routing."""

import importlib.util

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.local_moe import (  # noqa: E501
    m31_local_moe_op as local_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import check_routed_part, hidden_states, layer_for  # noqa: E402


def _two_domain_sm107() -> bool:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or importlib.util.find_spec("cutlass") is None
    ):
        return False
    return torch.cuda.get_device_properties(0).multi_processor_count == 2 * local_op.LOCAL_PD + 20


# The variant needs a two-domain sm_107 GPU (212 SMs: 96 + 96 routed CTAs by domain + 20 shared) and CuTe DSL.
_NEEDS_TWO_DOMAINS = pytest.mark.skipif(
    not _two_domain_sm107(),
    reason="needs a 212-SM sm_107 GPU with two locality domains and CuTe DSL",
)


def _stock_state():
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(
        dev, num_tokens=8, num_ctas=sms - moe_op.shared_ctas_for(8, True), shared=True
    )


def _ctas(tokens: int, shared: bool = True):
    """The routed CTA count the target passes (all SMs less the shared-expert CTAs)."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return sms - moe_op.shared_ctas_for(tokens, shared)


def _local_state(monkeypatch: pytest.MonkeyPatch, placement: str = "domain"):
    monkeypatch.setattr(local_op, "local_moe_enabled", lambda: True)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return local_op.LocalMoeState(
        dev, 8, num_ctas=sms - moe_op.shared_ctas_for(8, True), placement=placement
    )


@_NEEDS_TWO_DOMAINS
def test_make_state_takes_the_variant_at_8_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm.logger import logger

    seen = []
    monkeypatch.setattr(local_op, "_LOGGED", set())
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: seen.append(str(msg)))
    monkeypatch.setattr(local_op, "local_moe_enabled", lambda: True)
    dev = torch.device("cuda")
    st = local_op.make_state(dev, 8, num_ctas=_ctas(8), shared=True)
    assert isinstance(st, local_op.LocalMoeState)
    for tokens, shared in ((7, True), (8, False)):
        other = local_op.make_state(dev, tokens, num_ctas=_ctas(tokens, shared), shared=shared)
        assert type(other) is moe_op.M3MoeState
    taken = [m for m in seen if "LOCAL_MOE taken" in m]
    assert (
        len(taken) == 1 and "tokens=8" in taken[0] and "96+96 routed CTAs by domain" in taken[0]
    ), seen
    assert sum("LOCAL_MOE off" in m for m in seen) == 2, seen


@_NEEDS_TWO_DOMAINS
def test_kill_switch_builds_the_stock_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm.logger import logger

    seen = []
    monkeypatch.setattr(local_op, "_LOGGED", set())
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: seen.append(str(msg)))
    monkeypatch.setattr(local_op, "local_moe_enabled", lambda: False)
    st = local_op.make_state(torch.device("cuda"), 8, num_ctas=_ctas(8), shared=True)
    assert type(st) is moe_op.M3MoeState
    assert any("LOCAL_MOE off (TRTLLM_M31_DISABLE_LOCAL_MOE)" in m for m in seen), seen


@_NEEDS_TWO_DOMAINS
def test_receipt_and_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _local_state(monkeypatch)
    r = st.receipt
    assert r["grid"] == r["num_sms"] == 212 and r["per_domain"] == local_op.LOCAL_PD
    assert sorted(r["smap"]) == list(range(212))
    layer = layer_for(st, "random")
    w31, s31, w2, s2 = layer.local_weights
    lm = st.locality
    for t in (w31, s31, w2, s2):
        half = t.numel() // 2
        for k in range(2):
            p = t.data_ptr() + k * half
            assert lm.domain_of(p) == k and lm.domain_of(p + half - 1) == k
    assert r["placed_bytes"] == sum(t.numel() for t in (w31, s31, w2, s2))


@_NEEDS_TWO_DOMAINS
@pytest.mark.parametrize("placement", ["domain", "swapped", "plain"])
@pytest.mark.parametrize("case", ["random", "concentrated"])
@pytest.mark.parametrize("seed", [31, 107])
def test_bitwise_vs_stock_engine(
    monkeypatch: pytest.MonkeyPatch, placement: str, case: str, seed: int
) -> None:
    hidden = hidden_states(8, seed=seed)
    stock = _stock_state()
    ref = stock.run(layer_for(stock, case), hidden).clone()
    st = _local_state(monkeypatch, placement)
    layer = layer_for(st, case)
    for rep in range(3):
        got = st.run(layer, hidden)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
            placement,
            case,
            seed,
            rep,
        )


@_NEEDS_TWO_DOMAINS
def test_routed_part_is_trtllm_gen(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _local_state(monkeypatch)
    check_routed_part(lambda h, layer: st.run(layer, h), st, 8, seed=5)


@_NEEDS_TWO_DOMAINS
def test_graph_replay_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = hidden_states(8, seed=7)
    stock = _stock_state()
    ref = stock.run(layer_for(stock, "random"), hidden).clone()
    st = _local_state(monkeypatch)
    layer = layer_for(st, "random")
    out = torch.empty_like(ref)
    st.run(layer, hidden, out)  # compiles outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        st.run(layer, hidden, out)
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
