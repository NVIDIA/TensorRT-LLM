# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-model DSpark on the tp4 target, phase W0 (``dspark.py``): the shell's calls on the DSpark worker take no
fused path, the capture is the in-tree DFlash capture, the drafter's borrowed embedding is the target's exact gather,
and the FP8 draft-KV override (an Eagle3 need) is never applied to a DSpark run, so the drafter's context pool keeps
the projection dtype (BF16) and binds to the separate draft KV cache.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
    m3_target_head,
)
from tensorrt_llm.quantization.mode import QuantAlgo


def _spec(
    *, dspark_mode: bool, eagle3: bool = False, one_engine: bool = True, embedded: bool = False
):
    mode = SimpleNamespace(
        use_one_engine=lambda: one_engine,
        is_dspark=lambda: dspark_mode,
        is_eagle3_one_model=lambda: eagle3,
    )
    return SimpleNamespace(spec_dec_mode=mode, draft_is_embedded_in_target=embedded)


def _worker() -> dspark.M3DSparkWorker:
    # The shell-facing calls read no worker state, so the worker is not constructed (that needs a mapping and a
    # DSpark spec config).
    return dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)


@pytest.mark.parametrize(
    "spec, expected",
    [
        (_spec(dspark_mode=True), True),
        (_spec(dspark_mode=True, embedded=True), False),
        (_spec(dspark_mode=True, one_engine=False), False),
        (_spec(dspark_mode=False, eagle3=True), False),
        (None, False),
    ],
)
def test_standalone_dspark_detection(spec, expected):
    assert dspark.is_standalone_dspark(spec) is expected


def test_fp8_draft_kv_override_is_eagle3_only():
    assert dspark.draft_kv_override_applies(_spec(dspark_mode=False, eagle3=True))
    assert dspark.draft_kv_override_applies(None)
    assert not dspark.draft_kv_override_applies(_spec(dspark_mode=True))


@pytest.mark.parametrize("target_algo", [QuantAlgo.NVFP4, QuantAlgo.FP8, None])
def test_drafter_kv_pool_keeps_the_drafter_dtype(target_algo):
    from tensorrt_llm.models.modeling_utils import QuantConfig

    whole, layer = (
        QuantConfig(kv_cache_quant_algo=target_algo),
        QuantConfig(kv_cache_quant_algo=target_algo),
    )
    _ = whole.quant_mode  # cached before the drop, as a loaded config may have it
    cfg = SimpleNamespace(quant_config=whole, quant_config_dict={"model.layers.0": layer})
    dspark.keep_drafter_kv_dtype(cfg)
    assert whole.kv_cache_quant_algo is None and layer.kv_cache_quant_algo is None
    assert not whole.quant_mode.has_fp8_kv_cache() and not whole.quant_mode.has_fp4_kv_cache()


def test_shell_calls_take_no_fused_path(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("W0 must never reach the coded target head")

    monkeypatch.setattr(m3_target_head, "prepare_head", _forbidden)
    monkeypatch.setattr(m3_target_head, "logits", _forbidden)
    worker = _worker()
    assert worker.tail_eligible(None, None, None) is False
    assert worker.arm_capture_norm(None, None, None, None) is None
    assert worker.prebuild(None, None) is None


def test_capture_is_the_in_tree_capture():
    calls = []
    meta = SimpleNamespace(maybe_capture_hidden_states=lambda *a: calls.append(a))
    hidden, residual = object(), object()
    _worker().capture_hidden_states(meta, 17, hidden, residual)
    _worker().capture_hidden_states(meta, 59, hidden)
    assert calls == [(17, hidden, residual), (59, hidden, None)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_embed_tokens_is_the_target_gather():
    gen = torch.Generator(device="cuda").manual_seed(31)
    table = torch.randn(1000, 64, generator=gen, device="cuda").to(torch.bfloat16)
    core = SimpleNamespace(w={"embed": table})
    embed = dspark.M3EmbedTokens(core)
    ids = torch.tensor([0, 5, 999, 200, -1], dtype=torch.long, device="cuda")
    out = embed(ids)
    ref = table[ids.clamp(min=0)] * (ids >= 0).unsqueeze(-1).to(table.dtype)
    assert embed.weight is table
    assert torch.equal(out, ref)
    assert not list(embed.parameters()), (
        "the table belongs to the target, not to the borrowed module"
    )


def test_borrowed_embedding_owns_no_parameters():
    # The weights walker asserts every shell parameter is fed: the borrowed module must add none.
    model = torch.nn.Module()
    model.w = torch.nn.ParameterDict(
        {"embed": torch.nn.Parameter(torch.zeros(8, 4), requires_grad=False)}
    )
    before = {n for n, _ in model.named_parameters()}
    model.embed_tokens = dspark.M3EmbedTokens(model)
    assert {n for n, _ in model.named_parameters()} == before == {"w.embed"}
    assert model.embed_tokens.weight is model.w["embed"]


class _FakeDrafter:
    def __init__(self, borrows: bool = True) -> None:
        self.loaded = None
        self.borrows = borrows
        self.lm_head = None
        self.draft_model_full = SimpleNamespace(model=SimpleNamespace(embed_tokens=None))

    def load_weights(self, weights, weight_mapper=None):
        self.loaded = dict(weights)

    def load_weights_from_target_model(self, target):
        if self.borrows:
            self.lm_head = target.lm_head
            self.draft_model_full.model.embed_tokens = target.model.embed_tokens


def _shell(drafter):
    return SimpleNamespace(
        draft_model=drafter,
        lm_head=object(),
        model=SimpleNamespace(embed_tokens=object()),
        named_parameters=lambda: iter([("lm_head.weight", None), ("model.w.embed", None)]),
    )


@pytest.mark.parametrize(
    "name",
    ["model.embed_tokens.weight", "language_model.model.embed_tokens.weight", "lm_head.weight"],
)
def test_draft_load_refuses_borrowed_tensors(name):
    shell = _shell(_FakeDrafter())
    with pytest.raises(ValueError, match="lends it"):
        dspark.load_dspark_draft_weights(shell, {name: torch.zeros(1), "fc.weight": torch.zeros(1)})
    assert shell.draft_model.loaded is None


def test_draft_load_borrows_target_modules():
    shell = _shell(_FakeDrafter())
    weights = {
        "fc.weight": torch.zeros(1),
        "markov_head.markov_w1.weight": torch.zeros(1),
        "confidence_head.proj.weight": torch.zeros(1),
    }
    dspark.load_dspark_draft_weights(shell, weights)
    assert shell.draft_model.loaded == weights
    assert shell.draft_model.lm_head is shell.lm_head
    with pytest.raises(RuntimeError, match="does not borrow"):
        dspark.load_dspark_draft_weights(_shell(_FakeDrafter(borrows=False)), weights)
