# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's drafter block decode (``drafter_block_decode``) on the drafter.

Host-only. The shell decides its plan once, at construction. Under
``NO_PATHS`` its drafter load is the built-in one and the drafter's block
attention stays FlashAttention's ``flash_attn_with_kvcache``; with the block
decode planned the drafter's ``_dflash_flash_attention`` is wrapped by
``draft.WindowedBlockAttention``, its ``dflash_forward`` is the written-out
block decode, and no module or parameter changes.
"""

from __future__ import annotations

import inspect
import types

import pytest
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime as _runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM, DFlashLagunaForCausalLM
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_runtime_hooks(monkeypatch):
    """The executor hooks pin host staging blocks; these shells are built without them."""
    monkeypatch.setattr(_runtime, "make_runtime_hooks", lambda model: None)


_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep


def _flash_attn_with_kvcache(**kwargs):
    return "flash-attn"


class _Drafter(nn.Module):
    def __init__(self, backend: str = "VANILLA") -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.dflash_attention_backend = backend
        self._dflash_flash_attention = _flash_attn_with_kvcache if backend == "VANILLA" else None


class _Llm(nn.Module):
    def __init__(self, drafter) -> None:
        super().__init__()
        self.model = nn.Module()
        self.draft_model = drafter


@pytest.fixture
def build(monkeypatch):
    """Build shells through their own constructor, the built-in one and the load stubbed."""
    events = []

    def builtin_init(self, model_config, *args, **kwargs):
        events.append("built-in model")
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = _Llm(_Drafter())

    def builtin_load_draft_weights(self, *args, **kwargs):
        events.append(("built-in draft load", args, kwargs))
        return "loaded"

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration, "load_draft_weights", builtin_load_draft_weights
    )
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(
            lambda cls, model, plan=dispatch.NO_PATHS, compute_capability=None: events.append(
                "adopt"
            )
        ),
    )

    def _build(*paths):
        plan = dispatch.plan_of(*paths)
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        return _SHELL(model_config=None)

    _build.events = events
    return _build


@pytest.fixture
def block_decode(monkeypatch):
    """The written-out block decode bound as ``draft.install_block_decode`` binds it; the
    stand-in drafter is not a ``DFlashForCausalLM``, which the real check refuses."""

    def install(draft_model, enable_pdl):
        draft_model.dflash_forward = types.MethodType(draft.windowed_block_decode, draft_model)

    monkeypatch.setattr(draft, "install_block_decode", install)


def test_the_drafter_load_keeps_the_builtin_signature():
    assert (
        inspect.getfullargspec(_SHELL.load_draft_weights).args
        == inspect.getfullargspec(MuseSparkOpenOmniForConditionalGeneration.load_draft_weights).args
    )


def test_no_feature_is_the_builtin_drafter_load(build):
    shell = build()
    assert shell._plan == dispatch.NO_PATHS
    drafter = shell.llm.draft_model
    params = {n: p.data_ptr() for n, p in drafter.named_parameters()}
    assert shell.load_draft_weights("weights", "mapper") == "loaded"
    assert build.events == [
        "built-in model",
        "adopt",
        ("built-in draft load", ("weights", "mapper"), {}),
    ]
    assert drafter._dflash_flash_attention is _flash_attn_with_kvcache
    assert {n: p.data_ptr() for n, p in drafter.named_parameters()} == params


def test_the_block_decode_wraps_the_drafter_block_attention_and_nothing_else(
    build, block_decode, monkeypatch
):
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    shell = build("drafter_block_decode")
    assert shell._plan == dispatch.plan_of("drafter_block_decode")
    drafter = shell.llm.draft_model
    modules = list(drafter.modules())
    params = {n: p.data_ptr() for n, p in drafter.named_parameters()}
    assert shell.load_draft_weights("weights") == "loaded"
    attention = drafter._dflash_flash_attention
    assert isinstance(attention, draft.WindowedBlockAttention)
    assert attention.builtin is _flash_attn_with_kvcache and attention.enable_pdl
    assert drafter.__dict__["dflash_forward"].__func__ is draft.windowed_block_decode
    assert list(drafter.modules()) == modules
    assert {n: p.data_ptr() for n, p in drafter.named_parameters()} == params
    # A second load (the engine may reload draft weights) does not wrap twice.
    shell.load_draft_weights("weights")
    assert drafter._dflash_flash_attention is attention


def test_the_block_attention_always_launches_with_pdl(build, block_decode, monkeypatch):
    # PDL is a precondition of the composition; the install reads no switch.
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    shell = build("drafter_block_decode")
    shell.load_draft_weights("weights")
    assert shell.llm.draft_model._dflash_flash_attention.enable_pdl is True


def test_the_plan_is_decided_once_at_construction(monkeypatch, build, block_decode):
    shell = build("drafter_block_decode")
    monkeypatch.setattr(modeling._dispatch, "compose", lambda *a, **k: dispatch.NO_PATHS)
    shell.load_draft_weights("weights")
    assert isinstance(shell.llm.draft_model._dflash_flash_attention, draft.WindowedBlockAttention)

    shell = build()
    monkeypatch.setattr(
        modeling._dispatch, "compose", lambda *a, **k: dispatch.plan_of("drafter_block_decode")
    )
    shell.load_draft_weights("weights")
    assert shell.llm.draft_model._dflash_flash_attention is _flash_attn_with_kvcache


@pytest.mark.parametrize("backend", ["TRTLLM", "FA4"])
def test_a_paged_backend_drafter_is_refused_and_left_alone(backend):
    """No VANILLA block attention to wrap: the block decode raises instead of
    running the built-in attention."""
    drafter = _Drafter(backend)
    assert not draft.install_windowed_block_attention(drafter, enable_pdl=False)
    assert drafter._dflash_flash_attention is None
    for model in (drafter, None):
        with pytest.raises(
            ValueError, match="drafter_block_decode cannot run here: .*no VANILLA block attention"
        ):
            draft.install(model, False, True)
    assert drafter._dflash_flash_attention is None


# --------------------------------------------------------------------------
# The written-out block decode
# --------------------------------------------------------------------------
def test_block_decode_installs_the_windowed_attention_attention_with_it(monkeypatch):
    logged = []
    # Only this target's lines: the shared logger also says "PDL enabled" on first use.
    monkeypatch.setattr(
        draft.logger, "info", lambda msg: msg.startswith("[musespark_open]") and logged.append(msg)
    )
    drafter = _windowed()
    drafter._dflash_flash_attention = _flash_attn_with_kvcache
    draft.install(drafter, False, True)
    attention = drafter._dflash_flash_attention
    assert isinstance(attention, draft.WindowedBlockAttention)
    assert drafter.__dict__["dflash_forward"].__func__ is draft.windowed_block_decode
    assert [msg.split(":")[0] for msg in logged] == [
        "[musespark_open] drafter_block_decode",
        "[musespark_open] drafter_block_decode",
    ]
    # The shell installs again from the post-load walk: nothing changes, nothing is logged.
    forward = drafter.__dict__["dflash_forward"]
    draft.install(drafter, False, True)
    assert drafter._dflash_flash_attention is attention
    assert drafter.__dict__["dflash_forward"] is forward
    assert len(logged) == 2


def test_block_decode_on_a_drafter_it_does_not_write_out_fails_the_load(build):
    """The stand-in is not a DFlashForCausalLM: the windowed attention installs,
    then the written-out block decode refuses, and the drafter load raises instead of serving the built-in
    forward."""
    shell = build("drafter_block_decode")
    drafter = shell.llm.draft_model
    with pytest.raises(
        ValueError, match="drafter_block_decode cannot run here: not a DFlashForCausalLM"
    ):
        shell.load_draft_weights("weights")
    assert "dflash_forward" not in drafter.__dict__


def _bare(cls):
    """An instance of ``cls`` without running its constructor (class attributes only)."""
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


def test_block_decode_refuses_drafters_whose_forward_it_does_not_write_out():
    class OwnForward(DFlashForCausalLM):
        def dflash_forward(self, *args, **kwargs):
            raise AssertionError

    assert "not a DFlashForCausalLM" in draft.block_decode_unsupported_reason(_Drafter())
    assert "overrides dflash_forward" in draft.block_decode_unsupported_reason(_bare(OwnForward))
    assert "gates the attention output" in draft.block_decode_unsupported_reason(
        _bare(DFlashLagunaForCausalLM)
    )
    plain = _bare(DFlashForCausalLM)
    plain.dflash_attention_backend = "TRTLLM"
    assert "not VANILLA" in draft.block_decode_unsupported_reason(plain)
    assert draft.install_block_decode(plain, enable_pdl=False) is not None
    assert "dflash_forward" not in plain.__dict__


def _windowed(heads=64, kv_heads=8, head_dim=128, window=(2047, 2047), block_size=8):
    """A VANILLA DFlashForCausalLM stand-in with five layers of the given attention geometry."""
    drafter = _bare(DFlashForCausalLM)
    drafter.dflash_attention_backend = "VANILLA"
    drafter.block_size = block_size
    attn = types.SimpleNamespace(num_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim)
    drafter.model = nn.Module()
    drafter.model.layers = [types.SimpleNamespace(self_attn=attn) for _ in range(5)]
    drafter._resolve_block_attention = lambda layer_idx: (False, window)
    return drafter


def test_block_decode_takes_the_published_drafter_geometry():
    # TP1 (64 / 8 heads) and one TP8 rank (8 / 1 heads), width 128, window 2047.
    assert draft.block_decode_unsupported_reason(_windowed()) is None
    assert draft.block_decode_unsupported_reason(_windowed(heads=8, kv_heads=1)) is None


@pytest.mark.parametrize(
    "geometry, match",
    [
        (dict(head_dim=96), "head width"),
        (dict(heads=136, kv_heads=8), "block rows"),
        (dict(heads=12, kv_heads=8), "multiple of KV heads"),
        (dict(window=(70_000, 70_000)), "spans"),
        (dict(window=(-1, -1)), "causal or unwindowed"),
        (dict(block_size=None), "no block size"),
    ],
    ids=["width", "tile", "gqa-ratio", "too-wide", "unwindowed", "no-block"],
)
def test_block_decode_refuses_an_attention_geometry_the_op_does_not_take(geometry, match):
    """The block decode calls the op with no per-call fallback, so the install checks every layer."""
    drafter = _windowed(**geometry)
    assert match in draft.block_decode_unsupported_reason(drafter)
    assert draft.install_block_decode(drafter, enable_pdl=False) is not None
    assert "dflash_forward" not in drafter.__dict__
