# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``full_attention_prep_append`` on the attention lattice: its refusals, the classes and the
call chain; ``drop_o_proj_prefetch``: the full-attention layers' QKV sites lose their attention
output prefetch.

Host-only, on a stand-in text model whose layers own built-in
``MuseSparkOpenAttention`` objects created without their constructor (layer
``i`` of ``n`` is a full-attention NoPE layer when ``(n - 1 - i) % 4 == 0``).
The built-in forward, prep and attention op are recorders that keep the
built-in order (prep, then attention op); the attention op, as the built-in
TRTLLM backend does, calls ``thop.trtllm_gen_generation_preprocess`` through
``flashinfer_trtllm_gen``'s module global. What the kernels compute is the GPU
tests' business (``attention/test_modeling_v2_preattn_prep_append.py`` and
``test_modeling_v2_musespark_open_compose_prep_append.py`` on the real weights).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention_forward,
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi_gen
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenAttention,
    MuseSparkOpenModel,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.bindings.internal import thop

pytestmark = pytest.mark.cpu_only

_PREP = target_attention.MuseSparkOpenTepAttention
_BOTH = target_attention.MuseSparkOpenTepPrepDecodeAttention
_PLAN = target_attention.DecodeAttentionPlan


def _full(index: int, num_layers: int) -> bool:
    return (num_layers - 1 - index) % 4 == 0


def _builtin_attention(layer_idx: int, full: bool) -> MuseSparkOpenAttention:
    attention = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(attention)
    attention.layer_idx = layer_idx
    attention._fuse_qk_norm_rope_gate = False
    attention.fuse_qk_norm_rope = False
    attention.is_qk_norm = False
    attention.q_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.k_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.rope_fusion = False
    attention.attn_backend = "TRTLLM"
    attention.support_fused_qkv = True
    attention.head_dim = 64
    attention.num_heads = 16
    attention.num_key_value_heads = 2
    attention.scale_query_by = 1.0
    attention._qkv_layout_inverse = None
    attention.is_sliding = not full
    attention.attention_window_size = None if full else 2048
    attention.skip_rope = full
    rotary = None
    if not full:
        rotary = RotaryEmbedding.__new__(RotaryEmbedding)
        nn.Module.__init__(rotary)
        rotary.is_neox = False
        rotary.inverse = False
        rotary.rotary_cos_sin = torch.zeros(16, 2, 32)
    attention.rotary_emb = rotary
    return attention


class _Layer(nn.Module):
    def __init__(self, index: int, num_layers: int) -> None:
        super().__init__()
        self.self_attn = _builtin_attention(index, full=_full(index, num_layers))
        # The built-in layer's parallel mapping; None: a group of one (no TP agreement).
        self.mapping = None


def _Model(num_layers: int = 8) -> MuseSparkOpenModel:
    """A built-in text model as far as the core's adoption reads it."""
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_Layer(i, num_layers) for i in range(num_layers)])
    return model


def _adopted(cls, num_layers: int = 8) -> MuseSparkOpenModel:
    model = _Model(num_layers)
    for layer in model.layers:
        cls.adopt(layer.self_attn)
    return model


@pytest.fixture
def adoptable(monkeypatch):
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    monkeypatch.setattr(full_attention, "preprocess_reachable", lambda: None)


# --------------------------------------------------------------------------
# The id
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The classes
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "base", (_PREP, _BOTH), ids=("preattn_prep", "preattn_prep+swa_paged_decode")
)
def test_install_puts_the_mixin_in_front_of_the_full_layers_only(adoptable, base):
    model = _adopted(base)
    assert full_attention.install(model) == 2
    for index, layer in enumerate(model.layers):
        cls = type(layer.self_attn)
        if _full(index, len(model.layers)):
            assert cls.__mro__[1:3] == (full_attention.FullAppendAttention, base), cls.__mro__
            assert cls.__name__ == f"{base.__name__}FullAppend"
            assert (
                layer.self_attn._prep_append_folded is None
                and layer.self_attn._prep_append_metadata is None
            )
        else:
            assert cls is base
    # One class per base: a second model shares it.
    other = _adopted(base)
    full_attention.install(other)
    assert type(other.layers[3].self_attn) is type(model.layers[3].self_attn)


def test_install_refuses_attention_without_preattn_prep_and_changes_nothing(adoptable):
    model = _adopted(target_attention.MuseSparkOpenTepDecodeAttention)
    before = [type(layer.self_attn) for layer in model.layers]
    with pytest.raises(
        ValueError, match="full_attention_prep_append cannot fold .*not the preattn_prep class"
    ):
        full_attention.install(model)
    assert [type(layer.self_attn) for layer in model.layers] == before


def test_install_refuses_when_the_builtin_preprocess_call_moved(monkeypatch):
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    monkeypatch.setattr(full_attention, "preprocess_reachable", lambda: "it moved")
    model = _adopted(_PREP)
    with pytest.raises(ValueError, match="cannot skip the built-in preprocess: it moved"):
        full_attention.install(model)
    assert all(type(layer.self_attn) is _PREP for layer in model.layers)


@pytest.mark.parametrize(
    "paths, base",
    (
        (("full_attention_prep_append",), _PREP),
        (("full_attention_prep_append", "swa_paged_decode"), _BOTH),
    ),
)
def test_the_core_installs_prep_append_after_the_attention_lattice(
    adoptable, monkeypatch, paths, base
):
    """``full_attention_prep_append`` sees every attention already re-classed by the lattice, then re-classes the
    full-attention layers in front of that class."""
    seen = []
    real_install = full_attention.install

    def install(model):
        seen.append([type(layer.self_attn) for layer in model.layers])
        return real_install(model)

    monkeypatch.setattr(full_attention, "install", install)
    model = _Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of(*paths))
    assert seen == [[base] * 4]
    assert [type(layer.self_attn).__mro__[1] for layer in model.layers[3:]] == [
        full_attention.FullAppendAttention
    ]
    assert [type(layer.self_attn) for layer in model.layers[:3]] == [base] * 3


def test_the_core_does_not_import_prep_append_unless_selected(adoptable, monkeypatch):
    monkeypatch.setattr(full_attention, "install", lambda model: pytest.fail("installed"))
    model = _Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("preattn_prep"))
    assert all(type(layer.self_attn) is _PREP for layer in model.layers)


def test_the_check_reads_the_builtin_call():
    """On this tree the built-in call is the one ``eliding`` replaces."""
    assert full_attention.preprocess_reachable() is None
    assert len(full_attention.PREPROCESS_ARGS) == 43


# --------------------------------------------------------------------------
# The call chain
# --------------------------------------------------------------------------
@pytest.fixture
def recorded(monkeypatch):
    """Recorders for the built-in forward, prep and attention op; the folded prep stubbed.

    The built-in attention op calls the generation preprocess through
    ``flashinfer_trtllm_gen.thop`` the way ``run_generation`` does, with a stub
    real ``thop`` behind it, and records what came back.
    """
    taken: list = []

    def builtin_forward(self, position_ids, hidden_states, attn_metadata, **kwargs):
        q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
        return self._attn_impl(q, k, v, attn_metadata, None, None, None, None, None)

    def builtin_prep(self, qkv, position_ids):
        taken.append(("builtin_prep", self.layer_idx))
        return "builtin-qkv", None, None, None

    def builtin_attn(self, q, *args, **kwargs):
        result = fi_gen.thop.trtllm_gen_generation_preprocess("call", q)
        taken.append(("preprocess", self.layer_idx, result))
        return "attn", None

    class _RealThop:
        def trtllm_gen_generation_preprocess(self, *args, **kwargs):
            return "real-preprocess"

    class _Folded:
        def __init__(self, layer_idx, match):
            self.layer_idx = layer_idx
            self.match = match
            self.consumed = False

        def mismatch(self, args, kwargs):
            return None if self.match else "stub mismatch"

        def outputs(self, args):
            return f"views-{self.layer_idx}"

    def folded_prep(self, qkv, md):
        taken.append(("folded_prep", self.layer_idx, md))
        return "folded-qkv", "gate", _Folded(self.layer_idx, md != "mismatch")

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", builtin_forward)
    monkeypatch.setattr(MuseSparkOpenAttention, "preprocess_qkv", builtin_prep)
    monkeypatch.setattr(MuseSparkOpenAttention, "_attn_impl", builtin_attn)
    monkeypatch.setattr(_PREP, "_fits_preattn_prep", lambda self, qkv, position_ids: True)
    monkeypatch.setattr(
        full_attention.FullAppendAttention,
        "_prep_append_decline",
        lambda self, qkv, md: "stub decline" if md == "decline" else None,
    )
    monkeypatch.setattr(full_attention.FullAppendAttention, "_prep_append_prep", folded_prep)
    monkeypatch.setattr(fi_gen, "thop", _RealThop())
    return taken


def _preattn_prep_entry(taken):
    def entry(qkv, *args):
        taken.append(("preattn_prep_prep",))
        return "preattn-prep-qkv", "gate"

    return entry


def test_a_decode_call_on_a_full_layer_runs_the_folded_prep_and_skips_the_preprocess(
    adoptable, recorded
):
    model = _adopted(_PREP, 4)
    full_attention.install(model)
    full, swa = model.layers[3].self_attn, model.layers[0].self_attn
    entry = _preattn_prep_entry(recorded)
    full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=entry)
    swa(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=entry)
    assert recorded == [
        ("folded_prep", 3, "md"),
        ("preprocess", 3, "views-3"),
        ("preattn_prep_prep",),
        ("preprocess", 0, "real-preprocess"),
    ]
    assert full._prep_append_folded is None and full._prep_append_metadata is None
    assert isinstance(fi_gen.thop, object) and not isinstance(
        fi_gen.thop, attention_forward._ElidingThop
    )


def test_a_prefill_or_declined_or_mismatched_call_runs_the_builtin_path(adoptable, recorded):
    model = _adopted(_PREP, 4)
    full_attention.install(model)
    full = model.layers[3].self_attn
    # No preattn_prep: a prefill step (PrefillTarget passes none).
    full(position_ids=None, hidden_states="qkv", attn_metadata="md")
    # Declined: the preattn_prep prep and the real preprocess.
    entry = _preattn_prep_entry(recorded)
    full(position_ids=None, hidden_states="qkv", attn_metadata="decline", preattn_prep=entry)
    # The preprocess call does not match the folded prep: the real call runs.
    full(position_ids=None, hidden_states="qkv", attn_metadata="mismatch", preattn_prep=entry)
    assert recorded == [
        ("builtin_prep", 3),
        ("preprocess", 3, "real-preprocess"),
        ("preattn_prep_prep",),
        ("preprocess", 3, "real-preprocess"),
        ("folded_prep", 3, "mismatch"),
        ("preprocess", 3, "real-preprocess"),
    ]


def test_the_folded_state_is_scoped_to_one_forward_call(adoptable, recorded, monkeypatch):
    model = _adopted(_PREP, 4)
    full_attention.install(model)
    full = model.layers[3].self_attn

    def failing(self, q, *args, **kwargs):
        assert isinstance(fi_gen.thop, attention_forward._ElidingThop)
        raise RuntimeError("attention failed")

    monkeypatch.setattr(MuseSparkOpenAttention, "_attn_impl", failing)
    entry = _preattn_prep_entry(recorded)
    with pytest.raises(RuntimeError, match="attention failed"):
        full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=entry)
    assert full._prep_append_folded is None and full._prep_append_metadata is None
    assert not isinstance(fi_gen.thop, attention_forward._ElidingThop)


def test_the_proxy_forwards_every_other_attribute():
    class _Real:
        marker = 7

        def trtllm_gen_generation_preprocess(self, *args, **kwargs):
            return "real"

    class _Folded:
        consumed = False

        def mismatch(self, args, kwargs):
            return None

        def outputs(self, args):
            return "views"

    proxy = attention_forward._ElidingThop(_Real(), _Folded())
    assert proxy.marker == 7
    assert proxy.trtllm_gen_generation_preprocess(1) == "views"
    # A second call in the same window is never the folded one.
    assert proxy.trtllm_gen_generation_preprocess(1) == "real"
    assert attention_forward._ElidingThop(thop, _Folded()).build_trtllm_gen_kv_cache_metadata is (
        thop.build_trtllm_gen_kv_cache_metadata
    )


# --------------------------------------------------------------------------
# drop_o_proj_prefetch: the full-attention layers' attention output prefetch
# --------------------------------------------------------------------------
def _with_dense_gemms(model: MuseSparkOpenModel, prefetch: bool = True):
    """Give each stand-in attention a QKV projection carrying the dense_gemms dispatcher, as
    ``dense_gemms.install`` wires it: the QKV site prefetches the layer's o_proj weight."""
    plan = dense_gemms.DenseGemmPlan({}, prefetch=prefetch, enable_pdl=False)
    for layer in model.layers:
        attention = layer.self_attn
        attention.qkv_proj = nn.Module()
        attention.o_proj = nn.Module()
        attention.o_proj.weight = nn.Parameter(torch.empty(4, 4))
        site = dense_gemms._Site(
            plan, "qkv", attention.layer_idx, (4, 4), lambda a=attention: a.o_proj.weight
        )
        attention.qkv_proj.__dict__["_low_m_dispatcher"] = dense_gemms.DenseGemmDispatcher(site)
    return plan


def _qkv_next(model: MuseSparkOpenModel, missing=None) -> list:
    """Each layer's QKV site's next weight (``missing`` where the projection has no dispatcher)."""
    nexts = []
    for layer in model.layers:
        dispatcher = vars(layer.self_attn.qkv_proj).get("_low_m_dispatcher")
        nexts.append(missing if dispatcher is None else dispatcher.site.next_weight)
    return nexts


def test_drop_clears_the_full_layers_qkv_prefetch_only(adoptable):
    model = _adopted(_PREP)
    _with_dense_gemms(model)
    before = _qkv_next(model)
    assert full_attention.drop_o_proj_prefetch(model) == 2
    for index, (old, new) in enumerate(zip(before, _qkv_next(model))):
        if _full(index, len(model.layers)):
            assert new is None
        else:
            assert new is old and new() is model.layers[index].self_attn.o_proj.weight


@pytest.mark.parametrize("broken", ("no_dispatcher", "no_prefetch", "other_weight", "no_next"))
def test_drop_refuses_and_changes_nothing(adoptable, broken):
    model = _adopted(_PREP)
    _with_dense_gemms(model, prefetch=broken != "no_prefetch")
    full = model.layers[7].self_attn
    if broken == "no_dispatcher":
        del vars(full.qkv_proj)["_low_m_dispatcher"]
    elif broken == "other_weight":
        site = vars(full.qkv_proj)["_low_m_dispatcher"].site
        site.next_weight = lambda: model.layers[0].self_attn.o_proj.weight
    elif broken == "no_next":
        vars(full.qkv_proj)["_low_m_dispatcher"].site.next_weight = None
    before = _qkv_next(model, missing="no dispatcher")
    with pytest.raises(ValueError, match="drop_o_proj_prefetch cannot drop .*layer 7"):
        full_attention.drop_o_proj_prefetch(model)
    assert _qkv_next(model, missing="no dispatcher") == before


def test_the_core_drops_the_prefetch_after_prep_append_only_with_the_prefetch(
    adoptable, monkeypatch
):
    order = []
    real_install, real_drop = full_attention.install, full_attention.drop_o_proj_prefetch

    def install(model):
        order.append("prep_append")
        return real_install(model)

    def drop(model):
        order.append("prep_append_prefetch_trim")
        return real_drop(model)

    monkeypatch.setattr(
        dense_gemms, "install", lambda model, prefetch: _with_dense_gemms(model, prefetch)
    )
    monkeypatch.setattr(full_attention, "install", install)
    monkeypatch.setattr(full_attention, "drop_o_proj_prefetch", drop)

    model = _Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(
        model, dispatch.plan_of("full_attention_prep_append", "dense_gemm_prefetch")
    )
    assert order == ["prep_append", "prep_append_prefetch_trim"]
    assert [n is None for n in _qkv_next(model)] == [False, False, False, True]

    order.clear()
    model = _Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(
        model, dispatch.plan_of("full_attention_prep_append", "dense_gemms")
    )
    assert order == ["prep_append"]
    assert all(n is not None for n in _qkv_next(model))


# --------------------------------------------------------------------------
# The capture census: one line per captured graph
# --------------------------------------------------------------------------
def _forward_all(model, md, entry):
    """One forward over the model's full-attention layers, in order."""
    for layer in model.layers:
        attention = layer.self_attn
        if not attention.is_sliding:
            attention(position_ids=None, hidden_states="qkv", attn_metadata=md, preattn_prep=entry)


def test_the_census_reports_each_captured_forward_once(adoptable, recorded, monkeypatch):
    model = _adopted(_PREP, 8)
    full_attention.install(model)
    census = model.layers[3].self_attn._prep_append_census
    assert census is model.layers[7].self_attn._prep_append_census and census.layers == (3, 7)
    entry = _preattn_prep_entry(recorded)
    # Eager forwards report nothing, whatever they did.
    _forward_all(model, "md", entry)
    _forward_all(model, "decline", entry)
    assert list(census.reports) == [] and census.captures == 0

    monkeypatch.setattr(full_attention, "_capturing", lambda: True)
    _forward_all(model, "md", entry)
    number, folded, total, level, line = census.reports[-1]
    assert (number, folded, total, level) == (1, 2, 2, "info")
    assert line.startswith(
        "[musespark_open] full_attention_prep_append: CUDA graph capture 1: 2 of 2 full-attention "
        "layers fold"
    )
    assert ";" not in line


@pytest.mark.parametrize(
    "md, decline, level, words",
    (
        # A step the fold does not cover by its shape: info, nothing folded.
        ("shape", "9 rows", "info", "not the folded shape (9 rows) on layers 3, 7"),
        ("shape", "not one generation sequence", "info", "(not one generation sequence)"),
        # A step of the folded shape that stays on the built-in path: a warning.
        ("decline", "stub decline", "warning", "declined (stub decline) on layers 3, 7"),
        ("mismatch", None, "warning", "built-in preprocess ran (stub mismatch) on layers 3, 7"),
    ),
)
def test_the_census_warns_only_when_a_step_of_the_folded_shape_is_not_folded(
    adoptable, recorded, monkeypatch, md, decline, level, words
):
    model = _adopted(_PREP, 8)
    full_attention.install(model)
    monkeypatch.setattr(
        full_attention.FullAppendAttention,
        "_prep_append_decline",
        lambda self, qkv, m: decline if m in ("shape", "decline") else None,
    )
    monkeypatch.setattr(full_attention, "_capturing", lambda: True)
    _forward_all(model, md, _preattn_prep_entry(recorded))
    census = model.layers[3].self_attn._prep_append_census
    _, folded, total, got_level, line = census.reports[-1]
    assert (folded, total, got_level) == (0, 2, level), line
    assert words in line, line


def test_the_census_counts_a_prefill_and_a_partial_fold(adoptable, recorded, monkeypatch):
    model = _adopted(_PREP, 8)
    full_attention.install(model)
    monkeypatch.setattr(full_attention, "_capturing", lambda: True)
    entry = _preattn_prep_entry(recorded)
    first, last = model.layers[3].self_attn, model.layers[7].self_attn
    # A prefill graph (no decode prep anywhere): recorded, not logged, nothing folded.
    first(position_ids=None, hidden_states="qkv", attn_metadata="md")
    last(position_ids=None, hidden_states="qkv", attn_metadata="md")
    census = first._prep_append_census
    assert census.reports[-1][1:4] == (0, 2, "silent"), census.reports[-1]
    assert "no decode prep on layers 3, 7" in census.reports[-1][4]
    # One layer folds, the other's preprocess call does not match: a warning naming layer 7.
    first(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=entry)
    last(position_ids=None, hidden_states="qkv", attn_metadata="mismatch", preattn_prep=entry)
    number, folded, total, level, line = census.reports[-1]
    assert (number, folded, total, level) == (2, 1, 2, "warning"), line
    assert "built-in preprocess ran (stub mismatch) on layers 7" in line


def test_eager_decode_steps_are_counted(adoptable, recorded, monkeypatch):
    """Eager forwards are counted host-side (a graph replay runs no Python: it does what
    its capture recorded), without a log line."""
    model = _adopted(_PREP, 8)
    full_attention.install(model)
    grows = "a workspace the attention grows first"
    monkeypatch.setattr(
        full_attention.FullAppendAttention,
        "_prep_append_decline",
        lambda self, qkv, m: grows if m == "decline" else None,
    )
    entry = _preattn_prep_entry(recorded)
    first, last = model.layers[3].self_attn, model.layers[7].self_attn
    census = first._prep_append_census
    # A prefill (no decode prep anywhere) is not a decode step.
    first(position_ids=None, hidden_states="qkv", attn_metadata="md")
    last(position_ids=None, hidden_states="qkv", attn_metadata="md")
    assert census.eager_steps == 0
    _forward_all(model, "md", entry)
    _forward_all(model, "decline", entry)
    assert (census.eager_steps, census.eager_fallback_steps) == (2, 1)
    assert census.eager_reasons == {f"declined ({grows})": 2}
    assert list(census.reports) == [] and census.captures == 0
    logged = []
    monkeypatch.setattr(full_attention.logger, "info", logged.append)
    _forward_all(model, "md", entry)
    _forward_all(model, "decline", entry)
    assert (census.eager_steps, census.eager_fallback_steps) == (4, 2)
    assert logged == []


def test_the_folded_prep_launch_carries_the_trigger(adoptable, monkeypatch):
    """``_prep_append_prep`` launches the op with programmatic dependent launch as the
    environment says (host stand-ins for the metadata, the cache view and the
    workspace layout)."""
    from types import SimpleNamespace

    calls = []

    def op(*args):
        calls.append(args)
        return "qkv-out", "gate"

    layout = dict(
        q_buf_offset=0,
        q_buf_size=1024,
        bmm1_scale_offset=1024,
        bmm1_scale_size=8,
        bmm2_scale_offset=1032,
        bmm2_scale_size=4,
    )
    monkeypatch.setattr(full_attention, "preattn_prep_append", op)
    monkeypatch.setattr(full_attention, "generation_layout", lambda *args: layout)
    monkeypatch.setattr(full_attention, "get_bmm1_scale", lambda attn: 0.125)
    monkeypatch.setattr(
        full_attention,
        "thop",
        SimpleNamespace(
            build_trtllm_gen_kv_cache_metadata=lambda *args: (
                "kv-pool",
                torch.zeros(1, 2, 4, dtype=torch.int32),
                None,
            )
        ),
    )
    model = _adopted(_PREP)
    full_attention.install(model)
    attention = model.layers[3].self_attn
    attention.attn = SimpleNamespace(
        get_local_layer_idx=lambda md: 3,
        quant_mode=0,
        kv_scale_orig_quant=torch.ones(1),
        kv_scale_quant_orig=torch.ones(1),
        rope_params=SimpleNamespace(dim=0),
    )
    monkeypatch.setattr(
        full_attention.FullAppendAttention,
        "_prep_append_fmha_lib",
        lambda self: SimpleNamespace(_get_total_num_blocks=lambda md: 4),
    )
    md = SimpleNamespace(
        host_kv_cache_pool_pointers=torch.zeros(1, 2, dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        kv_cache_block_offsets=torch.zeros(1, 1, 2, 4, dtype=torch.int32),
        tokens_per_block=32,
        effective_workspace=torch.zeros(2048, dtype=torch.uint8),
        kv_lens_cuda_runtime=torch.zeros(1, dtype=torch.int32),
    )
    qkv = torch.zeros(8, 2 * (16 + 2) * 64, dtype=torch.bfloat16)
    qkv_out, gate, folded = attention._prep_append_prep(qkv, md)
    assert (qkv_out, gate, folded.rows) == ("qkv-out", "gate", 8)
    assert len(calls) == 1
    *_, enable_pdl = calls[0]
    assert enable_pdl is True
