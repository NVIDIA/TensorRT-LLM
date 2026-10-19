# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``full_attention_split_kv`` on the prep-append full-attention layers: its refusals, the classes
and the call chain; the kernel's spill-free build (``install_spill_free``) on top.

Host-only, on ``test_modeling_v2_musespark_open_prep_append.py``'s stand-in text model (its
helpers are loaded from beside this file). The built-in attention op is a recorder that, as
``run_generation`` does, calls the generation preprocess through ``flashinfer_trtllm_gen.thop``
and then the decode FMHA through ``flashinfer_trtllm_gen.flashinfer`` with ``run_generation``'s
keyword arguments; ``full_attention_split_kv`` and ``combine_norm_gate`` are recorders. What the kernels
compute is the GPU tests' business (``attention/test_modeling_v2_full_attention_split_kv.py`` and
``test_modeling_v2_musespark_open_compose_split_kv.py`` on the real weights).
"""

from __future__ import annotations

import importlib.util
import os
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention_forward,
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_split_kv import (  # noqa: E501
    attention as split_kv_attention,
)
from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi_gen
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention

pytestmark = pytest.mark.cpu_only


def _prep_append_features():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        here, "..", "full_attention_prep_append", "test_modeling_v2_musespark_open_prep_append.py"
    )
    spec = importlib.util.spec_from_file_location("_prep_append_tests_for_split_kv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP_APPEND = _prep_append_features()
_PREP = target_attention.MuseSparkOpenTepAttention
ROWS = 8


def _model(num_layers: int = 4):
    """The stand-in model with the preattn_prep class, full_attention_prep_append installed, and
    what the split-KV checks read."""
    model = PREP_APPEND._adopted(_PREP, num_layers)
    for layer in model.layers:
        attention = layer.self_attn
        attention.attn_output_gate = True
        attention.attn = types.SimpleNamespace(
            head_dim=64, q_scaling=1.0, kv_scale_orig_quant=torch.ones(1)
        )
        attention.o_head_norm = types.SimpleNamespace(variance_epsilon=1e-5)
    full_attention.install(model)
    return model


@pytest.fixture
def adoptable(monkeypatch):
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    monkeypatch.setattr(full_attention, "preprocess_reachable", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 3))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: types.SimpleNamespace(multi_processor_count=152),
    )
    monkeypatch.setattr(split_kv_attention, "_capturing", lambda: False)


# --------------------------------------------------------------------------
# The built-in call it intercepts
# --------------------------------------------------------------------------
def test_the_check_reads_the_builtin_decode_call():
    """On this tree the built-in decode FMHA call is the one ``intercepting`` replaces."""
    assert split_kv_attention.decode_reachable() is None


# --------------------------------------------------------------------------
# The classes
# --------------------------------------------------------------------------
def test_install_puts_the_mixin_in_front_of_prep_append_on_the_full_layers_only(adoptable):
    model = _model()
    folded = [type(layer.self_attn) for layer in model.layers]
    assert split_kv_attention.install(model) == 1
    for index, layer in enumerate(model.layers):
        cls = type(layer.self_attn)
        if PREP_APPEND._full(index, len(model.layers)):
            assert cls.__mro__[1:3] == (split_kv_attention.FullAttnSplitKvAttention, folded[index])
            assert layer.self_attn._split_kv_splits == 152
            assert layer.self_attn._split_kv_pending is None
        else:
            assert cls is folded[index]


def test_install_refuses_without_prep_append_or_off_sm103_and_changes_nothing(
    adoptable, monkeypatch
):
    model = PREP_APPEND._adopted(_PREP, 4)
    for layer in model.layers:
        layer.self_attn.attn_output_gate = True
        layer.self_attn.attn = types.SimpleNamespace(head_dim=64, q_scaling=1.0)
    before = [type(layer.self_attn) for layer in model.layers]
    with pytest.raises(
        ValueError, match="full_attention_prep_append's folded prep is not installed"
    ):
        split_kv_attention.install(model)
    assert [type(layer.self_attn) for layer in model.layers] == before
    model = _model()
    before = [type(layer.self_attn) for layer in model.layers]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    with pytest.raises(ValueError, match="sm_103 and sm_107 only"):
        split_kv_attention.install(model)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 3))
    model.layers[3].self_attn.attn.q_scaling = 2.0
    with pytest.raises(ValueError, match="softmax scale"):
        split_kv_attention.install(model)
    monkeypatch.setattr(split_kv_attention, "decode_reachable", lambda: "it moved")
    with pytest.raises(ValueError, match="cannot replace the full-attention decode FMHA: it moved"):
        split_kv_attention.install(model)
    assert [type(layer.self_attn) for layer in model.layers] == before


def test_the_core_installs_split_kv_after_prep_append_only_when_selected(adoptable, monkeypatch):
    seen = []
    real_install = split_kv_attention.install

    def install(model):
        seen.append([type(layer.self_attn).__mro__[1] for layer in model.layers])
        for layer in model.layers:
            layer.self_attn.attn_output_gate = True
            layer.self_attn.attn = types.SimpleNamespace(head_dim=64, q_scaling=1.0)
        return real_install(model)

    monkeypatch.setattr(split_kv_attention, "install", install)
    model = PREP_APPEND._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("full_attention_split_kv"))
    # The prep-append mixin is on the full layer when the split-KV attention installs, and the
    # split-KV mixin goes in front of it.
    assert seen[0][3] is full_attention.FullAppendAttention
    assert type(model.layers[3].self_attn).__mro__[1] is split_kv_attention.FullAttnSplitKvAttention
    model = PREP_APPEND._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("full_attention_prep_append"))
    assert len(seen) == 1
    assert not isinstance(model.layers[3].self_attn, split_kv_attention.FullAttnSplitKvAttention)


# --------------------------------------------------------------------------
# The call chain
# --------------------------------------------------------------------------
def _decode_kwargs(rows: int = ROWS, **change) -> dict:
    kwargs = dict(
        query=torch.zeros(rows, 16, 64),
        kv_cache=None,
        workspace_buffer=torch.zeros(4),
        block_tables=torch.zeros(1, 2, 64, dtype=torch.int32),
        seq_lens=None,
        max_seq_len=4096,
        bmm1_scale=torch.ones(2),
        bmm2_scale=torch.ones(1),
        window_left=-1,
        out=torch.zeros(rows, 16, 64, dtype=torch.bfloat16),
        sinks=None,
        kv_layout="HND",
        enable_pdl=True,
        backend="trtllm-gen",
        q_len_per_req=rows,
        max_q_len=None,
        cum_seq_lens_q=None,
        kv_cache_sf=None,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=None,
        multi_ctas_kv_counter_buffer=torch.zeros(4, dtype=torch.int32),
    )
    kwargs.update(change)
    return kwargs


@pytest.fixture
def recorded(monkeypatch):
    """Recorders for the built-in forward, prep, attention op and output gate, the folded prep,
    ``full_attention_split_kv`` and ``combine_norm_gate``. ``state["decode"]`` sets what the built-in
    attention op passes the decode FMHA."""
    taken: list = []
    pool = torch.zeros(10, 2, 32, 64, dtype=torch.float8_e4m3fn)
    seq_lens = torch.tensor([4096], dtype=torch.int32)
    state = {"decode": {}}

    def builtin_forward(self, position_ids, hidden_states, attn_metadata, **kwargs):
        q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
        output, _ = self._attn_impl(q, k, v, attn_metadata, None, None, None, None, None)
        return self.apply_output_gate(output, gate)

    def builtin_attn(self, q, *args, **kwargs):
        fi_gen.thop.trtllm_gen_generation_preprocess("call", q)
        decode = _decode_kwargs(kv_cache=(pool, pool), seq_lens=seq_lens)
        decode.update(state["decode"])
        fi_gen.flashinfer.decode.trtllm_batch_decode_with_kv_cache(**decode)
        return decode["out"].view(ROWS, -1), None

    def builtin_gate(self, output, gate):
        taken.append(("builtin_gate", self.layer_idx))
        return "gated"

    class _RealDecode:
        marker = 11

        def trtllm_batch_decode_with_kv_cache(self, **kwargs):
            taken.append(("builtin_fmha", kwargs["window_left"]))

    class _RealThop:
        def trtllm_gen_generation_preprocess(self, *args, **kwargs):
            return "real-preprocess"

    def folded_prep(self, qkv, md):
        folded = full_attention.FoldedPreprocess.__new__(full_attention.FoldedPreprocess)
        folded.qkv_out = torch.zeros(ROWS, 1280, dtype=torch.bfloat16)
        folded.rows = ROWS
        folded.seq_lens_ptr = seq_lens.data_ptr()
        folded.bmm1_host = 0.125
        folded.consumed = False
        folded.fallback = None if md != "mismatch" else None
        folded.mismatch = lambda args, kwargs: "stub mismatch" if md == "mismatch" else None
        folded.outputs = lambda args: "views"
        return "folded-qkv", "gate", folded

    def v4(qkv, kv_pool, tables, lens, scale, splits, pdl, prefetch, evict_first, rescale_cols):
        taken.append(
            (
                "full_attention_split_kv",
                tuple(tables.shape),
                splits,
                pdl,
                prefetch,
                evict_first,
                rescale_cols,
            )
        )
        return "part_o", "part_lse"

    def cng(part_o, part_lse, rows, gate, eps, trigger, seq_lens, heads=16):
        taken.append(("combine_norm_gate", part_o, rows, gate, trigger, seq_lens is not None))
        state.setdefault("cng_heads", []).append(heads)
        return "merged-gated"

    def builtin_prep(self, qkv, position_ids):
        return "builtin-qkv", None, None, "gate"

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", builtin_forward)
    monkeypatch.setattr(MuseSparkOpenAttention, "preprocess_qkv", builtin_prep)
    monkeypatch.setattr(MuseSparkOpenAttention, "_attn_impl", builtin_attn)
    monkeypatch.setattr(MuseSparkOpenAttention, "apply_output_gate", builtin_gate)
    monkeypatch.setattr(_PREP, "_fits_preattn_prep", lambda self, qkv, position_ids: True)
    monkeypatch.setattr(full_attention.FullAppendAttention, "_prep_append_decline", lambda *a: None)
    monkeypatch.setattr(full_attention.FullAppendAttention, "_prep_append_prep", folded_prep)
    monkeypatch.setattr(fi_gen, "thop", _RealThop())
    monkeypatch.setattr(fi_gen, "flashinfer", types.SimpleNamespace(decode=_RealDecode()))
    monkeypatch.setattr(split_kv_attention._v4, "full_attention_split_kv", v4)
    monkeypatch.setattr(split_kv_attention._cng, "combine_norm_gate", cng)
    return types.SimpleNamespace(taken=taken, state=state)


def _entry(qkv, *args):
    return "preattn-prep-qkv", "gate"


def test_a_folded_decode_call_runs_the_partials_and_the_merge(adoptable, recorded):
    model = _model()
    split_kv_attention.install(model)
    full = model.layers[3].self_attn
    out = full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)
    assert out == "merged-gated"
    assert recorded.taken == [
        ("full_attention_split_kv", (2, 64), 152, True, False, True, 64),
        ("combine_norm_gate", "part_o", ROWS, "gate", "exit", True),
    ]
    assert full._split_kv_pending is None
    assert not isinstance(fi_gen.flashinfer, attention_forward._FlashinferStandIn)


@pytest.mark.parametrize(
    "change, reason",
    (
        (dict(window_left=2047), "a sliding window"),
        (dict(q_len_per_req=None, max_q_len=ROWS), "another query length"),
        (dict(sinks=torch.zeros(16)), "attention sinks"),
        (dict(block_tables=torch.zeros(2, 2, 64, dtype=torch.int32)), "not one sequence's"),
        (dict(block_tables=torch.zeros(1, 2, 152 * 256 + 1, dtype=torch.int32)), "more pages"),
        (dict(kv_layout="NHD"), "another KV layout"),
    ),
)
def test_a_call_that_is_not_the_plain_case_runs_the_builtin_fmha(
    adoptable, recorded, change, reason
):
    model = _model()
    split_kv_attention.install(model)
    full = model.layers[3].self_attn
    recorded.state["decode"] = change
    out = full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)
    assert out == "gated"
    assert recorded.taken == [("builtin_fmha", change.get("window_left", -1)), ("builtin_gate", 3)]
    assert full._split_kv_pending is None


def test_a_prefill_or_a_mismatched_preprocess_runs_the_builtin_path(adoptable, recorded):
    model = _model()
    split_kv_attention.install(model)
    full = model.layers[3].self_attn
    # A prefill (no preattn_prep): no folded prep, the built-in FMHA and gate.
    full(position_ids=None, hidden_states="qkv", attn_metadata="md")
    # The built-in preprocess ran (its call did not match the folded prep): the built-in FMHA.
    full(position_ids=None, hidden_states="qkv", attn_metadata="mismatch", preattn_prep=_entry)
    assert recorded.taken == [
        ("builtin_fmha", -1),
        ("builtin_gate", 3),
        ("builtin_fmha", -1),
        ("builtin_gate", 3),
    ]


def test_unmerged_partials_raise(adoptable, recorded, monkeypatch):
    model = _model()
    split_kv_attention.install(model)
    full = model.layers[3].self_attn

    def no_gate(self, position_ids, hidden_states, attn_metadata, **kwargs):
        q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
        return self._attn_impl(q, k, v, attn_metadata, None, None, None, None, None)

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", no_gate)
    with pytest.raises(RuntimeError, match="left full_attention_split_kv partials unmerged"):
        full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)
    assert full._split_kv_pending is None


def test_the_gate_refuses_another_attention_output(adoptable, recorded, monkeypatch):
    model = _model()
    split_kv_attention.install(model)
    full = model.layers[3].self_attn

    def other_output(self, position_ids, hidden_states, attn_metadata, **kwargs):
        q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
        # The attention output stays alive, so the other tensor cannot take its address.
        output, _ = self._attn_impl(q, k, v, attn_metadata, None, None, None, None, None)
        other = torch.zeros(ROWS, 1024, dtype=torch.bfloat16)
        assert other.data_ptr() != output.data_ptr()
        return self.apply_output_gate(other, gate)

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", other_output)
    with pytest.raises(RuntimeError, match="did not get the attention output"):
        full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)


def test_the_stand_in_forwards_every_other_attribute_and_launches_once():
    class _Decode:
        marker = 5

        def trtllm_batch_decode_with_kv_cache(self, **kwargs):
            return "real"

    call = types.SimpleNamespace(launched=True, fallback=None)
    stand_in = attention_forward._FlashinferStandIn(
        types.SimpleNamespace(decode=_Decode(), x=3), call
    )
    assert stand_in.x == 3 and stand_in.decode.marker == 5
    # A second decode call in one window is never the one the partials stand for.
    assert stand_in.decode.trtllm_batch_decode_with_kv_cache(window_left=-1) == "real"
    assert call.fallback == "a second call"


def test_the_census_reports_each_captured_forward(adoptable, recorded, monkeypatch):
    model = _model(8)
    split_kv_attention.install(model)
    census = model.layers[3].self_attn._split_kv_census
    assert census.layers == (3, 7)
    monkeypatch.setattr(full_attention, "_capturing", lambda: False)
    monkeypatch.setattr(split_kv_attention, "_capturing", lambda: True)
    monkeypatch.setattr(
        split_kv_attention._v4,
        "_compiled_keys",
        {(None, True, split_kv_attention.PREFETCH, split_kv_attention.EVICT_FIRST)},
    )
    for index in (3, 7):
        model.layers[index].self_attn(
            position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry
        )
    assert census.reports[-1][1:4] == (2, 2, "info")
    recorded.state["decode"] = dict(window_left=5)
    for index in (3, 7):
        model.layers[index].self_attn(
            position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry
        )
    assert census.reports[-1][1:4] == (0, 2, "warning")
    recorded.state["decode"] = {}
    for index in (3, 7):
        model.layers[index].self_attn(position_ids=None, hidden_states="qkv", attn_metadata="md")
    assert census.reports[-1][3] == "silent"
    # A build not compiled before the capture is not launched inside it.
    monkeypatch.setattr(split_kv_attention._v4, "_compiled_keys", set())
    for index in (3, 7):
        model.layers[index].self_attn(
            position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry
        )
    assert census.reports[-1][1:4] == (0, 2, "warning")
    assert "not compiled before this capture" in census.reports[-1][4]


# --------------------------------------------------------------------------
# The kernel's spill-free build
# --------------------------------------------------------------------------
def test_the_spill_free_key_is_its_own_and_the_default_key_is_unchanged():
    v4 = split_kv_attention._v4
    assert v4.compiled_key(0, True, False, True) == (0, True, False, True)
    assert v4.compiled_key(0, True, False, True, 64) == (0, True, False, True)
    assert v4.compiled_key(0, True, False, True, v4.SPILL_FREE_RESCALE_COLS) == (
        0,
        True,
        False,
        True,
        ("rescale_cols", 16),
    )


def test_install_spill_free_sets_every_split_kv_layer_and_refuses_without_split_kv(adoptable):
    model = _model(8)
    with pytest.raises(
        ValueError, match=r"layers \[3, 7\] do not run full_attention_split_kv; install it first"
    ):
        split_kv_attention.install_spill_free(model)
    split_kv_attention.install(model)
    assert {model.layers[i].self_attn._split_kv_rescale_cols for i in (3, 7)} == {64}
    assert split_kv_attention.install_spill_free(model) == 2
    for index, layer in enumerate(model.layers):
        if index in (3, 7):
            assert layer.self_attn._split_kv_rescale_cols == 16
        else:
            assert not hasattr(layer.self_attn, "_split_kv_rescale_cols")


def test_the_core_installs_spill_free_right_after_split_kv(adoptable, monkeypatch):
    seen = []
    real_install = split_kv_attention.install
    real_spill_free = split_kv_attention.install_spill_free

    def install(model):
        for layer in model.layers:
            layer.self_attn.attn_output_gate = True
            layer.self_attn.attn = types.SimpleNamespace(head_dim=64, q_scaling=1.0)
        return real_install(model)

    def install_spill_free(model):
        seen.append(type(model.layers[3].self_attn).__mro__[1])
        return real_spill_free(model)

    monkeypatch.setattr(split_kv_attention, "install", install)
    monkeypatch.setattr(split_kv_attention, "install_spill_free", install_spill_free)
    model = PREP_APPEND._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("full_attention_split_kv"))
    assert seen == [split_kv_attention.FullAttnSplitKvAttention]
    assert model.layers[3].self_attn._split_kv_rescale_cols == 16
    model = PREP_APPEND._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("full_attention_prep_append"))
    assert len(seen) == 1
    assert not hasattr(model.layers[3].self_attn, "_split_kv_rescale_cols")


def test_a_spill_free_call_launches_the_spill_free_build(adoptable, recorded, monkeypatch):
    model = _model(8)
    split_kv_attention.install(model)
    split_kv_attention.install_spill_free(model)
    full = model.layers[3].self_attn
    out = full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)
    assert out == "merged-gated"
    assert recorded.taken == [
        ("full_attention_split_kv", (2, 64), 152, True, False, True, 16),
        ("combine_norm_gate", "part_o", ROWS, "gate", "exit", True),
    ]
    # Inside a capture only a compiled build launches: the default build's key does not admit
    # the spill-free build.
    census = full._split_kv_census
    monkeypatch.setattr(full_attention, "_capturing", lambda: False)
    monkeypatch.setattr(split_kv_attention, "_capturing", lambda: True)
    default = (None, True, split_kv_attention.PREFETCH, split_kv_attention.EVICT_FIRST)
    monkeypatch.setattr(split_kv_attention._v4, "_compiled_keys", {default})
    for index in (3, 7):
        model.layers[index].self_attn(
            position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry
        )
    assert census.reports[-1][1:4] == (0, 2, "warning")
    assert "not compiled before this capture" in census.reports[-1][4]
    monkeypatch.setattr(
        split_kv_attention._v4, "_compiled_keys", {default + (("rescale_cols", 16),)}
    )
    for index in (3, 7):
        model.layers[index].self_attn(
            position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry
        )
    assert census.reports[-1][1:4] == (2, 2, "info")


# --------------------------------------------------------------------------
# A tep4 rank: 32 query / 4 KV heads
# --------------------------------------------------------------------------
def _tep4(model):
    """The stand-in's full-attention layers at a tep4 rank's heads."""
    for layer in model.layers:
        layer.self_attn.num_heads = 32
        layer.self_attn.num_key_value_heads = 4
    return model


def test_install_picks_the_tep4_build_on_a_tep4_rank(adoptable):
    model = _tep4(_model(8))
    assert split_kv_attention.install(model) == 2
    for index in (3, 7):
        attention = model.layers[index].self_attn
        # Two pairs of KV heads share the SMs: half of them per pair.
        assert attention._split_kv_splits == 76
        assert attention._split_kv_kv_heads == 4
    tep8 = _model(8)
    split_kv_attention.install(tep8)
    assert tep8.layers[3].self_attn._split_kv_splits == 152
    assert tep8.layers[3].self_attn._split_kv_kv_heads == 2


@pytest.mark.parametrize("heads, kv_heads", ((8, 1), (64, 8), (24, 3), (32, 2), (16, 4)), ids=str)
def test_install_refuses_a_rank_shape_with_no_build(adoptable, heads, kv_heads):
    model = _model()
    for layer in model.layers:
        layer.self_attn.num_heads = heads
        layer.self_attn.num_key_value_heads = kv_heads
    before = [type(layer.self_attn) for layer in model.layers]
    with pytest.raises(ValueError, match="neither a tep8 rank's"):
        split_kv_attention.install(model)
    assert [type(layer.self_attn) for layer in model.layers] == before


def test_install_refuses_layers_of_mixed_rank_shapes(adoptable):
    model = _model(8)
    model.layers[7].self_attn.num_heads = 32
    model.layers[7].self_attn.num_key_value_heads = 4
    with pytest.raises(ValueError, match="mix rank shapes"):
        split_kv_attention.install(model)


def test_a_tep4_folded_decode_call_launches_the_tep4_build(adoptable, recorded, monkeypatch):
    """The decline reads the tep4 shapes (a [blocks, 4, 32, 64] pool, a 2560-wide operand, a
    [rows, 32, 64] output, 512 pages per split) and the merge runs at 32 heads."""
    model = _tep4(_model())
    split_kv_attention.install(model)
    full = model.layers[3].self_attn
    real_prep = full_attention.FullAppendAttention._prep_append_prep

    def tep4_prep(self, qkv, md):
        q, gate, folded = real_prep(self, qkv, md)
        folded.qkv_out = torch.zeros(ROWS, 2560, dtype=torch.bfloat16)
        return q, gate, folded

    monkeypatch.setattr(full_attention.FullAppendAttention, "_prep_append_prep", tep4_prep)
    pool = torch.zeros(10, 4, 32, 64, dtype=torch.float8_e4m3fn)
    recorded.state["decode"] = dict(
        kv_cache=(pool, pool),
        out=torch.zeros(ROWS, 32, 64, dtype=torch.bfloat16),
        block_tables=torch.zeros(1, 2, 76 * 512, dtype=torch.int32),
    )
    out = full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)
    assert out == "merged-gated"
    assert recorded.taken == [
        ("full_attention_split_kv", (2, 76 * 512), 76, True, False, True, 64),
        ("combine_norm_gate", "part_o", ROWS, "gate", "exit", True),
    ]
    assert recorded.state["cng_heads"] == [32]
    # A tep8 pool, a tep8 operand or a table past 512 pages per split: the built-in FMHA.
    for change in (
        dict(kv_cache=(torch.zeros(10, 2, 32, 64, dtype=torch.float8_e4m3fn),) * 2),
        dict(block_tables=torch.zeros(1, 2, 76 * 512 + 1, dtype=torch.int32)),
        dict(out=torch.zeros(ROWS, 16, 64, dtype=torch.bfloat16)),
    ):
        recorded.taken.clear()
        recorded.state["decode"] = dict(recorded.state["decode"], **change)
        assert (
            full(position_ids=None, hidden_states="qkv", attn_metadata="md", preattn_prep=_entry)
            == "gated"
        )
        assert recorded.taken[0][0] == "builtin_fmha", change
        recorded.state["decode"] = dict(
            kv_cache=(pool, pool),
            out=torch.zeros(ROWS, 32, 64, dtype=torch.bfloat16),
            block_tables=torch.zeros(1, 2, 76 * 512, dtype=torch.int32),
        )


def test_the_tep4_key_is_its_own():
    v4 = split_kv_attention._v4
    assert v4.compiled_key(0, True, False, True, 64, 2) == (0, True, False, True)
    assert v4.compiled_key(0, True, False, True, 16, 4) == (
        0,
        True,
        False,
        True,
        ("rescale_cols", 16),
        ("kv_heads", 4),
    )
    assert v4.build_options(v4.compiled_key(0, True, False, True, 16, 4)[4:]) == {
        "rescale_cols": 16,
        "kv_heads": 4,
    }
