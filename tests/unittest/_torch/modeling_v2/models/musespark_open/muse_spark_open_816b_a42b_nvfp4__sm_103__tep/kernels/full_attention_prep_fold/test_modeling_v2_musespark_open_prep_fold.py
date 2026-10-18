# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``full_attention_prep_fold`` on the split-KV full-attention layers: its refusals, the classes
and the routing.

Host-only, on ``test_modeling_v2_musespark_open_split_kv.py``'s stand-in model and recorders
(loaded from beside this file): the built-in attention op calls the generation preprocess and
the decode FMHA as ``run_generation`` does, and ``full_attention_split_kv`` / ``combine_norm_gate`` are
recorders. Here ``full_attention_prep_fold``, the folded prep and the cache metadata builder are recorders
too, and the built-in forward passes ``_attn_impl`` what ``Attention.forward_impl`` passes it.
What the kernel computes is the GPU tests' business
(``attention/test_modeling_v2_full_attention_prep_fold.py`` and the compose tests' ``full_attention_prep_fold`` table).
"""

from __future__ import annotations

import importlib.util
import os
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_fold import (  # noqa: E501
    attention as prep_fold_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_split_kv import (  # noqa: E501
    attention as split_kv_attention,
)
from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.interface import MLAParams, PredefinedAttentionMask
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention

pytestmark = pytest.mark.cpu_only


def _split_kv_features():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        here, "..", "full_attention_split_kv", "test_modeling_v2_musespark_open_split_kv.py"
    )
    spec = importlib.util.spec_from_file_location("_split_kv_tests_for_prep_fold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPLIT_KV = _split_kv_features()
PREP_APPEND = SPLIT_KV.PREP_APPEND
ROWS = SPLIT_KV.ROWS
RAW = 2304
_entry = SPLIT_KV._entry
# The split-KV test's fixtures, by name.
adoptable = SPLIT_KV.adoptable
recorded = SPLIT_KV.recorded


def _model(num_layers: int = 4):
    """The folded prep and the split-KV attention installed on the stand-in model, with what the
    fold's route reads."""
    model = SPLIT_KV._model(num_layers)
    for layer in model.layers:
        attn = layer.self_attn.attn
        attn.quant_mode = 0
        attn.get_local_layer_idx = lambda md, _i=layer.self_attn.layer_idx: _i
        # A TrtllmAttention's (``MLAParams``' default; no chunked attention).
        attn.predicted_tokens_per_seq = MLAParams().predicted_tokens_per_seq
        attn.attention_chunk_size = None
    split_kv_attention.install(model)
    return model


def _folded(num_layers: int = 4):
    model = _model(num_layers)
    prep_fold_attention.install(model)
    return model


def _metadata(seq_lens=None):
    return types.SimpleNamespace(
        host_kv_cache_pool_pointers="pointers",
        host_kv_cache_pool_mapping="mapping",
        kv_cache_block_offsets="offsets",
        tokens_per_block=32,
        kv_lens_cuda_runtime=torch.tensor([4096], dtype=torch.int32)
        if seq_lens is None
        else seq_lens,
    )


def _forward_impl_chain(self, position_ids, hidden_states, attn_metadata, **kwargs):
    """``Attention.forward`` from the projection on: ``preprocess_qkv``, ``convert_qkv`` (the
    fused layout passes q through), ``forward_impl``'s ``_attn_impl`` call, the output gate."""
    q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
    output, _ = self._attn_impl(
        q,
        k,
        v,
        attn_metadata,
        PredefinedAttentionMask.CAUSAL,
        None,
        None,
        None,
        None,
        attention_sinks=None,
        relative_attention_bias=None,
        relative_attention_max_distance=0,
        has_lora=False,
    )
    return self.apply_output_gate(output, gate)


@pytest.fixture
def routed(adoptable, recorded, monkeypatch):
    """The split-KV test's recorders, plus: the forward_impl chain, a recording folded prep, the cache
    metadata builder (a plain FP8 pool view and one sequence's page table) and a recording
    ``full_attention_prep_fold``. ``state["pool"]`` / ``state["tables"]`` set what the builder
    returns."""
    taken = recorded.taken
    state = recorded.state
    state["pool"] = torch.zeros(10, 2, 32, 64, dtype=torch.float8_e4m3fn)
    state["tables"] = torch.zeros(1, 2, 64, dtype=torch.int32)
    real_prep = full_attention.FullAppendAttention._prep_append_prep

    def prep(self, qkv, md):
        taken.append(("prep_append_prep", self.layer_idx))
        return real_prep(self, qkv, md)

    def build(*args):
        taken.append(("kv_cache_metadata", args[3]))
        return state["pool"], state["tables"], None

    def fold(qkv, eps, q_scale, permuted, kv_pool, tables, seq_lens, kv_scale, splits, pdl,
             evict_first=True, rescale_cols=64):  # fmt: skip
        state["fold_qkv"] = qkv
        state.setdefault("fold_rescale_cols", []).append(rescale_cols)
        taken.append(
            (
                "full_attention_prep_fold",
                eps,
                q_scale,
                permuted,
                tuple(tables.shape),
                splits,
                pdl,
                evict_first,
            )  # fmt: skip
        )
        return "part_o", "part_lse"

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", _forward_impl_chain)
    monkeypatch.setattr(full_attention.FullAppendAttention, "_prep_append_prep", prep)
    monkeypatch.setattr(
        full_attention.FullAppendAttention,
        "_prep_append_fmha_lib",
        lambda self: types.SimpleNamespace(_get_total_num_blocks=lambda md: 10),
    )
    monkeypatch.setattr(
        prep_fold_attention, "thop", types.SimpleNamespace(build_trtllm_gen_kv_cache_metadata=build)
    )
    monkeypatch.setattr(prep_fold_attention._fold, "full_attention_prep_fold", fold)
    monkeypatch.setattr(prep_fold_attention, "_capturing", lambda: False)
    # The stand-in projection is a CPU tensor (device index None); a CUDA tensor's is an int.
    real_key = prep_fold_attention._fold.compiled_key
    monkeypatch.setattr(
        prep_fold_attention._fold,
        "compiled_key",
        lambda device, *rest: real_key(0 if device is None else device, *rest),
    )
    monkeypatch.setattr(full_attention, "_capturing", lambda: False)
    # The stand-in has no o_proj; ``state["quantize_output"]`` sets what ``_attn_impl`` would.
    monkeypatch.setattr(
        MuseSparkOpenAttention,
        "_use_quantize_output",
        lambda self: state.get("quantize_output", False),
    )
    return recorded


def _qkv(rows: int = ROWS) -> torch.Tensor:
    return torch.randn(rows, RAW).to(torch.bfloat16)


def _step(attention, qkv, md=None):
    return attention(
        position_ids=None,
        hidden_states=qkv,
        attn_metadata=_metadata() if md is None else md,
        preattn_prep=_entry,
    )


# --------------------------------------------------------------------------
# The classes
# --------------------------------------------------------------------------
def test_install_puts_the_mixin_in_front_of_split_kv_on_the_full_layers_only(adoptable):
    model = _model(8)
    before = [type(layer.self_attn) for layer in model.layers]
    assert prep_fold_attention.install(model) == 2
    for index, layer in enumerate(model.layers):
        cls = type(layer.self_attn)
        if PREP_APPEND._full(index, len(model.layers)):
            assert cls.__mro__[1:3] == (
                prep_fold_attention.FullAttnSplitKvFoldAttention,
                before[index],
            )
            assert issubclass(before[index], split_kv_attention.FullAttnSplitKvAttention)
            assert layer.self_attn._prep_fold_call is None
            assert layer.self_attn._prep_fold_census.layers == (3, 7)
        else:
            assert cls is before[index]


def test_install_refuses_without_split_kv_or_a_gate_and_changes_nothing(adoptable):
    model = SPLIT_KV._model()  # full_attention_prep_append only
    before = [type(layer.self_attn) for layer in model.layers]
    with pytest.raises(ValueError, match="full_attention_split_kv is not installed"):
        prep_fold_attention.install(model)
    assert [type(layer.self_attn) for layer in model.layers] == before
    model = _model()
    before = [type(layer.self_attn) for layer in model.layers]
    model.layers[3].self_attn.attn_output_gate = False
    with pytest.raises(ValueError, match="layer 3: it has no output gate"):
        prep_fold_attention.install(model)
    model.layers[3].self_attn.attn_output_gate = True
    del model.layers[3].self_attn.scale_query_by
    with pytest.raises(ValueError, match="no norm epsilon, query pre-scale or projection layout"):
        prep_fold_attention.install(model)
    assert [type(layer.self_attn) for layer in model.layers] == before


def test_the_core_installs_prep_fold_after_split_kv_only_when_selected(adoptable, monkeypatch):
    seen = []
    real_v4_install = split_kv_attention.install
    real_install = prep_fold_attention.install

    def v4_install(model):
        for layer in model.layers:
            layer.self_attn.attn_output_gate = True
            layer.self_attn.attn = types.SimpleNamespace(head_dim=64, q_scaling=1.0)
        return real_v4_install(model)

    def install(model):
        seen.append([type(layer.self_attn).__mro__[1] for layer in model.layers])
        return real_install(model)

    monkeypatch.setattr(split_kv_attention, "install", v4_install)
    monkeypatch.setattr(prep_fold_attention, "install", install)
    model = PREP_APPEND._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("full_attention_prep_fold"))
    # The split-KV mixin is on the full layer when the fold installs, and the fold's goes in front.
    assert seen[0][3] is split_kv_attention.FullAttnSplitKvAttention
    cls = type(model.layers[3].self_attn)
    assert cls.__mro__[1] is prep_fold_attention.FullAttnSplitKvFoldAttention
    model = PREP_APPEND._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("full_attention_split_kv"))
    assert len(seen) == 1
    assert not isinstance(
        model.layers[3].self_attn, prep_fold_attention.FullAttnSplitKvFoldAttention
    )


# --------------------------------------------------------------------------
# The routing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("permuted", (False, True), ids=("ckpt", "permuted"))
def test_a_fold_routed_step_runs_the_fold_kernel_and_the_merge_only(routed, permuted):
    """No prep, no built-in preprocess or FMHA, no full_attention_split_kv: the fold kernel on the raw
    projection, then combine_norm_gate on its partials with the gate read from the projection."""
    model = _folded()
    full = model.layers[3].self_attn
    if permuted:
        full._qkv_layout_inverse = torch.arange(RAW)
    qkv = _qkv()
    out = _step(full, qkv)
    assert out == "merged-gated"
    assert routed.taken[:2] == [
        ("kv_cache_metadata", 3),
        ("full_attention_prep_fold", 1e-5, 1.0, permuted, (2, 64), 152, True, True),
    ]
    assert len(routed.taken) == 3
    name, part_o, rows, gate, trigger, masked = routed.taken[2]
    assert (name, part_o, rows, trigger, masked) == (
        "combine_norm_gate",
        "part_o",
        ROWS,
        "exit",
        True,
    )
    assert routed.state["fold_qkv"] is qkv
    # The gate is a view of the projection: checkpoint order [q_h | gate_h] per head, or the
    # permuted layout's last 1024 columns.
    assert gate.data_ptr() == qkv.data_ptr() + (1280 if permuted else 64) * qkv.element_size()
    want = qkv[:, 1280:] if permuted else qkv[:, :2048].view(ROWS, 16, 2, 64)[:, :, 1, :]
    assert torch.equal(gate.reshape(ROWS, -1), want.reshape(ROWS, -1))
    assert full._prep_fold_call is None and full._split_kv_pending is None
    assert full._prep_fold_census.outcomes[3] == (prep_fold_attention.FOLD, "")
    assert full._split_kv_census.outcomes[3][0] == split_kv_attention.V4
    assert full._prep_append_census.outcomes[3][0] == full_attention.FOLDED


def _builtin_route(rows: int = ROWS):
    """What the folded prep + split-KV attention record on a step they serve (the split-KV
    recorders)."""
    return [
        ("prep_append_prep", 3),
        ("full_attention_split_kv", (2, 64), 152, True, False, True, 64),
        ("combine_norm_gate", "part_o", rows, "gate", "exit", True),
    ]


@pytest.mark.parametrize(
    "change, reason",
    (
        ("tables", "more pages per split than the kernel stages"),
        ("pool", "another cache view"),
        ("two_sequences", "not one sequence's K/V page table"),
        ("scale", "another softmax scale"),
        ("width", "not a raw projection the fold reads"),
    ),
)
def test_a_step_the_fold_does_not_serve_runs_prep_append_and_split_kv(routed, change, reason):
    model = _folded()
    full = model.layers[3].self_attn
    qkv = _qkv()
    if change == "tables":
        routed.state["tables"] = torch.zeros(1, 2, 152 * 256 + 1, dtype=torch.int32)
    elif change == "pool":
        routed.state["pool"] = torch.zeros(10, 2, 32, 64, dtype=torch.bfloat16)
    elif change == "two_sequences":
        routed.state["tables"] = torch.zeros(2, 2, 64, dtype=torch.int32)
    elif change == "scale":
        full.attn.q_scaling = 2.0
    elif change == "width":
        qkv = torch.randn(ROWS, RAW + 64).to(torch.bfloat16)[:, 4 : 4 + RAW]  # 8-byte aligned
        assert qkv.data_ptr() % 16 == 8
    out = _step(full, qkv)
    assert out == "merged-gated"
    # The folded prep and full_attention_split_kv (its recorder serves whatever the prep's
    # stand-in hands it), then the norm-gate.
    assert [e for e in routed.taken if e[0] != "kv_cache_metadata"] == _builtin_route()
    assert full._prep_fold_census.outcomes[3] == ("declined", reason)
    assert full._prep_fold_call is None


def test_with_spill_free_the_fold_launches_the_spill_free_build(routed):
    """``install_spill_free`` sets the split-KV layers to the kernel's spill-free build; a
    fold-routed step launches the fold kernel with it (and without it with the whole-head
    rescale)."""
    model = _folded()
    _step(model.layers[3].self_attn, _qkv())
    split_kv_attention.install_spill_free(model)
    _step(model.layers[3].self_attn, _qkv())
    assert routed.state["fold_rescale_cols"] == [64, 16]
    assert prep_fold_attention._fold.compiled_key(0, True, True, 1e-5, 1.0, False, 16) == (
        prep_fold_attention._fold.compiled_key(0, True, True, 1e-5, 1.0, False)
        + (("rescale_cols", 16),)
    )


def test_a_build_not_compiled_before_the_capture_is_not_launched_inside_it(routed, monkeypatch):
    model = _folded()
    full = model.layers[3].self_attn
    monkeypatch.setattr(prep_fold_attention, "_capturing", lambda: True)
    monkeypatch.setattr(prep_fold_attention._v4, "_compiled_keys", set())
    _step(full, _qkv())
    assert full._prep_fold_census.outcomes[3] == (
        "declined",
        "full_attention_prep_fold was not compiled before this capture",
    )
    assert [e for e in routed.taken if e[0] != "kv_cache_metadata"] == _builtin_route()
    key = prep_fold_attention._fold.compiled_key(0, True, True, 1e-5, 1.0, False)
    monkeypatch.setattr(prep_fold_attention._v4, "_compiled_keys", {key})
    routed.taken.clear()
    _step(full, _qkv())
    assert full._prep_fold_census.outcomes[3] == (prep_fold_attention.FOLD, "")


def test_a_prefill_or_a_step_prep_append_declines_runs_the_builtin_path(routed, monkeypatch):
    model = _folded()
    full = model.layers[3].self_attn
    # A prefill (no preattn_prep): no fold, no prep, the built-in FMHA and gate.
    full(position_ids=None, hidden_states=_qkv(), attn_metadata=_metadata())
    assert routed.taken == [("builtin_fmha", -1), ("builtin_gate", 3)]
    # A step the prep append declines: its decline path decides, the fold is not consulted.
    routed.taken.clear()
    monkeypatch.setattr(
        full_attention.FullAppendAttention, "_prep_append_decline", lambda *a: "9 rows"
    )
    _step(full, _qkv())
    assert routed.taken == [("builtin_fmha", -1), ("builtin_gate", 3)]
    assert full._prep_fold_census.outcomes[3][0] == prep_fold_attention.NO_DECODE_PREP


@pytest.mark.parametrize(
    "change, reason",
    (
        (dict(attention_window_size=2047), "an attention window"),
        (dict(output=torch.zeros(ROWS, 1024)), "an output buffer"),
        (dict(attention_sinks=torch.zeros(16)), "attention sinks"),
        (dict(has_lora=True), "LoRA"),
        (dict(k=torch.zeros(1)), "separate K / V"),
        (dict(attention_mask=PredefinedAttentionMask.FULL), "not the causal mask"),
    ),
)
def test_an_attention_call_that_is_not_the_plain_one_raises(routed, monkeypatch, change, reason):
    """The prep was skipped: the fold-routed attention call cannot fall back."""
    model = _folded()
    full = model.layers[3].self_attn

    def forward(self, position_ids, hidden_states, attn_metadata, **kwargs):
        q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
        args = dict(
            q=q, k=k, v=v, attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL, mrope_rotary_cos_sin=None,
            mrope_position_deltas=None, attention_window_size=None, attention_mask_data=None,
        )  # fmt: skip
        args.update(change)
        output, _ = self._attn_impl(**args)
        return self.apply_output_gate(output, gate)

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", forward)
    with pytest.raises(RuntimeError, match=f"not the plain one the fold serves \\({reason}"):
        _step(full, _qkv())
    assert ("builtin_fmha", -1) not in routed.taken
    assert full._prep_fold_call is None


def test_a_fold_routed_step_whose_attention_never_ran_raises(routed, monkeypatch):
    model = _folded()
    full = model.layers[3].self_attn

    def no_attention(self, position_ids, hidden_states, attn_metadata, **kwargs):
        self.preprocess_qkv(hidden_states, position_ids)
        return "nothing"

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", no_attention)
    with pytest.raises(RuntimeError, match="the attention never launched it"):
        _step(full, _qkv())
    assert full._prep_fold_call is None


def test_a_second_attention_call_on_one_step_raises(routed, monkeypatch):
    model = _folded()
    full = model.layers[3].self_attn

    def twice(self, position_ids, hidden_states, attn_metadata, **kwargs):
        q, k, v, gate = self.preprocess_qkv(hidden_states, position_ids)
        mask = PredefinedAttentionMask.CAUSAL
        self._attn_impl(q, k, v, attn_metadata, mask, None, None, None, None)
        self._attn_impl(q, k, v, attn_metadata, mask, None, None, None, None)

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", twice)
    with pytest.raises(RuntimeError, match="ran twice on one fold-routed step"):
        _step(full, _qkv())


def test_the_census_reports_each_captured_forward(routed, monkeypatch):
    model = _folded(8)
    census = model.layers[3].self_attn._prep_fold_census
    assert census.layers == (3, 7)
    for module in (prep_fold_attention, split_kv_attention, full_attention):
        monkeypatch.setattr(module, "_capturing", lambda: True)
    key = prep_fold_attention._fold.compiled_key(0, True, True, 1e-5, 1.0, False)
    monkeypatch.setattr(prep_fold_attention._v4, "_compiled_keys", {key})
    for index in (3, 7):
        _step(model.layers[index].self_attn, _qkv())
    assert census.reports[-1][1:4] == (2, 2, "info")
    assert (
        "2 of 2 full-attention layers run the prep inside full_attention_prep_fold"
        in (census.reports[-1][4])
    )
    # Not compiled: both layers run the folded prep + split-KV attention (whose own build is not compiled either, so the
    # built-in FMHA), and the line warns.
    monkeypatch.setattr(prep_fold_attention._v4, "_compiled_keys", set())
    for index in (3, 7):
        _step(model.layers[index].self_attn, _qkv())
    assert census.reports[-1][1:4] == (0, 2, "warning")
    assert "not compiled before this capture" in census.reports[-1][4]
    # A prefill capture: nothing to report.
    for index in (3, 7):
        model.layers[index].self_attn(
            position_ids=None, hidden_states=_qkv(), attn_metadata=_metadata()
        )
    assert census.reports[-1][3] == "silent"


# --------------------------------------------------------------------------
# Steps the built-in chain runs in another mode than the fold's
# --------------------------------------------------------------------------
_SPEC_FIELDS = (
    "is_spec_decoding_enabled",
    "use_spec_decoding",
    "is_spec_dec_tree",
    "is_spec_dec_dynamic_tree",
    "spec_decoding_generation_lengths",
    "spec_decoding_bl_tree_mask",
)


def _served_linear_chain_metadata(monkeypatch):
    """The step metadata of the served target verify step: DFlash runs in one engine, so it
    asks ``update_spec_dec_param`` for spec-dec mode with a linear chain of K = 7 drafts, which
    sm_103's trtllm-gen kernels leave off (they take spec-dec mode for a dynamic tree only)."""
    monkeypatch.setattr(trtllm_backend, "get_sm_version", lambda: 103)
    real = trtllm_backend.TrtllmAttentionMetadata.__new__(trtllm_backend.TrtllmAttentionMetadata)
    real.update_spec_dec_param(
        batch_size=1,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        max_draft_len=7,
        max_total_draft_tokens=7,
    )
    md = _metadata()
    for name in _SPEC_FIELDS:
        setattr(md, name, getattr(real, name))
    return md


def test_the_served_linear_chain_step_is_fold_routed(routed, monkeypatch):
    md = _served_linear_chain_metadata(monkeypatch)
    assert (md.is_spec_decoding_enabled, md.use_spec_decoding, md.is_spec_dec_tree) == (
        False,
        False,
        False,
    )
    assert md.spec_decoding_generation_lengths is None and md.spec_decoding_bl_tree_mask is None
    model = _folded()
    full = model.layers[3].self_attn
    assert full.attn.predicted_tokens_per_seq == 1
    assert _step(full, _qkv(), md) == "merged-gated"
    assert full._prep_fold_census.outcomes[3] == (prep_fold_attention.FOLD, "")
    assert ("prep_append_prep", 3) not in routed.taken


@pytest.mark.parametrize(
    "change",
    (
        # A linear chain that turns use_spec_decoding on (as Eagle3 does) without spec-dec
        # mode: the built-in chain runs it as causal attention over the rows.
        dict(use_spec_decoding=True),
        # Multi-token generation needs spec-dec mode: without it the preprocess is handed no
        # generation lengths.
        dict(predicted_tokens_per_seq=8, spec_decoding_generation_lengths=torch.tensor([8])),
        dict(attention_chunk_size=0),
    ),
    ids=("use_spec_decoding", "predicted_without_spec_dec", "chunk_0"),
)
def test_a_step_the_builtin_chain_runs_as_plain_decode_is_fold_routed(routed, change):
    model = _folded()
    full = model.layers[3].self_attn
    md = _metadata()
    for name, value in change.items():
        target = full.attn if name in ("predicted_tokens_per_seq", "attention_chunk_size") else md
        setattr(target, name, value)
    assert _step(full, _qkv(), md) == "merged-gated"
    assert full._prep_fold_census.outcomes[3] == (prep_fold_attention.FOLD, "")
    assert ("prep_append_prep", 3) not in routed.taken


@pytest.mark.parametrize(
    "change, reason",
    (
        (
            dict(
                is_spec_decoding_enabled=True,
                predicted_tokens_per_seq=8,
                spec_decoding_generation_lengths=torch.tensor([8]),
            ),
            "multi-token generation",
        ),
        # A dynamic tree on sm_103: spec-dec mode on, a tree mask, one predicted token (MLA's).
        (
            dict(is_spec_decoding_enabled=True, use_spec_decoding=True, is_spec_dec_tree=True),
            "spec-dec tree masks",
        ),
        (dict(is_spec_decoding_enabled=True), "spec-dec tree masks"),
        (dict(is_spec_dec_tree=True), "spec-dec tree masks"),
        (dict(spec_decoding_bl_tree_mask=torch.zeros(4, dtype=torch.int32)), "spec-dec tree masks"),
        (dict(quantize_output=True), "a quantized output"),
        (dict(attention_chunk_size=8192), "chunked attention"),
    ),
    ids=(
        "multi_token",
        "dynamic_tree",
        "spec_dec_mode",
        "tree",
        "bl_tree_mask",
        "quantized_output",
        "chunked",
    ),
)
def test_a_step_the_builtin_chain_runs_in_another_mode_runs_prep_append_and_split_kv(
    routed, change, reason
):
    """The fold computes causal attention over the step's rows into BF16 partials: a step the
    built-in chain would run with a spec-decoding mask, a quantized output or chunked attention
    goes back to the folded prep and the built-in chain (the split-KV attention behind it),
    exactly as without the fold."""
    model = _folded()
    full = model.layers[3].self_attn
    md = _metadata()
    for name, value in change.items():
        if name in ("predicted_tokens_per_seq", "attention_chunk_size"):
            setattr(full.attn, name, value)
        elif name == "quantize_output":
            routed.state[name] = value
        else:
            setattr(md, name, value)
    assert _step(full, _qkv(), md) == "merged-gated"
    assert [e for e in routed.taken if e[0] != "kv_cache_metadata"] == _builtin_route()
    assert full._prep_fold_census.outcomes[3] == ("declined", reason)
    assert full._prep_fold_call is None
