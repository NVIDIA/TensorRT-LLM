# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host test for the SWA block (``swa_block``, ``kernels/swa_block/attention.py``): what it builds
on, the attention class it composes over ``preattn_prep`` + ``swa_paged_decode`` (and over a
norm-gate class such as ``boundary_early_release``'s), and the per-call scoping of the mixin's
state. No kernel runs here; the op's own GPU test and the combined real-weight test
(``MODELING_V2_COMPOSE_TIER=swa_block``) cover the numbers."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.swa_block import (  # noqa: E501
    attention as swa_attention_block,
)


def test_the_block_builds_on_the_prep_and_the_paged_decode():
    assert dispatch.closure({"swa_block"}) == {"swa_block", "preattn_prep", "swa_paged_decode"}
    for missing in ("preattn_prep", "swa_paged_decode"):
        with pytest.raises(ValueError, match="builds on"):
            dispatch.Plan(dispatch.closure({"swa_block"}) - {missing})


def test_the_class_needs_preattn_prep_and_paged_decode():
    for base in (
        None,
        target_attention.MuseSparkOpenTepAttention,
        target_attention.MuseSparkOpenTepDecodeAttention,
    ):
        with pytest.raises(ValueError, match="needs both"):
            swa_attention_block.attention_class(base)


def test_the_class_puts_the_swa_block_methods_first():
    base = target_attention.MuseSparkOpenTepPrepDecodeAttention
    cls = swa_attention_block.attention_class(base)
    assert cls is swa_attention_block.attention_class(base), "one class per base"
    assert cls.__mro__[1] is swa_attention_block._SwaBlock and cls.__mro__[2] is base
    for name in ("forward", "preprocess_qkv", "_attn_impl", "apply_output_gate"):
        assert getattr(cls, name) is getattr(swa_attention_block._SwaBlock, name), name


def test_the_class_composes_over_a_norm_gate_class():
    """Over a class that overrides ``apply_output_gate`` (as ``boundary_early_release``'s does),
    the swa_block gate skip comes first, and every other call reaches the base's override."""

    class NormGate(target_attention.MuseSparkOpenTepPrepDecodeAttention):
        def apply_output_gate(self, attention_output, gate):
            return ("base", attention_output, gate)

    cls = swa_attention_block.attention_class(NormGate)
    obj = cls.__new__(cls)
    obj._swa_gate = None
    obj._swa_gated = True
    assert obj.apply_output_gate("out", "gate") == "out"
    assert obj._swa_gated is False
    assert obj.apply_output_gate("out", "gate") == ("base", "out", "gate")


def test_the_class_composes_over_the_early_release_norm_gate_class():
    """With ``boundary_early_release`` planned, the attention class picked is its PDL norm-gate
    class, and swa_block wraps it: the swa_block methods first, then the norm-gate class's
    ``forward`` / ``apply_output_gate``, then the prep + paged decode class's."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
        triggers as pdl_triggers,
    )

    plan = target_attention.DecodeAttentionPlan(sliding_window=True, q_fp8=True)
    base = target_attention._attention_class(True, plan, True)
    assert base is pdl_triggers.MuseSparkOpenTepPrepDecodeNormGateAttention
    cls = swa_attention_block.attention_class(base)
    assert cls.__name__ == "MuseSparkOpenTepPrepDecodeNormGateAttentionSwaBlock"
    mro = cls.__mro__
    assert (
        mro.index(swa_attention_block._SwaBlock)
        < mro.index(pdl_triggers._NormGatePdl)
        < mro.index(target_attention.MuseSparkOpenTepPrepDecodeAttention)
    )
    obj = cls.__new__(cls)
    obj._swa_gate = None
    obj._swa_gated = True
    assert obj.apply_output_gate("out", "gate") == "out", "a gated call skips the PDL gate"
    assert obj._swa_gated is False


def test_the_adopt_check_is_the_bases():
    cls = swa_attention_block.attention_class(target_attention.MuseSparkOpenTepPrepDecodeAttention)
    with pytest.raises(TypeError):
        cls.check(object())


# --------------------------------------------------------------------------
# Launch order of a sliding-window decode step: prep's entry trigger only in front of
# swa_block's split, never in front of the paged decode / built-in attention of a declined step.
# --------------------------------------------------------------------------
_H, _G, _D = 16, 2, 64


class _Recorded(target_attention.MuseSparkOpenTepPrepDecodeAttention):
    """The prep + paged decode class, built-in attention and output gate replaced by recorders."""

    def _attn_impl(self, q, k, v, *args, **kwargs):
        self._calls.append(("builtin_attention", q))
        return q, None

    def apply_output_gate(self, attention_output, gate):
        self._calls.append(("norm_gate", gate))
        return attention_output


def _layer(plan):
    """A swa_block layer object with stand-ins for everything the decode path reads on the host."""
    import types

    import torch

    cls = swa_attention_block.attention_class(_Recorded)
    obj = cls.__new__(cls)
    calls = []
    fused = torch.zeros(8, (_H + 2 * _G) * _D, dtype=torch.bfloat16)
    prep_gate = torch.zeros(8, _H * _D, dtype=torch.bfloat16)

    def prep(qkv, cos_sin, positions, heads, kv_heads, eps, q_scale, permuted, pdl, **kwargs):
        calls.append(("preattn_prep", kwargs.get("trigger_at_entry", False), positions))
        return fused[: qkv.shape[0]], prep_gate[: qkv.shape[0]]

    def swa_block(q, md, window, q_fp8, raw, gate):
        calls.append(("swa_block", q, raw, gate))
        return q

    obj.__dict__.update(
        _calls=calls,
        _swa_block=plan,
        _swa_raw=False,
        _swa_gate=None,
        _swa_gated=False,
        _swa_position_ids=None,
        _preattn_prep=prep,
        _decode_attention=target_attention.DecodeAttentionPlan(True, True),
        _fits_preattn_prep=lambda qkv, position_ids: True,
        _qkv_layout_inverse=None,
        _swa_block_call=swa_block,
        _use_quantize_output=lambda: False,
        _total_num_blocks_fmha=lambda: object(),
        num_heads=_H,
        num_key_value_heads=_G,
        head_dim=_D,
        skip_rope=False,
        rotary_emb=types.SimpleNamespace(rotary_cos_sin="cos_sin"),
        q_norm=types.SimpleNamespace(variance_epsilon=1e-6),
        o_head_norm=types.SimpleNamespace(variance_epsilon=1e-5),
        scale_query_by=0.125,
        attn=types.SimpleNamespace(has_fp8_kv_cache=True),
        mapping=types.SimpleNamespace(has_cp_helix=lambda: False),
    )
    return obj, calls, fused, prep_gate


def _step(obj, rows, sequences):
    """``Attention.forward``'s calls between the QKV projection and o_proj, for one decode step."""
    import types

    import torch

    from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask

    qkv = torch.zeros(rows, 2 * (_H + _G) * _D, dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int32).reshape(1, rows)
    md = types.SimpleNamespace(
        num_tokens=rows,
        num_seqs=sequences,
        num_generations=sequences,
        kv_cache_manager=object(),
        kv_cache_block_offsets=object(),
        tokens_per_block=32,
        beam_width=1,
        is_cross=False,
    )
    q, k, v, gate = obj.preprocess_qkv(qkv, positions)
    out, _ = obj._attn_impl(q, k, v, md, PredefinedAttentionMask.CAUSAL, None, None, 2048, None)
    obj.apply_output_gate(out, gate)
    return positions


@pytest.mark.parametrize("rows", (1, 8))
def test_swa_prewait_prep_triggers_at_entry_only_right_before_the_split(rows):
    """One sequence: with pre-wait loads only, ``preattn_prep`` launches with its entry trigger,
    and the next launch is swa_block's split on prep's output; the norm-gate then reads prep's
    gate copy."""
    obj, calls, fused, prep_gate = _layer(swa_attention_block.SwaBlockPlan(False, False))
    positions = _step(obj, rows, 1)
    assert [c[0] for c in calls] == ["preattn_prep", "swa_block", "norm_gate"], calls
    assert calls[0][1] is True and calls[0][2] is positions
    assert calls[1][1].data_ptr() == fused.data_ptr() and calls[1][2] is False
    assert calls[1][3] is None
    assert calls[2][1].data_ptr() == prep_gate.data_ptr()


@pytest.mark.parametrize("rows", (2, 8))
@pytest.mark.parametrize(
    "plan",
    (swa_attention_block.SwaBlockPlan(False, False), swa_attention_block.SwaBlockPlan(True, True)),
    ids=("swa_prewait", "fused_prep_and_gate"),
)
def test_a_declined_step_launches_prep_with_its_exit_trigger(plan, rows):
    """Two sequences with at most 8 rows in all (``num_seqs != 1``: the step does not fit
    swa_block): prep runs as the base class launches it -- no entry trigger -- and the next
    launch is the paged decode / built-in attention on prep's output; the norm-gate reads
    prep's gate copy."""
    obj, calls, fused, prep_gate = _layer(plan)
    _step(obj, rows, 2)
    assert [c[0] for c in calls] == ["preattn_prep", "builtin_attention", "norm_gate"], calls
    assert calls[0][1] is False
    assert calls[1][1].data_ptr() == fused.data_ptr()
    assert calls[2][1].data_ptr() == prep_gate.data_ptr()


def test_swa_fused_prep_launches_no_prep_on_a_fitting_step():
    """With the fused prep the split runs the prep; with the fused gate the combine applies the
    gate (a view of the projection) and the norm-gate does not run."""
    obj, calls, _fused, _prep_gate = _layer(swa_attention_block.SwaBlockPlan(True, False))
    _step(obj, 8, 1)
    assert [c[0] for c in calls] == ["swa_block", "norm_gate"], calls
    assert calls[0][2] is True
    obj, calls, _fused, _prep_gate = _layer(swa_attention_block.SwaBlockPlan(True, True))
    obj._fits_fused_gate = lambda q: True
    _step(obj, 8, 1)
    assert [c[0] for c in calls] == ["swa_block"], calls
    assert calls[0][2] is True and calls[0][3] is not None
