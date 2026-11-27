# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 DEP target's runtime hooks: the built hooks equal the expected ``RuntimeHooks`` (every other
field at its default), and without the family step-hook ops no native launch is armed. CPU only (the ops' presence is
stubbed; test_m31_step_hooks_gpu.py runs them)."""

from dataclasses import fields

import pytest

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa import engine as msa_engine
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import step_hooks
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy, RuntimeHooks

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
        modeling,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_STEP_ORDER = dict(
    loop_on_execution_stream=True,
    sampler_d2h_side_stream=True,
    iteration_timing_one_event=True,
    defer_gen_reuse_commit=True,
)
_STEP_INPUTS = dict(
    pack_step_inputs=True,
    pack_parallel_scatter=True,
    pack_in_graph=True,
    pack_early_copy=True,
    skip_unchanged_h2d=True,
    h2d_changed_span=True,
    kv_lens_update_once=True,
    pack_spec_rng_uploads=True,
    spec_small_h2d=True,
    skip_unchanged_draft_block_offsets=True,
    patch_unchanged_draft_block_offsets=True,
    pack_cpp_scatter=True,
)
# The DSpark step fold the hooks build (one per call): a fixed stand-in so the hooks compare.
_FOLD = object()
# The native launches without the input path (DEP_STEP_NATIVE) ...
_NATIVE = dict(spec_offsets_op=step_hooks.spec_offsets, spec_step_fold=_FOLD)
# ... and the in-graph prologue, which needs the pack's early copy.
_PROLOGUE = dict(pack_step_epoch=True, pack_prologue_op=step_hooks.step_prologue)
_NO_PROLOGUE = dict(pack_step_epoch=False, pack_prologue_op=None)
_MSA_FIELDS = dict(
    skip_unchanged_block_offsets=True,
    reuse_pool_views=True,
)
_MSA = dict(**_MSA_FIELDS, kv_update=step_hooks.msa_kv_update)
_MSA_KV_ONLY = dict(kv_update=step_hooks.msa_kv_update)
_KERNELS = dict(mxfp8_pinned_tactics=True, spec_tail_bookkeep=True)
_FIXES = dict(
    draft_pool_fixes=True,
    draft_kv_reserve_guard=True,
)
_BASE = dict(
    late_compile_detect=True,
    merge_tuned_tactics_across_ranks=True,
    persist_flashinfer_mxfp8_tactics=True,
    install_tactic_pins=modeling._install_tactic_bundle,
    disagg_draft_kv=True,
    ctx_batch_wait_skip=True,
    postlaunch_admission=True,
    gpu_timestamps=True,
    sparse_attention=msa_engine.sparse_attention_classes(),
    context_onehot_index_fill=True,
    dflash_host_ctx_alloc=True,
    kv_fabric_write=True,
    bounce_scatter_on_order_stream=True,
    disagg_gen_completion_reservation=True,
    executor_thread_pin_cores=2,
    dspark_ancestral_override=True,
    ctx_srpt_aging=True,
    ctx_oop_retry_gate=True,
    response_guard=True,
    kv_cache_gil_fast_path=True,
    async_stats_rpc=True,
    moe_flashinfer_fp4_quantize=True,
)
_INPUTS_OFF = dict(**_BASE, **_STEP_ORDER, **_KERNELS, **_FIXES, **_NATIVE, msa=_MSA_KV_ONLY)
_DEFAULT = dict(**_INPUTS_OFF, **_STEP_INPUTS, **_PROLOGUE)
_DEFAULT["msa"] = _MSA


def _on(value) -> bool:
    return value is not None and value is not False


def _view(hooks: dict) -> dict:
    """The hooks a field mapping turns on (``msa`` as the MSA policy fields it turns on)."""
    out = {name: value for name, value in hooks.items() if name != "msa" and _on(value)}
    out["msa"] = {name: value for name, value in hooks.get("msa", {}).items() if _on(value)}
    return out


def _fields_of(hooks: RuntimeHooks) -> dict:
    """The fields ``hooks`` sets away from their defaults, ``msa`` as a mapping of its own."""
    default, default_msa = RuntimeHooks(), MsaRuntimePolicy()
    out = {
        f.name: getattr(hooks, f.name)
        for f in fields(hooks)
        if f.name != "msa" and getattr(hooks, f.name) != getattr(default, f.name)
    }
    out["msa"] = {
        f.name: getattr(hooks.msa, f.name)
        for f in fields(hooks.msa)
        if getattr(hooks.msa, f.name) != getattr(default_msa, f.name)
    }
    return out


def _arm(monkeypatch, disabled=(), ops=True):
    monkeypatch.setattr(m31_switches, "_DISABLED", frozenset(disabled))
    monkeypatch.setattr(step_hooks, "ops_available", lambda: ops)
    monkeypatch.setattr(step_hooks, "M3DSparkStepFold", lambda: _FOLD)
    return modeling.dep_runtime_hooks()


def _hooks(monkeypatch, disabled=(), ops=True):
    """What the target arms under the switches (None: the built-in model's hooks)."""
    hooks = _arm(monkeypatch, disabled, ops)
    if hooks is None:
        return None
    return _view(_fields_of(hooks))


def test_the_built_hooks_are_the_expected_runtime_hooks(monkeypatch):
    built = _arm(monkeypatch)
    expected = dict(_DEFAULT, msa=MsaRuntimePolicy(**_MSA))
    assert built == RuntimeHooks(**expected)


def test_without_the_ops_no_native_launch_is_armed_and_the_pack_stays_triton_free(monkeypatch):
    expected = dict(_DEFAULT, **_NO_PROLOGUE, spec_offsets_op=None, spec_step_fold=None)
    expected["msa"] = dict(_MSA_FIELDS)
    assert _hooks(monkeypatch, ops=False) == _view(expected)
    assert _arm(monkeypatch, ops=False).pack_cpp_scatter is True


@pytest.mark.parametrize("disabled", [("DEP_FIX_HOOKS",), ("DEP_HOOKS",)])
def test_kill_switches_keep_the_draft_pool_fixes(monkeypatch, disabled):
    """Without them a windowed drafter's mirror can run out of pages mid-serving."""
    hooks = _arm(monkeypatch, disabled)
    assert hooks.draft_pool_fixes is True
    if disabled == ("DEP_FIX_HOOKS",):
        assert hooks.draft_kv_reserve_guard is False
    else:
        assert _fields_of(hooks) == {"draft_pool_fixes": True, "msa": {}}
