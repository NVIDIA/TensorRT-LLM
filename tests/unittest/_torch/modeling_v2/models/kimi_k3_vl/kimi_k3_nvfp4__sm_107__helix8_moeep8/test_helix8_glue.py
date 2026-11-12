# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix8 targets' glue (each target's ``modeling.py``): built for exactly their constants.

Both helix8 targets (``helix8_moeep8``, ``helix8_moetp2ep4``) declare the same constants and each
carries its own copy of the glue, which has them folded in: no glue class takes a constants object,
the target class refuses any other set, the attention-DP layouts are refused, and what the engine
reads off the model (DSpark path, decode head, hooks, the drafter's role) is the helix8 one.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_FAMILY_DIR = Path(__file__).resolve().parents[7] / _FAMILY.replace(".", "/")
_TARGETS = {
    "helix8_moeep8": "ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8",
    "helix8_moetp2ep4": "ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4",
}


def _glue(target="helix8_moeep8"):
    return _target(target, "modeling")


def _target(target, name):
    return importlib.import_module(f"{_FAMILY}.kimi_k3_nvfp4__sm_107__{target}.{name}")


@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_both_targets_declare_the_folded_constants(target):
    modeling = _glue(target)
    constants = modeling.CONSTANTS
    assert constants == modeling.HELIX8_CONSTANTS
    cls = getattr(modeling, _TARGETS[target])
    assert issubclass(cls, modeling.KimiK3Sm107ForConditionalGeneration)
    assert cls.K3_CONSTANTS is constants


@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_target_class_refuses_other_constants(target):
    modeling = _glue(target)
    other = dataclasses.replace(modeling.HELIX8_CONSTANTS, shared_sm=0)

    class Other(modeling.KimiK3Sm107ForConditionalGeneration):
        K3_CONSTANTS = other

    with pytest.raises(ValueError, match="HELIX8_CONSTANTS"):
        Other(SimpleNamespace())


# Every field of K3Sm107Constants: none is read in the glue except off HELIX8_CONSTANTS.
def _constant_fields():
    return {f.name for f in dataclasses.fields(_glue().K3Sm107Constants)}


def _glue_files():
    files = [
        p
        for t in sorted(_TARGETS)
        for p in sorted((_FAMILY_DIR / f"kimi_k3_nvfp4__sm_107__{t}").glob("*.py"))
    ]
    assert len(files) == 3 * len(_TARGETS), files
    return files


def test_glue_reads_no_target_constants():
    fields = _constant_fields()
    hits = []
    for path in _glue_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in fields:
                base = node.value
                if isinstance(base, ast.Name) and base.id == "HELIX8_CONSTANTS":
                    continue
                hits.append(f"{path.name}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.Attribute) and node.attr in (
                "k3_constants",
                "_constants",
                "megamoe_experts",
            ):
                hits.append(f"{path.name}:{node.lineno} .{node.attr}")
    assert not hits, hits


def test_glue_reads_no_environment_variable():
    """No model switch from the environment; the one read is the MNNVL transport's fabric-handle
    override, which main's MNNVL all-reduce reads the same way."""
    hits = [
        f"{path.name}:{lineno}"
        for path in _glue_files()
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if ("os.environ" in line or "getenv(" in line) and "TRTLLM_FORCE_MNNVL_AR" not in line
    ]
    assert not hits, hits


@pytest.mark.parametrize(
    "name",
    [
        "KimiK3DefaultDecoderLayer",
        "KimiK3DefaultModelingV2",
        "KimiK3DefaultForCausalLM",
        "K3MoERuntime",
        "K3MLARuntime",
        "K3MLAAttention",
        "K3KDALinearAttention",
        "_swap_linear_to_fp8_weight_read",
        "_convert_moe_mlps_to_fp8_weight_read",
    ],
)
def test_glue_takes_no_constants(name):
    obj = getattr(_glue(), name)
    params = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj).parameters
    assert "constants" not in params


@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_glue_shares_the_family_allreduce_module(target):
    """The MNNVL workspace caches of ``ops/comm/sm107_allreduce`` are shared with ``dspark/``: one
    module, which no target copies."""
    allreduce = importlib.import_module(f"{_FAMILY}.ops.comm.sm107_allreduce")
    assert _glue(target).lamport_push_target is allreduce.lamport_push_target
    assert _glue(target).LamportPending is allreduce.LamportPending
    target_dir = _FAMILY_DIR / f"kimi_k3_nvfp4__sm_107__{target}"
    assert sorted(p.name for p in target_dir.glob("*.py")) == [
        "__init__.py",
        "modeling.py",
        "weights.py",
    ]


def test_attention_dp_is_refused():
    from tensorrt_llm.mapping import Mapping

    adp = Mapping(world_size=2, rank=0, tp_size=2, enable_attention_dp=True)
    with pytest.raises(NotImplementedError, match="attention DP"):
        _glue()._refuse_attention_dp(adp)
    _glue()._refuse_attention_dp(Mapping(world_size=2, rank=0, tp_size=2))
    with pytest.raises(NotImplementedError, match="attention DP"):
        _glue().K3MoERuntime(SimpleNamespace(mapping=adp), None, 1, {})


def test_moe_options_are_helix8():
    moe_runtime = _glue()
    assert moe_runtime._k3_fc12_options("CUTEDSL_FC12") == {
        "sm_reserve": 16,
        "die_major": True,
        "die_sched_min_tokens": 24,
    }
    assert moe_runtime._k3_fc12_options("CUTEDSL") == {}
    assert moe_runtime._K3_SHARED_SM == 16
    with pytest.raises(ValueError, match="MEGAMOE_CUTEDSL"):
        moe_runtime.K3MoERuntime._routed_moe_model_config(
            SimpleNamespace(moe_backend="MEGAMOE_CUTEDSL")
        )


def test_attn_res_pdl_serves_up_to_128_rows():
    attn_res = _glue()
    assert attn_res._defer_moe_residual(1)
    assert attn_res._defer_moe_residual(128)
    assert not attn_res._defer_moe_residual(129)


def test_engine_declarations_are_helix8():
    """What the engine and the DSpark shell read off the text model and the target class."""
    import torch
    from torch import nn

    from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
        DEFAULT_RUNTIME_HOOKS,
        HybridCachePolicy,
        runtime_hooks_of,
    )

    modeling = _glue()
    drafter = importlib.import_module(f"{_FAMILY}.dspark.drafter")
    head = importlib.import_module(f"{_FAMILY}.kernels.k3r_spec.spec_decode_head")
    text_cls = modeling.KimiK3DefaultForCausalLM
    assert text_cls.dspark_path == drafter.K3DSparkPath(spec_tail_fusion=True, tail_v2=True)
    assert text_cls.dspark_context_kv_only is False

    text = text_cls.__new__(text_cls)
    nn.Module.__init__(text)
    assert text.spec_decode_prologue is head.spec_decode_head
    text_hooks = runtime_hooks_of(text)
    assert text_hooks.decode_prologue is head.spec_decode_head
    cache = importlib.import_module(f"{_FAMILY}.ops.cache.k3_sm107_cache_manager")
    for target, cls_name in _TARGETS.items():
        cls = getattr(_target(target, "modeling"), cls_name)
        outer = cls.__new__(cls)
        nn.Module.__init__(outer)
        outer.llm = text
        hooks = runtime_hooks_of(outer)
        assert hooks.torch_mem_safety_margin == 4 * 1024**3
        assert hooks.hybrid_cache == HybridCachePolicy(state_index_upload_kernel=True)
        assert hooks.kv_cache_manager_cls is cache.KimiK3Sm107CacheManager
        rest = dataclasses.replace(
            hooks,
            torch_mem_safety_margin=None,
            hybrid_cache=DEFAULT_RUNTIME_HOOKS.hybrid_cache,
            kv_cache_manager_cls=None,
        )
        assert rest == text_hooks
    assert torch.float8_e4m3fn == drafter.K3DSparkDrafter.draft_kv_dtype
