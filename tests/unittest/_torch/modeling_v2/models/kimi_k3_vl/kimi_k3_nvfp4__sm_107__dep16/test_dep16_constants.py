# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kimi_k3_nvfp4__sm_107__dep16: its constants are the dep16 generation profile, it reads no
environment variable, and its class declares no model defaults (the recipe carries the engine
settings)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_TARGET = "kimi_k3_nvfp4__sm_107__dep16"
_CLASS = "ModelingV2KimiK3Nvfp4Sm107Dep16"
_TARGET_DIR = Path(__file__).resolve().parents[7] / _FAMILY.replace(".", "/") / _TARGET

_EXPECTED = {
    "spec_head_fusion": False,
    "fuse_moe_residual": True,
    "fp8w_lowm_gemm": True,
    "ld_fused": False,
    "shared_sm": 0,
    "fc12_die_major": False,
    # The MegaMoE module class, compared by name.
    "megamoe_experts": "K3MegaMoE",
    "router_exact_kda": False,
    "router_bf16_gate": False,
    "fp8_weight_read_moe_mlp": True,
    "latent_ar_own_ws": False,
    "attn_res_pdl": True,
    "attn_res_pdl_max_m": 64,
    "attn_res_mxfp8": False,
    "attn_res_early_trigger": False,
    "kda_prepad_prefill": False,
    "kda_fused_conv_prefill": False,
    "kda_l2_prefetch": False,
    "mla_q_norm_pdl": False,
    "fp4_mla_page_plan": False,
    "fp4_mla_reduce_pdl": False,
    "spec_tail_fusion": True,
    "dspark_tail_v2": False,
    "dspark_context_kv_only": False,
    "torch_mem_safety_margin": 4 * 1024**3,
}


def test_constants_are_the_reference_profile():
    import importlib

    CONSTANTS = importlib.import_module(f"{_FAMILY}.{_TARGET}.modeling").CONSTANTS

    got = dataclasses.asdict(CONSTANTS)
    megamoe = got["megamoe_experts"]
    assert megamoe.__module__ == f"{_FAMILY}.{_TARGET}.modeling", megamoe.__module__
    got["megamoe_experts"] = megamoe.__qualname__
    assert got == _EXPECTED


def test_target_reads_no_environment_variable():
    assert (_TARGET_DIR / "modeling.py").is_file(), _TARGET_DIR
    readers = re.compile(r"os\.environ|getenv\(")
    hits = [
        f"{path.name}:{lineno}"
        for path in sorted(_TARGET_DIR.rglob("*.py"))
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if readers.search(line)
    ]
    assert not hits, hits


def test_target_class_declares_its_constants_and_no_model_defaults():
    import importlib

    module = importlib.import_module(f"{_FAMILY}.{_TARGET}.modeling")
    cls = getattr(module, _CLASS)
    # The target owns its model: no class of another SM107 target's glue in its bases.
    others = (
        ".kimi_k3_nvfp4__sm_107__tp1.",
        ".kimi_k3_nvfp4__sm_107__helix8_",
    )
    assert not any(o in base.__module__ for base in cls.__mro__ for o in others), cls.__mro__
    assert cls.K3_CONSTANTS is importlib.import_module(f"{_FAMILY}.{_TARGET}.modeling").CONSTANTS
    # The model loader applies the model defaults of the checkpoint's built-in class before it
    # routes a Modeling V2 target, so a target's get_model_defaults would never run: the recipe
    # carries the engine settings.
    assert not hasattr(cls, "get_model_defaults")
