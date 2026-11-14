# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_DSPARK_STORE on the dep2ctx target: ``cache_derived_state`` runs the built-in model's derived state first,
then installs the prefill store on the target's DSpark worker, behind TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE
(CPU: no model is built; the worker is the hook surface the install reads)."""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        modeling as dep2ctx,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (  # noqa: E402
    dspark_ctx_store as dcs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (  # noqa: E402
    dspark_prefill_store as dps,
)

pytestmark = pytest.mark.cpu_only

Target = dep2ctx.ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx


class _Worker:
    def __init__(self):
        self.got = {}

    def enable_lean_glue(self, draft_model, enabled=True):
        self.got["S10"] = bool(enabled)

    def enable_fused_prefill_store(self, store):
        self.got["STORE"] = store
        return store is not None


def _target(monkeypatch, off=()):
    order = []
    monkeypatch.setattr(
        dep2ctx.MiniMaxM3ForCausalLM, "cache_derived_state", lambda self: order.append("super")
    )
    monkeypatch.setattr(dep2ctx, "m31_disabled", lambda item: item in off)
    real_install = dps.install
    monkeypatch.setattr(
        dep2ctx._dps, "install", lambda *a, **k: order.append("install") or real_install(*a, **k)
    )
    # The pad-row wiring runs in the same cache_derived_state; give it an empty model and no parameters.
    monkeypatch.setattr(dep2ctx._ppr, "wire", lambda layers, pad_rows: (0, ""))
    monkeypatch.setattr(Target, "parameters", lambda self, recurse=True: iter(()))
    t = object.__new__(Target)
    t.__dict__["model"] = types.SimpleNamespace(layers=[])
    t.__dict__["_pad_rows"] = None
    t.__dict__["spec_worker"] = _Worker()
    t.__dict__["draft_model"] = types.SimpleNamespace()
    return t, order


@pytest.mark.parametrize(
    "off, want",
    [
        ((), dict(S10=True, STORE=True, FOLD=True)),
        (("DEP2CTX_DSPARK_STORE",), dict(S10=False, STORE=False, FOLD=False)),
        (("DEP2CTX_DSPARK_STORE_FOLD",), dict(S10=True, STORE=True, FOLD=False)),
    ],
)
def test_cache_derived_state_installs_after_the_built_in_state(monkeypatch, off, want):
    t, order = _target(monkeypatch, off)
    try:
        Target.cache_derived_state(t)
        assert order == ["super", "install"]
        assert t._dspark_prefill == want
        assert dcs._fold_w[0] is want["FOLD"]
        worker = t.__dict__["spec_worker"]
        if want["STORE"]:
            assert isinstance(worker.got["STORE"], dps.PrefillStore)
        else:
            assert worker.got == {}
    finally:
        dcs.fold_k_norm_weight(False)


def test_cache_derived_state_is_idempotent(monkeypatch):
    """The loader may run it more than once: each run reinstalls the same items."""
    t, _ = _target(monkeypatch)
    try:
        Target.cache_derived_state(t)
        first = dict(t._dspark_prefill)
        Target.cache_derived_state(t)
        assert t._dspark_prefill == first == dict(S10=True, STORE=True, FOLD=True)
    finally:
        dcs.fold_k_norm_weight(False)
