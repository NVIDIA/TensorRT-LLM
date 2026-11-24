# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three Rubin targets: the server role at construction, the lever-variable check, the startup
lines and the forced-acceptance counts. CPU only."""

import importlib
import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_ROLES = {
    "kimi_k31_pro_nvfp4__sm_107__dep16": ("context", "generation"),
    "kimi_k31_pro_nvfp4__sm_107__helix8": ("generation",),
    "kimi_k31_pro_nvfp4__sm_107__tep8": ("context", "generation"),
}
_FORCE_ENV = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"

# The targets import the family kernels, which need the CuTe DSL.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        importlib.util.find_spec("cutlass") is None, reason="this environment lacks cutlass"
    ),
]


def _module(target: str, name: str = "modeling"):
    return importlib.import_module(f"{_FAMILY}.{target}.{name}")


def _role_env() -> str:
    return importlib.import_module(f"{_FAMILY}.routing").DISAGG_ROLE_ENV


@pytest.mark.parametrize("env", [None, "context", "generation"])
@pytest.mark.parametrize("target", sorted(_ROLES))
def test_a_target_without_routings_record_refuses(target, env, monkeypatch):
    # The role is routing's record in extra_attrs (the loader carries it). The target reads no
    # environment variable, so the rank's role does not stand in for a missing record.
    if env is None:
        monkeypatch.delenv(_role_env(), raising=False)
    else:
        monkeypatch.setenv(_role_env(), env)
    mod = _module(target)
    with pytest.raises(AssertionError, match="routing.py records it"):
        mod._server_role(SimpleNamespace(extra_attrs={}))


@pytest.mark.parametrize("target", sorted(_ROLES))
def test_recorded_role_wins(target, monkeypatch):
    monkeypatch.setenv(_role_env(), "generation")
    mod = _module(target)
    role = _ROLES[target][0]
    assert mod._server_role(SimpleNamespace(extra_attrs={mod._ROLE_ATTR: role})) == role


@pytest.mark.parametrize(
    "target", sorted(t for t, roles in _ROLES.items() if "context" not in roles)
)
def test_a_generation_only_prefill_target_refuses_every_step(target):
    cls = _module(target).PrefillTarget
    with pytest.raises(RuntimeError, match="generation steps only"):
        cls.__new__(cls).forward(SimpleNamespace(num_contexts=1), None, None, None, None)


@pytest.mark.parametrize("target", sorted(_ROLES))
def test_constants_line_states_the_drafter_load(target):
    mod = _module(target)
    core = next(
        obj
        for name, obj in vars(mod).items()
        if name.endswith("Core") and inspect.isclass(obj) and obj.__module__ == mod.__name__
    )
    assert "draft_context_kv_only=self.dims.draft_context_kv_only" in inspect.getsource(
        core.post_load_weights
    )
    if "context" in _ROLES[target]:
        assert "draft_ctx_weights" in mod._CONSTANTS["context"]
        full = mod.constants_line("context", draft_context_kv_only=False)
        assert "draft_ctx_weights=full" in full
        history = mod.constants_line("context", draft_context_kv_only=True)
        assert "draft_ctx_weights=history" in history


def _worker(mod, monkeypatch, value: str):
    from tensorrt_llm._torch.speculative.interface import get_force_num_accepted_tokens_float

    monkeypatch.setenv(_FORCE_ENV, value)
    worker = mod.DFlashWorker.__new__(mod.DFlashWorker)
    torch.nn.Module.__init__(worker)
    worker.force_num_accepted_tokens = get_force_num_accepted_tokens_float()
    worker._forced, worker._forced_calls = None, 0
    return worker


@pytest.mark.parametrize("target", sorted(_ROLES))
def test_forced_acceptance_line(target, monkeypatch):
    mod = _module(target)
    lines = []
    monkeypatch.setattr(mod, "logger", SimpleNamespace(info=lines.append))
    worker = _worker(mod, monkeypatch, "2")
    for _ in range(mod._FORCED_PERIOD):
        # One context row, four generation rows; draft length 7.
        out = worker._apply_force_accepted_tokens(torch.ones(5, dtype=torch.int32), 1, 7)
        assert out.tolist() == [1, 3, 3, 3, 3]
    assert lines == [
        f"Kimi K3.1 Rubin forced acceptance: {target} value=2.0 calls=200 rows=800 "
        "accepted=2400 mean=3.0000"
    ]


@pytest.mark.parametrize("target", sorted(_ROLES))
def test_no_forced_counts_without_the_setting(target, monkeypatch):
    mod = _module(target)
    worker = _worker(mod, monkeypatch, "0")
    out = worker._apply_force_accepted_tokens(torch.ones(3, dtype=torch.int32), 0, 7)
    assert out.tolist() == [1, 1, 1]
    assert worker._forced is None and worker._forced_calls == 0


# The lever variables each server checks at construction: (target, role) -> (passes, refused),
# each refused set holding the one variable the check names.
_LEVER_ENV = {
    ("kimi_k31_pro_nvfp4__sm_107__tep8", "generation"): (
        {"TRTLLM_K3_FC12_SM_RESERVE": "16", "TRTLLM_K3_FC12_DIE_SCHED": "1"},
        ({"TRTLLM_K3_FC12_SM_RESERVE": "8"},),
    ),
    ("kimi_k31_pro_nvfp4__sm_107__tep8", "context"): (
        {"TLLM_MOE_ASYNC_INPUT_QUANT": "0"},
        ({"TRTLLM_CUTEDSL_FC2_N_TILE_SIZE": "256"},),
    ),
    ("kimi_k31_pro_nvfp4__sm_107__helix8", "generation"): (
        {"TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS": "24", "TRTLLM_K3_FC12_DIE_LAYOUT": "1"},
        ({"TRTLLM_K3_FC12_SCHED_LOOKAHEAD": "2"},),
    ),
    ("kimi_k31_pro_nvfp4__sm_107__dep16", "context"): (
        {"TLLM_ROUTING_DISABLE_COOP_LAUNCH": "1"},
        ({"TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": "1"},),
    ),
    ("kimi_k31_pro_nvfp4__sm_107__dep16", "generation"): (
        {"TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET": "536870912", "MEGAMOE_COMBINE_FORMAT": "bf16"},
        (
            {"TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET": "268435456"},
            {"MEGAMOE_COMBINE_FORMAT": "32e4m3xe8m0"},
            {"MEGAMOE_TACTIC_AUTOTUNE": "1"},
            {"MEGAMOE_PRIME_LADDER": "0"},
        ),
    ),
}


@pytest.mark.parametrize(
    "target, role",
    sorted(_LEVER_ENV),
    ids=[f"{t.rsplit('__', 1)[-1]}-{r}" for t, r in sorted(_LEVER_ENV)],
)
def test_the_lever_variables_must_match_the_targets_constants(target, role):
    """A lever variable the target does not read must be unset or hold the target's value; the
    logits budget the engine's KV cache estimate reserves must be the target's."""
    mod = _module(target)
    passes, refused = _LEVER_ENV[(target, role)]
    mod._check_lever_env(role, {})
    mod._check_lever_env(role, passes)
    for bad in refused:
        (name,) = bad
        with pytest.raises(ValueError, match=name):
            mod._check_lever_env(role, {**passes, **bad})


@pytest.mark.parametrize("target", sorted(_ROLES))
def test_the_startup_line_states_the_lever_variables(target, monkeypatch):
    mod = _module(target)
    lines, warnings = [], []
    monkeypatch.setattr(mod, "logger", SimpleNamespace(info=lines.append, warning=warnings.append))
    role = _ROLES[target][0]
    mod.log_constants(role, lever_env={"TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET": "536870912"})
    assert lines[-1] == (
        f"Kimi K3.1 Rubin lever env: {target}/{role}: TLLM_ROUTING_DISABLE_COOP_LAUNCH=unset, "
        "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET=536870912"
    )
    # A context server's MoE routing kernels run without cooperative launch (read in C++).
    assert bool(warnings) is (role == "context")
