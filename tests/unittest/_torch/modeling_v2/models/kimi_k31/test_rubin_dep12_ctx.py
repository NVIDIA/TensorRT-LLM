# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep12 and dep8 on the dep16 target, CPU only: attention DP and MoE EP over 12 or 8 ranks on a
context server.

The target binds its world size from the Mapping: the build check admits 16 ranks on both server
roles and 12 or 8 on a context server only, against the role routing recorded, and a step checks
the engine's per-rank row counts against the bound size. Each rank logs its layout and expert window
at startup. The expert windows and the weight shares are in ``test_rubin_weights_modelopt.py``,
the routing in ``test_routing_sm107.py``.
"""

import importlib
import importlib.util
import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_a2a import (
    a2a_expert_window,
)
from tensorrt_llm.mapping import Mapping

_MODELING = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__dep16.modeling"
)

# The target imports the family kernels, which need the CuTe DSL.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        importlib.util.find_spec("cutlass") is None, reason="this environment lacks cutlass"
    ),
]

_DEP12 = dict(world_size=12, tp_size=12, enable_attention_dp=True, moe_ep_size=12)
_DEP16 = dict(world_size=16, tp_size=16, enable_attention_dp=True, moe_ep_size=16)
_DEP8 = dict(world_size=8, tp_size=8, enable_attention_dp=True, moe_ep_size=8)
_DEP24 = dict(world_size=24, tp_size=24, enable_attention_dp=True, moe_ep_size=24)
_ROLES = [(_DEP12, "context"), (_DEP8, "context"), (_DEP16, "context"), (_DEP16, "generation")]
_ROLE_IDS = ["dep12-context", "dep8-context", "dep16-context", "dep16-generation"]


def _mod():
    return importlib.import_module(_MODELING)


def _mapping(layout: dict, rank: int = 0) -> Mapping:
    return Mapping(rank=rank, **layout)


def test_the_world_sizes_by_role():
    """16 ranks on both roles, 12 or 8 on a context server, each world size named as routing
    names the layout."""
    mod = _mod()
    assert dict(mod._WORLD_SIZES) == {"context": (16, 12, 8), "generation": (16,)}
    assert dict(mod._LAYOUT_NAMES) == {16: "dep16", 12: "dep12", 8: "dep8"}


@pytest.mark.parametrize("layout, role", _ROLES, ids=_ROLE_IDS)
def test_the_build_takes_its_layouts(layout, role):
    mod = _mod()
    for rank in (0, layout["world_size"] - 1):
        mod._check_layout(_mapping(layout, rank), role)


_CONTEXT_RUNS = (
    "a context server of this target runs dep16 (attention DP 16) or dep12 (attention DP 12) or "
    "dep8 (attention DP 8)"
)
_FACTS = "; (world, tp, pp, cp, attention DP) is"


@pytest.mark.parametrize(
    "layout, role, named",
    [
        (
            _DEP12,
            "generation",
            f"a generation server of this target runs dep16 (attention DP 16){_FACTS} "
            "(12, 12, 1, 1, True)",
        ),
        (
            _DEP8,
            "generation",
            f"a generation server of this target runs dep16 (attention DP 16){_FACTS} "
            "(8, 8, 1, 1, True)",
        ),
        (_DEP24, "context", f"{_CONTEXT_RUNS}{_FACTS} (24, 24, 1, 1, True)"),
        (dict(world_size=12, tp_size=12, moe_ep_size=12), "context", "(12, 12, 1, 1, False)"),
        (
            dict(_DEP12, moe_tp_size=2, moe_ep_size=6),
            "context",
            "dep12 runs MoE EP 12; (moe_tp, moe_ep) is (2, 6)",
        ),
    ],
    ids=["dep12-generation", "dep8-generation", "dep24", "tp12", "dep12-moe-tp2"],
)
def test_the_build_refuses_other_layouts(layout, role, named):
    with pytest.raises(AssertionError, match=re.escape(named)):
        _mod()._check_layout(_mapping(layout), role)


class _PastLayout(Exception):
    """The build check read the recipe's quantization: it got past the layout."""


class _Quant:
    @property
    def quant_algo(self):
        raise _PastLayout


def _built_config(layout: dict, role: str | None) -> SimpleNamespace:
    """A model config that passes ``_check_build`` up to its layout check: the K3.1 text config
    the target pins, ``layout``'s Mapping and the role routing records (None: no record)."""
    mod = _mod()
    text = SimpleNamespace(
        **dict(mod._K31),
        linear_attn_config=dict(mod._K31_KDA),
        dsa_config=dict(mod._K31_DSA),
    )
    return SimpleNamespace(
        pretrained_config=text,
        mapping=_mapping(layout),
        extra_attrs={} if role is None else {mod._ROLE_ATTR: role},
        quant_config=_Quant(),
    )


@pytest.fixture
def sm107(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 7))


@pytest.mark.parametrize("layout, role", _ROLES, ids=_ROLE_IDS)
def test_the_build_check_takes_the_layouts_of_the_recorded_role(sm107, layout, role):
    with pytest.raises(_PastLayout):
        _mod()._check_build(_built_config(layout, role))


def test_the_build_check_refuses_dep12_on_a_generation_server(sm107):
    """The role routing recorded decides: dep12 passes on a context server and stops at the
    layout on a generation server; a config without a recorded role stops too."""
    mod = _mod()
    with pytest.raises(AssertionError, match=re.escape("a generation server of this target runs")):
        mod._check_build(_built_config(_DEP12, "generation"))
    with pytest.raises(AssertionError, match="no server role"):
        mod._check_build(_built_config(_DEP12, None))


def _metadata(counts, num_contexts=1, num_ctx_tokens=5, num_generations=0):
    return SimpleNamespace(
        num_tokens=counts[0],
        all_rank_num_tokens=list(counts),
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        num_generations=num_generations,
        mamba_metadata=None,
        kv_cache_manager=None,
        is_cuda_graph=False,
    )


@pytest.mark.parametrize("dp_size", [8, 12, 16])
def test_a_step_counts_every_rank_of_the_bound_size(monkeypatch, dp_size):
    """A step takes the engine's row count of each of ``dp_size`` ranks, and refuses another
    number of ranks."""
    mod = _mod()
    monkeypatch.setattr(mod, "_dsa_step_args", lambda *args: None)
    counts = (5,) + (0,) * (dp_size - 1)
    st = mod._build_step_args(_metadata(counts), torch.arange(5), None, None, dp_size)
    assert st.all_rank_num_tokens == counts
    assert (st.num_tokens, st.num_contexts, st.num_ctx_tokens) == (5, 1, 5)
    decode = _metadata((8,) * dp_size, num_contexts=0, num_ctx_tokens=0, num_generations=2)
    st = mod._build_step_args(decode, torch.arange(8), None, None, dp_size, decode=True)
    assert (st.num_generations, st.rows_per_gen, st.all_rank_num_tokens) == (2, 4, (8,) * dp_size)
    other = 28 - dp_size  # 20 for 8, 16 for 12, 12 for 16
    with pytest.raises(AssertionError):
        mod._build_step_args(_metadata(counts), torch.arange(5), None, None, other)


class _Stop(Exception):
    """Raised by the step-builder stand-in once it has seen the call."""


def test_the_targets_pass_the_cores_dp_size_to_each_step(monkeypatch):
    """Both targets bind the core's DP size at construction and build each step's arguments
    with it."""
    mod = _mod()
    seen = []

    def build_step_args(md, position_ids, spec_metadata, pools, dp_size, *, decode=False):
        seen.append((dp_size, decode))
        raise _Stop

    monkeypatch.setattr(mod, "_build_step_args", build_step_args)
    rt = SimpleNamespace(dsa_pools=SimpleNamespace(bind=lambda manager: None))
    core = SimpleNamespace(
        _layers=(),
        _head=None,
        _rt=rt,
        dims=SimpleNamespace(dp_size=12),
        _dense_hooks=mod._NoDenseHooks(),
    )
    for cls in (mod.PrefillTarget, mod.DecodeTarget):
        with pytest.raises(_Stop):
            cls(core).forward(
                SimpleNamespace(kv_cache_manager=None), None, torch.arange(1), None, None
            )
    assert seen == [(12, False), (12, True)]


@pytest.mark.parametrize(
    "world, rank, role, want",
    [
        (12, 0, "context", "dep12 (attention DP 12, MoE EP 12), EP rank 0 holds experts [0, 75)"),
        (12, 8, "context", "EP rank 8 holds experts [600, 674) (74)"),
        (12, 11, "context", "EP rank 11 holds experts [822, 896) (74)"),
        (16, 15, "generation", "dep16 (attention DP 16, MoE EP 16), EP rank 15 holds"),
        (8, 0, "context", "dep8 (attention DP 8, MoE EP 8), EP rank 0 holds experts [0, 112)"),
        (8, 7, "context", "EP rank 7 holds experts [784, 896) (112)"),
    ],
    ids=["dep12-rank0", "dep12-rank8", "dep12-rank11", "dep16-rank15", "dep8-rank0", "dep8-rank7"],
)
def test_the_layout_line_states_the_layout_and_the_rank_s_experts(world, rank, role, want):
    """One startup line per rank tells a dep12 server from a dep16 one and states the rank's
    expert window."""
    mod = _mod()
    first, count = a2a_expert_window(896, world, rank)
    d = SimpleNamespace(
        dp_size=world, moe_ep=world, moe_ep_rank=rank, expert_offset=first, local_experts=count
    )
    line = mod.layout_line(role, d)
    assert line.startswith(f"Kimi K3.1 Rubin layout: kimi_k31_pro_nvfp4__sm_107__dep16/{role}: ")
    assert want in line
    assert line.endswith(f"EP rank {rank} holds experts [{first}, {first + count}) ({count})")
