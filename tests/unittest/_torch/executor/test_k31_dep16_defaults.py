# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defaults of the Kimi K3.1 attention-DP (DEP16) knobs, per disaggregated role.

Where the env leaves a knob unset, a Kimi K3.1 attention-DP server of the knob's role takes the
DEP16 value; every other server (other models and Kimi K3, attention TP, the other role,
aggregated serving) keeps the previous default; a value set in the env always wins.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor import input_commit
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ConversationAwareADPRouter,
    DefaultADPRouter,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import gen_init_decode_headroom_tokens

# The L0 CPU stage invokes registered files with `-m cpu_only`.
pytestmark = pytest.mark.cpu_only

_KNOB_ENVS = (
    "TLLM_ADP_NEW_CONV_BALANCE",
    "TRTLLM_ADP_NEW_CONV_BY_QUEUED_UNCACHED",
    "TLLM_ADP_STICKY_WAIT_ITERS",
    "TRTLLM_ADP_KV_BALANCE_COUNT_FIRST",
    "TRTLLM_DISAGG_GEN_INIT_DECODE_HEADROOM_TOKENS",
    "KIMI_K31_ROUTER_TAIL_DEP",
    "KIMI_K3_ATTN_RES_TOPOLOGY",
    "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS",
)

# Server -> (Kimi K3.1, attention DP, TRTLLM_DISAGG_ROLE; None = unset).
_SERVERS = {
    "k31_dep_ctx": (True, True, "context"),
    "k31_dep_gen": (True, True, "generation"),
    "k31_dep_agg": (True, True, None),
    "k31_tep_ctx": (True, False, "context"),
    "k31_tep_gen": (True, False, "generation"),
    "k31_tep_agg": (True, False, None),
    "other_dep_ctx": (False, True, "context"),
    "other_dep_gen": (False, True, "generation"),
    "other_tp_agg": (False, False, None),
}


def _servers_except(*names):
    return [name for name in _SERVERS if name not in names]


@pytest.fixture
def server(monkeypatch):
    """Configure this process as one of ``_SERVERS``, with every knob env unset, the way the model
    loader does (``configure_k31_engine``); the K3.1 scope is restored after the test."""
    for name in ("_K31", "_K31_DRAFTS", "_K31_ATTENTION_DP", "_K31_GENERATION"):
        monkeypatch.setattr(input_commit, name, getattr(input_commit, name))
    for env in _KNOB_ENVS:
        monkeypatch.delenv(env, raising=False)

    def configure(name, **env):
        k31, attention_dp, role = _SERVERS[name]
        if role is None:
            monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
        else:
            monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        input_commit.configure_k31_engine(
            k31, None, SimpleNamespace(enable_attention_dp=attention_dp)
        )

    return configure


def _dist():
    dist = MagicMock()
    dist.tp_rank = 0
    dist.tp_size = 16
    dist.has_cp_helix = False
    return dist


def _conversation_knobs():
    router = ConversationAwareADPRouter(dist=_dist(), new_conv_placement="least_queued")
    return (
        router._balance_new_conv,
        router._new_conv_by_queued_uncached,
        router._sticky_wait_iters,
    )


def _count_first_slack():
    return DefaultADPRouter(dist=_dist(), kv_balanced_admission=True).kv_balance_count_first_slack


# ---------------------------------------------------------------------------
# Context role: TLLM_ADP_NEW_CONV_BALANCE, TRTLLM_ADP_NEW_CONV_BY_QUEUED_UNCACHED,
# TLLM_ADP_STICKY_WAIT_ITERS (ConversationAwareADPRouter)
# ---------------------------------------------------------------------------


def test_conversation_knobs_k31_attention_dp_context(server):
    server("k31_dep_ctx")
    assert _conversation_knobs() == (True, True, 400)


@pytest.mark.parametrize("name", _servers_except("k31_dep_ctx"))
def test_conversation_knobs_elsewhere_keep_the_old_default(server, name):
    server(name)
    assert _conversation_knobs() == (False, False, 0)


def test_conversation_knobs_env_wins_on_k31_attention_dp_context(server):
    server(
        "k31_dep_ctx",
        TLLM_ADP_NEW_CONV_BALANCE="0",
        TRTLLM_ADP_NEW_CONV_BY_QUEUED_UNCACHED="0",
        TLLM_ADP_STICKY_WAIT_ITERS="7",
    )
    assert _conversation_knobs() == (False, False, 7)


@pytest.mark.parametrize("name", ["k31_tep_ctx", "other_dep_ctx", "k31_dep_agg"])
def test_conversation_knobs_env_wins_elsewhere(server, name):
    server(
        name,
        TLLM_ADP_NEW_CONV_BALANCE="1",
        TRTLLM_ADP_NEW_CONV_BY_QUEUED_UNCACHED="1",
        TLLM_ADP_STICKY_WAIT_ITERS="400",
    )
    assert _conversation_knobs() == (True, True, 400)


# ---------------------------------------------------------------------------
# Generation role: TRTLLM_ADP_KV_BALANCE_COUNT_FIRST (DefaultADPRouter)
# ---------------------------------------------------------------------------


def test_count_first_slack_k31_attention_dp_generation(server):
    server("k31_dep_gen")
    assert _count_first_slack() == 1.1


@pytest.mark.parametrize("name", _servers_except("k31_dep_gen"))
def test_count_first_slack_elsewhere_keeps_the_old_default(server, name):
    server(name)
    assert _count_first_slack() is None


@pytest.mark.parametrize("raw, expected", [("0", None), ("1.3", 1.3)])
def test_count_first_slack_env_wins_on_k31_attention_dp_generation(server, raw, expected):
    server("k31_dep_gen", TRTLLM_ADP_KV_BALANCE_COUNT_FIRST=raw)
    assert _count_first_slack() == expected


@pytest.mark.parametrize("name", ["k31_tep_gen", "other_dep_gen", "k31_dep_ctx"])
def test_count_first_slack_env_wins_elsewhere(server, name):
    server(name, TRTLLM_ADP_KV_BALANCE_COUNT_FIRST="1.1")
    assert _count_first_slack() == 1.1


# ---------------------------------------------------------------------------
# Generation role: TRTLLM_DISAGG_GEN_INIT_DECODE_HEADROOM_TOKENS (KVCacheV2Scheduler)
# ---------------------------------------------------------------------------


def test_decode_headroom_k31_attention_dp_generation(server):
    server("k31_dep_gen")
    assert gen_init_decode_headroom_tokens() == 1024


@pytest.mark.parametrize("name", _servers_except("k31_dep_gen"))
def test_decode_headroom_elsewhere_keeps_the_old_default(server, name):
    server(name)
    assert gen_init_decode_headroom_tokens() == 0


@pytest.mark.parametrize("raw, expected", [("0", 0), ("512", 512)])
def test_decode_headroom_env_wins_on_k31_attention_dp_generation(server, raw, expected):
    server("k31_dep_gen", TRTLLM_DISAGG_GEN_INIT_DECODE_HEADROOM_TOKENS=raw)
    assert gen_init_decode_headroom_tokens() == expected


@pytest.mark.parametrize("name", ["k31_tep_gen", "other_dep_gen", "k31_dep_ctx"])
def test_decode_headroom_env_wins_elsewhere(server, name):
    server(name, TRTLLM_DISAGG_GEN_INIT_DECODE_HEADROOM_TOKENS="1024")
    assert gen_init_decode_headroom_tokens() == 1024


# ---------------------------------------------------------------------------
# Context role: KIMI_K31_ROUTER_TAIL_DEP, KIMI_K3_ATTN_RES_TOPOLOGY (_configure_k31_decode)
# ---------------------------------------------------------------------------

# Module globals of modeling_kimi_linear that _configure_k31_decode writes.
_DECODE_GLOBALS = (
    "_K31_LATENT_DOWN_STREAM",
    "_K31_FUSED_MOE_TAIL",
    "_K31_SEPARATED_ROUTING",
    "_K31_L2PF",
    "_K31_PACKED_MOE_FRONT",
    "_K31_SHARDED_MOE_FRONT",
    "_K31_SHARDED_MOE_FRONT_SHARED",
    "_K31_SHARDED_MOE_FRONT_SORT",
    "_K3_AR_PUSH_AMID",
    "_K3_AR_FINISH_FUSED",
    "_K3_ULL_OPROJ_SANDWICH",
    "_K3_ATTN_RES_PDL_MODE",
    "_K3_ATTN_RES_PDL",
    "_K3_ATTN_RES_PDL_MAX_M",
    "_K3_FUSE_MOE_RESIDUAL",
    "_K31_ROUTER_TAIL_DEP",
    "_ATTN_RES_TOPOLOGY",
    "_FUSED_ATTN_RES_MAX_TOKENS",
)


def _decode_knobs(monkeypatch, name, **env):
    """Build-time resolution of (router tail DEP, attn_res topology, fused attn_res max tokens) for
    the server ``name`` (Kimi K3.1: a text config with ``dsa_config``; else Kimi K3)."""
    from tensorrt_llm._torch.distributed import ops
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    for global_name in _DECODE_GLOBALS:
        monkeypatch.setattr(mkl, global_name, getattr(mkl, global_name))
    for global_name in ("_K3_AR_PUSH", "_K31_BF16_OPROJ_AR_FINISH", "_K31_BF16_OPROJ_AR_PUSH"):
        monkeypatch.setattr(ops, global_name, getattr(ops, global_name))
    for key in _KNOB_ENVS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    k31, attention_dp, role = _SERVERS[name]
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    cfg = SimpleNamespace(dsa_config={"dsa_layers": [4], "index_topk": 2048} if k31 else None)
    # TP 32: the attention-TP generation server then leaves k3_attn_res's early release alone.
    model_config = SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=attention_dp, tp_size=32),
        max_num_tokens=8192,
    )
    mkl._configure_k31_decode(cfg, model_config)
    return mkl._K31_ROUTER_TAIL_DEP, mkl._ATTN_RES_TOPOLOGY, mkl._FUSED_ATTN_RES_MAX_TOKENS


def test_decode_knobs_k31_attention_dp_context(monkeypatch):
    assert _decode_knobs(monkeypatch, "k31_dep_ctx") == (False, "split", 32)


@pytest.mark.parametrize("name", _servers_except("k31_dep_ctx"))
def test_decode_knobs_elsewhere_keep_the_old_default(monkeypatch, name):
    assert _decode_knobs(monkeypatch, name) == (True, "per_token", 1)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"KIMI_K31_ROUTER_TAIL_DEP": "1"}, (True, "split", 32)),
        ({"KIMI_K3_ATTN_RES_TOPOLOGY": "per_token"}, (False, "per_token", 1)),
        ({"KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS": "8"}, (False, "split", 8)),
    ],
)
def test_decode_knobs_env_wins_on_k31_attention_dp_context(monkeypatch, env, expected):
    assert _decode_knobs(monkeypatch, "k31_dep_ctx", **env) == expected


@pytest.mark.parametrize("name", ["k31_tep_ctx", "k31_dep_gen", "other_dep_ctx"])
def test_decode_knobs_env_wins_elsewhere(monkeypatch, name):
    env = {"KIMI_K31_ROUTER_TAIL_DEP": "0", "KIMI_K3_ATTN_RES_TOPOLOGY": "1"}
    assert _decode_knobs(monkeypatch, name, **env) == (False, "split", 32)
