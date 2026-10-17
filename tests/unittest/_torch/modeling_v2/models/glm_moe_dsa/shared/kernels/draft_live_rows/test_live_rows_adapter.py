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
"""The draft_live_rows call sites in the target and their host-side decisions.

* the plan runs draft_live_rows with its closure (post_attention_hop, draft_loop), next to
  the served GEN set without deferred_input_prep, and turns it off with either base; draft_live_rows has
  one numerical form; the package reads no environment variable and
  imports no other package (no deferred_input_prep side-selector or package-directory entry);
* the call sites: ``GlmMtpLayer`` adoption builds the live-row hop only when draft_live_rows
  is selected and its forward tries it only on an armed call that the post_attention_hop
  serves, before the post_attention_hop; the draft loop (``GlmMtpWorker``) decides once per
  forward, arms step 0 only, and makes draft_loop's step-0 gather the identity only when the
  layer compacted;
* the host-side eligibility of the live-row hop and of the step-0 arming,
  case by case, with no device work; the arming is cleared even when the
  step raises;
* the live-row hop is built on every rank of the post_attention_hop's TP group or on none:
  whether the operator is loaded is agreed over the group's communicator
  (mocked here; one collective per post_attention_hop, also on a rank without the
  operator), with a negative control showing that the rank-local rule
  disagrees whenever one rank lacks the operator.
"""

import ast
import inspect
import os
import re
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_testlib  # noqa: E402

GEN_FAST_PATHS_WITHOUT_INPUT_PREP = (
    "absorbed_query,indexer_topk,attention_output_tail,post_moe_hop,draft_front,"
    "router_gemm,post_attention_hop,draft_tail,sparse_mla_decode,draft_absorbed_query,"
    "rope_kv_append,draft_loop,shared_indexer_absorbed_query,routed_moe,"
    "cross_rank_topk,attention_output_tail_prefetch,attention_output_into_hop,"
    "routed_moe_into_hop,deferred_norm,dense_mlp,hop_routing,oproj_weight_staging,"
    "fused_kv_indexer_projection,host_page_retention,shared_expert_prefetch,"
    "draft_routed_moe,cross_layer_expert_prefetch"
)


def _dispatch():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    return dispatch


def test_selector_facts(monkeypatch):
    d = _dispatch()
    monkeypatch.setattr(d, "_log_info", lambda message: None)
    assert "draft_live_rows" in d.SELECTOR_PATHS
    assert set(d.REQUIRED_FEATURES["draft_live_rows"]) == {"post_attention_hop", "draft_loop"}
    assert "draft_live_rows" not in d.NUMERICAL_VARIANTS
    assert plan_testlib.select(
        d, "post_attention_hop,draft_loop,draft_live_rows", monkeypatch
    ) == frozenset({"post_attention_hop", "draft_loop", "draft_live_rows"})
    no_input_prep = frozenset(GEN_FAST_PATHS_WITHOUT_INPUT_PREP.split(","))
    assert plan_testlib.select(
        d, GEN_FAST_PATHS_WITHOUT_INPUT_PREP + ",draft_live_rows", monkeypatch
    ) == no_input_prep | {"draft_live_rows"}
    assert plan_testlib.select(
        d,
        GEN_FAST_PATHS_WITHOUT_INPUT_PREP + ",cross_rank_topk_cluster_merge,draft_live_rows",
        monkeypatch,
    ) >= {
        "cross_rank_topk_cluster_merge",
        "draft_live_rows",
    }
    for base in ("post_attention_hop", "draft_loop"):
        plan = d.resolve(d.ModelingV2Mode.AUTO, {d.DISABLE_ENV: base})
        assert not plan.enabled("draft_live_rows"), base


def _package_dir() -> Path:
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (
        draft_live_rows,
    )

    return Path(draft_live_rows.__file__).parent


def test_package_is_self_contained():
    """No side selector and no import of another package's code."""
    for path in sorted(_package_dir().glob("*.py")):
        src = path.read_text()
        assert "environ" not in src and "getenv" not in src, path.name
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                assert node.module == "dispatch", (path.name, node.module)


def test_mtp_call_site():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmMtpLayer,
    )

    init = inspect.getsource(GlmMtpLayer._adopt_draft)
    assert 'is_enabled("draft_live_rows")' in init and re.search(
        r'get_impl\(\s*"draft_live_rows",\s*"make_live_row_hop"\s*\)', init
    )
    fwd = inspect.getsource(GlmMtpLayer.forward)
    draft_live_rows = fwd.index("self.draft_live_row_hop.can_run(")
    # The post_attention_hop decision is made once, before either hop; the live-row hop is tried
    # first and only when armed.
    assert (
        fwd.index("use_post_attn_hop = ") < draft_live_rows < fwd.index("elif use_post_attn_hop:")
    )
    assert fwd.index("live_rows is not None") < draft_live_rows
    assert fwd.count("self._can_use_post_attn_hop(") == 1
    assert "self._draft_live_rows_compacted = True" in fwd


def test_draft_loop_call_site():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.spec_worker import (  # noqa: E501
        GlmMtpWorker,
    )

    src = inspect.getsource(GlmMtpWorker._forward_linear_draft_loop)
    match = re.search(r'get_impl\(\s*"draft_live_rows",\s*"select_step0"\s*\)', src)
    assert match is not None
    sel = match.start()
    assert sel < src.index("for i in range(runtime_draft_len):")
    assert "if i == 0 and live_rows_step0 is not None:" in src
    ident = src.index("if live_rows_compacted:")
    assert ident < src.index("draft_loop_ops.step0_rows(hidden_states)")
    assert "gather_ids, draft_rows = draft_loop_ops.gather_ids, hidden_states" in src


class _AttnHopStub:
    enabled = True
    rank = 2
    uc_buffer = None
    mc_ptr = 1
    flags = None
    min_sm_count = 148


def test_live_row_hop_eligibility():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_live_rows.hop import (  # noqa: E501
        LiveRowHop,
        required_sms,
    )

    hop = LiveRowHop(_AttnHopStub())
    h6 = torch.zeros(6, 6144, dtype=torch.bfloat16)
    w = torch.zeros(256, 6144, dtype=torch.bfloat16)
    ids = torch.zeros(1, dtype=torch.int64)
    assert hop.call_ineligibility(h6, ids, w) is None
    assert hop.call_ineligibility(h6, torch.zeros(2, dtype=torch.int64), w) is None
    assert hop.call_ineligibility(h6, ids, None) == "no router weight"
    assert hop.call_ineligibility(h6, None, w) == "no live rows"
    assert hop.call_ineligibility(h6, ids.int(), w) == "live-row ids layout"
    assert hop.call_ineligibility(h6, ids.view(1, 1), w) == "live-row ids layout"
    assert (
        hop.call_ineligibility(h6, torch.zeros(4, dtype=torch.int64)[::2], w)
        == "live-row ids layout"
    )
    assert hop.call_ineligibility(h6, torch.zeros(7, dtype=torch.int64), w).startswith(
        "live rows 7 of 6"
    )
    assert hop.call_ineligibility(h6, torch.zeros(0, dtype=torch.int64), w).startswith(
        "live rows 0"
    )
    h12 = torch.zeros(12, 6144, dtype=torch.bfloat16)
    assert hop.call_ineligibility(h12, ids, w).startswith("live rows 1 of 12")
    small = _AttnHopStub()
    small.min_sm_count = 24
    assert LiveRowHop(small).call_ineligibility(h6, ids, w) == "SM count"
    assert [required_sms(r) for r in (1, 2, 3, 8)] == [32, 64, 96, 96]


class _Comm:
    """Mocked TP communicator (MPI style): this rank's value plus fixed peer values."""

    def __init__(self, peers):
        self.peers = [int(v) for v in peers]
        self.calls = []

    def allreduce(self, value, op=None):
        self.calls.append(int(value))
        return int(value) + sum(self.peers)

    def Get_size(self):
        return len(self.peers) + 1


class _FakeAttnHop(_AttnHopStub):
    def __init__(self, comm, rank=2):
        self.comm = comm
        self.rank = rank


@pytest.fixture
def live_row_hop(monkeypatch):
    """The hop module with an empty decision table and the MPI agreement path."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_live_rows import (  # noqa: E501
        hop,
    )

    monkeypatch.setattr(hop.LiveRowHop, "_instances", {})
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)
    return hop


@pytest.mark.parametrize(
    "local, peers, built, reason",
    [
        (True, [1] * 7, True, ""),
        (True, [1] * 6 + [0], False, "another rank lacks trtllm::glm_moe_dsa_draft_live_row_hop"),
        (
            True,
            [1] * 3 + [0] * 4,
            False,
            "another rank lacks trtllm::glm_moe_dsa_draft_live_row_hop",
        ),
        (False, [1] * 7, False, "trtllm::glm_moe_dsa_draft_live_row_hop is not loaded"),
        (False, [0] * 7, False, "trtllm::glm_moe_dsa_draft_live_row_hop is not loaded"),
    ],
)
def test_live_row_hop_agreed_over_tp_group(live_row_hop, monkeypatch, local, peers, built, reason):
    monkeypatch.setattr(live_row_hop, "op_available", lambda: local)
    comm = _Comm(peers)
    w2 = _FakeAttnHop(comm)
    hop = live_row_hop.LiveRowHop.for_post_attn_hop(w2)
    assert (hop is not None) == built
    # This rank takes part in the agreement whatever its own answer: a rank that
    # skipped it would leave its peers waiting in the collective.
    assert comm.calls == [int(local)]
    assert live_row_hop.LiveRowHop._instances[id(w2)] == (w2, hop, reason)
    if built:
        assert hop.w2 is w2 and hop.rank == w2.rank
    # Later calls for the same post_attention_hop return the recorded decision without
    # communicating.
    assert live_row_hop.LiveRowHop.for_post_attn_hop(w2) is hop
    assert comm.calls == [int(local)]


def test_live_row_hop_without_attn_hop_group(live_row_hop, monkeypatch):
    """No enabled post_attention_hop (identical on all ranks) or no communicator: no hop, no collective."""
    monkeypatch.setattr(live_row_hop, "op_available", lambda: True)
    assert live_row_hop.LiveRowHop.for_post_attn_hop(None) is None
    comm = _Comm([1] * 7)
    off = _FakeAttnHop(comm)
    off.enabled = False
    assert live_row_hop.LiveRowHop.for_post_attn_hop(off) is None
    lone = _FakeAttnHop(None)
    assert live_row_hop.LiveRowHop.for_post_attn_hop(lone) is None
    assert (
        live_row_hop.LiveRowHop._instances[id(lone)][2]
        == "post_attention_hop hop has no communicator"
    )
    assert comm.calls == []


class _ThreadComm:
    """One in-process TP group of ``size`` threads; a single integer-sum allreduce."""

    def __init__(self, size):
        self.size = size
        self.total = 0
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(size, timeout=60)

    def allreduce(self, value, op=None):
        with self.lock:
            self.total += int(value)
        self.barrier.wait()
        return self.total

    def Get_size(self):
        return self.size


@pytest.mark.parametrize("lacking", [(), (0,), (3,), (5,), (7,), (4, 5, 6, 7), tuple(range(8))])
def test_live_row_hop_rank_agreement_eight_ranks(live_row_hop, monkeypatch, lacking):
    """Eight ranks (threads) build the hop concurrently: every rank gets it, or none does."""
    size = 8
    tls = threading.local()
    monkeypatch.setattr(live_row_hop, "op_available", lambda: tls.has_op)
    comm = _ThreadComm(size)
    built, local_rule, errors = [None] * size, [None] * size, []

    def rank_main(r):
        tls.has_op = r not in lacking
        try:
            built[r] = (
                live_row_hop.LiveRowHop.for_post_attn_hop(_FakeAttnHop(comm, rank=r)) is not None
            )
            # The construction rule before the agreement: this rank's own operator only.
            local_rule[r] = live_row_hop.op_available()
        except Exception as error:  # noqa: BLE001  (reported below with the rank)
            errors.append((r, repr(error)))

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads) and not errors, errors
    assert built == [not lacking] * size
    # Negative control: the rank-local rule splits the group whenever some but not all ranks lack the op.
    split = 0 < len(lacking) < size
    assert (len(set(local_rule)) == 2) == split


class _Layer:
    """The draft layer's live-row state and its arming API (``GlmMtpLayer``)."""

    def __init__(self, hop):
        self.draft_live_row_hop = hop
        self._draft_live_rows = None
        self._draft_live_rows_compacted = False

    def arm_live_rows(self, rows):
        self._draft_live_rows = rows
        self._draft_live_rows_compacted = False

    def disarm_live_rows(self):
        compacted = bool(self._draft_live_rows_compacted)
        self._draft_live_rows = None
        self._draft_live_rows_compacted = False
        return compacted


class _Draft:
    def __init__(self, hop):
        self.mtp_layers = [_Layer(hop)]


class _Spec:
    layers_to_capture = ()


class _DRAFT_LOOP:
    def __init__(self, b):
        self.batch_size = b
        self.gather_ids = torch.zeros(b, dtype=torch.int64)


class _Worker:
    is_mtp_eagle = True


def test_step0_eligibility():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_live_rows.step0 import (  # noqa: E501
        select_step0,
        step0_ineligibility,
    )

    hop = object()
    draft_loop = _DRAFT_LOOP(1)
    assert step0_ineligibility(_Worker(), _Draft(hop), _Spec(), draft_loop) is None
    other = _Worker()
    other.is_mtp_eagle = False
    assert step0_ineligibility(other, _Draft(hop), _Spec(), draft_loop) == "not MTP Eagle one-model"
    assert (
        step0_ineligibility(_Worker(), _Draft(hop), _Spec(), None) == "no draft_loop draft-loop ops"
    )
    cap = _Spec()
    cap.layers_to_capture = (0,)
    assert (
        step0_ineligibility(_Worker(), _Draft(hop), cap, draft_loop) == "MTP layer output captured"
    )
    none = _Spec()
    none.layers_to_capture = None
    assert step0_ineligibility(_Worker(), _Draft(hop), none, draft_loop) is None
    assert (
        step0_ineligibility(_Worker(), _Draft(None), _Spec(), draft_loop)
        == "MTP layer has no live-row hop"
    )
    bad = _DRAFT_LOOP(2)
    bad.gather_ids = torch.zeros(3, dtype=torch.int64)
    assert step0_ineligibility(_Worker(), _Draft(hop), _Spec(), bad) == "draft_loop gather ids"
    bad.gather_ids = None
    assert step0_ineligibility(_Worker(), _Draft(hop), _Spec(), bad) == "draft_loop gather ids"
    assert select_step0(_Worker(), _Draft(None), _Spec(), draft_loop) is None
    assert select_step0(_Worker(), _Draft(hop), _Spec(), draft_loop) is not None


def test_arming_is_cleared():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_live_rows.step0 import (  # noqa: E501
        select_step0,
    )

    draft = _Draft(object())
    draft_loop = _DRAFT_LOOP(1)
    layer = draft.mtp_layers[0]
    s0 = select_step0(_Worker(), draft, _Spec(), draft_loop)
    with s0.armed() as live:
        assert layer._draft_live_rows is draft_loop.gather_ids
        layer._draft_live_rows_compacted = True
    assert (
        live.compacted and layer._draft_live_rows is None and not layer._draft_live_rows_compacted
    )
    with s0.armed() as live:
        pass
    assert not live.compacted
    with pytest.raises(RuntimeError):
        with s0.armed():
            layer._draft_live_rows_compacted = True
            raise RuntimeError("step failed")
    assert layer._draft_live_rows is None and not layer._draft_live_rows_compacted


def test_mtp_layer_default_off(monkeypatch):
    """Without draft_live_rows the MTP layer's draft_live_rows attributes stay inert (no hop, never armed)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmMtpLayer,
    )

    d = _dispatch()
    monkeypatch.setattr(d, "_log_info", lambda message: None)
    plan_testlib.select(d, GEN_FAST_PATHS_WITHOUT_INPUT_PREP, monkeypatch)
    assert not d.is_enabled("draft_live_rows")
    src = inspect.getsource(GlmMtpLayer._adopt_draft)
    # The hop is created only under is_enabled("draft_live_rows"); the class defaults keep
    # the attributes inert either way.
    assert 'if self.post_attn_hop is not None and dispatch.is_enabled("draft_live_rows"):' in src
    assert GlmMtpLayer.draft_live_row_hop is None and GlmMtpLayer._draft_live_rows is None


@pytest.mark.skipif(
    not os.environ.get("GLM_MOE_DSA_DRAFT_LIVE_ROWS_OP_LIB") and not torch.cuda.is_available(),
    reason="needs the draft_live_rows op",
)
def test_op_registered_with_fake():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_live_rows import (  # noqa: E501
        hop as live_row_hop,
    )

    if not live_row_hop.op_available():
        pytest.skip("draft_live_rows op not loaded in this process")
    live_row_hop._register_fake()  # idempotent
    with torch._subclasses.fake_tensor.FakeTensorMode():
        x = torch.empty(6, 6144, dtype=torch.bfloat16, device="cuda")
        ids = torch.empty(2, dtype=torch.int64, device="cuda")
        out = torch.ops.trtllm.glm_moe_dsa_draft_live_row_hop(
            x,
            x,
            ids,
            torch.empty(6144, dtype=torch.bfloat16, device="cuda"),
            1e-5,
            torch.empty(256, 6144, dtype=torch.bfloat16, device="cuda"),
            torch.empty(10, device="cuda"),
            1,
            torch.empty(4, dtype=torch.int32, device="cuda"),
            0,
            8,
        )
        assert [tuple(t.shape) for t in out] == [(2, 6144), (2, 6144), (2, 256)]
