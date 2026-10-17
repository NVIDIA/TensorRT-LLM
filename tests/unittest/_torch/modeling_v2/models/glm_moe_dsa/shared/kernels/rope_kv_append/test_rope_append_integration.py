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
"""rope_kv_append through the target's DSA call sites (the adopted attention, on decode steps).

Real MLA/DSA modules at GLM-5.2 TP8 attention geometry (8 local heads,
q-LoRA 2048, kv-LoRA 512, nope 192, rope 64, v 256, 32 x 128 indexer, top-k
2048, FP8 KV cache, block 64, heuristic GVR top-k, CuTe DSL logits), four
target layers in the GLM pattern full/shared/shared/shared plus the MTP layer,
a real ``DSACacheManager`` filled with FP8 context and real generation
metadata (verify M = 6, MTP draft rows 6 then 1), for one request and for
two requests with different cache lengths. Each step runs the production
``_forward_dsa_attn`` natively (no fast path running) and with rope_kv_append
from identical state. Every KV and indexer-K pool byte must be identical, the
shared top-k buffer must hold the same per-row selection and the attention
outputs must be within the native run-to-run spread.
"""

import pytest
import torch
from rope_append_stack import (
    CTX_REQ_TOKENS,
    HEADS,
    KV_LORA,
    MTP_LAYER,
    PATTERN,
    ROPE,
    Stack,
    load_dev_library,
    set_features,
)

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="rope_kv_append targets Blackwell decode",
    ),
]


@pytest.fixture(scope="module", params=[32768, 491_061], ids=["ctx32k", "ctx491k"])
def stack(request):
    load_dev_library()
    s = Stack(request.param)
    yield s
    s.shutdown()


@pytest.fixture(scope="module")
def stack32k():
    """32K context stack for the call-site fallback cases."""
    load_dev_library()
    s = Stack(32768)
    s.add_context_request()
    yield s
    s.shutdown()


def test_native_state_characterization(stack):
    """Native run-to-run behaviour: cache bytes are deterministic; the top-k selection is reported.

    The heuristic GVR top-k emits its rows in a varying order and, on some
    inputs, selects different sets in two runs from identical state; the
    per-row symmetric difference is printed so it is characterized, not hidden.
    """
    md = stack.metadata(6, stack.ctx_len)
    state0 = stack.state(md)
    set_features("")
    stack.run_layers(md, range(len(PATTERN)), 6, stack.ctx_len, seed=1)
    st_a = stack.state(md)
    stack.restore(md, state0)
    stack.run_layers(md, range(len(PATTERN)), 6, stack.ctx_len, seed=1)
    st_b = stack.state(md)
    topk_idx = len(st_a) - 1 if md.shared_topk_indices is not None else -1
    for i, (x, y) in enumerate(zip(st_a, st_b)):
        if i != topk_idx:
            assert torch.equal(x, y), f"native cache state {i} differs"
    if topk_idx >= 0:
        diff = Stack._topk_set_diff(st_a[topk_idx][:6], st_b[topk_idx][:6])
        print(
            f"ctx {stack.ctx_len}: native top-k per-row symmetric difference between runs: {diff}"
        )


def test_target_verify_layers(stack):
    """Full layer (deferred scatter, grouped bridge with the reuse copy) + 3 shared layers."""
    md = stack.metadata(6, stack.ctx_len)
    stack.compare(md, lambda: stack.run_layers(md, range(len(PATTERN)), 6, stack.ctx_len, seed=2))
    native, _, rope_append_counts = stack.engagement
    assert native.get("native_scatter") == 1 and native.get("native_rope") == 4
    assert native.get("native_remap") == 4 and "rope_append_mla_rope_generation" not in native
    # rope_kv_append: one fused rope per layer (the full layer carries the scatter), one grouped
    # bridge launch for the whole group, no native rope/scatter/remap.
    assert (
        rope_append_counts.get("rope_append_mla_rope_generation") == 4
        and rope_append_counts.get("rope_append_fused_scatter") == 1
    )
    assert (
        rope_append_counts.get("rope_append_remap_topk_to_global") == 4
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts


@pytest.mark.parametrize("accepted", [1, 4, 6])
def test_mtp_draft_steps(stack, accepted):
    """Draft step 0 (row stash of the last accepted row) and a reuse step (rows = 1)."""
    md0 = stack.metadata(6, stack.ctx_len)
    md0.set_in_mtp_draft_loop(True)
    md0.set_mtp_num_accepted(torch.tensor([accepted], dtype=torch.int32, device=stack.dev))
    md0.set_skip_topk(False)
    stack.compare(md0, lambda: stack.run_layers(md0, [MTP_LAYER], 6, stack.ctx_len, seed=3))
    rope_append_counts = stack.engagement[2]
    assert (
        rope_append_counts.get("rope_append_fused_scatter") == 1
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts

    md1 = stack.metadata(1, stack.ctx_len + 5)
    md1.set_in_mtp_draft_loop(True)
    md1.set_skip_topk(True)
    md1.shared_topk_indices.copy_(md0.shared_topk_indices)
    stack.compare(md1, lambda: stack.run_layers(md1, [MTP_LAYER], 1, stack.ctx_len + 5, seed=4))
    rope_append_counts = stack.engagement[2]
    assert (
        rope_append_counts.get("rope_append_fused_scatter") == 1
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts


def test_target_verify_layers_two_requests(stack):
    """Two generation requests with different cache lengths (request-major rows 0..5, 6..11)."""
    md = stack.metadata(6, stack.ctx_len, nreq=2)
    assert md.num_generations == 2 and md.num_tokens == 12
    stack.compare(
        md, lambda: stack.run_layers(md, range(len(PATTERN)), 6, stack.ctx_len, seed=5, nreq=2)
    )
    rope_append_counts = stack.engagement[2]
    assert (
        rope_append_counts.get("rope_append_mla_rope_generation") == 4
        and rope_append_counts.get("rope_append_fused_scatter") == 1
    )
    assert (
        rope_append_counts.get("rope_append_remap_topk_to_global") == 4
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts


@pytest.mark.parametrize("accepted", [(1, 6), (4, 2)])
def test_mtp_draft_steps_two_requests(stack, accepted):
    """Draft step 0 stashes each request's own last accepted row; reuse step with 2 x 1 rows."""
    md0 = stack.metadata(6, stack.ctx_len, nreq=2)
    md0.set_in_mtp_draft_loop(True)
    md0.set_mtp_num_accepted(torch.tensor(list(accepted), dtype=torch.int32, device=stack.dev))
    md0.set_skip_topk(False)
    stack.compare(md0, lambda: stack.run_layers(md0, [MTP_LAYER], 6, stack.ctx_len, seed=6, nreq=2))
    rope_append_counts = stack.engagement[2]
    assert (
        rope_append_counts.get("rope_append_fused_scatter") == 1
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts

    md1 = stack.metadata(1, stack.ctx_len + 5, nreq=2)
    md1.set_in_mtp_draft_loop(True)
    md1.set_skip_topk(True)
    md1.shared_topk_indices.copy_(md0.shared_topk_indices)
    stack.compare(
        md1, lambda: stack.run_layers(md1, [MTP_LAYER], 1, stack.ctx_len + 5, seed=7, nreq=2)
    )
    rope_append_counts = stack.engagement[2]
    assert (
        rope_append_counts.get("rope_append_fused_scatter") == 1
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts


def test_lost_deferred_scatter_raises(stack, monkeypatch):
    """A call site that defers the indexer scatter but drops the packet fails at once.

    Reproduces a call-site placement defect (the packet never reached the
    generation rope, so nothing wrote the step's indexer keys): the deferring
    call site's consumed check must raise during that forward.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (
        attention_forward,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        RopeKvAppendContractError,
    )

    orig = attention_forward.forward_generation_sparse_attn

    def dropping(*args, deferred_indexer_k=None, **kwargs):
        return orig(*args, **kwargs)

    monkeypatch.setattr(attention_forward, "forward_generation_sparse_attn", dropping)
    md = stack.metadata(6, stack.ctx_len)
    state0 = stack.state(md)
    set_features("rope_kv_append")
    try:
        with pytest.raises(RopeKvAppendContractError, match="not consumed"):
            stack.run_layers(md, [0], 6, stack.ctx_len, seed=8)
    finally:
        set_features("")
        # The raise skipped the layer's join of the forked prev-top-k copy.
        stack.layers[0].mqa.indexer.maybe_join_prev_topk_copy()
        torch.cuda.synchronize()
        stack.restore(md, state0)


def test_deferred_scatter_contract(stack):
    """The packet is single use, the check sees a missing consumer, undeclared callers get None."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (
        rope_kv_append,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
        RopeKvAppendContractError,
    )

    md = stack.metadata(6, stack.ctx_len)
    state0 = stack.state(md)
    mla = stack.layers[0]
    k_fp8, k_scale = stack.indexer_k(mla, 6, stack.ctx_len, seed=9)
    set_features("rope_kv_append")
    try:
        assert rope_kv_append.defer_indexer_k_scatter(mla, k_fp8, k_scale, md) is None
        pkt = rope_kv_append.defer_indexer_k_scatter(mla, k_fp8, k_scale, md, forwards_packet=True)
        assert pkt is not None and pkt.consumer is None and pkt.layer_idx == 0
        with pytest.raises(RopeKvAppendContractError, match="not consumed"):
            rope_kv_append.check_indexer_k_scatter_consumed(pkt)
        rope_kv_append.run_indexer_k_scatter_native(pkt)
        rope_kv_append.check_indexer_k_scatter_consumed(pkt)
        with pytest.raises(RopeKvAppendContractError, match="consumed twice"):
            rope_kv_append.run_indexer_k_scatter_native(pkt)
        rope_kv_append.check_indexer_k_scatter_consumed(None)
        torch.cuda.synchronize()
        # The native fallback wrote exactly what the native scatter writes.
        after_rope_append = stack.state(md)
        stack.restore(md, state0)
        set_features("")
        mla.mqa.indexer._update_k_cache(k_fp8, k_scale, md)
        torch.cuda.synchronize()
        native = stack.state(md)
        for i, (a, b) in enumerate(zip(after_rope_append, native)):
            assert torch.equal(a, b), f"state {i} differs"
    finally:
        set_features("")
        stack.restore(md, state0)


def test_pool_view_memo_is_lazy(stack):
    """The fused bridge leaves the pool-view memo unpopulated; a later reader populates it.

    Accepted deviation from the native remap's side effect: every reader of
    ``_cached_*`` calls ``_ensure_pool_view_cached()`` first, so values are
    unchanged and only the memo's ``req_idx - num_contexts`` kernel moves to
    the first such reader of the step (none on the rope_kv_append/sparse_mla_decode/native-FMHA
    decode path).
    """
    md = stack.metadata(6, stack.ctx_len)
    state0 = stack.state(md)
    set_features("rope_kv_append")
    try:
        stack.run_layers(md, range(len(PATTERN)), 6, stack.ctx_len, seed=10)
        assert not md._pool_cache_valid
        md._ensure_pool_view_cached()
        nc = md.num_contexts
        want_req = md.req_idx_per_token[md.num_ctx_tokens : md.num_tokens] - nc
        assert torch.equal(md._cached_req_idx_gen, want_req)
        assert torch.equal(md._cached_block_table_gen, md.block_table[nc : md.num_seqs])
    finally:
        set_features("")
        stack.restore(md, state0)


def test_default_off_dispatch(stack):
    """With the selector empty, no rope_kv_append helper is consulted (native call graph)."""
    set_features("")
    md = stack.metadata(6, stack.ctx_len)
    assert dispatch.enabled_features() == frozenset()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (
        rope_kv_append,
    )

    assert rope_kv_append.defer_indexer_k_scatter(stack.layers[0], None, None, md) is None
    assert rope_kv_append.new_bridge_packet(md, True) is None


# ------------------------------------------------------------------------------------------
# Call-site fallbacks: every rejection selects the native path before any rope_kv_append side
# effect, the pools stay byte-identical to the rope_kv_append-off run and the engagement counters
# show which path ran.

ROPE_APPEND_ENTRY_KEYS = (
    "rope_append_mla_rope_generation",
    "rope_append_remap_topk_to_global",
    "rope_append_bridge_kernel",
    "rope_append_deferred_scatter",
    "rope_append_fused_scatter",
    "rope_append_run_indexer_k_scatter_native",
)


def _no_rope_append_engaged(counts):
    return not any(counts.get(k) for k in ROPE_APPEND_ENTRY_KEYS)


def test_fallback_rope_ineligible_after_deferral(stack32k):
    """Rope outside the fused domain (row-strided latent): the deferred scatter runs natively."""
    st = stack32k
    md = st.metadata(6, st.ctx_len)
    segs = [(6, st.ctx_len)]
    st.compare(
        md, lambda: st.run_seg(md, range(len(PATTERN)), segs, seed=31, latent_row_stride=640)
    )
    native, _, scatter_counts = st.engagement
    assert native.get("native_scatter") == 1 and native.get("native_rope") == 4
    assert (
        scatter_counts.get("rope_append_deferred_scatter") == 1
        and scatter_counts.get("rope_append_run_indexer_k_scatter_native") == 1
    )
    assert scatter_counts.get("rope_append_check_indexer_k_scatter_consumed") == 1
    assert not scatter_counts.get("rope_append_mla_rope_generation") and not scatter_counts.get(
        "native_scatter"
    )
    assert scatter_counts.get("native_rope") == 4
    # The bridge does not depend on the latent layout and stays engaged.
    assert (
        scatter_counts.get("rope_append_remap_topk_to_global") == 4
        and scatter_counts.get("rope_append_bridge_kernel") == 1
    )


@pytest.fixture(scope="module")
def stack_g9():
    """One full layer leading 8 shared layers: a group larger than the fused bridge supports."""
    load_dev_library()
    s = Stack(32768, pattern=["F"] + ["S"] * 8)
    yield s
    s.shutdown()


def test_fallback_group_larger_than_8(stack_g9):
    """Group of 9: the leader copies and remaps natively, followers slice the native batch."""
    st = stack_g9
    md = st.metadata(6, st.ctx_len)
    st.compare(md, lambda: st.run_layers(md, range(len(st.pattern)), 6, st.ctx_len, seed=32))
    _, _, scatter_counts = st.engagement
    assert scatter_counts.get("native_remap") == 1 and not scatter_counts.get(
        "rope_append_bridge_kernel"
    )
    assert scatter_counts.get("rope_append_remap_topk_to_global") == 9
    assert (
        scatter_counts.get("rope_append_mla_rope_generation") == 9
        and scatter_counts.get("rope_append_fused_scatter") == 1
    )


def test_fallback_context_batch(stack32k):
    """Pure context batch: no rope_kv_append entry engages."""
    st = stack32k
    md = st.metadata(0, st.ctx_len, nreq=0, ctx_tokens=CTX_REQ_TOKENS)
    assert md.num_contexts == 1 and md.num_generations == 0
    st.compare(md, lambda: st.run_seg(md, range(len(PATTERN)), [(CTX_REQ_TOKENS, 0)], seed=33))
    native, _, scatter_counts = st.engagement
    assert _no_rope_append_engaged(scatter_counts), scatter_counts
    assert native.get("native_scatter") == 1 and scatter_counts.get("native_scatter") == 1


def test_fallback_mixed_batch(stack32k):
    """Context + generation in one batch: no rope_kv_append entry engages."""
    st = stack32k
    md = st.metadata(6, st.ctx_len, nreq=1, ctx_tokens=CTX_REQ_TOKENS)
    assert md.num_contexts == 1 and md.num_generations == 1
    segs = [(CTX_REQ_TOKENS, 0), (6, st.ctx_len)]
    st.compare(md, lambda: st.run_seg(md, range(len(PATTERN)), segs, seed=34))
    native, _, scatter_counts = st.engagement
    assert _no_rope_append_engaged(scatter_counts), scatter_counts
    assert scatter_counts.get("native_scatter") == native.get("native_scatter") == 1
    assert scatter_counts.get("native_rope") == native.get("native_rope")


def test_fallback_ragged_verify(stack32k):
    """Ragged verification (token-major rows): rope and scatter native; bridge per-call exact."""
    st = stack32k
    md = st.metadata(6, st.ctx_len, nreq=2, ragged_lens=[6, 3])
    assert md.is_ragged_verify and md.token_major_gen_view() is not None
    c0, c1 = st.cached(st.ctx_len, 2)
    st.compare(md, lambda: st.run_seg(md, range(len(PATTERN)), [(6, c0), (3, c1)], seed=35))
    native, _, scatter_counts = st.engagement
    assert not scatter_counts.get("rope_append_deferred_scatter") and not scatter_counts.get(
        "rope_append_mla_rope_generation"
    )
    assert scatter_counts.get("native_scatter") == native.get("native_scatter") == 1
    assert scatter_counts.get("native_rope") == native.get("native_rope")


class _Over:
    """Attribute overlay over a real object (unset attributes forward to it)."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def test_eligibility_rejections(stack32k):
    """Host-side predicates reject every unsupported mode before device work.

    Starts from a real eligible GLM decode call and changes one condition at
    a time (NVFP4 / FP4 cache or attention, beam > 1, helix, ragged, strided
    latent, non-packed fused_q, residual-dim cache, missing KV manager).
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (
        rope_kv_append,
    )

    st = stack32k
    md = st.metadata(6, st.ctx_len)
    mla = st.layers[0]
    mqa = mla.mqa
    n = 6
    fused_q = torch.zeros(n, HEADS, KV_LORA + ROPE, dtype=torch.bfloat16, device=st.dev)
    q_pe = torch.zeros(n, HEADS, ROPE, dtype=torch.bfloat16, device=st.dev)
    latent = torch.zeros(n, KV_LORA + ROPE, dtype=torch.bfloat16, device=st.dev)
    quant_q = torch.zeros(n, HEADS, KV_LORA + ROPE, dtype=torch.uint8, device=st.dev)
    s1 = torch.zeros(2, dtype=torch.float32, device=st.dev)
    s2 = torch.zeros(1, dtype=torch.float32, device=st.dev)
    k_fp8, k_scale = st.indexer_k(mla, 6, st.ctx_len, seed=36)

    def rope(mqa_=mqa, md_=md, latent_=latent, fused_q_=fused_q):
        return rope_kv_append.rope_eligible(mqa_, md_, fused_q_, q_pe, latent_, quant_q, s1, s2)

    def defer(mla_=mla, md_=md):
        return rope_kv_append.defer_indexer_k_scatter(
            mla_, k_fp8, k_scale, md_, forwards_packet=True
        )

    set_features("rope_kv_append")
    try:
        assert rope() and rope_kv_append.bridge_eligible(md, True) and defer() is not None
        nvfp4 = tensorrt_llm.bindings.DataType.NVFP4
        cases = {
            "fp4_kv_cache": dict(mqa_=_Over(mqa, has_fp4_kv_cache=True)),
            "no_fp8_kv_cache": dict(mqa_=_Over(mqa, has_fp8_kv_cache=False)),
            "fp4_mla_attention": dict(mqa_=_Over(mqa, uses_fp4_mla_attention=True)),
            "beam2": dict(md_=_Over(md, beam_width=2)),
            "helix": dict(md_=_Over(md, helix_position_offsets=torch.zeros(1))),
            "ragged": dict(md_=_Over(md, is_ragged_verify=True)),
            "token_major": dict(md_=_Over(md, token_major_gen_view=lambda: object())),
            "context": dict(md_=_Over(md, num_contexts=1)),
            "no_kv_manager": dict(md_=_Over(md, kv_cache_manager=None)),
            "residual_dim_cache": dict(
                md_=_Over(
                    md, kv_cache_manager=_Over(md.kv_cache_manager, mla_kv_cache_residual_dim=64)
                )
            ),
        }
        for name, kw in cases.items():
            assert not rope(**kw), name
        strided = torch.zeros(n, 640, dtype=torch.bfloat16, device=st.dev)[:, : KV_LORA + ROPE]
        assert not rope(latent_=strided)
        assert not rope(fused_q_=fused_q.transpose(0, 1).contiguous().transpose(0, 1))
        for name in ("fp4_kv_cache", "beam2", "helix", "ragged", "token_major", "context"):
            kw = cases[name]
            assert (
                defer(
                    mla_=_Over(mla, mqa=kw["mqa_"]) if "mqa_" in kw else mla, md_=kw.get("md_", md)
                )
                is None
            ), name
        assert not rope_kv_append.bridge_eligible(
            _Over(md, kv_cache_manager=_Over(md.kv_cache_manager, dtype=nvfp4)), True
        )
        assert not rope_kv_append.bridge_eligible(_Over(md, num_contexts=1), True)
        assert not rope_kv_append.bridge_eligible(md, False)
    finally:
        set_features("")
