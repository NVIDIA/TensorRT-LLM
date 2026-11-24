# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the packed DSA projections: the packed path returns the unpacked path's outputs.

The operands are exact in BF16 GEMMs (small integers with sparse weights, so every partial sum is an integer far
below 2^8), which makes every GEMM kernel -- cuBLAS on the separate weights, cuBLAS on the packed weight, the
low-M CuTe DSL kernels -- produce identical columns. What is left to differ is the plumbing (row order of the
packed weight, the column split, the views, the indexer finishing, the gate hand-off), and that is checked bit
for bit against the general target's ``forward_dsa_proj``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    K31MLAAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_graph_ops import (
    forward_dsa_proj,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import (
    K31Indexer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_packed_proj import (
    PACKED_HIDDEN_WEIGHT,
    forward_dsa_proj_packed,
    pack_dsa_hidden_projection,
    packed_output_gate,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    k31_dsa_params,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    k31_low_m_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
    K31_LOW_M_PLANS,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

assert torch.cuda.is_available(), "the packed DSA projections require a CUDA device"

HIDDEN, Q_LORA, KV_LORA, ROPE = 7168, 1536, 512, 64
QK_HEAD = 128 + 64  # qk_nope + rope per MLA head
V_HEAD = 128


def _indexer() -> K31Indexer:
    sparse = DeepSeekSparseAttentionConfig(index_head_dim=128, index_n_heads=32, index_topk=2048)
    pretrained = SimpleNamespace(index_rope_dim=0)
    params = k31_dsa_params(sparse.to_sparse_params(pretrained_config=pretrained), pretrained)
    mla = SimpleNamespace(hidden_size=HIDDEN, q_lora_rank=Q_LORA, qk_rope_head_dim=ROPE)
    return K31Indexer(mla, params, torch.bfloat16).cuda()


def _exact(rows, cols, gen, density):
    """Small-integer BF16 values, mostly zero: GEMM sums stay exact integers in any order."""
    vals = torch.randint(-2, 3, (rows, cols), generator=gen, device="cuda").float()
    keep = torch.rand(rows, cols, generator=gen, device="cuda") < density
    return (vals * keep).to(torch.bfloat16)


class _FakeMLA(torch.nn.Module):
    """The attributes and BF16 methods of the target's MLA layer that ``forward_dsa_proj`` reads."""

    dsa_deferred_q_b = None

    def __init__(self, heads: int, gen):
        super().__init__()
        self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim = Q_LORA, KV_LORA, ROPE
        self.num_heads_tp, self.qk_head_dim = heads, QK_HEAD
        self.use_output_gate = True
        lin = lambda n, k: torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")  # noqa: E731
        self.kv_a_proj_with_mqa = lin(Q_LORA + KV_LORA + ROPE, HIDDEN)
        self.q_b_proj = lin(heads * QK_HEAD, Q_LORA)
        self.g_proj = lin(heads * V_HEAD, HIDDEN)
        self.q_a_layernorm = SimpleNamespace(
            weight=torch.ones(Q_LORA, dtype=torch.bfloat16, device="cuda"), variance_epsilon=1e-6
        )
        self.kv_a_layernorm = SimpleNamespace(
            weight=torch.ones(KV_LORA, dtype=torch.bfloat16, device="cuda"), variance_epsilon=1e-6
        )
        self.mqa = SimpleNamespace(indexer=_indexer())
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.aux_stream = None
        idx = self.mqa.indexer
        with torch.no_grad():
            for m in (self.kv_a_proj_with_mqa, self.g_proj, idx.wk, idx.weights_proj):
                m.weight.copy_(_exact(*m.weight.shape, gen, 0.002))
            for m in (self.q_b_proj, idx.wq_b):
                m.weight.copy_(_exact(*m.weight.shape, gen, 0.01))
        idx.post_load_weights()  # the fused [wk | weights_proj] weight

    def _project_qkv_a(self, hidden_states):
        return F.linear(hidden_states, self.kv_a_proj_with_mqa.weight)

    def _q_a_layernorm(self, q, return_norm_out=False):
        n = flashinfer_rmsnorm(q, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon)
        return (n, n) if return_norm_out else n

    def _kv_a_layernorm(self, compressed_kv):
        return flashinfer_rmsnorm(
            compressed_kv, self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon
        )


def _bits(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize(
    "layout,heads,rows", (("tep32", 3, 8), ("tep32", 3, 40), ("tep16", 6, 16), ("dep16", 96, 5))
)
@pytest.mark.parametrize("planned", (False, True))
def test_packed_matches_unpacked(layout, heads, rows, planned) -> None:
    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + heads)
    attn = _FakeMLA(heads, gen)
    x = _exact(rows, HIDDEN, gen, 0.05)
    with torch.no_grad():
        ref = forward_dsa_proj(attn, None, x, None)
        ref_gate = F.linear(x, attn.g_proj.weight)
        weights_before = {n: p.detach().clone() for n, p in attn.named_parameters()}
        assert pack_dsa_hidden_projection(attn, pack_gate=True)
        packed = getattr(attn, PACKED_HIDDEN_WEIGHT)
        assert packed.shape == (Q_LORA + KV_LORA + ROPE + 128 + 32 + heads * V_HEAD, HIDDEN)
        # The module weights are row views of the packed weight, with their values unchanged.
        for n, p in attn.named_parameters():
            assert torch.equal(p, weights_before[n]), n
        assert attn.kv_a_proj_with_mqa.weight.data_ptr() == packed.data_ptr()
        plan = K31_LOW_M_PLANS[layout] if planned else None
        out = forward_dsa_proj_packed(attn, None, x, None, plan, True)
        gate = packed_output_gate(attn, x)
    assert len(out) == len(ref) == 8
    for i, (o, r) in enumerate(zip(out, ref)):
        assert o.shape == r.shape and o.dtype == r.dtype, i
        assert torch.equal(_bits(o), _bits(r)), f"output {i} differs"
    assert gate is not None and torch.equal(gate, ref_gate)
    assert packed_output_gate(attn, x) is None  # consumed


def test_gate_is_not_handed_to_another_input() -> None:
    gen = torch.Generator(device="cuda").manual_seed(7)
    attn = _FakeMLA(3, gen)
    x = _exact(8, HIDDEN, gen, 0.05)
    with torch.no_grad():
        assert pack_dsa_hidden_projection(attn, pack_gate=True)
        forward_dsa_proj_packed(attn, None, x, None, None, True)
        assert packed_output_gate(attn, x.clone()) is None


def test_no_gate_rows_without_pack_gate() -> None:
    gen = torch.Generator(device="cuda").manual_seed(9)
    attn = _FakeMLA(3, gen)
    x = _exact(8, HIDDEN, gen, 0.05)
    with torch.no_grad():
        ref = forward_dsa_proj(attn, None, x, None)
        assert pack_dsa_hidden_projection(attn, pack_gate=False)
        assert getattr(attn, PACKED_HIDDEN_WEIGHT).shape[0] == Q_LORA + KV_LORA + ROPE + 128 + 32
        out = forward_dsa_proj_packed(attn, None, x, None, K31_LOW_M_PLANS["tep32"], True)
        assert packed_output_gate(attn, x) is None
    for o, r in zip(out, ref):
        assert torch.equal(_bits(o), _bits(r))


# ---------------------------------------------------------------------------------------------
# The perf targets' deferred index quantization: BF16 index projections in Op 1, the fused FP8
# prologue in Op 2 (dsa_graph_ops._fused_index_prologue).
# ---------------------------------------------------------------------------------------------

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E402
    dsa_graph_ops,
)

_PAGE = 64
_KEY_BYTES = 128 + 4


def _cache_and_slots(tokens: int, gen):
    """A random-byte [pages, 64, 1, 132] index-key cache view (padded page stride) and distinct slot
    mappings of ``tokens`` tokens, as the DSA metadata computes them."""
    pages = tokens // _PAGE + 8
    page_bytes = _PAGE * _KEY_BYTES
    base = torch.randint(0, 256, (pages, page_bytes + 512), generator=gen, device="cuda").to(
        torch.uint8
    )
    view = base[:, :page_bytes].view(pages, _PAGE, 1, _KEY_BYTES)
    slots = torch.randperm(pages * _PAGE, generator=gen, device="cuda")[:tokens]
    block = (slots // _PAGE) * page_bytes
    fp8 = (block + (slots % _PAGE) * 128).to(torch.int64)
    scale = (block + _PAGE * 128 + (slots % _PAGE) * 4).to(torch.int64)
    return base, view, fp8, scale


class _Manager:
    def __init__(self, view):
        self.view = view

    def get_indexer_k_cache_buffers(self, layer_idx):
        return self.view


@pytest.mark.parametrize("heads,rows,tokens", ((3, 8, 8), (6, 40, 27), (96, 5, 5), (6, 300, 300)))
def test_deferred_index_quant_matches_op1_quant_and_cache_update(heads, rows, tokens) -> None:
    """Op 1 BF16 projections + Op 2 fused prologue == Op 1 FP8 quantization + Op 2 _update_k_cache:
    every indexer intermediate, the latent rows and every cache byte, bit for bit (``rows`` padded rows,
    ``tokens`` of them the step's)."""
    gen = torch.Generator(device="cuda").manual_seed(rows * 7 + heads)
    attn = _FakeMLA(heads, gen)
    x = _exact(rows, HIDDEN, gen, 0.05)
    indexer = attn.mqa.indexer
    indexer.layer_idx = 0
    base_a, view_a, slot_fp8, slot_scale = _cache_and_slots(tokens, gen)
    base_b = base_a.clone()
    view_b = base_b[:, : view_a.shape[1] * _KEY_BYTES].view(view_a.shape)
    with torch.no_grad():
        assert pack_dsa_hidden_projection(attn, pack_gate=True)
        ref = forward_dsa_proj_packed(attn, None, x, None, None, True)
        deferred = forward_dsa_proj_packed(attn, None, x, None, None, True, defer_index_quant=True)
        assert len(deferred) == 6
        # A custom op's returns may not alias each other: one tensor per storage.
        storages = [t.untyped_storage().data_ptr() for t in deferred]
        assert len(set(storages)) == len(storages), "two Op 1 returns share a storage"
        kpe_tail = deferred[2]
        k_pe = kpe_tail[:, :ROPE]
        index_weights = kpe_tail[:, ROPE + 128 : ROPE + 160]
        for i, o in enumerate((deferred[0], deferred[1], k_pe)):
            assert torch.equal(_bits(o), _bits(ref[i])), f"output {i} differs"
        md_a = SimpleNamespace(
            kv_cache_manager=_Manager(view_a),
            slot_mapping_fp8=slot_fp8,
            slot_mapping_scale=slot_scale,
        )
        q_fp8, k_fp8, k_scale, weights = (t[:tokens] for t in ref[4:])
        indexer._update_k_cache(k_fp8, k_scale, md_a)
        md_b = SimpleNamespace(
            kv_cache_manager=_Manager(view_b),
            slot_mapping_fp8=slot_fp8,
            slot_mapping_scale=slot_scale,
        )
        got, latent = dsa_graph_ops._fused_index_prologue(
            attn,
            deferred[4],
            deferred[5],
            index_weights,
            deferred[1][:tokens],
            k_pe[:tokens],
            md_b,
            tokens,
        )
    torch.cuda.synchronize()
    for i, (o, r) in enumerate(zip(got, [q_fp8, k_fp8, k_scale, weights])):
        assert o.shape == r.shape and o.dtype == r.dtype, i
        assert torch.equal(_bits(o), _bits(r)), f"indexer intermediate {i} differs"
    assert torch.equal(_bits(latent), _bits(ref[3][:tokens]))
    assert torch.equal(base_a, base_b), "index-key cache bytes differ"


def test_deferred_index_quant_without_cache_update() -> None:
    """No slot mappings this step: the ops chain runs, the cache is untouched, the bits match."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    attn = _FakeMLA(3, gen)
    x = _exact(8, HIDDEN, gen, 0.05)
    attn.mqa.indexer.layer_idx = 0
    with torch.no_grad():
        assert pack_dsa_hidden_projection(attn, pack_gate=True)
        ref = forward_dsa_proj_packed(attn, None, x, None, None, True)
        deferred = forward_dsa_proj_packed(attn, None, x, None, None, True, defer_index_quant=True)
        md = SimpleNamespace(kv_cache_manager=None, slot_mapping_fp8=None, slot_mapping_scale=None)
        tail = deferred[2]
        got, latent = dsa_graph_ops._fused_index_prologue(
            attn,
            deferred[4],
            deferred[5],
            tail[:, ROPE + 128 : ROPE + 160],
            deferred[1],
            tail[:, :ROPE],
            md,
            8,
        )
    for i, (o, r) in enumerate(zip(got, ref[4:])):
        assert torch.equal(_bits(o), _bits(r)), f"indexer intermediate {i} differs"
    assert torch.equal(_bits(latent), _bits(ref[3]))


def test_fused_index_prologue_gate() -> None:
    """Only a packed layer of the perf backend (mqa._k31_perf) with a full indexer defers the quantization."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    attn = _FakeMLA(3, gen)
    assert not dsa_graph_ops._fused_index_prologue_layer(attn)  # not packed, not perf
    with torch.no_grad():
        assert pack_dsa_hidden_projection(attn, pack_gate=True)
    assert not dsa_graph_ops._fused_index_prologue_layer(attn)  # packed, general backend
    attn.mqa._k31_perf = True
    assert dsa_graph_ops._fused_index_prologue_layer(attn)
    indexer, attn.mqa.indexer = attn.mqa.indexer, None
    assert not dsa_graph_ops._fused_index_prologue_layer(attn)  # shared-indexer layer
    attn.mqa.indexer = indexer


@pytest.mark.parametrize("layout,heads,rows", (("tep32", 3, 8), ("tep16", 6, 14), ("tep16", 6, 25)))
@pytest.mark.parametrize("defer_index_quant", (False, True))
def test_deferred_q_b_is_the_same_gemm_later(layout, heads, rows, defer_index_quant) -> None:
    """With ``defer_q_b`` Op 1 returns the normed q_a in the q slot; the q_b projection Op 2 runs on it (the same
    planned GEMM) is bitwise Op 1's own q output, and every other output is unchanged."""
    gen = torch.Generator(device="cuda").manual_seed(rows * 13 + heads)
    attn = _FakeMLA(heads, gen)
    x = _exact(rows, HIDDEN, gen, 0.05)
    plan = K31_LOW_M_PLANS[layout]
    with torch.no_grad():
        assert pack_dsa_hidden_projection(attn, pack_gate=True)
        ref = forward_dsa_proj_packed(
            attn, None, x, None, plan, True, defer_index_quant=defer_index_quant
        )
        out = forward_dsa_proj_packed(
            attn, None, x, None, plan, True, defer_index_quant=defer_index_quant, defer_q_b=True
        )
        assert out[0].shape == (rows, Q_LORA)
        late = k31_low_m_linear(out[0], attn.q_b_proj.weight, plan, True)
    assert torch.equal(_bits(late), _bits(ref[0]))
    for i, (o, r) in enumerate(zip(out[1:], ref[1:]), start=1):
        if defer_index_quant and i == 3:
            continue  # the uninitialized latent rows the fused prologue writes
        assert torch.equal(_bits(o), _bits(r)), f"output {i} differs"


def test_predicted_topk_skips_the_indexer() -> None:
    """A Top-K the caller predicted ahead of the attention call (``_k31_predicted_topk``) is what that call's
    ``sparse_attn_predict`` returns, without running the indexer or the remap."""

    class _Unreachable:
        def __getattr__(self, name):
            raise AssertionError(f"the indexer ran ({name})")

    predicted = torch.arange(12, dtype=torch.int32, device="cuda").view(3, 4)
    backend = SimpleNamespace(_k31_predicted_topk=predicted, indexer=_Unreachable())
    indices = K31MLAAttention.sparse_attn_predict(backend, None, _Unreachable(), True, None)
    assert indices is predicted


# ---------------------------------------------------------------------------------------------
# Op 1's fake implementation (piecewise CUDA graph tracing) against its real outputs.
# ---------------------------------------------------------------------------------------------

import weakref  # noqa: E402

from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata  # noqa: E402
from tensorrt_llm._torch.utils import model_extra_attrs  # noqa: E402


class _NoMetadata(TrtllmAttentionMetadata):
    """Op 1 reads no step metadata; the layer registry only checks the type."""

    def __init__(self):
        pass


@pytest.mark.parametrize(
    "variant",
    ("unpacked", "shared_indexer", "packed", "packed_perf", "deferred_q_b", "deferred_q_b_perf"),
)
@pytest.mark.parametrize("rows", (5, 40))
def test_op1_fake_matches_real(variant, rows) -> None:
    """``trtllm::k31_mla_dsa_proj``'s fake returns the real op's output list: the count (4 attention inputs and 0,
    2 or 4 indexer intermediates), and each output's shape, dtype and device."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    attn = _FakeMLA(3, gen)
    x = _exact(rows, HIDDEN, gen, 0.05)
    with torch.no_grad():
        if variant == "shared_indexer":
            attn.mqa.indexer = None
        if variant.startswith(("packed", "deferred")):
            assert pack_dsa_hidden_projection(attn, pack_gate=True)
        if variant.endswith("perf"):
            attn.mqa._k31_perf = True
        if variant.startswith("deferred"):
            attn.dsa_deferred_q_b = dsa_graph_ops.DeferredQB(
                rows, lambda q: q, lambda q, attn_metadata: False
            )
    metadata = _NoMetadata()
    attrs = {"attention_metadata": weakref.ref(metadata), "mla_layers": {"3": weakref.ref(attn)}}
    with model_extra_attrs(attrs), torch.no_grad():
        real = torch.ops.trtllm.k31_mla_dsa_proj(x, None, "3")
        with FakeTensorMode():
            fake = torch.ops.trtllm.k31_mla_dsa_proj(
                torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda"), None, "3"
            )
    expected_len = {"shared_indexer": 4, "packed_perf": 6, "deferred_q_b_perf": 6}.get(variant, 8)
    assert len(real) == len(fake) == expected_len
    for i, (r, f) in enumerate(zip(real, fake)):
        assert (f.shape, f.dtype, f.device) == (r.shape, r.dtype, r.device), f"output {i}"
    q_width = Q_LORA if variant.startswith("deferred") else 3 * QK_HEAD
    assert real[0].shape == (rows, q_width)
