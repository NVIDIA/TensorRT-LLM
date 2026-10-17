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
"""``AttnTail`` protocol on a real DSA ``MLA`` module with synthetic latents.

Builds the GLM-5.2 attention module of TP8 rank 0 (DSA sparse attention, FP8
KV cache, BF16 projections) with real ``v_b_proj``/``o_proj`` weights and
drives the three ``AttnTail`` entry points directly (``arm``, ``defer_w_uv``,
``project``) with synthetic latents (``tail_fixtures.mix_latent``) and
``SimpleNamespace`` metadata. No attention kernel runs here; the real
``MLA.forward`` path with the native attention producer is covered by
``test_attn_tail_mla_forward.py`` and ``test_attn_tail_mla_forward_mpi.py``.

* the static capability check accepts the module;
* ``arm`` + ``defer_w_uv`` + ``project`` equal the native GEN tail
  (``MLA._bmm_bf16_out`` into ``attn_output[0]`` followed by the native
  ``MLA._project_output``) within the frozen attention_output_tail budget, for every row count;
* ``defer_w_uv`` acts only on an armed call; ``arm`` declines (and touches no
  state) under torch.compile tracing, TensorRT-LLM's compile policy and
  breakable CUDA graphs; unsupported calls (context present, more than
  ``MAX_ENGAGED_ROWS`` rows) are declined before any GPU work; a mismatched or
  leftover packet raises; draft (MTP) layers get no tail;
* the tail can be captured in a CUDA graph and replayed.
"""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E501
    AttnTail,
    make_attn_tail,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

_CKPT = tail_fixtures.checkpoint_dir()

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="attention_output_tail targets SM100/SM103",
    ),
    pytest.mark.skipif(not (_CKPT and os.path.isdir(_CKPT)), reason="checkpoint not available"),
]

LAYER = 3


def _rope_config():
    return SimpleNamespace(
        hidden_size=6144,
        num_attention_heads=64,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        max_position_embeddings=4096,
        rope_theta=10000.0,
        qk_rope_head_dim=64,
        model_type="deepseek_v2",
    )


def _build_mla(layer_idx=LAYER, num_hidden_layers=None):
    from tensorrt_llm._torch.attention.mla import MLA
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant = QuantConfig()
    quant.kv_cache_quant_algo = QuantAlgo.FP8
    pretrained = SimpleNamespace(rms_norm_eps=1e-5)
    if num_hidden_layers is not None:
        pretrained.num_hidden_layers = num_hidden_layers
    model_config = ModelConfig(
        mapping=Mapping(world_size=8, tp_size=8, rank=0),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=32, index_head_dim=128, index_topk=2048
        ),
        pretrained_config=pretrained,
        quant_config=quant,
    )
    model_config.extra_attrs["kv_cache_dtype"] = "fp8"
    # One process stands in for TP8 rank 0: the COLUMN Linears would build
    # all-reduce workspaces over 8 local GPUs, which this test never uses.
    import tensorrt_llm._torch.distributed as distributed

    saved_all_reduce = distributed.AllReduce
    distributed.AllReduce = lambda *args, **kwargs: None
    try:
        mla = _construct(MLA, model_config, layer_idx)
    finally:
        distributed.AllReduce = saved_all_reduce
    w_uv, w_o = tail_fixtures.load_tail_weights(_CKPT, LAYER, [0])[0]
    with torch.no_grad():
        mla.v_b_proj.data = w_uv.cuda().contiguous()
        mla.o_proj.weight.data.copy_(w_o.cuda())
    return mla


def _build_adopted_mla(layer_idx=LAYER, num_hidden_layers=None):
    """``_build_mla``, adopted as the target's ``GlmMLA`` (which attaches the attention tail)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
        GlmMLA,
    )
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention

    mla = _build_mla(layer_idx=layer_idx, num_hidden_layers=num_hidden_layers)
    # The model's DeepseekV32Attention adds only its constructor to MLA.
    mla.__class__ = DeepseekV32Attention
    return GlmMLA.adopt(mla)


def _construct(MLA, model_config, layer_idx):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm.functional import PositionEmbeddingType

    return MLA(
        hidden_size=6144,
        num_attention_heads=64,
        num_key_value_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        predicted_tokens_per_seq=6,
        max_position_embeddings=4096,
        bias=False,
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams.from_config(_rope_config()),
            is_neox=False,
        ),
        layer_idx=layer_idx,
        dtype=torch.bfloat16,
        config=model_config,
        reduce_output=False,
    ).cuda()


_MLA = {}


def _mla():
    if "m" not in _MLA:
        _MLA["m"] = _build_mla()
    return _MLA["m"]


def _native(mla, latent, out_rows):
    from tensorrt_llm._torch.attention.mla import MLA
    from tensorrt_llm._torch.distributed import AllReduceParams

    rows = latent.shape[0]
    attn_out = torch.zeros(
        (out_rows, mla.num_heads_tp * mla.v_head_dim), dtype=torch.bfloat16, device="cuda"
    )
    v_view = attn_out[:rows].view(rows, mla.num_heads_tp, mla.v_head_dim)
    MLA._bmm_bf16_out(
        mla,
        latent.transpose(0, 1),
        mla.v_b_proj,
        mla.v_b_proj.transpose(1, 2),
        v_view.transpose(0, 1),
    )
    meta = SimpleNamespace(num_contexts=0)
    return mla._project_output([attn_out], None, meta, AllReduceParams(enable_allreduce=False))


def _meta(rows, num_contexts=0):
    return SimpleNamespace(num_contexts=num_contexts, num_tokens=rows)


def _latent(rows, seed):
    if "comp" not in _MLA:
        _MLA["comp"] = tail_fixtures.compressed_kv_rows(_CKPT, LAYER)
    return tail_fixtures.mix_latent(_MLA["comp"], rows, 8, seed=seed)


def test_static_capability_accepts_dsa_mla():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E501
        _static_reason,
    )

    mla = _mla()
    assert mla.sparse_attn_hooks is not None
    assert _static_reason(mla) is None


@pytest.mark.parametrize("rows,out_rows", [(1, 1), (2, 2), (4, 4), (6, 6), (6, 8), (6, 20), (8, 8)])
def test_defer_and_project_match_native(rows, out_rows):
    from tensorrt_llm._torch.distributed import AllReduceParams

    mla = _mla()
    latent = _latent(rows, seed=rows)
    native = _native(mla, latent, out_rows)
    tail = make_attn_tail(mla)
    attn_out = torch.empty((out_rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = _meta(rows)
    assert tail.arm(mla, meta)
    assert tail.defer_w_uv(mla, meta, latent, attn_out[:rows])
    attention_output_tail = tail.project(mla, attn_out, AllReduceParams(enable_allreduce=False))
    assert tail.project(mla, attn_out, None) is None  # packet consumed
    ref, _ = tail_reference.oracle_attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, rows)
    e_attn_tail = tail_reference.error_stats(attention_output_tail[:rows], ref)
    e_nat = tail_reference.error_stats(native[:rows], ref)
    assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6
    torch.testing.assert_close(
        attention_output_tail[:rows], native[:rows], rtol=2**-6, atol=2**-6 * e_nat["ref_rms"]
    )
    assert bool((attention_output_tail[rows:] == 0).all())
    assert tail.engaged_calls == 1 and tail.declined_calls == 0


def test_declines_before_any_work():
    mla = _mla()
    tail = make_attn_tail(mla)
    lat = _latent(6, seed=0)
    out = torch.empty((6, 2048), dtype=torch.bfloat16, device="cuda")
    # Not armed: the op-side entry never engages on its own.
    assert not tail.defer_w_uv(mla, _meta(6), lat, out)
    assert not tail.arm(mla, _meta(6, num_contexts=1))
    assert not tail.defer_w_uv(mla, _meta(6, num_contexts=1), lat, out)
    assert not tail.arm(mla, _meta(9))
    # Armed, but the latent the op hands over does not fit the call.
    assert tail.arm(mla, _meta(6))
    big = torch.zeros((9, 8, 512), dtype=torch.bfloat16, device="cuda")
    big_out = torch.empty((9, 2048), dtype=torch.bfloat16, device="cuda")
    assert not tail.defer_w_uv(mla, _meta(6), big, big_out)
    assert tail.declined_calls == 3 and tail.engaged_calls == 0
    assert not tail._armed
    assert tail.project(mla, out, None) is None


def test_arm_declines_under_compile_and_breakable_graphs():
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import enable_breakable_cuda_graph
    from tensorrt_llm._torch.utils import torch_compiling

    mla = _mla()
    tail = make_attn_tail(mla)
    with torch_compiling(True):
        assert not tail.arm(mla, _meta(6))
    with enable_breakable_cuda_graph():
        assert not tail.arm(mla, _meta(6))
    assert tail.declined_calls == 1  # the compile check touches no state

    meta = _meta(6)

    def traced(x):
        return x + 1 if tail.arm(mla, meta) else x - 1

    x = torch.zeros(4, device="cuda")
    out = torch.compile(traced, backend="eager", fullgraph=True)(x)
    assert torch.equal(out, x - 1)
    assert not tail._armed and tail.declined_calls == 1


def test_packet_errors_are_loud():
    mla = _mla()
    tail = make_attn_tail(mla)
    meta = _meta(6)
    lat = _latent(6, seed=1)
    out = torch.empty((6, 2048), dtype=torch.bfloat16, device="cuda")
    other = torch.empty((6, 2048), dtype=torch.bfloat16, device="cuda")
    assert tail.arm(mla, meta)
    assert tail.defer_w_uv(mla, meta, lat, out)
    with pytest.raises(RuntimeError, match="not consumed"):
        tail.arm(mla, meta)
    with pytest.raises(RuntimeError, match="does not match"):
        tail.project(mla, other, None)


def test_draft_layers_keep_native_tail():
    mla = _build_mla(layer_idx=78, num_hidden_layers=78)
    assert make_attn_tail(mla) is None


def test_graph_capture_and_replay():
    from tensorrt_llm._torch.distributed import AllReduceParams

    mla = _mla()
    tail = make_attn_tail(mla)
    rows = 6
    meta = _meta(rows)
    lat = _latent(rows, seed=2).contiguous()
    out = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    params = AllReduceParams(enable_allreduce=False)
    # Eager warm-up (JIT, per-layer state) before capture, like the engine.
    assert tail.arm(mla, meta)
    assert tail.defer_w_uv(mla, meta, lat, out)
    tail.project(mla, out, params)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert tail.arm(mla, meta)
        assert tail.defer_w_uv(mla, meta, lat, out)
        captured = tail.project(mla, out, params)
    for seed in range(8):
        lat.copy_(_latent(rows, seed=100 + seed))
        graph.replay()
        native = _native(mla, lat, rows)
        torch.testing.assert_close(
            captured.clone(),
            native,
            rtol=2**-6,
            atol=2**-6 * native.float().pow(2).mean().sqrt().item(),
        )
    assert not math.isnan(captured.float().sum().item())


@pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_ADAPTER_PATCH_APPLIED") != "1",
    reason="opt-in",
)
def test_adoption_attaches_tail_to_target_layers(monkeypatch):
    """With attention_output_tail running, the adopted attention (``GlmMLA``) owns an attention_output_tail.

    The call sites themselves run in ``test_attn_tail_mla_forward.py``.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
        dispatch,
    )

    plan_testlib.select(dispatch, "attention_output_tail", monkeypatch)
    mla = _build_adopted_mla(layer_idx=LAYER, num_hidden_layers=78)
    assert isinstance(mla._attn_tail, AttnTail)
    draft = _build_adopted_mla(layer_idx=78, num_hidden_layers=78)
    assert draft._attn_tail is None
