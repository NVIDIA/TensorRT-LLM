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
"""One DSA ``MLA`` module through the target's call sites, for composition tests of sparse_mla_decode.

Run as a script in a fresh interpreter: ``python sparse_mla_case.py <output.pt>`` with the fast
paths to run in ``GLM_MOE_DSA_TEST_FAST_PATHS`` (``plan_testlib.FAST_PATHS_ENV``). It builds
one GLM-shaped DSA MLA layer (8 local heads, TP1, FP8 latent cache, deterministic weights keyed
by parameter name), adopts it as the target's ``GlmMLA``,
runs a native context step and then MTP verify steps of 6 rows as decode steps through
``GlmMLA.forward``, so that every package at the MLA / DSA call sites
(absorbed_query/shared_indexer_absorbed_query
projections, rope_kv_append RoPE/append and top-k bridge, attention_output_tail, sparse_mla_decode
attention) runs as in the model. With ``GLM_MOE_DSA_SPARSE_MLA_DECODE_MLA_TOPK=fixed`` (default) the
indexer's generation selection is replaced by a given local top-k, so that every process attends to
the same rows; with ``GLM_MOE_DSA_SPARSE_MLA_DECODE_MLA_TOPK=indexer`` the indexer runs unmodified
(needed when rope_kv_append's top-k bridge is a producer).

Saved: per step the FMHA inputs (FP8 query bytes, bmm scales, global indices,
KV lengths), the latent attention output and the module output, for an eager
step, a step whose sparse_mla_decode launch is followed by a device synchronization (no
programmatic overlap with the next kernel), and replays of a captured CUDA
graph over alternating inputs (with the global indices each replay used);
plus the per-package capability decisions.
"""

import json
import math
import os
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plan_testlib  # noqa: E402
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E402, E501
    GlmMLA,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode.sparse_mla import (  # noqa: E402, E501
    SparseMlaDecodeFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha  # noqa: E402
from tensorrt_llm._torch.attention.backends.interface import (  # noqa: E402
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager  # noqa: E402
from tensorrt_llm._torch.attention.mla import MLA  # noqa: E402
from tensorrt_llm._torch.metadata import KVCacheParams  # noqa: E402
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention  # noqa: E402
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str  # noqa: E402
from tensorrt_llm.bindings.executor import KvCacheConfig  # noqa: E402
from tensorrt_llm.functional import PositionEmbeddingType  # noqa: E402
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

HIDDEN = 6144
HEADS = 8
Q_LORA = 2048
KV_LORA = 512
NOPE = 192
ROPE = 64
V_HEAD = 256
TOP_K = 2048
ROWS = 6
TOKENS_PER_BLOCK = 64


def _rope_config(max_positions):
    return SimpleNamespace(
        max_position_embeddings=max_positions,
        hidden_size=HIDDEN,
        num_attention_heads=64,
        qk_rope_head_dim=ROPE,
        rope_theta=8.0e6,
        rope_scaling=None,
        model_type="deepseek_v32",
    )


def _fill(param: torch.Tensor, name: str) -> None:
    """Deterministic values keyed by the parameter name (independent of creation order)."""
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(name.encode()))
    if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return
    if param.dim() == 1:
        values = 1.0 + 0.05 * torch.randn(param.shape, device="cuda", generator=gen)
    else:
        fan_in = param.shape[-1]
        values = torch.randn(param.shape, device="cuda", generator=gen) / math.sqrt(fan_in)
    with torch.no_grad():
        param.copy_(values.to(param.dtype))


def build(ctx_len: int):
    max_seq_len = ctx_len + 4 * ROWS
    max_positions = 1 << max(16, math.ceil(math.log2(max_seq_len + 1)))
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=32, index_head_dim=128, index_topk=TOP_K, skip_indexer_for_short_seqs=False
    )
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=SimpleNamespace(rms_norm_eps=1e-5),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value),
    )
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(_rope_config(max_positions)),
        is_neox=False,
    )
    mla = MLA(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        # GLM-5.2: num_key_value_heads == num_attention_heads (sizes o_proj's input).
        num_key_value_heads=HEADS,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        predicted_tokens_per_seq=1,
        max_position_embeddings=max_positions,
        bias=False,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=model_config,
        reduce_output=False,
        rms_norm_eps=1e-5,
    )
    # The model's DeepseekV32Attention adds only its constructor to MLA; the case builds the MLA
    # with GLM shapes directly and adopts it as the model's attention.
    mla.__class__ = DeepseekV32Attention
    GlmMLA.adopt(mla)
    # The model builds the FMHA libraries in ``create_weights`` after adoption; a module built
    # alone rebuilds them for the library adoption declared.
    for backend in (mla.mha, mla.mqa):
        if backend is not None and backend.prepended_fmha_lib_classes:
            backend.update_quant_config(backend.quant_config)
    mla = mla.cuda()
    # Eager module path (the custom-op path resolves layers through the model engine).
    mla.register_to_config = False
    indexer = mla.mqa.indexer
    indexer.cuda()
    for name, param in list(mla.named_parameters()) + [
        ("indexer." + n, p) for n, p in indexer.named_parameters()
    ]:
        _fill(param, name)
    with torch.no_grad():
        kv_b = mla.kv_b_proj.weight.view(HEADS, NOPE + V_HEAD, KV_LORA)
        mla.kv_b_proj.weight.copy_(
            torch.cat(
                [kv_b[:, :NOPE].reshape(-1, KV_LORA), kv_b[:, NOPE:].reshape(-1, KV_LORA)], dim=0
            )
        )
        k_nope = mla.kv_b_proj.weight[: HEADS * NOPE].view(HEADS, NOPE, KV_LORA)
        v = mla.kv_b_proj.weight[HEADS * NOPE :].view(HEADS, V_HEAD, KV_LORA)
        # In place, so that any packed view a package took at construction stays valid.
        mla.k_b_proj_trans.copy_(k_nope.transpose(1, 2))
        mla.v_b_proj.copy_(v)
    indexer.post_load_weights()
    mla.post_load_weights()

    blocks = (max_seq_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    cache = DSACacheManager(
        KvCacheConfig(max_tokens=blocks * TOKENS_PER_BLOCK, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA + ROPE,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    cache.add_dummy_requests([0], [ctx_len])
    return mla, cache, sparse_config, mapping


def metadata(mla, cache, sparse_config, mapping, ctx_len, cached, rows, generation):
    meta = type(mla.mqa).Metadata(
        seq_lens=torch.tensor([rows], dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=0 if generation else 1,
        prompt_lens=[ctx_len],
        max_num_tokens=max(rows, ROWS),
        kv_cache_manager=cache,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    if generation:
        meta.update_spec_dec_param(
            batch_size=1,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=ROWS - 1,
            max_total_draft_tokens=ROWS - 1,
        )
    meta.prepare()
    return meta


def local_topk(ctx_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk = torch.full((ROWS, TOP_K), -1, dtype=torch.int32)
    for row in range(ROWS):
        visible = ctx_len + row + 1
        recent = torch.arange(visible - 32, visible)
        rest = torch.randperm(visible - 32, generator=gen)[: TOP_K - 32]
        topk[row] = torch.cat([recent, rest]).to(torch.int32)
    return topk.cuda()


class Capture:
    """Records the FMHA inputs and the latent output of each attention call."""

    def __init__(self):
        self.calls = []
        self.record_inputs = True
        self.sync_after_sparse_decode = False
        self._originals = {}

    def install(self):
        for cls in (SparseMlaDecodeFmha, FallbackFmha):
            original = cls.forward
            self._originals[cls] = original

            def forward(fmha, q, k, v, md, args, _original=original, _cls=cls):
                entry = {"library": _cls.__name__, "output": args.output}
                if self.record_inputs:
                    sparse = args.sparse_runtime_params
                    entry.update(
                        quant_q=args.quant_q_buffer.clone(),
                        bmm1=args.mla_bmm1_scale.clone(),
                        bmm2=args.mla_bmm2_scale.clone(),
                        indices=sparse.sparse_attn_indices.clone(),
                        kv_lens=md.kv_lens_cuda_runtime[md.num_contexts :].clone(),
                    )
                _original(fmha, q, k, v, md, args)
                if self.sync_after_sparse_decode and _cls is SparseMlaDecodeFmha:
                    torch.cuda.synchronize()
                self.calls.append(entry)

            cls.forward = forward


def main(out_path: str) -> None:
    plan_testlib.select_from_env(dispatch)
    ctx_len = int(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_MLA_CTX", "32768"))
    replays = int(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_MLA_REPLAYS", "8"))
    topk_mode = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_MLA_TOPK", "fixed")
    assert topk_mode in ("fixed", "indexer"), topk_mode
    torch.manual_seed(0)
    mla, cache, sparse_config, mapping = build(ctx_len)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    hidden_ctx = torch.randn(ctx_len, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    hidden_gen = [
        torch.randn(ROWS, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16) for _ in range(2)
    ]
    topks = [local_topk(ctx_len, 77), local_topk(ctx_len, 78)]
    pool = cache.get_unique_primary_pool()

    ctx_meta = metadata(mla, cache, sparse_config, mapping, ctx_len, 0, ctx_len, False)
    with torch.no_grad():
        mla.forward(torch.arange(ctx_len, device="cuda")[None], hidden_ctx, ctx_meta)
    torch.cuda.synchronize()
    for _ in range(ROWS):
        cache.impl.add_token(0)
    gen_meta = metadata(mla, cache, sparse_config, mapping, ctx_len, ctx_len, ROWS, True)
    positions = torch.arange(ctx_len, ctx_len + ROWS, device="cuda")[None]

    capture = Capture()
    capture.install()
    static_hidden = hidden_gen[0].clone()
    static_topk = topks[0].clone()
    if topk_mode == "fixed":
        mla.mqa.indexer.forward_from_projected = Mock(return_value=static_topk)

    def step(index):
        static_hidden.copy_(hidden_gen[index])
        static_topk.copy_(topks[index])
        with torch.no_grad(), dispatch.step(dispatch.Phase.DECODE):
            return mla.forward(positions, static_hidden, gen_meta)

    result = {"eager": [], "serialized": [], "graph": []}
    # Eager steps (inputs recorded), each input set once.
    for index in (0, 1):
        out = step(index).clone()
        torch.cuda.synchronize()
        call = capture.calls[-1]
        result["eager"].append(
            {
                "input": index,
                "module_output": out.cpu(),
                "latent": call["output"].clone().cpu(),
                "library": call["library"],
                "quant_q": call["quant_q"].cpu(),
                "bmm1": call["bmm1"].cpu(),
                "bmm2": call["bmm2"].cpu(),
                "indices": call["indices"].cpu(),
                "kv_lens": call["kv_lens"].cpu(),
            }
        )
    # The attended pool rows of each input, for an FP64 oracle in the parent.
    pool_rows = pool.view(torch.uint8).reshape(-1, KV_LORA + ROPE)
    for entry in result["eager"]:
        idx = entry["indices"].long().clamp_min(0).cuda()
        entry["rows"] = pool_rows[idx.reshape(-1)].reshape(ROWS, TOP_K, -1).cpu()
    # Same inputs with the attention kernel completed before the next kernel is launched.
    capture.sync_after_sparse_decode = True
    for index in (0, 1):
        out = step(index).clone()
        torch.cuda.synchronize()
        result["serialized"].append(
            {
                "input": index,
                "module_output": out.cpu(),
                "latent": capture.calls[-1]["output"].clone().cpu(),
                "indices": capture.calls[-1]["indices"].cpu(),
            }
        )
    capture.sync_after_sparse_decode = False
    # A captured CUDA graph of one verify step, replayed over alternating inputs. The
    # recorded indices are clones made inside the graph, refreshed by every replay.
    step(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad(), dispatch.step(dispatch.Phase.DECODE):
            graph_out = mla.forward(positions, static_hidden, gen_meta)
    graph_call = capture.calls[-1]
    for it in range(replays):
        index = it % 2
        static_hidden.copy_(hidden_gen[index])
        static_topk.copy_(topks[index])
        graph.replay()
        torch.cuda.synchronize()
        result["graph"].append(
            {
                "input": index,
                "module_output": graph_out.clone().cpu(),
                "latent": graph_call["output"].clone().cpu(),
                "indices": graph_call["indices"].cpu(),
            }
        )
    result["capability"] = {
        f"{feature}/{key}": value for (feature, key), value in dispatch._CAPABILITY_LOGGED.items()
    }
    result["features"] = sorted(dispatch.enabled_features())
    result["topk_mode"] = topk_mode
    result["libraries"] = sorted({c["library"] for c in capture.calls})
    torch.save(result, out_path)
    print(
        json.dumps(
            {
                "features": result["features"],
                "topk_mode": topk_mode,
                "libraries": result["libraries"],
                "capability": result["capability"],
            }
        )
    )
    cache.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
