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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_attention import (
    _build_full_compressed_kv,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_kernels import (
    deepseek_v4_local_window_topk_idxs,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import InsertCachedAttention
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REGISTRY_DIR = _REPO_ROOT / "examples" / "auto_deploy" / "model_registry"
_CONFIGS_DIR = _REGISTRY_DIR / "configs"


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _small_config(**overrides) -> DeepseekV4Config:
    values = dict(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=4,
        compress_ratios=(4, 128),
        compress_rope_theta=10000.0,
        moe_intermediate_size=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.25,
        swiglu_limit=0.0,
        max_position_embeddings=256,
        ad_rope_cache_len=256,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 0,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        rms_norm_eps=1e-6,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
    )
    values.update(overrides)
    return DeepseekV4Config(**values)


def _batch_info(num_prefill: int, num_prefill_tokens: int, num_decode: int) -> torch.Tensor:
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo

    batch_info = BatchInfo()
    batch_info.update([num_prefill, num_prefill_tokens, 0, 0, num_decode, num_decode])
    batch_info.update_tokens_gather_info(num_prefill_tokens + num_decode, False)
    return batch_info.serialize()


def _paged_cache_metadata(
    seq_lens: list[int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_loc = []
    cu_num_pages = [0]
    next_page = 0
    for seq_len in seq_lens:
        num_pages = (seq_len + block_size - 1) // block_size
        cache_loc.extend(range(next_page, next_page + num_pages))
        next_page += num_pages
        cu_num_pages.append(len(cache_loc))
    return torch.tensor(cache_loc, dtype=torch.int), torch.tensor(cu_num_pages, dtype=torch.int)


def _freqs_cis_table(max_position: int, rope_dim: int) -> torch.Tensor:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    phases = torch.arange(max_position, dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.polar(torch.ones_like(phases), phases)


def _compressed_topk_idxs(
    ratio: int,
    batch_size: int,
    seq_len: int,
    offset: int,
    max_compressed_len: int,
) -> torch.Tensor:
    compressed = torch.arange(max_compressed_len)
    matrix = compressed.unsqueeze(0).expand(seq_len, -1)
    valid_lengths = torch.arange(1, seq_len + 1).unsqueeze(1) // ratio
    matrix = torch.where(matrix < valid_lengths, matrix + offset, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int32)


def _compressed_source_topk(
    ratio: int,
    batch_size: int,
    seq_len: int,
    window_size: int,
    max_compressed_len: int,
) -> torch.Tensor:
    local = deepseek_v4_local_window_topk_idxs(
        window_size, batch_size, seq_len, torch.device("cpu")
    )
    compressed = _compressed_topk_idxs(ratio, batch_size, seq_len, seq_len, max_compressed_len)
    return torch.cat([local.to(torch.int32), compressed], dim=-1)


def _compressed_caches(
    total_len: int,
    block_size: int,
    head_dim: int,
    compressor_state_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    cache_loc, cu_num_pages = _paged_cache_metadata([total_len], block_size)
    num_pages = len(cache_loc)
    caches = (
        torch.zeros(num_pages, block_size, 1, 1, head_dim),
        torch.zeros(num_pages, block_size, 1, 1, head_dim),
        torch.zeros(num_pages, block_size, 1, 1, compressor_state_dim),
        torch.zeros(num_pages, block_size, 1, 1, compressor_state_dim),
    )
    return cache_loc, cu_num_pages, caches


def _run_sparse_attention_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    ratio: int,
    max_compressed_len: int,
    window_size: int,
    rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        softmax_scale,
        window_size=window_size,
        compress_ratio=ratio,
        max_compressed_len=max_compressed_len,
        rope_dim=rope_dim,
        rms_norm_eps=1e-6,
    )


def _run_cached_sparse_attention_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_pages: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ratio: int,
    max_compressed_len: int,
    window_size: int,
    rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        batch_info,
        seq_len,
        input_pos,
        cu_seqlen,
        cache_loc,
        cu_num_pages,
        *caches,
        softmax_scale,
        window_size,
        ratio,
        max_compressed_len,
        1e-6,
        rope_dim,
    )


def _prefill_then_decode_fixture(ratio: int, prefix_len: int, total_len: int) -> dict:
    torch.manual_seed(500 + ratio)
    batch_size = 1
    num_heads = 2
    head_dim = 6
    rope_dim = 2
    window_size = 3
    block_size = 4
    softmax_scale = 0.375
    channels = 2 if ratio == 4 else 1
    max_compressed_len = (total_len + ratio - 1) // ratio

    q = torch.randn(batch_size, total_len, num_heads, head_dim)
    kv = torch.randn(batch_size, total_len, head_dim)
    compressor_kv = torch.randn(batch_size, total_len, channels * head_dim)
    compressor_gate = torch.randn(batch_size, total_len, channels * head_dim)
    compressor_ape = torch.randn(ratio, channels * head_dim)
    compressor_norm_weight = torch.randn(head_dim)
    freqs_cis_table = _freqs_cis_table(total_len + ratio, rope_dim)
    position_ids = torch.arange(total_len).view(1, -1)
    attn_sink = torch.tensor([-0.2, 0.35])

    full_topk = _compressed_source_topk(
        ratio, batch_size, total_len, window_size, max_compressed_len
    )
    full = _run_sparse_attention_v2(
        q,
        kv,
        attn_sink,
        full_topk,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        ratio,
        max_compressed_len,
        window_size,
        rope_dim,
        softmax_scale,
    )

    cache_loc, cu_num_pages, caches = _compressed_caches(
        total_len, block_size, head_dim, channels * head_dim
    )
    prefix_topk = _compressed_source_topk(
        ratio, batch_size, prefix_len, window_size, max_compressed_len
    )
    prefix = _run_cached_sparse_attention_v2(
        q[:, :prefix_len],
        kv[:, :prefix_len],
        attn_sink,
        prefix_topk,
        compressor_kv[:, :prefix_len],
        compressor_gate[:, :prefix_len],
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids[:, :prefix_len],
        _batch_info(num_prefill=1, num_prefill_tokens=prefix_len, num_decode=0),
        torch.tensor([prefix_len], dtype=torch.int),
        torch.tensor([0], dtype=torch.int),
        torch.tensor([0, prefix_len], dtype=torch.int),
        cache_loc,
        cu_num_pages,
        caches,
        ratio,
        max_compressed_len,
        window_size,
        rope_dim,
        softmax_scale,
    )

    decode_outputs = []
    dummy_topk = torch.zeros(batch_size, 1, window_size, dtype=torch.int32)
    for pos in range(prefix_len, total_len):
        decode_outputs.append(
            _run_cached_sparse_attention_v2(
                q[:, pos : pos + 1],
                kv[:, pos : pos + 1],
                attn_sink,
                dummy_topk,
                compressor_kv[:, pos : pos + 1],
                compressor_gate[:, pos : pos + 1],
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                position_ids[:, pos : pos + 1],
                _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=1),
                torch.tensor([1], dtype=torch.int),
                torch.tensor([pos], dtype=torch.int),
                torch.tensor([0, 1], dtype=torch.int),
                cache_loc,
                cu_num_pages,
                caches,
                ratio,
                max_compressed_len,
                window_size,
                rope_dim,
                softmax_scale,
            )
        )

    return {
        "full": full,
        "prefix": prefix,
        "decode": torch.cat(decode_outputs, dim=1),
        "caches": caches,
        "compressor_kv": compressor_kv,
        "compressor_gate": compressor_gate,
        "compressor_ape": compressor_ape,
        "compressor_norm_weight": compressor_norm_weight,
        "freqs_cis_table": freqs_cis_table,
        "position_ids": position_ids,
        "prefix_len": prefix_len,
        "ratio": ratio,
        "max_compressed_len": max_compressed_len,
        "rope_dim": rope_dim,
    }


def test_registry_config_selects_deepseek_v4_sparse_cached_attention() -> None:
    registry = _load_yaml(_REGISTRY_DIR / "models.yaml")
    dsv4_entry = next(
        entry for entry in registry["models"] if entry["name"] == "deepseek-ai/DeepSeek-V4-Flash"
    )
    config = _load_yaml(_CONFIGS_DIR / "deepseek_v4_flash.yaml")

    assert dsv4_entry["yaml_extra"] == [
        "dashboard_default.yaml",
        "world_size_8.yaml",
        "deepseek_v4_flash.yaml",
    ]
    assert config["model_factory"] == "DeepseekV4AutoModelForCausalLM"
    assert config["model_kwargs"]["skip_mtp"] is True
    assert config["transforms"]["insert_cached_attention"]["backend"] == "deepseek_v4_sparse"


@pytest.mark.parametrize(
    "ratio,prefix_len,total_len",
    [
        pytest.param(4, 3, 9, id="ratio4-overlap-decode"),
        pytest.param(128, 127, 130, id="ratio128-boundary-decode"),
    ],
)
def test_compressed_cached_decode_matches_full_source_integration(
    ratio: int, prefix_len: int, total_len: int
) -> None:
    fixture = _prefill_then_decode_fixture(ratio, prefix_len, total_len)

    torch.testing.assert_close(
        fixture["prefix"],
        fixture["full"][:, :prefix_len],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        fixture["decode"],
        fixture["full"][:, prefix_len:],
        rtol=1e-5,
        atol=1e-5,
    )

    expected_compressed = _build_full_compressed_kv(
        fixture["compressor_kv"],
        fixture["compressor_gate"],
        fixture["compressor_ape"],
        fixture["compressor_norm_weight"],
        fixture["freqs_cis_table"],
        fixture["position_ids"],
        1e-6,
        fixture["rope_dim"],
        fixture["ratio"],
        fixture["max_compressed_len"],
    )
    _, mhc_cache, _, _ = fixture["caches"]
    last_completed_row = (total_len // ratio) - 1
    if last_completed_row >= 0:
        page = last_completed_row // 4
        offset = last_completed_row % 4
        torch.testing.assert_close(
            mhc_cache[page, offset, 0, 0],
            expected_compressed[0, last_completed_row],
            rtol=1e-5,
            atol=1e-5,
        )


def test_exported_dsv4_graph_cache_insertion_preserves_compressed_ratios() -> None:
    torch.manual_seed(123)
    config = _small_config()
    model = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 129))
    position_ids = torch.arange(129).unsqueeze(0).expand(2, -1)

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        dynamic_shapes={
            "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        },
        num_moe_experts_for_export=2,
    )

    source_nodes = [
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2)
    ]
    assert len(source_nodes) == 2

    cm = CachedSequenceInterface(
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=256,
        device="cpu",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=4,
            max_tokens=512,
            free_gpu_memory_fraction=0.0,
        ),
    )
    registry_config = _load_yaml(_CONFIGS_DIR / "deepseek_v4_flash.yaml")
    transform = InsertCachedAttention.from_kwargs(
        stage=Stages.CACHE_INIT,
        backend=registry_config["transforms"]["insert_cached_attention"]["backend"],
        run_graph_cleanup=False,
        requires_clean_graph=False,
    )

    transformed, info = transform._apply(gm, cm, SimpleNamespace(), SharedConfig())

    cached_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache)
    ]
    source_after = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2)
    ]
    placeholder_names = {node.name for node in transformed.graph.nodes if node.op == "placeholder"}
    ratio_to_max_len = {}
    for node in cached_nodes:
        _, compress_ratio, max_compressed_len, _, _ = extract_op_args(
            node,
            "window_size",
            "compress_ratio",
            "max_compressed_len",
            "rms_norm_eps",
            "rope_dim",
        )
        ratio_to_max_len[compress_ratio] = max_compressed_len

    assert info.num_matches == 2
    assert len(cached_nodes) == 2
    assert source_after == []
    assert ratio_to_max_len == {4: 64, 128: 2}
    assert any("mhc_cache" in name for name in placeholder_names)
    assert any("compressor_kv_cache" in name for name in placeholder_names)
    assert any("compressor_gate_cache" in name for name in placeholder_names)
