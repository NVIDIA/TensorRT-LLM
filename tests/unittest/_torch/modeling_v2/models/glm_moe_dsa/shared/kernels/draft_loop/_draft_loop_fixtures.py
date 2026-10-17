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
"""Shared helpers for the draft_loop tests and microbenchmarks."""

import json
import os
from pathlib import Path
from typing import Optional

import torch
from utils.llm_data import llm_models_root

GLM52_CKPT_ENV = "GLM52_CKPT"
_DEFAULT_CKPT = os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4")


def checkpoint_dir() -> Optional[Path]:
    """Return the GLM-5.2 NVFP4 checkpoint directory if it is readable."""
    path = Path(os.environ.get(GLM52_CKPT_ENV, _DEFAULT_CKPT))
    if (path / "model.safetensors.index.json").is_file():
        return path
    return None


def load_checkpoint_tensor(name: str, device: str = "cuda") -> Optional[torch.Tensor]:
    """Load one tensor of the GLM-5.2 checkpoint, or None if unavailable."""
    ckpt = checkpoint_dir()
    if ckpt is None:
        return None
    from safetensors import safe_open

    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(ckpt / weight_map[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device)


def real_token_ids(num_tokens: int) -> Optional[torch.Tensor]:
    """Tokenize a fixed tool-use style text with the checkpoint tokenizer."""
    ckpt = checkpoint_dir()
    if ckpt is None:
        return None
    # The raw tokenizer avoids loading the model config through transformers.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    text = (
        "<tool_call>search_files(path='/repo/src', pattern='def forward')</tool_call>\n"
        "The draft loop embeds each proposed token, normalizes it, and projects the "
        "concatenation with the previous hidden state; 你好，世界。 0x7f3a 3.14159 "
    ) * 64
    ids = tokenizer.encode(text, add_special_tokens=True).ids
    assert len(ids) >= num_tokens
    return torch.tensor(ids[:num_tokens], dtype=torch.int32)


def bits(t: torch.Tensor) -> torch.Tensor:
    """Bit view of a floating tensor (for bitwise equality incl. -0.0/NaN)."""
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def graph_time_us(fn, reps: int = 50, replays: int = 20, warmup: int = 3) -> float:
    """Mean device time of ``fn`` per call, measured as a CUDA graph of ``reps`` calls.

    The graph holds ``reps`` back-to-back invocations so the result includes
    the inter-kernel launch gaps a captured decode graph pays.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    del graph
    return elapsed_ms * 1000.0 / (replays * reps)


# ----------------------------------------------------------------------------
# Real DSA generation metadata at the served GEN configuration
# ----------------------------------------------------------------------------

MTP_K = 5
TOKENS_PER_BLOCK = 64


def served_tp8_sparse_config():
    """DSA config of the served GEN role (GLM-5.2 indexer geometry)."""
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    return DeepSeekSparseAttentionConfig(
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        enable_heuristic_topk=True,
        use_cute_dsl_paged_mqa_logits=True,
        use_cute_dsl_topk=True,
        indexer_k_dtype="fp8",
    )


def make_dsa_cache_manager(batch_size: int, max_kv_len: int, draft_len: int = MTP_K):
    """A one-layer FP8 DSA cache manager sized for ``batch_size`` long requests."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.executor import KvCacheConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    sparse_config = served_tp8_sparse_config()
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=SimpleNamespace(rms_norm_eps=1e-5),
    )
    max_seq_len = max_kv_len + 2 * (draft_len + 1) + TOKENS_PER_BLOCK
    blocks = (max_seq_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK + 2
    return (
        DSACacheManager(
            KvCacheConfig(
                max_tokens=batch_size * blocks * TOKENS_PER_BLOCK, enable_block_reuse=False
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=DataType.FP8,
            sparse_attn_config=sparse_config,
            model_config=model_config,
        ),
        mapping,
        sparse_config,
    )


def make_gen_dsa_metadata(
    cache_manager,
    mapping,
    sparse_config,
    request_ids,
    past_kv_lens,
    draft_len: int = MTP_K,
    draft_cache_manager=None,
):
    """Prepared generation metadata for one MTP verify step (K+1 query tokens each).

    Follows the decoder runner's order: ``update_spec_dec_param`` (one-model MTP,
    linear tree), then ``KVCacheParams`` with the one-model extra KV tokens and
    ``prepare()``.
    """
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    batch_size = len(request_ids)
    md = DSAtrtllmAttentionMetadata(
        seq_lens=torch.full((batch_size,), draft_len + 1, dtype=torch.int),
        request_ids=list(request_ids),
        max_num_requests=batch_size,
        num_contexts=0,
        prompt_lens=list(past_kv_lens),
        max_num_tokens=batch_size * (draft_len + 1),
        kv_cache_manager=cache_manager,
        draft_kv_cache_manager=draft_cache_manager,
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    md.update_spec_dec_param(
        batch_size=batch_size,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        max_draft_len=draft_len,
        max_total_draft_tokens=draft_len,
        spec_metadata=None,
        spec_tree_manager=None,
        num_contexts=0,
    )
    md.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=list(past_kv_lens),
        num_extra_kv_tokens=draft_len - 1,
    )
    md.prepare()
    return md


def add_gen_requests(cache_manager, request_ids, past_kv_lens, draft_len: int = MTP_K):
    """Allocate KV blocks for generation requests with the given past lengths."""
    cache_manager.add_dummy_requests(
        list(request_ids),
        token_nums=[n + 1 for n in past_kv_lens],
        is_gen=True,
        prepare_resource=True,
        max_num_draft_tokens=draft_len,
    )


def tensor_state(obj) -> dict:
    """Clone every tensor and scalar attribute of ``obj`` (for state comparison)."""
    state = {}
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            state[name] = value.detach().clone()
        elif isinstance(value, (bool, int, float, str, type(None))):
            state[name] = value
    return state


def changed_keys(before: dict, after: dict) -> set:
    """Attributes whose value differs between two ``tensor_state`` snapshots."""
    keys = set()
    for name in set(before) | set(after):
        a, b = before.get(name, "<missing>"), after.get(name, "<missing>")
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
                keys.add(name)
        elif type(a) is not type(b) or a != b:
            keys.add(name)
    return keys


def assert_same_state(native: dict, candidate: dict, keys, what: str) -> None:
    """Assert that ``keys`` hold identical values in both snapshots."""
    for name in sorted(keys):
        a, b = native.get(name, "<missing>"), candidate.get(name, "<missing>")
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor), f"{what}: {name} type differs"
            assert a.shape == b.shape and a.dtype == b.dtype, f"{what}: {name} shape/dtype"
            assert torch.equal(a.cpu(), b.cpu()), f"{what}: {name} differs"
        else:
            assert a == b, f"{what}: {name} differs ({a!r} vs {b!r})"


def native_draft_step_tail(
    md,
    i,
    batch_size,
    num_contexts,
    runtime_draft_len,
    num_accepted_tokens,
    position_ids,
    gather_ids,
):
    """Literal copy of the native loop tail for draft step ``i`` (MTP Eagle, DSA)."""
    from tensorrt_llm._torch.speculative.mtp import _select_mtp_position_ids

    new_position_ids = None
    if i + 1 < runtime_draft_len:
        new_position_ids = _select_mtp_position_ids(position_ids, gather_ids) + 1
    attn_metadata = md
    if i == 0:
        attn_metadata._seq_lens[:batch_size].fill_(1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
        attn_metadata.on_update()
        has_kv_cache = attn_metadata.kv_cache_manager is not None
        if has_kv_cache and hasattr(attn_metadata, "host_request_types"):
            attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
        if has_kv_cache:
            attn_metadata.num_contexts = 0
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                runtime_draft_len - num_accepted_tokens[num_contexts:]
            )
            attn_metadata.kv_lens_cuda[:num_contexts] += 1
        # _prepare_flash_mla_generation_layout is a no-op without context rows.
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.update_for_spec_dec()
        attn_metadata.use_spec_decoding = False
    else:
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[:batch_size] += 1
            attn_metadata.update_for_spec_dec()
    return new_position_ids
