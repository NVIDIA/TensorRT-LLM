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
"""Native oracle of the decode_prologue target prologue (shared by the tests and the benchmark).

The native reference is the literal ``DecoderRunner._preprocess_inputs``
(called unbound on a runner stub that holds the overlap-scheduler offset
buffers, the configuration and the mapping) followed by the target model's
native ``Embedding``. It runs on a real CUDA-graph ``DSAtrtllmAttentionMetadata``
prepared from a real ``DSACacheManager`` at the served GEN DSA
configuration (draft_loop's fixtures), exactly as the decoder runner prepares a decode
step: ``create_cuda_graph_metadata``, ``update_spec_dec_param``,
``KVCacheParams``, ``prepare()``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch

_DRAFT_LOOP_TESTS = Path(__file__).resolve().parent.parent / "draft_loop"
sys.path.insert(0, str(_DRAFT_LOOP_TESTS))

from _draft_loop_fixtures import (  # noqa: E402
    MTP_K,
    add_gen_requests,
    load_checkpoint_tensor,
    make_dsa_cache_manager,
)

VOCAB = 154880
HIDDEN = 6144
# Served GEN: max_num_tokens 128, max_batch_size 1 (2 for the padded case).
MAX_NUM_TOKENS = 128
POISON = 0xA5
PACKET_ATTR = "_decode_prologue_embedding"

# Every metadata buffer the native prologue writes; poisoned identically in
# both arms before each run and compared in full afterwards.
DEST_FIELDS = (
    "kv_lens_cuda_2d",
    "scheduler_metadata_buffer",
    "scheduler_metadata_buffer_full_next_n",
    "kv_lens_expanded_cuda",
    "scheduler_metadata_buffer_expanded",
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
    "topk_indices_buffer",
)


def make_graph_metadata(
    cache_manager, mapping, sparse_config, request_ids, past_kv_lens, draft_len: int = MTP_K
):
    """A prepared CUDA-graph decode metadata (the runner's per-step preparation)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    batch_size = len(request_ids)
    base = DSAtrtllmAttentionMetadata(
        seq_lens=torch.full((batch_size,), draft_len + 1, dtype=torch.int),
        request_ids=list(request_ids),
        max_num_requests=batch_size,
        num_contexts=0,
        prompt_lens=list(past_kv_lens),
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=cache_manager,
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    md = base.create_cuda_graph_metadata(batch_size, False, draft_len)
    md.request_ids = list(request_ids)
    md.prompt_lens = list(past_kv_lens)
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
    # Set by the runner's input preparation (no chunked-context rows in a decode step).
    md.num_chunked_ctx_requests = 0
    return md, base


def make_runner_stub(model, num_tokens: int, batch_size: int):
    """The attributes ``_preprocess_inputs`` and ``target_prologue`` read from a runner."""
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS
    from tensorrt_llm.mapping import Mapping

    return SimpleNamespace(
        _config=SimpleNamespace(disable_overlap_scheduler=False),
        previous_pos_id_offsets_cuda=torch.zeros(
            (MAX_NUM_TOKENS,), dtype=torch.int32, device="cuda"
        ),
        previous_kv_lens_offsets_cuda=torch.zeros(
            (max(batch_size, 1),), dtype=torch.int32, device="cuda"
        ),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        guided_decoder=None,
        model=model,
        runtime_hooks=DEFAULT_RUNTIME_HOOKS,
    )


def make_target_model(weight: torch.Tensor):
    """A ``DeepseekV3Model`` shell holding only the native replicated ``Embedding``."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Model
    from tensorrt_llm._torch.modules.embedding import Embedding

    embed = Embedding(weight.shape[0], weight.shape[1], dtype=weight.dtype).cuda()
    embed.weight.data = weight
    target = DeepseekV3Model.__new__(DeepseekV3Model)
    torch.nn.Module.__init__(target)
    target.embed_tokens = embed
    return SimpleNamespace(model=target), embed


def embedding_weight(synthetic: bool = False) -> torch.Tensor:
    """The GLM-5.2 embedding table (synthetic only when asked or unavailable)."""
    if not synthetic:
        weight = load_checkpoint_tensor("model.embed_tokens.weight")
        if weight is not None:
            assert weight.shape == (VOCAB, HIDDEN) and weight.dtype == torch.bfloat16
            return weight
    gen = torch.Generator(device="cuda").manual_seed(12)
    weight = torch.randn((VOCAB, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=gen)
    weight[3, :4] = torch.tensor([-0.0, float("inf"), float("-inf"), float("nan")])
    return weight


class Arm:
    """One arm (native or decode_prologue): metadata, runner stub, static inputs."""

    def __init__(self, name, past_kv_lens, weight, cache_manager, mapping, sparse_config):
        self.name = name
        self.batch_size = len(past_kv_lens)
        ids = list(range(self.batch_size))
        self.md, self._base = make_graph_metadata(
            cache_manager, mapping, sparse_config, ids, past_kv_lens
        )
        self.num_tokens = self.md.num_tokens
        self.model, self.embed = make_target_model(weight)
        self.runner = make_runner_stub(self.model, self.num_tokens, self.batch_size)
        self.input_ids = torch.zeros((MAX_NUM_TOKENS,), dtype=torch.int32, device="cuda")
        self.position_ids = torch.zeros((1, MAX_NUM_TOKENS), dtype=torch.int32, device="cuda")
        self.out = None

    def inputs(self) -> dict:
        n = self.num_tokens
        return {
            "attn_metadata": self.md,
            "input_ids": self.input_ids[:n],
            "position_ids": self.position_ids[:, :n],
        }

    def set_step(self, kv_lens, positions, ids, kv_off, pos_off) -> None:
        """Install one step's inputs (eager copies outside any graph)."""
        b, n = self.batch_size, self.num_tokens
        self.md.kv_lens_cuda[:b].copy_(kv_lens)
        self.position_ids[0, :n].copy_(positions)
        self.input_ids[:n].copy_(ids)
        self.runner.previous_kv_lens_offsets_cuda[:b].copy_(kv_off)
        self.runner.previous_pos_id_offsets_cuda[:n].copy_(pos_off)

    def poison(self) -> None:
        for name in DEST_FIELDS:
            t = getattr(self.md, name, None)
            if isinstance(t, torch.Tensor):
                t.view(torch.uint8).fill_(POISON) if t.is_contiguous() else t.fill_(-1)


def run_native(
    arm: Arm, inputs: Optional[dict] = None, enable_spec_decode: bool = True
) -> torch.Tensor:
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    inputs = arm.inputs() if inputs is None else inputs
    DecoderRunner._preprocess_inputs(
        arm.runner, inputs, enable_spec_decode=enable_spec_decode, runtime_draft_len=MTP_K
    )
    return arm.embed(inputs["input_ids"])


def run_decode_prologue(
    arm: Arm, inputs: Optional[dict] = None, enable_spec_decode: bool = True
) -> torch.Tensor:
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.decode_prologue import (  # noqa: E501
        prologue,
    )

    inputs = arm.inputs() if inputs is None else inputs
    done = prologue.target_prologue(
        arm.runner, inputs, enable_spec_decode=enable_spec_decode, runtime_draft_len=MTP_K
    )
    assert done, prologue.prologue_ineligibility(arm.runner, inputs, enable_spec_decode)
    rows = prologue.take_embedding(inputs["attn_metadata"], inputs["input_ids"], arm.embed)
    assert rows is not None
    return rows


def state(arm: Arm) -> Dict[str, object]:
    """Every tensor (cloned) and scalar attribute of the metadata, plus the step tensors."""
    out = {}
    for name, value in vars(arm.md).items():
        if isinstance(value, torch.Tensor):
            out[name] = value.detach().clone()
        elif isinstance(value, (bool, int, float, str, type(None))):
            out[name] = value
    out["@position_ids"] = arm.position_ids.clone()
    out["@input_ids"] = arm.input_ids.clone()
    out["@kv_off"] = arm.runner.previous_kv_lens_offsets_cuda.clone()
    out["@pos_off"] = arm.runner.previous_pos_id_offsets_cuda.clone()
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(_bits(a).cpu(), _bits(b).cpu()))
    return type(a) is type(b) and a == b


def changed(before: dict, after: dict) -> set:
    return {
        k
        for k in set(before) | set(after)
        if not _same(before.get(k, "<missing>"), after.get(k, "<missing>"))
    }


def compare(native_before, native_after, z_before, z_after) -> List[str]:
    """Names whose bytes (or host values) differ between the arms after the run.

    Compared: every key either arm changed, every destination buffer in full,
    and the step tensors.
    """
    keys = changed(native_before, native_after) | changed(z_before, z_after)
    keys |= {k for k in DEST_FIELDS if k in native_after}
    keys |= {"@position_ids", "@input_ids", "@kv_off", "@pos_off", "kv_lens_cuda"}
    bad = []
    for k in sorted(keys):
        if not _same(native_after.get(k, "<missing>"), z_after.get(k, "<missing>")):
            bad.append(k)
    return bad


def step_inputs(
    gen: torch.Generator,
    kv_base: Sequence[int],
    accepted: Sequence[int],
    num_tokens: int,
    draft_len: int = MTP_K,
    id_edges: bool = False,
):
    """One step's (kv_lens, positions, input ids, kv offsets, position offsets).

    ``kv_base[r]`` is request r's KV length after the correction; the eager
    preparation wrote ``kv_base[r] - offset`` (the overlap scheduler assumed
    every draft accepted), and the correction adds the offset
    ``accepted[r] - draft_len`` back. Positions follow the same rule per token.
    """
    b = len(kv_base)
    kv_off = torch.tensor([a - draft_len for a in accepted], dtype=torch.int32)
    kv_lens = torch.tensor(kv_base, dtype=torch.int32) - kv_off
    per_req = num_tokens // b
    pos = []
    pos_off = []
    for r in range(b):
        start = kv_base[r] - per_req
        for t in range(per_req):
            pos.append(start + t - (accepted[r] + 1))
            pos_off.append(accepted[r] + 1)
    ids = torch.randint(0, VOCAB, (num_tokens,), generator=gen, dtype=torch.int32)
    if id_edges:
        edge = torch.tensor([-1, VOCAB, VOCAB + 7, 0, VOCAB - 1, 3], dtype=torch.int32)
        ids[: min(num_tokens, edge.numel())] = edge[: min(num_tokens, edge.numel())]
    return (
        kv_lens.cuda(),
        torch.tensor(pos, dtype=torch.int32).cuda(),
        ids.cuda(),
        kv_off.cuda(),
        torch.tensor(pos_off, dtype=torch.int32).cuda(),
    )


def make_pair(
    past_kv_lens, weight, max_kv: Optional[int] = None, sparse_update: Optional[dict] = None
):
    """A cache manager and two identically prepared arms on it.

    ``sparse_update`` overrides fields of the arms' DSA configuration (the
    metadata's parameters), e.g. the paged-MQA logits kernel choice.
    """
    batch_size = len(past_kv_lens)
    cache_manager, mapping, sparse = make_dsa_cache_manager(
        batch_size, max_kv or max(past_kv_lens) + 4096
    )
    if sparse_update:
        sparse = sparse.model_copy(update=sparse_update)
    add_gen_requests(cache_manager, list(range(batch_size)), past_kv_lens)
    native = Arm("native", past_kv_lens, weight, cache_manager, mapping, sparse)
    decode_prologue = Arm("decode_prologue", past_kv_lens, weight, cache_manager, mapping, sparse)
    return cache_manager, native, decode_prologue


def capture(arm: Arm, fn, debug_dot: Optional[str] = None):
    """Warm up ``fn(arm)`` eagerly once, then capture it; returns (graph, output)."""
    fn(arm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if debug_dot:
        graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        out = fn(arm)
    if debug_dot:
        graph.debug_dump(debug_dot)
    return graph, out
