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
"""Native-state environment shared by the rope_kv_append tests and microbenchmark.

It allocates a real ``DSACacheManager`` (FP8 MLA latent pool + FP8 indexer
K-cache pool, tokens_per_block 64) at GLM-5.2 TP8 decode geometry and builds
the per-step tensors the native ops read: cache lengths, KV block offsets,
indexer slot mappings, the DSA block table and request indices. Both the
native ops (``mla_rope_generation``, ``indexer_k_cache_scatter_op``,
``convert_req_index_to_global[_grouped]``, the top-k reuse ``copy_``) and the
rope_kv_append ops are called on the same state so that results can be compared byte for
byte.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _compute_slot_mappings
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

NUM_HEADS = 8  # GLM-5.2 64 heads at TP8
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
LATENT = KV_LORA_RANK + QK_ROPE
INDEX_HEAD_DIM = 128
TOPK = 2048
TOKENS_PER_BLOCK = 64
QUANT_MODE_FP8_KV = 128
ROPE_THETA = 8_000_000.0
MAX_POSITIONS = 1 << 20


def load_dev_library() -> None:
    """Load a development build of the rope_kv_append ops when ``GLM_MOE_DSA_ROPE_KV_APPEND_DEV_LIB`` points to one."""
    path = os.environ.get("GLM_MOE_DSA_ROPE_KV_APPEND_DEV_LIB")
    if path and not hasattr(torch.ops.trtllm, "glm_moe_dsa_topk_index_bridge"):
        torch.ops.load_library(path)


CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1  # tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py
PAD = None  # batch entry standing for the CUDA-graph padding dummy request
PAD_DRAFT_LEN = 5  # served MTP depth; the padding dummy is created per draft length


@dataclass
class Step:
    """One generation step of a batch of requests (``rows`` query tokens each).

    Request-major like the native metadata: request ``i`` owns token rows
    ``[i * rows, (i + 1) * rows)`` and row ``i`` of the block offsets, the
    block table and the indexer block offsets.
    """

    request_ids: List[int]
    ctx_lens: List[int]  # cached tokens before this step, per batch entry
    rows: int
    is_pad: List[bool]  # entries that are the CUDA-graph padding dummy
    kv_lens_cuda: torch.Tensor  # [N] int32, cache length after the append
    kv_lens_cpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    block_offsets: torch.Tensor  # [num_pools, N, 2, max_blocks]
    block_table: torch.Tensor  # [N, max_blocks] pool slots, -1 padding
    req_idx: torch.Tensor  # [N * rows] int32 request index of each token
    slot_fp8: torch.Tensor  # [N * rows] int64
    slot_scale: torch.Tensor  # [N * rows] int64
    idx_block_offsets: torch.Tensor  # [N, max_blocks] indexer K-cache pages

    @property
    def num_seqs(self) -> int:
        return len(self.request_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_seqs * self.rows

    @property
    def request_id(self) -> int:
        return self.request_ids[0]

    @property
    def ctx_len(self) -> int:
        return self.ctx_lens[0]

    def kv_lens(self) -> List[int]:
        return [c + self.rows for c in self.ctx_lens]

    def slot_mappings(self, kv_lens):
        """Native ``_compute_slot_mappings`` for each request's ``rows`` tokens ending at ``kv_lens``."""
        if isinstance(kv_lens, int):
            kv_lens = [kv_lens] * self.num_seqs
        dev = self.idx_block_offsets.device
        positions = torch.cat(
            [torch.arange(k - self.rows, k, dtype=torch.int64, device=dev) for k in kv_lens]
        )
        req_ind = torch.arange(self.num_seqs, dtype=torch.int64, device=dev).repeat_interleave(
            self.rows
        )
        return _compute_slot_mappings(
            positions,
            self.idx_block_offsets,
            req_ind,
            INDEX_HEAD_DIM,
            TOKENS_PER_BLOCK,
            128,
            data_bytes_per_token=INDEX_HEAD_DIM,
        )

    def set_kv_lens(self, kv_lens: List[int]) -> None:
        """Advance the step in place (static addresses, as across CUDA-graph replays)."""
        self.kv_lens_cuda.copy_(torch.tensor(kv_lens, dtype=torch.int32))
        self.kv_lens_cpu.copy_(torch.tensor(kv_lens, dtype=torch.int32))
        fp8, scl = self.slot_mappings(kv_lens)
        self.slot_fp8.copy_(fp8)
        self.slot_scale.copy_(scl)


class RopeAppendEnv:
    """Real DSA cache manager plus helpers to call native and rope_kv_append ops."""

    def __init__(
        self,
        num_layers: int = 4,
        max_tokens: int = (1 << 20) + 8192,
        max_seq_len: int = (1 << 20) + 64,
        max_batch_size: int = 40,
    ):
        self.device = torch.device("cuda")
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        self.sparse_config = DeepSeekSparseAttentionConfig(
            index_n_heads=32, index_head_dim=INDEX_HEAD_DIM, index_topk=TOPK
        )
        model_config = ModelConfig(
            mapping=mapping,
            sparse_attention_config=self.sparse_config,
            pretrained_config=SimpleNamespace(rms_norm_eps=1e-5),
        )
        self.kvm = DSACacheManager(
            KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=LATENT,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.FP8,
            sparse_attention_config=self.sparse_config,
            model_config=model_config,
        )
        rope = RopeParams(
            dim=QK_ROPE, theta=ROPE_THETA, max_positions=max_seq_len, duplicate_data=True
        )
        _, self.cos_sin = rope.create_rope_const_params()
        self.kv_scale_orig_quant = torch.ones(1, dtype=torch.float32, device=self.device)
        self._next_request = 1000
        self._live: List = []
        self._pad = {}
        self._layer_ids = {}

    # ----------------------------------------------------------------- state
    def pools(self) -> List[torch.Tensor]:
        """Every cache byte the ops may write: the MLA pool and the indexer pools."""
        pools = [self.kvm.get_unique_primary_pool().view(torch.uint8)]
        for pool in self.kvm.indexer_k_cache_pool_per_layer:
            if pool is not None:
                pools.append(pool.view(torch.uint8))
        return pools

    def snapshot(self) -> List[torch.Tensor]:
        return [p.clone() for p in self.pools()]

    def restore(self, snap: List[torch.Tensor]) -> None:
        for p, s in zip(self.pools(), snap):
            p.copy_(s)

    def _padding_dummy(self, rows: int):
        """The CUDA-graph padding request, created as ``CUDAGraphRunner._get_or_create_padding_dummy``."""
        rid = CUDA_GRAPH_DUMMY_REQUEST_ID - PAD_DRAFT_LEN
        if rid not in self._pad:
            reqs = self.kvm.add_dummy_requests(
                [rid], token_nums=None, is_gen=True, max_num_draft_tokens=PAD_DRAFT_LEN
            )
            assert reqs, "KV pool exhausted"
            self._live.extend(reqs)
            self._pad[rid] = reqs[0]
        # Engine view of a generation dummy: past = prompt_len = token_num - 1 = draft_len.
        assert rows <= 1 + PAD_DRAFT_LEN
        return rid, PAD_DRAFT_LEN

    def new_step(self, ctx_len: int, rows: int) -> Step:
        """One generation request whose cache holds ``ctx_len`` tokens plus this step."""
        return self.new_batch([ctx_len], rows)

    def new_batch(self, ctx_lens: List[Optional[int]], rows: int) -> Step:
        """A generation batch; ``PAD`` entries are the CUDA-graph padding dummy (repeated as padding is)."""
        rids, ctxs, is_pad = [], [], []
        for ctx in ctx_lens:
            if ctx is PAD:
                rid, past = self._padding_dummy(rows)
                rids.append(rid)
                ctxs.append(past)
                is_pad.append(True)
                continue
            rid = self._next_request
            self._next_request += 1
            reqs = self.kvm.add_dummy_requests(
                [rid], token_nums=[ctx + 1], is_gen=True, max_num_draft_tokens=rows - 1
            )
            assert reqs, "KV pool exhausted"
            self._live.extend(reqs)
            rids.append(rid)
            ctxs.append(ctx)
            is_pad.append(False)
        n = len(rids)
        kv_lens = [c + rows for c in ctxs]
        dev = self.device
        offsets = torch.zeros(
            (self.kvm.num_pools, n, 2, self.kvm.max_blocks_per_seq), dtype=torch.int32, device=dev
        )
        self.kvm.copy_batch_block_offsets(offsets, rids, 1, 0, n)
        pool_idx = self.kvm.get_pool_block_indices(n)  # [n, max_blocks], host
        bt = pool_idx.clone()
        for i, kv in enumerate(kv_lens):
            bt[i, (kv + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK :] = -1
        idx_offsets = (pool_idx * self.kvm.indexer_k_cache_page_scale).to(dev)
        step = Step(
            request_ids=rids,
            ctx_lens=ctxs,
            rows=rows,
            is_pad=is_pad,
            kv_lens_cuda=torch.tensor(kv_lens, dtype=torch.int32, device=dev),
            kv_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32),
            prompt_lens_cpu=torch.tensor(ctxs, dtype=torch.int32),
            block_offsets=offsets,
            block_table=bt.to(dev),
            req_idx=torch.arange(n, dtype=torch.int32, device=dev).repeat_interleave(rows),
            slot_fp8=torch.empty(n * rows, dtype=torch.int64, device=dev),
            slot_scale=torch.empty(n * rows, dtype=torch.int64, device=dev),
            idx_block_offsets=idx_offsets,
        )
        fp8, scl = step.slot_mappings(kv_lens)
        step.slot_fp8.copy_(fp8)
        step.slot_scale.copy_(scl)
        return step

    def release(self) -> None:
        for req in self._live:
            self.kvm.free_resources(req)
        self._live = []
        self._pad = {}

    def shutdown(self) -> None:
        self.release()
        self.kvm.shutdown()

    # ------------------------------------------------------------ native ops
    def native_rope(
        self, step: Step, layer: int, inp: "RopeInputs", out: "RopeOutputs", q_scaling: float = 1.0
    ) -> None:
        """``TrtllmAttention._mla_rope_generation_impl`` with the DSA/MLA decode arguments."""
        torch.ops.trtllm.mla_rope_generation(
            inp.fused_q,
            inp.q_pe,
            inp.latent,
            self.cos_sin,
            out.cu_q,
            out.cu_kv,
            out.counter,
            out.bmm1,
            out.bmm2,
            out.quant_q,
            step.kv_lens_cuda,
            step.kv_lens_cpu,
            step.prompt_lens_cpu,
            0,
            step.block_offsets,
            self.kvm.kv_cache_pool_pointers,
            self.kvm.kv_cache_pool_mapping,
            None,
            None,
            self.kv_scale_orig_quant,
            None,
            None,
            [None, None],
            1,
            layer,
            NUM_HEADS,
            1,
            LATENT,
            0,
            TOKENS_PER_BLOCK,
            self.max_seq_len,
            1,
            QUANT_MODE_FP8_KV,
            q_scaling,
            Q_LORA_RANK,
            KV_LORA_RANK,
            QK_NOPE,
            QK_ROPE,
            V_HEAD,
            True,
            None,
            1e-6,
            False,
            False,
            False,
            False,
            None,
        )

    def native_scatter(self, step: Step, layer: int, inp: "RopeInputs") -> None:
        """``Indexer._update_k_cache``."""
        k_cache = self.kvm.get_indexer_k_cache_buffers(layer)
        torch.ops.trtllm.indexer_k_cache_scatter_op(
            inp.k_fp8, inp.k_scale, k_cache, step.slot_fp8, step.slot_scale, inp.k_fp8.shape[0]
        )

    def layer_ids(self, layers: List[int]) -> torch.Tensor:
        """Device layer-offset tensor of a group, built once (as the native group struct does)."""
        key = tuple(layers)
        if key not in self._layer_ids:
            self._layer_ids[key] = torch.tensor(
                [self.kvm.get_primary_pool_page_index_params(m)[1] for m in layers],
                dtype=torch.int32,
                device=self.device,
            )
        return self._layer_ids[key]

    def native_bridge(
        self,
        step: Step,
        topk: torch.Tensor,
        layers: List[int],
        shared: Optional[torch.Tensor],
        grouped: bool,
    ) -> torch.Tensor:
        """Top-k reuse copy + ``transform_local_topk_and_prepare_pool_view[_grouped]``."""
        if shared is not None:
            shared[0 : topk.shape[0], : topk.shape[1]].copy_(topk)
        scale, _ = self.kvm.get_primary_pool_page_index_params(layers[0])
        stride = scale * TOKENS_PER_BLOCK
        if grouped:
            layer_ids = self.layer_ids(layers)
            return torch.ops.trtllm.convert_req_index_to_global_grouped(
                step.req_idx,
                step.block_table,
                topk,
                TOKENS_PER_BLOCK,
                topk.shape[1],
                stride,
                layer_ids,
            )
        _, off = self.kvm.get_primary_pool_page_index_params(layers[0])
        return torch.ops.trtllm.convert_req_index_to_global(
            step.req_idx, step.block_table, topk, TOKENS_PER_BLOCK, topk.shape[1], stride, off
        )

    # --------------------------------------------------------------- rope_kv_append ops
    def rope_append_rope(
        self,
        step: Step,
        layer: int,
        inp: "RopeInputs",
        out: "RopeOutputs",
        with_scatter: bool,
        q_scaling: float = 1.0,
    ) -> None:
        idx = [None, None, None, None, None]
        n_idx = 0
        if with_scatter:
            idx = [
                inp.k_fp8,
                inp.k_scale,
                self.kvm.get_indexer_k_cache_buffers(layer),
                step.slot_fp8,
                step.slot_scale,
            ]
            n_idx = inp.k_fp8.shape[0]
        torch.ops.trtllm.glm_moe_dsa_rope_kv_append(
            inp.fused_q,
            inp.q_pe,
            inp.latent,
            self.cos_sin,
            out.cu_q,
            out.cu_kv,
            out.counter,
            out.bmm1,
            out.bmm2,
            out.quant_q,
            step.kv_lens_cuda,
            step.prompt_lens_cpu,
            0,
            step.block_offsets,
            self.kvm.kv_cache_pool_pointers,
            self.kvm.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            layer,
            NUM_HEADS,
            TOKENS_PER_BLOCK,
            self.max_seq_len,
            QUANT_MODE_FP8_KV,
            q_scaling,
            QK_NOPE,
            QK_ROPE,
            KV_LORA_RANK,
            *idx,
            n_idx,
        )

    def rope_append_bridge(
        self,
        step: Step,
        topk: torch.Tensor,
        layers: List[int],
        shared: Optional[torch.Tensor],
        grouped: bool,
        copy_mode: int = 0,
        next_n: int = 1,
        num_accepted=None,
        copy_rows: Optional[int] = None,
        num_rows: Optional[int] = None,
    ) -> torch.Tensor:
        scale, off = self.kvm.get_primary_pool_page_index_params(layers[0])
        layer_ids = None
        if grouped:
            layer_ids = self.layer_ids(layers)
        rows = topk.shape[0] if num_rows is None else num_rows
        crows = 0 if shared is None else (rows if copy_rows is None else copy_rows)
        return torch.ops.trtllm.glm_moe_dsa_topk_index_bridge(
            topk,
            rows,
            step.req_idx,
            0,
            step.block_table,
            TOKENS_PER_BLOCK,
            scale * TOKENS_PER_BLOCK,
            layer_ids,
            off,
            shared,
            0,
            crows,
            copy_mode,
            next_n,
            num_accepted,
        )


@dataclass
class RopeInputs:
    fused_q: torch.Tensor  # [R, H, 576] bf16
    q: torch.Tensor  # [R, H * 256] bf16, q_pe is a strided view of it
    q_pe: torch.Tensor
    latent: torch.Tensor  # [R, 576] bf16
    k_fp8: torch.Tensor  # [R, 128] fp8
    k_scale: torch.Tensor  # [R, 1] fp32


@dataclass
class RopeOutputs:
    quant_q: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    counter: torch.Tensor
    bmm1: torch.Tensor
    bmm2: torch.Tensor

    @staticmethod
    def make(rows: int, device, fill: int = 0xA5, num_seqs: int = 1) -> "RopeOutputs":
        """Outputs pre-filled with sentinels; the cu_seqlens buffers have 3 slack entries."""

        def i32(n: int) -> torch.Tensor:
            return torch.full((n,), -7, dtype=torch.int32, device=device)

        return RopeOutputs(
            quant_q=torch.full((rows, NUM_HEADS, LATENT), fill, dtype=torch.uint8, device=device),
            cu_q=i32(num_seqs + 4),
            cu_kv=i32(num_seqs + 4),
            counter=torch.full((1,), 99, dtype=torch.int32, device=device).view(torch.uint32),
            bmm1=torch.full((2,), -3.0, dtype=torch.float32, device=device),
            bmm2=torch.full((1,), -3.0, dtype=torch.float32, device=device),
        )

    def tensors(self):
        return [
            self.quant_q,
            self.cu_q,
            self.cu_kv,
            self.counter.view(torch.int32),
            self.bmm1,
            self.bmm2,
        ]


def make_inputs(
    rows: int, device, scale: float = 1.0, seed: int = 0, specials: bool = False
) -> RopeInputs:
    g = torch.Generator(device=device).manual_seed(seed)
    fused_q = (torch.randn((rows, NUM_HEADS, LATENT), generator=g, device=device) * scale).to(
        torch.bfloat16
    )
    q = (
        torch.randn((rows, NUM_HEADS * (QK_NOPE + QK_ROPE)), generator=g, device=device) * scale
    ).to(torch.bfloat16)
    latent = (torch.randn((rows, LATENT), generator=g, device=device) * scale).to(torch.bfloat16)
    if specials:
        vals = torch.tensor(
            [
                0.0,
                -0.0,
                1e-40,
                -1e-38,
                448.0,
                449.0,
                1e4,
                -1e4,
                float("inf"),
                float("-inf"),
                float("nan"),
                2.0**-10,
                3.0e-3,
            ],
            device=device,
        ).to(torch.bfloat16)
        for t in (fused_q.view(-1), q.view(-1), latent.view(-1)):
            pos = torch.randint(0, t.numel(), (vals.numel() * 8,), generator=g, device=device)
            t[pos] = vals.repeat(8)
    q_pe = q.view(rows, NUM_HEADS, QK_NOPE + QK_ROPE)[..., QK_NOPE:]
    k_fp8 = (
        torch.randint(0, 256, (rows, INDEX_HEAD_DIM), generator=g, device=device, dtype=torch.int32)
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
    )
    k_scale = torch.rand((rows, 1), generator=g, device=device, dtype=torch.float32) + 0.01
    return RopeInputs(fused_q=fused_q, q=q, q_pe=q_pe, latent=latent, k_fp8=k_fp8, k_scale=k_scale)


def make_topk(
    rows: int, kv_len: int, max_blocks: int, device, seed: int = 0, invalid_frac: float = 0.05
) -> torch.Tensor:
    """Top-k ids in [0, kv_len) plus -1 padding and out-of-table ids."""
    g = torch.Generator(device=device).manual_seed(seed)
    topk = torch.randint(
        0, max(kv_len, 1), (rows, TOPK), generator=g, device=device, dtype=torch.int32
    )
    mask = torch.rand((rows, TOPK), generator=g, device=device) < invalid_frac
    topk[mask] = -1
    beyond = torch.rand((rows, TOPK), generator=g, device=device) < invalid_frac / 2
    topk[beyond] = torch.randint(
        kv_len,
        kv_len + 4 * TOKENS_PER_BLOCK,
        (int(beyond.sum()),),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    far = torch.rand((rows, TOPK), generator=g, device=device) < invalid_frac / 4
    topk[far] = max_blocks * TOKENS_PER_BLOCK + 5
    return topk.contiguous()


def make_batch_inputs(
    step: Step, device, scale: float = 1.0, seed: int = 0, specials: bool = False
) -> RopeInputs:
    """Inputs for every token of ``step``.

    Repeated padding entries are the same request writing the same cache slots,
    so they carry identical rows (as padded graph replays feed identical dummy
    tokens); otherwise the final cache bytes would depend on write order for
    native and rope_kv_append alike.
    """
    inp = make_inputs(step.num_tokens, device, scale=scale, seed=seed, specials=specials)
    pads = [i for i, p in enumerate(step.is_pad) if p]
    r = step.rows
    for i in pads[1:]:
        src, dst = slice(pads[0] * r, pads[0] * r + r), slice(i * r, i * r + r)
        for t in (inp.fused_q, inp.q, inp.latent, inp.k_fp8.view(torch.uint8), inp.k_scale):
            t[dst] = t[src]
    return inp


def make_batch_topk(
    step: Step,
    device,
    seed: int = 0,
    rows: Optional[int] = None,
    kv_lens: Optional[List[int]] = None,
) -> torch.Tensor:
    """Per-request top-k rows (``rows`` per request, default ``step.rows``) inside each cache length."""
    rows = step.rows if rows is None else rows
    kv_lens = step.kv_lens() if kv_lens is None else kv_lens
    parts = [
        make_topk(rows, kv, step.block_table.shape[1], device, seed=seed * 131 + i)
        for i, kv in enumerate(kv_lens)
    ]
    return torch.cat(parts).contiguous()
