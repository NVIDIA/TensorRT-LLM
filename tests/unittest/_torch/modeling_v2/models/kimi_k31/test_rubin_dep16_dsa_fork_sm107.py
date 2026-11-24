# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's DSA decode streams on one sm_107 GPU (``dsa_idx_fork``, ``dsa_gate_early``,
``dsa_rope_before_idx_join``, ``dsa_logits_first``): bit for bit.

One dep16 DSA decode layer, every kernel real (attention DP: the layer has no collective): its
weights at dep16's shapes, an FP8 latent pool and an index-key cache paged as the KV cache manager
pages them, a decode step of one-row requests whose KV lengths pass the Top-K's 2048, in fixed
tensors. The step state is the target's own projection (``_dsa_step_args``) of the metadata; the
indexer runs: the FP8 DSL paged logits with the DeepGEMM schedule, the GVR Top-K, the remap, the
sparse MLA. On CPU the test collects and skips.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
TARGET = "kimi_k31_pro_nvfp4__sm_107__dep16"
HIDDEN, Q_LORA, KV_LORA, ROPE, NOPE, V = 7168, 1536, 512, 64, 128, 128
HEADS, IDX_HEADS, IDX_DIM = 96, 32, 128
LATENT = KV_LORA + ROPE
TOPK = 2048
TPB = 64  # tokens per block
LAYER = 3  # a DSA layer
SCHEDULE_BLOCK_KV = 64


def module():
    return importlib.import_module(f"{PKG}.{TARGET}.modeling")


def layer_weights(mod, gen):
    """A DSA layer's weight views at dep16's shapes, seeded (the views ``_layer_views`` takes)."""

    def rnd(rows, cols, dtype=torch.bfloat16):
        return (torch.randn(rows, cols, generator=gen, device="cuda") * cols**-0.5).to(dtype)

    def norm(size, dtype=torch.bfloat16):
        return (1 + 0.1 * torch.randn(size, generator=gen, device="cuda")).to(dtype)

    qkv_a = rnd(Q_LORA + KV_LORA + ROPE, HIDDEN)
    q_b = rnd(HEADS * (NOPE + ROPE), Q_LORA)
    kv_b = rnd(HEADS * (NOPE + V), KV_LORA)
    g, o = rnd(HEADS * V, HIDDEN), rnd(HIDDEN, HEADS * V)
    idx_wq_b = rnd(IDX_HEADS * IDX_DIM, Q_LORA)
    return mod._MlaW(
        q_a_norm=norm(Q_LORA),
        kv_a_norm=norm(KV_LORA),
        q_b=q_b,
        g=g,
        o=o,
        idx_wk_wp=rnd(IDX_DIM + IDX_HEADS, HIDDEN, torch.float32),
        idx_k_norm_w=norm(IDX_DIM, torch.float32),
        idx_k_norm_b=0.1 * torch.randn(IDX_DIM, generator=gen, device="cuda"),
        k_b=kv_b[: HEADS * NOPE].view(HEADS, NOPE, KV_LORA),
        v_b_t=kv_b[HEADS * NOPE :].view(HEADS, V, KV_LORA).transpose(1, 2),
        qkv_a_t=qkv_a.t(),
        idx_wq_b_t=idx_wq_b.t(),
    )


class Metadata(SimpleNamespace):
    """What dep16's step projection reads of the family DSA metadata on a decode step."""

    def get_indexer_max_seq_len(self):
        return self.max_seq_len

    def get_empty(self, buffers, shape, cache_name, dtype, capture_graph):
        return self.topk_buffer[: shape[0]]


def kv_lens(rows, low, span, seed=7):
    """Each request's KV length (its new token included), spread over [low, low + span)."""
    g = torch.Generator().manual_seed(seed)
    return (low + torch.randint(0, span, (rows,), generator=g)).to(torch.int32)


class Rig:
    """One rank's DSA layer, its caches and a decode step of ``rows`` one-row requests."""

    def __init__(self, mod, rows, kv, max_seq, layer=None, seed=0):
        self.mod, self.rows = mod, rows
        dev = "cuda"
        self.layer = layer or layer_weights(mod, torch.Generator(device=dev).manual_seed(seed))
        pages_per = max_seq // TPB
        blocks = rows * pages_per + 2
        order = torch.randperm(blocks, generator=torch.Generator().manual_seed(seed + 1)).tolist()
        pages = [[order.pop() for _ in range(-(-int(k) // TPB))] for k in kv.tolist()]
        table = torch.zeros(rows, pages_per, dtype=torch.int32)
        for r, row in enumerate(pages):
            table[r, : len(row)] = torch.tensor(row, dtype=torch.int32)
        offsets = torch.zeros(1, rows, 2, pages_per, dtype=torch.int32)
        offsets[0, :, 0] = table
        offsets[0, :, 1] = table
        # The FP8 latent pool of the layer, and its index-key cache: a page holds 64 keys of 128
        # bytes, then their 64 FP32 scales.
        self.pool = torch.zeros(blocks, TPB, LATENT, dtype=torch.float8_e4m3fn, device=dev)
        self.idx_cache = torch.zeros(blocks, TPB, 1, IDX_DIM + 4, dtype=torch.uint8, device=dev)
        self.x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.kv = kv
        pos = kv - 1  # the new token's position
        page = torch.tensor([row[p // TPB] for row, p in zip(pages, pos.tolist())])
        slot = pos.long() % TPB
        page_bytes = TPB * (IDX_DIM + 4)
        self.new_rows = (page.long(), slot)  # where the step writes the latent and index-key rows
        slot_fp8 = page * page_bytes + slot * IDX_DIM
        slot_scale = page * page_bytes + TPB * IDX_DIM + slot * 4
        self.topk_out = torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev)
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        sched = torch.zeros(num_sms + 1, 2, dtype=torch.int32, device=dev)
        meta = importlib.import_module(f"{PKG}.ops.attention.get_paged_mqa_logits_metadata")
        kv_dev = kv.to(dev)
        meta.get_paged_mqa_logits_metadata(kv_dev.view(-1, 1), SCHEDULE_BLOCK_KV, num_sms, sched)
        cache = SimpleNamespace(
            kv_cache_pool_pointers=torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64),
            kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
            tokens_per_block=TPB,
            max_attention_window_vec=[None],
        )
        md = Metadata(
            is_ragged_verify=False,
            skip_indexer_for_gen_reqs=False,
            skip_indexer_for_ctx_reqs=False,
            is_cuda_graph=True,
            cuda_graph_buffers=None,
            topk_buffer=self.topk_out,
            kv_cache_manager=cache,
            kv_lens_cuda_runtime=kv_dev,
            kv_lens_runtime=kv.clone(),
            gen_indexer_kv_lens_cuda_runtime=kv_dev,
            prompt_lens_cuda_runtime=pos.to(dev),
            prompt_lens_cpu_runtime=pos.clone(),
            host_total_kv_lens=torch.tensor([0, int(kv.sum())], dtype=torch.int32),
            host_request_types_runtime=torch.ones(rows, dtype=torch.int32),
            max_context_q_len_override=None,
            kv_cache_block_offsets=offsets.to(dev),
            host_kv_cache_pool_pointers=cache.kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=cache.kv_cache_pool_mapping,
            cache_indirection=None,
            block_ids_per_seq=None,
            tokens_per_block=TPB,
            max_num_requests=rows,
            max_num_sequences=rows,
            beam_width=1,
            effective_beam_width=1,
            use_paged_context_fmha=False,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets_for_cpp=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            max_total_draft_tokens=None,
            force_prepare_spec_dec_tree_mask=False,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            num_sparse_topk=TOPK,
            flash_mla_tile_scheduler_metadata=None,
            flash_mla_num_splits=None,
            max_context_length=max_seq,
            max_seq_len=max_seq,
            trtllm_gen_jit_warmup=False,
            is_cross=False,
            effective_workspace=torch.empty(64 << 20, dtype=torch.int8, device=dev),
            indexer_k_cache_block_offsets=table.to(dev),
            scheduler_metadata_buffer=sched,
            topk_indices_buffer=torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev),
            radix_aux_indices=None,
            radix_aux_logits=None,
            req_idx_per_token=torch.arange(rows, dtype=torch.int32, device=dev),
            block_table=table.to(dev),
            slot_mapping_fp8=slot_fp8.to(dev),
            slot_mapping_scale=slot_scale.to(dev),
        )
        width = -(-max_seq // 256) * 256
        self.arena = torch.zeros(rows, width, dtype=torch.float32, device=dev)
        pools = SimpleNamespace(
            idx_k_cache={LAYER: self.idx_cache},
            pool_params={LAYER: (1, 0)},
            local={LAYER: 0},
            rope_positions=2 * max_seq,
            tokens_per_block=TPB,
            gen_requests=rows,
            idx_logits=self.arena,
        )
        self.rt = SimpleNamespace(
            dsa_pools=pools,
            mla_rope=mod._identity_rope_table(pools.rope_positions, ROPE, dev),
            aux_stream=torch.cuda.Stream(),
            fork_stream=torch.cuda.Stream(),
            idx_stream=torch.cuda.Stream(),
            events=tuple((torch.cuda.Event(), torch.cuda.Event()) for _ in range(4)),
            # the decode logits' PDL launch, as the generation constant sets it (dsa_idx_logits_pdl)
            idx_logits_pdl=mod._CONSTANTS["generation"]["dsa_idx_logits_pdl"],
            # a generation server's decode levers (_levers: its constants)
            levers=mod._levers("generation"),
        )
        dsa = mod._dsa_step_args(md, pools, rows, 0, 0, rows)
        self.steps = {
            graph: mod._Step(
                md=md,
                kda_md=None,
                cache=cache,
                position_ids=pos.to(dev),
                num_tokens=rows,
                num_contexts=0,
                num_ctx_tokens=0,
                num_generations=rows,
                rows_per_gen=1,
                all_rank_num_tokens=(rows,) * 16,
                spec_md=None,
                cuda_graph=graph,
                dsa=dsa,
            )
            for graph in (False, True)
        }

    def run(self, cuda_graph: bool) -> torch.Tensor:
        return self.mod.mla_generate(self.layer, LAYER, self.x, self.steps[cuda_graph], self.rt)

    def inputs(self, seed: int) -> dict:
        """A step's data: the layer input and the caches' contents (the KV history)."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        pool = 0.5 * torch.randn(self.pool.shape, generator=gen, device="cuda")
        keys = torch.randint(
            0, 256, self.idx_cache.shape, generator=gen, device="cuda", dtype=torch.uint8
        )
        # A page is 64 keys of 128 FP8 bytes, then their 64 FP32 scales: finite keys (no 0x7f /
        # 0xff NaN codes) and finite positive scales.
        blocks = self.idx_cache.shape[0]
        page = keys.view(blocks, TPB * (IDX_DIM + 4))
        page[:, : TPB * IDX_DIM] &= 0x77
        scales = 0.01 + torch.rand(blocks, TPB, generator=gen, device="cuda")
        page[:, TPB * IDX_DIM :] = scales.view(torch.uint8)
        return dict(
            x=torch.randn(self.x.shape, generator=gen, device="cuda").to(torch.bfloat16),
            pool=pool.to(torch.float8_e4m3fn),
            idx_cache=keys,
        )

    def load(self, inputs: dict) -> None:
        self.x.copy_(inputs["x"])
        self.pool.copy_(inputs["pool"])
        self.idx_cache.copy_(inputs["idx_cache"])


def topk_sets_equal(a_topk, b_topk, a_logits, b_logits, kv):
    """Top-K rows as sorted sets (the kernel may write a row's indices in any order); a row whose
    sets differ counts as equal only when the multiset of its selected logits is bitwise equal (a
    boundary tie). Returns (equal, rows that differ as sets, tie rows)."""
    sa, _ = torch.sort(a_topk, dim=1)
    sb, _ = torch.sort(b_topk, dim=1)
    rows = (sa != sb).any(dim=1).nonzero().flatten().tolist()
    ties = 0
    for r in rows:
        n = int(kv[r])
        ia, ib = (
            a_topk[r][(a_topk[r] >= 0) & (a_topk[r] < n)],
            b_topk[r][(b_topk[r] >= 0) & (b_topk[r] < n)],
        )
        if ia.numel() != ib.numel():
            return False, len(rows), ties
        va, _ = torch.sort(a_logits[r, ia.long()])
        vb, _ = torch.sort(b_logits[r, ib.long()])
        if not torch.equal(va.view(torch.int32), vb.view(torch.int32)):
            return False, len(rows), ties
        ties += 1
    return True, len(rows), ties


# --- one sm_107 GPU: the captured lever against today's in-order step ---

_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="dep16's kernels run on sm_107",
)
# A delay's spin cycles: milliseconds, far longer than any stream's work after a fork.
_SPIN = 4_000_000
# Requests' KV lengths: past the Top-K's 2048, so the GVR Top-K selects, around page boundaries.
_KV_LOW, _KV_SPAN, _MAX_SEQ = 2100, 1900, 4096


def _keep(kept: dict, name: str, t: torch.Tensor) -> None:
    raw = t.view(torch.uint8)
    if name not in kept:
        kept[name] = torch.empty_like(raw)
    kept[name].copy_(raw)


def _spy(mod, kept: dict, monkeypatch) -> None:
    """Keep the sparse attention's output and the indexer's FP8 queries and logits weights."""
    attention, decode = mod.thop_attention, mod._idx_decode

    def thop_attention(**kw):
        out = attention(**kw)
        _keep(kept, "fmha output", out)
        return out

    def idx_decode(i, q8, w, st, rt):
        _keep(kept, "indexer q8", q8)
        _keep(kept, "indexer weights", w)
        return decode(i, q8, w, st, rt)

    monkeypatch.setattr(mod, "thop_attention", thop_attention)
    monkeypatch.setattr(mod, "_idx_decode", idx_decode)


def _delay(mod, rig: Rig, where: str, monkeypatch) -> None:
    """A spin kernel at the head of the idx stream, of the fork stream, or before the indexer's
    query projection (``main``: on the current stream, or on the aux stream under
    dsa_logits_first): the other streams' work goes first, so a missing edge corrupts every run
    instead of some."""
    if where == "idx":
        keys = mod._idx_keys

        def idx_keys(*args):
            if torch.cuda.current_stream() == rig.rt.idx_stream:
                torch.cuda._sleep(_SPIN)
            return keys(*args)

        monkeypatch.setattr(mod, "_idx_keys", idx_keys)
    elif where == "fork":
        linear = mod.rubin_linear

        def rubin_linear(*args):
            if torch.cuda.current_stream() == rig.rt.fork_stream:
                torch.cuda._sleep(_SPIN)
            return linear(*args)

        monkeypatch.setattr(mod, "rubin_linear", rubin_linear)
    elif where == "main":
        queries = mod._idx_queries

        def idx_queries(*args):
            torch.cuda._sleep(_SPIN)
            return queries(*args)

        monkeypatch.setattr(mod, "_idx_queries", idx_queries)


def _snapshot(rig: Rig, out: torch.Tensor, kept: dict) -> dict:
    cols = torch.arange(rig.arena.shape[1], device="cuda")[None, :]
    valid = cols < rig.kv.to("cuda")[:, None]
    snap = {
        "o_proj partial": out.view(torch.uint8),
        "latent pool": rig.pool.view(torch.uint8),
        "index-key cache": rig.idx_cache,
        "logits (valid columns)": torch.where(valid, rig.arena, 0.0).view(torch.int32),
        **kept,
    }
    snap = {name: t.clone() for name, t in snap.items()}
    snap["topk"] = rig.topk_out.clone()
    snap["logits"] = rig.arena.clone()
    return snap


@_ON_SM107
@pytest.mark.parametrize("logits_first", [True, False])
@pytest.mark.parametrize("rows", [192, 96, 64, 32])
@pytest.mark.parametrize("delay", ["none", "idx", "fork", "main"])
def test_the_captured_step_equals_todays_in_order_step(rows, delay, logits_first, monkeypatch):
    """``mla_generate`` of one dep16 DSA layer, every kernel real, captured in a CUDA graph on one
    set of inputs and replayed on two others, equals today's in-order step (the eager path: every
    kernel on the current stream, ``x.float()`` and the weights' two products) on those inputs bit
    for bit: the o_proj partial, the latent pool, the index-key cache, the logits on each row's
    keys, the sparse attention's output, the indexer's FP8 queries and logits weights; the Top-K
    rows as sets (a row that differs counts only when its selected logits are the same multiset).
    The graph step runs a generation server's levers with ``dsa_logits_first`` on and off; rows 96
    and 192 are dep16's b12 and b24 verify buckets."""
    mod = module()
    rig = Rig(mod, rows, kv_lens(rows, _KV_LOW, _KV_SPAN), _MAX_SEQ)
    rig.rt.levers = rig.rt.levers._replace(dsa_logits_first=logits_first)
    kept: dict = {}
    _spy(mod, kept, monkeypatch)
    _delay(mod, rig, delay, monkeypatch)
    a, b, c = (rig.inputs(seed) for seed in (1, 2, 3))
    for cuda_graph in (False, True):  # kernel builds, workspaces, the warm-up of the graph path
        rig.load(a)
        rig.run(cuda_graph)
    rig.load(a)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rig.run(True)
    torch.cuda.synchronize()
    for inputs in (b, c):
        rig.load(inputs)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _snapshot(rig, out, kept)
        rig.load(inputs)
        in_order = rig.run(False)
        torch.cuda.synchronize()
        want = _snapshot(rig, in_order, kept)
        assert torch.isfinite(in_order.float()).all() and in_order.abs().sum() > 0
        ok, rows_differ, ties = topk_sets_equal(
            replayed["topk"], want["topk"], replayed["logits"], want["logits"], rig.kv
        )
        assert ok, f"{rows_differ} Top-K rows differ as sets ({ties} of them boundary ties)"
        differ = [
            name
            for name in want
            if name not in ("topk", "logits") and not torch.equal(replayed[name], want[name])
        ]
        assert not differ, f"the captured step differs from today's in-order step in {differ}"
