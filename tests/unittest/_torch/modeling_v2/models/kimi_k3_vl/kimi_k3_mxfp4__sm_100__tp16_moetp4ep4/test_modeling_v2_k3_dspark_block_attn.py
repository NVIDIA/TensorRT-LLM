# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's attention/k3_dspark_block_attn op (sm_100).

The arguments are built exactly as the built-in DFlash drafter builds them for its TRTLLM backend on a paged context
cache (``DFlashForCausalLM.dflash_forward``, ``is_paged``, unwindowed non-causal layer): a slot page table, the
batch's rows of it, the append indices and positions, the kernel's lengths and the workspace / counters sized as
``_prepare_dflash_trtllm_gen_buffers`` sizes them. Each cell then checks

* ``out`` against ``reference`` within the band the kernel's bf16 P allows, and the cache after the call bit for bit
  against the reference's (the append is a copy);
* ``out`` and the cache bit for bit against the built-in drafter's flashinfer calls in its order (the paged append,
  then ``dflash_trtllm_folded_block_attention``; the entry adds nothing);

blocks that take more than one folded launch are checked the same way, and one case replays a CUDA graph captured after
an eager call, bit for bit against eager. The op's buffer sizes, folded chunk plan and availability probe are checked
against the DFlash speculative runtime's (``speculative.dflash_attention``).
"""

import importlib
import zlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_dspark_block_attn requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("k3_dspark_block_attn is certified on sm_100 and sm_103", allow_module_level=True)

_attn = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.attention.k3_dspark_block_attn"
)
k3_dspark_block_attn = _attn.k3_dspark_block_attn

if _attn.unavailability_reason() is not None:
    pytest.skip(f"k3_dspark_block_attn: {_attn.unavailability_reason()}", allow_module_level=True)

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402

D = 64
PAGE = 64
# max |err| / max |ref|: the kernel rounds P and the output to bf16 (~4e-3 each), as for the k3_drafter kernel.
TOL = 1e-2
# Context lengths before the block, cycled over the requests: none, a page boundary, a block that crosses a page,
# several pages, a long context, a tile boundary.
LENGTHS = (0, 64, 60, 127, 1000, 5, 2041, 128)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a.double() - ref.double()).abs().max().item() / max(ref.abs().max().item(), 1e-6)


class _Case:
    """B requests of a T-row block over a paged HND cache [P, 2, KV, 64, 64], with the built-in drafter's argument
    tensors.

    ``slot_table`` plays the built-in drafter's ``ctx_page_table`` (one row of distinct random pages per slot,
    ``num_slots`` rows); ``slots`` plays ``ctx_cache_batch_idx``. Pages no slot owns are NaN; owned pages hold
    random rows, so a kernel that read past a request's length or into another request's pages would move the output.
    """

    def __init__(self, seed: int, heads: int, kv: int, batch: int, block: int, lengths=None):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        cpu_gen = torch.Generator().manual_seed(seed)
        self.heads, self.kv, self.b, self.t = heads, kv, batch, block
        self.num_slots = batch + 2
        self.lengths = (
            list(lengths)
            if lengths is not None
            else [LENGTHS[(seed + 3 * i) % len(LENGTHS)] for i in range(batch)]
        )
        # One page beyond the longest request, so a replay may grow a request.
        self.width = max((c + block + PAGE - 1) // PAGE for c in self.lengths) + 1
        n_pages = self.num_slots * self.width + 4
        self.cache = torch.full(
            (n_pages, 2, kv, PAGE, D), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        perm = torch.randperm(n_pages, generator=cpu_gen)[: self.num_slots * self.width]
        self.slot_table = perm.view(self.num_slots, self.width).to(torch.int32).cuda()
        owned = self.slot_table.flatten().long()
        self.cache[owned] = (
            torch.randn(owned.numel(), 2, kv, PAGE, D, generator=gen, device="cuda") * 0.5
        ).bfloat16()
        self.slots = (
            torch.randperm(self.num_slots, generator=cpu_gen)[:batch].to(torch.int32).cuda()
        )
        m = batch * block
        self.q = (torch.randn(m, heads, D, generator=gen, device="cuda") * 0.5).bfloat16()
        self.k = (torch.randn(m, kv, D, generator=gen, device="cuda") * 0.5).bfloat16()
        self.v = (torch.randn(m, kv, D, generator=gen, device="cuda") * 0.5).bfloat16()
        self.num_ctx = torch.tensor(self.lengths, dtype=torch.int32, device="cuda")
        self._buffers()
        self._metadata()

    def _buffers(self) -> None:
        """``_prepare_dflash_trtllm_gen_buffers`` and ``_prepare_dflash_index_buffers`` at max batch = num_slots: the
        workspace and counters at the larger of the block's tokens and a folded launch's heads."""
        ops = _attn.trtllm_gen_ops()
        max_batch_size = self.num_slots
        chunk = _attn.folded_decode_chunk_size(self.t, self.heads, self.kv)
        shapes = ((max_batch_size * self.t, self.heads), (max_batch_size, self.heads * chunk))
        workspace_bytes = max(
            ops.get_workspace_size(
                dtype=torch.bfloat16,
                num_tokens=tokens,
                num_gen_tokens=tokens,
                num_heads=heads,
                num_kv_heads=self.kv,
                head_size=D,
                max_num_requests=max_batch_size,
                rotary_embedding_dim=0,
                fp8_context_fmha=False,
            )
            for tokens, heads in shapes
        )
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        counter_bytes = ops.get_multi_ctas_kv_counter_size(
            max(heads for _, heads in shapes), max_batch_size, sm_count
        )
        self.counters = torch.zeros(counter_bytes, dtype=torch.uint8, device="cuda")
        self.block_offsets = torch.arange(self.t, dtype=torch.int32, device="cuda")
        self.batch_indices = (
            torch.arange(max_batch_size, dtype=torch.int32, device="cuda")
            .view(-1, 1)
            .expand(-1, self.t)
            .contiguous()
        )

    def _metadata(self) -> None:
        """``dflash_forward``'s paged-branch tensors, from ``num_ctx`` and ``slots`` (TRTLLM, non-causal, no window)."""
        b, t = self.b, self.t
        cache_seqlens = self.num_ctx[:b].to(torch.int32)
        self.seq_lens_after = cache_seqlens + t
        self.block_tables = self.slot_table.index_select(0, self.slots.long())
        pages_per_slot = self.block_tables.size(1)
        self.kv_indices = self.block_tables.flatten()
        self.kv_indptr = torch.arange(
            0, (b + 1) * pages_per_slot, pages_per_slot, dtype=torch.int32, device="cuda"
        )
        self.kv_last_page_len = ((self.seq_lens_after - 1) % PAGE) + 1
        self.append_batch_indices = self.batch_indices[:b].reshape(-1)
        self.append_positions = (
            (cache_seqlens.view(-1, 1) + self.block_offsets).reshape(-1).contiguous()
        )
        self.max_kv_len = pages_per_slot * PAGE

    def refresh(self, lengths, slots) -> None:
        """New lengths and slots, every metadata tensor rewritten in place (what a graph replay reads)."""
        old = {
            n: getattr(self, n)
            for n in (
                "seq_lens_after",
                "block_tables",
                "kv_indices",
                "kv_indptr",
                "kv_last_page_len",
                "append_batch_indices",
                "append_positions",
            )
        }
        width = self.max_kv_len
        self.lengths = list(lengths)
        self.num_ctx.copy_(torch.tensor(lengths, dtype=torch.int32))
        self.slots.copy_(torch.tensor(slots, dtype=torch.int32))
        self._metadata()
        assert self.max_kv_len == width
        for n, buf in old.items():
            buf.copy_(getattr(self, n))
            setattr(self, n, buf)

    def args(self, cache: torch.Tensor, out: torch.Tensor) -> tuple:
        return (
            self.q,
            self.k,
            self.v,
            cache,
            self.block_tables,
            self.append_batch_indices,
            self.append_positions,
            self.kv_indices,
            self.kv_indptr,
            self.kv_last_page_len,
            self.seq_lens_after,
            self.max_kv_len,
            self.workspace,
            self.counters,
            out,
        )

    def out(self) -> torch.Tensor:
        return torch.empty(self.b * self.t, self.heads, D, dtype=torch.bfloat16, device="cuda")


def _builtin_sequence(case: _Case, cache: torch.Tensor, out: torch.Tensor) -> None:
    """The flashinfer calls of the built-in ``dflash_forward`` (TRTLLM branch, unwindowed non-causal layer), in
    order: the paged append, then ``dflash_trtllm_folded_block_attention``."""
    import flashinfer

    from tensorrt_llm._torch.models.modeling_dflash import dflash_trtllm_folded_block_attention
    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_trtllm_gen_ops

    flashinfer.page.append_paged_kv_cache(
        append_key=case.k.reshape(-1, case.kv, D).contiguous(),
        append_value=case.v.reshape(-1, case.kv, D).contiguous(),
        batch_indices=case.append_batch_indices,
        positions=case.append_positions,
        paged_kv_cache=cache,
        kv_indices=case.kv_indices,
        kv_indptr=case.kv_indptr,
        kv_last_page_len=case.kv_last_page_len,
        kv_layout="HND",
    )
    dflash_trtllm_folded_block_attention(
        get_dflash_trtllm_gen_ops(),
        case.q.view(case.b, case.t, case.heads, D),
        cache,
        case.block_tables,
        case.seq_lens_after,
        case.max_kv_len,
        case.workspace,
        case.counters,
        out.view(-1, case.heads, D),
    )


@pytest.mark.parametrize(
    "cell", k3_dspark_block_attn.CELLS, ids=[c.why for c in k3_dspark_block_attn.CELLS]
)
def test_k3_dspark_block_attn(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    with torch.inference_mode():
        case = _Case(seed, spec["heads"], spec["kv"], spec["batch"], spec["block"])
        cache0 = case.cache.clone()

        cache, out = cache0.clone(), case.out()
        with validating(k3_dspark_block_attn):
            assert k3_dspark_block_attn(*case.args(cache, out)) is None

        cache_ref, want = cache0.clone(), case.out()
        k3_dspark_block_attn.reference(*case.args(cache_ref, want))
        assert torch.equal(_bits(cache), _bits(cache_ref)), (
            "the cache after the call is not the reference's"
        )
        err = _rel_err(out, want)
        assert err <= TOL, f"rel err {err:.3e}"

        cache_builtin, out_builtin = cache0.clone(), case.out()
        _builtin_sequence(case, cache_builtin, out_builtin)
        assert torch.equal(_bits(cache), _bits(cache_builtin)), (
            "the cache differs from the built-in drafter's flashinfer calls"
        )
        assert torch.equal(_bits(out), _bits(out_builtin)), (
            "out differs from the built-in drafter's flashinfer calls"
        )


# (heads, KV heads, requests, block rows) whose block takes more than one folded launch: 32 heads per KV head fold 4
# rows per launch (two launches, a short last one); 16 per KV head over 4 KV heads fold 8 (a one-row tail).
_MULTI_CHUNK = [(32, 1, 3, 7), (64, 4, 2, 9)]


@pytest.mark.parametrize("heads, kv, batch, block", _MULTI_CHUNK)
def test_k3_dspark_block_attn_multi_chunk(heads, kv, batch, block) -> None:
    """A block split over several folded launches: within the band of the reference, and bit for bit the built-in
    drafter's calls."""
    assert _attn.folded_decode_chunk_size(block, heads, kv) < block
    with torch.inference_mode():
        case = _Case(71 + heads, heads, kv, batch, block)
        cache0 = case.cache.clone()
        cache, out = cache0.clone(), case.out()
        with validating(k3_dspark_block_attn):
            k3_dspark_block_attn(*case.args(cache, out))
        want = case.out()
        k3_dspark_block_attn.reference(*case.args(cache0.clone(), want))
        err = _rel_err(out, want)
        assert err <= TOL, f"rel err {err:.3e}"
        cache_builtin, out_builtin = cache0.clone(), case.out()
        _builtin_sequence(case, cache_builtin, out_builtin)
        assert torch.equal(_bits(cache), _bits(cache_builtin))
        assert torch.equal(_bits(out), _bits(out_builtin))


def test_k3_dspark_block_attn_graph_replay() -> None:
    """Captured after an eager call, replayed on the eager call's inputs and then on rewritten q / k / v, lengths and
    slots: every replay bit for bit an eager call on the same inputs, out and cache."""
    with torch.inference_mode():
        case = _Case(53, 4, 1, 8, 7, lengths=(60, 0, 1000, 64, 127, 2041, 5, 128))
        cache0 = case.cache.clone()
        graph_cache, graph_out = cache0.clone(), case.out()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            k3_dspark_block_attn(*case.args(graph_cache, graph_out))  # eager, outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                k3_dspark_block_attn(*case.args(graph_cache, graph_out))
        torch.cuda.synchronize()

        gen = torch.Generator(device="cuda").manual_seed(54)
        rounds = [
            None,
            ((2041, 5, 128, 60, 0, 1000, 64, 127), (9, 8, 7, 6, 5, 4, 3, 2)),
            ((0, 0, 64, 64, 1, 63, 1900, 300), (1, 3, 5, 7, 9, 0, 2, 4)),
        ]
        for rnd, change in enumerate(rounds):
            if change is not None:
                case.refresh(*change)
                for x in (case.q, case.k, case.v):
                    x.copy_((torch.randn(x.shape, generator=gen, device="cuda") * 0.5).bfloat16())
            graph_cache.copy_(cache0)
            graph.replay()
            torch.cuda.synchronize()
            eager_cache, eager_out = cache0.clone(), case.out()
            k3_dspark_block_attn(*case.args(eager_cache, eager_out))
            assert torch.equal(_bits(graph_cache), _bits(eager_cache)), f"round {rnd}: cache"
            assert torch.equal(_bits(graph_out), _bits(eager_out)), f"round {rnd}: out"
            want = case.out()
            k3_dspark_block_attn.reference(*case.args(cache0.clone(), want))
            assert _rel_err(graph_out, want) <= TOL, f"round {rnd}"


_OUT_OF_CONTRACT = {
    "fp16": lambda c: dict(q=c.q.half(), k=c.k.half(), v=c.v.half()),
    "non-contiguous k": lambda c: dict(k=c.k.repeat(1, 2, 1)[:, ::2]),
    "uneven block rows": lambda c: dict(q=c.q[:-1]),
    "more query heads per KV head than the generation kernels take": lambda c: dict(
        q=c.q.repeat(1, 9, 1)
    ),
}


@pytest.mark.parametrize("case_name", list(_OUT_OF_CONTRACT))
def test_rejects_out_of_contract(case_name) -> None:
    """The guard refuses the inputs it lists (it raises before the kernel runs)."""
    with torch.inference_mode():
        case = _Case(5, 4, 1, 2, 7)
        for name, value in _OUT_OF_CONTRACT[case_name](case).items():
            setattr(case, name, value)
        with validating(k3_dspark_block_attn), pytest.raises(AssertionError):
            k3_dspark_block_attn(*case.args(case.cache, case.out()))


# (heads, KV heads, requests, block rows): the drafters' layouts at the batch sizes the worker prepares, and the
# generation-only and context-larger corners of the two workspace layouts.
_SIZES = [(4, 1, 1, 7), (6, 1, 8, 7), (6, 1, 16, 7), (4, 1, 64, 8), (96, 16, 3, 1), (8, 8, 1, 0)]


@pytest.mark.parametrize("heads, kv, batch, block", _SIZES)
def test_buffer_sizes_match_the_dflash_runtime(heads, kv, batch, block) -> None:
    """The op's workspace and counter sizes equal the DFlash speculative runtime's (``get_dflash_trtllm_gen_ops``,
    which reads them from the TRTLLM-Gen attention backend), at the arguments the drafter passes."""
    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_trtllm_gen_ops

    builtin = get_dflash_trtllm_gen_ops()
    ops = _attn.trtllm_gen_ops()
    kwargs = dict(
        dtype=torch.bfloat16,
        num_tokens=batch * block,
        num_gen_tokens=batch * block,
        num_heads=heads,
        num_kv_heads=kv,
        head_size=D,
        max_num_requests=batch,
        rotary_embedding_dim=0,
        fp8_context_fmha=False,
    )
    assert ops.get_workspace_size(**kwargs) == builtin.get_workspace_size(**kwargs)
    for sm_count in (1, 148, 1000):
        assert ops.get_multi_ctas_kv_counter_size(
            heads, batch, sm_count
        ) == builtin.get_multi_ctas_kv_counter_size(heads, batch, sm_count)
    assert ops.append_paged_kv_cache is builtin.append_paged_kv_cache
    assert ops.batch_decode_with_kv_cache is builtin.batch_decode_with_kv_cache


@pytest.mark.parametrize(
    "block, heads, kv", [(7, 4, 1), (8, 6, 1), (1, 96, 16), (7, 32, 1), (9, 64, 4), (200, 2, 2)]
)
def test_folded_chunk_plan_matches_the_dflash_runtime(block, heads, kv) -> None:
    """The op's folded chunk plan and its limits equal the DFlash speculative runtime's."""
    from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
        FlashInferTrtllmGenFmha,
    )
    from tensorrt_llm._torch.speculative import dflash_attention

    assert _attn.LOWERING == dflash_attention.DFLASH_FOLDED_DECODE_LOWERING
    assert _attn.MAX_HEADS_RATIO_GENERATION == FlashInferTrtllmGenFmha.MAX_HEADS_RATIO_GENERATION
    assert (
        _attn.FOLDED_DECODE_MAX_RATIO_PER_LAUNCH
        == dflash_attention.DFLASH_FOLDED_DECODE_MAX_RATIO_PER_LAUNCH
    )
    assert _attn.folded_decode_chunk_size(
        block, heads, kv
    ) == dflash_attention.get_dflash_folded_decode_chunk_size(block, heads, kv)
    for bad in ((0, 4, 1), (7, 6, 4), (7, 66, 1)):
        with pytest.raises(RuntimeError):
            _attn.folded_decode_chunk_size(*bad)
        with pytest.raises(RuntimeError):
            dflash_attention.get_dflash_folded_decode_chunk_size(*bad)


def test_unavailability_reason_matches_the_dflash_runtime(monkeypatch) -> None:
    """The op's availability probe answers as the DFlash speculative runtime's on this build, and names the SM it
    refuses."""
    from tensorrt_llm._torch.speculative.dflash_attention import (
        dflash_trtllm_gen_unavailability_reason,
    )

    assert _attn.unavailability_reason() == dflash_trtllm_gen_unavailability_reason()
    monkeypatch.setattr(_attn, "get_sm_version", lambda: 90)
    assert _attn.unavailability_reason() == "requires SM100 or SM103, got SM90"
