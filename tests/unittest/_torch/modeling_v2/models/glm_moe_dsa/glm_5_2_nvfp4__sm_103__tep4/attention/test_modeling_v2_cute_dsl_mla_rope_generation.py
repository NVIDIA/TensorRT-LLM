# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mla_rope_generation entry.

The op takes TRT-LLM's serving ABI: pool pointers, a pool mapping and a block
table from a KV cache manager, and per-sequence lengths from prepared
attention metadata. The test builds both for real (an MLA SELFKONLY manager
over an fp8 latent pool, and a prepared TrtllmAttentionMetadata), drives the
target's operand layout (`q_pe` the rope half of a `[T, 16, 256]` query,
the latent the tail of a `[T, 2624]` projection output), and checks every
buffer the call writes against the entry's reference.
"""

import inspect
import zlib
from typing import List

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mla_rope_generation as entry,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams  # noqa: E402
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata  # noqa: E402
from tensorrt_llm._torch.metadata import KVCacheParams  # noqa: E402
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager  # noqa: E402
from tensorrt_llm.bindings import DataType  # noqa: E402
from tensorrt_llm.bindings.internal.batch_manager import CacheType  # noqa: E402
from tensorrt_llm.llmapi.llm_args import KvCacheConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_mla_rope_generation

NUM_HEADS = 16
KV_LORA_RANK = 512
ROPE = 64
NOPE = 192
Q_LORA_RANK = 2048
HEAD_SIZE = KV_LORA_RANK + ROPE
QK_DIM = NOPE + ROPE
#: The kv_a projection's output width: q_lora + kv_lora + rope.
QKVA_WIDTH = Q_LORA_RANK + HEAD_SIZE
#: GLM-5.2's rope base: which base the table was built at shows in every roped byte.
ROPE_THETA = 8_000_000.0
MAX_SEQ_LEN = 1024
QUANT_MODE_FP8_KV_CACHE = 128

#: One e4m3 step is 2**-3 of the value. The roped halves can differ from the
#: reference by one step: both evaluate `x*cos -+ y*sin` in fp32-or-better and
#: round to bf16, and a contraction the kernel's compiler chooses can move a
#: value across a bf16 rounding midpoint, which e4m3 then rounds to a
#: neighbouring step on rare elements. The fraction gate keeps that rare: a
#: wrong pairing, angle or position moves most bytes.
E4M3_STEP = 2**-3
MAX_INEXACT_FRACTION = 1e-3


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Bit-exact, for the parts that are a single bf16 -> e4m3 conversion."""
    assert got.shape == want.shape, (what, got.shape, want.shape)
    diff = int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())
    assert diff == 0, f"{what}: {diff}/{got.numel()} bytes differ"


def _assert_roped(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """One e4m3 step per element, and at most 1e-3 of the bytes off bit-exact."""
    assert got.shape == want.shape, (what, got.shape, want.shape)
    torch.testing.assert_close(got.float(), want.float(), rtol=E4M3_STEP, atol=0.0, msg=what)
    diff = int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())
    allowed = max(1, int(MAX_INEXACT_FRACTION * got.numel()))
    assert diff <= allowed, f"{what}: {diff}/{got.numel()} bytes differ (allowed {allowed})"


class _MlaEnv:
    """An MLA KV cache manager over an fp8 latent pool, and the rope table."""

    def __init__(self, max_batch_size: int, num_layers: int) -> None:
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=65536, enable_block_reuse=False),
            CacheType.SELFKONLY,  # MLA latent cache: kv_factor=1, one kv head
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=HEAD_SIZE,
            tokens_per_block=entry.TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8,
        )
        assert self.kv_cache_manager.tokens_per_block == entry.TOKENS_PER_BLOCK
        # The duplicated-layout fp32 (cos, sin) table the MLA backend builds.
        rope = RopeParams(
            dim=ROPE, theta=ROPE_THETA, max_positions=MAX_SEQ_LEN, duplicate_data=True
        )
        _, self.rotary_cos_sin = rope.create_rope_const_params()

    def prepare_metadata(self, request_ids, seq_lens, cached_lens) -> TrtllmAttentionMetadata:
        md = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=8192,
            kv_cache_manager=self.kv_cache_manager,
        )
        md.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        md.num_contexts = 0
        md.request_ids = request_ids
        md.prompt_lens = cached_lens
        md.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens)
        md.prepare()
        return md

    def layer(self, layer_idx: int) -> torch.Tensor:
        pool = self.kv_cache_manager.get_buffers(layer_idx)
        assert pool is not None and pool.dtype == torch.float8_e4m3fn
        return pool

    def cache_rows(self, layer_idx: int, request_ids: List[int], positions: List[int]):
        """`([T, 576] rows, {(page, offset)} slots)` the tokens' positions address."""
        tpb = entry.TOKENS_PER_BLOCK
        pool = self.layer(layer_idx)
        out, slots = [], set()
        for rid, pos in zip(request_ids, positions):
            page = self.kv_cache_manager.get_batch_cache_indices([rid])[0][pos // tpb]
            out.append(pool[page, 0, pos % tpb, 0])
            slots.add((page, pos % tpb))
        return torch.stack(out), slots

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    spec = cell.spec
    assert (spec["num_heads"], spec["kv_lora_rank"], spec["qk_rope_head_dim"]) == (
        NUM_HEADS,
        KV_LORA_RANK,
        ROPE,
    )
    assert (spec["qk_nope_head_dim"], spec["latent_row_stride"]) == (NOPE, QKVA_WIDTH)
    p, n, layer = spec["predicted_tokens_per_seq"], spec["num_seqs"], spec["layer_idx"]
    cached = spec["cached"]
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    dev = torch.cuda.current_device()

    env = _MlaEnv(max_batch_size=max(4, n), num_layers=spec["num_layers"])
    try:
        ids = list(range(n))
        env.kv_cache_manager.add_dummy_requests(ids, token_nums=cached)
        for rid in ids:
            for _ in range(p):
                env.kv_cache_manager.impl.add_token(rid)
        md = env.prepare_metadata(ids, [p] * n, cached)
        rows = n * p

        # The target's operand layout: q_pe is the rope half of the q_b output,
        # the latent the tail of the kv_a output.
        q = torch.randn(rows, NUM_HEADS, QK_DIM, dtype=torch.bfloat16, device="cuda")
        q_pe = q.split([NOPE, ROPE], -1)[1]
        qkva = torch.randn(rows, QKVA_WIDTH, dtype=torch.bfloat16, device="cuda")
        latent = qkva[:, Q_LORA_RANK:]
        fused_q = torch.randn(rows, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
        inputs = (q.clone(), qkva.clone(), fused_q.clone())
        # Sentinels: every buffer starts at a value the op must overwrite or
        # must leave alone. The target sizes cu_* by token, not by sequence.
        quant_q = torch.full((rows, NUM_HEADS, HEAD_SIZE), 0x7F, dtype=torch.uint8, device="cuda")
        quant_q = quant_q.view(torch.float8_e4m3fn)
        cu_q = torch.full((rows + 1,), -1, dtype=torch.int32, device="cuda")
        cu_kv = torch.full((rows + 1,), -1, dtype=torch.int32, device="cuda")
        counter = torch.full((1,), 7, dtype=torch.uint32, device="cuda")
        bmm1 = torch.full((2,), -99.0, dtype=torch.float32, device="cuda")
        bmm2 = torch.full((1,), -99.0, dtype=torch.float32, device="cuda")

        compiled = entry.compile(
            max_blocks_per_seq=int(md.kv_cache_block_offsets.shape[-1]),
            q_pe_head_stride=q_pe.stride(1),
            q_pe_token_stride=q_pe.stride(0),
            latent_row_stride=latent.stride(0),
            device_index=dev,
        )
        # The target's positional call, argument for argument.
        args = (
            fused_q, q_pe, latent, env.rotary_cos_sin, cu_q, cu_kv, counter, bmm1, bmm2,
            quant_q, md.kv_lens_cuda_runtime, md.kv_lens_runtime, md.prompt_lens_cpu_runtime,
            0, md.kv_cache_block_offsets, env.kv_cache_manager.kv_cache_pool_pointers,
            env.kv_cache_manager.kv_cache_pool_mapping, None, None, None, None, [None, None], p,
            layer, NUM_HEADS, 1, HEAD_SIZE, entry.TOKENS_PER_BLOCK, MAX_SEQ_LEN, 1,
            QUANT_MODE_FP8_KV_CACHE, 1.0, Q_LORA_RANK, KV_LORA_RANK, NOPE, ROPE, KV_LORA_RANK,
            True,
        )  # fmt: skip
        before = [env.layer(i).clone() for i in range(env.num_layers)]

        op(*args, compiled=compiled)
        torch.cuda.synchronize()
        ref = op.reference(*args, compiled=compiled)

        # Read-only inputs come back untouched.
        for got, was in zip((q, qkva, fused_q), inputs):
            assert torch.equal(got, was), "an input the op only reads was written"

        # The query: the absorbed half is one bf16 -> e4m3 conversion, the
        # rope half goes through the rotation.
        _assert_bytes_equal(
            quant_q[..., :KV_LORA_RANK], ref["quant_q_buffer"][..., :KV_LORA_RANK], "q base"
        )
        _assert_roped(
            quant_q[..., KV_LORA_RANK:], ref["quant_q_buffer"][..., KV_LORA_RANK:], "q rope"
        )

        # The appended rows, read back through the manager's block table at
        # the positions the reference states, and nothing else in any layer.
        positions = ref["positions"].tolist()
        want_positions = [cached[g] + t for g in range(n) for t in range(p)]
        assert positions == want_positions, (positions, want_positions)
        owners = [g for g in range(n) for _ in range(p)]
        got_rows, slots = env.cache_rows(layer, owners, positions)
        _assert_bytes_equal(
            got_rows[:, :KV_LORA_RANK], ref["latent_cache"][:, :KV_LORA_RANK], "kv latent"
        )
        _assert_roped(got_rows[:, KV_LORA_RANK:], ref["latent_cache"][:, KV_LORA_RANK:], "kv rope")
        for i in range(env.num_layers):
            changed = env.layer(i).view(torch.uint8) != before[i].view(torch.uint8)
            idx = changed.nonzero().cpu()
            touched = set(zip(idx[:, 0].tolist(), idx[:, 2].tolist()))
            # A subset, not equality: a byte rewritten with its old value is
            # invisible to a diff. The rows' contents are pinned above.
            assert touched <= (slots if i == layer else set()), (
                f"layer {i}: the op touched slots {sorted(touched - slots)}"
            )

        # Scheduler buffers: G + 1 entries written, the rest left alone.
        g1 = n + 1
        assert torch.equal(cu_q[:g1], ref["cu_q_seqlens"]), (cu_q[:g1], ref["cu_q_seqlens"])
        assert torch.equal(cu_kv[:g1], ref["cu_kv_seqlens"]), (cu_kv[:g1], ref["cu_kv_seqlens"])
        assert bool((cu_q[g1:] == -1).all()) and bool((cu_kv[g1:] == -1).all())
        assert int(counter.item()) == int(ref["fmha_scheduler_counter"].item())
        # The scales are compiled constants converted to fp32 once, as the
        # reference rounds its double expression once: bit-exact.
        assert torch.equal(bmm1, ref["mla_bmm1_scale"]), (bmm1, ref["mla_bmm1_scale"])
        assert torch.equal(bmm2, ref["mla_bmm2_scale"]), (bmm2, ref["mla_bmm2_scale"])
    finally:
        env.shutdown()


def _guard_args(**overrides) -> dict:
    """Every `is_valid` parameter, set to a call that passes, with the named
    ones replaced. Read off the signature, so a parameter added to the op
    cannot shift a hand-counted list."""
    args = dict.fromkeys(inspect.signature(op.is_valid).parameters)
    q = torch.zeros(4, NUM_HEADS, QK_DIM, dtype=torch.bfloat16, device="cuda")
    args.update(
        q_pe=q.split([NOPE, ROPE], -1)[1],
        rotary_cos_sin=torch.zeros(MAX_SEQ_LEN * ROPE * 2, dtype=torch.float32, device="cuda"),
        sequence_length=torch.tensor([40, 700], dtype=torch.int32, device="cuda"),
        host_context_lengths=torch.zeros(2, dtype=torch.int32),
        num_contexts=0,
        kv_cache_block_offsets=torch.zeros(1, 2, 2, 32, dtype=torch.int32, device="cuda"),
        predicted_tokens_per_seq=2,
        tokens_per_block=entry.TOKENS_PER_BLOCK,
        attention_window_size=MAX_SEQ_LEN,
        qk_rope_head_dim=ROPE,
    )
    args.update(overrides)
    return args


def test_is_valid_admits_a_well_formed_call() -> None:
    op.is_valid(**_guard_args())


def test_is_valid_refuses_a_sequence_shorter_than_its_new_tokens() -> None:
    """Position -1 is clamped to 0: the row overwrites the sequence's first token."""
    lens = torch.tensor([1, 700], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="predicted_tokens_per_seq"):
        op.is_valid(**_guard_args(sequence_length=lens))


def test_is_valid_refuses_a_position_past_the_rope_table() -> None:
    table = torch.zeros(512 * ROPE * 2, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="rotary_cos_sin"):
        op.is_valid(**_guard_args(rotary_cos_sin=table))


def test_is_valid_refuses_a_page_past_the_block_table() -> None:
    """The kernel skips that KV write and reports nothing."""
    narrow = torch.zeros(1, 2, 2, 16, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="block table"):
        op.is_valid(**_guard_args(kv_cache_block_offsets=narrow))


def test_is_valid_refuses_a_window_below_the_sequence() -> None:
    with pytest.raises(AssertionError, match="attention_window_size"):
        op.is_valid(**_guard_args(attention_window_size=512))


def test_is_valid_refuses_a_misaligned_q_pe() -> None:
    """A 16-byte vector load from an unaligned address faults the context."""
    q = torch.zeros(4, NUM_HEADS, QK_DIM + 4, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="aligned"):
        op.is_valid(**_guard_args(q_pe=q[..., NOPE + 4 : NOPE + 4 + ROPE]))
