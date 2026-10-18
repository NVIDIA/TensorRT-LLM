# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: ``drafter_block_decode`` over the DFlash worker's windowed context arena.

The windowed arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA=1``) keeps only the last
``keep`` context columns of each slot: column ``c`` holds the logical token
``_ctx_base + c``, and after each step's accepted tokens are written the
built-in ``DFlashWorker._compact_ctx_window`` shifts every slot left so at most
``keep`` columns remain. The draft forward then gets the physical length
``_ctx_len - _ctx_base`` and the logical query positions.

The written-out block decode reads only that physical length: it appends the
block at ``[phys, phys + M)`` and attends over the window ending there, whose
first key is ``phys + M - 1 - window_left`` -- the same tokens, in the same
order and split into the same 32-key tiles, as on the full-length arena with
``ctx_len`` in place of ``phys``. So over several steps that cross
compactions, every request's output and its arena columns are bitwise the
full-length arena's, eagerly and under CUDA-graph replay.

The arenas are the worker's own layout ``[slots, layers, columns, kv heads,
width]``; the compaction is the built-in method on a stand-in carrying the
attributes it reads. Two requests (one starting below the window and crossing
it, one starting past it) and a padding row on the dummy slot.

Needs ``MUSESPARK_REFERENCE_DRAFTER`` (the published drafter checkpoint) and
flash-attn (the drafter's built-in VANILLA attention).
"""

import os
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

assert torch.cuda.is_available(), "the windowed arena test requires a CUDA device"

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")

#: K drafts per step; the block is the last accepted token and the K drafts.
K = 7
BLOCK = K + 1
#: The two requests' slots, their starting contexts and the dummy slot of the padding row.
SLOTS = (1, 2)
STARTS = (2_020, 2_060)
DUMMY = 0
#: Accepted tokens per step for each request: full, partial and single acceptances.
ACCEPTED = ((8, 3), (8, 8), (3, 1), (8, 5), (8, 8), (1, 8), (5, 2), (8, 8))


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _DRAFTER_DIR or not _has_flash_attn(),
    reason="needs MUSESPARK_REFERENCE_DRAFTER and flash_attn",
)


@pytest.fixture(scope="module")
def drafter():
    """The published drafter (TP1) with the served drafter paths: block conv + block decode."""
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda"):
        model = DFlashForCausalLM(ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM"))
    model.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert model.has_block_conv and model.dflash_attention_backend == "VANILLA"
    draft.install(model, True, True)
    assert model.dflash_forward.__func__ is draft.windowed_block_decode
    return model


class _Arenas:
    """A full-length arena and the worker's windowed arena, fed the same tokens."""

    def __init__(self, model) -> None:
        self.model = model
        # The arena geometry is set when the drafter builds its fused K/V buffers (lazily).
        if model._fused_kv_weight is None:
            model._build_fused_kv_buffers()
        layers, heads, width = model._num_attn_layers, model._num_kv_heads, model._head_dim
        assert layers and heads and width, (layers, heads, width)
        steps = sum(max(a) for a in ACCEPTED)
        self.max_ctx = max(STARTS) + steps + BLOCK
        # The built-in keep over the drafter's decoder layers: the widest left window plus one.
        probe = types.SimpleNamespace(windowed_ctx_arena=True, _max_ctx=self.max_ctx)
        decoder_layers = len(model.model.layers)
        self.keep = DFlashWorker._windowed_ctx_keep(probe, model, decoder_layers)
        windows = [model._resolve_block_attention(i)[1] for i in range(decoder_layers)]
        assert 0 < self.keep < min(STARTS) + 64, (self.keep, layers, windows)
        capacity = self.keep + K + 1 + BLOCK
        slots = max(SLOTS) + 1

        def buffers(columns):
            shape = (slots, layers, columns, heads, width)
            return torch.zeros(shape, dtype=torch.bfloat16, device="cuda"), torch.zeros(
                shape, dtype=torch.bfloat16, device="cuda"
            )

        self.full_k, self.full_v = buffers(self.max_ctx)
        win_k, win_v = buffers(capacity)
        self.worker = types.SimpleNamespace(
            _ctx_k_buf=win_k,
            _ctx_v_buf=win_v,
            _ctx_len=torch.zeros(slots, dtype=torch.long, device="cuda"),
            _ctx_base=torch.zeros(slots, dtype=torch.long, device="cuda"),
            _ctx_keep=self.keep,
        )
        self.gen = torch.Generator(device="cuda").manual_seed(2026)
        self.all_slots = torch.tensor((*SLOTS, DUMMY), dtype=torch.long, device="cuda")
        self.shifted = [0] * len(SLOTS)
        for slot, start in zip(SLOTS, STARTS):
            self.append(slot, start)
        self.compact()
        # Shifts counted from here on: the compactions the steps cause.
        self.shifted = [0] * len(SLOTS)

    @property
    def ctx_len(self):
        return self.worker._ctx_len

    @property
    def base(self):
        return self.worker._ctx_base

    def append(self, slot: int, count: int) -> None:
        """``count`` accepted tokens of ``slot``: K/V at their logical positions, both arenas."""
        model = self.model
        start = int(self.ctx_len[slot])
        captured = torch.randn(
            count,
            model.config.hidden_size * len(model.target_layer_ids),
            generator=self.gen,
            device="cuda",
        ).to(torch.bfloat16)
        positions = torch.arange(start, start + count, device="cuda")
        k, v = model.precompute_context_kv(model.project_target_hidden(captured), positions)
        phys = start - int(self.base[slot])
        assert phys + count <= self.worker._ctx_k_buf.size(2), "the windowed arena overflowed"
        self.full_k[slot, :, start : start + count] = k.permute(1, 0, 2, 3)
        self.full_v[slot, :, start : start + count] = v.permute(1, 0, 2, 3)
        self.worker._ctx_k_buf[slot, :, phys : phys + count] = k.permute(1, 0, 2, 3)
        self.worker._ctx_v_buf[slot, :, phys : phys + count] = v.permute(1, 0, 2, 3)
        self.ctx_len[slot] += count

    def compact(self) -> None:
        """The built-in compaction over every slot of the batch, the dummy slot included."""
        before = self.base.clone()
        DFlashWorker._compact_ctx_window(self.worker, self.all_slots)
        for i, slot in enumerate(SLOTS):
            self.shifted[i] += int(self.base[slot] - before[slot])
        for slot in SLOTS:
            assert int(self.ctx_len[slot] - self.base[slot]) <= self.keep

    def step(self, accepted) -> None:
        for slot, count in zip(SLOTS, accepted):
            self.append(slot, count)
        self.compact()

    def assert_same_columns(self, what: str) -> None:
        """Every request's windowed columns, through the block just appended, are the full arena's."""
        for slot in SLOTS:
            base, end = int(self.base[slot]), int(self.ctx_len[slot]) + BLOCK
            for name, full, win in (
                ("K", self.full_k, self.worker._ctx_k_buf),
                ("V", self.full_v, self.worker._ctx_v_buf),
            ):
                assert torch.equal(win[slot, :, : end - base], full[slot, :, base:end]), (
                    f"{what}: slot {slot} arena {name} differs"
                )


class _Inputs:
    """Static draft-forward inputs for one arena (CUDA-graph safe: refilled in place)."""

    def __init__(self, model, k_cache, v_cache) -> None:
        rows = len(SLOTS) + 1
        self.k_cache, self.v_cache = k_cache, v_cache
        self.noise = torch.empty(
            rows, BLOCK, model.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        self.positions = torch.zeros(rows, BLOCK, dtype=torch.long, device="cuda")
        self.num_ctx = torch.zeros(rows, dtype=torch.long, device="cuda")
        self.batch_idx = torch.tensor((*SLOTS, DUMMY), dtype=torch.long, device="cuda")

    def fill(self, noise, positions, num_ctx) -> None:
        self.noise.copy_(noise)
        self.positions.copy_(positions)
        self.num_ctx.copy_(num_ctx)

    def forward(self, model) -> torch.Tensor:
        return model.dflash_forward(
            noise_embedding=self.noise,
            query_positions=self.positions,
            num_ctx_per_req=self.num_ctx,
            ctx_k_cache=self.k_cache,
            ctx_v_cache=self.v_cache,
            ctx_cache_batch_idx=self.batch_idx,
        )


def _step_inputs(arenas: _Arenas, model, seed: int):
    """Noise, logical query positions and each arena's context length for this step."""
    rows = len(SLOTS) + 1
    noise = torch.randn(
        rows,
        BLOCK,
        model.config.hidden_size,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        device="cuda",
    ).to(torch.bfloat16)
    lengths = arenas.ctx_len[arenas.all_slots]
    positions = lengths.unsqueeze(1) + torch.arange(BLOCK, device="cuda").unsqueeze(0)
    physical = lengths - arenas.base[arenas.all_slots]
    return noise, positions, lengths, physical


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_block_decode_over_the_windowed_arena_is_bitwise_the_full_arena_across_compactions(
    drafter, graph
):
    arenas = _Arenas(drafter)
    full = _Inputs(drafter, arenas.full_k, arenas.full_v)
    windowed = _Inputs(drafter, arenas.worker._ctx_k_buf, arenas.worker._ctx_v_buf)
    real = len(SLOTS)
    replay = None
    if graph:
        noise, positions, lengths, physical = _step_inputs(arenas, drafter, 0)
        windowed.fill(noise, positions, physical)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                windowed.forward(drafter)
        torch.cuda.current_stream().wait_stream(side)
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            static_out = windowed.forward(drafter)

    for index, accepted in enumerate(ACCEPTED):
        arenas.step(accepted)
        noise, positions, lengths, physical = _step_inputs(arenas, drafter, 100 + index)
        full.fill(noise, positions, lengths)
        expected = full.forward(drafter).clone()
        windowed.fill(noise, positions, physical)
        if replay is not None:
            replay.replay()
            got = static_out.clone()
        else:
            got = windowed.forward(drafter).clone()
        torch.cuda.synchronize()
        what = f"step {index} ctx {lengths.tolist()} phys {physical.tolist()}"
        rows = real * BLOCK
        assert torch.equal(got[:rows], expected[:rows]), (
            f"{what}: output differs in {(got[:rows] != expected[:rows]).sum().item()} elements"
        )
        arenas.assert_same_columns(what)

    # Every request crossed the window: its arena was compacted at least once after the start.
    assert all(int(arenas.base[slot]) > 0 for slot in SLOTS), arenas.base.tolist()
    assert all(shift > 0 for shift in arenas.shifted), arenas.shifted
