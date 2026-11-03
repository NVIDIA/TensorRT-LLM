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
"""One DSpark draft step of DeepSeek-V4.1-Flash against its plain path: the acceptance with the
slot-state advance in one launch, and the draft attention with its K/V half on a side stream and the
q RoPE in place."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100/SM103 with CuTe DSL required",
)

HIDDEN, CAPTURES = 5120, 3
HEADS, HEAD_DIM, ROPE_DIM, Q_RANK, GROUPS, O_RANK, WINDOW = 64, 512, 64, 1280, 8, 1024, 128
# The checkpoint's square FP8 weight blocks.
BLOCK = 32


def _checkpoint_fp8(gen, rows, cols):
    """An E4M3 weight with square ``BLOCK`` UE8M0 scales, as the checkpoint stores it."""
    weight = (torch.randn(rows, cols, device="cuda", generator=gen) * 64).to(torch.float8_e4m3fn)
    exponents = torch.randint(
        110, 122, (rows // BLOCK, cols // BLOCK), device="cuda", generator=gen
    )
    return weight, exponents.to(torch.uint8).view(torch.float8_e8m0fnu)


def _draft_model(gen):
    """Stage 0 of the V4.1 draft on an attention-DP rank, loaded from FP8 checkpoint tensors."""
    from tensorrt_llm._torch.models.modeling_dspark_v41 import DSv41DSparkDraftModel
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    src = {"mtp.0.attn.attn_sink": torch.randn(HEADS, device="cuda", generator=gen) * 0.1}
    for name, shape in {
        "attn.wq_a": (Q_RANK, HIDDEN),
        "attn.wq_b": (HEADS * HEAD_DIM, Q_RANK),
        "attn.wkv": (HEAD_DIM, HIDDEN),
        "attn.wo_a": (GROUPS * O_RANK, HEADS * HEAD_DIM // GROUPS),
        "attn.wo_b": (HIDDEN, GROUPS * O_RANK),
        "main_proj": (HIDDEN, CAPTURES * HIDDEN),
    }.items():
        src[f"mtp.0.{name}.weight"], src[f"mtp.0.{name}.scale"] = _checkpoint_fp8(gen, *shape)
    src["mtp.0.attn.q_norm.weight"] = torch.ones(Q_RANK, device="cuda")
    src["mtp.0.attn.kv_norm.weight"] = torch.ones(HEAD_DIM, device="cuda")

    model = DSv41DSparkDraftModel.__new__(DSv41DSparkDraftModel)
    nn.Module.__init__(model)
    model.model_config = SimpleNamespace(
        mapping=Mapping(world_size=4, tp_size=4, rank=0, gpus_per_node=4, enable_attention_dp=True)
    )
    model.checkpoint_fp8_block_size = BLOCK
    stage = nn.Module()
    stage.input_layernorm = RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16)
    stage.main_proj = nn.Linear(CAPTURES * HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)
    stage.main_proj.weight.copy_(
        model._block_dequant(src["mtp.0.main_proj.weight"], src["mtp.0.main_proj.scale"], BLOCK)
    )
    stage.main_norm = RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16)
    stage.main_norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda"))
    model.mtp_layers = nn.ModuleList([stage]).cuda()
    model._bind_bf16_attn_projections(src)
    return model


def _accept(gen, gens, draft_len):
    """The acceptance of the previous drafts with the slot-state advance in one launch equals
    ``dspark_strict_accept`` + ``dspark_generation_prepare``; returns the bonus rows."""
    from tensorrt_llm._torch.attention.kernels.blackwell.dspark.generation_prepare import (
        dspark_accept_prepare,
        dspark_generation_prepare,
        dspark_strict_accept,
    )

    width, num_slots, cap = draft_len + 1, gens + 8, 4096
    scratch, rows = num_slots - 1, gens * width
    target = torch.randint(0, 50, (rows,), dtype=torch.int64, device="cuda", generator=gen)
    draft = target.view(gens, width)[:, :draft_len].clone().to(torch.int32)
    for g in range(gens):
        draft[g, g % width :] += 1
    slots = torch.randperm(num_slots - 1, device="cuda", generator=gen)[:gens]
    # Two padding requests share the scratch slot.
    slots[-2:] = scratch
    state = (
        torch.randint(0, cap + 10, (num_slots,), device="cuda", generator=gen),
        torch.randint(0, WINDOW + 1, (num_slots,), device="cuda", generator=gen),
        torch.rand(num_slots, device="cuda", generator=gen) < 0.5,
    )
    positions = torch.randint(0, cap + 10, (rows,), dtype=torch.int32, device="cuda", generator=gen)
    two, one = [t.clone() for t in state], [t.clone() for t in state]
    common = dict(gen_start=0, num_rows=rows, scratch_slot=scratch, window=WINDOW, position_cap=cap)
    accepted = dspark_strict_accept(target, draft)
    want = (
        *accepted,
        *dspark_generation_prepare(
            *accepted, positions, slots, *two, draft_len=draft_len, **common
        ),
    )
    got = dspark_accept_prepare(target, draft, positions, slots, *one, **common)
    racy = slots == scratch
    for index, (g, w) in enumerate(zip(got, want)):
        assert g.dtype == w.dtype and g.shape == w.shape
        if index == 5:  # the window lengths re-read after the writes
            g, w = g[~racy], w[~racy]
        assert torch.equal(g, w), index
    for g, w in zip(one, two):
        g[scratch] = w[scratch]
        assert torch.equal(g, w)
    return got[3]


@pytest.mark.parametrize("gens,draft_len", [(25, 5), (320, 3)], ids=["mid", "hi"])
@torch.inference_mode()
def test_dspark_draft_step(monkeypatch, gens, draft_len):
    """Bitwise against the plain path: the one-launch acceptance and the K/V side stream with the
    in-place q RoPE."""
    from tensorrt_llm._torch.models import modeling_dspark as dspark

    gen = torch.Generator(device="cuda").manual_seed(gens)
    width = draft_len + 1
    main_rows = _accept(gen, gens, draft_len)
    all_rows = torch.arange(gens * width, device="cuda").view(gens, width)
    assert ((main_rows >= all_rows[:, 0]) & (main_rows <= all_rows[:, -1])).all()

    stage = _draft_model(gen).mtp_layers[0]

    params = {
        "n_heads": HEADS,
        "head_dim": HEAD_DIM,
        "rope_head_dim": ROPE_DIM,
        "n_groups": GROUPS,
        "o_lora_rank": O_RANK,
        "window_size": WINDOW,
        "eps": 1e-6,
        "softmax_scale": HEAD_DIM**-0.5,
        "freqs_cis": dspark.precompute_dspark_freqs_cis(ROPE_DIM, 1024, device="cuda"),
        "persist": True,
        "q_b_norm_enabled": False,
        **stage._dspark_attn,
    }
    x = (torch.randn(gens, width, HIDDEN, generator=gen, device="cuda") * 0.1).bfloat16()
    bonus = (torch.randn(gens, 1, HIDDEN, generator=gen, device="cuda") * 0.1).bfloat16()
    start_pos = torch.randint(5, 400, (gens,), generator=gen, device="cuda")
    slots = torch.randperm(gens, generator=gen, device="cuda")
    valid_len = torch.minimum(start_pos, torch.full_like(start_pos, WINDOW))
    cache = (torch.randn(gens, WINDOW, HEAD_DIM, generator=gen, device="cuda") * 0.1).bfloat16()
    caches = [cache.clone() for _ in range(2)]
    inputs = (x, bonus)

    def attention(cache, **kwargs):
        return dspark.dspark_attention_forward_batched(
            *inputs, start_pos, cache, slots, valid_len, **params, **kwargs
        )

    out = attention(caches[0])
    with monkeypatch.context() as patch:
        patch.setattr(dspark, "private_stream", lambda *a: torch.cuda.current_stream())
        patch.setattr(dspark, "_mla_rope_into", lambda *a: False)
        serial = attention(caches[1])
    torch.cuda.synchronize()
    assert torch.equal(out, serial) and torch.equal(caches[0], caches[1])
