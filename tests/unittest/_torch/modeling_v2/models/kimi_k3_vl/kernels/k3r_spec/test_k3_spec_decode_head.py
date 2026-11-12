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
"""The private one-kernel speculative decode step head writes what main's native head writes.

``test_head_matches_native_preprocess_inputs`` runs main's ``DecoderInputPreparer.preprocess_inputs``
on the same step, so it also pins the preparer and metadata names the head reads.
"""

import functools
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_spec.spec_decode_head import (  # noqa: E501
    SpecDecodeHead,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec.k3_spec_decode_head import (  # noqa: E501
    k3_spec_decode_head,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _helix_metadata(kv_lens, helix_pos, tokens_per_block, cp_size, cp_rank, max_tokens, num_gen):
    def buffer(dtype, fill):
        return torch.full((max_tokens,), fill, dtype=dtype, device="cuda")

    return SimpleNamespace(
        num_contexts=0,
        num_generations=num_gen,
        kv_lens_cuda=kv_lens,
        helix_position_offsets=helix_pos,
        helix_local_slots=buffer(torch.int32, -7),
        helix_kv_bounds=buffer(torch.int32, -7),
        kv_cache_manager=SimpleNamespace(tokens_per_block=tokens_per_block),
        mapping=SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank),
    )


@requires_cuda
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize(
    "num_seqs, tokens_per_seq, tokens_per_block, cp_size",
    [
        (1, 7, 32, 16),
        (2, 7, 32, 16),
        (4, 8, 64, 16),
        (3, 7, 32, 4),
        (16, 7, 32, 16),
        (96, 8, 32, 8),
    ],
)
def test_fused_head_matches_native(overlap, num_seqs, tokens_per_seq, tokens_per_block, cp_size):
    gen = torch.Generator(device="cuda").manual_seed(num_seqs * 131 + tokens_per_block + overlap)
    max_tokens = 1024
    n = num_seqs * tokens_per_seq
    for cp_rank in sorted({0, 1, cp_size // 2, cp_size - 1}):
        # Groups start near page boundaries, so a group straddles owners.
        base = torch.randint(0, 1 << 18, (num_seqs,), device="cuda", generator=gen)
        base = base // tokens_per_block * tokens_per_block
        base += 2 + torch.randint(0, 3, (num_seqs,), device="cuda", generator=gen)
        offsets = torch.arange(tokens_per_seq, device="cuda")
        positions = (base.view(-1, 1) + offsets).view(-1).to(torch.int32)
        pos_offsets = torch.randint(-2, 1, (max_tokens,), device="cuda", generator=gen)
        pos_offsets = pos_offsets.to(torch.int32)
        kv_offsets = torch.randint(-2, 1, (128,), device="cuda", generator=gen).to(torch.int32)
        kv_lens = torch.randint(0, 1 << 16, (128,), device="cuda", generator=gen).to(torch.int32)
        position_ids = torch.randint(0, 1 << 18, (1, n), device="cuda", generator=gen)
        position_ids = position_ids.to(torch.int32)
        helix_pos = torch.zeros(max_tokens, dtype=torch.int32, device="cuda")
        helix_pos[:n] = positions

        # Native: DecoderInputPreparer.preprocess_inputs + recompute_helix_spec_buffers.
        ref_pos_ids = position_ids.clone()
        ref = _helix_metadata(
            kv_lens.clone(),
            helix_pos.clone(),
            tokens_per_block,
            cp_size,
            cp_rank,
            max_tokens,
            num_seqs,
        )
        if overlap:
            ref_pos_ids[0, :] += pos_offsets[:n]
            ref.kv_lens_cuda[:num_seqs] += kv_offsets[:num_seqs]
            ref.helix_position_offsets[:n] += pos_offsets[:n]
        TrtllmAttentionMetadata.recompute_helix_spec_buffers(ref, n, tokens_per_seq)

        out_pos_ids = position_ids.clone()
        out = _helix_metadata(
            kv_lens.clone(),
            helix_pos.clone(),
            tokens_per_block,
            cp_size,
            cp_rank,
            max_tokens,
            num_seqs,
        )
        zero_kv = torch.ones(max_tokens, dtype=torch.bool, device="cuda")
        k3_spec_decode_head(
            out_pos_ids[0, :],
            pos_offsets,
            n,
            out.kv_lens_cuda,
            kv_offsets,
            0,
            num_seqs,
            has_overlap=overlap,
            helix_positions=out.helix_position_offsets,
            helix_slots=out.helix_local_slots,
            helix_bounds=out.helix_kv_bounds,
            helix_zero_kv=zero_kv,
            num_helix=n,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_bound_start=0,
            tokens_per_seq=tokens_per_seq,
        )
        torch.cuda.synchronize()
        assert torch.equal(out_pos_ids, ref_pos_ids)
        assert torch.equal(out.kv_lens_cuda, ref.kv_lens_cuda)
        for name in ("helix_position_offsets", "helix_local_slots", "helix_kv_bounds"):
            assert torch.equal(getattr(out, name)[:n], getattr(ref, name)[:n]), name
        assert torch.equal(zero_kv[:n], ref.helix_kv_bounds[:n] == 0)


@requires_cuda
@pytest.mark.parametrize("num_gen_tokens", [28, 1500])
def test_fused_head_overlap_only_chunked_range(num_gen_tokens):
    """Without Helix buffers only the overlap correction runs, on any KV range."""
    gen = torch.Generator(device="cuda").manual_seed(num_gen_tokens)
    position_ids = torch.randint(0, 1000, (1, 12 + num_gen_tokens), device="cuda", generator=gen)
    position_ids = position_ids.to(torch.int32)
    pos_offsets = torch.randint(-3, 4, (2048,), device="cuda", generator=gen).to(torch.int32)
    kv_lens = torch.randint(0, 1000, (16,), device="cuda", generator=gen).to(torch.int32)
    kv_offsets = torch.randint(-3, 4, (16,), device="cuda", generator=gen).to(torch.int32)
    ref_pos, ref_kv = position_ids.clone(), kv_lens.clone()
    ref_pos[0, 12:] += pos_offsets[:num_gen_tokens]
    ref_kv[3:5] += kv_offsets[:2]
    k3_spec_decode_head(
        position_ids[0, 12:],
        pos_offsets,
        num_gen_tokens,
        kv_lens,
        kv_offsets,
        3,
        2,
        has_overlap=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(position_ids, ref_pos)
    assert torch.equal(kv_lens, ref_kv)


def _step_metadata(num_ctx, num_ctx_tokens, num_gen, num_chunked, helix, gen_positions, gen):
    """TRT-LLM attention metadata of a mixed step: ``num_ctx`` context rows (the last
    ``num_chunked`` of them extend a context) and ``num_gen`` verify groups."""
    max_tokens = 256
    md = Mock(spec=TrtllmAttentionMetadata)
    md.num_contexts, md.num_generations = num_ctx, num_gen
    md.num_seqs = num_ctx + num_gen
    md.num_ctx_tokens, md.num_chunked_ctx_requests = num_ctx_tokens, num_chunked
    md.kv_lens_cuda = torch.randint(0, 1 << 12, (num_ctx + num_gen,), device="cuda", generator=gen)
    md.kv_lens_cuda = md.kv_lens_cuda.to(torch.int32)
    md.kv_cache_manager = SimpleNamespace(tokens_per_block=32)
    md.mapping = SimpleNamespace(cp_size=4, cp_rank=1)
    md._helix_spec_tokens_valid = helix
    md.helix_position_offsets = torch.zeros(max_tokens, dtype=torch.int32, device="cuda")
    md.helix_position_offsets[: gen_positions.numel()] = gen_positions
    md.helix_local_slots = torch.full((max_tokens,), -7, dtype=torch.int32, device="cuda")
    md.helix_kv_bounds = torch.full((max_tokens,), -7, dtype=torch.int32, device="cuda")
    md.recompute_helix_spec_buffers = (
        lambda num_tokens, width: TrtllmAttentionMetadata.recompute_helix_spec_buffers(
            md, num_tokens, width
        )
    )
    return md


def _preparer(mapping, pos_offsets, kv_offsets):
    """The fields of ``DecoderInputPreparer`` that ``preprocess_inputs`` and the head read."""
    preparer = SimpleNamespace(
        decode_prologue=None,
        guided_decoder=None,
        runtime_hooks=SimpleNamespace(spec_offsets_op=None),
        _config=SimpleNamespace(disable_overlap_scheduler=False),
        _buffers=SimpleNamespace(
            previous_pos_id_offsets_cuda=pos_offsets, previous_kv_lens_offsets_cuda=kv_offsets
        ),
        mapping=mapping,
        get_runtime_tokens_per_gen_step=lambda d: d + 1,
    )
    preparer._kv_lens_corrected_in_preprocess = functools.partial(
        DecoderInputPreparer._kv_lens_corrected_in_preprocess, preparer
    )
    return preparer


@requires_cuda
@pytest.mark.parametrize(
    "num_ctx, num_gen, num_chunked, helix",
    [(2, 3, 0, False), (2, 3, 0, True), (3, 0, 2, False), (0, 4, 0, True)],
    ids=["mixed", "mixed_helix", "chunked", "gen_helix"],
)
def test_head_matches_native_preprocess_inputs(num_ctx, num_gen, num_chunked, helix):
    """The head through its RuntimeHooks entry writes what main's
    DecoderInputPreparer.preprocess_inputs writes: the KV-length range of mixed and
    extend-context batches, and the helix buffers past the context rows."""
    draft_len = 3
    width = draft_len + 1
    num_ctx_tokens = 37 * num_ctx
    gen_tokens = num_gen * width if num_gen else num_chunked * width
    gen = torch.Generator(device="cuda").manual_seed(num_ctx * 7 + num_gen + num_chunked + helix)
    base = torch.randint(0, 1 << 12, (max(num_gen, 1),), device="cuda", generator=gen)
    gen_positions = (base.view(-1, 1) * 32 + 30 + torch.arange(width, device="cuda")).view(-1)
    gen_positions = gen_positions[: num_gen * width].to(torch.int32)
    pos_offsets = torch.randint(-2, 1, (256,), device="cuda", generator=gen).to(torch.int32)
    kv_offsets = torch.randint(-2, 1, (16,), device="cuda", generator=gen).to(torch.int32)
    position_ids = torch.randint(
        0, 1 << 16, (1, num_ctx_tokens + gen_tokens), device="cuda", generator=gen
    ).to(torch.int32)
    mapping = SimpleNamespace(has_cp_helix=lambda: helix)

    def step(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        md = _step_metadata(num_ctx, num_ctx_tokens, num_gen, num_chunked, helix, gen_positions, g)
        return {
            "attn_metadata": md,
            "position_ids": position_ids.clone(),
            "input_ids": torch.zeros(num_ctx_tokens + gen_tokens, device="cuda"),
        }

    native = _preparer(mapping, pos_offsets, kv_offsets)
    ref = DecoderInputPreparer.preprocess_inputs(
        native, step(11), enable_spec_decode=True, runtime_draft_len=draft_len
    )
    head = SpecDecodeHead()
    out = step(11)
    assert head(
        _preparer(mapping, pos_offsets, kv_offsets),
        out,
        enable_spec_decode=True,
        runtime_draft_len=draft_len,
    )
    torch.cuda.synchronize()
    assert torch.equal(out["position_ids"], ref["position_ids"])
    out_md, ref_md = out["attn_metadata"], ref["attn_metadata"]
    assert torch.equal(out_md.kv_lens_cuda, ref_md.kv_lens_cuda)
    if helix:
        n = gen_tokens
        for name in ("helix_position_offsets", "helix_local_slots", "helix_kv_bounds"):
            assert torch.equal(getattr(out_md, name)[:n], getattr(ref_md, name)[:n]), name
        assert torch.equal(head._zero_kv[:n], ref_md.helix_kv_bounds[:n] == 0)


def test_head_keeps_the_native_prologue_outside_its_cases():
    preparer = SimpleNamespace(
        _config=SimpleNamespace(disable_overlap_scheduler=False),
        mapping=SimpleNamespace(has_cp_helix=lambda: False),
    )
    head = SpecDecodeHead()
    other_metadata = {"attn_metadata": SimpleNamespace(kv_cache_manager=object())}
    assert not head(preparer, other_metadata, enable_spec_decode=True, runtime_draft_len=3)
    assert not head(preparer, {}, enable_spec_decode=True, runtime_draft_len=3)
    assert not head(preparer, other_metadata, enable_spec_decode=False, runtime_draft_len=3)


__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402, F401 — path above; fixture by name


def _on_archs(op) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in op.ARCHS}


@pytest.mark.skipif(not _on_archs(k3_spec_decode_head), reason="not an ARCHS architecture")
@pytest.mark.parametrize(
    "cell", k3_spec_decode_head.CELLS, ids=[c.why[:48] for c in k3_spec_decode_head.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    """Every tensor the head writes equals the entry's reference, and nothing past the ranges."""
    certifying(k3_spec_decode_head)
    spec = cell.spec
    num_seqs, tokens_per_seq, tokens_per_block = spec["num_seqs"], spec["tokens_per_seq"], 128
    n = num_seqs * tokens_per_seq
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)

    def ints(low, high, size):
        return torch.randint(low, high, (size,), device="cuda", generator=gen).to(torch.int32)

    # Groups start a few tokens past a page boundary, so a group straddles owners.
    base = ints(0, 1 << 11, num_seqs).long() * tokens_per_block - ints(0, tokens_per_seq, num_seqs)
    base = base.clamp(min=0)
    positions = (base.view(-1, 1) + torch.arange(tokens_per_seq, device="cuda")).view(-1)
    capacity = n + 16
    helix_positions = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    helix_positions[:n] = positions.to(torch.int32)
    args = (
        ints(0, 1 << 18, capacity),  # position_ids
        ints(-2, 1, capacity),  # pos_offsets
        n,
        ints(0, 1 << 16, num_seqs + 8),  # kv_lens
        ints(-2, 1, num_seqs + 8),  # kv_offsets
        0,
        num_seqs,
    )
    kwargs = dict(
        has_overlap=spec["overlap"],
        helix_positions=helix_positions,
        helix_slots=torch.full((capacity,), -7, dtype=torch.int32, device="cuda"),
        helix_bounds=torch.full((capacity,), -7, dtype=torch.int32, device="cuda"),
        helix_zero_kv=torch.ones(capacity, dtype=torch.bool, device="cuda"),
        num_helix=n,
        tokens_per_block=tokens_per_block,
        cp_size=spec["cp_size"],
        cp_rank=spec["cp_rank"],
        kv_bound_start=0,
        tokens_per_seq=tokens_per_seq,
    )
    expected = k3_spec_decode_head.reference(*args, **kwargs)
    pos_offsets, kv_offsets = args[1].clone(), args[4].clone()
    k3_spec_decode_head(*args, **kwargs)
    torch.cuda.synchronize()
    written = dict(position_ids=args[0], kv_lens=args[3], **kwargs)
    for name, value in expected.items():
        assert torch.equal(written[name], value), name
    assert torch.equal(args[1], pos_offsets) and torch.equal(args[4], kv_offsets)
