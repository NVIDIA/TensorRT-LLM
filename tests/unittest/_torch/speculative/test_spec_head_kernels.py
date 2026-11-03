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
"""The one-kernel speculative decode step head writes what the native head writes."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.spec_head_kernels import fused_spec_head_prep, spec_decode_head

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
        helix_zero_kv_tokens=buffer(torch.bool, True),
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
        fused_spec_head_prep(
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
            helix_zero_kv=out.helix_zero_kv_tokens,
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
        for name in (
            "helix_position_offsets",
            "helix_local_slots",
            "helix_kv_bounds",
            "helix_zero_kv_tokens",
        ):
            assert torch.equal(getattr(out, name)[:n], getattr(ref, name)[:n]), name


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
    fused_spec_head_prep(
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


def test_head_keeps_the_native_prologue_outside_its_cases():
    preparer = SimpleNamespace(
        overlap_scheduler_enabled=True, mapping=SimpleNamespace(has_cp_helix=lambda: False)
    )
    other_metadata = {"attn_metadata": SimpleNamespace(kv_cache_manager=object())}
    assert not spec_decode_head(
        preparer, other_metadata, enable_spec_decode=True, runtime_draft_len=3
    )
    assert not spec_decode_head(preparer, {}, enable_spec_decode=True, runtime_draft_len=3)
    assert not spec_decode_head(
        preparer, other_metadata, enable_spec_decode=False, runtime_draft_len=3
    )
