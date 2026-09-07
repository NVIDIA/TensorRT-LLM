# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ragged (per-request) new-token counts for ``compressor_paged_kv_compress``.

The generation compressor used to take ``NEXT_N`` as a C++ template constant and
derive each request's starting position from it (``sp = kv_len - NEXT_N``). Under
DSpark's ragged verification a request appends only as many tokens as it was
given to verify, so ``next_n`` was demoted to a compile-time *upper bound* and
the true per-request counts now arrive in an optional ``new_tokens_per_seq``.

The property that matters: a ragged call whose ``new_tokens_per_seq`` is filled
with the uniform value must be **bitwise identical** to passing ``None``. That
is what makes the optional argument safe to add -- it says the ragged code path
did not perturb the uniform one, which every non-DSpark model still uses.

The second test covers the case the first cannot: genuinely differing counts.
It checks the ragged call against per-request uniform calls, which is the only
independent statement of what "correct" means here.
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for compressor tests", allow_module_level=True)


def _register_ops() -> None:
    """Make ``torch.ops.trtllm.compressor_paged_kv_compress`` available.

    Mirrors ``test_indexer_topk_ragged.py``: importing ``tensorrt_llm`` pulls in
    the whole model zoo including optional CuTe-DSL paths that some containers
    lack, and this file tests one C++ kernel.
    """
    try:
        import tensorrt_llm  # noqa: F401

        return
    except ImportError:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, *[os.pardir] * 5))
    for candidate in (
        os.path.join(repo, "cpp", "build", "tensorrt_llm", "thop", "libth_common.so"),
        os.path.join(repo, "tensorrt_llm", "libs", "libth_common.so"),
    ):
        if os.path.exists(candidate):
            torch.ops.load_library(candidate)
            return
    pytest.skip("neither tensorrt_llm nor libth_common.so is importable", allow_module_level=True)


_register_ops()

HEAD_DIM = 512
PAGE_SIZE = 64
COMPRESS_RATIO = 4
# The paged state is twice as wide as the head for compress_ratio=4: that ratio
# selects the kernel's "overlap" layout, where `STATE_DIM = 2 * HEAD_DIM` (see
# compressorKernels.cu, `IS_OVERLAP`). `kv_score` is twice as wide again -- it
# carries the kv and the score side by side. Getting these wrong does not
# produce a shape error anywhere; the kernel simply reads past the end.
STATE_DIM = 2 * HEAD_DIM if COMPRESS_RATIO == 4 else HEAD_DIM
KV_SCORE_DIM = 2 * STATE_DIM


class _CompressorCase:
    """Inputs for one ``compressor_paged_kv_compress`` call.

    Built once and reused across the calls being compared, so that a difference
    in output can only come from the arguments under test. The paged buffers are
    written in place by the kernel, so each call gets a fresh clone.
    """

    def __init__(
        self,
        kv_lens: list[int],
        new_tokens: list[int],
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 0,
    ) -> None:
        if len(kv_lens) != len(new_tokens):
            raise ValueError("kv_lens and new_tokens must have the same length")
        self.batch_size = len(kv_lens)
        self.kv_lens = list(kv_lens)
        self.new_tokens = list(new_tokens)
        self.device = device
        self.dtype = dtype

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # One block table row per request, sized for the longest sequence.
        max_blocks = max((kv_len + PAGE_SIZE - 1) // PAGE_SIZE for kv_len in kv_lens) + 1
        num_blocks = self.batch_size * max_blocks
        self.block_table = (
            torch.arange(num_blocks, dtype=torch.int32)
            .reshape(self.batch_size, max_blocks)
            .to(device)
        )

        self.paged_kv = (
            torch.empty([num_blocks, PAGE_SIZE, STATE_DIM], dtype=dtype)
            .uniform_(-1, 1, generator=generator)
            .to(device)
        )
        self.paged_score = (
            torch.empty([num_blocks, PAGE_SIZE, STATE_DIM], dtype=dtype)
            .uniform_(-1, 1, generator=generator)
            .to(device)
        )

        # kv_score is indexed by the flat token offsets in cu_seq_lens, so it has
        # to cover every request's appended tokens.
        total_new = sum(new_tokens)
        self.kv_score = (
            torch.empty([max(total_new, 1), KV_SCORE_DIM], dtype=dtype)
            .uniform_(-1, 1, generator=generator)
            .to(device)
        )
        self.ape = (
            torch.empty([COMPRESS_RATIO, STATE_DIM], dtype=torch.float32)
            .uniform_(-1, 1, generator=generator)
            .to(device)
        )

        cu_seq_lens = [0]
        for count in new_tokens:
            cu_seq_lens.append(cu_seq_lens[-1] + count)
        self.cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32, device=device)

        # How many compressed tokens each request produces this step: the count
        # of compression boundaries its new tokens crossed.
        cu_kv_comp = [0]
        for kv_len, count in zip(kv_lens, new_tokens):
            start = kv_len - count
            produced = (kv_len // COMPRESS_RATIO) - (start // COMPRESS_RATIO)
            cu_kv_comp.append(cu_kv_comp[-1] + produced)
        self.cu_kv_comp = torch.tensor(cu_kv_comp, dtype=torch.int32, device=device)
        self.total_outputs = cu_kv_comp[-1]

        self.kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    def run(
        self, next_n: int, new_tokens_per_seq: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        paged_kv = self.paged_kv.clone()
        paged_score = self.paged_score.clone()
        # Filled, not zeroed: a kernel that skips an output row would otherwise
        # be indistinguishable from one that legitimately writes zeros there.
        output = torch.full(
            [max(self.total_outputs, 1), HEAD_DIM],
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        torch.ops.trtllm.compressor_paged_kv_compress(
            self.kv_score,
            self.ape,
            paged_kv,
            paged_score,
            self.block_table,
            self.block_table,
            output,
            self.kv_lens_t,
            self.cu_seq_lens,
            self.cu_kv_comp,
            self.batch_size,
            PAGE_SIZE,
            HEAD_DIM,
            COMPRESS_RATIO,
            next_n,
            new_tokens_per_seq,
        )
        return output, paged_kv, paged_score


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "kv_lens,next_n",
    [
        # next_n == 1: the degenerate case the uniform path uses most.
        ([256, 257, 512], 1),
        # next_n == 6: DeepSeek-V4-Pro-DSpark's full block (max_draft_len=5).
        ([256, 300, 512, 1024], 6),
        # Lengths deliberately not multiples of compress_ratio, so requests
        # produce differing numbers of compressed tokens.
        ([255, 258, 261, 1023], 4),
    ],
)
def test_uniform_new_tokens_matches_none(
    kv_lens: list[int], next_n: int, dtype: torch.dtype
) -> None:
    """Filling ``new_tokens_per_seq`` uniformly must equal passing None.

    This is the differential that licenses the optional argument. If it fails,
    the ragged plumbing changed the uniform path -- which every non-DSpark model
    on this kernel still relies on.
    """
    device = torch.device("cuda")
    new_tokens = [next_n] * len(kv_lens)
    case = _CompressorCase(kv_lens, new_tokens, device, dtype, seed=1234)

    reference = case.run(next_n, None)
    uniform_vector = torch.tensor(new_tokens, dtype=torch.int32, device=device)
    ragged = case.run(next_n, uniform_vector)

    reference_output, reference_kv, reference_score = reference
    ragged_output, ragged_kv, ragged_score = ragged
    assert not torch.isnan(reference_output[: case.total_outputs]).any(), (
        "reference call left output rows unwritten; the test's cu_kv_comp does "
        "not match what the kernel produces"
    )
    assert torch.equal(
        reference_output[: case.total_outputs], ragged_output[: case.total_outputs]
    ), (
        "passing a uniformly-filled new_tokens_per_seq changed the result; the "
        "ragged branch is not a strict generalization of the uniform one. Max "
        f"abs diff: {(reference_output.float() - ragged_output.float()).abs().max().item()}"
    )
    assert torch.equal(reference_kv, ragged_kv)
    assert torch.equal(reference_score, ragged_score)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ragged_new_tokens_matches_per_request_uniform(dtype: torch.dtype) -> None:
    """Genuinely differing counts, checked against single-request uniform calls.

    The test above cannot catch a kernel that ignores ``new_tokens_per_seq``
    entirely -- with a uniform vector that is the correct answer. This one can:
    a batch of one request is unambiguously uniform, so running each request
    alone with its own ``next_n`` is an independent statement of the expected
    result, with no shared code path to the ragged call.
    """
    device = torch.device("cuda")
    kv_lens = [256, 300, 512, 1024]
    new_tokens = [6, 3, 1, 4]
    next_n = max(new_tokens)

    batched = _CompressorCase(kv_lens, new_tokens, device, dtype, seed=99)
    ragged_out, ragged_kv, ragged_score = batched.run(
        next_n, torch.tensor(new_tokens, dtype=torch.int32, device=device)
    )
    assert not torch.isnan(ragged_out).any(), "ragged call left output rows unwritten"

    cu_kv_comp = batched.cu_kv_comp.tolist()
    for index, (kv_len, count) in enumerate(zip(kv_lens, new_tokens)):
        single = _CompressorCase([kv_len], [count], device, dtype, seed=99)
        # Point the one-request case at the same slice of the shared inputs the
        # batched call read for this request, so the only difference is the
        # batching itself.
        token_start = batched.cu_seq_lens[index].item()
        single.kv_score = batched.kv_score[token_start : token_start + count]
        single.ape = batched.ape
        block_row = batched.block_table[index : index + 1]
        single.block_table = block_row - block_row.min()
        block_start = int(block_row.min().item())
        num_blocks = block_row.shape[1]
        single.paged_kv = batched.paged_kv[block_start : block_start + num_blocks].clone()
        single.paged_score = batched.paged_score[block_start : block_start + num_blocks].clone()

        expected, expected_kv, expected_score = single.run(count, None)
        start, end = cu_kv_comp[index], cu_kv_comp[index + 1]
        actual = ragged_out[start:end]
        assert torch.equal(expected, actual), (
            f"request {index} (kv_len={kv_len}, new_tokens={count}) differs "
            f"between the ragged batch and a uniform batch of one. Max abs "
            f"diff: {(expected.float() - actual.float()).abs().max().item()}"
        )
        assert torch.equal(expected_kv, ragged_kv[block_start : block_start + num_blocks])
        assert torch.equal(expected_score, ragged_score[block_start : block_start + num_blocks])


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ragged_append_without_output_updates_paged_state(dtype: torch.dtype) -> None:
    """A request crossing no compression boundary still updates paged state."""
    device = torch.device("cuda")
    kv_lens = [257, 260]
    new_tokens = [1, 4]
    batched = _CompressorCase(kv_lens, new_tokens, device, dtype, seed=7)

    ragged_out, ragged_kv, ragged_score = batched.run(
        max(new_tokens), torch.tensor(new_tokens, dtype=torch.int32, device=device)
    )
    assert batched.cu_kv_comp.tolist() == [0, 0, 1]
    assert not torch.isnan(ragged_out[:1]).any()

    for index, (kv_len, count) in enumerate(zip(kv_lens, new_tokens)):
        single = _CompressorCase([kv_len], [count], device, dtype, seed=7)
        token_start = batched.cu_seq_lens[index].item()
        single.kv_score = batched.kv_score[token_start : token_start + count]
        single.ape = batched.ape
        block_row = batched.block_table[index : index + 1]
        block_start = int(block_row.min().item())
        num_blocks = block_row.shape[1]
        single.block_table = block_row - block_start
        single.paged_kv = batched.paged_kv[block_start : block_start + num_blocks].clone()
        single.paged_score = batched.paged_score[block_start : block_start + num_blocks].clone()

        expected_out, expected_kv, expected_score = single.run(count, None)
        start, end = batched.cu_kv_comp[index : index + 2].tolist()
        assert torch.equal(expected_out[: end - start], ragged_out[start:end])
        assert torch.equal(expected_kv, ragged_kv[block_start : block_start + num_blocks])
        assert torch.equal(expected_score, ragged_score[block_start : block_start + num_blocks])
