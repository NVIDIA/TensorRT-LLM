# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``k3_helix_tail_ull`` entry on its ``.CELLS`` on one GPU (``loopback`` for ``P > 1``),
under ``validating``: ``out`` within BF16 rounding of the entry's FP64 reference (fold, V
up-projection, combine over the P sources of the looped-back group) and o_proj's MXFP8 operand
within one E4M3 step of the reference's, over two launches that use both receive halves and re-arm
them. The split state is the packed decode's (FP32 max-relative partials, or the FP16 normalized
split outputs with ``output_scale`` 1), planned by the makespan plan the decode runs; the band is
the one of ``kernels/k3r_helix/test_k3_helix_tail.py``.
"""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

K3 = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl"
NUM_HEADS = 96
SM_SCALE = 0.1
#: ``output_scale`` of FP32 max-relative partials; FP16 normalized partials take 1.
FP32_OUTPUT_SCALE = 0.25

k3_helix_tail_ull = importlib.import_module(f"{K3}.ops.attention.k3_helix_tail_ull").k3_helix_tail

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or _ARCH.get(torch.cuda.get_device_capability()) not in k3_helix_tail_ull.ARCHS,
    reason="k3_helix_tail_ull is certified on its ARCHS",
)


def _modules():
    reduce_mod = importlib.import_module(f"{K3}.kernels.k3r_fp4_mla.fp4_mla_packed_reduce_ull")
    tail_mod = importlib.import_module(f"{K3}.kernels.k3r_helix.k3_helix_tail_ull")
    return reduce_mod, tail_mod


def _split_case(reduce_mod, kv_tiles, splits, wave_tasks, query_len, partial_dtype, seed):
    """Random decode split state of ``len(kv_tiles)`` requests under the makespan plan: each
    request's first task is empty, unplanned slots hold garbage, rows of single-task requests are
    stored directly in ``rows``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    requests = len(kv_tiles)
    valid_m = NUM_HEADS * query_len
    physical_m = (valid_m + 127) // 128 * 128
    tasks = splits * requests
    kv_lens = torch.tensor([t * 256 - 3 for t in kv_tiles], dtype=torch.int32, device="cuda")
    plan = torch.empty(
        reduce_mod.packed_plan_length(requests, tasks), dtype=torch.int32, device="cuda"
    )
    reduce_mod.launch_packed_plan(
        kv_lens,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=wave_tasks,
        max_tiles_per_task=max(kv_tiles),
        query_len=query_len,
        lpt_m_tiles=physical_m // 128,
        lpt_wave_clusters=max(1, torch.cuda.get_device_properties(0).multi_processor_count // 2),
    )
    begins = plan[2 : 3 + requests].tolist()
    partial = torch.randn((tasks, physical_m, 512), device="cuda", generator=gen)
    maxima = torch.randn((tasks, physical_m), device="cuda", generator=gen) * 12
    sums = torch.rand((tasks, physical_m), device="cuda", generator=gen) * 16 + 0.5
    for begin, end in zip(begins, begins[1:]):
        if end - begin > 1:
            sums[begin] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    partial.masked_fill_((sums == 0).unsqueeze(-1), torch.nan)
    sums[begins[-1] :] = 3.0
    maxima[begins[-1] :] = 1e9
    partial[begins[-1] :] = torch.nan
    rows = torch.randn((requests * valid_m, 512), device="cuda", generator=gen).to(torch.bfloat16)
    return dict(
        partial=partial.to(partial_dtype),
        maxima=maxima,
        sums=sums,
        plan=plan,
        rows=rows,
        requests=requests,
        valid_m=valid_m,
        physical_m=physical_m,
    )


def _mx_dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim=-1)


@pytest.mark.parametrize(
    "cell", k3_helix_tail_ull.CELLS, ids=[c.why[:48] for c in k3_helix_tail_ull.CELLS]
)
def test_k3_helix_tail_ull_cells(cell):
    op = k3_helix_tail_ull
    spec = cell.spec
    reduce_mod, tail_mod = _modules()
    cp_size = spec["cp_size"]
    num_tokens = len(spec["kv_tiles"]) * spec["query_len"]
    seed = zlib.crc32(cell.why.encode()) % 2**31
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if spec["partial"] is None:
        rows = torch.randn((num_tokens * NUM_HEADS, 512), device="cuda", generator=gen).to(
            torch.bfloat16
        )
        stats = torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
        stats[:, 0] = torch.randn(num_tokens * NUM_HEADS, device="cuda", generator=gen) * 3
        stats[:, 1] = torch.rand(num_tokens * NUM_HEADS, device="cuda", generator=gen) * 50 + 1
        fold = None
    else:
        fp16 = spec["partial"] == "fp16"
        case = _split_case(
            reduce_mod,
            spec["kv_tiles"],
            spec["splits"],
            spec["wave_tasks"],
            spec["query_len"],
            torch.float16 if fp16 else torch.float32,
            seed,
        )
        rows, stats = case["rows"], torch.empty((num_tokens * NUM_HEADS, 2), device="cuda")
        fold = tail_mod.FoldInputs(
            case["partial"],
            case["maxima"],
            case["sums"],
            case["plan"],
            torch.tensor(1.25, device="cuda"),
            torch.tensor(0.75, device="cuda"),
            num_requests=case["requests"],
            valid_m=case["valid_m"],
            physical_m=case["physical_m"],
            sm_scale=SM_SCALE,
            output_scale=1.0 if fp16 else FP32_OUTPUT_SCALE,
        )
    w_uv = (torch.randn((NUM_HEADS, 128, 512), device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    cols = NUM_HEADS // cp_size * 128
    out = torch.full((num_tokens, cols), torch.nan, dtype=torch.bfloat16, device="cuda")
    mx_out = None
    if spec["mx_out"]:
        gate = torch.sigmoid(torch.randn((num_tokens, cols), device="cuda", generator=gen)).to(
            torch.bfloat16
        )
        mx_out = (
            gate,
            torch.empty((num_tokens, cols), dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty((num_tokens, cols // 32), dtype=torch.uint8, device="cuda"),
        )
    half_bytes = num_tokens * cp_size * tail_mod.record_bytes(NUM_HEADS // cp_size, 128) + 4096
    buffer = torch.full((2 * half_bytes // 4,), -1, dtype=torch.int32, device="cuda")
    args = (
        rows,
        stats,
        w_uv,
        out,
        fold,
        None,
        0,
        cp_size,
        buffer.data_ptr(),
        buffer.numel() * 4,
        half_bytes,
        torch.zeros(8, dtype=torch.int64, device="cuda"),
        torch.cuda.get_device_properties(0).multi_processor_count,
        True,
        mx_out,
        None,
        cp_size > 1,
        False,
        spec["ull2"],
    )
    for _ in range(2):
        out.fill_(torch.nan)
        with validating(op):
            op(*args)
        torch.cuda.synchronize()
        expected = op.reference(*args)
        ref_out = expected[0] if mx_out is not None else expected
        # The V up-projection rounds each source row once to BF16 (half an ulp of the row's
        # scale), the fold's latent rows another half ulp; the combine adds one rounding.
        assert_within_ulp(out, ref_out, element_ulp=4, rms_ulp=1)
        if mx_out is not None:
            got = _mx_dequant(mx_out[1], mx_out[2])
            ref = _mx_dequant(expected[1], expected[2])
            # One E4M3 step (2^-3 of the group's power-of-two scale) where out differs by an ulp.
            group = ref.abs().view(num_tokens, -1, 32).amax(-1, keepdim=True)
            step = (group * 2.0**-3).expand(-1, -1, 32).reshape(num_tokens, -1)
            assert bool(((got - ref).abs() <= step + 1e-30).all()), cell.why
    assert bool((buffer == -1).all()), "the receive buffer was not re-armed"
