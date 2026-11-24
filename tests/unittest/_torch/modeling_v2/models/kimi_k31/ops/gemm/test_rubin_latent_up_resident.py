# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_latent_up_resident entry (helix8's latent_up, weight on chip before the
grid wait).

* the cells: the op against its reference (the three K chunks' FP32 products summed in rank
  order), within the band of one BF16 rounding of an FP32 sum taken in another order;
* every output word equals FlashInfer's cluster split-K over the same chunks (``splitk_chunked``)
  up to 96 rows; each element's error against an FP64 product is cuBLAS's plus one BF16 ulp and
  the FP32 reordering allowance, its relative L2 error at most 1.02 x cuBLAS's, at most 10 % of
  its words differ from cuBLAS's;
* only the M output rows are stored; repeated calls and a replayed graph give the same words;
* behind a PDL producer that writes the input after releasing its dependents, every graph replay
  reads the new input;
* the launcher's configurations fit the budget at every row count; the guard refuses what the
  launcher refuses.
"""

import pytest
import torch
from _rubin_gemm_util import generator, requires_sm107

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_latent_up_resident import (  # noqa: E501
    HIDDEN,
    LATENT,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_latent_up_resident import (  # noqa: E501
    rubin_latent_up_resident as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = requires_sm107

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

_BF16_ULP = 2.0**-7  # one BF16 ulp of a value in [1, 2): 7 stored mantissa bits
_ROWS = list(range(6, 129, 6)) + [128]
_KINDS = ("normal", "wide", "massive", "signed_zero", "denormal", "ties")


def _kernel():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        rubin_latent_up_resident as kernel,
    )

    return kernel


def _weight(gen, n=HIDDEN, k=LATENT) -> torch.Tensor:
    return (torch.randn(n, k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)


def _activation(rows: int, kind: str, gen, k=LATENT) -> torch.Tensor:
    """BF16 ``[rows, k]`` of one input kind (what the normed latent can hold)."""
    x = torch.randn(rows, k, generator=gen, device="cuda")
    if kind == "wide":
        x = x * torch.exp(torch.randn(rows, k, generator=gen, device="cuda") * 4)
    elif kind == "massive":
        x[:, torch.randint(0, k, (8,), generator=gen, device="cuda")] *= 1000.0
    elif kind == "signed_zero":
        x[torch.rand(rows, k, generator=gen, device="cuda") < 0.3] = -0.0
    elif kind == "denormal":
        x = x * 1e-38
    elif kind == "ties":
        x = torch.randint(-4, 5, (rows, k), generator=gen, device="cuda").float() / 4
    return x.to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _cublas(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return cublas_mm(x, w.t())


def _flashinfer(x: torch.Tensor, w: torch.Tensor, split_k: int) -> torch.Tensor:
    """FlashInfer's cluster split-K (``splitk_chunked``) on the resident kernel's token tile, with
    the deepest ring that fits."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        splitk_chunked,
    )

    mma_n = 16 * ((x.shape[0] + 15) // 16)
    config = None
    for stages in range(2, 13):
        try:
            splitk_chunked.validate_config((128, mma_n, split_k, stages), x.shape[0], x.shape[1])
        except ValueError:
            continue
        config = (128, mma_n, split_k, stages)
    assert config is not None, (x.shape, split_k)
    out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
    return splitk_chunked.run_chunked_splitk(x, w.t(), out, False, config)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    s = cell.spec
    gen = generator(cell.why)
    w = _weight(gen, s["n"], s["k"])
    x = _activation(s["rows"], "normal", gen, s["k"])
    out = op(x, w)
    ref = op.reference(x, w)
    # One BF16 rounding of an FP32 sum: the reference's order inside a chunk is torch's.
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.mark.parametrize("kind", _KINDS)
def test_split_is_flashinfer_chunked_split_k_word_for_word(kind) -> None:
    """split's words are FlashInfer's cluster split-K over the same front-heavy chunks
    (``splitk_chunked``), wherever that kernel's ring fits (up to 96 tokens per tile)."""
    gen = generator(f"split chunks {kind}")
    w = _weight(gen)
    for rows in [r for r in _ROWS if r <= 96]:
        x = _activation(rows, kind, gen)
        got, want = op(x, w), _flashinfer(x, w, split_k=3)
        diff = (_bits(got) != _bits(want)).sum().item()
        assert diff == 0, f"{kind}, {rows} rows: {diff} of {got.numel()} words differ"


@pytest.mark.parametrize("rows", [6, 12, 24, 54, 66, 96, 126, 128])
def test_split_error_is_within_cublas_class(rows) -> None:
    gen = generator(f"split {rows}")
    w = _weight(gen)
    w64 = w.double()
    for _ in range(4):
        x = _activation(rows, "normal", gen)
        y64 = x.double() @ w64.t()
        got, base = op(x, w), _cublas(x, w)
        ulp = (
            torch.ldexp(torch.ones_like(y64), torch.frexp(y64.abs().clamp_min(1e-30))[1] - 1)
            * _BF16_ULP
        )
        # An FP32 sum taken in another order moves an output by up to a few FP32 ulps of the
        # products' magnitude: an allowance where cancellation leaves the output far below them.
        order = 2.0**-20 * (x.double().abs() @ w64.abs().t())
        err, base_err = (got.double() - y64).abs(), (base.double() - y64).abs()
        assert bool((err <= base_err + ulp + order).all()), (err - base_err - ulp - order).max()
        rel = (got.double() - y64).norm() / y64.norm()
        base_rel = (base.double() - y64).norm() / y64.norm()
        assert rel <= 1.02 * base_rel, (rel.item(), base_rel.item())
        assert (_bits(got) != _bits(base)).float().mean().item() <= 0.10


@pytest.mark.parametrize("rows", [6, 54, 66, 128])
def test_only_the_rows_are_stored(rows) -> None:
    gen = generator(f"poison {rows}")
    w, x = _weight(gen), _activation(rows, "normal", gen)
    buf = torch.full((rows + 16, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    _kernel().run_latent_up_resident(x, w, buf[:rows])
    torch.cuda.synchronize()
    assert bool(buf[rows:].isnan().all()), "a row past M was stored"
    assert torch.equal(_bits(buf[:rows]), _bits(op(x, w)))


def test_repeats_and_graph_replays_give_the_same_words() -> None:
    gen = generator("replay")
    w = _weight(gen)
    x = _activation(54, "normal", gen)
    first = op(x, w)
    for _ in range(20):
        assert torch.equal(_bits(op(x, w)), _bits(first))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, w)
    for _ in range(3):
        x.copy_(_activation(54, "normal", gen))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(op(x, w)))


@pytest.mark.parametrize("rows", [24, 54])
def test_behind_a_pdl_producer_every_replay_reads_the_new_input(rows) -> None:
    from _rubin_pdl_spin import spin_copy

    gen = generator(f"pdl {rows}")
    w = _weight(gen)
    sets = [_activation(rows, "normal", gen) for _ in range(8)]
    want = [op(s, w) for s in sets]
    src = torch.empty_like(sets[0])
    x = torch.zeros_like(sets[0])
    src.copy_(sets[0])
    spin_copy(src, x, 3000)
    op(x, w)  # compiled outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        spin_copy(src, x, 3000)
        out = op(x, w)
    for i in range(200):
        src.copy_(sets[i % 8])
        x.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(want[i % 8])), f"replay {i}"


def test_every_row_count_has_a_config_within_the_budget() -> None:
    kernel = _kernel()
    for rows in range(1, kernel.MAX_M + 1):
        config = kernel.select_config(rows, LATENT)
        assert config.mma_n >= rows and config.mma_n % 16 == 0
        assert config.a_stages >= 2
        assert kernel.launch_smem_bytes(config) <= kernel.DEFAULT_SMEM_BUDGET
    big = kernel.select_config(54, LATENT, kernel.SMEM_CAPACITY_BYTES)
    assert big.a_stages >= kernel.select_config(54, LATENT).a_stages
    assert [c for _, c in kernel.k_chunks(LATENT)] == [10, 9, 9]


@pytest.mark.parametrize(
    "rows, k, n, dtype",
    [
        (0, LATENT, HIDDEN, torch.bfloat16),
        (129, LATENT, HIDDEN, torch.bfloat16),
        (8, LATENT + 64, HIDDEN, torch.bfloat16),
        (8, LATENT, HIDDEN + 64, torch.bfloat16),
        (8, LATENT, HIDDEN, torch.float16),
    ],
    ids=["no_rows", "past_128", "k_tail", "n_tail", "fp16"],
)
def test_the_guard_refuses_what_the_launcher_refuses(rows, k, n, dtype) -> None:
    x = torch.zeros(rows, k, dtype=dtype, device="cuda")
    w = torch.zeros(n, k, dtype=dtype, device="cuda")
    with pytest.raises(ValueError):
        op.is_valid(x, w)
    with pytest.raises(ValueError):
        with validating(op):
            op(x, w)
    with pytest.raises(ValueError):
        _kernel().run_latent_up_resident(x, w, torch.empty(rows, n, dtype=dtype, device="cuda"))
