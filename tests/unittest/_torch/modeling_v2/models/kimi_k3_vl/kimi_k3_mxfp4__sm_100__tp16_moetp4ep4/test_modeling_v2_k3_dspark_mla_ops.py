# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's attention/k3_dspark_mla ops (sm_100).

``k3_dspark_mla_paged_decode`` (flashinfer TRTLLM-Gen, the built-in TRTLLM backend) is checked at the end of the file.
The other three entries wrap the CuTe DSL kernels the built-in MLA DSpark drafter
(``modeling_dspark.MLADSparkForCausalLM``) calls. The inputs are built as the built-in drafter builds them: the phases
come from ``build_dspark_mla_yarn_freqs_cis`` with the Inferact-Kimi-K3-DSpark YaRN settings, gathered at the positions
and flattened to one row per (G, s) the way ``_rmsnorm_rope_batched`` / ``_mla_rope_into`` flatten them; the latent pool
is a view of the built-in drafter's 5-D layer cache; the lengths, KV bounds and page table come from
``_build_mla_block_fixup``. Each cell then checks

* the entry bit for bit against the direct ``torch.ops.trtllm`` call and against the built-in drafter's own function
  (the entry adds nothing);
* the entry against its ``reference`` within a band stated where it is checked;

and one case replays a CUDA graph of the decode captured after an eager call, bit for bit against eager.
"""

import importlib
import math
import zlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_dspark_mla requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("k3_dspark_mla is certified on sm_100 and sm_103", allow_module_level=True)

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE  # noqa: E402

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("k3_dspark_mla needs the CuTe DSL build", allow_module_level=True)

_mla = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.attention.k3_dspark_mla"
)
k3_dspark_rmsnorm_rope = _mla.k3_dspark_rmsnorm_rope
k3_dspark_rope_into = _mla.k3_dspark_rope_into
k3_dspark_mla_decode = _mla.k3_dspark_mla_decode
k3_dspark_mla_paged_decode = _mla.k3_dspark_mla_paged_decode

from tensorrt_llm._torch.models.modeling_dspark import (  # noqa: E402
    _MLA_DECODE_WORKSPACE_BYTES,
    _build_mla_block_fixup,
    _mla_rope_into,
    _rmsnorm_rope_batched,
    _yarn_get_mscale,
    build_dspark_mla_yarn_freqs_cis,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

ROPE = 64
LATENT = 512
NOPE = 128
PAGE = 64
MAX_POS = 4096
EPS = 1e-6
# Inferact-Kimi-K3-DSpark's rope_scaling; the table is capped at MAX_POS positions as the built-in drafter caps it at
# the runtime ceiling (the YaRN correction range reads original_max_position_embeddings, not the table length).
YARN = dict(
    dim=ROPE,
    base=50000.0,
    scaling_factor=32.0,
    original_max_position_embeddings=32768,
    max_position_embeddings=MAX_POS,
    beta_fast=32.0,
    beta_slow=1.0,
    mscale=1.0,
    mscale_all_dim=1.0,
)
# The built-in MLADSparkForCausalLM.softmax_scale: mscale^2 / sqrt(qk_nope + qk_rope).
MSCALE = _yarn_get_mscale(YARN["scaling_factor"], YARN["mscale_all_dim"])
SOFTMAX_SCALE = MSCALE * MSCALE / math.sqrt(NOPE + ROPE)
# max |err| / max |ref| of the decode. The kernel accumulates q.k in fp32 from the same bf16 operands the reference
# reads (exact products), runs the softmax in fp32, then rounds P to bf16 for the P.V MMA (each weight off by at
# most 2^-9 relative) and the output to bf16 (2^-9 relative). The P rounding errors have random sign, so their
# sum moves the output by about 2^-9 of its own size, not of max |V|; the two roundings stay near 2 * 2^-9 = 3.9e-3
# of max |ref|, and 1e-2 leaves a factor 2.5 for the exp2 approximation and the order of the fp32 sums.
DECODE_TOL = 1e-2
# Context lengths before the block, cycled over the requests: none, a page boundary, a block that crosses a page,
# a block that crosses two pages' boundary, a long context, a short one, the longest, two pages.
LENGTHS = (0, 64, 60, 127, 1000, 5, 2041, 128)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a.double() - ref.double()).abs().max().item() / max(ref.abs().max().item(), 1e-6)


_TABLE: dict = {}


def _table() -> torch.Tensor:
    """[MAX_POS, ROPE / 2] complex64, the built-in drafter's ``_mla_freqs_cis``."""
    if "t" not in _TABLE:
        _TABLE["t"] = build_dspark_mla_yarn_freqs_cis(device="cuda", **YARN)
    return _TABLE["t"]


def _phases(gen: torch.Generator, lead: tuple[int, int]):
    """Random positions [G, s] -> (freqs_cis [G, s, rd / 2] complex, the built-in drafter's operand; freqs_real
    [G * s, rd / 2, 2] fp32, the kernels' operand, flattened exactly as ``_rmsnorm_rope_batched`` does)."""
    positions = torch.randint(0, MAX_POS, lead, generator=gen, device="cuda")
    freqs_cis = _table()[positions.long()]
    freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
    return freqs_cis, freqs_real


def _gate_rope(got: torch.Tensor, want: torch.Tensor) -> None:
    # Both sides work in fp32 and round to bf16 once: the RMSNorm (sum of squares, rsqrt, weight) and the rotation
    # (re * cos - im * sin) differ only in summation order, rsqrt's last bits and fma contraction, a few fp32 ulps.
    # That flips the bf16 rounding of an element by at most one spacing, <= 2^-7 of its own magnitude and so of the
    # row's largest (2 units of 2^-8). Flips need the fp32 value within a few ulps of a rounding boundary, so they
    # are rare and the relative RMS stays far below one unit.
    assert_within_ulp(got, want, element_ulp=2.0, rms_ulp=1.0)


# --------------------------------------------------------------------------------------------------------------
# k3_dspark_rmsnorm_rope
# --------------------------------------------------------------------------------------------------------------


def _rmsnorm_rope_inputs(spec, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = spec["shape"]
    t = torch.randn(shape, generator=gen, device="cuda").bfloat16()
    width = spec["norm_dim"] or shape[-1]
    weight = (
        (1.0 + 0.1 * torch.randn(width, generator=gen, device="cuda")).bfloat16()
        if spec["weight"]
        else None
    )
    freqs_cis, freqs_real = _phases(gen, shape[:2])
    # The built-in drafter passes eps 0 for the rope-only q call and the layernorm's eps otherwise.
    eps = EPS if spec["norm"] else 0.0
    return t, weight, freqs_cis, freqs_real, eps


@pytest.mark.parametrize(
    "cell", k3_dspark_rmsnorm_rope.CELLS, ids=[c.why for c in k3_dspark_rmsnorm_rope.CELLS]
)
def test_k3_dspark_rmsnorm_rope(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    with torch.inference_mode():
        t, weight, freqs_cis, freqs_real, eps = _rmsnorm_rope_inputs(spec, seed)
        args = (
            t,
            weight,
            freqs_real,
            spec["heads"],
            spec["rope"],
            eps,
            spec["weight"],
            spec["norm"],
            spec["norm_dim"],
        )
        t0 = t.clone()
        with validating(k3_dspark_rmsnorm_rope):
            got = k3_dspark_rmsnorm_rope(*args)
        assert torch.equal(_bits(t), _bits(t0)), "the input was written"

        direct = torch.ops.trtllm.cute_dsl_dspark_rmsnorm_rope(
            t,
            weight,
            freqs_real,
            spec["heads"],
            spec["rope"],
            eps,
            spec["weight"],
            spec["norm"],
            False,
            spec["norm_dim"],
        )
        assert torch.equal(_bits(got), _bits(direct)), (
            "the entry differs from the direct kernel call"
        )

        builtin = _rmsnorm_rope_batched(
            t,
            weight,
            eps,
            spec["rope"],
            freqs_cis,
            num_heads=spec["heads"],
            apply_weight=spec["weight"],
            apply_rmsnorm=spec["norm"],
            norm_dim=spec["norm_dim"],
        )
        assert torch.equal(_bits(got), _bits(builtin)), (
            "the entry differs from the built-in _rmsnorm_rope_batched"
        )

        _gate_rope(got, k3_dspark_rmsnorm_rope.reference(*args))
        assert not torch.equal(_bits(got[..., -ROPE:]), _bits(t[..., -ROPE:])), (
            "nothing was rotated"
        )


# --------------------------------------------------------------------------------------------------------------
# k3_dspark_rope_into
# --------------------------------------------------------------------------------------------------------------


def _rope_into_inputs(spec, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b, t, h = spec["batch"], spec["block"], spec["heads"]
    q = torch.randn(b, t, h, spec["q_dim"], generator=gen, device="cuda").bfloat16()
    freqs_cis, freqs_real = _phases(gen, (b, t))
    return q, freqs_cis, freqs_real


def _sentinel_out(spec) -> torch.Tensor:
    """The built-in drafter's ``query`` buffer, filled with NaN: any column the op writes outside its rope slice
    shows."""
    shape = (spec["batch"], spec["block"], spec["heads"], spec["out_dim"])
    return torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")


@pytest.mark.parametrize(
    "cell", k3_dspark_rope_into.CELLS, ids=[c.why for c in k3_dspark_rope_into.CELLS]
)
def test_k3_dspark_rope_into(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    heads, rope, offset = spec["heads"], spec["rope"], spec["offset"]
    cols = slice(offset, offset + rope)
    with torch.inference_mode():
        q, freqs_cis, freqs_real = _rope_into_inputs(spec, seed)
        q0 = q.clone()
        sentinel = _sentinel_out(spec)

        got = sentinel.clone()
        with validating(k3_dspark_rope_into):
            assert k3_dspark_rope_into(q, freqs_real, got, heads, rope, offset) is None
        assert torch.equal(_bits(q), _bits(q0)), "q was written"
        rest = torch.ones(spec["out_dim"], dtype=torch.bool, device="cuda")
        rest[cols] = False
        assert torch.equal(_bits(got[..., rest]), _bits(sentinel[..., rest])), (
            "a column outside the rope slice was written"
        )
        assert not got[..., cols].isnan().any(), "the rope slice was not written"

        direct = sentinel.clone()
        torch.ops.trtllm.cute_dsl_dspark_rope_into(q, freqs_real, direct, heads, rope, offset)
        assert torch.equal(_bits(got), _bits(direct)), (
            "the entry differs from the direct kernel call"
        )

        builtin = sentinel.clone()
        assert _mla_rope_into(q, freqs_cis, builtin, heads, rope, offset), (
            "the built-in path took its eager path"
        )
        assert torch.equal(_bits(got), _bits(builtin)), (
            "the entry differs from the built-in _mla_rope_into"
        )

        want = sentinel.clone()
        k3_dspark_rope_into.reference(q, freqs_real, want, heads, rope, offset)
        assert torch.equal(_bits(want[..., rest]), _bits(sentinel[..., rest]))
        _gate_rope(got[..., cols], want[..., cols])


_ROPE_OUT_OF_CONTRACT = {
    # [G, s, rd / 2, 2] unflattened: the kernel wants one freqs row per (G, s).
    "4-D freqs": lambda q, f: (q, f.view(q.shape[0], q.shape[1], *f.shape[1:])),
    "bf16 freqs": lambda q, f: (q, f.bfloat16()),
    "fp16 q": lambda q, f: (q.half(), f),
}


@pytest.mark.parametrize("case", list(_ROPE_OUT_OF_CONTRACT))
def test_rope_rejects_out_of_contract(case) -> None:
    """Both rope guards refuse the inputs listed (they raise before the kernel runs)."""
    spec = k3_dspark_rope_into.CELLS[0].spec
    heads, rope, offset = spec["heads"], spec["rope"], spec["offset"]
    with torch.inference_mode():
        q, _, freqs_real = _rope_into_inputs(spec, 3)
        q, freqs_real = _ROPE_OUT_OF_CONTRACT[case](q, freqs_real)
        out = torch.empty(*q.shape[:-1], spec["out_dim"], dtype=q.dtype, device="cuda")
        with validating(k3_dspark_rope_into), pytest.raises(AssertionError):
            k3_dspark_rope_into(q, freqs_real, out, heads, rope, offset)
        with validating(k3_dspark_rmsnorm_rope), pytest.raises(AssertionError):
            k3_dspark_rmsnorm_rope(q, None, freqs_real, heads, rope, 0.0, False, False, None)


# --------------------------------------------------------------------------------------------------------------
# k3_dspark_mla_decode
# --------------------------------------------------------------------------------------------------------------

_DECODE_REASON = _mla.mla_decode_unavailability_reason()
_needs_decode = pytest.mark.skipif(
    _DECODE_REASON is not None, reason=f"k3_dspark_mla_decode: {_DECODE_REASON}"
)


class _DecodeCase:
    """B requests of a T-row block over the built-in drafter's paged latent cache [P, 1, 1, PAGE, 576] (``pool`` is
    ``layer_cache[:, 0, 0]``, as the drafter passes it), with the fixup the built-in drafter builds from ``ctx_len``.

    Each request owns a row of distinct random pages, one more than its longest length needs, so a replay may
    grow a request. Pages nobody owns are NaN; owned pages hold random rows (block rows included: the built-in drafter
    writes the block's latents there before the call), so a kernel that read another request's pages or past the page
    table would move the output.
    """

    def __init__(self, seed: int, heads: int, batch: int, block: int, lengths=None):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        cpu_gen = torch.Generator().manual_seed(seed)
        self.gen = gen
        self.heads, self.b, self.t = heads, batch, block
        lengths = (
            list(lengths)
            if lengths is not None
            else [LENGTHS[(seed + i) % len(LENGTHS)] for i in range(batch)]
        )
        self.width = max((c + block + PAGE - 1) // PAGE for c in LENGTHS) + 1
        n_pages = batch * self.width + 4
        self.layer_cache = torch.full(
            (n_pages, 1, 1, PAGE, LATENT + ROPE), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        self.pool = self.layer_cache[:, 0, 0]
        perm = torch.randperm(n_pages, generator=cpu_gen)[: batch * self.width]
        self.page_table = perm.view(batch, self.width).to(torch.int32).cuda()
        self.owned = self.page_table.flatten().long()
        self.query = torch.empty(
            batch, block, heads, LATENT + ROPE, dtype=torch.bfloat16, device="cuda"
        )
        self.ctx_len = torch.empty(batch, dtype=torch.int32, device="cuda")
        self.seq_lens = torch.empty(batch, dtype=torch.int32, device="cuda")
        self.kv_bounds = torch.empty(batch * block, dtype=torch.int32, device="cuda")
        self.refresh(lengths)

    def refresh(self, lengths, rows=None) -> None:
        """New query and pool values, new lengths, and (with ``rows``) the page-table rows permuted, all in place."""
        self.query.copy_(torch.randn(self.query.shape, generator=self.gen, device="cuda") * 0.5)
        self.layer_cache[self.owned] = (
            torch.randn(
                self.owned.numel(), 1, 1, PAGE, LATENT + ROPE, generator=self.gen, device="cuda"
            )
            * 0.5
        ).bfloat16()
        if rows is not None:
            self.page_table.copy_(self.page_table[torch.tensor(rows, device="cuda")])
        self.ctx_len.copy_(torch.tensor(lengths, dtype=torch.int32))
        self.fixup = _build_mla_block_fixup(self.ctx_len, self.page_table, self.t, PAGE)
        # In place, so a captured graph sees them.
        self.seq_lens.copy_(self.ctx_len + self.t)
        self.kv_bounds.copy_(self.seq_lens.repeat_interleave(self.t))
        # The entry's operands are the built-in drafter's, value for value.
        assert torch.equal(self.fixup.seq_lens_i32, self.seq_lens)
        assert torch.equal(self.fixup.kv_bounds_i32, self.kv_bounds)
        assert torch.equal(self.fixup.page_tables_i32, self.page_table)

    def workspace(self) -> torch.Tensor:
        """The built-in ``_mla_decode_workspace``: zeros of max(256 MiB, the runner's padded size)."""
        need = _mla.mla_decode_workspace_bytes(self.heads, self.t, LATENT, self.b)
        return torch.zeros(max(_MLA_DECODE_WORKSPACE_BYTES, need), dtype=torch.uint8, device="cuda")

    def args(self, workspace: torch.Tensor):
        return (
            self.query,
            self.pool,
            self.page_table,
            self.seq_lens,
            self.kv_bounds,
            LATENT,
            SOFTMAX_SCALE,
            workspace,
        )


def _builtin_block_decode(query, layer_cache, fixup, block_size, workspace):
    """The body of the built-in ``MLADSparkForCausalLM._cute_dsl_block_decode``, its ``self`` fields as module
    constants (kv_lora_rank, softmax_scale) and its cached workspace as the ``workspace`` argument."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    batch, _, num_heads, _ = query.shape
    device = query.device
    pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
    page_size = pool.shape[1]
    d_latent = LATENT

    q_latent = query.detach()[..., :d_latent].permute(2, 3, 1, 0)
    q_rope = query.detach()[..., d_latent:].permute(2, 3, 1, 0)
    c_latent = pool.detach()[..., :d_latent].permute(1, 2, 0)
    c_rope = pool.detach()[..., d_latent:].permute(1, 2, 0)

    out = torch.empty(batch, block_size, num_heads, d_latent, dtype=query.dtype, device=device)
    workspace_numel = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        num_heads, block_size, d_latent, batch, cutlass.Float32
    )
    assert workspace.numel() >= max(_MLA_DECODE_WORKSPACE_BYTES, workspace_numel)
    torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        fixup.page_tables_i32.t(),
        fixup.seq_lens_i32,
        out.permute(2, 3, 1, 0),
        workspace,
        num_heads,
        block_size,
        page_size,
        SOFTMAX_SCALE,
        1.0,
        batch,
        None,
        fixup.kv_bounds_i32,
    )
    return out


@_needs_decode
@pytest.mark.parametrize(
    "cell", k3_dspark_mla_decode.CELLS, ids=[c.why for c in k3_dspark_mla_decode.CELLS]
)
def test_k3_dspark_mla_decode(cell) -> None:
    spec = cell.spec
    assert (spec["latent"], spec["rope"], spec["page"]) == (LATENT, ROPE, PAGE)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    with torch.inference_mode():
        case = _DecodeCase(seed, spec["heads"], spec["batch"], spec["block"])
        pool0 = case.layer_cache.clone()

        with validating(k3_dspark_mla_decode):
            got = k3_dspark_mla_decode(*case.args(case.workspace()))
        assert torch.equal(_bits(case.layer_cache), _bits(pool0)), "the pool was written"
        assert not got.isnan().any(), "a NaN page was read"

        builtin = _builtin_block_decode(
            case.query, case.layer_cache, case.fixup, case.t, case.workspace()
        )
        assert torch.equal(_bits(got), _bits(builtin)), (
            "the entry differs from the built-in _cute_dsl_block_decode"
        )

        err = _rel_err(got, k3_dspark_mla_decode.reference(*case.args(case.workspace())))
        assert err <= DECODE_TOL, f"rel err {err:.3e}"


@_needs_decode
def test_k3_dspark_mla_decode_graph_replay() -> None:
    """Captured after an eager call, replayed on the eager call's inputs and then on rewritten query / pool,
    lengths and page-table rows: every replay bit for bit an eager call on the same inputs."""
    with torch.inference_mode():
        case = _DecodeCase(53, 4, 8, 7, lengths=(60, 0, 1000, 64, 127, 2041, 5, 128))
        graph_ws = case.workspace()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            k3_dspark_mla_decode(*case.args(graph_ws))  # eager, outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out = k3_dspark_mla_decode(*case.args(graph_ws))
        torch.cuda.synchronize()

        rounds = [
            None,
            ((2041, 5, 128, 60, 0, 1000, 64, 127), (7, 6, 5, 4, 3, 2, 1, 0)),
            ((0, 0, 64, 64, 1, 63, 1900, 300), (1, 3, 5, 7, 0, 2, 4, 6)),
        ]
        for rnd, change in enumerate(rounds):
            if change is not None:
                case.refresh(*change)
            graph.replay()
            torch.cuda.synchronize()
            eager = k3_dspark_mla_decode(*case.args(case.workspace()))
            assert torch.equal(_bits(graph_out), _bits(eager)), f"round {rnd}"
            want = k3_dspark_mla_decode.reference(*case.args(case.workspace()))
            assert _rel_err(graph_out, want) <= DECODE_TOL, f"round {rnd}"


@_needs_decode
def test_mla_decode_can_implement() -> None:
    """The probe passes the drafter's shape and refuses a 256-token page (the kernel's 128-wide KV tile must be a
    multiple of the page, the built-in drafter's ``_assert_cute_dsl_can_implement`` docstring)."""
    args = (2, 7, PAGE, 4, LATENT, ROPE, torch.bfloat16)
    assert _mla.mla_decode_can_implement(*args)
    assert not _mla.mla_decode_can_implement(2, 7, 256, 4, LATENT, ROPE, torch.bfloat16)


_DECODE_OUT_OF_CONTRACT = {
    "page 256": lambda c: dict(
        pool=torch.zeros(4, 256, LATENT + ROPE, dtype=torch.bfloat16, device="cuda")
    ),
    "int64 kv_bounds": lambda c: dict(kv_bounds=c.kv_bounds.long()),
    "kv_bounds per request": lambda c: dict(kv_bounds=c.seq_lens),
    "int64 page table": lambda c: dict(page_table=c.page_table.long()),
}


@_needs_decode
@pytest.mark.parametrize("case_name", list(_DECODE_OUT_OF_CONTRACT))
def test_decode_rejects_out_of_contract(case_name) -> None:
    """The guard refuses the inputs it lists (it raises before the kernel runs)."""
    with torch.inference_mode():
        case = _DecodeCase(5, 4, 2, 7)
        for name, value in _DECODE_OUT_OF_CONTRACT[case_name](case).items():
            setattr(case, name, value)
        with validating(k3_dspark_mla_decode), pytest.raises(AssertionError):
            k3_dspark_mla_decode(*case.args(case.workspace()))


# --------------------------------------------------------------------------------------------------------------
# k3_dspark_mla_paged_decode: flashinfer TRTLLM-Gen, the built-in TRTLLM backend.
# --------------------------------------------------------------------------------------------------------------

_PAGED_REASON = _mla.mla_paged_decode_unavailability_reason()
_needs_paged = pytest.mark.skipif(
    _PAGED_REASON is not None, reason=f"k3_dspark_mla_paged_decode: {_PAGED_REASON}"
)
# max |err| / max |ref|: the kernel rounds P and the output to bf16 (~2^-9 each), as for the CuTe DSL decode.
PAGED_TOL = 1e-2
# The LSE (base 2) of fp32 scores: the kernel sums bf16-rounded P's exponents in fp32; 1e-2 absolute in log2 units.
LSE_TOL = 1e-2


def _paged_args(case: _DecodeCase, workspace: torch.Tensor):
    """The built-in ``_mla_paged_attention`` operands: ``layer_cache[:, 0]``, the int32 page table and lengths (the
    case's own tensors, rewritten in place by ``refresh`` so a captured graph sees new values; equal to the fixup's,
    which ``refresh`` asserts), max_kv_len the pages' capacity per request, fresh out / lse."""
    out = torch.empty(case.b, case.t, case.heads, LATENT, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(case.b * case.t, case.heads, dtype=torch.float32, device="cuda")
    return (
        case.query,
        case.layer_cache[:, 0],
        workspace,
        NOPE,
        LATENT,
        ROPE,
        case.page_table,
        case.seq_lens,
        case.page_table.shape[1] * PAGE,
        SOFTMAX_SCALE,
        out,
        lse,
    )


def _builtin_paged_decode(case: _DecodeCase, workspace: torch.Tensor):
    """The built-in flashinfer call in ``MLADSparkForCausalLM._mla_paged_attention``, positionally as the built-in
    drafter makes it."""
    import flashinfer

    out = torch.empty(case.b, case.t, case.heads, LATENT, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(case.b * case.t, case.heads, dtype=torch.float32, device="cuda")
    flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        case.query,
        case.layer_cache[:, 0],
        workspace,
        NOPE,
        LATENT,
        ROPE,
        case.fixup.page_tables_i32,
        case.fixup.seq_lens_i32,
        case.page_table.shape[1] * PAGE,
        0,
        out,
        SOFTMAX_SCALE,
        1.0,
        None,
        None,
        None,
        backend="trtllm-gen",
        is_var_seq=True,
        uses_shared_paged_kv_idx=True,
        lse=lse,
        return_lse=True,
    )
    return out, lse


def _paged_workspace() -> torch.Tensor:
    """The built-in ``_mla_decode_workspace(device)``: 256 MiB of zeros."""
    return torch.zeros(_MLA_DECODE_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")


@_needs_paged
@pytest.mark.parametrize(
    "cell", k3_dspark_mla_paged_decode.CELLS, ids=[c.why for c in k3_dspark_mla_paged_decode.CELLS]
)
def test_k3_dspark_mla_paged_decode(cell) -> None:
    spec = cell.spec
    assert (spec["latent"], spec["rope"], spec["page"], spec["nope"]) == (LATENT, ROPE, PAGE, NOPE)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    with torch.inference_mode():
        case = _DecodeCase(seed, spec["heads"], spec["batch"], spec["block"])
        pool0 = case.layer_cache.clone()
        args = _paged_args(case, _paged_workspace())
        with validating(k3_dspark_mla_paged_decode):
            k3_dspark_mla_paged_decode(*args)
        got, got_lse = args[-2], args[-1]
        assert torch.equal(_bits(case.layer_cache), _bits(pool0)), "the pool was written"
        assert not got.isnan().any() and not got_lse.isnan().any(), "a NaN page was read"

        builtin_out, builtin_lse = _builtin_paged_decode(case, _paged_workspace())
        assert torch.equal(_bits(got), _bits(builtin_out)), (
            "the entry differs from the built-in flashinfer call"
        )
        assert torch.equal(got_lse, builtin_lse), "the entry's LSE differs from the built-in call's"

        ref_args = _paged_args(case, _paged_workspace())
        k3_dspark_mla_paged_decode.reference(*ref_args)
        err = _rel_err(got, ref_args[-2])
        assert err <= PAGED_TOL, f"rel err {err:.3e}"
        lse_err = (got_lse.double() - ref_args[-1].double()).abs().max().item()
        assert lse_err <= LSE_TOL, f"LSE err {lse_err:.3e} (base 2)"


@_needs_paged
def test_k3_dspark_mla_paged_decode_graph_replay() -> None:
    """Captured after an eager call, replayed on rewritten query / pool, lengths and page-table rows: every replay
    bit for bit an eager call on the same inputs."""
    with torch.inference_mode():
        case = _DecodeCase(57, 4, 8, 7, lengths=(60, 0, 1000, 64, 127, 2041, 5, 128))
        workspace = _paged_workspace()
        graph_args = _paged_args(case, workspace)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            k3_dspark_mla_paged_decode(*graph_args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                k3_dspark_mla_paged_decode(*graph_args)
        torch.cuda.synchronize()
        rounds = [
            None,
            ((2041, 5, 128, 60, 0, 1000, 64, 127), (7, 6, 5, 4, 3, 2, 1, 0)),
            ((0, 0, 64, 64, 1, 63, 1900, 300), (1, 3, 5, 7, 0, 2, 4, 6)),
        ]
        for rnd, change in enumerate(rounds):
            if change is not None:
                case.refresh(*change)
            graph.replay()
            torch.cuda.synchronize()
            eager_args = _paged_args(case, _paged_workspace())
            k3_dspark_mla_paged_decode(*eager_args)
            assert torch.equal(_bits(graph_args[-2]), _bits(eager_args[-2])), f"round {rnd}"
            assert torch.equal(graph_args[-1], eager_args[-1]), f"round {rnd} LSE"
