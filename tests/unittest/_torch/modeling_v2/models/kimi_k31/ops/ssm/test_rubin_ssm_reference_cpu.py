# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The plain-torch references and guards of the Rubin KDA entries, on the CPU with small shapes.

The GPU tests gate each kernel against its entry's reference; these check the references among
themselves where two of them state the same recurrence (convolution + prefill against the verify,
a chunked prefill against a whole one, a snapshot against a shorter prefill, a BF16 state against
an FP32 one) and run every guard. The kernel modules need the GPU toolchain, so where one does not
import here it is replaced by a stand-in whose launchers raise: nothing here launches a kernel.
"""

import importlib
import math
import sys
import types

import pytest
import torch
from _rubin_kda_prefill_inputs import (
    HEAD_DIM,
    assert_scaled_close,
    conv_args,
    prefill_args,
    stacked_columns,
    stacked_projection,
    step_schedule,
)
from _rubin_kda_verify_inputs import MUTATED, K, clone_mutated, fab_args, verify_args

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="Rubin entries: on a GPU only sm_107 runs them",
    ),
]

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
# Kernel module -> constants an entry reads at import.
_KERNELS = {
    "rubin_kda_prefill_ops": {"KDA_PREFILL_SNAPSHOT_CHUNK": 64},
    "rubin_kda_mtp_ops": {},
    "rubin_kda_fab": {},
    "rubin_rms_norm_gated": {},
}
_ENTRIES = (
    "rubin_kda_prefill",
    "rubin_kda_prefill_conv",
    "rubin_kda_mtp_verify",
    "rubin_kda_fab",
    "rubin_rms_norm_gated_token_major",
)
DEV = "cpu"


def _stand_in(name: str, constants: dict) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(constants)

    def launcher(attr):
        def unavailable(*args, **kwargs):
            raise RuntimeError(f"{name}.{attr} is a CPU stand-in")

        return unavailable

    module.__getattr__ = launcher
    return module


@pytest.fixture(scope="module")
def E():
    """The entry instances by name."""
    saved = {}
    for kernel, constants in _KERNELS.items():
        name = f"{_PKG}.kernels.ssm.{kernel}"
        try:
            importlib.import_module(name)
        except Exception:  # the GPU toolchain, or the kernel not at this path in the tree
            saved[name] = sys.modules.get(name)
            sys.modules[name] = _stand_in(name, constants)
    fresh = [e for e in _ENTRIES if f"{_PKG}.ops.ssm.{e}" not in sys.modules]
    try:
        yield {e: getattr(importlib.import_module(f"{_PKG}.ops.ssm.{e}"), e) for e in _ENTRIES}
    finally:
        for e in fresh:  # entries bound to stand-ins must not outlive the module
            sys.modules.pop(f"{_PKG}.ops.ssm.{e}", None)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _scaled_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


# prefill


def test_prefill_reference_carries_state_across_chunks(E) -> None:
    """Two chunks through the pool equal one whole prefill (output and final state)."""
    op = E["rubin_kda_prefill"]
    spec = dict(heads=2, seq_lens=(40,), slots=(1,), fresh=(False,), state_dtype="float32")
    whole = prefill_args(spec, seed=1, device=DEV)
    first = clone_written_pool(whole)
    out_whole = op.reference(**whole)
    outs = []
    for lo, hi in ((0, 17), (17, 40)):
        part = dict(first)
        for name in ("q", "k", "v", "g", "beta"):
            part[name] = first[name][:, lo:hi]
        part["cu_seqlens"] = torch.tensor([0, hi - lo])
        outs.append(op.reference(**part))
    assert torch.equal(torch.cat(outs, dim=1), out_whole)
    assert torch.equal(first["state_pool"][1], whole["state_pool"][1])
    assert out_whole.dtype == torch.bfloat16 and out_whole.shape == whole["q"].shape


def clone_written_pool(args: dict) -> dict:
    return dict(args, state_pool=args["state_pool"].clone())


def test_prefill_reference_snapshot_is_the_prefix_state(E) -> None:
    op = E["rubin_kda_prefill"]
    spec = dict(
        heads=2,
        seq_lens=(70, 64),
        slots=(0, 1),
        fresh=(True, False),
        state_dtype="float32",
        snapshot_rows=(4, 5),
        snapshot_chunks=(1, 1),
    )
    args = prefill_args(spec, seed=2, device=DEV)
    prefix = clone_written_pool(args)
    op.is_valid(**args)
    op.reference(**args)
    # Sequence 0's first 64 tokens alone, from its fresh row, into row 3.
    prefix.update(
        {n: prefix[n][:, :64] for n in ("q", "k", "v", "g", "beta")},
        cu_seqlens=torch.tensor([0, 64]),
        state_indices=torch.tensor([3], dtype=torch.int32),
        snapshot_rows=None,
        snapshot_chunks=None,
    )
    prefix["state_pool"][3].zero_()
    op.reference(**prefix)
    assert torch.equal(args["state_pool"][4], prefix["state_pool"][3])
    # Sequence 1 is exactly one snapshot chunk long: its snapshot is its final state.
    assert torch.equal(args["state_pool"][5], args["state_pool"][1])
    untouched = [r for r in range(args["state_pool"].shape[0]) if r not in (0, 1, 4, 5)]
    before = prefill_args(spec, seed=2, device=DEV)["state_pool"]
    assert torch.equal(args["state_pool"][untouched], before[untouched])


def test_prefill_guard(E) -> None:
    op = E["rubin_kda_prefill"]
    spec = dict(heads=2, seq_lens=(5, 70), slots=(0, 1), fresh=(True, True), state_dtype="float32")
    args = prefill_args(spec, seed=3, device=DEV)
    op.is_valid(**args)
    with pytest.raises(AssertionError, match="unique slot"):
        op.is_valid(**dict(args, state_indices=torch.tensor([1, 1], dtype=torch.int32)))
    with pytest.raises(AssertionError, match="partition"):
        op.is_valid(**dict(args, cu_seqlens=torch.tensor([0, 5, 74])))
    snap = dict(args, snapshot_chunks=torch.tensor([0, 2], dtype=torch.int32))
    snap["snapshot_rows"] = torch.tensor([-1, 2], dtype=torch.int32)
    with pytest.raises(AssertionError, match="inside its sequence"):
        op.is_valid(**snap)
    snap["snapshot_rows"] = torch.tensor([-1, 0], dtype=torch.int32)
    snap["snapshot_chunks"] = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(AssertionError, match="does not update"):
        op.is_valid(**snap)


def test_prefill_cells_have_the_kernel_layout(E) -> None:
    """Each prefill cell's pool has the layout the kernel is compiled for (`_unified_fakes` in
    kda_chunked_bt16): 16-byte aligned [128, 128] state rows, a slot stride of 8-element multiples.
    With snapshots the pool is also the checkpoint tensor, compiled compact: the slot stride must be
    whole 128 x 128 blocks, and a 64-byte pad fails the call on the GPU."""
    block = 128 * 128
    for cell in E["rubin_kda_prefill"].CELLS:
        pool = prefill_args(cell.spec, seed=0, device=DEV)["state_pool"]
        assert pool.stride()[1:] == (block, 128, 1) and pool.data_ptr() % 16 == 0, cell.why
        assert pool.stride(0) % (block if "snapshot_rows" in cell.spec else 8) == 0, cell.why


# convolution


def test_conv_reference_carries_the_window(E) -> None:
    """A sequence split in two (the window carried in the pool) equals the whole; a sequence
    shorter than the window keeps the old window's tail."""
    op = E["rubin_kda_prefill_conv"]
    spec = dict(heads=1, seq_lens=(9,), has_init=(True,), pool_slots=3, src_width_of=4)
    whole = conv_args(spec, seed=4, device=DEV)
    split = dict(whole, conv_state=whole["conv_state"].clone())
    out_whole = op.reference(**whole)
    parts = []
    for lo, hi in ((0, 2), (2, 9)):
        parts.append(
            op.reference(
                **dict(
                    split,
                    src=split["src"][lo:hi],
                    starts=torch.tensor([0, hi - lo]),
                    max_seqlen=hi - lo,
                )
            )
        )
    for i in range(3):
        assert torch.equal(torch.cat([p[i] for p in parts], dim=1), out_whole[i])
    assert torch.equal(split["conv_state"], whole["conv_state"])

    short = conv_args(dict(spec, seq_lens=(1,)), seed=5, device=DEV)
    slot = int(short["slots"][0])
    old = short["conv_state"][slot].clone()
    op.reference(**short)
    assert torch.equal(short["conv_state"][slot][:, :2], old[:, 1:])
    assert torch.equal(short["conv_state"][slot][:, 2], short["src"][0, : 3 * 128])


def test_conv_guard(E) -> None:
    op = E["rubin_kda_prefill_conv"]
    spec = dict(heads=1, seq_lens=(3, 4), has_init=(True, False), pool_slots=4, src_width_of=4)
    args = conv_args(spec, seed=6, device=DEV)
    op.is_valid(**args)
    with pytest.raises(AssertionError, match="unique"):
        op.is_valid(**dict(args, slots=torch.tensor([2, 2], dtype=torch.int32)))
    with pytest.raises(AssertionError, match="out of range"):
        op.is_valid(**dict(args, slots=torch.tensor([0, 4], dtype=torch.int32)))


# verify


def test_verify_golden_row_is_conv_then_prefill(E) -> None:
    """With nothing to replay, the verify's committed state is one convolution + prefill step of
    its first row (the two other references; the convolution rounds its outputs to bf16)."""
    verify, conv, prefill = (
        E["rubin_kda_mtp_verify"],
        E["rubin_kda_prefill_conv"],
        E["rubin_kda_prefill"],
    )
    heads = 2
    dim = heads * K
    args = verify_args(dict(N=1, H=heads, num_spec=2, pool=2), seed=7, device=DEV)
    args["num_accepted_tokens"].zero_()
    for name in ("w_q", "w_k", "w_v"):  # bf16-representable: the convolution op takes bf16
        args[name] = args[name].to(torch.bfloat16).float()
    slot = int(args["ssm_state_indices"][0])
    ref = clone_mutated(args)
    out = verify.reference(**args)
    assert out.shape == (1, 3, heads, K) and out.dtype == torch.bfloat16

    # The golden row's raw q | k | v, channel-major like the convolution's src.
    row = torch.cat([args[n][0, :1].reshape(1, dim) for n in ("x_q", "x_k", "x_v")], dim=-1)
    window = torch.cat(
        [ref["cs_q"][slot, :, :3], ref["cs_k"][slot, :, :3], ref["cs_v"][slot, :, :3]]
    )
    pool = window.unsqueeze(0).clone()
    q, k, v = conv.reference(
        src=row,
        weight=torch.cat([args["w_q"], args["w_k"], args["w_v"]]).to(torch.bfloat16),
        conv_state=pool,
        starts=torch.tensor([0, 1], dtype=torch.int32),
        slots=torch.tensor([0], dtype=torch.int32),
        has_init=torch.tensor([True]),
        max_seqlen=1,
    )
    state = ref["recurrent_state"].clone()
    prefill.reference(
        q=q,
        k=k,
        v=v,
        g=args["g"][:, :1],
        beta=args["beta"][:, :1],
        state_pool=state,
        state_indices=torch.tensor([slot], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 1]),
        A_log=args["A_log"],
        dt_bias=args["dt_bias"],
        scale=args["scale"],
        lower_bound=args["lower_bound"],
    )
    assert _scaled_err(args["recurrent_state"][slot], state[slot]) < 2e-2
    # The committed window is the old one shifted by the golden row's raw inputs.
    assert torch.equal(args["cs_q"][slot, :, :3], pool[0, :dim])


def test_verify_replays_accepted_drafts_first(E) -> None:
    """A round that accepted the previous round's drafts replays them before its new rows, so its
    committed state differs from the same round with nothing accepted; the replay caches are what
    the previous round stored for its draft rows."""
    verify = E["rubin_kda_mtp_verify"]
    a = verify_args(dict(N=1, H=1, num_spec=2, pool=1), seed=8, device=DEV)
    a["num_accepted_tokens"].zero_()
    verify.reference(**a)  # round 1 commits row 0 and caches rows 1 and 2 as drafts
    assert a["g_cache"].le(0).all() and a["g_cache"].ge(a["lower_bound"]).all()
    b = clone_mutated(a)
    a["num_accepted_tokens"].fill_(2)
    verify.reference(**a)
    b["num_accepted_tokens"] = torch.zeros_like(a["num_accepted_tokens"])
    verify.reference(**b)
    assert not torch.equal(a["recurrent_state"], b["recurrent_state"])


def test_verify_fused_gate_equals_the_projected_gate(E) -> None:
    verify = E["rubin_kda_mtp_verify"]
    spec = dict(N=2, H=2, num_spec=1, pool=3, gate="fa")
    fused = verify_args(spec, seed=9, device=DEV)
    plain = clone_mutated(fused)
    rows = fused["fa"].shape[1]
    gate = (fused["fa"][0].float() @ fused["w_fb"].float().t()).to(torch.bfloat16)
    plain.update(g=gate.view(1, rows, 2, K), fa=None, w_fb=None)
    assert torch.equal(verify.reference(**fused), verify.reference(**plain))


def test_verify_mxfp8_output_dequantizes_to_the_bf16_output(E) -> None:
    verify = E["rubin_kda_mtp_verify"]
    spec = dict(N=3, H=2, num_spec=2, pool=4)
    bf16 = verify_args(spec, seed=10, device=DEV)
    mx = dict(clone_mutated(bf16), quantize_output=True)
    out = verify.reference(**bf16).reshape(-1, 2, K).float()
    codes, scales = verify.reference(**mx)
    rows = out.shape[0]
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == (rows, 2 * K)
    assert scales.dtype == torch.uint8 and scales.numel() == (rows + 127) // 128 * 128 * 2 * 4
    r = torch.arange(rows).view(-1, 1)
    h = torch.arange(2).view(1, -1)
    off = (r // 128) * 2 * 512 + h * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
    for replica in range(1, 4):
        assert torch.equal(scales[off + replica], scales[off])
    sf = torch.pow(2.0, scales[off].float() - 127).unsqueeze(-1)
    values = codes.float().view(rows, 2, K) * sf
    amax = out.abs().amax(dim=-1, keepdim=True)
    assert bool((amax / sf <= 448).all()) and bool((amax / sf > 224).all()), (
        "scale is 2^ceil(log2(amax/448))"
    )
    assert ((values - out).abs() <= sf * 16 + 1e-12).all(), (
        "within half an e4m3 step of the top binade"
    )


def test_verify_mxfp8_exact_output_is_the_catalog_quantization(E) -> None:
    """With mxfp8_exact the MXFP8 output is the catalog quantizer's form of the BF16 output: the
    values and scale bytes of `quantize_mxfp8_reference`, the bytes in R128c4 order, one scale per
    32 values (not one per row and head)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
        mxfp8_scales_to_r128c4,
        quantize_mxfp8_reference,
    )

    verify = E["rubin_kda_mtp_verify"]
    spec = dict(N=3, H=2, num_spec=2, pool=4)
    bf16 = verify_args(spec, seed=18, device=DEV)
    mx = dict(clone_mutated(bf16), quantize_output=True, mxfp8_exact=True)
    out = verify.reference(**bf16).reshape(-1, 2 * K)
    codes, scales = verify.reference(**mx)
    values, byte = quantize_mxfp8_reference(out)
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == out.shape
    assert torch.equal(codes.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(scales, mxfp8_scales_to_r128c4(byte))
    groups = byte.view(out.shape[0], 2, 4)
    assert bool((groups != groups[..., :1]).any()), "a head's four groups keep their own scales"
    verify.is_valid(**mx)
    with pytest.raises(AssertionError, match="needs quantize_output"):
        verify.is_valid(**dict(bf16, mxfp8_exact=True))


def test_verify_guard(E) -> None:
    verify = E["rubin_kda_mtp_verify"]
    args = verify_args(dict(N=2, H=1, num_spec=3, pool=4), seed=11, device=DEV)
    verify.is_valid(**args)
    slots = args["ssm_state_indices"].long()
    bad = dict(args, num_accepted_tokens=args["num_accepted_tokens"].clone())
    bad["num_accepted_tokens"][slots[0]] = 4
    with pytest.raises(AssertionError, match="num_accepted_tokens"):
        verify.is_valid(**bad)
    with pytest.raises(AssertionError, match="distinct"):
        verify.is_valid(**dict(args, ssm_state_indices=slots[:1].repeat(2).to(torch.int32)))
    with pytest.raises(AssertionError, match=r"\[0, pool\)"):
        verify.is_valid(**dict(args, ssm_state_indices=torch.tensor([0, 4], dtype=torch.int32)))


# fab, output gate


def test_fab_reference_modes_agree(E) -> None:
    fab = E["rubin_kda_fab"]
    args = fab_args(
        dict(rows=3, heads=4, hidden=64, fa_only=False, multi_tile=False), seed=12, device=DEV
    )
    beta, gate = fab.reference(**args)
    bfa = fab.reference(**dict(args, fa_only=True))
    assert bfa.shape == (3, 136) and beta.shape == (3, 4) and gate.shape == (3, 4 * K)
    assert torch.equal(bfa[:, K : K + 4], beta)
    assert torch.equal(bfa[:, K + 4 :], torch.zeros_like(bfa[:, K + 4 :]))  # zero weight rows
    want = (bfa[:, :K].double() @ args["w_fb"].double().t()).to(torch.bfloat16)
    assert _scaled_err(gate, want) <= 2.0**-8


def test_output_gate_reference(E) -> None:
    norm = E["rubin_rms_norm_gated_token_major"]
    tokens, heads, n = 3, 2, 8
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(tokens * heads, n, generator=gen).to(torch.bfloat16)
    proj = torch.randn(tokens, 4 * heads * n, generator=gen).to(torch.bfloat16)
    z_view = proj[:, 3 * heads * n :].view(tokens, heads, n)
    w = torch.rand(n, generator=gen).to(torch.bfloat16)
    sig = norm.reference(x, z_view, w, 1e-5, None, "sigmoid")
    assert torch.equal(sig, norm.reference(x, z_view.contiguous(), w, 1e-5, None, "sigmoid"))
    xf, zf = x.float(), z_view.reshape(-1, n).float()
    rms = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()
    assert _scaled_err(sig, rms * torch.sigmoid(zf)) < 2.0**-7
    silu = norm.reference(x, z_view, w, 1e-5, None, "silu")
    assert _scaled_err(silu, rms * torch.sigmoid(zf) * zf) < 2.0**-7
    fp8 = norm.reference(x, z_view, w, 1e-5, torch.tensor([0.5]), "sigmoid")
    assert fp8.dtype == torch.float8_e4m3fn
    norm.is_valid(x, z_view, w, 1e-5, torch.tensor([0.5]), "sigmoid")
    with pytest.raises(AssertionError, match="exactly one element"):
        norm.is_valid(x, z_view, w, 1e-5, torch.ones(2), "sigmoid")


# stacked projection

#: entry -> (input builder, heads key, {argument: (section of the projection, token dimension)})
_STACKED_VIEWS = {
    "rubin_kda_prefill_conv": (conv_args, "heads", {"src": ("q", 0)}),
    "rubin_kda_prefill": (prefill_args, "heads", {"beta": ("b", 1)}),
    "rubin_kda_mtp_verify": (
        verify_args,
        "H",
        dict(x_q=("q", 1), x_k=("k", 1), x_v=("v", 1), onorm_g=("g", 1), beta=("b", 1)),
    ),
}


def test_stacked_projection_is_the_targets_layout() -> None:
    """q, k, v, g (H * 128 columns each), f_a (128), b (H), then zero columns to a multiple of 16:
    the row pitch is 6288 at 12 heads (TP8) and 49376 at 96 (DEP16)."""
    gen = torch.Generator().manual_seed(14)
    for heads, pitch in ((12, 6288), (96, 49376)):
        d = heads * K
        cols = stacked_columns(heads)
        want = dict(q=0, k=d, v=2 * d, g=3 * d, f_a=4 * d, b=4 * d + K, pad=4 * d + K + heads)
        assert cols == dict(want, N=pitch)
        out = stacked_projection(3, heads, gen, DEV)
        assert out.shape == (3, pitch) and out.dtype == torch.bfloat16 and out.is_contiguous()
        assert not out[:, cols["pad"] :].ne(0).any() and out[:, : cols["pad"]].ne(0).any()


def test_stacked_cells_view_one_projection(E) -> None:
    """Each `stacked` cell hands its entry views of one such output, at the specialized targets' row
    pitch and columns, at 12 and at 96 heads (the verify has several cells at 12: helix8's rows)."""
    for name, (build, heads_key, views) in _STACKED_VIEWS.items():
        cells = [c for c in E[name].CELLS if c.spec.get("stacked")]
        assert sorted({c.spec[heads_key] for c in cells}) == [12, 96], name
        for cell in cells:
            cols = stacked_columns(cell.spec[heads_key])
            args = build(cell.spec, seed=0, device=DEV)
            storage = {args[arg].untyped_storage().data_ptr() for arg in views}
            assert len(storage) == 1, cell.why
            for arg, (section, dim) in views.items():
                t = args[arg]
                assert (t.stride(dim), t.storage_offset()) == (cols["N"], cols[section]), arg
            if name == "rubin_kda_prefill_conv":  # the whole output, as the targets pass it
                assert args["src"].shape[1] == cols["N"], cell.why
    norm = E["rubin_rms_norm_gated_token_major"].CELLS
    assert sorted({c.spec["heads"] for c in norm if c.spec.get("stacked")}) == [12, 96]


def test_references_read_the_stacked_views(E) -> None:
    """On the views each reference computes what it computes on dense copies of them."""
    conv, prefill, verify = (
        E["rubin_kda_prefill_conv"],
        E["rubin_kda_prefill"],
        E["rubin_kda_mtp_verify"],
    )
    seqs = dict(heads=2, seq_lens=(5, 3), stacked=True)
    args = conv_args(dict(seqs, has_init=(True, False), pool_slots=3), seed=15, device=DEV)
    dense = dict(args, src=args["src"][:, : 3 * 2 * K].contiguous())
    dense["conv_state"] = args["conv_state"].clone()
    for got, want in zip(conv.reference(**args), conv.reference(**dense)):
        assert torch.equal(got, want)
    assert torch.equal(args["conv_state"], dense["conv_state"])
    spec = dict(seqs, slots=(0, 1), fresh=(True, False), state_dtype="float32")
    args = prefill_args(spec, seed=16, device=DEV)
    dense = dict(clone_written_pool(args), beta=args["beta"].contiguous())
    assert torch.equal(prefill.reference(**args), prefill.reference(**dense))
    assert torch.equal(args["state_pool"], dense["state_pool"])
    args = verify_args(dict(N=2, H=2, num_spec=2, stacked=True), seed=17, device=DEV)
    dense = clone_mutated(args)
    dense.update({n: args[n].contiguous() for n in _STACKED_VIEWS["rubin_kda_mtp_verify"][2]})
    assert torch.equal(verify.reference(**args), verify.reference(**dense))
    assert torch.equal(args["recurrent_state"], dense["recurrent_state"])


def test_entry_declarations(E) -> None:
    """Every entry lists its cells and says why each one exists."""
    for name, entry in E.items():
        assert entry.CELLS and all(cell.why for cell in entry.CELLS), name
        specs = [repr(sorted(cell.spec.items())) for cell in entry.CELLS]
        assert len(set(specs)) == len(specs), f"{name}: a repeated cell spec"


# BF16 state

#: One rounding to BF16 moves a value by at most this fraction of it (8 significant bits).
BF16_ROUNDING = 2.0**-9
# The entries' output bands (their GPU tests): the verify's BF16 output elementwise and in relative
# RMS; the prefill's output within 2e-2 of the largest magnitude.
VERIFY_OUT_ATOL, VERIFY_OUT_RTOL, VERIFY_OUT_RMS = 3.2e-2, 8e-3, 2.5e-3
PREFILL_TOL_OUT = 2e-2
#: The request of the drift check: 12 heads, two requests, three drafts, six verify rounds.
_CHAIN = dict(H=12, N=2, num_spec=3, rounds=6)


def _bf16_values(t: torch.Tensor) -> None:
    """Round an FP32 tensor's values to BF16 in place, so that a BF16 pool can hold them exactly."""
    t.copy_(t.to(torch.bfloat16))


def _rel_rms(got: torch.Tensor, want: torch.Tensor) -> float:
    diff = (got.float() - want.float()).pow(2).mean().sqrt()
    return (diff / want.float().pow(2).mean().sqrt().clamp_min(1e-12)).item()


@torch.inference_mode()
def test_verify_reference_rounds_only_the_committed_state(E) -> None:
    """From the same state values, a BF16 pool gives the output, conv windows and replay caches of
    an FP32 pool bit for bit, and holds the FP32 pool's committed state rounded to BF16."""
    op = E["rubin_kda_mtp_verify"]
    spec = dict(N=2, H=12, num_spec=3, pool=4, stacked=True)
    f32 = verify_args(dict(spec, state_dtype="float32"), seed=18, device=DEV)
    b16 = verify_args(dict(spec, state_dtype="bfloat16"), seed=18, device=DEV)
    _bf16_values(f32["recurrent_state"])
    assert torch.equal(f32["recurrent_state"].to(torch.bfloat16), b16["recurrent_state"])
    assert torch.equal(op.reference(**f32), op.reference(**b16))
    for name in MUTATED:
        if name != "recurrent_state":
            assert torch.equal(f32[name], b16[name]), name
    assert torch.equal(f32["recurrent_state"].to(torch.bfloat16), b16["recurrent_state"])


@torch.inference_mode()
def test_prefill_reference_rounds_only_the_committed_state(E) -> None:
    """From the same state values, the prefill's output on a BF16 pool is the FP32 pool's bit for
    bit, and its final states and snapshots are the FP32 ones rounded to BF16."""
    op = E["rubin_kda_prefill"]
    spec = dict(
        heads=12,
        seq_lens=(70, 9),
        slots=(0, 1),
        fresh=(False, True),
        snapshot_rows=(2, -1),
        snapshot_chunks=(1, 0),
        stacked=True,
    )
    schedule = dict(zip(("seq_order", "cu_chunks"), step_schedule(spec["seq_lens"], DEV)))
    f32 = dict(prefill_args(dict(spec, state_dtype="float32"), seed=19, device=DEV), **schedule)
    b16 = dict(prefill_args(dict(spec, state_dtype="bfloat16"), seed=19, device=DEV), **schedule)
    _bf16_values(f32["state_pool"])
    assert torch.equal(f32["state_pool"].to(torch.bfloat16), b16["state_pool"])
    assert torch.equal(op.reference(**f32), op.reference(**b16))
    assert torch.equal(f32["state_pool"].to(torch.bfloat16), b16["state_pool"])


def _run_request(op_prefill, op_verify, chain: dict, state_dtype: torch.dtype) -> tuple:
    """One batch's life on a pool of `state_dtype`: two prefill chunks (new, then continued), then
    verify rounds whose accepted counts cycle. Returns the outputs per step, the FP32 values of the
    batch's state rows, and their norm after each commit."""
    heads, n, k = chain["H"], chain["N"], chain["num_spec"]
    slots = (1, 3)[:n]
    pool = torch.zeros(4, heads, HEAD_DIM, HEAD_DIM, dtype=state_dtype)
    outputs, norms = [], []
    for chunk, (lens, fresh) in enumerate((((40, 25)[:n], True), ((30, 17)[:n], False))):
        spec = dict(heads=heads, seq_lens=lens, slots=slots, fresh=(fresh,) * n, stacked=True)
        args = prefill_args(dict(spec, state_dtype="float32"), seed=10 + chunk, device=DEV)
        seq_order, cu_chunks = step_schedule(lens, DEV)
        args.update(state_pool=pool, seq_order=seq_order, cu_chunks=cu_chunks)
        outputs.append(op_prefill.reference(**args))
        norms.append(pool[list(slots)].float().norm().item())
    spec = dict(N=n, H=heads, num_spec=k, pool=4, stacked=True)
    first = verify_args(dict(spec, state_dtype="float32"), seed=99, device=DEV)
    carried = {name: first[name] for name in MUTATED if name != "recurrent_state"}
    indices = torch.tensor(slots, dtype=torch.int32)
    accepted = torch.zeros(4, dtype=torch.int32)  # nothing pending after the prefill
    for r in range(chain["rounds"]):
        args = verify_args(dict(spec, state_dtype="float32"), seed=200 + r, device=DEV)
        args.update(carried, recurrent_state=pool, ssm_state_indices=indices)
        args["num_accepted_tokens"] = accepted
        outputs.append(op_verify.reference(**args))
        norms.append(pool[list(slots)].float().norm().item())
        accepted = accepted.clone()
        accepted[list(slots)] = torch.tensor([(r + i) % (k + 1) for i in range(n)]).to(torch.int32)
    return outputs, pool[list(slots)].float(), norms


@torch.inference_mode()
def test_a_request_on_a_bf16_state_stays_within_the_bands(E) -> None:
    """Prefill chunks and verify rounds on a BF16 pool against the same steps on an FP32 one.

    Both runs read the same inputs, so their states differ only through the roundings of the BF16
    run's commits (one per step, each at most 2^-9 of the state). A step maps that difference
    through a decay of at most one and a delta-rule update `I - beta k k^T` (`|k| = 1`, `0 < beta <
    1`), neither of which grows it, so after `c` commits it is at most `c * 2^-9` of the largest
    state norm along the way. The outputs stay within the entries' own bands.
    """
    prefill, verify = E["rubin_kda_prefill"], E["rubin_kda_mtp_verify"]
    out_f32, state_f32, norms = _run_request(prefill, verify, _CHAIN, torch.float32)
    out_b16, state_b16, _ = _run_request(prefill, verify, _CHAIN, torch.bfloat16)
    for step, (got, want) in enumerate(zip(out_b16, out_f32)):
        if step < 2:
            assert_scaled_close(f"prefill chunk {step}", got, want, PREFILL_TOL_OUT)
            continue
        assert _rel_rms(got, want) < VERIFY_OUT_RMS, f"verify round {step - 2}"
        torch.testing.assert_close(
            got.float(), want.float(), atol=VERIFY_OUT_ATOL, rtol=VERIFY_OUT_RTOL
        )
    commits = len(norms)
    # 1 + 2^-9 * commits: the BF16 run's own norm, which its roundings act on.
    bound = commits * BF16_ROUNDING * max(norms) * (1 + commits * BF16_ROUNDING)
    drift = (state_b16 - state_f32).norm().item()
    assert drift <= bound, f"state drift {drift:.3e} against {bound:.3e} after {commits} commits"
    assert math.isfinite(drift) and drift > 0, "the BF16 run must round something"
