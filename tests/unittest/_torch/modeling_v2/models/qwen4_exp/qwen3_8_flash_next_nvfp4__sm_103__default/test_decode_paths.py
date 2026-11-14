# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's decode-side paths at one rank (tp1): the draft layer's front end, the heads
and their logits, and the Gated-DeltaNet verify step's fused conv.

* **The draft front end**: the token embedding under rejection sampling, and the dense
  projections' dispatch between `fp8_gemv` and the bf16 projection.
* **The logits**: the target's logits processor returns the shell's logits bit for bit (the
  same row gather, the same fp32 values, its head's bf16 logits widened by one vectorized
  kernel); a consumer of the row argmax alone may take the bf16 logits; the dense FP8 verify
  head reads the NVFP4 copy at decode rows.
* **The verify step's conv** (`DecodeTarget._gdn_block`) under the dense FP8 mode and over the
  in_proj's NVFP4 copy (`TRTLLM_MODELING_V2_GDN_INPROJ_NVFP4`, up to
  `_INPROJ_NVFP4_MAX_ROWS`). A captured verify step runs the in_proj with the multi-token conv
  step as its epilogue (`fp8_gemv_conv1d_verify` / `nvfp4_split_gemv_conv1d_verify`); an eager
  one runs the projection and then `causal_conv1d_triton_update`. Both must hand the cached
  replay the same thing -- the conv output (contiguous, as the replay reads it), `a`, `b` and
  `z` bitwise -- and leave the conv pool and the per-step snapshot buffer bitwise the same. The
  replay is stubbed to record its inputs (it is the same call on both sides), the epilogue to
  record `z`. Over a bf16 pool (the CuTe replay) `a` and `b` reach it as views of the
  projection, not copies.
"""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.fp8_gemv import (  # noqa: E501
    pack_from_bf16,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (  # noqa: E501
    pack_split_from_bf16,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative.interface import rejection_sampling_one_model

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs on a CUDA device")
DEV, BF16 = "cuda", torch.bfloat16


# ---- the draft front end ------------------------------------------------------------------


@_cuda
def test_draft_embedding_reads_row_0_for_rejected_slots() -> None:
    """Rejection sampling pads the slots past a request's accepted count with -1 (here every
    draft is rejected at slot 0); the front end embeds those as row 0 and every valid id as
    the plain lookup does."""
    batch, draft, vocab, hidden = 8, 3, 4096, 256
    draft_tokens = torch.arange(batch * draft, dtype=torch.int32, device=DEV).reshape(batch, draft)
    draft_probs = torch.zeros(batch, draft, vocab, device=DEV)
    draft_probs.scatter_(2, draft_tokens.long().unsqueeze(-1), 1.0)
    target_probs = torch.zeros(batch, draft + 1, vocab, device=DEV)
    target_probs[..., vocab - 1] = 1.0
    seed = torch.zeros(batch, dtype=torch.int64, device=DEV)
    offset = torch.arange(batch, dtype=torch.int64, device=DEV)
    accepted, _ = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, seed, offset
    )
    assert bool((accepted[:, 1:] == -1).all()) and bool((accepted[:, 0] == vocab - 1).all())

    g = torch.Generator(device=DEV).manual_seed(0)
    table = torch.randn(vocab, hidden, generator=g, device=DEV).to(BF16)
    valid_ids = torch.randint(0, vocab, (257,), generator=g, device=DEV, dtype=torch.int32)
    for ids in (accepted.flatten(), valid_ids, valid_ids.long()):
        e = M._draft_embedding(ids, table)
        valid = ids >= 0
        assert torch.equal(e[valid], table[ids[valid].long()])
        assert torch.equal(e[~valid], table[0].expand(int((~valid).sum()), -1))


@_cuda
@pytest.mark.parametrize("rows", [1, 16, 17])
def test_draft_projection_dispatch(rows: int) -> None:
    """`_draft_mm`: `fp8_gemv` over the packed copy up to `_MTP_DENSE_FP8_MAX_ROWS` rows, the
    bf16 projection past them and without a copy."""
    g = torch.Generator(device=DEV).manual_seed(rows)
    w = (torch.randn(2560, 2560, generator=g, device=DEV) * 0.02).to(BF16)
    x = torch.randn(rows, 2560, generator=g, device=DEV).to(BF16)
    packed = pack_from_bf16(w)
    got = M._draft_mm(x, w.t(), packed, cublas_mm)
    if rows <= M._MTP_DENSE_FP8_MAX_ROWS:
        assert torch.equal(got, M.fp8_gemv_op(x, packed))
        ref = x.float() @ w.float().t()
        assert ((got.float() - ref).norm() / ref.norm()).item() < 0.05
    else:
        assert torch.equal(got, cublas_mm(x, w.t()))
    assert torch.equal(M._draft_mm(x, w.t(), None, cublas_mm), cublas_mm(x, w.t()))


# ---- the logits ---------------------------------------------------------------------------

VOCAB, HIDDEN = 248320, 2560


@pytest.fixture(scope="module")
def head():
    g = torch.Generator(device=DEV).manual_seed(0)
    lin = torch.nn.Linear(HIDDEN, VOCAB, bias=False, device=DEV, dtype=BF16)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=g, device=DEV) * 0.02)
    return lin


@_cuda
@pytest.mark.parametrize("seq_lens", [[1] * 128, [5000, 3000, 1] + [1] * 121])
def test_logits_match_the_shell_processor_bitwise(head, seq_lens) -> None:
    g = torch.Generator(device=DEV).manual_seed(len(seq_lens))
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    hidden = torch.randn(int(lens.sum()), HIDDEN, generator=g, device=DEV).to(BF16)
    md = SimpleNamespace(seq_lens_cuda=lens)
    with torch.no_grad():
        want = LogitsProcessor().forward(hidden, head, md)
        got = M._WideningLogitsProcessor().forward(hidden, head, md)
    assert got.dtype == torch.float32 and got.shape == (len(seq_lens), VOCAB)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@_cuda
def test_context_logits_and_no_metadata(head) -> None:
    g = torch.Generator(device=DEV).manual_seed(9)
    hidden = torch.randn(33, HIDDEN, generator=g, device=DEV).to(BF16)
    with torch.no_grad():
        for md, ctx in ((None, False), (SimpleNamespace(seq_lens_cuda=None), True)):
            want = LogitsProcessor().forward(hidden, head, md, ctx)
            got = M._WideningLogitsProcessor().forward(hidden, head, md, ctx)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@_cuda
@pytest.mark.parametrize("rows", [1, 4, 256])
def test_unwidened_logits_give_the_same_greedy_tokens(head, rows) -> None:
    """`widen=False` returns the head's own bf16 logits, whose greedy samplers (the verify
    sampler's `greedy_argmax`, the draft sampler's int32 one) pick the widened logits' tokens."""
    from tensorrt_llm._torch.speculative.greedy_argmax import greedy_argmax

    g = torch.Generator(device=DEV).manual_seed(rows)
    hidden = torch.randn(rows, HIDDEN, generator=g, device=DEV).to(BF16)
    proc = M._WideningLogitsProcessor()
    with torch.no_grad():
        wide = proc.forward(hidden, head, None, True)
        narrow = proc.forward(hidden, head, None, True, widen=False)
        # Without return_context_logits the shell's gather and fp32 contract stay.
        gathered = proc.forward(hidden, head, None, False, widen=False)
    assert narrow.dtype == BF16 and wide.dtype == torch.float32
    assert gathered.dtype == torch.float32
    assert torch.equal(narrow.float().view(torch.int32), wide.view(torch.int32))
    narrow[0, 17] = narrow[0].max()  # an exact tie, broken to the lower index either way
    wide = narrow.float()
    assert torch.equal(greedy_argmax(narrow), greedy_argmax(wide))
    assert torch.equal(greedy_argmax(narrow, torch.int32), greedy_argmax(wide, torch.int32))
    assert torch.equal(greedy_argmax(narrow), torch.argmax(wide, dim=-1))


@_cuda
@pytest.mark.parametrize("rows", [1, 3, 16, 17])
def test_fp8_verify_head_reads_the_nvfp4_copy_at_decode_rows(head, rows) -> None:
    """With an NVFP4 copy (`_TARGET_HEAD_NVFP4_ENV`) the dense FP8 verify head's decode rows
    are `nvfp4_gemv` over it; past `_FP8_GEMV_HEAD_ROWS` they take the per-row FP8 copy as
    the head without it does."""
    rowwise = M._fp8_rowwise(head.weight.data)
    nv = M.nvfp4_pack_from_bf16(head.weight.data)
    g = torch.Generator(device=DEV).manual_seed(100 + rows)
    hidden = torch.randn(rows, HIDDEN, generator=g, device=DEV).to(BF16)
    with torch.no_grad():
        got = M._Fp8Head(*rowwise, None, nv)(hidden)
        if rows <= M._FP8_GEMV_HEAD_ROWS:
            want = M.nvfp4_gemv_op(hidden, nv)
        else:
            want = M._Fp8Head(*rowwise, None, None)(hidden)
    assert got.shape == (rows, VOCAB) and torch.equal(got, want)


def _gate(guided=None, greedy=True, rejection=False, penalty=False, relaxed=False):
    worker = SimpleNamespace(
        guided_decoder=guided,
        spec_config=SimpleNamespace(use_relaxed_acceptance_for_thinking=relaxed),
    )
    md = SimpleNamespace(
        is_all_greedy_sample=greedy,
        wants_advanced_draft_sampling=rejection and not greedy,
        enable_penalty=penalty,
    )
    return M._spec_argmax_only(worker, md)


@pytest.mark.cpu_only
def test_argmax_only_gate() -> None:
    assert _gate() == (True, True)
    # A non-greedy batch: the verify sampler samples; the draft stays argmax unless
    # rejection sampling needs its distribution.
    assert _gate(greedy=False) == (False, True)
    assert _gate(greedy=False, rejection=True) == (False, False)
    # The occurrence penalties and relaxed acceptance read the verify logits only.
    assert _gate(penalty=True) == (False, True)
    assert _gate(relaxed=True) == (False, True)
    # A guided decoder edits both.
    assert _gate(guided=object()) == (False, False)


# ---- the Gated-DeltaNet verify step's conv ------------------------------------------------

KEY, HV, V, CONV, VD, SLOTS, ROWS = 16, 48, 128, 10240, 6144, 461, 32


@pytest.fixture(scope="module", params=["fp8", "nvfp4"])
def gdn_layer(request):
    g = torch.Generator(device=DEV).manual_seed(0)
    w = (torch.randn(CONV + VD + 2 * HV, HIDDEN, generator=g, device=DEV) * HIDDEN**-0.5).to(BF16)
    nv = request.param == "nvfp4"
    return types.SimpleNamespace(
        in_proj=w.t(),
        in_proj8=None,
        in_proj_m=None,
        in_proj_g=None if nv else pack_from_bf16(w),
        in_proj_nv=pack_split_from_bf16(w) if nv else None,
        conv_w=(torch.randn(CONV, 4, generator=g, device=DEV) * 0.5).to(BF16),
        a_log=None,
        dt_bias=None,
    )


def _gdn_block(layer, x, si, int_idx, conv0, win0, p, captured, monkeypatch):
    """One verify `_gdn_block` on copies of the pools; returns what the cached replay and the
    epilogue were handed, and the pools."""
    seen = {}

    def replay(**kw):
        mq_out = kw["packed_qkv"]
        n, p_ = kw["g"].shape[:2]
        seen.update(
            mq_out=mq_out.clone(),
            a=kw["g"].clone(),
            b=kw["beta"].clone(),
            pdl=kw["launch_with_pdl"],
        )
        assert mq_out.is_contiguous() and mq_out.shape == (n * p_, CONV)
        # The CuTe replay reads a / b through their row stride: views, not copies.
        assert kw["g"].stride(1) > HV and kw["beta"].stride(1) > HV
        return torch.zeros(n, p_, HV, V, dtype=BF16, device=DEV)

    monkeypatch.setattr(M, "gdn_cached_replay_cute", replay)
    history = torch.zeros(1, device=DEV)
    core = types.SimpleNamespace(
        conv_dim=CONV,
        key_dim=KEY * 128,
        value_dim=VD,
        num_k_heads=KEY,
        head_k_dim=128,
        num_v_heads=HV,
        head_v_dim=V,
        in_proj_split=(CONV, VD, HV, HV),
        decode_in_mm=cublas_mm,
        _bound=types.SimpleNamespace(
            pools=types.SimpleNamespace(
                conv={0: conv0.clone()}, ssm={0: torch.zeros(1, dtype=BF16, device=DEV)}
            ),
            int_conv={0: win0.clone()},
            replay=types.SimpleNamespace(
                old={0: (history,) * 4},
                cache_buf_idx=None,
                prev_num_accepted=None,
                history=16,
                ring=None,
                cache_base=None,
            ),
        ),
    )
    target = object.__new__(M.DecodeTarget)
    target.core = core
    target._gdn_epilogue = lambda out, z, L, num_tokens: z.clone()
    verify = (int_idx, None, None, False, None)
    z = target._gdn_block(x, 0, layer, si, verify, p, captured=captured)
    pools = core._bound.pools
    return dict(seen, z=z, conv=pools.conv[0], win=core._bound.int_conv[0])


@_cuda
@pytest.mark.parametrize("n,p", [(1, 4), (2, 4), (8, 2), (2, 8)])
def test_fused_verify_conv_matches_the_two_launches(gdn_layer, n, p, monkeypatch) -> None:
    if gdn_layer.in_proj_nv is not None and n * p > M._INPROJ_NVFP4_MAX_ROWS:
        pytest.skip("past the NVFP4 copy's rows")
    g = torch.Generator(device=DEV).manual_seed(10 * n + p)
    conv0 = torch.randn(SLOTS, CONV, 3, generator=g, device=DEV).to(BF16)
    win0 = torch.randn(ROWS, p, CONV, 3, generator=g, device=DEV).to(BF16)
    x = torch.randn(n * p, HIDDEN, generator=g, device=DEV).to(BF16)
    si = torch.randperm(SLOTS, generator=g, device=DEV)[:n].to(torch.int32)
    int_idx = torch.arange(ROWS, dtype=torch.int32, device=DEV)[:n]
    eager = _gdn_block(gdn_layer, x, si, int_idx, conv0, win0, p, False, monkeypatch)
    fused = _gdn_block(gdn_layer, x, si, int_idx, conv0, win0, p, True, monkeypatch)
    for key in ("mq_out", "a", "b", "z", "conv", "win"):
        assert torch.equal(eager[key].view(torch.int16), fused[key].view(torch.int16)), key
    assert fused["pdl"] == eager["pdl"] == M._GDN_REPLAY_PDL
    assert not torch.equal(fused["conv"], conv0) and not torch.equal(fused["win"], win0)
