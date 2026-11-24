# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's sparse MLA decode at 128 heads (``mla_gen_pad128``) on one sm_107 GPU.

One dep16 DSA decode layer, every kernel real (the rig of ``test_rubin_dep16_dsa_fork_sm107.py``),
its eager step run with the lever off (96 heads) and on on the same inputs. From 32 rows on the
lever runs 128 heads: it leaves the latent pool, the Top-K rows and the real heads' FP8 query as
they are, and zeroes the padding heads' FP8 query. The attention output comes from another
TRTLLM-Gen kernel (at 32 rows both head counts split the keys over CTAs, in different splits), so
each output is checked against an FP32 attention of the same FP8 query over the same selected
latent rows: the 128-head output's error is no larger than the 96-head output's, within a small
margin, and the o_proj partials differ no more than a linear map of the two attention outputs
would make them. Below 32 rows the lever's step is today's, bit for bit. On CPU the test
collects and skips.
"""

import math

import pytest
import torch

__extra_import_path__ = ["."]
import test_rubin_dep16_dsa_fork_sm107 as S  # noqa: E402

HEADS, PAD, KV_LORA = 96, 128, 512
SOFTMAX_SCALE = 1.0 / math.sqrt(S.NOPE + S.ROPE)


def _reference(q8: torch.Tensor, indices: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """FP32 attention of each row's 96 real heads over its selected latent rows: ``[rows, 96, 512]``.

    ``q8``: the FP8 query the kernels read (``[rows, heads, latent]`` bytes); ``indices``: the
    attention op's global latent rows (``-1``: none); ``pool``: the FP8 latent pool after the step.
    """
    rows = pool.view(-1, S.LATENT)
    out = []
    for r in range(q8.shape[0]):
        sel = indices[r][indices[r] >= 0].long()
        k = rows[sel].float()
        q = q8[r, :HEADS].view(torch.float8_e4m3fn).float()
        p = torch.softmax((q @ k.t()) * SOFTMAX_SCALE, dim=-1)
        out.append(p @ k[:, :KV_LORA])
    return torch.stack(out)


def _steps(mod, rig, seen: dict) -> dict:
    """The eager step with the lever off, then on, on the same inputs."""
    inputs = rig.inputs(1)
    got = {}
    for pad in (False, True):
        rig.rt.levers = mod._levers("generation")._replace(mla_gen_pad128=pad)
        for _ in range(2):  # kernel builds and workspaces
            rig.load(inputs)
            rig.run(False)
        seen.clear()
        rig.load(inputs)
        out = rig.run(False)
        torch.cuda.synchronize()
        got[pad] = dict(
            out=out.clone(),
            pool=rig.pool.view(torch.uint8).clone(),
            topk=rig.topk_out.clone(),
            attention=dict(seen),
        )
    return got


@S._ON_SM107
@pytest.mark.parametrize("rows", [96, 32])
def test_the_padded_decode_is_as_accurate_as_the_96_head_decode(rows, monkeypatch):
    mod = S.module()
    rig = S.Rig(mod, rows, S.kv_lens(rows, S._KV_LOW, S._KV_SPAN), S._MAX_SEQ)
    seen: dict = {}
    attention = mod.thop_attention

    def spy(**kw):
        out = attention(**kw)
        seen[kw["num_heads"]] = (
            kw["quant_q_buffer"].clone(),
            out.clone(),
            kw["sparse_attn_indices"].clone(),
        )
        return out

    monkeypatch.setattr(mod, "thop_attention", spy)
    got = _steps(mod, rig, seen)
    q96, o96, i96 = got[False]["attention"][HEADS]
    q128, o128, i128 = got[True]["attention"][PAD]
    assert q96.shape[1] == HEADS and q128.shape[1] == PAD
    assert torch.equal(q128[:, :HEADS], q96), "the real heads' FP8 query changed"
    assert not q128[:, HEADS:].any(), "the padding heads' FP8 query is not zero"
    assert torch.equal(got[False]["pool"], got[True]["pool"]), "the latent pool changed"
    assert torch.equal(i96, i128), "the attention's latent rows changed"
    assert torch.equal(
        torch.sort(got[False]["topk"], 1).values, torch.sort(got[True]["topk"], 1).values
    ), "the Top-K rows changed"
    a = o96.view(-1, HEADS, KV_LORA).float()
    b = o128.view(-1, PAD, KV_LORA)[:, :HEADS].float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    ref = _reference(q96, i96, got[False]["pool"].view(torch.float8_e4m3fn))
    err96, err128 = (a - ref).abs(), (b - ref).abs()
    rms96, rms128 = err96.pow(2).mean().sqrt(), err128.pow(2).mean().sqrt()
    assert rms96 <= 2e-2 * ref.abs().max(), f"96-head error rms {rms96:.3e}"
    assert rms128 <= 1.1 * rms96, f"error rms {rms128:.3e} (128 heads) vs {rms96:.3e} (96 heads)"
    assert err128.max() <= 1.5 * err96.max(), (
        f"error max {err128.max():.3e} (128 heads) vs {err96.max():.3e} (96 heads)"
    )
    # The o_proj partial is a linear map of the attention output (W_UV, the gate, o_proj): the
    # two partials differ relatively no more than the attention outputs do, up to a factor 3.
    x, y = got[False]["out"].float(), got[True]["out"].float()
    d_attn = ((a - b).norm() / a.norm()).item()
    d_out = ((x - y).norm() / x.norm()).item()
    assert torch.isfinite(y).all() and d_out <= max(1e-2, 3 * d_attn), (d_out, d_attn)


@S._ON_SM107
@pytest.mark.parametrize("rows", [8, 31])
def test_fewer_rows_keep_todays_decode(rows, monkeypatch):
    """Below mla_gen_pad128_min_rows (32) the lever's step runs the 96-head attention and returns
    today's bits."""
    mod = S.module()
    rig = S.Rig(mod, rows, S.kv_lens(rows, S._KV_LOW, S._KV_SPAN), S._MAX_SEQ)
    seen: dict = {}
    attention = mod.thop_attention

    def spy(**kw):
        seen[kw["num_heads"]] = True
        return attention(**kw)

    monkeypatch.setattr(mod, "thop_attention", spy)
    got = _steps(mod, rig, seen)
    assert list(got[False]["attention"]) == list(got[True]["attention"]) == [HEADS]
    assert torch.equal(got[False]["out"], got[True]["out"])
    assert torch.equal(got[False]["pool"], got[True]["pool"])
