# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE TP2 x EP8 routed-expert bank of ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` (CPU only,
synthetic ModelOpt-shaped checkpoints at shrunk dimensions, the 16-rank split unchanged): the split
is the engine Mapping's MoE order; a rank reads its share exactly once (the ledger fires otherwise);
``pread`` reads the lazy slices' bytes; the bank is the window at the half in FC12's layout and
names its window; the FP32 scalars; the packer's refusals.
"""

from __future__ import annotations

import pytest
import torch
from _helix16_wide_cpu import checkpoint, load_weights_module, tiny_experts, ungate_up16, unswizzle

pytestmark = pytest.mark.cpu_only

PREFIX = "language_model."
LAYER = 5


@pytest.fixture(scope="module")
def w():
    return load_weights_module()


def test_the_split_is_the_engine_mappings_moe_order(w):
    """moe_ep_rank = tp_rank % 8, moe_tp_rank = tp_rank // 8 (``Mapping.moe_ep_rank`` /
    ``moe_tp_rank`` at moe_tp 2, moe_ep 8); the 8 windows partition the 896 experts and every
    (window, half) has one rank."""
    assert (w.MOE_TP, w.MOE_EP, w.LOCAL, w.HALF) == (2, 8, 112, 1536)
    pairs = [w.expert_split(r) for r in range(16)]
    assert pairs == [(r % 8, r // 8) for r in range(16)]
    assert len(set(pairs)) == 16
    covered = sorted(e for ep in range(8) for e in w.expert_window(ep))
    assert covered == list(range(896))
    with pytest.raises(ValueError):
        w.expert_split(16)


def _read(w, ckpt, rank):
    reads = w.ExpertReads()
    raw = w.read_routed_experts_ep(ckpt, LAYER, rank, prefix=PREFIX, device="cpu", reads=reads)
    return raw, reads


def test_a_rank_reads_its_share_exactly_once(w):
    with tiny_experts(w):
        ckpt = checkpoint(w, LAYER)
        taken = {}
        for rank in range(16):
            raw, reads = _read(w, ckpt, rank)
            reads.check(PREFIX, LAYER, rank)
            assert set(reads.counts.values()) == {1}
            assert set(reads.counts) == w.expert_keys(PREFIX, LAYER, rank)
            assert set(ckpt) - set(reads.counts) == w.unread_expert_keys(PREFIX, LAYER, rank)
            for key in reads.counts:
                taken[key] = taken.get(key, 0) + 1
        # Over the 16 ranks: every key read; a window key by its two halves' ranks, an
        # input_scale by every rank.
        assert set(taken) == set(ckpt)
        for key, n in taken.items():
            assert n == (16 if key.endswith(".input_scale") else 2), key


def test_the_ledger_fires(w):
    """A key read twice, a key of the share not read, a key outside it read: each one raises,
    naming the key in its own list."""
    with tiny_experts(w):
        ckpt = checkpoint(w, LAYER)
        _, reads = _read(w, ckpt, 3)
        key = sorted(reads.counts)[0]
        outside = sorted(w.unread_expert_keys(PREFIX, LAYER, 3))[0]
        cases = (
            ("read more than once", lambda r: r.take(key), key),
            ("not read", lambda r: r.counts.pop(key), key),
            ("outside this rank's share", lambda r: r.take(outside), outside),
        )
        for what, edit, named in cases:
            _, reads = _read(w, ckpt, 3)
            reads.check(PREFIX, LAYER, 3)
            edit(reads)
            with pytest.raises(ValueError) as error:
                reads.check(PREFIX, LAYER, 3)
            assert f"{what} ['{named}']" in str(error.value), (what, str(error.value))


def test_the_pread_reader_reads_the_slices_bytes(w, tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    with tiny_experts(w):
        ckpt = checkpoint(w, LAYER)
        path = tmp_path / "experts.safetensors"
        safetensors_torch.save_file({k: v.contiguous() for k, v in ckpt.items()}, str(path))
        for rank in (0, 7, 8, 15):
            lazy, _ = _read(w, ckpt, rank)
            reads = w.ExpertReads()
            fast = w.pread_routed_experts_ep(
                str(path), LAYER, rank, prefix=PREFIX, device="cpu", threads=4, reads=reads
            )
            reads.check(PREFIX, LAYER, rank)
            assert fast.keys() == lazy.keys()
            for name, value in lazy.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(fast[name], value), (rank, name)
                else:
                    assert fast[name] == value, (rank, name)


def test_the_bank_is_the_window_at_the_half(w):
    """Un-interleaving and un-swizzling a rank's bank gives back the checkpoint's rows / columns
    of its window at its half; the two halves of every expert give back the whole expert."""
    with tiny_experts(w) as dims:
        ckpt = checkpoint(w, LAYER)
        half_rows, sf = dims["HALF"], w.SF_VEC
        whole = {}
        for rank in range(16):
            ep, half = w.expert_split(rank)
            raw, _ = _read(w, ckpt, rank)
            bank = w.pack_routed_experts_fc12(**raw)
            # The bank names its window (a ``range``, not a tensor: the weight walks pass it by).
            assert bank["window"] == w.expert_window(rank)
            assert sorted(k for k, v in bank.items() if not isinstance(v, torch.Tensor)) == [
                "window"
            ]
            assert bank["fc1_w"].shape == (dims["LOCAL"], 2 * half_rows, dims["LATENT"] // 2)
            assert bank["fc2_w"].shape == (dims["LOCAL"], dims["LATENT"], half_rows // 2)
            for local, e in enumerate(w.expert_window(rank)):
                key = lambda proj, kind: w.expert_key(PREFIX, LAYER, e, proj, kind)  # noqa: E731
                rows = slice(half * half_rows, (half + 1) * half_rows)
                gate, up = ungate_up16(bank["fc1_w"][local])
                assert torch.equal(gate, ckpt[key("w1", "weight")][rows])
                assert torch.equal(up, ckpt[key("w3", "weight")][rows])
                gate_s, up_s = ungate_up16(unswizzle(bank["fc1_s"][local]))
                assert torch.equal(gate_s, ckpt[key("w1", "weight_scale")][rows].view(torch.uint8))
                assert torch.equal(up_s, ckpt[key("w3", "weight_scale")][rows].view(torch.uint8))
                cols = slice(half * half_rows // 2, (half + 1) * half_rows // 2)
                scols = slice(half * half_rows // sf, (half + 1) * half_rows // sf)
                assert torch.equal(bank["fc2_w"][local], ckpt[key("w2", "weight")][:, cols])
                assert torch.equal(
                    unswizzle(bank["fc2_s"][local]),
                    ckpt[key("w2", "weight_scale")][:, scols].view(torch.uint8),
                )
                whole.setdefault(e, {})[half] = (gate, up, bank["fc2_w"][local])
        for e, halves in whole.items():
            assert sorted(halves) == [0, 1]
            key = lambda proj: w.expert_key(PREFIX, LAYER, e, proj, "weight")  # noqa: E731
            assert torch.equal(torch.cat([halves[0][0], halves[1][0]]), ckpt[key("w1")])
            assert torch.equal(torch.cat([halves[0][1], halves[1][1]]), ckpt[key("w3")])
            assert torch.equal(torch.cat([halves[0][2], halves[1][2]], dim=1), ckpt[key("w2")])


def test_the_scales_and_alphas_are_the_ull_targets(w):
    """The FP32 scalars FC12 runs on are the ULL target's packer's (``pack_routed_experts_tp16``):
    ``fc31_in_scale`` / ``fc2_in_scale`` = 1 / input_scale (its ``x_gsf`` / ``h_gsf``), the alphas
    input_scale x weight_scale_2 (its alpha table), here 1.0 and weight_scale_2 exactly."""
    with tiny_experts(w):
        ckpt = checkpoint(w, LAYER)
        raw, _ = _read(w, ckpt, 9)
        bank = w.pack_routed_experts_fc12(**raw)
        window = w.expert_window(9)
        ws2 = lambda proj: torch.stack(  # noqa: E731
            [ckpt[w.expert_key(PREFIX, LAYER, e, proj, "weight_scale_2")] for e in window]
        )
        one = torch.ones(1, dtype=torch.float32)
        assert torch.equal(bank["fc31_in_scale"], one) and torch.equal(bank["fc2_in_scale"], one)
        # The ULL target's ``alphas[:E, 0] = in1 * s13`` and ``alphas[:E, 1] = in2 * s2``.
        assert torch.equal(bank["fc31_alpha"], ws2("w1"))
        assert torch.equal(bank["fc2_alpha"], ws2("w2"))
        assert all(bank[k].dtype == torch.float32 for k in ("fc31_alpha", "fc2_alpha"))


def test_the_packer_refuses_what_it_cannot_represent(w):
    with tiny_experts(w):
        ckpt = checkpoint(w, LAYER)
        raw, _ = _read(w, ckpt, 0)
        bad = dict(raw, w3_scale2=raw["w3_scale2"] * 2)
        with pytest.raises(ValueError, match="gate/up scales differ"):
            w.pack_routed_experts_fc12(**bad)
        uneven = raw["fc2_input_scale"].clone()
        uneven[3] = 0.5
        with pytest.raises(ValueError, match="unit activation scales"):
            w.pack_routed_experts_fc12(**dict(raw, fc2_input_scale=uneven))
        # A uniform scale other than 1.0 too: FC12's quantization rounds as the ULL target's
        # routing kernel does only at the unit scale.
        with pytest.raises(ValueError, match="unit activation scales"):
            w.pack_routed_experts_fc12(**dict(raw, fc1_input_scale=raw["fc1_input_scale"] * 2))
        with pytest.raises(TypeError):
            w.pack_routed_experts_fc12(**dict(raw, w2_scale2=raw["w2_scale2"].double()))
        ckpt[w.expert_key(PREFIX, LAYER, 2, "w3", "input_scale")] = torch.tensor(0.5)
        with pytest.raises(ValueError, match="activation scales differ"):
            _read(w, ckpt, 0)
