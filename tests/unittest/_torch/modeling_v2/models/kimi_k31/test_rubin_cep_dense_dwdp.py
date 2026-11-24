# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense DWDP on the cep target: its flat weight table sharded, gathered and read back bit-exact.

The cep core keeps every layer's weights in one ``ParameterDict`` (``core.w``) and reads them
through the views ``_layer_views`` builds, some of them derived (``k_b``, ``v_b_t``, ``qkv_a_t``,
``idx_wq_b_t``). Under dense DWDP the engine's setup (``modules/dwdp/dense``) moves the dense
weights each layer names (``dense_dwdp_layers``) into a row store and staging slots, repoints the
parameters and asks the core to rebuild its views (``dense_dwdp_rebind``). The test runs the real
setup on a toy core of the cep shape (on the host, one rank of an ``N``-rank group whose peers
report the same: the loopback store holds every rank's rows) and then two forwards' hook
sequence of ``PrefillTarget`` (``_DenseHooks``): inside each layer's read windows every view the
forward reads is bit-equal to the whole weight it replaced, the moved parameters are exactly the
target's dense names and the originals are gone.

Importing the target needs the CuTe DSL; the copies and reads run on the host.
"""

from __future__ import annotations

__extra_import_path__ = ["~/tests/unittest/_torch/modules/dwdp"]

import ctypes
import gc
import weakref
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from test_dense_protocol import FakeOps
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__cep import (  # noqa: E501
    modeling as cep,
)
from tensorrt_llm._torch.modules.dwdp.dense import setup as dense_setup
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
from tensorrt_llm._torch.modules.dwdp.dense.setup import setup_dense_dwdp
from tensorrt_llm._torch.modules.dwdp.dense.store import LoopbackShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode, LayerCopyPlan

#: The toy layer stack: layer 0 dense MLP (KDA mixer), MLA at 3 and 6, KDA elsewhere; layers
#: 1..7 are MoE layers, which dense DWDP covers.
NUM_LAYERS = 8
KDA_LAYERS = (0, 1, 2, 4, 5, 7)
MLA_LAYERS = (3, 6)
MOE_LAYERS = tuple(range(1, NUM_LAYERS))
#: Toy widths (the views read heads, nope, v and kv_lora).
H, HEADS, NOPE, V, KV_LORA, Q_LORA, ROPE = 64, 4, 8, 8, 16, 32, 4
#: The test moves every parameter of the dense names, however small.
MIN_BYTES = 1
#: The parts of a layer view (``_LayerW``) that hold tensors.
_PARTS = ("kda", "mla", "moe", "dense", "pre", "mid")


def _toy_dims() -> SimpleNamespace:
    """What ``_layer_views`` and ``_dense_roots`` read off ``core.dims``."""
    return SimpleNamespace(
        num_layers=NUM_LAYERS,
        kda_layers=KDA_LAYERS,
        mla_layers=MLA_LAYERS,
        moe_layers=MOE_LAYERS,
        dense_layers=(0,),
        mla_heads=HEADS,
        nope=NOPE,
        kv_lora=KV_LORA,
        v_dim=V,
        attn_res_block=3,
        expert_offset=0,
        cp=4,
    )


def _shapes() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every ``core.w`` key the views read, at toy shapes, in the cep dtypes."""
    bf16, fp32, fp8, u8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.uint8
    per_layer = {
        "attn_norm": ((H,), bf16),
        "attn_res_proj": ((1, H), bf16),
        "attn_res_norm": ((H,), bf16),
        "mlp_norm": ((H,), bf16),
        "mlp_res_proj": ((1, H), bf16),
        "mlp_res_norm": ((H,), bf16),
    }
    kda = {
        "kda_qkvgfab_w": ((80, H), fp8),
        "kda_qkvgfab_s": ((80 * H // 32,), u8),
        "kda_fb": ((32, 16), bf16),
        "kda_o_w": ((H, 32), fp8),
        "kda_o_s": ((H,), u8),
        "kda_conv": ((24, 4), bf16),
        "kda_a_log": ((2,), fp32),
        "kda_dt_bias": ((8,), fp32),
        "kda_o_norm": ((16,), bf16),
    }
    mla = {
        "mla_qkv_a": ((Q_LORA + KV_LORA + ROPE, H), bf16),
        "mla_q_a_norm": ((Q_LORA,), bf16),
        "mla_kv_a_norm": ((KV_LORA,), bf16),
        "mla_q_b": ((HEADS * (NOPE + ROPE), Q_LORA), bf16),
        "mla_kv_b": ((HEADS * (NOPE + V), KV_LORA), bf16),
        "mla_g": ((HEADS * V, H), bf16),
        "mla_o": ((H, HEADS * V), bf16),
        "mla_idx_wq_b": ((32, Q_LORA), bf16),
        "mla_idx_wk_wp": ((12, H), fp32),
        "mla_idx_k_norm_w": ((8,), fp32),
        "mla_idx_k_norm_b": ((8,), fp32),
    }
    moe = {
        "moe_router": ((16, H), bf16),
        "moe_router_bias": ((16,), fp32),
        "moe_shared_gu_w": ((64, H), fp8),
        "moe_shared_gu_s": ((64 * H // 32,), u8),
        "moe_shared_dn_w": ((H, 32), fp8),
        "moe_shared_dn_s": ((H,), u8),
        "moe_latent_down": ((32, H), bf16),
        "moe_latent_norm": ((32,), bf16),
        "moe_latent_up": ((H, 32), bf16),
        "moe_fc1_w": ((4, 32, 16), u8),
        "moe_fc1_s": ((4, 32, 2), u8),
        "moe_fc2_w": ((4, 32, 8), u8),
        "moe_fc2_s": ((4, 32, 1), u8),
        "moe_fc31_in_scale": ((1,), fp32),
        "moe_fc2_in_scale": ((1,), fp32),
        "moe_fc31_alpha": ((4,), fp32),
        "moe_fc2_alpha": ((4,), fp32),
        "moe_fc1_norm_const": ((4,), fp32),
    }
    out = {}
    for i in range(NUM_LAYERS):
        table = dict(per_layer)
        table.update(kda if i in KDA_LAYERS else mla)
        if i in MOE_LAYERS:
            table.update(moe)
        else:
            table.update({"dense_gu": ((64, H), bf16), "dense_dn": ((H, 32), bf16)})
        out.update({f"l{i}_{name}": shape for name, shape in table.items()})
    return out


class ToyCepCore(nn.Module):
    """A cep core reduced to what dense DWDP touches: ``w``, ``dims``, the views and targets."""

    dense_dwdp_layers = cep.KimiK31ProNvfp4Sm107CepCore.dense_dwdp_layers
    dense_dwdp_rebind = cep.KimiK31ProNvfp4Sm107CepCore.dense_dwdp_rebind

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.dims = _toy_dims()
        self.model_config = SimpleNamespace(mapping=SimpleNamespace(cp_rank=0))
        self.w = nn.ParameterDict()
        gen = torch.Generator().manual_seed(seed)
        for name, (shape, dtype) in _shapes().items():
            raw = torch.randint(
                0, 256, (torch.Size(shape).numel() * dtype.itemsize,), generator=gen
            )
            value = raw.to(torch.uint8).view(dtype).view(shape)
            self.w[name] = nn.Parameter(value.clone(), requires_grad=False)
        self._head = None
        # The one runtime field the prefill target reads at construction: the CP push region.
        self._rt = SimpleNamespace(cp_push=SimpleNamespace(region=0))
        self.eplb = None
        self.kda_split = "ulysses_fp8"
        self._dense_hooks = cep._NoDenseHooks()
        self._layers = cep._layer_views(self)
        self._prefill = cep.PrefillTarget(self)
        self._decode = cep.DecodeTarget(self)


class GroupOfIdenticalRanks:
    """``GroupComm`` of rank ``rank`` in a group of ``size`` whose peers report what it reports
    (every rank holds the same weights, and the loopback store holds every rank's rows)."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank, self.size = rank, size

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def allgather(self, obj: object) -> list[object]:
        return [obj] * self.size


class HostLauncher:
    """Runs a layer's copy plan as host memmoves when the runtime enqueues it."""

    def __init__(self, plans: Mapping[int, LayerCopyPlan]) -> None:
        self.plans = dict(plans)

    def __call__(self, layer: int) -> None:
        for dst, src, nbytes in self.plans[layer].descriptors:
            ctypes.memmove(dst, src, nbytes)


def _view_bytes(view: tuple) -> dict[str, torch.Tensor]:
    """Each tensor field of a layer view (a NamedTuple) as a copy of its bytes, by field."""
    return {
        field: value.detach().contiguous().reshape(-1).view(torch.uint8).clone()
        for field, value in view._asdict().items()
        if isinstance(value, torch.Tensor)
    }


def _expected_moved() -> set[str]:
    names = set()
    for i in MOE_LAYERS:
        attn = cep._DENSE_ATTN_KDA if i in KDA_LAYERS else cep._DENSE_ATTN_MLA
        names |= {f"l{i}_{n}" for n in (*attn, *cep._DENSE_MOE)}
    return names


@pytest.fixture
def host_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The setup's device calls on the host: no CUDA cache, the test's ops seam."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.mark.parametrize("rank", [0, 3])
def test_dense_dwdp_reads_back_the_whole_weights(host_setup: None, rank: int) -> None:
    size = 4
    core = ToyCepCore(seed=1234)
    model = nn.Module()
    model.model = core
    reference = {
        (i, part): _view_bytes(getattr(view, part))
        for i, view in enumerate(core._layers)
        for part in _PARTS
        if getattr(view, part) is not None
    }
    originals = {name: weakref.ref(p.data.untyped_storage()) for name, p in core.w.items()}
    pointers = {name: p.data_ptr() for name, p in core.w.items()}

    ops = FakeOps()
    transfer = SimpleNamespace(
        copy_stream=ops.copy, use_gather=False, use_batched_prefetch=False, use_tma=False
    )
    runtime = setup_dense_dwdp(
        model,
        transfer,
        GroupOfIdenticalRanks(rank, size),
        dwdp_rank=rank,
        dwdp_size=size,
        device_id=0,
        moe_layers=None,
        config=DenseConfig(enabled=True, min_bytes=MIN_BYTES),
        store_factory=LoopbackShardStore,
        ops=ops,
        device=torch.device("cpu"),
        launcher_factory=lambda plans, _transfer, _device: HostLauncher(plans),
        mem_get_info=lambda: (64 << 30, 64 << 30),
    )
    try:
        assert runtime is not None and runtime.layers == MOE_LAYERS
        assert core._dense_dwdp_active and model._dense_dwdp_active

        # Exactly the target's dense names moved; their original storages are gone.
        gc.collect()
        moved = {name for name, p in core.w.items() if p.data_ptr() != pointers[name]}
        assert moved == _expected_moved()
        assert [name for name in moved if originals[name]() is not None] == []

        manager = DenseDwdpManager(DenseMode.ATTACHED)
        manager.attach(runtime, expert=SimpleNamespace(start_forward=lambda: None))
        hooks = cep._DenseHooks(manager)
        mismatches = []
        for _ in range(2):  # the second forward primes from the executor's H0 again
            ops.current = ops.compute
            hooks.dense_begin()
            for i, view in enumerate(core._layers):
                hooks.dense_wait(i)
                # Reads of the mixer's weights happen between H1 and H2, of the MLP's between H1
                # and H3 (H2 refills the mixer's slot with layer i + 2).
                mismatches += _mismatches(
                    i, view, "kda" if view.kda is not None else "mla", reference
                )
                hooks.dense_attn_done(i)
                for part in ("moe" if view.moe is not None else "dense", "pre", "mid"):
                    mismatches += _mismatches(i, view, part, reference)
                hooks.dense_moe_done(i)
        assert mismatches == []
        assert runtime.demand_fetches == 0
    finally:
        runtime.release()


def _mismatches(layer: int, view: tuple, part: str, reference: dict) -> list[str]:
    """The fields of layer ``layer``'s ``view.<part>`` whose bytes differ from the reference taken
    before setup."""
    got, want = _view_bytes(getattr(view, part)), reference[(layer, part)]
    assert got.keys() == want.keys(), (layer, part, sorted(got), sorted(want))
    return [f"l{layer}.{part}.{f}" for f in got if not torch.equal(got[f], want[f])]


def test_the_dense_names_are_parameters_the_views_read() -> None:
    """Every name the target moves is a ``core.w`` key its views read (a renamed weight row would
    otherwise drop out of dense DWDP silently), and no routed-expert row is among them."""
    core = ToyCepCore(seed=7)
    assert _expected_moved() <= set(core.w.keys())
    assert not any("_moe_fc" in name or name.endswith("router_bias") for name in _expected_moved())
    roots = core.dense_dwdp_layers()
    assert [bool(r.dense_dwdp_roots("attn")) for r in roots] == [i in MOE_LAYERS for i in range(8)]
    with pytest.raises(ValueError, match="segment must be one of"):
        roots[1].dense_dwdp_roots("experts")
    assert dense_setup.dense_roots_available(nn.ModuleDict({"model": core}))
