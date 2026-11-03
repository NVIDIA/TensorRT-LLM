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
"""``TRTLLM_K3_L2PF_V2``: the Kimi-K3 decode L2 weight prefetch.

* Off (the default), the model builds no controller, wires no hook and its join points do
  nothing.
* On, the model wires the decoder layers, the MoE blocks and the attention blocks of MoE layers,
  and selects each point's weights in read order without those an older prefetch switch covers.
* The table kernel issues every chunk of its ranges exactly once (traced), leaves memory
  untouched, and the controller's side-stream forks capture into and replay from a CUDA graph.
"""

import types
from types import SimpleNamespace

import pytest
import torch

_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)
needs_sm100 = pytest.mark.skipif(not _SM100, reason="needs an SM100+ GPU")

_MIB = 1 << 20


def _mkl():
    pytest.importorskip("fla")
    from tensorrt_llm._torch.models import modeling_kimi_linear

    return modeling_kimi_linear


def _l2pf():
    from tensorrt_llm._torch.modules import kimi_k3_l2_prefetch

    return kimi_k3_l2_prefetch


def _table_mod():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.kernels.blackwell import k3_l2_prefetch_table

    return k3_l2_prefetch_table


# ---------------------------------------------------------------------------
# Switch and tables.
# ---------------------------------------------------------------------------


def test_switch_parsing():
    pf = _l2pf()
    assert pf.L2PrefetchConfig.from_env({}).points == frozenset()
    assert pf.parse_points("0") == frozenset()
    assert pf.parse_points("1") == frozenset(pf.POINTS)
    assert pf.parse_points("next_attention, oproj_down") == {"next_attention", "oproj_down"}
    with pytest.raises(ValueError):
        pf.parse_points("front")
    config = pf.L2PrefetchConfig.from_env(
        {"TRTLLM_K3_L2PF_V2": "1", "TRTLLM_K3_L2PF_V2_NEXT_ATTENTION": "1:96.100,2-3:96"}
    )
    assert config.tables["next_attention"] == {1: (96, 100), 2: (96, None), 3: (96, None)}
    assert config.ctas == pf.DEFAULT_CTAS
    # Default tables: nothing above 64 tokens per step.
    defaults = pf.L2PrefetchConfig.from_env({}).tables
    assert defaults["next_attention"][8] == (64, None)
    assert defaults["next_attention"][16] == defaults["next_attention"][64] == (50, None)
    assert all(65 not in table for table in defaults.values())
    for bad in ("8", "0-8:4", "8-4:4", "8:x", "8:4.101"):
        with pytest.raises(ValueError):
            pf.batch_table(bad)
    assert pf.parse_ctas("1,2,3") == {"attention_moe": 1, "oproj_down": 2, "next_attention": 3}
    with pytest.raises(ValueError):
        pf.parse_ctas("1,2")


def test_capped_ranges():
    tm = _table_mod()
    ranges = [(0x1000, 3 * _MIB), (0x900000, 0), (0x2000000, 2 * _MIB)]
    assert tm.capped_ranges(ranges) == [(0x1000, 3 * _MIB), (0x2000000, 2 * _MIB)]
    assert tm.capped_ranges(ranges, 4 * _MIB) == [(0x1000, 3 * _MIB), (0x2000000, _MIB)]
    assert tm.capped_ranges(ranges, 2 * _MIB + 40) == [(0x1000, 2 * _MIB + 32)]
    assert tm.table_chunks([(0, tm.CHUNK_BYTES), (0, tm.CHUNK_BYTES + 16)]) == 3


# ---------------------------------------------------------------------------
# Switch off: nothing wired, nothing run.
# ---------------------------------------------------------------------------


class _StandInModel(SimpleNamespace):
    """A namespace the controller can hold a weak reference to, like the real model."""


def _stand_in_model(num_layers=4):
    """Decoder-layer stand-ins: layer 0 dense KDA, then MoE layers alternating KDA / MLA."""
    layers = []
    for i in range(num_layers):
        is_kda = i % 2 == 0
        layers.append(
            SimpleNamespace(
                layer_idx=i,
                is_kda=is_kda,
                is_moe=i > 0,
                linear_attn=SimpleNamespace() if is_kda else None,
                self_attn=None if is_kda else SimpleNamespace(mixer=SimpleNamespace()),
                block_sparse_moe=SimpleNamespace() if i > 0 else None,
            )
        )
    return _StandInModel(layers=layers)


def test_switch_off_wires_nothing(monkeypatch):
    mkl = _mkl()
    from tensorrt_llm._torch.modules.kimi_k3_mla import KimiK3MLAAttention
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention

    monkeypatch.delenv("TRTLLM_K3_L2PF_V2", raising=False)
    # A malformed table is not even read while the switch is off.
    monkeypatch.setenv("TRTLLM_K3_L2PF_V2_NEXT_ATTENTION", "garbage")
    model = _stand_in_model()
    assert mkl._k3_l2pf_controller(model) is None
    for layer in model.layers:
        assert not hasattr(layer, "_l2pf")
        if layer.is_moe:
            assert not hasattr(layer.block_sparse_moe, "_l2pf")
    attns = [layer.linear_attn or layer.self_attn.mixer for layer in model.layers]
    assert not any(hasattr(a, "l2pf_attention_hook") for a in attns)
    for cls in (mkl.KimiLinearModel, mkl.KimiLinearDecoderLayer, mkl.KimiK3MoERuntime):
        assert cls._l2pf is None
    assert KimiKDALinearAttention.l2pf_attention_hook is None
    assert KimiK3MLAAttention.l2pf_attention_hook is None


def test_switch_off_join_does_nothing(monkeypatch):
    """The MoE block's shared join point (also the V2 join) touches no stream when both prefetch
    switches are off."""
    mkl = _mkl()
    rt = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
    torch.nn.Module.__init__(rt)

    def no_stream(*args, **kwargs):
        raise AssertionError("no stream may be touched with the prefetch switches off")

    monkeypatch.setattr(torch.cuda, "current_stream", no_stream)
    rt._join_expert_l2_prefetch()


def test_switch_on_wires_moe_layers(monkeypatch):
    mkl = _mkl()
    monkeypatch.setattr(mkl, "_K31_L2PF", True)
    model = _stand_in_model()
    controller = mkl._k3_l2pf_controller(model)
    assert controller is not None and controller.config.points == frozenset(_l2pf().POINTS)
    assert all(layer._l2pf is controller for layer in model.layers)
    dense, kda_moe, mla_moe = model.layers[0], model.layers[2], model.layers[1]
    assert not hasattr(dense.linear_attn, "l2pf_attention_hook")
    for layer, attn in ((kda_moe, kda_moe.linear_attn), (mla_moe, mla_moe.self_attn.mixer)):
        assert layer.block_sparse_moe._l2pf is controller
        hook = attn.l2pf_attention_hook
        assert hook.func == controller.fork and hook.args == ("attention_moe", layer.layer_idx)


# ---------------------------------------------------------------------------
# Which weights each point prefetches.
# ---------------------------------------------------------------------------


def _linear(rows, cols):
    """An FP8 projection stand-in: weight and one block-scale row, no low-M scale."""
    weight = torch.empty(rows, cols, dtype=torch.uint8)
    scale = torch.empty(1, cols // 32, dtype=torch.uint8)
    return SimpleNamespace(weight=weight, weight_scale=scale, _low_m_weight_scale=None)


def _weights_model():
    """Layers 0 (KDA, MoE), 1 (MLA, MoE), 2 (KDA, MoE) with small CPU weights; the sharded MoE
    front serves none of their forwards (no ``_k31f_weight``)."""
    runtime = _mkl().KimiK3MoERuntime
    moe_group = {}
    latent_stream = object()
    layers = {}
    for i, is_kda in enumerate((True, False, True)):
        moe = SimpleNamespace(
            layer_idx=i,
            gate=SimpleNamespace(weight=torch.empty(64, 64)),
            routed_expert_down_proj=_linear(32, 64),
            shared_experts=SimpleNamespace(gate_up_proj=_linear(16, 64), down_proj=_linear(64, 64)),
            latent_down_stream=latent_stream,
            _moe_group=moe_group,
            _k31f_weight=None,
        )
        # The point weights ask the MoE block whether its sharded front serves the rows.
        for name in ("k31_sharded_front_weight", "k31_sharded_front_reads_shared_gate_up"):
            setattr(moe, name, types.MethodType(getattr(runtime, name), moe))
        moe_group[i] = moe
        if is_kda:
            attn = SimpleNamespace(
                layer_idx=i,
                _qkvg_proj_weight=None,
                qkvg_proj=_linear(128, 64),
                _bfa_proj_weight=torch.empty(8, 64),
                f_b_proj=SimpleNamespace(weight=torch.empty(64, 8)),
                o_proj=_linear(64, 64),
            )
            layers[i] = SimpleNamespace(
                layer_idx=i, is_kda=True, is_moe=True, linear_attn=attn, block_sparse_moe=moe
            )
        else:
            mixer = SimpleNamespace(
                fuse_qkv_a_proj=True,
                kv_a_proj_with_mqa=_linear(96, 64),
                q_b_proj=_linear(192, 32),
                k_b_proj_trans=torch.empty(4, 16, 8, dtype=torch.uint8),
                k_b_proj_trans_scale=torch.empty(4, 1, 1),
                k_b_proj_trans_dequant=None,
                helix_v_b_proj_full=torch.empty(4, 8, 16, dtype=torch.bfloat16),
                g_proj=_linear(32, 64),
                o_proj=_linear(64, 32),
            )
            layers[i] = SimpleNamespace(
                layer_idx=i,
                is_kda=False,
                is_moe=True,
                self_attn=SimpleNamespace(mixer=mixer),
                block_sparse_moe=moe,
            )
    for i, moe in moe_group.items():
        moe._next_moe_layer = (lambda j: (lambda: moe_group.get(j)))(i + 1)
    return SimpleNamespace(layers=[layers[i] for i in sorted(layers)])


def _ids(tensors):
    return [id(t) for t in tensors]


def test_point_weights_in_read_order(monkeypatch):
    mkl = _mkl()
    monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH", frozenset())
    monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH_NEXT_MLA", False)
    model = _weights_model()

    def tensors_for(point, layer_idx, m=8):
        return mkl._k3_l2pf_tensors(lambda: model, point, layer_idx, m)

    def lin(module):
        return [module.weight_scale, module.weight]

    kda0, mla1, kda2 = model.layers
    moe1 = mla1.block_sparse_moe
    front = [moe1.gate.weight] + lin(moe1.routed_expert_down_proj)
    shared = lin(moe1.shared_experts.gate_up_proj) + lin(moe1.shared_experts.down_proj)
    assert _ids(tensors_for("oproj_down", 1)) == _ids(front)
    assert _ids(tensors_for("attention_moe", 1)) == _ids(front + shared)
    # Where the sharded MoE front serves the rows it reads its own weight (this rank's head rows,
    # plus the shared gate_up rows where it computes them) instead of the router and latent_down.
    head = torch.empty(128, 64, dtype=torch.bfloat16)
    for reads_shared_gate_up in (False, True):
        moe1.k31_sharded_front_weight = lambda m: head
        moe1.k31_sharded_front_reads_shared_gate_up = lambda m, r=reads_shared_gate_up: r
        assert _ids(tensors_for("oproj_down", 1)) == _ids([head])
        rest = lin(moe1.shared_experts.down_proj) if reads_shared_gate_up else shared
        assert _ids(tensors_for("attention_moe", 1)) == _ids([head] + rest)
    mixer = mla1.self_attn.mixer
    mla = lin(mixer.kv_a_proj_with_mqa) + lin(mixer.q_b_proj)
    mla += [mixer.k_b_proj_trans_scale, mixer.k_b_proj_trans, mixer.helix_v_b_proj_full]
    mla += lin(mixer.g_proj) + lin(mixer.o_proj)
    assert _ids(tensors_for("next_attention", 0)) == _ids(mla)
    attn = kda2.linear_attn
    kda = lin(attn.qkvg_proj) + [attn._bfa_proj_weight, attn.f_b_proj.weight] + lin(attn.o_proj)
    assert _ids(tensors_for("next_attention", 1)) == _ids(kda)
    # No next layer.
    assert tensors_for("next_attention", 2) == []


# ---------------------------------------------------------------------------
# Kernel and controller on the GPU.
# ---------------------------------------------------------------------------


def _expected_chunks(ranges, chunk):
    out = []
    for address, nbytes in ranges:
        for offset in range(0, nbytes, chunk):
            out.append((address + offset, min(chunk, nbytes - offset)))
    return out


def _buffers(sizes, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randint(-1000, 1000, (n // 4,), generator=gen, device="cuda", dtype=torch.int32)
        for n in sizes
    ]


@needs_sm100
@pytest.mark.parametrize("ctas", [1, 3, 32])
def test_table_kernel_issues_every_chunk_once(ctas):
    tm = _table_mod()
    chunk = tm.CHUNK_BYTES
    # One 16-byte range, partial last chunks, many small ranges, and a multi-MiB range.
    sizes = [16, 3 * chunk + 48] + [16 * (1 + 97 * i % 2600) for i in range(40)] + [(5 << 20) + 16]
    bufs = _buffers(sizes)
    refs = [b.clone() for b in bufs]
    ranges = [(b.data_ptr(), n) for b, n in zip(bufs, sizes)]
    table = tm.L2PrefetchTable(ranges, torch.device("cuda"), ctas=ctas, record=True)
    expected = _expected_chunks(ranges, chunk)
    assert table.chunks == len(expected) and table.nbytes == sum(sizes)
    trace = torch.full((table.chunks, 2), -1, dtype=torch.int64, device="cuda")
    table.launch(trace)
    torch.cuda.synchronize()
    assert [tuple(row) for row in trace.tolist()] == expected
    for b, r in zip(bufs, refs):
        assert torch.equal(b, r)
    with pytest.raises(ValueError):
        table.launch()  # a record table needs its trace
    with pytest.raises(ValueError):
        tm.L2PrefetchTable([(bufs[0].data_ptr() + 4, 16)], torch.device("cuda"))
    with pytest.raises(ValueError):
        tm.L2PrefetchTable([(bufs[0].data_ptr(), 24)], torch.device("cuda"))


def _controller(tables, tensors_for, points=None):
    pf = _l2pf()
    config = pf.L2PrefetchConfig(
        points=frozenset(points or pf.POINTS),
        tables={p: pf.batch_table(tables.get(p, "")) for p in pf.POINTS},
        ctas=dict(pf.DEFAULT_CTAS),
    )
    return pf.L2PrefetchController(config, tensors_for)


@needs_sm100
def test_controller_caps_shares_and_gates():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    bufs = _buffers([_MIB, 2 * _MIB, 3 * _MIB])
    calls = []

    def tensors_for(point, layer_idx, m):
        calls.append((point, layer_idx, m))
        return bufs if point != "oproj_down" else bufs[:1]

    controller = _controller({"next_attention": "1-8:4,9-16:0", "oproj_down": "1-8:1"}, tensors_for)
    # Not armed outside multi-stream steps, nor for steps with prompts or no table entry.
    controller.begin_step(8, True)
    assert not controller.active
    with with_multi_stream(True):
        controller.begin_step(8, False)
        assert not controller.active
        controller.begin_step(100, True)
        assert not controller.active
        controller.begin_step(8, True)
        assert controller.active
        ranges = controller.table_ranges("next_attention", 3, 8)
        assert [n for _, n in ranges] == [_MIB, 2 * _MIB, _MIB]
        assert controller.table_ranges("next_attention", 3, 16) == ()
        # Equal ranges share one table; each (point, layer, tokens) asks for its tensors once.
        assert controller._table("next_attention", 4, 8) is controller._table(
            "next_attention", 3, 8
        )
        controller.table_ranges("next_attention", 3, 8)
        assert calls.count(("next_attention", 3, 8)) == 1
        assert controller.fork("next_attention", 3, 8)
        assert not controller.fork("next_attention", 3, 16)
        assert not controller.fork("attention_moe", 3, 8)  # no table entry for that point
        assert controller.fork("oproj_down", 3, 8)
        assert controller.pending
        # Front / shared points: a stream of higher priority, the max-shared carveout.
        assert controller._streams[True].priority < controller._streams[False].priority
        assert controller._table("oproj_down", 3, 8).carveout == _l2pf().URGENT_CARVEOUT
        assert controller._table("next_attention", 3, 8).carveout is None
        controller.join()
        assert not controller.pending
    torch.cuda.synchronize()


@needs_sm100
def test_controller_forks_capture_and_replay():
    """Forks on the two prefetch streams next to main-stream work, joined at the end: eager (the
    warmup builds the tables), then captured into a CUDA graph and replayed."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    bufs = _buffers([16, 4096 + 48, 3 << 20, (5 << 20) + 16])
    refs = [b.clone() for b in bufs]
    controller = _controller(
        {p: "1-64:4" for p in _l2pf().POINTS}, lambda point, layer_idx, m: bufs
    )
    x = torch.randn(1 << 16, device="cuda")
    y = torch.empty_like(x)

    def step():
        controller.begin_step(16, True)
        torch.mul(x, 2.0, out=y)
        for layer in range(3):
            assert controller.fork("attention_moe", layer, 16)
            y.add_(1.0)
            assert controller.fork("oproj_down", layer, 16)
            y.mul_(0.5)
            assert controller.fork("next_attention", layer, 16)
        controller.join()

    def reference():
        r = x * 2.0
        for _ in range(3):
            r = (r + 1.0) * 0.5
        return r

    main = torch.cuda.Stream()
    with with_multi_stream(True):
        with torch.cuda.stream(main):
            step()
        torch.cuda.synchronize()
        assert torch.equal(y, reference())
        y.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main):
            step()
        assert not controller.pending
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, reference())
    for b, r in zip(bufs, refs):
        assert torch.equal(b, r)
