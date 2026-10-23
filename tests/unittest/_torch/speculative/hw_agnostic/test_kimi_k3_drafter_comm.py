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
"""The Kimi K3 target ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``'s DSpark drafter on the TP group's collective state
(host-side: CPU tensors and fakes, the collective replaced by a recorder).

* ``fc_columns`` / ``K3FcSlice``: the context projection's ``fc`` split by input feature over 1, 2, 4 and 16 ranks.
  Each rank's block is its columns of the full weight, copied; the blocks tile the columns in rank order; the ranks'
  partial products sum to the full product. Uneven splits are refused.
* ``K3DSparkDrafter._k3_split_fc`` splits ``fc`` only on the collective state, for a bias-free bf16 weight whose
  columns split evenly, with the layers' TP all-reduce. ``project_target_hidden`` hands this rank's partial product,
  a zero residual and ``hidden_norm`` to the fused all-reduce up to a decode step's rows (the ranks' partials sum to
  the full product), and above them sums the partials with the drafter's TP all-reduce, then applies
  ``hidden_norm``; either way the result is the stock projection's.
* ``_gate_drafter_comm`` hands the state to a ``K3DSparkDrafter`` only, and only where the target built it.
* ``K3DecodeComm.takes_allreduce_norm`` / ``allreduce_norm``: which calls the TP16 MNNVL workspace holds, and the
  catalog call's arguments.
* The fused norms: ``_k3_norms_take_comm`` holds for plain bf16 RMSNorms of the hidden width and row-parallel output
  and down projections with their all-reduce, and for nothing else; ``use_decode_comm`` compiles each
  ``comm/k3_sandwich_plain`` form whose shape every layer shares (one zero-row call each, on the group's sandwich
  workspace), and none without the fused norms; ``K3DecodeComm.compile_plain`` / ``takes_plain`` /
  ``sandwich_plain`` hand the catalog entry its arguments; ``skip_all_reduce`` turns a module's all-reduce off.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    decode_comm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as target,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.cpu_only

# A small context projection: 4 captured layers of 32 features into 32 (K3's is 5 x 7168 into 7168), and the input
# widths of the layers' output and down projections.
IN, OUT = 128, 32
K_O, K_DOWN = 16, 24
DECODE_ROWS = target.MAX_REQUESTS * target.MAX_TOKENS_PER_REQUEST


def _fc(dtype=torch.bfloat16, seed=3, bias=False):
    g = torch.Generator().manual_seed(seed)
    fc = nn.Linear(IN, OUT, bias=bias, dtype=dtype)
    with torch.no_grad():
        fc.weight.copy_(torch.randn(OUT, IN, generator=g) * 0.1)
    return fc


def _hidden_norm(eps=1e-6):
    norm = nn.RMSNorm(OUT, eps=eps, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, OUT))
    return norm


class _Sum:
    """A TP all-reduce over the fake group: records its inputs, returns ``total`` (or the input)."""

    def __init__(self, total=None):
        self.inputs = []
        self.total = total

    def __call__(self, x):
        self.inputs.append(x)
        return x if self.total is None else self.total


def _comm(world=1, buffer_bytes=4 << 20):
    return decode_comm.K3DecodeComm(
        mnnvl=SimpleNamespace(world_size=world, buffer_bytes=buffer_bytes),
        sandwich=SimpleNamespace(world_size=world),
    )


def _norm(**kwargs):
    """A stock RMSNorm of the hidden width."""
    kwargs.setdefault("dtype", torch.bfloat16)
    return RMSNorm(hidden_size=kwargs.pop("hidden_size", OUT), eps=1e-6, **kwargs)


def _projection(k, all_reduce=True, row_parallel=True, reduce_output=True):
    """A row-parallel projection's fields: its mode, its all-reduce and its [OUT, k] slice."""
    return SimpleNamespace(
        tp_mode=SimpleNamespace(name="ROW" if row_parallel else "COLUMN"),
        reduce_output=reduce_output,
        all_reduce=_Sum() if all_reduce else None,
        weight=torch.full((OUT, k), 0.5, dtype=torch.bfloat16),
    )


def _layer(all_reduce=True, row_parallel=True):
    return SimpleNamespace(
        input_layernorm=_norm(),
        post_attention_layernorm=_norm(),
        self_attn=SimpleNamespace(o_proj=_projection(K_O, all_reduce, row_parallel)),
        mlp=SimpleNamespace(down_proj=_projection(K_DOWN)),
    )


def _drafter(tp_size=1, tp_rank=0, comm=None, fc=None, all_reduce=True, row_parallel=True):
    """A K3DSparkDrafter without ``__init__`` (that needs a GPU drafter checkpoint): the fields its context projection
    and fused norms read, two layers."""
    drafter = target.K3DSparkDrafter.__new__(target.K3DSparkDrafter)
    nn.Module.__init__(drafter)
    drafter.model_config = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank)
    )
    drafter.config = SimpleNamespace(hidden_size=OUT)
    drafter.model = SimpleNamespace(
        layers=[_layer(all_reduce, row_parallel) for _ in range(2)], norm=_norm()
    )
    drafter.fc = _fc() if fc is None else fc
    drafter.hidden_norm = _hidden_norm()
    drafter.decode_comm = comm
    drafter._k3_zero_rows = None
    drafter._k3_norms_fuse = False
    drafter._k3_sandwich_forms = frozenset()
    return drafter


@pytest.fixture
def fused_calls(monkeypatch):
    """Records ``comm/mnnvl_fusion_allreduce`` calls; each returns the one-rank result (the sum is the input)."""
    calls = []

    def one_rank(input, workspace, one_shot_max_bytes, residual=None, norm_weight=None, eps=None):
        calls.append(
            dict(
                input=input,
                workspace=workspace,
                one_shot_max_bytes=one_shot_max_bytes,
                residual=residual,
                norm_weight=norm_weight,
                eps=eps,
            )
        )
        updated = (input.float() + residual.float()).to(input.dtype)
        normed = torch.nn.functional.rms_norm(updated.float(), (updated.shape[-1],), eps=eps)
        return (normed * norm_weight.float()).to(input.dtype), updated

    monkeypatch.setattr(decode_comm, "mnnvl_fusion_allreduce", one_rank)
    return calls


@pytest.fixture
def sandwich_calls(monkeypatch):
    """Records ``comm/k3_sandwich_plain`` calls (each returns its residual twice) and lets the kernel's shape check
    pass on CPU tensors unless ``unsupported`` holds the call's weight width; device syncs are no-ops."""
    calls = []
    unsupported = set()

    def sandwich(x, weight, residual, norm_weight, eps, workspace, swiglu=False):
        calls.append(
            dict(
                x=x,
                weight=weight,
                residual=residual,
                norm_weight=norm_weight,
                eps=eps,
                workspace=workspace,
                swiglu=swiglu,
            )
        )
        return residual, residual

    def supports(x, weight, residual, norm_weight, swiglu=False):
        return weight.shape[1] not in unsupported

    monkeypatch.setattr(decode_comm, "k3_sandwich_plain", sandwich)
    monkeypatch.setattr(decode_comm._sandwich_op, "supports_plain", supports)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return SimpleNamespace(calls=calls, unsupported=unsupported)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 16])
def test_fc_columns_tile_the_inputs_in_rank_order(tp_size):
    in_features = 5 * 7168
    columns = [target.fc_columns(in_features, tp_size, rank) for rank in range(tp_size)]
    assert columns[0][0] == 0 and columns[-1][1] == in_features
    assert all(end - start == in_features // tp_size for start, end in columns)
    assert all(a[1] == b[0] for a, b in zip(columns, columns[1:]))
    assert target.fc_columns(5 * 7168, 16, 3) == (6720, 8960)


@pytest.mark.parametrize(
    "in_features,tp_size,tp_rank",
    [(130, 4, 0), (128, 0, 0), (128, 4, 4), (128, 4, -1), (0, 4, 0)],
    ids=["uneven", "no ranks", "rank past the group", "negative rank", "no columns"],
)
def test_fc_columns_refuse_what_does_not_split(in_features, tp_size, tp_rank):
    assert target.fc_columns(in_features, tp_size, tp_rank) is None


@pytest.mark.parametrize("tp_size", [1, 2, 4, 16])
def test_each_ranks_slice_is_its_columns_and_the_partials_sum_to_fc(tp_size):
    g = torch.Generator().manual_seed(tp_size)
    weight = torch.randn(OUT, IN, generator=g, dtype=torch.float64)
    x = torch.randn(9, IN, generator=g, dtype=torch.float64)
    slices = [target.K3FcSlice.of(weight, tp_size, rank) for rank in range(tp_size)]
    for rank, block in enumerate(slices):
        start, end = target.fc_columns(IN, tp_size, rank)
        assert (block.start, block.end) == (start, end)
        assert torch.equal(block.weight, weight[:, start:end])
        assert block.weight.is_contiguous() and not block.weight.requires_grad
        # A block of some columns is a copy; one rank's block is the whole weight.
        assert (block.weight.data_ptr() == weight.data_ptr()) == (tp_size == 1)
    # Reassembled: the blocks side by side are the full weight, and the partial products sum to the full product.
    assert torch.equal(torch.cat([block.weight for block in slices], dim=1), weight)
    total = sum(block(x) for block in slices)
    torch.testing.assert_close(total, x @ weight.T, rtol=1e-12, atol=1e-12)
    assert target.K3FcSlice.of(weight[:, :-1], tp_size, 0) is None or tp_size == 1


def test_fc_splits_only_on_the_collective_state():
    drafter = _drafter(tp_size=4, tp_rank=1)
    drafter._k3_split_fc()
    assert isinstance(drafter.fc, nn.Linear) and drafter._k3_zero_rows is None

    full = drafter.fc.weight.detach().clone()
    drafter.decode_comm = _comm(world=4)
    drafter._k3_split_fc()
    assert isinstance(drafter.fc, target.K3FcSlice)
    assert (drafter.fc.start, drafter.fc.end) == (32, 64)
    assert torch.equal(drafter.fc.weight, full[:, 32:64])
    assert drafter._k3_zero_rows.shape == (DECODE_ROWS, OUT)
    assert drafter._k3_zero_rows.dtype == torch.bfloat16 and not drafter._k3_zero_rows.any()
    # Split once: a second call keeps the block.
    block = drafter.fc
    drafter._k3_split_fc()
    assert drafter.fc is block


@pytest.mark.parametrize(
    "case",
    ["bias", "fp32 weight", "uneven columns", "no TP all-reduce", "column-parallel o_proj"],
)
def test_fc_stays_replicated_where_it_does_not_split(case):
    kwargs = dict(tp_size=4, tp_rank=0, comm=_comm(world=4))
    if case == "bias":
        kwargs["fc"] = _fc(bias=True)
    elif case == "fp32 weight":
        kwargs["fc"] = _fc(dtype=torch.float32)
    elif case == "uneven columns":
        kwargs["tp_size"], kwargs["comm"] = 3, _comm(world=3)
    elif case == "no TP all-reduce":
        kwargs["all_reduce"] = False
    else:
        kwargs["row_parallel"] = False
    drafter = _drafter(**kwargs)
    fc = drafter.fc
    drafter._k3_split_fc()
    assert drafter.fc is fc and drafter._k3_zero_rows is None


def test_reload_splits_fc_again(monkeypatch):
    drafter = _drafter(tp_size=2, tp_rank=1, comm=_comm(world=2))
    drafter._k3_split_fc()
    reloaded = _fc(seed=9)

    def stock_load(self, weights, weight_mapper=None, **kwargs):
        self.fc = reloaded

    monkeypatch.setattr(target.GQADSparkForCausalLM, "load_weights", stock_load)
    drafter.load_weights({})
    assert isinstance(drafter.fc, target.K3FcSlice)
    assert torch.equal(drafter.fc.weight, reloaded.weight[:, IN // 2 :])


@pytest.mark.parametrize("rows", [1, 8, DECODE_ROWS])
def test_split_projection_takes_the_fused_all_reduce(fused_calls, rows):
    stock = _drafter()
    drafter = _drafter(comm=_comm())
    drafter.fc = stock.fc
    drafter._k3_split_fc()
    x = torch.randn(rows, IN, generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16)

    out = drafter.project_target_hidden(x)
    ref = stock.project_target_hidden(x)
    assert len(fused_calls) == 1
    call = fused_calls[0]
    torch.testing.assert_close(call["input"], drafter.fc(x), rtol=0, atol=0)
    assert call["workspace"] is drafter.decode_comm.mnnvl
    assert call["one_shot_max_bytes"] == decode_comm.DECODE_AR_ONE_SHOT_MAX_BYTES
    assert call["residual"].shape == (rows, OUT) and not call["residual"].any()
    assert call["norm_weight"] is drafter.hidden_norm.weight and call["eps"] == 1e-6
    assert drafter.model.layers[0].self_attn.o_proj.all_reduce.inputs == []
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_split_projection_above_a_decode_step_uses_the_tp_all_reduce(fused_calls):
    stock = _drafter()
    drafter = _drafter(comm=_comm())
    drafter.fc = stock.fc
    drafter._k3_split_fc()
    rows = DECODE_ROWS + 1
    x = torch.randn(rows, IN, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)

    out = drafter.project_target_hidden(x)
    assert fused_calls == []
    all_reduce = drafter.model.layers[0].self_attn.o_proj.all_reduce
    assert len(all_reduce.inputs) == 1 and all_reduce.inputs[0].shape == (rows, OUT)
    torch.testing.assert_close(out, stock.project_target_hidden(x), rtol=2e-2, atol=2e-2)


def test_split_projection_where_the_workspace_does_not_hold_the_rows(fused_calls):
    drafter = _drafter(comm=_comm(buffer_bytes=16))
    drafter._k3_split_fc()
    out = drafter.project_target_hidden(torch.ones(2, IN, dtype=torch.bfloat16))
    assert fused_calls == [] and out.shape == (2, OUT)
    assert len(drafter.model.layers[0].self_attn.o_proj.all_reduce.inputs) == 1


def test_split_partials_of_a_tp_group_sum_to_the_projection(fused_calls):
    """Four ranks, each with its block: the partial each hands the all-reduce sums, over the ranks, to the full
    product; the all-reduce's result is the stock projection."""
    tp = 4
    stock = _drafter()
    x = torch.randn(8, IN, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    for rank in range(tp):
        drafter = _drafter(tp_size=tp, tp_rank=rank, comm=_comm(world=tp))
        drafter.fc = stock.fc
        drafter._k3_split_fc()
        drafter.project_target_hidden(x)
    assert len(fused_calls) == tp
    total = sum(call["input"].float() for call in fused_calls)
    full = x.float() @ stock.fc.weight.float().T
    assert ((total - full).norm() / full.norm()).item() < 1e-2


def test_unsplit_projection_is_the_stock_one(fused_calls):
    drafter = _drafter()
    x = torch.randn(4, IN, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    out = drafter.project_target_hidden(x)
    assert fused_calls == []
    torch.testing.assert_close(out, drafter.hidden_norm(drafter.fc(x)), rtol=0, atol=0)


def test_gate_hands_the_state_to_the_k3_drafter_only():
    comm = _comm(world=16)
    taken = []
    drafter = _drafter()
    drafter.use_decode_comm = taken.append
    assert target.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4._gate_drafter_comm(
        SimpleNamespace(draft_model=drafter), comm
    )
    assert taken == [comm]
    assert not target.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4._gate_drafter_comm(
        SimpleNamespace(draft_model=drafter), None
    )
    assert taken == [comm]
    for other in (None, SimpleNamespace(use_decode_comm=taken.append)):
        assert not target.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4._gate_drafter_comm(
            SimpleNamespace(draft_model=other), comm
        )
    assert taken == [comm]


def test_use_decode_comm_keeps_the_state_and_splits_fc(sandwich_calls):
    comm = _comm(world=2)
    drafter = _drafter(tp_size=2, tp_rank=0)
    drafter.use_decode_comm(comm)
    assert drafter.decode_comm is comm
    assert isinstance(drafter.fc, target.K3FcSlice)
    assert (drafter.fc.start, drafter.fc.end) == (0, 64)


def test_tp16_workspace_holds_a_decode_steps_rows():
    """The target's 4 MiB MNNVL buffers at TP16: rows of 7168 go one-shot up to 18, two-shot up to 144."""
    comm = _comm(world=16, buffer_bytes=decode_comm.MNNVL_BUFFER_BYTES)
    assert all(comm.takes_allreduce_norm(rows, 7168) for rows in range(1, 145))
    assert not comm.takes_allreduce_norm(145, 7168)
    assert not comm.takes_allreduce_norm(0, 7168)
    assert not comm.takes_allreduce_norm(8, 7164)
    assert 18 * 7168 * 16 * 2 <= decode_comm.DECODE_AR_ONE_SHOT_MAX_BYTES < 19 * 7168 * 16 * 2
    # A two-shot call needs a buffer of whole 32-byte units.
    assert not _comm(world=16, buffer_bytes=(4 << 20) + 16).takes_allreduce_norm(64, 7168)
    assert _comm(world=16, buffer_bytes=(4 << 20) + 16).takes_allreduce_norm(8, 7168)


@pytest.mark.parametrize("eps,expected", [(1e-5, 1e-5), (None, torch.finfo(torch.bfloat16).eps)])
def test_allreduce_norm_passes_the_workspace_ceiling_and_norm(fused_calls, eps, expected):
    comm = _comm(world=16)
    norm = nn.RMSNorm(OUT, eps=eps, dtype=torch.bfloat16)
    partial = torch.ones(3, OUT, dtype=torch.bfloat16)[:, :]
    residual = torch.zeros(3, OUT, dtype=torch.bfloat16)
    comm.allreduce_norm(partial, residual, norm)
    (call,) = fused_calls
    assert call["input"] is partial and call["residual"] is residual
    assert call["workspace"] is comm.mnnvl and call["norm_weight"] is norm.weight
    assert call["one_shot_max_bytes"] == decode_comm.DECODE_AR_ONE_SHOT_MAX_BYTES
    assert call["eps"] == pytest.approx(expected)
    stock = SimpleNamespace(variance_epsilon=1e-6, weight=norm.weight)
    comm.allreduce_norm(partial, residual, stock)
    assert fused_calls[1]["eps"] == pytest.approx(1e-6)


def test_norms_take_the_fused_all_reduces():
    assert _drafter()._k3_norms_take_comm()


@pytest.mark.parametrize(
    "case",
    [
        "gemma input norm",
        "fp32 post-attention norm",
        "final norm of another width",
        "nvfp4 norm output",
        "high-precision norm output",
        "o_proj without its all-reduce",
        "column-parallel o_proj",
        "down projection that does not reduce",
    ],
)
def test_norms_do_not_take_the_fused_all_reduces(case):
    drafter = _drafter()
    layer = drafter.model.layers[1]
    if case == "gemma input norm":
        layer.input_layernorm = _norm(use_gemma=True)
    elif case == "fp32 post-attention norm":
        layer.post_attention_layernorm = _norm(dtype=torch.float32)
    elif case == "final norm of another width":
        drafter.model.norm = _norm(hidden_size=OUT + 8)
    elif case == "nvfp4 norm output":
        layer.input_layernorm.is_nvfp4 = True
    elif case == "high-precision norm output":
        layer.post_attention_layernorm.return_hp_output = True
    elif case == "o_proj without its all-reduce":
        layer.self_attn.o_proj = _projection(K_O, all_reduce=False)
    elif case == "column-parallel o_proj":
        layer.self_attn.o_proj = _projection(K_O, row_parallel=False)
    else:
        layer.mlp.down_proj = _projection(K_DOWN, reduce_output=False)
    assert not drafter._k3_norms_take_comm()


def test_use_decode_comm_compiles_each_sandwich_form(sandwich_calls):
    comm = _comm(world=16)
    drafter = _drafter(tp_size=16, tp_rank=3)
    drafter.use_decode_comm(comm)
    assert drafter._k3_norms_fuse
    assert drafter._k3_sandwich_forms == {"o_proj", "down"}
    assert isinstance(drafter.fc, target.K3FcSlice)
    assert (drafter.fc.start, drafter.fc.end) == (24, 32)
    plain, swiglu = sandwich_calls.calls
    assert not plain["swiglu"] and swiglu["swiglu"]
    assert plain["x"].shape == (1, K_O) and plain["weight"].shape == (OUT, K_O)
    assert swiglu["x"].shape == (1, 2 * K_DOWN) and swiglu["weight"].shape == (OUT, K_DOWN)
    for call in (plain, swiglu):
        # One zero row of a zero weight on the group's sandwich workspace: it compiles the kernel and adds nothing.
        assert call["workspace"] is comm.sandwich
        assert call["residual"].shape == (1, OUT) and call["norm_weight"].shape == (OUT,)
        assert not call["x"].any() and not call["weight"].any() and not call["residual"].any()


def test_use_decode_comm_skips_a_form_the_layers_do_not_share(sandwich_calls):
    drafter = _drafter()
    drafter.model.layers[1].self_attn.o_proj = _projection(2 * K_O)
    drafter.use_decode_comm(_comm())
    assert drafter._k3_sandwich_forms == {"down"}
    assert [call["swiglu"] for call in sandwich_calls.calls] == [True]


def test_use_decode_comm_skips_a_form_the_kernel_does_not_take(sandwich_calls):
    sandwich_calls.unsupported.add(K_DOWN)
    drafter = _drafter()
    drafter.use_decode_comm(_comm())
    assert drafter._k3_norms_fuse and drafter._k3_sandwich_forms == {"o_proj"}
    assert [call["swiglu"] for call in sandwich_calls.calls] == [False]


def test_use_decode_comm_without_the_fused_norms_compiles_nothing(sandwich_calls):
    drafter = _drafter()
    drafter.model.norm = _norm(use_gemma=True)
    drafter.use_decode_comm(_comm())
    assert not drafter._k3_norms_fuse and drafter._k3_sandwich_forms == frozenset()
    assert sandwich_calls.calls == []
    # The context projection's split does not depend on the norms.
    assert isinstance(drafter.fc, target.K3FcSlice)


def test_compile_plain_declines_a_shape_the_kernel_does_not_take(sandwich_calls):
    sandwich_calls.unsupported.add(K_O)
    assert not _comm().compile_plain(torch.ones(OUT, K_O, dtype=torch.bfloat16))
    assert sandwich_calls.calls == []


def test_sandwich_plain_passes_its_arguments(sandwich_calls):
    comm = _comm(world=16)
    norm = _norm()
    x = torch.ones(3, 2 * K_DOWN, dtype=torch.bfloat16)
    weight = torch.ones(OUT, K_DOWN, dtype=torch.bfloat16)
    residual = torch.zeros(3, OUT, dtype=torch.bfloat16)
    comm.sandwich_plain(x, weight, residual, norm, swiglu=True)
    (call,) = sandwich_calls.calls
    assert call["x"] is x and call["weight"] is weight and call["residual"] is residual
    assert call["norm_weight"] is norm.weight and call["eps"] == pytest.approx(1e-6)
    assert call["swiglu"] and call["workspace"] is comm.sandwich


def test_takes_plain_asks_the_kernel_with_the_norm_weight(monkeypatch):
    asked = []
    monkeypatch.setattr(
        decode_comm._sandwich_op,
        "supports_plain",
        lambda *args: asked.append(args) or True,
    )
    norm = _norm()
    x, weight, residual = torch.ones(2, K_O), torch.ones(OUT, K_O), torch.zeros(2, OUT)
    assert decode_comm.K3DecodeComm.takes_plain(x, weight, residual, norm, swiglu=True)
    (args,) = asked
    assert all(a is b for a, b in zip(args, (x, weight, residual, norm.weight)))
    assert args[4] is True


def test_skip_all_reduce_turns_the_modules_all_reduce_off():
    params = decode_comm.skip_all_reduce()
    assert params.enable_allreduce is False and params.residual is None
    assert decode_comm.skip_all_reduce() is not params
