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
"""The Kimi-K3 side of dense DWDP: roots, model tables, hook placement and the load guard.

Covers ``modeling_kimi_linear``: the model tables, the load guard and the hook placement (M0, M1,
H0-H3, R3).  Layers are real modules built from a tiny Kimi-Linear config.  A real forward of the
attention and MoE submodules needs attention metadata, KV-cache pools and the routed-expert
kernels, so the hook tests replace exactly those submodules with recorders; everything else in the
decoder-layer and model forward runs for real, on CPU in fp32 (the eager norm paths).  The dense
runtime is either a recording fake or the real ``DenseDWDP`` with CPU stand-ins for its streams
and events.

KDA layers, dense-MLP and MoE, build on CPU.  The routed experts of a MoE layer are a stand-in
without tensors, built as in ``test_kimi_k3_mlp.py``: ``create_moe`` resolves the backend for the
GPU's SM version and puts the SiTU constants of a CUTLASS backend on the current CUDA device, and
the experts never run here.  The MLA mixer (``KimiK3MLAAttention``) allocates its attention
backends' RoPE tables and scale tensors on CUDA, so the MLA cases run twice: with the real mixer,
which needs a GPU and skips without one (the parameters still stay on CPU), and on CPU with only
that mixer replaced by a stand-in.  The stand-in keeps the layer's real MLA branch and its
``KimiMLARuntime`` wrapper, which are what the roots and the hooks depend on.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.models.modeling_utils import remove_weights
from tensorrt_llm._torch.modules import kimi_k3_mla
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    RING_DEPTH,
    ROW_ALIGN,
    SEGMENT_ORDER,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE, CutlassFusedMoE
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

# The model tables, verbatim.
_DESIGN_ATTR_ALLOW = frozenset(
    {
        "_bfa_proj_weight",
        "_qkvg_proj_weight",
        "_gate_b_t",
        "_packed_conv_weight",
        "_mtp_conv_weights",
        "_w_q_t",
        "_w_k_t",
        "_w_v_t",
        "_A_log_f32",
        "_dt_bias_f32",
        "_onorm_w_f32",
    }
)
_DESIGN_EXCLUDE = frozenset(
    {
        "_o_dense",
        "_fused_q_placeholder",
        "softmax_stats_tensor",
        "temp_softmax_stats_tensor",
        "e_score_correction_bias",
    }
)
# The MoE-dense roots, in walk order.
_DESIGN_MOE_ROOTS = (
    "gate",
    "shared_experts",
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "routed_expert_norm",
)
# Table names no Kimi-K3 module of this branch defines: the KDA mixer has no low-rank gate
# transpose (``_gate_b_t``) and no decode scratch buffer (``_o_dense``), the MLA mixer no fused Q
# placeholder (``_fused_q_placeholder``).  The tables keep them; here they match nothing.
_UNDEFINED_TABLE_NAMES = frozenset({"_gate_b_t", "_o_dense", "_fused_q_placeholder"})

_HIDDEN = 256
_TOKENS = 3

# Layers of the default ``_kimi_config``: 0 is a dense-MLP KDA layer, then the KDA, MLA and KDA
# MoE layers.
_KDA_MOE_LAYER = 1
_MLA_MOE_LAYER = 2
_MOE_LAYERS = (1, 2, 3)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the MLA mixer's attention backends allocate their tables on CUDA",
)
# The MoE layers the root and hook tests build: KDA, MLA with its real mixer, and MLA with the
# CPU stand-in mixer.
_MOE_LAYER_KINDS = [
    pytest.param("kda", marks=pytest.mark.cpu_only),
    pytest.param("mla", marks=_requires_cuda),
    pytest.param("mla_cpu_mixer", marks=pytest.mark.cpu_only),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _kimi_config(
    *,
    kda_layers: tuple[int, ...] = (1, 2, 4),
    mla_layers: tuple[int, ...] = (3,),
    first_k_dense_replace: int = 1,
    full_rank_gate: bool = True,
) -> KimiLinearConfig:
    """A tiny Kimi-K3: by default a dense-MLP KDA layer 0, then KDA, MLA and KDA MoE layers.

    ``kda_layers`` and ``mla_layers`` are 1-indexed, as in the checkpoint config.
    """
    return KimiLinearConfig(
        vocab_size=128,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        num_hidden_layers=len(kda_layers) + len(mla_layers),
        num_attention_heads=2,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        attn_res_block_size=2,
        mla_use_output_gate=True,
        max_position_embeddings=64,
        linear_attn_config={
            "kda_layers": list(kda_layers),
            "full_attn_layers": list(mla_layers),
            "num_heads": 2,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": full_rank_gate,
        },
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=128,
        num_shared_experts=1,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        first_k_dense_replace=first_k_dense_replace,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )


def _kda_config(num_layers: int, *, first_k_dense_replace: int = 1) -> KimiLinearConfig:
    """``num_layers`` KDA layers: a whole model that builds on CPU."""
    return _kimi_config(
        kda_layers=tuple(range(1, num_layers + 1)),
        mla_layers=(),
        first_k_dense_replace=first_k_dense_replace,
    )


def _model_config(cfg: KimiLinearConfig) -> ModelConfig:
    """The deployed context-worker shape: tp1 with attention DP (fp32 router, replicated MLPs)."""
    return ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=1, tp_size=1, rank=0, enable_attention_dp=True),
        quant_config=QuantConfig(),
        moe_backend="CUTLASS",
    )


def _aux_streams() -> dict:
    # Construction only stores the auxiliary streams; nothing in these tests runs on them.
    return {kind: object() for kind in AuxStreamType}


def _build_layer(layer_idx: int, cfg: KimiLinearConfig | None = None) -> mkl.KimiLinearDecoderLayer:
    cfg = _kimi_config() if cfg is None else cfg
    return mkl.KimiLinearDecoderLayer(_model_config(cfg), cfg, layer_idx, _aux_streams())


def _build_model(monkeypatch: pytest.MonkeyPatch, cfg: KimiLinearConfig) -> mkl.KimiLinearModel:
    # The model creates its auxiliary streams eagerly; CPU stand-ins keep CUDA out of it.
    monkeypatch.setattr(torch.cuda, "Stream", object)
    return mkl.KimiLinearModel(_model_config(cfg))


class _CpuMlaMixer(nn.Module):
    """Stands in for ``KimiK3MLAAttention``, whose attention backends allocate CUDA tensors.

    Takes the mixer's keyword arguments and holds one projection, so ``KimiMLARuntime`` wraps it
    as it wraps the real mixer.
    """

    def __init__(self, *, hidden_size: int, num_heads: int, v_head_dim: int, **kwargs) -> None:
        super().__init__()
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)


def _use_cpu_mla_mixer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build MLA layers with ``_CpuMlaMixer``; ``KimiMLARuntime`` imports the mixer per call."""
    monkeypatch.setattr(kimi_k3_mla, "KimiK3MLAAttention", _CpuMlaMixer)


class _CpuRoutedBackend(CutlassFusedMoE):
    """The CUTLASS backend the layers request, without its constructor and its tensors."""

    def __init__(self, num_experts: int) -> None:
        nn.Module.__init__(self)
        self.initial_local_expert_ids = list(range(num_experts))


class _CpuRoutedExperts(ConfigurableMoE):
    """Stands in for the routed experts ``create_moe`` builds.

    Holds what ``KimiK3MoERuntime`` reads at construction: a backend with this rank's experts
    (all of them at tp1), no load balancer and no all-reduce.
    """

    def __init__(self, *, num_experts: int, **kwargs) -> None:
        nn.Module.__init__(self)
        self.backend = _CpuRoutedBackend(num_experts)
        self.comm = None
        self.layer_load_balancer = None
        self.all_reduce = None


@pytest.fixture
def cpu_routed_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the routed experts as ``_CpuRoutedExperts``, so MoE layers build on CPU."""
    monkeypatch.setattr(mkl, "create_moe", lambda **kwargs: _CpuRoutedExperts(**kwargs))


@pytest.fixture(params=_MOE_LAYER_KINDS)
def moe_layer(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, cpu_routed_experts: None
) -> mkl.KimiLinearDecoderLayer:
    """A MoE layer of the default ``_kimi_config``, of each kind in ``_MOE_LAYER_KINDS``."""
    kind = request.param
    if kind == "mla_cpu_mixer":
        _use_cpu_mla_mixer(monkeypatch)
    layer = _build_layer(_KDA_MOE_LAYER if kind == "kda" else _MLA_MOE_LAYER)
    assert layer.is_moe
    assert layer.is_kda == (kind == "kda")
    if not layer.is_kda:
        assert isinstance(layer.self_attn, mkl.KimiMLARuntime)
        assert isinstance(layer.self_attn.mixer, _CpuMlaMixer) == (kind == "mla_cpu_mixer")
    return layer


@contextlib.contextmanager
def _global_dwdp_manager(manager: object):
    set_global_dwdp_manager(manager)
    try:
        yield
    finally:
        set_global_dwdp_manager(None)


@pytest.fixture(autouse=True)
def _no_global_dwdp_manager():
    """Layers capture the global manager when built; every test starts and ends without one."""
    previous = get_global_dwdp_manager()
    set_global_dwdp_manager(None)
    yield
    set_global_dwdp_manager(previous)


# ---------------------------------------------------------------------------
# Recorders and fakes
# ---------------------------------------------------------------------------


class _Recorder(nn.Module):
    """Stands in for an attention, MoE or MLP submodule: logs the call, returns a scaled input."""

    def __init__(self, log: list, name: str, layer_idx: int) -> None:
        super().__init__()
        self._log = log
        self._name = name
        self._layer_idx = layer_idx
        self.fail = False

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._log.append((self._name, self._layer_idx))
        if self.fail:
            raise RuntimeError(f"{self._name} of layer {self._layer_idx} failed")
        return hidden_states * 0.5


class _Embedding(nn.Module):
    """Stands in for the token embedding: logs the call, returns fp32 embeddings."""

    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._log.append(("embed",))
        return torch.randn(input_ids.shape[0], _HIDDEN)


class _RecordingDense:
    """The ``DenseDWDP`` calls the model makes, recorded; it covers the layers in ``covered``.

    Any other runtime call raises ``AttributeError``.
    """

    def __init__(self, log: list, covered: tuple[int, ...]) -> None:
        self._log = log
        self._covered = frozenset(covered)

    def covers(self, layer: int) -> bool:
        self._log.append(("covers", layer))
        return layer in self._covered

    def wait_layer(self, layer: int) -> None:
        self._log.append(("wait_layer", layer))

    def attention_done(self, layer: int) -> None:
        self._log.append(("attention_done", layer))

    def layer_done(self, layer: int) -> None:
        self._log.append(("layer_done", layer))


class _RecordingManager:
    """``DwdpManager.dense`` and ``DwdpManager.ensure_primed``, recorded."""

    def __init__(self, log: list, dense: object | None) -> None:
        self._log = log
        self._dense = dense

    @property
    def dense(self) -> object | None:
        return self._dense

    def ensure_primed(self) -> None:
        self._log.append(("ensure_primed",))


class _PrimingManager:
    """``DwdpManager.dense`` and ``ensure_primed`` as the manager runs them.

    The eager-execution check first; after an aborted forward a WAR barrier, then ``L(first),
    E(first), L(second), E(second)`` unless already primed.  The expert half is recorded instead
    of run; the covered layers stand in for the expert DWDP MoE layers, which setup requires to
    be the same list.
    """

    def __init__(self, log: list, dense: DenseDWDP) -> None:
        self._log = log
        self.dense = dense

    def ensure_primed(self) -> None:
        dense = self.dense
        dense.require_eager("DwdpManager.ensure_primed (H0)")
        if dense.forward_open:
            dense.war_barrier()
            self._log.append(("war_barrier",))
        if not dense.primed:
            for layer in dense.layers[:2]:
                dense.prefetch(layer)
                self._log.append(("expert_prefetch", layer))
            dense.mark_primed()


class _CpuStream:
    def wait_event(self, event: object) -> None:
        pass


class _CpuEvent:
    def record(self, stream: object = None) -> None:
        pass


class _CpuOps:
    """``DeviceOps`` without CUDA: the runtime's event protocol is checked on the host."""

    def __init__(self) -> None:
        self._current = _CpuStream()

    def new_event(self) -> _CpuEvent:
        return _CpuEvent()

    def current_stream(self) -> _CpuStream:
        return self._current

    @contextlib.contextmanager
    def stream(self, stream: _CpuStream):
        previous, self._current = self._current, stream
        try:
            yield
        finally:
            self._current = previous

    def synchronize(self) -> None:
        pass

    def is_current_stream_capturing(self) -> bool:
        return False

    def do_multi_stream(self) -> bool:
        return False

    def nvtx_range(self, message: str) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()


def _install_recorders(layer: mkl.KimiLinearDecoderLayer, log: list) -> _Recorder:
    """Replace the layer's attention and MoE (or MLP) submodules; log its norms' calls too.

    Returns:
        The recorder standing in for the attention.
    """
    attention = _Recorder(log, "attention", layer.layer_idx)
    setattr(layer, "linear_attn" if layer.is_kda else "self_attn", attention)
    if layer.is_moe:
        layer.block_sparse_moe = _Recorder(log, "moe", layer.layer_idx)
    else:
        layer.mlp = _Recorder(log, "mlp", layer.layer_idx)
    for norm in ("input_layernorm", "post_attention_layernorm"):
        getattr(layer, norm).register_forward_pre_hook(
            lambda module, args, name=norm: log.append((name, layer.layer_idx))
        )
    return attention


def _attn_metadata() -> SimpleNamespace:
    # The model forward checks the batch against ``num_tokens``; the recorders read nothing.
    return SimpleNamespace(num_tokens=_TOKENS)


def _run_layer(layer: mkl.KimiLinearDecoderLayer) -> None:
    # fp32 activations keep the residual mixing and the norms on their eager CPU paths.
    hidden_states = torch.randn(_TOKENS, _HIDDEN)
    block_residual = torch.zeros(2, _TOKENS, _HIDDEN)
    layer(hidden_states, block_residual, 0, _attn_metadata())


def _layer_reads(layer_idx: int, ffn: str) -> list[tuple]:
    """The submodule and norm calls of one decoder-layer forward, in program order."""
    return [
        ("input_layernorm", layer_idx),
        ("attention", layer_idx),
        ("post_attention_layernorm", layer_idx),
        (ffn, layer_idx),
    ]


def _hooked_layer_reads(layer_idx: int, ffn: str) -> list[tuple]:
    """``_layer_reads`` of a covered layer, with H1-H3 where the protocol puts them."""
    return [
        ("covers", layer_idx),
        ("wait_layer", layer_idx),
        ("input_layernorm", layer_idx),
        ("attention", layer_idx),
        ("attention_done", layer_idx),
        ("post_attention_layernorm", layer_idx),
        (ffn, layer_idx),
        ("layer_done", layer_idx),
    ]


# ---------------------------------------------------------------------------
# M1: tables and roots
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_tables_match_the_design_and_sit_on_the_decoder_layer():
    assert isinstance(mkl.DWDP_DENSE_ATTR_ALLOW, frozenset)
    assert isinstance(mkl.DWDP_DENSE_EXCLUDE, frozenset)
    assert mkl.DWDP_DENSE_ATTR_ALLOW == _DESIGN_ATTR_ALLOW
    assert mkl.DWDP_DENSE_EXCLUDE == _DESIGN_EXCLUDE
    assert not mkl.DWDP_DENSE_ATTR_ALLOW & mkl.DWDP_DENSE_EXCLUDE

    # Setup reads the tables off each decoder layer, with an empty default.
    layer = _build_layer(0)
    assert getattr(layer, "dense_dwdp_attr_allow", frozenset()) is mkl.DWDP_DENSE_ATTR_ALLOW
    assert getattr(layer, "dense_dwdp_exclude", frozenset()) is mkl.DWDP_DENSE_EXCLUDE
    assert mkl.KimiLinearDecoderLayer.dense_dwdp_attr_allow is mkl.DWDP_DENSE_ATTR_ALLOW
    assert mkl.KimiLinearDecoderLayer.dense_dwdp_exclude is mkl.DWDP_DENSE_EXCLUDE


@pytest.mark.cpu_only
@pytest.mark.parametrize("full_rank_gate", [True, False])
def test_allowed_names_are_plain_attributes_of_the_kda_mixer(full_rank_gate):
    """Every allowed name is spelled as the KDA mixer spells its plain attributes."""
    attention = _build_layer(0, _kimi_config(full_rank_gate=full_rank_gate)).linear_attn
    attributes = vars(attention)
    registered = set(attention._parameters) | set(attention._buffers) | set(attention._modules)
    for name in sorted(mkl.DWDP_DENSE_ATTR_ALLOW - _UNDEFINED_TABLE_NAMES):
        assert name in attributes, f"{name} is not an attribute of {type(attention).__name__}"
        assert name not in registered, f"{name} is a parameter, buffer or submodule"
    # The table names this branch's mixer lacks name nothing on it.
    assert not any(hasattr(attention, name) for name in _UNDEFINED_TABLE_NAMES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA decode constants are built on CUDA")
def test_allowed_names_hold_the_built_decode_weights():
    """After ``finalize_decode_weights`` every allowed name holds a tensor or a tensor tuple."""
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import is_kda_optimized_supported

    if not is_kda_optimized_supported():
        pytest.skip("the KDA decode constants are built only on SM100/SM103/SM107")
    built = set()
    for full_rank_gate in (True, False):
        attention = _build_layer(0, _kimi_config(full_rank_gate=full_rank_gate)).linear_attn
        attention.cuda().finalize_decode_weights()
        for name in mkl.DWDP_DENSE_ATTR_ALLOW - _UNDEFINED_TABLE_NAMES:
            value = getattr(attention, name)
            if value is None:
                continue
            built.add(name)
            items = value if isinstance(value, tuple) else (value,)
            assert items and all(isinstance(item, torch.Tensor) for item in items), name
    # The full-rank gate builds every allowed name the mixer defines; with the low-rank gate
    # ``finalize_decode_weights`` builds nothing.
    assert built == mkl.DWDP_DENSE_ATTR_ALLOW - _UNDEFINED_TABLE_NAMES


@pytest.mark.cpu_only
def test_dense_mlp_layer_has_no_roots():
    layer = _build_layer(0)
    assert not layer.is_moe
    assert layer.dense_dwdp_roots("attn") == []
    assert layer.dense_dwdp_roots("moe") == []


@pytest.mark.cpu_only
@pytest.mark.parametrize("segment", ["", "ATTN", "mlp", "experts", "attn,moe"])
def test_unknown_segment_raises(segment):
    with pytest.raises(ValueError, match="segment"):
        _build_layer(0).dense_dwdp_roots(segment)


def test_moe_layer_roots_cover_their_segments(moe_layer):
    layer = moe_layer
    attention_path = "linear_attn" if layer.is_kda else "self_attn"

    attn_roots = layer.dense_dwdp_roots("attn")
    assert attn_roots == [(attention_path, layer.get_submodule(attention_path))]

    moe = layer.block_sparse_moe
    moe_roots = layer.dense_dwdp_roots("moe")
    assert [path for path, _ in moe_roots] == [
        f"block_sparse_moe.{name}" for name in _DESIGN_MOE_ROOTS
    ]
    for path, module in attn_roots + moe_roots:
        assert layer.get_submodule(path) is module
    assert isinstance(moe.gate, mkl.KimiK3MoEGate)

    # Every child of the MoE block except the routed experts is a dense root: nothing dense is
    # left replicated, and nothing expert DWDP owns is reached.
    assert set(dict(moe.named_children())) == set(_DESIGN_MOE_ROOTS) | {"routed_experts"}

    def subtree(roots):
        return {id(module) for _, root in roots for module in root.modules()}

    attn_modules, moe_modules = subtree(attn_roots), subtree(moe_roots)
    assert not attn_modules & moe_modules
    assert not (attn_modules | moe_modules) & set(map(id, moe.routed_experts.modules()))
    replicated = [
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.self_attention_res_norm,
        layer.self_attention_res_proj,
        layer.mlp_res_norm,
        layer.mlp_res_proj,
    ]
    assert not (attn_modules | moe_modules) & set(map(id, replicated))

    # Excluded names the built modules carry: the routing bias.  The real MLA mixer has no Q
    # placeholder on this branch.
    assert "e_score_correction_bias" in moe.gate._parameters
    if not layer.is_kda and not isinstance(layer.self_attn.mixer, _CpuMlaMixer):
        assert not any(hasattr(layer.self_attn.mixer, name) for name in _UNDEFINED_TABLE_NAMES)


def test_layer_without_weights_has_no_roots(moe_layer):
    layer = moe_layer
    assert layer.dense_dwdp_roots("attn") and layer.dense_dwdp_roots("moe")
    remove_weights(layer)
    assert layer.dense_dwdp_roots("attn") == []
    assert layer.dense_dwdp_roots("moe") == []


def _fp8_weight_read_on_cpu(
    cls: type[mkl._Fp8BlockScaleWeightReadLinear], linear: nn.Module
) -> mkl._Fp8BlockScaleWeightReadLinear:
    """``_Fp8BlockScaleWeightReadLinear.from_linear`` without its GPU block-scale quantizer."""
    return cls(linear.weight.detach().to(torch.float8_e4m3fn), torch.ones(1), linear.out_features)


@pytest.mark.cpu_only
def test_moe_roots_are_the_modules_the_load_left(cpu_routed_experts, monkeypatch):
    """The roots are the MoE block's children when setup asks, not when the layer was built.

    The load's FP8 weight-read conversion replaces the latent projections, which are roots
    themselves, and empties the replaced modules.  It runs here as in ``_finalize_weight_load``,
    with only the quantizer stubbed.
    """
    monkeypatch.setattr(
        mkl._Fp8BlockScaleWeightReadLinear, "from_linear", classmethod(_fp8_weight_read_on_cpu)
    )
    layer = _build_layer(_KDA_MOE_LAYER)
    moe = layer.block_sparse_moe
    built = dict(layer.dense_dwdp_roots("moe"))
    converted = mkl._convert_moe_mlps_to_fp8_weight_read(SimpleNamespace(layers=[layer]))

    roots = dict(layer.dense_dwdp_roots("moe"))
    assert roots == {f"block_sparse_moe.{name}": getattr(moe, name) for name in _DESIGN_MOE_ROOTS}
    for name in ("routed_expert_down_proj", "routed_expert_up_proj"):
        path = f"block_sparse_moe.{name}"
        assert isinstance(roots[path], mkl._Fp8BlockScaleWeightReadLinear)
        assert roots[path] is not built[path]
    # Every converted projection, the shared expert's included, is reached from the roots.
    swapped = [
        module for module in moe.modules() if isinstance(module, mkl._Fp8BlockScaleWeightReadLinear)
    ]
    reached = {id(module) for root in roots.values() for module in root.modules()}
    assert len(swapped) == converted
    assert all(id(module) in reached for module in swapped)


# ---------------------------------------------------------------------------
# M0, H1-H3: the decoder-layer hooks
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_layers_and_model_capture_the_manager_when_built(monkeypatch):
    manager = _RecordingManager([], None)
    with _global_dwdp_manager(manager):
        layer = _build_layer(0)
        model = _build_model(monkeypatch, _kda_config(2, first_k_dense_replace=2))
    assert layer._dwdp is manager
    assert model._dwdp is manager
    assert all(decoder._dwdp is manager for decoder in model.layers)

    assert _build_layer(0)._dwdp is None
    assert _build_model(monkeypatch, _kda_config(1))._dwdp is None


@pytest.mark.cpu_only
def test_hooks_bracket_the_attention_and_mlp_reads():
    log = []
    layer = _build_layer(0)
    _install_recorders(layer, log)
    layer._dwdp = _RecordingManager(log, _RecordingDense(log, covered=(0,)))
    _run_layer(layer)
    assert log == _hooked_layer_reads(0, "mlp")


def test_hooks_bracket_the_attention_and_moe_reads(moe_layer):
    log = []
    _install_recorders(moe_layer, log)
    moe_layer._dwdp = _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS))
    _run_layer(moe_layer)
    assert log == _hooked_layer_reads(moe_layer.layer_idx, "moe")


def test_fused_moe_residual_still_closes_the_moe_window(moe_layer, monkeypatch):
    """``TRTLLM_K3_FUSE_MOE_RESIDUAL`` returns from the layer forward early; H3 still runs."""
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    log = []
    _install_recorders(moe_layer, log)
    moe_layer._dwdp = _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS))
    _run_layer(moe_layer)
    assert log == _hooked_layer_reads(moe_layer.layer_idx, "moe")


@pytest.mark.parametrize("segment", ["attn", "moe"])
def test_fused_moe_residual_refuses_dense_roots(moe_layer, monkeypatch, segment):
    """Dense DWDP setup is abandoned while the fused MoE residual is on (not validated)."""
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    with pytest.raises(ValueError, match="TRTLLM_K3_FUSE_MOE_RESIDUAL"):
        moe_layer.dense_dwdp_roots(segment)


@pytest.mark.parametrize("fused", [True, False], ids=["above_cap", "at_cap"])
def test_ctx_fused_moe_residual_still_closes_the_moe_window(moe_layer, monkeypatch, fused):
    """``TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX``: above the row cap the layer hands the residual to the
    MoE block and returns early, never deferred; H1-H3 keep their places either way."""
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL_CTX", True)
    monkeypatch.setattr(mkl, "_MOE_RESIDUAL_MAX_FUSED_ROWS", _TOKENS - 1 if fused else _TOKENS)
    log = []
    _install_recorders(moe_layer, log)
    moe_calls = []
    recorder_forward = moe_layer.block_sparse_moe.forward

    def recording_forward(hidden_states, *args, **kwargs):
        moe_calls.append((kwargs.get("residual") is not None, kwargs.get("defer_residual")))
        return recorder_forward(hidden_states, *args, **kwargs)

    monkeypatch.setattr(moe_layer.block_sparse_moe, "forward", recording_forward)
    moe_layer._dwdp = _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS))
    _run_layer(moe_layer)
    assert log == _hooked_layer_reads(moe_layer.layer_idx, "moe")
    assert moe_calls == ([(True, False)] if fused else [(False, None)])


@pytest.mark.parametrize("segment", ["attn", "moe"])
def test_ctx_fused_moe_residual_keeps_dense_roots(moe_layer, monkeypatch, segment):
    """The context fusion runs with dense DWDP: the roots are the same with it on; with
    ``TRTLLM_K3_FUSE_MOE_RESIDUAL`` on as well, setup is still refused."""
    roots = moe_layer.dense_dwdp_roots(segment)
    assert roots
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL_CTX", True)
    assert moe_layer.dense_dwdp_roots(segment) == roots
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    with pytest.raises(ValueError, match="TRTLLM_K3_FUSE_MOE_RESIDUAL"):
        moe_layer.dense_dwdp_roots(segment)


@pytest.mark.cpu_only
@pytest.mark.parametrize("state", ["no_manager", "no_dense", "uncovered"])
def test_no_hook_runs_without_a_covering_runtime(state):
    log = []
    layer = _build_layer(0)
    _install_recorders(layer, log)
    if state == "no_manager":
        layer._dwdp = None
    elif state == "no_dense":
        layer._dwdp = _RecordingManager(log, None)
    else:
        layer._dwdp = _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS))
    _run_layer(layer)
    covers = [("covers", 0)] if state == "uncovered" else []
    assert log == covers + _layer_reads(0, "mlp")


# ---------------------------------------------------------------------------
# H0: priming in the model forward
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("state", ["dense", "no_dense", "no_manager"])
def test_model_primes_before_the_embedding_only_with_dense(monkeypatch, state):
    log = []
    model = _build_model(monkeypatch, _kda_config(1))
    _install_recorders(model.layers[0], log)
    model.embed_tokens = _Embedding(log)
    manager = {
        "dense": _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS)),
        "no_dense": _RecordingManager(log, None),
        "no_manager": None,
    }[state]
    model._dwdp = manager
    model.layers[0]._dwdp = manager

    output = model(_attn_metadata(), input_ids=torch.zeros(_TOKENS, dtype=torch.int32))

    assert output.shape == (_TOKENS, _HIDDEN)
    if state == "dense":
        assert log == [("ensure_primed",), ("embed",), ("covers", 0)] + _layer_reads(0, "mlp")
    else:
        assert log == [("embed",)] + _layer_reads(0, "mlp")

    # The input check runs first: a rejected call does not prime.
    log.clear()
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model(_attn_metadata())
    assert log == []


@pytest.mark.cpu_only
def test_model_forward_drives_the_dense_runtime(cpu_routed_experts, monkeypatch):
    """A real ``DenseDWDP`` in strict mode accepts the hooks of real model forwards.

    The model mixes KDA and MLA layers as Kimi-K3 does.  Strict mode raises on any demand fetch
    or fence, so a missing or misplaced H0, H2 or H3 fails here; the copy of layer ``l + 2`` is
    enqueued between the attention and the MoE block of layer ``l``.  An aborted forward is
    recovered by the next H0 with a WAR barrier.
    """
    log = []
    _use_cpu_mla_mixer(monkeypatch)
    model = _build_model(monkeypatch, _kimi_config())
    assert [layer.is_kda for layer in model.layers] == [True, True, False, True]
    covered = tuple(layer.layer_idx for layer in model.layers if layer.dense_dwdp_roots("attn"))
    assert covered == _MOE_LAYERS
    attention = {layer.layer_idx: _install_recorders(layer, log) for layer in model.layers}
    slots = {
        kind: allocate_slot_ring(SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2), "cpu")
        for kind in SEGMENT_ORDER
    }
    dense = DenseDWDP(covered, SEGMENT_ORDER, slots, _CpuStream(), ops=_CpuOps(), strict=True)
    dense.set_launcher(lambda layer: log.append(("launch", layer)))
    dense.commit(None, fill_byte=0)
    manager = _PrimingManager(log, dense)
    model._dwdp = manager
    for layer in model.layers:
        layer._dwdp = manager

    def forward():
        return model(_attn_metadata(), inputs_embeds=torch.randn(_TOKENS, _HIDDEN))

    def copies_and_reads():
        kinds = {"launch", "expert_prefetch", "war_barrier", "attention", "mlp", "moe"}
        return [entry for entry in log if entry[0] in kinds]

    one_forward = [
        ("launch", 1),
        ("expert_prefetch", 1),
        ("launch", 2),
        ("expert_prefetch", 2),
        ("attention", 0),
        ("mlp", 0),
        ("attention", 1),
        ("launch", 3),
        ("moe", 1),
        ("attention", 2),
        ("moe", 2),
        ("attention", 3),
        ("moe", 3),
    ]
    try:
        for _ in range(2):
            log.clear()
            forward()
            assert copies_and_reads() == one_forward
            assert not dense.forward_open and not dense.primed

        # Abort inside the MLA layer 2: its read windows are open and layer 3 never runs.
        attention[_MLA_MOE_LAYER].fail = True
        log.clear()
        with pytest.raises(RuntimeError, match="attention of layer 2 failed"):
            forward()
        assert dense.forward_open
        attention[_MLA_MOE_LAYER].fail = False
        log.clear()
        forward()
        assert copies_and_reads() == [("war_barrier",)] + one_forward
        assert dense.demand_fetches == 0 and dense.war_fences == 0
    finally:
        dense.release()


# ---------------------------------------------------------------------------
# R3: the load guard
# ---------------------------------------------------------------------------


class _LoadReached(Exception):
    """Raised by the stubbed loader once ``load_weights`` got past its guard."""


def _causal_lm(active: bool | None) -> mkl.KimiLinearForCausalLM:
    # The guard reads only ``self.model``; the first loader step is stubbed to report reaching it.
    model = mkl.KimiLinearForCausalLM.__new__(mkl.KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    if active is not None:
        model.model._dense_dwdp_active = active

    def trunk_parameters():
        raise _LoadReached

    model._trunk_parameters = trunk_parameters
    return model


@pytest.mark.cpu_only
def test_load_weights_refuses_while_dense_is_active():
    with pytest.raises(RuntimeError, match="dense DWDP is active"):
        _causal_lm(active=True).load_weights({})
    for active in (False, None):
        with pytest.raises(_LoadReached):
            _causal_lm(active=active).load_weights({})


class _VisionTower(nn.Module):
    """Counts the loads that reach the vision tower."""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load_weights(self, weights) -> None:
        self.loads += 1


def _vl_wrapper(active: bool | None, tower: nn.Module | None):
    from tensorrt_llm._torch.models.modeling_kimi_k3_vl import KimiK3ForConditionalGeneration

    # The wrapper's guard reads ``self.model``, which is ``self.llm.model``.
    wrapper = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.mm_encoder = tower
    wrapper.llm = _causal_lm(active=active)
    return wrapper


@pytest.mark.cpu_only
@pytest.mark.parametrize("deferred", [False, True], ids=["tower-built", "tower-deferred"])
def test_vl_wrapper_load_weights_refuses_before_the_vision_tower(deferred):
    """A load through the deployed VL wrapper is refused before it touches the vision tower.

    Neither a built tower is loaded nor a deferred one (``None`` after meta init) rebuilt; the
    wrapper has no ``model_config`` here, so a rebuild would fail with an ``AttributeError``.
    Past the guards, the stubbed text loader raises ``_LoadReached``, not a ``RuntimeError``.
    """
    tower = None if deferred else _VisionTower()
    wrapper = _vl_wrapper(active=True, tower=tower)
    with pytest.raises(
        RuntimeError,
        match=r"^KimiK3ForConditionalGeneration load_weights is not supported while dense DWDP is "
        r"active",
    ):
        wrapper.load_weights({})
    assert wrapper.mm_encoder is tower
    if tower is not None:
        assert tower.loads == 0


@pytest.mark.cpu_only
def test_vl_wrapper_load_weights_runs_without_dense_dwdp():
    for active in (False, None):
        wrapper = _vl_wrapper(active=active, tower=_VisionTower())
        with pytest.raises(_LoadReached):
            wrapper.load_weights({})
        assert wrapper.mm_encoder.loads == 1
