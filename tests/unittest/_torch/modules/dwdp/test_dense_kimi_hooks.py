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
"""The Kimi-K3 side of dense DWDP: hook placement, the load guard and what the collector moves.

Covers the hook calls H0-H3 of ``modeling_kimi_linear``, its load guard and the dense layout of its
real decoder layers, on a real model built from a tiny Kimi-Linear config.  A real forward of the
attention and MoE submodules needs attention metadata, KV-cache pools and the routed-expert kernels,
so recorders replace exactly those submodules; everything else in the model forward runs for real,
on CPU in fp32.  Stand-ins without tensors replace the routed experts and the MLA mixer
(``KimiK3MLAAttention``, whose attention backends allocate on CUDA); the MLA layer keeps its real
MLA branch and ``KimiMLARuntime`` wrapper.  The model reaches dense DWDP through the dense DWDP
manager it captures when built (``get_global_dense_dwdp_manager``), in both modes: attached and
standalone.
"""

from types import SimpleNamespace
from typing import NoReturn

import pytest
import torch
from torch import nn

pytest.importorskip("fla")
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules import kimi_k3_mla
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout
from tensorrt_llm._torch.modules.dwdp.dense.manager import set_global_dense_dwdp_manager
from tensorrt_llm._torch.modules.dwdp.dense.types import SegmentKind
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, CutlassFusedMoE
from tensorrt_llm._torch.pyexecutor.dwdp import set_global_dwdp_manager
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

pytestmark = pytest.mark.cpu_only

_HIDDEN = 256
_TOKENS = 3
# Layers of the default ``_kimi_config``: 0 is a dense-MLP KDA layer, then KDA, MLA and KDA MoE.
_MOE_LAYERS = (1, 2, 3)


def _kimi_config(
    kda_layers: tuple[int, ...] = (1, 2, 4), mla_layers: tuple[int, ...] = (3,)
) -> KimiLinearConfig:
    """A tiny Kimi-K3; ``kda_layers`` and ``mla_layers`` are 1-indexed, as in the checkpoint."""
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
            "use_full_rank_gate": True,
        },
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=128,
        num_shared_experts=1,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        first_k_dense_replace=1,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )


def _model_config(cfg: KimiLinearConfig) -> ModelConfig:
    """The deployed context-worker shape: tp1 with attention DP (fp32 router, replicated MLPs)."""
    return ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=1, tp_size=1, rank=0, enable_attention_dp=True),
        quant_config=QuantConfig(),
        moe_backend="CUTLASS",
    )


class _CpuMlaMixer(nn.Module):
    """Stands in for ``KimiK3MLAAttention``; ``KimiMLARuntime`` wraps it as the real mixer."""

    def __init__(
        self, *, hidden_size: int, num_heads: int, v_head_dim: int, **kwargs: object
    ) -> None:
        super().__init__()
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)


class _CpuRoutedBackend(CutlassFusedMoE):
    """The CUTLASS backend the layers request, without its constructor and its tensors."""

    def __init__(self, num_experts: int) -> None:
        nn.Module.__init__(self)
        self.initial_local_expert_ids = list(range(num_experts))

    @classmethod
    def can_implement(cls, p: object, d: object) -> NoReturn:
        raise NotImplementedError("CPU stand-in")

    def _get_quant_method(self) -> NoReturn:
        raise NotImplementedError("CPU stand-in")


class _CpuRoutedExperts(ConfigurableMoE):
    """What ``KimiK3MoERuntime`` reads of the routed experts at construction, at tp1."""

    def __init__(self, *, num_experts: int, **kwargs: object) -> None:
        nn.Module.__init__(self)
        self.backend = _CpuRoutedBackend(num_experts)
        self.comm = None
        self.layer_load_balancer = None
        self.all_reduce = None


@pytest.fixture(autouse=True)
def _cpu_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU stand-ins for the routed experts, the MLA mixer and the model's auxiliary streams.

    Layers capture the global managers when built: every test starts and ends without one.
    """
    monkeypatch.setattr(mkl, "create_moe", lambda **kwargs: _CpuRoutedExperts(**kwargs))
    monkeypatch.setattr(kimi_k3_mla, "KimiK3MLAAttention", _CpuMlaMixer)
    monkeypatch.setattr(torch.cuda, "Stream", object)
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------


class _Recorder(nn.Module):
    """Stands in for an attention, MoE, MLP or embedding: logs the call, returns fp32 states."""

    def __init__(self, log: list, *entry: object) -> None:
        super().__init__()
        self._log = log
        self._entry = entry

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        self._log.append(self._entry)
        if hidden_states.dtype == torch.int32:  # the token ids of the embedding
            return torch.randn(hidden_states.shape[0], _HIDDEN)
        return hidden_states * 0.5


class _RecordingDense:
    """The ``DenseDWDP`` calls the model makes, recorded; any other call raises."""

    def __init__(self, log: list) -> None:
        self._log = log

    def covers(self, layer: int) -> bool:
        self._log.append(("covers", layer))
        return layer in _MOE_LAYERS

    def wait_layer(self, layer: int) -> None:
        self._log.append(("wait_layer", layer))

    def attention_done(self, layer: int) -> None:
        self._log.append(("attention_done", layer))

    def layer_done(self, layer: int) -> None:
        self._log.append(("layer_done", layer))


def _manager(log: list, dense: object | None) -> SimpleNamespace:
    """``DenseDwdpManager.dense`` and ``DenseDwdpManager.ensure_primed``, recorded."""
    return SimpleNamespace(dense=dense, ensure_primed=lambda: log.append(("ensure_primed",)))


def _install_recorders(layer: mkl.KimiLinearDecoderLayer, log: list) -> None:
    """Replace the layer's attention and MoE (or MLP) submodules; log its norms' calls too."""
    index = layer.layer_idx
    setattr(
        layer, "linear_attn" if layer.is_kda else "self_attn", _Recorder(log, "attention", index)
    )
    ffn = "block_sparse_moe" if layer.is_moe else "mlp"
    setattr(layer, ffn, _Recorder(log, "ffn", index))
    for norm in ("input_layernorm", "post_attention_layernorm"):
        getattr(layer, norm).register_forward_pre_hook(
            lambda module, args, name=norm: log.append((name, index))
        )


def _layer_reads(index: int, dense: bool) -> list[tuple]:
    """The calls of one decoder-layer forward; a covered layer brackets its reads with H1-H3."""
    hooked = dense and index in _MOE_LAYERS
    return [
        *([("covers", index)] if dense else []),
        *([("wait_layer", index)] if hooked else []),
        ("input_layernorm", index),
        ("attention", index),
        *([("attention_done", index)] if hooked else []),
        ("post_attention_layernorm", index),
        ("ffn", index),
        *([("layer_done", index)] if hooked else []),
    ]


@pytest.mark.parametrize("registered", [True, False], ids=["registered", "unregistered"])
def test_the_model_and_its_layers_capture_the_registered_dense_manager(registered: bool) -> None:
    """Built with or without a registered dense DWDP manager, the model and each layer keep it."""
    manager = _manager([], None) if registered else None
    set_global_dense_dwdp_manager(manager)

    model = mkl.KimiLinearModel(_model_config(_kimi_config()))

    assert model._dense_dwdp is manager
    assert model.layers and all(layer._dense_dwdp is manager for layer in model.layers)


@pytest.mark.parametrize("state", ["dense", "no_dense", "no_manager"])
def test_the_model_primes_first_and_brackets_the_reads_of_covered_layers(state: str) -> None:
    """H0 primes before the embedding; H1-H3 bracket the attention and MoE reads of KDA and MLA."""
    log = []
    model = mkl.KimiLinearModel(_model_config(_kimi_config()))
    kinds = [(layer.is_moe, layer.is_kda) for layer in model.layers]
    assert kinds == [(False, True), (True, True), (True, False), (True, True)]
    model.embed_tokens = _Recorder(log, "embed")
    manager = {
        "dense": _manager(log, _RecordingDense(log)),
        "no_dense": _manager(log, None),
        "no_manager": None,
    }[state]
    model._dense_dwdp = manager
    for layer in model.layers:
        _install_recorders(layer, log)
        layer._dense_dwdp = manager
    # fp32 activations keep the residual mixing and the norms on their eager CPU paths.
    metadata = SimpleNamespace(num_tokens=_TOKENS)

    output = model(metadata, input_ids=torch.zeros(_TOKENS, dtype=torch.int32))

    assert output.shape == (_TOKENS, _HIDDEN)
    dense = state == "dense"
    reads = [("ensure_primed",)] * dense + [("embed",)]
    assert log == reads + sum((_layer_reads(i, dense) for i in range(len(kinds))), [])
    # The input check runs first: a rejected call does not prime.
    log.clear()
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model(metadata)
    assert log == []


class _LoadReached(Exception):
    """Raised by the stubbed loader once ``load_weights`` got past its guard."""


def _causal_lm(active: bool | None) -> mkl.KimiLinearForCausalLM:
    # The guard reads only ``self.model``; the first loader step is stubbed to report reaching it.
    model = mkl.KimiLinearForCausalLM.__new__(mkl.KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    if active is not None:
        model.model._dense_dwdp_active = active

    def trunk_parameters() -> NoReturn:
        raise _LoadReached

    model._trunk_parameters = trunk_parameters
    return model


def test_load_weights_refuses_while_dense_is_active() -> None:
    with pytest.raises(RuntimeError, match="dense DWDP is active"):
        _causal_lm(active=True).load_weights({})
    for active in (False, None):
        with pytest.raises(_LoadReached):
            _causal_lm(active=active).load_weights({})


# ---------------------------------------------------------------------------
# What the collector moves from the real layers
# ---------------------------------------------------------------------------


def _finish_loading(model: mkl.KimiLinearModel) -> None:
    """What the loader does after the checkpoint: transform each Linear, build the KDA tensors.

    ``finalize_decode_weights`` skips CPU, so its fused projections and constants are built here.
    """
    for module in model.modules():
        if isinstance(module, Linear):
            module.post_load_weights()
    for layer in model.layers:
        if layer.is_kda:
            kda = layer.linear_attn
            kda._qkvg_proj_weight = kda._merge_projection_weights(
                (kda.q_proj, kda.k_proj, kda.v_proj, kda.g_proj)
            )
            kda._bfa_proj_weight = kda._merge_projection_weights(
                (kda.f_a_proj, kda.b_proj), pad_rows_to=8
            )
            kda._build_decode_kernel_constants()
            kda._build_mtp_conv_weights()


def _root_tensors(roots: list[tuple[str, nn.Module]]) -> set[str]:
    """The names dense DWDP binds under ``roots``: the tensors that may move and their aliases.

    Parameters, buffers and the tensor attributes ``DWDP_DENSE_ATTR_ALLOW`` names may move unless
    ``DWDP_DENSE_EXCLUDE`` names them; any other tensor reference under the roots moves only as a
    view of a storage that moves.
    """
    references = []
    for root_path, root in roots:
        for path, module in root.named_modules(prefix=root_path):
            for table in (module._parameters, module._buffers):
                references += [
                    (f"{path}.{name}", tensor, name not in mkl.DWDP_DENSE_EXCLUDE)
                    for name, tensor in table.items()
                    if tensor is not None
                ]
            for name, value in vars(module).items():
                allowed = name in mkl.DWDP_DENSE_ATTR_ALLOW - mkl.DWDP_DENSE_EXCLUDE
                if isinstance(value, torch.Tensor):
                    references.append((f"{path}.{name}", value, allowed))
                elif isinstance(value, (tuple, list)) and value:
                    if all(isinstance(item, torch.Tensor) for item in value):
                        references += [
                            (f"{path}.{name}[{index}]", item, allowed)
                            for index, item in enumerate(value)
                        ]
    storage = lambda tensor: tensor.untyped_storage().data_ptr()  # noqa: E731
    moving = {storage(tensor) for _, tensor, may_move in references if may_move}
    return {name for name, tensor, _ in references if storage(tensor) in moving}


def test_the_collector_moves_every_root_tensor_of_the_moe_layers_and_nothing_else() -> None:
    """Each tensor under the attention and MoE-dense roots moves; layer 0 and the experts stay.

    Strict collection with a 1-byte threshold also fails on any plain tensor attribute under the
    roots that ``DWDP_DENSE_ATTR_ALLOW`` and ``DWDP_DENSE_EXCLUDE`` do not account for.
    """
    model = mkl.KimiLinearModel(_model_config(_kimi_config()))
    _finish_loading(model)

    layout = collect_dense_layout(
        list(enumerate(model.layers)),
        DenseConfig(enabled=True, min_bytes=1, strict=True),
        num_ranks=2,
        device=torch.device("cpu"),
        attr_allow=mkl.DWDP_DENSE_ATTR_ALLOW,
        exclude=mkl.DWDP_DENSE_EXCLUDE,
    )

    assert layout.layers == _MOE_LAYERS
    assert layout.warnings == []
    for index in _MOE_LAYERS:
        layer = model.layers[index]
        attention = "linear_attn" if layer.is_kda else "self_attn"
        assert [path for path, _ in layer.dense_dwdp_roots("attn")] == [attention]
        assert [path for path, _ in layer.dense_dwdp_roots("moe")] == [
            "block_sparse_moe.gate",
            "block_sparse_moe.shared_experts",
            "block_sparse_moe.routed_expert_down_proj",
            "block_sparse_moe.routed_expert_up_proj",
            "block_sparse_moe.routed_expert_norm",
        ]
        for kind in (SegmentKind.ATTN, SegmentKind.MOE):
            moved = {
                binding.qualified_name
                for anchor in layout.segment(index, kind).anchors
                for binding in anchor.bindings
            }
            assert moved == _root_tensors(layer.dense_dwdp_roots(kind.value)), (index, kind)
            assert not any(name.startswith("block_sparse_moe.routed_experts") for name in moved)
