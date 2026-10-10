# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_dflash as dflash
from tensorrt_llm._torch.models import modeling_lilicorr as lili
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.speculative.dflash import (
    DFlashWorker,
    dflash_context_dtype,
    last_accepted_hidden,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

HEAD_CONFIG = dict(
    model_hidden_size=32,
    hidden_size=8,
    num_layers=2,
    num_heads=2,
    mlp_ratio=2,
    block_size=4,
    candidate_topk=3,
    factor_dim=8,
    rms_norm_eps=1e-5,
    vector_eps=1e-4,
    logit_scale=8,
)


def _head() -> lili.LiLiCorrHead:
    head = lili.LiLiCorrHead(**HEAD_CONFIG).eval()
    for name, parameter in head.named_parameters():
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        parameter.data.copy_(torch.sin(values * 0.17 + sum(name.encode()) * 0.013) * 0.2)
    return head


def test_head_matches_reference_scores() -> None:
    """Compare all start/transition potentials with a frozen Model Optimizer reference."""
    head = _head()
    embeddings = torch.linspace(-1.2, 1.3, 512).reshape(16, 32)
    ids = torch.tensor([[[2, 5, 9], [6, 3, 1], [4, 7, 8]]])
    logs = torch.tensor([[[-0.4, -1.5, -3.0], [-0.6, -1.7, -2.7], [-0.2, -2.1, -3.4]]])
    hidden = torch.linspace(-0.7, 0.9, 96).reshape(1, 3, 32)
    anchor = torch.linspace(-0.5, 0.4, 32).reshape(1, 32)
    start, pairs = head(embeddings[ids], logs, hidden, anchor)
    expected_start = torch.tensor([[-2.00565147, -2.00631571, -2.01010752]])
    expected_pairs = torch.tensor(
        [
            [
                [
                    [1.16437435, 1.16812551, 1.15701485],
                    [1.16656375, 1.17030466, 1.15921211],
                    [1.17068172, 1.17445230, 1.16334319],
                ],
                [
                    [1.14806044, 1.14459574, 1.12698877],
                    [1.14481199, 1.14134955, 1.12373281],
                    [1.14475155, 1.14125943, 1.12357390],
                ],
            ]
        ]
    )
    torch.testing.assert_close(start, expected_start, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(pairs, expected_pairs, atol=2e-5, rtol=2e-5)


def test_candidate_walk_follows_the_selected_predecessor() -> None:
    start = torch.tensor([[1.0, 4.0], [5.0, 1.0]])
    pairs = torch.tensor(
        [
            [[[8.0, 0.0], [2.0, 7.0]], [[4.0, 3.0], [6.0, 1.0]]],
            [[[1.0, 9.0], [8.0, 0.0]], [[7.0, 2.0], [3.0, 8.0]]],
        ]
    )
    expected = torch.tensor(
        [[[1.0, 4.0], [2.0, 7.0], [6.0, 1.0]], [[5.0, 1.0], [1.0, 9.0], [3.0, 8.0]]]
    )
    torch.testing.assert_close(lili.lilicorr_proposal_logits(start, pairs), expected)
    torch.testing.assert_close(lili.lilicorr_proposal_logits(start, pairs[:, :0]), start[:, None])


@pytest.mark.parametrize("tp_size", [1, 4])
def test_worker_normalizes_full_vocabulary(tp_size: int) -> None:
    torch.manual_seed(21)
    logits = torch.randn(2, 3, 32)
    shards = logits.chunk(tp_size, dim=-1)
    expected_values, expected_ids = logits.topk(3, dim=-1)
    for rank, shard in enumerate(shards):
        worker = DFlashWorker.__new__(DFlashWorker)
        worker.mapping = SimpleNamespace(tp_size=tp_size, tp_rank=rank, enable_attention_dp=False)
        worker._d2t = None

        def gather(local: torch.Tensor, mapping: object, dim: int) -> torch.Tensor:
            if local.shape[-1] == 1:
                payloads = [x.logsumexp(-1, keepdim=True) for x in shards]
            else:
                payloads = []
                for index, values in enumerate(shards):
                    top, ids = values.topk(3, dim=-1)
                    payloads.append(
                        torch.stack(((ids + index * shard.shape[-1]).float(), top), -1).flatten(-2)
                    )
            torch.testing.assert_close(local, payloads[rank])
            return torch.cat(payloads, dim=dim)

        def select(
            ids: torch.Tensor, probs: torch.Tensor, hidden: torch.Tensor, anchor: torch.Tensor
        ) -> torch.Tensor:
            torch.testing.assert_close(ids, expected_ids)
            torch.testing.assert_close(probs, logits.log_softmax(-1).gather(-1, ids))
            return expected_values

        model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=32),
            lilicorr=SimpleNamespace(candidate_topk=3),
            select_lilicorr_path=select,
        )
        with patch("tensorrt_llm._torch.distributed.ops.allgather", side_effect=gather):
            actual = worker._apply_lilicorr(
                model,
                shard.clone(),
                torch.zeros(2, 3, 8),
                torch.zeros(2, 8),
                SimpleNamespace(draft_vocab_size=32, vocab_size=32),
            )
        torch.testing.assert_close(actual.gather(-1, expected_ids), expected_values)
        assert (actual > -torch.inf).sum() == expected_ids.numel()


def test_projection_cache_uses_activation_dtype() -> None:
    packed = SimpleNamespace(weight=torch.zeros(8, 8, dtype=torch.uint8), dtype=torch.bfloat16)
    assert dflash_context_dtype(SimpleNamespace(fc=packed)) == torch.bfloat16
    assert dflash_context_dtype(SimpleNamespace(fc=nn.Linear(8, 8))) == torch.float32


def test_partial_acceptance_selects_normalized_target_row() -> None:
    model = lili.LiLiCorrForCausalLM.__new__(lili.LiLiCorrForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(torch_dtype=torch.float32)
    model.fc = nn.Linear(64, 32, bias=False)
    model.fc.weight.data.copy_(torch.linspace(-0.3, 0.5, 2048).reshape(32, 64))
    model.hidden_norm = lili.LiLiCorrRMSNorm(32, 1e-5)
    model.hidden_norm.weight.data.copy_(torch.linspace(0.5, 1.5, 32))
    captured = torch.sin(torch.arange(3 * 4 * 64).float() * 0.17).reshape(3, 4, 64)
    counts = torch.tensor([1, 3, 4])

    projected = model.project_target_hidden(captured)
    actual = last_accepted_hidden(projected, counts)
    raw = captured[torch.arange(3), counts - 1] @ model.fc.weight.T
    expected = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-5)
    expected = expected * model.hidden_norm.weight
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, raw)


def test_worker_rejects_unsupported_layout_before_collectives() -> None:
    worker = DFlashWorker.__new__(DFlashWorker)
    worker.mapping = SimpleNamespace(tp_size=2, tp_rank=0, enable_attention_dp=False)
    worker._d2t = None
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=16), lilicorr=SimpleNamespace(candidate_topk=3)
    )
    logits = torch.zeros(1, 3, 9)
    with patch("tensorrt_llm._torch.distributed.ops.allgather") as gather:
        with pytest.raises(NotImplementedError, match="plain TP column shard"):
            worker._apply_lilicorr(
                model,
                logits,
                torch.zeros(1, 3, 32),
                torch.zeros(1, 32),
                SimpleNamespace(draft_vocab_size=16, vocab_size=16),
            )
        gather.assert_not_called()


@pytest.fixture
def lightweight_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    class Backbone(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()
            self.model.norm = nn.LayerNorm(32)
            self.lm_head = nn.Linear(32, 16, bias=False)

    monkeypatch.setattr(dflash, "get_model_architecture", lambda config: (Backbone, None))
    monkeypatch.setattr(dflash, "get_dflash_flash_attention", lambda: None)


def _draft_config() -> ModelConfig:
    return ModelConfig(
        pretrained_config=SimpleNamespace(
            architectures=["LiLiCorrDraftModel"],
            hidden_size=32,
            vocab_size=16,
            torch_dtype=torch.float32,
            rms_norm_eps=1e-5,
            num_hidden_layers=0,
            dflash_config={"block_size": 4, **{"lilicorr_" + k: v for k, v in HEAD_CONFIG.items()}},
        )
    )


@pytest.mark.usefixtures("lightweight_backbone")
@pytest.mark.parametrize("marker", ["architecture", "projector_type", "lilicorr_enabled", "plain"])
def test_checkpoint_markers_route_to_the_correct_drafter(marker: str) -> None:
    config = _draft_config()
    if marker == "architecture":
        config.pretrained_config.block_size = config.pretrained_config.dflash_config.pop(
            "block_size"
        )
        config.pretrained_config.dflash_config.update(conv_kernel_size=2, conv_group_size=8)
    else:
        config.pretrained_config.architectures = ["Qwen3ForCausalLM"]
    if marker == "projector_type":
        config.pretrained_config.dflash_config[marker] = "lilicorr"
    elif marker == "lilicorr_enabled":
        config.pretrained_config.dflash_config[marker] = True
    assert dflash.declares_lilicorr(config.pretrained_config) == (marker != "plain")
    model = dflash._build_dflash_draft(
        SimpleNamespace(spec_config=SimpleNamespace(attention_backend="VANILLA")),
        config,
        None,
        None,
    )
    expected_type = DFlashForCausalLM if marker == "plain" else lili.LiLiCorrForCausalLM
    assert type(model) is expected_type
    assert not model.is_dflash2
    if marker == "architecture":
        assert model.block_size == 4
        assert model._dflash2_conv_taps == 2 and model.candidate_selector is None


def test_laguna_lilicorr_is_rejected() -> None:
    config = _draft_config()
    config.pretrained_config.architectures = ["LagunaForCausalLM", "LiLiCorrDraftModel"]
    with pytest.raises(NotImplementedError, match="generic GQA DFlash backbone"):
        dflash._build_dflash_draft(
            SimpleNamespace(spec_config=SimpleNamespace(attention_backend="VANILLA")),
            config,
            None,
            None,
        )


@pytest.mark.usefixtures("lightweight_backbone")
@pytest.mark.parametrize("missing", ["dflash_config", "block_size"])
def test_missing_checkpoint_settings_have_clear_errors(missing: str) -> None:
    config = _draft_config()
    if missing == "dflash_config":
        del config.pretrained_config.dflash_config
        message = "missing config fields"
    else:
        del config.pretrained_config.dflash_config["block_size"]
        message = "requires dflash_config.block_size or block_size"
    with pytest.raises(ValueError, match=message):
        lili.LiLiCorrForCausalLM(config, dflash_attention_backend="VANILLA")


@pytest.mark.usefixtures("lightweight_backbone")
@pytest.mark.parametrize(
    "settings, message",
    [
        ({"conv_kernel_size": -1}, "conv_kernel_size"),
        ({"conv_kernel_size": 5}, "conv_kernel_size"),
        ({"conv_kernel_size": 2, "conv_group_size": 0}, "conv_group_size"),
        ({"causal": True}, "non-causal"),
    ],
)
def test_invalid_convolution_and_causal_settings(settings: dict, message: str) -> None:
    config = _draft_config()
    config.pretrained_config.dflash_config.update(settings)
    with pytest.raises(ValueError, match=message):
        lili.LiLiCorrForCausalLM(config, dflash_attention_backend="VANILLA")


@pytest.mark.usefixtures("lightweight_backbone")
@pytest.mark.parametrize("is_causal", [None, False, True])
def test_lilicorr_attention_is_noncausal_with_symmetric_windows(is_causal: bool | None) -> None:
    config = _draft_config()
    config.pretrained_config.is_causal = is_causal
    config.pretrained_config.layer_types = ["sliding_attention", "full_attention"]
    config.pretrained_config.sliding_window = 8
    config.pretrained_config.use_sliding_window = True
    if is_causal:
        with pytest.raises(ValueError, match="non-causal"):
            lili.LiLiCorrForCausalLM(config, dflash_attention_backend="VANILLA")
        return
    model = lili.LiLiCorrForCausalLM(config, dflash_attention_backend="VANILLA")
    assert model._get_attention_mask_args(0) == (False, (7, 7))
    assert model._get_attention_mask_args(1) == (False, (-1, -1))


@pytest.mark.usefixtures("lightweight_backbone")
@pytest.mark.parametrize("algo", [None, QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4])
def test_projection_width_comes_from_checkpoint(algo: QuantAlgo | None) -> None:
    config = _draft_config()
    config.quant_config = QuantConfig(quant_algo=algo)
    model = lili.LiLiCorrForCausalLM(config, dflash_attention_backend="VANILLA")
    assert model.target_layer_ids is None
    packed = algo in (QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4)
    weight = (
        torch.zeros(32, 48, dtype=torch.uint8)
        if packed
        else torch.linspace(-0.5, 0.5, 32 * 96).reshape(32, 96)
    )
    weights = {"fc.weight": weight}
    if algo is None:
        model._load_target_projection(weights)
        assert model.fc.in_features == 96
        rows = torch.randn(2, 96)
        torch.testing.assert_close(model.fc(rows), rows @ weight.T)
    else:
        with patch.object(lili, "_load_linear", return_value=nn.Identity()) as load:
            model._load_target_projection(weights)
        assert load.call_args.args[1:3] == (96, 32)
    assert not weights


def test_checkpoint_loading_preserves_metadata_scales_and_own_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock native GEMMs; exercise checkpoint routing, finalization and sharing."""

    class RecordingLinear(nn.Module):
        def __init__(
            self, *args: object, quant_config: QuantConfig | None = None, **kwargs: object
        ) -> None:
            super().__init__()
            self.quant_config, self.loaded, self.scale = quant_config, {}, None

        def create_weights(self) -> None:
            pass

        def load_weights(self, weights: list[dict]) -> None:
            self.loaded = weights[0]

        def process_weights_after_loading(self) -> None:
            self.scale = self.loaded.get("weight_scale_2")

    class MLP(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.__dict__.update(
                dict(
                    hidden_size=32,
                    intermediate_size=64,
                    split_gate_up=False,
                    activation=torch.nn.functional.silu,
                    layer_idx=None,
                    swiglu_limit=None,
                    swiglu_alpha=None,
                    swiglu_beta=None,
                )
                | kwargs
            )
            self.gate_up_proj = SimpleNamespace(has_bias=False)
            if self.split_gate_up:
                self.gate_proj, self.up_proj = RecordingLinear(), RecordingLinear()

    def initialize(self: nn.Module, config: ModelConfig, **kwargs: object) -> None:
        nn.Module.__init__(self)
        self.model_config, self.config = config, config.pretrained_config
        self.target_layer_ids, self.block_size, self._dflash2_conv_taps = [0, 1], 4, 0
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(32)
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.mlp = MLP()
        self.lm_head = RecordingLinear()
        self.draft_model_full = SimpleNamespace(model=self.model, lm_head=self.lm_head)

    def load_backbone(self: nn.Module, weights: dict, **kwargs: object) -> None:
        self._load_target_projection(weights)
        for layer_idx, layer in enumerate(self.model.layers):
            for name in ("gate", "up"):
                getattr(layer.mlp, name + "_proj").load_weights(
                    [
                        {
                            "weight_scale_2": weights[
                                f"layers.{layer_idx}.mlp.{name}_proj.weight_scale_2"
                            ]
                        }
                    ]
                )

    monkeypatch.setattr(lili, "Linear", RecordingLinear)
    monkeypatch.setattr(lili, "GatedMLP", MLP)
    monkeypatch.setattr(DFlashForCausalLM, "__init__", initialize)
    monkeypatch.setattr(DFlashForCausalLM, "load_weights", load_backbone)
    config = ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=32,
            vocab_size=16,
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            has_own_lm_head=True,
            dflash_config={"lilicorr_" + k: v for k, v in HEAD_CONFIG.items()},
        ),
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.W4A16_NVFP4,
            group_size=16,
            exclude_modules=["lilicorr.feature_mlp*"],
        ),
        quant_config_dict={
            "fc": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=32),
            "lilicorr.feature_mlp.1": QuantConfig(quant_algo=QuantAlgo.FP8),
            "layers.0.mlp.gate_proj": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=32),
            "layers.0.mlp.up_proj": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16),
            "layers.1.mlp.gate_proj": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16),
            "layers.1.mlp.up_proj": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=32),
        },
    )
    model = lili.LiLiCorrForCausalLM(config)
    source_head = _head().state_dict()
    weights = {
        "lilicorr."
        + k.replace("feature_norm.", "feature_mlp.0.").replace(
            "feature_mlp.up_proj.", "feature_mlp.1."
        ): v.unsqueeze(1) if k in ("slot_embedding", "rank_embedding") else v
        for k, v in source_head.items()
    }
    weights.update(
        {
            "fc.weight": torch.zeros(32, 32, dtype=torch.uint8),
            "fc.weight_scale": torch.ones(32, 2).to(torch.float8_e4m3fn),
            "fc.weight_scale_2": torch.tensor(0.125),
            "lm_head.weight": torch.ones(16, 32),
            "layers.0.mlp.gate_proj.weight_scale_2": torch.tensor(0.25),
            "layers.0.mlp.up_proj.weight_scale_2": torch.tensor(0.5),
            "layers.1.mlp.gate_proj.weight_scale_2": torch.tensor(0.75),
            "layers.1.mlp.up_proj.weight_scale_2": torch.tensor(1.0),
        }
    )
    model.load_weights(weights)
    assert model.fc.quant_config.group_size == 32
    assert isinstance(model.lilicorr.feature_mlp.up_proj, nn.Linear)
    for name, value in model.lilicorr.state_dict().items():
        torch.testing.assert_close(value, source_head[name].to(torch.bfloat16))
    mlp = model.model.layers[0].mlp
    assert mlp.layer_idx == 0
    assert mlp.split_gate_up and mlp.gate_proj.scale == 0.25 and mlp.up_proj.scale == 0.5
    assert mlp.gate_proj.quant_config.group_size == 32 and mlp.up_proj.quant_config.group_size == 16
    second_mlp = model.model.layers[1].mlp
    assert second_mlp.layer_idx == 1
    assert second_mlp.gate_proj.quant_config.group_size == 16
    assert second_mlp.up_proj.quant_config.group_size == 32
    own_head = model.lm_head
    embedding = nn.Embedding(16, 32)
    model.load_weights_from_target_model(
        SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding), lm_head=object())
    )
    assert model.model.embed_tokens is embedding and model.lm_head is own_head
    torch.testing.assert_close(own_head.loaded["weight"], weights["lm_head.weight"])
