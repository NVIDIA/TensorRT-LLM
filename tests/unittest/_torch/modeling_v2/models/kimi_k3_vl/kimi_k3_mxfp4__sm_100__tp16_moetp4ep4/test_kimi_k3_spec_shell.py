# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp4ep4 target class as a causal LM (``modeling.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4`` with
``spec_shell``'s speculative decoding) and the drafter's weight sharding, host-side: the real target ``__init__`` and
``forward`` on a tiny text model (its construction checks, text config and text model stubbed), meta-initialized
parameters, recording workers. (tp16_moetp16ep1's class: ``test_tp16_moetp16ep1_causal_lm.py`` in its directory.)

* The class inherits ``DecoderModelForCausalLM`` directly (README "Model lifecycle").
* Without speculation: the KDA Mamba metadata class, no drafter or worker, and a forward that drops the padded rows
  before the logits processor.
* SA: the speculative runtime's worker, built with the arguments the built-in one-engine model gives it, handed no
  drafter and the target.
* DSpark with a standalone GQA drafter: the drafter config from the built-in ``external_drafter_config_kwargs``,
  ``K3DSparkDrafter`` with every field the worker reads, sharing the shell's logits processor, and this target's
  ``KimiK3DSparkWorker`` holding it (not handed the target); both on the epilogue. ``load_draft_weights`` loads the
  drafter, then shares the target's embedding and LM head.
* DSpark with a standalone MLA drafter: ``K3MLADSparkDrafter`` on its CUTEDSL block decode, sharing only the LM head.
* The GQA drafter on ``attention_backend`` VANILLA (flash-attn over the worker's arena), and AUTO falling back to
  VANILLA where TRTLLM-Gen is unavailable, as the built-in GQA DSpark drafter does.
* The MLA drafter on VANILLA (the eager decode) and on TRTLLM (flashinfer's MLA decode); AUTO with CUTEDSL
  unavailable, or an explicit backend unavailable, fails as the built-in MLA DSpark drafter does (it does not
  degrade).
* Declined with ``NoTarget``: DFlash, an embedded DSpark draft, an FA4 GQA drafter attention (which
  ``DSparkDecodingConfig`` cannot request). The drafter config is loaded with the kwargs ``route`` loads it with.
* The drafter's load cuts each projection for its TP rank exactly as the built-in loader (``load_weight_shard``) does,
  for TP 1, 2, 4 and 16.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, Qwen3Config

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    drafter_config as family_drafter_config,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    drafter as target_drafter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    drafter_mla as target_drafter_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as target_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    spec_shell,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_speculative
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, MetaInitMode
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

HIDDEN, VOCAB = 64, 128


def _text_config():
    return Qwen3Config.from_dict(
        dict(
            architectures=["KimiLinearForCausalLM"],
            model_type="qwen3",
            hidden_size=HIDDEN,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=VOCAB,
            torch_dtype="bfloat16",
            tie_word_embeddings=False,
        )
    )


def _draft_config(heads=64, kv_heads=16, inter=14336, mapping=None, **extra):
    cfg = Qwen3Config.from_dict(
        dict(
            architectures=["DSparkDraftModel"],
            model_type="qwen3",
            hidden_size=256,
            intermediate_size=inter,
            num_hidden_layers=2,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=8,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 256,
            },
            attention_bias=False,
            torch_dtype="bfloat16",
            markov_rank=4,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            block_size=7,
            dflash_config={"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]},
            **extra,
        )
    )
    return ModelConfig(pretrained_config=cfg, mapping=mapping or Mapping())


class _TextModel(nn.Module):
    """A text model that records its inputs and returns ``hidden`` rows."""

    def __init__(self, rows=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.rows = rows
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.arange(self.rows * HIDDEN, dtype=torch.float32).view(self.rows, HIDDEN)


class _Logits(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        self.calls.append((hidden_states, lm_head, return_context_logits))
        return "logits"


def _model_config(spec_config=None):
    config = ModelConfig(pretrained_config=_text_config())
    config.spec_config = spec_config
    return config


TARGET = target_modeling.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4


@pytest.fixture(autouse=True)
def tiny_target(monkeypatch):
    """The target's construction checks pass, the text config is the tiny one itself, the text model is `_TextModel`."""
    monkeypatch.setattr(target_modeling, "_check_construction", lambda model_config: None)
    monkeypatch.setattr(target_modeling, "_text_model_config", lambda model_config: model_config)
    monkeypatch.setattr(target_modeling, "KimiLinearModel", lambda text: _TextModel())


def _shell(spec_config=None):
    """A target built by its own ``__init__`` around the tiny text model, its step contract already checked."""
    model_config = _model_config(spec_config)
    model_config.pretrained_config.text_config = model_config.pretrained_config
    with MetaInitMode():
        model = TARGET(model_config)
    model._step_checked = True
    return model


class _Worker(nn.Module):
    """A one-engine worker that records how it was built and wired."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init = (args, kwargs)
        self.calls = []

    def set_draft_model(self, draft_model):
        self.calls.append(("set_draft_model", draft_model))

    def set_target_model(self, target_model):
        self.calls.append(("set_target_model", target_model))


@pytest.fixture
def dspark(monkeypatch, tmp_path):
    """A DSpark spec config with a standalone drafter, the drafter config ``from_pretrained`` builds (recorded), the
    target's worker recording, and the drafter attention available."""
    built = {}
    draft_config = _draft_config()

    def from_pretrained(path, **kwargs):
        built["path"], built["kwargs"] = path, kwargs
        return draft_config

    monkeypatch.setattr(spec_shell.ModelConfig, "from_pretrained", staticmethod(from_pretrained))
    monkeypatch.setattr(spec_shell._spec_worker, "KimiK3DSparkWorker", _Worker)
    monkeypatch.setattr(target_drafter._block_attn_op, "unavailability_reason", lambda: None)
    spec_config = SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        draft_is_embedded_in_target=False,
        speculative_model=str(tmp_path),
        attention_backend="AUTO",
        moe_backend=None,
        _use_shared_kv_cache=False,
        # Read by should_use_separate_draft_kv_cache (DSparkDecodingConfig's default).
        _allow_separate_draft_kv_cache=True,
        mask_token_id=None,
    )
    return SimpleNamespace(spec_config=spec_config, built=built, draft_config=draft_config)


def test_target_inherits_the_decoder_lm_interface_directly():
    cls = TARGET
    assert cls.__bases__ == (DecoderModelForCausalLM,)
    assert cls.mamba_metadata_cls is KimiK3MambaMetadata and cls.spec_decode_fast_paths is False


def test_base_without_speculation():
    shell = _shell()
    assert isinstance(shell, DecoderModelForCausalLM)
    assert shell.mamba_metadata_cls is KimiK3MambaMetadata
    assert not shell.spec_decode_fast_paths
    assert (shell.spec_config, shell.draft_config, shell.draft_model, shell.spec_worker) == (
        None,
    ) * 4
    assert shell.use_separate_draft_kv_cache is False and shell.layer_idx == -1
    assert isinstance(shell.logits_processor, target_modeling._decode_gemv.K3LogitsProcessor)
    shell.logits_processor = _Logits()
    attn = SimpleNamespace(padded_num_tokens=8, num_tokens=4)
    assert (
        shell.forward(attn, input_ids="ids", position_ids="pos", return_context_logits=True)
        == "logits"
    )
    (kwargs,) = shell.model.calls
    assert (
        kwargs["input_ids"] == "ids"
        and kwargs["position_ids"] == "pos"
        and kwargs["spec_metadata"] is None
    )
    ((hidden, head, ctx),) = shell.logits_processor.calls
    assert hidden.shape[0] == 4 and head is shell.lm_head and ctx is True


def test_spec_shell_without_speculation_is_the_base():
    shell = _shell()
    assert shell.spec_worker is None and shell.draft_model is None
    assert shell.epilogue == [shell.lm_head]


def test_sa_takes_the_runtime_worker(monkeypatch):
    built = []
    monkeypatch.setattr(
        spec_shell, "get_spec_worker", lambda *a, **k: built.append((a, k)) or _Worker(*a, **k)
    )
    spec_config = SimpleNamespace(spec_dec_mode=SpeculativeDecodingMode.SA)
    shell = _shell(spec_config)
    ((args, kwargs),) = built
    assert args[0] is spec_config and args[1].spec_config is spec_config
    assert args[2] == shell.model_config.mapping.mapping_for_full_rows()
    assert kwargs == dict(
        use_separate_draft_kv_cache=False, key_exchange=None, spec_decode_fast_paths=False
    )
    worker = shell.spec_worker
    assert worker.calls == [("set_draft_model", None), ("set_target_model", shell)]
    assert shell.draft_model is None and shell.epilogue == [shell.lm_head, worker]


def test_dspark_builds_the_target_drafter_and_worker(dspark):
    shell = _shell(dspark.spec_config)
    model_config = shell.model_config
    # The drafter config: the built-in kwargs for a standalone external drafter.
    assert dspark.built["path"] == dspark.spec_config.speculative_model
    assert dspark.built["kwargs"] == modeling_speculative.external_drafter_config_kwargs(
        model_config, dspark.spec_config
    )
    assert shell.draft_config is dspark.draft_config
    assert shell.draft_config.extra_attrs is model_config.extra_attrs
    drafter = shell.draft_model
    assert type(drafter) is target_drafter.K3DSparkDrafter
    for name in target_drafter.WORKER_FIELDS:
        assert hasattr(drafter, name), name
    assert drafter.logits_processor is shell.logits_processor
    assert drafter.dflash_attention_backend == "TRTLLM"
    # The drafter is built without the speculative config's mask id.
    assert drafter.mask_token_id == VOCAB - 2
    worker = shell.spec_worker
    assert type(worker) is _Worker
    assert worker.init == ((dspark.spec_config, model_config.mapping, True), {})
    assert worker.calls == [("set_draft_model", drafter)]
    assert shell.epilogue == [shell.lm_head, drafter, worker]
    assert dict(shell.named_children())["draft_model"] is drafter
    assert dict(shell.named_children())["spec_worker"] is worker


def test_dspark_load_draft_weights_then_shares_the_target_head(dspark, monkeypatch):
    shell = _shell(dspark.spec_config)
    loaded = []
    monkeypatch.setattr(
        target_drafter.K3DSparkDrafter,
        "load_weights",
        lambda self, weights, weight_mapper=None: loaded.append((weights, weight_mapper)),
    )
    shell.load_draft_weights({"x": 1}, weight_mapper="mapper")
    assert loaded == [({"x": 1}, "mapper")]
    assert shell.draft_model.model.embed_tokens is shell.model.embed_tokens
    assert shell.draft_model.draft_model_full.model.embed_tokens is shell.model.embed_tokens
    assert shell.draft_model.lm_head is shell.lm_head
    assert shell.draft_model.draft_model_full.lm_head is shell.lm_head


def test_route_loads_the_drafter_config_the_shell_builds_from():
    """The family's ``route`` reads the drafter checkpoint with the shell's ``from_pretrained`` kwargs, and the
    shell dispatches the drafter class on ``drafter_config``'s MLA test."""
    assert (
        spec_shell.external_drafter_config_kwargs
        is family_drafter_config.external_drafter_config_kwargs
    )
    assert spec_shell._drafter_config is family_drafter_config


@pytest.mark.parametrize(
    "case",
    [
        "dflash",
        "embedded dspark",
        # DSparkDecodingConfig cannot ask for FA4; the drafter still refuses it if constructed with it directly.
        "fa4 attention",
    ],
)
def test_declined_speculation_raises_no_target(dspark, monkeypatch, case):
    spec_config = dspark.spec_config
    if case == "dflash":
        spec_config.spec_dec_mode = SpeculativeDecodingMode.DFLASH
    elif case == "embedded dspark":
        spec_config.draft_is_embedded_in_target = True
    else:
        spec_config.attention_backend = "FA4"
    with pytest.raises(NoTarget):
        _shell(spec_config)


def _mla_draft_config():
    """Inferact-Kimi-K3-DSpark's config at small widths (MLA, YaRN)."""
    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "mscale_all_dim": 1.0,
        },
        mask_token_id=VOCAB - 3,
        markov_rank=4,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        torch_dtype=torch.bfloat16,
    )
    return ModelConfig(pretrained_config=cfg)


def _use_mla_drafter(dspark, monkeypatch):
    """``from_pretrained`` returns the MLA drafter config; its block decode is available."""
    mla_config = _mla_draft_config()
    monkeypatch.setattr(
        spec_shell.ModelConfig, "from_pretrained", staticmethod(lambda path, **kwargs: mla_config)
    )
    monkeypatch.setattr(
        target_drafter_mla._mla_ops, "mla_decode_unavailability_reason", lambda: None
    )
    return mla_config


def test_dspark_mla_drafter_builds_the_target_mla_drafter(dspark, monkeypatch):
    """An MLA drafter checkpoint (Inferact-Kimi-K3-DSpark) gets the target's MLA drafter (Modeling V1 serves it with
    its built-in MLA drafter and the same worker); AUTO resolves to its CUTEDSL block decode."""
    mla_config = _use_mla_drafter(dspark, monkeypatch)
    shell = _shell(dspark.spec_config)
    drafter = shell.draft_model
    assert type(drafter) is target_drafter_mla.K3MLADSparkDrafter
    assert shell.draft_config is mla_config
    for name in target_drafter.WORKER_FIELDS:
        assert hasattr(drafter, name), name
    assert drafter.dflash_attention_backend == "CUTEDSL"
    assert (drafter._kv_factor, drafter._num_kv_heads, drafter._head_dim) == (1, 1, 576)
    assert drafter._paged_ctx_cache and not drafter._uses_worker_attention_backend
    assert drafter.mask_token_id == VOCAB - 3
    assert drafter.logits_processor is shell.logits_processor
    worker = shell.spec_worker
    assert worker.calls == [("set_draft_model", drafter)]
    assert shell.epilogue == [shell.lm_head, drafter, worker]
    # Only the LM head is shared: the drafter keeps its own embedding.
    own_embedding = drafter.model.embed_tokens
    drafter.load_weights_from_target_model(shell)
    assert drafter.lm_head is shell.lm_head and drafter.model.embed_tokens is own_embedding


def test_set_guided_decoder_goes_to_the_worker(dspark):
    shell = _shell(dspark.spec_config)
    shell.spec_worker.set_guided_decoder = lambda decoder: decoder == "guided"
    assert shell.set_guided_decoder("guided")


class _NoAllReduce:
    def __init__(self, *args, **kwargs):
        self.args = (args, kwargs)

    def __call__(self, x):
        return x


@pytest.mark.parametrize("tp", [1, 2, 4, 16])
def test_drafter_load_cuts_each_projection_like_the_builtin_loader(monkeypatch, tp):
    """Every rank's shard of the draft checkpoint's projections is what the built-in loader cuts (``load_weight_shard``:
    q / k / v and gate / up by rows, fused in that order; o_proj and down_proj by columns); norms and fc whole."""
    monkeypatch.setattr(target_drafter, "K3AllReduce", _NoAllReduce)
    monkeypatch.setattr(target_drafter._block_attn_op, "unavailability_reason", lambda: None)
    g = torch.Generator().manual_seed(tp)
    base = _draft_config(heads=16, kv_heads=16, inter=64)
    cfg = base.pretrained_config
    hd, hidden = cfg.head_dim, cfg.hidden_size

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    weights = {
        "fc.weight": rnd(hidden, 2 * hidden),
        "hidden_norm.weight": rnd(hidden),
        "norm.weight": rnd(hidden),
        "markov_head.markov_w1.weight": rnd(VOCAB, 4),
        "markov_head.markov_w2.weight": rnd(VOCAB, 4),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        for name, shape in (
            ("self_attn.q_proj", (cfg.num_attention_heads * hd, hidden)),
            ("self_attn.k_proj", (cfg.num_key_value_heads * hd, hidden)),
            ("self_attn.v_proj", (cfg.num_key_value_heads * hd, hidden)),
            ("self_attn.o_proj", (hidden, cfg.num_attention_heads * hd)),
            ("self_attn.q_norm", (hd,)),
            ("self_attn.k_norm", (hd,)),
            ("input_layernorm", (hidden,)),
            ("post_attention_layernorm", (hidden,)),
            ("mlp.gate_proj", (cfg.intermediate_size, hidden)),
            ("mlp.up_proj", (cfg.intermediate_size, hidden)),
            ("mlp.down_proj", (hidden, cfg.intermediate_size)),
        ):
            weights[p + name + ".weight"] = rnd(*shape)
    col, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW
    for rank in range(tp):
        config = _draft_config(
            heads=16, kv_heads=16, inter=64, mapping=Mapping(world_size=tp, tp_size=tp, rank=rank)
        )
        drafter = target_drafter.K3DSparkDrafter(config, dflash_attention_backend="TRTLLM")
        drafter.load_weights(dict(weights))

        def shard(name, mode):
            return load_weight_shard(weights[name], tp, rank, mode)

        for i, layer in enumerate(drafter.model.layers):
            p = f"layers.{i}."
            attn, mlp = layer.self_attn, layer.mlp
            qkv = torch.cat([shard(p + f"self_attn.{n}_proj.weight", col) for n in "qkv"])
            gate_up = torch.cat([shard(p + f"mlp.{n}_proj.weight", col) for n in ("gate", "up")])
            assert torch.equal(attn.qkv_proj.weight, qkv)
            assert torch.equal(attn.o_proj.weight, shard(p + "self_attn.o_proj.weight", row))
            assert torch.equal(mlp.gate_up_proj.weight, gate_up)
            assert torch.equal(mlp.down_proj.weight, shard(p + "mlp.down_proj.weight", row))
            assert torch.equal(attn.q_norm.weight, weights[p + "self_attn.q_norm.weight"])
            assert torch.equal(layer.input_layernorm.weight, weights[p + "input_layernorm.weight"])
            assert (attn.num_heads, attn.num_key_value_heads) == (16 // tp, 16 // tp)
            assert (attn.o_proj.all_reduce is None) == (tp == 1)
            assert (mlp.down_proj.all_reduce is None) == (tp == 1)
        assert torch.equal(drafter.fc.weight, weights["fc.weight"])
        assert torch.equal(drafter.model.norm.weight, weights["norm.weight"])
        assert torch.equal(drafter.markov_w1, weights["markov_head.markov_w1.weight"])


def test_drafter_load_requires_every_layer_tensor(monkeypatch):
    monkeypatch.setattr(target_drafter._block_attn_op, "unavailability_reason", lambda: None)
    drafter = target_drafter.K3DSparkDrafter(_draft_config(heads=4, kv_heads=1, inter=64))
    with pytest.raises(ValueError, match="missing"):
        drafter.load_weights({"norm.weight": torch.ones(256)})


@pytest.mark.parametrize("requested,trtllm", [("VANILLA", True), ("AUTO", False)])
def test_dspark_gqa_drafter_on_vanilla(dspark, monkeypatch, requested, trtllm):
    """VANILLA as asked, or AUTO where TRTLLM-Gen is unavailable (the built-in fallback): the drafter's block attention
    is flash-attn over the worker's private arena, which the worker pages only for TRTLLM."""
    loaded = []
    monkeypatch.setattr(target_drafter._dense_attn_op, "flash_attention", lambda: loaded.append(1))
    if not trtllm:
        monkeypatch.setattr(
            target_drafter._block_attn_op, "unavailability_reason", lambda: "no TRTLLM-Gen here"
        )
    dspark.spec_config.attention_backend = requested
    shell = _shell(dspark.spec_config)
    drafter = shell.draft_model
    assert type(drafter) is target_drafter.K3DSparkDrafter
    assert drafter.dflash_attention_backend == "VANILLA" and loaded == [1]
    assert not drafter._paged_ctx_cache and drafter.validate_block_attention_windows() == []
    assert shell.spec_worker.calls == [("set_draft_model", drafter)]


def test_dspark_gqa_drafter_refuses_an_unavailable_explicit_trtllm(dspark, monkeypatch):
    monkeypatch.setattr(
        target_drafter._block_attn_op, "unavailability_reason", lambda: "no TRTLLM-Gen here"
    )
    dspark.spec_config.attention_backend = "TRTLLM"
    with pytest.raises(ValueError, match="TRTLLM"):
        _shell(dspark.spec_config)


def test_dspark_mla_drafter_on_vanilla(dspark, monkeypatch):
    """VANILLA: the eager MLA decode; the CUTEDSL kernel is not probed."""
    _use_mla_drafter(dspark, monkeypatch)
    monkeypatch.setattr(
        target_drafter_mla._mla_ops, "mla_decode_unavailability_reason", lambda: "no CuTe DSL here"
    )
    dspark.spec_config.attention_backend = "VANILLA"
    shell = _shell(dspark.spec_config)
    drafter = shell.draft_model
    assert type(drafter) is target_drafter_mla.K3MLADSparkDrafter
    assert drafter.dflash_attention_backend == "VANILLA"
    assert drafter._block_decode_variant(True) == "eager"
    assert drafter._paged_ctx_cache and not drafter._uses_worker_attention_backend


@pytest.mark.parametrize(
    "requested,error",
    [
        ("AUTO", "has no usable attention backend: its default 'CUTEDSL' is unavailable"),
        ("CUTEDSL", "attention_backend='CUTEDSL' was requested but it is unavailable"),
    ],
)
def test_dspark_mla_drafter_without_cutedsl_fails_like_the_builtin(
    dspark, monkeypatch, requested, error
):
    _use_mla_drafter(dspark, monkeypatch)
    monkeypatch.setattr(
        target_drafter_mla._mla_ops, "mla_decode_unavailability_reason", lambda: "no CuTe DSL here"
    )
    dspark.spec_config.attention_backend = requested
    with pytest.raises(ValueError, match=error):
        _shell(dspark.spec_config)


def test_dspark_mla_drafter_on_trtllm(dspark, monkeypatch):
    """TRTLLM: flashinfer's TRTLLM-Gen MLA decode with the upper-triangle fixup."""
    _use_mla_drafter(dspark, monkeypatch)
    monkeypatch.setattr(
        target_drafter_mla._mla_ops, "mla_paged_decode_unavailability_reason", lambda: None
    )
    monkeypatch.setattr(target_drafter_mla._mla_ops, "mla_paged_decode_kernel", lambda: object())
    dspark.spec_config.attention_backend = "TRTLLM"
    shell = _shell(dspark.spec_config)
    drafter = shell.draft_model
    assert type(drafter) is target_drafter_mla.K3MLADSparkDrafter
    assert drafter.dflash_attention_backend == "TRTLLM"
    assert drafter._block_decode_variant(True) == "trtllm_gen"
    assert drafter._block_decode_variant(False) == "eager"


def test_dspark_mla_drafter_without_flashinfer_mla_fails_like_the_builtin(dspark, monkeypatch):
    _use_mla_drafter(dspark, monkeypatch)
    monkeypatch.setattr(
        target_drafter_mla._mla_ops,
        "mla_paged_decode_unavailability_reason",
        lambda: "no flashinfer MLA here",
    )
    dspark.spec_config.attention_backend = "TRTLLM"
    with pytest.raises(
        ValueError, match="attention_backend='TRTLLM' was requested but it is unavailable"
    ):
        _shell(dspark.spec_config)
