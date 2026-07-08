# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level configuration tests for Qwen-Image."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.utils import gelu_tanh

# Importing the models package applies the Qwen-Image registration side effect.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
    Attention2DAttention,
    RingAttention,
    UlyssesAttention,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenImagePipeline,
    QwenImageTransformerBlock,
    QwenJointAttention,
    apply_rotary_emb_qwen,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
    FeedForward,
    QwenImageTransformer2DModel,
    _build_joint_attention_mask,
    _get_feedforward_activation,
    _is_qwen_sequence_parallel_attention,
    _supports_qwen_key_padding_mask,
    qwen_complex_freqs_to_cos_sin,
    qwen_joint_freqs_to_cos_sin,
)
from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode, apply_rotary_emb
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CudaGraphConfig,
    ParallelConfig,
    TorchCompileConfig,
    VisualGenArgs,
)


def _write_minimal_qwen_checkpoint(tmp_path):
    """Create the minimum diffusers layout needed by PipelineLoader config code."""
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImagePipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            }
        )
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "QwenImageTransformer2DModel"})
    )
    return tmp_path


def test_qwen_pipeline_config_defaults_to_empty_dict(tmp_path):
    checkpoint_dir = _write_minimal_qwen_checkpoint(tmp_path)

    args = VisualGenArgs(model=str(checkpoint_dir))
    resolved = PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))

    assert resolved == {}


def test_qwen_pipeline_config_rejects_unknown_keys(tmp_path):
    checkpoint_dir = _write_minimal_qwen_checkpoint(tmp_path)

    args = VisualGenArgs(
        model=str(checkpoint_dir),
        pipeline_config={"text_encoder_path": "/tmp/not-a-qwen-knob"},
    )
    with pytest.raises(ValueError, match="Unknown pipeline_config keys for QwenImagePipeline"):
        PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(
            VisualGenArgs(model="Qwen/Qwen-Image"),
            {
                "backend": "VANILLA",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 1,
            },
            id="default",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
                cuda_graph_config=CudaGraphConfig(enable=True),
            ),
            {
                "backend": "VANILLA",
                "compile": False,
                "autotune": False,
                "cuda_graph": True,
                "n_workers": 1,
            },
            id="cuda-graph",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                attention_config=AttentionConfig(backend="FA4"),
                parallel_config=ParallelConfig(attn2d_size=(2, 2)),
            ),
            {
                "backend": "FA4",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 4,
            },
            id="attention2d",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                attention_config=AttentionConfig(backend="FA4"),
                parallel_config=ParallelConfig(ring_size=2),
            ),
            {
                "backend": "FA4",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 2,
            },
            id="ring",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                attention_config=AttentionConfig(backend="TRTLLM"),
            ),
            {
                "backend": "TRTLLM",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 1,
            },
            id="trtllm-backend",
        ),
    ],
)
def test_qwen_pipeline_feature_args(args, expected):
    assert args.attention_config.backend == expected["backend"]
    assert args.torch_compile_config.enable is expected["compile"]
    assert args.torch_compile_config.enable_autotune is expected["autotune"]
    assert args.cuda_graph_config.enable is expected["cuda_graph"]
    assert args.parallel_config.n_workers == expected["n_workers"]


@pytest.mark.parametrize(
    ("quant_config", "quant_algo", "group_size", "force_dynamic_quantization"),
    [
        pytest.param({}, None, 128, False, id="bf16"),
        pytest.param(
            {"quant_algo": "FP8", "dynamic": True},
            QuantAlgo.FP8,
            None,
            False,
            id="dynamic-fp8-tensor",
        ),
        pytest.param(
            {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            QuantAlgo.FP8_BLOCK_SCALES,
            128,
            False,
            id="dynamic-fp8-blockwise",
        ),
        pytest.param(
            {"quant_algo": "NVFP4", "dynamic": True},
            QuantAlgo.NVFP4,
            16,
            True,
            id="dynamic-fp4",
        ),
    ],
)
def test_qwen_pipeline_quant_config_parses_from_args(
    tmp_path, quant_config, quant_algo, group_size, force_dynamic_quantization
):
    checkpoint_dir = _write_minimal_qwen_checkpoint(tmp_path)
    args = VisualGenArgs(model=str(checkpoint_dir), quant_config=quant_config)

    config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_dir), args=args)

    assert config.quant_config.quant_algo == quant_algo
    assert config.quant_config.group_size == group_size
    assert config.dynamic_weight_quant is (quant_algo is not None)
    assert config.force_dynamic_quantization is force_dynamic_quantization


@pytest.mark.parametrize(
    ("visual_gen_mapping", "not_wrapped_as"),
    [
        pytest.param(
            SimpleNamespace(
                attn2d_row_size=2,
                attn2d_col_size=2,
                ring_size=1,
                ulysses_size=1,
            ),
            "Attention2DAttention",
            id="attention2d",
        ),
        pytest.param(
            SimpleNamespace(
                attn2d_row_size=1,
                attn2d_col_size=1,
                ring_size=2,
                ulysses_size=1,
            ),
            "RingAttention",
            id="ring",
        ),
    ],
)
def test_qwen_joint_attention_keeps_separate_qkv_path_unwrapped(visual_gen_mapping, not_wrapped_as):
    config = DiffusionModelConfig(
        attention=AttentionConfig(backend="VANILLA"),
        visual_gen_mapping=visual_gen_mapping,
    )

    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=config,
    )

    assert attention.qkv_mode == QKVMode.SEPARATE_QKV
    assert attention.attn.__class__.__name__ != not_wrapped_as


class _FakeTokenBatch:
    def __init__(self, attention_mask):
        self.attention_mask = attention_mask
        self.input_ids = torch.arange(attention_mask.numel()).view_as(attention_mask)

    def to(self, device):
        self.attention_mask = self.attention_mask.to(device)
        self.input_ids = self.input_ids.to(device)
        return self


class _FakeTokenizer:
    def __init__(self, attention_mask):
        self.attention_mask = attention_mask

    def __call__(self, *args, **kwargs):
        return _FakeTokenBatch(self.attention_mask.clone())


class _FakeTextEncoder:
    def __call__(self, input_ids, **kwargs):
        hidden = torch.arange(
            input_ids.numel() * 4,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(*input_ids.shape, 4)
        return SimpleNamespace(hidden_states=[hidden])


@pytest.mark.parametrize(
    ("attention_mask", "expect_none"),
    [
        pytest.param(torch.ones(2, 36, dtype=torch.long), True, id="all-valid"),
        pytest.param(
            torch.tensor([[1] * 36, [1] * 35 + [0]], dtype=torch.long),
            False,
            id="has-padding",
        ),
    ],
)
def test_qwen_encode_prompt_returns_none_for_all_valid_masks(attention_mask, expect_none):
    pipeline = QwenImagePipeline(SimpleNamespace(torch_dtype=torch.float32))
    pipeline.tokenizer = _FakeTokenizer(attention_mask)
    pipeline.text_encoder = _FakeTextEncoder()

    prompt_embeds, prompt_embeds_mask = pipeline._encode_prompt(
        ["one", "two"],
        torch.device("cpu"),
        max_sequence_length=8,
    )

    assert prompt_embeds.shape == (2, 2, 4)
    if expect_none:
        assert prompt_embeds_mask is None
    else:
        assert prompt_embeds_mask is not None
        assert prompt_embeds_mask.shape == (2, 2)
        assert not prompt_embeds_mask.bool().all()


def test_qwen_joint_attention_passes_padding_mask_to_backend(monkeypatch):
    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=DiffusionModelConfig(attention=AttentionConfig(backend="VANILLA")),
    )
    captured = {}

    def fake_attn_impl(q, k, v, **kwargs):
        captured["key_padding_mask"] = kwargs.get("key_padding_mask")
        return q.new_zeros(q.shape)

    monkeypatch.setattr(attention, "_attn_impl", fake_attn_impl)

    hidden_states = torch.randn(1, 4, 16)
    encoder_hidden_states = torch.randn(1, 3, 16)
    attention_mask = torch.tensor([[True, False, True, True, True, True, True]])

    attention(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
    )

    assert captured["key_padding_mask"] is attention_mask


def test_qwen_joint_attention_rejects_unsupported_masked_sequence_parallel(monkeypatch):
    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=DiffusionModelConfig(attention=AttentionConfig(backend="VANILLA")),
    )
    attention.attn_backend = "FA4"
    attention._supports_key_padding_mask = False
    attention._uses_sequence_parallel_attention = True
    monkeypatch.setattr(
        attention,
        "_prepare_qkv",
        lambda *args, **kwargs: (
            torch.empty(1, 7, 16),
            torch.empty(1, 7, 16),
            torch.empty(1, 7, 16),
        ),
    )

    with pytest.raises(NotImplementedError, match="Padded Qwen-Image prompts"):
        attention(
            hidden_states=torch.empty(1, 4, 16),
            encoder_hidden_states=torch.empty(1, 3, 16),
            attention_mask=torch.tensor([[True, False, True, True, True, True, True]]),
        )


def test_qwen_joint_attention_tp2_shards_both_streams():
    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=DiffusionModelConfig(
            mapping=Mapping(world_size=2, rank=0, tp_size=2),
            attention=AttentionConfig(backend="VANILLA"),
        ),
    )

    joint_q, joint_k, joint_v = attention._prepare_qkv_unfused(
        hidden_states=torch.randn(1, 4, 16),
        encoder_hidden_states=torch.randn(1, 3, 16),
        image_rotary_emb=None,
    )

    assert not attention.fuse_qk_norm_rope
    assert attention.add_q_proj.tp_mode == TensorParallelMode.COLUMN
    assert attention.add_k_proj.tp_mode == TensorParallelMode.COLUMN
    assert attention.add_v_proj.tp_mode == TensorParallelMode.COLUMN
    assert attention.to_add_out.tp_mode == TensorParallelMode.ROW
    assert joint_q.shape == (1, 7, 8)
    assert joint_k.shape == (1, 7, 8)
    assert joint_v.shape == (1, 7, 8)


def test_qwen_complex_freqs_convert_to_shared_rope_format():
    torch.manual_seed(0)
    seq_len = 8
    head_dim = 16
    x = torch.randn(2, seq_len, 3, head_dim)
    phases = torch.randn(seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(phases), phases)

    freqs_cos, freqs_sin = qwen_complex_freqs_to_cos_sin(freqs_cis)

    ref = apply_rotary_emb_qwen(x, freqs_cis)
    out = apply_rotary_emb(x, freqs_cos, freqs_sin)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_qwen_joint_attention_fused_rope_passes_2d_freqs_to_kernel(monkeypatch):
    torch.manual_seed(0)
    txt_seq = 5
    img_seq = 7
    batch_size = 2
    head_dim = 8
    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=head_dim,
        config=DiffusionModelConfig(),
    )
    captured = {}

    def fake_apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin, **kwargs):
        captured["cos_shape"] = tuple(freqs_cos.shape)
        captured["sin_shape"] = tuple(freqs_sin.shape)

    monkeypatch.setattr(attention, "apply_packed_qk_norm_rope", fake_apply_packed_qk_norm_rope)

    hidden_states = torch.randn(batch_size, img_seq, 16)
    encoder_hidden_states = torch.randn(batch_size, txt_seq, 16)
    img_phases = torch.randn(img_seq, head_dim // 2)
    txt_phases = torch.randn(txt_seq, head_dim // 2)
    image_rotary_emb = (
        torch.polar(torch.ones_like(img_phases), img_phases),
        torch.polar(torch.ones_like(txt_phases), txt_phases),
    )

    attention._prepare_qkv_fused(hidden_states, encoder_hidden_states, image_rotary_emb)

    assert captured == {
        "cos_shape": (batch_size * (txt_seq + img_seq), head_dim),
        "sin_shape": (batch_size * (txt_seq + img_seq), head_dim),
    }


def test_qwen_joint_attention_reuses_precomputed_fused_rope(monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        transformer_qwen_image as qwen_transformer,
    )

    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=DiffusionModelConfig(),
    )
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 5, 16)
    image_rotary_emb = (
        torch.polar(torch.ones(7, 4), torch.randn(7, 4)),
        torch.polar(torch.ones(5, 4), torch.randn(5, 4)),
    )
    fused_rotary_emb = qwen_joint_freqs_to_cos_sin(image_rotary_emb, batch_size=2)
    captured = {}

    def fake_apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin, **kwargs):
        captured["freqs_cos"] = freqs_cos
        captured["freqs_sin"] = freqs_sin

    def fail_recompute(*args, **kwargs):
        raise AssertionError("precomputed RoPE should bypass complex-to-real conversion")

    monkeypatch.setattr(attention, "apply_packed_qk_norm_rope", fake_apply_packed_qk_norm_rope)
    monkeypatch.setattr(qwen_transformer, "qwen_joint_freqs_to_cos_sin", fail_recompute)

    attention._prepare_qkv_fused(
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,
        fused_rotary_emb,
    )

    assert captured["freqs_cos"] is fused_rotary_emb[0]
    assert captured["freqs_sin"] is fused_rotary_emb[1]


def test_qwen_joint_attention_fused_rope_requires_qk_norm():
    attention = QwenJointAttention(
        dim=128,
        num_attention_heads=2,
        attention_head_dim=64,
        config=DiffusionModelConfig(),
    )
    hidden_states = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    image_rotary_emb = (object(), object())

    assert attention._use_fused_qk_norm_rope(hidden_states, image_rotary_emb)

    attention.qk_norm = False
    assert not attention._use_fused_qk_norm_rope(hidden_states, image_rotary_emb)


def test_qwen_key_padding_mask_support_matrix():
    vanilla = object()
    attention_2d = Attention2DAttention.__new__(Attention2DAttention)
    ring = RingAttention.__new__(RingAttention)
    ulysses = UlyssesAttention.__new__(UlyssesAttention)

    assert _supports_qwen_key_padding_mask("VANILLA", vanilla)
    assert _supports_qwen_key_padding_mask("VANILLA", ulysses)
    # Qwen's joint mask is [valid text | padded text | valid image], while
    # FA4's key_padding_mask path assumes valid tokens form one prefix.
    assert not _supports_qwen_key_padding_mask("FA4", vanilla)
    assert not _supports_qwen_key_padding_mask("FA4", ulysses)
    assert not _supports_qwen_key_padding_mask("TRTLLM", vanilla)
    assert not _supports_qwen_key_padding_mask("VANILLA", attention_2d)
    assert not _supports_qwen_key_padding_mask("VANILLA", ring)
    assert _is_qwen_sequence_parallel_attention(attention_2d)
    assert _is_qwen_sequence_parallel_attention(ring)
    assert _is_qwen_sequence_parallel_attention(ulysses)
    assert not _is_qwen_sequence_parallel_attention(vanilla)


def test_qwen_build_joint_attention_mask_appends_valid_image_tokens():
    hidden_states = torch.empty(2, 4, 8)
    text_mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.int64)

    joint_mask = _build_joint_attention_mask(text_mask, hidden_states)

    expected = torch.tensor(
        [
            [True, False, True, True, True, True, True],
            [True, True, False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(joint_mask, expected)


def test_qwen_build_joint_attention_mask_none_stays_none():
    assert _build_joint_attention_mask(None, torch.empty(2, 4, 8)) is None


def test_qwen_sequence_sharding_pads_streams_and_interleaves_rank_masks():
    class RankZeroSharder:
        is_active = True
        size = 2

        def __init__(self, rank_one_mask):
            self.rank_one_mask = rank_one_mask

        def shard(self, tensor, dim=1, pad_to_multiple=False):
            if pad_to_multiple and tensor.shape[dim] % self.size:
                pad_shape = list(tensor.shape)
                pad_shape[dim] = self.size - tensor.shape[dim] % self.size
                tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)
            return tensor.narrow(dim, 0, tensor.shape[dim] // self.size).contiguous()

        def gather(self, tensor, dim=1, unpad_to=None):
            assert unpad_to is None
            return torch.cat([tensor, self.rank_one_mask], dim=dim)

    model = QwenImageTransformer2DModel(
        model_config=DiffusionModelConfig(),
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
    )
    rank_one_mask = torch.tensor([[True, False, False, True, True, True, False]])
    model.sharder = RankZeroSharder(rank_one_mask)
    hidden_states = torch.randn(1, 7, 16)
    encoder_hidden_states = torch.randn(1, 5, 16)
    encoder_mask = torch.tensor([[True, False, True, True, False]])
    image_rotary_emb = (
        torch.polar(torch.ones(7, 4), torch.randn(7, 4)),
        torch.polar(torch.ones(5, 4), torch.randn(5, 4)),
    )

    hidden_shard, text_shard, rope_shard, joint_mask, image_seq_len = model._shard_sequences(
        hidden_states,
        encoder_hidden_states,
        encoder_mask,
        image_rotary_emb,
    )

    assert hidden_shard.shape == (1, 4, 16)
    assert text_shard.shape == (1, 3, 16)
    assert rope_shard[0].shape == (4, 4)
    assert rope_shard[1].shape == (3, 4)
    assert image_seq_len == 7
    torch.testing.assert_close(
        joint_mask,
        torch.tensor(
            [
                [
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                    False,
                ]
            ]
        ),
    )


def test_qwen_sequence_sharding_rejects_padding_without_key_mask_support():
    class CachedBlocksWrapper(torch.nn.Module):
        def __init__(self, transformer_blocks):
            super().__init__()
            self.transformer_blocks = transformer_blocks

    model = QwenImageTransformer2DModel(
        model_config=DiffusionModelConfig(),
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
    )
    model.sharder = SimpleNamespace(is_active=True, size=2)
    first_block_attn = model.transformer_blocks[0].attn
    first_block_attn._supports_key_padding_mask = False
    model.transformer_blocks = torch.nn.ModuleList([CachedBlocksWrapper(model.transformer_blocks)])

    assert model._first_block_attn() is first_block_attn

    with pytest.raises(NotImplementedError, match="requires VANILLA Ulysses"):
        model._shard_sequences(
            torch.randn(1, 7, 16),
            torch.randn(1, 4, 16),
            None,
            (
                torch.polar(torch.ones(7, 4), torch.randn(7, 4)),
                torch.polar(torch.ones(4, 4), torch.randn(4, 4)),
            ),
        )


def test_qwen_joint_attention_cpu_fallback_uses_unfused_qk_norm_rope():
    torch.manual_seed(0)
    attention = (
        QwenJointAttention(
            dim=16,
            attention_head_dim=8,
            num_attention_heads=2,
        )
        .to(torch.bfloat16)
        .eval()
    )
    hidden_states = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    img_phases = torch.randn(4, 4)
    txt_phases = torch.randn(5, 4)
    image_rotary_emb = (
        torch.polar(torch.ones_like(img_phases), img_phases),
        torch.polar(torch.ones_like(txt_phases), txt_phases),
    )

    assert attention.fuse_qk_norm_rope
    assert not attention._use_fused_qk_norm_rope(hidden_states, image_rotary_emb)
    q, k, v = attention._prepare_qkv(
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,
    )

    assert q.shape == k.shape == v.shape == (1, 9, 16)


def test_qwen_transformer_block_modulation_helpers():
    x = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])
    mod_params = torch.tensor([[0.5, -1.0, 2.0, -0.5, 0.25, 0.75]])

    modulated, gate = QwenImageTransformerBlock._modulate(x, mod_params)

    expected_modulated = x * torch.tensor([[[3.0, 0.5]]]) + torch.tensor([[[0.5, -1.0]]])
    expected_gate = torch.tensor([[[0.25, 0.75]]])
    torch.testing.assert_close(modulated, expected_modulated)
    torch.testing.assert_close(gate, expected_gate)


def test_qwen_transformer_block_gate_residual_helper():
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gate = torch.tensor([[[0.25, 0.5]]])
    residual = torch.tensor([[[8.0, 6.0], [4.0, 2.0]]])

    output = QwenImageTransformerBlock._apply_gate_residual(hidden_states, gate, residual)

    torch.testing.assert_close(output, hidden_states + gate * residual)


def test_qwen_feedforward_uses_shared_gelu_tanh():
    assert _get_feedforward_activation("gelu-approximate") is gelu_tanh
    assert _get_feedforward_activation("gelu") is F.gelu

    with pytest.raises(ValueError, match="Unsupported activation_fn=relu"):
        _get_feedforward_activation("relu")


def test_qwen_feedforward_tp2_reduces_row_parallel_output():
    feedforward = FeedForward(
        dim=16,
        config=DiffusionModelConfig(mapping=Mapping(world_size=2, rank=0, tp_size=2)),
    )

    assert feedforward.down_proj.tp_mode == TensorParallelMode.ROW
    assert feedforward.down_proj.reduce_output
