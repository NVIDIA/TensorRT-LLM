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
"""Tests for the DeCo-Attention CuTe DSL backend of VisualGen.

``require_deco_kernels`` (conftest.py) skips a test on a GPU without a kernel for its V format
and fails it when the kernel package does not import.
"""

import importlib
import itertools
import multiprocessing
from typing import get_args

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.backends import (
    CuTeDSLAttention,
    create_attention,
    get_visual_gen_attention_backend,
)
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import DeCoAttention
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import deco as deco_backend
from tensorrt_llm._torch.visual_gen.attention.backends.utils import (
    get_padding_support,
    get_quant_algorithm,
)
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.deco_attention.runtime import num_sms
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    BaseQuantAttentionConfig,
    DeCoAttentionConfig,
    QuantAttentionConfig,
)

HEAD_DIM = 128
V_DTYPES = ("fp8", "nvfp4")
# Sequence lengths that no other forward of the process uses, one per request.
_UNSEEN_LENGTHS = itertools.count(2001)
MIN_SEQ_LEN = deco_backend.MIN_SEQ_LEN


@pytest.fixture
def kernel_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """The shapes of q in the calls of the kernels' ``prepare`` during the test."""
    calls: list = []
    prepare = deco_backend.deco_attention.prepare

    def counting(q, *args, **kwargs):
        calls.append(tuple(q.shape))
        return prepare(q, *args, **kwargs)

    monkeypatch.setattr(deco_backend.deco_attention, "prepare", counting)
    return calls


@pytest.fixture
def kernel_inputs(monkeypatch: pytest.MonkeyPatch) -> list:
    """The ``(rows, valid_seq_lens)`` of the calls of the kernels' ``prepare`` during the test,
    newest last."""
    calls: list = []
    prepare = deco_backend.deco_attention.prepare

    def recording(*args, **kwargs):
        calls.append((kwargs.get("rows"), kwargs.get("valid_seq_lens")))
        return prepare(*args, **kwargs)

    monkeypatch.setattr(deco_backend.deco_attention, "prepare", recording)
    return calls


@pytest.fixture(autouse=True)
def _kernels_serve_short_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests run the kernels on short sequences; ``test_deco_delegates_short_calls`` puts
    the backend's limit back."""
    monkeypatch.setattr(deco_backend, "MIN_SEQ_LEN", 1)


def _backend(num_heads: int, recipe: DeCoAttentionConfig) -> DeCoAttention:
    """A DeCo layer from the factory."""
    attn = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        attention_config=AttentionConfig(backend="CUTEDSL", quant_attention_config=recipe),
    )
    assert isinstance(attn, DeCoAttention)
    return attn


def _random_qkv(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    device: torch.device,
    seed: int = 0,
    q_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal Q and K, V with a per-channel offset, as [B, S, H, D] bfloat16.

    The scores have standard deviation ``q_scale``: 1 gives nearly uniform attention weights,
    4 gives rows that a few keys dominate.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (batch_size, seq_len, num_heads, HEAD_DIM)
    q = torch.randn(shape, generator=gen, device=device) * q_scale
    k = torch.randn(shape, generator=gen, device=device)
    v = torch.randn(shape, generator=gen, device=device)
    v = v + torch.randn((1, 1, num_heads, HEAD_DIM), generator=gen, device=device)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    """FP32 SDPA reference on [B, S, H, D] tensors."""
    out = F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2), **kwargs
    )
    return out.transpose(1, 2)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm())


def _kernels(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, v_dtype: str = "fp8", **kwargs
) -> torch.Tensor:
    """The output of the kernel package on a workspace of its own; ``kwargs`` go to
    ``prepare``, and a ``rank`` or ``bases`` among them size the workspace as well."""
    batch_size, seq_len, num_heads, _ = q.shape
    num_bytes = deco_attention.workspace_size(
        batch_size,
        seq_len,
        num_heads,
        q.device,
        v_dtype,
        kwargs.get("rank"),
        bases=kwargs.get("bases", "shared"),
    )
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=q.device)
    operands = deco_attention.prepare(q, k, v, workspace, v_dtype, **kwargs)
    return deco_attention.attention(operands, torch.empty_like(q))


def _assert_follows_the_scores(
    out: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    """The error of ``out`` against FP32 SDPA stays below half the error of an attention that
    ignores Q and K (uniform weights) on rows that a few keys dominate."""
    assert torch.isfinite(out).all()
    ref = _sdpa(q, k, v)
    uniform_weights = v.float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(out, ref) < max(0.5 * _relative_error(uniform_weights, ref), 0.02)


def _split_plan(kernel: str):
    """Return ``split_plan`` of a kernel package of the GPU's family, such as ``tcgen05.fp8pv``."""
    return importlib.import_module(f"{deco_attention.__name__}.{kernel}.launch").split_plan


def _multi_launch_call(device: torch.device, v_dtype: str = "fp8") -> tuple[int, int, int, int]:
    """Return (batch, rows, heads, valid rows) of a call sized from the SM count of the device.

    Its plan, asserted from ``split_plan`` of the kernels:

    * tcgen05: two waves of whole units in the main launch and a tail, a last partial wave. A
      GPU whose kernels have the split launch (``kv_split`` of their tuning) runs the tail
      there, in pieces of the key sequence; the others run it in the main launch. The valid
      rows end in the first tail unit, so the tail has units without a valid query and, where
      it is split, pieces without a valid key.
    * SM120: whole units and a split tail in the attention launch, and more (batch, head) pairs
      than a quarter of the SMs, so the V pass of the preparation is a launch of its own. The
      valid rows end in the first key half of the sequence.
    """
    sms = num_sms(device)
    if deco_attention.target.kernel_family() == "tcgen05":
        seq_len, num_heads = 256 * (sms // 4 + 1), 8
        plan = _split_plan("tcgen05.fp8pv")(seq_len, num_heads, 1, sms)
        units, main_units, pieces = plan
        tail_units = units % sms
        assert units >= 2 * sms and tail_units > 0, (plan, sms)
        tuning = importlib.import_module(f"{deco_attention.__name__}.tcgen05.fp8pv.tuning")
        if tuning.tuning().kv_split:
            assert main_units == units - tail_units and pieces > 1, (plan, sms)
        else:
            assert main_units == units and pieces == 1, (plan, sms)
        return 1, seq_len, num_heads, seq_len - 256 * tail_units + 100
    seq_len, num_heads = 3072, sms // 4 + 1
    tiles = seq_len // 128
    if v_dtype == "fp8":
        plan = _split_plan("sm120.fp8pv")(tiles * num_heads, tiles, sms)
    else:
        plan = _split_plan("sm120.nvfp4pv")(1, seq_len, num_heads, sms)
    whole_units, split_units, _ = plan
    assert whole_units > 0 and split_units > 0 and 4 * num_heads > sms, (plan, sms)
    return 1, seq_len, num_heads, seq_len // 2 - 156


def test_deco_selection(deco_recipe) -> None:
    """CUTEDSL keeps its dense backend unless the quantization config is that of
    DeCo-Attention, which takes the padding of a call as a key padding mask with the padding of
    every sample behind its tokens."""
    plain = QuantAttentionConfig(qk_dtype="nvfp4")
    assert get_quant_algorithm(None) is None and get_quant_algorithm(plain) is None
    assert get_quant_algorithm(deco_recipe()) == "deco_attention"
    assert get_quant_algorithm(QuantAttentionConfig(vc_attention=True)) == "vc_attention"
    assert get_visual_gen_attention_backend("CUTEDSL") is CuTeDSLAttention
    assert (
        get_visual_gen_attention_backend("CUTEDSL", quant_algorithm="deco_attention")
        is DeCoAttention
    )
    # The algorithm selects a class on CUTEDSL only, as the config validator binds it, and
    # VC-Attention runs in the class of its backend.
    assert get_visual_gen_attention_backend(
        "VANILLA", quant_algorithm="deco_attention"
    ) is get_visual_gen_attention_backend("VANILLA")
    assert get_visual_gen_attention_backend(
        "TRTLLM", quant_algorithm="vc_attention"
    ) is get_visual_gen_attention_backend("TRTLLM")
    assert DeCoAttention.support_key_padding_mask()
    assert DeCoAttention.requires_trailing_padding()
    assert not DeCoAttention.support_valid_seq_len()

    # (supports_mask, supports_length, requires_trailing): a quantized FLASHINFER recipe rejects
    # a mask and a quantized TRTLLM recipe the valid prefix, which their classes alone do not say.
    # The configs skip the recipe validator, which binds a recipe to the GPU generation.
    def config(backend: str, quant_attention_config: BaseQuantAttentionConfig | None = None):
        return AttentionConfig.model_construct(
            backend=backend, quant_attention_config=quant_attention_config
        )

    quantized = QuantAttentionConfig(qk_dtype="nvfp4", v_dtype="fp8")
    assert get_padding_support(config("FLASHINFER")) == (True, False, False)
    assert get_padding_support(config("FLASHINFER", quantized)) == (False, False, False)
    assert get_padding_support(config("TRTLLM")) == (False, True, True)
    assert get_padding_support(config("TRTLLM", quantized)) == (False, False, False)
    assert get_padding_support(config("CUTEDSL", deco_recipe())) == (True, False, True)


def test_deco_cross_attention_layers_run_on_vanilla(require_deco_kernels, deco_recipe) -> None:
    """The Attention module builds a cross-attention layer on VANILLA under a DeCo-Attention
    recipe, with separate or fused K and V projections, and a self-attention layer on the DeCo
    backend: the backend reads q, k and v of one shape as one sequence, which a cross-attention
    layer with as many queries as keys is not."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode
    from tensorrt_llm._torch.visual_gen.attention.backends import VanillaAttention
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

    require_deco_kernels()
    num_heads, hidden_size = 2, 2 * HEAD_DIM
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=hidden_size, num_attention_heads=num_heads, attention_head_dim=HEAD_DIM
        ),
        attention=AttentionConfig(backend="CUTEDSL", quant_attention_config=deco_recipe()),
        skip_create_weights_in_init=True,
    )

    for cross in (
        Attention(hidden_size, num_heads, qkv_mode=QKVMode.SEPARATE_QKV, config=config),
        Attention(hidden_size, num_heads, qkv_mode=QKVMode.FUSE_KV, config=config),
    ):
        assert type(cross.attn) is VanillaAttention
    for self_attention in (
        Attention(hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=config),
        Attention(
            hidden_size,
            num_heads,
            qkv_mode=QKVMode.SEPARATE_QKV,
            separate_qkv_is_self_attention=True,
            config=config,
        ),
        Attention(
            hidden_size,
            num_heads,
            qkv_mode=QKVMode.FUSE_KV,
            separate_qkv_is_self_attention=True,
            config=config,
        ),
    ):
        assert type(self_attention.attn) is DeCoAttention


def test_deco_layer_construction(require_deco_kernels, deco_recipe, kernel_calls: list) -> None:
    """A layer needs a DeCo-Attention config with a kernel on this GPU (NVFP4 PV: SM120 only)
    and no more compensation channels than its kernels offer. A layer whose head dimension or
    K/V head count the kernels do not take is built and runs all its calls on VANILLA."""
    device = require_deco_kernels()
    args = dict(num_heads=4, head_dim=HEAD_DIM, quant_attention_config=deco_recipe())
    with pytest.raises(ValueError, match="needs a DeCoAttentionConfig"):
        DeCoAttention(**{**args, "quant_attention_config": QuantAttentionConfig(qk_dtype="nvfp4")})
    if "nvfp4" not in deco_backend.supported_v_dtypes():
        with pytest.raises(ValueError, match="no kernel for v_dtype='nvfp4'"):
            DeCoAttention(**{**args, "quant_attention_config": deco_recipe("nvfp4")})
    # A term the config leaves open takes half of the channels of the GPU, which is too much
    # next to a term that takes them all.
    max_channels = deco_attention.max_comp_channels()
    assert DeCoAttention(**args)._rank == (max_channels // 2, max_channels // 2)
    with pytest.raises(ValueError, match=f"has {max_channels} compensation channels"):
        DeCoAttention(**{**args, "quant_attention_config": deco_recipe(q_channels=max_channels)})

    for geometry in ({"head_dim": 64}, {"num_kv_heads": 2}):
        attn = DeCoAttention(**{**args, **geometry})
        head_dim, num_kv_heads = attn.head_dim, attn.num_kv_heads

        gen = torch.Generator(device=device).manual_seed(0)
        q = torch.randn((1, 2048, 4, head_dim), generator=gen, device=device).bfloat16()
        k = torch.randn((1, 2048, num_kv_heads, head_dim), generator=gen, device=device).bfloat16()
        v = torch.randn((1, 2048, num_kv_heads, head_dim), generator=gen, device=device).bfloat16()
        out = attn.forward(q, k, v)

        ref = _sdpa(q, k, v, enable_gqa=num_kv_heads != 4)
        assert out.shape == q.shape and _relative_error(out, ref) < 2e-2
        assert not kernel_calls


@pytest.mark.parametrize("v_dtype", V_DTYPES)
@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads",
    [(1, 1024, 4), (2, 333, 3), (1, 1, 1)],
    ids=["tiles", "partial_tile", "one_token"],
)
def test_deco_forward_matches_sdpa(
    require_deco_kernels,
    deco_recipe,
    kernel_calls: list,
    v_dtype: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
) -> None:
    """The output follows the scores: its error against FP32 SDPA stays below half the error of
    an attention that ignores Q and K (uniform weights) on rows that a few keys dominate."""
    device = require_deco_kernels(v_dtype)
    if seq_len < deco_attention.min_seq_len(v_dtype):
        pytest.skip(f"The {v_dtype} PV kernel of this GPU takes at least one key tile.")
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    attn = _backend(num_heads, deco_recipe(v_dtype))

    out = attn.forward(q, k, v)

    assert kernel_calls == [tuple(q.shape)]
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    _assert_follows_the_scores(out, q, k, v)
    # A repeated call returns the same bits.
    assert torch.equal(attn.forward(q, k, v), out)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
@pytest.mark.parametrize("with_lengths", [False, True], ids=["full", "lengths"])
def test_deco_whole_units_and_split_tail_follow_the_scores(
    require_deco_kernels, v_dtype: str, with_lengths: bool
) -> None:
    """A call whose plan has whole units and a tail, split where the kernels of the GPU split it
    (``_multi_launch_call``), follows the scores, with and without valid lengths, and a second
    call returns the same bits."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads, valid_seq_len = _multi_launch_call(device, v_dtype)
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    lens = None
    if with_lengths:
        lens = torch.tensor([valid_seq_len], dtype=torch.int32, device=device)
    else:
        valid_seq_len = seq_len

    out = _kernels(q, k, v, v_dtype, valid_seq_lens=lens)

    assert not out[:, valid_seq_len:].any()
    tokens = slice(valid_seq_len)
    _assert_follows_the_scores(out[:, tokens], q[:, tokens], k[:, tokens], v[:, tokens])
    assert torch.equal(_kernels(q, k, v, v_dtype, valid_seq_lens=lens), out)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_compensation_follows_the_shared_directions(
    require_deco_kernels, v_dtype: str
) -> None:
    """Q and K whose energy sits in 32 directions that every token shares put the quantization
    error of the scores on those directions. The compensation channels, estimated on the sampled
    tokens, remove most of it; a basis estimated on one token cannot, and its error is larger.
    NVFP4 PV adds an error of its own to both outputs, so its two bounds are wider."""
    device = require_deco_kernels(v_dtype)
    num_directions = 32
    q, k, v = _random_qkv(1, 4096, 2, device)
    gen = torch.Generator(device=device).manual_seed(2)
    directions = torch.randn((num_directions, HEAD_DIM), generator=gen, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True)
    shared = []
    for x in (q, k):
        weights = torch.randn((*x.shape[:3], num_directions), generator=gen, device=device)
        shared.append((0.1 * x.float() + 3.0 * weights @ directions).bfloat16())
    q, k = shared
    ref = _sdpa(q, k, v)

    error = _relative_error(_kernels(q, k, v, v_dtype), ref)
    one_token = torch.zeros(64, dtype=torch.int32, device=device)
    error_one_token = _relative_error(_kernels(q, k, v, v_dtype, rows=one_token), ref)

    max_error, max_share = {"fp8": (0.04, 0.5), "nvfp4": (0.07, 0.85)}[v_dtype]
    assert error < max_error
    assert error < max_share * error_one_token


@pytest.mark.parametrize("v_dtype", V_DTYPES)
@pytest.mark.parametrize(
    "fields, kernel_arguments",
    [
        (dict(q_channels=0, k_channels=48), dict(rank=(0, 48))),
        (dict(basis="separate"), dict(bases="separate")),
        (dict(gram_weight=1.0), dict(gram_weight=1.0)),
        (dict(q_channels=96, k_channels=32), dict(rank=(96, 32))),
    ],
    ids=["uneven_split", "separate_bases", "gram_weight", "128_channels"],
)
def test_deco_compensation_fields_follow_the_scores(
    require_deco_kernels, deco_recipe, v_dtype: str, fields: dict, kernel_arguments: dict
) -> None:
    """Each compensation field of the config reaches the kernels through the backend: an
    uneven split with a term of no channels and a zero block, separate bases, a Gram weight
    that takes Q alone, and 128 channels where the GPU offers them. The output follows the
    scores, is the output of the kernels under the same arguments, and differs from that of
    the default config."""
    device = require_deco_kernels(v_dtype)
    channels = fields.get("q_channels", 0) + fields.get("k_channels", 0)
    if channels > deco_attention.max_comp_channels():
        pytest.skip(f"{channels} compensation channels have no kernel on this GPU.")
    num_heads = 2
    q, k, v = _random_qkv(1, 1024, num_heads, device, q_scale=4.0)

    out = _backend(num_heads, deco_recipe(v_dtype, **fields)).forward(q, k, v)

    _assert_follows_the_scores(out, q, k, v)
    assert torch.equal(out, _kernels(q, k, v, v_dtype, **kernel_arguments))
    assert not torch.equal(out, _backend(num_heads, deco_recipe(v_dtype)).forward(q, k, v))


def test_deco_scale_rules_follow_the_scores(require_deco_kernels, deco_recipe) -> None:
    """The scale rule of the config reaches the kernels through the backend: each rule gives
    the output of the kernels under that rule, which follows the scores, and no two rules give
    the same output. The config names the rules of the kernel package and its default."""
    device = require_deco_kernels()
    rules = deco_attention.constants.SCALE_RULES
    field = DeCoAttentionConfig.model_fields["scale_rule"]
    assert get_args(field.annotation) == rules
    assert field.default == deco_attention.constants.DEFAULT_SCALE_RULE
    num_heads = 2
    q, k, v = _random_qkv(1, 1024, num_heads, device, q_scale=4.0)

    outs = []
    for rule in rules:
        out = _backend(num_heads, deco_recipe(scale_rule=rule)).forward(q, k, v)
        _assert_follows_the_scores(out, q, k, v)
        assert torch.equal(out, _kernels(q, k, v, scale_rule=rule))
        outs.append(out)

    assert not any(torch.equal(a, b) for a, b in itertools.combinations(outs, 2))


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_sample_rates_and_minimum_set_the_basis_rows(
    require_deco_kernels, deco_recipe, kernel_inputs: list, v_dtype: str
) -> None:
    """The sample rates and the minimum of sampled tokens of the config set the rows that reach
    the preparation: the kernels' own draw under the defaults, one set for equal rates and a
    pair (rows of Q, rows of K) for different ones, each of its rate of the sequence and at
    least the minimum. Under a mask every sample draws its rows from its own tokens, and both
    draws are kept for the next call that passes the same mask tensor. With token tags half of
    each draw comes from the text tokens of the sample."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 2, 4096, 2
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)

    _backend(num_heads, deco_recipe(v_dtype)).forward(q, k, v)
    assert kernel_inputs[-1][0] is None
    equal_rates = deco_recipe(v_dtype, q_sample_rate=1 / 32, k_sample_rate=1 / 32)
    _backend(num_heads, equal_rates).forward(q, k, v)
    assert kernel_inputs[-1][0].shape == (seq_len // 32,)

    num_q_rows, num_k_rows = 100, seq_len // 16
    attn = _backend(
        num_heads,
        deco_recipe(
            v_dtype, q_sample_rate=1 / 128, k_sample_rate=1 / 16, min_sample_rows=num_q_rows
        ),
    )
    out = attn.forward(q, k, v)
    rows_q, rows_k = kernel_inputs[-1][0]
    assert rows_q.shape == (num_q_rows,) and rows_k.shape == (num_k_rows,)
    _assert_follows_the_scores(out, q, k, v)

    lengths = torch.tensor([3000, 500], device=device)
    key_padding_mask = torch.arange(seq_len, device=device) < lengths[:, None]
    out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)
    (rows_q, rows_k), valid_seq_lens = kernel_inputs[-1]
    assert valid_seq_lens.tolist() == lengths.tolist()
    assert rows_q.shape == (batch_size, num_q_rows) and rows_k.shape == (batch_size, num_k_rows)
    assert (rows_q < lengths[:, None]).all() and (rows_k < lengths[:, None]).all()
    assert not out[~key_padding_mask].any() and torch.isfinite(out).all()
    attn.forward(q, k, v, key_padding_mask=key_padding_mask)
    assert kernel_inputs[-1][0][0] is rows_q and kernel_inputs[-1][0][1] is rows_k

    num_text = 200
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:num_text] = 1
    out = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask)
    for rows, num_rows in zip(kernel_inputs[-1][0], (num_q_rows, num_k_rows)):
        assert rows.shape == (batch_size, num_rows) and (rows < lengths[:, None]).all()
        assert ((rows < num_text).sum(dim=1) == num_rows // 2).all()
    assert not out[~key_padding_mask].any() and torch.isfinite(out).all()


def _spectrum_tokens(
    num_tokens: int, num_heads: int, device: torch.device, seed: int
) -> torch.Tensor:
    """Tokens ``[1, num_tokens, H, 128]`` of a chosen spectrum per head, as bfloat16.

    Token i is direction ``i % 128`` of a random orthonormal set of its head, with a random
    sign: 16 directions of decaying scale, then 112 directions of one scale. Whole rounds of
    the 128 directions give a Gram matrix with 16 distinct leading eigenvalues above a tail
    of nearly equal ones.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    directions = torch.linalg.qr(
        torch.randn((num_heads, HEAD_DIM, HEAD_DIM), generator=gen, device=device)
    ).Q
    scales = torch.full((HEAD_DIM,), 2.0**-3, device=device)
    scales[:16] = 2.0 ** (-torch.arange(16, device=device) / 8)
    index = torch.arange(num_tokens, device=device) % HEAD_DIM
    signs = torch.randint(0, 2, (num_tokens,), generator=gen, device=device) * 2 - 1
    tokens = (signs * scales[index])[:, None, None] * directions[:, index].transpose(0, 1)
    return tokens[None].bfloat16().contiguous()


def _basis_gaps(
    q: torch.Tensor,
    k: torch.Tensor,
    rows: torch.Tensor,
    basis: tuple[torch.Tensor, ...],
    terms: list[tuple[float, tuple[int, ...]]],
) -> tuple[float, float]:
    """Hold a basis of the preparation against ``torch.linalg.eigh`` on the same rows.

    * Reference: the float64 Gram matrices of the sampled tokens of Q and of K in the rotated
      frame, each over its trace, summed per basis tensor with its weight of Q.
    * terms: Per basis tensor, the weight of Q and the channels of the terms that take its
      leading directions, which are its last rows.

    Returns:
        The largest share of the trace that the leading directions of a term miss against as
        many leading eigenvectors, and the largest deviation of those directions from
        orthonormal rows.
    """
    rotation = deco_attention.inputs.hadamard(q.device).double()
    grams = []
    for x in (q, k):
        tokens = x[:, rows.long()].double().transpose(1, 2) @ rotation
        gram = tokens.transpose(2, 3) @ tokens
        grams.append(gram / gram.diagonal(dim1=2, dim2=3).sum(-1)[..., None, None])
    share_gap, deviation = 0.0, 0.0
    for directions, (q_weight, channels) in zip(basis, terms):
        gram = q_weight * grams[0] + (1.0 - q_weight) * grams[1]
        eigenvalues = torch.linalg.eigvalsh(gram)
        for num_channels in channels:
            leading = directions[:, :, -num_channels:].double()
            span = torch.linalg.qr(leading.transpose(2, 3)).Q
            share = (span.transpose(2, 3) @ gram @ span).diagonal(dim1=2, dim2=3).sum(-1)
            missed = eigenvalues[..., -num_channels:].sum(-1) - share
            share_gap = max(share_gap, float(missed.max()))
            eye = torch.eye(num_channels, dtype=torch.float64, device=q.device)
            off = leading @ leading.transpose(2, 3) - eye
            deviation = max(deviation, float(off.abs().max()))
    return share_gap, deviation


@pytest.mark.parametrize(
    "rank, bases",
    [(None, "shared"), ((16, 48), "shared"), (None, "separate"), ((96, 32), "shared")],
    ids=["default", "wide_term", "separate_bases", "128_channels"],
)
def test_deco_basis_spans_the_leading_eigenspace(
    require_deco_kernels, rank: tuple[int, int] | None, bases: str
) -> None:
    """The compensation basis of the preparation holds against ``torch.linalg.eigh`` on the
    same rows: the directions of every term are orthonormal and carry the share of the trace
    of as many leading eigenvectors. The rows give a Gram matrix whose nearly equal eigenvalues
    run across the last direction of a term, and then fewer distinct rows than directions."""
    device = require_deco_kernels()
    max_channels = deco_attention.max_comp_channels()
    r_q, r_k = rank or (max_channels // 2, max_channels // 2)
    if r_q + r_k > max_channels:
        pytest.skip(f"{r_q + r_k} compensation channels have no kernel on this GPU.")
    seq_len, num_heads = 2048, 3
    q, k = (_spectrum_tokens(seq_len, num_heads, device, seed) for seed in (0, 1))
    v = _random_qkv(1, seq_len, num_heads, device)[2]
    # Shared bases: one tensor of the equal-weight sum for both terms. Separate bases: the
    # Q-residual term on the Gram matrix of K, the K-residual term on that of Q.
    terms = [(0.0, (r_q,)), (1.0, (r_k,))] if bases == "separate" else [(0.5, (r_q, r_k))]
    num_bytes = deco_attention.workspace_size(
        1, seq_len, num_heads, device, "fp8", rank, bases=bases
    )
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    rounds = torch.arange(3 * HEAD_DIM, dtype=torch.int32, device=device)
    for rows in (rounds, rounds % 5):
        operands = deco_attention.prepare(q, k, v, workspace, "fp8", rank, rows=rows, bases=bases)
        basis = operands.basis if isinstance(operands.basis, tuple) else (operands.basis,)
        share_gap, deviation = _basis_gaps(q, k, rows, basis, terms)
        assert share_gap <= 1e-5 and deviation <= 1e-5, (share_gap, deviation)


def test_deco_large_caller_draw_follows_the_scores(require_deco_kernels) -> None:
    """A caller's draw of more than 512 rows reaches the later sampling chunks of the
    preparation: two draws that share their first 512 rows and differ in the 513th, an outlier
    row, give different bases."""
    device = require_deco_kernels()
    seq_len = 1024
    q, k, v = _random_qkv(1, seq_len, 2, device, q_scale=4.0)
    gen = torch.Generator(device=device).manual_seed(1)
    perm = torch.randperm(seq_len, generator=gen, device=device)
    rows = perm[:513].to(torch.int32)
    other = torch.cat([perm[:512], perm[513:514]]).to(torch.int32)
    q[:, perm[513]] *= 16  # the row only the second draw samples weighs on the basis
    out, out_other = (_kernels(q, k, v, rows=r) for r in (rows, other))
    _assert_follows_the_scores(out, q, k, v)
    assert not torch.equal(out, out_other)


def _run_on_two_streams(device: torch.device, shape: tuple[int, int, int]) -> None:
    """Run two calls of a shape on two streams, four rounds, and compare with the calls alone."""
    calls = [_random_qkv(*shape, device, seed=seed) for seed in (0, 1)]
    expected = [_kernels(q, k, v) for q, k, v in calls]
    num_bytes = deco_attention.workspace_size(*shape, device, "fp8")
    workspaces = [torch.empty(num_bytes, dtype=torch.uint8, device=device) for _ in calls]
    rounds = 4
    outs = [[torch.full_like(q, float("nan")) for q, _, _ in calls] for _ in range(rounds)]
    streams = [torch.cuda.Stream(device=device) for _ in calls]
    # The inputs and the NaN fills are on the current stream: both streams wait for them.
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    for round_outs in outs:
        for stream, (q, k, v), workspace, out in zip(streams, calls, workspaces, round_outs):
            with torch.cuda.stream(stream):
                deco_attention.attention(deco_attention.prepare(q, k, v, workspace, "fp8"), out)
    torch.cuda.synchronize(device)
    for round_outs in outs:
        for out, want in zip(round_outs, expected):
            assert torch.equal(out, want), f"{shape}: the output differs from the call alone"


def _two_stream_rounds(device_index: int, result) -> None:
    """Run the two-stream rounds of every shape and send True or the failure text."""
    try:
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
        shapes = [(1, 16384, 4)]
        if deco_attention.target.kernel_family() == "tcgen05":
            # The tail of this shape runs in the split launch, whose pieces merge, on a GPU
            # whose kernels have one.
            shapes.append(_multi_launch_call(device)[:3])
        for shape in shapes:
            _run_on_two_streams(device, shape)
        result.send(True)
    except Exception as error:  # noqa: BLE001 - the child reports to the test process
        result.send(f"{type(error).__name__}: {error}")


def test_deco_preparations_on_two_streams_finish(require_deco_kernels) -> None:
    """Two calls with workspaces of their own run their preparations side by side on two
    streams and come back as alone: no CTA of a launch waits for another CTA of its grid, so
    the launches need not be resident together. On the tcgen05 kernels a second shape adds a
    tail, which the split launch runs where the GPU has one: its pieces merge next to the
    launches of the other stream. The rounds run
    in a child process that a hang cannot leave stuck on the device: the test kills it on the
    timeout."""
    device = require_deco_kernels()
    context = multiprocessing.get_context("spawn")
    result, child_end = context.Pipe(duplex=False)
    child = context.Process(target=_two_stream_rounds, args=(device.index, child_end))
    child.start()
    child_end.close()
    # The kernels may compile in the child; the rounds themselves take under a second.
    finished = result.poll(600)
    outcome = result.recv() if finished else None
    child.join(timeout=30 if finished else 0)
    if child.is_alive():
        child.kill()
        child.join()
    assert finished, "the preparations on two streams did not finish"
    assert outcome is True, outcome
    assert child.exitcode == 0


def test_deco_kernel_api_host_checks(
    require_deco_kernels, deco_recipe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernels move q, k, v and out through TMA descriptors and vector stores that assume
    256-byte aligned base addresses: a view into an allocation at an offset of 128 elements is
    served, one at an offset of one or of eight elements is rejected.
    Beside the alignment, the host side resolves its devices: see the checks at the end."""
    device = require_deco_kernels()
    shape = (1, 256, 2, HEAD_DIM)
    q, k, v = _random_qkv(*shape[:3], device)
    expected = _kernels(q, k, v)
    backing = torch.empty(q.numel() + 128, dtype=torch.bfloat16, device=device)

    aligned = backing[128:].view(shape)
    aligned.copy_(q)
    assert torch.equal(_kernels(aligned, k, v), expected)
    out = backing[128:].view(shape)
    num_bytes = deco_attention.workspace_size(*shape[:3], device, "fp8")
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    assert torch.equal(
        deco_attention.attention(deco_attention.prepare(q, k, v, workspace, "fp8"), out), expected
    )

    attn = _backend(shape[2], deco_recipe())
    for offset in (1, 8):
        misaligned = backing[offset : offset + q.numel()].view(shape)
        assert misaligned.is_contiguous()
        with pytest.raises(ValueError, match="aligned"):
            deco_attention.prepare(misaligned, k, v, workspace, "fp8")
        with pytest.raises(ValueError, match="aligned"):
            deco_attention.attention(deco_attention.prepare(q, k, v, workspace, "fp8"), misaligned)
        # The backend copies such a view before the kernels.
        misaligned.copy_(q)
        assert torch.equal(attn.forward(misaligned, k, v), attn.forward(q, k, v))
    with pytest.raises(ValueError, match="empty dimension"):
        deco_attention.prepare(q[:, :0], k[:, :0], v[:, :0], workspace, "fp8")
    # rows and valid_seq_lens are read through descriptors that assume 16-byte addresses: a view at an offset
    # of one element is rejected, not misread.
    ints = torch.zeros(9, dtype=torch.int32, device=device)
    lens = ints[1:2].fill_(shape[1])
    rows = ints[1:9].copy_(torch.arange(8, dtype=torch.int32, device=device))
    with pytest.raises(ValueError, match="aligned"):
        deco_attention.prepare(q, k, v, workspace, "fp8", valid_seq_lens=lens)
    with pytest.raises(ValueError, match="aligned"):
        deco_attention.prepare(q, k, v, workspace, "fp8", rows=rows)

    # The rotation matrix is built on the CPU whatever the default device is, and the SM count of
    # "cuda" is that of the current device, not a value cached under the string.
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.deco_attention import runtime

    rotation = deco_attention.inputs.hadamard(device).clone()
    deco_attention.inputs.hadamard.cache_clear()
    torch.set_default_device(device)
    try:
        assert torch.equal(deco_attention.inputs.hadamard(device), rotation)
    finally:
        torch.set_default_device("cpu")
    assert runtime.num_sms("cuda") == runtime.num_sms(torch.device("cuda", device.index))
    counts = {0: 10, 1: 20}
    monkeypatch.setattr(runtime, "_num_sms", lambda index: counts[index])
    for index in counts:
        monkeypatch.setattr(torch.cuda, "current_device", lambda index=index: index)
        assert runtime.num_sms("cuda") == counts[index]


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_reads_the_slices_of_a_fused_projection(
    require_deco_kernels, deco_recipe, v_dtype: str
) -> None:
    """The preparation reads q, k and v in place from a fused QKV projection, whose slices are
    strided over the heads: the bits of contiguous copies, with and without valid lengths, and
    the backend hands such slices to the kernels without a copy. A slice of another layout is
    rejected by the kernel API, and the output must be contiguous."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 2, 1024, 3
    gen = torch.Generator(device=device).manual_seed(0)
    fused = torch.randn(
        (batch_size, seq_len, 3 * num_heads, 128), generator=gen, device=device
    ).bfloat16()
    q, k, v = (fused[:, :, i * num_heads : (i + 1) * num_heads] for i in range(3))
    assert not q.is_contiguous() and deco_attention.token_rows(q)
    lens = torch.tensor([1024, 700], dtype=torch.int32, device=device)
    for valid_seq_lens in (None, lens):
        expected = _kernels(
            q.contiguous(), k.contiguous(), v.contiguous(), v_dtype, valid_seq_lens=valid_seq_lens
        )
        assert torch.equal(_kernels(q, k, v, v_dtype, valid_seq_lens=valid_seq_lens), expected)
    attn = _backend(num_heads, deco_recipe(v_dtype))
    contiguous_out = attn.forward(q.contiguous(), k.contiguous(), v.contiguous())
    assert torch.equal(attn.forward(q, k, v), contiguous_out)
    num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, device, v_dtype)
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    channels_outer = torch.empty((batch_size, 128, num_heads, seq_len), device=device).bfloat16()
    with pytest.raises(ValueError, match="fused projection"):
        deco_attention.prepare(q, k, channels_outer.permute(0, 3, 2, 1), workspace, v_dtype)
    with pytest.raises(ValueError, match="contiguous"):
        deco_attention.attention(deco_attention.prepare(q, k, v, workspace, v_dtype), q)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_constant_v_comes_back_as_the_constant(require_deco_kernels, v_dtype: str) -> None:
    """A V of one large value in every row has block means of that value and deviations of
    zero; the smoothed V keeps its scaled means finite, and the output is the constant."""
    device = require_deco_kernels(v_dtype)
    q, k, _ = _random_qkv(1, 512, 2, device, q_scale=4.0)
    for value in (2.0**30, -(2.0**32)):
        v = torch.full_like(q, value)
        out = _kernels(q, k, v, v_dtype)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, v, rtol=2e-2, atol=0)


def test_deco_call_leaves_no_workspace_behind(require_deco_kernels, deco_recipe) -> None:
    """The workspace of a call comes from the caching allocator and lives for the call: the
    layers of a component hold no buffer between forwards, and calls of other sizes in between
    leave nothing a later call depends on."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads = 1, 512, 4
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    first, second = _backend(num_heads, deco_recipe()), _backend(num_heads, deco_recipe())
    out = first.forward(q, k, v)
    others = [_random_qkv(batch_size, n, num_heads, device, seed=1) for n in (200, 1024)]
    # The first call of a sequence length draws its basis rows, which the kernel package keeps.
    for call in others:
        second.forward(*call)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)

    for call in others:
        second.forward(*call)
    assert torch.equal(first.forward(q, k, v), out)
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated


def test_deco_forward_keeps_its_caller_compiled(require_deco_kernels, deco_recipe) -> None:
    """A torch.compile'd caller keeps its own operations compiled around the backend. The one
    graph break is the call that launches the kernels: a tensor address read in traced code
    would make the compiler give up the whole frame of the caller."""
    import torch._dynamo

    device = require_deco_kernels()
    seq_len, num_heads = 512, 2
    attn = _backend(num_heads, deco_recipe())

    class Caller(torch.nn.Module):
        def forward(self, x, key_padding_mask, token_tags):
            q, k, v = x * 0.5, x * 0.25, x + 1.0
            out = attn.forward(q, k, v, key_padding_mask=key_padding_mask, token_tags=token_tags)
            return out * 2.0

    x = _random_qkv(1, seq_len, num_heads, device)[0]
    key_padding_mask = torch.arange(seq_len, device=device)[None] < 400
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[300:] = 1
    caller = Caller()
    expected = caller(x, key_padding_mask, token_tags)

    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(caller)(x, key_padding_mask, token_tags)
        assert (explanation.graph_count, explanation.graph_break_count) == (2, 1), (
            explanation.break_reasons
        )
        compiled = torch.compile(caller, backend="eager")
        assert torch.equal(compiled(x, key_padding_mask, token_tags), expected)
    finally:
        torch._dynamo.reset()


def test_deco_delegates_unsupported_calls(
    require_deco_kernels, deco_recipe, kernel_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-attention, a causal mask and other dtypes run on VANILLA, unquantized; the layer
    notes its first delegated call with what keeps it from the kernels."""
    device = require_deco_kernels()
    num_heads = 2
    q, k, v = _random_qkv(1, 256, num_heads, device)
    attn = _backend(num_heads, deco_recipe())

    # The note names the dtype of K or V when Q is bfloat16. VANILLA takes one dtype for all.
    notes = []
    monkeypatch.setattr(
        deco_backend.logger, "warning_once", lambda message, key=None: notes.append(message)
    )
    with pytest.raises(RuntimeError, match="same dtype"):
        attn.forward(q, k.float(), v.float())
    assert attn._noted_unquantized

    out = attn.forward(q, k[:, :128], v[:, :128])
    assert out.shape == q.shape
    assert _relative_error(out, _sdpa(q, k[:, :128], v[:, :128])) < 2e-2
    # A layer whose first delegated call is cross-attention notes that.
    _backend(num_heads, deco_recipe()).forward(q, k[:, :128], v[:, :128])
    assert len(notes) == 2 and "cross-attention" in notes[1]

    out = attn.forward(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert _relative_error(out, _sdpa(q, k, v, is_causal=True)) < 2e-2

    out = attn.forward(q.half(), k.half(), v.half())
    assert out.dtype == torch.float16
    assert _relative_error(out, _sdpa(q, k, v)) < 2e-2

    # A delegated call reads a key padding mask as a prefix of its count of tokens, as the kernels
    # do; the rows after the prefix are zero.
    key_padding_mask = torch.ones(1, 256, dtype=torch.bool, device=device)
    key_padding_mask[:, 100:156] = False
    out = attn.forward(q.half(), k.half(), v.half(), key_padding_mask=key_padding_mask)
    prefix = torch.arange(256, device=device)[None] < 200
    ref = _sdpa(q, k, v, attn_mask=prefix[:, None, None, :])
    assert not out[~prefix].any()
    assert _relative_error(out[prefix], ref[prefix]) < 2e-2

    assert not kernel_calls
    # The layer notes once, whatever its later calls are delegated for.
    assert len(notes) == 2 and "torch.float32" in notes[0]


def test_deco_delegates_short_calls(
    require_deco_kernels, deco_recipe, kernel_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-attention call over fewer than ``MIN_SEQ_LEN`` rows, such as the text sequence
    of a token refiner, runs on VANILLA, with the note; a call of that many rows runs on the
    kernels."""
    device = require_deco_kernels()
    monkeypatch.setattr(deco_backend, "MIN_SEQ_LEN", MIN_SEQ_LEN)
    assert MIN_SEQ_LEN > 256
    num_heads = 2
    attn = _backend(num_heads, deco_recipe())

    q, k, v = _random_qkv(2, 160, num_heads, device)
    key_padding_mask = torch.ones(2, 160, dtype=torch.bool, device=device)
    key_padding_mask[0, 130:] = False
    ref = _sdpa(q, k, v, attn_mask=key_padding_mask[:, None, None, :])
    # The padded rows hold nothing that reaches the valid rows, NaN included.
    for padding in (None, float("nan")):
        if padding is not None:
            for tensor in (q, k, v):
                tensor[~key_padding_mask] = padding
        out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)
        assert not out[~key_padding_mask].any()
        assert _relative_error(out[key_padding_mask], ref[key_padding_mask]) < 2e-2
    assert not kernel_calls and attn._noted_unquantized

    # The boundary is 8192 rows: one row fewer delegates, that many runs the kernels.
    assert MIN_SEQ_LEN == 8192
    attn.forward(*_random_qkv(1, MIN_SEQ_LEN - 1, num_heads, device))
    assert not kernel_calls
    attn.forward(*_random_qkv(1, MIN_SEQ_LEN, num_heads, device))
    assert kernel_calls == [(1, MIN_SEQ_LEN, num_heads, 128)]


def test_deco_token_tags_draw_the_basis_rows_by_modality(
    require_deco_kernels, deco_recipe, kernel_inputs: list
) -> None:
    """With the token tags of a packed sequence the preparation takes half of its rows from the
    text and audio tokens and none from the padding. The rows are drawn on the device, the same
    for the same tags, and kept while the tags tensor stays as it is; with a mask they come from
    the tokens of the mask."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads = 1, 4096, 2
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:100] = 1  # text
    token_tags[100:160] = 2  # audio
    token_tags[-8:] = -1  # padding
    num_rows = deco_attention.inputs.sample_count(seq_len)
    attn = _backend(num_heads, deco_recipe())

    out = attn.forward(q, k, v, token_tags=token_tags)

    rows, valid_seq_lens = kernel_inputs[-1]
    assert valid_seq_lens is None and rows.shape == (num_rows,)
    drawn = token_tags[rows.long()]
    assert (drawn >= 0).all() and rows.unique().numel() == num_rows
    assert int((drawn > 0).sum()) == num_rows // 2

    # The kernels give the same output for these rows, and another one for their own draw.
    assert torch.equal(out, _kernels(q, k, v, rows=rows))
    assert not torch.equal(out, attn.forward(q, k, v))

    # The same tags in another tensor give the same rows, in inference mode as well.
    with torch.inference_mode():
        assert torch.equal(attn.forward(q, k, v, token_tags=token_tags.clone()), out)
    assert torch.equal(kernel_inputs[-1][0], rows)
    with pytest.raises(ValueError, match="token tags"):
        attn.forward(q, k, v, token_tags=token_tags[:-1])

    # The draw of the last tags tensor is kept until the tensor is written in place: the next
    # call then draws from the new text.
    attn.forward(q, k, v, token_tags=token_tags)
    kept = kernel_inputs[-1][0]
    attn.forward(q, k, v, token_tags=token_tags)
    assert kernel_inputs[-1][0] is kept
    token_tags[:1000] = 1
    attn.forward(q, k, v, token_tags=token_tags)
    assert int((kernel_inputs[-1][0] < 1000).sum()) == num_rows // 2

    # A group short of its half gives all its tokens and the other group fills the draw; tags
    # with fewer tokens than the draw repeat them; a sequence of padding draws row 0.
    mostly_text = torch.ones(seq_len, dtype=torch.int32, device=device)
    mostly_text[:5] = 0
    draw = deco_attention.sample_rows(seq_len, device, mostly_text)
    assert draw.unique().numel() == num_rows
    assert int((mostly_text[draw.long()] == 0).sum()) == 5
    few_tokens = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    few_tokens[:50] = 1
    assert torch.isfinite(attn.forward(q, k, v, token_tags=few_tokens)).all()
    assert kernel_inputs[-1][0].unique().tolist() == list(range(50))
    padding = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    draw = deco_attention.sample_rows(seq_len, device, padding)
    assert draw.numel() == num_rows and not draw.any()

    # The preparation takes a draw of fewer rows than twice its rank and rejects an empty one.
    lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    for short, valid_seq_lens in ((rows[:16], None), (rows[:16][None], lens)):
        out_short = _kernels(q, k, v, rows=short, valid_seq_lens=valid_seq_lens)
        assert torch.isfinite(out_short).all()
    with pytest.raises(ValueError, match="at least one token"):
        _kernels(q, k, v, rows=rows[:0])

    # A row outside the sequence stands for its nearest token.
    beyond, last = rows.clone(), rows.clone()
    beyond[0], last[0] = seq_len + 5, seq_len - 1
    assert torch.equal(_kernels(q, k, v, rows=beyond), _kernels(q, k, v, rows=last))

    # With a mask the rows come from the tokens of the mask.
    text_last = torch.zeros(seq_len, dtype=torch.int32, device=device)
    text_last[3000:] = 1  # text at the end; the tags mark no padding
    positions = torch.arange(seq_len, device=device)[None]
    for num_tokens in (3500, 3100):
        key_padding_mask = positions < num_tokens
        out = attn.forward(q, k, v, token_tags=text_last, key_padding_mask=key_padding_mask)
        assert not out[:, num_tokens:].any()
        rows, valid_seq_lens = kernel_inputs[-1]
        assert valid_seq_lens.tolist() == [num_tokens]
        assert rows.shape == (batch_size, num_rows) and int(rows.max()) < num_tokens
        assert int((rows >= 3000).sum()) == num_rows // 2


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_prefix_mask_attends_the_prefix(
    require_deco_kernels, deco_recipe, kernel_inputs: list, v_dtype: str
) -> None:
    """The tokens of a sample are its first rows and the rows after them padding: the tokens
    attend their own keys only, whatever the padding holds, and the padded rows come back zero.
    A mask without padding gives the bits of the call without a mask."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_tokens, num_heads = 2, 1024, 700, 3
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    attn = _backend(num_heads, deco_recipe(v_dtype))

    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
    prefix_mask = positions < num_tokens
    expected = attn.forward(q, k, v, key_padding_mask=prefix_mask)

    assert expected.shape == q.shape and expected.dtype == torch.bfloat16
    assert not expected[:, num_tokens:].any()
    prefix = expected[:, :num_tokens]
    ref = _sdpa(q[:, :num_tokens], k[:, :num_tokens], v[:, :num_tokens])
    uniform_weights = v[:, :num_tokens].float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(prefix, ref) < max(0.5 * _relative_error(uniform_weights, ref), 0.02)

    full = torch.ones_like(prefix_mask)
    assert torch.equal(attn.forward(q, k, v, key_padding_mask=full), attn.forward(q, k, v))

    # The token counts follow the content of the mask: a write into the tensor is seen.
    full[:, num_tokens:] = False
    assert torch.equal(attn.forward(q, k, v, key_padding_mask=full), expected)
    assert kernel_inputs[-1][1].tolist() == [num_tokens] * batch_size

    # Every sample has its own tokens: a shorter sample 0 changes that sample only.
    shorter = prefix_mask.clone()
    shorter[0, 500:] = False
    out = attn.forward(q, k, v, key_padding_mask=shorter)
    assert torch.equal(out[1], expected[1]) and not out[0, 500:].any()
    assert not torch.equal(out[0, :500], expected[0, :500])

    for padding in (0.0, 1e30, float("nan")):
        for tensor in (q, k, v):
            tensor[:, num_tokens:] = padding
        out = attn.forward(q, k, v, key_padding_mask=prefix_mask)
        assert torch.equal(out[:, :num_tokens], prefix)
        assert not out[:, num_tokens:].any()

    # A batch element without valid tokens comes back zero, whatever its padding holds.
    lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    out = _kernels(q, k, v, v_dtype, valid_seq_lens=lens)
    assert not out[0].any()
    assert torch.equal(out[1, :num_tokens], prefix[1])
    assert not out[1, num_tokens:].any()


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_skipped_tile_leaves_zero_scales(require_deco_kernels, v_dtype: str) -> None:
    """On SM120 the Q/K pass skips the tiles past the valid length and the encode pass reads the
    provisional scales of every tile: a skipped tile holds zero scales, also on a workspace that
    a longer call wrote before, and the output of the valid tile is that of a fresh workspace."""
    device = require_deco_kernels(v_dtype)
    if deco_attention.target.kernel_family() != "sm120":
        pytest.skip("The provisional scales of the Q/K pass are an SM120 kernel detail.")
    seq_len, num_heads = 256, 1
    q, k, v = _random_qkv(1, seq_len, num_heads, device, q_scale=4.0)
    num_bytes = deco_attention.workspace_size(1, seq_len, num_heads, device, v_dtype)
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    lens = torch.tensor([128], dtype=torch.int32, device=device)

    deco_attention.attention(
        deco_attention.prepare(q, k, v, workspace, v_dtype), torch.empty_like(q)
    )
    operands = deco_attention.prepare(q, k, v, workspace, v_dtype, valid_seq_lens=lens)
    out = deco_attention.attention(operands, torch.empty_like(q))

    prov = operands.buffers["prov"].reshape(-1).view(torch.uint8)
    tiles = seq_len // 128
    for side in range(2):
        skipped = (side * tiles + 1) * 3072
        assert not prov[skipped : skipped + 3072].any()
    assert torch.equal(out, _kernels(q, k, v, v_dtype, valid_seq_lens=lens))


def test_deco_batch_stride_past_2_to_the_24_tokens(require_deco_kernels) -> None:
    """A call whose tokens times heads exceed 2^24 addresses its second batch element from its own
    rows: the batch strides of the attention's views are 64-bit products. Short valid lengths
    keep the run quick; the second element's prefix is checked against its own SDPA and must
    differ from the first's."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads, valid = 2, 419456, 40, 1024
    assert seq_len * num_heads > 2**24
    need = 5 * batch_size * seq_len * num_heads * 128 * 2
    if torch.cuda.get_device_properties(device).total_memory < need:
        pytest.skip("the inputs, the output and the workspace of this call need more memory")
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn((batch_size, seq_len, num_heads, 128), generator=gen, device=device).bfloat16()
        for _ in range(3)
    )
    q[:, :valid] *= 4
    lens = torch.full((batch_size,), valid, dtype=torch.int32, device=device)
    out = _kernels(q, k, v, valid_seq_lens=lens)
    for b in range(batch_size):
        ref = _sdpa(q[b : b + 1, :valid], k[b : b + 1, :valid], v[b : b + 1, :valid])
        uniform = v[b : b + 1, :valid].float().mean(dim=1, keepdim=True).expand_as(ref)
        assert _relative_error(out[b : b + 1, :valid], ref) < max(
            0.5 * _relative_error(uniform, ref), 0.02
        )
    assert not torch.equal(out[0, :valid], out[1, :valid])
    assert not out[:, valid:].any()


def test_deco_sm120_fp8_takes_at_least_one_key_tile(require_deco_kernels) -> None:
    """FP8 PV on SM120 has no kernel for a sequence shorter than one key tile: the kernel API
    names the limit and rejects a shorter sequence, while NVFP4 PV takes any length."""
    device = require_deco_kernels()
    if deco_attention.target.kernel_family() != "sm120":
        pytest.skip("The limit is that of the SM120 FP8-PV kernel.")
    assert deco_attention.min_seq_len("fp8") == 128
    assert deco_attention.min_seq_len("nvfp4") == 1
    num_heads = 2
    q, k, v = _random_qkv(1, 127, num_heads, device)
    with pytest.raises(ValueError, match="at least 128 tokens"):
        deco_attention.workspace_size(1, 127, num_heads, device, "fp8")
    num_bytes = deco_attention.workspace_size(1, 128, num_heads, device, "fp8")
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="at least 128 tokens"):
        deco_attention.prepare(q, k, v, workspace, "fp8")


def test_deco_sm120_nvfp4_split_units_merge_a_padded_half(require_deco_kernels) -> None:
    """The SM120 NVFP4 PV kernel splits the units of a long sequence into two key halves when
    they do not fill the SMs; with a valid length in the second half the merged output of the
    valid rows follows FP32 SDPA, the padded rows are zero, and a second call on the same
    workspace returns the same bits."""
    device = require_deco_kernels("nvfp4")
    seq_len, valid_seq_len, num_heads = 3200, 2049, 1
    split_plan = _split_plan("sm120.nvfp4pv")
    full_units, split_units, split_tile = split_plan(1, seq_len, num_heads, num_sms(device))
    # The split units are the last query tiles: the first of them starts below the valid length, and the valid
    # keys reach past the split point, so the merge of a valid query tile has two non-empty halves.
    assert split_units > 0 and full_units * 128 < valid_seq_len and split_tile * 128 < valid_seq_len
    q, k, v = _random_qkv(1, seq_len, num_heads, device, q_scale=4.0)
    lens = torch.tensor([valid_seq_len], dtype=torch.int32, device=device)
    num_bytes = deco_attention.workspace_size(1, seq_len, num_heads, device, "nvfp4")
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)

    def run() -> torch.Tensor:
        operands = deco_attention.prepare(q, k, v, workspace, "nvfp4", valid_seq_lens=lens)
        return deco_attention.attention(operands, torch.empty_like(q))

    out = run()
    assert not out[:, valid_seq_len:].any()
    ref = _sdpa(q[:, :valid_seq_len], k[:, :valid_seq_len], v[:, :valid_seq_len])
    uniform_weights = v[:, :valid_seq_len].float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(out[:, :valid_seq_len], ref) < max(
        0.5 * _relative_error(uniform_weights, ref), 0.02
    )
    assert torch.equal(run(), out)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_key_padding_mask_attends_the_tokens(
    require_deco_kernels, deco_recipe, kernel_inputs: list, v_dtype: str
) -> None:
    """With a ``key_padding_mask`` every sample attends its own tokens only and the masked rows
    come back zero. A sample whose mask excludes every row comes back zero, on the kernels and
    on a delegated call, with and without token tags, next to a sample that is served."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 2, 1024, 3
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    attn = _backend(num_heads, deco_recipe(v_dtype))

    # The samples hold different numbers of tokens, each followed by its padding.
    lengths = torch.tensor([717, 600], device=device)
    key_padding_mask = torch.arange(seq_len, device=device) < lengths[:, None]

    out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)

    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert not out[~key_padding_mask].any()
    assert kernel_inputs[-1][1].tolist() == lengths.tolist()
    ref = _sdpa(q, k, v, attn_mask=key_padding_mask[:, None, None, :])
    for sample in range(batch_size):
        valid = key_padding_mask[sample]
        uniform_weights = v[sample, valid].float().mean(dim=0, keepdim=True)
        uniform_error = _relative_error(
            uniform_weights.expand_as(ref[sample, valid]), ref[sample, valid]
        )
        error = _relative_error(out[sample, valid], ref[sample, valid])
        assert error < max(0.5 * uniform_error, 0.02)

    with pytest.raises(ValueError, match="key_padding_mask"):
        attn.forward(q, k, v, key_padding_mask=key_padding_mask[:, :-1])

    # Sample 0 without tokens, sample 1 with 900.
    empty_sample = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    empty_sample[0] = False
    empty_sample[1, 900:] = False
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[800:] = 1
    ref = _sdpa(q[1:], k[1:], v[1:], attn_mask=empty_sample[1:, None, None, :])[0, :900]

    for kwargs in ({}, {"token_tags": token_tags}):
        out = attn.forward(q, k, v, key_padding_mask=empty_sample, **kwargs)
        assert not out[0].any() and not out[1, 900:].any()
        assert torch.isfinite(out).all()
        assert _relative_error(out[1, :900], ref) < 0.2
    rows = kernel_inputs[-1][0]
    assert rows.shape[0] == batch_size and (rows[1] < 900).all()
    # The draw of sample 1 keeps its rows, half of them text, next to the empty sample.
    assert int((rows[1] >= 800).sum()) == rows.shape[1] // 2

    out = attn.forward(q.half(), k.half(), v.half(), key_padding_mask=empty_sample)
    assert not out[0].any() and not out[1, 900:].any()
    assert _relative_error(out[1, :900], ref) < 2e-2

    # The mask is read on the device on every call, in inference mode as well: a write into it
    # between two calls is followed.
    with torch.inference_mode():
        inference_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        attn.forward(q, k, v, key_padding_mask=inference_mask)
        inference_mask[:, 700:] = False
        out = attn.forward(q, k, v, key_padding_mask=inference_mask)
    assert torch.equal(out, attn.forward(q, k, v, key_padding_mask=inference_mask.clone()))


def test_deco_token_tags_with_a_key_padding_mask(
    require_deco_kernels, deco_recipe, kernel_inputs: list
) -> None:
    """With token tags and a mask every sample gets the draw of its own tokens, half of it from
    its text: the samples of a batch may hold text of different lengths."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads = 2, 4096, 2
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    attn = _backend(num_heads, deco_recipe())

    # Video, then text with the padding of every sample last: the tags mark no padding.
    num_video = 3600
    token_tags = torch.ones(seq_len, dtype=torch.int32, device=device)
    token_tags[:num_video] = 0
    lengths = torch.tensor([num_video + 400, num_video + 20], device=device)
    key_padding_mask = torch.arange(seq_len, device=device) < lengths[:, None]

    out = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask)

    rows, valid_seq_lens = kernel_inputs[-1]
    assert valid_seq_lens.tolist() == lengths.tolist()
    assert rows.shape[0] == batch_size and (rows < lengths[:, None]).all()
    text_rows = (rows >= num_video).sum(dim=1)
    assert text_rows.tolist() == [rows.shape[1] // 2, 20]
    assert not out[~key_padding_mask].any()
    assert torch.equal(out, _kernels(q, k, v, rows=rows, valid_seq_lens=lengths.to(torch.int32)))

    # The layers of a forward pass the same two tensors: the draw is kept for them.
    assert torch.equal(
        attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask), out
    )
    assert kernel_inputs[-1][0] is rows


@pytest.mark.parametrize("v_dtype", V_DTYPES)
@pytest.mark.parametrize(
    "sampling",
    [dict(), dict(q_sample_rate=0.0078125, k_sample_rate=0.00390625, min_sample_rows=512)],
    ids=["default_sampling", "one_count_for_two_rates"],
)
def test_deco_cuda_graph_with_a_key_padding_mask(
    require_deco_kernels, deco_recipe, kernel_inputs: list, v_dtype: str, sampling: dict
) -> None:
    """A captured forward with a mask and token tags computes its token counts and basis rows
    inside the graph: a replay follows the content of the mask and tags tensors of the capture,
    which the eager forward before it has not seen. Two sample rates that give one count of
    rows for the sequence draw one set of rows, in the eager forward and in the capture alike."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 2, 2048, 2
    attn = _backend(num_heads, deco_recipe(v_dtype, **sampling))
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[1800:] = 1
    lengths = torch.tensor([[2000], [1850]], device=device)
    positions = torch.arange(seq_len, device=device)
    key_padding_mask = positions < lengths

    attn.forward(q, k, v, token_tags=token_tags.clone(), key_padding_mask=key_padding_mask.clone())
    assert torch.is_tensor(kernel_inputs[-1][0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask)

    new_q, new_k, new_v = _random_qkv(batch_size, seq_len, num_heads, device, seed=1)
    audio_first = torch.zeros_like(token_tags)
    audio_first[:300] = 2
    for new_mask, new_tags in (
        (key_padding_mask.clone(), token_tags.clone()),
        (positions < lengths - 700, token_tags.clone()),
        (torch.ones_like(key_padding_mask), audio_first),
    ):
        expected = attn.forward(
            new_q, new_k, new_v, token_tags=new_tags, key_padding_mask=new_mask
        ).clone()
        q.copy_(new_q)
        k.copy_(new_k)
        v.copy_(new_v)
        key_padding_mask.copy_(new_mask)
        token_tags.copy_(new_tags)
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out, expected)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_cuda_graph(require_deco_kernels, deco_recipe, v_dtype: str) -> None:
    """A forward is captured after an eager forward of its shape and replays onto new inputs,
    also after eager forwards of other shapes and other token tags. A sequence length first met
    under capture is rejected with a clear error."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 1, 384, 2
    attn = _backend(num_heads, deco_recipe(v_dtype))
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:64] = 1

    attn.forward(q, k, v, token_tags=token_tags)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = attn.forward(q, k, v, token_tags=token_tags)

    for other_len in (128, 1024, 256, 512, 640, 768):
        other_tags = torch.zeros(other_len, dtype=torch.int32, device=device)
        other_tags[:64] = 1
        attn.forward(*_random_qkv(batch_size, other_len, num_heads, device), token_tags=other_tags)
    new_q, new_k, new_v = _random_qkv(batch_size, seq_len, num_heads, device, seed=1)
    expected = attn.forward(new_q, new_k, new_v, token_tags=token_tags).clone()
    q.copy_(new_q)
    k.copy_(new_k)
    v.copy_(new_v)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, expected)

    # The preparation builds its inputs for a sequence length on first use, outside capture.
    unseen_len = next(_UNSEEN_LENGTHS)
    unseen = _random_qkv(batch_size, unseen_len, num_heads, device)
    num_bytes = deco_attention.workspace_size(batch_size, unseen_len, num_heads, device, v_dtype)
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    with pytest.raises(RuntimeError, match="eagerly once before capturing"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            deco_attention.prepare(*unseen, workspace, v_dtype)
