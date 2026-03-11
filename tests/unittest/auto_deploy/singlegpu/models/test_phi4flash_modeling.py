"""Tests for Phi-4 Flash custom model implementation.

Hierarchical test levels:
1. Block equivalence — MLP, attention, Mamba
2. Layer equivalence — standard and YOCO/shared layers
3. Full model equivalence — end-to-end logits comparison
4. Export test — torch_export_to_gm with dynamic shapes
"""

import copy
import math
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_phi4flash import (
    Phi4FlashAttention,
    Phi4FlashConfig,
    Phi4FlashDecoderLayer,
    Phi4FlashForCausalLM,
    Phi4FlashMamba,
    Phi4FlashMLP,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def assert_rmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rmse_ratio_tol: float,
    msg: str = "",
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff_rmse = torch.sqrt(torch.mean((actual_f - expected_f) ** 2))
    expected_rmse = torch.sqrt(torch.mean(expected_f**2))
    ratio = diff_rmse / max(expected_rmse, torch.tensor(1e-12, device=expected_f.device))
    assert ratio <= rmse_ratio_tol, (
        f"{msg}rmse(actual-expected)/rmse(expected)={ratio.item():.6f} > {rmse_ratio_tol}"
    )


def _create_small_config() -> Phi4FlashConfig:
    return Phi4FlashConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        sliding_window=4,
        mb_per_layer=2,
        mamba_d_state=8,
        mamba_d_conv=3,
        mamba_expand=2,
        mamba_dt_rank=4,
    )


def _lambda_init_fn(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def _swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return value * F.silu(gate)


def _hf_phi4flash_ssm_prefill(
    x: torch.Tensor,
    z: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    yoco_kv: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch_size, seq_len, d_inner = x.shape
    d_state = B.shape[-1]
    state = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=torch.float32)
    A_fp32 = A.to(torch.float32)
    D_fp32 = D.to(torch.float32)
    dt_bias_fp32 = dt_bias.to(torch.float32)

    outputs = []
    yoco_key_values = []
    for token_idx in range(seq_len):
        x_t = x[:, token_idx].to(torch.float32)
        z_t = z[:, token_idx].to(torch.float32)
        dt_t = F.softplus(dt[:, token_idx].to(torch.float32) + dt_bias_fp32)
        B_t = B[:, token_idx].to(torch.float32)
        C_t = C[:, token_idx].to(torch.float32)

        dA = torch.exp(dt_t.unsqueeze(-1) * A_fp32.unsqueeze(0))
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
        state = state * dA + x_t.unsqueeze(-1) * dB

        y_t = torch.einsum("bdn,bn->bd", state, C_t) + D_fp32.unsqueeze(0) * x_t
        if yoco_kv:
            yoco_key_values.append(y_t.to(dtype=x.dtype))
            y_t = _swiglu(z_t, y_t)
        else:
            y_t = y_t * F.silu(z_t)
        outputs.append(y_t.to(dtype=x.dtype))

    y = torch.stack(outputs, dim=1)
    yoco_out = torch.stack(yoco_key_values, dim=1) if yoco_kv else None
    return y, yoco_out


def _split_heads(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.view(*x.shape[:-2], x.shape[-2] // 2, 2, x.shape[-1])
    return x[..., 0, :], x[..., 1, :]


def _repeat_kv(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    n_heads = q.shape[2]
    bs, slen, n_kv_heads, head_dim = kv.shape
    n_rep = n_heads // n_kv_heads
    return (
        kv[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_heads, head_dim)
    )


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    if q.shape[2] != k.shape[2]:
        k = _repeat_kv(q, k)
        v = _repeat_kv(q, v)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_mask = None
    is_causal = sliding_window is None
    if sliding_window is not None:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        q_pos = torch.arange(q_len, device=q.device)[:, None]
        k_pos = torch.arange(k_len, device=q.device)[None, :]
        blocked = (k_pos > q_pos) | ((q_pos - k_pos) >= sliding_window)
        attn_mask = torch.zeros((q_len, k_len), dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(blocked, float("-inf"))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    return out.transpose(1, 2).contiguous()


class RefPhi4FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RefPhi4FlashMLP(nn.Module):
    def __init__(self, config: Phi4FlashConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.mlp_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(up * F.silu(gate))


class RefPhi4FlashDiffAttention(nn.Module):
    def __init__(self, head_dim: int, depth: int):
        super().__init__()
        self.head_dim = head_dim
        self.lambda_init = _lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.subln = RefPhi4FlashRMSNorm(2 * head_dim, eps=1e-5)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        q1, q2 = _split_heads(q)
        k1, k2 = _split_heads(k)
        v1, v2 = _split_heads(v)

        attn11 = _reference_attention(q1, k1, v1, sliding_window)
        attn12 = _reference_attention(q1, k1, v2, sliding_window)
        attn21 = _reference_attention(q2, k2, v1, sliding_window)
        attn22 = _reference_attention(q2, k2, v2, sliding_window)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = self.subln(attn1 - lambda_full * attn2)
        attn = attn * (1 - self.lambda_init)
        return attn.view(*attn.shape[:-2], attn.shape[-2] * 2, self.head_dim)


class RefPhi4FlashAttention(nn.Module):
    def __init__(self, config: Phi4FlashConfig, layer_idx: int, yoco_cross: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.yoco_cross = yoco_cross
        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        self.Wqkv = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim if yoco_cross else op_size,
            bias=True,
        )
        self.inner_cross_attn = RefPhi4FlashDiffAttention(self.head_dim, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        if self.yoco_cross:
            q = self.Wqkv(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
            k, v = yoco_key_values
            sliding_window = None
        else:
            qkv = self.Wqkv(hidden_states)
            q_pos = self.num_heads * self.head_dim
            kv_pos = self.num_key_value_heads * self.head_dim
            q = qkv[..., :q_pos].view(bsz, q_len, self.num_heads, self.head_dim)
            k = qkv[..., q_pos : q_pos + kv_pos].view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
            v = qkv[..., q_pos + kv_pos :].view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            sliding_window = self.config.sliding_window[self.layer_idx]
        out = self.inner_cross_attn(q, k, v, sliding_window)
        return self.out_proj(out.reshape(bsz, q_len, self.hidden_size)), (k, v)


class RefPhi4FlashMamba(nn.Module):
    def __init__(self, config: Phi4FlashConfig, yoco_cross: bool = False, yoco_kv: bool = False):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.d_inner = config.mamba_expand * config.hidden_size
        self.num_heads = self.d_inner
        self.head_dim = 1
        self.n_groups = 1
        self.dt_rank = config.mamba_dt_rank
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner if yoco_cross else self.d_inner * 2,
            bias=config.mamba_proj_bias,
        )
        if not yoco_cross:
            self.conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=self.d_conv,
                groups=self.d_inner,
                padding=self.d_conv - 1,
                bias=config.mamba_conv_bias,
            )
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
            self.A_log = nn.Parameter(torch.log(A.expand(self.d_inner, -1).contiguous()))
            self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.mamba_proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dtype = hidden_states.dtype
        if self.yoco_cross:
            out = self.in_proj(hidden_states)
            out = _swiglu(out, yoco_key_values)
            return self.out_proj(out), yoco_key_values

        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., : hidden_states.shape[1]].transpose(1, 2)
        x = F.silu(x)
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        y, yoco_out = _hf_phi4flash_ssm_prefill(
            x=x,
            z=z,
            A=-torch.exp(self.A_log.float()),
            B=B,
            C=C,
            D=self.D,
            dt=torch.matmul(dt, self.dt_proj.weight.t()),
            dt_bias=self.dt_proj.bias,
            yoco_kv=self.yoco_kv,
        )
        return self.out_proj(y.to(dtype)), yoco_out


class RefPhi4FlashDecoderLayer(nn.Module):
    def __init__(self, config: Phi4FlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = RefPhi4FlashMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        yoco_mb = layer_idx >= config.num_hidden_layers // 2
        yoco_cross = layer_idx >= (config.num_hidden_layers // 2 + 2)
        attn_config = config
        if layer_idx >= (config.num_hidden_layers // 2 + 1):
            attn_config = Phi4FlashConfig(**config.to_dict())
            attn_config.sliding_window = [None] * config.num_hidden_layers

        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            self.attn = RefPhi4FlashMamba(attn_config, yoco_cross=yoco_cross, yoco_kv=yoco_mb)
        else:
            self.attn = RefPhi4FlashAttention(
                attn_config, layer_idx=layer_idx, yoco_cross=yoco_cross
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ssm_output: Optional[torch.Tensor] = None,
        yoco_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(hidden_states, yoco_key_values=ssm_output)
        else:
            attn_outputs, yoco_key_values = self.attn(
                hidden_states, yoco_key_values=yoco_key_values
            )
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, ssm_output, yoco_key_values


class RefPhi4FlashForCausalLM(nn.Module):
    def __init__(self, config: Phi4FlashConfig):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.model.layers = nn.ModuleList(
            [RefPhi4FlashDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.model.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        del position_ids
        hidden_states = self.model.embed_tokens(input_ids)
        ssm_output = None
        yoco_key_values = None
        for layer in self.model.layers:
            hidden_states, ssm_output, yoco_key_values = layer(
                hidden_states,
                ssm_output=ssm_output,
                yoco_key_values=yoco_key_values,
            )
        hidden_states = self.model.final_layernorm(hidden_states)
        return self.lm_head(hidden_states)


def _assert_same_state_dict(custom: nn.Module, ref: nn.Module):
    missing, unexpected = ref.load_state_dict(custom.state_dict(), strict=False)
    assert not missing
    assert not unexpected


def _load_hf_reference_state_dict(custom: nn.Module, ref: nn.Module):
    missing, unexpected = custom.load_state_dict(copy.deepcopy(ref.state_dict()), strict=False)
    assert not missing
    assert not unexpected


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4flash_mlp_equivalence(B, S):
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    custom = Phi4FlashMLP(config).to(device=device, dtype=dtype)
    ref = RefPhi4FlashMLP(config).to(device=device, dtype=dtype)
    _assert_same_state_dict(custom, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom(x), ref(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4flash_attention_equivalence(B, S):
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    custom = Phi4FlashAttention(config, layer_idx=1).to(device=device, dtype=dtype)
    ref = RefPhi4FlashAttention(config, layer_idx=1).to(device=device, dtype=dtype)
    _assert_same_state_dict(custom, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    custom_out, _ = custom(x)
    ref_out, _ = ref(x)
    assert_rmse_close(custom_out, ref_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4flash_mamba_equivalence(B, S):
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    custom = Phi4FlashMamba(config, layer_idx=0, yoco_cross=False, yoco_kv=False).to(
        device=device, dtype=dtype
    )
    ref = RefPhi4FlashMamba(config, yoco_cross=False, yoco_kv=False).to(device=device, dtype=dtype)
    _load_hf_reference_state_dict(custom, ref)
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    custom_out, _ = custom(x)
    ref_out, _ = ref(x)
    assert_rmse_close(custom_out, ref_out, rmse_ratio_tol=0.05, msg="Mamba: ")


@torch.no_grad()
def test_phi4flash_mamba_load_preserves_hf_a_log():
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    custom = Phi4FlashMamba(config, layer_idx=0, yoco_cross=False, yoco_kv=False).to(
        device=device, dtype=dtype
    )
    ref = RefPhi4FlashMamba(config, yoco_cross=False, yoco_kv=False).to(device=device, dtype=dtype)
    ref_state = copy.deepcopy(ref.state_dict())
    expected = ref_state["A_log"]

    missing, unexpected = custom.load_state_dict(ref_state, strict=False)
    assert not missing
    assert not unexpected
    torch.testing.assert_close(custom.A_log, expected, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_phi4flash_shared_mamba_equivalence():
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    custom = Phi4FlashMamba(config, layer_idx=4, yoco_cross=True, yoco_kv=True).to(
        device=device, dtype=dtype
    )
    ref = RefPhi4FlashMamba(config, yoco_cross=True, yoco_kv=True).to(device=device, dtype=dtype)
    _load_hf_reference_state_dict(custom, ref)
    x = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    ssm = torch.randn(2, 6, config.mamba_expand * config.hidden_size, device=device, dtype=dtype)
    custom_out, _ = custom(x, yoco_key_values=ssm)
    ref_out, _ = ref(x, yoco_key_values=ssm)
    torch.testing.assert_close(custom_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("layer_idx", [1, 5])
@torch.no_grad()
def test_phi4flash_decoder_layer_equivalence(layer_idx):
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    layer = Phi4FlashDecoderLayer(config, layer_idx=layer_idx).to(device=device, dtype=dtype)
    ref = RefPhi4FlashDecoderLayer(config, layer_idx=layer_idx).to(device=device, dtype=dtype)
    _load_hf_reference_state_dict(layer, ref)
    x = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    ssm = torch.randn(2, 6, config.mamba_expand * config.hidden_size, device=device, dtype=dtype)
    head_dim = config.hidden_size // config.num_attention_heads
    kv = (
        torch.randn(2, 6, config.num_key_value_heads, head_dim, device=device, dtype=dtype),
        torch.randn(2, 6, config.num_key_value_heads, head_dim, device=device, dtype=dtype),
    )
    out, next_ssm, next_kv = layer(x, ssm_output=ssm, yoco_key_values=kv)
    ref_out, ref_ssm, ref_kv = ref(x, ssm_output=ssm, yoco_key_values=kv)
    assert_rmse_close(out, ref_out, rmse_ratio_tol=0.05, msg=f"Layer {layer_idx}: ")
    if next_ssm is not None and ref_ssm is not None:
        assert_rmse_close(next_ssm, ref_ssm, rmse_ratio_tol=0.05, msg=f"Layer {layer_idx} ssm: ")
    if next_kv is not None and ref_kv is not None:
        assert_rmse_close(next_kv[0], ref_kv[0], rmse_ratio_tol=0.05, msg=f"Layer {layer_idx} k: ")
        assert_rmse_close(next_kv[1], ref_kv[1], rmse_ratio_tol=0.05, msg=f"Layer {layer_idx} v: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4flash_full_model_equivalence(B, S):
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    custom = Phi4FlashForCausalLM(config).to(device=device, dtype=dtype)
    ref = RefPhi4FlashForCausalLM(config).to(device=device, dtype=dtype)
    _load_hf_reference_state_dict(custom, ref)
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    custom_out = custom(input_ids=input_ids, position_ids=position_ids)
    ref_out = ref(input_ids=input_ids, position_ids=position_ids)
    assert_rmse_close(custom_out.logits, ref_out, rmse_ratio_tol=0.05, msg="Full model: ")


@torch.no_grad()
def test_phi4flash_model_can_be_exported():
    config = _create_small_config()
    dtype = torch.bfloat16
    device = "cpu"
    model = Phi4FlashForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )
    move_to_device(gm, device)
    out = gm(input_ids=input_ids, position_ids=position_ids)
    assert "logits" in out
    assert out["logits"].shape == (B, S, config.vocab_size)
    assert torch.isfinite(out["logits"]).all()

    input_ids2 = torch.randint(0, config.vocab_size, (1, 4), device=device)
    position_ids2 = torch.arange(4, device=device).unsqueeze(0)
    out2 = gm(input_ids=input_ids2, position_ids=position_ids2)
    assert out2["logits"].shape == (1, 4, config.vocab_size)
    assert torch.isfinite(out2["logits"]).all()
