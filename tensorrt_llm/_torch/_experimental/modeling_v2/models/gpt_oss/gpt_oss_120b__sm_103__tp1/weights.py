# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: gpt-oss-120b / sm_103 / tp1.

One `W` per weight role, carrying its shape, checkpoint source, dtype and
load-time transform. tp1: no sharding -- every checkpoint tensor reaches
exactly one parameter whole. Storage layout is HF [out, in] row-major, so the
attention, router and embed copies are layout-preserving.

The parity trap in the MXFP4 expert operands: this checkpoint's `2I` axis
runs (gate, up, gate, up, ...) -- HF reads `gate = gate_up[..., ::2]`, `up =
gate_up[..., 1::2]` -- while the kernel's interleave wants destination row
`2i` = **up** `i` and `2i+1` = gate `i`. Getting it backwards is finite,
plausibly scaled and invisible to a boot check.
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2._weights import ModelWeights, W

SCALE_BLOCK = 32  # mxfp4: one E8M0 exponent per 32 elements along K

# FC1's K alignment for the trtllm-gen MXFP4 weight family. It sizes the
# declared expert operands here and is the alignment mxfp8_quantize pads the
# hidden states up to in modeling.py; the two must be the same number.
FC1_K_ALIGN = 512


def _pad_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _pad_rows_cols(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a `[E, R, C]` stack to `[E, rows, cols]`."""
    out = torch.zeros(t.shape[0], rows, cols, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1], : t.shape[2]] = t
    return out


def _pad_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    """Zero-pad a `[E, C]` stack to `[E, cols]`."""
    out = torch.zeros(t.shape[0], cols, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1]] = t
    return out


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of
    32 rows, source row `4u + v` (0<=u<8, 0<=v<4) lands at destination row
    `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I_pad`
    row stack into up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def _swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 block-scale swizzle, viewed back as `[E, M, C]`: the
    byte at `(m, c)` moves to flat offset `(m//128)*512*(C//4) + (c//4)*512
    + (m%32)*16 + ((m%128)//32)*4 + (c%4)`."""
    e, m, c = s.shape
    assert m % 128 == 0 and c % 4 == 0, (m, c)
    v = s.reshape(e, m // 128, 4, 32, c // 4, 4)  # (e, m/128, (m%128)/32, m%32, c/4, c%4)
    return v.permute(0, 1, 4, 3, 2, 5).reshape(e, m, c).contiguous()


def _split_gate_up(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the checkpoint's interleaved `2I` output axis into (up, gate)."""
    return t[:, 1::2], t[:, 0::2]


def _fc1_perm(rows: int, device) -> torch.Tensor:
    """Interleave then 32-row block shuffle, composed into one gather."""
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _prep_fc1_weight(blocks: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I, H/32, 16] uint8 blocks -> kernel FC1 operand."""
    cfg = core.model_config.pretrained_config
    b = blocks.reshape(cfg.num_local_experts, 2 * cfg.intermediate_size, cfg.hidden_size // 2)
    up, gate = _split_gate_up(b)
    cols = core.fc1_k_pad // 2
    w = torch.cat(
        [
            _pad_rows_cols(up, core.inter_pad, cols),
            _pad_rows_cols(gate, core.inter_pad, cols),
        ],
        dim=1,
    )
    return torch.index_select(w, 1, _fc1_perm(w.shape[1], w.device)).contiguous()


def _prep_fc1_scale(scales: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I, H/32] uint8 E8M0 scales -> kernel FC1 scale operand."""
    up, gate = _split_gate_up(scales)
    cols = core.fc1_k_pad // SCALE_BLOCK
    s = torch.cat(
        [
            _pad_rows_cols(up, core.inter_pad, cols),
            _pad_rows_cols(gate, core.inter_pad, cols),
        ],
        dim=1,
    )
    return _swizzle_scales(torch.index_select(s, 1, _fc1_perm(s.shape[1], s.device)))


def _prep_fc1_bias(bias: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I] bf16 biases -> fp32 kernel FC1 bias (same row permutation)."""
    up, gate = _split_gate_up(bias.float())
    b = torch.cat([_pad_cols(up, core.inter_pad), _pad_cols(gate, core.inter_pad)], dim=1)
    return torch.index_select(b, 1, _fc1_perm(b.shape[1], b.device)).contiguous()


def _prep_fc2_weight(blocks: torch.Tensor, core) -> torch.Tensor:
    """[E, H, I/32, 16] uint8 blocks -> kernel FC2 operand (no interleave)."""
    cfg = core.model_config.pretrained_config
    w = _pad_rows_cols(
        blocks.reshape(cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size // 2),
        core.fc2_rows_pad,
        core.inter_pad // 2,
    )
    return torch.index_select(w, 1, _block32_perm(core.fc2_rows_pad, w.device)).contiguous()


def _prep_fc2_scale(scales: torch.Tensor, core) -> torch.Tensor:
    """[E, H, I/32] uint8 E8M0 scales -> kernel FC2 scale operand."""
    s = _pad_rows_cols(scales, core.fc2_rows_pad, core.inter_pad // SCALE_BLOCK)
    return _swizzle_scales(torch.index_select(s, 1, _block32_perm(core.fc2_rows_pad, s.device)))


def _prep_fc2_bias(bias: torch.Tensor, core) -> torch.Tensor:
    """[E, H] bf16 biases -> fp32 kernel FC2 bias (same row permutation)."""
    b = _pad_cols(bias.float(), core.fc2_rows_pad)
    return torch.index_select(b, 1, _block32_perm(core.fc2_rows_pad, b.device)).contiguous()


def _to_fp32(t: torch.Tensor, core) -> torch.Tensor:
    """bf16 -> fp32 promotion for tensors the kernels require in fp32."""
    return t.float()


class GptOssWeights(ModelWeights):
    """gpt-oss-120b / sm_103 / tp1."""

    WEIGHTS: tuple[W, ...] = (
        W("norm1", shape=lambda d: (d.hidden,), src="{p}.input_layernorm.weight"),
        W(
            "qkv",
            shape=lambda d: (d.q_width + 2 * d.kv_width, d.hidden),
            src=(
                ("{p}.self_attn.q_proj.weight", lambda d: (slice(0, d.q_width),)),
                (
                    "{p}.self_attn.k_proj.weight",
                    lambda d: (slice(d.q_width, d.q_width + d.kv_width),),
                ),
                (
                    "{p}.self_attn.v_proj.weight",
                    lambda d: (slice(d.q_width + d.kv_width, d.q_width + 2 * d.kv_width),),
                ),
            ),
        ),
        W(
            "qkv_bias",
            shape=lambda d: (d.q_width + 2 * d.kv_width,),
            src=(
                ("{p}.self_attn.q_proj.bias", lambda d: (slice(0, d.q_width),)),
                (
                    "{p}.self_attn.k_proj.bias",
                    lambda d: (slice(d.q_width, d.q_width + d.kv_width),),
                ),
                (
                    "{p}.self_attn.v_proj.bias",
                    lambda d: (slice(d.q_width + d.kv_width, d.q_width + 2 * d.kv_width),),
                ),
            ),
        ),
        W(
            "sinks",
            shape=lambda d: (d.heads_q,),
            dtype=torch.float32,
            src="{p}.self_attn.sinks",
            transform=_to_fp32,
        ),
        W("o", shape=lambda d: (d.hidden, d.q_width), src="{p}.self_attn.o_proj.weight"),
        W("o_bias", shape=lambda d: (d.hidden,), src="{p}.self_attn.o_proj.bias"),
        W("norm2", shape=lambda d: (d.hidden,), src="{p}.post_attention_layernorm.weight"),
        W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.router.weight"),
        W("router_bias", shape=lambda d: (d.num_experts,), src="{p}.mlp.router.bias"),
        W(
            "fc1_w",
            shape=lambda d: (d.num_experts, d.fc1_rows, d.fc1_k_pad // 2),
            dtype=torch.uint8,
            src="{p}.mlp.experts.gate_up_proj_blocks",
            transform=_prep_fc1_weight,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.num_experts, d.fc1_rows, d.fc1_k_pad // 32),
            dtype=torch.uint8,
            src="{p}.mlp.experts.gate_up_proj_scales",
            transform=_prep_fc1_scale,
        ),
        W(
            "fc1_b",
            shape=lambda d: (d.num_experts, d.fc1_rows),
            dtype=torch.float32,
            src="{p}.mlp.experts.gate_up_proj_bias",
            transform=_prep_fc1_bias,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.num_experts, d.fc2_rows_pad, d.inter_pad // 2),
            dtype=torch.uint8,
            src="{p}.mlp.experts.down_proj_blocks",
            transform=_prep_fc2_weight,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.num_experts, d.fc2_rows_pad, d.inter_pad // 32),
            dtype=torch.uint8,
            src="{p}.mlp.experts.down_proj_scales",
            transform=_prep_fc2_scale,
        ),
        W(
            "fc2_b",
            shape=lambda d: (d.num_experts, d.fc2_rows_pad),
            dtype=torch.float32,
            src="{p}.mlp.experts.down_proj_bias",
            transform=_prep_fc2_bias,
        ),
        W("final_norm", shape=lambda d: (d.hidden,), src="model.norm.weight", per_layer=False),
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src="model.embed_tokens.weight",
            per_layer=False,
        ),
    )

    RELEASE_AFTER = "_fc2_b"

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        inter_pad = _pad_up(cfg.intermediate_size, 128)
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            q_width=cfg.num_attention_heads * cfg.head_dim,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            heads_q=cfg.num_attention_heads,
            num_experts=cfg.num_local_experts,
            vocab=cfg.vocab_size,
            fc1_rows=2 * inter_pad,
            fc1_k_pad=_pad_up(cfg.hidden_size, FC1_K_ALIGN),
            inter_pad=inter_pad,
            fc2_rows_pad=_pad_up(cfg.hidden_size, 128),
            dtype=core.model_config.torch_dtype,
        )


MODEL_WEIGHTS = GptOssWeights()


def _manifest(core) -> dict:
    """The hand-written manifest `ModelWeights.manifest` replaces.

    Dead code, kept for one commit so the equivalence test can compare the
    generated manifest against the thing it is replacing rather than against a
    transcription of it. Deleted once that test has run green.
    """
    cfg = core.model_config.pretrained_config
    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    rows: dict = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        rows[f"l{i}_norm1"] = [(f"{p}.input_layernorm.weight", None, None)]
        rows[f"l{i}_qkv"] = [
            (f"{p}.self_attn.q_proj.weight", (slice(0, q_width),), None),
            (
                f"{p}.self_attn.k_proj.weight",
                (slice(q_width, q_width + kv_width),),
                None,
            ),
            (
                f"{p}.self_attn.v_proj.weight",
                (slice(q_width + kv_width, q_width + 2 * kv_width),),
                None,
            ),
        ]
        rows[f"l{i}_qkv_bias"] = [
            (f"{p}.self_attn.q_proj.bias", (slice(0, q_width),), None),
            (f"{p}.self_attn.k_proj.bias", (slice(q_width, q_width + kv_width),), None),
            (
                f"{p}.self_attn.v_proj.bias",
                (slice(q_width + kv_width, q_width + 2 * kv_width),),
                None,
            ),
        ]
        rows[f"l{i}_sinks"] = [(f"{p}.self_attn.sinks", None, _to_fp32)]
        rows[f"l{i}_o"] = [(f"{p}.self_attn.o_proj.weight", None, None)]
        rows[f"l{i}_o_bias"] = [(f"{p}.self_attn.o_proj.bias", None, None)]
        rows[f"l{i}_norm2"] = [(f"{p}.post_attention_layernorm.weight", None, None)]
        rows[f"l{i}_router"] = [(f"{p}.mlp.router.weight", None, None)]
        rows[f"l{i}_router_bias"] = [(f"{p}.mlp.router.bias", None, None)]
        rows[f"l{i}_fc1_w"] = [(f"{p}.mlp.experts.gate_up_proj_blocks", None, _prep_fc1_weight)]
        rows[f"l{i}_fc1_s"] = [(f"{p}.mlp.experts.gate_up_proj_scales", None, _prep_fc1_scale)]
        rows[f"l{i}_fc1_b"] = [(f"{p}.mlp.experts.gate_up_proj_bias", None, _prep_fc1_bias)]
        rows[f"l{i}_fc2_w"] = [(f"{p}.mlp.experts.down_proj_blocks", None, _prep_fc2_weight)]
        rows[f"l{i}_fc2_s"] = [(f"{p}.mlp.experts.down_proj_scales", None, _prep_fc2_scale)]
        rows[f"l{i}_fc2_b"] = [(f"{p}.mlp.experts.down_proj_bias", None, _prep_fc2_bias)]
    rows["final_norm"] = [("model.norm.weight", None, None)]
    rows["embed"] = [("model.embed_tokens.weight", None, None)]
    return rows
