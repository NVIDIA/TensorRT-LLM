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

"""Run a truncated DeepSeek V4 Flash reference model with real quantized weights.

The DeepSeek V4 Flash checkpoint ships an ``inference/model.py`` reference that
expects TileLang kernels for FP8/FP4 matmuls, sparse attention, and HC mixing.
This script imports that checkpoint model and replaces those kernel entry points
with eager PyTorch emulation so a small, 5-layer model can be loaded directly
from the original safetensor shards and run through ``forward``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import util
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

_DEFAULT_CHECKPOINT_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
    "bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash"
)

_FP4_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


@contextmanager
def _torch_defaults(dtype: torch.dtype, device: torch.device) -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    old_device = torch.get_default_device()
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)


def _missing_kernel(*args, **kwargs):
    raise RuntimeError("DeepSeek V4 kernel stub was called before eager emulation was installed.")


def _import_checkpoint_model(checkpoint_dir: Path):
    inference_dir = checkpoint_dir / "inference"
    model_path = inference_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"DeepSeek V4 reference model not found: {model_path}")

    kernel_stub = types.ModuleType("kernel")
    for name in (
        "act_quant",
        "fp4_act_quant",
        "fp8_gemm",
        "fp4_gemm",
        "sparse_attn",
        "hc_split_sinkhorn",
    ):
        setattr(kernel_stub, name, _missing_kernel)

    old_kernel = sys.modules.get("kernel")
    sys.modules["kernel"] = kernel_stub
    sys.path.insert(0, str(inference_dir))
    try:
        spec = util.spec_from_file_location("deepseek_v4_checkpoint_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {model_path}")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(inference_dir))
        if old_kernel is None:
            sys.modules.pop("kernel", None)
        else:
            sys.modules["kernel"] = old_kernel
    return module


def _maybe_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and scale.dtype == e8m0_dtype:
        fp32_bits = scale.view(torch.uint8).to(torch.int32) << 23
        return fp32_bits.view(torch.float32)
    return scale.to(torch.float32)


def _round_up_power_of_two(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x)))


def _fake_quant_dequant_fp8_activation(
    x: torch.Tensor,
    block_size: int,
    scale_fmt: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"activation K={x.shape[-1]} must be divisible by block_size={block_size}")

    x_shape = x.shape
    x_blocks = (
        x.contiguous().view(-1, x.shape[-1]).float().view(-1, x.shape[-1] // block_size, block_size)
    )
    amax = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True), min=1e-4)
    if scale_fmt is None:
        scale = amax / 448.0
    else:
        scale = _round_up_power_of_two(amax / 448.0)

    q = torch.clamp(x_blocks / scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
    dequant = (q.to(torch.float32) * scale).view(*x_shape)
    return dequant.to(x.dtype), scale.squeeze(-1).view(*x_shape[:-1], x_shape[-1] // block_size)


def _emulated_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    del scale_dtype
    dequant, scale = _fake_quant_dequant_fp8_activation(x, block_size, scale_fmt)
    if inplace:
        x.copy_(dequant)
        return x
    q = torch.clamp(
        x.contiguous().view(-1, x.shape[-1]).float().view(-1, x.shape[-1] // block_size, block_size)
        / scale.view(-1, x.shape[-1] // block_size, 1),
        min=-448.0,
        max=448.0,
    ).to(torch.float8_e4m3fn)
    return q.view(*x.shape), scale


def _cast_to_fp4_indices(x: torch.Tensor) -> torch.Tensor:
    bounds = _FP4_BOUNDS.to(device=x.device)
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.bool, device=x.device)
    x_abs = x.abs()
    ordinal = torch.searchsorted(bounds, x_abs, out_int32=True).to(torch.uint8)
    round_up = (x_abs.unsqueeze(-1) == bounds).logical_and(mask).any(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    return sign + ordinal + round_up


def _fake_quant_dequant_fp4_activation(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"activation K={x.shape[-1]} must be divisible by block_size={block_size}")

    x_shape = x.shape
    x_blocks = (
        x.contiguous().view(-1, x.shape[-1]).float().view(-1, x.shape[-1] // block_size, block_size)
    )
    amax = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True), min=6.0 * (2.0**-126))
    scale = _round_up_power_of_two(amax / 6.0)
    indices = _cast_to_fp4_indices(torch.clamp(x_blocks / scale, min=-6.0, max=6.0))
    values = _FP4_VALUES.to(device=x.device)[indices.long()]
    return (values * scale).view(*x_shape).to(x.dtype), scale.squeeze(-1).view(
        *x_shape[:-1], x_shape[-1] // block_size
    )


def _emulated_fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    dequant, scale = _fake_quant_dequant_fp4_activation(x, block_size)
    if inplace:
        x.copy_(dequant)
        return x

    indices = _cast_to_fp4_indices(
        x.contiguous().view(-1, x.shape[-1]).float().view(-1, x.shape[-1] // block_size, block_size)
        / scale.view(-1, x.shape[-1] // block_size, 1)
    ).view(*x.shape)
    packed = (indices[..., 1::2] << 4) | indices[..., 0::2]
    return packed.contiguous(), scale


def _dequant_fp8_weight(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    rows, cols = weight.shape
    scale_fp32 = _maybe_e8m0_to_fp32(scale)
    scale_rows, scale_cols = scale_fp32.shape
    block_rows = math.ceil(rows / scale_rows)
    block_cols = math.ceil(cols / scale_cols)
    expanded_scale = scale_fp32.repeat_interleave(block_rows, dim=0).repeat_interleave(
        block_cols, dim=1
    )
    return (weight.to(torch.float32) * expanded_scale[:rows, :cols]).to(dtype)


def _dequant_fp4_weight(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    raw = weight.view(torch.uint8)
    rows, packed_cols = raw.shape
    cols = packed_cols * 2
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    indices = torch.empty(rows, cols, dtype=torch.long, device=raw.device)
    indices[..., 0::2] = low.long()
    indices[..., 1::2] = high.long()
    values = _FP4_VALUES.to(device=raw.device)[indices]

    scale_fp32 = _maybe_e8m0_to_fp32(scale)
    if scale_fp32.shape != (rows, cols // 32):
        raise ValueError(
            f"FP4 scale shape {tuple(scale_fp32.shape)} does not match packed weight "
            f"shape {tuple(raw.shape)}; expected {(rows, cols // 32)}"
        )
    expanded_scale = scale_fp32.repeat_interleave(32, dim=1)[:, :cols]
    return (values * expanded_scale).to(dtype)


def _linear_compute_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.dtype
    return torch.bfloat16


def _emulated_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    scale = getattr(weight, "scale", None)
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise RuntimeError("FP8 weight is missing its per-block .scale tensor.")
        compute_dtype = _linear_compute_dtype(x)
        x_dequant, _ = _fake_quant_dequant_fp8_activation(x, 128, "ue8m0")
        weight_dequant = _dequant_fp8_weight(weight, scale, compute_dtype)
        out = F.linear(x_dequant.to(compute_dtype), weight_dequant, bias)
        return out.to(x.dtype)

    if fp4_dtype is not None and weight.dtype == fp4_dtype:
        if scale is None:
            raise RuntimeError("FP4 weight is missing its per-block .scale tensor.")
        compute_dtype = _linear_compute_dtype(x)
        x_dequant, _ = _fake_quant_dequant_fp8_activation(x, 128, "ue8m0")
        weight_dequant = _dequant_fp4_weight(weight, scale, compute_dtype)
        out = F.linear(x_dequant.to(compute_dtype), weight_dequant, bias)
        return out.to(x.dtype)

    return F.linear(x, weight, bias)


def _emulated_grouped_wo_a(o: torch.Tensor, wo_a: nn.Module) -> torch.Tensor:
    scale = getattr(wo_a.weight, "scale", None)
    if wo_a.weight.dtype != torch.float8_e4m3fn or scale is None:
        weight = wo_a.weight.view(o.shape[-2], -1, o.shape[-1])
        return torch.einsum("bsgd,grd->bsgr", o, weight)

    compute_dtype = _linear_compute_dtype(o)
    o_dequant, _ = _fake_quant_dequant_fp8_activation(o, 128, "ue8m0")
    weight_dequant = _dequant_fp8_weight(wo_a.weight, scale, compute_dtype)
    weight = weight_dequant.view(o.shape[-2], -1, o.shape[-1])
    return torch.einsum("bsgd,grd->bsgr", o_dequant.to(compute_dtype), weight).to(o.dtype)


def _emulated_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, _, head_dim = q.shape
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0)
    gather = gather_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    selected_kv = torch.gather(kv.unsqueeze(1).expand(-1, seq_len, -1, -1), 2, gather)
    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected_kv.float()) * softmax_scale
    scores = torch.where(topk_idxs.unsqueeze(2) < 0, torch.full_like(scores, float("-inf")), scores)
    sink = attn_sink.view(1, 1, -1, 1).expand(batch_size, seq_len, -1, -1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs.to(selected_kv.dtype), selected_kv).to(q.dtype)


def _emulated_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb_base = hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = torch.softmax(comb * hc_scale[2] + comb_base, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _emulated_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dim, got {x.shape[-1]}")
    orig_shape = x.shape
    y = x.float().reshape(-1, x.shape[-1])
    width = 1
    while width < y.shape[-1]:
        y = y.view(-1, y.shape[-1] // (2 * width), 2 * width)
        left = y[..., :width].clone()
        right = y[..., width : 2 * width].clone()
        y[..., :width] = left + right
        y[..., width : 2 * width] = left - right
        y = y.view(-1, orig_shape[-1])
        width *= 2
    return (y.view(orig_shape) * (orig_shape[-1] ** -0.5)).to(x.dtype)


def _make_attention_forward(model_module):
    def _attention_forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        model_module.apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        model_module.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        model_module.act_quant(
            kv[..., :-rd], 64, model_module.scale_fmt, model_module.scale_dtype, True
        )
        topk_idxs = model_module.get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = model_module.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = model_module.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = model_module.sparse_attn(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
            )

        model_module.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        o = _emulated_grouped_wo_a(o, self.wo_a)
        return self.wo_b(o.flatten(2))

    return _attention_forward


def _patch_checkpoint_model(model_module) -> None:
    model_module.act_quant = _emulated_act_quant
    model_module.fp4_act_quant = _emulated_fp4_act_quant
    model_module.linear = _emulated_linear
    model_module.sparse_attn = _emulated_sparse_attn
    model_module.hc_split_sinkhorn = _emulated_hc_split_sinkhorn
    model_module.rotate_activation = _emulated_rotate_activation
    model_module.Attention.forward = _make_attention_forward(model_module)


def _convert_wo_a_to_fp8(model: nn.Module) -> None:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        raise RuntimeError("torch.float8_e8m0fnu is required for DeepSeek V4 FP8 scales.")

    for layer in model.layers:
        wo_a = layer.attn.wo_a
        rows, cols = wo_a.weight.shape
        wo_a.weight = nn.Parameter(
            torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=wo_a.weight.device),
            requires_grad=False,
        )
        wo_a.scale = nn.Parameter(
            torch.empty(
                math.ceil(rows / 128),
                math.ceil(cols / 128),
                dtype=e8m0_dtype,
                device=wo_a.weight.device,
            ),
            requires_grad=False,
        )
        wo_a.weight.scale = wo_a.scale


def _prepare_tensor_for_load(key: str, tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if (
        fp4_dtype is not None
        and target.dtype == fp4_dtype
        and tensor.dtype in (torch.int8, torch.uint8)
    ):
        return tensor.view(torch.uint8).view(fp4_dtype)
    if tensor.dtype == torch.float8_e4m3fn and target.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{key} is FP8 in the checkpoint but the instantiated model expects {target.dtype}; "
            "the model patch missed a quantized path."
        )
    return tensor


def _load_checkpoint_subset(model: nn.Module, checkpoint_dir: Path, device: torch.device) -> None:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensor index: {index_path}")

    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    target_state = model.state_dict()
    wanted = {key for key in target_state if key in weight_map}
    missing_from_checkpoint = sorted(set(target_state) - set(weight_map))
    if missing_from_checkpoint:
        raise KeyError(
            "The checkpoint is missing keys required by the truncated model, first few: "
            f"{missing_from_checkpoint[:10]}"
        )

    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key in sorted(wanted):
        keys_by_shard[weight_map[key]].append(key)

    loaded: set[str] = set()
    device_arg = "cpu" if device.type == "cpu" else str(device)
    for shard_name, keys in sorted(keys_by_shard.items()):
        shard_path = checkpoint_dir / shard_name
        chunk = {}
        with safe_open(shard_path, framework="pt", device=device_arg) as f:
            for key in keys:
                tensor = f.get_tensor(key)
                chunk[key] = _prepare_tensor_for_load(key, tensor, target_state[key])
        model.load_state_dict(chunk, strict=False)
        loaded.update(chunk)
        print(f"loaded {len(keys):5d} tensors from {shard_name}")
        del chunk

    missing_after_load = sorted(wanted - loaded)
    if missing_after_load:
        raise RuntimeError(f"Failed to load expected tensors, first few: {missing_after_load[:10]}")


def _load_model_args(checkpoint_dir: Path, layers: int, batch_size: int, seq_len: int):
    config_path = checkpoint_dir / "inference" / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["n_layers"] = layers
    config["n_mtp_layers"] = 0
    config["max_batch_size"] = batch_size
    config["max_seq_len"] = seq_len
    return config


def _make_input_ids(
    args: argparse.Namespace, checkpoint_dir: Path, device: torch.device
) -> torch.Tensor:
    if args.prompt:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[:, : args.seq_len]
        if input_ids.shape[-1] == 0:
            raise ValueError("The prompt produced no tokens.")
        return input_ids.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    return torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_len), generator=generator, device=device
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=_DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--load-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model_module = _import_checkpoint_model(checkpoint_dir)
    _patch_checkpoint_model(model_module)
    model_config = _load_model_args(checkpoint_dir, args.layers, args.batch_size, args.seq_len)
    args.vocab_size = model_config["vocab_size"]

    with _torch_defaults(torch.bfloat16, device):
        model_args = model_module.ModelArgs(**model_config)
        model = model_module.Transformer(model_args).eval()
        _convert_wo_a_to_fp8(model)
        _load_checkpoint_subset(model, checkpoint_dir, device)

        if args.load_only:
            print("load-only requested; skipping forward")
            return

        input_ids = _make_input_ids(args, checkpoint_dir, device)
        with torch.inference_mode():
            logits = model(input_ids, start_pos=0)
            if not torch.isfinite(logits).all():
                raise RuntimeError("forward produced non-finite logits")
            top_values, top_indices = logits.float().topk(args.top_k, dim=-1)

        print(f"input_ids shape: {tuple(input_ids.shape)}")
        print(f"logits shape:    {tuple(logits.shape)}")
        print(f"logits dtype:     {logits.dtype}")
        print(f"top-{args.top_k} token ids for batch 0: {top_indices[0].tolist()}")
        print(f"top-{args.top_k} logits for batch 0:    {top_values[0].tolist()}")


if __name__ == "__main__":
    main()
