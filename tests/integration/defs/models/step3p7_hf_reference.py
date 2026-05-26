#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF reference subprocess for Step3p7 source parity.

Two modes:

* ``--mode layer_outputs`` (fast, default for activation replay):  Builds a
  minimal sub-model containing **only** the requested attention layers (and,
  optionally, one or two MoE layers).  Loads just those weights — typically
  under 2 GB — and runs hidden-state activations through each named layer.
  Writes a ``.pt`` with per-layer activation snapshots so the parity driver
  can compare them element-wise against TensorRT-LLM.

* ``--mode full_forward`` (slow, used by full source_logit_replay / generation
  parity):  Builds the full model, dequantizes every FP8 routed expert tensor
  on GPU per shard, distributes layers across all visible CUDA devices, and
  writes per-prompt final logits / step-by-step argmax tokens.

The FP8 block-scale dequantization is GPU-side, so the dominant cost is the
data movement across PCIe rather than CPU arithmetic — typical full-load
times on 8 B200s should be on the order of minutes rather than the half-hour
the previous CPU-side dequant required.
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# FP8 block-scale dequantization helpers
# ---------------------------------------------------------------------------


def _fp8_e4m3_to_bf16_block(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = 128, device: str = "cuda:0"
) -> torch.Tensor:
    """Dequantize a 2D FP8 e4m3 block-scale tensor to bfloat16 on GPU.

    Moves both ``weight_fp8`` (FP8) and ``scale_inv`` (FP32) to ``device`` and
    performs the expansion + multiply on the GPU.  Returns a bf16 tensor on
    ``device``.  This is ~100x faster than the CPU equivalent for the 288
    routed experts per MoE layer in Step3p7.
    """
    M, K = weight_fp8.shape
    w_dev = weight_fp8.to(device).to(torch.float32)
    s_dev = scale_inv.to(device).to(torch.float32)
    scale_expanded_m = s_dev.repeat_interleave(block, dim=0)[:M, :]
    scale_expanded = scale_expanded_m.repeat_interleave(block, dim=1)[:, :K]
    return (w_dev * scale_expanded).to(torch.bfloat16)


def _maybe_dequant_to_bf16(
    tensor: torch.Tensor, scale: torch.Tensor | None, device: str
) -> torch.Tensor:
    """Move ``tensor`` to ``device``; if FP8 + scale provided, block-dequant."""
    if tensor.dtype != torch.float8_e4m3fn:
        return tensor.to(device).to(torch.bfloat16)
    if scale is None:
        # Cannot dequant without scale; cast bits to bf16 (incorrect math,
        # caller should flag).
        return tensor.to(device).to(torch.bfloat16)
    if tensor.dim() == 2:
        return _fp8_e4m3_to_bf16_block(tensor, scale, device=device)
    if tensor.dim() == 3:
        return torch.stack(
            [
                _fp8_e4m3_to_bf16_block(tensor[i], scale[i], device=device)
                for i in range(tensor.shape[0])
            ]
        )
    return tensor.to(device).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Prompts / device map / config patching
# ---------------------------------------------------------------------------


def _load_prompts(prompts_arg: str) -> list[dict]:
    if prompts_arg and Path(prompts_arg).exists():
        with open(prompts_arg) as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
            if all(isinstance(p, dict) for p in prompts):
                return prompts
            return [{"id": f"p{i}", "prompt": str(p)} for i, p in enumerate(prompts)]
        if isinstance(data, list):
            if all(isinstance(p, dict) for p in data):
                return data
            return [{"id": f"p{i}", "prompt": str(p)} for i, p in enumerate(data)]
    return [
        {"id": "default_p0", "prompt": "The capital of France is"},
        {"id": "default_p1", "prompt": "Q: 2 + 3 = ? A:"},
    ]


def _build_layer_device_map(n_text_layers: int, gpu_ids: list[int]) -> dict:
    device_map = {}
    primary = f"cuda:{gpu_ids[0]}"
    device_map["model.language_model.embed_tokens"] = primary
    device_map["model.language_model.norm"] = primary
    device_map["model.vision_model"] = primary
    device_map["model.vit_large_projector"] = primary
    device_map["lm_head"] = primary
    n_gpus = len(gpu_ids)
    for li in range(n_text_layers):
        device_map[f"model.language_model.layers.{li}"] = (
            f"cuda:{gpu_ids[li * n_gpus // n_text_layers]}"
        )
    return device_map


def _patch_rotary_for_list_theta(step3p7_mod) -> None:
    Step3p7RotaryEmbedding = getattr(step3p7_mod, "Step3p7RotaryEmbedding", None)
    if Step3p7RotaryEmbedding is None or getattr(
        Step3p7RotaryEmbedding, "_step3p7_hf_ref_patched", False
    ):
        return
    _orig_init = Step3p7RotaryEmbedding.__init__

    def _patched_init(self, config, device=None, layer_idx=None):
        import copy as _copy

        if isinstance(getattr(config, "rope_theta", None), list):
            cfg = _copy.copy(config)
            theta_list = cfg.rope_theta
            cfg.rope_theta = float(theta_list[layer_idx])
            rp = getattr(cfg, "rope_parameters", None)
            if isinstance(rp, dict):
                rp = dict(rp)
                if "rope_type" in rp or rp.get("type") == "llama3":
                    rp["rope_theta"] = float(theta_list[layer_idx])
                cfg.rope_parameters = rp
            config = cfg
        return _orig_init(self, config, device=device, layer_idx=layer_idx)

    Step3p7RotaryEmbedding.__init__ = _patched_init
    Step3p7RotaryEmbedding._step3p7_hf_ref_patched = True


def _load_config_and_modeling(checkpoint: str):
    """Return ``(config, step3p7_module, model_class, tokenizer_class)``."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    print(f"[step3p7-hf-ref] loading config from {checkpoint}", flush=True)
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        config.quantization_config = None
    if hasattr(config, "text_config") and hasattr(config.text_config, "torch_dtype"):
        config.text_config.torch_dtype = torch.bfloat16

    auto_map = getattr(config, "auto_map", {})
    model_class_ref = auto_map.get(
        "AutoModelForCausalLM", "modeling_step3p7.Step3p7ForConditionalGeneration"
    )
    model_class = get_class_from_dynamic_module(model_class_ref, checkpoint)
    step3p7_mod = importlib.import_module(model_class.__module__)
    _patch_rotary_for_list_theta(step3p7_mod)
    print(
        "[step3p7-hf-ref] patched Step3p7RotaryEmbedding for list "
        "rope_theta + transformers 5.x rope_parameters",
        flush=True,
    )

    # Tokenizer alignment with TRT-LLM is non-trivial: the checkpoint
    # carries the Mistral regex bug
    # (https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84)
    # plus differences in BPE merges/digit handling under ``use_fast=True``.
    # For short prompts ``fix_mistral_regex=True`` is enough to match
    # TRT-LLM's tokenization, but for longer GSM8K-style prompts the two
    # diverge in numeric token grouping (e.g. "16" as one BPE token vs
    # two digit tokens).  Use TRT-LLM's own ``tokenizer_factory`` here so
    # HF reference always sees the exact same input_ids TRT-LLM does.
    try:
        from tensorrt_llm.tokenizer import tokenizer_factory

        trt_tokenizer = tokenizer_factory(checkpoint, trust_remote_code=True)
        # ``tokenizer_factory`` returns a ``TransformersTokenizer`` wrapper;
        # callers in this file use ``.input_ids`` and ``.decode``, so we
        # surface the inner HF tokenizer to keep call sites unchanged.
        tokenizer = getattr(trt_tokenizer, "tokenizer", trt_tokenizer)
    except ImportError:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return config, step3p7_mod, model_class, tokenizer


# ---------------------------------------------------------------------------
# Layer-only (fast) mode: minimal HF reference for activation replay
# ---------------------------------------------------------------------------


def _shard_grouped_keys(shard_keys: list[str]) -> dict[str, dict[str, str]]:
    """Group keys by base path so weight + weight_scale_inv pair up."""
    grouped: dict[str, dict[str, str]] = {}
    for k in shard_keys:
        if k.endswith(".weight_scale_inv"):
            grouped.setdefault(k[: -len("_scale_inv")], {})["scale"] = k
        elif k.endswith(".weight"):
            grouped.setdefault(k, {})["main"] = k
        else:
            grouped[k] = {"main": k}
    return grouped


_RENAME_RULES = [
    (re.compile(r"^vision_model"), "model.vision_model"),
    (re.compile(r"^model(?!\.(language_model|vision_model))"), "model.language_model"),
    (re.compile(r"^vit_large_projector"), "model.vit_large_projector"),
]


def _remap_key(k: str) -> str:
    for pat, repl in _RENAME_RULES:
        new = pat.sub(repl, k, count=1)
        if new != k:
            return new
    return k


def _matches_layer_target(
    target_k: str, requested_layer_idxs: list[int], include_global: bool
) -> bool:
    """Return True if the target key belongs to a requested layer or global module."""
    if include_global:
        if target_k.startswith("model.language_model.embed_tokens"):
            return True
        if target_k.startswith("model.language_model.norm"):
            return True
        if target_k.startswith("lm_head"):
            return True
    m = re.match(r"^model\.language_model\.layers\.(\d+)\.", target_k)
    if m:
        return int(m.group(1)) in requested_layer_idxs
    return False


def _materialize_module_param(
    model: torch.nn.Module, target_k: str, tensor_dev_bf: torch.Tensor
) -> None:
    """Install ``tensor_dev_bf`` as the parameter at ``target_k`` on ``model``."""
    parts = target_k.split(".")
    mod_obj = model
    for p in parts[:-1]:
        mod_obj = getattr(mod_obj, p)
    attr = parts[-1]
    existing = getattr(mod_obj, attr, None)
    if existing is None:
        return
    new_param = torch.nn.Parameter(tensor_dev_bf, requires_grad=False)
    setattr(mod_obj, attr, new_param)


def _zero_remaining_meta(model: torch.nn.Module, primary: str) -> None:
    """Replace any remaining meta-device parameters/buffers with zeros."""
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            stub = torch.zeros_like(p, device=primary)
            parts = name.split(".")
            mod_obj = model
            for pp in parts[:-1]:
                mod_obj = getattr(mod_obj, pp)
            setattr(mod_obj, parts[-1], torch.nn.Parameter(stub, requires_grad=False))
    for name, b in model.named_buffers():
        if b.device.type == "meta":
            stub = torch.zeros_like(b, device=primary)
            parts = name.split(".")
            mod_obj = model
            for pp in parts[:-1]:
                mod_obj = getattr(mod_obj, pp)
            mod_obj.register_buffer(parts[-1], stub, persistent=False)


def _load_partial_state_dict(
    checkpoint: str, model: torch.nn.Module, requested_layer_idxs: list[int], primary_device: str
) -> tuple[int, int]:
    """Load weights only for the requested decoder layers + global modules.

    Returns (text_keys_filled, fp8_blocks_dequantized).
    """
    import safetensors.torch

    target_keys = set(model.state_dict().keys())
    files = sorted(glob.glob(os.path.join(checkpoint, "model-*.safetensors")))
    text_keys_filled = 0
    fp8_blocks_dequantized = 0
    for f in files:
        with safetensors.torch.safe_open(f, framework="pt") as h:
            grouped = _shard_grouped_keys(list(h.keys()))
            for base, paths in grouped.items():
                k = paths.get("main")
                if k is None:
                    continue
                target_k = _remap_key(k)
                if target_k not in target_keys:
                    continue
                if not _matches_layer_target(target_k, requested_layer_idxs, include_global=True):
                    continue
                w = h.get_tensor(k)
                scale = h.get_tensor(paths["scale"]) if "scale" in paths else None
                bf = _maybe_dequant_to_bf16(w, scale, primary_device)
                if scale is not None:
                    fp8_blocks_dequantized += 1
                _materialize_module_param(model, target_k, bf)
                text_keys_filled += 1
        gc.collect()
        torch.cuda.empty_cache()
    return text_keys_filled, fp8_blocks_dequantized


def _layer_outputs_mode(args, config, step3p7_mod, model_class, tokenizer) -> int:
    """Fast HF reference: load only requested layers, capture activations."""
    requested_layer_idxs = sorted(int(x) for x in args.capture_layers.split(",") if x.strip())
    # The manual forward below runs sequentially from layer 0 to
    # max(requested_layer_idxs).  We need EVERY layer in that range loaded —
    # not just the requested ones — otherwise un-loaded layers on the meta
    # device fail with "Tensor on device meta is not on the expected device".
    max_li = max(requested_layer_idxs) if requested_layer_idxs else 0
    layers_to_load = list(range(0, max_li + 1))
    print(
        f"[step3p7-hf-ref] layer_outputs mode, "
        f"capture_layers={requested_layer_idxs}, "
        f"loading layers 0..{max_li} ({len(layers_to_load)} total)",
        flush=True,
    )

    primary = "cuda:0"
    # Construct on meta, then materialize requested layers + globals.
    print(f"[step3p7-hf-ref] constructing {model_class.__name__} on meta (layer-only)", flush=True)
    with torch.device("meta"):
        model = model_class(config)
    text_keys_filled, fp8_count = _load_partial_state_dict(
        args.checkpoint, model, layers_to_load, primary
    )
    # capture_layers stays as the requested set (we only install hooks on those).
    layer_idxs = requested_layer_idxs
    print(
        f"[step3p7-hf-ref] partial load: {text_keys_filled} tensors ({fp8_count} FP8-dequantized).",
        flush=True,
    )
    # NOTE: layer_outputs mode intentionally does NOT call _zero_remaining_meta.
    # Doing so would materialize ~190 GiB of empty tensors for the un-loaded
    # layers (45 decoder layers, of which we only use the first few).  Instead
    # we just only step through ``layers[:max_li + 1]`` manually below, never
    # touching meta-device params for the un-loaded layers.
    # Re-initialize ``rotary_emb.inv_freq`` on real GPU for ALL loaded layers
    # (the Step3p7RotaryEmbedding constructor ran on meta, so inv_freq is
    # meta).  Note: we re-init the rotary buffer for every layer we need to
    # forward through, not only those we capture from.
    text_model = model.model.language_model
    for li in layers_to_load:
        if li >= len(text_model.layers):
            continue
        layer = text_model.layers[li]
        rot = getattr(layer.self_attn, "rotary_emb", None)
        if rot is None:
            continue
        new_inv, _ = rot.rope_init_fn(rot.config, torch.device(primary))
        # ``register_buffer`` replaces the existing meta buffer.
        rot.register_buffer("inv_freq", new_inv.to(primary), persistent=False)
        rot.original_inv_freq = rot.inv_freq
    model.eval()

    prompts_dicts = _load_prompts(args.prompts)
    prompts = [p["prompt"] for p in prompts_dicts]

    out: dict = {
        "mode": "layer_outputs",
        "prompts": prompts,
        "prompt_ids": [p.get("id", f"p{i}") for i, p in enumerate(prompts_dicts)],
        "capture_layers": layer_idxs,
        "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
        "input_ids": [],
        "activations": {},  # key (layer_idx, tag) -> list per prompt
    }

    # Resolve the text decoder module path.
    text_model = model.model.language_model

    # Install forward hooks on the requested layers.  We capture:
    #   * post_input_ln       (pre-self_attn input to attention)
    #   * attn_q/k/v post Q/K-norm (pre-RoPE)
    #   * attn_output_post_gate (post head-wise gate, post o_proj)
    #
    # The HF source attention.forward does q_norm/k_norm first, then RoPE,
    # then attention, then gate, then o_proj.  We can install hooks on
    # q_norm/k_norm and on the attention module forward output.
    captures = {layer_idx: {} for layer_idx in layer_idxs}

    def _save(layer_idx: int, tag: str, tensor: torch.Tensor) -> None:
        captures[layer_idx].setdefault(tag, []).append(tensor.detach().to("cpu").to(torch.float32))

    def _attn_pre_hook(layer_idx: int):
        def hook(module, inputs, kwargs):
            # inputs[0] = hidden_states (post input_layernorm)
            h = inputs[0] if inputs else kwargs.get("hidden_states")
            if isinstance(h, torch.Tensor):
                _save(layer_idx, "attn_input_post_ln", h[0])

        return hook

    def _attn_post_hook(layer_idx: int):
        def hook(module, inputs, output):
            # attention output = (attn_output, attn_weights)
            if isinstance(output, tuple):
                a = output[0]
            else:
                a = output
            if isinstance(a, torch.Tensor):
                _save(layer_idx, "attn_output_post_o_proj", a[0])

        return hook

    def _qnorm_post_hook(layer_idx: int, which: str):
        def hook(module, inputs, output):
            _save(layer_idx, f"{which}_norm_out", output[0])

        return hook

    def _gate_post_hook(layer_idx: int):
        def hook(module, inputs, output):
            # g_proj(hidden_states) raw output (before sigmoid).
            _save(layer_idx, "g_proj_raw", output[0])

        return hook

    handles = []
    for li in layer_idxs:
        if li >= len(text_model.layers):
            continue
        layer = text_model.layers[li]
        handles.append(
            layer.self_attn.register_forward_pre_hook(_attn_pre_hook(li), with_kwargs=True)
        )
        handles.append(layer.self_attn.register_forward_hook(_attn_post_hook(li)))
        if hasattr(layer.self_attn, "q_norm"):
            handles.append(layer.self_attn.q_norm.register_forward_hook(_qnorm_post_hook(li, "q")))
        if hasattr(layer.self_attn, "k_norm"):
            handles.append(layer.self_attn.k_norm.register_forward_hook(_qnorm_post_hook(li, "k")))
        if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
            handles.append(layer.self_attn.g_proj.register_forward_hook(_gate_post_hook(li)))

    # If MoE layers are requested, capture router logits + MoE output
    # post-residual (sum of routed + shared) at the decoder-layer level.
    def _moe_post_hook(layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                _save(layer_idx, "moe_layer_output", output[0])

        return hook

    def _gate_router_hook(layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                _save(layer_idx, "router_logits", output[0])

        return hook

    def _share_expert_hook(layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                _save(layer_idx, "shared_expert_output", output[0])

        return hook

    def _moe_routed_hook(layer_idx: int):
        """Capture the MoE module's own forward output.

        ``= routed + shared`` in HF source so we can compare the pre-residual
        MoE sum vs TRT-LLM's ``post_moe_plus_shared``.
        """

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                _save(layer_idx, "moe_module_output", output[0])

        return hook

    for li in layer_idxs:
        if li >= len(text_model.layers):
            continue
        layer = text_model.layers[li]
        if getattr(layer, "use_moe", False):
            handles.append(layer.register_forward_hook(_moe_post_hook(li)))
            if hasattr(layer, "moe"):
                handles.append(layer.moe.register_forward_hook(_moe_routed_hook(li)))
                if hasattr(layer.moe, "gate"):
                    handles.append(layer.moe.gate.register_forward_hook(_gate_router_hook(li)))
            if hasattr(layer, "share_expert"):
                handles.append(layer.share_expert.register_forward_hook(_share_expert_hook(li)))

    # Run forward (only up through the last requested layer is enough, but
    # the model's text forward doesn't support partial layer execution
    # easily; just run the whole text model — empty layers return zeros).
    embed_dev = next(text_model.embed_tokens.parameters()).device
    n_compared = 0
    for prompt_idx, prompt in enumerate(prompts):
        t0 = time.time()
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(embed_dev)
        if ids.shape[1] > args.max_prompt_tokens:
            ids = ids[:, : args.max_prompt_tokens]
        out["input_ids"].append(ids[0].tolist())

        # Run forward through just the embedding + the first
        # (max layer idx) + 1 layers manually.  This is faster than the
        # full forward because we exit early after the last requested
        # capture and avoids accessing un-loaded later layers.
        with torch.no_grad():
            hidden_states = text_model.embed_tokens(ids)
            cache_position = torch.arange(ids.shape[1], device=embed_dev)
            position_ids = cache_position.unsqueeze(0)
            mask_kwargs = {
                "config": text_model.config,
                "attention_mask": None,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            mask_kwargs[step3p7_mod._MASK_INPUT_EMBEDS_ARG] = hidden_states
            causal_mask_mapping = {
                "full_attention": step3p7_mod.create_causal_mask(**mask_kwargs),
            }
            if getattr(text_model, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = (
                    step3p7_mod.create_sliding_window_causal_mask(**mask_kwargs)
                )
            max_li = max(layer_idxs)
            for li in range(max_li + 1):
                if li >= len(text_model.layers):
                    break
                layer = text_model.layers[li]
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=None,
                    cache_position=cache_position,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
        n_compared += 1
        print(
            f"[step3p7-hf-ref] layer_outputs p{prompt_idx} "
            f"prompt_len={ids.shape[1]} took {time.time() - t0:.2f}s",
            flush=True,
        )

    for h in handles:
        h.remove()

    # Convert captures dict to a serializable form.
    serialized = {}
    for li, tag_lists in captures.items():
        for tag, tensors in tag_lists.items():
            serialized[f"layer_{li}::{tag}"] = tensors

    out["activations"] = serialized
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(
        f"[step3p7-hf-ref] saved layer_outputs reference to {args.output} "
        f"({n_compared} prompts, "
        f"{len(serialized)} activation keys).",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Full forward mode (slow, full model)
# ---------------------------------------------------------------------------


def _load_full_state_dict_gpu_dequant(
    checkpoint: str, model: torch.nn.Module, device_map: dict, gpu_ids: list[int]
) -> tuple[int, int, int]:
    """Stream every checkpoint shard, GPU-dequantize FP8, materialize params."""
    import safetensors.torch

    target_keys = set(model.state_dict().keys())
    n_text_layers = int(getattr(model.config.text_config, "num_hidden_layers", 45))
    n_mtp = int(getattr(model.config.text_config, "num_nextn_predict_layers", 0))
    mtp_layer_idxs = set(range(n_text_layers, n_text_layers + n_mtp))

    def is_ignored(k: str) -> bool:
        for li in mtp_layer_idxs:
            if f"model.layers.{li}." in k or f"model.language_model.layers.{li}." in k:
                return True
        return False

    files = sorted(glob.glob(os.path.join(checkpoint, "model-*.safetensors")))
    text_keys_filled = 0
    skipped = 0
    fp8_blocks_dequantized = 0
    primary = f"cuda:{gpu_ids[0]}"
    for f in files:
        with safetensors.torch.safe_open(f, framework="pt") as h:
            grouped = _shard_grouped_keys(list(h.keys()))
            for base, paths in grouped.items():
                k = paths.get("main")
                if k is None:
                    continue
                if is_ignored(k):
                    skipped += 1
                    continue
                target_k = _remap_key(k)
                if target_k not in target_keys:
                    skipped += 1
                    continue
                # Resolve target device per the device_map.
                target_dev = device_map.get(".".join(target_k.split(".")[:-1]))
                if target_dev is None:
                    candidates = [d for d in device_map if target_k.startswith(d + ".")]
                    if candidates:
                        target_dev = device_map[max(candidates, key=len)]
                    else:
                        target_dev = primary
                w = h.get_tensor(k)
                scale = h.get_tensor(paths["scale"]) if "scale" in paths else None
                # Dequant on the target GPU directly to avoid a round trip.
                bf = _maybe_dequant_to_bf16(w, scale, target_dev)
                if scale is not None:
                    fp8_blocks_dequantized += 1
                _materialize_module_param(model, target_k, bf)
                text_keys_filled += 1
        gc.collect()
        torch.cuda.empty_cache()
    return text_keys_filled, fp8_blocks_dequantized, skipped


def _full_forward_mode(args, config, step3p7_mod, model_class, tokenizer) -> int:
    from accelerate import dispatch_model

    n_visible = torch.cuda.device_count()
    if n_visible == 0:
        raise RuntimeError("HF reference requires CUDA; saw 0 devices.")
    gpu_ids = list(range(n_visible))
    n_text_layers = int(getattr(config.text_config, "num_hidden_layers", 45))
    device_map = _build_layer_device_map(n_text_layers, gpu_ids)
    print(
        f"[step3p7-hf-ref] device_map across {n_visible} GPUs ({len(device_map)} module groups)",
        flush=True,
    )

    print(f"[step3p7-hf-ref] constructing {model_class.__name__} on meta", flush=True)
    with torch.device("meta"):
        model = model_class(config)

    t0 = time.time()
    text_keys, fp8_dequant, skipped = _load_full_state_dict_gpu_dequant(
        args.checkpoint, model, device_map, gpu_ids
    )
    print(
        f"[step3p7-hf-ref] loaded {text_keys} text-path tensors "
        f"({fp8_dequant} FP8 block-dequantized); skipped {skipped}. "
        f"took {time.time() - t0:.1f}s",
        flush=True,
    )

    # Re-initialize ``rotary_emb.inv_freq`` for every decoder layer BEFORE
    # ``_zero_remaining_meta`` overwrites them.  Step3p7RotaryEmbedding's
    # constructor ran under the meta-device context and only set ``inv_freq``
    # symbolically, so the real values must be computed here.  If we let
    # ``_zero_remaining_meta`` zero ``inv_freq``, every token is treated as
    # if it were at position 0 — the model loses positional information and
    # arithmetic answers collapse to "0"/"2"/etc. regardless of operands.
    primary_dev = f"cuda:{gpu_ids[0]}"
    text_model = model.model.language_model
    for li in range(len(text_model.layers)):
        layer = text_model.layers[li]
        rot = getattr(layer.self_attn, "rotary_emb", None)
        if rot is None:
            continue
        new_inv, _ = rot.rope_init_fn(rot.config, torch.device(primary_dev))
        rot.register_buffer("inv_freq", new_inv.to(primary_dev), persistent=False)
        rot.original_inv_freq = rot.inv_freq

    _zero_remaining_meta(model, primary_dev)
    model.eval()
    model = dispatch_model(model, device_map=device_map)
    print("[step3p7-hf-ref] dispatch_model installed cross-device hooks", flush=True)

    prompts_dicts = _load_prompts(args.prompts)
    prompts = [p["prompt"] for p in prompts_dicts]
    out: dict = {
        "mode": "full_forward",
        "prompts": prompts,
        "prompt_ids": [p.get("id", f"p{i}") for i, p in enumerate(prompts_dicts)],
        "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
        "top_k": args.top_k,
        "input_ids": [],
        "logit_argmax": [],
        "logit_topk_ids": [],
        "logit_topk_values": [],
        "generated_token_ids": [],
        "per_step_logit_argmax": [],
        # Per-step top-K logits captured during the manual greedy loop so
        # the parity driver can emit per-step max_abs / cosine metrics for
        # criterion #7 (generation_parity).
        "per_step_logit_topk_ids": [],
        "per_step_logit_topk_values": [],
        "device_summary": torch.cuda.get_device_name(0),
        # Per-layer cumulative-output activations (only filled when
        # ``--capture-all-layer-outputs`` is set).  Tag form
        # ``layer_<i>::layer_output`` matches the HF decoder layer's
        # post-residual output, i.e. the same tensor TRT-LLM saves as
        # ``layer_<i>_moe_layer_output``.  Also captures
        # ``layer_<i>::attn_input_post_ln`` (post-input_layernorm input
        # to self_attn) and ``layer_<i>::attn_output_post_o_proj``
        # (self_attn return value) for finer-grained localization.
        "activations": {},
    }
    capture_all_layers = bool(getattr(args, "capture_all_layer_outputs", False))
    captures_all: dict = {}

    def _save_all(tag: str, tensor: torch.Tensor) -> None:
        captures_all.setdefault(tag, []).append(tensor.detach().to("cpu").to(torch.float32))

    handles_all: list = []
    if capture_all_layers:
        text_model = model.model.language_model
        n_layers = len(text_model.layers)
        print(
            f"[step3p7-hf-ref] installing per-layer-output hooks on {n_layers} decoder layers",
            flush=True,
        )

        def _layer_output_hook(layer_idx: int):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    _save_all(f"layer_{layer_idx}::layer_output", output[0])

            return hook

        def _attn_pre_hook_all(layer_idx: int):
            def hook(module, inputs, kwargs):
                h = inputs[0] if inputs else kwargs.get("hidden_states")
                if isinstance(h, torch.Tensor):
                    _save_all(f"layer_{layer_idx}::attn_input_post_ln", h[0])

            return hook

        def _attn_post_hook_all(layer_idx: int):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    a = output[0]
                else:
                    a = output
                if isinstance(a, torch.Tensor):
                    _save_all(f"layer_{layer_idx}::attn_output_post_o_proj", a[0])

            return hook

        for li in range(n_layers):
            layer = text_model.layers[li]
            handles_all.append(layer.register_forward_hook(_layer_output_hook(li)))
            handles_all.append(
                layer.self_attn.register_forward_pre_hook(_attn_pre_hook_all(li), with_kwargs=True)
            )
            handles_all.append(layer.self_attn.register_forward_hook(_attn_post_hook_all(li)))

        # Also capture the embed-token output and final-norm output so the
        # comparison can attribute drift to embedding, layer 0..N-1, final
        # norm, or LM head.
        def _embed_post_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                _save_all("embed_tokens::output", output[0])

        def _norm_post_hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                _save_all("final_norm::output", output[0])

        handles_all.append(text_model.embed_tokens.register_forward_hook(_embed_post_hook))
        handles_all.append(text_model.norm.register_forward_hook(_norm_post_hook))

    embed_dev = next(model.model.language_model.embed_tokens.parameters()).device
    for prompt_idx, prompt in enumerate(prompts):
        t1 = time.time()
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(embed_dev)
        if ids.shape[1] > args.max_prompt_tokens:
            ids = ids[:, : args.max_prompt_tokens]
        out["input_ids"].append(ids[0].tolist())
        with torch.no_grad():
            try:
                fwd_out = model(input_ids=ids, use_cache=False)
            except TypeError:
                fwd_out = model.forward(input_ids=ids, use_cache=False)
        logits = fwd_out.logits if hasattr(fwd_out, "logits") else fwd_out[0]
        last = logits[0, -1].float().cpu()
        topk_count = max(1, int(args.top_k))
        topk_vals, topk_ids = torch.topk(last, topk_count)
        out["logit_argmax"].append(int(topk_ids[0]))
        out["logit_topk_ids"].append(topk_ids[: args.top_k].tolist())
        out["logit_topk_values"].append([float(x) for x in topk_vals[: args.top_k].tolist()])
        # Manual greedy-decoding loop.  Now that rotary_emb.inv_freq is
        # re-initialised before _zero_remaining_meta, the per-step forward
        # produces the correct logits (the previous "step 1 predicts '0'"
        # symptom was caused by RoPE inv_freq being zeroed; with that
        # fixed the model knows positional ordering and arithmetic works).
        # ``model.generate()`` is harder to wire here because it depends
        # on multimodal config defaults that conflict with our text-only
        # forward, so the explicit loop is the more reliable path.
        gen_ids = ids.clone()
        per_step_argmax: list[int] = []
        per_step_topk_ids: list[list[int]] = []
        per_step_topk_values: list[list[float]] = []
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                try:
                    step_out = model(input_ids=gen_ids, use_cache=False)
                except Exception as e:
                    print(f"[step3p7-hf-ref] generation step failed: {e}", flush=True)
                    break
                step_logits = step_out.logits if hasattr(step_out, "logits") else step_out[0]
                last = step_logits[0, -1].float().cpu()
                topk_vals, topk_ids = torch.topk(last, topk_count)
                next_token = int(topk_ids[0])
                per_step_argmax.append(next_token)
                per_step_topk_ids.append(topk_ids[: args.top_k].tolist())
                per_step_topk_values.append([float(x) for x in topk_vals[: args.top_k].tolist()])
                gen_ids = torch.cat(
                    [gen_ids, torch.tensor([[next_token]], device=embed_dev)], dim=1
                )
        out["per_step_logit_argmax"].append(per_step_argmax)
        out["per_step_logit_topk_ids"].append(per_step_topk_ids)
        out["per_step_logit_topk_values"].append(per_step_topk_values)
        out["generated_token_ids"].append(gen_ids[0, ids.shape[1] :].tolist())
        print(
            f"[step3p7-hf-ref] full_forward p{prompt_idx} "
            f"prompt_len={ids.shape[1]} took {time.time() - t1:.1f}s",
            flush=True,
        )
        # When capturing all-layer activations, only keep the FIRST prompt's
        # captures.  Cumulative replay only needs one prompt; per-prompt
        # accumulation across 45 layers x 3 tags x N prompts can blow up host
        # memory and serialization size.
        if capture_all_layers and prompt_idx == 0:
            out["activations"] = dict(captures_all)
            print(
                f"[step3p7-hf-ref] saved per-layer activations: "
                f"{len(out['activations'])} tags from prompt 0",
                flush=True,
            )

    for h in handles_all:
        try:
            h.remove()
        except Exception:
            pass

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"[step3p7-hf-ref] saved full_forward reference to {args.output}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step3p7 HF reference subprocess")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompts", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["layer_outputs", "full_forward"], default="layer_outputs")
    p.add_argument(
        "--capture-layers",
        default="0,1",
        help="Comma-separated layer indices for layer_outputs mode.",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument(
        "--capture-all-layer-outputs",
        action="store_true",
        help="In full_forward mode, also capture each decoder "
        "layer's post-residual output (layer_<i>::layer_output) "
        "plus embed/final-norm outputs for cumulative parity "
        "comparison vs TRT-LLM.",
    )
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("[step3p7-hf-ref] ERROR: CUDA not available", flush=True)
        return 2
    print(f"[step3p7-hf-ref] mode={args.mode}", flush=True)
    config, step3p7_mod, model_class, tokenizer = _load_config_and_modeling(args.checkpoint)
    if args.mode == "layer_outputs":
        return _layer_outputs_mode(args, config, step3p7_mod, model_class, tokenizer)
    return _full_forward_mode(args, config, step3p7_mod, model_class, tokenizer)


if __name__ == "__main__":
    sys.exit(main())
