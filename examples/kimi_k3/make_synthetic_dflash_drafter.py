# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit a synthetic (random-weight) Kimi K3 DFlash/DSpark drafter checkpoint.

The real K3 drafter (training in progress) is a DSpark drafter — DeepSeek's
DFlash follow-up (arXiv 2607.05147): a dense Qwen3-style parallel block
backbone (q/k-norm attention, SiLU MLP) plus the DFlash pooling projection,
extended with a low-rank Markov head (token-conditioned intra-block logit
bias) and a confidence head (per-position acceptance prediction). This
generator materializes that schema — verified against the training team's
dummy-weight checkpoint (dummy-dspark0724, 73 tensors) — so the structural
wiring (DFlashForCausalLM construction via the model_type=qwen3 fallback,
load_weights key remapping, target-layer hidden-state capture in
KimiLinearModel.forward) can be exercised end-to-end before real weights
drop. Outputs are gibberish by construction.

Checkpoint contents (no embed_tokens / lm_head; shared with the target):

* fc.weight                 [hidden, hidden * len(target_layer_ids)]
* hidden_norm.weight        [hidden]
* layers.{i}.self_attn.{q,k,v,o}_proj.weight, {q,k}_norm.weight
* layers.{i}.mlp.{gate,up,down}_proj.weight
* layers.{i}.{input,post_attention}_layernorm.weight
* norm.weight               [hidden]
* markov_w1.weight, markov_w2.weight   [vocab, markov_rank]   (dspark)
* confidence_proj.weight    [1, hidden + markov_rank]         (dspark)
* confidence_proj.bias      [1]                               (dspark)

Three modes:

* --config: adopt a REAL drafter config.json verbatim (authoritative).
* --ckpt-dir: read hidden_size / vocab_size / num_hidden_layers from a
  REAL K3 target checkpoint's config.json; drafter dims default to the
  dummy-dspark0724 drafter's.
* --tiny: minimal dims, no checkpoint access — for unit tests.

target_layer_ids defaults to len-6 even spacing over the target stack —
confirmed by the real config: [1, 19, 37, 54, 72, 90] over K3's 93 layers
(same convention as K2.7's [1, 12, 24, 35, 47, 58] over 61).

Usage:
  python make_synthetic_dflash_drafter.py --config <drafter_cfg> --out <dir>
  python make_synthetic_dflash_drafter.py --ckpt-dir <k3_ckpt> --out <dir>
  python make_synthetic_dflash_drafter.py --tiny --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

# torch/safetensors are imported lazily in main() so the schema helpers
# (drafter_tensor_plan, drafter_config, even_target_layer_ids) stay
# importable on hosts without the container venv.

# nvidia/Kimi-K2.7-Code-DFlash dims (plain DFlash, no dspark heads): kept
# as the schema-compat reference the unit tests pin against.
K27_DRAFTER = dict(
    num_hidden_layers=6,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=18432,
    block_size=8,
)

# Drafter defaults = the K3 DSpark drafter (dummy-dspark0724 config from
# the training team, 2026-07-24).
K3_DRAFTER = dict(
    num_hidden_layers=6,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=12288,
    block_size=8,
    markov_rank=256,
    use_confidence_head=True,
    swa_window_size=1024,
)

TINY = dict(
    hidden_size=64,
    vocab_size=512,
    num_target_layers=8,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    block_size=4,
    markov_rank=8,
    use_confidence_head=True,
    swa_window_size=32,
)


def even_target_layer_ids(num_target_layers: int, k: int = 6) -> list[int]:
    """Evenly spaced capture layers, K2.7 convention (first=1, last=L-3)."""
    lo, hi = 1, max(1, num_target_layers - 3)
    if k == 1:
        return [lo]
    return sorted({round(lo + i * (hi - lo) / (k - 1)) for i in range(k)})


def validate_target_layer_ids(
    target_layer_ids: list[int], num_target_layers: int | None = None
) -> None:
    """Validate capture-layer ids in every mode.

    Uses real exceptions rather than ``assert`` so the checks survive
    ``python -O`` (which strips asserts). ``num_target_layers`` is the
    target stack depth when known (unavailable in --config mode, where the
    target checkpoint is not read); the upper-bound check is skipped when it
    is None.
    """
    if not target_layer_ids:
        raise ValueError("target_layer_ids must be a non-empty list")
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError(f"target_layer_ids {target_layer_ids} contains duplicates")
    if min(target_layer_ids) < 0:
        raise ValueError(f"target_layer_ids {target_layer_ids} contains negative ids")
    if num_target_layers is not None and max(target_layer_ids) >= num_target_layers:
        raise ValueError(
            f"target_layer_ids {target_layer_ids} out of range [0, {num_target_layers})"
        )


def drafter_tensor_plan(
    hidden: int,
    cfg: dict,
    num_capture: int,
    vocab: int | None = None,
    markov_rank: int | None = None,
    use_confidence_head: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Return {key: shape} for the drafter checkpoint.

    Base keys follow the K2.7 DFlash schema; the dspark heads (markov_w1/w2
    and confidence_proj) are added when requested. The confidence head reads
    the concatenation [hidden, markov_features], hence its in-dim.
    """
    heads, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, inter = cfg["head_dim"], cfg["intermediate_size"]
    plan = {
        "fc.weight": (hidden, hidden * num_capture),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    if markov_rank:
        if vocab is None:
            raise ValueError("markov head tensors need vocab_size")
        plan["markov_w1.weight"] = (vocab, markov_rank)
        plan["markov_w2.weight"] = (vocab, markov_rank)
    if use_confidence_head:
        plan["confidence_proj.weight"] = (1, hidden + (markov_rank or 0))
        plan["confidence_proj.bias"] = (1,)
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        plan.update(
            {
                p + "self_attn.q_proj.weight": (heads * hd, hidden),
                p + "self_attn.k_proj.weight": (kv * hd, hidden),
                p + "self_attn.v_proj.weight": (kv * hd, hidden),
                p + "self_attn.o_proj.weight": (hidden, heads * hd),
                p + "self_attn.q_norm.weight": (hd,),
                p + "self_attn.k_norm.weight": (hd,),
                p + "mlp.gate_proj.weight": (inter, hidden),
                p + "mlp.up_proj.weight": (inter, hidden),
                p + "mlp.down_proj.weight": (hidden, inter),
                p + "input_layernorm.weight": (hidden,),
                p + "post_attention_layernorm.weight": (hidden,),
            }
        )
    return plan


def drafter_config(
    hidden: int,
    vocab: int,
    num_target_layers: int,
    target_layer_ids: list[int],
    mask_token_id: int,
    cfg: dict,
) -> dict:
    dflash_cfg = {
        "mask_token_id": mask_token_id,
        "target_layer_ids": target_layer_ids,
    }
    if cfg.get("markov_rank"):
        # DSpark extras, mirroring the dummy-dspark0724 config.
        dflash_cfg.update(
            {
                "use_swa": True,
                "swa_window_size": cfg["swa_window_size"],
                "causal": False,
                "projector_type": "dspark",
                "shift_label": True,
                "markov_rank": cfg["markov_rank"],
                "markov_head_type": "vanilla",
                "use_confidence_head": cfg.get("use_confidence_head", False),
            }
        )
    is_dspark = bool(cfg.get("markov_rank"))
    return {
        "architectures": ["DFlashDraftModel"],
        # model_type drives DFlashForCausalLM's backbone fallback
        # (Qwen3ForCausalLM), matching the K2.7 and K3 drafters.
        "model_type": "qwen3",
        "block_size": cfg["block_size"],
        "dflash_config": dflash_cfg,
        "hidden_size": hidden,
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "intermediate_size": cfg["intermediate_size"],
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "max_position_embeddings": 1048576 if is_dspark else 262144,
        "initializer_range": 0.02,
        "attention_bias": False,
        "attention_dropout": 0.0,
        # RoPE per the dummy-dspark0724 config: plain 1e4 theta, no scaling.
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "num_target_layers": num_target_layers,
        "layer_types": (["sliding_attention"] if is_dspark else ["full_attention"])
        * cfg["num_hidden_layers"],
        **({"sliding_window": cfg["swa_window_size"]} if is_dspark else {}),
        "synthetic_random_weights": True,
    }


def target_dims_from_ckpt(ckpt_dir: str) -> tuple[int, int, int]:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    return (text["hidden_size"], text["vocab_size"], text["num_hidden_layers"])


def drafter_cfg_from_real_config(path: str) -> tuple[dict, dict]:
    """--config mode: adopt the REAL drafter config.json verbatim.

    Random weights, exact real module structure — no schema guessing.
    Returns (real_cfg_dict, drafter_dims_dict). Errors clearly on fields
    the tensor plan needs; anything else in the file is passed through
    untouched so TRT-LLM sees exactly what the trained checkpoint will
    ship.
    """
    with open(path) as f:
        real = json.load(f)
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
    )
    missing = [k for k in required if k not in real]
    if missing:
        raise KeyError(
            f"real drafter config is missing {missing}; "
            "ask the training team for the full HF config.json"
        )
    dflash_cfg = real.get("dflash_config") or {}
    if "target_layer_ids" not in dflash_cfg:
        raise KeyError(
            "real drafter config has no "
            "dflash_config.target_layer_ids — the capture wiring "
            "cannot be derived without it"
        )
    dims = dict(
        num_hidden_layers=real["num_hidden_layers"],
        num_attention_heads=real["num_attention_heads"],
        num_key_value_heads=real["num_key_value_heads"],
        head_dim=real.get("head_dim", real["hidden_size"] // real["num_attention_heads"]),
        intermediate_size=real["intermediate_size"],
        block_size=real.get("block_size", 8),
        # DSpark heads: emitted only if the real config declares them.
        markov_rank=dflash_cfg.get("markov_rank"),
        use_confidence_head=dflash_cfg.get("use_confidence_head", False),
    )
    archs = real.get("architectures", [])
    if not any("Laguna" in a for a in archs) and real.get("model_type") not in ("qwen3", "llama"):
        print(
            f"WARNING: architectures={archs} model_type="
            f"{real.get('model_type')} may not resolve through the "
            "generic DFlashForCausalLM fallback — drafter-side code "
            "changes may be needed. Generating anyway."
        )
    return real, dims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ckpt-dir", help="real K3 target checkpoint dir (reads config.json)")
    mode.add_argument(
        "--config",
        help="REAL drafter config.json from the training team: "
        "adopt it verbatim and emit random weights matching "
        "its exact module structure (no schema guessing)",
    )
    mode.add_argument(
        "--tiny", action="store_true", help="minimal dims for unit tests; no checkpoint access"
    )
    ap.add_argument("--out", required=True, help="output drafter dir")
    ap.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="override capture layers (default: even spacing)",
    )
    ap.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="default: the real config's value in --config mode, "
        "else vocab_size - 2 (NB: the real K3 drafter uses "
        "163606, NOT vocab-2)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    real_cfg = None
    if args.config:
        real_cfg, cfg = drafter_cfg_from_real_config(args.config)
        hidden, vocab = real_cfg["hidden_size"], real_cfg["vocab_size"]
        target_layer_ids = args.target_layer_ids or real_cfg["dflash_config"]["target_layer_ids"]
        # No target checkpoint is read in --config mode, so the stack depth
        # is only known if the real config carries num_target_layers.
        validate_target_layer_ids(target_layer_ids, real_cfg.get("num_target_layers"))
        mask_token_id = (
            args.mask_token_id
            if args.mask_token_id is not None
            else real_cfg["dflash_config"].get("mask_token_id", vocab - 2)
        )
    else:
        if args.tiny:
            cfg = dict(TINY)
            hidden, vocab = cfg["hidden_size"], cfg["vocab_size"]
            num_target_layers = cfg["num_target_layers"]
            default_k = 2
        else:
            hidden, vocab, num_target_layers = target_dims_from_ckpt(args.ckpt_dir)
            cfg = dict(K3_DRAFTER)
            default_k = 6
        target_layer_ids = args.target_layer_ids or even_target_layer_ids(
            num_target_layers, default_k
        )
        validate_target_layer_ids(target_layer_ids, num_target_layers)
        mask_token_id = args.mask_token_id if args.mask_token_id is not None else vocab - 2

    torch.manual_seed(args.seed)
    plan = drafter_tensor_plan(
        hidden,
        cfg,
        len(target_layer_ids),
        vocab=vocab,
        markov_rank=cfg.get("markov_rank"),
        use_confidence_head=cfg.get("use_confidence_head", False),
    )
    tensors = {
        k: (torch.randn(s, dtype=torch.float32) * 0.02).to(torch.bfloat16) for k, s in plan.items()
    }

    if real_cfg is not None:
        out_cfg = dict(real_cfg)
        out_cfg["dflash_config"] = {
            **real_cfg.get("dflash_config", {}),
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids,
        }
        out_cfg["synthetic_random_weights"] = True
    else:
        out_cfg = drafter_config(
            hidden, vocab, num_target_layers, target_layer_ids, mask_token_id, cfg
        )

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    total = sum(t.numel() for t in tensors.values())
    print(
        f"wrote {len(tensors)} tensors ({total / 1e6:.1f}M params) to "
        f"{args.out} (target_layer_ids={target_layer_ids}, "
        f"mask_token_id={mask_token_id}) — SYNTHETIC RANDOM WEIGHTS"
    )


if __name__ == "__main__":
    main()
