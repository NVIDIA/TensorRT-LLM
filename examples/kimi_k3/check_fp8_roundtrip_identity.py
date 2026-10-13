# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Is `ckpt FP8 -> BF16 -> FP8` bitwise identity on the REAL checkpoint?

`test_bf16_roundtrip_is_identity_so_both_fp8_routes_agree` answers yes, but it
builds its "checkpoint" pair with ``per_block_cast_to_fp8`` -- i.e. it assumes
the checkpoint was quantized by the very function the round trip re-applies.
nvidia/Kimi-K3-NVFP4 was quantized by ModelOpt. If the two disagree at all,
dropping the `_K3_CKPT_FP8_ATTR` stash silently changes the MLA weights.

Compares, per MLA projection:
  A  prepare_checkpoint_scale(ckpt_fp8, ckpt_scale)   <- what the stash gave
  B  quantize_weight(dequantize(ckpt_fp8, ckpt_scale)) <- what from_linear does

Writes its verdict to --output; stdout is polluted by the tensorrt_llm import
banner and must not be parsed.
"""

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layers", type=int, default=4, help="how many MLA layers to check")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    from tensorrt_llm._torch.models.modeling_kimi_linear import _FP8_BLOCK
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as FP8Linear,
    )

    index = json.load(open(os.path.join(args.ckpt, "model.safetensors.index.json")))["weight_map"]

    # The four projections the (restored) MLA conversion actually swaps.
    targets = []
    for key in sorted(index):
        if not key.endswith(".weight"):
            continue
        if not any(f".self_attn.{p}." in key for p in ("q_a_proj", "q_b_proj", "o_proj", "g_proj")):
            continue
        if key + "_scale" in index:
            targets.append(key)

    # Spread across layers rather than taking the first N of one layer.
    by_layer: dict[str, list[str]] = {}
    for key in targets:
        by_layer.setdefault(key.split(".self_attn.")[0], []).append(key)
    chosen = [k for layer in sorted(by_layer)[: args.layers] for k in by_layer[layer]]

    lines = [f"checkpoint: {args.ckpt}", f"FP8 MLA projections with a scale: {len(targets)}", ""]
    n_same = n_diff = 0

    for key in chosen:
        shard = index[key]
        with safe_open(os.path.join(args.ckpt, shard), framework="pt") as f:
            w = f.get_tensor(key).cuda()
            s = f.get_tensor(key + "_scale").cuda().float()
        if s.dim() == 4:
            s = s.reshape(s.shape[0], s.shape[2])

        # A: the checkpoint-direct route (what the stash fed from_linear).
        a_w, a_s = FP8Linear.prepare_checkpoint_scale(w, s)

        # B: dequantize then re-quantize (what from_linear does without a stash).
        expanded = s.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
        bf16 = (w.to(torch.float32) * expanded[: w.shape[0], : w.shape[1]]).to(torch.bfloat16)
        b_w, b_s = FP8Linear.quantize_weight(bf16)

        # float8 has almost no ops, so compare its bytes; the prepared scale is
        # int32 and may be non-contiguous, so compare it directly instead.
        a_bytes = a_w.contiguous().view(torch.uint8)
        b_bytes = b_w.contiguous().view(torch.uint8)
        w_same = a_bytes.equal(b_bytes)
        s_same = a_s.shape == b_s.shape and a_s.contiguous().equal(b_s.contiguous())
        if w_same and s_same:
            n_same += 1
            verdict = "IDENTICAL"
        else:
            n_diff += 1
            nbytes = (a_bytes != b_bytes).sum().item() if a_bytes.shape == b_bytes.shape else -1
            total = a_bytes.numel()
            verdict = (
                f"DIFFERS weight_same={w_same} scale_same={s_same} "
                f"bytes_differing={nbytes}/{total} ({100.0 * nbytes / total:.4f}%)"
            )
        lines.append(f"{key}  {tuple(w.shape)}  {verdict}")

    lines += [
        "",
        f"checked {len(chosen)} projections: {n_same} identical, {n_diff} differing",
        "",
        (
            "VERDICT: round trip is bitwise identity on the real checkpoint -- "
            "dropping the stash does NOT change MLA weights."
            if n_diff == 0
            else "VERDICT: round trip is NOT identity on the real checkpoint -- "
            "dropping the stash DOES change MLA weights."
        ),
    ]
    with open(args.output, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
