# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diff two MiniMax-M3 layer-activation dumps and report where they diverge.

Consumes two directories written by dump_layers_one_request.py and prints, per
forward step, the relative error of every dumped tensor in layer order. The
first layer whose error jumps well above the surrounding noise floor is where
the divergence originates; a step-0 (prefill) match with a step-1 (first decode)
mismatch localizes the fault to the decode path instead.

    python examples/minimax_m3/compare_layer_dumps.py /tmp/m3_fp8 /tmp/m3_nvfp4

Tensors are compared per rank. --threshold sets the relative error that counts
as divergence for the summary line; the full table is always printed unless
--only-diverging is passed.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import torch

_NAME_RE = re.compile(
    r"^step(?P<step>\d+)_tok(?P<tokens>\d+)"
    r"_layer(?P<layer>-?\d+)_(?P<tag>.+)_rank(?P<rank>\d+)\.pt$"
)


def _index(directory: str) -> dict:
    entries = {}
    for name in os.listdir(directory):
        match = _NAME_RE.match(name)
        if match is None:
            continue
        key = (
            int(match["step"]),
            int(match["layer"]),
            match["tag"],
            int(match["rank"]),
        )
        entries[key] = (os.path.join(directory, name), int(match["tokens"]))
    if not entries:
        raise SystemExit(
            f"no dump files matched in {directory} (expected "
            "step<N>_tok<N>_layer<N>_<tag>_rank<N>.pt)"
        )
    return entries


def _format_values(values: list) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"


def _compare_logits(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Compare two logit tensors by the decision the sampler would make.

    Relative error on logits is a poor guide on its own: a large error that
    leaves the ranking intact costs nothing, while a small one that closes a
    narrow top-1/top-2 gap changes the generated token. Reported instead are
    how often the argmax agrees and how much headroom the reference had.
    """
    if left.shape != right.shape:
        return {"shape_mismatch": (tuple(left.shape), tuple(right.shape))}
    rows_left = left.reshape(-1, left.shape[-1]).double()
    rows_right = right.reshape(-1, right.shape[-1]).double()
    top2 = rows_left.topk(2, dim=-1).values
    agree = rows_left.argmax(-1) == rows_right.argmax(-1)
    return {
        "rows": int(rows_left.shape[0]),
        "argmax_agree": agree.double().mean().item(),
        "ref_margin": (top2[:, 0] - top2[:, 1]).mean().item(),
        "max_abs": (rows_left - rows_right).abs().max().item(),
    }


def _compare_selection(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Compare two top-k block-selection tables from the sparse indexer.

    The last axis is the top-k list for one (token, head), whose order carries
    no meaning, so the sets are compared by sorting each row. Negative entries
    are the padding used for rows with fewer valid blocks than k, and sort to
    the front consistently on both sides.
    """
    if left.shape != right.shape:
        return {"shape_mismatch": (tuple(left.shape), tuple(right.shape))}
    rows_left = left.reshape(-1, left.shape[-1]).to(torch.int64).sort(dim=-1).values
    rows_right = right.reshape(-1, right.shape[-1]).to(torch.int64).sort(dim=-1).values
    differing = rows_left != rows_right
    return {
        "row_mismatch": differing.any(dim=-1).double().mean().item(),
        "changed_per_row": differing.sum(dim=-1).double().mean().item(),
        "topk": int(left.shape[-1]),
        "rows": int(rows_left.shape[0]),
    }


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Summarize the difference between two activations from the same position.

    The denominator is the larger of the two norms, so a tensor that is zero on
    one side and populated on the other reports 1.0 rather than an infinity that
    hides its magnitude. Both norms are returned because a one-sided zero means
    something quite different from a numerical drift.
    """
    if left.shape != right.shape:
        return {"shape_mismatch": (tuple(left.shape), tuple(right.shape))}
    left = left.flatten().double()
    right = right.flatten().double()
    left_norm = left.norm().item()
    right_norm = right.norm().item()
    scale = max(left_norm, right_norm)
    delta = left - right
    return {
        "rel_l2": (delta.norm().item() / scale) if scale > 0 else 0.0,
        "max_abs": delta.abs().max().item(),
        "ref_norm": left_norm,
        "cand_norm": right_norm,
        "nonfinite": int((~torch.isfinite(left)).sum() + (~torch.isfinite(right)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=None,
                        help="restrict to one rank (default: all)")
    parser.add_argument("--only-diverging", action="store_true")
    args = parser.parse_args()

    reference = _index(args.reference_dir)
    candidate = _index(args.candidate_dir)
    shared = sorted(set(reference) & set(candidate))
    if args.rank is not None:
        shared = [key for key in shared if key[3] == args.rank]
    if not shared:
        raise SystemExit("the two dumps share no (step, layer, tag, rank) keys")

    missing = sorted(set(reference) ^ set(candidate))
    if missing:
        print(f"note: {len(missing)} keys present in only one dump, e.g. {missing[:3]}\n")

    first_diverging = {}
    by_step = defaultdict(list)
    for key in shared:
        by_step[key[0]].append(key)

    for step in sorted(by_step):
        tokens = {reference[key][1] for key in by_step[step]} | {
            candidate[key][1] for key in by_step[step]
        }
        if len(tokens) > 1:
            print(f"===== step {step}: SKIPPED, token counts disagree "
                  f"({sorted(tokens)}); the two runs are not aligned =====\n")
            continue
        token_count = tokens.pop()
        print(f"===== step {step} "
              f"({'prefill' if step == 0 else f'decode {step}'}, "
              f"{token_count} token(s)) =====")
        print(f"{'layer':>6} {'tag':<24} {'rank':>4} {'rel_l2':>11} "
              f"{'max_abs':>11} {'ref_norm':>11} {'cand_norm':>11} {'nonfin':>7}")
        selection_rows = []
        logit_rows = []
        scalar_rows = []
        for key in sorted(by_step[step], key=lambda k: (k[1], k[2], k[3])):
            left = torch.load(reference[key][0], map_location="cpu")
            right = torch.load(candidate[key][0], map_location="cpu")
            if not left.is_floating_point():
                selection_rows.append((key, _compare_selection(left, right)))
                continue
            if key[2] == "logits":
                logit_rows.append((key, _compare_logits(left, right)))
                continue
            if left.numel() <= 8:
                # Per-layer quantization scales, worth reading as values.
                scalar_rows.append((key, left.flatten().tolist(), right.flatten().tolist()))
                continue
            stats = _compare(left, right)
            if "shape_mismatch" in stats:
                print(f"{key[1]:>6} {key[2]:<24} {key[3]:>4}  shape mismatch "
                      f"{stats['shape_mismatch']}")
                continue
            rel = stats["rel_l2"]
            diverging = rel >= args.threshold or stats["nonfinite"] > 0
            if diverging and step not in first_diverging:
                first_diverging[step] = (key, rel)
            if args.only_diverging and not diverging:
                continue
            print(f"{key[1]:>6} {key[2]:<24} {key[3]:>4} {rel:11.3e} "
                  f"{stats['max_abs']:11.3e} {stats['ref_norm']:11.3e} "
                  f"{stats['cand_norm']:11.3e} {stats['nonfinite']:>7}"
                  f"{'  <== diverges' if diverging else ''}")
        print()

        if selection_rows:
            print(f"----- step {step} indexer block selection -----")
            print(f"{'layer':>6} {'rank':>4} {'topk':>5} {'rows':>7} "
                  f"{'rows_differ':>12} {'blocks_changed_per_row':>23}")
            for key, stats in selection_rows:
                if "shape_mismatch" in stats:
                    print(f"{key[1]:>6} {key[3]:>4}  shape mismatch "
                          f"{stats['shape_mismatch']}")
                    continue
                print(f"{key[1]:>6} {key[3]:>4} {stats['topk']:>5} "
                      f"{stats['rows']:>7} {stats['row_mismatch']:>11.1%} "
                      f"{stats['changed_per_row']:>23.2f}")
            print()

        if scalar_rows:
            print(f"----- step {step} small tensors (values) -----")
            for key, left_values, right_values in scalar_rows:
                same = "same" if left_values == right_values else "DIFFER"
                print(f"{key[1]:>6} {key[2]:<24} {key[3]:>4} {same:>7} "
                      f"ref={_format_values(left_values)} "
                      f"cand={_format_values(right_values)}")
            print()

        if logit_rows:
            print(f"----- step {step} logits -----")
            print(f"{'rank':>4} {'rows':>6} {'argmax_agree':>13} "
                  f"{'ref_top1_top2_gap':>18} {'max_abs_diff':>13}")
            for key, stats in logit_rows:
                if "shape_mismatch" in stats:
                    print(f"{key[3]:>4}  shape mismatch {stats['shape_mismatch']}")
                    continue
                print(f"{key[3]:>4} {stats['rows']:>6} {stats['argmax_agree']:>12.1%} "
                      f"{stats['ref_margin']:>18.4f} {stats['max_abs']:>13.4f}")
            print()

    print("===== summary =====")
    for step in sorted(by_step):
        if step in first_diverging:
            (_, layer, tag, rank), rel = first_diverging[step]
            print(f"step {step}: first divergence at layer {layer} "
                  f"tag={tag} rank={rank} rel_l2={rel:.3e}")
        else:
            print(f"step {step}: no tensor above threshold {args.threshold}")


if __name__ == "__main__":
    main()
