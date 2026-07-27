#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.5 DECISIVE vision-vs-decode localizer for the MMMU Accounting
accuracy gap.

``reference_tier=real_source``, ``validation_tier=unit`` (CUDA/GPU + CPU numba).

The baseline deterministic 27-item MMMU run (job 5586765, TP=4,
``cuda_graph=false, overlap_scheduler=false``, ``enable_autotuner=False``, bs=1,
maxrep=1 / NO collapse) is systematically WRONG on ``validation_Accounting_2..7``
(cache offsets ``Accounting:1..6``) while SGLang gets 5/6 right, yet TRT is
CORRECT on the surrounding ``validation_Accounting_1`` (offset 0) and
``validation_Accounting_8`` (offset 7). Goal-1.3's tower replay (job 5579921)
only ever covered ``Accounting:0`` -- the ONE Accounting item TRT gets right --
so the failing dense financial-table images were never vision-verified.

This driver settles the human-feedback-#2 localization question with ONE cheap,
single-GPU job by REUSING the two already-proven replay paths (no new vision
math) over the DISCRIMINATING item set (E2E-correct controls ``Accounting:0,7``
vs E2E-wrong ``Accounting:1..6``):

  * PREPROCESS parity -- ``inkling_input_processor_test.run_preprocess_parity``:
    TRT ``InklingImagePreprocessor.encode_one`` vs SGLang's REAL numba
    ``InklingImageProcessor._encode_image_bytes`` per image (patch grid /
    ``num_patches`` + per-patch bf16 tensor). This catches the risk-register
    width-padding (``width // patch + 1``) / normalization / temporal-duplication
    bugs that only certain image geometries exercise.
  * TOWER parity -- ``inkling_vision_tower_test.run_tower_replay``: TRT
    ``InklingVisionModel`` vs SGLang's REAL ``HMLPPatchEncoder`` on CUDA per image
    (same real ``model.visual.*`` bf16 weights loaded into both).

Together these cover the ENTIRE vision path (bytes -> patches -> text-hidden
embedding). The verdict is decisive:

  * If the E2E-WRONG items diverge (``num_patches`` mismatch, or cos below the
    proven ~1.0 floor) while the E2E-CORRECT controls do NOT, the Accounting gap
    is a Python-fixable vision preprocessing/encoding bug -> fix + rerun M1b/M1c.
  * If ALL items -- wrong and control alike -- match at the same ~1.0 floor and
    identical ``num_patches``, the vision path is RULED OUT and the Accounting
    gap lives in the decode-side path (the accepted, documented, out-of-scope
    fa4(SGLang)-vs-Triton(TRT) attention-kernel-family residual, Goal 1.4) ->
    teacher-force / escalate as the same-class BLOCKER, not a vision fix.

Env (mirrors Goal 1.2/1.3):
  * ``MMMU_ALIGN_CACHE``   -- 27-item warmed cache (has Accounting_0..11)
  * ``MMMU_ALIGN_ITEMS``   -- ``Config:offset`` list; defaults to the canary +
                              Accounting:0..7 discriminating set below.
  * ``INKLING_CKPT`` / ``SGLANG_PY`` -- as Goal 1.3.
  * ``INKLING_ACCT_LOCALIZE_ARTIFACT`` -- combined per-item JSON output path.

Non-skipping: if CUDA / the checkpoint ``model.visual.*`` weights / the SGLang
references cannot be resolved, this FAILS (a skip would hide missing evidence).
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The discriminating set: the cross-discipline canary + Accounting:0..7. Offsets
# 0 (validation_Accounting_1) and 7 (validation_Accounting_8) are the E2E-CORRECT
# controls; offsets 1..6 (validation_Accounting_2..7) are the E2E-WRONG items.
# Set BEFORE importing the test modules so their first ``load_fixed_items`` call
# (via ``R._selected_items`` reading this env) picks up the widened set.
DEFAULT_ITEMS = (
    "Math:0,Math:1,Physics:0,Chemistry:0,"
    "Accounting:0,Accounting:1,Accounting:2,Accounting:3,"
    "Accounting:4,Accounting:5,Accounting:6,Accounting:7"
)
os.environ.setdefault("MMMU_ALIGN_ITEMS", DEFAULT_ITEMS)

ARTIFACT = os.environ.get(
    "INKLING_ACCT_LOCALIZE_ARTIFACT",
    os.path.join(HERE, "inkling_accounting_vision_localize_artifact.json"),
)

# E2E verdicts from the deterministic baseline run (job 5586765) -- used only to
# LABEL the table (correct/wrong), not to gate anything.
E2E_WRONG = {
    "validation_Accounting_2",
    "validation_Accounting_3",
    "validation_Accounting_4",
    "validation_Accounting_5",
    "validation_Accounting_6",
    "validation_Accounting_7",
}
E2E_CORRECT = {"validation_Accounting_1", "validation_Accounting_8"}

# Same numeric floor the two proven replays already pass at (Goal 1.2/1.3 hit
# cos=1.0 / max_abs=0.0 exactly; keep a tiny bf16 margin).
COS_FLOOR = 0.9999


def _label(item_id: str) -> str:
    if item_id in E2E_WRONG:
        return "WRONG"
    if item_id in E2E_CORRECT:
        return "ctrl-OK"
    return "canary"


def main() -> int:
    import inkling_input_processor_test as IP
    import inkling_vision_tower_test as VT
    import torch

    if not torch.cuda.is_available():
        print(
            "INKLING_ACCT_VISION_LOCALIZE FAIL: CUDA required (a skip would "
            "hide missing GPU evidence).",
            flush=True,
        )
        return 2

    items = os.environ["MMMU_ALIGN_ITEMS"]
    print(
        "=== ACCOUNTING VISION LOCALIZATION (preprocess + tower parity vs SGLang) ===", flush=True
    )
    print(f"items={items}", flush=True)
    print(f"cos_floor={COS_FLOOR} artifact={ARTIFACT}", flush=True)

    # 1. PREPROCESS parity (CPU numba kernel vs TRT preprocessor), per item.
    print(
        "\n--- STEP 1: preprocess parity (TRT encode_one vs SGLang _encode_image_bytes) ---",
        flush=True,
    )
    pre = IP.run_preprocess_parity()
    pre_by_id = {r["id"]: r for r in pre["records"]}

    # 2. TOWER parity (CUDA InklingVisionModel vs SGLang HMLPPatchEncoder).
    print(
        "--- STEP 2: tower parity (TRT InklingVisionModel vs SGLang HMLPPatchEncoder, CUDA) ---",
        flush=True,
    )
    tow = VT.run_tower_replay()
    tow_by_id = {r["id"]: r for r in tow["records"]}

    # 3. Combined per-item table + decisive verdict.
    ids = [r["id"] for r in tow["records"]]
    print("\n=== COMBINED PER-ITEM VISION PARITY (Accounting localizer) ===", flush=True)
    print(
        f"{'id':<28} {'e2e':<8} {'np_trt/np_sg':<14} "
        f"{'pre_max_abs':<12} {'pre_cos':<11} "
        f"{'tow_max_abs':<12} {'tow_cos':<11} verdict",
        flush=True,
    )
    combined = []
    diverging = []
    for item_id in ids:
        pr = pre_by_id.get(item_id, {})
        tr = tow_by_id.get(item_id, {})
        np_trt = pr.get("trt_num_patches")
        np_sg = pr.get("sglang_num_patches")
        pre_cos = pr.get("cosine")
        pre_abs = pr.get("max_abs")
        tow_cos = tr.get("cosine")
        tow_abs = tr.get("max_abs")
        np_ok = (
            np_trt is not None
            and np_trt == np_sg
            and tr.get("feature_rows_trt") == tr.get("num_patches")
        )
        pre_ok = pre_cos is not None and pre_cos >= COS_FLOOR
        tow_ok = tow_cos is not None and tow_cos >= COS_FLOOR
        vis_ok = bool(np_ok and pre_ok and tow_ok)
        verdict = "vision_clean" if vis_ok else "VISION_DIVERGES"
        if not vis_ok:
            diverging.append(item_id)
        print(
            f"{item_id:<28} {_label(item_id):<8} "
            f"{str(np_trt) + '/' + str(np_sg):<14} "
            f"{str(pre_abs):<12} {str(pre_cos):<11} "
            f"{str(tow_abs):<12} {str(tow_cos):<11} {verdict}",
            flush=True,
        )
        combined.append(
            {
                "id": item_id,
                "e2e": _label(item_id),
                "trt_num_patches": np_trt,
                "sglang_num_patches": np_sg,
                "num_patches_ok": bool(np_ok),
                "preprocess_max_abs": pre_abs,
                "preprocess_cos": pre_cos,
                "preprocess_ok": bool(pre_ok),
                "tower_max_abs": tow_abs,
                "tower_cos": tow_cos,
                "tower_ok": bool(tow_ok),
                "vision_ok": vis_ok,
            }
        )

    # Decisive verdict: do the E2E-WRONG items diverge in vision while the
    # E2E-CORRECT controls stay clean?
    wrong_div = [i for i in diverging if i in E2E_WRONG]
    ctrl_div = [i for i in diverging if i in E2E_CORRECT]
    if wrong_div and not ctrl_div:
        verdict = "VISION_LOCALIZED"  # wrong items diverge, controls clean
    elif diverging:
        verdict = "VISION_DIVERGES_MIXED"  # divergence not aligned to E2E-wrong
    else:
        verdict = "VISION_CLEAN"  # all items match -> decode-side, not vision

    summary = {
        "items": items,
        "cos_floor": COS_FLOOR,
        "verdict": verdict,
        "diverging_ids": diverging,
        "wrong_diverging": wrong_div,
        "ctrl_diverging": ctrl_div,
        "records": combined,
    }
    os.makedirs(os.path.dirname(ARTIFACT) or ".", exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nINKLING_ACCT_VISION_LOCALIZE verdict={verdict} "
        f"n_items={len(ids)} diverging={diverging} "
        f"wrong_diverging={wrong_div} ctrl_diverging={ctrl_div} "
        f"artifact={ARTIFACT}",
        flush=True,
    )
    if verdict == "VISION_CLEAN":
        print(
            "INTERPRETATION: vision path (preprocess + tower) is bitwise-clean "
            "for the E2E-WRONG Accounting images -> the accuracy gap is "
            "DECODE-SIDE (accepted fa4-vs-Triton kernel-family residual), NOT a "
            "Python-fixable vision bug.",
            flush=True,
        )
    elif verdict == "VISION_LOCALIZED":
        print(
            "INTERPRETATION: the E2E-WRONG Accounting images DIVERGE in the "
            "vision path while the E2E-CORRECT controls do not -> a "
            "Python-fixable vision preprocessing/encoding bug; fix and rerun "
            "M1b/M1c.",
            flush=True,
        )
    else:
        print(
            "INTERPRETATION: vision divergence is not cleanly aligned to the "
            "E2E-wrong set; inspect the per-item table before concluding.",
            flush=True,
        )
    # rc=0 means evidence was produced (this is a diagnostic, not a pass gate).
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("INKLING_ACCT_VISION_LOCALIZE FAIL: exception producing evidence", flush=True)
        sys.exit(1)
