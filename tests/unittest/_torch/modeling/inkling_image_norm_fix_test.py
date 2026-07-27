#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.5 regression guard for the multimodal ``embed_norm`` fusion
fix (iter 22 root cause of the MMMU Accounting accuracy gap).

``reference_tier=real_source`` (SGLang inkling.py contract), ``validation_tier=unit``.

ROOT CAUSE (overturns the iter21 "decode-side fa4-vs-Triton near-tie residual"
BLOCKER): the Accounting gap was NOT a decode-kernel residual. On every failing
item TRT's reasoning said "image not visible" and hallucinated numbers, while the
SGLang reference -- SAME NVFP4 checkpoint, SAME image -- read the real table
values and answered correctly. The vision tower is bytewise-clean
(jobs 5588462/5588190), so the corruption is AFTER the tower, in the fusion.

SGLang ``inkling.py`` applies ``embed_norm`` to the TEXT embeddings ONLY
(``get_input_embeddings`` folds it in) and scatters the RAW vision-tower rows --
which "keep their own norm" -- in AFTER the norm; its decoder ``forward`` then
skips ``embed_norm`` when ``input_embeds`` is supplied ("embed_norm was already
applied during the MM embed; don't re-norm here"). TRT instead fused RAW text +
RAW vision and let the decoder ``embed_norm`` the WHOLE fused stream, pushing the
image rows through an extra RMSNorm SGLang never applies -> the decoder could not
read the image.

This test locks the corrected contract with the REAL ``fuse_input_embeds`` +
REAL ``RMSNorm``:

  1. FUSION: text positions of the fused stream equal
     ``embed_norm(embed_tokens(text_ids))``; image positions equal the RAW vision
     rows bytewise (NOT re-normed).
  2. LOAD-BEARING: the pre-fix behavior (``embed_norm`` applied to the vision
     rows) WOULD have changed them -- so the fix is not cosmetic.
  3. WIRING: the model forwards expose ``inputs_embeds_prenormed`` and the
     multimodal wrapper carries the normed text embedder ``_embed_tokens_with_norm``.

CPU-only; no checkpoint or GPU needed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _fusion_contract() -> int:
    """Real ``fuse_input_embeds`` + real ``RMSNorm``: normed text, raw vision."""
    import torch
    import torch.nn as nn

    from tensorrt_llm._torch.models.modeling_multimodal_utils import fuse_input_embeds
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [fusion] PASS {name}")
        else:
            fail += 1
            print(f"  [fusion] FAIL {name}")

    torch.manual_seed(0)
    hidden, vocab, img_id = 16, 100, 200054
    embed_tokens = nn.Embedding(vocab, hidden).to(torch.float32)
    embed_norm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.float32)
    # A non-trivial (non-identity) norm weight so re-norming the vision rows is a
    # detectable change, not a no-op.
    with torch.no_grad():
        embed_norm.weight.copy_(1.0 + 0.1 * torch.randn(hidden))

    def normed_embedder(ids):
        # Mirror InklingForConditionalGeneration._embed_tokens_with_norm /
        # SGLang InklingModel.get_input_embeddings.
        return embed_norm(embed_tokens(ids))

    # Sequence: [t, t, IMG, IMG, IMG, IMG, t] -- one image expanded to 4 patches.
    input_ids = torch.tensor([5, 6, img_id, img_id, img_id, img_id, 7], dtype=torch.int64)
    text_idx = torch.tensor([0, 1, 6], dtype=torch.int64)
    mm_idx = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    raw_vision = torch.randn(4, hidden)  # hMLP tower output (already tower-normed)

    ret = fuse_input_embeds(
        normed_embedder,
        input_ids,
        [raw_vision],
        mm_token_ids=None,  # explicit indices -> OOV branch, placeholder unembedded
        text_token_indices=text_idx,
        mm_token_indices=mm_idx,
    )
    fused = ret[1]
    chk("fuse returns inputs_embeds tensor", fused is not None)
    chk("fused shape [seq, hidden]", tuple(fused.shape) == (7, hidden))

    # (1) TEXT positions are embed_norm(embed_tokens(text)).
    expect_text = normed_embedder(input_ids[text_idx])
    chk(
        "text positions == embed_norm(embed_tokens(text))",
        torch.allclose(fused[text_idx].float(), expect_text.float(), atol=1e-5),
    )

    # (1) IMAGE positions are the RAW vision rows (bypass embed_norm).
    chk(
        "image positions == RAW vision rows (bypass embed_norm)",
        torch.allclose(fused[mm_idx].float(), raw_vision.float(), atol=1e-6),
    )

    # (2) The pre-fix path (embed_norm on the vision rows) WOULD differ -> the fix
    #     is load-bearing. If this were a no-op the whole bug could not exist.
    buggy_vision = embed_norm(raw_vision)
    chk(
        "re-norming vision rows is a real change (fix is load-bearing)",
        not torch.allclose(buggy_vision.float(), raw_vision.float(), atol=1e-3),
    )

    # Sanity: text rows are NOT equal to the raw (unnormed) token embeddings --
    # the text side really is normed.
    raw_text = embed_tokens(input_ids[text_idx])
    chk(
        "text rows are normed (differ from raw token embeddings)",
        not torch.allclose(fused[text_idx].float(), raw_text.float(), atol=1e-3),
    )

    return 0 if fail == 0 else 1


def _wiring_contract() -> int:
    """Signature/attribute guards so the prenormed skip stays wired end-to-end."""
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForCausalLM,
        InklingForConditionalGeneration,
        InklingModel,
    )

    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [wiring] PASS {name}")
        else:
            fail += 1
            print(f"  [wiring] FAIL {name}")

    chk(
        "InklingModel.forward exposes inputs_embeds_prenormed",
        "inputs_embeds_prenormed" in inspect.signature(InklingModel.forward).parameters,
    )
    chk(
        "InklingForCausalLM.forward exposes inputs_embeds_prenormed",
        "inputs_embeds_prenormed" in inspect.signature(InklingForCausalLM.forward).parameters,
    )
    chk(
        "wrapper carries normed text embedder _embed_tokens_with_norm",
        hasattr(InklingForConditionalGeneration, "_embed_tokens_with_norm"),
    )
    # Default must preserve the accepted text path (embed_norm applied).
    default = inspect.signature(InklingModel.forward).parameters["inputs_embeds_prenormed"].default
    chk("inputs_embeds_prenormed defaults False (text path unchanged)", default is False)

    return 0 if fail == 0 else 1


def _selftest() -> int:
    print("=== inkling_image_norm_fix: fusion contract (real fuse_input_embeds + RMSNorm) ===")
    rc_f = _fusion_contract()
    print("=== inkling_image_norm_fix: wiring contract (prenormed skip) ===")
    rc_w = _wiring_contract()
    rc = rc_f | rc_w
    print(f"\nINKLING_IMAGE_NORM_FIX {'OK' if rc == 0 else 'FAIL'}")
    return rc


# ---- pytest entrypoints ---------------------------------------------------
def test_fusion_contract():
    assert _fusion_contract() == 0


def test_wiring_contract():
    assert _wiring_contract() == 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
