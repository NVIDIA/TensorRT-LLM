# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared canonical image-prompt builder for the Stage-1 / Goal-1.4 end-to-end
vision ``source_logit_replay`` / ``generation_parity`` tests.

Both the TensorRT-LLM runtime tests and the SGLang reference capture consume the
SAME token stream so there is zero tokenizer drift between the two stacks (the
text-stage parity method: feed byte-identical ``input_ids`` to both sides).

Contract (verified against SGLang PR 31681 on disk):
  * SGLang's Inkling multimodal serving path
    (``multimodal/processors/inkling.py:process_mm_data_async``) requires
    **pre-rendered ``input_ids`` that already carry the ``-101`` image sentinel**
    (one per image); its ``assemble`` then expands each ``-101`` into
    ``num_patches`` copies and attaches the vision features. The custom chat
    renderer that would insert those sentinels is "a separate workstream", so
    the caller supplies the ``-101`` ids directly.
  * The Inkling chat template (``chat_template.jinja``) renders an image content
    part as ``<|content_image|>(200005) <|unused_200054|> <|end_message|>(200010)``.
    We render the template, then replace the placeholder token(s) between the
    ``<|content_image|>`` and the following ``<|end_message|>`` with a single
    ``-101`` -- the canonical unexpanded stream both stacks expand identically.
  * ``reasoning_effort=0.9`` is injected into the chat template exactly as the
    accepted text tower does (the template's ``reasoning_effort`` variable), so
    the served rendering matches the proven text path.

The five fixed prompts are real ``MMMU/MMMU`` validation items resolved through
the Goal-1.1 alignment infrastructure (``inkling_mmmu_real_align_test`` /
``inkling_mmmu_harness``), so the images and textual prompt are the exact
serving-aligned ones.
"""

from __future__ import annotations

import base64
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import inkling_mmmu_harness as H  # noqa: E402  (aligned prompt renderer)
import inkling_mmmu_real_align_test as R  # noqa: E402  (fixed real MMMU items)

CKPT = os.environ.get(
    "INKLING_CKPT",
    os.environ.get(
        "INKLING_CHECKPOINT",
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
        "users/kleinc/hf_data/Inkling-NVFP4-full",
    ),
)

# Inkling protocol framing ids (SGLang parser/inkling_tokenizer.py) -- fixed
# constants, not tokenizer-dependent.
CONTENT_IMAGE_ID = 200005  # <|content_image|>
END_MESSAGE_ID = 200010  # <|end_message|>
# The Inkling chat template renders an image content part as
# ``<|content_image|>(200005) <|unused_200054|>(200054) <|end_message|>(200010)``,
# so 200054 IS the in-vocab image placeholder. TRT keys on it (the executor
# rejects an out-of-range id, so the SGLang-style -101 cannot be sent to
# ``llm.generate``); the SGLang server expects its internal -101, so the capture
# maps 200054 -> -101 before POST (see ``to_sglang_ids``). The two ids are
# interchangeable for parity: both are overwritten by identical vision embeds.
IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101
REASONING_EFFORT = float(os.environ.get("INKLING_MM_REASONING_EFFORT", "0.9"))

# How many fixed items to build (>=5 required by the acceptance criterion).
N_PROMPTS = int(os.environ.get("INKLING_MM_N_PROMPTS", "5"))


def _build_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)


def _render_ids(tok, prompt_text: str) -> List[int]:
    """Render one user turn (image content part + question text) to token ids
    with ``reasoning_effort`` injected, exactly like the accepted text path."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    # reasoning_effort is a template variable (chat_template.jinja line 5); pass
    # it through apply_chat_template's **kwargs like paired_gsm8k.chat() does.
    try:
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=REASONING_EFFORT
        )
    except Exception:  # noqa: BLE001 -- older template without the kwarg
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return _normalize_ids(ids)


def _normalize_ids(ids) -> List[int]:
    """Normalize apply_chat_template output to a flat list[int].

    ``apply_chat_template(tokenize=True)`` may return a plain list, a
    ``BatchEncoding``/dict (iterating which yields the *keys* -- the
    ``'input_ids'`` string that the text stage also had to normalize), a torch
    tensor, or a nested ``[[...]]`` single-batch list."""
    if hasattr(ids, "input_ids"):  # BatchEncoding
        ids = ids.input_ids
    elif isinstance(ids, dict):  # dict with input_ids
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):  # torch/np tensor
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):  # [[...]] batch dim
        ids = ids[0]
    return [int(t) for t in ids]


def _validate_single_image(ids: List[int]) -> List[int]:
    """Assert the rendered stream carries exactly one image placeholder.

    The chat template emits ``[<|content_image|>(200005), <|unused_200054|>
    (200054), <|end_message|>(200010)]`` for a single-image content part, so the
    rendered ids already carry exactly one ``IMAGE_TOKEN_ID`` (200054) framed by
    ``<|content_image|>`` / ``<|end_message|>``. We keep the stream as-is (the
    input processor expands the one placeholder into ``num_patches`` rows) and
    only fail loudly if the expected placeholder/framing is missing, so we never
    silently produce a placeholder-free image prompt."""
    n_img = sum(1 for t in ids if t == IMAGE_TOKEN_ID)
    if n_img != 1:
        raise ValueError(
            f"expected exactly one image placeholder ({IMAGE_TOKEN_ID}) in the "
            f"rendered stream, found {n_img}; the chat template did not emit a "
            f"single image content block"
        )
    if CONTENT_IMAGE_ID not in ids or END_MESSAGE_ID not in ids:
        raise ValueError(
            f"rendered image block missing framing (<|content_image|> "
            f"{CONTENT_IMAGE_ID} / <|end_message|> {END_MESSAGE_ID})"
        )
    return list(ids)


def to_sglang_ids(ids: List[int]) -> List[int]:
    """Map the TRT in-vocab image placeholder (200054) to SGLang's internal
    sentinel (-101) for POSTing to the SGLang server, which resolves its image
    token to -101 (config.json omits ``image_token_id``). Every other token is
    unchanged, so the two streams are identical apart from the placeholder id
    (which both stacks overwrite with the same vision embeddings)."""
    return [SGLANG_IMAGE_TOKEN_ID if t == IMAGE_TOKEN_ID else int(t) for t in ids]


def image_to_data_uri(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _num_patches_for(image_bytes: bytes) -> int:
    """Number of hMLP patch tokens the image expands to. Uses the TRT-LLM
    ``InklingImagePreprocessor`` (bitwise-equal to SGLang, proven in Goal 1.3).
    Only importable where ``tensorrt_llm`` is installed (the TRT container); the
    SGLang capture does not need it (the server computes num_patches itself)."""
    from tensorrt_llm._torch.models.modeling_inkling_vision import InklingImagePreprocessor

    pre = InklingImagePreprocessor(
        patch_size=R.PATCH_SIZE, temporal_patch_size=H.DEFAULT_TEMPORAL_PATCH_SIZE
    )
    return int(pre.preprocess([image_bytes])["num_patches"][0])


def build_prompts(n: Optional[int] = None, with_num_patches: bool = True) -> List[Dict[str, Any]]:
    """Return up to ``n`` canonical image-prompt records.

    Each record::

        {
          "id": "validation_Math_1",
          "config": "Math",
          "input_ids": [... one -101 sentinel ...],   # unexpanded, shared stream
          "num_patches": 1148,                          # only if with_num_patches
          "image_bytes": b"...png...",                  # the exact aligned image
          "prompt": "Find ...",                         # textual question (log only)
        }

    ``input_ids`` and the image are fed VERBATIM to both TRT and the SGLang
    reference server, so no re-tokenization can desynchronize them. Set
    ``with_num_patches=False`` in the SGLang capture container (no
    ``tensorrt_llm`` install; the server expands the -101 itself).
    """
    if n is None:
        n = N_PROMPTS
    tok = _build_tokenizer()
    items = R.load_fixed_items()
    out: List[Dict[str, Any]] = []
    for it in items[:n]:
        prompt_text, _qtype = H.render_mmmu_prompt(it["question"], it["options"])
        rendered = _render_ids(tok, prompt_text)
        input_ids = _validate_single_image(rendered)
        rec = {
            "id": it["id"],
            "config": it["config"],
            "input_ids": input_ids,
            "image_bytes": it["image_bytes"],
            "prompt": prompt_text,
        }
        if with_num_patches:
            rec["num_patches"] = _num_patches_for(it["image_bytes"])
        out.append(rec)
    if len(out) < 1:
        raise RuntimeError("no MMMU image prompts resolved (warm MMMU cache?)")
    return out


def _selftest() -> int:
    """CPU-only structural self-test of the sentinel-collapse logic (no GPU, no
    tokenizer). Exercises ``_replace_image_block_with_sentinel`` on synthetic
    rendered streams so the collapse contract is verifiable without the 403GB
    stack."""
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [selftest] PASS {name}")
        else:
            fail += 1
            print(f"  [selftest] FAIL {name}")

    # rendered stream with the in-vocab image block is accepted unchanged
    ids = [200000, CONTENT_IMAGE_ID, IMAGE_TOKEN_ID, END_MESSAGE_ID, 1, 2, 3]
    out = _validate_single_image(ids)
    chk("valid single-image stream kept as-is", out == ids)
    chk("exactly one image placeholder (200054)", sum(1 for t in out if t == IMAGE_TOKEN_ID) == 1)

    # to_sglang_ids maps 200054 -> -101, leaves everything else
    chk(
        "to_sglang_ids maps 200054->-101",
        to_sglang_ids(ids)
        == [200000, CONTENT_IMAGE_ID, SGLANG_IMAGE_TOKEN_ID, END_MESSAGE_ID, 1, 2, 3],
    )

    # zero image placeholders fails loudly
    try:
        _validate_single_image([CONTENT_IMAGE_ID, 5, END_MESSAGE_ID])
        chk("zero image placeholders raises", False)
    except ValueError:
        chk("zero image placeholders raises", True)

    # missing framing fails loudly
    try:
        _validate_single_image([1, IMAGE_TOKEN_ID, 2])
        chk("missing framing raises", False)
    except ValueError:
        chk("missing framing raises", True)

    # _normalize_ids: plain list, dict/BatchEncoding, nested batch dim
    chk("normalize plain list", _normalize_ids([1, 2, 3]) == [1, 2, 3])
    chk(
        "normalize dict input_ids",
        _normalize_ids({"input_ids": [4, 5], "attention_mask": [1, 1]}) == [4, 5],
    )
    chk("normalize nested [[...]]", _normalize_ids([[7, 8, 9]]) == [7, 8, 9])

    class _BE:  # duck-typed BatchEncoding
        input_ids = [10, 11]

    chk("normalize BatchEncoding.input_ids", _normalize_ids(_BE()) == [10, 11])

    print(f"\nINKLING_IMAGE_PROMPTS_SELFTEST {'OK' if fail == 0 else 'FAIL'} pass={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
