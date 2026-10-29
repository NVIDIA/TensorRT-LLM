# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MuseSparkOpen vision embedding merge, checked against the frozen oracle.

The reference implementation norms vision rows inside its encoder with
`perception_emb_norm` and merges them *after* the text embeddings have had
their own `tok_embeddings_norm`. The decoder instead applies one
`embed_norm` to whatever `inputs_embeds` it is handed, so the encoder hands
over **un-normed** projected rows and the decoder norms text and vision alike,
exactly once.

That is only correct if the reference's two norms are the same function. They
are, and this file proves it against the oracle's own captured tensors rather
than against a reimplementation of them:

* the reference's `tok_embeddings_norm` is a weightless RMSNorm over
  `hidden_size` with `eps=rms_norm_eps`;
* its `perception_emb_norm` is a weightless RMSNorm over the same width with
  `eps=1e-5`;
* the checkpoint has `rms_norm_eps = 1e-05` and `hidden_size = 8192`, so the
  two are the same weightless RMSNorm over the same width;
* the decoder's `embed_norm` is `RMSNorm(hidden_size, eps=rms_norm_eps,
  has_weights=False)` — the same again.

RMSNorm is per-row, so merge-then-norm and norm-then-merge agree row by row.
The tests below check that claim on real captured activations, and the last one
is the negative control: applying the norm a second time, which is what the
default fuse-then-norm wiring would do if the encoder normed, moves the answer.
"""

import json
import os

import pytest
import torch

from tensorrt_llm._torch.modules.rms_norm import RMSNorm

ORACLE_ENV = "MUSESPARK_OPEN_VISION_ORACLE"

# From the checkpoint's config.json. Asserted against the manifest below rather
# than trusted, so a refset captured from a different config fails loudly.
LLM_HIDDEN = 8192
RMS_NORM_EPS = 1e-5

# The capture is fp32 and the engine ran bf16, so the tensors carry bf16's
# resolution in an fp32 container. bf16 unit roundoff is 2**-8 = 3.9e-3; the
# norm is a division by a mean-square over 8192 elements, which does not
# amplify it. Declared here, before the run.
NORM_IDENTITY_RTOL = 2.0e-2
NORM_IDENTITY_ATOL = 2.0e-2


def _oracle_dir():
    root = os.environ.get(ORACLE_ENV)
    return root if root and os.path.isdir(root) else None


pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        _oracle_dir() is None, reason=f"set {ORACLE_ENV} to the captured refset directory"
    ),
]


@pytest.fixture(scope="module")
def manifest():
    root = _oracle_dir()
    with open(os.path.join(root, "vision-manifest.json")) as handle:
        return root, json.load(handle)


@pytest.fixture(scope="module")
def weightless_norm():
    """TRT-LLM's RMSNorm, configured the way the decoder configures it."""
    norm = RMSNorm(hidden_size=LLM_HIDDEN, eps=RMS_NORM_EPS, dtype=torch.float32, has_weights=False)
    return norm


def _records(manifest):
    root, data = manifest
    return root, [r for r in data["records"] if r.get("tensors_file")]


def _load(root, record):
    return torch.load(
        os.path.join(root, record["tensors_file"]), map_location="cpu", weights_only=False
    )


def _tags(manifest):
    _, records = _records(manifest)
    return [r["tag"] for r in records]


def test_refset_was_captured_from_the_config_these_tests_assume(manifest):
    """A refset from a different checkpoint must not silently pass."""
    root, data = manifest
    del root
    for record in data["records"]:
        shapes = record.get("tensor_shapes") or {}
        for name in ("vision_projection.out", "perception_emb_norm.out"):
            if name in shapes:
                assert shapes[name][0][-1] == LLM_HIDDEN, (
                    f"{record['tag']}: {name} is width {shapes[name][0][-1]}, "
                    f"these tests assume the decoder hidden size {LLM_HIDDEN}"
                )


@pytest.mark.parametrize("index", range(9))
def test_perception_norm_is_the_weightless_rmsnorm_we_apply(manifest, weightless_norm, index):
    """`perception_emb_norm.out` == the decoder's embed_norm applied to the projection.

    This is the whole basis for handing un-normed rows to the decoder. If it
    fails, the encoder must norm and the decoder must not, and the design in
    `modeling_musespark_open_omni.py` is wrong.
    """
    root, records = _records(manifest)
    if index >= len(records):
        pytest.skip("fewer records than the parametrization")
    record = records[index]
    tensors = _load(root, record)

    projected = tensors["vision_projection.out"][0].float()
    reference = tensors["perception_emb_norm.out"][0].float()
    assert projected.shape == reference.shape

    ours = weightless_norm(projected)

    max_abs = (ours - reference).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        ours.flatten(), reference.flatten(), dim=0
    ).item()
    assert torch.allclose(ours, reference, rtol=NORM_IDENTITY_RTOL, atol=NORM_IDENTITY_ATOL), (
        f"{record['tag']}: max_abs {max_abs:.3e}, cosine {cosine:.6f}"
    )
    assert cosine > 0.999, f"{record['tag']}: cosine {cosine:.6f}"


#: Images fill with `<|patch|>`, video with `<|video|>`. Counting only the
#: image filler reports every video record as inconsistent -- the first version
#: of the consistency test did exactly that and flagged `vmme_short_1`, whose
#: own counts (6912 marked, 6912 rows) agree perfectly.
FILLER_TOKEN_IDS = {200092, 200091}

#: A record's captured window is the part of its prompt the forward actually
#: computed, which is not the whole prompt: the oracle ran with vLLM's prefix
#: cache on, so `merged_input_ids` is the prompt's *uncached suffix*. Eight
#: records lose the 112-token template header they all share. `mc_multi_4`
#: loses 512, because its first image is byte-identical to `mc_single_1`'s
#: (the only repeated image in the 15-file
#: media set), so the two prompts agree for 516 tokens and 32 of vLLM's
#: 16-token blocks hit.
#:
#: The same duplicate image also hits the multimodal embedding cache, so the
#: encoder never ran for `mc_multi_4`'s image 0 and its tensors hold 1173 rows
#: (3 x 391) while the window's mask marks 1176 positions. The extra 3 are the
#: tail of image 0's 391-token run, which starts at prompt position 124 and so
#: ends 3 tokens past the 512-token prefix boundary. The encoder rows therefore
#: align to the LAST `rows` marked positions, not the first.
#:
#: With that alignment the merge is bitwise identical (max_abs 0.0) on all
#: nine records of the reference capture set; at offset 0, `mc_multi_4`
#: differs by 21.78. Its four counts (1564 prompt fillers, 1286 window tokens,
#: 1176 marked, 1173 rows) are a faithful capture of a partially cached
#: forward, not a record disagreeing with itself.
#:
#: Pinned per record so a genuinely bad capture cannot hide behind the
#: mechanism: how many leading marked positions have no captured encoder row.
ENCODER_CACHE_HIT_ROWS = {"mc_multi_4": 3}


def _aligned_vision_rows(record, tensors):
    """The merged vision rows the encoder tensors refer to.

    Checks the capture against itself first -- the window is the prompt's own
    suffix and the mask agrees with the window's own tokens -- then drops the
    leading marked positions whose embedding came from the multimodal cache and
    so has no captured encoder row. See `ENCODER_CACHE_HIT_ROWS`.
    """
    prompt = record["prompt_token_ids"]
    window_ids = tensors["merged_input_ids"][0].tolist()
    assert window_ids == prompt[len(prompt) - len(window_ids) :], (
        f"{record['tag']}: the captured window is not this prompt's suffix "
        f"({len(window_ids)} of {len(prompt)} tokens)"
    )

    mask = tensors["merged_is_multimodal"][0].bool()
    marked = int(mask.sum())
    window_fillers = sum(1 for t in window_ids if t in FILLER_TOKEN_IDS)
    assert marked == window_fillers, (
        f"{record['tag']}: mask marks {marked} positions but the captured window "
        f"carries {window_fillers} filler tokens"
    )

    rows = tensors["perception_emb_norm.out"][0].shape[0]
    uncovered = marked - rows
    assert uncovered == ENCODER_CACHE_HIT_ROWS.get(record["tag"], 0), (
        f"{record['tag']}: {marked} marked positions against {rows} encoder rows "
        f"leaves {uncovered} uncovered, not the pinned "
        f"{ENCODER_CACHE_HIT_ROWS.get(record['tag'], 0)}"
    )

    merged = tensors["merged_input_embeddings"][0].float()
    return merged[mask][uncovered:]


@pytest.mark.parametrize("index", range(9))
def test_the_oracle_merges_vision_rows_already_normed(manifest, index):
    """At vision positions the merged tensor carries the NORMED rows.

    Establishes what the merge has to reproduce: the reference's merged
    embedding holds `perception_emb_norm.out` at the multimodal positions, not
    the raw projection. Uses the capture's own `merged_is_multimodal` mask, so
    the positions are the oracle's.
    """
    root, records = _records(manifest)
    if index >= len(records):
        pytest.skip("fewer records than the parametrization")
    record = records[index]
    tensors = _load(root, record)
    if "merged_is_multimodal" not in tensors:
        pytest.skip("this record has no merged capture")

    normed = tensors["perception_emb_norm.out"][0].float()
    projected = tensors["vision_projection.out"][0].float()

    vision_rows = _aligned_vision_rows(record, tensors)
    to_normed = (vision_rows - normed).abs().max().item()
    to_projected = (vision_rows - projected).abs().max().item()

    assert to_normed < to_projected, (
        f"{record['tag']}: merged vision rows are closer to the raw projection "
        f"({to_projected:.3e}) than to the normed rows ({to_normed:.3e})"
    )
    assert to_normed < 1e-2, f"{record['tag']}: merged vs normed max_abs {to_normed:.3e}"


@pytest.mark.parametrize("index", range(9))
def test_norming_twice_is_not_a_no_op(manifest, weightless_norm, index):
    """The negative control that makes the trap real rather than theoretical.

    If the encoder normed AND the decoder normed, vision rows would be normed
    twice. On a unit-RMS row that is close to idempotent, which is exactly why
    it is easy to assume it is harmless. Measure it instead: the second norm
    must change the rows, and the drift is printed so a reader can set it
    against the identity tolerance above.
    """
    root, records = _records(manifest)
    if index >= len(records):
        pytest.skip("fewer records than the parametrization")
    record = records[index]
    tensors = _load(root, record)

    once = tensors["perception_emb_norm.out"][0].float()
    twice = weightless_norm(once)

    drift = (twice - once).abs().max().item()
    relative = drift / once.abs().max().item()
    # Only that it is not zero: that is the claim the trap rests on. The
    # magnitude is reported, not asserted.
    assert drift > 0.0, f"{record['tag']}: second norm was a bitwise no-op"
    print(f"{record['tag']}: second-norm drift max_abs {drift:.3e}, relative {relative:.3e}")


def test_every_stable_prompt_is_covered(manifest):
    """The parametrization must not silently cover fewer prompts than exist."""
    _, records = _records(manifest)
    assert len(records) == 9, f"expected 9 captured records, found {len(records)}"


def test_every_record_agrees_with_itself_once_the_caches_are_accounted_for(manifest):
    """Pin the cache accounting, so a genuinely bad record cannot hide in it.

    A record agrees with itself when its captured window is its own prompt's
    suffix, its mask marks exactly the filler tokens that window carries, and
    the marked positions its encoder rows do not cover are the pinned number.
    The earlier version of this test compared the whole prompt's filler count
    against the mask and the rows, read the prefix-cache gap as corruption, and
    excluded `mc_multi_4` on it; this one measures what the forward actually
    saw, so every record is refereed rather than one being skipped.
    """
    root, records = _records(manifest)
    measured = {}
    for record in records:
        tensors = _load(root, record)
        if "merged_is_multimodal" not in tensors:
            continue
        prompt = record["prompt_token_ids"]
        window_ids = tensors["merged_input_ids"][0].tolist()
        marked = int(tensors["merged_is_multimodal"][0].bool().sum())
        rows = tensors["perception_emb_norm.out"][0].shape[0]
        measured[record["tag"]] = {
            "cached_prefix": len(prompt) - len(window_ids),
            "window_is_prompt_suffix": window_ids == prompt[len(prompt) - len(window_ids) :],
            "mask_matches_window_fillers": marked
            == sum(1 for t in window_ids if t in FILLER_TOKEN_IDS),
            "uncovered": marked - rows,
        }

    assert all(m["window_is_prompt_suffix"] for m in measured.values()), (
        f"a captured window is not its prompt's suffix; measured {measured}"
    )
    assert all(m["mask_matches_window_fillers"] for m in measured.values()), (
        f"a mask disagrees with its own window's filler tokens; measured {measured}"
    )
    uncovered = {tag: m["uncovered"] for tag, m in measured.items() if m["uncovered"]}
    assert uncovered == ENCODER_CACHE_HIT_ROWS, (
        "the set of records whose encoder rows do not cover every marked position "
        f"changed; measured {uncovered}, pinned {ENCODER_CACHE_HIT_ROWS}"
    )
