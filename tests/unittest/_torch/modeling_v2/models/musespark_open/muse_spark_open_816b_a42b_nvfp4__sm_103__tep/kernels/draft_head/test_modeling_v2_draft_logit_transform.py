# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the draft head's drafter logit transform tables (``kernels/draft_head/logit_transform.py``).

The claim is that ``values`` is the drafter's own transform of every bf16 value
bit for bit, in any layout, and that ``canon`` groups exactly the bf16 values
whose transformed values have the same bits, in runs of consecutive keys, with
the identity on ``[lo, hi]``. Checked exhaustively over the 65536 bf16 values
on this device, for the Muse Spark Open drafter's transform and three others.
"""

import pytest
import torch

__extra_import_path__ = ["."]
from _draft_head_cases import (  # noqa: E402
    MUSESPARK,
    OTHER_TRANSFORMS,
    builtin_transform,
    drafter,
    logit_transform,
    transform,
)

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"

CONFIGS = {"musespark": MUSESPARK, **OTHER_TRANSFORMS}
ALL = torch.arange(1 << 16, dtype=torch.int32)


def _radix32(v: torch.Tensor) -> torch.Tensor:
    word = v.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(word >= 0x80000000, word ^ 0xFFFFFFFF, word ^ 0x80000000)
    return torch.where(v.isnan(), torch.full_like(key, 0xFFFFFFFF), key)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_values_are_the_builtin_transform_of_every_bf16_value(name):
    config = CONFIGS[name]
    t = transform(config)
    x = logit_transform.bf16_of_key16(ALL.cuda())
    assert t.values.dtype == torch.float32 and t.values.shape == (1 << 16,)
    assert logit_transform.same_bits(t.values, builtin_transform(config, x))


@pytest.mark.parametrize("requests", [1, 2, 8, 24])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_values_hold_in_the_runtime_layout(name, requests):
    """``values[key16(x)]`` equals the transform of x laid out as ``[B * 7, shard]``, shuffled."""
    config = CONFIGS[name]
    t = transform(config)
    rows, shard = requests * 7, 25256
    g = torch.Generator(device="cuda").manual_seed(requests)
    keys = torch.randint(0, 1 << 16, (rows, shard), generator=g, device="cuda", dtype=torch.int32)
    keys.view(-1)[: 1 << 16] = ALL.cuda()[torch.randperm(1 << 16, generator=g, device="cuda")]
    x = logit_transform.bf16_of_key16(keys)
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.draft_shard_topk import (  # noqa: E501
        radix_key16,
    )

    got = t.values[radix_key16(x.view(torch.int16)).long()]
    assert logit_transform.same_bits(got, builtin_transform(config, x))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_canon_groups_exactly_the_ties(name):
    t = transform(CONFIGS[name])
    canon, values = t.canon.cpu().long(), t.values.cpu()
    radix = _radix32(values)
    finite = torch.arange(0x007F, 0xFF81)
    # Non-decreasing over the non-NaN keys, so the tie groups are runs of consecutive keys.
    assert (radix[finite][1:] >= radix[finite][:-1]).all()
    # Same canonical key exactly when the transformed values have the same bits.
    assert torch.equal(radix[canon[finite]], radix[finite])
    split = canon[finite][1:] != canon[finite][:-1]
    assert torch.equal(split, radix[finite][1:] != radix[finite][:-1])
    # The band: canon is the identity on [lo, hi] and on no larger interval around zero.
    assert torch.equal(canon[t.lo : t.hi + 1], torch.arange(t.lo, t.hi + 1))
    assert t.lo == 0 or canon[t.lo - 1] != t.lo - 1
    assert t.hi == 0xFFFF or canon[t.hi + 1] != t.hi + 1
    # NaN maps to itself; +0.0 and -0.0 stay apart and keep their sign.
    assert int(canon[0xFFFF]) == 0xFFFF
    assert int(canon[0x8000]) == 0x8000 and int(canon[0x7FFF]) == 0x7FFF
    assert values[0x8000].item() == 0.0 and not torch.signbit(values[0x8000])
    assert torch.signbit(values[0x7FFF]) and values[0x7FFF].item() == 0.0
    assert logit_transform.unsupported_reason(t, 25256) is None


def test_musespark_band_and_saturation_on_this_device():
    """The identity band holds every logit of magnitude up to 512; +-inf saturate to +-softcap."""
    t = transform(MUSESPARK)
    lo_value = float(logit_transform.bf16_of_key16(torch.tensor([t.lo])).float())
    hi_value = float(logit_transform.bf16_of_key16(torch.tensor([t.hi])).float())
    groups = logit_transform.tie_groups(t)
    first_up = min(g[0] for g in groups if g[0] >= 0x8000)
    print(
        f"identity band [{lo_value}, {hi_value}], first tie above zero at bf16 "
        f"{float(logit_transform.bf16_of_key16(torch.tensor([first_up])).float())}, "
        f"{len(groups)} tie groups"
    )
    assert lo_value <= -512.0 and hi_value >= 512.0
    softcap = MUSESPARK[1]
    assert t.values[0xFF80].item() == softcap and t.values[0x007F].item() == -softcap


def test_identity_transform_builds_nothing():
    assert logit_transform.build(drafter(1.0, None), torch.device("cuda")) is None


def test_an_fp32_probe_would_not_have_found_the_transform():
    """Why build probes in bf16: on fp32 the built-in transform runs in place and returns its input."""
    x = torch.ones(4, device="cuda")
    assert builtin_transform(MUSESPARK, x) is x
    assert logit_transform.build(drafter(*MUSESPARK), torch.device("cuda")) is not None


def test_ensure_transformed_applies_once():
    t = transform(MUSESPARK)
    x = torch.randn(3, 9, device="cuda").bfloat16()
    y = logit_transform.ensure_transformed(t, x)
    assert y.dtype == torch.float32 and logit_transform.same_bits(
        y, builtin_transform(MUSESPARK, x)
    )
    assert logit_transform.ensure_transformed(t, y) is y
    assert logit_transform.ensure_transformed(None, x) is x


def test_the_tables_follow_the_drafter_attributes():
    t = transform(MUSESPARK)
    model = drafter(*MUSESPARK)
    assert t.matches(model)
    model.logit_softcap = 30.0
    try:
        assert not t.matches(model)
    finally:
        model.logit_softcap = MUSESPARK[1]


def test_a_transform_that_decreases_is_refused():
    t = transform(MUSESPARK)
    bad = logit_transform.DraftLogitTransform(
        drafter(*MUSESPARK), t.function, t.canon, -t.values, t.lo, t.hi
    )
    reason = logit_transform.unsupported_reason(bad, 25256)
    assert reason is not None


def test_another_method_is_refused():
    t = transform(MUSESPARK)

    def other(model, logits):
        return logits.float()

    bad = logit_transform.DraftLogitTransform(
        drafter(*MUSESPARK), other, t.canon, t.values, t.lo, t.hi
    )
    assert "not DFlashForCausalLM's" in logit_transform.unsupported_reason(bad, 25256)


def test_digest_tells_tables_apart():
    a = logit_transform.digest(transform(MUSESPARK))
    assert a == logit_transform.digest(transform(MUSESPARK))
    assert a != logit_transform.digest(transform(OTHER_TRANSFORMS["softcap 30"]))
    assert logit_transform.digest(None) != a
