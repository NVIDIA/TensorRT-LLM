# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterization of the attention-plugin rotary table's float32 precision.

``RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin`` rounds
``inv_freq`` to float32 before multiplying it by the position. The stored angle
is ``position * inv_freq``, so a relative error of about 6e-8 in ``inv_freq``
becomes an *absolute* angle error that grows linearly with position: invisible
in short contexts, about a milliradian at 32k.

That is a characterization, not a complaint. Building ``inv_freq`` in double
precision and rounding once, at the end, was measured and deliberately not
adopted: it makes this side exact, but reference implementations round their
own frequency table to single precision too, so removing one side of a
two-sided rounding difference lands on a different set of near-ties rather
than on better agreement. Below 8k nothing moves either way.

So these tests pin the *current* behaviour with numbers, and pin what the
double-precision construction would buy, so that a future long-context
investigation does not have to rediscover either. If the shortcut is ever
removed, ``test_the_float32_shortcut_drifts_with_position`` is the test that
should be updated -- deliberately, and with a re-measurement, because the table
is built for every model.
"""

import numpy as np
import pytest

from tensorrt_llm.functional import RopeEmbeddingUtils

# A 64-wide head at theta 5e5: a representative long-context configuration.
HEAD_DIM = 64
THETA = 5.0e5

# Measured deltas of the stored cos/sin table against a float64 reference,
# max over the row at that position. Linear in position, as the analysis says.
MEASURED_FLOAT32_DELTAS = {
    1024: 1.5e-5,
    8127: 1.3e-4,
    32768: 5.6e-4,
}
# What building inv_freq in float64 and rounding once achieves instead: flat,
# and at the limit of what float32 storage can hold.
DOUBLE_PRECISION_BOUND = 3.0e-8


def _reference_row(position: int, dim: int, theta: float) -> np.ndarray:
    """The float64 cos/sin row at ``position``, in the plugin's interleaving."""
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angle = np.float64(position) * inv_freq
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1).reshape(-1)


def _plugin_row(position: int, dim: int, theta: float) -> np.ndarray:
    """The row the production table stores at ``position``."""
    _, table = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
        position + 1, dim, theta
    )
    per_position = dim  # dim/2 frequencies, fused into (cos, sin) pairs
    return table.reshape(-1)[position * per_position : (position + 1) * per_position].astype(
        np.float64
    )


def _double_precision_row(position: int, dim: int, theta: float) -> np.ndarray:
    """The same table built in double precision.

    ``inv_freq`` in float64, the angle formed in float64, and a single round to
    float32 when the array is stored.
    """
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angle = np.float64(position) * inv_freq
    row = np.stack([np.cos(angle), np.sin(angle)], axis=-1).reshape(-1)
    return row.astype(np.float32).astype(np.float64)


def test_the_table_is_exact_at_position_zero():
    """Position 0 has no angle to get wrong: every entry is cos 0 / sin 0."""
    row = _plugin_row(0, HEAD_DIM, THETA)
    np.testing.assert_allclose(row, _reference_row(0, HEAD_DIM, THETA), atol=1e-12)


def test_the_table_is_accurate_enough_below_eight_thousand():
    """Short contexts are unaffected, which is why this is easy to miss."""
    for position in (128, 1024):
        error = np.abs(
            _plugin_row(position, HEAD_DIM, THETA) - _reference_row(position, HEAD_DIM, THETA)
        ).max()
        assert error < 1e-4, f"position {position}: {error:.3e}"


@pytest.mark.parametrize("position,measured", sorted(MEASURED_FLOAT32_DELTAS.items()))
def test_the_float32_shortcut_drifts_with_position(position, measured):
    """The stored table is off by about the measured amount, and no more.

    Both bounds matter. The lower one is the point: at 32k the table is wrong
    by 5.6e-4, roughly a milliradian, four orders of magnitude worse than
    float32 storage. The upper one catches a regression that makes it worse
    still.
    """
    error = np.abs(
        _plugin_row(position, HEAD_DIM, THETA) - _reference_row(position, HEAD_DIM, THETA)
    ).max()
    assert measured / 4.0 < error < measured * 4.0, (
        f"position {position}: measured {measured:.1e}, got {error:.3e}"
    )


def test_the_drift_is_linear_in_position():
    """The signature of a *frequency* rounding, not a per-entry one.

    A per-entry rounding error would be flat at the float32 epsilon. Growing
    proportionally to the position is what says the error is in ``inv_freq``
    and is then multiplied by the position.
    """

    def error_at(position):
        return np.abs(
            _plugin_row(position, HEAD_DIM, THETA) - _reference_row(position, HEAD_DIM, THETA)
        ).max()

    near, far = error_at(4096), error_at(32768)
    assert far / near == pytest.approx(8.0, rel=0.5)


@pytest.mark.parametrize("position", [1024, 32768, 131072])
def test_double_precision_construction_is_flat_and_at_the_storage_limit(position):
    """What the double-precision construction buys: 3e-8 at every position.

    Kept as an executable record of the alternative. It costs nothing at run
    time -- the table is built once when the engine is constructed -- so if a
    long-context investigation ever wants it, this is the bound to expect and
    the reason a wider intermediate is enough.
    """
    error = np.abs(
        _double_precision_row(position, HEAD_DIM, THETA) - _reference_row(position, HEAD_DIM, THETA)
    ).max()
    assert error < DOUBLE_PRECISION_BOUND, f"position {position}: {error:.3e}"


def test_the_returned_dtypes_and_shapes_are_unchanged():
    """Whatever the arithmetic inside, the stored arrays stay float32.

    A change to the intermediate precision must not also widen what is
    returned, because both arrays are consumed as float32 by the attention
    plugin.
    """
    num_pos = 256
    inv_freq, table = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
        num_pos, HEAD_DIM, THETA
    )
    assert inv_freq.dtype == np.float32
    assert table.dtype == np.float32
    assert inv_freq.shape == (HEAD_DIM // 2,)
    assert table.shape == (1, num_pos * HEAD_DIM)
