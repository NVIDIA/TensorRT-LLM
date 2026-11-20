# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""Grid and route-mailbox policy for persistent routed experts."""

import torch

from ...k3r_common.ull.policy import batch_table


# Rubin (cc 10.7) TP16 batches on ``RubinExpertsKernel``: the routing kernel writes
# the latent as the experts' MXFP8 SMEM image (``experts_rubin.image_chunk``), which
# they bulk-copy, and the last quarter of FC2(B) is claimed tile by tile. B8 only
# (the image needs every token row), on the grid that spans every SM (``GRID``).
RUBIN = {16: "8:1"}


def rubin(tp, batch, device):
    """Batches on the Rubin experts kernel; the routing kernel then writes the plan."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    if not every_sm(tp, batch, device):  # other SM counts: the 152-SM grids
        return False
    return bool(batch_table(RUBIN.get(tp, "")).get(batch, 0))


# Rubin TP16 batches whose routing kernel publishes the ids and the image plan's
# completion into the experts' route mailbox, so the experts dedupe, partition and
# stream their first weights while routing (and the shared expert) still run
# (``RubinExpertsKernel.polled``; the routing kernel's ``signal``).
POLLED = {16: "8:1"}


def polled(tp, batch, device):
    """Batches whose experts poll the route mailbox (routing must then signal)."""
    if not rubin(tp, batch, device):
        return False
    return bool(batch_table(POLLED.get(tp, "")).get(batch, 0))


# Persistent grid and per-batch overrides by (TP, SMs); other SM counts take the 152-SM
# entries. 152 SMs: TP8 B3-4 on 148 CTAs; B5-8 publish on 128-136 CTAs, leaving SMs for
# the one-wave tail; grouped B9-64 on 152. 212 SMs (TP16): B5-64 on every SM (B5-8: the
# shared expert then finishes first; split batches leave their down SMs, see
# ``moe.prepared``); B2-4 on 192 with the shared expert's down rows beside them (B3-4
# ~0.7 us per layer faster than 212); B1 on 152 (TP16 B1-2 run the small-batch chain).
GRID = {(8, 152): 136, (16, 152): 152, (16, 212): 212}
GRIDS = {(8, 152): "3-4:148,5-6:128,7:132,9-64:152", (16, 212): "1:152,2-4:192"}


def grids(tp, device):
    """The persistent grid and its per-batch overrides for the device's SM count."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    key = (tp, sms) if (tp, sms) in GRID else (tp, 152)
    return GRID[key], batch_table(GRIDS.get(key, ""))


def every_sm(tp, batch, device):
    """Whether the persistent grid at ``batch`` spans every SM of the device (before
    ``moe.prepared`` gives a split shared expert's down rows their SMs, B>8)."""
    grid, overrides = grids(tp, device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return overrides.get(batch, grid) >= sms
