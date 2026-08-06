# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA decode fast path's three conv-state tiers must agree.

``KimiKDARuntime._forward_decode`` gets each request's short-convolution
window to the kernel, and rolls the layer's pool forward by the generated
token, in one of three ways:

* the decode kernel reads the window out of the pool and stores it back
  rolled, so neither step costs a pass of its own;
* ``kda_conv_state_decode_step`` stages the window into a dense per-section
  buffer and rolls the pool in one indexed pass, for the cases the kernel
  cannot own the pool;
* the ATen gather / repack / concat / scatter sequence.

They are three implementations of one contract, so they have to produce the
same output and leave the same pool behind — including across a run of steps,
where a one-column drift in the in-place roll would accumulate rather than
cancel. Op-level parity tests cannot see this: the tier is chosen here, and
choosing wrong is silent.

Needs 1 GPU + fla-core; runs with random weights (no checkpoint).
"""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.models import modeling_kimi_linear  # noqa: E402
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiKDARuntime  # noqa: E402

HIDDEN_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 128  # the fused decode kernel only supports 128
CONV_WIDTH = 4


class _Cfg:
    hidden_size = HIDDEN_SIZE
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "short_conv_kernel_size": CONV_WIDTH,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


class _LayerCache:
    """Only what the plain-decode path reads; no fused-verify replay caches."""

    kda_qkg_cache = None


class _MambaMetadata:
    def __init__(self, max_batch_size: int, device: str) -> None:
        self._arange_buffer = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)


def _skip_without_supported_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the fused KDA decode kernel requires a CUDA device")
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3)}:
        pytest.skip("Kimi K3 is supported only on Blackwell (SM100/SM103)")


def _make_runtime() -> KimiKDARuntime:
    runtime = KimiKDARuntime(_Cfg(), layer_idx=0).to("cuda")
    # dt_bias is declared but not initialized (it expects a checkpoint), so
    # without this it holds whatever the caching allocator last left there —
    # large enough values overflow the decay to NaN, and every comparison
    # below then fails on NaN != NaN rather than on a real difference.
    # finalize_decode_weights snapshots it, so seed it first.
    torch.nn.init.normal_(runtime.mixer.dt_bias, std=0.1)
    runtime.finalize_decode_weights()
    if runtime._in_proj_weight is None:
        pytest.skip("decode fast path unavailable for this build")
    return runtime


def _run_steps(runtime, tier, x, conv_pool, ssm_pool, slot_indices, monkeypatch):
    """Drive ``steps`` decode steps through one conv-state tier.

    The tiers are selected the way the runtime selects them: the kernel-owned
    roll reads the module flag per call, while the fused staging pass is bound
    onto the instance at construction, so clearing the flag afterwards leaves
    the staging tier in place and clearing the bound pass drops to ATen.
    """
    if tier != "kernel-roll":
        monkeypatch.setattr(modeling_kimi_linear, "_FUSED_KDA_CONV_STATE_ENABLED", False)
    if tier == "aten":
        monkeypatch.setattr(runtime, "_fused_conv_state_step", None)

    metadata = _MambaMetadata(conv_pool.shape[0], "cuda")
    outputs = []
    for step in range(x.shape[1]):
        outputs.append(
            runtime._forward_decode(
                x[:, step],
                conv_pool,
                ssm_pool,
                slot_indices.long(),
                metadata,
                _LayerCache(),
                ssm_state_indices=slot_indices,
            )
        )
    return outputs


@torch.no_grad()
@pytest.mark.parametrize("tier", ["fused-stage", "aten"])
@pytest.mark.parametrize("batch", [1, 3])
def test_decode_conv_tiers_agree_over_many_steps(tier, batch, monkeypatch):
    _skip_without_supported_gpu()
    torch.manual_seed(0)
    slots, steps = batch + 3, 6
    dim = NUM_HEADS * HEAD_DIM

    runtime = _make_runtime()
    # Slots deliberately out of order and not starting at 0, so a tier that
    # confused the cache slot with the batch row would read the wrong window.
    slot_indices = torch.tensor(
        [slots - 1 - i for i in range(batch)], dtype=torch.int32, device="cuda"
    )
    conv_pool = torch.randn(slots, 3 * dim, CONV_WIDTH, dtype=torch.bfloat16, device="cuda") * 0.05
    ssm_pool = (
        torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.05
    )
    x = torch.randn(batch, steps, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1

    rolled_conv, rolled_ssm = conv_pool.clone(), ssm_pool.clone()
    rolled = _run_steps(
        runtime, "kernel-roll", x, rolled_conv, rolled_ssm, slot_indices, monkeypatch
    )

    staged_conv, staged_ssm = conv_pool.clone(), ssm_pool.clone()
    staged = _run_steps(runtime, tier, x, staged_conv, staged_ssm, slot_indices, monkeypatch)

    for step, (got, want) in enumerate(zip(rolled, staged)):
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"output diverged at step {step}")
    torch.testing.assert_close(rolled_conv, staged_conv, rtol=0, atol=0)
    torch.testing.assert_close(rolled_ssm, staged_ssm, rtol=0, atol=0)


@torch.no_grad()
def test_kernel_roll_leaves_unadmitted_slots_untouched(monkeypatch):
    """The roll is in place, so it must reach only the admitted requests' rows."""
    _skip_without_supported_gpu()
    torch.manual_seed(1)
    batch, slots = 2, 6
    dim = NUM_HEADS * HEAD_DIM

    runtime = _make_runtime()
    slot_indices = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    untouched = torch.tensor([0, 2, 3, 5], device="cuda")
    conv_pool = torch.randn(slots, 3 * dim, CONV_WIDTH, dtype=torch.bfloat16, device="cuda") * 0.05
    ssm_pool = (
        torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.05
    )
    x = torch.randn(batch, 3, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1

    conv_before = conv_pool.index_select(0, untouched).clone()
    ssm_before = ssm_pool.index_select(0, untouched).clone()
    _run_steps(runtime, "kernel-roll", x, conv_pool, ssm_pool, slot_indices, monkeypatch)

    torch.testing.assert_close(conv_pool.index_select(0, untouched), conv_before, rtol=0, atol=0)
    torch.testing.assert_close(ssm_pool.index_select(0, untouched), ssm_before, rtol=0, atol=0)


@torch.no_grad()
def test_kernel_roll_skips_the_staging_buffer(monkeypatch):
    """The kernel-owned roll exists to not allocate per-layer staging at all.

    ``_cs_dense`` is sized at the pool's slot count and pinned for the life of
    the process (captured CUDA graphs hold the pointer), so leaving it
    unallocated is a large part of what this tier buys.
    """
    _skip_without_supported_gpu()
    torch.manual_seed(2)
    batch, slots = 2, 5
    dim = NUM_HEADS * HEAD_DIM

    runtime = _make_runtime()
    slot_indices = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    conv_pool = torch.randn(slots, 3 * dim, CONV_WIDTH, dtype=torch.bfloat16, device="cuda") * 0.05
    ssm_pool = (
        torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.05
    )
    x = torch.randn(batch, 2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1

    _run_steps(runtime, "kernel-roll", x, conv_pool, ssm_pool, slot_indices, monkeypatch)
    assert runtime._cs_dense is None

    _run_steps(runtime, "fused-stage", x, conv_pool, ssm_pool, slot_indices, monkeypatch)
    assert runtime._cs_dense is not None
